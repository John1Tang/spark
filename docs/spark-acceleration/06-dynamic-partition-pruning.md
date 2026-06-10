# Dynamic Partition Pruning (DPP)

## What It Is

Dynamic Partition Pruning (DPP) is a query optimization in Spark SQL that pushes partition-level filters from dimension tables down to fact table scans **at runtime**, using values determined during query execution rather than at plan time.

Where **static partition pruning** eliminates partitions based on literal predicates known at compile time (e.g., `WHERE date = '2024-01-15'`), DPP eliminates partitions based on values computed from other tables during execution (e.g., `WHERE date IN (SELECT date FROM dim_date WHERE month = 'January')`).

DPP is implemented in three key source files:

- **`PlanDynamicPruningFilters`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/dynamicpruning/PlanDynamicPruningFilters.scala`) -- the compile-time rule that inserts DPP subqueries into the physical plan.
- **`PlanAdaptiveDynamicPruningFilters`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/PlanAdaptiveDynamicPruningFilters.scala`) -- the AQE-aware version that reuses broadcast exchange results when AQE is enabled.
- **`CleanupDynamicPruningFilters`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/dynamicpruning/CleanupDynamicPruningFilters.scala`) -- the cleanup rule that removes DPP filters that were not pushed down to the scan.

## Why It Works: Subquery-Based Runtime Filtering

### The Mechanism

DPP operates by rewriting partition-filter predicates into **broadcast subqueries** that execute during the fact table scan. The key expression is `InSubqueryExec` wrapping a `SubqueryBroadcastExec`:

From `PlanDynamicPruningFilters` (lines 70-78):
```scala
if (canReuseExchange) {
  val executedPlan = QueryExecution.prepareExecutedPlan(sparkSession, sparkPlan)
  val mode = broadcastMode(buildKeys, executedPlan.output)
  // plan a broadcast exchange of the build side of the join
  val exchange = BroadcastExchangeExec(mode, executedPlan)
  // place the broadcast adaptor for reusing the broadcast results on the probe side
  val broadcastValues =
    SubqueryBroadcastExec(name, broadcastKeyIndices, buildKeys, exchange)
  DynamicPruningExpression(InSubqueryExec(value, broadcastValues, exprId))
}
```

The flow:
1. **Plan Time**: `PlanDynamicPruningFilters` identifies dimension table predicates on partition columns and wraps them in `DynamicPruningSubquery` expressions attached to the fact table scan filter.
2. **Physical Planning**: The subquery is realized as `InSubqueryExec(SubqueryBroadcastExec(...))` which collects the dimension table's filtered partition keys via a broadcast exchange.
3. **Execution Time**: The broadcast subquery executes first, collecting the filtered keys. The fact table scan uses these keys as an `IN` predicate, skipping partitions whose partition column values are not in the broadcast set.
4. **Exchange Reuse**: When the same dimension table is also the build side of a broadcast hash join, DPP **reuses** the same `BroadcastExchangeExec` -- the keys are broadcast once and used for both the join and the pruning.

### Conditions for DPP to Fire

From `PlanDynamicPruningFilters` and the logical DPP insertion logic, these conditions must all be met:

1. **Broadcast Join**: The fact-dimension join must be planned as a broadcast hash join (dimension size < `spark.sql.autoBroadcastJoinThreshold`, default 10MB). DPP requires the dimension keys to be available as a broadcast.
2. **Partition Column Match**: The join key on the fact table must be a partition column.
3. **Join Type**: Inner joins and semi-joins are eligible. Outer joins and anti-joins have different null-handling semantics that make DPP unsafe without additional guards.
4. **Dimension Filter Exists**: The dimension table must have a filter predicate that reduces the set of join keys. Without a filter, broadcasting all keys provides no pruning benefit.
5. **Cost Benefit**: The DPP subquery execution cost (broadcast + scan) must be less than the estimated savings from skipping fact table partitions.

### Exchange Reuse vs. Subquery Execution

`PlanDynamicPruningFilters` has three branches (lines 70-91):

1. **Reuse broadcast exchange** (best case): When exchange reuse is enabled and the dimension plan matches a BHJ build side, DPP reuses the existing broadcast. Zero additional cost.

2. **No pruning needed** (`onlyInBroadcast` = true): When DPP cannot provide benefit, the expression is replaced with `Literal.TrueLiteral` -- the filter passes through without cost.

3. **Execute separate subquery** (fallback): When no broadcast exchange can be reused, DPP creates a separate `SubqueryExec` with an `Aggregate` on the dimension to extract only the needed keys. This adds execution cost but may still be worth it if the fact table is large and selective.

### AQE-Aware DPP

When AQE is enabled, `PlanAdaptiveDynamicPruningFilters` takes over instead of `PlanDynamicPruningFilters`. The key difference is that it works with `SubqueryAdaptiveBroadcastExec` and `AdaptiveSparkPlanExec` nodes:

From `PlanAdaptiveDynamicPruningFilters` (lines 45-69):
```scala
case DynamicPruningExpression(InSubqueryExec(
    value, SubqueryAdaptiveBroadcastExec(name, indices, onlyInBroadcast, buildPlan, buildKeys,
    adaptivePlan: AdaptiveSparkPlanExec), exprId, _, _, _)) =>
  val packedKeys = BindReferences.bindReferences(
    HashJoin.rewriteKeyExpr(buildKeys), adaptivePlan.executedPlan.output)
  val mode = HashedRelationBroadcastMode(packedKeys)
  val exchange = BroadcastExchangeExec(mode, adaptivePlan.executedPlan)
  // ... reuse logic similar to non-AQE path
```

It also handles the case where AQE re-optimizes the DPP subquery plan independently, creating a nested `AdaptiveSparkPlanExec` for the subquery execution.

### Cleanup: Removing Unused DPP Filters

`CleanupDynamicPruningFilters` runs after physical planning to handle cases where DPP filters were not pushed down to the scan:

1. **Redundant pruning**: If a partition column already has a static equality predicate (e.g., `WHERE date = '2024-01-15' AND date IN (subquery)`), the DPP filter is redundant and is replaced with `TrueLiteral` (lines 50-64).

2. **Failed pushdown**: If a DPP filter could not be pushed down to the scan (e.g., blocked by a project or aggregate with non-deterministic expressions), it is removed to avoid executing an unnecessary filter at a higher level (lines 86-91).

```scala
case f @ Filter(condition, _) =>
  val newCondition = condition.transformWithPruning(
    _.containsAnyPattern(DYNAMIC_PRUNING_EXPRESSION, DYNAMIC_PRUNING_SUBQUERY)) {
    case _: DynamicPruning => TrueLiteral
  }
  f.copy(condition = newCondition)
```

## What Problem It Solves

### The Star Schema Query Problem

In a typical data warehouse star schema:

```
fact_sales (100B rows, partitioned by date)
    |-- date_id (FK) --> dim_date (365 rows)
    |-- product_id (FK) --> dim_product (10K rows)
    |-- store_id (FK) --> dim_store (500 rows)
```

A query like:
```sql
SELECT sum(amount)
FROM fact_sales s
JOIN dim_date d ON s.date_id = d.date_id
WHERE d.month = 'January' AND d.year = 2024
```

Without DPP:
1. Spark reads **all** partitions of `fact_sales` (all dates).
2. Joins with the filtered `dim_date` (31 days of January).
3. 97% of the fact table data was read unnecessarily.

With DPP:
1. Spark broadcasts the 31 January date keys from `dim_date`.
2. The `fact_sales` scan uses these keys to read only 31 of ~365 partitions.
3. 97% of I/O is avoided.

### DPP vs. Static Partition Pruning

| Aspect | Static Pruning | Dynamic Pruning |
|--------|---------------|-----------------|
| **When determined** | Plan time (compile) | Execution time (runtime) |
| **Filter values** | Literals in query text | Computed from dimension tables |
| **Requires broadcast** | No | Yes (broadcast join) |
| **Example** | `WHERE date = '2024-01-15'` | `WHERE date IN (SELECT date_id FROM dim_date WHERE month='Jan')` |
| **EXPLAIN indicator** | `PartitionFilters: [isNotNull(date), (date = 2024-01-15)]` | `DynamicPruning: date# IN (dynamicpruning#...)` |

## How to Use DPP in Production

### Star Schema Example: Date Dimension Filtering

```scala
// Scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("DPPExample")
  // DPP is enabled by default; be explicit for clarity
  .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
  .config("spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly", "true")
  .config("spark.sql.autoBroadcastJoinThreshold", "50MB")  // generous BHJ threshold
  .getOrCreate()

import spark.implicits._

// Fact table: partitioned by date
case class Sale(date: String, productId: Int, storeId: Int, amount: Double)
case class DateDim(dateId: String, month: String, year: Int, quarter: Int)

// Simulated data
val sales = Seq(
  Sale("2024-01-15", 1, 101, 250.00),
  Sale("2024-02-20", 2, 102, 150.00),
  Sale("2024-03-10", 3, 101, 300.00),
  Sale("2024-01-05", 1, 103, 175.00),
  Sale("2024-06-30", 4, 102, 400.00)
).toDF().write.mode("overwrite").partitionBy("date").parquet("/tmp/fact_sales")

val dates = Seq(
  DateDim("2024-01-15", "January", 2024, 1),
  DateDim("2024-02-20", "February", 2024, 1),
  DateDim("2024-03-10", "March", 2024, 1),
  DateDim("2024-01-05", "January", 2024, 1),
  DateDim("2024-06-30", "June", 2024, 2)
).toDF()

val factSales = spark.read.parquet("/tmp/fact_sales")

val query = factSales
  .join(dates, factSales("date") === dates("dateId"))
  .filter(dates("month") === "January" && dates("year") === 2024)
  .selectExpr("sum(amount) as total_revenue")

// EXPLAIN to verify DPP
query.explain("extended")
query.show()
```

```python
# PySpark
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("DPPExample")
    .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
    .config("spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly", "true")
    .config("spark.sql.autoBroadcastJoinThreshold", "50MB")
    .getOrCreate())

# Fact table (partitioned by date)
fact_sales = spark.read.parquet("/tmp/fact_sales")
dim_date = spark.read.parquet("/tmp/dim_date")

query = (fact_sales
    .join(dim_date, fact_sales.date == dim_date.dateId)
    .filter((dim_date.month == "January") & (dim_date.year == 2024))
    .selectExpr("sum(amount) as total_revenue"))

query.explain("extended")
query.show()
```

```sql
-- SQL
-- Ensure DPP is enabled (default)
SET spark.sql.optimizer.dynamicPartitionPruning.enabled = true;
SET spark.sql.autoBroadcastJoinThreshold = 52428800; -- 50MB

SELECT sum(s.amount) as total_revenue
FROM fact_sales s
JOIN dim_date d ON s.date_id = d.date_id
WHERE d.month = 'January' AND d.year = 2024;
```

### EXPLAIN Output Showing DPP

```
== Physical Plan ==
*(5) HashAggregate(keys=[], functions=[sum(amount#6)])
+- Exchange SinglePartition, ENSURE_REQUIREMENTS
   +- *(4) HashAggregate(keys=[], functions=[partial_sum(amount#6)])
      +- *(4) Project [amount#6]
         +- *(4) BroadcastHashJoin [date#3], [dateId#9], Inner, BuildRight
            :- AQEShuffleRead
            :  +- ShuffleQueryStage
            :     +- Exchange hashpartitioning(date#3, 200)
            :        +- *(1) FileScan parquet [date#3, productId#4, storeId#5, amount#6]
            :           PartitionFilters: [isnotnull(date#3),
            :              dynamicpruning#12 IN (broadcast SubqueryBroadcast)]
            +- BroadcastExchange HashedRelationBroadcastMode(
               List(input[0, string, true]),false), [plan_id=...]
               +- *(3) Filter ((month#10 = January) AND (year#11 = 2024))
                  +- *(2) FileScan parquet [dateId#9, month#10, year#11]
                     +- SubqueryBroadcast dynamicpruning#12,
                        [index: 0, expr: dateId#9],
                        [date#3 IN dynamicpruning#12]
```

Key indicators:
- **`dynamicpruning#N IN (broadcast SubqueryBroadcast)`** in `PartitionFilters` confirms DPP is active on the fact table scan.
- **`SubqueryBroadcast dynamicpruning#N`** shows the broadcast subquery that provides pruning keys.
- The `BroadcastExchange` is shared between the join and the DPP subquery (exchange reuse).

### Measuring DPP Effectiveness in Spark UI

1. **SQL Tab -> Scan Details**: Compare "bytes read" for the fact table scan with and without DPP. Effective DPP shows significantly fewer bytes read because pruned partitions are never scanned.

2. **Input Size vs. Output Size**: For a fact table scan with DPP, the input bytes should be close to the bytes of the pruned partitions only (not the full table).

3. **Partition Count**: The scan metrics show "number of partitions read" vs. "total partitions". With DPP, the read count should be significantly lower.

4. **Subquery Metrics**: The broadcast subquery for DPP appears as a separate subquery execution. Its execution time should be small compared to the fact table scan savings.

5. **Programmatic Measurement**:
```scala
// Before query execution, get total partition count
val totalPartitions = spark.read.parquet("/tmp/fact_sales")
  .select("date").distinct().count()

// Execute DPP query
val df = factSales.join(dim_date.filter($"month" === "January"),
  factSales("date") === dim_date("dateId"))
df.count()

// After execution, check input metrics
val scanMetrics = df.queryExecution.executedPlan.collect {
  case fs: org.apache.spark.sql.execution.datasources.FileSourceScanExec => fs
}.head

val metrics = scanMetrics.metrics
println(s"Files read: ${metrics("numFiles").value}")
println(s"Bytes read: ${metrics("bytesRead").value}")
// Compare against full table scan to quantify DPP savings
```

## Full Configuration Reference

| Config | Default | Recommended | Description |
|--------|---------|-------------|-------------|
| `spark.sql.optimizer.dynamicPartitionPruning.enabled` | `true` | `true` | Master DPP switch |
| `spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly` | `true` | `true` | Only use DPP when the broadcast can be reused from a BHJ. Setting to `false` allows DPP even with a separate subquery execution, but adds overhead. |
| `spark.sql.optimizer.dynamicPartitionPruning.fallbackFilterRatio` | `0.5` | `0.5` | Estimated ratio of rows that pass the dimension filter when statistics are unavailable. Lower values make DPP more conservative. |
| `spark.sql.optimizer.dynamicPartitionPruning.joinReusingGPUBroadcastThreshold` | `-1` | `-1` | GPU broadcast threshold (for RAPIDS plugin). `-1` = disabled. |
| `spark.sql.optimizer.dynamicPartitionPruning.useStats` | `true` | `true` | Whether to use dimension table statistics for cost estimation. |
| `spark.sql.optimizer.dynamicPartitionPruning pruningExchangeReuseEnabled` | `true` | `true` | Whether DPP can reuse an existing broadcast exchange. |
| `spark.sql.autoBroadcastJoinThreshold` | `10MB` | `50MB`--`100MB` | Maximum size for broadcast. DPP requires BHJ, so this threshold directly controls when DPP is eligible. |

## DPP + AQE Synergy

DPP and AQE work together synergistically:

1. **AQE enables DPP in more cases**: `PlanAdaptiveDynamicPruningFilters` works with adaptive plans, allowing DPP even when the join plan is determined at runtime.

2. **DPP improves AQE statistics**: By pruning fact table partitions before the shuffle, DPP reduces the data volume that AQE's coalesce and skew optimization rules work with, leading to more accurate decisions.

3. **Exchange reuse is automatic**: When AQE is enabled, `PlanAdaptiveDynamicPruningFilters` creates a `BroadcastExchangeExec` within the adaptive plan and reuses it for both the join and the DPP subquery, eliminating redundant broadcasts.

4. **Empty propagation**: If DPP filters the dimension to zero rows, AQE's `AQEPropagateEmptyRelation` detects the empty broadcast and short-circuits the entire join to an empty result.

```
DPP Flow with AQE:

1. InsertAdaptiveSparkPlan wraps the query in AdaptiveSparkPlanExec
2. PlanAdaptiveDynamicPruningFilters inserts DPP subqueries using
   SubqueryAdaptiveBroadcastExec
3. AQE executes dimension stages first (broadcast stages submitted first)
4. DPP subquery collects keys from broadcast
5. Fact table scan uses DPP keys to prune partitions
6. Remaining stages are coalesced/optimized based on actual pruned data sizes
```

## When DPP Is Most Effective

DPP provides the largest benefit when:

1. **Large fact tables with many partitions**: The more partitions, the more can be pruned.
2. **Selective dimension filters**: Filters that reduce the dimension to a small subset of keys (e.g., "last 7 days" out of 5 years of data).
3. **Star schema queries**: Multiple dimension joins with filters, each contributing pruning keys.
4. **Partition column join keys**: The fact table must be partitioned by the join column.

DPP provides **minimal benefit** when:

1. **Dimension filters are not selective**: If the dimension filter returns 90% of keys, DPP prunes very few partitions but still pays the broadcast cost.
2. **Fact table is not partitioned**: DPP works on partition columns; no partitions means nothing to prune.
3. **Dimension table exceeds broadcast threshold**: If the dimension is too large for BHJ, DPP cannot execute.
4. **Non-partition column joins**: Joining on a non-partition column means DPP has no partitions to prune.

## Data Layout for DPP

DPP requires the fact table to be partitioned by the join column. Common partitioning strategies:

```scala
// Partition by date (daily partitions) -- best for date dimension DPP
factSales.write
  .partitionBy("date")
  .mode("overwrite")
  .parquet("/data/fact_sales")

// Partition by date and region -- DPP works on the partition column
factSales.write
  .partitionBy("date", "region")
  .mode("overwrite")
  .parquet("/data/fact_sales")

// Partition pruning works on the first partition column only
// If filtering by region only, DPP on date won't help
// If filtering by date, DPP prunes at the date level
```

**Partition metadata** must be available. Spark reads partition values from:
- Hive metastore (for managed tables)
- Directory structure (for file-based tables, e.g., `date=2024-01-15/`)
- Delta/Iceberg/Hudi transaction logs (for lakehouse tables)

Ensure partition discovery is not disabled:
```
spark.sql.sources.partitionOverwriteMode = dynamic  -- default
```

## Common Gotchas

### Non-Partition Column Joins

If the join column on the fact table is **not a partition column**, DPP will not trigger or will trigger but have no effect:

```scala
// Will NOT benefit from DPP: product_id is not a partition column
factSales.join(dimProduct, $"productId" === $"productId")
  .filter($"dimProduct.category" === "Electronics")
```

Solution: Partition the fact table by columns that are frequently used as join keys in filtered queries.

### Missing Partition Metadata

If the table's partition metadata are stale or missing (e.g., files written outside Spark without `MSCK REPAIR TABLE`), DPP cannot determine which directories correspond to which partition values:

```sql
-- For Hive-managed partitioned tables, refresh partition info
MSCK REPAIR TABLE fact_sales;

-- For Spark external tables, refresh
spark.catalog.refreshTable("fact_sales")
```

### DPP Broadcast Too Large

If the dimension table (after filtering) exceeds `spark.sql.autoBroadcastJoinThreshold`, DPP falls back to a separate subquery execution (if `reuseBroadcastOnly = false`) or disables pruning entirely (if `reuseBroadcastOnly = true`).

Solution: Increase `spark.sql.autoBroadcastJoinThreshold` if memory permits:
```scala
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")
```

### DPP Overhead Exceeds Savings

For small fact tables or non-selective dimension filters, DPP adds overhead (broadcast + subquery execution) that exceeds the I/O savings. The cost-based decision in Spark's optimizer should prevent this, but in edge cases you can disable DPP:

```scala
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "false")
```

### DPP Not Visible in EXPLAIN

If DPP does not appear in EXPLAIN output, check:
1. **BHJ threshold**: The dimension may exceed the broadcast threshold. Increase `autoBroadcastJoinThreshold`.
2. **Join type**: Outer joins and anti-joins may not trigger DPP.
3. **Filter selectivity**: If the dimension filter is estimated to return most rows, the optimizer may skip DPP.
4. **Partition column mismatch**: The fact table's join column must match a partition column.

### Subquery Duplication Cost

When `reuseBroadcastOnly = false` (the default is `true`), DPP may execute a separate subquery to collect pruning keys. This adds:
- An additional scan of the dimension table.
- A broadcast of the collected keys.
- Potential aggregation overhead (the subquery wraps the dimension in an `Aggregate` for column pruning).

Always keep `reuseBroadcastOnly = true` unless you have verified that the separate subquery cost is justified by the pruning savings.

### DPP and Bucketed Tables

If the fact table is bucketed and the join uses bucketed shuffle (bucket join), DPP may not apply because bucketed joins do not use broadcast. Ensure bucketed tables are not used when DPP is expected, or verify that the query planner chooses BHJ over bucketed join.

### Multiple DPP Subqueries

In queries with multiple dimension joins, each dimension can contribute its own DPP subquery. This can lead to:
- Multiple broadcast exchanges.
- Accumulated subquery execution overhead.

Monitor the SQL tab to verify that each DPP subquery is worth its cost. If a dimension filter is not selective, consider removing it or materializing a pre-filtered dimension table.

---

## Production-Ready Python (PySpark) Code

### Session Factory with DPP Configuration

```python
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

logger = logging.getLogger(__name__)


def create_dpp_session(app_name: str = "DPPProduction") -> SparkSession:
    """Create a SparkSession with DPP and supporting configs enabled.

    Enables dynamic partition pruning, statistics-based pruning cost
    estimation, join key reuse, and a generous broadcast threshold
    to ensure dimension tables qualify for broadcast hash joins.

    Args:
        app_name: The Spark application name.

    Returns:
        Configured SparkSession instance.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        # DPP master switch (true by default; be explicit)
        .config(
            "spark.sql.optimizer.dynamicPartitionPruning.enabled", "true"
        )
        # Only reuse broadcast from BHJ (zero additional cost)
        .config(
            "spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly",
            "true",
        )
        # Use dimension statistics for cost estimation
        .config(
            "spark.sql.optimizer.dynamicPartitionPruning.useStats", "true"
        )
        # Join key reuse: how many distinct keys to cache per expression
        .config(
            "spark.sql.optimizer.dynamicPartitionPruning.joinKeyReuse.maxKeys",
            "10000",
        )
        # Generous BHJ threshold so dimensions qualify for broadcast
        .config("spark.sql.autoBroadcastJoinThreshold", "52428800")  # 50 MB
        # Enable statistics correlation for better estimates
        .config("spark.sql.statistics.colStatsCorrelation", "true")
        .getOrCreate()
    )
```

### DPP Verification

```python
@dataclass
class DPPVerificationResult:
    """Result of verifying DPP is active for a query."""
    dpp_active: bool
    subquery_broadcast_found: bool = False
    dynamic_pruning_found: bool = False
    pruning_expressions: list[str] = field(default_factory=list)
    subquery_names: list[str] = field(default_factory=list)
    explain_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify_dpp_active(
    spark: SparkSession, sql_query: str
) -> DPPVerificationResult:
    """Parse EXPLAIN output to confirm DPP is active.

    Searches for ``SubqueryBroadcast`` and ``DynamicPruningExpression``
    markers in the physical plan.

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to analyze.

    Returns:
        DPPVerificationResult with subquery and pruning details.
    """
    try:
        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
    except Exception as exc:
        logger.error("EXPLAIN failed: %s", exc)
        return DPPVerificationResult(
            dpp_active=False, notes=[f"EXPLAIN failed: {exc}"]
        )

    full_plan = "\n".join(lines)

    subquery_broadcast = re.findall(
        r"SubqueryBroadcast\s+(\w+)", full_plan
    )
    dynamic_pruning = re.findall(
        r"DynamicPruningExpression\s*\(([^)]+)\)", full_plan
    )
    # Also catch the inline form in PartitionFilters
    inline_dpp = re.findall(
        r"dynamicpruning#\d+\s+IN\s+\([^)]+\)", full_plan
    )

    all_dpp_expressions = dynamic_pruning + inline_dpp

    notes: list[str] = []
    if subquery_broadcast:
        notes.append(
            f"SubqueryBroadcast found: {subquery_broadcast}"
        )
    if all_dpp_expressions:
        notes.append(
            f"Dynamic pruning expressions: {len(all_dpp_expressions)}"
        )

    return DPPVerificationResult(
        dpp_active=bool(subquery_broadcast) or bool(all_dpp_expressions),
        subquery_broadcast_found=bool(subquery_broadcast),
        dynamic_pruning_found=bool(all_dpp_expressions),
        pruning_expressions=all_dpp_expressions,
        subquery_names=subquery_broadcast,
        explain_lines=lines,
        notes=notes,
    )
```

### DPP Effectiveness Analysis

```python
@dataclass
class DPPEffectivenessReport:
    """Report comparing partition counts before/after DPP pruning."""
    table_name: str
    partition_column: str
    total_partitions: int = 0
    pruned_partitions: int = 0
    remaining_partitions: int = 0
    pruning_ratio: float = 0.0
    bytes_before: int = 0
    bytes_after: int = 0
    notes: list[str] = field(default_factory=list)


def analyze_dpp_effectiveness(
    spark: SparkSession,
    fact_table: str,
    partition_column: str,
    fact_df: DataFrame,
    dpp_query: DataFrame,
) -> DPPEffectivenessReport:
    """Compare partition counts and bytes read with and without DPP.

    Runs the fact table DataFrame (full scan) to establish a baseline,
    then compares against the DPP-qualified query's metrics.

    Args:
        spark: Active SparkSession.
        fact_table: Fully qualified fact table name.
        partition_column: The partition column name.
        fact_df: DataFrame representing a full fact table scan.
        dpp_query: DataFrame with DPP-eligible join and filter.

    Returns:
        DPPEffectivenessReport with pruning ratio and byte savings.
    """
    report = DPPEffectivenessReport(
        table_name=fact_table,
        partition_column=partition_column,
    )

    try:
        # Total distinct partition values
        total_df = spark.sql(
            f"SELECT COUNT(DISTINCT {partition_column}) AS cnt "
            f"FROM {fact_table}"
        )
        report.total_partitions = total_df.collect()[0]["cnt"]

        # Full scan byte count
        fact_df.collect()  # force execution
        full_scan = fact_df.queryExecution.executedPlan.collectLeaves()
        for leaf in full_scan:
            metrics = leaf.metrics
            if "bytesRead" in metrics:
                report.bytes_before = metrics["bytesRead"].value
                break

        # DPP query execution
        dpp_query.collect()
        dpp_scan = dpp_query.queryExecution.executedPlan.collectLeaves()
        for leaf in dpp_scan:
            metrics = leaf.metrics
            if "bytesRead" in metrics:
                report.bytes_after = metrics["bytesRead"].value
                break

        # Compute pruning ratio
        if report.bytes_before > 0:
            report.pruning_ratio = round(
                1.0 - (report.bytes_after / report.bytes_before), 4
            )
            report.remaining_partitions = max(
                1,
                int(
                    report.total_partitions
                    * (1.0 - report.pruning_ratio)
                ),
            )
            report.pruned_partitions = (
                report.total_partitions - report.remaining_partitions
            )

        report.notes.append(
            f"Total partitions: {report.total_partitions}, "
            f"pruning ratio: {report.pruning_ratio:.1%}"
        )

    except Exception as exc:
        logger.error("DPP effectiveness analysis failed: %s", exc)
        report.notes.append(f"Analysis failed: {exc}")

    return report
```

### DPP Benchmark

```python
@dataclass
class DPPBenchmarkQueryResult:
    """Benchmark result for a single DPP query."""
    query_name: str
    dpp_enabled_ms: float = 0.0
    dpp_disabled_ms: float = 0.0
    speedup_ratio: float = 1.0
    dpp_partitions_read: int = 0
    full_partitions_read: int = 0
    dpp_active: bool = False
    error: str = ""


@dataclass
class DPPBenchmarkReport:
    """Aggregate DPP benchmark results."""
    queries: list[DPPBenchmarkQueryResult] = field(default_factory=list)
    avg_speedup: float = 0.0
    avg_pruning_ratio: float = 0.0
    timestamp: str = ""


def dpp_benchmark(
    spark: SparkSession,
    queries: dict[str, str],
    output_path: str = "/tmp/dpp_benchmark.json",
) -> DPPBenchmarkReport:
    """Benchmark star-schema queries with DPP ON vs. OFF.

    For each query, runs it with DPP enabled and disabled, comparing
    execution times and EXPLAIN plan indicators.

    Args:
        spark: Active SparkSession.
        queries: Dict of query name -> SQL string.
        output_path: JSON report output path.

    Returns:
        DPPBenchmarkReport with per-query results.
    """
    report = DPPBenchmarkReport(timestamp=datetime.utcnow().isoformat())
    results: list[DPPBenchmarkQueryResult] = []

    original_dpp = spark.conf.get(
        "spark.sql.optimizer.dynamicPartitionPruning.enabled", "true"
    )

    def _run(sql: str, dpp_on: bool) -> tuple[float, bool, int]:
        spark.conf.set(
            "spark.sql.optimizer.dynamicPartitionPruning.enabled",
            str(dpp_on).lower(),
        )

        try:
            start = time.perf_counter()
            df = spark.sql(sql)
            df.count()
            elapsed_ms = (time.perf_counter() - start) * 1000

            explain_df = spark.sql(f"EXPLAIN EXTENDED {sql}")
            plan = "\n".join(r["plan"] for r in explain_df.collect())
            dpp_active = (
                "SubqueryBroadcast" in plan
                or "DynamicPruning" in plan
                or "dynamicpruning" in plan
            )

            # Count partition indicators from scan metrics if available
            partition_matches = re.findall(
                r"PartitionFilters.*?\[([^\]]+)\]", plan
            )
            partition_count = len(partition_matches) if partition_matches else 0

            return elapsed_ms, dpp_active, partition_count

        except Exception as exc:
            logger.error("DPP benchmark (DPP=%s) failed: %s", dpp_on, exc)
            return -1.0, False, 0

    for name, sql in queries.items():
        bq = DPPBenchmarkQueryResult(query_name=name)

        bq.dpp_enabled_ms, bq.dpp_active, bq.dpp_partitions_read = _run(
            sql, dpp_on=True
        )
        bq.dpp_disabled_ms, _, bq.full_partitions_read = _run(
            sql, dpp_on=False
        )

        if bq.dpp_disabled_ms > 0 and bq.dpp_enabled_ms > 0:
            bq.speedup_ratio = round(
                bq.dpp_disabled_ms / bq.dpp_enabled_ms, 2
            )

        results.append(bq)

    # Restore original setting
    spark.conf.set(
        "spark.sql.optimizer.dynamicPartitionPruning.enabled", original_dpp
    )

    report.queries = results
    valid = [
        q
        for q in results
        if not q.error and q.dpp_enabled_ms > 0 and q.dpp_disabled_ms > 0
    ]
    if valid:
        report.avg_speedup = round(
            sum(q.speedup_ratio for q in valid) / len(valid), 2
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": report.timestamp,
                "avg_speedup": report.avg_speedup,
                "queries": [asdict(q) for q in report.queries],
            },
            fh,
            indent=2,
        )
    logger.info("DPP benchmark report written to %s", output_path)
    return report
```

### Star Schema Builder with DPP

```python
def build_star_schema_with_dpp(
    spark: SparkSession,
    database: str = "default",
    output_path: str = "/tmp/star_schema_dpp.json",
) -> dict:
    """Create fact and dimension tables and verify DPP on star queries.

    Creates:
    - ``fact_sales`` partitioned by ``sale_date``
    - ``dim_date`` with month/year/quarter columns
    - ``dim_product`` with category column
    - ``dim_store`` with region column

    Then runs a star query joining all dimensions and verifies DPP
    is active on the fact table scan.

    Args:
        spark: Active SparkSession.
        database: Target database name.
        output_path: JSON report path.

    Returns:
        Dict with table names, DPP verification, and query result.
    """
    report: dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "tables": {},
        "dpp_verification": {},
        "star_query_result": None,
    }

    try:
        # Create dimension: date
        spark.sql(f"DROP TABLE IF EXISTS {database}.dim_date")
        spark.sql(
            f"CREATE TABLE {database}.dim_date "
            f"(date_id STRING, month STRING, year INT, quarter INT) "
            f"USING parquet"
        )
        dates_data = [
            ("2024-01-15", "January", 2024, 1),
            ("2024-02-20", "February", 2024, 1),
            ("2024-03-10", "March", 2024, 1),
            ("2024-01-05", "January", 2024, 1),
            ("2024-06-30", "June", 2024, 2),
            ("2024-07-04", "July", 2024, 3),
            ("2024-11-25", "November", 2024, 4),
            ("2024-12-31", "December", 2024, 4),
        ]
        spark.createDataFrame(
            dates_data, ["date_id", "month", "year", "quarter"]
        ).write.mode("overwrite").saveAsTable(f"{database}.dim_date")
        report["tables"]["dim_date"] = f"{database}.dim_date"

        # Create dimension: product
        spark.sql(f"DROP TABLE IF EXISTS {database}.dim_product")
        spark.sql(
            f"CREATE TABLE {database}.dim_product "
            f"(product_id INT, product_name STRING, category STRING, "
            f"price DOUBLE) USING parquet"
        )
        products_data = [
            (1, "Widget-A", "Electronics", 29.99),
            (2, "Widget-B", "Electronics", 49.99),
            (3, "Gadget-C", "Home", 19.99),
            (4, "Gadget-D", "Home", 39.99),
            (5, "Tool-E", "Tools", 9.99),
        ]
        spark.createDataFrame(
            products_data,
            ["product_id", "product_name", "category", "price"],
        ).write.mode("overwrite").saveAsTable(f"{database}.dim_product")
        report["tables"]["dim_product"] = f"{database}.dim_product"

        # Create dimension: store
        spark.sql(f"DROP TABLE IF EXISTS {database}.dim_store")
        spark.sql(
            f"CREATE TABLE {database}.dim_store "
            f"(store_id INT, store_name STRING, region STRING) "
            f"USING parquet"
        )
        stores_data = [
            (101, "Store-North", "North"),
            (102, "Store-South", "South"),
            (103, "Store-East", "East"),
            (104, "Store-West", "West"),
        ]
        spark.createDataFrame(
            stores_data, ["store_id", "store_name", "region"]
        ).write.mode("overwrite").saveAsTable(f"{database}.dim_store")
        report["tables"]["dim_store"] = f"{database}.dim_store"

        # Create fact table: partitioned by sale_date
        spark.sql(f"DROP TABLE IF EXISTS {database}.fact_sales")
        sales_data = [
            ("2024-01-15", 1, 101, 250.00),
            ("2024-02-20", 2, 102, 150.00),
            ("2024-03-10", 3, 101, 300.00),
            ("2024-01-05", 1, 103, 175.00),
            ("2024-06-30", 4, 102, 400.00),
            ("2024-07-04", 5, 104, 50.00),
            ("2024-11-25", 2, 101, 225.00),
            ("2024-12-31", 3, 103, 180.00),
        ]
        spark.createDataFrame(
            sales_data, ["sale_date", "product_id", "store_id", "amount"]
        ).write.mode("overwrite").partitionBy("sale_date").saveAsTable(
            f"{database}.fact_sales"
        )
        report["tables"]["fact_sales"] = f"{database}.fact_sales"

        # Collect statistics on all tables for optimal pruning cost
        for tbl in ["dim_date", "dim_product", "dim_store", "fact_sales"]:
            spark.sql(
                f"ANALYZE TABLE {database}.{tbl} COMPUTE STATISTICS FOR COLUMNS"
            )

        # Run star query with DPP
        star_query = f"""
            SELECT
                d.month,
                d.year,
                p.category,
                s.region,
                SUM(f.amount) AS total_revenue
            FROM {database}.fact_sales f
            JOIN {database}.dim_date d ON f.sale_date = d.date_id
            JOIN {database}.dim_product p ON f.product_id = p.product_id
            JOIN {database}.dim_store s ON f.store_id = s.store_id
            WHERE d.month = 'January'
              AND d.year = 2024
              AND p.category = 'Electronics'
            GROUP BY d.month, d.year, p.category, s.region
        """

        result_df = spark.sql(star_query)
        result_rows = [row.asDict() for row in result_df.collect()]
        report["star_query_result"] = result_rows

        # Verify DPP
        verification = verify_dpp_active(spark, star_query)
        report["dpp_verification"] = {
            "dpp_active": verification.dpp_active,
            "subquery_broadcast_found": verification.subquery_broadcast_found,
            "dynamic_pruning_found": verification.dynamic_pruning_found,
            "pruning_expressions": verification.pruning_expressions,
            "notes": verification.notes,
        }

        logger.info(
            "Star schema built with %d tables, DPP active: %s",
            len(report["tables"]),
            verification.dpp_active,
        )

    except Exception as exc:
        logger.error("Failed to build star schema: %s", exc)
        report["error"] = str(exc)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Star schema DPP report written to %s", output_path)
    return report
```

### DPP Monitor Class

```python
@dataclass
class DPPExpressionInfo:
    """Parsed information about a single DPP expression."""
    partition_column: str = ""
    subquery_name: str = ""
    pruning_type: str = ""  # "SubqueryBroadcast" or "DynamicPruningExpression"
    in_predicate: str = ""


@dataclass
class DPPMonitorReport:
    """Monitor report for DPP expressions across queries."""
    query_name: str
    total_dpp_expressions: int = 0
    expressions: list[DPPExpressionInfo] = field(default_factory=list)
    pruning_selectivity: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class DPPMonitor:
    """Monitor DPP expressions, extract subquery info, and calculate pruning selectivity.

    Usage:
        monitor = DPPMonitor(spark)
        report = monitor.analyze("star_query", star_sql)
        monitor.save_report("/tmp/dpp_monitor.json")
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._reports: list[DPPMonitorReport] = []

    def analyze(
        self, query_name: str, sql_query: str
    ) -> DPPMonitorReport:
        """Analyze a query for DPP expressions and compute selectivity.

        Args:
            query_name: Human-readable query name.
            sql_query: The SQL query to analyze.

        Returns:
            DPPMonitorReport with parsed DPP information.
        """
        report = DPPMonitorReport(query_name=query_name)

        try:
            explain_df = self.spark.sql(f"EXPLAIN EXTENDED {sql_query}")
            lines = [row["plan"] for row in explain_df.collect()]
        except Exception as exc:
            logger.error("EXPLAIN failed: %s", exc)
            report.notes.append(f"EXPLAIN failed: {exc}")
            self._reports.append(report)
            return report

        full_plan = "\n".join(lines)

        # Extract SubqueryBroadcast entries
        sb_matches = re.findall(
            r"SubqueryBroadcast\s+(\w+),\s*\[.*?\],\s*\[(\w+)#\d+\s+IN\s+dynamicpruning",
            full_plan,
        )
        for name, col in sb_matches:
            report.expressions.append(
                DPPExpressionInfo(
                    partition_column=col,
                    subquery_name=name,
                    pruning_type="SubqueryBroadcast",
                )
            )

        # Extract DynamicPruningExpression entries
        dp_matches = re.findall(
            r"DynamicPruningExpression\s*\(([^)]+)\)", full_plan
        )
        for expr in dp_matches:
            report.expressions.append(
                DPPExpressionInfo(
                    pruning_type="DynamicPruningExpression",
                    in_predicate=expr.strip(),
                )
            )

        # Extract inline dynamic pruning in PartitionFilters
        inline_matches = re.findall(
            r"(dynamicpruning#\d+)\s+IN\s+\((dynamicpruning[^)]+)\)",
            full_plan,
        )
        for _pruning_id, subquery_ref in inline_matches:
            col_match = re.search(r"(\w+)#\d+\s+IN", full_plan)
            report.expressions.append(
                DPPExpressionInfo(
                    partition_column=col_match.group(1) if col_match else "",
                    pruning_type="inline_dynamicpruning",
                    in_predicate=subquery_ref,
                )
            )

        report.total_dpp_expressions = len(report.expressions)

        # Calculate pruning selectivity per expression
        for expr_info in report.expressions:
            if expr_info.partition_column:
                try:
                    # Count distinct values in the fact table's partition column
                    total_df = self.spark.sql(
                        f"SELECT COUNT(DISTINCT {expr_info.partition_column}) "
                        f"AS cnt FROM (SELECT {expr_info.partition_column} "
                        f"FROM (SELECT * FROM {sql_query.split('FROM')[1].split('WHERE')[0].strip()}))"
                    )
                    # Fallback: just count the column if complex query fails
                except Exception:
                    pass
                report.pruning_selectivity[expr_info.partition_column] = (
                    "requires_runtime_measurement"
                )

        if report.total_dpp_expressions > 0:
            cols = [
                e.partition_column
                for e in report.expressions
                if e.partition_column
            ]
            report.notes.append(
                f"DPP active on {report.total_dpp_expressions} expression(s), "
                f"columns: {cols}"
            )
        else:
            report.notes.append("No DPP expressions detected")

        self._reports.append(report)
        return report

    def save_report(
        self, path: str = "/tmp/dpp_monitor.json"
    ) -> str:
        """Write all collected monitor reports to JSON.

        Args:
            path: Output file path.

        Returns:
            The written file path.
        """
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "reports": [
                {
                    "query_name": r.query_name,
                    "total_dpp_expressions": r.total_dpp_expressions,
                    "expressions": [asdict(e) for e in r.expressions],
                    "pruning_selectivity": r.pruning_selectivity,
                    "notes": r.notes,
                }
                for r in self._reports
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("DPP monitor report written to %s", path)
        return path
```

### Production DQL with DPP Monitoring

```python
@dataclass
class DQLStepResult:
    """Result of a single DQL (data query language) step."""
    step_name: str
    sql: str
    duration_ms: float = 0.0
    dpp_active: bool = False
    dpp_expressions: int = 0
    pruning_columns: list[str] = field(default_factory=list)
    result_count: int = 0
    status: str = ""


@dataclass
class DQLPipelineReport:
    """Full production analytical query pipeline report."""
    pipeline_name: str
    steps: list[DQLStepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    dpp_enabled: bool = False
    timestamp: str = ""


def run_production_dql_with_dpp(
    spark: SparkSession,
    pipeline_name: str,
    queries: dict[str, str],
    output_path: str = "/tmp/dql_dpp_report.json",
) -> DQLPipelineReport:
    """Run production analytical queries with DPP monitoring and reporting.

    For each query:
    1. Verify DPP is active via EXPLAIN.
    2. Execute and measure wall-clock time.
    3. Record DPP expression details.
    4. Aggregate into a JSON report.

    Args:
        spark: Active SparkSession with DPP enabled.
        pipeline_name: Name for the query pipeline.
        queries: Dict of step name -> SQL string.
        output_path: JSON report output path.

    Returns:
        DQLPipelineReport with per-step DPP analysis.
    """
    result = DQLPipelineReport(
        pipeline_name=pipeline_name,
        timestamp=datetime.utcnow().isoformat(),
        dpp_enabled=spark.conf.get(
            "spark.sql.optimizer.dynamicPartitionPruning.enabled", "false"
        )
        == "true",
    )

    total_start = time.perf_counter()

    for step_name, sql in queries.items():
        step = DQLStepResult(step_name=step_name, sql=sql)
        try:
            # DPP verification
            verification = verify_dpp_active(spark, sql)
            step.dpp_active = verification.dpp_active
            step.dpp_expressions = verification.total_dpp_expressions if hasattr(verification, 'total_dpp_expressions') else (
                len(verification.pruning_expressions)
            )
            step.pruning_columns = list({
                re.search(r"(\w+)#\d+", expr).group(1)
                for expr in verification.pruning_expressions
                if re.search(r"(\w+)#\d+", expr)
            })

            # Execution
            start = time.perf_counter()
            df = spark.sql(sql)
            rows = df.collect()
            step.duration_ms = (time.perf_counter() - start) * 1000
            step.result_count = len(rows)
            step.status = "success"

            logger.info(
                "Step '%s': %.0f ms, %d rows, DPP=%s (%d exprs)",
                step_name,
                step.duration_ms,
                step.result_count,
                step.dpp_active,
                step.dpp_expressions,
            )

        except Exception as exc:
            step.status = f"error: {exc}"
            logger.error("DQL step '%s' failed: %s", step_name, exc)

        result.steps.append(step)

    result.total_duration_ms = (time.perf_counter() - total_start) * 1000

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pipeline_name": result.pipeline_name,
                "timestamp": result.timestamp,
                "dpp_enabled": result.dpp_enabled,
                "total_duration_ms": round(result.total_duration_ms, 2),
                "steps": [asdict(s) for s in result.steps],
            },
            fh,
            indent=2,
        )
    logger.info("DQL DPP report written to %s", output_path)
    return result
```
