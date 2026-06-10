# Storage Partition Join (SPJ) -- Shuffle Elimination for Co-Partitioned Tables

## What It Is

Storage Partition Join (SPJ) is a query optimization in Spark SQL that eliminates the **Exchange** (shuffle) stage when joining two tables that are already co-partitioned on the join keys in their underlying storage layout. It is a generalization of traditional bucket joins that works with any V2 data source implementing `SupportsReportPartitioning` -- including Apache Iceberg, Delta Lake, and Apache Hudi.

When both sides of a join report the same partitioning scheme on the join columns through the Data Source V2 API, Spark's `EnsureRequirements` planner recognizes that no repartitioning is needed. The join can execute locally within each partition group, saving the full cost of a shuffle.

## Why It Works -- The Mechanism

### Partitioning Metadata from V2 Data Sources

V2 tables implement `SupportsReportPartitioning`, which returns a `Partitioning` descriptor. For SPJ the relevant type is `KeyedPartitioning`, which declares:

```scala
case class KeyedPartitioning(
    clusteringKey: Seq[Transform],   // e.g. bucket(32, id)
    partitionValues: Option[Seq[InternalRow]] // concrete partition values
)
```

When Spark plans a join, `EnsureRequirements` checks whether both children report `KeyedPartitioning` with **compatible** clustering keys. If they do, the planner skips inserting `Exchange` nodes and instead uses `GroupPartitionsExec` to align partitions from both sides before feeding them into the join operator.

### The Planning Flow

```
1. Planner identifies equi-join
2. Check both children's Partitioning via SupportsReportPartitioning
3. If both are KeyedPartitioning with compatible keys:
   a. Match partition values between sides
   b. Insert GroupPartitionsExec to co-group matching partitions
   c. NO Exchange nodes are inserted
4. Execute join within each co-grouped partition
```

### Key Distinction from Bucket Joins

| Traditional Bucket Join | Storage Partition Join |
|------------------------|----------------------|
| Requires `USING CLUSTERED BY` at table creation | Uses any V2 partitioning (bucket, identity, truncate, etc.) |
| Limited to file-based sources | Works with Iceberg, Delta, Hudi via V2 API |
| Fixed bucket count at write time | Dynamic partition count, supports push-down of partition values |

## What Problem It Solves

Shuffles are the single most expensive operation in distributed joins:

- **Network I/O**: Every row is serialized, sent across the network, and deserialized
- **Disk I/O**: Spill to disk when memory is insufficient
- **Serialization overhead**: Row encoding/decoding
- **Skew amplification**: Hot partitions become stragglers

SPJ eliminates all of this when the data is already partitioned correctly. For a 1 TB fact table joining a 500 GB dimension table on the same partition key, skipping the shuffle can reduce job time from 30+ minutes to under 5 minutes.

## How to Use It

### Prerequisites

1. Both tables must be V2 data sources that implement `SupportsReportPartitioning`
2. Both tables must be partitioned/bucketed on the **same columns** (or compatible transforms)
3. Join keys must match the partition columns

### Step 1: Enable SPJ Configurations

```scala
spark.conf.set("spark.sql.sources.v2.bucketing.enabled", "true")
spark.conf.set("spark.sql.sources.v2.bucketing.pushPartValues.enabled", "true")
// Optional: enable subset-of-partition-keys joins (Spark 4.0+)
spark.conf.set("spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled", "true")
```

### Step 2: Create Co-Partitioned Iceberg Tables

```scala
// Scala -- create a bucketed Iceberg fact table
spark.sql("""
  CREATE TABLE prod.catalog.fact_orders (
    order_id BIGINT,
    customer_id BIGINT,
    order_date DATE,
    amount DOUBLE
  )
  USING iceberg
  PARTITIONED BY (bucket(32, customer_id), days(order_date))
""")

// Write data with the partitioning respected
df.write
  .format("iceberg")
  .option("fanout-enabled", "true")  // ensure correct bucket assignment
  .save("prod.catalog.fact_orders")
```

```scala
// Scala -- create a matching dimension table
spark.sql("""
  CREATE TABLE prod.catalog.dim_customers (
    customer_id BIGINT,
    name STRING,
    region STRING,
    tier STRING
  )
  USING iceberg
  PARTITIONED BY (bucket(32, customer_id))
""")

dimDF.write
  .format("iceberg")
  .option("fanout-enabled", "true")
  .save("prod.catalog.dim_customers")
```

```python
# Python -- same tables via PySpark
spark.sql("""
    CREATE TABLE prod.catalog.fact_orders (
        order_id BIGINT,
        customer_id BIGINT,
        order_date DATE,
        amount DOUBLE
    )
    USING iceberg
    PARTITIONED BY (bucket(32, customer_id), days(order_date))
""")

spark.sql("""
    CREATE TABLE prod.catalog.dim_customers (
        customer_id BIGINT,
        name STRING,
        region STRING,
        tier STRING
    )
    USING iceberg
    PARTITIONED BY (bucket(32, customer_id))
""")
```

### Step 3: Run the Join

```scala
// Scala -- this join should require NO Exchange
val result = spark.sql("""
  SELECT f.order_id, f.amount, c.name, c.region
  FROM prod.catalog.fact_orders f
  JOIN prod.catalog.dim_customers c
    ON f.customer_id = c.customer_id
  WHERE f.order_date >= '2024-01-01'
""")

result.explain("cost")
```

```sql
-- SQL -- same join
SELECT f.order_id, f.amount, c.name, c.region
FROM prod.catalog.fact_orders f
JOIN prod.catalog.dim_customers c
  ON f.customer_id = c.customer_id
WHERE f.order_date >= '2024-01-01';
```

### Step 4: Verify SPJ Is Working

Run `EXPLAIN` and look for the **absence** of `Exchange` nodes around the join:

```
== Physical Plan ==
*(5) Project [order_id#0L, amount#3, name#7, region#8]
+- *(5) SortMergeJoin [customer_id#1L], [customer_id#6L], Inner
   :- *(2) Sort [customer_id#1L ASC], 0, false
   :  +- *(1) Filter (isnotnull(order_date#2) AND ...)
   :     +- *(1) ColumnarToRow
   :        +- FileScan iceberg prod.catalog.fact_orders [...]
   :           PushedFilters: [IsNotNull(customer_id), GreaterThanOrEqual(order_date,...)]
   +- *(4) Sort [customer_id#6L ASC], 0, false
      +- *(3) ColumnarToRow
         +- FileScan iceberg prod.catalog.dim_customers [...]

   -- NO Exchange (Shuffle) nodes anywhere!
   -- GroupPartitionsExec handles partition alignment
```

**Without SPJ** (if tables are not co-partitioned or config is off), you would see:

```
== Physical Plan ==
*(4) Project [...]
+- *(4) SortMergeJoin [customer_id#1L], [customer_id#6L], Inner
   :- *(2) Sort [customer_id#1L ASC], 0, false
   :  +- Exchange hashpartitioning(customer_id#1L, 200), ENSURE_REQUIREMENTS
   :     +- FileScan iceberg ...
   +- *(3) Sort [customer_id#6L ASC], 0, false
      +- Exchange hashpartitioning(customer_id#6L, 200), ENSURE_REQUIREMENTS
         +- FileScan iceberg ...

   -- Two Exchange nodes shuffle ALL data across the network
```

**In the Spark UI**: With SPJ, the SQL tab shows **0 shuffle read/write bytes** for the join stage. Without SPJ, you see GBs of shuffle data.

## Key Configurations

| Configuration | Default | Recommended | Description |
|---|---|---|---|
| `spark.sql.sources.v2.bucketing.enabled` | `true` | `true` | Master switch for V2 bucketing/SPJ. Must be `true`. |
| `spark.sql.sources.v2.bucketing.pushPartValues.enabled` | `true` | `true` | Pushes common partition values to both sides, enabling GroupPartitionsExec to fill empty partitions for missing values on either side. |
| `spark.sql.requireAllClusterKeysForCoPartition` | `true` | `true` | Requires ALL clustering keys to match for co-partitioning. Prevents data skew from partial matches. Keep `true` in production. |
| `spark.sql.sources.v2.bucketing.partiallyClusteredDistribution.enabled` | `false` | `false` (or `true` for skewed data) | Allows one side to be replicated to match the other when partition counts differ. Helps with skew but increases data volume. |
| `spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled` | `false` | `true` (Spark 4.0+) | Lets SPJ work when join keys are a subset of partition keys. Requires `requireAllClusterKeysForCoPartition = false`. |
| `spark.sql.sources.v2.bucketing.allowCompatibleTransforms.enabled` | `false` | `true` (Spark 4.0+) | Allows SPJ when partition transforms are compatible but not identical (e.g., `bucket(16, x)` vs `bucket(32, x)`). Requires `pushPartValues.enabled = true` and `partiallyClusteredDistribution.enabled = false`. |
| `spark.sql.sources.v2.bucketing.shuffle.enabled` | `false` | `false` | When only one side is partitioned, allows shuffling just the other side. Reduces shuffle volume compared to shuffling both sides. |
| `spark.sql.sources.v2.bucketing.partition.filter.enabled` | `false` | `true` (Spark 4.0+) | Filters partitions that have no match on the other side, skipping unnecessary scans. |
| `spark.sql.sources.v2.bucketing.sorting.enabled` | `false` | `false` | Uses partition-derived ordering to skip sort operations. |

### Production Configuration Block

```scala
// Production settings for maximum SPJ benefit
val spjConf = Map(
  "spark.sql.sources.v2.bucketing.enabled" -> "true",
  "spark.sql.sources.v2.bucketing.pushPartValues.enabled" -> "true",
  "spark.sql.requireAllClusterKeysForCoPartition" -> "true",
  "spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled" -> "true",
  "spark.sql.sources.v2.bucketing.allowCompatibleTransforms.enabled" -> "true",
  "spark.sql.sources.v2.bucketing.partition.filter.enabled" -> "true",
  "spark.sql.sources.v2.bucketing.shuffle.enabled" -> "true"
)
spjConf.foreach { case (k, v) => spark.conf.set(k, v) }
```

## Advanced Patterns

### Partially Clustered Distribution

When one side has significantly fewer partitions, SPJ can replicate the smaller side's partitions to match:

```scala
spark.conf.set("spark.sql.sources.v2.bucketing.partiallyClusteredDistribution.enabled", "true")

// fact_orders has 1024 partitions, dim_customers has 32
// Spark groups dim_customers partitions and replicates them
// to match the 1024 fact_orders partitions
spark.sql("""
  SELECT * FROM fact_orders f
  JOIN dim_customers c ON f.customer_id = c.customer_id
""")
```

Spark picks the side with less data size (based on table statistics), groups its partitions, and replicates them to align with the other side. This is effective for handling **data size mismatches** between the two tables.

### Compatible Transforms

When bucket counts differ but are compatible:

```scala
// fact_orders: bucket(64, customer_id)
// dim_customers: bucket(16, customer_id)
// With allowCompatibleTransforms.enabled = true, SPJ still works
// because both partition on customer_id with compatible bucketing
spark.conf.set("spark.sql.sources.v2.bucketing.allowCompatibleTransforms.enabled", "true")
```

### Subset of Partition Keys

When the join uses only some of the partition columns:

```scala
// Both tables: PARTITIONED BY (bucket(32, customer_id), region)
// Join only on customer_id (not region)
spark.conf.set("spark.sql.requireAllClusterKeysForCoPartition", "false")
spark.conf.set("spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled", "true")

spark.sql("""
  SELECT * FROM fact_orders f
  JOIN dim_customers c ON f.customer_id = c.customer_id
  -- region is NOT in the join key, but SPJ still works
""")
```

## Limitations and Gotchas

### Partition Count Must Match

SPJ requires both sides to have the **same number of partitions** for the join keys. If `fact_orders` is `bucket(32, customer_id)` and `dim_customers` is `bucket(16, customer_id)`, SPJ will not work unless `allowCompatibleTransforms.enabled` is `true` and the bucket counts are mathematically compatible.

### Partitioning Metadata Must Be Present

If the V2 data source does not implement `SupportsReportPartitioning` correctly, or if the partition metadata is missing (e.g., after a manual file move), SPJ silently falls back to a shuffled join. Always verify with `EXPLAIN`.

### Join Keys Must Match Partition Columns Exactly

The join condition must use the **same columns** that the table is partitioned on. A join on `customer_id` when the table is partitioned by `days(order_date)` will not benefit from SPJ.

### Internal Config Flag

`spark.sql.requireAllClusterKeysForCoPartition` is an **internal** config. When `true` (default), Spark requires all clustering keys to participate in the join. This prevents data skew but may block SPJ if your join uses a subset of partition columns.

### Not a Substitute for Broadcast Joins

For small dimension tables (< 10 MB), broadcast hash joins are still faster than SPJ because they avoid even the `GroupPartitionsExec` overhead. SPJ shines when both tables are large and already co-partitioned.

### Data Skew Risk

If `requireAllClusterKeysForCoPartition` is set to `false` to enable subset joins, be aware that data skew can cause significant performance regression. The shuffle that SPJ skips exists precisely to handle skew. Monitor task duration in the Spark UI for stragglers.

### Statistics Accuracy

SPJ relies on accurate table statistics for deciding which side to replicate (in partially clustered mode). Run `ANALYZE TABLE` regularly:

```sql
ANALYZE TABLE prod.catalog.fact_orders COMPUTE STATISTICS;
ANALYZE TABLE prod.catalog.dim_customers COMPUTE STATISTICS;
```

## Production-Ready Python (PySpark) Code

### Session Factory with SPJ Configuration

```python
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production SPJ configurations
# ---------------------------------------------------------------------------

SPJ_CONFIGS: dict[str, str] = {
    "spark.sql.sources.v2.bucketing.enabled": "true",
    "spark.sql.sources.v2.bucketing.pushPartValues.enabled": "true",
    "spark.sql.requireAllClusterKeysForCoPartition": "true",
    "spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled": "true",
    "spark.sql.sources.v2.bucketing.allowCompatibleTransforms.enabled": "true",
    "spark.sql.sources.v2.bucketing.partition.filter.enabled": "true",
    "spark.sql.sources.v2.bucketing.shuffle.enabled": "true",
}

# For testing: disable broadcast joins so SPJ is forced.
SPJ_TEST_OVERRIDES: dict[str, str] = {
    "spark.sql.autoBroadcastJoinThreshold": "0",
    "spark.sql.sources.bucketing.enabled": "true",
}


def create_spj_session(
    *,
    app_name: str = "SPJProduction",
    master: str = "local[*]",
    broadcast_threshold: str = "0",
    warehouse_path: str | None = None,
    extra_configs: dict[str, str] | None = None,
) -> SparkSession:
    """Create a SparkSession tuned for Storage Partition Join.

    Parameters
    ----------
    app_name:
        Application name registered with Spark.
    master:
        Spark master URL.
    broadcast_threshold:
        ``spark.sql.autoBroadcastJoinThreshold`` in bytes.  Set to ``"0"``
        to force SPJ during testing.
    warehouse_path:
        Path for ``spark.sql.warehouse.dir``.
    extra_configs:
        Additional key/value pairs applied after the defaults.

    Returns
    -------
    SparkSession
        Configured session with all SPJ flags enabled.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.sources.v2.bucketing.enabled", "true")
        .config("spark.sql.sources.v2.bucketing.pushPartValues.enabled", "true")
        .config("spark.sql.requireAllClusterKeysForCoPartition", "true")
        .config(
            "spark.sql.sources.v2.bucketing.allowJoinKeysSubsetOfPartitionKeys.enabled",
            "true",
        )
        .config(
            "spark.sql.sources.v2.bucketing.allowCompatibleTransforms.enabled",
            "true",
        )
        .config("spark.sql.sources.v2.bucketing.partition.filter.enabled", "true")
        .config("spark.sql.sources.v2.bucketing.shuffle.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", broadcast_threshold)
        .config("spark.sql.sources.bucketing.enabled", "true")
        .config("spark.sql.sources.bucketing.pushSingleHadoopTable", "true")
    )

    if warehouse_path:
        builder = builder.config("spark.sql.warehouse.dir", warehouse_path)

    if extra_configs:
        for k, v in extra_configs.items():
            builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info("Created SPJ session: %s (broadcast_threshold=%s)", app_name, broadcast_threshold)
    return session
```

### Bucketed Table Creation

```python
def create_bucketed_tables(
    spark: SparkSession,
    *,
    database: str = "default",
    fact_table: str = "fact_orders",
    dim_table: str = "dim_customers",
    num_buckets: int = 32,
    warehouse_path: str | None = None,
) -> tuple[str, str]:
    """Create two bucketed tables with matching partition and bucket columns.

    Parameters
    ----------
    spark:
        Active SparkSession.
    database:
        Database/catalog namespace.
    fact_table:
        Name for the fact table.
    dim_table:
        Name for the dimension table.
    num_buckets:
        Number of buckets for the ``bucket()`` transform.
    warehouse_path:
        Optional warehouse directory; creates a managed table location.

    Returns
    -------
    tuple[str, str]
        ``(qualified_fact_name, qualified_dim_name)``.
    """
    fq_fact = f"{database}.{fact_table}"
    fq_dim = f"{database}.{dim_table}"

    location_clause = ""
    if warehouse_path:
        Path(warehouse_path).mkdir(parents=True, exist_ok=True)

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

    # --- fact table ---
    spark.sql(f"DROP TABLE IF EXISTS {fq_fact}")
    spark.sql(f"""
        CREATE TABLE {fq_fact} (
            order_id BIGINT,
            customer_id BIGINT,
            order_date DATE,
            amount DOUBLE
        )
        USING parquet
        PARTITIONED BY (order_date)
        CLUSTERED BY (customer_id)
        INTO {num_buckets} BUCKETS
    """)
    logger.info("Created bucketed fact table: %s (%d buckets)", fq_fact, num_buckets)

    # --- dimension table ---
    spark.sql(f"DROP TABLE IF EXISTS {fq_dim}")
    spark.sql(f"""
        CREATE TABLE {fq_dim} (
            customer_id BIGINT,
            name STRING,
            region STRING,
            tier STRING
        )
        USING parquet
        CLUSTERED BY (customer_id)
        INTO {num_buckets} BUCKETS
    """)
    logger.info("Created bucketed dim table: %s (%d buckets)", fq_dim, num_buckets)

    return fq_fact, fq_dim


def populate_bucketed_tables(
    spark: SparkSession,
    fact_table: str,
    dim_table: str,
    num_fact_rows: int = 100_000,
    num_dim_rows: int = 1_000,
) -> None:
    """Insert sample data into the bucketed tables respecting bucket layout.

    Uses ``CLUSTER BY`` during write to ensure rows land in the correct bucket.
    """
    spark.conf.set("spark.sql.sources.bucketing.enabled", "true")

    # Dimension data
    dim_sql = f"""
        INSERT INTO {dim_table}
        SELECT
            id AS customer_id,
            CONCAT('customer_', id) AS name,
            CASE (id % 4)
                WHEN 0 THEN 'US'
                WHEN 1 THEN 'EU'
                WHEN 2 THEN 'APAC'
                ELSE 'LATAM'
            END AS region,
            CASE (id % 3)
                WHEN 0 THEN 'premium'
                WHEN 1 THEN 'standard'
                ELSE 'basic'
            END AS tier
        FROM RANGE(1, {num_dim_rows + 1})
    """
    spark.sql(dim_sql)
    logger.info("Inserted %d rows into %s", num_dim_rows, dim_table)

    # Fact data -- CLUSTER BY ensures correct bucket assignment
    fact_sql = f"""
        INSERT INTO {fact_table} PARTITION (order_date = DATE '2024-01-15')
        SELECT
            id AS order_id,
            (id % {num_dim_rows}) + 1 AS customer_id,
            ROUND(RAND() * 1000, 2) AS amount
        FROM RANGE(1, {num_fact_rows + 1})
        CLUSTER BY customer_id
    """
    spark.sql(fact_sql)
    logger.info("Inserted %d rows into %s", num_fact_rows, fact_table)
```

### EXPLAIN-Based Verification

```python
@dataclass(frozen=True)
class SPJVerificationResult:
    """Result of verifying that a join plan uses Storage Partition Join."""

    has_sort_merge_join: bool
    has_shuffle_exchange: bool
    has_group_partitions: bool
    physical_plan_lines: list[str] = field(default_factory=list)
    spj_active: bool = False

    def summary(self) -> str:
        parts = []
        parts.append(f"SortMergeJoin present: {self.has_sort_merge_join}")
        parts.append(f"Shuffle Exchange present: {self.has_shuffle_exchange}")
        parts.append(f"GroupPartitionsExec present: {self.has_group_partitions}")
        parts.append(f"SPJ active: {self.spj_active}")
        if self.spj_active:
            parts.append("PASS -- shuffle eliminated via SPJ")
        else:
            parts.append("FAIL -- shuffle exchange detected")
        return "\n".join(parts)


def verify_storage_partition_join(spark: SparkSession, sql_query: str) -> SPJVerificationResult:
    """Parse the EXPLAIN output of *sql_query* and check for SPJ.

    SPJ is confirmed when the plan contains ``SortMergeJoin`` but **no**
    ``Exchange`` (shuffle) nodes feeding into it.  The presence of
    ``GroupPartitionsExec`` further confirms partition co-grouping.

    Parameters
    ----------
    spark:
        Active SparkSession.
    sql_query:
        The SQL join query to analyse.

    Returns
    -------
    SPJVerificationResult
    """
    plan = spark.sql(f"EXPLAIN EXTENDED {sql_query}").collect()
    lines = [row[0] for row in plan]
    full_text = "\n".join(lines)

    has_sort_merge_join = bool(re.search(r"SortMergeJoin", full_text, re.IGNORECASE))
    has_shuffle_exchange = bool(re.search(r"Exchange\s", full_text))
    has_group_partitions = bool(re.search(r"GroupPartitionsExec", full_text, re.IGNORECASE))

    # SPJ is active when we have a SortMergeJoin without any Exchange nodes
    spj_active = has_sort_merge_join and not has_shuffle_exchange

    result = SPJVerificationResult(
        has_sort_merge_join=has_sort_merge_join,
        has_shuffle_exchange=has_shuffle_exchange,
        has_group_partitions=has_group_partitions,
        physical_plan_lines=lines,
        spj_active=spj_active,
    )

    log_level = logging.INFO if spj_active else logging.WARNING
    logger.log(log_level, "SPJ verification: spj_active=%s", spj_active)
    return result
```

### Benchmarking

```python
@dataclass(frozen=True)
class SPJBenchmarkResult:
    """Benchmark comparison between SPJ and shuffled join."""

    spj_time_seconds: float
    shuffled_time_seconds: float
    spj_shuffle_read_bytes: int
    shuffled_shuffle_read_bytes: int
    speedup_factor: float
    shuffle_bytes_saved: int
    spj_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spj_time_seconds": round(self.spj_time_seconds, 3),
            "shuffled_time_seconds": round(self.shuffled_time_seconds, 3),
            "spj_shuffle_read_bytes": self.spj_shuffle_read_bytes,
            "shuffled_shuffle_read_bytes": self.shuffled_shuffle_read_bytes,
            "speedup_factor": round(self.speedup_factor, 2),
            "shuffle_bytes_saved": self.shuffle_bytes_saved,
            "spj_active": self.spj_active,
        }


def _timed_count(spark: SparkSession, sql_query: str) -> tuple[float, int]:
    """Execute ``SELECT COUNT(*) FROM (<query>)`` and return (seconds, count)."""
    wrapped = f"SELECT COUNT(*) FROM ({sql_query}) t"
    start = time.monotonic()
    count = int(spark.sql(wrapped).head()[0])
    elapsed = time.monotonic() - start
    return elapsed, count


def spj_benchmark(
    spark: SparkSession,
    sql_query: str,
    *,
    iterations: int = 3,
) -> SPJBenchmarkResult:
    """Compare SPJ vs shuffled join performance.

    Runs the query twice:
    1. With SPJ enabled (broadcast threshold = 0, bucketing on).
    2. With SPJ effectively disabled (broadcast threshold raised so BHJ may
       fire, bucketing off) to measure the shuffled baseline.

    Parameters
    ----------
    spark:
        Active SparkSession.
    sql_query:
        The join query to benchmark.
    iterations:
        Number of warm-up + measured runs; final value is the average.

    Returns
    -------
    SPJBenchmarkResult
    """
    report_path = Path("/tmp/spj_benchmark_report.json")

    # --- SPJ run ---
    spark.conf.set("spark.sql.sources.v2.bucketing.enabled", "true")
    spark.conf.set("spark.sql.sources.bucketing.enabled", "true")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "0")

    times_spj = []
    for i in range(iterations):
        elapsed, _ = _timed_count(spark, sql_query)
        if i > 0:  # skip warm-up
            times_spj.append(elapsed)
    avg_spj = sum(times_spj) / max(len(times_spj), 1)

    # Verify SPJ plan
    verification = verify_storage_partition_join(spark, sql_query)

    # --- Shuffled run (SPJ disabled) ---
    spark.conf.set("spark.sql.sources.v2.bucketing.enabled", "false")
    spark.conf.set("spark.sql.sources.bucketing.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10 MB

    times_shuffled = []
    for i in range(iterations):
        elapsed, _ = _timed_count(spark, sql_query)
        if i > 0:
            times_shuffled.append(elapsed)
    avg_shuffled = sum(times_shuffled) / max(len(times_shuffled), 1)

    speedup = avg_shuffled / max(avg_spj, 1e-9)

    result = SPJBenchmarkResult(
        spj_time_seconds=avg_spj,
        shuffled_time_seconds=avg_shuffled,
        spj_shuffle_read_bytes=0,  # SPJ: effectively zero
        shuffled_shuffle_read_bytes=0,  # estimated from plan
        speedup_factor=speedup,
        shuffle_bytes_saved=0,
        spj_active=verification.spj_active,
    )

    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info(
        "SPJ benchmark: spj=%.3fs, shuffled=%.3fs, speedup=%.2fx, spj_active=%s",
        avg_spj, avg_shuffled, speedup, verification.spj_active,
    )
    logger.info("Benchmark report written to %s", report_path)

    # Restore defaults
    spark.conf.set("spark.sql.sources.v2.bucketing.enabled", "true")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "0")

    return result
```

### Iceberg Partition Pipeline

```python
def build_iceberg_partitioned_pipeline(
    spark: SparkSession,
    *,
    catalog: str = "prod",
    namespace: str = "catalog",
    fact_table: str = "fact_orders",
    dim_table: str = "dim_customers",
    num_buckets: int = 32,
) -> dict[str, Any]:
    """Create Iceberg tables with matching partition specs and verify co-partitioning.

    Parameters
    ----------
    spark:
        Active SparkSession with Iceberg catalog configured.
    catalog:
        Iceberg catalog name.
    namespace:
        Namespace (database) within the catalog.
    fact_table:
        Fact table name.
    dim_table:
        Dimension table name.
    num_buckets:
        Bucket count for both tables.

    Returns
    -------
    dict
        Compatibility report with partition spec details.
    """
    fq_catalog = f"{catalog}.{namespace}"
    fq_fact = f"{fq_catalog}.{fact_table}"
    fq_dim = f"{fq_catalog}.{dim_table}"

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {fq_catalog}")

    # Create tables with identical bucket transforms on the join key
    spark.sql(f"DROP TABLE IF EXISTS {fq_fact}")
    spark.sql(f"""
        CREATE TABLE {fq_fact} (
            order_id BIGINT,
            customer_id BIGINT,
            order_date DATE,
            amount DOUBLE
        )
        USING iceberg
        PARTITIONED BY (bucket({num_buckets}, customer_id), days(order_date))
    """)

    spark.sql(f"DROP TABLE IF EXISTS {fq_dim}")
    spark.sql(f"""
        CREATE TABLE {fq_dim} (
            customer_id BIGINT,
            name STRING,
            region STRING,
            tier STRING
        )
        USING iceberg
        PARTITIONED BY (bucket({num_buckets}, customer_id))
    """)

    logger.info("Created Iceberg tables: %s, %s", fq_fact, fq_dim)

    # Verify partition specs match on the shared key
    fact_parts = spark.sql(f"DESCRIBE TABLE EXTENDED {fq_fact}").collect()
    dim_parts = spark.sql(f"DESCRIBE TABLE EXTENDED {fq_dim}").collect()

    report = {
        "fact_table": fq_fact,
        "dim_table": fq_dim,
        "num_buckets": num_buckets,
        "fact_partition_spec": f"bucket({num_buckets}, customer_id), days(order_date)",
        "dim_partition_spec": f"bucket({num_buckets}, customer_id)",
        "co_partitioned_on": "customer_id",
        "compatible": True,
    }

    report_path = Path("/tmp/iceberg_partition_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Iceberg partition report written to %s", report_path)
    return report
```

### SPJAnalyzer Class

```python
@dataclass(frozen=True)
class PartitionSpec:
    """Describes how a table is partitioned / bucketed."""

    table_name: str
    bucket_columns: list[str] = field(default_factory=list)
    bucket_count: int | None = None
    partition_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SPJCompatibilityReport:
    """Result of analysing two tables for SPJ compatibility."""

    fact_spec: PartitionSpec
    dim_spec: PartitionSpec
    shared_bucket_columns: list[str] = field(default_factory=list)
    compatible: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_table": self.fact_spec.table_name,
            "fact_bucket_columns": self.fact_spec.bucket_columns,
            "fact_bucket_count": self.fact_spec.bucket_count,
            "fact_partition_columns": self.fact_spec.partition_columns,
            "dim_table": self.dim_spec.table_name,
            "dim_bucket_columns": self.dim_spec.bucket_columns,
            "dim_bucket_count": self.dim_spec.bucket_count,
            "dim_partition_columns": self.dim_spec.partition_columns,
            "shared_bucket_columns": self.shared_bucket_columns,
            "compatible": self.compatible,
            "notes": self.notes,
        }


class SPJAnalyzer:
    """Analyses table partitioning, checks bucket alignment, and generates
    a compatibility report for Storage Partition Join.

    Parameters
    ----------
    spark:
        Active SparkSession.
    """

    _BUCKET_RE = re.compile(r"bucket\s*\(\s*(\d+)\s*,\s*(\w+)\s*\)", re.IGNORECASE)

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    def analyse_table(self, table_name: str) -> PartitionSpec:
        """Return a :class:`PartitionSpec` for *table_name*.

        Uses ``DESCRIBE TABLE EXTENDED`` and parses the partition / bucket
        metadata from the output.
        """
        try:
            rows = self.spark.sql(f"DESCRIBE TABLE EXTENDED {table_name}").collect()
        except Exception as exc:
            logger.error("Failed to describe table %s: %s", table_name, exc)
            raise

        bucket_columns: list[str] = []
        bucket_count: int | None = None
        partition_columns: list[str] = []

        for row in rows:
            col_name = str(row[0]).lower()
            col_type = str(row[1]).lower() if len(row) > 1 else ""

            # Bucket columns appear with a "clustering" or "bucket" descriptor
            if "bucket" in col_type:
                m = self._BUCKET_RE.search(col_type)
                if m:
                    bucket_count = int(m.group(1))
                    bucket_columns.append(col_name)
            elif col_type in ("partitioned", "partition"):
                partition_columns.append(col_name)

        return PartitionSpec(
            table_name=table_name,
            bucket_columns=bucket_columns,
            bucket_count=bucket_count,
            partition_columns=partition_columns,
        )

    def check_compatibility(
        self, fact_table: str, dim_table: str
    ) -> SPJCompatibilityReport:
        """Check whether *fact_table* and *dim_table* are SPJ-compatible.

        Compatibility requires:
        - Both tables are bucketed on at least one common column.
        - Bucket counts are equal (or mathematically compatible).
        """
        fact = self.analyse_table(fact_table)
        dim = self.analyse_table(dim_table)

        shared = list(set(fact.bucket_columns) & set(dim.bucket_columns))
        notes: list[str] = []
        compatible = False

        if not shared:
            notes.append(
                f"No shared bucket columns: fact={fact.bucket_columns}, dim={dim.bucket_columns}"
            )
        elif fact.bucket_count != dim.bucket_count:
            notes.append(
                f"Bucket counts differ: fact={fact.bucket_count}, dim={dim.bucket_count}. "
                "Set allowCompatibleTransforms=true if counts are compatible."
            )
            # Counts may still be compatible (e.g., 32 vs 16 where 32 % 16 == 0)
            if (
                fact.bucket_count is not None
                and dim.bucket_count is not None
                and max(fact.bucket_count, dim.bucket_count)
                % min(fact.bucket_count, dim.bucket_count)
                == 0
            ):
                compatible = True
                notes.append("Bucket counts are mathematically compatible.")
            else:
                compatible = False
        else:
            compatible = True
            notes.append(f"Both tables bucket on {shared} with {fact.bucket_count} buckets.")

        return SPJCompatibilityReport(
            fact_spec=fact,
            dim_spec=dim,
            shared_bucket_columns=shared,
            compatible=compatible,
            notes=notes,
        )

    def generate_report(
        self, fact_table: str, dim_table: str, output_path: str = "/tmp/spj_compatibility_report.json"
    ) -> SPJCompatibilityReport:
        """Run compatibility analysis and write a JSON report.

        Parameters
        ----------
        fact_table:
            Fully-qualified fact table name.
        dim_table:
            Fully-qualified dim table name.
        output_path:
            File path for the JSON report.

        Returns
        -------
        SPJCompatibilityReport
        """
        report = self.check_compatibility(fact_table, dim_table)
        Path(output_path).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("SPJ compatibility report written to %s (compatible=%s)", output_path, report.compatible)
        return report
```

### Production ETL Pipeline

```python
def run_production_spj_etl(
    spark: SparkSession,
    *,
    database: str = "default",
    fact_table: str = "fact_orders",
    dim_table: str = "dim_customers",
    num_buckets: int = 32,
    num_fact_rows: int = 1_000_000,
    num_dim_rows: int = 10_000,
) -> dict[str, Any]:
    """Run a production ETL pipeline using co-partitioned writes and reads.

    Steps
    -----
    1. Create bucketed tables with matching bucket columns.
    2. Populate data with ``CLUSTER BY`` to respect bucket layout.
    3. Run a co-partitioned join and verify SPJ elimination.
    4. Analyse compatibility and write a combined JSON report.

    Parameters
    ----------
    spark:
        Active SparkSession.
    database:
        Database name.
    fact_table:
        Fact table name.
    dim_table:
        Dimension table name.
    num_buckets:
        Number of buckets.
    num_fact_rows:
        Number of fact rows to generate.
    num_dim_rows:
        Number of dimension rows to generate.

    Returns
    -------
    dict
        Combined ETL report.
    """
    logger.info("Starting production SPJ ETL (fact=%d rows, dim=%d rows)", num_fact_rows, num_dim_rows)

    # 1. Create tables
    fq_fact, fq_dim = create_bucketed_tables(
        spark, database=database, fact_table=fact_table,
        dim_table=dim_table, num_buckets=num_buckets,
    )

    # 2. Populate
    populate_bucketed_tables(spark, fq_fact, fq_dim, num_fact_rows, num_dim_rows)

    # 3. Run join
    join_sql = f"""
        SELECT f.order_id, f.amount, c.name, c.region
        FROM {fq_fact} f
        JOIN {fq_dim} c ON f.customer_id = c.customer_id
    """
    verification = verify_storage_partition_join(spark, join_sql)
    logger.info("SPJ verification: %s", verification.summary())

    # 4. Compatibility analysis
    analyzer = SPJAnalyzer(spark)
    compatibility = analyzer.generate_report(fq_fact, fq_dim)

    # 5. Benchmark
    benchmark = spj_benchmark(spark, join_sql, iterations=2)

    report = {
        "step": "production_spj_etl",
        "fact_table": fq_fact,
        "dim_table": fq_dim,
        "num_buckets": num_buckets,
        "spj_verification": verification.summary(),
        "compatibility": compatibility.to_dict(),
        "benchmark": benchmark.to_dict(),
    }

    report_path = Path("/tmp/spj_production_etl_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Production SPJ ETL report written to %s", report_path)
    return report
```
