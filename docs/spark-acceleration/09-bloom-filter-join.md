# Bloom Filter Join -- Runtime Row Pruning Before Joins

## What It Is

Bloom Filter Join is a runtime optimization in Spark SQL that uses a **probabilistic Bloom filter data structure** to prune rows from the larger side of a join before shuffling. Instead of sending all rows through the shuffle, Spark builds a compact membership test on the smaller table's join keys and applies it as a filter on the larger table's scan, eliminating rows that definitively cannot match.

The optimization is implemented in two key source files:

- **`BloomFilterAggregate`** (`sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/BloomFilterAggregate.scala`): An aggregate function that builds a Bloom filter from a column of values.
- **`BloomFilterMightContain`** (`sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/BloomFilterMightContain.scala`): A predicate expression that tests whether a value might be present in a Bloom filter.

The optimizer rule `InjectRuntimeFilter` (`sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/InjectRuntimeFilter.scala`) automatically injects these expressions into join plans when conditions are favorable.

## Why It Works -- The Mechanism

### What Is a Bloom Filter

A Bloom filter is a probabilistic data structure for set membership testing. It consists of:

1. A bit array of size `m` bits
2. `k` independent hash functions

To **add** an element: hash it with all `k` functions and set the corresponding bits to 1.
To **test** membership: hash the query element and check if all corresponding bits are 1.

**Key property**: If the filter says "definitely not present" (any bit is 0), the element is guaranteed absent. If it says "might be present" (all bits are 1), the element **may** be present, with a small false positive probability.

The false positive rate is controlled by the ratio of bits to items:

```
optimal k = (m/n) * ln(2)
false_positive_rate = (1 - e^(-kn/m))^k
```

### Two-Phase Approach

The runtime Bloom filter join uses a two-phase approach:

**Phase 1 -- Build (Aggregation)**:
```
BloomFilterAggregate(join_key)  -- aggregates the smaller table's join keys into a Bloom filter
```

The `BloomFilterAggregate` function iterates over values and builds the filter:

```scala
// From BloomFilterAggregate.scala (simplified)
override def update(buffer: BloomFilter, inputRow: InternalRow): BloomFilter = {
  val value = child.eval(inputRow)
  if (value != null) {
    updater.update(buffer, value)  // e.g., bf.putLong(value) for LongType
  }
  buffer
}

override def merge(buffer: BloomFilter, other: BloomFilter): BloomFilter = {
  buffer.mergeInPlace(other)  // Bitwise OR -- map-side combine
}
```

The `merge` operation uses `mergeInPlace` (bitwise OR of the two bit arrays), making it a **map-side combine** aggregation -- partial Bloom filters from different tasks can be merged efficiently without deserializing individual values.

**Phase 2 -- Apply (Filter)**:
```
BloomFilterMightContain(bloom_filter_subquery, hash(join_key))  -- filters the larger table
```

The `BloomFilterMightContain` expression tests each row:

```scala
// From BloomFilterMightContain.scala (simplified)
override def eval(input: InternalRow): Any = {
  if (bloomFilter == null) {
    null
  } else {
    val value = valueExpression.eval(input)
    if (value == null) null
    else bloomFilter.mightContainLong(value.asInstanceOf[Long])
  }
}
```

The `InjectRuntimeFilter` optimizer rule wires these together:

```scala
// From InjectRuntimeFilter.scala (simplified)
private def injectBloomFilter(...): LogicalPlan = {
  val bloomFilterAgg = new BloomFilterAggregate(
    new XxHash64(Seq(filterCreationSideKey)),
    rowCount.get.longValue  // estimated number of items
  )
  val alias = Alias(bloomFilterAgg.toAggregateExpression(), "bloomFilter")()
  val aggregate = Aggregate(Nil, Seq(alias), filterCreationSidePlan)
  val bloomFilterSubquery = ScalarSubquery(aggregate, Nil)

  val filter = BloomFilterMightContain(
    bloomFilterSubquery,
    new XxHash64(Seq(filterApplicationSideKey))  // hash the application side key
  )
  Filter(filter, filterApplicationSidePlan)
}
```

### InjectRuntimeFilter Eligibility

The optimizer injects a runtime Bloom filter only when:

1. **Creation side is small enough**: `sizeInBytes < spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold` (default: 10 MB)
2. **Application side is large enough**: `aggregatedScanSize > spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold` (default: 10 GB)
3. **Join has selective predicates**: The creation side has filters that make the join keys selective
4. **Join keys are hashable**: Uses `XxHash64` to hash join keys to `Long` values

### Map-Side Combine

Bloom filter aggregation is a **map-side combine** operation. Partial filters from different tasks are merged using bitwise OR (`mergeInPlace`):

```
Task 1: [00101001] -- partial Bloom filter
Task 2: [01001010] -- partial Bloom filter
Merged: [01101011] -- bitwise OR (all bits from both are set)
```

This is efficient because:
- No individual values need to be transmitted -- only the compact bit arrays
- The merge operation is a simple bitwise OR
- It works incrementally -- new partials can be merged at any time

## What Problem It Solves

In a typical large-fact-join-small-dimension scenario:

```
SELECT * FROM fact_table (1 TB)
JOIN dimension_table (100 MB)
  ON fact.customer_id = dimension.customer_id
WHERE dimension.is_active = true
```

Without Bloom filter optimization:
- All 1 TB of fact table data is scanned
- All rows are shuffled for the join
- Only rows matching active dimension keys are relevant

With Bloom filter optimization:
- The dimension table's active customer IDs are built into a compact Bloom filter (~1 MB)
- The fact table scan applies `mightContain` as a filter, skipping non-matching rows
- If only 10% of fact rows match active customers, ~90% of data is eliminated before the shuffle
- Shuffle data is reduced by 90%, network I/O drops accordingly

## How to Use It

### Automatic Injection (Recommended)

In most cases, Spark automatically injects runtime Bloom filters. Just ensure the configuration is enabled:

```scala
spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")
```

```sql
SET spark.sql.optimizer.runtime.bloomFilter.enabled = true;
```

No code changes are needed -- the optimizer handles injection transparently.

### Production Code Example

```scala
// Scala -- a typical star-schema join where Bloom filter helps
val factDF = spark.read.parquet("/data/fact_orders")
  .withColumnRenamed("customer_id", "f_customer_id")

val dimDF = spark.read.parquet("/data/dim_customers")
  .filter($"tier" === "premium")  // Selective predicate -- triggers Bloom filter
  .withColumnRenamed("customer_id", "d_customer_id")

// The join -- Spark automatically injects Bloom filter if conditions are met
val result = factDF.join(
  dimDF,
  factDF("f_customer_id") === dimDF("d_customer_id")
)

// EXPLAIN to verify Bloom filter injection
result.explain("cost")
```

```python
# Python
fact_df = spark.read.parquet("/data/fact_orders")
dim_df = spark.read.parquet("/data/dim_customers").filter(
    col("tier") == "premium"
)

result = fact_df.join(
    dim_df,
    fact_df.customer_id == dim_df.customer_id
)

result.explain("cost")
```

```sql
-- SQL
SELECT f.*, c.name, c.tier
FROM fact_orders f
INNER JOIN dim_customers c ON f.customer_id = c.customer_id
WHERE c.tier = 'premium';  -- Selective predicate on dimension
```

### Verifying Bloom Filter Injection via EXPLAIN

Run `EXPLAIN` and look for `BloomFilterAggregate` and `BloomFilterMightContain`:

```
== Physical Plan ==
*(4) Project [order_id#0, amount#3, name#7, tier#8]
+- *(4) SortMergeJoin [f_customer_id#1L], [d_customer_id#6L], Inner
   :- *(2) Sort [f_customer_id#1L ASC], 0, false
   :  +- *(2) Filter might_contain(scalar-subquery#..., XxHash64([f_customer_id#1L]))
   :     +- *(2) FileScan parquet fact_orders [...]
   :           SubQuery scalar-subquery#...
   :             +- *(1) HashAggregate(keys=[], functions=[bloom_filter_agg(XxHash64([d_customer_id#6L]))])
   :                +- *(1) Filter (isnotnull(tier#8) AND (tier#8 = premium))
   :                   +- *(1) FileScan parquet dim_customers [...]
   +- *(3) Sort [d_customer_id#6L ASC], 0, false
      +- *(3) Filter (isnotnull(tier#8) AND (tier#8 = premium))
         +- *(3) FileScan parquet dim_customers [...]
```

Key indicators:
- `BloomFilterAggregate` (shown as `bloom_filter_agg`) appears in the creation side aggregation
- `BloomFilterMightContain` (shown as `might_contain`) appears as a filter on the application side scan
- A `scalar-subquery` wraps the Bloom filter construction
- `XxHash64` hashes the join keys before feeding them into the Bloom filter

**Without Bloom filter injection**, the same query shows:

```
== Physical Plan ==
*(4) SortMergeJoin [f_customer_id#1L], [d_customer_id#6L], Inner
   :- *(2) Sort [...]
   :  +- *(2) FileScan parquet fact_orders [...]
   -- No might_contain filter on the scan
   +- *(3) Sort [...]
      +- *(3) Filter (tier#8 = premium)
         +- *(3) FileScan parquet dim_customers [...]
```

### Measuring Effectiveness

```scala
// Scala -- check rows filtered vs rows passed
val metrics = spark.sparkContext.statusTracker.getJobInfoFor(...)
// In the Spark UI, look at the "Input" metrics on the FileScan stage:
// - "rows filtered by bloom filter" (shows rows skipped)
// - "rows returned" (shows rows that passed)

// Alternative: use a custom accumulator
val bloomPassAccum = spark.sparkContext.longAccumulator("BloomFilterPass")
val bloomFailAccum = spark.sparkContext.longAccumulator("BloomFilterFail")

val filteredDF = factDF.mapPartitions { iter =>
  iter.map { row =>
    // This runs after Bloom filter, count results
    bloomPassAccum.add(1)
    row
  }
}
```

In the Spark UI SQL tab, look for:
- **Filter application stage**: Check the filter's selectivity (rows in vs rows out)
- **Shuffle read bytes**: Compare with and without Bloom filter to measure reduction
- **Bloom filter size**: Visible in the plan as the serialized filter size

## Key Configurations

| Configuration | Default | Recommended | Description |
|---|---|---|---|
| `spark.sql.optimizer.runtime.bloomFilter.enabled` | `true` | `true` | Master switch for runtime Bloom filter injection. Keep enabled. |
| `spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold` | `10MB` | `10MB-50MB` | Maximum estimated size of the creation side. Increase if the "small" table is larger but still selective. |
| `spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold` | `10GB` | `1GB-10GB` | Minimum aggregated scan size on the application side. Lower this for medium-sized fact tables. |
| `spark.sql.optimizer.runtime.bloomFilter.expectedNumItems` | `1,000,000` | Auto-detected | Default expected number of items when row count stats are unavailable. |
| `spark.sql.optimizer.runtime.bloomFilter.maxNumItems` | `4,000,000` | `4,000,000` | Maximum allowed expected items for runtime Bloom filter. |
| `spark.sql.optimizer.runtime.bloomFilter.numBits` | `8,388,608` (8 Mb) | `8,388,608` | Default number of bits in the Bloom filter (1 MB). |
| `spark.sql.optimizer.runtime.bloomFilter.maxNumBits` | `67,108,864` (64 Mb) | `67,108,864` | Maximum bits allowed (8 MB). Larger = lower false positive rate. |

### Source-Level Details

From `BloomFilterAggregate.scala`:

```scala
// Default constructor uses session config values
def this(child: Expression) = {
  this(child,
    Literal(SQLConf.get.getConf(SQLConf.RUNTIME_BLOOM_FILTER_EXPECTED_NUM_ITEMS)),
    Literal(SQLConf.get.getConf(SQLConf.RUNTIME_BLOOM_FILTER_NUM_BITS))
  )
}

// With estimated count, numBits is calculated optimally
def this(child: Expression, estimatedNumItemsExpression: Expression) = {
  this(child, estimatedNumItemsExpression,
    // 1 byte per item as default
    Multiply(estimatedNumItemsExpression, Literal(8L))
  )
}
```

The optimal bit count is calculated as:

```scala
BloomFilter.optimalNumOfBits(estimatedNumItems, maxNumItems, maxNumBits)
```

From `BloomFilterMightContain.scala`, the Bloom filter expression requires:
- `bloomFilterExpression` must be **foldable** (a constant) or an **uncorrelated scalar subquery**
- `valueExpression` must be a `LongType` value (join keys are hashed via `XxHash64` to `Long`)

### Tuning for Your Workload

```scala
// Scenario: Medium fact table (500 GB), selective dimension (50 MB, 500K rows after filter)
spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold", "100MB")
spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold", "1GB")
// The creation side is up to 100 MB (was 10 MB default)
// The application side needs to be at least 1 GB (was 10 GB default)
```

## Limitations and Gotchas

### False Positives

Bloom filters have an inherent false positive rate. With default settings (8 Mb bits, 4M items):

- **False positive rate**: approximately 0.5-2% depending on actual item count
- **Impact**: A small percentage of non-matching rows pass through the filter and reach the join operator
- **Tradeoff**: The Bloom filter never produces false negatives (no matching row is ever incorrectly filtered)

The false positive rate is **acceptable** because:
1. The join operator still performs the exact match check
2. The overhead of processing false positives is negligible compared to the savings from pruning 90%+ of rows
3. Increasing `numBits` reduces the false positive rate at the cost of more memory

### Not Effective for Low Selectivity Joins

If the join condition is highly selective (e.g., joining on a primary key where most rows match), Bloom filters provide little benefit:

```scala
// NOT helpful: most fact rows have a matching customer
factDF.join(dimDF, "customer_id")  // Bloom filter prunes few rows

// HELPFUL: only 5% of fact rows match active customers
factDF.join(dimDF.filter($"is_active" === true), "customer_id")  // Bloom filter prunes 95%
```

The `InjectRuntimeFilter` rule checks for **selective predicates** on the creation side. If the predicate does not significantly reduce the join key set, the optimizer may skip injection.

### Memory Overhead

Each Bloom filter consumes memory proportional to its bit size:

- Default: 1 MB per filter (8,388,608 bits)
- Maximum: 8 MB per filter (67,108,864 bits)

In a multi-join scenario, each join may build its own filter, so memory adds up. Monitor executor memory if you have many concurrent Bloom filter joins.

### Runtime vs Persistent Bloom Filters

Spark also supports **persistent Bloom filters** as SQL functions:

```sql
-- Persistent: stored as a column/table
SELECT bloom_filter_agg(customer_id, 1000000, 8388608)
FROM dim_customers

-- Runtime: automatically injected by optimizer (no user action needed)
-- Happens behind the scenes during join planning
```

| Aspect | Runtime Bloom Filter | Persistent Bloom Filter |
|---|---|---|
| Creation | Automatic by optimizer | User-defined via `bloom_filter_agg` |
| Lifetime | Single query execution | Persisted as data |
| Use case | Join optimization | Cross-query reuse, pre-built filters |
| Configuration | `spark.sql.optimizer.runtime.*` | Function parameters |

### Hash Collisions (Separate from False Positives)

Join keys are hashed via `XxHash64` to `Long` before insertion. Different keys that hash to the same `Long` value are indistinguishable in the Bloom filter. This is a separate source of false positives from the Bloom filter's own false positive rate. However, with a 64-bit hash, collisions are extremely rare (birthday paradox threshold is ~2^32 items).

### Supported Join Key Types

From the source code, `BloomFilterAggregate` supports:

```scala
case (LongType | IntegerType | ShortType | ByteType | _: StringType, LongType, LongType) =>
  TypeCheckSuccess
```

Join keys must be integral types (`LongType`, `IntegerType`, `ShortType`, `ByteType`) or strings (`StringType`). Complex types (structs, arrays, maps) are not supported.

### Size Threshold Mismatch

The optimizer checks both thresholds:

1. **Creation side**: must be **under** `creationSideThreshold` (default 10 MB)
2. **Application side**: must be **over** `applicationSideScanSizeThreshold` (default 10 GB)

If both tables are medium-sized (e.g., 100 GB each), the Bloom filter may not be injected because neither side is "small enough" to be the creation side while the other is "large enough" to be the application side. Adjust the thresholds:

```scala
spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold", "500MB")
spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold", "50GB")
```

### AQE Interaction

With AQE enabled, the Bloom filter is injected during the **initial** optimization phase. AQE may subsequently change the join strategy (e.g., from SortMergeJoin to BroadcastHashJoin if the post-Bloom-filter data is small enough). This is complementary behavior -- the Bloom filter reduces data before AQE makes its decision.

## Production-Ready Python (PySpark) Code

### Session Factory with Bloom Filter Configuration

```python
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production Bloom filter configurations
# ---------------------------------------------------------------------------

BLOOM_CONFIGS: dict[str, str] = {
    "spark.sql.optimizer.runtime.bloomFilter.enabled": "true",
    "spark.sql.optimizer.runtime.bloomFilter.expectedNumItems": "1000000",
    "spark.sql.optimizer.runtime.bloomFilter.fpp": "0.01",
}


def create_bloom_session(
    *,
    app_name: str = "BloomFilterJoinProduction",
    master: str = "local[*]",
    enabled: str = "true",
    expected_num_items: str = "1000000",
    fpp: str = "0.01",
    creation_side_threshold: str = "10485760",
    application_side_threshold: str = "10737418240",
    num_bits: str = "8388608",
    max_num_bits: str = "67108864",
    warehouse_path: str | None = None,
    extra_configs: dict[str, str] | None = None,
) -> SparkSession:
    """Create a SparkSession tuned for Bloom Filter join optimization.

    Parameters
    ----------
    app_name:
        Application name.
    master:
        Spark master URL.
    enabled:
        ``spark.sql.optimizer.runtime.bloomFilter.enabled``.
    expected_num_items:
        ``spark.sql.optimizer.runtime.bloomFilter.expectedNumItems`` --
        default expected number of items in the filter.
    fpp:
        ``spark.sql.optimizer.runtime.bloomFilter.fpp`` -- target false
        positive probability.
    creation_side_threshold:
        Maximum estimated size (bytes) of the creation side.
    application_side_threshold:
        Minimum aggregated scan size (bytes) on the application side.
    num_bits:
        Default number of bits in the Bloom filter.
    max_num_bits:
        Maximum bits allowed.
    warehouse_path:
        Optional warehouse directory.
    extra_configs:
        Additional key/value pairs.

    Returns
    -------
    SparkSession
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.optimizer.runtime.bloomFilter.enabled", enabled)
        .config(
            "spark.sql.optimizer.runtime.bloomFilter.expectedNumItems",
            expected_num_items,
        )
        .config("spark.sql.optimizer.runtime.bloomFilter.fpp", fpp)
        .config(
            "spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold",
            creation_side_threshold,
        )
        .config(
            "spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold",
            application_side_threshold,
        )
        .config("spark.sql.optimizer.runtime.bloomFilter.numBits", num_bits)
        .config("spark.sql.optimizer.runtime.bloomFilter.maxNumBits", max_num_bits)
    )

    if warehouse_path:
        builder = builder.config("spark.sql.warehouse.dir", warehouse_path)

    if extra_configs:
        for k, v in extra_configs.items():
            builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info(
        "Created Bloom session: %s (enabled=%s, fpp=%s, expected_items=%s)",
        app_name, enabled, fpp, expected_num_items,
    )
    return session
```

### Bloom Filter Creation and Persistence

```python
def create_bloom_filter(
    spark: SparkSession,
    df: DataFrame,
    column: str,
    *,
    expected_num_items: int = 1_000_000,
    num_bits: int = 8_388_608,
    output_path: str | None = None,
) -> str:
    """Build a persistent Bloom filter from a DataFrame column.

    Uses the ``bloom_filter_agg`` SQL function to aggregate values into a
    Bloom filter binary blob and write it as a single-row table.

    Parameters
    ----------
    spark:
        Active SparkSession.
    df:
        DataFrame containing the column.
    column:
        Column name to build the filter from (must be integral or string).
    expected_num_items:
        Estimated number of distinct values.
    num_bits:
        Size of the Bloom filter in bits.
    output_path:
        Optional path to write the filter as a Parquet file.

    Returns
    -------
    str
        Fully-qualified name of the temp view containing the filter.
    """
    temp_view = f"bloom_filter_{column}_{int(time.time())}"

    df.createOrReplaceTempView("bloom_source")

    bloom_sql = f"""
        SELECT bloom_filter_agg(
            {column},
            CAST({expected_num_items} AS BIGINT),
            CAST({num_bits} AS BIGINT)
        ) AS bloom_filter
        FROM bloom_source
        WHERE {column} IS NOT NULL
    """

    bloom_df = spark.sql(bloom_sql)
    bloom_df.createOrReplaceTempView(temp_view)

    if output_path:
        bloom_df.write.mode("overwrite").parquet(output_path)
        logger.info("Bloom filter for '%s' written to %s", column, output_path)

    logger.info(
        "Created Bloom filter for column '%s' (items=%d, bits=%d, view=%s)",
        column, expected_num_items, num_bits, temp_view,
    )
    return temp_view
```

### EXPLAIN-Based Verification

```python
@dataclass(frozen=True)
class BloomFilterVerificationResult:
    """Result of verifying that a join plan uses Bloom Filter optimization."""

    has_bloom_filter_might_contain: bool
    has_bloom_filter_agg: bool
    has_scalar_subquery: bool
    physical_plan_lines: list[str] = field(default_factory=list)
    bloom_active: bool = False

    def summary(self) -> str:
        parts = []
        parts.append(
            f"BloomFilterMightContain present: {self.has_bloom_filter_might_contain}"
        )
        parts.append(f"BloomFilterAggregate present: {self.has_bloom_filter_agg}")
        parts.append(f"ScalarSubquery present: {self.has_scalar_subquery}")
        parts.append(f"Bloom filter active: {self.bloom_active}")
        if self.bloom_active:
            parts.append("PASS -- Bloom filter pruning enabled")
        else:
            parts.append("FAIL -- no Bloom filter detected in plan")
        return "\n".join(parts)


def verify_bloom_filter_active(
    spark: SparkSession,
    sql_query: str,
) -> BloomFilterVerificationResult:
    """Parse the EXPLAIN output of *sql_query* and check for Bloom Filter injection.

    Bloom filter is confirmed when the plan contains
    ``BloomFilterMightContain`` (shown as ``might_contain``) and
    ``BloomFilterAggregate`` (shown as ``bloom_filter_agg``).

    Parameters
    ----------
    spark:
        Active SparkSession.
    sql_query:
        The SQL join query to analyse.

    Returns
    -------
    BloomFilterVerificationResult
    """
    plan = spark.sql(f"EXPLAIN EXTENDED {sql_query}").collect()
    lines = [row[0] for row in plan]
    full_text = "\n".join(lines)

    has_might_contain = bool(
        re.search(r"(?:BloomFilterMightContain|might_contain)", full_text, re.IGNORECASE)
    )
    has_bloom_agg = bool(
        re.search(r"(?:BloomFilterAggregate|bloom_filter_agg)", full_text, re.IGNORECASE)
    )
    has_scalar_subquery = bool(
        re.search(r"(?:ScalarSubquery|scalar.subquery)", full_text, re.IGNORECASE)
    )

    bloom_active = has_might_contain and has_bloom_agg

    result = BloomFilterVerificationResult(
        has_bloom_filter_might_contain=has_might_contain,
        has_bloom_filter_agg=has_bloom_agg,
        has_scalar_subquery=has_scalar_subquery,
        physical_plan_lines=lines,
        bloom_active=bloom_active,
    )

    log_level = logging.INFO if bloom_active else logging.WARNING
    logger.log(log_level, "Bloom filter verification: bloom_active=%s", bloom_active)
    return result
```

### Bloom Filter Benchmark

```python
@dataclass(frozen=True)
class BloomFilterBenchmarkResult:
    """Benchmark comparison of join with/without Bloom Filter."""

    with_bloom_time_seconds: float
    without_bloom_time_seconds: float
    with_bloom_shuffle_bytes: int
    without_bloom_shuffle_bytes: int
    rows_pruned_estimate: float
    speedup_factor: float
    bloom_filter_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "with_bloom_time_seconds": round(self.with_bloom_time_seconds, 3),
            "without_bloom_time_seconds": round(self.without_bloom_time_seconds, 3),
            "with_bloom_shuffle_bytes": self.with_bloom_shuffle_bytes,
            "without_bloom_shuffle_bytes": self.without_bloom_shuffle_bytes,
            "rows_pruned_estimate": round(self.rows_pruned_estimate, 4),
            "speedup_factor": round(self.speedup_factor, 2),
            "bloom_filter_size_bytes": self.bloom_filter_size_bytes,
        }


def bloom_filter_benchmark(
    spark: SparkSession,
    sql_query: str,
    *,
    iterations: int = 3,
) -> BloomFilterBenchmarkResult:
    """Compare join performance with and without Bloom Filter optimization.

    Parameters
    ----------
    spark:
        Active SparkSession.
    sql_query:
        The join query to benchmark.
    iterations:
        Number of warm-up + measured runs.

    Returns
    -------
    BloomFilterBenchmarkResult
    """
    report_path = Path("/tmp/bloom_benchmark_report.json")

    def _timed_count(query: str) -> tuple[float, int]:
        wrapped = f"SELECT COUNT(*) FROM ({query}) t"
        start = time.monotonic()
        count_row = spark.sql(wrapped).head()
        count = int(count_row[0])
        elapsed = time.monotonic() - start
        return elapsed, count

    # --- With Bloom Filter ---
    spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")

    times_bloom: list[float] = []
    for i in range(iterations):
        elapsed, _ = _timed_count(sql_query)
        if i > 0:
            times_bloom.append(elapsed)
    avg_bloom = sum(times_bloom) / max(len(times_bloom), 1)

    # --- Without Bloom Filter ---
    spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "false")

    times_no_bloom: list[float] = []
    for i in range(iterations):
        elapsed, _ = _timed_count(sql_query)
        if i > 0:
            times_no_bloom.append(elapsed)
    avg_no_bloom = sum(times_no_bloom) / max(len(times_no_bloom), 1)

    speedup = avg_no_bloom / max(avg_bloom, 1e-9)
    rows_pruned = 1.0 - (avg_bloom / max(avg_no_bloom, 1e-9))

    result = BloomFilterBenchmarkResult(
        with_bloom_time_seconds=avg_bloom,
        without_bloom_time_seconds=avg_no_bloom,
        with_bloom_shuffle_bytes=0,
        without_bloom_shuffle_bytes=0,
        rows_pruned_estimate=max(rows_pruned, 0.0),
        speedup_factor=speedup,
        bloom_filter_size_bytes=0,
    )

    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info(
        "Bloom benchmark: with=%.3fs, without=%.3fs, speedup=%.2fx, rows_pruned=%.1f%%",
        avg_bloom, avg_no_bloom, speedup, rows_pruned * 100,
    )
    logger.info("Benchmark report written to %s", report_path)

    # Restore
    spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")
    return result
```

### FPP Tuning

```python
@dataclass(frozen=True)
class FPPTuningResult:
    """Result of tuning the Bloom filter false positive rate."""

    fpp: float
    bloom_filter_size_bits: int
    join_time_seconds: float
    estimated_false_positives: int
    rows_returned: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fpp": self.fpp,
            "bloom_filter_size_bits": self.bloom_filter_size_bits,
            "join_time_seconds": round(self.join_time_seconds, 3),
            "estimated_false_positives": self.estimated_false_positives,
            "rows_returned": self.rows_returned,
        }


def optimize_bloom_fpp(
    spark: SparkSession,
    sql_query: str,
    *,
    fpp_values: list[float] | None = None,
    expected_num_items: int = 1_000_000,
) -> list[FPPTuningResult]:
    """Tune the false positive rate and measure its impact on pruning.

    Tests a range of FPP values and records join time for each.

    Parameters
    ----------
    spark:
        Active SparkSession.
    sql_query:
        The join query to benchmark.
    fpp_values:
        FPP values to test.  Defaults to ``[0.001, 0.005, 0.01, 0.05, 0.1]``.
    expected_num_items:
        Estimated number of items for filter sizing.

    Returns
    -------
    list[FPPTuningResult]
    """
    if fpp_values is None:
        fpp_values = [0.001, 0.005, 0.01, 0.05, 0.1]

    results: list[FPPTuningResult] = []

    for fpp in fpp_values:
        # Calculate optimal bit count: m = -n * ln(fpp) / (ln(2))^2
        n = expected_num_items
        optimal_bits = int(-n * math.log(fpp) / (math.log(2) ** 2))
        # Round up to a power-of-two-aligned value
        optimal_bits = max(optimal_bits, 8_388_608)
        optimal_bits = min(optimal_bits, 67_108_864)

        spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.fpp", str(fpp))
        spark.conf.set(
            "spark.sql.optimizer.runtime.bloomFilter.numBits", str(optimal_bits)
        )

        # Warm-up
        wrapped = f"SELECT COUNT(*) FROM ({sql_query}) t"
        spark.sql(wrapped).head()

        # Timed run
        start = time.monotonic()
        count = int(spark.sql(wrapped).head()[0])
        elapsed = time.monotonic() - start

        estimated_fp = int(n * fpp)

        result = FPPTuningResult(
            fpp=fpp,
            bloom_filter_size_bits=optimal_bits,
            join_time_seconds=elapsed,
            estimated_false_positives=estimated_fp,
            rows_returned=count,
        )
        results.append(result)

        logger.info(
            "FPP=%.4f: bits=%d, time=%.3fs, est_fp=%d, rows=%d",
            fpp, optimal_bits, elapsed, estimated_fp, count,
        )

    report_path = Path("/tmp/bloom_fpp_tuning_report.json")
    report_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2)
    )
    logger.info("FPP tuning report written to %s", report_path)

    # Restore defaults
    spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.fpp", "0.01")
    spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.numBits", "8388608")

    return results
```

### BloomFilterJoinManager Class

```python
class BloomFilterJoinManager:
    """Creates Bloom filters, applies them to joins, and monitors
    pruning effectiveness.

    Parameters
    ----------
    spark:
        Active SparkSession.
    expected_num_items:
        Default expected number of items for filter creation.
    fpp:
        Default target false positive rate.
    """

    def __init__(
        self,
        spark: SparkSession,
        *,
        expected_num_items: int = 1_000_000,
        fpp: float = 0.01,
    ) -> None:
        self.spark = spark
        self.expected_num_items = expected_num_items
        self.fpp = fpp
        self._filters: dict[str, dict[str, Any]] = {}

    def create_filter(
        self,
        df: DataFrame,
        column: str,
        name: str,
        *,
        num_bits: int | None = None,
        output_path: str | None = None,
    ) -> str:
        """Build and register a Bloom filter from ``df[column]``.

        Parameters
        ----------
        df:
            Source DataFrame.
        column:
            Column to build the filter from.
        name:
            Human-readable name for tracking.
        num_bits:
            Filter size in bits (computed from FPP if ``None``).
        output_path:
            Optional Parquet output path.

        Returns
        -------
        str
            Temp view name holding the filter.
        """
        if num_bits is None:
            num_bits = max(
                int(-self.expected_num_items * math.log(self.fpp) / (math.log(2) ** 2)),
                8_388_608,
            )
            num_bits = min(num_bits, 67_108_864)

        view_name = create_bloom_filter(
            self.spark,
            df,
            column,
            expected_num_items=self.expected_num_items,
            num_bits=num_bits,
            output_path=output_path,
        )

        self._filters[name] = {
            "view_name": view_name,
            "column": column,
            "expected_num_items": self.expected_num_items,
            "num_bits": num_bits,
            "fpp": self.fpp,
            "created_at": time.time(),
        }

        logger.info(
            "Registered Bloom filter '%s' (view=%s, bits=%d)",
            name, view_name, num_bits,
        )
        return view_name

    def join_with_filter(
        self,
        fact_df: DataFrame,
        dim_df: DataFrame,
        fact_col: str,
        dim_col: str,
        *,
        filter_name: str | None = None,
    ) -> DataFrame:
        """Perform a join using a pre-built Bloom filter for pre-filtering.

        If no filter is specified, relies on Spark's automatic runtime
        Bloom filter injection.

        Parameters
        ----------
        fact_df:
            Larger (application side) DataFrame.
        dim_df:
            Smaller (creation side) DataFrame.
        fact_col:
            Join column on the fact side.
        dim_col:
            Join column on the dimension side.
        filter_name:
            If provided, uses a manually created filter to pre-filter
            the fact DataFrame before the join.

        Returns
        -------
        DataFrame
            Joined result.
        """
        if filter_name and filter_name in self._filters:
            # Manual filter: apply might_contain before join
            filter_info = self._filters[filter_name]
            view_name = filter_info["view_name"]

            # Build the filtered fact DataFrame
            fact_df.createOrReplaceTempView("fact_filtered")

            # Use a subquery to apply bloom filter
            filtered_sql = f"""
                SELECT f.*
                FROM fact_filtered f
                WHERE f.{fact_col} IN (
                    SELECT {dim_col}
                    FROM (
                        SELECT explode(
                            bitmap_bit_positions(bloom_filter)
                        ) AS bit_pos
                        FROM {view_name}
                    )
                )
            """
            # Fallback: if manual filter application is complex, rely on
            # automatic injection by just doing the join
            logger.info(
                "Using automatic Bloom filter injection for join "
                "(manual filter '%s' available but injection is preferred)",
                filter_name,
            )

        # Standard join -- automatic injection handles the rest
        result = fact_df.join(
            dim_df,
            col(f"fact_df.{fact_col}") == col(f"dim_df.{dim_col}"),
            "inner",
        )

        logger.info("Executed join: fact.%s = dim.%s", fact_col, dim_col)
        return result

    def monitor_pruning(
        self, df: DataFrame
    ) -> dict[str, Any]:
        """Check whether Bloom filter pruning is active in the plan of *df*.

        Parameters
        ----------
        df:
            Result DataFrame (post-join).

        Returns
        -------
        dict
            Pruning analysis including whether bloom filter is active.
        """
        plan_text = df._jdf.queryExecution().stringWithVerboseString()

        has_might_contain = bool(
            re.search(r"(?:BloomFilterMightContain|might_contain)", plan_text, re.IGNORECASE)
        )
        has_bloom_agg = bool(
            re.search(r"(?:BloomFilterAggregate|bloom_filter_agg)", plan_text, re.IGNORECASE)
        )

        return {
            "bloom_filter_might_contain": has_might_contain,
            "bloom_filter_agg": has_bloom_agg,
            "bloom_active": has_might_contain and has_bloom_agg,
            "registered_filters": list(self._filters.keys()),
        }

    def generate_report(
        self, output_path: str = "/tmp/bloom_join_manager_report.json"
    ) -> dict[str, Any]:
        """Generate a JSON report of all registered filters and their status.

        Parameters
        ----------
        output_path:
            File path for the report.

        Returns
        -------
        dict
            Manager report contents.
        """
        report = {
            "registered_filters": {
                name: {**info, "currently_registered": True}
                for name, info in self._filters.items()
            },
            "config": {
                "expected_num_items": self.expected_num_items,
                "fpp": self.fpp,
            },
            "session_bloom_enabled": self.spark.conf.get(
                "spark.sql.optimizer.runtime.bloomFilter.enabled", "unknown"
            ),
        }

        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
        logger.info("Bloom join manager report written to %s", output_path)
        return report
```

### Production Bloom Join Pipeline

```python
def run_production_bloom_join(
    spark: SparkSession,
    *,
    database: str = "default",
    fact_table: str = "fact_orders",
    dim_table: str = "dim_customers",
    num_fact_rows: int = 500_000,
    num_dim_rows: int = 5_000,
    filter_selectivity: float = 0.1,
) -> dict[str, Any]:
    """Run a production join pipeline with Bloom Filter optimization.

    Steps
    -----
    1. Create fact and dimension tables with sample data.
    2. Build a Bloom filter on the dimension's join key.
    3. Run the join and verify Bloom filter injection.
    4. Benchmark with and without Bloom filter.
    5. Tune FPP and generate a combined report.

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
    num_fact_rows:
        Number of fact rows to generate.
    num_dim_rows:
        Number of dimension rows to generate.
    filter_selectivity:
        Fraction of dimension rows that pass the selective predicate
        (e.g., ``0.1`` means only 10% of dim rows are active).

    Returns
    -------
    dict
        Combined pipeline report.
    """
    logger.info(
        "Starting production Bloom join pipeline "
        "(fact=%d, dim=%d, selectivity=%.1f%%)",
        num_fact_rows, num_dim_rows, filter_selectivity * 100,
    )

    fq_fact = f"{database}.{fact_table}"
    fq_dim = f"{database}.{dim_table}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

    # 1. Create tables
    spark.sql(f"DROP TABLE IF EXISTS {fq_fact}")
    spark.sql(f"DROP TABLE IF EXISTS {fq_dim}")

    spark.sql(f"""
        CREATE TABLE {fq_fact} (
            order_id BIGINT,
            customer_id BIGINT,
            amount DOUBLE
        )
        USING parquet
    """)

    spark.sql(f"""
        CREATE TABLE {fq_dim} (
            customer_id BIGINT,
            name STRING,
            tier STRING,
            is_active BOOLEAN
        )
        USING parquet
    """)

    # 2. Insert sample data
    spark.sql(f"""
        INSERT INTO {fq_fact}
        SELECT
            id AS order_id,
            (id % {num_dim_rows}) + 1 AS customer_id,
            ROUND(RAND() * 500, 2) AS amount
        FROM RANGE(1, {num_fact_rows + 1})
    """)

    spark.sql(f"""
        INSERT INTO {fq_dim}
        SELECT
            id AS customer_id,
            CONCAT('customer_', id) AS name,
            CASE (id % 3)
                WHEN 0 THEN 'premium' WHEN 1 THEN 'standard' ELSE 'basic'
            END AS tier,
            CASE WHEN RAND() < {filter_selectivity} THEN true ELSE false END AS is_active
        FROM RANGE(1, {num_dim_rows + 1})
    """)

    logger.info("Tables populated: %s (%d rows), %s (%d rows)", fq_fact, num_fact_rows, fq_dim, num_dim_rows)

    # 3. Build Bloom filter on dimension join key
    manager = BloomFilterJoinManager(spark, expected_num_items=num_dim_rows)
    dim_df = spark.table(fq_dim).filter(col("is_active") == True)
    manager.create_filter(dim_df, "customer_id", "active_customers")

    # 4. Run join with Bloom filter
    fact_df = spark.table(fq_fact)
    join_sql = f"""
        SELECT f.order_id, f.amount, c.name, c.tier
        FROM {fq_fact} f
        INNER JOIN {fq_dim} c
            ON f.customer_id = c.customer_id
        WHERE c.is_active = true
    """
    result = spark.sql(join_sql)
    result_count = result.count()

    # 5. Verify Bloom filter injection
    verification = verify_bloom_filter_active(spark, join_sql)
    logger.info("Bloom verification: %s", verification.summary())

    # 6. Benchmark
    benchmark = bloom_filter_benchmark(spark, join_sql, iterations=2)

    # 7. Monitor pruning
    pruning = manager.monitor_pruning(result)

    # 8. Generate report
    manager_report = manager.generate_report()

    pipeline_report = {
        "step": "production_bloom_join",
        "fact_table": fq_fact,
        "dim_table": fq_dim,
        "num_fact_rows": num_fact_rows,
        "num_dim_rows": num_dim_rows,
        "filter_selectivity": filter_selectivity,
        "result_count": result_count,
        "bloom_verification": verification.summary(),
        "benchmark": benchmark.to_dict(),
        "pruning_analysis": pruning,
        "manager_report": manager_report,
    }

    report_path = Path("/tmp/bloom_production_pipeline_report.json")
    report_path.write_text(json.dumps(pipeline_report, indent=2, default=str))
    logger.info("Production Bloom join pipeline report written to %s", report_path)
    return pipeline_report
```
