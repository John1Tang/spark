# DataFrame Caching -- In-Memory Columnar Storage for Repeated Computations

## What It Is

DataFrame caching in Spark SQL stores the results of a computed DataFrame in memory using an **in-memory columnar format**. Once cached, subsequent actions on the same DataFrame read from memory instead of recomputing the entire lineage. The cache is managed by the `CacheManager` class and is automatically substituted into query plans during the analysis phase via `useCachedData`.

Caching is most effective when:
- The same DataFrame is used in **multiple downstream queries** (e.g., a pre-computed feature set used by several ML models)
- The DataFrame is **expensive to compute** (joins, aggregations, window functions)
- The DataFrame fits within available executor memory

## Why It Works -- The Mechanism

### CacheManager and CachedData

The `CacheManager` maintains a list of `CachedData` entries, each holding:

```scala
case class CachedData(
    plan: LogicalPlan,                // Normalized logical plan for matching
    cachedRepresentation: InMemoryRelation  // The cached data wrapper
)
```

When you call `df.cache()` followed by an action, Spark:

1. Computes the DataFrame and stores the result as an `InMemoryRelation`
2. Registers a `CachedData` entry in the `CacheManager`

### Plan Substitution via useCachedData

During the analysis phase of **subsequent queries**, `SparkSession.sessionState.useCachedData` traverses the logical plan and substitutes matching sub-plans with `InMemoryRelation`:

```scala
// CacheManager.useCachedData (simplified)
def useCachedData(plan: LogicalPlan): LogicalPlan = plan match {
  case p if lookupCachedData(p).isDefined =>
    // Replace matching sub-plan with InMemoryRelation
    cachedRepr.withOutput(p.output)
  case p =>
    // Recurse into children
    p.mapChildren(useCachedData)
}
```

The plan matching uses `sameResult` comparison on **normalized** logical plans (canonicalized expressions, resolved attributes). This means a query written differently but logically equivalent will still hit the cache.

### InMemoryRelation and InMemoryTableScanExec

The cached data is represented as:

```
InMemoryRelation
  +- InMemoryTableScanExec [col1, col2, ...]
     +- (original computed RDD)
```

`InMemoryTableScanExec` reads the columnar data from `CachedBatch` objects stored in the BlockManager. The columnar format enables:

- **Column pruning**: Only requested columns are deserialized
- **Predicate pushdown**: Filters are applied directly on cached batches
- **Compression**: Automatic codec selection per column

### In-Memory Columnar Format

Spark uses `InMemoryColumnVector` to store data in a columnar layout. The format automatically selects compression codecs based on column statistics:

- **Integer columns with small range**: Run-length encoding (RLE)
- **String columns with repeated values**: Dictionary encoding
- **General numeric**: Bit-packed encoding
- **Low-cardinality columns**: High compression ratios

The `CachedBatch` serializer (`InMemoryColumnarStorage`) processes data in configurable batches (default: 10,000 rows per batch).

### Interaction with AQE

Adaptive Query Execution (AQE) and caching have an important interaction:

- **AQE cannot change shuffle partitions for cached queries**. Once data is cached with a certain partition count, that partitioning is fixed.
- If you cache before AQE can optimize shuffle partitions, you may cache with suboptimal parallelism.
- **Recommendation**: Let AQE optimize the query first, then cache the result for reuse.

## What Problem It Solves

Without caching, every action on a DataFrame triggers full recomputation of its lineage. For a pipeline that:

1. Reads a 500 GB fact table
2. Joins with three dimension tables
3. Applies complex transformations and window functions
4. Then runs five different aggregations on the result

Each aggregation independently recomputes steps 1-3. Caching the intermediate result after step 3 eliminates this redundant work, reducing total compute time by up to 60-80%.

## How to Use It

### Proper Caching Pattern

The critical rule: **cache, then action**. Caching is lazy -- it only materializes data when an action triggers computation.

```scala
// CORRECT: cache() followed by an action
val df = spark.read
  .parquet("/data/fact_table")
  .join(dimDF, Seq("customer_id"))
  .withColumn("rank", rank().over(Window.partitionBy("region").orderBy($"amount".desc)))

df.cache()        // Marks the DataFrame for caching
df.count()        // ACTION: triggers computation and materializes the cache

// Subsequent queries hit the cache
df.filter($"rank" <= 10).count()        // Fast - reads from cache
df.groupBy("region").agg(sum("amount")).show()  // Fast - reads from cache
```

```python
# Python -- correct pattern
from pyspark.sql.functions import rank, sum as _sum
from pyspark.sql.window import Window

df = (spark.read.parquet("/data/fact_table")
      .join(dim_df, ["customer_id"])
      .withColumn("rank", rank().over(
          Window.partitionBy("region").orderBy(col("amount").desc())
      )))

df.cache()
df.count()  # Must trigger an action to materialize the cache

df.filter(col("rank") <= 10).count()
df.groupBy("region").agg(_sum("amount")).show()
```

```sql
-- SQL -- cache table (catalog-level cache)
CACHE TABLE cached_fact_table AS
SELECT f.*, r.rank
FROM fact_table f
JOIN (
  SELECT order_id, RANK() OVER (PARTITION BY region ORDER BY amount DESC) as rank
  FROM fact_table
) r ON f.order_id = r.order_id;

-- Subsequent queries use the cached table
SELECT * FROM cached_fact_table WHERE rank <= 10;
```

### Verifying Caching via EXPLAIN

Run `EXPLAIN` on a query that should hit the cache:

```scala
df.filter($"rank" <= 10).explain("extended")
```

Look for `InMemoryTableScanExec` in the physical plan:

```
== Physical Plan ==
*(1) Filter (rank#5 <= 10)
   +- *(1) Project [order_id#0, customer_id#1, amount#3, rank#5]
      +- InMemoryTableScan [order_id#0, customer_id#1, amount#3, rank#5]
            +- InMemoryRelation [order_id#0, customer_id#1, amount#3, rank#5],
               StorageLevel(disk, memory, deserialized, 1 replicas)
                  +- *(2) SortMergeJoin [customer_id#1], [customer_id#1L], Inner
                     ... (original computation plan) ...
```

The `InMemoryTableScan` shows that data is read from cache. The original computation plan appears as the child of `InMemoryRelation` but is **not executed** -- it's just the lineage metadata.

**Without caching**, the same query shows:

```
== Physical Plan ==
*(4) Filter (rank#5 <= 10)
   +- *(4) Project [...]
      +- *(4) SortMergeJoin ...
         -- Full recomputation, no InMemoryTableScan
```

### Listing Cached Tables

```scala
// Scala -- list all cached tables
spark.sharedState.cacheManager.cacheMetrics.foreach { metrics =>
  println(s"Cached: ${metrics.cacheName}, Size: ${metrics.cachedStorageSize}")
}

// Or check via catalog
spark.catalog.listTables().filter(_.isTemporary).show()

// Check if a specific table is cached
val isCached = spark.catalog.isCached("default.cached_fact_table")
println(s"Is cached: $isCached")
```

```python
# Python
spark.catalog.isCached("cached_fact_table")  # True/False
spark.catalog.listTables()  # Shows cached tables
```

```sql
-- SQL
SHOW TABLES LIKE '*cached*';
```

### Uncaching Properly

```scala
// Uncache a specific DataFrame
spark.catalog.uncacheTable("cached_fact_table")

// Uncache all cached tables
spark.catalog.clearCache()

// Or on the DataFrame itself
df.unpersist()         // Non-blocking: marks for async removal
df.unpersist(blocking = true)  // Blocking: waits for removal
```

```python
# Python
spark.catalog.uncacheTable("cached_fact_table")
spark.catalog.clearCache()

df.unpersist()
df.unpersist(blocking=True)
```

```sql
-- SQL
UNCACHE TABLE cached_fact_table;
-- Or clear all
CLEAR CACHE;
```

## Key Configurations

| Configuration | Default | Recommended | Description |
|---|---|---|---|
| `spark.sql.inMemoryColumnarStorage.compressed` | `true` | `true` | Automatically selects compression codec per column based on data statistics. Keep enabled for memory efficiency. |
| `spark.sql.inMemoryColumnarStorage.batchSize` | `10000` | `10000-20000` | Rows per columnar batch. Larger batches improve compression but increase OOM risk. Increase only if you have ample memory. |
| `spark.sql.inMemoryColumnarStorage.enableVectorizedReader` | `true` | `true` | Enables vectorized reader for cached data scans. Significant performance improvement. |
| `spark.sql.inMemoryColumnarStorage.partitionPruning` | `true` (internal) | `true` | Enables partition pruning on cached tables. Internal config, do not disable. |
| `spark.sql.inMemoryColumnarStorage.hugeVectorThreshold` | `-1` (disabled) | `-1` or `256MB` | Threshold for switching to huge vector memory management. Set to a byte value if caching very large batches. |
| `spark.sql.inMemoryColumnarStorage.hugeVectorReserveRatio` | `1.2` | `1.2` | Memory reserve ratio for huge vectors. Only relevant when `hugeVectorThreshold` is set. |

### Production Configuration Block

```scala
// Production settings for DataFrame caching
val cacheConf = Map(
  "spark.sql.inMemoryColumnarStorage.compressed" -> "true",
  "spark.sql.inMemoryColumnarStorage.batchSize" -> "10000",
  "spark.sql.inMemoryColumnarStorage.enableVectorizedReader" -> "true"
)
cacheConf.foreach { case (k, v) => spark.conf.set(k, v) }
```

## Storage Levels

Unlike RDD caching with custom `StorageLevel`, DataFrame caching uses a fixed storage level but you can control it through `persist()`:

```scala
import org.apache.spark.storage.StorageLevel

// MEMORY_ONLY (default for cache()) -- fastest, but OOM risk
df.cache()  // Equivalent to df.persist(StorageLevel.MEMORY_ONLY)

// MEMORY_AND_DISK -- spills to disk if memory is full (safest)
df.persist(StorageLevel.MEMORY_AND_DISK)

// MEMORY_AND_DISK_SER -- serialized storage, smaller footprint
df.persist(StorageLevel.MEMORY_AND_DISK_SER)

// DISK_ONLY -- only if memory is critically constrained
df.persist(StorageLevel.DISK_ONLY)
```

| Storage Level | Memory | Disk | Serialization | Use Case |
|---|---|---|---|---|
| `MEMORY_ONLY` (default) | Yes | No | Deserialized | Data fits in memory, max performance |
| `MEMORY_AND_DISK` | Yes | Spill | Deserialized | Data may exceed memory, safe choice |
| `MEMORY_AND_DISK_SER` | Yes | Spill | Serialized | Large data, want smaller footprint |
| `DISK_ONLY` | No | Yes | Serialized | Memory-constrained clusters |

## Advanced Patterns

### Catalog Cache vs DataFrame Cache

There are two ways to cache tables:

```scala
// Method 1: DataFrame API (cache())
val df = spark.read.parquet("/data/fact_table")
df.cache()
df.count()

// Method 2: Catalog API (CACHE TABLE)
spark.sql("CACHE TABLE fact_table")

// Difference: CACHE TABLE caches the entire table as registered in the catalog.
// df.cache() caches the specific DataFrame, which may be a filtered/transformed subset.
```

| Aspect | `df.cache()` | `CACHE TABLE` |
|---|---|---|
| Scope | Specific DataFrame (may be filtered) | Entire catalog table |
| Laziness | Requires action to materialize | Triggers immediately |
| Substitution | Plan-based matching (sameResult) | Table name matching |
| Flexibility | Cache any transformation | Only full table scans |

### Memory Monitoring for Cached Data

```scala
// Scala -- monitor cache usage
val cacheManager = spark.sharedState.cacheManager
val metrics = cacheManager.cacheMetrics

metrics.foreach { m =>
  println(s"Cache hits: ${m.cacheHits}")
  println(s"Cache misses: ${m.cacheMisses}")
  println(s"Cached size: ${m.cachedStorageSize} bytes")
  println(s"Cache name: ${m.cacheName}")
}

// Also check Spark UI -> Storage tab for cached RDD details
spark.sparkContext.getRDDStorageInfo.foreach { info =>
  println(s"RDD ${info.id}: ${info.numCachedPartitions} partitions, " +
    s"${info.memSize} bytes in memory, ${info.diskSize} bytes on disk")
}
```

```python
# Python
sc = spark.sparkContext
for info in sc._jsc.sc().getRDDStorageInfo():
    print(f"RDD {info.id()}: mem={info.memSize()}, disk={info.diskSize()}")
```

## Limitations and Gotchas

### Caching Before an Action Does Nothing

The most common mistake: calling `df.cache()` and assuming data is cached.

```scala
// WRONG: no action, nothing is cached
df.cache()
val count = df.filter($"x" > 10).count()  // Recomputes everything!

// CORRECT: action after cache()
df.cache()
df.count()  // Materializes the cache
val count = df.filter($"x" > 10).count()  // Uses the cache
```

`cache()` is **lazy**. It only marks the DataFrame for caching. The data is materialized only when an action (count, collect, write, show, etc.) triggers computation.

### Stale Cache

If the underlying data changes after caching, the cache is **not automatically invalidated**. You must manually uncache and recache:

```scala
// After underlying data changes:
spark.catalog.refreshTable("fact_table")  // For catalog-cached tables
df.unpersist()
df.cache()
df.count()  // Recache with fresh data
```

### OOM from Over-Caching

Caching too many DataFrames or caching data larger than available memory causes OutOfMemoryError:

```
java.lang.OutOfMemoryError: Java heap space
  at org.apache.spark.sql.execution.columnar.InMemoryColumnarStorage$.compress(…)
```

**Prevention**:
- Monitor memory usage via the Storage tab
- Use `MEMORY_AND_DISK` as a safety net
- Uncache DataFrames that are no longer needed
- Do not cache data larger than 50% of executor memory

### When Caching Is NOT Beneficial

1. **Single-pass computations**: If a DataFrame is used only once, caching adds overhead (serialization, storage management) without benefit.

2. **Very small DataFrames**: Data under ~100 MB is fast to recompute. The caching overhead outweighs the benefit.

3. **Upstream data changes frequently**: If the source data changes between every query, the cache is constantly stale.

4. **Before AQE optimization**: Caching before AQE can optimize shuffle partitions means you cache with potentially wrong partition counts.

### Cache Substitution Is Plan-Based

The cache is substituted based on **logical plan matching**, not DataFrame identity. This means:

```scala
val df1 = spark.read.parquet("/data/fact_table").filter($"x" > 10)
df1.cache()
df1.count()  // Materializes cache

// This query WILL hit the cache because the logical plan matches
val df2 = spark.read.parquet("/data/fact_table").filter($"x" > 10)
df2.count()  // Uses cached data

// But this query will NOT hit the cache (different filter)
val df3 = spark.read.parquet("/data/fact_table").filter($"x" > 20)
df3.count()  // Recomputes
```

### Column Pruning on Cached Data

When you select only a subset of columns from a cached DataFrame, Spark performs **column pruning** -- only the requested columns are read from the cache:

```scala
// Only reads col1 and col2 from cache, not col3
df.select("col1", "col2").count()
```

This is handled by `InMemoryTableScanExec` which projects only the needed columns.

### Vectorized Reader Requirement

The vectorized reader (`spark.sql.inMemoryColumnarStorage.enableVectorizedReader`) requires:
- `spark.sql.adaptive.enabled = false` (for some older versions), OR
- The cached data was written with vectorized encoding enabled

If the vectorized reader cannot be used, Spark falls back to row-based reading, which is significantly slower.

## Production-Ready Python (PySpark) Code

### Session Factory with Cache Configuration

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
from pyspark.storage import StorageLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production cache configurations
# ---------------------------------------------------------------------------

CACHE_CONFIGS: dict[str, str] = {
    "spark.sql.inMemoryColumnarStorage.compressed": "true",
    "spark.sql.inMemoryColumnarStorage.batchSize": "10000",
    "spark.sql.inMemoryColumnarStorage.enableVectorizedReader": "true",
}


def create_cache_session(
    *,
    app_name: str = "DataFrameCacheProduction",
    master: str = "local[*]",
    storage_fraction: str = "0.5",
    batch_size: str = "10000",
    compressed: str = "true",
    vectorized_reader: str = "true",
    warehouse_path: str | None = None,
    extra_configs: dict[str, str] | None = None,
) -> SparkSession:
    """Create a SparkSession tuned for DataFrame caching.

    Parameters
    ----------
    app_name:
        Application name.
    master:
        Spark master URL.
    storage_fraction:
        ``spark.memory.storageFraction`` -- fraction of storage space
        reserved for cached data (default ``0.5``).
    batch_size:
        ``spark.sql.inMemoryColumnarStorage.batchSize`` -- rows per batch.
    compressed:
        ``spark.sql.inMemoryColumnarStorage.compressed`` -- enable per-column
        compression codec selection.
    vectorized_reader:
        ``spark.sql.inMemoryColumnarStorage.enableVectorizedReader``.
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
        .config("spark.sql.inMemoryColumnarStorage.compressed", compressed)
        .config("spark.sql.inMemoryColumnarStorage.batchSize", batch_size)
        .config(
            "spark.sql.inMemoryColumnarStorage.enableVectorizedReader",
            vectorized_reader,
        )
        .config("spark.memory.storageFraction", storage_fraction)
    )

    if warehouse_path:
        builder = builder.config("spark.sql.warehouse.dir", warehouse_path)

    if extra_configs:
        for k, v in extra_configs.items():
            builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info(
        "Created cache session: %s (storage_fraction=%s, batch_size=%s)",
        app_name, storage_fraction, batch_size,
    )
    return session
```

### StorageLevel Strategy Picker

```python
@dataclass(frozen=True)
class CacheStrategy:
    """Recommended caching strategy for a DataFrame."""

    storage_level: StorageLevel
    storage_level_name: str
    reason: str
    estimated_size_bytes: int
    access_count: int
    fits_in_memory: bool


# Mapping from human-readable names to StorageLevel instances.
_STORAGE_LEVELS: dict[str, StorageLevel] = {
    "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
    "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
    "MEMORY_AND_DISK_SER": StorageLevel.MEMORY_AND_DISK_SER,
    "DISK_ONLY": StorageLevel.DISK_ONLY,
}


def cache_strategy_picker(
    *,
    estimated_size_bytes: int,
    available_executor_memory_bytes: int,
    access_pattern_count: int = 1,
    recomputation_cost_seconds: float = 0.0,
) -> CacheStrategy:
    """Select the optimal ``StorageLevel`` based on data size, memory, and access pattern.

    Decision logic
    --------------
    - If the data fits comfortably in executor memory (< 70% of available)
      and will be accessed multiple times, use ``MEMORY_ONLY``.
    - If it fits but is close to the limit, use ``MEMORY_AND_DISK`` as a safety net.
    - If it exceeds executor memory, use ``MEMORY_AND_DISK_SER`` for serialized
      storage to shrink the footprint.
    - If memory is critically constrained or data is very large, use ``DISK_ONLY``.
    - Single-access DataFrames with cheap recomputation are not cached.

    Parameters
    ----------
    estimated_size_bytes:
        Approximate size of the DataFrame in memory.
    available_executor_memory_bytes:
        Total executor memory available for caching.
    access_pattern_count:
        Expected number of downstream queries using the cached DataFrame.
    recomputation_cost_seconds:
        Time to recompute the DataFrame from scratch.

    Returns
    -------
    CacheStrategy
    """
    if access_pattern_count < 2 and recomputation_cost_seconds < 5:
        return CacheStrategy(
            storage_level=StorageLevel.NONE,
            storage_level_name="NONE",
            reason="Single access with cheap recomputation -- skip caching.",
            estimated_size_bytes=estimated_size_bytes,
            access_count=access_pattern_count,
            fits_in_memory=True,
        )

    ratio = estimated_size_bytes / max(available_executor_memory_bytes, 1)

    if ratio < 0.5:
        return CacheStrategy(
            storage_level=StorageLevel.MEMORY_ONLY,
            storage_level_name="MEMORY_ONLY",
            reason="Data fits comfortably in memory; maximum read speed.",
            estimated_size_bytes=estimated_size_bytes,
            access_count=access_pattern_count,
            fits_in_memory=True,
        )
    elif ratio < 0.8:
        return CacheStrategy(
            storage_level=StorageLevel.MEMORY_AND_DISK,
            storage_level_name="MEMORY_AND_DISK",
            reason="Data fits but close to limit; spill to disk if needed.",
            estimated_size_bytes=estimated_size_bytes,
            access_count=access_pattern_count,
            fits_in_memory=True,
        )
    elif ratio < 2.0:
        return CacheStrategy(
            storage_level=StorageLevel.MEMORY_AND_DISK_SER,
            storage_level_name="MEMORY_AND_DISK_SER",
            reason="Serialized storage to reduce memory footprint.",
            estimated_size_bytes=estimated_size_bytes,
            access_count=access_pattern_count,
            fits_in_memory=False,
        )
    else:
        return CacheStrategy(
            storage_level=StorageLevel.DISK_ONLY,
            storage_level_name="DISK_ONLY",
            reason="Memory critically constrained; cache on disk only.",
            estimated_size_bytes=estimated_size_bytes,
            access_count=access_pattern_count,
            fits_in_memory=False,
        )
```

### Cache Verification via EXPLAIN

```python
@dataclass(frozen=True)
class CacheVerificationResult:
    """Result of verifying that a query plan hits the DataFrame cache."""

    has_in_memory_table_scan: bool
    has_in_memory_relation: bool
    storage_level: str
    physical_plan_lines: list[str] = field(default_factory=list)
    cache_active: bool = False

    def summary(self) -> str:
        parts = []
        parts.append(f"InMemoryTableScan present: {self.has_in_memory_table_scan}")
        parts.append(f"InMemoryRelation present: {self.has_in_memory_relation}")
        parts.append(f"Storage level: {self.storage_level}")
        parts.append(f"Cache active: {self.cache_active}")
        if self.cache_active:
            parts.append("PASS -- reading from in-memory cache")
        else:
            parts.append("FAIL -- no cache hit detected")
        return "\n".join(parts)


def verify_cache_active(spark: SparkSession, df: DataFrame) -> CacheVerificationResult:
    """Parse the EXPLAIN output of *df* and check for cache hits.

    A cache hit is confirmed when the physical plan contains
    ``InMemoryTableScan`` with an ``InMemoryRelation`` parent.

    Parameters
    ----------
    spark:
        Active SparkSession (used for catalog checks).
    df:
        DataFrame to analyse.

    Returns
    -------
    CacheVerificationResult
    """
    plan_text = df._jdf.queryExecution().stringWithVerboseString()
    lines = plan_text.split("\n")

    has_in_memory_table_scan = bool(
        re.search(r"InMemoryTableScan", plan_text, re.IGNORECASE)
    )
    has_in_memory_relation = bool(
        re.search(r"InMemoryRelation", plan_text, re.IGNORECASE)
    )

    # Extract storage level
    storage_level = "unknown"
    m = re.search(r"StorageLevel\(([^)]+)\)", plan_text)
    if m:
        storage_level = m.group(1)

    cache_active = has_in_memory_table_scan and has_in_memory_relation

    result = CacheVerificationResult(
        has_in_memory_table_scan=has_in_memory_table_scan,
        has_in_memory_relation=has_in_memory_relation,
        storage_level=storage_level,
        physical_plan_lines=lines,
        cache_active=cache_active,
    )

    log_level = logging.INFO if cache_active else logging.WARNING
    logger.log(log_level, "Cache verification: cache_active=%s", cache_active)
    return result
```

### Cache Benchmark

```python
@dataclass(frozen=True)
class CacheBenchmarkResult:
    """Benchmark results comparing different storage levels."""

    no_cache_time_seconds: float
    memory_only_time_seconds: float
    memory_and_disk_time_seconds: float
    disk_only_time_seconds: float
    cache_size_bytes: dict[str, int] = field(default_factory=dict)
    hit_rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "no_cache_time_seconds": round(self.no_cache_time_seconds, 3),
            "memory_only_time_seconds": round(self.memory_only_time_seconds, 3),
            "memory_and_disk_time_seconds": round(self.memory_and_disk_time_seconds, 3),
            "disk_only_time_seconds": round(self.disk_only_time_seconds, 3),
            "cache_size_bytes": self.cache_size_bytes,
            "hit_rates": self.hit_rates,
        }


def cache_benchmark(
    spark: SparkSession,
    df_factory,  # Callable[[], DataFrame] -- produces a fresh DataFrame
    *,
    query_fn,   # Callable[[DataFrame], DataFrame] -- applies downstream work
    iterations: int = 3,
) -> CacheBenchmarkResult:
    """Benchmark different storage levels by caching and re-running *query_fn*.

    Parameters
    ----------
    spark:
        Active SparkSession.
    df_factory:
        Callable that returns a fresh DataFrame to cache.
    query_fn:
        Callable that takes a DataFrame and returns a new DataFrame
        (e.g. filter + aggregation).
    iterations:
        Number of warm-up + measured runs per storage level.

    Returns
    -------
    CacheBenchmarkResult
    """
    report_path = Path("/tmp/cache_benchmark_report.json")

    def _run(label: str, storage_level: StorageLevel | None) -> tuple[float, int]:
        """Cache with given level, run query_fn, return (avg_seconds, count)."""
        # Clear any previous cache
        spark.catalog.clearCache()

        times: list[float] = []
        count = 0
        for i in range(iterations):
            df = df_factory()
            if storage_level is not None:
                df = df.persist(storage_level)

            # Action to materialize the cache
            q = query_fn(df)
            start = time.monotonic()
            count = int(q.head()[0])
            elapsed = time.monotonic() - start
            if i > 0:
                times.append(elapsed)
            # Unpersist to isolate iterations
            if storage_level is not None:
                df.unpersist(blocking=True)

        avg = sum(times) / max(len(times), 1)
        logger.info("  %s: avg=%.3fs (n=%d)", label, avg, len(times))
        return avg, count

    # No-cache baseline
    baseline_time, _ = _run("no_cache", None)

    # MEMORY_ONLY
    mo_time, _ = _run("MEMORY_ONLY", StorageLevel.MEMORY_ONLY)

    # MEMORY_AND_DISK
    mad_time, _ = _run("MEMORY_AND_DISK", StorageLevel.MEMORY_AND_DISK)

    # DISK_ONLY
    do_time, _ = _run("DISK_ONLY", StorageLevel.DISK_ONLY)

    result = CacheBenchmarkResult(
        no_cache_time_seconds=baseline_time,
        memory_only_time_seconds=mo_time,
        memory_and_disk_time_seconds=mad_time,
        disk_only_time_seconds=do_time,
        cache_size_bytes={
            "MEMORY_ONLY": 0,
            "MEMORY_AND_DISK": 0,
            "DISK_ONLY": 0,
        },
        hit_rates={
            "MEMORY_ONLY": 1.0 if mo_time < baseline_time else 0.0,
            "MEMORY_AND_DISK": 1.0 if mad_time < baseline_time else 0.0,
            "DISK_ONLY": 1.0 if do_time < baseline_time else 0.0,
        },
    )

    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Cache benchmark report written to %s", report_path)
    return result
```

### Cache Metrics Monitor

```python
@dataclass(frozen=True)
class CacheMetrics:
    """Snapshot of cache performance from the SparkContext status tracker."""

    cache_hits: int = 0
    cache_misses: int = 0
    cached_storage_size_bytes: int = 0
    num_cached_partitions: int = 0
    memory_size_bytes: int = 0
    disk_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cached_storage_size_bytes": self.cached_storage_size_bytes,
            "num_cached_partitions": self.num_cached_partitions,
            "memory_size_bytes": self.memory_size_bytes,
            "disk_size_bytes": self.disk_size_bytes,
        }


def monitor_cache_metrics(spark: SparkSession) -> CacheMetrics:
    """Read cache hit/miss metrics from the Spark UI and BlockManager.

    Uses the Java API via ``_jsc`` to access ``getRDDStorageInfo`` and
    cache manager metrics.

    Parameters
    ----------
    spark:
        Active SparkSession.

    Returns
    -------
    CacheMetrics
    """
    sc = spark.sparkContext

    hits = 0
    misses = 0
    cached_size = 0
    mem_size = 0
    disk_size = 0
    num_partitions = 0

    try:
        # RDD storage info from the status tracker
        storage_info = sc._jsc.sc().getRDDStorageInfo()
        for info in storage_info:
            num_partitions += info.numCachedPartitions()
            mem_size += info.memSize()
            disk_size += info.diskSize()

        # Cache metrics from sharedState
        cache_manager = spark.sharedState.cacheManager
        metrics_seq = cache_manager.cacheMetrics()
        if metrics_seq is not None:
            for m in list(metrics_seq):
                hits += int(m.cacheHits())
                misses += int(m.cacheMisses())
                cached_size += int(m.cachedStorageSize())
    except Exception as exc:
        logger.warning("Could not read cache metrics: %s", exc)

    result = CacheMetrics(
        cache_hits=hits,
        cache_misses=misses,
        cached_storage_size_bytes=cached_size,
        num_cached_partitions=num_partitions,
        memory_size_bytes=mem_size,
        disk_size_bytes=disk_size,
    )
    logger.info(
        "Cache metrics: hits=%d, misses=%d, size=%d bytes, partitions=%d",
        hits, misses, cached_size, num_partitions,
    )
    return result
```

### CacheManager Class

```python
class PySparkCacheManager:
    """Manages DataFrame lifecycle (cache, check, unpersist) and generates
    usage reports.

    Wraps Spark's ``catalog.isCached``, ``catalog.uncacheTable``, and
    ``DataFrame.cache`` / ``unpersist`` with structured reporting.

    Parameters
    ----------
    spark:
        Active SparkSession.
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self._registered: dict[str, dict[str, Any]] = {}

    def cache(
        self,
        df: DataFrame,
        name: str,
        storage_level: StorageLevel | None = None,
        blocking: bool = True,
    ) -> None:
        """Cache *df* under *name* and materialise it with ``count()``.

        Parameters
        ----------
        df:
            DataFrame to cache.
        name:
            Registration name used for catalog lookup.
        storage_level:
            Custom ``StorageLevel``; defaults to ``MEMORY_ONLY``.
        blocking:
            Whether to block until cache materialisation completes.
        """
        if storage_level is not None:
            df = df.persist(storage_level)
        else:
            df = df.cache()

        if blocking:
            df.count()  # materialise

        # Also register as a temp view for catalog-level tracking
        df.createOrReplaceTempView(name)
        self._registered[name] = {
            "storage_level": str(storage_level or StorageLevel.MEMORY_ONLY),
            "is_cached": True,
            "timestamp": time.time(),
        }
        logger.info("Cached DataFrame as '%s' (storage_level=%s)", name, storage_level)

    def is_cached(self, name: str) -> bool:
        """Check whether *name* is cached."""
        return self.spark.catalog.isCached(name)

    def unpersist(self, name: str, *, blocking: bool = True) -> None:
        """Unpersist the cached DataFrame registered as *name*.

        Parameters
        ----------
        name:
            Registration name.
        blocking:
            Wait for removal to complete.
        """
        try:
            self.spark.catalog.uncacheTable(name)
        except Exception:
            # May not be catalog-cached; try dropping temp view
            pass
        try:
            self.spark.catalog.dropTempView(name)
        except Exception:
            pass
        self._registered.pop(name, None)
        logger.info("Uncached DataFrame '%s'", name)

    def clear_all(self) -> None:
        """Uncache all registered DataFrames and clear the Spark cache."""
        for name in list(self._registered.keys()):
            self.unpersist(name, blocking=False)
        self.spark.catalog.clearCache()
        logger.info("Cleared all cached DataFrames")

    def generate_report(self, output_path: str = "/tmp/cache_manager_report.json") -> dict[str, Any]:
        """Generate a JSON report of all cached DataFrames.

        Parameters
        ----------
        output_path:
            File path for the report.

        Returns
        -------
        dict
            Report contents.
        """
        report: dict[str, Any] = {
            "registered_tables": {},
            "catalog_cached": [],
            "cache_metrics": None,
        }

        for name, info in self._registered.items():
            info["currently_cached"] = self.is_cached(name)
            report["registered_tables"][name] = info

        # List all cached tables from the catalog
        try:
            cached = self.spark.catalog.listTables()
            for t in cached:
                if t.isTemporary:
                    report["catalog_cached"].append(t.name)
        except Exception:
            pass

        # Cache metrics
        metrics = monitor_cache_metrics(self.spark)
        report["cache_metrics"] = metrics.to_dict()

        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
        logger.info("Cache manager report written to %s", output_path)
        return report
```

### Production Cache Pipeline

```python
def run_production_cache_pipeline(
    spark: SparkSession,
    *,
    fact_table: str = "fact_orders",
    dim_table: str = "dim_customers",
    database: str = "default",
    num_rows: int = 100_000,
) -> dict[str, Any]:
    """Run a production pipeline with smart caching, memory monitoring, and
    automatic unpersist.

    Steps
    -----
    1. Create sample data (or read from existing tables).
    2. Apply an expensive transformation (join + window).
    3. Select a cache strategy based on estimated size.
    4. Cache, run multiple downstream queries, verify cache hits.
    5. Monitor metrics and automatically unpersist when done.

    Parameters
    ----------
    spark:
        Active SparkSession.
    fact_table:
        Name for the fact table.
    dim_table:
        Name for the dimension table.
    database:
        Database name.
    num_rows:
        Number of rows to generate for demo.

    Returns
    -------
    dict
        Pipeline report.
    """
    from pyspark.sql.functions import col, rank, sum as _sum
    from pyspark.sql.window import Window

    logger.info("Starting production cache pipeline")
    manager = PySparkCacheManager(spark)

    # 1. Create sample tables
    fq_fact = f"{database}.{fact_table}"
    fq_dim = f"{database}.{dim_table}"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    spark.sql(f"DROP TABLE IF EXISTS {fq_fact}")
    spark.sql(f"DROP TABLE IF EXISTS {fq_dim}")

    spark.sql(f"""
        CREATE TABLE {fq_fact} (
            order_id BIGINT,
            customer_id BIGINT,
            amount DOUBLE,
            region STRING
        )
        USING parquet
    """)

    spark.sql(f"""
        CREATE TABLE {fq_dim} (
            customer_id BIGINT,
            name STRING,
            tier STRING
        )
        USING parquet
    """)

    # Insert sample data
    spark.sql(f"""
        INSERT INTO {fq_fact}
        SELECT
            id AS order_id,
            (id % 1000) + 1 AS customer_id,
            ROUND(RAND() * 500, 2) AS amount,
            CASE (id % 4)
                WHEN 0 THEN 'US' WHEN 1 THEN 'EU'
                WHEN 2 THEN 'APAC' ELSE 'LATAM'
            END AS region
        FROM RANGE(1, {num_rows + 1})
    """)

    spark.sql(f"""
        INSERT INTO {fq_dim}
        SELECT
            id AS customer_id,
            CONCAT('customer_', id) AS name,
            CASE (id % 3)
                WHEN 0 THEN 'premium' WHEN 1 THEN 'standard' ELSE 'basic'
            END AS tier
        FROM RANGE(1, 1001)
    """)

    # 2. Expensive transformation
    fact_df = spark.table(fq_fact)
    dim_df = spark.table(fq_dim)

    joined = fact_df.join(dim_df, ["customer_id"])
    enriched = joined.withColumn(
        "rank",
        rank().over(
            Window.partitionBy("region").orderBy(col("amount").desc())
        ),
    )

    # 3. Pick cache strategy
    estimated_bytes = estimated_bytes or enriched._jdf.queryExecution().executedPlan().doExecute().count() * 100

    strategy = cache_strategy_picker(
        estimated_size_bytes=estimated_bytes,
        available_executor_memory_bytes=1_073_741_824,  # 1 GB demo
        access_pattern_count=3,
        recomputation_cost_seconds=10.0,
    )
    logger.info("Selected cache strategy: %s", strategy.storage_level_name)

    # 4. Cache and run downstream queries
    manager.cache(enriched, "enriched_orders", storage_level=strategy.storage_level)

    # Downstream query 1: top 10 per region
    q1 = spark.sql("""
        SELECT * FROM enriched_orders WHERE rank <= 10 ORDER BY region, rank
    """)
    q1_count = q1.count()

    # Downstream query 2: regional totals
    q2 = spark.sql("""
        SELECT region, SUM(amount) AS total
        FROM enriched_orders
        GROUP BY region
    """)
    q2_count = q2.count()

    # Downstream query 3: tier distribution
    q3 = spark.sql("""
        SELECT tier, COUNT(*) AS cnt
        FROM enriched_orders
        GROUP BY tier
    """)
    q3_count = q3.count()

    # 5. Verify cache
    verification = verify_cache_active(spark, q1)
    logger.info("Cache verification: %s", verification.summary())

    # 6. Monitor metrics
    metrics = monitor_cache_metrics(spark)

    # 7. Generate report
    report = manager.generate_report()

    # 8. Automatic unpersist
    manager.unpersist("enriched_orders")
    logger.info("Automatically unpersisted 'enriched_orders'")

    pipeline_report = {
        "step": "production_cache_pipeline",
        "strategy": {
            "storage_level": strategy.storage_level_name,
            "reason": strategy.reason,
        },
        "downstream_queries": {
            "q1_top_10": q1_count,
            "q2_regional_totals": q2_count,
            "q3_tier_distribution": q3_count,
        },
        "cache_verification": verification.summary(),
        "cache_metrics": metrics.to_dict(),
        "manager_report": report,
    }

    report_path = Path("/tmp/cache_production_pipeline_report.json")
    report_path.write_text(json.dumps(pipeline_report, indent=2, default=str))
    logger.info("Production cache pipeline report written to %s", report_path)
    return pipeline_report
```
