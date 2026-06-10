# Configuration Tuning: Comprehensive Reference Guide

This is the definitive reference for Spark SQL performance configurations. Every setting is organized by category, with defaults, recommended values, and the reasoning behind each tuning decision.

## Production-Ready SparkConf Template

Start with this baseline and adjust per workload:

```scala
// Scala: Production SparkConf
val conf = new SparkConf()
  // === Serialization ===
  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  .set("spark.kryoserializer.buffer", "64m")

  // === Memory ===
  .set("spark.memory.fraction", "0.6")
  .set("spark.memory.storageFraction", "0.5")

  // === Shuffle & Partitions ===
  .set("spark.sql.shuffle.partitions", "200")

  // === File Reading ===
  .set("spark.sql.files.maxPartitionBytes", "134217728")  // 128 MB
  .set("spark.sql.files.openCostInBytes", "4194304")      // 4 MB

  // === Join Optimization ===
  .set("spark.sql.autoBroadcastJoinThreshold", "104857600")  // 100 MB
  .set("spark.sql.broadcastTimeout", "600")

  // === AQE (on by default since 3.2) ===
  .set("spark.sql.adaptive.enabled", "true")
  .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "67108864")  // 64 MB

  // === Dynamic Partition Pruning ===
  .set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

  // === Bloom Filter ===
  .set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")

val spark = SparkSession.builder()
  .config(conf)
  .getOrCreate()
```

```python
# Python: Production SparkSession
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer", "64m")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.5")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.files.maxPartitionBytes", "134217728")
    .config("spark.sql.files.openCostInBytes", "4194304")
    .config("spark.sql.autoBroadcastJoinThreshold", "104857600")
    .config("spark.sql.broadcastTimeout", "600")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "67108864")
    .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
    .config("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")
    .getOrCreate()
)
```

```sql
-- SQL: Set configurations at session level
SET spark.serializer = org.apache.spark.serializer.KryoSerializer;
SET spark.sql.autoBroadcastJoinThreshold = 104857600;
SET spark.sql.adaptive.enabled = true;
SET spark.sql.adaptive.coalescePartitions.enabled = true;
SET spark.sql.optimizer.dynamicPartitionPruning.enabled = true;
```

## Validating Configurations

```scala
// Verify active configurations
spark.conf.getAll.filter { case (k, _) =>
  k.startsWith("spark.sql") || k.startsWith("spark.serializer") ||
  k.startsWith("spark.memory") || k.startsWith("spark.kryo")
}.toSeq.sorted.foreach(println)

// Check a specific config
val threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
println(s"Broadcast threshold: ${threshold.toDouble / 1024 / 1024} MB")
```

---

## Serialization

Spark serializes data during shuffle, broadcast, caching, and task distribution. Serialization format directly impacts network I/O and memory footprint.

### spark.serializer

| Default | Recommended | Scope |
|---|---|---|
| `org.apache.spark.serializer.JavaSerializer` | `org.apache.spark.serializer.KryoSerializer` | Application-wide |

Kryo is 2-10x faster and produces smaller serialized data than Java serialization. Spark uses Kryo internally for shuffle of simple types even with the default serializer, but setting it explicitly ensures Kryo is used everywhere.

```scala
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### spark.kryoserializer.buffer

| Default | Recommended | Scope |
|---|---|---|
| `64k` | `64m` (for large objects) | Application-wide |

This buffer must be large enough to hold the largest single object you serialize. If you see `Kryo buffer overflow` errors, increase this value.

```scala
conf.set("spark.kryoserializer.buffer", "64m")
```

### spark.kryoserializer.buffer.max

| Default | Recommended | Scope |
|---|---|---|
| `64m` | `256m` (if needed) | Application-wide |

Maximum buffer size. Must be >= `spark.kryoserializer.buffer`.

### registerKryoClasses (API, not a config)

Registering classes eliminates the need to store full class names with each serialized object:

```scala
conf.registerKryoClasses(Array(
  classOf[org.apache.spark.sql.catalyst.InternalRow],
  classOf[MyCustomDataType],
  classOf[AnotherClass]
))
```

**When to register**: Only when you have many small objects of custom types. For DataFrames/Datasets with primitive and standard types, Spark's default Kryo registrars already cover most cases.

---

## Memory Management

Spark's unified memory manager divides JVM heap into execution and storage regions. Understanding this division is critical for avoiding OOM errors.

### spark.memory.fraction

| Default | Recommended | Scope |
|---|---|---|
| `0.6` | `0.6` (most workloads), `0.7-0.8` (heavy caching) | Application-wide |

Controls the fraction of (heap - 300 MiB) reserved for Spark's memory pool (execution + storage). The remaining 40% is for user data structures, internal metadata, and OOM safeguards.

```
Total Heap = 16 GB
Reserved (300 MiB) = 0.3 GB
Available for Spark = 16 - 0.3 = 15.7 GB
Spark Memory Pool = 15.7 * 0.6 = 9.42 GB
User Data Structures = 15.7 * 0.4 = 6.28 GB
```

**Increase to 0.7-0.8** when:
- You have well-controlled memory usage and want more caching space
- Running memory-bound workloads with predictable data sizes

**Keep at 0.6** when:
- Running mixed workloads
- User code allocates significant off-heap memory

### spark.memory.storageFraction

| Default | Recommended | Scope |
|---|---|---|
| `0.5` | `0.5` (most), `0.3` (compute-heavy), `0.7` (cache-heavy) | Application-wide |

Controls the fraction of the Spark memory pool that is **reserved** for cached data (immune to execution eviction). Within the 9.42 GB pool above:

```
Storage-Reserved = 9.42 * 0.5 = 4.71 GB (cached data never evicted from this)
Execution-Available = 9.42 GB (can use all, but evicts storage below 4.71 GB)
```

**Lower to 0.3** for compute-heavy workloads (joins, aggregations) that need more execution memory.

**Raise to 0.7** for cache-heavy workloads (repeated queries over cached tables).

### spark.memory.offHeap.enabled

| Default | Recommended | Scope |
|---|---|---|
| `false` | `true` (for large datasets with off-heap capable formats) | Application-wide |

Enables off-heap memory allocation. Spark's Unsafe memory manager uses off-heap memory for sort, shuffle, hash tables, and codegen. When enabled alongside `spark.memory.offHeap.size`, Spark can bypass the JVM heap for execution data.

```scala
conf.set("spark.memory.offHeap.enabled", "true")
conf.set("spark.memory.offHeap.size", "16g")  // Must set this too
```

**When to enable**:
- Running with large executor heap (> 32 GB) where GC pressure is high
- Using columnar formats (Parquet, ORC) with vectorized readers that allocate off-heap
- Using AQE with large advisory partition sizes

### spark.memory.offHeap.size

| Default | Recommended | Scope |
|---|---|---|
| `0` | `16g-32g` (when off-heap is enabled) | Application-wide |

Must be set when `spark.memory.offHeap.enabled` is `true`. This memory is outside the JVM heap and not subject to GC.

### spark.executor.memoryOverhead

| Default | Recommended | Scope |
|---|---|---|
| `exec-mem * 0.10`, min 384m | `2g-4g` (for large executors), higher for off-heap | Per-executor |

Additional memory allocated per executor outside the JVM heap. Covers:
- Off-heap memory (when `spark.memory.offHeap.enabled = true`)
- JVM overhead (thread stacks, metaspace, native libraries)
- Process overhead (Python workers, R processes)

```scala
// For 16g executor with 16g off-heap:
conf.set("spark.executor.memory", "16g")
conf.set("spark.executor.memoryOverhead", "4g")
// Total per executor: 20g
```

**Rule of thumb**: `memoryOverhead >= executor.memory * 0.15 + offHeapSize`. For JDK 17+ with G1GC, you may need even more overhead due to larger native memory usage.

---

## Shuffle & Partitions

The number of partitions determines parallelism. Too few partitions underutilize the cluster; too many create scheduling overhead.

### spark.sql.shuffle.partitions

| Default | Recommended | Scope |
|---|---|---|
| `200` | `2-4x total executor cores` (initial), let AQE coalesce | Session-level |

This is the **initial** number of shuffle partitions. With AQE enabled, Spark will coalesce small partitions at runtime.

```scala
// Calculate based on cluster size
val totalCores = numExecutors * coresPerExecutor
conf.set("spark.sql.shuffle.partitions", (totalCores * 3).toString)
```

**Without AQE**: Set to `2-3x total cores` to avoid straggler tasks from skew.

**With AQE**: Set higher (e.g., `4-5x total cores`) to give AQE fine-grained control over partition coalescing.

### spark.default.parallelism

| Default | Recommended | Scope |
|---|---|---|
| `total cores` (cluster mode), `8` (local) | Same as `spark.sql.shuffle.partitions` | Application-wide |

This applies to RDD operations (`groupByKey`, `reduceByKey`). For DataFrame/SQL, `spark.sql.shuffle.partitions` takes precedence.

```scala
conf.set("spark.default.parallelism", "400")
```

### spark.sql.adaptive.coalescePartitions.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

When enabled with AQE, Spark coalesces contiguous shuffle partitions at runtime to avoid too many small tasks. This is the most impactful AQE setting for partition tuning.

### spark.sql.adaptive.coalescePartitions.parallelismFirst

| Default | Recommended | Scope |
|---|---|---|
| `true` | `false` (for busy clusters) | Session-level |

When `true` (default), Spark ignores `advisoryPartitionSizeInBytes` during coalescing and maximizes parallelism. This can create many small tasks on busy clusters.

**Set to `false`** to let Spark create fewer, larger partitions that match the advisory size. This improves resource utilization on multi-tenant clusters.

```scala
conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
```

### spark.sql.adaptive.coalescePartitions.minPartitionSize

| Default | Recommended | Scope |
|---|---|---|
| `1MB` | `1MB` (most), `4MB` (many tiny partitions) | Session-level |

The minimum partition size after coalescing. Partitions smaller than this are merged with adjacent partitions.

### spark.sql.adaptive.coalescePartitions.initialPartitionNum

| Default | Recommended | Scope |
|---|---|---|
| Same as `spark.sql.shuffle.partitions` | Set higher if you want AQE to have more flexibility | Session-level |

The initial number of shuffle partitions **before** coalescing. Useful if you want `spark.sql.shuffle.partitions` to control the non-AQE path but want a higher initial number for AQE.

### spark.sql.adaptive.advisoryPartitionSizeInBytes

| Default | Recommended | Scope |
|---|---|---|
| `67108864` (64 MB) | `64MB-128MB` (ETL), `32MB-64MB` (interactive) | Session-level |

The target size for coalesced partitions. This is the single most important AQE partition setting.

```scala
// ETL: fewer, larger partitions for throughput
conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  // 128 MB

// Interactive: smaller partitions for lower latency
conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "33554432")  // 32 MB
```

---

## File Reading

These configurations control how Spark splits files into partitions during scans.

### spark.sql.files.maxPartitionBytes

| Default | Recommended | Scope |
|---|---|---|
| `134217728` (128 MB) | `128MB` (default), `256MB-512MB` (fast I/O), `64MB` (slow network) | Session-level |

Maximum bytes packed into a single partition when reading files. Larger partitions mean fewer tasks and less scheduling overhead, but each task processes more data.

```scala
// For clusters with fast local storage (NVMe):
conf.set("spark.sql.files.maxPartitionBytes", "268435456")  // 256 MB

// For object stores (S3, GCS) where network is the bottleneck:
conf.set("spark.sql.files.maxPartitionBytes", "67108864")  // 64 MB
```

**Rule of thumb**: Target 128-256 MB per partition. If tasks complete in under 10 seconds, increase this value. If tasks take over 5 minutes, decrease it.

### spark.sql.files.openCostInBytes

| Default | Recommended | Scope |
|---|---|---|
| `4194304` (4 MB) | `4MB` (most), `1MB` (many small files) | Session-level |

The estimated cost to open a file, measured in equivalent bytes that could be scanned in the same time. This is used when deciding whether to combine multiple small files into one partition.

**Lower to 1 MB** when reading from fast local storage where file open cost is low. Keep at 4 MB for HDFS or object stores.

### spark.sql.files.minPartitionNum

| Default | Recommended | Scope |
|---|---|---|
| `spark.sql.leafNodeDefaultParallelism` | Set explicitly if you want guaranteed minimum partitions | Session-level |

The suggested minimum number of file partitions. This is useful when reading a small number of large files where Spark would otherwise create too few partitions.

### spark.sql.files.maxPartitionNum

| Default | Recommended | Scope |
|---|---|---|
| None (no limit) | `10000-50000` (prevent task explosion) | Session-level |

Added in Spark 3.5.0. Limits the maximum number of file partitions. Spark rescales partitions if the initial count exceeds this value.

**Set this** when reading directories with millions of tiny files to prevent task explosion.

### spark.sql.sources.parallelPartitionDiscovery.threshold

| Default | Recommended | Scope |
|---|---|---|
| `32` | `32` (most), lower for more parallelism | Session-level |

When the number of input paths exceeds this threshold, Spark uses distributed parallel listing instead of sequential listing. Critical for object stores with deep directory structures.

### spark.sql.sources.parallelPartitionDiscovery.parallelism

| Default | Recommended | Scope |
|---|---|---|
| `10000` | `10000` (most) | Session-level |

Maximum parallelism for distributed file listing.

---

## Join Optimization

### spark.sql.autoBroadcastJoinThreshold

| Default | Recommended | Scope |
|---|---|---|
| `10485760` (10 MB) | `100MB-200MB` (data warehouse), `-1` (disable) | Session-level |

Maximum size for automatic broadcast. **This is the single highest-impact join configuration.**

```scala
// For a data warehouse with dimension tables in 50-200 MB range:
conf.set("spark.sql.autoBroadcastJoinThreshold", "209715200")  // 200 MB

// Disable all automatic broadcasts (use hints explicitly):
conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
```

### spark.sql.broadcastTimeout

| Default | Recommended | Scope |
|---|---|---|
| `300` (seconds) | `600` (for large broadcasts), `1200` (very large) | Session-level |

Timeout for broadcast collection and distribution. Increase for large broadcast tables or clusters with many executors.

### spark.sql.join.preferSortMergeJoin

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` (most), `false` (specific workloads) | Session-level |

When `true` (default), Spark prefers SortMergeJoin over ShuffledHashJoin when both are viable. SortMergeJoin is more robust to skew and memory pressure.

**Set to `false`** when:
- Build side of hash join fits comfortably in memory
- You want to avoid the sort overhead on join keys
- You have uniform data distribution (no skew)

### spark.sql.join.forceApplyShuffledHashJoin

| Default | Recommended | Scope |
|---|---|---|
| `false` | `false` (most), `true` (specific cases) | Session-level |

When `true`, forces ShuffledHashJoin even when statistics suggest SortMergeJoin would be better. Use only when you know the build side fits in memory.

---

## Adaptive Query Execution (Full Configuration)

AQE is enabled by default since Spark 3.2.0. Here is the complete configuration set.

### spark.sql.adaptive.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Master toggle for AQE. Should remain enabled for virtually all workloads.

### spark.sql.adaptive.coalescePartitions.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Coalesces shuffle partitions at runtime based on actual data sizes.

### spark.sql.adaptive.advisoryPartitionSizeInBytes

| Default | Recommended | Scope |
|---|---|---|
| `67108864` (64 MB) | `64MB-128MB` | Session-level |

Target size for coalesced partitions. Covered in detail in the Shuffle & Partitions section above.

### spark.sql.adaptive.skewJoin.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Dynamically handles data skew in SortMergeJoin by splitting skewed partitions into smaller ones. This is essential for production workloads where data skew is common.

### spark.sql.adaptive.skewJoin.skewedPartitionFactor

| Default | Recommended | Scope |
|---|---|---|
| `5.0` | `5.0` (most), `3.0` (aggressive skew handling) | Session-level |

A partition is skewed if its size exceeds this factor times the median partition size, AND exceeds `skewedPartitionThresholdInBytes`.

### spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes

| Default | Recommended | Scope |
|---|---|---|
| `274877906944` (256 MB) | `256MB` (most), `128MB` (aggressive) | Session-level |

Absolute threshold for skew detection. A partition must exceed this size to be considered skewed, regardless of the median.

### spark.sql.adaptive.forceOptimizeSkewedJoin

| Default | Recommended | Scope |
|---|---|---|
| `false` | `false` (most), `true` (known severe skew) | Session-level |

When `true`, forces skew optimization even if it introduces extra shuffle.

### spark.sql.adaptive.localShuffleReader.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Uses local shuffle reader when possible (e.g., after converting SMJ to BHJ). Eliminates remote shuffle reads for co-partitioned data.

### spark.sql.adaptive.autoBroadcastJoinThreshold

| Default | Recommended | Scope |
|---|---|---|
| Same as `spark.sql.autoBroadcastJoinThreshold` | Same or higher | Session-level |

AQE-specific threshold for runtime SMJ-to-BHJ conversion. Can be set higher than the planning-time threshold to catch tables that became small after filtering.

```scala
conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "209715200")  // 200 MB
```

### spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold

| Default | Recommended | Scope |
|---|---|---|
| `0` (disabled) | `67108864` (64 MB) to enable SMJ-to-SHJ conversion | Session-level |

When non-zero, AQE converts SortMergeJoin to ShuffledHashJoin if every post-shuffle partition is smaller than this threshold.

### spark.sql.adaptive.optimizer.excludedRules

| Default | Recommended | Scope |
|---|---|---|
| None | Specific rule names to disable | Session-level |

Comma-separated list of AQE optimizer rules to disable. Use only when a specific rule causes issues.

```scala
conf.set("spark.sql.adaptive.optimizer.excludedRules",
  "org.apache.spark.sql.execution.adaptive.DynamicJoinSelection")
```

### spark.sql.adaptive.customCostEvaluatorClass

| Default | Recommended | Scope |
|---|---|---|
| None (uses SimpleCostEvaluator) | Custom class name | Session-level |

Replace AQE's cost evaluator with a custom implementation for workload-specific plan selection.

### spark.sql.adaptive.optimizeSkewsInRebalancePartitions.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Splits skewed partitions during `RebalancePartitions` operations.

### spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor

| Default | Recommended | Scope |
|---|---|---|
| `0.2` | `0.2` | Session-level |

Partitions smaller than this factor times `advisoryPartitionSizeInBytes` are merged during rebalance splitting.

### spark.sql.nonEmptyPartitionRatioForBroadcastJoin

| Default | Recommended | Scope |
|---|---|---|
| `0.2` | `0.2` | Session-level |

Used by `DynamicJoinSelection` to detect joins with many empty partitions. If fewer than 20% of partitions have data, AQE may demote BHJ in favor of shuffle join (which short-circuits on empty partitions).

### spark.sql.adaptive.maxNumPostShufflePartitions

| Default | Recommended | Scope |
|---|---|---|
| `5000` | `5000` (most) | Session-level |

Upper bound on the number of post-shuffle partitions after AQE coalescing.

---

## Dynamic Partition Pruning

DPP filters the fact table using dimension table values at runtime, reducing the amount of data read.

### spark.sql.optimizer.dynamicPartitionPruning.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Master toggle for DPP. When enabled, Spark pushes partition filters from the dimension side of a join down to the fact table scan.

### spark.sql.optimizer.dynamicPartitionPruning.joinReusingGPW

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Enables DPP when the same dimension is joined multiple times (reusing the broadcast exchange).

### spark.sql.optimizer.dynamicPartitionPruning.broadcastExchangeReuse

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Reuses broadcast exchanges for DPP to avoid duplicate broadcast computation.

### spark.sql.optimizer.dynamicPartitionPruning.useStats

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Uses statistics to estimate DPP selectivity and decide whether DPP is beneficial.

### spark.sql.optimizer.dynamicPartitionPruning.fallbackFilterRatio

| Default | Recommended | Scope |
|---|---|---|
| `0.5` | `0.5` (most) | Session-level |

Fallback selectivity ratio when statistics are unavailable. Lower values make DPP more aggressive.

---

## Cost-Based Optimization (CBO)

CBO uses table and column statistics to choose optimal join orders and access paths.

### spark.sql.cbo.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Master toggle for CBO. Requires up-to-date statistics via `ANALYZE TABLE`.

```sql
ANALYZE TABLE orders COMPUTE STATISTICS;
ANALYZE TABLE customers COMPUTE STATISTICS FOR COLUMNS customer_id, region;
```

```scala
conf.set("spark.sql.cbo.enabled", "true")
```

### spark.sql.cbo.joinReorder.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Enables join reordering based on cost. For queries with 4+ tables, CBO can dramatically improve performance by choosing the optimal join order.

### spark.sql.cbo.joinReorder.dp.threshold

| Default | Recommended | Scope |
|---|---|---|
| `12` | `12` (most) | Session-level |

Maximum number of tables before switching to the dynamic programming join reorder algorithm. Below this threshold, exhaustive search is used.

### spark.sql.cbo.joinReorder.dp.star.filter

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Special handling for star-schema join order optimization.

---

## Whole-Stage Code Generation

Codegen compiles query plans into optimized JVM bytecode, eliminating virtual function calls.

### spark.sql.codegen.maxFields

| Default | Recommended | Scope |
|---|---|---|
| `100` | `100` (most), `500` (wide tables) | Session-level |

Maximum number of fields before codegen falls back to interpreted execution. If you have tables with more than 100 columns, increase this.

```scala
conf.set("spark.sql.codegen.maxFields", "500")
```

### spark.sql.codegen.fallback

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

When `true`, Spark falls back to non-codegen execution if codegen fails. When `false`, it throws an error.

### spark.sql.codegen.hugeMethodLimit

| Default | Recommended | Scope |
|---|---|---|
| `8000` | `8000` (most), `16000` (complex queries) | Session-level |

Maximum JVM bytecode method size. If codegen produces methods larger than this, they are split. Complex queries with many expressions may exceed the default.

### spark.sql.codegen.factoryMode

| Default | Recommended | Scope |
|---|---|---|
| `CODEGEN_FACTORY_MODE` | `CODEGEN_FACTORY_MODE` | Session-level |

Controls how generated code is compiled and cached.

---

## Caching

### spark.sql.inMemoryColumnarStorage.compressed

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Automatically selects a compression codec per column based on data statistics. Enabled by default and should remain enabled.

### spark.sql.inMemoryColumnarStorage.batchSize

| Default | Recommended | Scope |
|---|---|---|
| `10000` | `10000-20000` (memory-rich) | Session-level |

Number of rows per batch in columnar caching. Larger batches improve compression and memory utilization but increase per-batch memory pressure.

```scala
// For memory-rich clusters:
conf.set("spark.sql.inMemoryColumnarStorage.batchSize", "20000")
```

---

## Bloom Filter

Bloom filters accelerate join pruning by creating probabilistic membership tests at runtime.

### spark.sql.optimizer.runtime.bloomFilter.enabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Creates bloom filters from the smaller join side at runtime and uses them to filter the larger side before the join.

### spark.sql.optimizer.runtime.bloomFilter.buildSideJoinRowCountThreshold

| Default | Recommended | Scope |
|---|---|---|
| `10000000` (10 million) | `10000000` (most) | Session-level |

Minimum row count for bloom filter creation on the build side.

### spark.sql.optimizer.runtime.bloomFilter.maxNumBits

| Default | Recommended | Scope |
|---|---|---|
| `1048576` (1M bits = 128 KB) | `1048576` (most) | Session-level |

Maximum size of the bloom filter in bits. Larger filters have lower false positive rates but consume more memory.

### spark.sql.optimizer.runtime.bloomFilter.maxNumItems

| Default | Recommended | Scope |
|---|---|---|
| `1000000000` (1 billion) | `1000000000` (most) | Session-level |

Maximum number of items for bloom filter creation. Beyond this threshold, the false positive rate becomes too high.

### spark.sql.optimizer.runtime.bloomFilter.variantFallbackEnabled

| Default | Recommended | Scope |
|---|---|---|
| `true` | `true` | Session-level |

Falls back to a variant bloom filter when the standard filter would be too large.

---

## Data Locality

### spark.locality.wait

| Default | Recommended | Scope |
|---|---|---|
| `3s` | `3s` (most), `10s` (data-heavy) | Application-wide |

How long to wait for a data-local task before falling back to a less optimal locality level.

### spark.locality.wait.process

| Default | Recommended | Scope |
|---|---|---|
| `3s` | `3s` | Application-wide |

Wait time for PROCESS_LOCAL (data in same JVM).

### spark.locality.wait.node

| Default | Recommended | Scope |
|---|---|---|
| `3s` | `3s` | Application-wide |

Wait time for NODE_LOCAL (data on same node).

### spark.locality.wait.rack

| Default | Recommended | Scope |
|---|---|---|
| `3s` | `3s` | Application-wide |

Wait time for RACK_LOCAL (data on same rack).

**Increase these** when:
- Reading from HDFS with local replicas
- Tasks are long-running (minutes)
- You see many `ANY` locality tasks in the Spark UI

**Decrease these** when:
- Tasks are short-running (seconds)
- The cluster has uniform data distribution
- You want to minimize scheduling latency

---

## GC Tuning

Since Spark 4.0, JDK 17 is the default, and G1GC is the default garbage collector.

### spark.executor.defaultJavaOptions

| Default | Recommended | Scope |
|---|---|---|
| None | See templates below | Per-executor |

Set JVM options for executors. This is preferred over `spark.executor.extraJavaOptions` as it is applied after Spark's own JVM options.

```scala
// JDK 17+ with G1GC (recommended for executor heap >= 8g)
conf.set("spark.executor.defaultJavaOptions",
  "-XX:+UseG1GC " +
  "-XX:G1HeapRegionSize=16m " +           // 16 MB regions for 16-32g heaps
  "-XX:InitiatingHeapOccupancyPercent=35 " + // Start concurrent mark at 35% usage
  "-XX:+UnlockDiagnosticVMOptions " +
  "-XX:+G1SummarizeConcMark " +
  "-verbose:gc -Xlog:gc*:file=/tmp/gc-%p.log:time,uptime:filecount=5,filesize=100M")
```

### G1GC Configuration Breakdown

| Option | Purpose | Recommended |
|---|---|---|
| `-XX:+UseG1GC` | Enable G1 collector | Always on JDK 17+ |
| `-XX:G1HeapRegionSize` | Region size (power of 2, 1-32 MB) | `16m` for 16-32g heaps, `32m` for >32g |
| `-XX:InitiatingHeapOccupancyPercent` | Triggers concurrent mark | `35` (default 45, lower = earlier GC) |
| `-XX:MaxGCPauseMillis` | Target max pause | `200` (default) |
| `-XX:+ParallelRefProcEnabled` | Parallel reference processing | Add for better throughput |

### spark.driver.defaultJavaOptions

| Default | Recommended | Scope |
|---|---|---|
| None | G1GC with lower memory pressure settings | Per-driver |

```scala
conf.set("spark.driver.defaultJavaOptions",
  "-XX:+UseG1GC " +
  "-XX:G1HeapRegionSize=8m " +
  "-XX:InitiatingHeapOccupancyPercent=35")
```

---

## Monitoring Configuration Effectiveness

### Spark UI Pages to Check

1. **SQL Tab > Query Details**:
   - Look at "Optimized Logical Plan" vs "Physical Plan"
   - Check if BHJ is being selected over SMJ
   - Review AQE "Final Plan" for runtime optimizations

2. **Storage Tab**:
   - Check broadcast variable sizes (should match expected build-side sizes)
   - Review cached table memory usage

3. **Executors Tab**:
   - Monitor "On-Heap Memory" vs "Off-Heap Memory" usage
   - Check "GC Time" percentage (should be < 10% of task time)

4. **Environment Tab**:
   - Verify all configurations are applied correctly
   - Check `spark.executor.memoryOverhead` allocation

### Key Metrics to Watch

```sql
-- Check broadcast metrics after a join query
-- In Spark UI > SQL Tab > Query Details:
-- BroadcastExchange metrics:
--   dataSize: serialized size of broadcast table
--   numOutputRows: row count of build side
--   broadcastTime: time to distribute to all executors
--   collectTime: time to collect to driver
--   buildTime: time to construct HashedRelation
```

If `broadcastTime` is a large fraction of total query time, consider:
- Reducing the broadcast threshold
- Increasing `spark.sql.broadcastTimeout`
- Checking network bandwidth between driver and executors

### Common Misconfigurations and Symptoms

| Symptom | Likely Cause | Fix |
|---|---|---|
| Query fails with `OutOfMemoryError` during join | Broadcast table too large for executor memory | Lower `autoBroadcastJoinThreshold`, increase executor memory |
| Many small tasks completing instantly | `spark.sql.shuffle.partitions` too high or AQE coalescing disabled | Enable AQE coalescing, reduce initial partition count |
| GC time > 30% of task time | Executor heap too small or too many Java objects | Increase heap, enable off-heap, use Kryo, reduce caching |
| Broadcast timeout errors | `broadcastTimeout` too low for table/executors size | Increase `spark.sql.broadcastTimeout` to 600+ |
| Shuffle spill to disk | Not enough execution memory | Increase `spark.memory.fraction`, reduce `spark.sql.shuffle.partitions` |
| File scan is slow | Partition size too small or too many small files | Increase `maxPartitionBytes`, lower `openCostInBytes` |
| AQE not converting SMJ to BHJ | AQE threshold too low or statistics inaccurate | Increase `aqe.autoBroadcastJoinThreshold`, run `ANALYZE TABLE` |
| Driver OOM during broadcast | Build-side too large for driver to collect | Lower broadcast threshold, increase driver memory |
| Join is slow despite BHJ | Build side has duplicate keys causing explosion | Check data quality, consider `PREFER_SHUFFLE_HASH` hint |

---

## Workload Configuration Templates

### ETL / Batch Processing

Optimized for throughput over latency. Fewer, larger partitions and aggressive broadcast.

```scala
val etlConf = new SparkConf()
  // Serialization
  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  .set("spark.kryoserializer.buffer", "64m")

  // Memory - prioritize execution for heavy computation
  .set("spark.memory.fraction", "0.6")
  .set("spark.memory.storageFraction", "0.3")

  // Partitions - fewer, larger for throughput
  .set("spark.sql.shuffle.partitions", "400")
  .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  // 128 MB
  .set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")

  // Files - large partitions
  .set("spark.sql.files.maxPartitionBytes", "268435456")  // 256 MB
  .set("spark.sql.files.openCostInBytes", "4194304")      // 4 MB

  // Joins - aggressive broadcast for dimension tables
  .set("spark.sql.autoBroadcastJoinThreshold", "209715200")  // 200 MB
  .set("spark.sql.broadcastTimeout", "600")

  // AQE - all on
  .set("spark.sql.adaptive.enabled", "true")
  .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .set("spark.sql.adaptive.skewJoin.enabled", "true")

  // DPP and Bloom Filter
  .set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
  .set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")

  // CBO
  .set("spark.sql.cbo.enabled", "true")
  .set("spark.sql.cbo.joinReorder.enabled", "true")

  // GC for large executors
  .set("spark.executor.defaultJavaOptions",
    "-XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:InitiatingHeapOccupancyPercent=35")
```

### Interactive / Ad-Hoc Queries

Optimized for low latency. More partitions for parallelism, moderate broadcast.

```scala
val interactiveConf = new SparkConf()
  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  .set("spark.kryoserializer.buffer", "64m")

  // Memory - balance execution and caching for repeated queries
  .set("spark.memory.fraction", "0.6")
  .set("spark.memory.storageFraction", "0.5")

  // Partitions - more parallelism for quick response
  .set("spark.sql.shuffle.partitions", "200")
  .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "33554432")  // 32 MB
  .set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "true")

  // Files - moderate partition size
  .set("spark.sql.files.maxPartitionBytes", "134217728")  // 128 MB

  // Joins - moderate broadcast
  .set("spark.sql.autoBroadcastJoinThreshold", "104857600")  // 100 MB
  .set("spark.sql.broadcastTimeout", "300")

  // AQE
  .set("spark.sql.adaptive.enabled", "true")
  .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .set("spark.sql.adaptive.skewJoin.enabled", "true")

  // DPP
  .set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
```

### Streaming (Structured Streaming)

Optimized for low latency with small micro-batches. Minimal broadcast, fine-grained partitions.

```scala
val streamingConf = new SparkConf()
  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  // Memory - lower fraction for streaming's smaller working sets
  .set("spark.memory.fraction", "0.5")
  .set("spark.memory.storageFraction", "0.3")

  // Partitions - fine-grained for low-latency processing
  .set("spark.sql.shuffle.partitions", "200")  // Match to stream parallelism
  .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16777216")  // 16 MB

  // Joins - conservative broadcast (streaming data is often unpredictable)
  .set("spark.sql.autoBroadcastJoinThreshold", "52428800")  // 50 MB
  .set("spark.sql.broadcastTimeout", "120")  // Shorter timeout for streaming

  // AQE - enabled but with conservative settings
  .set("spark.sql.adaptive.enabled", "true")
  .set("spark.sql.adaptive.skewJoin.enabled", "true")

  // DPP - enabled
  .set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

  // Codegen - ensure it's enabled for streaming performance
  .set("spark.sql.codegen.fallback", "true")

  // GC - lower pause targets for streaming latency
  .set("spark.executor.defaultJavaOptions",
    "-XX:+UseG1GC -XX:MaxGCPauseMillis=100 -XX:InitiatingHeapOccupancyPercent=30")
```

---

## Configuration Priority and Override Order

Understanding how configurations are resolved is critical for debugging:

1. **Programmatic** (`SparkConf` / `SparkSession.builder.config`) -- Highest priority
2. **Session-level** (`spark.conf.set()` / `SET key=value` in SQL)
3. **Command-line** (`spark-submit --conf`)
4. **Configuration file** (`spark-defaults.conf`)
5. **Default values** -- Lowest priority

**Important**: Some configurations can only be set before the `SparkSession` is created (application-level), while others can be changed at runtime (session-level). When a configuration is marked as "Application-wide" above, it must be set via `SparkConf` or `spark-submit --conf`, not via `spark.conf.set()`.

## Production-Ready Python (PySpark) Code

### Profile-Based Session Factory

```python
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("config_tuning")

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

CONFIG_PROFILES: dict[str, dict[str, str]] = {
    "default": {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "64m",
        "spark.memory.fraction": "0.6",
        "spark.memory.storageFraction": "0.5",
        "spark.sql.shuffle.partitions": "200",
        "spark.sql.files.maxPartitionBytes": "134217728",
        "spark.sql.files.openCostInBytes": "4194304",
        "spark.sql.autoBroadcastJoinThreshold": "104857600",
        "spark.sql.broadcastTimeout": "300",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "67108864",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
        "spark.sql.optimizer.runtime.bloomFilter.enabled": "true",
        "spark.sql.cbo.enabled": "true",
        "spark.sql.cbo.joinReorder.enabled": "true",
    },
    "etl": {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "64m",
        "spark.memory.fraction": "0.6",
        "spark.memory.storageFraction": "0.3",
        "spark.sql.shuffle.partitions": "400",
        "spark.sql.files.maxPartitionBytes": "268435456",
        "spark.sql.files.openCostInBytes": "4194304",
        "spark.sql.autoBroadcastJoinThreshold": "209715200",
        "spark.sql.broadcastTimeout": "600",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.parallelismFirst": "false",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "134217728",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
        "spark.sql.optimizer.runtime.bloomFilter.enabled": "true",
        "spark.sql.cbo.enabled": "true",
        "spark.sql.cbo.joinReorder.enabled": "true",
        "spark.executor.defaultJavaOptions": (
            "-XX:+UseG1GC -XX:G1HeapRegionSize=16m "
            "-XX:InitiatingHeapOccupancyPercent=35"
        ),
    },
    "interactive": {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "64m",
        "spark.memory.fraction": "0.6",
        "spark.memory.storageFraction": "0.5",
        "spark.sql.shuffle.partitions": "200",
        "spark.sql.files.maxPartitionBytes": "134217728",
        "spark.sql.autoBroadcastJoinThreshold": "104857600",
        "spark.sql.broadcastTimeout": "300",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.parallelismFirst": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "33554432",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
    },
    "ml": {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "64m",
        "spark.kryoserializer.buffer.max": "512m",
        "spark.memory.fraction": "0.7",
        "spark.memory.storageFraction": "0.3",
        "spark.sql.shuffle.partitions": "400",
        "spark.sql.files.maxPartitionBytes": "268435456",
        "spark.sql.autoBroadcastJoinThreshold": "104857600",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "67108864",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.codegen.maxFields": "500",
    },
}

# Recommended checks: (key, expected_regex_or_value, severity, advice)
CONFIG_CHECKS: list[tuple[str, str | re.Pattern[str], str, str]] = [
    (
        "spark.serializer",
        "org.apache.spark.serializer.KryoSerializer",
        "WARNING",
        "KryoSerializer is 2-10x faster than Java serialization for shuffle and broadcast",
    ),
    (
        "spark.sql.adaptive.enabled",
        "true",
        "INFO",
        "AQE is enabled; runtime optimizations will improve performance",
    ),
    (
        "spark.sql.shuffle.partitions",
        re.compile(r"^(?:200)$"),
        "WARNING",
        "Default 200 partitions may be too low for large clusters; "
        "consider 2-4x total executor cores",
    ),
    (
        "spark.sql.autoBroadcastJoinThreshold",
        "10485760",
        "WARNING",
        "Default 10 MB broadcast threshold; raise for data warehouse workloads",
    ),
]
```

### Factory: Create an Optimized Session

```python
def create_optimized_session(
    profile: str = "default",
    app_name: str = "OptimizedSpark",
    master: Optional[str] = None,
    extra_conf: Optional[dict[str, str]] = None,
) -> SparkSession:
    """Create a SparkSession using a named configuration profile.

    Supported profiles: ``default``, ``etl``, ``interactive``, ``ml``.

    Parameters
    ----------
    profile :
        Configuration profile name.
    app_name :
        Spark application name.
    master :
        Spark master URL (e.g. ``yarn``, ``local[*]``). ``None`` uses
        the default.
    extra_conf :
        Additional key=value config overrides.

    Returns
    -------
    SparkSession

    Raises
    ------
    ValueError
        If *profile* is not one of the recognized names.
    """
    profile = profile.lower()
    if profile not in CONFIG_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Available: {', '.join(CONFIG_PROFILES)}"
        )

    conf = CONFIG_PROFILES[profile]
    if extra_conf:
        conf = {**conf, **extra_conf}

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)

    for k, v in conf.items():
        builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info(
        "Created SparkSession with profile='%s' (%d config entries)",
        profile, len(conf),
    )
    return session
```

### Cluster Resource Detection and Recommendations

```python
@dataclass
class ClusterResources:
    """Detected cluster resource summary."""
    total_executor_cores: int = 0
    num_executors: int = 0
    executor_memory_mb: int = 0
    driver_memory_mb: int = 0
    cluster_manager: str = "unknown"


@dataclass
class ConfigRecommendation:
    key: str
    current_value: Optional[str]
    recommended_value: str
    reason: str
    severity: str  # INFO, WARNING, CRITICAL


def get_config_recommendations(
    resources: Optional[ClusterResources] = None,
) -> list[ConfigRecommendation]:
    """Analyze cluster resources and return optimal configuration settings.

    If *resources* is not provided, the function attempts to detect
    environment variables and system defaults from a running SparkSession
    or the OS.

    Returns
    -------
    list[ConfigRecommendation]
    """
    if resources is None:
        # Heuristic detection from environment
        num_executors = int(os.environ.get("SPARK_EXECUTOR_INSTANCES", "4"))
        cores_per_executor = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))
        exec_mem_str = os.environ.get("SPARK_EXECUTOR_MEMORY", "16g")
        exec_mem_mb = _parse_memory(exec_mem_str)
        resources = ClusterResources(
            total_executor_cores=num_executors * cores_per_executor,
            num_executors=num_executors,
            executor_memory_mb=exec_mem_mb,
            cluster_manager=(
                "yarn" if os.environ.get("YARN_CONF_DIR") else
                "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else
                "standalone"
            ),
        )

    total_cores = resources.total_executor_cores
    recommendations: list[ConfigRecommendation] = []

    # Shuffle partitions
    recommended_partitions = max(200, total_cores * 3)
    recommendations.append(ConfigRecommendation(
        key="spark.sql.shuffle.partitions",
        current_value=None,
        recommended_value=str(recommended_partitions),
        reason=f"Based on {total_cores} total cores, recommend 3x partitions for AQE",
        severity="WARNING",
    ))

    # Advisory partition size
    if resources.executor_memory_mb >= 16384:
        advisory = "134217728"  # 128 MB
    else:
        advisory = "67108864"   # 64 MB
    recommendations.append(ConfigRecommendation(
        key="spark.sql.adaptive.advisoryPartitionSizeInBytes",
        current_value=None,
        recommended_value=advisory,
        reason=f"Executor memory {resources.executor_memory_mb // 1024}g "
               f"-> {'larger' if advisory == '134217728' else 'standard'} partitions",
        severity="INFO",
    ))

    # Broadcast threshold
    if resources.executor_memory_mb >= 32768:
        bhj_threshold = "209715200"  # 200 MB
    else:
        bhj_threshold = "104857600"  # 100 MB
    recommendations.append(ConfigRecommendation(
        key="spark.sql.autoBroadcastJoinThreshold",
        current_value=None,
        recommended_value=bhj_threshold,
        reason=f"Executor memory supports {'200' if bhj_threshold == '209715200' else '100'} MB broadcast",
        severity="WARNING",
    ))

    # Memory fraction
    recommendations.append(ConfigRecommendation(
        key="spark.memory.fraction",
        current_value=None,
        recommended_value="0.6",
        reason="Standard 60/40 split between Spark and user memory",
        severity="INFO",
    ))

    # Executor memory overhead
    overhead_mb = max(2048, int(resources.executor_memory_mb * 0.15))
    recommendations.append(ConfigRecommendation(
        key="spark.executor.memoryOverhead",
        current_value=None,
        recommended_value=f"{overhead_mb}m",
        reason="Overhead >= 15%% of executor memory for JVM and native allocations",
        severity="WARNING",
    ))

    # Serializer
    recommendations.append(ConfigRecommendation(
        key="spark.serializer",
        current_value=None,
        recommended_value="org.apache.spark.serializer.KryoSerializer",
        reason="Kryo is 2-10x faster than Java serialization",
        severity="WARNING",
    ))

    # File partition size
    recommendations.append(ConfigRecommendation(
        key="spark.sql.files.maxPartitionBytes",
        current_value=None,
        recommended_value="268435456",
        reason="256 MB partitions for throughput on modern storage systems",
        severity="INFO",
    ))

    return recommendations


def _parse_memory(mem_str: str) -> int:
    """Parse a memory string like '16g' or '4096m' into MB."""
    mem_str = mem_str.strip().lower()
    if mem_str.endswith("g"):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith("m"):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith("t"):
        return int(float(mem_str[:-1]) * 1024 * 1024)
    return int(mem_str)
```

### Apply Config Profile

```python
def apply_config_profile(
    spark: SparkSession, profile: str = "default"
) -> dict[str, tuple[str | None, str]]:
    """Apply a named configuration profile to a running SparkSession.

    Only session-scritable configs are changed.  Returns a mapping of
    ``key -> (old_value, new_value)`` for all applied settings.

    Parameters
    ----------
    spark : Active SparkSession.
    profile : Profile name (``default``, ``etl``, ``interactive``, ``ml``).

    Returns
    -------
    dict
        Mapping of config key to (old, new) value tuples.
    """
    if profile not in CONFIG_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'")

    conf = CONFIG_PROFILES[profile]
    changes: dict[str, tuple[str | None, str]] = {}

    # Only apply configs that are mutable at runtime (start with spark.sql)
    for key, value in conf.items():
        old = spark.conf.get(key, None)
        if old != value:
            try:
                spark.conf.set(key, value)
                changes[key] = (old, value)
                logger.info("Config change: %s = %s (was %s)", key, value, old)
            except Exception as exc:
                logger.warning("Failed to set %s=%s: %s", key, value, exc)
        else:
            logger.debug("Config %s already set to %s", key, value)

    logger.info(
        "Applied profile='%s': %d changes, %d unchanged",
        profile, len(changes), len(conf) - len(changes),
    )
    return changes
```

### Config Audit

```python
@dataclass
class ConfigAuditReport:
    """Result of a SparkConf audit."""
    timestamp: str = ""
    hostname: str = ""
    spark_version: str = ""
    total_configs: int = 0
    recommendations: list[dict] = field(default_factory=list)
    current_values: dict[str, str] = field(default_factory=dict)


def config_audit(spark: SparkSession) -> ConfigAuditReport:
    """Read the current SparkConf, compare against best-practice checks,
    and generate an audit report.

    Parameters
    ----------
    spark : Active SparkSession.

    Returns
    -------
    ConfigAuditReport
    """
    all_conf = spark.conf.getAll()
    report = ConfigAuditReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hostname=socket.gethostname(),
        spark_version=spark.version,
        total_configs=len(all_conf),
        current_values={k: v for k, v in all_conf.items()},
    )

    for key, expected, severity, advice in CONFIG_CHECKS:
        current = all_conf.get(key)
        if isinstance(expected, re.Pattern):
            is_ok = current is not None and expected.match(current)
        else:
            is_ok = current == expected

        if not is_ok:
            report.recommendations.append({
                "key": key,
                "current": current,
                "expected": expected.pattern if isinstance(expected, re.Pattern) else expected,
                "severity": severity,
                "advice": advice,
            })

    report_path = Path("/tmp/config_audit.json")
    report_path.write_text(json.dumps(asdict(report), indent=2, default=str))
    logger.info(
        "Config audit complete: %d recommendations written to %s",
        len(report.recommendations), report_path,
    )
    return report
```

### Benchmark Config Profiles

```python
@dataclass
class ConfigBenchmarkResult:
    profile: str
    query_name: str
    duration_s: float
    num_tasks: int = 0
    num_shuffle_bytes: int = 0


def benchmark_config_profiles(
    spark: SparkSession,
    fact_path: str,
    dim_path: str,
    join_column: str,
    profiles: Optional[list[str]] = None,
) -> list[ConfigBenchmarkResult]:
    """Run a standard join workload under different config profiles and
    compare performance.

    Parameters
    ----------
    spark : Active SparkSession.
    fact_path : Path to the fact table.
    dim_path : Path to the dimension table.
    join_column : Join column name.
    profiles : List of profiles to test. Defaults to ``["default", "etl", "interactive"]``.

    Returns
    -------
    list[ConfigBenchmarkResult]
    """
    if profiles is None:
        profiles = ["default", "etl", "interactive"]

    fact = spark.read.parquet(fact_path)
    dim = spark.read.parquet(dim_path)
    fact.createOrReplaceTempView("bench_fact")
    dim.createOrReplaceTempView("bench_dim")

    query = (
        f"SELECT f.*, d.* FROM bench_fact f "
        f"JOIN bench_dim d ON f.{join_column} = d.{join_column}"
    )

    original_conf = dict(spark.conf.getAll())
    results: list[ConfigBenchmarkResult] = []

    for profile in profiles:
        logger.info("Benchmarking profile='%s'", profile)
        # Reset to defaults then apply profile
        for key, value in original_conf.items():
            try:
                spark.conf.set(key, value)
            except Exception:
                pass  # some keys may be read-only at runtime

        apply_config_profile(spark, profile)

        start = time.monotonic()
        df = spark.sql(query)
        count = df.count()
        dur = time.monotonic() - start

        # Best-effort task / shuffle metrics from accumulator
        results.append(ConfigBenchmarkResult(
            profile=profile,
            query_name="fact_dim_join",
            duration_s=round(dur, 3),
        ))
        logger.info(
            "Profile=%s count=%d duration=%.3fs", profile, count, dur,
        )

    report_path = Path("/tmp/config_benchmark.json")
    report_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info("Benchmark results written to %s", report_path)

    # Identify fastest profile
    fastest = min(results, key=lambda r: r.duration_s)
    logger.info(
        "Fastest profile: %s (%.3fs)", fastest.profile, fastest.duration_s,
    )
    return results
```

### ConfigManager Class

```python
@dataclass
class ConfigChange:
    key: str
    old_value: Optional[str]
    new_value: str
    applied_at: str = ""


class ConfigManager:
    """Apply, validate, and roll back Spark configuration changes with a
    full audit trail.

    Usage::

        mgr = ConfigManager(spark)
        mgr.set("spark.sql.shuffle.partitions", "400")
        mgr.set("spark.sql.autoBroadcastJoinThreshold", "209715200")
        mgr.apply()
        # If something goes wrong:
        mgr.rollback()
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self._original: dict[str, str] = {}
        self._pending: dict[str, str] = {}
        self._applied: list[ConfigChange] = []
        self._snapshot()

    def _snapshot(self) -> None:
        """Capture current configuration for rollback."""
        self._original = dict(self.spark.conf.getAll())

    def set(self, key: str, value: str) -> None:
        """Stage a configuration change (not applied yet)."""
        old = self._original.get(key)
        self._pending[key] = value
        logger.debug("Staged: %s = %s (was %s)", key, value, old)

    def apply(self) -> list[ConfigChange]:
        """Apply all staged configuration changes."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        applied: list[ConfigChange] = []

        for key, value in self._pending.items():
            old = self.spark.conf.get(key, None)
            try:
                self.spark.conf.set(key, value)
                applied.append(ConfigChange(key, old, value, timestamp))
                logger.info("Applied: %s = %s", key, value)
            except Exception as exc:
                logger.error("Failed to apply %s=%s: %s", key, value, exc)
                # Roll back already-applied changes
                self._rollback_list(applied)
                raise

        self._applied.extend(applied)
        self._pending.clear()
        return applied

    def rollback(self) -> list[ConfigChange]:
        """Roll back all applied changes to their original values."""
        if not self._applied:
            logger.info("Nothing to roll back")
            return []

        # Clear pending first
        self._pending.clear()

        reversed_changes = list(reversed(self._applied))
        self._rollback_list(reversed_changes)
        rolled_back = reversed_changes
        self._applied.clear()

        logger.info("Rolled back %d changes", len(rolled_back))
        return rolled_back

    def _rollback_list(self, changes: list[ConfigChange]) -> None:
        """Roll back a list of changes."""
        for change in changes:
            if change.old_value is not None:
                try:
                    self.spark.conf.set(change.key, change.old_value)
                    logger.info(
                        "Rollback: %s = %s (restored)",
                        change.key, change.old_value,
                    )
                except Exception as exc:
                    logger.error("Rollback failed for %s: %s", change.key, exc)

    def audit_trail(self) -> list[dict]:
        """Return the audit trail of all applied changes."""
        return [asdict(c) for c in self._applied]

    def write_audit(self, path: str = "/tmp/config_manager_audit.json") -> str:
        Path(path).write_text(
            json.dumps(
                {
                    "original_config": self._original,
                    "applied_changes": self.audit_trail(),
                    "pending_changes": self._pending,
                },
                indent=2,
            )
        )
        logger.info("Config audit trail written to %s", path)
        return path


### Production ETL with Dynamic Config

```python
def run_production_tuned_etl(
    spark: SparkSession,
    workload_type: str = "etl",
    fact_path: str = "/data/warehouse/fact",
    dim_paths: dict[str, str] | None = None,
    output_path: str = "/data/warehouse/output",
) -> dict:
    """Run a production ETL pipeline with dynamic configuration selection
    based on *workload_type*.

    Supported workload types: ``etl``, ``interactive``, ``ml``, ``default``.

    Parameters
    ----------
    spark : Active SparkSession.
    workload_type : Named config profile to apply.
    fact_path : Fact table path.
    dim_paths : Optional dimension table alias->path mapping.
    output_path : Destination for the result.

    Returns
    -------
    dict
        ETL metadata with config changes and performance metrics.
    """
    if dim_paths is None:
        dim_paths = {}

    mgr = ConfigManager(spark)
    start = time.monotonic()
    errors: list[str] = []

    try:
        # Apply profile
        profile_conf = CONFIG_PROFILES.get(workload_type, CONFIG_PROFILES["default"])
        for k, v in profile_conf.items():
            mgr.set(k, v)
        applied = mgr.apply()
        logger.info("Applied %d config changes for workload='%s'", len(applied), workload_type)

        # Register tables
        spark.read.parquet(fact_path).createOrReplaceTempView("etl_fact")
        for alias, path in dim_paths.items():
            spark.read.parquet(path).createOrReplaceTempView(alias)

        # Build and execute the join
        select_cols = "etl_fact.*"
        from_clause = "etl_fact"
        for alias in dim_paths:
            from_clause += f" LEFT JOIN {alias} ON etl_fact.id = {alias}.id"

        query = f"SELECT {select_cols} FROM {from_clause}"
        logger.info("Executing ETL query with %d dimension tables", len(dim_paths))

        result_df = spark.sql(query)
        result_df.write.mode("overwrite").parquet(output_path)

        dur = time.monotonic() - start
        logger.info("ETL completed in %.3fs", dur)

    except Exception as exc:
        logger.error("ETL failed: %s", exc)
        errors.append(str(exc))
        mgr.rollback()
        logger.info("Configuration rolled back")
        dur = time.monotonic() - start
    else:
        dur = time.monotonic() - start

    audit_path = mgr.write_audit()
    report = {
        "workload_type": workload_type,
        "duration_s": round(dur, 3),
        "config_changes_applied": len(applied) if 'applied' in dir() else 0,
        "output_path": output_path,
        "audit_trail": audit_path,
        "errors": errors,
    }

    report_path = Path("/tmp/production_tuned_etl.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("ETL report written to %s", report_path)
    return report
```

### Quick Usage Example

```python
# 1. Create session with a profile
spark = create_optimized_session(profile="etl", app_name="MyETL")

# 2. Audit current config
audit = config_audit(spark)
for rec in audit.recommendations:
    print(f"[{rec['severity']}] {rec['key']}: {rec['advice']}")

# 3. Get resource-based recommendations
resources = ClusterResources(
    total_executor_cores=64,
    num_executors=8,
    executor_memory_mb=32768,
)
recs = get_config_recommendations(resources)
for r in recs:
    print(f"{r.key}: {r.recommended_value} ({r.severity})")

# 4. Apply a profile to a running session
changes = apply_config_profile(spark, "interactive")
for key, (old, new) in changes.items():
    print(f"{key}: {old} -> {new}")

# 5. Use ConfigManager for safe changes with rollback
mgr = ConfigManager(spark)
mgr.set("spark.sql.shuffle.partitions", "400")
mgr.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")
mgr.apply()
# On error: mgr.rollback()

# 6. Run tuned ETL
report = run_production_tuned_etl(
    spark,
    workload_type="etl",
    fact_path="/data/warehouse/orders",
    dim_paths={
        "customers": "/data/warehouse/customers",
        "products": "/data/warehouse/products",
    },
    output_path="/data/warehouse/enriched_orders",
)
```
