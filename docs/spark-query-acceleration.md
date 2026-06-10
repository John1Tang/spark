# Spark Query Acceleration Research

This document covers the built-in and third-party acceleration mechanisms available in Apache Spark SQL for speeding up query execution.

---

## Part 1: Built-in Spark Acceleration

Apache Spark has multiple layers of query acceleration built into its core. These require no external dependencies and are either enabled by default or activated via configuration.

### 1.1 Tungsten Execution Engine

Tungsten is Spark's low-level execution backend that provides:

- **UnsafeRow**: Compact binary row format that stores data contiguously in memory, eliminating JVM object headers and pointer overhead.
- **Off-heap memory management**: Direct memory allocation outside the JVM heap, reducing garbage collection pressure.
- **ColumnarBatch**: Batch processing of rows for better CPU cache utilization. When file formats support it (Parquet, ORC), Spark processes rows in batches rather than one at a time.
- **Vectorized Parquet/ORC readers**: `VectorizedReaderBase.java` reads entire column batches at once, using SIMD-friendly memory access patterns.

Key files:
- `sql/catalyst/src/main/java/org/apache/spark/sql/catalyst/expressions/UnsafeRow.java`
- `sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ColumnarBatch.java`
- `sql/core/src/main/java/org/apache/spark/sql/execution/datasources/parquet/VectorizedReaderBase.java`

### 1.2 Whole-Stage Code Generation

Spark compiles chains of physical operators (Project + Filter + Aggregate, etc.) into a **single Java function** at runtime using the Janino compiler. This eliminates virtual function calls inherent in the Volcano iterator model and enables CPU-level optimizations like register allocation and loop unrolling.

```
Before:  Project -> Filter -> HashAggregate  (3 virtual calls per row)
After:   WholeStageCodegenExec               (1 fused tight loop)
```

Key file: `sql/core/src/main/scala/org/apache/spark/sql/execution/WholeStageCodegenExec.scala`

Configuration:
- `spark.sql.codegen.maxFields` (default 100): Disable codegen for wide schemas
- `spark.sql.codegen.fallback` (default false): Allow fallback to interpreted execution

### 1.3 Catalyst Optimizer — Rule-Based Optimizations

The Catalyst optimizer applies 100+ transformation rules in batches. Key optimizations that directly accelerate queries:

| Optimization | Effect |
|---|---|
| **Predicate Pushdown** (`PushDownPredicates`) | Push filters to data source level, reducing rows read |
| **Column Pruning** (`ColumnPruning`) | Read only needed columns from storage |
| **Constant Folding** | Pre-compute static expressions at plan time |
| **Null Propagation** (`NullPropagation`) | Short-circuit null-aware expressions |
| **Boolean Simplification** | Simplify `x AND TRUE` → `x` |
| **Limit PushDown** | Stop scanning after N rows |
| **Join Reorder** (CBO) | Reorder multi-way joins to minimize intermediate data |
| **Inline CTE** (`InlineCTE`) | Inline common table expressions |
| **Propagate Empty Relation** (`PropagateEmptyRelation`) | Eliminate entire branches when one side is empty |
| **Eliminate Outer Join** | Convert to inner join when nulls are eliminated |
| **PushDownLeftSemiAntiJoin** | Push semi/anti-join predicates deeper |
| **RemoveRedundantAggregates** | Remove duplicate aggregation steps |
| **ConstantPropagation** | Propagate known constant values through the plan |
| **FoldablePropagation** | Propagate foldable expressions through aliases |
| **CombineFilters** | Merge adjacent Filter operators |
| **PruneFilters** | Remove redundant filter conditions |

### 1.4 Cost-Based Optimizer (CBO)

CBO enables join reordering based on actual table statistics. Unlike rule-based heuristics, CBO uses a selectivity model to estimate output row counts and selects the lowest-cost join order.

```sql
-- Required: collect statistics first
ANALYZE TABLE sales COMPUTE STATISTICS FOR ALL COLUMNS;
ANALYZE TABLE products COMPUTE STATISTICS FOR ALL COLUMNS;

-- Enable CBO
SET spark.sql.cbo.enabled = true;
```

Configuration:
- `spark.sql.cbo.enabled`: Enable cost-based optimization
- `spark.sql.cbo.joinReorder.enabled`: Enable join reorder optimization
- `spark.sql.cbo.joinReorder.dp.threshold`: Max tables for dynamic programming (default 12)

### 1.5 Adaptive Query Execution (AQE)

AQE is the most impactful built-in acceleration feature. Enabled by default since Spark 3.2.0, it re-optimizes the physical plan mid-execution using runtime statistics.

**Five core features:**

1. **Coalesce Partitions**: Merges small shuffle partitions into larger ones based on actual data size. Eliminates the "too many small tasks" problem.

2. **Dynamic Join Switching**: Converts SortMergeJoin → BroadcastHashJoin when runtime data shows one side is smaller than the broadcast threshold. Also converts SortMergeJoin → ShuffledHashJoin when partition sizes allow.

3. **Skew Join Optimization**: Detects skewed partitions (size > 5x median) and splits them into smaller sub-partitions for parallel processing.

4. **Empty Relation Propagation**: Detects empty tables at runtime and eliminates downstream computation entirely.

5. **Local Shuffle Reader**: After converting SMJ → BHJ, reads shuffle files locally to save network traffic.

```scala
// Key AQE configurations
spark.sql.adaptive.enabled = true                        // enabled by default since 3.2
spark.sql.adaptive.coalescePartitions.enabled = true     // merge small partitions
spark.sql.adaptive.skewJoin.enabled = true               // handle data skew
spark.sql.adaptive.autoBroadcastJoinThreshold            // runtime broadcast threshold
spark.sql.adaptive.advisoryPartitionSizeInBytes = 64MB   // target partition size
spark.sql.adaptive.skewJoin.skewedPartitionFactor = 5.0  // skew detection threshold
spark.sql.adaptive.localShuffleReader.enabled = true     // local shuffle reading
```

Implementation files:
- `sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/AdaptiveSparkPlanExec.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/CoalesceShufflePartitions.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/OptimizeSkewedJoin.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/OptimizeShuffleWithLocalRead.scala`

### 1.6 Dynamic Partition Pruning (DPP)

When a large partitioned fact table is joined with a smaller dimension table on the partition column, DPP pushes the dimension table's filtered keys down to the fact table's file scan. This eliminates entire partition reads before any shuffle begins.

```sql
-- Star schema query: only relevant date partitions are scanned
SELECT * FROM sales s
JOIN date_dim d ON s.date_key = d.date_key
WHERE d.year = 2025 AND d.quarter = 'Q1';
```

Configuration:
- `spark.sql.optimizer.dynamicPartitionPruning.enabled = true` (default)
- `spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly = true` (default)
- `spark.sql.optimizer.dynamicPartitionPruning.joinFilterMultiJoinFactor = 3.0`

Implementation files:
- `sql/core/src/main/scala/org/apache/spark/sql/execution/dynamicpruning/PlanDynamicPruningFilters.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/PlanAdaptiveDynamicPruningFilters.scala`

### 1.7 Storage Partition Join (SPJ)

SPJ eliminates shuffle for joins when both tables are co-partitioned on the join keys. This is a generalization of bucket joins to any V2 data source (Iceberg, Delta, Hudi) that reports partitioning metadata.

```sql
SET spark.sql.sources.v2.bucketing.enabled = true;
SET spark.sql.sources.v2.bucketing.pushPartValues.enabled = true;
SET spark.sql.sources.v2.bucketing.partiallyClusteredDistribution.enabled = true;
```

### 1.8 DataFrame Caching

Spark SQL can cache tables using an in-memory columnar format with automatic codec selection per column:

```scala
df.cache()  // or spark.catalog.cacheTable("tableName")
```

Configuration:
- `spark.sql.inMemoryColumnarStorage.compressed = true` (auto-select codec)
- `spark.sql.inMemoryColumnarStorage.batchSize = 10000`

### 1.9 Bloom Filter Join

Spark supports runtime Bloom filter generation to prune rows before joins. When enabled, Spark builds Bloom filters during aggregation and applies them during scan to skip non-matching rows.

```sql
SET spark.sql.optimizer.runtime.bloomFilter.enabled = true;
SET spark.sql.optimizer.runtime.bloomFilter.maxNumBits = 1048576;
```

Implementation files:
- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/BloomFilterMightContain.scala`
- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/BloomFilterAggregate.scala`

### 1.10 Broadcast Hash Join

For small tables (default < 10MB), Spark broadcasts the entire table to all executors, eliminating shuffle entirely:

```sql
SET spark.sql.autoBroadcastJoinThreshold = 10485760;  -- 10MB default
SELECT /*+ BROADCAST(dim) */ * FROM fact JOIN dim ON fact.key = dim.key;
```

### 1.11 Configuration-Level Tuning

Key performance configurations:

| Configuration | Default | Effect |
|---|---|---|
| `spark.sql.shuffle.partitions` | 200 | Number of shuffle partitions |
| `spark.sql.files.maxPartitionBytes` | 128MB | Max bytes per file partition |
| `spark.sql.files.minPartitionNum` | leafNodeDefaultParallelism | Min file partitions |
| `spark.serializer` | KryoSerializer | Serialization format |
| `spark.memory.fraction` | 0.6 | Unified memory for execution + storage |
| `spark.sql.join.preferSortMergeJoin` | true | Prefer SMJ over SHJ |
| `spark.sql.files.maxPartitionNum` | None | Max file partitions (3.5+) |

---

## Part 2: Third-Party Accelerators

Three major open-source projects provide significant query acceleration for Spark with zero or minimal code changes.

### 2.1 NVIDIA RAPIDS Accelerator for Apache Spark

**What it is:** A drop-in plugin that offloads Spark SQL execution to NVIDIA GPUs using the RAPIDS cuDF library.

**Architecture:**
- Replaces Spark's physical plan operators with GPU equivalents (`GpuProject`, `GpuFileGpuScan`, etc.)
- Data is transferred to GPU memory and processed in parallel using CUDA
- Unsupported operators fall back to CPU execution transparently
- Works with AQE, DPP, and other built-in Spark optimizations

**Setup:**
```bash
spark-submit \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.executor.resource.gpu.amount=1 \
  --jars rapids-4-spark_2.12-26.06.0.jar
```

**Performance:** 3x-7x speedup typical for computation-heavy workloads (joins, aggregations, sorts). Up to 100x for specific queries.

**Supported operations:**
- Joins (BroadcastHashJoin, SortMergeJoin, ShuffledHashJoin)
- Aggregations (hash, object)
- Sorts and sorts with limit
- Window functions
- File I/O (Parquet, ORC, CSV read; Parquet, ORC write)
- Pandas UDFs with GPU acceleration
- MLlib via spark-rapids-ml (zero code change)

**GitHub:** https://github.com/NVIDIA/spark-rapids

**Best for:** GPU-equipped clusters, heavy ETL/ML workloads, existing Spark code with no changes desired.

### 2.2 Apache Gluten (with Velox or ClickHouse backend)

**What it is:** A middle-layer plugin that offloads Spark SQL physical plan execution to native C++ engines via JNI. "Gluten" is Latin for glue — it "glues" native libraries with Spark SQL.

**Architecture:**
- Intercepts the physical plan after Catalyst optimization
- Transforms the plan to Substrait format and sends to native engine via JNI
- Uses Apache Arrow for zero-copy data transfer between JVM and native code
- Returns `ColumnarBatch` results using Spark's columnar API
- Unsupported operators fall back to JVM execution

**Two backends:**
- **Velox** (Meta's C++ execution library): Primary backend, most actively developed
- **ClickHouse** (ported as native library): Alternative backend

**Setup:**
```bash
spark-shell \
  --conf spark.plugins=org.apache.gluten.GlutenPlugin \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=20g \
  --conf spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager \
  --jars gluten-velox-bundle-spark3.5_2.12-1.5.0.jar
```

**Performance (Velox backend, TPC benchmarks):**
- TPC-H: 3.34x overall speedup, up to 23.45x on individual queries
- TPC-DS: 3.02x overall speedup, up to 13.75x on individual queries

**Key advantages:**
- Runs on commodity CPU hardware (no GPUs needed)
- SIMD-optimized native C++ execution (Velox)
- Unified off-heap memory management
- Supports Spark 3.2 through 4.1
- Adopted by Microsoft Fabric, Google Cloud Dataproc, IBM

**GitHub:** https://github.com/apache/gluten

**Best for:** CPU clusters wanting native vectorized execution, SIMD optimization, organizations already invested in Spark ecosystem.

### 2.3 Apache DataFusion Comet

**What it is:** A Spark accelerator built on Apache DataFusion (Rust-based query engine). Translates Spark physical plans to DataFusion physical plans for native execution.

**Architecture:**
- Spark plugin that intercepts the physical plan after Catalyst
- Translates Spark plan operators to DataFusion plan operators
- Executes in native Rust code via Apache Arrow data format
- Uses JNI for JVM ↔ native communication
- Native columnar shuffle (hash and range partitioning)
- Falls back to Spark for unsupported operators

**Setup:**
```bash
export COMET_JAR=comet-spark-spark3.5_2.12-0.15.0.jar

spark-shell \
  --jars $COMET_JAR \
  --conf spark.plugins=org.apache.spark.CometPlugin \
  --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=4g
```

**Performance (TPC-H @ SF1000):** ~2x speedup, translating to 50% cost savings.

**Key features:**
- Runs on commodity hardware (no GPUs)
- Rust-based native execution (memory safe)
- Native Parquet reader
- Native Iceberg reader (enabled by default in 0.15.0)
- Hash joins replace sort-merge joins (experimental)
- Supports Spark 3.4, 3.5, 4.0 (experimental 4.1, 4.2)
- Bloom filter support (native aggregation + filter)

**GitHub:** https://github.com/apache/datafusion-comet

**Best for:** Teams wanting Rust-based execution, Iceberg users, organizations preferring Apache-licensed native engines without GPU requirements.

---

## Part 3: Comparison Matrix

| Feature | Built-in Spark | RAPIDS GPU | Gluten+Velox | DataFusion Comet |
|---|---|---|---|---|
| **Hardware** | CPU (JVM) | NVIDIA GPU | CPU (native C++) | CPU (native Rust) |
| **Code changes** | None | None | None | None |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Typical speedup** | Baseline | 3-7x | 3-3.3x | 2x |
| **Vectorized execution** | ColumnarBatch | GPU cuDF | Velox SIMD | DataFusion |
| **Native shuffle** | No | No | ColumnarShuffleManager | CometShuffleManager |
| **AQE compatible** | Yes | Yes | Yes | Yes |
| **Parquet acceleration** | Vectorized reader | GPU reader | Velox native | Rust native |
| **Iceberg support** | V2 API | GPU via plugin | Via Gluten | Native (iceberg-rust) |
| **ANSI mode** | Supported | Supported | Falls back to Spark | Partial (0.10.0+) |
| **Memory model** | JVM heap + off-heap | GPU VRAM | Off-heap native | Off-heap native |

---

## Part 4: Recommended Configuration for Maximum Built-in Acceleration

For maximum performance without third-party plugins:

```scala
val conf = new SparkConf()
  // Serialization
  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  // AQE (enabled by default since 3.2, but be explicit)
  .set("spark.sql.adaptive.enabled", "true")
  .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
  .set("spark.sql.adaptive.skewJoin.enabled", "true")
  .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
  .set("spark.sql.adaptive.autoBroadcastJoinThreshold", "10MB")

  // DPP
  .set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
  .set("spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly", "true")

  // CBO (requires ANALYZE TABLE first)
  .set("spark.sql.cbo.enabled", "true")
  .set("spark.sql.cbo.joinReorder.enabled", "true")

  // Broadcast joins
  .set("spark.sql.autoBroadcastJoinThreshold", "10485760")

  // Bloom filter join
  .set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")

  // Shuffle partitions (set high, let AQE coalesce down)
  .set("spark.sql.shuffle.partitions", "500")

  // File partitioning
  .set("spark.sql.files.maxPartitionBytes", "128MB")
  .set("spark.sql.files.minPartitionNum", "1000")

  // Storage partition join (for Iceberg/Delta)
  .set("spark.sql.sources.v2.bucketing.enabled", "true")
  .set("spark.sql.sources.v2.bucketing.pushPartValues.enabled", "true")
```

---

## Part 5: When to Use Which Accelerator

| Scenario | Recommendation |
|---|---|
| Already have GPU-equipped cluster | RAPIDS Accelerator |
| CPU cluster, want 3x+ speedup on compute-heavy ETL | Gluten + Velox |
| CPU cluster, want Rust-based execution, Iceberg-heavy workloads | DataFusion Comet |
| No infrastructure changes allowed, need immediate gains | Built-in AQE + DPP + CBO |
| Mixed workload (ETL + ML on GPU) | RAPIDS + spark-rapids-ml |
| Cloud-native, cost-sensitive, commodity hardware | DataFusion Comet |
| Enterprise-grade, multi-cloud, vendor-supported | Gluten (backed by IBM, Microsoft, Google) |
