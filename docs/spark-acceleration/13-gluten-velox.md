# Apache Gluten with Velox Backend

## Overview

Apache Gluten is a **middle-layer plugin** that offloads Spark SQL physical plan execution to native C++ engines via JNI, achieving significant CPU-based performance improvements with **zero code changes**. The name "Gluten" = "glue" -- it glues Spark's JVM execution model to high-performance native engines.

**GitHub:** https://github.com/apache/gluten
**Documentation:** https://gluten.apache.org/
**Status:** Apache Incubating project (graduated to top-level Apache project)

### What It Is

Gluten intercepts Spark's physical execution plan after Catalyst optimization, transforms it into the Substrait intermediate representation, and dispatches it to a native execution engine. The primary backend is **Velox** (Meta's open-source C++ data processing library), with **ClickHouse** available as an alternative backend.

Key properties:
- **Drop-in replacement**: Add configuration flags and a JAR; no application code changes
- **Native execution**: C++ SIMD-optimized operators run on CPU, not JVM
- **Zero-copy data transfer**: Apache Arrow columnar format enables direct memory sharing between JVM and native engine
- **Transparent fallback**: Unsupported operators fall back to vanilla Spark execution
- **Columnar-first**: Uses Spark's columnar execution API throughout the pipeline
- **SIMD acceleration**: Leverages AVX2/AVX-512/NEON vectorization for parallel processing

### Why It Works

The architecture eliminates JVM overhead and leverages native code optimization:

1. **Physical plan interception**: After Catalyst produces the optimized physical plan, Gluten's `GlutenStrategy` scans for replaceable subtrees
2. **Substrait translation**: Replaceable plan subtrees are converted to Substrait, a cross-language query intermediate representation
3. **JNI dispatch**: The Substrait plan is sent to the native engine via JNI
4. **Native execution**: Velox (or ClickHouse) executes the plan using:
   - **SIMD vectorization**: Processes 8-64 elements per CPU instruction via AVX2/AVX-512
   - **Columnar processing**: Data stays in Arrow columnar format end-to-end
   - **Cache-aware algorithms**: Optimized for CPU cache line sizes and prefetching
   - **Multi-threading**: Native thread pools bypass JVM thread overhead
   - **Zero-copy**: Arrow IPC enables shared memory between JVM and native
5. **ColumnarBatch return**: Results are returned as Spark `ColumnarBatch` objects
6. **Mixed execution**: Native and JVM operators coexist in the same plan

The performance gains come from:
- **Eliminating JVM overhead**: No JIT warmup, no GC pauses, no object allocation overhead
- **SIMD parallelism**: AVX-512 processes 8 doubles or 16 ints per instruction
- **Better cache utilization**: Columnar layout with prefetching maximizes L1/L2 cache hit rates
- **Native memory management**: Off-heap memory avoids GC pressure
- **Optimized algorithms**: Velox implements hand-tuned hash tables, sort algorithms, and join strategies

### What Problem It Solves

| Problem | Gluten Solution |
|---------|----------------|
| JVM overhead in Spark SQL execution | Native C++ eliminates JIT, GC, object allocation overhead |
| Slow analytical queries on CPU clusters | SIMD vectorization achieves 3-13x speedup on TPC benchmarks |
| Row-to-row conversion overhead | End-to-end columnar Arrow pipeline eliminates conversion |
| Memory pressure from JVM heap | Off-heap native memory management bypasses GC |
| GC pauses causing tail latency spikes | Deterministic native memory management eliminates GC |
| Cost of scaling CPU clusters for analytical workloads | Same hardware achieves 3x throughput improvement |
| Vendor lock-in for native SQL execution | Open-source Apache-licensed alternative to proprietary accelerators |

---

## Installation and Deployment

### Prerequisites

- Linux x86_64 (ARM64 supported with limited testing)
- GCC 11+ or Clang 14+ (for building from source)
- CMake 3.23+
- Apache Spark 3.2.x through 4.1.x
- Java 8, 11, or 17
- AVX2-capable CPU (most CPUs since 2013); AVX-512 for maximum performance

### Supported Spark Versions

| Spark Version | Gluten Support | Notes |
|---------------|----------------|-------|
| 3.2.x | Yes | LTS |
| 3.3.x | Yes | LTS |
| 3.4.x | Yes | |
| 3.5.x | Yes | Recommended LTS |
| 4.0.x | Yes | |
| 4.1.x | Yes | Latest |

### Current Version

As of mid-2026, the current stable release is **1.5.0**. Always verify the latest version on [GitHub Releases](https://github.com/apache/gluten/releases).

### Option 1: Pre-built JARs (Recommended)

Download pre-built bundles from Maven Central:

```bash
# For Spark 3.5 with Velox backend (Scala 2.12)
wget https://repo1.maven.org/maven2/org/apache/gluten/gluten-velox-bundle-spark3.5_2.12/1.5.0/gluten-velox-bundle-spark3.5_2.12-1.5.0.jar

# For Spark 3.5 with ClickHouse backend
wget https://repo1.maven.org/maven2/org/apache/gluten/gluten-clickhouse-bundle-spark3.5_2.12/1.5.0/gluten-clickhouse-bundle-spark3.5_2.12-1.5.0.jar
```

**Available bundles:**
- `gluten-velox-bundle-spark3.5_2.12-1.5.0.jar` -- Velox backend, Spark 3.5, Scala 2.12
- `gluten-velox-bundle-spark3.5_2.13-1.5.0.jar` -- Velox backend, Spark 3.5, Scala 2.13
- `gluten-velox-bundle-spark3.4_2.12-1.5.0.jar` -- Velox backend, Spark 3.4
- `gluten-velox-bundle-spark3.3_2.12-1.5.0.jar` -- Velox backend, Spark 3.3

### Option 2: Build from Source

**Static build** (bundles all native libraries into the JAR -- recommended for deployment):

```bash
git clone https://github.com/apache/gluten.git
cd gluten

# Build Velox bundle for Spark 3.5 (static, self-contained JAR)
./dev/buildbundle-veloxbe.sh --spark-version 3.5 --scala-version 2.12

# Output: backends-velox/bundle/target/gluten-velox-bundle-spark3.5_2.12-1.5.0.jar
```

**Dynamic build** (separate native `.so` file -- smaller JAR, requires native library on each node):

```bash
# Build native library first
./dev/build-lib.sh --backend velox --build-type release

# Build the bundle JAR (references the native library)
./dev/buildbundle-veloxbe.sh --spark-version 3.5 --build-type dynamic

# You must distribute the native .so to all cluster nodes:
# backends-velox/velox/lib/libvelox_spark.so -> /usr/lib/ on each node
```

**Build dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential cmake ninja-build \
  libssl-dev libboost-all-dev libdouble-conversion-dev \
  libgflags-dev libgoogle-glog-dev libevent-dev \
  libsnappy-dev libzstd-dev liblz4-dev \
  flex bison

# RHEL/CentOS/AlmaLinux
sudo yum install -y gcc gcc-c++ make cmake3 ninja-build \
  openssl-devel boost-devel double-conversion-devel \
  gflags-devel glog-devel libevent-devel \
  snappy-devel zstd-devel lz4-devel \
  flex bison
```

### Basic Spark Usage

```bash
spark-shell \
  --conf spark.plugins=org.apache.gluten.GlutenPlugin \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=20g \
  --conf spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager \
  --jars gluten-velox-bundle-spark3.5_2.12-1.5.0.jar
```

---

## Deployment Platforms

### YARN Deployment

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 8 \
  --executor-cores 16 \
  --executor-memory 64g \
  --conf spark.plugins=org.apache.gluten.GlutenPlugin \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=20g \
  --conf spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager \
  --conf spark.gluten.sql.columnar.shuffle.enabled=true \
  --conf spark.gluten.memory.mode=unsafe \
  --conf spark.sql.files.maxPartitionBytes=256m \
  --jars gluten-velox-bundle-spark3.5_2.12-1.5.0.jar \
  your-application.jar
```

**YARN with dynamic native library** (if using dynamic build):

```bash
# Include the native .so as a distributed file
spark-submit \
  --conf spark.yarn.dist.archives=libvelox_spark.so#velox-lib \
  --conf spark.executorEnv.LD_LIBRARY_PATH=./velox-lib \
  --jars gluten-velox-bundle-spark3.5_2.12-1.5.0.jar \
  your-application.jar
```

### Kubernetes Deployment

```yaml
apiVersion: "sparkoperator.k8s.io/v1beta2"
kind: SparkApplication
metadata:
  name: gluten-spark-app
spec:
  type: Scala
  mode: cluster
  image: "spark:3.5-gluten"
  sparkVersion: "3.5.0"
  driver:
    cores: 4
    memory: "16g"
  executor:
    cores: 16
    memory: "64g"
    instances: 8
  sparkConf:
    "spark.plugins": "org.apache.gluten.GlutenPlugin"
    "spark.memory.offHeap.enabled": "true"
    "spark.memory.offHeap.size": "20g"
    "spark.shuffle.manager": "org.apache.spark.shuffle.sort.ColumnarShuffleManager"
    "spark.gluten.sql.columnar.shuffle.enabled": "true"
    "spark.gluten.memory.mode": "unsafe"
    "spark.sql.files.maxPartitionBytes": "256m"
  deps:
    jars:
      - "local:///opt/spark/jars/gluten-velox-bundle-spark3.5_2.12-1.5.0.jar"
```

**Custom Dockerfile**:

```dockerfile
FROM apache/spark:3.5.0-scala2.12-java17-ubuntu

# Install runtime dependencies for Velox native library
RUN apt-get update && apt-get install -y \
    libsnappy1v5 libzstd1 liblz4-1 libgflags2.2 \
    libgoogle-glog0v5 libdouble-conversion3 libevent-2.1-7 \
    libssl3 libboost-program-options1.74.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Gluten JAR
COPY gluten-velox-bundle-spark3.5_2.12-1.5.0.jar /opt/spark/jars/
```

### Google Cloud Dataproc

Gluten is available as an optional component on Dataproc:

```bash
gcloud dataproc clusters create gluten-cluster \
  --region us-central1 \
  --image-version 2.2 \
  --master-machine-type n2-standard-16 \
  --worker-machine-type n2-standard-16 \
  --num-workers 8 \
  --optional-components SPARK \
  --initialization-actions gs://my-bucket/install-gluten.sh \
  --properties spark:spark.plugins=org.apache.gluten.GlutenPlugin,\
spark:spark.memory.offHeap.enabled=true,\
spark:spark.memory.offHeap.size=20g,\
spark:spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager,\
spark:spark.gluten.sql.columnar.shuffle.enabled=true
```

**Dataproc with pre-installed Gluten**: Some Dataproc images include Gluten as a built-in optional component. Check the [Dataproc documentation](https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/optional-components) for availability.

### Microsoft Fabric

Microsoft Fabric includes Gluten as part of its Spark runtime optimization layer. No manual installation is required -- Gluten is automatically enabled for compatible workloads. Configuration is done through Fabric Spark pool settings.

### AWS EMR

```bash
aws emr create-cluster \
  --name "Gluten Spark Cluster" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --instance-type m6i.4xlarge \
  --instance-count 8 \
  --ec2-attributes KeyName=my-key,InstanceProfile=EMR_EC2_DefaultRole \
  --service-role EMR_DefaultRole \
  --configurations '[
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.plugins": "org.apache.gluten.GlutenPlugin",
        "spark.memory.offHeap.enabled": "true",
        "spark.memory.offHeap.size": "20g",
        "spark.shuffle.manager": "org.apache.spark.shuffle.sort.ColumnarShuffleManager",
        "spark.gluten.sql.columnar.shuffle.enabled": "true",
        "spark.gluten.memory.mode": "unsafe",
        "spark.sql.files.maxPartitionBytes": "256m"
      }
    }
  ]' \
  --bootstrap-actions Path=s3://my-bucket/install-gluten.sh,Name="Install Gluten"
```

---

## Usage

### Basic Usage (Zero Code Change)

Existing Spark SQL and DataFrame code runs with native acceleration:

```scala
// Scala - existing code, now runs on native engine
val df = spark.read.parquet("s3://data-bucket/large-dataset/")

val result = df
  .filter($"value" > 100)
  .groupBy($"category")
  .agg(
    sum($"amount").as("total"),
    avg($"quantity").as("avg_qty"),
    count($"*").as("cnt")
  )
  .orderBy($"total".desc)

result.write.mode("overwrite").parquet("s3://output-bucket/results/")
```

```python
# PySpark - existing code, now runs on native engine
from pyspark.sql.functions import col, sum, avg, count

df = spark.read.parquet("s3://data-bucket/large-dataset/")

result = (df
  .filter(col("value") > 100)
  .groupBy("category")
  .agg(
    sum("amount").alias("total"),
    avg("quantity").alias("avg_qty"),
    count("*").alias("cnt")
  )
  .orderBy(col("total").desc())
)

result.write.mode("overwrite").parquet("s3://output-bucket/results/")
```

### SQL Queries

```scala
spark.sql("""
  SELECT
    department,
    COUNT(*) as emp_count,
    AVG(salary) as avg_salary,
    PERCENTILE_APPROX(salary, 0.5) as median_salary
  FROM employees
  JOIN departments ON employees.dept_id = departments.id
  WHERE hire_date >= '2020-01-01'
  GROUP BY department
  ORDER BY avg_salary DESC
""").show()
```

---

## Configuration Reference

### Core Configuration

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.plugins` | (none) | `org.apache.gluten.GlutenPlugin` | Enable Gluten plugin |
| `spark.memory.offHeap.enabled` | `false` | `true` | Required for native execution |
| `spark.memory.offHeap.size` | `0` | `executor-memory * 0.3` | Off-heap memory for native engine (e.g., 20g for 64g executor) |
| `spark.shuffle.manager` | `org.apache.spark.shuffle.sort.SortShuffleManager` | `org.apache.spark.shuffle.sort.ColumnarShuffleManager` | Columnar shuffle manager for native shuffle |

### Gluten SQL Execution

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.gluten.sql.enabled` | `true` | `true` | Master switch for Gluten SQL acceleration |
| `spark.gluten.sql.columnar.shuffle.enabled` | `false` | `true` | Enable columnar shuffle (significant performance impact) |
| `spark.gluten.sql.columnar.shuffle.codec` | `lz4` | `lz4` | Shuffle compression codec: `lz4`, `zstd`, `snappy` |
| `spark.gluten.sql.fallback.enforced` | `false` | `false` | Set `true` to force all operations to fall back to JVM (debugging) |
| `spark.gluten.sql.native.writer.enabled` | `true` | `true` | Enable native file writer |
| `spark.gluten.sql.validation.enabled` | `false` | `false` | Validate native results match JVM results (debugging; adds overhead) |
| `spark.gluten.sql.validation.mode` | `CORRECTNESS` | `CORRECTNESS` | Validation mode: `CORRECTNESS`, `PERFORMANCE` |

### Memory Management

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.gluten.memory.mode` | `unsafe` | `unsafe` | Memory mode: `unsafe` (direct, fastest) or `safe` (managed, slower) |
| `spark.gluten.memory.dynamic.enabled` | `false` | `true` (production) | Enable dynamic off-heap memory sizing based on query demand |
| `spark.gluten.memory.dynamic.overallocationFactor` | `1.0` | `1.2` | Over-allocation factor for dynamic memory |
| `spark.gluten.memory.retain.size` | `0` | `0` | Bytes of native memory to retain between queries |
| `spark.gluten.memory.allocator.policy` | `default` | `default` | Memory allocation policy: `default`, `caching`, `arena` |

### Operator Control

| Property | Default | Description |
|----------|---------|-------------|
| `spark.gluten.sql.native.operator.*` | `true` | Fine-grained control per native operator; set to `false` to force JVM fallback |
| `spark.gluten.sql.columnar.sqlMetrics` | `true` | Enable native SQL metrics in Spark UI |
| `spark.gluten.sql.explain.mode` | `NONE` | `ALL` prints native/JVM plan decisions to stderr |
| `spark.gluten.sql.backend` | `auto` | Backend selection: `auto`, `velox`, `clickhouse` |

### Performance Tuning

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.sql.files.maxPartitionBytes` | `128m` | `256m` | Larger partitions reduce task scheduling overhead for native execution |
| `spark.sql.adaptive.enabled` | `true` | `true` | AQE works with Gluten; native AQE supported |
| `spark.sql.adaptive.coalescePartitions.enabled` | `true` | `true` | Recommended with Gluten |
| `spark.gluten.sql.columnar.maxBatchSize` | `4096` | `4096` | Max rows per ColumnarBatch for native processing |
| `spark.gluten.velox.cast.expr.enabled` | `true` | `true` | Enable native cast expression execution |
| `spark.gluten.velox.fs.enabled` | `false` | `false` | Enable Velox native file system I/O |
| `spark.gluten.velox.fs.cache.enabled` | `false` | `true` (read-heavy) | Enable Velox file cache |
| `spark.gluten.velox.fs.cache.size` | `1g` | `4g`-`8g` | Velox file cache size |
| `spark.gluten.velox.http.enabled` | `false` | `false` | Enable Velox HTTP transport for shuffle |

### Debugging

| Property | Default | Description |
|----------|---------|-------------|
| `spark.gluten.sql.explain.mode` | `NONE` | `NONE`, `SIMPLE`, `ALL` -- verbosity of plan explanation |
| `spark.gluten.log.level` | `INFO` | `DEBUG` for detailed native execution logs |
| `spark.gluten.sql.native.listener.enabled` | `false` | `true` to enable native process debugging |
| `spark.gluten.sql.validation.throwException` | `false` | `true` to throw on result mismatch (debugging) |

---

## Supported Operations Matrix (Velox Backend)

### SQL Operators (Exec Level)

| Operator | Native Support | Notes |
|----------|---------------|-------|
| **Project** | Full | Arithmetic, expressions, case statements, casts |
| **Filter** | Full | All comparison and boolean operators |
| **Sort** | Full | Single and multi-column, null ordering |
| **Hash Aggregate** | Full | All aggregate functions including partial |
| **Sort Aggregate** | Full | Streaming aggregation |
| **Broadcast Hash Join** | Full | Inner, left, right, full outer, semi, anti |
| **Shuffled Hash Join** | Full | All join types |
| **Sort Merge Join** | Full | All join types |
| **Cartesian Product** | Partial | Native support in recent versions |
| **Window** | Full | Row/range frames, ranking, window aggregates |
| **Union** | Full | |
| **Expand** | Full | Grouping sets, rollup, cube |
| **Generate** | Partial | `explode` supported; limited UDTF support |
| **Limit** | Full | |
| **Sample** | Full | |
| **Coalesce** | Full | |
| **Collect Limit** | Full | |
| **TakeOrderedAndProject** | Full | |
| **Subquery (Broadcast)** | Full | DPP-compatible |
| **Subquery (In-Set)** | Partial | |

### File Formats

| Format | Native Read | Native Write | Notes |
|--------|-------------|--------------|-------|
| Parquet | Yes | Yes | Full support including nested types |
| ORC | Yes | Yes | |
| CSV | Yes | Yes | |
| JSON | Partial | No | Basic JSON read supported |
| Delta Lake | Partial | Partial | Depends on Delta version |
| Iceberg | Partial | Partial | Community support improving |
| Text | Yes | Yes | |

### Aggregate Functions

| Function | Native Support | Notes |
|----------|---------------|-------|
| `COUNT` | Full | `COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT col)` |
| `SUM` | Full | Numeric types |
| `AVG` | Full | |
| `MIN`/`MAX` | Full | All comparable types |
| `STDDEV` | Full | `STDDEV_POP`, `STDDEV_SAMP` |
| `VARIANCE` | Full | `VAR_POP`, `VAR_SAMP` |
| `APPROX_COUNT_DISTINCT` | Full | HyperLogLog |
| `COLLECT_LIST` | Full | |
| `COLLECT_SET` | Full | |
| `FIRST`/`LAST` | Full | With and without ignoreNulls |
| `PERCENTILE` | Full | `PERCENTILE_APPROX`, `PERCENTILE_CONT` |
| `BIT_AND`/`BIT_OR`/`BIT_XOR` | Full | |
| `BOOL_AND`/`BOOL_OR` | Full | |
| `EVERY`/`ANY` | Full | |
| `REGR_*` (regression) | Full | `REGR_SLOPE`, `REGR_INTERCEPT`, etc. |
| `CORR`/`COVAR_*` | Full | |
| `HISTOGRAM_NUMERIC` | Full | |

### Expressions

| Category | Native Support | Examples |
|----------|---------------|----------|
| **Arithmetic** | Full | `+`, `-`, `*`, `/`, `%`, `pmod`, `abs`, `round`, `ceil`, `floor`, `signum` |
| **Comparison** | Full | `=`, `!=`, `<`, `>`, `<=`, `>=`, `<=>`, `IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL` |
| **Boolean** | Full | `AND`, `OR`, `NOT`, `XOR` |
| **String** | Most | `concat`, `concat_ws`, `substr`, `trim`, `ltrim`, `rtrim`, `lower`, `upper`, `length`, `replace`, `split`, `reverse`, `lpad`, `rpad`, `repeat`, `translate` |
| **String (Regex)** | Partial | `regexp_extract`, `regexp_replace` (limited patterns) |
| **Date/Time** | Most | `date_add`, `date_sub`, `datediff`, `year`, `month`, `day`, `hour`, `minute`, `second`, `unix_timestamp`, `from_unixtime`, `to_date`, `to_timestamp`, `date_format` |
| **Conditional** | Full | `CASE WHEN`, `IF`, `COALESCE`, `NULLIF`, `NVL`, `NVL2` |
| **Bitwise** | Full | `&`, `|`, `^`, `~`, `<<`, `>>`, `bit_count`, `shiftleft`, `shiftright` |
| **Array** | Most | `size`, `element_at`, `array_contains`, `sort_array`, `arrays_overlap`, `array_union`, `array_except`, `array_intersect`, `array_join`, `array_position`, `array_remove`, `array_repeat` |
| **Map** | Partial | `size`, `keys`, `values`, `map_keys`, `map_values`, `element_at` |
| **Struct** | Partial | Field access, `named_struct` |
| **Cast** | Full | Numeric, string, date/time, boolean casts |
| **Hash** | Full | `hash`, `md5`, `sha1`, `sha2`, `crc32`, `xxhash64` |
| **JSON** | Partial | `get_json_object`, `from_json` (limited) |
| **Window Functions** | Full | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`, window aggregates |
| **Decimal** | Full | Decimal arithmetic and comparisons |
| **Lambda** | Partial | `transform`, `filter`, `exists`, `forall` on arrays |

### Columnar Shuffle

| Feature | Support | Notes |
|---------|---------|-------|
| Hash partitioning | Full | Default shuffle |
| Range partitioning | Full | With sort |
| Round-robin | Full | |
| Broadcast | Full | Native broadcast exchange |
| Compression | Full | LZ4, ZSTD, SNAPPY |
| Encryption | Partial | Depends on Spark version |

### Unsupported (JVM Fallback)

Operations that typically fall back to vanilla Spark:
- **ANSI mode operations**: When `spark.sql.ansi.enabled=true`, many operations fall back
- **Complex regex patterns**: Backreferences, named groups
- **Some Hive UDFs**: Legacy Hive functions not yet ported to native
- **Custom Scala/Python UDFs**: Must execute in JVM/Python
- **Certain subquery patterns**: Correlated subqueries with complex conditions
- **MergeInto/Update/Delete**: DML operations (read-only supported)
- **Some struct operations**: Deeply nested struct manipulation

---

## Verification

### Query Plan Inspection

Check that operators are running natively by examining the physical plan:

```scala
// Enable verbose explain
spark.conf.set("spark.gluten.sql.explain.mode", "ALL")

// Check the plan - look for native operator indicators
val df = spark.read.parquet("/data/large-dataset/")
df.groupBy($"category").agg(sum($"amount")).explain(true)
```

**Expected output** (Gluten-accelerated plan):

```
== Physical Plan ==
AdaptiveSparkPlan (4)
+- ColumnarToRow (3)
   +- VeloxColumnarToRow
      +- VeloxColumnarExchange
         +- VeloxColumnarHashAggregate (2)
            +- VeloxColumnarHashAggregate (1)
               +- VeloxColumnarFileScan parquet
```

Key indicators of native execution:
- `VeloxColumnarFileScan` instead of `FileScan` (native Parquet/ORC read)
- `VeloxColumnarHashAggregate` instead of `HashAggregate` (native aggregation)
- `VeloxColumnarExchange` instead of `ColumnarExchange` (native shuffle)
- `Transformer` nodes in the plan (plan transformation markers)
- `VeloxColumnarToRow` at the boundary between native and JVM execution

If you see fallback markers, the `ALL` explain mode shows why:

```
== Gluten Fallback Operators ==
<SortExec> fell back to JVM because: ANSI mode is enabled
  <expression> ComplexRegex not supported in native engine
```

### Spark UI Verification

In the Spark UI (`http://<driver>:4040`):

1. **SQL Tab**: Physical plan shows `VeloxColumnar*` or `Transformer` nodes for native execution
2. **Stage Details**: Native stages show `ColumnarShuffleManager` and reduced execution times
3. **SQL Metrics**: Native metrics display native operator timing (when `spark.gluten.sql.columnar.sqlMetrics=true`)
4. **Environment Tab**: Verify `spark.plugins=org.apache.gluten.GlutenPlugin` is listed

### Programmatic Verification

```scala
// Check if Gluten is active
val glutenEnabled = spark.conf.get("spark.gluten.sql.enabled", "true").toBoolean
println(s"Gluten SQL enabled: $glutenEnabled")

// Check off-heap memory configuration
val offHeapEnabled = spark.conf.get("spark.memory.offHeap.enabled").toBoolean
val offHeapSize = spark.conf.get("spark.memory.offHeap.size")
println(s"Off-heap: enabled=$offHeapEnabled, size=$offHeapSize")

// Check shuffle manager
val shuffleManager = spark.conf.get("spark.shuffle.manager")
println(s"Shuffle manager: $shuffleManager")
```

---

## Performance Benchmarks

### TPC-H (Velox Backend)

| Metric | Value |
|--------|-------|
| **Overall speedup** | 3.34x |
| **Best individual query** | 23.45x |
| **Queries with speedup** | 21/22 |

### TPC-DS (Velox Backend)

| Metric | Value |
|--------|-------|
| **Overall speedup** | 3.02x |
| **Best individual query** | 13.75x |
| **Queries with speedup** | 93/99 |

These benchmarks were run on identical hardware comparing vanilla Spark vs. Spark + Gluten/Velox. Results vary based on data distribution, cluster size, and query complexity.

---

## Best Practices

### When Gluten Is the Right Choice

- **Analytical queries on CPU clusters**: TPC-H/DS style workloads with scans, joins, aggregations
- **Cost-sensitive environments**: 3x throughput improvement on existing CPU hardware
- **GPU-unavailable environments**: Cloud instances without GPU support, on-prem CPU-only clusters
- **JVM GC pressure**: Workloads suffering from GC pauses and heap fragmentation
- **Large-scale ETL**: Multi-terabyte data pipelines with heavy SQL transformations
- **Mixed workloads**: Coexistence with other JVM applications on the same cluster

### When Gluten Is NOT Appropriate

- **ANSI mode required**: Many native operators fall back when `spark.sql.ansi.enabled=true`
- **Heavy custom UDFs**: Custom Scala/Python UDFs cannot be offloaded to native
- **Graph algorithms**: Not columnar-processing-friendly
- **Small interactive queries**: Native engine warmup adds latency for sub-second queries
- **Streaming micro-batches**: Very small batch sizes may not amortize JNI overhead
- **ARM64 production**: Limited testing and optimization for ARM architectures
- **Workloads requiring full operator coverage**: If your queries hit many unsupported operators, fallback overhead may negate benefits

### Production Deployment Checklist

- [ ] Set `spark.memory.offHeap.enabled=true` and `spark.memory.offHeap.size` to 30% of executor memory
- [ ] Set `spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager`
- [ ] Enable `spark.gluten.sql.columnar.shuffle.enabled=true` for shuffle-heavy workloads
- [ ] Use `spark.gluten.sql.explain.mode=ALL` in development to verify native execution
- [ ] Test with production data volumes before cutover
- [ ] Enable `spark.gluten.memory.dynamic.enabled=true` for variable workload patterns
- [ ] Monitor off-heap memory usage; adjust `spark.memory.offHeap.size` as needed
- [ ] Disable ANSI mode for maximum native operator coverage (if business logic permits)
- [ ] Use LZ4 shuffle compression for best performance; ZSTD for compression ratio
- [ ] Verify native `.so` library is present on all executor nodes (dynamic build only)

### Performance Tuning

```bash
# Aggressive tuning (large executors, shuffle-heavy workloads)
--conf spark.memory.offHeap.size=32g
--conf spark.gluten.sql.columnar.shuffle.enabled=true
--conf spark.gluten.sql.columnar.shuffle.codec=lz4
--conf spark.gluten.memory.dynamic.enabled=true
--conf spark.gluten.memory.dynamic.overallocationFactor=1.2
--conf spark.sql.files.maxPartitionBytes=256m
--conf spark.gluten.velox.fs.cache.enabled=true
--conf spark.gluten.velox.fs.cache.size=8g

# Conservative tuning (moderate executors, mixed workloads)
--conf spark.memory.offHeap.size=16g
--conf spark.gluten.sql.columnar.shuffle.enabled=true
--conf spark.gluten.sql.columnar.shuffle.codec=lz4
--conf spark.gluten.memory.mode=unsafe
--conf spark.sql.files.maxPartitionBytes=128m

# Minimal overhead (testing/validation)
--conf spark.memory.offHeap.size=8g
--conf spark.gluten.sql.columnar.shuffle.enabled=false
--conf spark.gluten.memory.mode=safe
--conf spark.gluten.sql.validation.enabled=true
```

### Off-Heap Memory Sizing

Rule of thumb: `spark.memory.offHeap.size = executor-memory * 0.3`

| Executor Memory | Recommended Off-Heap Size |
|-----------------|---------------------------|
| 16g | 4-6g |
| 32g | 10-12g |
| 64g | 20-24g |
| 128g | 40-48g |

Dynamic memory sizing (`spark.gluten.memory.dynamic.enabled=true`) automatically adjusts native memory allocation based on query demand, reducing the need for manual tuning.

---

## Limitations and Gotchas

### Known Limitations

1. **ANSI mode fallback**: When `spark.sql.ansi.enabled=true`, many arithmetic and cast operations fall back to vanilla Spark because native engine behavior differs from ANSI semantics
2. **No native Python UDF execution**: Python UDFs must execute in the Python interpreter; cannot be offloaded
3. **DML operations limited**: `INSERT`, `UPDATE`, `DELETE`, `MERGE INTO` have limited or no native support
4. **Dynamic partition pruning**: Supported but may have partial native coverage depending on the DPP plan shape
5. **AQE skew join handling**: Native skew join handling is improving but may fall back for complex skew patterns
6. **Hive compatibility**: Some Hive-compatible features and SerDes are not available in native mode
7. **Streaming**: Structured Streaming has partial support; continuous streaming mode not supported

### Common Gotchas

1. **Off-heap memory must be configured**: Without `spark.memory.offHeap.enabled=true` and a non-zero size, Gluten cannot execute natively and will silently fall back to JVM
2. **ColumnarShuffleManager required**: Columnar shuffle only works with `ColumnarShuffleManager`; using the default `SortShuffleManager` causes shuffle to fall back
3. **Silent JVM fallback**: Unsupported operators fall back to JVM without errors. Always check `explain` output in development
4. **Static vs dynamic build confusion**: Static builds bundle native code into the JAR (larger, simpler deployment). Dynamic builds require distributing `.so` files to all nodes
5. **Native library versioning**: The native `.so` must match the JAR version exactly; mixing versions causes cryptic errors
6. **Result validation overhead**: `spark.gluten.sql.validation.enabled=true` doubles execution time (runs both native and JVM paths and compares); use only for debugging
7. **Memory mode `safe` vs `unsafe`**: `unsafe` mode is faster but can cause segfaults on edge cases. `safe` mode adds safety checks with ~5-10% overhead
8. **Velox file cache requires warmup**: File cache shows benefits only after the first run populates the cache
9. **Spark 4.x compatibility**: Gluten supports Spark 4.x but some operators may have gaps due to API changes between 3.x and 4.x
10. **Native thread count**: Velox uses its own thread pool; high `spark.executor.cores` with small partitions may underutilize native threads

### Troubleshooting

#### Native Library Not Found

```bash
# For dynamic builds, verify .so is accessible
ls -la /usr/lib/libvelox_spark.so

# Check LD_LIBRARY_PATH on executors
spark-shell --conf spark.executorEnv.LD_LIBRARY_PATH=/usr/lib
```

#### All Operators Falling Back to JVM

```bash
# Check explain output for reasons
--conf spark.gluten.sql.explain.mode=ALL

# Verify off-heap memory is configured
spark-shell --conf spark.memory.offHeap.enabled=true --conf spark.memory.offHeap.size=8g

# Verify plugin is loaded
scala> Class.forName("org.apache.gluten.GlutenPlugin")
```

#### Off-Heap Memory Errors

```
java.lang.OutOfMemoryError: Direct buffer memory
```

1. Increase `spark.memory.offHeap.size`
2. Enable dynamic memory: `spark.gluten.memory.dynamic.enabled=true`
3. Reduce `spark.sql.files.maxPartitionBytes` to lower per-task memory pressure
4. Check for memory leaks: compare native memory usage across multiple queries

#### Performance Worse Than Vanilla Spark

1. Check for excessive JVM fallback: `spark.gluten.sql.explain.mode=ALL`
2. Verify columnar shuffle is enabled: `spark.gluten.sql.columnar.shuffle.enabled=true`
3. Verify `ColumnarShuffleManager` is set
4. Check partition sizes (256m recommended for native execution)
5. Disable validation mode: `spark.gluten.sql.validation.enabled=false`
6. Ensure off-heap memory is sized appropriately (30% of executor memory)

---

## Adopters

Apache Gluten is used in production by major cloud providers and enterprises:

| Organization | Usage |
|--------------|-------|
| **Microsoft Fabric** | Integrated into Fabric Spark runtime as a transparent optimization layer |
| **Google Cloud Dataproc** | Available as an optional component for Dataproc clusters |
| **IBM** | Integrated into IBM Cloud Pak for Data and Analytics services |
| **Alibaba Cloud** | Used in MaxCompute and E-MapReduce services |
| **Tencent Cloud** | Deployed for internal analytical workloads |

These adopters contribute back to the Apache Gluten project, ensuring ongoing development and enterprise-grade reliability.

---

## Production-Ready Python (PySpark) Code

### Gluten Session Factory

```python
from dataclasses import dataclass, field
from typing import Optional
import json
import logging
import os
import re
import time
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gluten_pyspark")


def create_gluten_session(
    *,
    app_name: str = "GlutenAcceleratedApp",
    off_heap_size: str = "20g",
    shuffle_codec: str = "lz4",
    memory_mode: str = "unsafe",
    dynamic_memory: bool = True,
    velox_file_cache: bool = False,
    velox_file_cache_size: str = "4g",
    max_partition_bytes: str = "256m",
    columnar_max_batch_size: int = 4096,
    extra_confs: Optional[dict[str, str]] = None,
) -> SparkSession:
    """Create a SparkSession configured with Gluten plugin and Velox backend.

    Args:
        app_name: Application name shown in Spark UI.
        off_heap_size: Off-heap memory size (e.g. '20g').
        shuffle_codec: Compression codec for columnar shuffle.
        memory_mode: 'unsafe' (fastest) or 'safe' (with safety checks).
        dynamic_memory: Enable dynamic off-heap memory sizing.
        velox_file_cache: Enable Velox file cache.
        velox_file_cache_size: Size of Velox file cache.
        max_partition_bytes: Max bytes per file partition.
        columnar_max_batch_size: Max rows per ColumnarBatch.
        extra_confs: Additional Spark configurations.

    Returns:
        Configured SparkSession with Gluten plugin enabled.

    Raises:
        RuntimeError: If off-heap memory is not enabled.
    """
    conf = {
        # --- Plugin registration ---
        "spark.plugins": "org.apache.gluten.GlutenPlugin",
        "spark.gluten.sql.enabled": "true",
        "spark.gluten.sql.backend": "velox",

        # --- Off-heap memory (required for native execution) ---
        "spark.memory.offHeap.enabled": "true",
        "spark.memory.offHeap.size": off_heap_size,

        # --- Columnar shuffle ---
        "spark.shuffle.manager":
            "org.apache.spark.shuffle.sort.ColumnarShuffleManager",
        "spark.gluten.sql.columnar.shuffle.enabled": "true",
        "spark.gluten.sql.columnar.shuffle.codec": shuffle_codec,

        # --- Memory management ---
        "spark.gluten.memory.mode": memory_mode,
        "spark.gluten.memory.dynamic.enabled": str(dynamic_memory).lower(),
        "spark.gluten.memory.dynamic.overallocationFactor": "1.2",

        # --- Performance tuning ---
        "spark.sql.files.maxPartitionBytes": max_partition_bytes,
        "spark.gluten.sql.columnar.maxBatchSize": str(columnar_max_batch_size),
        "spark.gluten.velox.fs.cache.enabled": str(
            velox_file_cache
        ).lower(),
        "spark.gluten.velox.fs.cache.size": velox_file_cache_size,

        # --- SQL / native writer ---
        "spark.gluten.sql.native.writer.enabled": "true",
        "spark.gluten.sql.columnar.sqlMetrics": "true",

        # --- AQE ---
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
    }

    if extra_confs:
        conf.update(extra_confs)

    builder = SparkSession.builder.appName(app_name)
    for key, value in conf.items():
        builder = builder.config(key, value)

    session = builder.getOrCreate()
    logger.info(
        "Gluten session created: app=%s, off_heap=%s, memory_mode=%s",
        app_name,
        off_heap_size,
        memory_mode,
    )
    return session
```

### Verify Gluten Is Active

```python
def verify_gluten_active(spark: SparkSession) -> dict[str, bool]:
    """Check whether Gluten is active and parsing EXPLAIN for native operators.

    Runs a small query, captures EXPLAIN output, and searches for Gluten /
    Velox operator signatures.

    Args:
        spark: Active SparkSession.

    Returns:
        Dict with keys:
          - plugin_loaded: whether spark.plugins contains GlutenPlugin
          - off_heap_enabled: whether off-heap memory is configured
          - columnar_shuffle: whether columnar shuffle is enabled
          - native_operators_found: whether Velox operators appear in EXPLAIN
          - operators: list of Velox operator names found
    """
    result: dict[str, bool | list[str]] = {
        "plugin_loaded": False,
        "off_heap_enabled": False,
        "columnar_shuffle": False,
        "native_operators_found": False,
        "operators": [],
    }

    # --- Configuration checks ---
    try:
        plugins = spark.conf.get("spark.plugins", "")
        result["plugin_loaded"] = "GlutenPlugin" in plugins
    except Exception:
        logger.warning("Could not read spark.plugins")

    try:
        off_heap = spark.conf.get("spark.memory.offHeap.enabled", "false")
        result["off_heap_enabled"] = off_heap.lower() == "true"
    except Exception:
        logger.warning("Could not read off-heap config")

    try:
        shuffle_mgr = spark.conf.get("spark.shuffle.manager", "")
        result["columnar_shuffle"] = "ColumnarShuffleManager" in shuffle_mgr
    except Exception:
        logger.warning("Could not read shuffle manager config")

    # --- EXPLAIN parsing ---
    velox_pattern = re.compile(
        r"(Velox\w+|Gluten\w+|Transformer|ColumnarToRow)",
    )
    try:
        spark.conf.set("spark.gluten.sql.explain.mode", "ALL")
        plan = spark.sql(
            """
            SELECT category, SUM(amount) AS total
            FROM (
                SELECT 'A' AS category, 100 AS amount
            ) t
            GROUP BY category
            """
        )._jdf.queryExecution().executedPlan().toString()
        plan_text = str(plan) if plan else ""
        operators = velox_pattern.findall(plan_text)
        result["operators"] = list(set(operators))
        result["native_operators_found"] = len(operators) > 0
    except Exception as exc:
        logger.warning("EXPLAIN parse failed: %s", exc)

    logger.info("Gluten verification result: %s", json.dumps(result))
    return result  # type: ignore[return-value]
```

### Gluten Benchmark

```python
@dataclass
class GlutenBenchmarkResult:
    """Benchmark comparison between Gluten+Velox and vanilla Spark."""
    query_name: str
    gluten_time_seconds: float
    spark_time_seconds: float
    speedup_ratio: float
    gluten_rows: int
    spark_rows: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_faster(self) -> bool:
        return self.speedup_ratio > 1.0

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "gluten_time_seconds": round(self.gluten_time_seconds, 3),
            "spark_time_seconds": round(self.spark_time_seconds, 3),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "gluten_rows": self.gluten_rows,
            "spark_rows": self.spark_rows,
            "timestamp": self.timestamp,
        }


def gluten_benchmark(
    spark: SparkSession,
    *,
    query_name: str = "aggregation_benchmark",
    query: str = "",
    gluten_enabled: bool = True,
    data_path: Optional[str] = None,
) -> GlutenBenchmarkResult:
    """Benchmark Gluten+Velox vs vanilla Spark execution.

    Runs the same query with Gluten enabled and disabled, comparing
    wall-clock times and result row counts.

    Args:
        spark: Active SparkSession.
        query_name: Human-readable name for this benchmark run.
        query: SQL query to benchmark. If empty, uses a default aggregation.
        gluten_enabled: Whether Gluten is currently enabled in the session.
        data_path: Optional path to Parquet data. If given, creates a temp view.

    Returns:
        GlutenBenchmarkResult with timing and speedup metrics.
    """
    default_query = """
        SELECT category,
               SUM(amount) AS total,
               AVG(quantity) AS avg_qty,
               COUNT(*) AS cnt
        FROM benchmark_data
        WHERE value > 100
        GROUP BY category
        ORDER BY total DESC
    """
    sql = query if query else default_query

    if data_path:
        spark.read.parquet(data_path).createOrReplaceTempView("benchmark_data")

    def _timed_run() -> tuple[float, int]:
        start = time.monotonic()
        df = spark.sql(sql)
        rows = df.count()
        elapsed = time.monotonic() - start
        return elapsed, rows

    # --- Run with Gluten ---
    if gluten_enabled:
        spark.conf.set("spark.gluten.sql.enabled", "true")
    gluten_time, gluten_rows = _timed_run()
    logger.info("Gluten run: %.3fs, %d rows", gluten_time, gluten_rows)

    # --- Run without Gluten ---
    spark.conf.set("spark.gluten.sql.enabled", "false")
    spark_time, spark_rows = _timed_run()
    logger.info("Spark run: %.3fs, %d rows", spark_time, spark_rows)

    # --- Restore original state ---
    if gluten_enabled:
        spark.conf.set("spark.gluten.sql.enabled", "true")

    speedup = spark_time / gluten_time if gluten_time > 0 else float("inf")
    result = GlutenBenchmarkResult(
        query_name=query_name,
        gluten_time_seconds=gluten_time,
        spark_time_seconds=spark_time,
        speedup_ratio=speedup,
        gluten_rows=gluten_rows,
        spark_rows=spark_rows,
    )

    # --- Write JSON report ---
    report_path = os.path.join(
        "/tmp", f"gluten_benchmark_{query_name}_{int(time.time())}.json"
    )
    with open(report_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("Benchmark report written to %s", report_path)

    logger.info(
        "Benchmark %s: speedup=%.2fx (Gluten=%.3fs, Spark=%.3fs)",
        query_name,
        speedup,
        gluten_time,
        spark_time,
    )
    return result
```

### Monitor Gluten Metrics

```python
@dataclass
class GlutenMetrics:
    """Snapshot of Gluten runtime metrics."""
    timestamp: str
    columnar_conversion_rate: float
    native_memory_used_mb: float
    fallback_operators: list[str]
    total_queries_accelerated: int
    total_queries_fallback: int
    avg_speedup: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "columnar_conversion_rate": round(
                self.columnar_conversion_rate, 4
            ),
            "native_memory_used_mb": round(self.native_memory_used_mb, 1),
            "fallback_operators": self.fallback_operators,
            "total_queries_accelerated": self.total_queries_accelerated,
            "total_queries_fallback": self.total_queries_fallback,
            "avg_speedup": round(self.avg_speedup, 2),
        }


def monitor_gluten_metrics(spark: SparkSession) -> GlutenMetrics:
    """Collect current Gluten execution metrics.

    Parses Spark metrics and EXPLAIN output to determine:
      - Columnar conversion rate (native vs JVM operators)
      - Native memory usage from off-heap pool
      - Fallback operator names
      - Acceleration statistics

    Args:
        spark: Active SparkSession.

    Returns:
        GlutenMetrics dataclass with current metrics.
    """
    fallback_pattern = re.compile(
        r"<(\w+Exec)>\s*fell back",
    )
    velox_operators = re.compile(
        r"(Velox\w+)",
    )
    row_operators = re.compile(
        r"(ColumnarToRow|RowToColumnar)",
    )

    fallback_ops: list[str] = []
    velox_count = 0
    row_count = 0

    try:
        # Capture recent executed plan for analysis
        last_plan = spark._jdf.queryExecution().executedPlan().toString()
        plan_text = str(last_plan) if last_plan else ""

        velox_count = len(velox_operators.findall(plan_text))
        row_count = len(row_operators.findall(plan_text))
        fallback_ops = list(set(fallback_pattern.findall(plan_text)))
    except Exception as exc:
        logger.warning("Could not parse plan: %s", exc)

    total_ops = velox_count + row_count
    conversion_rate = (
        velox_count / total_ops if total_ops > 0 else 0.0
    )

    # Off-heap memory usage (approximate from Spark conf + JVM metrics)
    try:
        off_heap_size_str = spark.conf.get("spark.memory.offHeap.size", "0")
        # Parse sizes like '20g', '4096m'
        size_match = re.match(r"(\d+)\s*([gGmM])", off_heap_size_str)
        if size_match:
            val = int(size_match.group(1))
            unit = size_match.group(2).lower()
            off_heap_mb = val * 1024 if unit == "g" else val
        else:
            off_heap_mb = 0
    except Exception:
        off_heap_mb = 0

    metrics = GlutenMetrics(
        timestamp=datetime.utcnow().isoformat(),
        columnar_conversion_rate=conversion_rate,
        native_memory_used_mb=off_heap_mb,
        fallback_operators=fallback_ops,
        total_queries_accelerated=velox_count,
        total_queries_fallback=len(fallback_ops),
        avg_speedup=0.0,  # updated after benchmark runs
    )

    logger.info("Gluten metrics: %s", json.dumps(metrics.to_dict()))
    return metrics
```

### Velox Optimization Config & Tuning

```python
@dataclass
class VeloxOptimizationConfig:
    """Structured configuration for Velox-specific optimizations."""
    # Memory
    off_heap_size: str = "20g"
    memory_mode: str = "unsafe"
    dynamic_memory: bool = True
    overallocation_factor: float = 1.2

    # Shuffle
    columnar_shuffle: bool = True
    shuffle_codec: str = "lz4"

    # Velox features
    velox_file_cache: bool = False
    velox_file_cache_size: str = "4g"
    velox_cast_enabled: bool = True
    velox_fs_enabled: bool = False

    # Batch / partitioning
    max_batch_size: int = 4096
    max_partition_bytes: str = "256m"

    # AQE
    aqe_enabled: bool = True
    aqe_coalesce_partitions: bool = True

    def to_conf_dict(self) -> dict[str, str]:
        """Convert to Spark configuration key-value pairs."""
        return {
            "spark.memory.offHeap.enabled": "true",
            "spark.memory.offHeap.size": self.off_heap_size,
            "spark.gluten.memory.mode": self.memory_mode,
            "spark.gluten.memory.dynamic.enabled": str(
                self.dynamic_memory
            ).lower(),
            "spark.gluten.memory.dynamic.overallocationFactor": str(
                self.overallocation_factor
            ),
            "spark.gluten.sql.columnar.shuffle.enabled": str(
                self.columnar_shuffle
            ).lower(),
            "spark.gluten.sql.columnar.shuffle.codec": self.shuffle_codec,
            "spark.gluten.velox.fs.cache.enabled": str(
                self.velox_file_cache
            ).lower(),
            "spark.gluten.velox.fs.cache.size": self.velox_file_cache_size,
            "spark.gluten.velox.cast.expr.enabled": str(
                self.velox_cast_enabled
            ).lower(),
            "spark.gluten.velox.fs.enabled": str(
                self.velox_fs_enabled
            ).lower(),
            "spark.gluten.sql.columnar.maxBatchSize": str(
                self.max_batch_size
            ),
            "spark.sql.files.maxPartitionBytes": self.max_partition_bytes,
            "spark.sql.adaptive.enabled": str(self.aqe_enabled).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(
                self.aqe_coalesce_partitions
            ).lower(),
        }


def apply_velox_tuning(
    spark: SparkSession,
    config: Optional[VeloxOptimizationConfig] = None,
    profile: Optional[str] = None,
) -> VeloxOptimizationConfig:
    """Apply Velox-specific optimizations to a running SparkSession.

    Either apply a custom ``VeloxOptimizationConfig`` or use a named profile.

    Profiles:
        - 'aggressive': large executors, shuffle-heavy workloads
        - 'conservative': moderate executors, mixed workloads
        - 'testing': minimal overhead, validation mode on

    Args:
        spark: Active SparkSession.
        config: Custom optimization config. Ignored if profile is given.
        profile: Named tuning profile.

    Returns:
        The applied VeloxOptimizationConfig.
    """
    if profile:
        profiles = {
            "aggressive": VeloxOptimizationConfig(
                off_heap_size="32g",
                columnar_shuffle=True,
                shuffle_codec="lz4",
                dynamic_memory=True,
                overallocation_factor=1.2,
                max_batch_size=8192,
                max_partition_bytes="256m",
                velox_file_cache=True,
                velox_file_cache_size="8g",
            ),
            "conservative": VeloxOptimizationConfig(
                off_heap_size="16g",
                columnar_shuffle=True,
                shuffle_codec="lz4",
                memory_mode="unsafe",
                max_batch_size=4096,
                max_partition_bytes="128m",
            ),
            "testing": VeloxOptimizationConfig(
                off_heap_size="8g",
                columnar_shuffle=False,
                memory_mode="safe",
                max_batch_size=1024,
                max_partition_bytes="64m",
            ),
        }
        if profile not in profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Choose from: {list(profiles.keys())}"
            )
        config = profiles[profile]
    elif config is None:
        config = VeloxOptimizationConfig()

    conf_dict = config.to_conf_dict()
    for key, value in conf_dict.items():
        try:
            spark.conf.set(key, value)
        except Exception as exc:
            logger.warning("Failed to set %s=%s: %s", key, value, exc)

    logger.info(
        "Applied Velox tuning profile=%s, %d configs set",
        profile or "custom",
        len(conf_dict),
    )
    return config
```

### GlutenPipeline Class

```python
@dataclass
class PipelineReport:
    """End-to-end pipeline execution report."""
    pipeline_name: str
    status: str  # 'success', 'partial_fallback', 'failure'
    total_time_seconds: float
    stages_completed: int
    stages_total: int
    gluten_accelerated_stages: int
    fallback_stages: int
    error_message: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "stages_completed": self.stages_completed,
            "stages_total": self.stages_total,
            "gluten_accelerated_stages": self.gluten_accelerated_stages,
            "fallback_stages": self.fallback_stages,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class GlutenPipeline:
    """End-to-end native pipeline with fallback handling and performance reporting.

    Manages a sequence of SQL / DataFrame transformations, verifies Gluten
    acceleration at each stage, and produces a structured report.
    """

    def __init__(
        self,
        spark: SparkSession,
        pipeline_name: str = "gluten_pipeline",
    ) -> None:
        self.spark = spark
        self.pipeline_name = pipeline_name
        self._stages: list[dict] = []
        self._start_time: float = 0.0
        self._accelerated = 0
        self._fallback = 0

    def add_stage(self, name: str, sql: str) -> "GlutenPipeline":
        """Add a SQL stage to the pipeline.

        Args:
            name: Human-readable stage name.
            sql: SQL query to execute.

        Returns:
            Self for chaining.
        """
        self._stages.append({"name": name, "sql": sql})
        return self

    def add_df_stage(
        self,
        name: str,
        func: callable,
    ) -> "GlutenPipeline":
        """Add a DataFrame-function stage to the pipeline.

        Args:
            name: Human-readable stage name.
            func: Callable that takes (SparkSession, DataFrame) and returns
                  a new DataFrame. The DataFrame is passed from the previous
                  stage (or None for the first stage).
        """
        self._stages.append({"name": name, "df_func": func})
        return self

    def _check_acceleration(self) -> bool:
        """Check whether the last executed plan used native operators."""
        try:
            plan = str(
                self.spark._jdf.queryExecution()
                .executedPlan()
                .toString()
            )
            return bool(
                re.search(r"Velox\w+|Gluten\w+|Columnar", plan)
            )
        except Exception:
            return False

    def execute(self) -> PipelineReport:
        """Execute all pipeline stages sequentially.

        Returns:
            PipelineReport with execution summary.
        """
        self._start_time = time.monotonic()
        df: Optional[DataFrame] = None
        errors: list[str] = []

        for i, stage in enumerate(self._stages):
            stage_name = stage["name"]
            logger.info("Executing stage %d/%d: %s", i + 1, len(self._stages), stage_name)

            try:
                if "sql" in stage:
                    df = self.spark.sql(stage["sql"])
                elif "df_func" in stage:
                    df = stage["df_func"](self.spark, df)
                else:
                    errors.append(f"Stage '{stage_name}' has no SQL or df_func")
                    self._fallback += 1
                    continue

                # Trigger execution
                df.count()

                accelerated = self._check_acceleration()
                if accelerated:
                    self._accelerated += 1
                    logger.info("Stage '%s' accelerated by Gluten", stage_name)
                else:
                    self._fallback += 1
                    logger.warning(
                        "Stage '%s' fell back to JVM execution", stage_name
                    )

            except Exception as exc:
                msg = f"Stage '{stage_name}' failed: {exc}"
                logger.error(msg)
                errors.append(msg)
                self._fallback += 1

        elapsed = time.monotonic() - self._start_time
        completed = len(self._stages) - len(errors)

        if errors:
            status = "failure" if completed == 0 else "partial_fallback"
        elif self._fallback > 0:
            status = "partial_fallback"
        else:
            status = "success"

        report = PipelineReport(
            pipeline_name=self.pipeline_name,
            status=status,
            total_time_seconds=elapsed,
            stages_completed=completed,
            stages_total=len(self._stages),
            gluten_accelerated_stages=self._accelerated,
            fallback_stages=self._fallback,
            error_message="; ".join(errors) if errors else None,
        )

        # Write report to /tmp
        report_path = os.path.join(
            "/tmp",
            f"gluten_pipeline_{self.pipeline_name}_{int(time.time())}.json",
        )
        with open(report_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Pipeline report written to %s", report_path)

        return report
```

### Production ETL with Gluten Acceleration

```python
def run_production_gluten_etl(
    spark: SparkSession,
    *,
    source_path: str,
    output_path: str,
    partition_column: str = "date",
    tuning_profile: str = "conservative",
    enable_monitoring: bool = True,
) -> dict:
    """Run a production ETL pipeline with Gluten acceleration and monitoring.

    This function orchestrates a complete ETL workflow:
      1. Apply Velox tuning profile
      2. Verify Gluten is active
      3. Execute multi-stage ETL (read, transform, aggregate, write)
      4. Collect metrics and produce a JSON report

    Args:
        spark: Active SparkSession.
        source_path: Path to source Parquet data.
        output_path: Path for output Parquet files.
        partition_column: Column to partition output by.
        tuning_profile: Velox tuning profile name.
        enable_monitoring: Whether to collect runtime metrics.

    Returns:
        Dict with ETL summary and performance metrics.
    """
    logger.info("Starting production Gluten ETL: source=%s", source_path)

    # --- Step 1: Apply tuning ---
    try:
        config = apply_velox_tuning(spark, profile=tuning_profile)
    except Exception as exc:
        logger.error("Failed to apply Velox tuning: %s", exc)
        return {"status": "error", "message": str(exc)}

    # --- Step 2: Verify Gluten ---
    verification = verify_gluten_active(spark)
    if not verification.get("plugin_loaded"):
        logger.error("Gluten plugin is not loaded. Aborting ETL.")
        return {"status": "error", "message": "Gluten plugin not loaded"}

    # --- Step 3: Build pipeline ---
    pipeline = GlutenPipeline(spark, pipeline_name="production_etl")

    # Stage 1: Read and register temp view
    pipeline.add_stage(
        "register_source",
        f"CREATE OR REPLACE TEMP VIEW raw_data AS "
        f"SELECT * FROM parquet.`{source_path}`",
    )

    # Stage 2: Filter and transform
    pipeline.add_stage(
        "filter_transform",
        """
        SELECT
            category,
            amount,
            quantity,
            value,
            CASE
                WHEN value > 1000 THEN 'high'
                WHEN value > 100  THEN 'medium'
                ELSE 'low'
            END AS value_tier
        FROM raw_data
        WHERE amount > 0 AND quantity > 0
        """,
    )

    # Stage 3: Aggregate
    pipeline.add_stage(
        "aggregate",
        """
        SELECT
            category,
            value_tier,
            SUM(amount) AS total_amount,
            AVG(quantity) AS avg_quantity,
            COUNT(*) AS record_count,
            PERCENTILE_APPROX(value, 0.5) AS median_value,
            STDDEV(value) AS stddev_value
        FROM raw_data
        GROUP BY category, value_tier
        """,
    )

    # --- Step 4: Execute pipeline ---
    report = pipeline.execute()

    # --- Step 5: Write output ---
    try:
        agg_df = spark.sql("""
            SELECT * FROM (
                SELECT
                    category, value_tier, total_amount,
                    avg_quantity, record_count, median_value, stddev_value
                FROM raw_data
                GROUP BY category, value_tier
            )
        """)
        (
            agg_df.write.mode("overwrite")
            .partitionBy(partition_column)
            .parquet(output_path)
        )
        logger.info("Output written to %s", output_path)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        report.status = "failure"
        report.error_message = str(exc)

    # --- Step 6: Monitor metrics ---
    metrics = None
    if enable_monitoring:
        try:
            metrics = monitor_gluten_metrics(spark)
        except Exception as exc:
            logger.warning("Failed to collect metrics: %s", exc)

    # --- Build summary ---
    summary = {
        "status": report.status,
        "pipeline_report": report.to_dict(),
        "velox_config": config.to_conf_dict(),
        "gluten_verification": verification,
        "metrics": metrics.to_dict() if metrics else None,
    }

    # Write combined report
    report_path = os.path.join(
        "/tmp", f"gluten_etl_report_{int(time.time())}.json"
    )
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("ETL report written to %s", report_path)

    return summary
```

### Usage Examples

```python
# ── Example: Create session and verify ───────────────────────────────────────
# spark = create_gluten_session(
#     app_name="MyGlutenApp",
#     off_heap_size="16g",
#     velox_file_cache=True,
# )
# verification = verify_gluten_active(spark)
# print(f"Gluten active: {verification['native_operators_found']}")
# print(f"Operators found: {verification['operators']}")

# ── Example: Run benchmark ───────────────────────────────────────────────────
# result = gluten_benchmark(
#     spark,
#     query_name="tpch_q1",
#     query="SELECT ... FROM lineitem GROUP BY ...",
#     data_path="/data/tpch/lineitem/",
# )
# print(f"Speedup: {result.speedup_ratio:.2f}x")

# ── Example: Production ETL ──────────────────────────────────────────────────
# summary = run_production_gluten_etl(
#     spark,
#     source_path="s3://data-lake/raw/transactions/",
#     output_path="s3://data-lake/curated/transactions_agg/",
#     partition_column="txn_date",
#     tuning_profile="aggressive",
# )
# print(f"ETL status: {summary['status']}")
```
