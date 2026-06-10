# NVIDIA RAPIDS Accelerator for Apache Spark

## Overview

The NVIDIA RAPIDS Accelerator for Apache Spark is a **drop-in plugin** that transparently offloads Spark SQL and DataFrame operations to NVIDIA GPUs. It requires **zero code changes** -- existing Spark SQL, DataFrame, and Dataset code runs on GPU without modification.

**GitHub:** https://github.com/NVIDIA/spark-rapids
**Documentation:** https://docs.nvidia.com/spark-rapids/

### What It Is

The RAPIDS Accelerator is a Spark SQL plugin that intercepts the physical execution plan after Catalyst optimization and replaces supported operators with GPU equivalents powered by the [RAPIDS cuDF](https://github.com/rapidsai/cudf) library. Data is transferred to GPU memory and processed in parallel via CUDA kernels.

Key properties:
- **Drop-in replacement**: Add configuration flags and a JAR; no application code changes
- **Transparent fallback**: Unsupported operators silently fall back to CPU execution
- **Compatible with Spark optimizations**: Works alongside AQE, DPP, broadcast joins, and all built-in Spark optimizations
- **Ecosystem integration**: Pandas UDFs, spark-rapids-ml (MLlib), and Structured Streaming are supported

### Why It Works

The accelerator exploits the massive parallelism of NVIDIA GPUs for data-parallel workloads:

1. **Physical plan interception**: After Catalyst produces the optimized physical plan, a SparkStrategy scans for replaceable operators
2. **Operator substitution**: Supported executors are replaced with GPU counterparts (`GpuProject`, `GpuFileGpuScan`, `GpuHashAggregateExec`, `GpuSortExec`, etc.)
3. **Data transfer**: Input data is batched and transferred to GPU memory via PCIe/NVLink
4. **Parallel execution**: cuDF executes operations as CUDA kernels, processing thousands of rows in parallel
5. **Result transfer**: Processed data returns to the JVM as `ColumnarBatch` objects
6. **Mixed CPU/GPU execution**: Operators that cannot run on GPU execute on CPU in the same query plan

The performance gains come from:
- **SIMT parallelism**: GPUs execute the same instruction across thousands of threads simultaneously
- **Memory bandwidth**: GPU memory bandwidth (1-3 TB/s) exceeds CPU memory bandwidth (100-300 GB/s)
- **Columnar-native processing**: cuDF operates directly on columnar Arrow-compatible data
- **Reduced serialization**: Data stays in columnar format end-to-end on the GPU

### What Problem It Solves

| Problem | RAPIDS Solution |
|---------|----------------|
| Slow analytical queries on large datasets | GPU parallelism accelerates scans, joins, aggregations |
| CPU-bound Spark workloads bottlenecked by core count | Thousands of GPU threads vs. tens of CPU cores |
| ETL pipelines exceeding SLA windows | 3x-7x typical speedup, up to 100x for specific query patterns |
| ML training on Spark limited by CPU throughput | spark-rapids-ml provides GPU-accelerated MLlib |
| Pandas UDF overhead in Spark | GPU-accelerated pandas UDFs eliminate Python serialization bottleneck |
| Cost-inefficient CPU-only clusters | Fewer GPU nodes can replace larger CPU clusters |

---

## Installation and Deployment

### Prerequisites

- NVIDIA GPU with compute capability 7.0+ (Volta or newer recommended)
- NVIDIA driver >= 535.00
- CUDA 12.x
- Apache Spark 3.1.1 - 4.0.x (check [compatibility matrix](https://docs.nvidia.com/spark-rapids/release-matrix.html) for exact version support)
- Linux x86_64 or ARM64

### Current Version

As of mid-2026, the current stable release is **26.06.0** (following the YY.MM convention). Always verify the latest version on [GitHub Releases](https://github.com/NVIDIA/spark-rapids/releases).

### Download the JAR

```bash
# Download from Maven Central
wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.06.0/rapids-4-spark_2.12-26.06.0.jar

# Or build from source
git clone https://github.com/NVIDIA/spark-rapids.git
cd spark-rapids
mvn clean package -DskipTests
```

### YARN Deployment

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 4 \
  --executor-cores 8 \
  --executor-memory 32g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.task.resource.gpu.amount=0.125 \
  --conf spark.sql.files.maxPartitionBytes=512m \
  --jars rapids-4-spark_2.12-26.06.0.jar \
  --files getGpusResources.sh \
  your-application.jar
```

**`getGpusResources.sh`** (required by YARN for GPU discovery):

```bash
#!/bin/bash
# This script outputs GPU resources as JSON for YARN
# Place this file alongside your application
echo '{"addressedGPUs": [{"minorID": 0}]}'
```

For multi-GPU executors:

```bash
# 4 GPUs per executor, 8 tasks per GPU
--conf spark.executor.resource.gpu.amount=4 \
--conf spark.task.resource.gpu.amount=0.125 \
--conf spark.rapids.sql.concurrentGpuTasks=4
```

### Kubernetes Deployment

```yaml
# Spark application spec
apiVersion: "sparkoperator.k8s.io/v1beta2"
kind: SparkApplication
metadata:
  name: rapids-spark-app
spec:
  type: Scala
  mode: cluster
  image: "spark:3.5-rapids"
  sparkVersion: "3.5.0"
  driver:
    cores: 4
    memory: "16g"
  executor:
    cores: 8
    memory: "32g"
    instances: 4
    gpu:
      name: "nvidia.com/gpu"
      quantity: 1
  sparkConf:
    "spark.plugins": "com.nvidia.spark.SQLPlugin"
    "spark.rapids.sql.enabled": "true"
    "spark.executor.resource.gpu.amount": "1"
    "spark.task.resource.gpu.amount": "0.125"
    "spark.rapids.sql.concurrentGpuTasks": "1"
    "spark.sql.files.maxPartitionBytes": "512m"
  deps:
    jars:
      - "local:///opt/spark/jars/rapids-4-spark_2.12-26.06.0.jar"
```

**Custom Dockerfile** for Kubernetes:

```dockerfile
FROM apache/spark:3.5.0-scala2.12-java17-python3-ubuntu

# Install NVIDIA container toolkit dependencies
RUN apt-get update && apt-get install -y \
    wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-runtime-12-4 && \
    rm -rf /var/lib/apt/lists/*

# Copy RAPIDS JAR
COPY rapids-4-spark_2.12-26.06.0.jar /opt/spark/jars/
```

### Databricks Deployment

**Via Init Script** (recommended for production):

```bash
# /dbfs/databricks/scripts/rapids-init.sh
#!/bin/bash
VERSION="26.06.0"
SPARK_VERSION=$(spark-submit --version 2>&1 | grep -oP 'version \K[0-9]+\.[0-9]+')
SCALA_VERSION="2.12"

# Download RAPIDS JAR to cluster nodes
RAPIDS_JAR="/databricks/jars/rapids-4-spark_${SCALA_VERSION}-${VERSION}.jar"
if [ ! -f "$RAPIDS_JAR" ]; then
  wget -O "$RAPIDS_JAR" \
    "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_${SCALA_VERSION}/${VERSION}/rapids-4-spark_${SCALA_VERSION}-${VERSION}.jar"
fi

# Configure Spark
cat >> /databricks/spark/conf/spark-defaults.conf << EOF
spark.plugins com.nvidia.spark.SQLPlugin
spark.rapids.sql.enabled true
spark.executor.resource.gpu.amount 1
spark.task.resource.gpu.amount 0.125
spark.rapids.sql.concurrentGpuTasks 1
spark.sql.files.maxPartitionBytes 512m
spark.rapids.memory.pinnedPool.size 2g
EOF
```

**Via Cluster UI**:
1. Go to Compute > Create Cluster > Advanced Options
2. Under Init Scripts, add the path to your init script
3. Under Spark Config, add the configuration keys and values above

**Databricks Runtime with GPU**: Select a GPU-enabled instance type (e.g., `g5.2xlarge` on AWS) and choose the "GPU" runtime version which includes pre-installed RAPIDS support.

### AWS EMR Deployment

```bash
aws emr create-cluster \
  --name "RAPIDS Spark Cluster" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --instance-type g5.2xlarge \
  --instance-count 4 \
  --ec2-attributes KeyName=my-key,InstanceProfile=EMR_EC2_DefaultRole \
  --service-role EMR_DefaultRole \
  --configurations '[
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.plugins": "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled": "true",
        "spark.executor.resource.gpu.amount": "1",
        "spark.task.resource.gpu.amount": "0.125",
        "spark.rapids.sql.concurrentGpuTasks": "1",
        "spark.sql.files.maxPartitionBytes": "512m",
        "spark.rapids.memory.pinnedPool.size": "2g"
      }
    },
    {
      "Classification": "spark",
      "Properties": {
        "maximizeResourceAllocation": "true"
      }
    }
  ]' \
  --bootstrap-actions Path=s3://my-bucket/bootstrap-rapids.sh,Name="Install RAPIDS"
```

**Bootstrap action** (`bootstrap-rapids.sh`):

```bash
#!/bin/bash
set -ex
VERSION="26.06.0"
SCALA_VERSION="2.12"
JARS_DIR="/usr/lib/spark/jars/"

wget -O "${JARS_DIR}rapids-4-spark_${SCALA_VERSION}-${VERSION}.jar" \
  "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_${SCALA_VERSION}/${VERSION}/rapids-4-spark_${SCALA_VERSION}-${VERSION}.jar"

# Distribute to all executors
for host in $(awk '{print $1}' /mnt/var/lib/info/instances.json | grep -v "privateIp"); do
  scp "${JARS_DIR}rapids-4-spark_${SCALA_VERSION}-${VERSION}.jar" \
    "$host:${JARS_DIR}/" 2>/dev/null || true
done
```

### Google Cloud Dataproc Deployment

```bash
gcloud dataproc clusters create rapids-cluster \
  --region us-central1 \
  --master-machine-type n1-standard-8 \
  --master-boot-disk-size 200GB \
  --worker-machine-type n1-standard-8 \
  --worker-boot-disk-size 200GB \
  --worker-accelerator type=nvidia-tesla-t4,count=2 \
  --num-workers 4 \
  --image-version 2.2 \
  --optional-components SPARK \
  --initialization-actions gs://my-bucket/init-rapids.sh \
  --properties spark:spark.plugins=com.nvidia.spark.SQLPlugin,\
spark:spark.rapids.sql.enabled=true,\
spark:spark.executor.resource.gpu.amount=2,\
spark:spark.task.resource.gpu.amount=0.125,\
spark:spark.rapids.sql.concurrentGpuTasks=2,\
spark:spark.rapids.memory.pinnedPool.size=4g
```

### Azure Synapse Analytics

In Synapse Spark pool configuration:
1. Select GPU node size (e.g., `NCasT4_v3` series)
2. Add RAPIDS JAR as a workspace package
3. Configure Spark pool settings with the properties above

---

## Usage

### Basic Usage (Zero Code Change)

Your existing Spark SQL and DataFrame code runs on GPU without any modifications:

```scala
// Scala - existing code, now runs on GPU
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
# PySpark - existing code, now runs on GPU
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

### Pandas UDF Acceleration

RAPIDS accelerates Pandas UDFs using GPU vectorized execution:

```python
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import numpy as np

# Standard Pandas UDF - accelerated on GPU
@pandas_udf("double")
def gpu_compute_score(values: pd.Series) -> pd.Series:
    # This runs on GPU via cuDF
    return (values - values.mean()) / values.std()

df.withColumn("normalized_score", gpu_compute_score(df["value"])) \
  .show()

# Grouped map Pandas UDF
@pandas_udf("id long, value double, normalized double", PandasUDFType.GROUPED_MAP)
def gpu_normalize(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf["normalized"] = (pdf["value"] - pdf["value"].mean()) / pdf["value"].std()
    return pdf

df.groupby("category").apply(gpu_normalize).show()
```

### MLlib Acceleration via spark-rapids-ml

The [spark-rapids-ml](https://github.com/NVIDIA/spark-rapids-ml) library provides GPU-accelerated MLlib algorithms with zero code change for most estimators:

```python
from spark_rapids_ml.classification import RandomForestClassifier
from spark_rapids_ml.regression import LinearRegression
from spark_rapids_ml.clustering import KMeans

# Drop-in replacement for Spark MLlib
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy="sqrt"
)
model = rf.fit(training_data)

# Linear regression on GPU
lr = LinearRegression(
    regParam=0.01,
    maxIter=100
)
lr_model = lr.fit(training_data)

# K-Means clustering
kmeans = KMeans(k=10, maxIter=20)
km_model = kmeans.fit(training_data)
```

Install spark-rapids-ml:

```bash
pip install spark-rapids-ml
```

---

## Configuration Reference

### Core Configuration

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.plugins` | (none) | `com.nvidia.spark.SQLPlugin` | Enable the RAPIDS SQL plugin |
| `spark.rapids.sql.enabled` | `true` | `true` | Master switch for GPU acceleration |
| `spark.executor.resource.gpu.amount` | `0` | `1` | Number of GPUs per executor |
| `spark.task.resource.gpu.amount` | `1.0` | `1/executor-cores` | Fraction of GPU per task (controls parallelism) |

### GPU Memory Management

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.rapids.memory.pinnedPool.size` | `0` | `2g`-`4g` | Host pinned (page-locked) memory for faster CPU-GPU transfer |
| `spark.rapids.memory.gpu.allocFraction` | `1.0` | `0.95` | Fraction of GPU memory available for RAPIDS |
| `spark.rapids.memory.gpu.maxAllocFraction` | `1.0` | `0.95` | Maximum fraction of GPU memory for allocation |
| `spark.rapids.memory.gpu.minAllocFraction` | `0.25` | `0.25` | Minimum fraction; queries failing below this threshold are rejected |
| `spark.rapids.sql.batchSizeBytes` | `1g` | `512m`-`1g` | Max batch size for GPU operations; reduce if OOM occurs |
| `spark.rapids.sql.reader.batchSizeBytes` | `1g` | `512m` | Max batch size for file readers |

### GPU Task Scheduling

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.rapids.sql.concurrentGpuTasks` | `1` | `1`-`4` (depends on GPU memory) | Max concurrent GPU tasks per executor |
| `spark.rapids.sql.multiThreadedRead.numThreads` | `20` | `20`-`50` | Threads for concurrent file reads |
| `spark.rapids.sql.hashAgg.replace.mode` | `AUTO` | `AUTO` | Hash aggregate replacement mode |

### File I/O

| Property | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `spark.rapids.sql.format.parquet.reader.type` | `MULTITHREADED` | `MULTITHREADED` | Parquet read mode: COALESCING, MULTITHREADED, or PERFILE |
| `spark.rapids.sql.format.parquet.reader.numThreads` | `20` | `20` | Threads for coalescing parquet reader |
| `spark.rapids.sql.format.orc.enabled` | `true` | `true` | Enable ORC read on GPU |
| `spark.rapids.sql.format.csv.enabled` | `true` | `true` | Enable CSV read on GPU |
| `spark.rapids.sql.format.json.enabled` | `false` | `false` | Enable JSON read on GPU (limited support) |

### Expression and Operator Control

| Property | Default | Description |
|----------|---------|-------------|
| `spark.rapids.sql.expression.*` | varies | Fine-grained control per expression; set to `false` to force CPU fallback for specific operations |
| `spark.rapids.sql.exec.*` | varies | Fine-grained control per exec type |
| `spark.rapids.sql.hasNans` | `true` | Set to `false` if data has no NaN values for faster floating-point ops |
| `spark.rapids.sql.decimalOverflowGuaranteed` | `true` | Set to `false` to allow faster decimal operations when overflow is impossible |

### Debugging and Diagnostics

| Property | Default | Description |
|----------|---------|-------------|
| `spark.rapids.sql.explain` | `NONE` | `ALL` prints GPU/CPU plan decisions to stderr |
| `spark.rapids.sql.enabled` | `true` | Master toggle |
| `spark.rapids.sql.profiler.enabled` | `false` | Enable profiling output |
| `spark.rapids.sql.debug.log` | `false` | Enable debug logging for operator replacement |

---

## Supported Operations Matrix

### SQL Operators (Exec Level)

| Operator | GPU Support | Notes |
|----------|-------------|-------|
| **Project** | Full | Arithmetic, expressions, case statements |
| **Filter** | Full | All comparison and boolean operators |
| **Sort** | Full | Single and multi-column, with null ordering |
| **Hash Aggregate** | Full | Sum, count, avg, min, max, stddev, variance |
| **Sort Aggregate** | Full | Streaming aggregation |
| **Broadcast Hash Join** | Full | Inner, left, right, full outer, semi, anti |
| **Shuffled Hash Join** | Full | All join types |
| **Sort Merge Join** | Full | All join types |
| **Cartesian Product** | Full | |
| **Window** | Full | Row/range frames, ranking, aggregation windows |
| **Union** | Full | |
| **Expand** | Full | Grouping sets, rollup, cube |
| **Generate** | Partial | `explode`, `posexplode`; limited support for UDTFs |
| **Limit** | Full | |
| **Sample** | Full | |
| **Coalesce** | Full | |
| **Collect Limit** | Full | |
| **TakeOrderedAndProject** | Full | |

### File Formats

| Format | GPU Read | GPU Write | Notes |
|--------|----------|-----------|-------|
| Parquet | Yes | Yes | Full support, including nested types |
| ORC | Yes | Yes | Full support |
| CSV | Yes | Yes | |
| JSON | Partial | No | Limited nested JSON support |
| AVRO | Partial | No | Basic support |
| Delta Lake | Yes | Yes | With Delta Lake plugin |
| Iceberg | Partial | Partial | Depends on Iceberg version |
| Hudi | Partial | Partial | Community support |

### Expressions

| Category | GPU Support | Examples |
|----------|-------------|----------|
| **Arithmetic** | Full | `+`, `-`, `*`, `/`, `%`, `abs`, `round`, `ceil`, `floor` |
| **Comparison** | Full | `=`, `!=`, `<`, `>`, `<=`, `>=`, `IN`, `BETWEEN` |
| **Boolean** | Full | `AND`, `OR`, `NOT`, `XOR` |
| **String** | Most | `concat`, `substr`, `trim`, `lower`, `upper`, `replace`, `split`, `regexp` (partial) |
| **Date/Time** | Most | `date_add`, `date_sub`, `datediff`, `year`, `month`, `unix_timestamp` |
| **Aggregate** | Full | `sum`, `count`, `avg`, `min`, `max`, `stddev`, `variance`, `approx_count_distinct`, `collect_list`, `collect_set`, `first`, `last`, `percentile_approx` |
| **Conditional** | Full | `CASE WHEN`, `IF`, `COALESCE`, `NULLIF`, `NVL` |
| **Bitwise** | Full | `&`, `|`, `^`, `<<`, `>>`, `bit_count` |
| **Array** | Most | `size`, `element_at`, `array_contains`, `sort_array`, `explode` |
| **Map** | Partial | Basic map operations |
| **Struct** | Partial | Field access, creation |
| **Hash** | Full | `hash`, `md5`, `sha1`, `sha2`, `crc32` |
| **JSON** | Partial | `get_json_object`, `json_tuple` (limited) |
| **Window Functions** | Full | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, window aggregates |
| **Decimal** | Full | Decimal arithmetic and comparisons |
| **Cast** | Most | Numeric, string, date/time casts |

### Unsupported (CPU Fallback)

Operations that typically fall back to CPU:
- Complex regular expressions with backreferences
- Certain nested JSON parsing
- Custom UDFs (non-Pandas UDFs)
- Some Hive UDFs
- Certain subquery execution patterns
- Some complex struct operations

---

## Verification

### Query Plan Inspection

Check that operators are running on GPU by examining the physical plan:

```scala
// Enable verbose explain mode
spark.conf.set("spark.rapids.sql.explain", "ALL")

// Check the plan - look for Gpu* prefixes
val df = spark.read.parquet("/data/large-dataset/")
df.groupBy($"category").agg(sum($"amount")).explain(true)
```

**Expected output** (GPU-accelerated plan):

```
== Physical Plan ==
AdaptiveSparkPlan (4)
+- GpuHashAggregate (3)
   +- GpuShuffleCoalesce
      +- GpuColumnarExchange
         +- GpuHashAggregate (2)
            +- GpuFileGpuScan parquet (1)
```

The key indicators are:
- `GpuFileGpuScan` instead of `FileScan` (GPU reading Parquet/ORC)
- `GpuHashAggregate` instead of `HashAggregate` (GPU aggregation)
- `GpuProject` instead of `Project` (GPU projection)
- `GpuSortExec` instead of `SortExec` (GPU sorting)
- `GpuBroadcastHashJoinExec` instead of `BroadcastHashJoinExec`

If you see `! <operator>` markers, that operator fell back to CPU. The `ALL` explain mode shows why:

```
! <SortExec> cannot run on GPU because ...
  @ <expression> SomeExpression is not supported on GPU because ...
```

### Spark UI Verification

In the Spark UI (`http://<driver>:4040`):

1. **SQL Tab**: Look at the Physical Plan section -- GPU operators show `Gpu` prefix
2. **Stage Details**: GPU stages show significantly faster execution times
3. **Executor Tab**: GPU utilization metrics appear when RAPIDS is active

### Spark UI with Qualification Tool

After a run, check the RAPIDS qualification output:

```bash
# The qualification tool generates reports showing which queries benefit from GPU
java -cp rapids-4-spark-tools_2.12-26.06.0.jar \
  com.nvidia.spark.rapids.tool.qualification.QualificationMain \
  --output-directory /tmp/rapids-qual \
  /path/to/spark-event-log.json
```

This produces:
- `rapids_4_spark_qualification_output.csv` -- per-application GPU speedup estimates
- `rapids_4_spark_qualification_output.log` -- detailed operator analysis

### Metrics and Monitoring

Enable GPU metrics collection:

```scala
// In application code
spark.conf.set("spark.ui.prometheus.enabled", "true")
spark.conf.set("spark.metrics.conf.*.source.sink.servlet.class",
  "org.apache.spark.metrics.source.RapidsShuffleMetricsSource")
```

Check GPU utilization on the host:

```bash
# Monitor GPU usage during Spark job
watch -n 1 nvidia-smi

# Detailed GPU metrics
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free \
  --format=csv -l 1
```

---

## GPU OOM Handling

GPU memory is limited (typically 16-80 GB per GPU). When a query exceeds GPU memory:

### Symptoms
- `CudaException: out of memory` errors
- Jobs stuck or killed
- Partial results followed by failure

### Mitigation Strategies

**1. Reduce batch size:**

```bash
--conf spark.rapids.sql.batchSizeBytes=256m
--conf spark.rapids.sql.reader.batchSizeBytes=256m
```

**2. Reduce concurrent GPU tasks:**

```bash
--conf spark.rapids.sql.concurrentGpuTasks=1
```

**3. Increase spill-to-disk capability:**

```bash
--conf spark.rapids.memory.gpu.maxAllocFraction=0.9
--conf spark.locality.wait=0
```

**4. Enable operator-level spill:**

```bash
--conf spark.rapids.sql.hashAgg.replace.mode=COMPLETE
--conf spark.rapids.sql.sortAggregation.enabled=true
```

**5. Increase GPU memory (hardware):**
- Use A100 (40/80 GB) or H100 (80 GB) instead of T4 (16 GB)
- Enable multi-GPU per executor

### RMM (RAPIDS Memory Manager)

RMM manages GPU memory allocation. Key configurations:

```bash
# Arena-based memory allocator (recommended)
--conf spark.rapids.memory.gpu.pool=ARENA

# Use async allocator for CUDA 12+
--conf spark.rapids.memory.gpu.pool=ASYNC

# Pinned host memory for faster transfers
--conf spark.rapids.memory.pinnedPool.size=4g
```

---

## Qualification and Profiling Tools

### Qualification Tool

Assesses which Spark workloads benefit from GPU acceleration:

```bash
# Analyze Spark event logs
java -cp rapids-4-spark-tools_2.12-26.06.0.jar \
  com.nvidia.spark.rapids.tool.qualification.QualificationMain \
  --output-directory /tmp/qual-output \
  --order-by desc \
  --num-output-rows 50 \
  /path/to/eventlog1 /path/to/eventlog2

# Analyze running cluster via Spark History Server
java -cp rapids-4-spark-tools_2.12-26.06.0.jar \
  com.nvidia.spark.rapids.tool.qualification.QualificationMain \
  --output-directory /tmp/qual-output \
  --platform dataproc \
  --connection jdbc:spark://<history-server>:18080
```

Output includes estimated speedup per application and recommended GPU cluster sizing.

### Profiling Tool

Deep-dive performance analysis:

```bash
java -cp rapids-4-spark-tools_2.12-26.06.0.jar \
  com.nvidia.spark.rapids.tool.profiling.ProfilerMain \
  --output-directory /tmp/profile-output \
  /path/to/eventlog1
```

Generates detailed reports on:
- SQL plan operator execution times
- GPU vs. CPU breakdown
- Shuffle statistics
- Job and stage timelines

---

## Best Practices

### When RAPIDS Is the Right Choice

- **Large-scale ETL**: Multi-terabyte scans, joins, and aggregations
- **Data exploration/BI**: Repeated analytical queries over large datasets
- **Feature engineering**: DataFrame transformations at scale
- **ML preprocessing**: Data preparation pipelines before model training
- **Pandas UDF-heavy workloads**: GPU eliminates Python serialization overhead
- **Cost-sensitive workloads**: Fewer GPU nodes replacing larger CPU clusters

### When RAPIDS Is NOT Appropriate

- **Small datasets**: Overhead of GPU transfer outweighs computation benefits (< 10 GB)
- **Heavy custom UDFs**: Non-Pandas UDFs fall back to CPU; performance gains limited
- **Graph algorithms**: Not data-parallel; GPUs poorly suited
- **Highly selective queries**: If most data is filtered early, GPU parallelism underutilized
- **String-heavy workloads**: String operations have partial GPU support; performance gains modest
- **Interactive/low-latency queries**: GPU kernel launch latency adds 100-500ms overhead
- **Memory-bound operations exceeding GPU VRAM**: No GPU spill-to-disk for all operators

### Production Deployment Checklist

- [ ] Verify GPU driver and CUDA version compatibility
- [ ] Set `spark.rapids.memory.pinnedPool.size` to 2-4g for faster transfers
- [ ] Configure `spark.rapids.sql.concurrentGpuTasks` based on GPU memory
- [ ] Use `spark.rapids.sql.explain=ALL` in development to verify GPU usage
- [ ] Run qualification tool on existing workloads before migration
- [ ] Monitor GPU memory with `nvidia-smi` during peak load
- [ ] Set `spark.sql.files.maxPartitionBytes=512m` for optimal GPU batch sizes
- [ ] Test with production data volumes before cutover
- [ ] Configure alerts for GPU OOM errors
- [ ] Maintain CPU-only fallback path for unsupported operations

### Performance Tuning

```bash
# Aggressive tuning for A100/H100 GPUs
--conf spark.rapids.sql.batchSizeBytes=1g
--conf spark.rapids.sql.concurrentGpuTasks=4
--conf spark.rapids.memory.pinnedPool.size=4g
--conf spark.rapids.memory.gpu.pool=ASYNC
--conf spark.sql.files.maxPartitionBytes=512m
--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=32m

# Conservative tuning for T4 GPUs (16 GB)
--conf spark.rapids.sql.batchSizeBytes=256m
--conf spark.rapids.sql.concurrentGpuTasks=1
--conf spark.rapids.memory.pinnedPool.size=2g
--conf spark.rapids.memory.gpu.pool=ARENA
--conf spark.sql.files.maxPartitionBytes=256m
```

### Partitioning Strategy

- Target 128-512 MB partitions for GPU processing
- Coalesce small partitions: `df.coalesce(numPartitions)`
- Avoid too many tiny partitions (GPU kernel launch overhead)
- For file scans, use `MULTITHREADED` reader for many small files

---

## Limitations and Gotchas

### Known Limitations

1. **No GPU spill-to-disk for all operators**: Some operators (e.g., sorts, joins) may fail rather than spill when GPU memory is exceeded
2. **UDF limitations**: Only Pandas UDFs are GPU-accelerated; standard Python UDFs fall back to CPU
3. **Regex limitations**: Complex regular expressions with backreferences are not GPU-supported
4. **JSON parsing**: Limited nested JSON support; `from_json` may fall back to CPU
5. **Single-node GPU per task**: A task cannot span multiple GPUs
6. **Multi-GPU synchronization**: Requires explicit configuration for multi-GPU executors
7. **Kubernetes GPU scheduling**: Requires Kubernetes 1.23+ with GPU device plugins

### Common Gotchas

1. **Silent CPU fallback**: Unsupported operators fall back to CPU without errors. Always check `explain(true)` in development
2. **Driver-only GPU issues**: If the driver has no GPU but executors do, certain driver-side operations may fail
3. **Mixed Spark versions**: RAPIDS JAR must match the Spark version exactly
4. **Scala version mismatch**: Use `_2.12` JARs for Spark 3.x with Scala 2.12, `_2.13` for Spark 4.0+
5. **Container runtime**: NVIDIA Container Toolkit must be installed for Docker/Kubernetes deployments
6. **Pinned memory overhead**: Large `pinnedPool.size` reduces available host RAM for other operations
7. **AQE interaction**: AQE may change partition counts at runtime, affecting GPU batch sizes
8. **Shuffle compression**: GPU shuffle benefits from LZ4 compression; set `spark.shuffle.compress=true`

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify GPU visibility
nvidia-smi

# Check Spark GPU discovery
spark-shell --conf spark.executor.resource.gpu.amount=1
# In shell: spark.conf.get("spark.executor.resource.gpu.amount")
```

### All Operators Falling Back to CPU

```bash
# Check explain output for reasons
--conf spark.rapids.sql.explain=ALL

# Verify RAPIDS JAR is on classpath
spark-shell --jars rapids-4-spark_2.12-26.06.0.jar
scala> Class.forName("com.nvidia.spark.SQLPlugin")
```

### GPU Out of Memory

1. Check current GPU memory usage: `nvidia-smi`
2. Reduce `spark.rapids.sql.batchSizeBytes`
3. Reduce `spark.rapids.sql.concurrentGpuTasks`
4. Switch to ARENA or ASYNC memory pool

### Performance Worse Than CPU

1. Check for excessive CPU fallback: `spark.rapids.sql.explain=ALL`
2. Verify partition sizes are GPU-appropriate (128-512 MB)
3. Check for GPU memory thrashing (repeated allocation/deallocation)
4. Verify data formats (Parquet > CSV for GPU I/O)
5. Run qualification tool to assess workload suitability

## Production-Ready Python (PySpark) Code

### Factory: Create a RAPIDS-Optimized Session

```python
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rapids")


def create_rapids_session(
    gpu_memory_gb: int = 16,
    concurrent_gpu_tasks: int = 1,
    pinned_pool_size_gb: int = 2,
    max_partition_bytes_mb: int = 512,
    rapids_version: str = "26.06.0",
    app_name: str = "RapidsSpark",
    memory_pool: str = "ARENA",
    extra_conf: Optional[dict[str, str]] = None,
) -> SparkSession:
    """Build a SparkSession with NVIDIA RAPIDS Accelerator enabled.

    Parameters
    ----------
    gpu_memory_gb :
        GPU VRAM in GB (used to size batch sizes and concurrent tasks).
    concurrent_gpu_tasks :
        Max concurrent GPU tasks per executor.  For T4 (16 GB) use 1;
        for A100 (40-80 GB) use 2-4.
    pinned_pool_size_gb :
        Host pinned (page-locked) memory for faster CPU-GPU transfers.
    max_partition_bytes_mb :
        Target partition size in MB for file reads.
    rapids_version :
        RAPIDS plugin version string (informational).
    app_name :
        Spark application name.
    memory_pool :
        RMM memory pool allocator: ``ARENA`` or ``ASYNC`` (for CUDA 12+).
    extra_conf :
        Additional key=value config overrides.

    Returns
    -------
    SparkSession

    Notes
    -----
    The RAPIDS plugin JAR must already be on the classpath (via
    ``spark-submit --jars`` or a cluster init script).
    """
    # Batch size scales with GPU memory
    if gpu_memory_gb <= 16:
        batch_bytes = "256m"
        reader_batch = "256m"
    elif gpu_memory_gb <= 40:
        batch_bytes = "512m"
        reader_batch = "512m"
    else:
        batch_bytes = "1g"
        reader_batch = "1g"

    conf_map = {
        # Core RAPIDS plugin
        "spark.plugins": "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled": "true",

        # GPU resources
        "spark.executor.resource.gpu.amount": "1",
        "spark.task.resource.gpu.amount": "0.125",
        "spark.rapids.sql.concurrentGpuTasks": str(concurrent_gpu_tasks),

        # Memory management
        "spark.rapids.memory.pinnedPool.size": f"{pinned_pool_size_gb}g",
        "spark.rapids.memory.gpu.pool": memory_pool,
        "spark.rapids.sql.batchSizeBytes": batch_bytes,
        "spark.rapids.sql.reader.batchSizeBytes": reader_batch,

        # File I/O
        "spark.sql.files.maxPartitionBytes": f"{max_partition_bytes_mb}m",
        "spark.rapids.sql.format.parquet.reader.type": "MULTITHREADED",
        "spark.rapids.sql.multiThreadedRead.numThreads": "20",

        # NaN handling (set to False if data is clean for faster ops)
        "spark.rapids.sql.hasNans": "true",
    }
    if extra_conf:
        conf_map.update(extra_conf)

    builder = SparkSession.builder.appName(app_name)
    for k, v in conf_map.items():
        builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info(
        "RAPIDS session created: gpu=%dGB, tasks=%d, pinned=%dGB, "
        "pool=%s, batch=%s",
        gpu_memory_gb, concurrent_gpu_tasks, pinned_pool_size_gb,
        memory_pool, batch_bytes,
    )
    return session
```

### Verify RAPIDS Is Active

```python
@dataclass
class RapidsVerificationResult:
    """Structured result from verifying RAPIDS Accelerator status."""
    plugin_enabled: bool = False
    gpu_ops_found: int = 0
    cpu_fallbacks_found: int = 0
    gpu_operators: list[str] = field(default_factory=list)
    cpu_fallback_operators: list[str] = field(default_factory=list)
    raw_explain: str = ""
    notes: list[str] = field(default_factory=list)


def verify_rapids_active(
    spark: SparkSession, query: Optional[str] = None
) -> RapidsVerificationResult:
    """Check RAPIDS plugin status and parse EXPLAIN for GPU operators.

    Parameters
    ----------
    spark : Active SparkSession.
    query : SQL string to analyze.  If ``None``, a simple scan is used.

    Returns
    -------
    RapidsVerificationResult
    """
    result = RapidsVerificationResult()

    # Check plugin status
    plugin_value = spark.conf.get("spark.plugins", "")
    result.plugin_enabled = "com.nvidia.spark.SQLPlugin" in plugin_value

    if not result.plugin_enabled:
        result.notes.append(
            "RAPIDS plugin not found in spark.plugins. "
            "Ensure the JAR is on the classpath."
        )
        return result

    # Enable verbose explain
    original_explain = spark.conf.get("spark.rapids.sql.explain", "NONE")
    spark.conf.set("spark.rapids.sql.explain", "ALL")

    if query is None:
        query = "SELECT 1 AS x"

    plan = spark.sql(f"EXPLAIN {query}").collect()[0][0]
    result.raw_explain = plan
    spark.conf.set("spark.rapids.sql.explain", original_explain)

    # Parse GPU operators
    gpu_op_pattern = re.compile(r"(Gpu\w+)")
    gpu_matches = gpu_op_pattern.findall(plan)
    result.gpu_operators = list(set(gpu_matches))
    result.gpu_ops_found = len(result.gpu_operators)

    # Parse CPU fallbacks: "! <Operator> cannot run on GPU because"
    fallback_pattern = re.compile(r"!\s*<(\w+)>.*?cannot run on GPU", re.DOTALL)
    fallback_matches = fallback_pattern.findall(plan)
    result.cpu_fallback_operators = list(set(fallback_matches))
    result.cpu_fallbacks_found = len(result.cpu_fallback_operators)

    if result.gpu_ops_found > 0:
        result.notes.append(
            f"Found {result.gpu_ops_found} GPU operator types: "
            f"{', '.join(sorted(result.gpu_operators))}"
        )
    else:
        result.notes.append(
            "No GPU operators found in plan. "
            "Check that RAPIDS JAR is on classpath and version matches Spark."
        )

    if result.cpu_fallbacks_found > 0:
        result.notes.append(
            f"{result.cpu_fallbacks_found} CPU fallback operator(s): "
            f"{', '.join(sorted(result.cpu_fallback_operators))}"
        )

    return result
```

### CPU vs GPU Benchmark

```python
@dataclass
class RapidsBenchmarkResult:
    mode: str  # "gpu" or "cpu"
    query_name: str = ""
    duration_s: float = 0.0
    rows_written: int = 0
    gpu_memory_peak_mb: float = 0.0


def rapids_benchmark(
    spark: SparkSession,
    input_path: str,
    query_name: str = "aggregation",
    query: Optional[str] = None,
) -> list[RapidsBenchmarkResult]:
    """Compare CPU-only vs GPU-accelerated execution for the same query.

    The GPU run uses the current session config (assumes RAPIDS enabled).
    The CPU run temporarily disables RAPIDS via ``spark.rapids.sql.enabled=false``.

    Parameters
    ----------
    spark : Active SparkSession.
    input_path : Path to the input Parquet/ORC dataset.
    query_name : Label for the benchmark.
    query : SQL query to execute.  Defaults to a standard aggregation.

    Returns
    -------
    list[RapidsBenchmarkResult] with GPU and CPU entries.
    """
    df = spark.read.parquet(input_path)
    df.createOrReplaceTempView("rapids_bench")

    if query is None:
        query = (
            "SELECT category, SUM(amount) AS total, "
            "AVG(quantity) AS avg_qty, COUNT(*) AS cnt "
            "FROM rapids_bench GROUP BY category ORDER BY total DESC"
        )

    results: list[RapidsBenchmarkResult] = []

    for mode in ["gpu", "cpu"]:
        if mode == "cpu":
            spark.conf.set("spark.rapids.sql.enabled", "false")
            logger.info("Benchmarking CPU mode (RAPIDS disabled)")
        else:
            spark.conf.set("spark.rapids.sql.enabled", "true")
            logger.info("Benchmarking GPU mode (RAPIDS enabled)")

        # Warm up
        spark.sql(f"EXPLAIN {query}").collect()

        start = time.monotonic()
        out_df = spark.sql(query)
        row_count = out_df.count()
        dur = time.monotonic() - start

        results.append(RapidsBenchmarkResult(
            mode=mode,
            query_name=query_name,
            duration_s=round(dur, 3),
            rows_written=row_count,
        ))
        logger.info("Mode=%s duration=%.3fs rows=%d", mode, dur, row_count)

    # Ensure RAPIDS is re-enabled
    spark.conf.set("spark.rapids.sql.enabled", "true")

    # Calculate speedup
    gpu_dur = next((r.duration_s for r in results if r.mode == "gpu"), 0)
    cpu_dur = next((r.duration_s for r in results if r.mode == "cpu"), 0)
    speedup = cpu_dur / gpu_dur if gpu_dur > 0 else 0.0

    report_path = Path("/tmp/rapids_benchmark.json")
    report_data = {
        "query": query_name,
        "results": [asdict(r) for r in results],
        "speedup_ratio": round(speedup, 2),
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    logger.info(
        "RAPIDS benchmark: GPU=%.3fs  CPU=%.3fs  speedup=%.2fx",
        gpu_dur, cpu_dur, speedup,
    )
    logger.info("Report written to %s", report_path)
    return results
```

### GPU Memory Monitor

```python
@dataclass
class GpuMemorySnapshot:
    """Point-in-time GPU memory snapshot from nvidia-smi."""
    gpu_index: int = 0
    gpu_name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_pct: float = 0.0
    timestamp: str = ""


def monitor_gpu_memory() -> list[GpuMemorySnapshot]:
    """Read GPU memory usage via ``nvidia-smi`` and return a list of
    per-GPU snapshots.

    Returns
    -------
    list[GpuMemorySnapshot]

    Raises
    ------
    RuntimeError
        If ``nvidia-smi`` is not found on PATH.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found on PATH")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc.stderr.strip()}") from exc

    snapshots: list[GpuMemorySnapshot] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            snapshots.append(GpuMemorySnapshot(
                gpu_index=int(parts[0]),
                gpu_name=parts[1],
                memory_total_mb=int(float(parts[2])),
                memory_used_mb=int(float(parts[3])),
                memory_free_mb=int(float(parts[4])),
                utilization_pct=float(parts[5]),
                timestamp=now,
            ))
        except (ValueError, IndexError) as exc:
            logger.warning("Failed to parse nvidia-smi line: %s (%s)", line, exc)

    report_path = Path("/tmp/gpu_memory_monitor.json")
    report_path.write_text(
        json.dumps([asdict(s) for s in snapshots], indent=2)
    )
    logger.info(
        "GPU memory snapshot: %d GPU(s), report at %s",
        len(snapshots), report_path,
    )
    for s in snapshots:
        pct = (s.memory_used_mb / s.memory_total_mb * 100) if s.memory_total_mb > 0 else 0
        logger.info(
            "  GPU %d [%s]: %d/%d MB used (%.1f%%), util=%.1f%%",
            s.gpu_index, s.gpu_name, s.memory_used_mb,
            s.memory_total_mb, pct, s.utilization_pct,
        )

    return snapshots
```

### GPU Fallback Handler

```python
@dataclass
class FallbackEvent:
    query_id: str = ""
    operator: str = ""
    reason: str = ""
    severity: str = "WARNING"
    timestamp: str = ""


class GPUFallbackHandler:
    """Detect and log GPU-to-CPU fallback events by parsing Spark EXPLAIN
    output and event logs.

    Usage::

        handler = GPUFallbackHandler(spark)
        handler.check_query("q1", "SELECT ...")
        print(handler.get_fallbacks())
        handler.write_report()
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.fallbacks: list[FallbackEvent] = []

    def check_query(self, query_id: str, query: str) -> list[FallbackEvent]:
        """Analyze *query* for CPU fallback operators and record events.

        Parameters
        ----------
        query_id : Unique identifier for this query.
        query : SQL string to analyze.

        Returns
        -------
        list[FallbackEvent]
        """
        original = self.spark.conf.get("spark.rapids.sql.explain", "NONE")
        self.spark.conf.set("spark.rapids.sql.explain", "ALL")

        try:
            plan = self.spark.sql(f"EXPLAIN {query}").collect()[0][0]
        finally:
            self.spark.conf.set("spark.rapids.sql.explain", original)

        # Pattern: "! <Operator> cannot run on GPU because <reason>"
        pattern = re.compile(
            r"!\s*<(\w+)>\s+cannot run on GPU because\s+([^\n]+)",
            re.DOTALL,
        )
        found: list[FallbackEvent] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        for match in pattern.finditer(plan):
            operator = match.group(1)
            reason = match.group(2).strip()
            event = FallbackEvent(
                query_id=query_id,
                operator=operator,
                reason=reason,
                timestamp=now,
            )
            found.append(event)
            logger.warning(
                "GPU fallback: query=%s operator=%s reason=%s",
                query_id, operator, reason[:120],
            )

        self.fallbacks.extend(found)
        return found

    def get_fallbacks(self) -> list[dict]:
        return [asdict(f) for f in self.fallbacks]

    def summary(self) -> dict:
        total = len(self.fallbacks)
        by_operator: dict[str, int] = {}
        for f in self.fallbacks:
            by_operator[f.operator] = by_operator.get(f.operator, 0) + 1

        return {
            "total_fallback_events": total,
            "fallback_operators": by_operator,
            "affected_queries": len(set(f.query_id for f in self.fallbacks)),
            "events": self.get_fallbacks(),
        }

    def write_report(self, path: str = "/tmp/gpu_fallback_report.json") -> str:
        Path(path).write_text(json.dumps(self.summary(), indent=2))
        logger.info("GPU fallback report written to %s", path)
        return path
```

### RapidsPipeline Class

```python
@dataclass
class RapidsPipelineResult:
    """End-to-end GPU pipeline execution result."""
    pipeline_name: str = ""
    total_duration_s: float = 0.0
    stages: list[dict] = field(default_factory=list)
    gpu_operators_used: list[str] = field(default_factory=list)
    cpu_fallbacks: list[str] = field(default_factory=list)
    gpu_memory_snapshot: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


class RapidsPipeline:
    """End-to-end GPU pipeline with fallback handling and performance
    reporting.

    Usage::

        pipeline = RapidsPipeline(spark, "etl_gpu_pipeline")
        pipeline.add_stage("read", "SELECT * FROM parquet.`/data/input`")
        pipeline.add_stage("transform", "SELECT a, b, SUM(c) FROM t GROUP BY a, b")
        pipeline.add_stage("write", "INSERT OVERWRITE ...")
        result = pipeline.execute()
        print(f"GPU operators: {result.gpu_operators_used}")
    """

    def __init__(
        self,
        spark: SparkSession,
        name: str = "rapids_pipeline",
        monitor_gpu: bool = True,
    ) -> None:
        self.spark = spark
        self.name = name
        self.monitor_gpu = monitor_gpu
        self.stages: list[tuple[str, str]] = []
        self.fallback_handler = GPUFallbackHandler(spark)

    def add_stage(self, name: str, query: str) -> None:
        """Register a pipeline stage."""
        self.stages.append((name, query))
        logger.info("Added stage: %s", name)

    def execute(self) -> RapidsPipelineResult:
        """Execute all pipeline stages and collect metrics.

        Returns
        -------
        RapidsPipelineResult
        """
        start = time.monotonic()
        stages_results: list[dict] = []
        all_gpu_ops: set[str] = set()
        all_fallbacks: set[str] = set()
        warnings: list[str] = []

        # Capture initial GPU memory
        gpu_snap: Optional[dict] = None
        if self.monitor_gpu:
            try:
                snapshots = monitor_gpu_memory()
                if snapshots:
                    gpu_snap = asdict(snapshots[0])
            except RuntimeError:
                warnings.append("nvidia-smi not available; skipping GPU memory monitor")

        for stage_name, query in self.stages:
            stage_start = time.monotonic()

            # Check for fallbacks
            fallbacks = self.fallback_handler.check_query(stage_name, query)
            for fb in fallbacks:
                all_fallbacks.add(fb.operator)

            # Execute the stage
            try:
                df = self.spark.sql(query)
                df.collect()  # materialize
                stage_dur = time.monotonic() - stage_start

                # Parse plan for GPU operators
                plan = self.spark.sql(f"EXPLAIN {query}").collect()[0][0]
                gpu_ops = set(re.findall(r"Gpu(\w+)", plan))
                all_gpu_ops.update(gpu_ops)

                stages_results.append({
                    "stage": stage_name,
                    "duration_s": round(stage_dur, 3),
                    "gpu_operators": sorted(gpu_ops),
                    "fallbacks": [fb.operator for fb in fallbacks],
                    "status": "success",
                })
                logger.info(
                    "Stage '%s' completed in %.3fs (GPU ops: %s)",
                    stage_name, stage_dur, sorted(gpu_ops),
                )
            except Exception as exc:
                stage_dur = time.monotonic() - stage_start
                stages_results.append({
                    "stage": stage_name,
                    "duration_s": round(stage_dur, 3),
                    "status": "error",
                    "error": str(exc),
                })
                warnings.append(f"Stage '{stage_name}' failed: {exc}")
                logger.error("Stage '%s' failed: %s", stage_name, exc)

        total_dur = time.monotonic() - start

        result = RapidsPipelineResult(
            pipeline_name=self.name,
            total_duration_s=round(total_dur, 3),
            stages=stages_results,
            gpu_operators_used=sorted(all_gpu_ops),
            cpu_fallbacks=sorted(all_fallbacks),
            gpu_memory_snapshot=gpu_snap,
            warnings=warnings,
        )

        report_path = Path("/tmp/rapids_pipeline_report.json")
        report_path.write_text(json.dumps(asdict(result), indent=2))
        logger.info(
            "Pipeline '%s' completed in %.3fs, report at %s",
            self.name, total_dur, report_path,
        )
        return result
```

### Production RAPIDS ETL

```python
def run_production_rapids_etl(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    query: Optional[str] = None,
    monitor_gpu: bool = True,
) -> dict:
    """Run a production ETL pipeline with GPU acceleration and monitoring.

    Parameters
    ----------
    spark : Active SparkSession (RAPIDS should already be enabled).
    input_path : Path to input data.
    output_path : Destination path for results.
    query : SQL query to execute.  Defaults to a standard ETL aggregation.
    monitor_gpu : Whether to capture GPU memory snapshots.

    Returns
    -------
    dict
        ETL metadata with GPU metrics and fallback information.
    """
    if query is None:
        query = (
            f"SELECT category, department, "
            f"SUM(amount) AS total_amount, "
            f"AVG(quantity) AS avg_quantity, "
            f"COUNT(*) AS row_count "
            f"FROM parquet.`{input_path}` "
            f"WHERE amount IS NOT NULL "
            f"GROUP BY category, department "
            f"ORDER BY total_amount DESC"
        )

    start = time.monotonic()
    errors: list[str] = []

    # GPU memory before
    gpu_before = None
    if monitor_gpu:
        try:
            gpu_before = monitor_gpu_memory()
        except RuntimeError:
            logger.warning("GPU monitor not available")

    try:
        df = spark.read.parquet(input_path)
        df.createOrReplaceTempView("rapids_etl_input")

        result_df = spark.sql(query)
        result_df.write.mode("overwrite").parquet(output_path)
        row_count = result_df.count()

    except Exception as exc:
        logger.error("RAPIDS ETL failed: %s", exc)
        errors.append(str(exc))
        row_count = 0
    finally:
        dur = time.monotonic() - start

    # GPU memory after
    gpu_after = None
    if monitor_gpu:
        try:
            gpu_after = monitor_gpu_memory()
        except RuntimeError:
            pass

    # Verify RAPIDS activity
    verification = verify_rapids_active(spark, query)

    # Check for fallbacks
    fallback_handler = GPUFallbackHandler(spark)
    fallbacks = fallback_handler.check_query("production_etl", query)

    report = {
        "input_path": input_path,
        "output_path": output_path,
        "row_count": row_count,
        "duration_s": round(dur, 3),
        "rapids_verification": asdict(verification),
        "gpu_fallbacks": [asdict(f) for f in fallbacks],
        "gpu_memory_before": [asdict(s) for s in gpu_before] if gpu_before else None,
        "gpu_memory_after": [asdict(s) for s in gpu_after] if gpu_after else None,
        "errors": errors,
    }

    report_path = Path("/tmp/rapids_production_etl.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "RAPIDS ETL report: %.3fs, %d rows, %d GPU operators, %d fallbacks",
        dur, row_count, verification.gpu_ops_found, len(fallbacks),
    )
    logger.info("Report written to %s", report_path)
    return report
```

### Quick Usage Example

```python
# 1. Create RAPIDS-enabled session
spark = create_rapids_session(
    gpu_memory_gb=16,
    concurrent_gpu_tasks=1,
    pinned_pool_size_gb=2,
    memory_pool="ARENA",
)

# 2. Verify RAPIDS is active
result = verify_rapids_active(spark, """
    SELECT category, SUM(amount) AS total
    FROM parquet.`/data/large-dataset`
    GROUP BY category
""")
for note in result.notes:
    print(note)

# 3. Run a CPU vs GPU benchmark
results = rapids_benchmark(
    spark,
    input_path="/data/large-dataset",
    query_name="aggregation",
)
gpu_dur = next(r.duration_s for r in results if r.mode == "gpu")
cpu_dur = next(r.duration_s for r in results if r.mode == "cpu")
print(f"Speedup: {cpu_dur / gpu_dur:.2f}x")

# 4. Monitor GPU memory
snapshots = monitor_gpu_memory()
for s in snapshots:
    print(f"GPU {s.gpu_index}: {s.memory_used_mb}/{s.memory_total_mb} MB")

# 5. Run end-to-end pipeline
pipeline = RapidsPipeline(spark, "etl_gpu")
pipeline.add_stage("scan", "SELECT * FROM parquet.`/data/input`")
pipeline.add_stage(
    "aggregate",
    "SELECT category, SUM(amount) AS total FROM temp GROUP BY category",
)
pipeline_result = pipeline.execute()
print(f"GPU operators: {pipeline_result.gpu_operators_used}")
print(f"Fallbacks: {pipeline_result.cpu_fallbacks}")

# 6. Production ETL
etl_report = run_production_rapids_etl(
    spark,
    input_path="/data/large-dataset",
    output_path="/data/output/enriched",
)
```
