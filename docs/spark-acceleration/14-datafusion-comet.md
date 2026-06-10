# Apache DataFusion Comet

> Rust-based native Spark accelerator running on commodity hardware -- zero code changes required.

## Overview

Apache DataFusion Comet is a high-performance accelerator for Apache Spark built on top of [Apache DataFusion](https://datafusion.apache.org), an extensible query engine written in Rust. Comet intercepts Spark physical plans after Catalyst optimization, translates supported operators into DataFusion plans, and executes them natively in Rust. The result is approximately 2x speedup on TPC-DS benchmarks at Scale Factor 1000 (1 TB), translating to roughly 50% cost savings on cloud clusters -- with no modifications to existing Spark SQL, DataFrame, or PySpark code.

Comet was originally donated to Apache Arrow in 2024 and later became a subproject of Apache DataFusion. It runs entirely on commodity CPU hardware: no GPUs, FPGAs, or other specialized accelerators are required.

| Property | Value |
|----------|-------|
| Native engine | Apache DataFusion (Rust) |
| Data format | Apache Arrow (zero-copy JNI) |
| Code changes required | Zero (drop-in plugin) |
| TPC-DS SF 1000 speedup | ~2x (632s -> 295s on TPC-H 100GB, similar ratio on TPC-DS) |
| Latest release | 0.14.0 (March 2025) |
| Main branch | 0.16.0-dev |
| GitHub | https://github.com/apache/datafusion-comet |
| Docs | https://datafusion.apache.org/comet/ |

## Architecture

### Plugin Architecture

Comet integrates into Spark as a plugin, intercepting query execution after Catalyst has produced the optimized physical plan:

```
Spark SQL Query
    |
    v
Catalyst Optimizer (Spark physical plan)
    |
    v
CometSparkSessionExtensions (physical plan rules)
    |
    +-- Walk plan bottom-up
    +-- Replace supported operators with Comet equivalents
    +-- Serialize consecutive Comet operators as protobuf
    +-- Execute fused native block on DataFusion
    |
    v
Mixed plan: Comet* nodes (native) + Spark nodes (fallback)
```

Two core classes form the integration surface:

- **`CometPlugin`** (`org.apache.spark.CometPlugin`): Registered via `spark.plugins`. Installs `CometSparkSessionExtensions` rules that traverse the physical plan and replace Spark operators with Comet-native equivalents.
- **`CometShuffleManager`** (`org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager`): Registered via `spark.shuffle.manager`. Replaces Spark's default shuffle manager with Comet's accelerated columnar and native shuffle implementations.

### Plan Translation and Native Execution

The `CometSparkSessionExtensions` rules walk the physical plan bottom-up. When a supported Spark operator is encountered (e.g., `ProjectExec`, `FilterExec`, `HashAggregateExec`), it is replaced with a Comet equivalent (`CometProject`, `CometFilter`, `CometHashAggregate`). Consecutive native operators are fused into a single block, serialized as Protocol Buffers, and sent across JNI to the native DataFusion engine for execution.

Operators that Comet does not support remain as their original Spark form. This means a single query plan can contain a mix of three node types:

1. **`Comet*` nodes running natively in Rust** -- e.g., `CometProject`, `CometHashAggregate`, `CometSortMergeJoin`.
2. **`Comet*` nodes running on the JVM** -- e.g., `CometBroadcastExchange`, `CometColumnarExchange`. These keep data on the JVM but participate in the Comet pipeline via Arrow batches.
3. **Standard Spark nodes** -- where Comet either does not support the operator or has fallen back due to an unsupported expression, data type, or configuration.

Wherever data crosses between columnar and row-based execution, Comet inserts a transition node such as `CometColumnarToRow` or `CometSparkRowToColumnar`. Frequent transitions usually indicate partial fallback within the plan.

### JNI and Apache Arrow Zero-Copy

Communication between the JVM and native Rust code uses Apache Arrow as the data interchange format. Arrow's in-memory columnar layout enables zero-copy data transfer: the JVM writes Arrow batches directly into off-heap memory, and the native engine reads them without serialization or deserialization overhead. This is the key to Comet's performance -- avoiding the row-to-column and column-to-row conversions that plague other accelerator approaches.

### Shuffle Implementations

Comet provides two shuffle implementations. The plan output tells you which one is in use:

| Shuffle Type | Node Name | Description |
|-------------|-----------|-------------|
| **Native Shuffle** | `CometExchange` | Fully native execution. Partition, encode, and compress steps run in Rust. Supports `HashPartitioning`, `RangePartitioning`, and `SinglePartitioning`. Partition keys must be primitive types; data columns may include complex types (`StructType`, `ArrayType`, `MapType`) since batches are serialized via Arrow IPC. |
| **Columnar (JVM) Shuffle** | `CometColumnarExchange` | JVM-based columnar shuffle. Supports `HashPartitioning`, `RoundRobinPartitioning`, `RangePartitioning`, and `SinglePartitioning`. Accepts both Spark row-based and Comet columnar input, making it the fallback when the child is not a Comet operator or when partition key types are unsupported by native shuffle (e.g., collated strings). |

Comet automatically chooses the fastest available option: it first attempts native shuffle, and if that is not possible it falls back to columnar shuffle. If neither can be applied, it reverts to Spark's built-in shuffle.

Comet also performs **automatic revert optimization**: when a columnar shuffle ends up between two non-Comet operators (e.g., a partial/final hash aggregate pair that Comet could not convert), Comet reverts it to Spark's built-in shuffle to avoid the cost of `row -> Arrow -> shuffle -> Arrow -> row` conversions with no Comet consumer on either side. Each revert is logged at INFO level as `Reverting Comet columnar shuffle to Spark shuffle between <parent> and <child>`.

By default, Comet compresses shuffle files using **ZSTD** instead of Spark's default LZ4.

### Scan Implementations

Comet provides multiple scan implementations for Parquet and Iceberg data:

| Scan Node | Description |
|-----------|-------------|
| `CometScan` | V1 Parquet scan using Comet's native Parquet reader. Decoding runs in native code; Arrow batches cross JNI into the native plan. Active scan implementation shown in brackets, e.g., `CometScan [native_iceberg_compat]`. |
| `CometBatchScan` | DataSource V2 scan, including Iceberg Parquet, producing Arrow batches consumed by Comet. |
| `CometNativeScan` | Fully native Parquet scan running entirely in DataFusion (no JVM Parquet reader involvement). |
| `CometIcebergNativeScan` | Fully native Iceberg Parquet scan using iceberg-rust. |
| `CometCsvNativeScan` | Fully native CSV scan (experimental). |

The active scan implementation is controlled by `spark.comet.scan.impl`:

| Value | Description |
|-------|-------------|
| `native_comet` | Comet's own native Parquet reader. |
| `native_datafusion` | DataFusion's native Parquet reader. |
| `native_iceberg_compat` | Native Iceberg reader (enabled by default since 0.15.0). Uses iceberg-rust for table metadata extraction and DataFusion for Parquet decoding. |

### Native Iceberg Reader

Comet's native Iceberg reader (enabled by default in 0.15.0+) uses the `iceberg-rust` crate to extract `FileScanTask`s from Iceberg tables via reflection, then serializes them to Comet's native execution engine. Key capabilities:

- **Table specs**: v1 and v2 (v3 falls back to Spark)
- **Data types**: All primitives including UUID; complex types (arrays, maps, structs)
- **Schema evolution**: Adding and dropping columns
- **Time travel**: `VERSION AS OF` queries for historical snapshots
- **Branch reads**: Accessing named branches
- **Delete handling**: Position deletes, equality deletes, mixed delete types (Merge-On-Read)
- **Filter pushdown**: Equality/comparison predicates, logical operators, NULL checks, IN/NOT IN, BETWEEN
- **Partitioning**: Standard partition pruning, date, bucket, truncate, hour transforms
- **Storage**: Local filesystem, HDFS, S3-compatible (AWS S3, MinIO)
- **REST catalogs**: Native support for REST catalog configurations
- **Parallel file fetching**: Configurable via `spark.comet.scan.icebergNative.dataFileConcurrencyLimit` (default 1; recommended 2-8)

Current limitations for the native Iceberg reader: Iceberg v3 scans, writes (reads are accelerated, writes use Spark), Avro/ORC data files (only Parquet), tables partitioned on BINARY or DECIMAL(precision > 28) columns, residual filters using transform functions (partition pruning still works, row-level filtering falls back), and Dynamic Partition Pruning under AQE.

## Supported Features

### Spark Version Compatibility

| Spark Version | Support Level | Scala Versions |
|--------------|---------------|----------------|
| 3.4 | Stable | 2.12, 2.13 |
| 3.5 | Stable | 2.12, 2.13 |
| 4.0 | Stable | 2.13 |
| 4.1 | Stable | 2.13 |
| 4.2 | Experimental | 2.13 |

### Supported Operators

The following Spark physical operators are replaced with native Comet implementations:

| Operator | Status | Notes |
|----------|--------|-------|
| `BatchScanExec` | Yes | Parquet and Iceberg Parquet scans |
| `BroadcastExchangeExec` | Yes | |
| `BroadcastHashJoinExec` | Yes | |
| `ExpandExec` | Yes | |
| `FileSourceScanExec` | Yes | Parquet files |
| `FilterExec` | Yes | |
| `GenerateExec` | Partial | `explode` generator only |
| `GlobalLimitExec` | Yes | |
| `HashAggregateExec` | Yes | Includes `FILTER (WHERE ...)` clauses |
| `LocalLimitExec` | Yes | |
| `ObjectHashAggregateExec` | Partial | Limited aggregates (e.g., `bloom_filter_agg`) |
| `ProjectExec` | Yes | |
| `ShuffleExchangeExec` | Yes | Native and columnar shuffle |
| `ShuffledHashJoinExec` | Yes | |
| `SortExec` | Yes | |
| `SortMergeJoinExec` | Yes | |
| `UnionExec` | Yes | |
| `WindowExec` | Partial | Disabled by default due to known correctness issues |
| `InsertIntoHadoopFsRelationCommand` | Experimental | Native Parquet writes, disabled by default |
| `LocalTableScanExec` | Experimental | Disabled by default |

### Supported Expressions

Comet supports hundreds of Spark expressions across the following categories:

- **Math**: arithmetic operators, comparison operators, rounding, trigonometry
- **String**: concatenation, substring, trim, replace, regex matching
- **Datetime**: date arithmetic, extraction, formatting, interval operations
- **Array**: element access, size, sorting, filtering, aggregation
- **Map**: key lookup, size, transformation
- **JSON**: parsing, extraction, validation
- **Hash**: CRC32, MD5, SHA1, SHA2, xxHash
- **Predicate**: AND, OR, NOT, IN, BETWEEN, IS NULL, IS NOT NULL
- **Complex types**: struct field access, array indexing, map key lookup

For the authoritative list of currently supported expressions, see the [Comet Expressions documentation](https://datafusion.apache.org/comet/user-guide/expressions.html).

### Supported Data Types

Comet supports all primitive types (boolean, byte, short, int, long, float, double, decimal, string, binary, date, timestamp), as well as complex types including arrays, maps, and structs. For the full compatibility matrix, see the [Comet Data Types documentation](https://datafusion.apache.org/comet/user-guide/datatypes.html).

## Setup and Installation

### Basic Installation

Add the Comet JAR to the Spark classpath and enable the plugin. A typical configuration:

```shell
export COMET_JAR=/path/to/comet-spark-spark3.5_2.12-0.14.0.jar

$SPARK_HOME/bin/spark-shell \
    --jars $COMET_JAR \
    --conf spark.driver.extraClassPath=$COMET_JAR \
    --conf spark.executor.extraClassPath=$COMET_JAR \
    --conf spark.plugins=org.apache.spark.CometPlugin \
    --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager \
    --conf spark.comet.explainFallback.enabled=true \
    --conf spark.memory.offHeap.enabled=true \
    --conf spark.memory.offHeap.size=4g
```

### JAR Selection

Select the Comet artifact that matches your Spark and Scala versions:

| Spark + Scala | Maven Artifact |
|--------------|----------------|
| Spark 3.4 / Scala 2.12 | `org.apache.datafusion:comet-spark-spark3.4_2.12:0.14.0` |
| Spark 3.4 / Scala 2.13 | `org.apache.datafusion:comet-spark-spark3.4_2.13:0.14.0` |
| Spark 3.5 / Scala 2.12 | `org.apache.datafusion:comet-spark-spark3.5_2.12:0.14.0` |
| Spark 3.5 / Scala 2.13 | `org.apache.datafusion:comet-spark-spark3.5_2.13:0.14.0` |
| Spark 4.0 / Scala 2.13 | `org.apache.datafusion:comet-spark-spark4.0_2.13:0.14.0` |
| Spark 4.1 / Scala 2.13 | `org.apache.datafusion:comet-spark-spark4.1_2.13:0.14.0` |

### Using --packages

Instead of downloading the JAR manually, use Spark's package resolver:

```shell
$SPARK_HOME/bin/spark-shell \
    --packages org.apache.datafusion:comet-spark-spark3.5_2.12:0.14.0 \
    --conf spark.plugins=org.apache.spark.CometPlugin \
    --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager \
    --conf spark.memory.offHeap.enabled=true \
    --conf spark.memory.offHeap.size=4g
```

## Configuration Reference

### Core Execution

| Configuration | Default | Description |
|--------------|---------|-------------|
| `spark.comet.exec.enabled` | `true` | Enable native execution. When `false`, Comet registers but does not replace any operators. |
| `spark.comet.batchSize` | `4096` | Number of rows per Arrow batch. Larger batches reduce JNI overhead but increase memory usage. |
| `spark.comet.exec.shuffle.fallbackToColumnar` | `true` | When native shuffle is not possible, fall back to columnar shuffle instead of Spark shuffle. |
| `spark.comet.exec.replaceSortMergeJoin` | `false` | Convert `SortMergeJoin` to `ShuffledHashJoin` for better vectorized performance. Comet does not yet provide spill-to-disk for `ShuffledHashJoin`, so this may cause OOM on large build sides. |

### Scan Configuration

| Configuration | Default | Description |
|--------------|---------|-------------|
| `spark.comet.scan.impl` | `native_iceberg_compat` | Scan implementation: `native_comet`, `native_datafusion`, or `native_iceberg_compat`. |
| `spark.comet.scan.icebergNative.enabled` | `true` (since 0.15.0) | Enable native Iceberg reader. |
| `spark.comet.scan.icebergNative.dataFileConcurrencyLimit` | `1` | Parallel file fetches for Iceberg scans. Recommended 2-8 for production. |

### Shuffle Configuration

| Configuration | Default | Description |
|--------------|---------|-------------|
| `spark.comet.exec.shuffle.enabled` | `true` | Enable Comet shuffle at runtime. Can be toggled dynamically. |
| `spark.comet.exec.shuffle.revertRedundantColumnar.enabled` | `true` | Revert columnar shuffle to Spark when between two non-Comet operators. |
| `spark.comet.exec.shuffle.convertFromSparkPlan.enabled` | `true` | Convert Spark plan to columnar shuffle even when child is non-Comet, enabling downstream native execution. |
| `spark.shuffle.compress` | `true` | Compress shuffle files. Comet defaults to ZSTD compression. |

### Fallback and Diagnostics

| Configuration | Default | Description |
|--------------|---------|-------------|
| `spark.comet.explainFallback.enabled` | `false` | Log a single WARN per query stage when fallback occurs, with aggregated reasons. Low-noise diagnostic. |
| `spark.comet.logFallbackReasons.enabled` | `false` | Log every fallback reason as a separate WARN. Higher volume but captures all reasons. |
| `spark.comet.explain.format` | `verbose` | Format for extended explain in Spark SQL UI (Spark 4.0+). `verbose` shows full annotated plan; `fallback` shows fallback reasons only. |
| `spark.comet.explain.native.enabled` | `false` | Log the DataFusion native plan and metrics per executor task. Verbose (one plan per task). |
| `spark.comet.ansi.ignore` | `false` | Ignore ANSI mode differences. Useful when running in Spark 4.0 where ANSI is default. |

### Memory and Tuning

| Configuration | Default | Description |
|--------------|---------|-------------|
| `spark.memory.offHeap.enabled` | `false` | Enable Spark off-heap memory mode. Required for Comet memory management. |
| `spark.memory.offHeap.size` | `0` | Size of off-heap memory pool. Comet shares this pool with Spark. |
| `spark.comet.exec.memoryPool` | `fair_unified` (when off-heap enabled) | Memory pool type: `fair_unified` (prevents any operator from using more than its even share) or `greedy_unified` (first-come first-serve). |
| `spark.comet.exec.memoryPool.fraction` | `1.0` | Fraction of off-heap pool Comet can reserve. Set below 1.0 to work around memory accounting inaccuracies. |
| `COMET_WORKER_THREADS` | number of cores | Number of tokio worker threads. Set to executor core count. |
| `COMET_MAX_BLOCKING_THREADS` | `512` | Maximum tokio blocking threads. |

### Expression and Operator Controls

Individual expressions and operators can be enabled or disabled via dynamic configurations:

- **Per-expression opt-in**: `spark.comet.expression.<ExpressionName>.allowIncompatible=true` enables an expression marked as incompatible.
- **Per-operator opt-in**: `spark.comet.operator.<OperatorName>.allowIncompatible=true` enables an operator marked as incompatible.
- **Per-operator enable/disable**: `spark.comet.operator.<OperatorName>.enabled=true/false`.
- **Floating-point sort**: `spark.comet.expression.SortOrder.allowIncompatible=true` enables sorting on floating-point types (incompatible with Spark when data contains both +0.0 and -0.0).

### Extended Explain (Spark 4.0+)

On Spark 4.0 and newer, Comet provides an extended explain provider that annotates query plans with fallback reasons directly in the Spark SQL UI:

```shell
--conf spark.sql.extendedExplainProviders=org.apache.comet.ExtendedExplainInfo
```

The format is controlled by `spark.comet.explain.format`:
- `verbose` (default): full plan annotated with fallback reasons, plus acceleration summary.
- `fallback`: list of fallback reasons only.

## Query Plan Verification

### Verifying Comet Is Active

After enabling Comet, run `EXPLAIN FORMATTED` on a query and inspect the output:

```sql
EXPLAIN FORMATTED SELECT * FROM parquet_table WHERE amount > 100 GROUP BY category;
```

Look for `Comet*` prefixed nodes in the plan:

```
== Physical Plan ==
CometHashAggregate (1)
+- CometExchange (2)
   +- CometHashAggregate (3)
      +- CometProject (4)
         +- CometFilter (5)
            +- CometScan parquet_table (6)
```

Nodes with the `Comet*` prefix are accelerated by Comet. Standard Spark node names (without prefix) indicate fallback.

### Understanding Transition Nodes

Transition nodes mark boundaries between columnar Comet execution and row-based Spark execution:

| Node | Direction | Meaning |
|------|-----------|---------|
| `CometColumnarToRow` | columnar -> row | Data leaves native execution and returns to Spark rows. |
| `CometNativeColumnarToRow` | columnar -> row | Native row conversion for broadcast Arrow batches. Zero-copy for variable-length types. |
| `CometSparkRowToColumnar` | row -> columnar | Spark rows are converted to Comet Arrow batches. |
| `CometSparkColumnarToColumnar` | columnar -> columnar | Spark ColumnarBatch converted to Comet Arrow batches. |

Frequent transitions indicate partial fallback -- the query is not fully accelerated.

### Debugging Fallback

When Comet falls back, use these configurations to understand why:

**Low-noise (recommended first step):**

```shell
--conf spark.comet.explainFallback.enabled=true
```

Logs a single WARN per query stage with aggregated fallback reasons. Nothing is logged when the entire stage runs in Comet.

**Detailed (when low-noise is insufficient):**

```shell
--conf spark.comet.logFallbackReasons.enabled=true
```

Logs every fallback reason as a separate WARN. Does not include surrounding plan context -- best for accumulating diagnostics across many queries.

**Native plan inspection:**

```shell
--conf spark.comet.explain.native.enabled=true
```

Each executor task logs the DataFusion native plan with metrics. Verbose (one plan per task), but the only way to see the plan as DataFusion actually executes it.

## Production Best Practices

### Memory Sizing

Comet requires off-heap memory in addition to the JVM heap. Real-world benchmarks from TPC-H 100GB workloads:

| Configuration | Cores | RAM | Comet Time | Spark Time |
|--------------|-------|-----|------------|------------|
| Spark baseline | 8 | 8 GB | N/A | 632s |
| Spark minimum | 8 | 3 GB | N/A | 744s |
| Comet minimum | 8 | 5 GB | 340s | -- |
| Comet recommended | 8 | 8 GB | 295s | -- |
| Comet half-resources | 4 | 4 GB | 520s | 632s |

Comet outperforms Spark even at half the resources (4 cores, 4 GB RAM: 520s vs 632s). The recommended starting point is to allocate the same total memory as your Spark baseline, with at least 5 GB for off-heap.

If Comet uses more memory than it reserves (due to imperfect memory accounting), reduce `spark.comet.exec.memoryPool.fraction` below 1.0.

### Shuffle Partition Tuning

Set `spark.sql.shuffle.partitions` appropriately for your workload. The default of 200 partitions is often too low for large datasets and too high for small ones. With Comet's native shuffle, fewer partitions may be sufficient because native execution processes each partition more efficiently.

For TPC-DS SF 1000, the upstream benchmarks use 1000 partitions. Start with partitions roughly equal to total data size (GB) divided by 128 MB (target partition size).

### ANSI Mode Handling (Spark 4.0)

Spark 4.0 enables ANSI mode by default, which changes overflow behavior for integer arithmetic. Comet handles ANSI mode natively for most expressions. If you encounter compatibility issues with ANSI mode, set `spark.comet.ansi.ignore=true` to disable ANSI enforcement in Comet (though Spark's ANSI checks still apply).

### Minimizing Fallbacks

To maximize the portion of your query that runs natively:

1. **Use Parquet format** -- Comet's native Parquet reader is its most mature scan implementation.
2. **Avoid Window functions** -- `WindowExec` is disabled by default due to known correctness issues. Enable with `spark.comet.operator.WindowExec.enabled=true` only after validating correctness.
3. **Prefer primitive partition keys** in shuffles -- native shuffle requires primitive partition key types. Complex types as partition keys force columnar shuffle fallback.
4. **Test `ShuffledHashJoin`** -- set `spark.comet.exec.replaceSortMergeJoin=true` and compare performance. Comet does not yet spill `ShuffledHashJoin` to disk, so this may cause OOM on large build sides.
5. **Enable native Iceberg reader** -- already the default in 0.15.0+. Verify with `CometBatchScan` or `CometIcebergNativeScan` nodes in the plan.
6. **Set appropriate batch size** -- `spark.comet.batchSize` (default 4096) balances JNI overhead against memory usage. Increase for memory-rich clusters, decrease for tight memory budgets.

### Iceberg Workload Acceleration

For Iceberg workloads, ensure:

```shell
--conf spark.comet.scan.impl=native_iceberg_compat
--conf spark.comet.scan.icebergNative.enabled=true
--conf spark.comet.scan.icebergNative.dataFileConcurrencyLimit=4
```

Use the `CometBatchScan` or `CometIcebergNativeScan` nodes in `EXPLAIN` output to verify native Iceberg scanning. Increase `dataFileConcurrencyLimit` from the default of 1 to 2-8 for production workloads to hide I/O latency through parallel file fetching.

Note that Iceberg writes are not accelerated -- only reads benefit from the native reader. Writes continue through Spark's native path.

### Kubernetes Deployment

For Kubernetes deployments, Comet provides specific guidance:

- Set `COMET_WORKER_THREADS` to match container CPU limits (Kubernetes CPU requests).
- Ensure the Comet JAR is available on all executor pods (use `--jars`, `--packages`, or bake into the container image).
- Off-heap memory must be within the pod memory limit.
- See the [Comet Kubernetes Guide](https://datafusion.apache.org/comet/user-guide/kubernetes.html) for detailed examples.

## Comparison with Gluten

Comet and [Gluten](https://github.com/oap-project/gluten) are both third-party Spark accelerators that replace Spark operators with native implementations. They differ in their underlying engines and design philosophies:

| Aspect | DataFusion Comet | Gluten |
|--------|-----------------|--------|
| Native engine | Apache DataFusion (Rust) | Apache Velox (C++) |
| Language | Rust | C++ |
| Data format | Apache Arrow | Apache Arrow (Velox-native) |
| Project status | Apache Software Foundation | Linux Foundation / OAP |
| Maturity | Younger, rapidly evolving | Older, broader ecosystem integrations |
| Shuffle | Native + columnar, auto-selected | Velco-shuffle (Velox-based) |
| Parquet reader | Native (Comet's own + DataFusion) | Velox reader |
| Iceberg | Native (iceberg-rust) | Via Velox |
| Target hardware | Commodity CPU | Commodity CPU |
| Target workloads | SQL, DataFrame, PySpark, Iceberg reads | SQL, ML, broader Velox ecosystem |

Both projects run on commodity CPU hardware and require zero code changes. The choice between them often depends on:

- **Ecosystem preference**: Comet is an Apache project with tight DataFusion integration; Gluten is part of the broader OAP/Velox ecosystem.
- **Language familiarity**: Rust (Comet) vs C++ (Gluten) for debugging and extending.
- **Maturity**: Gluten has a longer track record; Comet is newer but advancing rapidly.
- **Iceberg support**: Both support Iceberg; Comet's native iceberg-rust reader is enabled by default as of 0.15.0.

In practice, the best approach is to benchmark both accelerators against your specific workload. Results can vary significantly based on query patterns, data characteristics, and cluster configuration.

## Metrics and Monitoring

Comet exposes native execution metrics through Spark's standard metrics infrastructure. To see native plans with metrics:

```shell
--conf spark.comet.explain.native.enabled=true
```

Each executor task logs the DataFusion native plan with embedded metrics including:
- Rows processed per operator
- Memory usage and spill information
- Execution time breakdown

These metrics also appear in the Spark UI under the Stages and Tasks tabs. Comet populates Spark's task-level `inputMetrics.bytesRead` for native Iceberg scans using the `bytes_read` counter from iceberg-rust's `ScanMetrics`, which includes both data files and delete files.

## GitHub and Resources

- **Source code**: https://github.com/apache/datafusion-comet
- **Documentation**: https://datafusion.apache.org/comet/
- **Installation guide**: https://datafusion.apache.org/comet/user-guide/installation.html
- **Configuration reference**: https://datafusion.apache.org/comet/user-guide/configs.html
- **Supported operators**: https://datafusion.apache.org/comet/user-guide/operators.html
- **Supported expressions**: https://datafusion.apache.org/comet/user-guide/expressions.html
- **Understanding Comet plans**: https://datafusion.apache.org/comet/user-guide/understanding-comet-plans.html
- **Tuning guide**: https://datafusion.apache.org/comet/user-guide/tuning.html
- **Iceberg guide**: https://datafusion.apache.org/comet/user-guide/iceberg.html
- **Benchmarking guide**: https://datafusion.apache.org/comet/contributor-guide/benchmarking.html
- **Contributor guide**: https://datafusion.apache.org/comet/contributor-guide/contributing.html
- **Discord community**: https://discord.gg/3EAr4ZX6JK
- **Maven Central**: https://search.maven.org/search?q=g:org.apache.datafusion

---

## Production-Ready Python (PySpark) Code

### Comet Session Factory

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
logger = logging.getLogger("comet_pyspark")


def create_comet_session(
    *,
    app_name: str = "CometAcceleratedApp",
    off_heap_size: str = "4g",
    batch_size: int = 4096,
    enable_shuffle: bool = True,
    enable_columnar_cache: bool = True,
    memory_pool: str = "fair_unified",
    memory_pool_fraction: float = 1.0,
    scan_impl: str = "native_iceberg_compat",
    iceberg_concurrency: int = 4,
    explain_fallback: bool = True,
    extra_confs: Optional[dict[str, str]] = None,
) -> SparkSession:
    """Create a SparkSession configured with DataFusion Comet plugin.

    Args:
        app_name: Application name shown in Spark UI.
        off_heap_size: Off-heap memory size for native execution.
        batch_size: Rows per Arrow batch (larger = less JNI overhead).
        enable_shuffle: Enable Comet native shuffle.
        enable_columnar_cache: Enable Comet columnar cache.
        memory_pool: Memory pool type ('fair_unified' or 'greedy_unified').
        memory_pool_fraction: Fraction of off-heap pool Comet can reserve.
        scan_impl: Scan implementation ('native_comet', 'native_datafusion',
                   or 'native_iceberg_compat').
        iceberg_concurrency: Parallel file fetches for Iceberg scans.
        explain_fallback: Log fallback reasons at WARN level.
        extra_confs: Additional Spark configurations.

    Returns:
        Configured SparkSession with Comet plugin enabled.
    """
    conf = {
        # --- Plugin registration ---
        "spark.plugins": "org.apache.spark.CometPlugin",
        "spark.shuffle.manager":
            "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",

        # --- Core execution ---
        "spark.comet.exec.enabled": "true",
        "spark.comet.batchSize": str(batch_size),

        # --- Off-heap memory ---
        "spark.memory.offHeap.enabled": "true",
        "spark.memory.offHeap.size": off_heap_size,

        # --- Shuffle ---
        "spark.comet.exec.shuffle.enabled": str(enable_shuffle).lower(),
        "spark.comet.exec.shuffle.fallbackToColumnar": "true",
        "spark.comet.exec.shuffle.revertRedundantColumnar.enabled": "true",

        # --- Columnar cache ---
        "spark.comet.columnarCache.enabled": str(
            enable_columnar_cache
        ).lower(),

        # --- Memory management ---
        "spark.comet.exec.memoryPool": memory_pool,
        "spark.comet.exec.memoryPool.fraction": str(memory_pool_fraction),

        # --- Scan configuration ---
        "spark.comet.scan.impl": scan_impl,
        "spark.comet.scan.icebergNative.enabled": "true",
        "spark.comet.scan.icebergNative.dataFileConcurrencyLimit": str(
            iceberg_concurrency
        ),

        # --- Diagnostics ---
        "spark.comet.explainFallback.enabled": str(
            explain_fallback
        ).lower(),

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
        "Comet session created: app=%s, off_heap=%s, batch_size=%d",
        app_name,
        off_heap_size,
        batch_size,
    )
    return session
```

### Verify Comet Is Active

```python
def verify_comet_active(spark: SparkSession) -> dict:
    """Check whether Comet is active and parse EXPLAIN for native operators.

    Runs a small query, captures EXPLAIN output, and searches for Comet
    operator signatures (CometScan, CometProject, CometHashAggregate, etc.).

    Args:
        spark: Active SparkSession.

    Returns:
        Dict with keys:
          - plugin_loaded: whether spark.plugins contains CometPlugin
          - off_heap_enabled: whether off-heap memory is configured
          - comet_enabled: whether spark.comet.exec.enabled is true
          - native_operators_found: whether Comet operators appear in EXPLAIN
          - operators: list of Comet operator names found
          - transition_nodes: list of Columnar<->Row transition nodes found
    """
    result: dict = {
        "plugin_loaded": False,
        "off_heap_enabled": False,
        "comet_enabled": False,
        "native_operators_found": False,
        "operators": [],
        "transition_nodes": [],
    }

    # --- Configuration checks ---
    try:
        plugins = spark.conf.get("spark.plugins", "")
        result["plugin_loaded"] = "CometPlugin" in plugins
    except Exception:
        logger.warning("Could not read spark.plugins")

    try:
        off_heap = spark.conf.get("spark.memory.offHeap.enabled", "false")
        result["off_heap_enabled"] = off_heap.lower() == "true"
    except Exception:
        logger.warning("Could not read off-heap config")

    try:
        comet_enabled = spark.conf.get("spark.comet.exec.enabled", "false")
        result["comet_enabled"] = comet_enabled.lower() == "true"
    except Exception:
        logger.warning("Could not read comet.exec.enabled")

    # --- EXPLAIN parsing ---
    comet_pattern = re.compile(r"(Comet\w+)")
    transition_pattern = re.compile(
        r"(CometColumnarToRow|CometNativeColumnarToRow|"
        r"CometSparkRowToColumnar|CometSparkColumnarToColumnar)"
    )

    try:
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

        operators = comet_pattern.findall(plan_text)
        transitions = transition_pattern.findall(plan_text)

        result["operators"] = list(set(operators))
        result["transition_nodes"] = list(set(transitions))
        result["native_operators_found"] = len(operators) > 0
    except Exception as exc:
        logger.warning("EXPLAIN parse failed: %s", exc)

    logger.info("Comet verification result: %s", json.dumps(result))
    return result
```

### Comet Benchmark

```python
@dataclass
class CometBenchmarkResult:
    """Benchmark comparison between Comet and vanilla Spark execution."""
    query_name: str
    comet_time_seconds: float
    spark_time_seconds: float
    speedup_ratio: float
    comet_rows: int
    spark_rows: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_faster(self) -> bool:
        return self.speedup_ratio > 1.0

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "comet_time_seconds": round(self.comet_time_seconds, 3),
            "spark_time_seconds": round(self.spark_time_seconds, 3),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "comet_rows": self.comet_rows,
            "spark_rows": self.spark_rows,
            "timestamp": self.timestamp,
        }


def comet_benchmark(
    spark: SparkSession,
    *,
    query_name: str = "aggregation_benchmark",
    query: str = "",
    comet_enabled: bool = True,
    data_path: Optional[str] = None,
) -> CometBenchmarkResult:
    """Benchmark Comet vs vanilla Spark execution.

    Runs the same query with Comet enabled and disabled, comparing
    wall-clock times and result row counts.

    Args:
        spark: Active SparkSession.
        query_name: Human-readable name for this benchmark run.
        query: SQL query to benchmark. If empty, uses a default aggregation.
        comet_enabled: Whether Comet is currently enabled in the session.
        data_path: Optional path to Parquet data. If given, creates a temp view.

    Returns:
        CometBenchmarkResult with timing and speedup metrics.
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

    def _timed_run(enabled: bool) -> tuple[float, int]:
        spark.conf.set("spark.comet.exec.enabled", str(enabled).lower())
        start = time.monotonic()
        df = spark.sql(sql)
        rows = df.count()
        elapsed = time.monotonic() - start
        return elapsed, rows

    # --- Run with Comet ---
    comet_time, comet_rows = _timed_run(enabled=True)
    logger.info("Comet run: %.3fs, %d rows", comet_time, comet_rows)

    # --- Run without Comet ---
    spark_time, spark_rows = _timed_run(enabled=False)
    logger.info("Spark run: %.3fs, %d rows", spark_time, spark_rows)

    # --- Restore original state ---
    if comet_enabled:
        spark.conf.set("spark.comet.exec.enabled", "true")

    speedup = spark_time / comet_time if comet_time > 0 else float("inf")
    result = CometBenchmarkResult(
        query_name=query_name,
        comet_time_seconds=comet_time,
        spark_time_seconds=spark_time,
        speedup_ratio=speedup,
        comet_rows=comet_rows,
        spark_rows=spark_rows,
    )

    # --- Write JSON report ---
    report_path = os.path.join(
        "/tmp", f"comet_benchmark_{query_name}_{int(time.time())}.json"
    )
    with open(report_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("Benchmark report written to %s", report_path)

    logger.info(
        "Benchmark %s: speedup=%.2fx (Comet=%.3fs, Spark=%.3fs)",
        query_name,
        speedup,
        comet_time,
        spark_time,
    )
    return result
```

### Monitor Comet Metrics

```python
@dataclass
class CometMetrics:
    """Snapshot of Comet runtime metrics."""
    timestamp: str
    native_execution_ratio: float
    shuffle_native_ratio: float
    cache_hit_rate: float
    fallback_operators: list[str]
    transition_count: int
    total_comet_operators: int
    avg_speedup: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "native_execution_ratio": round(
                self.native_execution_ratio, 4
            ),
            "shuffle_native_ratio": round(self.shuffle_native_ratio, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "fallback_operators": self.fallback_operators,
            "transition_count": self.transition_count,
            "total_comet_operators": self.total_comet_operators,
            "avg_speedup": round(self.avg_speedup, 2),
        }


def monitor_comet_metrics(spark: SparkSession) -> CometMetrics:
    """Collect current Comet execution metrics.

    Parses Spark metrics and EXPLAIN output to determine:
      - Native execution ratio (Comet operators vs total)
      - Shuffle performance (native vs columnar vs Spark)
      - Cache hit rates for columnar cache
      - Fallback operator names and transition nodes

    Args:
        spark: Active SparkSession.

    Returns:
        CometMetrics dataclass with current metrics.
    """
    fallback_pattern = re.compile(
        r"Reverting Comet|fell back|not supported",
        re.IGNORECASE,
    )
    comet_operator_pattern = re.compile(r"(Comet\w+)")
    transition_pattern = re.compile(
        r"(CometColumnarToRow|CometNativeColumnarToRow|"
        r"CometSparkRowToColumnar|CometSparkColumnarToColumnar)"
    )
    exchange_pattern = re.compile(r"(CometExchange|CometColumnarExchange)")

    fallback_ops: list[str] = []
    comet_count = 0
    transition_count = 0
    exchange_count = 0

    try:
        last_plan = spark._jdf.queryExecution().executedPlan().toString()
        plan_text = str(last_plan) if last_plan else ""

        comet_count = len(comet_operator_pattern.findall(plan_text))
        transition_count = len(transition_pattern.findall(plan_text))
        exchange_count = len(exchange_pattern.findall(plan_text))
    except Exception as exc:
        logger.warning("Could not parse plan: %s", exc)

    # Compute native execution ratio
    total_ops = comet_count + transition_count
    native_ratio = (
        (comet_count - transition_count) / comet_count
        if comet_count > 0
        else 0.0
    )

    # Shuffle native ratio (native exchange / total exchange)
    shuffle_native = (
        1.0 if exchange_count > 0 else 0.0
    )

    # Cache hit rate (approximate from Spark metrics)
    try:
        cache_enabled = spark.conf.get(
            "spark.comet.columnarCache.enabled", "false"
        ).lower() == "true"
        cache_hit_rate = 0.0 if not cache_enabled else 0.5  # placeholder
    except Exception:
        cache_hit_rate = 0.0

    metrics = CometMetrics(
        timestamp=datetime.utcnow().isoformat(),
        native_execution_ratio=max(0.0, native_ratio),
        shuffle_native_ratio=shuffle_native,
        cache_hit_rate=cache_hit_rate,
        fallback_operators=fallback_ops,
        transition_count=transition_count,
        total_comet_operators=comet_count,
        avg_speedup=0.0,  # updated after benchmark runs
    )

    logger.info("Comet metrics: %s", json.dumps(metrics.to_dict()))
    return metrics
```

### Comet Optimization Config & Tuning

```python
@dataclass
class CometOptimizationConfig:
    """Structured configuration for Comet-specific optimizations."""
    # Core
    batch_size: int = 4096
    comet_exec_enabled: bool = True

    # Memory
    off_heap_size: str = "4g"
    memory_pool: str = "fair_unified"
    memory_pool_fraction: float = 1.0

    # Shuffle
    shuffle_enabled: bool = True
    shuffle_fallback_to_columnar: bool = True
    revert_redundant_columnar: bool = True

    # Scan
    scan_impl: str = "native_iceberg_compat"
    iceberg_concurrency: int = 4

    # Cache
    columnar_cache_enabled: bool = True

    # ANSI handling (Spark 4.0+)
    ansi_ignore: bool = False

    # Operator controls
    replace_sort_merge_join: bool = False
    window_enabled: bool = False

    # Diagnostics
    explain_fallback: bool = True

    def to_conf_dict(self) -> dict[str, str]:
        """Convert to Spark configuration key-value pairs."""
        return {
            "spark.comet.exec.enabled": str(
                self.comet_exec_enabled
            ).lower(),
            "spark.comet.batchSize": str(self.batch_size),
            "spark.memory.offHeap.enabled": "true",
            "spark.memory.offHeap.size": self.off_heap_size,
            "spark.comet.exec.memoryPool": self.memory_pool,
            "spark.comet.exec.memoryPool.fraction": str(
                self.memory_pool_fraction
            ),
            "spark.comet.exec.shuffle.enabled": str(
                self.shuffle_enabled
            ).lower(),
            "spark.comet.exec.shuffle.fallbackToColumnar": str(
                self.shuffle_fallback_to_columnar
            ).lower(),
            "spark.comet.exec.shuffle.revertRedundantColumnar.enabled": str(
                self.revert_redundant_columnar
            ).lower(),
            "spark.comet.scan.impl": self.scan_impl,
            "spark.comet.scan.icebergNative.enabled": "true",
            "spark.comet.scan.icebergNative.dataFileConcurrencyLimit": str(
                self.iceberg_concurrency
            ),
            "spark.comet.columnarCache.enabled": str(
                self.columnar_cache_enabled
            ).lower(),
            "spark.comet.ansi.ignore": str(self.ansi_ignore).lower(),
            "spark.comet.exec.replaceSortMergeJoin": str(
                self.replace_sort_merge_join
            ).lower(),
            "spark.comet.operator.WindowExec.enabled": str(
                self.window_enabled
            ).lower(),
            "spark.comet.explainFallback.enabled": str(
                self.explain_fallback
            ).lower(),
        }


def apply_comet_tuning(
    spark: SparkSession,
    config: Optional[CometOptimizationConfig] = None,
    profile: Optional[str] = None,
) -> CometOptimizationConfig:
    """Apply Comet-specific optimizations to a running SparkSession.

    Either apply a custom ``CometOptimizationConfig`` or use a named profile.

    Profiles:
        - 'aggressive': large batch sizes, hash joins, maximum native exec
        - 'conservative': moderate settings, safe defaults
        - 'iceberg': optimized for Iceberg table workloads
        - 'testing': minimal settings for validation

    Args:
        spark: Active SparkSession.
        config: Custom optimization config. Ignored if profile is given.
        profile: Named tuning profile.

    Returns:
        The applied CometOptimizationConfig.

    Raises:
        ValueError: If an unknown profile name is provided.
    """
    if profile:
        profiles = {
            "aggressive": CometOptimizationConfig(
                batch_size=8192,
                off_heap_size="8g",
                shuffle_enabled=True,
                replace_sort_merge_join=True,
                memory_pool="greedy_unified",
                columnar_cache_enabled=True,
            ),
            "conservative": CometOptimizationConfig(
                batch_size=4096,
                off_heap_size="4g",
                shuffle_enabled=True,
                replace_sort_merge_join=False,
                memory_pool="fair_unified",
            ),
            "iceberg": CometOptimizationConfig(
                batch_size=4096,
                off_heap_size="8g",
                scan_impl="native_iceberg_compat",
                iceberg_concurrency=8,
                shuffle_enabled=True,
                columnar_cache_enabled=True,
            ),
            "testing": CometOptimizationConfig(
                batch_size=1024,
                off_heap_size="2g",
                shuffle_enabled=False,
                explain_fallback=True,
                columnar_cache_enabled=False,
            ),
        }
        if profile not in profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Choose from: {list(profiles.keys())}"
            )
        config = profiles[profile]
    elif config is None:
        config = CometOptimizationConfig()

    conf_dict = config.to_conf_dict()
    for key, value in conf_dict.items():
        try:
            spark.conf.set(key, value)
        except Exception as exc:
            logger.warning("Failed to set %s=%s: %s", key, value, exc)

    logger.info(
        "Applied Comet tuning profile=%s, %d configs set",
        profile or "custom",
        len(conf_dict),
    )
    return config
```

### CometPipeline Class

```python
@dataclass
class CometPipelineReport:
    """End-to-end pipeline execution report."""
    pipeline_name: str
    status: str  # 'success', 'partial_fallback', 'failure'
    total_time_seconds: float
    stages_completed: int
    stages_total: int
    comet_accelerated_stages: int
    fallback_stages: int
    transition_nodes_found: int
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
            "comet_accelerated_stages": self.comet_accelerated_stages,
            "fallback_stages": self.fallback_stages,
            "transition_nodes_found": self.transition_nodes_found,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class CometPipeline:
    """End-to-end native pipeline with fallback handling and performance reporting.

    Manages a sequence of SQL / DataFrame transformations, verifies Comet
    acceleration at each stage, and produces a structured report.
    """

    def __init__(
        self,
        spark: SparkSession,
        pipeline_name: str = "comet_pipeline",
    ) -> None:
        self.spark = spark
        self.pipeline_name = pipeline_name
        self._stages: list[dict] = []
        self._start_time: float = 0.0
        self._accelerated = 0
        self._fallback = 0
        self._transitions = 0

    def add_stage(self, name: str, sql: str) -> "CometPipeline":
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
    ) -> "CometPipeline":
        """Add a DataFrame-function stage to the pipeline.

        Args:
            name: Human-readable stage name.
            func: Callable that takes (SparkSession, DataFrame) and returns
                  a new DataFrame.
        """
        self._stages.append({"name": name, "df_func": func})
        return self

    def _check_acceleration(self) -> tuple[bool, int]:
        """Check whether the last executed plan used Comet operators.

        Returns:
            Tuple of (is_accelerated, transition_node_count).
        """
        try:
            plan = str(
                self.spark._jdf.queryExecution()
                .executedPlan()
                .toString()
            )
            comet_ops = re.findall(r"Comet\w+", plan)
            transitions = re.findall(
                r"CometColumnarToRow|CometSparkRowToColumnar", plan
            )
            has_native = bool(comet_ops)
            return has_native, len(transitions)
        except Exception:
            return False, 0

    def execute(self) -> CometPipelineReport:
        """Execute all pipeline stages sequentially.

        Returns:
            CometPipelineReport with execution summary.
        """
        self._start_time = time.monotonic()
        df: Optional[DataFrame] = None
        errors: list[str] = []

        for i, stage in enumerate(self._stages):
            stage_name = stage["name"]
            logger.info(
                "Executing stage %d/%d: %s",
                i + 1,
                len(self._stages),
                stage_name,
            )

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

                accelerated, transitions = self._check_acceleration()
                self._transitions += transitions

                if accelerated:
                    self._accelerated += 1
                    logger.info("Stage '%s' accelerated by Comet", stage_name)
                    if transitions > 0:
                        logger.warning(
                            "Stage '%s' has %d transition nodes (partial "
                            "fallback indicators)",
                            stage_name,
                            transitions,
                        )
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

        report = CometPipelineReport(
            pipeline_name=self.pipeline_name,
            status=status,
            total_time_seconds=elapsed,
            stages_completed=completed,
            stages_total=len(self._stages),
            comet_accelerated_stages=self._accelerated,
            fallback_stages=self._fallback,
            transition_nodes_found=self._transitions,
            error_message="; ".join(errors) if errors else None,
        )

        # Write report to /tmp
        report_path = os.path.join(
            "/tmp",
            f"comet_pipeline_{self.pipeline_name}_{int(time.time())}.json",
        )
        with open(report_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Pipeline report written to %s", report_path)

        return report
```

### Production ETL with Comet Acceleration

```python
def run_production_comet_etl(
    spark: SparkSession,
    *,
    source_path: str,
    output_path: str,
    partition_column: str = "date",
    tuning_profile: str = "conservative",
    enable_monitoring: bool = True,
) -> dict:
    """Run a production ETL pipeline with Comet acceleration and monitoring.

    This function orchestrates a complete ETL workflow:
      1. Apply Comet tuning profile
      2. Verify Comet is active
      3. Execute multi-stage ETL (read, transform, aggregate, write)
      4. Collect metrics and produce a JSON report

    Args:
        spark: Active SparkSession.
        source_path: Path to source Parquet data.
        output_path: Path for output Parquet files.
        partition_column: Column to partition output by.
        tuning_profile: Comet tuning profile name.
        enable_monitoring: Whether to collect runtime metrics.

    Returns:
        Dict with ETL summary and performance metrics.
    """
    logger.info("Starting production Comet ETL: source=%s", source_path)

    # --- Step 1: Apply tuning ---
    try:
        config = apply_comet_tuning(spark, profile=tuning_profile)
    except Exception as exc:
        logger.error("Failed to apply Comet tuning: %s", exc)
        return {"status": "error", "message": str(exc)}

    # --- Step 2: Verify Comet ---
    verification = verify_comet_active(spark)
    if not verification.get("plugin_loaded"):
        logger.error("Comet plugin is not loaded. Aborting ETL.")
        return {"status": "error", "message": "Comet plugin not loaded"}

    # --- Step 3: Build pipeline ---
    pipeline = CometPipeline(spark, pipeline_name="production_etl")

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
            metrics = monitor_comet_metrics(spark)
        except Exception as exc:
            logger.warning("Failed to collect metrics: %s", exc)

    # --- Build summary ---
    summary = {
        "status": report.status,
        "pipeline_report": report.to_dict(),
        "comet_config": config.to_conf_dict(),
        "comet_verification": verification,
        "metrics": metrics.to_dict() if metrics else None,
    }

    # Write combined report
    report_path = os.path.join(
        "/tmp", f"comet_etl_report_{int(time.time())}.json"
    )
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("ETL report written to %s", report_path)

    return summary
```

### Usage Examples

```python
# ── Example: Create session and verify ───────────────────────────────────────
# spark = create_comet_session(
#     app_name="MyCometApp",
#     off_heap_size="8g",
#     batch_size=8192,
#     scan_impl="native_iceberg_compat",
#     iceberg_concurrency=4,
# )
# verification = verify_comet_active(spark)
# print(f"Comet active: {verification['native_operators_found']}")
# print(f"Operators found: {verification['operators']}")
# print(f"Transition nodes: {verification['transition_nodes']}")

# ── Example: Run benchmark ───────────────────────────────────────────────────
# result = comet_benchmark(
#     spark,
#     query_name="tpch_q6",
#     query="SELECT SUM(amount) FROM lineitem WHERE ...",
#     data_path="/data/tpch/lineitem/",
# )
# print(f"Speedup: {result.speedup_ratio:.2f}x")

# ── Example: Production ETL with Iceberg ─────────────────────────────────────
# summary = run_production_comet_etl(
#     spark,
#     source_path="s3://data-lake/iceberg/transactions/",
#     output_path="s3://data-lake/curated/transactions_agg/",
#     partition_column="txn_date",
#     tuning_profile="iceberg",
# )
# print(f"ETL status: {summary['status']}")
# print(f"Comet operators: {summary['metrics']['total_comet_operators']}")
```
