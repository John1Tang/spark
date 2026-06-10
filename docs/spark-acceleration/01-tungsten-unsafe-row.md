# Tungsten UnsafeRow, Off-Heap Memory, and Vectorized Readers

## What It Is

Project Tungsten is Spark SQL's low-level execution engine, introduced in Spark 1.5. It re-architects how Spark manages memory and processes data, replacing Java object-based row representations with a compact binary format called **UnsafeRow**, and enabling **vectorized (columnar) batch processing** for I/O-heavy operations.

Three pillars of Tungsten acceleration:

1. **UnsafeRow** -- a binary row format that stores data in contiguous memory, avoiding Java object overhead.
2. **Off-heap memory management** -- memory allocated outside the JVM heap, bypassing GC pressure.
3. **Vectorized readers** -- Parquet/ORC decoders that read data into `ColumnarBatch` structures, processing columns in bulk rather than row-by-row.

Source files:
- `UnsafeRow` -- `sql/catalyst/src/main/java/org/apache/spark/sql/catalyst/expressions/UnsafeRow.java`
- `ColumnarBatch` -- `sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ColumnarBatch.java`
- `VectorizedReaderBase` -- `sql/core/src/main/java/org/apache/spark/sql/execution/datasources/parquet/VectorizedReaderBase.java`
- `VectorizedParquetRecordReader` -- `sql/core/src/main/java/org/apache/spark/sql/execution/datasources/parquet/VectorizedParquetRecordReader.java`
- `SpecificInternalRow` -- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/SpecificInternalRow.scala`

## Why It Works: The Mechanism

### UnsafeRow Binary Memory Layout

`UnsafeRow` stores each row in three contiguous regions within a single memory block ([UnsafeRow.java](sql/catalyst/src/main/java/org/apache/spark/sql/catalyst/expressions/UnsafeRow.java)):

```
+------------------+------------------+---------------------------+
|  Null Bit Set    |   Fixed-Width     |   Variable-Length Portion |
|  (bit per field) |   Values (8B each)|   (strings, arrays, etc) |
+------------------+------------------+---------------------------+
```

**Region 1 -- Null Bit Set:**
- Aligned to 8-byte boundaries
- One bit per field: bit set (1) means NULL, bit clear (0) means not-null
- Width in bytes: `((numFields + 63) / 64) * 8`

**Region 2 -- Fixed-Width Values:**
- One 8-byte word per field
- For primitive types (int, long, double, boolean, etc.): the value is stored directly
- For variable-length types (string, binary, array, struct, map): stores a 64-bit compound value encoding `(offset << 32) | length`
- Offset is relative to the start of the row's backing memory

**Region 3 -- Variable-Length Data:**
- Actual bytes for strings, binary data, nested structs, arrays, maps
- Referenced by the offset+length pairs in Region 2

The entire row size is always a multiple of 8 bytes (see `pointTo()` assertion in UnsafeRow.java line 161).

Example: a row with schema `(id: INT, name: STRING, score: DOUBLE)` and 3 fields:
- Null bit set: 8 bytes (1 word, covers up to 64 fields)
- Fixed values: 3 fields * 8 bytes = 24 bytes
- Variable portion: depends on `name` length
- Total: `8 + 24 + variable_length`, rounded up to multiple of 8

### How UnsafeRow Avoids Java Object Overhead

A traditional `GenericInternalRow` holds an `Object[]` where each element is a boxed Java object:
- Each object has a 12-16 byte header
- References cost 4-8 bytes each
- Boxing primitives (int -> Integer) adds another 16-24 bytes
- Objects are scattered across the heap, hurting cache locality

An `UnsafeRow` instead:
- Stores primitives directly in memory (no boxing)
- Uses `sun.misc.Unsafe` (via `org.apache.spark.unsafe.Platform`) for direct memory access
- Keeps all data in contiguous memory, improving CPU cache utilization
- Acts as a **pointer** to memory -- the UnsafeRow instance itself is tiny; `pointTo(baseObject, baseOffset, sizeInBytes)` re-points it to different backing data without allocation

### Off-Heap Memory and GC Avoidance

Off-heap memory is allocated via `java.nio.DirectByteBuffer` / `sun.misc.Unsafe` and lives **outside** the JVM heap. The JVM garbage collector never scans it.

In Spark, Tungsten uses off-heap memory for:
- `TaskMemoryManager` page allocation (shuffle, sort, join buffers)
- `OffHeapColumnVector` in vectorized Parquet/ORC readers (see `VectorizedParquetRecordReader.java` line 165: `MEMORY_MODE = useOffHeap ? MemoryMode.OFF_HEAP : MemoryMode.ON_HEAP`)
- UnsafeRow backing storage when `spark.memory.offHeap.enabled` is true

**Why this matters:** In a typical Spark SQL query processing millions of rows, the interpreted path creates millions of Java row objects. The GC must scan, mark, and compact them. Off-heap memory + UnsafeRow means Spark manages memory explicitly -- it allocates pages, tracks usage, and frees them when the task completes, with zero GC involvement.

### Vectorized Parquet/ORC Readers

The traditional Parquet reader decodes one row at a time, calling `next()` on a `RecordReader`. The vectorized reader (`VectorizedParquetRecordReader`) works differently:

1. It reads **batches** of rows (default 4096) at once
2. Data is decoded into `WritableColumnVector` structures -- one column vector per column
3. These vectors are wrapped in a `ColumnarBatch` ([ColumnarBatch.java](sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ColumnarBatch.java))
4. The `ColumnarBatch` provides a row-wise view (`getRow(rowId)`) but the data is stored columnar internally
5. `VectorizedReaderBase` ([VectorizedReaderBase.java](sql/core/src/main/java/org/apache/spark/sql/execution/datasources/parquet/VectorizedReaderBase.java)) provides bulk read methods like `readIntegers(total, columnVector, rowId)`, `readLongs(...)`, etc. -- decoding entire columns in one call

This eliminates per-row method call overhead and enables SIMD-friendly columnar processing.

### SpecificInternalRow -- The Mutable Alternative

`SpecificInternalRow` ([SpecificInternalRow.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/SpecificInternalRow.scala)) is a **mutable** row container that uses `MutableValue` objects per column. It is designed to reduce garbage when modifying primitive column values -- instead of creating a new boxed object on each update, it mutates the existing `MutableInt`, `MutableLong`, etc. in place. This is useful for intermediate computation but is superseded by UnsafeRow for the main execution path.

## What Problem It Solves

| Problem | Tungsten Solution |
|---------|------------------|
| Java object overhead (headers, boxing, references) | UnsafeRow stores raw bytes, no objects |
| GC pauses from millions of row allocations | Off-heap memory + explicit lifecycle management |
| Poor CPU cache utilization (scattered objects) | Contiguous memory layout |
| Per-row decode overhead in Parquet/ORC | Vectorized batch decoding (4096 rows at a time) |
| Volcano iterator model function call overhead | Whole-stage codegen (covered in the companion doc) |

## Production Code Examples

### Verifying Tungsten Is Being Used

Run `EXPLAIN` and look for `WholeStageCodegen` (the `(N)` prefix) and `FileScan parquet` with `Batched: true`:

**Scala:**
```scala
val df = spark.read.parquet("/data/large_table")
  .filter($"age" > 30)
  .select($"name", $"salary")

df.explain("extended")
// Look for:
// *(1) Project [name#0, salary#1]
// +- *(1) Filter (age#2 > 30)
//    +- *(1) FileScan parquet [name#0,salary#1,age#2]
//       Batched: true,    <-- vectorized reader is active
//       ...
```

**Python:**
```python
df = (spark.read.parquet("/data/large_table")
      .filter("age > 30")
      .select("name", "salary"))

df.explain("extended")
```

**SQL:**
```sql
EXPLAIN EXTENDED
SELECT name, salary FROM parquet_table WHERE age > 30;
```

The `*(N)` prefix marks operators that are fused into WholeStageCodegen stage N. `Batched: true` confirms the vectorized Parquet reader is active.

### Configuring Off-Heap Memory

Off-heap memory is configured at the SparkSession/application level:

```scala
val spark = SparkSession.builder()
  .appName("OffHeapExample")
  .config("spark.memory.offHeap.enabled", "true")
  .config("spark.memory.offHeap.size", "16g")  // Must be positive when enabled
  .getOrCreate()
```

```python
spark = (SparkSession.builder
  .appName("OffHeapExample")
  .config("spark.memory.offHeap.enabled", "true")
  .config("spark.memory.offHeap.size", "16g")
  .getOrCreate())
```

SQL (session-level toggle only):
```sql
-- Off-heap size cannot be set via SQL; must be set at startup
-- However, you can verify the setting:
SET spark.memory.offHeap.enabled;
SET spark.memory.offHeap.size;
```

**Key configurations:**

| Config | Default | Recommended | Description |
|--------|---------|-------------|-------------|
| `spark.memory.offHeap.enabled` | `false` | `true` for large workloads | Enables off-heap memory |
| `spark.memory.offHeap.size` | `0` | 25-50% of executor memory | Total off-heap bytes per executor |
| `spark.sql.parquet.enableVectorizedReader` | `true` | `true` | Enables vectorized Parquet decoding |
| `spark.sql.parquet.enableNestedColumnVectorizedReader` | `true` | `true` | Vectorized decoding for nested structs/lists/maps |
| `spark.sql.orc.enableVectorizedReader` | `true` | `true` | Enables vectorized ORC decoding |
| `spark.sql.files.maxRecordsPerBatch` | `4096` | `4096` (tune for memory) | Rows per vectorized batch |

### Enabling Vectorized Parquet Reading

Vectorized reading is **on by default**. Explicitly ensure these are set:

```scala
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
spark.conf.set("spark.sql.parquet.enableNestedColumnVectorizedReader", "true")
```

```python
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
spark.conf.set("spark.sql.parquet.enableNestedColumnVectorizedReader", "true")
```

The vectorized reader is automatically selected when:
- Reading Parquet files with native Spark types
- Schema does not contain incompatible types (e.g., certain complex decimals)

### Memory Estimation Examples

**UnsafeRow size estimation:**

For a schema with `N` fields:
```
bitset_width  = ((N + 63) / 64) * 8   // bytes, rounded up
fixed_values  = N * 8                  // 8 bytes per field
variable      = sum of actual var-length data
total         = bitset_width + fixed_values + variable (rounded to multiple of 8)
```

Example: 50 fields, all primitives (no variable-length data):
```
bitset_width  = ((50 + 63) / 64) * 8 = 8 bytes
fixed_values  = 50 * 8 = 400 bytes
variable      = 0
total         = 408 bytes (already multiple of 8)
```

Example: 10 fields, 5 strings averaging 20 bytes each:
```
bitset_width  = ((10 + 63) / 64) * 8 = 8 bytes
fixed_values  = 10 * 8 = 80 bytes
variable      = 5 * 20 = 100 bytes
total         = 8 + 80 + 100 = 188 -> rounded to 192 bytes
```

**ColumnarBatch memory estimation:**

For a batch of 4096 rows with 20 columns of INT (4 bytes each):
```
per_column = 4096 * 4 = 16,384 bytes
total      = 20 * 16,384 = 327,680 bytes (~320 KB)
```

For 20 columns of STRING averaging 50 bytes:
```
per_column = 4096 * (4 byte length + 50 byte data) ~ 221,184 bytes
total      = 20 * 221,184 = 4,423,680 bytes (~4.2 MB)
```

### Off-Heap ColumnVector Selection

The `VectorizedParquetRecordReader` chooses column vector implementation based on configuration (lines 524-542 of VectorizedParquetRecordReader.java):

```
if (useOffHeap) {
  vectors[i] = new OffHeapColumnVector(capacity, fields[i].dataType());
} else {
  vectors[i] = new OnHeapColumnVector(capacity, fields[i].dataType());
}
```

Off-heap vectors allocate native memory and are managed by Spark's `MemoryManager`. On-heap vectors use Java byte arrays and are subject to GC.

## How to Verify It's Working

### EXPLAIN Output

```scala
spark.read.parquet("/data/sales").filter($"amount" > 100).explain("cost")
```

Look for:
- `*(N)` prefix on operators -- WholeStageCodegen is active
- `FileScan parquet [...] Batched: true` -- vectorized reader is active
- `Selected buffers: ...` -- Tungsten memory buffers are being used

### Spark UI Metrics

1. **SQL tab** -- Click on a query stage:
   - "WholeStageCodegen" stages show `pipelineTime` metric
   - Vectorized scans show significantly lower "Scan Time" than row-by-row

2. **Executor tab** -- Compare GC time:
   - With off-heap + UnsafeRow: GC time should be <5% of task time
   - Without: GC time can exceed 30% for I/O-heavy queries

3. **Storage tab** (if caching):
   - "Deserialized Storage Size" with Tungsten is much smaller than raw Java objects

### Programmatic Verification

```scala
import org.apache.spark.sql.execution.SQLExecution

// Capture the physical plan
val plan = df.queryExecution.executedPlan
plan.treeString.foreach(println)

// Check if vectorized reader is used
val scanNodes = plan.collect {
  case scan: org.apache.spark.sql.execution.datasources.FileSourceScanExec => scan
}
scanNodes.foreach { scan =>
  println(s"Vector types: ${scan.vectorTypes}")
  // Some(...) means vectorized reader is active
  // None means row-by-row reader
}
```

## Production-Ready Python (PySpark) Code

### Complete PySpark Session Setup with Tungsten Configuration

```python
"""
Production PySpark session configuration for Tungsten acceleration.

Sets up off-heap memory, vectorized readers, and comprehensive error handling.
"""

import logging
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_tungsten_session(
    app_name: str = "TungstenPipeline",
    off_heap_size: str = "16g",
    num_cores: int = 4,
) -> SparkSession:
    """Create a SparkSession optimized for Tungsten execution.

    Args:
        app_name: Application name shown in Spark UI.
        off_heap_size: Off-heap memory per executor (e.g. "16g", "8192m").
        num_cores: Number of executor cores.

    Returns:
        Configured SparkSession.

    Raises:
        RuntimeError: If the session cannot be created.
    """
    try:
        spark = (
            SparkSession.builder.appName(app_name)
            # --- Tungsten / off-heap ---
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", off_heap_size)
            # --- Memory management ---
            .config("spark.memory.fraction", "0.7")
            .config("spark.memory.storageFraction", "0.3")
            # --- Vectorized Parquet / ORC readers ---
            .config("spark.sql.parquet.enableVectorizedReader", "true")
            .config(
                "spark.sql.parquet.enableNestedColumnVectorizedReader", "true"
            )
            .config("spark.sql.orc.enableVectorizedReader", "true")
            # --- Vectorized batch tuning ---
            .config("spark.sql.files.maxRecordsPerBatch", "4096")
            # --- Code generation ---
            .config("spark.sql.codegen.wholeStage", "true")
            .config("spark.sql.codegen.factoryMode", "CODEGEN_ONLY")
            # --- Executor resources ---
            .config("spark.executor.cores", str(num_cores))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # --- Logging ---
            .config("spark.log.level", "WARN")
            .getOrCreate()
        )

        # Suppress noisy spark logs; user code uses its own logger
        spark.sparkContext.setLogLevel("WARN")

        # Verify critical settings
        off_heap_enabled = spark.conf.get(
            "spark.memory.offHeap.enabled", "false"
        )
        off_heap_size_actual = spark.conf.get("spark.memory.offHeap.size", "0")
        logger.info(
            "Tungsten session created: offHeap.enabled=%s, offHeap.size=%s",
            off_heap_enabled,
            off_heap_size_actual,
        )
        return spark

    except Exception as exc:
        logger.error("Failed to create Tungsten SparkSession: %s", exc)
        raise RuntimeError(
            "Could not initialize SparkSession with Tungsten config"
        ) from exc


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spark = create_tungsten_session(
        app_name="ProductionTungsten",
        off_heap_size="16g",
        num_cores=4,
    )
    try:
        # ... pipeline work ...
        pass
    finally:
        spark.stop()
        logger.info("Spark session stopped.")
```

### Vectorized Parquet Reading with Verification

```python
"""
Read Parquet data with vectorized readers and verify the reader is active.
"""

import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.execution.datasources.v2 import FileScan  # noqa (for type hints)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def verify_vectorized_reader(spark: SparkSession, parquet_path: str) -> bool:
    """Return True if the vectorized Parquet reader is active for *parquet_path*.

    Examines the executed physical plan for ``Batched: true`` and non-empty
    ``vectorTypes`` on the FileSourceScanExec node.
    """
    df = spark.read.parquet(parquet_path)
    executed_plan = df.queryExecution.executedPlan

    # Collect all FileSourceScanExec nodes
    scan_nodes = executed_plan.collect(
        lambda plan: "FileScan parquet" in plan.simpleString()
    )

    if not scan_nodes:
        logger.warning("No FileScan parquet node found in physical plan.")
        return False

    scan = scan_nodes[0]
    vector_types = getattr(scan, "vectorTypes", None)
    is_batched = vector_types is not None and len(vector_types) > 0

    logger.info("FileSourceScanExec: vectorTypes=%s", vector_types)
    logger.info("Batched (vectorized) reader active: %s", is_batched)

    # Also verify via EXPLAIN string as a fallback
    explain_text = df._jdf.queryExecution().executedPlan().toString()
    has_batched = "Batched: true" in explain_text
    logger.info("EXPLAIN 'Batched: true' found: %s", has_batched)

    return is_batched or has_batched


def read_parquet_vectorized(
    spark: SparkSession,
    parquet_path: str,
    filter_expr: str = "",
    select_cols: list[str] | None = None,
) -> tuple:
    """Read a Parquet file with vectorized reader and return (DataFrame, metadata).

    Args:
        spark: Active SparkSession.
        parquet_path: Path to the Parquet dataset.
        filter_expr: Optional SQL filter expression (e.g. "age > 30").
        select_cols: Optional list of columns to project.

    Returns:
        Tuple of (DataFrame, dict of metadata).

    Raises:
        ValueError: If the path is empty.
        RuntimeError: If the vectorized reader cannot be verified.
    """
    if not parquet_path:
        raise ValueError("parquet_path must be non-empty")

    reader = spark.read.parquet(parquet_path)

    if filter_expr:
        reader = reader.filter(filter_expr)
    if select_cols:
        reader = reader.select(*select_cols)

    # Force verification that vectorized reader is active
    is_vectorized = verify_vectorized_reader(
        spark.read.parquet(parquet_path), parquet_path
    )
    if not is_vectorized:
        logger.error("Vectorized reader is NOT active for %s", parquet_path)
        raise RuntimeError("Vectorized Parquet reader is not active")

    logger.info("Vectorized reader verified for %s", parquet_path)
    schema = reader.schema
    logger.info("Schema: %s", schema.simpleString())

    return reader, {"schema": schema, "vectorized": True}


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from session_setup import create_tungsten_session

    spark = create_tungsten_session()
    try:
        df, meta = read_parquet_vectorized(
            spark,
            parquet_path="s3://data-lake/sales/year=2025",
            filter_expr="amount > 100 AND region = 'US'",
            select_cols=["customer_id", "amount", "date"],
        )
        logger.info("Row count: %d", df.count())
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise
    finally:
        spark.stop()
```

### Off-Heap Memory Monitoring via Spark UI Metrics

```python
"""
Monitor off-heap memory usage and GC overhead programmatically.

Pulls executor metrics from SparkListener / Spark UI to compare
GC time against total task time.
"""

import logging
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_task_metrics(spark: SparkSession) -> list[dict]:
    """Return task-level metrics from the Spark UI REST API.

    Uses the internal ``/api/v1/applications/<appId>/stages`` endpoint
    to fetch task summary data including GC time.

    Returns:
        A list of dicts with keys: stage_id, task_time_ms, gc_time_ms,
        peak_exec_mem_bytes.
    """
    app_id = spark.sparkContext.applicationId
    base_url = spark.sparkContext._uiWebUrl  # type: ignore[attr-defined]
    if base_url is None:
        logger.warning("Spark UI not available; cannot fetch metrics.")
        return []

    import urllib.request
    import json

    url = f"{base_url}/api/v1/applications/{app_id}/stages?details=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            stages = json.loads(resp.read())
    except Exception as exc:
        logger.error("Failed to fetch stage metrics: %s", exc)
        return []

    metrics = []
    for stage in stages:
        summary = stage.get("taskMetrics", {})
        gc = summary.get("jvmGCTime", 0)  # milliseconds
        exec_runtime = summary.get("executorRunTime", 0)
        peak_mem = summary.get("peakExecutionMemory", 0)
        metrics.append(
            {
                "stage_id": stage.get("stageId"),
                "task_time_ms": exec_runtime,
                "gc_time_ms": gc,
                "peak_exec_mem_bytes": peak_mem,
            }
        )
    return metrics


def analyze_gc_overhead(metrics: list[dict]) -> dict:
    """Compute GC overhead ratio from task metrics.

    Args:
        metrics: Output of ``get_task_metrics``.

    Returns:
        Dict with total_task_time_ms, total_gc_time_ms, gc_ratio_pct,
        and a health assessment string.
    """
    if not metrics:
        return {"status": "no_data"}

    total_task = sum(m["task_time_ms"] for m in metrics)
    total_gc = sum(m["gc_time_ms"] for m in metrics)
    gc_ratio = (total_gc / total_task * 100) if total_task > 0 else 0.0

    if gc_ratio < 5:
        health = "HEALTHY -- GC overhead is minimal (<5%%)"
    elif gc_ratio < 15:
        health = "WARNING -- GC overhead is moderate (5-15%%)"
    else:
        health = "CRITICAL -- GC overhead is high (>15%%); consider off-heap"

    result = {
        "total_task_time_ms": total_task,
        "total_gc_time_ms": total_gc,
        "gc_ratio_pct": round(gc_ratio, 2),
        "num_stages": len(metrics),
        "status": "healthy" if gc_ratio < 5 else "warning" if gc_ratio < 15 else "critical",
        "assessment": health,
    }
    logger.info(
        "GC analysis: task=%dms, gc=%dms, ratio=%.2f%% -- %s",
        total_task,
        total_gc,
        gc_ratio,
        health,
    )
    return result


def monitor_offheap_health(spark: SparkSession) -> dict:
    """Full off-heap health check: config verification + GC analysis.

    Returns:
        Dict with off_heap_enabled, off_heap_size, and gc_analysis.
    """
    off_heap_enabled = spark.conf.get("spark.memory.offHeap.enabled", "false")
    off_heap_size = spark.conf.get("spark.memory.offHeap.size", "0")

    logger.info("Off-heap config: enabled=%s, size=%s", off_heap_enabled, off_heap_size)

    # Run a trivial action to populate metrics
    spark.sparkContext.parallelize(range(1)).count()

    metrics = get_task_metrics(spark)
    gc_analysis = analyze_gc_overhead(metrics)

    return {
        "off_heap_enabled": off_heap_enabled == "true",
        "off_heap_size": off_heap_size,
        "gc_analysis": gc_analysis,
    }


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from session_setup import create_tungsten_session

    spark = create_tungsten_session()
    try:
        health = monitor_offheap_health(spark)
        logger.info("Off-heap health: %s", health)
        if health["gc_analysis"]["status"] == "critical":
            logger.error("GC overhead is critical -- review off-heap config")
            sys.exit(1)
    finally:
        spark.stop()
```

### UnsafeRow Memory Size Calculator (Python)

```python
"""
Calculate the in-memory size of an UnsafeRow given field count and
variable-length data sizes.

Mirrors the layout documented in the UnsafeRow Java source:
  Region 1: Null bit set  -- ((numFields + 63) / 64) * 8 bytes
  Region 2: Fixed values  -- numFields * 8 bytes
  Region 3: Variable data -- sum of var-length field bytes
  Total: rounded up to the nearest multiple of 8.
"""

import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def unsafe_row_size(num_fields: int, variable_bytes: int = 0) -> int:
    """Compute the UnsafeRow size in bytes.

    Args:
        num_fields: Number of columns in the row.
        variable_bytes: Total bytes of variable-length data
                        (strings, arrays, binary, etc.).

    Returns:
        Size in bytes, rounded up to a multiple of 8.
    """
    if num_fields <= 0:
        raise ValueError("num_fields must be positive")
    if variable_bytes < 0:
        raise ValueError("variable_bytes must be non-negative")

    bitset_width = math.ceil((num_fields + 63) / 64) * 8
    fixed_values = num_fields * 8
    raw_total = bitset_width + fixed_values + variable_bytes
    # Round up to the nearest multiple of 8
    total = math.ceil(raw_total / 8) * 8
    return total


def unsafe_row_size_detailed(
    schema_fields: list[dict],
) -> dict:
    """Compute UnsafeRow size from a field-level schema description.

    Args:
        schema_fields: List of dicts like
            ``[{"name": "id", "type": "int", "var_bytes": 0}, ...]``.
            ``var_bytes`` is the expected average byte size for variable-length
            fields (0 for fixed-width types).

    Returns:
        Dict with bitset_width, fixed_values, variable_bytes, total_bytes,
        and per_field_breakdown.
    """
    num_fields = len(schema_fields)
    total_variable = sum(f.get("var_bytes", 0) for f in schema_fields)

    bitset_width = math.ceil((num_fields + 63) / 64) * 8
    fixed_values = num_fields * 8
    raw_total = bitset_width + fixed_values + total_variable
    total = math.ceil(raw_total / 8) * 8

    per_field = []
    running_offset = bitset_width + fixed_values
    for f in schema_fields:
        field_size = 8 if f.get("var_bytes", 0) == 0 else (8 + f["var_bytes"])
        per_field.append(
            {
                "name": f["name"],
                "type": f["type"],
                "fixed_bytes": 8,
                "variable_bytes": f.get("var_bytes", 0),
                "total_bytes": field_size,
            }
        )

    return {
        "num_fields": num_fields,
        "bitset_width": bitset_width,
        "fixed_values": fixed_values,
        "variable_bytes": total_variable,
        "total_bytes": total,
        "per_field": per_field,
    }


def estimate_dataset_memory(
    num_rows: int,
    num_fields: int,
    avg_variable_bytes: int = 0,
) -> dict:
    """Estimate total UnsafeRow memory for a dataset.

    Args:
        num_rows: Number of rows in the dataset.
        num_fields: Number of columns per row.
        avg_variable_bytes: Average variable-length bytes per row.

    Returns:
        Dict with row_size_bytes, total_bytes, total_human.
    """
    row_size = unsafe_row_size(num_fields, avg_variable_bytes)
    total = row_size * num_rows

    def human(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    logger.info(
        "Dataset memory: %d rows x %d fields -> %s per row, %s total",
        num_rows,
        num_fields,
        human(row_size),
        human(total),
    )
    return {
        "row_size_bytes": row_size,
        "total_bytes": total,
        "total_human": human(total),
    }


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: 50 primitive fields
    size = unsafe_row_size(num_fields=50)
    logger.info("50 primitive fields: %d bytes per row", size)

    # Example 2: 10 fields, 5 strings of 20 bytes each
    size2 = unsafe_row_size(num_fields=10, variable_bytes=5 * 20)
    logger.info("10 fields + 5x20B strings: %d bytes per row", size2)

    # Example 3: detailed breakdown
    schema = [
        {"name": "id", "type": "int", "var_bytes": 0},
        {"name": "name", "type": "string", "var_bytes": 50},
        {"name": "score", "type": "double", "var_bytes": 0},
    ]
    detail = unsafe_row_size_detailed(schema)
    logger.info("Detailed row: %s", detail)

    # Example 4: dataset estimate
    estimate_dataset_memory(num_rows=10_000_000, num_fields=20, avg_variable_bytes=200)
```

### Column Vector Inspection (OnHeap vs OffHeap)

```python
"""
Inspect whether Spark is using OnHeapColumnVector or OffHeapColumnVector
for vectorized reads.

This uses the internal JVM types exposed through PySpark's gateway.
"""

import logging
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_column_vectors(
    spark: SparkSession, parquet_path: str, limit: int = 1
) -> dict:
    """Inspect the column vector memory mode for a Parquet scan.

    Reads a tiny sample and inspects the underlying ColumnarBatch
    through the JVM gateway to determine OnHeap vs OffHeap allocation.

    Args:
        spark: Active SparkSession with off-heap configured.
        parquet_path: Path to the Parquet dataset.
        limit: Number of rows to sample (keep small).

    Returns:
        Dict with column names, vector class names, and memory mode summary.
    """
    try:
        jvm = spark._jvm  # type: ignore[attr-defined]

        # Build a small RDD to trigger a scan and inspect the internal batch
        df = spark.read.parquet(parquet_path).limit(limit)

        # Get the executed plan
        executed_plan = df.queryExecution.executedPlan
        plan_str = executed_plan.toString()
        logger.info("Physical plan: %s", plan_str)

        # Check the vectorTypes attribute on FileSourceScanExec
        scan_nodes = executed_plan.collect(
            lambda plan: "FileScan" in plan.simpleString()
        )

        results = {"columns": [], "memory_mode": "unknown"}

        for scan in scan_nodes:
            vector_types = getattr(scan, "vectorTypes", None)
            if vector_types:
                results["memory_mode"] = (
                    "off-heap"
                    if spark.conf.get("spark.memory.offHeap.enabled", "false")
                    == "true"
                    else "on-heap"
                )
                results["vector_types"] = [str(v) for v in vector_types]
                for i, vt in enumerate(vector_types):
                    results["columns"].append(
                        {"index": i, "vector_type": str(vt)}
                    )
                logger.info(
                    "Vector types: %s (mode=%s)",
                    results["vector_types"],
                    results["memory_mode"],
                )
            else:
                logger.warning("No vectorTypes found; row-by-row reader may be active")
                results["memory_mode"] = "row-by-row (not vectorized)"

        return results

    except Exception as exc:
        logger.error("Failed to inspect column vectors: %s", exc)
        raise


def verify_vector_mode(spark: SparkSession) -> dict:
    """Quick verification of on-heap vs off-heap column vector mode.

    Returns a dict summarising the active mode and confirming config alignment.
    """
    off_heap = spark.conf.get("spark.memory.offHeap.enabled", "false")
    off_heap_size = spark.conf.get("spark.memory.offHeap.size", "0")
    vectorized = spark.conf.get(
        "spark.sql.parquet.enableVectorizedReader", "false"
    )

    mode = "off-heap" if off_heap == "true" else "on-heap"
    logger.info(
        "Vector mode: %s (offHeap.enabled=%s, offHeap.size=%s, vectorized=%s)",
        mode,
        off_heap,
        off_heap_size,
        vectorized,
    )

    return {
        "memory_mode": mode,
        "off_heap_enabled": off_heap == "true",
        "off_heap_size": off_heap_size,
        "vectorized_reader_enabled": vectorized == "true",
    }


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from session_setup import create_tungsten_session

    spark = create_tungsten_session()
    try:
        mode = verify_vector_mode(spark)
        logger.info("Vector mode: %s", mode)

        results = inspect_column_vectors(
            spark, parquet_path="/data/sample_table"
        )
        logger.info("Column vector inspection: %s", results)
    finally:
        spark.stop()
```

### Performance Comparison: Interpreted vs Vectorized Reading

```python
"""
Benchmark: row-by-row (interpreted) vs vectorized Parquet reading.

Measures wall-clock time, GC overhead, and throughput for both modes
on the same dataset, then logs a comparison report.
"""

import json
import logging
import time
import sys
from datetime import datetime
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _run_query(spark: SparkSession, parquet_path: str, label: str) -> dict:
    """Execute a standard query and collect timing / GC metrics.

    Args:
        spark: Active SparkSession.
        parquet_path: Parquet dataset path.
        label: Human-readable label for the run.

    Returns:
        Dict with timing and metric details.
    """
    # Clear any cached data to ensure a fair comparison
    spark.catalog.clearCache()

    df = (
        spark.read.parquet(parquet_path)
        .filter("amount > 100")
        .select("customer_id", "amount", "transaction_date")
    )

    start = time.perf_counter()
    row_count = df.count()
    elapsed = time.perf_counter() - start

    # Fetch GC metrics from Spark UI
    app_id = spark.sparkContext.applicationId
    ui_url = spark.sparkContext._uiWebUrl  # type: ignore[attr-defined]
    gc_time_ms = 0
    exec_time_ms = 0

    if ui_url:
        import urllib.request
        api = f"{ui_url}/api/v1/applications/{app_id}/stages?details=true"
        try:
            with urllib.request.urlopen(api, timeout=30) as resp:
                stages = json.loads(resp.read())
                gc_time_ms = sum(
                    s.get("taskMetrics", {}).get("jvmGCTime", 0)
                    for s in stages
                )
                exec_time_ms = sum(
                    s.get("taskMetrics", {}).get("executorRunTime", 0)
                    for s in stages
                )
        except Exception as exc:
            logger.warning("Could not fetch Spark UI metrics: %s", exc)

    gc_ratio = (gc_time_ms / exec_time_ms * 100) if exec_time_ms > 0 else 0.0

    result = {
        "label": label,
        "row_count": row_count,
        "elapsed_seconds": round(elapsed, 3),
        "gc_time_ms": gc_time_ms,
        "exec_time_ms": exec_time_ms,
        "gc_ratio_pct": round(gc_ratio, 2),
        "throughput_rows_per_sec": round(row_count / elapsed, 0)
        if elapsed > 0
        else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(
        "[%s] rows=%d, time=%.3fs, gc_ratio=%.2f%%, throughput=%.0f rows/s",
        label,
        row_count,
        elapsed,
        gc_ratio,
        result["throughput_rows_per_sec"],
    )
    return result


def run_vectorized_benchmark(
    spark: SparkSession,
    parquet_path: str,
    output_report: str | None = None,
) -> dict:
    """Run both interpreted and vectorized benchmarks and compare.

    Args:
        spark: Active SparkSession.
        parquet_path: Path to the Parquet dataset.
        output_report: Optional file path to write the JSON report.

    Returns:
        Dict with both runs and a speedup ratio.
    """
    logger.info("=== Starting Tungsten Performance Benchmark ===")
    logger.info("Dataset: %s", parquet_path)

    # --- Run 1: Vectorized (Tungsten enabled, defaults) ---
    logger.info("--- Run 1: Vectorized reader (Tungsten ON) ---")
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
    vectorized_result = _run_query(spark, parquet_path, "vectorized")

    # --- Run 2: Row-by-row (vectorized disabled) ---
    logger.info("--- Run 2: Interpreted reader (vectorized OFF) ---")
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
    interpreted_result = _run_query(spark, parquet_path, "interpreted")

    # Re-enable vectorized reader
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")

    # --- Comparison ---
    v_time = vectorized_result["elapsed_seconds"]
    i_time = interpreted_result["elapsed_seconds"]
    speedup = round(i_time / v_time, 2) if v_time > 0 else float("inf")
    gc_improvement = (
        round(
            interpreted_result["gc_ratio_pct"] - vectorized_result["gc_ratio_pct"],
            2,
        )
    )

    report = {
        "vectorized": vectorized_result,
        "interpreted": interpreted_result,
        "speedup_x": speedup,
        "gc_ratio_improvement_pp": gc_improvement,
        "summary": (
            f"Vectorized is {speedup}x faster than interpreted. "
            f"GC ratio improved by {gc_improvement} percentage points."
        ),
    }

    logger.info("=== Benchmark Complete ===")
    logger.info("Speedup: %sx", speedup)
    logger.info("GC improvement: %s pp", gc_improvement)
    logger.info("%s", report["summary"])

    if output_report:
        with open(output_report, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report written to %s", output_report)

    return report


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from session_setup import create_tungsten_session

    spark = create_tungsten_session()
    try:
        report = run_vectorized_benchmark(
            spark,
            parquet_path="s3://data-lake/benchmark/large_table",
            output_report="tungsten_benchmark_report.json",
        )
        print(json.dumps(report, indent=2, default=str))
    finally:
        spark.stop()
```

### Production Pipeline: Large Parquet ETL with Error Handling and Retry

```python
"""
Production-grade ETL pipeline reading large Parquet datasets,
applying filters and transformations, with retry logic, logging,
and proper resource cleanup.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp, lit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Immutable configuration for the ETL pipeline."""

    app_name: str = "TungstenETLPipeline"
    input_path: str = ""
    output_path: str = ""
    off_heap_size: str = "16g"
    max_records_per_batch: int = 4096
    filter_expression: str = "status = 'active'"
    select_columns: list[str] = field(
        default_factory=lambda: ["id", "name", "amount", "date", "region"]
    )
    output_format: str = "parquet"
    output_mode: str = "overwrite"
    max_retries: int = 3
    retry_delay_seconds: int = 5
    checkpoint_path: str = ""

    def __post_init__(self) -> None:
        if not self.input_path:
            raise ValueError("input_path is required")
        if not self.output_path:
            raise ValueError("output_path is required")


def create_pipeline_session(cfg: PipelineConfig) -> SparkSession:
    """Create a SparkSession tuned for the ETL pipeline.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Configured SparkSession.
    """
    spark = (
        SparkSession.builder.appName(cfg.app_name)
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", cfg.off_heap_size)
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.parquet.enableNestedColumnVectorizedReader", "true")
        .config("spark.sql.files.maxRecordsPerBatch", str(cfg.max_records_per_batch))
        .config("spark.sql.codegen.wholeStage", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(
        "Pipeline session created: app=%s, offHeap=%s",
        cfg.app_name,
        cfg.off_heap_size,
    )
    return spark


def execute_with_retry(
    spark: SparkSession, cfg: PipelineConfig
) -> dict[str, Any]:
    """Execute the ETL pipeline with automatic retry on transient failures.

    Args:
        spark: Active SparkSession.
        cfg: Pipeline configuration.

    Returns:
        Dict with status, row counts, elapsed time, and attempt details.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    attempt = 0
    last_error: Exception | None = None
    attempts_log: list[dict] = []

    while attempt < cfg.max_retries:
        attempt += 1
        attempt_start = time.perf_counter()
        attempt_info: dict[str, Any] = {
            "attempt": attempt,
            "status": "pending",
        }

        try:
            logger.info(
                "=== Pipeline attempt %d/%d ===", attempt, cfg.max_retries
            )

            # --- Read ---
            logger.info("Reading Parquet from %s", cfg.input_path)
            df: DataFrame = spark.read.parquet(cfg.input_path)
            input_schema = df.schema
            logger.info("Input schema: %s", input_schema.simpleString())

            # --- Verify vectorized reader ---
            explain_text = df._jdf.queryExecution().executedPlan().toString()
            if "Batched: true" not in explain_text:
                logger.warning(
                    "Vectorized reader may not be active for this dataset"
                )

            # --- Filter ---
            logger.info("Applying filter: %s", cfg.filter_expression)
            df = df.filter(cfg.filter_expression)

            # --- Select ---
            available_cols = [
                c for c in cfg.select_columns if c in df.columns
            ]
            missing = set(cfg.select_columns) - set(available_cols)
            if missing:
                logger.warning("Columns not found in schema (skipping): %s", missing)
            df = df.select(*available_cols)

            # --- Transform: add audit columns ---
            df = (
                df.withColumn("processed_at", current_timestamp())
                .withColumn("pipeline_version", lit("1.0.0"))
            )

            # --- Write ---
            logger.info(
                "Writing to %s (format=%s, mode=%s)",
                cfg.output_path,
                cfg.output_format,
                cfg.output_mode,
            )

            if cfg.checkpoint_path:
                df.write.option("checkpointLocation", cfg.checkpoint_path)

            writer = df.write.mode(cfg.output_mode).format(cfg.output_format)
            writer.save(cfg.output_path)

            # --- Verification read ---
            verify_df = spark.read.parquet(cfg.output_path)
            output_count = verify_df.count()
            elapsed = time.perf_counter() - attempt_start

            attempt_info.update(
                {
                    "status": "success",
                    "output_rows": output_count,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            logger.info(
                "Pipeline succeeded on attempt %d: %d rows in %.3fs",
                attempt,
                output_count,
                elapsed,
            )
            attempts_log.append(attempt_info)
            return {
                "status": "success",
                "output_rows": output_count,
                "elapsed_seconds": round(elapsed, 3),
                "attempts": attempts_log,
                "completed_at": datetime.utcnow().isoformat(),
            }

        except Exception as exc:
            elapsed = time.perf_counter() - attempt_start
            attempt_info.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            attempts_log.append(attempt_info)
            last_error = exc

            if attempt < cfg.max_retries:
                delay = cfg.retry_delay_seconds * attempt  # exponential-ish
                logger.warning(
                    "Attempt %d failed (%s). Retrying in %ds...",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d attempts exhausted. Last error: %s",
                    cfg.max_retries,
                    exc,
                )

    raise RuntimeError(
        f"Pipeline failed after {cfg.max_retries} attempts"
    ) from last_error


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    """End-to-end pipeline entry point with resource cleanup.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Result dict from ``execute_with_retry``.
    """
    spark: SparkSession | None = None
    try:
        spark = create_pipeline_session(cfg)
        result = execute_with_retry(spark, cfg)
        return result
    except Exception as exc:
        logger.error("Pipeline terminated with error: %s", exc)
        raise
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped; pipeline resources released.")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tungsten ETL Pipeline")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument("--off-heap-size", default="16g")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        off_heap_size=args.off_heap_size,
        max_retries=args.max_retries,
        checkpoint_path=args.checkpoint,
    )

    result = run_pipeline(config)
    logger.info("Pipeline result: %s", result)

    if result["status"] != "success":
        sys.exit(1)
```

## Limitations and Gotchas

### UnsafeRow Limitations

1. **Not externally visible:** UnsafeRow is an internal format. You cannot directly inspect or manipulate UnsafeRow data from user code. The `get()` methods return deserialized values.

2. **No in-place mutation of variable-length fields:** `update()` in UnsafeRow throws `SparkUnsupportedOperationException` (line 209-211 of UnsafeRow.java). Only fixed-length fields (`setInt`, `setLong`, etc.) and specific types (`setDecimal`, `setInterval`) support mutation.

3. **Row copy on certain operations:** When a codegen operator produces multiple output rows from one input row (e.g., joins), the row must be copied (`.copy()`) before buffering, which allocates a new byte array (UnsafeRow.java line 502-514).

### Vectorized Reader Limitations

1. **Schema compatibility:** Vectorized reading is disabled automatically when:
   - Parquet schema has type mismatches with Spark schema
   - Complex decimals requiring more than 8 bytes (handled by fallback)
   - INT96 timestamps requiring timezone conversion (may fall back)

2. **Nested column vectorized reading:** Requires `spark.sql.parquet.enableNestedColumnVectorizedReader = true` (default true since Spark 3.3). Disabling this forces row-by-row reading for nested data, causing a significant performance drop.

3. **Batch size tuning:** `spark.sql.files.maxRecordsPerBatch` (default 4096) controls the vectorized batch size. Larger batches improve throughput but increase memory pressure. For wide tables with many string columns, consider reducing to 2048.

### Off-Heap Memory Gotchas

1. **Must set size when enabled:** `spark.memory.offHeap.enabled=true` without a positive `spark.memory.offHeap.size` will fail at startup.

2. **Not counted in heap metrics:** Off-heap memory usage does not appear in JVM heap metrics or standard GC logs. Monitor via Spark UI's "Memory" column or `spark.memory.offHeap.size`.

3. **Does not eliminate all GC:** Only Tungsten-managed structures (UnsafeRow backing, ColumnVector data) use off-heap. Catalyst expression objects, SparkPlan instances, and other JVM objects still create GC pressure.

4. **Native memory tracking:** Off-heap memory comes from the process's native memory space. Ensure the container/host has sufficient total memory (heap + off-heap + OS overhead).

### Performance Anti-Patterns

1. **Small files:** Vectorized readers gain little from small files (< 128MB) because setup overhead dominates. Compact small files first.

2. **Highly selective filters pushed down:** If a Parquet predicate filter eliminates >99% of rows, the vectorized batch decode cost may outweigh benefits since most decoded rows are immediately discarded.

3. **Disabling codegen:** Tungsten acceleration works best with WholeStageCodegen (see the companion document). Disabling `spark.sql.codegen.wholeStage` significantly reduces the benefit of UnsafeRow and vectorized readers.
