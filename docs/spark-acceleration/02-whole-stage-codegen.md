# Whole-Stage Code Generation

## What It Is

Whole-Stage Code Generation (WholeStageCodegen) is Spark SQL's technique for fusing an entire pipeline of physical operators into a **single compiled Java function**. Instead of executing operators as a chain of iterator-based stages (the Volcano model), Spark generates custom Java source code at query planning time, compiles it with Janino on the driver, ships it to executors, and runs the compiled bytecode.

Source files:
- `WholeStageCodegenExec` -- `sql/core/src/main/scala/org/apache/spark/sql/execution/WholeStageCodegenExec.scala`
- `CodegenContext` -- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/codegen/CodegenContext.scala`
- `CodeGenerator` -- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/codegen/CodeGenerator.scala`
- `CodegenSupport` trait -- defined in `WholeStageCodegenExec.scala` (line 47)
- `CollapseCodegenStages` rule -- defined in `WholeStageCodegenExec.scala` (line 914)

## Why It Works: The Mechanism

### The Volcano Iterator Model Overhead

The traditional Volcano model represents a query plan as a tree of operators, each implementing `next()`:

```
Project.next()
  -> Filter.next()
       -> Scan.next()
```

Each call to `next()` incurs:
- **Virtual function dispatch** -- polymorphic method call
- **Function call overhead** -- stack frame creation
- **CPU branch misprediction** -- the call chain is unpredictable
- **Instruction cache misses** -- code is scattered across many classes

For a query processing billions of rows, these overheads are measured in **seconds of pure call overhead**.

### The Produce/Consume Model

WholeStageCodegen replaces the pull-based Volcano model with a **push-based produce/consume model** built on the `CodegenSupport` trait:

```scala
trait CodegenSupport extends SparkPlan {
  def produce(ctx: CodegenContext, parent: CodegenSupport): String
  def doProduce(ctx: CodegenContext): String
  def consume(ctx: CodegenContext, outputVars: Seq[ExprCode], row: String): String
  def doConsume(ctx: CodegenContext, input: Seq[ExprCode], row: ExprCode): String
}
```

**`produce()`** -- Generates code that **produces** rows. A leaf operator (scan) generates a loop that iterates over input data. An internal operator (filter, project) generates logic that processes rows from its child.

**`consume()`** -- Generates code that **consumes** rows produced by the current operator and passes them to the parent. It calls `parent.doConsume()`.

The call graph (from `WholeStageCodegenExec.scala` lines 619-636):

```
WholeStageCodegen.execute()
  -> child.produce(ctx, this)        // Start from root, walk down
       -> doProduce(child)
            -> child.produce(ctx, self)   // Recursively produce from children
                 -> doProduce(grandchild)
                      -> consume(ctx, vars, row)  // Walk back up
                           -> parent.doConsume(ctx, input, row)
```

### Code Generation Walkthrough

Taking a simple `Project -> Filter -> Scan` pipeline:

**1. Scan (leaf) -- `InputRDDCodegen.doProduce()`** (line 471-501):
```java
while (input.hasNext()) {
  InternalRow row = (InternalRow) input.next();
  // consume: call Filter's doConsume
  if (shouldStop()) return;
}
```

**2. Filter -- `doProduce()` emits** the predicate check and **`doConsume()`** passes matching rows to Project:
```java
// Inside the scan loop (inlined by codegen):
boolean isNull_value = false;
int value = row.getInt(2);  // age column
if (!isNull_value && value > 30) {
  // consume: call Project's doConsume
}
```

**3. Project -- `doConsume()`** extracts the needed columns and appends to output buffer:
```java
UTF8String value_name = row.getUTF8String(0);
double value_salary = row.getDouble(1);
append(unsafeRow);
```

The final result is a **single Java class** with a `processNext()` method containing all three operators inlined:

```java
final class GeneratedIteratorForCodegenStage1 extends BufferedRowIterator {
  private Object[] references;
  private scala.collection.Iterator[] inputs;

  public void init(int index, scala.collection.Iterator[] inputs) {
    partitionIndex = index;
    this.inputs = inputs;
  }

  protected void processNext() throws java.io.IOException {
    while (input.hasNext()) {
      InternalRow scan_row = (InternalRow) input.next();
      // Filter predicate (inlined)
      int filter_value = scan_row.getInt(2);
      if (filter_value > 30) {
        // Project columns (inlined)
        append(unsafeRow);
      }
      if (shouldStop()) return;
    }
  }
}
```

### Janino Compilation

The generated Java source is compiled using [Janino](https://janino-compiler.github.io/janino/), a lightweight embeddable Java compiler. The compilation pipeline (from `CodeGenerator.scala` line 1510-1571):

1. `CodeGenerator.compile(code)` is called on the driver
2. Janino's `ClassBodyEvaluator` compiles the source to bytecode
3. The compiled class is cached (`NonFateSharingCache`) to avoid re-compilation of identical queries
4. The bytecode is shipped to executors as part of the task serialization
5. On the executor, `WholeStageCodegenEvaluatorFactory` instantiates the class and calls `processNext()` for each partition

Key compilation constants from `CodeGenerator.scala`:
- `DEFAULT_JVM_HUGE_METHOD_LIMIT = 8000` -- bytecode size beyond which HotSpot stops JIT-optimizing
- `MAX_JVM_METHOD_PARAMS_LENGTH = 255` -- max JVM method parameter length
- `MAX_JVM_CONSTANT_POOL_SIZE = 65535` -- max constant pool entries
- `GENERATED_CLASS_SIZE_THRESHOLD = 1000000` -- 1MB class size threshold before splitting to inner classes
- `MERGE_SPLIT_METHODS_THRESHOLD = 3` -- threshold for merging split methods

### Subexpression Elimination

`CodegenContext` tracks `EquivalentExpressions` to detect and eliminate common subexpressions. For a query like:

```sql
SELECT (a + b), (a + b) / c FROM t
```

The expression `a + b` is computed once and stored in a local variable, then reused. This is handled in `CodegenContext.subexpressionElimination()` (line 1273-1318) and `subexpressionEliminationForWholeStageCodegen()` (line 1152-1266).

### Mutable State Management

`CodegenContext` manages mutable states (line 295-320) that become member variables of the generated class:
- **Inlined states** -- primitive types placed directly as fields (e.g., `private int count;`)
- **Compacted array states** -- grouped into arrays when there are many states of the same type (e.g., `private long[] mutableStateArray = new long[32768];`)

This avoids creating Java objects for intermediate state, further reducing GC pressure.

### Codegen Stage IDs

The `CollapseCodegenStages` rule (line 914-995) walks the physical plan and wraps contiguous codegen-supporting subtrees in `WholeStageCodegenExec` nodes, each assigned a unique stage ID. The ID appears in EXPLAIN output as `*(N)`:

```
*(5) SortMergeJoin [x#3L], [y#9L], Inner
:- *(2) Sort [x#3L ASC NULLS FIRST]
:  +- Exchange hashpartitioning(x#3L, 200)
:     +- *(1) Project [(id#0L % 2) AS x#3L]
:        +- *(1) Filter isnotnull((id#0L % 2))
:           +- *(1) Range (0, 5, step=1, splits=8)
+- *(4) Sort [y#9L ASC NULLS FIRST]
   +- Exchange hashpartitioning(y#9L, 200)
      +- *(3) Project [(id#6L % 2) AS y#9L]
         +- *(3) Filter isnotnull((id#6L % 2))
            +- *(3) Range (0, 5, step=1, splits=8)
```

Stages 1, 2 are one pipeline; stages 3, 4 are another. Stage 5 (SortMergeJoin) is its own pipeline. The `Exchange` (shuffle) **breaks** the codegen pipeline because data must be materialized and repartitioned.

## What Problem It Solves

| Problem | WholeStageCodegen Solution |
|---------|---------------------------|
| Virtual function dispatch per row | Single compiled function, no dispatch |
| Per-operator function call overhead | All operators inlined into one method |
| CPU instruction cache misses | Tight, linear bytecode |
| Redundant expression evaluation | Subexpression elimination at compile time |
| JIT cannot optimize across operators | Generated method is small enough for C2 compilation |

**Measured improvement:** The Tungsten paper reports 5-10x speedup over the Volcano model for CPU-bound queries.

## Production Code Examples

### EXPLAIN CODEGEN Output

**Scala:**
```scala
val df = spark.range(10000)
  .toDF("id")
  .filter($"id" % 2 === 0)
  .select($"id" * 10 as "scaled_id")

// Show codegen stages in the physical plan
df.explain("codegen")
```

Example output:
```
Found 1 WholeStage codegen stages.

Stage 0:
*(1) Project [(id#0L * 10) AS scaled_id#3L]
+- *(1) Filter ((id#0L % 2) = 0)
   +- *(1) Range (0, 10000, step=1, splits=8)
```

**Extended EXPLAIN with code details:**
```scala
df.explain("extended")
// Shows the physical plan with codegen stage annotations
```

**Accessing the generated source code:**

```scala
import org.apache.spark.sql.execution.WholeStageCodegenExec

val plan = df.queryExecution.executedPlan
val codegenStages = plan.collect {
  case ws: WholeStageCodegenExec => ws
}

codegenStages.foreach { stage =>
  val (ctx, code) = stage.doCodeGen()
  println(s"=== Stage ${stage.codegenStageId} ===")
  println(code.body)
}
```

### Reading Generated Code from Spark UI

1. Run a query with WholeStageCodegen
2. Open the Spark Web UI (`http://<driver>:4040`)
3. Go to the **SQL** tab
4. Click on the query description
5. Click on a **WholeStageCodegen** stage
6. Click **"Generated Java Source Code"** (or "Generated Bytecode" for compiled output)

You will see the full generated class, including:
- The `GeneratedIteratorForCodegenStageN` class definition
- All `processNext()` logic
- Any nested inner classes (for large stages)
- Comments marking PRODUCE and CONSUME boundaries

### Spark Shell Quick Verification

```scala
spark.conf.set("spark.sql.codegen.logLevel", "INFO")
// This prints generated code to the driver log at INFO level

val df = spark.sql("SELECT /*+ COALESCE(1) */ id, id * 2 FROM range(100)")
df.collect()
// Check driver stdout for "Code generated in X ms" log line
```

## Key Configurations

| Config | Default | Recommended | Description |
|--------|---------|-------------|-------------|
| `spark.sql.codegen.wholeStage` | `true` | `true` | Master toggle for whole-stage codegen |
| `spark.sql.codegen.fallback` | `true` | `true` | If codegen fails to compile, fall back to interpreted execution |
| `spark.sql.codegen.maxFields` | `100` | `100` (increase for wide tables) | Max fields before codegen is disabled. Counts nested fields too. |
| `spark.sql.codegen.hugeMethodLimit` | `65535` | `8000` on HotSpot | Max bytecode size per method. Above 8000, HotStop skips JIT. Set to 8000 for better JIT behavior. |
| `spark.sql.codegen.methodSplitThreshold` | `1024` | `1024` | Source code char threshold for splitting a function into smaller ones |
| `spark.sql.codegen.splitConsumeFuncByOperator` | `true` | `true` | Put each operator's consume logic into a separate method |
| `spark.sql.codegen.useIdInClassName` | `true` | `true` | Embed stage ID in generated class name (helps debugging) |
| `spark.sql.codegen.factoryMode` | `FALLBACK` | `FALLBACK` | `FALLBACK`, `CODEGEN_ONLY`, or `NO_CODEGEN` |
| `spark.sql.codegen.logLevel` | `DEBUG` | `DEBUG` | Log level for generated code. Set to `INFO` to print code |
| `spark.sql.codegen.logging.maxLines` | `1000` | `1000` | Max lines to log on codegen errors (-1 for unlimited) |
| `spark.sql.codegen.aggregate.map.twolevel.enabled` | `true` | `true` | Use two-level hash map for codegen'd aggregates |
| `spark.sql.codegen.aggregate.splitAggregateFunc.enabled` | varies | `true` | Split aggregate functions into separate methods |

### Configuring for Wide Tables

If your table has > 100 columns (or > 100 nested fields), codegen is automatically disabled:

```scala
// From WholeStageCodegenExec.isTooManyFields():
//   numOfNestedFields(dataType) > conf.wholeStageMaxNumFields
// where numOfNestedFields recursively counts all fields in structs, maps, arrays
```

Increase the limit for wide tables:
```scala
spark.conf.set("spark.sql.codegen.maxFields", "500")
```

Be aware: wider codegen stages take longer to compile and may exceed the huge method limit.

## When Codegen Is Disabled

Whole-stage codegen is **automatically disabled** in these scenarios:

### 1. Too Many Fields

From `CollapseCodegenStages.supportCodegen()` (line 925-935):
```scala
val hasTooManyOutputFields =
  WholeStageCodegenExec.isTooManyFields(conf, plan.schema)
val hasTooManyInputFields =
  plan.children.exists(p => WholeStageCodegenExec.isTooManyFields(conf, p.schema))
```

If the schema (including nested fields) exceeds `spark.sql.codegen.maxFields`, codegen is disabled for that operator and all its children.

### 2. Complex Expressions (CodegenFallback)

Expressions that implement `CodegenFallback` cannot participate in codegen. If any expression in a plan node has `CodegenFallback`, the entire node falls back:
```scala
val willFallback = plan.expressions.exists(_.exists(e => !supportCodegen(e)))
```

### 3. Compilation Failure

If Janino fails to compile the generated source, and `spark.sql.codegen.fallback = true`, the stage falls back to interpreted execution:

```scala
// WholeStageCodegenExec.doExecute() line 741-749:
val (_, compiledCodeStats) = try {
  CodeGenerator.compile(cleanedSource)
} catch {
  case NonFatal(_) if !Utils.isTesting && conf.codegenFallback =>
    logWarning("Whole-stage codegen disabled for plan...")
    return child.execute()  // Interpreted fallback
}
```

### 4. Bytecode Too Large

If the compiled method exceeds `spark.sql.codegen.hugeMethodLimit`:
```scala
if (compiledCodeStats.maxMethodCodeSize > conf.hugeMethodLimit) {
  logInfo("Found too long generated codes and JIT optimization might not work...")
  return child.execute()  // Interpreted fallback
}
```

### 5. Domain Objects in Output

If an operator's output contains `ObjectType` (Java objects), WholeStageCodegen is not inserted because domain objects cannot be serialized into UnsafeRow:
```scala
case plan if plan.output.length == 1 && plan.output.head.dataType.isInstanceOf[ObjectType] =>
  plan.withNewChildren(plan.children.map(insertWholeStageCodegen))
```

### 6. Columnar Execution

If a plan `supportsColumnar` (e.g., certain AQE-optimized paths), it cannot participate in whole-stage codegen, which is row-based:
```scala
assert(!plan.supportsColumnar)  // line 981
```

### 7. Specific Operator Types

The following operators are never wrapped in WholeStageCodegen:
- `LocalTableScanExec` -- fast driver-local collect/take paths
- `EmptyRelationExec` -- empty results need no processing
- `CommandResultExec` -- DDL/DML commands

## Pipeline Breakers (What Breaks the Codegen Pipeline)

A codegen pipeline is a contiguous chain of `CodegenSupport` operators. The pipeline breaks (a new WholeStageCodegen stage starts) when:

### Shuffle (Exchange)

The most common breaker. `Exchange` operators repartition data, requiring full materialization:
```
*(2) Project
   +- Exchange hashpartitioning(key, 200)    // <-- Pipeline break
      +- *(1) Filter                         // <-- New stage
```

### Sort

`SortExec` is a blocking operator (`BlockingOperatorWithCodegen`). It must consume all input before producing output:
```
*(2) Sort                                    // <-- New pipeline (blocking)
   +- Exchange
      +- *(1) Scan                           // <-- Separate pipeline
```

### Aggregate

`HashAggregateExec` and `SortAggregateExec` are also blocking -- they buffer all input rows before emitting results.

### SortMergeJoin and ShuffledHashJoin

Join operators that require shuffled input wrap their children in `InputAdapter` (separate codegen stages):
```scala
case j: SortMergeJoinExec =>
  j.withNewChildren(j.children.map(
    child => InputAdapter(insertWholeStageCodegen(child))))  // Separate stages
```

### BroadcastHashJoin

Unlike sort-based joins, broadcast hash join CAN be in a single pipeline with its streamed side, because the broadcast side is already materialized:
```
*(3) BroadcastHashJoin
:- *(2) Project        // Streamed side - can be in same stage
   +- *(1) Scan
+- BroadcastExchange   // <-- Pipeline break for broadcast side
```

### InputAdapter

`InputAdapter` (line 511-577) wraps a non-codegen child and bridges it into a codegen pipeline. It generates the loop that iterates over the child's RDD iterator.

## Performance Comparison: Interpreted vs Codegen

### Micro-benchmark Pattern

```scala
// Disable codegen for baseline
spark.conf.set("spark.sql.codegen.wholeStage", "false")
val t1 = System.nanoTime()
df_no_codegen.collect()
val interpreted_ms = (System.nanoTime() - t1) / 1e6

// Enable codegen for comparison
spark.conf.set("spark.sql.codegen.wholeStage", "true")
val t2 = System.nanoTime()
df_with_codegen.collect()
val codegen_ms = (System.nanoTime() - t2) / 1e6

println(s"Interpreted: ${interpreted_ms}ms, Codegen: ${codegen_ms}ms, " +
        s"Speedup: ${interpreted_ms / codegen_ms}x")
```

### Typical Results by Query Type

| Query Pattern | Interpreted | Codegen | Speedup |
|--------------|-------------|---------|---------|
| Filter + Project (10M rows) | ~800ms | ~150ms | 5.3x |
| Aggregation (SUM, COUNT, GROUP BY) | ~1200ms | ~200ms | 6.0x |
| Join (hash, 5M x 5M) | ~2500ms | ~600ms | 4.2x |
| Complex expressions (nested CASE WHEN) | ~1500ms | ~300ms | 5.0x |

*Numbers are illustrative and depend on hardware, data distribution, and Spark version.*

### Why the Speedup

1. **No virtual dispatch:** The interpreter calls `next()` millions of times. Codegen inlines everything.
2. **JIT optimization:** HotSpot C2 compiler can inline, unroll loops, and apply SIMD to the generated method.
3. **Null check elimination:** Codegen knows nullability at compile time and elides unnecessary null checks.
4. **Specialized accessors:** `row.getInt(2)` directly reads from memory. The interpreter goes through generic `get(ordinal, DataType)`.
5. **Subexpression elimination:** Common expressions are computed once at compile time.

## Tuning Recommendations

### For CPU-Bound Queries (many expressions, filters)
- Keep `spark.sql.codegen.wholeStage = true` (default)
- Set `spark.sql.codegen.hugeMethodLimit = 8000` to match HotSpot JIT threshold
- Set `spark.sql.codegen.logLevel = DEBUG` to inspect generated code for inefficiencies

### For Wide Tables (> 100 columns)
- Increase `spark.sql.codegen.maxFields` to 500+
- Monitor compile time; if excessive, split queries into narrower projections
- Consider `spark.sql.codegen.splitConsumeFuncByOperator = true` to split consume logic

### For Complex Aggregates
- Enable `spark.sql.codegen.aggregate.map.twolevel.enabled = true` for large group-by cardinality
- Enable `spark.sql.codegen.aggregate.splitAggregateFunc.enabled = true` to avoid oversized methods

### For Debugging Compilation Failures
- Set `spark.sql.codegen.fallback = true` (default) to avoid query failures
- Set `spark.sql.codegen.logLevel = INFO` to see generated source
- Check executor logs for "Internal compiler error" messages

## Limitations and Gotchas

1. **Compilation overhead on small data:** For queries returning few rows, codegen compile time can exceed query execution time. Spark tracks total compile time in `WholeStageCodegenExec.codeGenTime` for monitoring.

2. **Janino cache misses:** Generated code is cached, but different query text (even semantically identical) produces different bytecode. Parameterized queries (prepared statements) help cache hits.

3. **Not all operators support codegen:** Certain complex operators (e.g., certain window functions, UDFs with complex types) fall back. Check the physical plan for `*(N)` prefixes.

4. **Codegen and columnar execution are mutually exclusive:** As seen in `WholeStageCodegenExec.scala` line 981: `assert(!plan.supportsColumnar)`. If an operator uses columnar execution, it cannot participate in whole-stage codegen.

5. **Large generated code and broadcasting:** When the generated source + comments exceed `spark.sql.codegen.broadcastCleanedSourceThreshold`, Spark broadcasts the code instead of serializing it with each task. The default is `-1` (disabled). For very complex queries, consider enabling this.

6. **Union codegen (Spark 4.2+):** `spark.sql.codegen.wholeStage.union.enabled` (default `false`) allows `UnionExec` to participate in whole-stage codegen fusion. When enabled, up to `spark.sql.codegen.wholeStage.union.maxChildren` (default 64) children can be fused into a single stage.

7. **No CodegenBoundary hint:** Unlike some other query engines, Spark does not provide a user-facing hint to manually break or force codegen pipelines. Pipeline boundaries are determined entirely by the `CollapseCodegenStages` rule based on operator semantics.

8. **Testing impact:** In test environments with small data, codegen compile time dominates runtime. The code tracks total compile time via `_compileTime` in `CodeGenerator.scala` for analysis. Consider disabling codegen in unit tests if compilation time is a bottleneck: `spark.conf.set("spark.sql.codegen.factoryMode", "NO_CODEGEN")`.

## Production-Ready Python (PySpark) Code

### 1. Complete PySpark Session Setup with Codegen Configuration

```python
"""Production PySpark session with whole-stage codegen tuning."""
import logging
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_spark_session(
    app_name: str = "whole-stage-codegen-etl",
    master: str = "yarn",
    num_executors: int = 50,
    executor_cores: int = 4,
    executor_memory: str = "16g",
) -> SparkSession:
    """Build a SparkSession tuned for whole-stage codegen performance.

    Parameters
    ----------
    app_name : str
        Application name shown in the Spark UI.
    master : str
        Deploy mode (yarn, k8s, local[*], etc.).
    num_executors : int
        Number of executors to request.
    executor_cores : int
        Cores per executor -- more cores = more parallel codegen eval.
    executor_memory : str
        Memory per executor.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # -- Core codegen toggles --
        .config("spark.sql.codegen.wholeStage", "true")
        .config("spark.sql.codegen.fallback", "true")
        # -- JVM / bytecode tuning --
        .config("spark.sql.codegen.hugeMethodLimit", "8000")
        .config("spark.sql.codegen.methodSplitThreshold", "1024")
        .config("spark.sql.codegen.splitConsumeFuncByOperator", "true")
        # -- Field / method limits --
        .config("spark.sql.codegen.maxFields", "200")
        .config("spark.sql.codegen.logging.maxLines", "2000")
        # -- Union codegen (Spark 4.2+) --
        .config("spark.sql.codegen.wholeStage.union.enabled", "true")
        .config("spark.sql.codegen.wholeStage.union.maxChildren", "64")
        # -- Aggregate optimisations --
        .config("spark.sql.codegen.aggregate.map.twolevel.enabled", "true")
        .config("spark.sql.codegen.aggregate.splitAggregateFunc.enabled", "true")
        # -- Source broadcasting for very large generated code --
        .config("spark.sql.codegen.broadcastCleanedSourceThreshold", "10240")
        # -- Debugging --
        .config("spark.sql.codegen.logLevel", "DEBUG")
        .config("spark.sql.codegen.useIdInClassName", "true")
        .config("spark.sql.factoryMode", "FALLBACK")
        # -- Executor resources --
        .config("spark.executor.instances", num_executors)
        .config("spark.executor.cores", executor_cores)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    # Runtime verification
    codegen = spark.conf.get("spark.sql.codegen.wholeStage")
    fallback = spark.conf.get("spark.sql.codegen.fallback")
    huge_limit = spark.conf.get("spark.sql.codegen.hugeMethodLimit")
    max_fields = spark.conf.get("spark.sql.codegen.maxFields")
    log_level = spark.conf.get("spark.sql.codegen.logLevel")

    logger.info(
        "Codegen config verified -- wholeStage=%s, fallback=%s, "
        "hugeMethodLimit=%s, maxFields=%s, logLevel=%s",
        codegen, fallback, huge_limit, max_fields, log_level,
    )

    return spark
```

### 2. EXPLAIN-Based Verification: Check WholeStageCodegen Activity and Parse *(N) Prefixes

```python
"""Verify that WholeStageCodegen is active by parsing EXPLAIN output."""
import logging
import re
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

# Pattern: *(N) where N is the codegen stage id
_CODEGEN_STAGE_RE = re.compile(r"^\s*\*\((\d+)\)\s+")


def parse_codegen_stages(explain_text: str) -> Dict[int, List[str]]:
    """Parse the *(N) stage prefixes from EXPLAIN output.

    Returns a dict mapping stage id -> list of operator descriptions
    belonging to that stage.

    Example
    -------
    >>> plan_text = df._jdf.queryExecution().executedPlan().toString()
    >>> stages = parse_codegen_stages(plan_text)
    >>> for sid, ops in sorted(stages.items()):
    ...     logger.info("Stage %d: %s", sid, ops[0])
    """
    stages: Dict[int, List[str]] = {}
    for line in explain_text.splitlines():
        m = _CODEGEN_STAGE_RE.match(line)
        if m:
            stage_id = int(m.group(1))
            operator = line[m.end():].strip()
            stages.setdefault(stage_id, []).append(operator)
    return stages


def verify_codegen_active(df: DataFrame) -> bool:
    """Return True if at least one WholeStageCodegen stage exists in the plan.

    Logs a summary of every discovered stage.
    """
    plan_text = df._jdf.queryExecution().executedPlan().toString()
    stages = parse_codegen_stages(plan_text)

    if not stages:
        logger.warning("No WholeStageCodegen stages found -- "
                       "operators may be running interpreted.")
        return False

    for sid, ops in sorted(stages.items()):
        logger.info("Codegen stage %d -- root operator: %s (%d ops total)",
                     sid, ops[0], len(ops))

    logger.info("WholeStageCodegen is ACTIVE with %d stage(s)", len(stages))
    return True


# ---- Usage ----
if __name__ == "__main__":
    spark: SparkSession = create_spark_session()
    try:
        df = spark.range(10_000_000).toDF("id") \
            .filter("id % 2 = 0") \
            .selectExpr("id * 10 AS scaled_id")
        verify_codegen_active(df)
    except Exception:
        logger.exception("EXPLAIN verification failed")
    finally:
        spark.stop()
```

### 3. Generated Code Inspection Utility

```python
"""Retrieve and log the generated Java source code from Spark's CodeGenerator."""
import logging
from typing import List, Tuple

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def inspect_generated_code(spark: SparkSession, sql_text: str) -> List[Tuple[int, str]]:
    """Execute a SQL query and return (stage_id, generated_java_source) pairs.

    Uses the internal CodeGenerator log path: when spark.sql.codegen.logLevel
    is set to INFO or above, generated source appears in the driver log.
    Here we also use the programmatic path via WholeStageCodegenExec.

    Parameters
    ----------
    spark : SparkSession
        Active session with codegen enabled.
    sql_text : str
        SQL statement to execute.

    Returns
    -------
    list of (stage_id, java_source)
        Empty list if no codegen stages were found.
    """
    df = spark.sql(sql_text)

    # Force execution to trigger codegen
    df.collect()

    # Access the Scala API to extract generated source
    jvm = spark.sparkContext._jvm
    plan = df._jdf.queryExecution().executedPlan()

    # Collect all WholeStageCodegenExec nodes
    ws_class = jvm.org.apache.spark.sql.execution.WholeStageCodegenExec

    def collect_codegen(p) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        if p.getClass() == ws_class or isinstance(p, ws_class):
            stage_id = p.codegenStageId()
            # doCodeGen returns a CodeGenResult with (CodegenContext, String)
            codegen_result = p.doCodeGen()
            ctx = codegen_result[0]
            code = codegen_result[1]
            results.append((stage_id, code))
        for child in java_as_list(p.children()):
            results.extend(collect_codegen(child))
        return results

    def java_as_list(jlist):
        """Convert a Scala Seq to a Python list."""
        return list(jlist.toSeq()) if hasattr(jlist, "toSeq") else list(jlist)

    stages = collect_codegen(plan)

    for stage_id, source in stages:
        logger.info("=== Generated Java Source for Stage %d ===", stage_id)
        logger.info(source)
        # Also write to a local file for deep inspection
        out_path = f"/tmp/spark_codegen_stage_{stage_id}.java"
        with open(out_path, "w") as f:
            f.write(source)
        logger.info("Written stage %d source to %s", stage_id, out_path)

    if not stages:
        logger.warning("No WholeStageCodegen stages found for: %s", sql_text)

    return stages


# ---- Usage ----
if __name__ == "__main__":
    spark = create_spark_session()
    spark.conf.set("spark.sql.codegen.logLevel", "INFO")
    try:
        inspect_generated_code(
            spark,
            "SELECT id, id * 2 AS doubled FROM range(1000) WHERE id % 3 = 0",
        )
    except Exception:
        logger.exception("Code inspection failed")
    finally:
        spark.stop()
```

### 4. Performance Benchmark: Interpreted vs Codegen-Enabled Execution

```python
"""Benchmark comparing interpreted (codegen off) vs codegen-enabled execution."""
import gc
import logging
import time
from dataclasses import dataclass
from typing import List

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    mode: str
    compile_ms: float
    execute_ms: float
    total_ms: float
    gc_before_pct: float
    gc_after_pct: float
    row_count: int


def _gc_usage() -> float:
    """Return approximate JVM GC overhead percentage (best-effort)."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 3) * 100  # GB
    except ImportError:
        return 0.0


def _force_gc():
    """Run a full garbage collection cycle."""
    gc.collect()


def run_query_timed(
    spark: SparkSession,
    build_df: callable,
    label: str,
    iterations: int = 3,
) -> List[BenchmarkResult]:
    """Execute the DataFrame builder multiple times and collect timing.

    Parameters
    ----------
    spark : SparkSession
    build_df : callable
        Function(spark) -> DataFrame that builds the query plan.
    label : str
        Human-readable label for this run (e.g. "codegen-on").
    iterations : int
        Number of warmup+measurement iterations.
    """
    results: List[BenchmarkResult] = []
    for i in range(iterations):
        _force_gc()
        gc_before = _gc_usage()

        t0 = time.perf_counter()
        df = build_df(spark)
        t_compile = time.perf_counter()

        rows = df.collect()
        t1 = time.perf_counter()

        gc_after = _gc_usage()

        compile_ms = (t_compile - t0) * 1000
        execute_ms = (t1 - t_compile) * 1000
        total_ms = (t1 - t0) * 1000

        results.append(BenchmarkResult(
            mode=label,
            compile_ms=round(compile_ms, 2),
            execute_ms=round(execute_ms, 2),
            total_ms=round(total_ms, 2),
            gc_before_pct=round(gc_before, 4),
            gc_after_pct=round(gc_after, 4),
            row_count=len(rows),
        ))
        logger.info(
            "[%s] iter %d -- compile=%.1fms, exec=%.1fms, total=%.1fms, "
            "rows=%d, gc_before=%.4f%%, gc_after=%.4f%%",
            label, i + 1, compile_ms, execute_ms, total_ms,
            len(rows), gc_before, gc_after,
        )

    return results


def benchmark_codegen(spark: SparkSession) -> None:
    """Run the full interpreted-vs-codegen benchmark and print a summary."""

    def build_query(s: SparkSession) -> DataFrame:
        return (
            s.range(10_000_000).toDF("id")
            .filter("id % 3 = 0")
            .filter("id > 1000")
            .selectExpr(
                "id",
                "id * 10 AS scaled",
                "CAST(id AS STRING) AS id_str",
            )
        )

    # Phase 1: Interpreted (codegen OFF)
    logger.info("=== Phase 1: Interpreted execution (codegen OFF) ===")
    spark.conf.set("spark.sql.codegen.wholeStage", "false")
    interpreted_results = run_query_timed(spark, build_query, "interpreted", 3)

    # Phase 2: Codegen ON
    logger.info("=== Phase 2: Codegen-enabled execution ===")
    spark.conf.set("spark.sql.codegen.wholeStage", "true")
    codegen_results = run_query_timed(spark, build_query, "codegen", 3)

    # Summary
    avg_interpreted = sum(r.total_ms for r in interpreted_results) / len(interpreted_results)
    avg_codegen = sum(r.total_ms for r in codegen_results) / len(codegen_results)
    speedup = avg_interpreted / avg_codegen if avg_codegen > 0 else float("inf")

    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info("Interpreted avg: %.2f ms", avg_interpreted)
    logger.info("Codegen avg:     %.2f ms", avg_codegen)
    logger.info("Speedup:         %.2fx", speedup)
    logger.info("=" * 60)


# ---- Usage ----
if __name__ == "__main__":
    spark = create_spark_session()
    try:
        benchmark_codegen(spark)
    except Exception:
        logger.exception("Benchmark failed")
    finally:
        spark.stop()
```

### 5. Production ETL Pipeline Using Whole-Stage Codegen

```python
"""Production ETL pipeline leveraging whole-stage codegen end-to-end.

Flow: read Parquet -> multi-filter -> aggregation -> join -> write Parquet.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum as _sum, count, lit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ETLConfig:
    """Immutable configuration container (follows the immutability principle)."""

    def __init__(
        self,
        input_path: str,
        reference_path: str,
        output_path: str,
        checkpoint_dir: str,
        min_revenue: float = 1000.0,
        min_quantity: int = 5,
        date_partition_col: str = "sale_date",
    ):
        self.input_path = input_path
        self.reference_path = reference_path
        self.output_path = output_path
        self.checkpoint_dir = checkpoint_dir
        self.min_revenue = min_revenue
        self.min_quantity = min_quantity
        self.date_partition_col = date_partition_col


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def read_transactions(spark: SparkSession, path: str) -> DataFrame:
    """Read the fact table from Parquet with schema inference and validation."""
    logger.info("Reading transactions from %s", path)
    df = spark.read.format("parquet").load(path)

    required_cols = {"transaction_id", "product_id", "revenue", "quantity", "sale_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in transactions: {missing}")

    logger.info("Transactions schema: %d columns, %d partitions",
                 len(df.columns), df.rdd.getNumPartitions())
    return df


def read_products(spark: SparkSession, path: str) -> DataFrame:
    """Read the dimension table from Parquet."""
    logger.info("Reading products from %s", path)
    df = spark.read.format("parquet").load(path)

    required_cols = {"product_id", "product_name", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in products: {missing}")

    return df


def filter_transactions(df: DataFrame, config: ETLConfig) -> DataFrame:
    """Apply multi-filter: revenue and quantity thresholds.

    These filter + any subsequent select/project will fuse into a single
    whole-stage codegen pipeline.
    """
    logger.info("Applying filters: revenue >= %s, quantity >= %s",
                 config.min_revenue, config.min_quantity)
    return (
        df
        .filter(col("revenue") >= config.min_revenue)
        .filter(col("quantity") >= config.min_quantity)
        .filter(col("revenue").isNotNull())
        .filter(col("quantity").isNotNull())
    )


def aggregate_sales(df: DataFrame) -> DataFrame:
    """Group by product_id and compute aggregates.

    HashAggregate is a blocking operator, so it starts a new codegen stage.
    """
    logger.info("Aggregating sales by product_id")
    return (
        df
        .groupBy("product_id")
        .agg(
            _sum("revenue").alias("total_revenue"),
            _sum("quantity").alias("total_quantity"),
            count("transaction_id").alias("transaction_count"),
        )
    )


def join_with_products(
    aggregated: DataFrame,
    products: DataFrame,
) -> DataFrame:
    """Join aggregated sales with product dimension.

    BroadcastHashJoin is preferred here since products is a dimension table.
    """
    logger.info("Joining aggregated sales with products (broadcast)")
    return aggregated.join(
        products.hint("broadcast"),
        on="product_id",
        how="inner",
    ).select(
        col("product_id"),
        col("product_name"),
        col("category"),
        col("total_revenue"),
        col("total_quantity"),
        col("transaction_count"),
    )


def write_output(df: DataFrame, path: str, partition_col: Optional[str] = None) -> None:
    """Write the final result as partitioned Parquet with Snappy compression."""
    logger.info("Writing output to %s", path)

    writer = (
        df.write
        .mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
    )
    if partition_col:
        writer = writer.partitionBy(partition_col)
    writer.save(path)
    logger.info("Output written successfully")


def verify_codegen_plan(df: DataFrame) -> None:
    """Log a warning if no codegen stages are found in the final plan."""
    plan = df._jdf.queryExecution().executedPlan().toString()
    has_codegen = "*(1)" in plan or "*(2)" in plan or "*(3)" in plan or "*(4)" in plan
    if not has_codegen:
        logger.warning("No WholeStageCodegen stages detected in final plan!")
    else:
        logger.info("WholeStageCodegen stages present in final execution plan")


def run_etl(spark: SparkSession, config: ETLConfig) -> None:
    """Execute the full ETL pipeline with codegen verification at each step."""
    logger.info("Starting ETL pipeline (codegen enabled)")

    spark.sparkContext.setCheckpointDir(config.checkpoint_dir)

    # Verify codegen is on
    assert spark.conf.get("spark.sql.codegen.wholeStage") == "true", \
        "wholeStage codegen must be enabled"

    try:
        # Step 1: Read
        transactions = read_transactions(spark, config.input_path)
        products = read_products(spark, config.reference_path)

        # Step 2: Filter (fuses with Scan into one codegen stage)
        filtered = filter_transactions(transactions, config)
        verify_codegen_plan(filtered)

        # Step 3: Aggregate (new codegen stage -- blocking operator)
        aggregated = aggregate_sales(filtered)
        verify_codegen_plan(aggregated)

        # Step 4: Join (may be a separate codegen stage with BroadcastHashJoin)
        joined = join_with_products(aggregated, products)
        verify_codegen_plan(joined)

        # Step 5: Write
        write_output(joined, config.output_path, partition_col="category")

        logger.info("ETL pipeline completed successfully")

    except ValueError as e:
        logger.error("ETL validation error: %s", e)
        raise
    except Exception:
        logger.exception("ETL pipeline failed")
        raise


# ---- Usage ----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session(app_name="codegen-etl-pipeline")
    try:
        cfg = ETLConfig(
            input_path="s3://data-lake/transactions/",
            reference_path="s3://data-lake/products/",
            output_path="s3://data-lake/output/sales_summary/",
            checkpoint_dir="s3://data-lake/checkpoints/",
            min_revenue=1000.0,
            min_quantity=5,
        )
        run_etl(spark, cfg)
    except Exception:
        logger.exception("Pipeline execution failed")
        sys.exit(1)
    finally:
        spark.stop()
```

### 6. Codegen Fallback Detection Utility

```python
"""Detect which operators fell back from whole-stage codegen to interpreted execution."""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

_CODEGEN_RE = re.compile(r"^\s*\*\((\d+)\)")
_ALL_OPERATORS_RE = re.compile(
    r"^\s*(?:[-+:|]*\s*)?([A-Za-z]+(?:Exec|Scan|Exchange|Join|Aggregate|Sort|Project|Filter|Union|Window))"
)


@dataclass
class FallbackReport:
    """Summary of codegen coverage for a DataFrame's physical plan."""
    total_operators: int = 0
    codegen_operators: int = 0
    interpreted_operators: int = 0
    codegen_stage_ids: List[int] = field(default_factory=list)
    interpreted_ops: List[str] = field(default_factory=list)
    codegen_coverage_pct: float = 0.0

    @property
    def has_fallback(self) -> bool:
        return self.interpreted_operators > 0


def detect_codegen_fallback(df: DataFrame) -> FallbackReport:
    """Analyse the physical plan and report codegen coverage.

    Operators prefixed with *(N) are inside a WholeStageCodegen stage.
    Operators without that prefix (and not Exchange / leaf nodes) may
    have fallen back to interpreted execution.

    Parameters
    ----------
    df : DataFrame
        A DataFrame whose plan should be inspected.

    Returns
    -------
    FallbackReport
    """
    plan = df._jdf.queryExecution().executedPlan().toString()
    logger.info("Physical plan:\n%s", plan)

    report = FallbackReport()
    codegen_stage_ids: set = set()
    all_lines = plan.splitlines()

    for line in all_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Determine operator name
        op_match = _ALL_OPERATORS_RE.match(stripped)
        if not op_match:
            continue

        op_name = op_match.group(1)
        report.total_operators += 1

        cg_match = _CODEGEN_RE.match(stripped)
        if cg_match:
            codegen_stage_ids.add(int(cg_match.group(1)))
            report.codegen_operators += 1
        else:
            # Known non-codegen operators (not fallbacks)
            if op_name in ("Exchange", "BroadcastExchange", "SubqueryBroadcast",
                           "CustomShuffleReader", "InputAdapter"):
                continue
            report.interpreted_operators += 1
            report.interpreted_ops.append(stripped)

    report.codegen_stage_ids = sorted(codegen_stage_ids)
    if report.total_operators > 0:
        report.codegen_coverage_pct = round(
            report.codegen_operators / report.total_operators * 100, 1
        )

    if report.has_fallback:
        logger.warning(
            "CODEGEN FALLBACK DETECTED: %d/%d operators interpreted (%.1f%% coverage). "
            "Fallback operators: %s",
            report.interpreted_operators,
            report.total_operators,
            report.codegen_coverage_pct,
            report.interpreted_ops,
        )
    else:
        logger.info(
            "FULL CODEGEN COVERAGE: %d operators across stage(s) %s",
            report.codegen_operators,
            report.codegen_stage_ids,
        )

    return report


# ---- Usage ----
if __name__ == "__main__":
    spark = create_spark_session()
    try:
        # A query with a UDF that forces CodegenFallback
        from pyspark.sql.functions import udf
        from pyspark.sql.types import IntegerType

        @udf(returnType=IntegerType())
        def complex_python_logic(x):
            """Python UDF forces interpreted fallback."""
            return x * 2 + 1

        df = (
            spark.range(1_000_000).toDF("id")
            .withColumn("computed", complex_python_logic(col("id")))
            .filter("computed > 100")
        )

        report = detect_codegen_fallback(df)
        if report.has_fallback:
            logger.info("Consider rewriting the UDF as a SQL expression "
                         "to restore full codegen coverage.")
    except Exception:
        logger.exception("Fallback detection failed")
    finally:
        spark.stop()
```

### 7. Union Codegen Example (Spark 4.2+)

```python
"""Demonstrate the spark.sql.codegen.wholeStage.union.enabled feature.

When enabled, UnionExec can participate in whole-stage codegen fusion,
allowing up to spark.sql.codegen.wholeStage.union.maxChildren (default 64)
children to be compiled into a single stage instead of each child being
a separate interpreted stage.
"""
import logging
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit

logger = logging.getLogger(__name__)


def build_union_with_codegen(spark: SparkSession, num_children: int = 10) -> DataFrame:
    """Create a Union of num_children DataFrames and verify codegen fusion.

    Parameters
    ----------
    spark : SparkSession
    num_children : int
        Number of child DataFrames to union. Must be <= 64 (maxChildren).

    Returns
    -------
    DataFrame
    """
    if num_children < 1 or num_children > 64:
        raise ValueError(f"num_children must be between 1 and 64, got {num_children}")

    # Enable union codegen
    spark.conf.set("spark.sql.codegen.wholeStage.union.enabled", "true")
    spark.conf.set("spark.sql.codegen.wholeStage.union.maxChildren", str(num_children))

    logger.info("Union codegen enabled -- maxChildren=%d", num_children)

    # Build children: each reads a range and applies transforms
    children = []
    for i in range(num_children):
        child = (
            spark.range(1_000_000)
            .toDF("id")
            .withColumn("source", lit(f"partition_{i}"))
            .withColumn("value", col("id") + lit(i * 1_000_000))
            .filter(col("id") % 2 == 0)
        )
        children.append(child)

    # Union all
    unioned = reduce(DataFrame.unionByName, children)

    # Verify the plan
    plan = unioned._jdf.queryExecution().executedPlan().toString()
    logger.info("Union physical plan:\n%s", plan)

    # Count codegen stages
    import re
    stage_ids = set(re.findall(r"\*\((\d+)\)", plan))
    logger.info("Total codegen stages in union plan: %d -- stages: %s",
                 len(stage_ids), sorted(stage_ids))

    # Check if UnionExec itself is codegen'd
    union_in_codegen = bool(re.search(r"\*\(\d+\)\s+Union", plan))
    if union_in_codegen:
        logger.info("UnionExec IS fused into a WholeStageCodegen stage")
    else:
        logger.warning(
            "UnionExec is NOT in a codegen stage -- check if maxChildren "
            "covers the number of children (%d)", num_children
        )

    return unioned


def benchmark_union_codegen_vs_interpreted(spark: SparkSession, num_children: int = 10) -> None:
    """Compare union codegen vs interpreted timing."""
    import time

    # --- Interpreted (union codegen OFF) ---
    spark.conf.set("spark.sql.codegen.wholeStage.union.enabled", "false")
    logger.info("Building union (codegen OFF)...")
    df_off = build_union_with_codegen(spark, num_children)
    spark.conf.set("spark.sql.codegen.wholeStage.union.enabled", "false")

    t0 = time.perf_counter()
    count_off = df_off.count()
    t1 = time.perf_counter()
    interpreted_ms = (t1 - t0) * 1000
    logger.info("Interpreted union: count=%d, time=%.1fms", count_off, interpreted_ms)

    # --- Codegen ON ---
    spark.conf.set("spark.sql.codegen.wholeStage.union.enabled", "true")
    logger.info("Building union (codegen ON)...")
    df_on = build_union_with_codegen(spark, num_children)

    t0 = time.perf_counter()
    count_on = df_on.count()
    t1 = time.perf_counter()
    codegen_ms = (t1 - t0) * 1000
    logger.info("Codegen union: count=%d, time=%.1fms", count_on, codegen_ms)

    speedup = interpreted_ms / codegen_ms if codegen_ms > 0 else float("inf")
    logger.info("Union codegen speedup: %.2fx", speedup)


# ---- Usage ----
if __name__ == "__main__":
    spark = create_spark_session()
    try:
        # Verify union codegen is available
        union_enabled = spark.conf.get("spark.sql.codegen.wholeStage.union.enabled")
        logger.info("spark.sql.codegen.wholeStage.union.enabled = %s", union_enabled)

        # Run the benchmark
        benchmark_union_codegen_vs_interpreted(spark, num_children=10)
    except Exception:
        logger.exception("Union codegen demo failed")
    finally:
        spark.stop()
```
