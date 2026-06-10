# Spark SQL Engine Architecture

## Overview

Spark SQL is the module in Apache Spark for processing structured data using SQL and the DataFrame API. It translates high-level SQL queries into distributed physical executions on a cluster. The engine is built on **Catalyst** (the optimizer) and **Tungsten** (the execution backend).

---

## 1. The Big Picture: End-to-End Flow

```
User calls spark.sql("SELECT ...")
         │
         ▼
┌─────────────────────────────────────────────────┐
│  SparkSession.sql()                              │
│  SparkSession.scala:680                          │
│  → parses SQL text → LogicalPlan                 │
│  → wraps in Dataset[Row] via QueryExecution      │
└─────────────────────────────────────────────────┘
         │
         ▼  [LAZY — nothing executes yet]
         │
User calls action: df.collect() / df.show()
         │
         ▼
┌─────────────────────────────────────────────────┐
│  QueryExecution pipeline (lazy vals chain)       │
│  QueryExecution.scala:64                        │
│                                                  │
│  logical ──► analyzed ──► optimizedPlan          │
│     │                                            │
│     ▼                                            │
│  sparkPlan ──► executedPlan ──► toRdd            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Physical Execution on Cluster                   │
│  SparkPlan.execute() → RDD[InternalRow]          │
│  SparkPlan.scala:197                             │
│  │                                               │
│  ├── Whole-Stage Codegen (compiled Java)         │
│  ├── Shuffle (data redistribution)               │
│  └── Data Source Scans (Parquet/ORC/CSV)        │
└─────────────────────────────────────────────────┘
         │
         ▼
    Array[Row] result
```

---

## 2. Phase 1: Parsing — SQL Text to Unresolved LogicalPlan

**Key file:** [SparkSqlParser.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkSqlParser.scala)

When you call `spark.sql("SELECT * FROM users WHERE age > 18")`:

```scala
// SparkSession.scala:680
override def sql(sqlText: String): DataFrame = sql(sqlText, Map.empty[String, Any])

// SparkSession.scala:571 — delegates to parser
val qe = sessionState.sqlParser.parsePlan(sqlText)

// SparkSqlParser.scala:81
override def parsePlanWithParameters(...): LogicalPlan = {
  parseInternal(sqlText) { parser =>
    astBuilder.visitCompoundOrSingleStatement(parser.compoundOrSingleStatement())
  }
}
```

**What happens:**

1. ANTLR4 `SqlBaseParser` tokenizes SQL text using grammar rules from [SqlBaseParser.g4](sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser/SqlBaseParser.g4)
2. `SparkSqlAstBuilder` visits the ANTLR parse tree and converts it into Catalyst `LogicalPlan` nodes
3. The result is an **unresolved** logical plan — table names and column references are not yet validated

Example output (unresolved):
```
'Project [*]
+- 'Filter ('age > 18)
   +- 'UnresolvedRelation [users]
```

The `'` prefix means "unresolved" — these are `UnresolvedAttribute` and `UnresolvedRelation` nodes.

---

## 3. Phase 2: Analysis — Resolving References

**Key file:** [Analyzer.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/analysis/Analyzer.scala)

```scala
// QueryExecution.scala:139
sparkSession.sessionState.analyzer.executeAndCheck(sqlScriptExecuted, tracker)
```

The Analyzer traverses the logical plan and resolves every unresolved reference using the catalog:

| Rule | What it does |
|------|-------------|
| `ResolveRelations` | Looks up table schemas from the catalog, replaces `UnresolvedRelation` with `UnresolvedPlan` → resolved `LogicalRelation` |
| `ResolveReferences` | Resolves column names to actual `AttributeReference` objects |
| `ResolveFunctions` | Resolves function names to implementations |
| `ResolveSubquery` | Resolves subquery references |
| `TypeCoercion` | Handles implicit type conversions (e.g., int → double) |

These rules run in **batches** with strategies:
- **Once** — apply the rule once per pass
- **FixedPoint** — repeat until the plan stops changing

After analysis:
```
Project [id#0, name#1, age#2]
+- Filter (age#2 > 18)
   +- Relation [id#0, name#1, age#2] parquet default.users
```

All attributes are now resolved with unique IDs (`#0`, `#1`, `#2`).

---

## 4. Phase 3: Optimization — Logical Plan Rewrites

**Key file:** [SparkOptimizer.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkOptimizer.scala)

```scala
// QueryExecution.scala:249
sparkSession.sessionState.optimizer.executeAndTrack(withCachedData.clone(), tracker)
```

The Optimizer applies a large set of rule-based transformations to produce a more efficient logical plan. Key optimizations:

| Optimization | Before | After |
|-------------|--------|-------|
| **Constant Folding** | `1 + 2` | `3` |
| **Predicate Pushdown** | Filter above Join | Filter pushed to both sides of Join |
| **Column Pruning** | `SELECT *` then project only needed columns | Only needed columns are read from the source |
| **Null Propagation** | `SUM(null)` | Returns `null` directly |
| **Eliminate Outer Join** | Left outer join when right side is guaranteed non-null | Inner join |
| **Boolean Simplification** | `x AND TRUE` | `x` |
| **Limit PushDown** | Take 10 after full scan | Stop after 10 rows |

Example of Predicate Pushdown:
```
Before:                        After:
Project [name]                 Project [name]
  └─ Filter [age > 18]          └─ Join
      └─ Join                     ├─ Filter [age > 18]
         ├─ Scan users               └─ Scan users
         └─ Scan orders              └─ Scan orders
```

The optimizer is a `RuleExecutor` ([RuleExecutor.scala:205](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/rules/RuleExecutor.scala#L205)) that iteratively applies batches of rules until a fixed point is reached.

---

## 5. Phase 4: Planning — Logical to Physical

**Key file:** [SparkPlanner.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkPlanner.scala)

```scala
// QueryExecution.scala:651-656
def createSparkPlan(planner: SparkPlanner, plan: LogicalPlan): SparkPlan = {
  planner.plan(ReturnAnswer(plan)).next()
}
```

The planner uses a list of **strategies** to convert `LogicalPlan` nodes into `SparkPlan` (physical) nodes. Each strategy handles a subset of logical operators:

```scala
// SparkPlanner.scala:38-57 — strategy list order matters (first match wins)
Seq(
  LogicalQueryStageStrategy,    // SubqueryStageDeserializer, BroadcastNestedLoopJoin
  PythonEvals,                  // Python UDF evaluation
  DataSourceV2Strategy,         // V2 data source scans
  V2CommandStrategy,            // V2 commands (INSERT, DELETE, MERGE)
  FileSourceStrategy,           // File-based scans (Parquet, ORC, JSON)
  DataSourceStrategy,           // Generic data source scans
  SpecialLimits,                // TakeOrderedAndProject, ScriptTransform
  Aggregation,                  // HashAggregate, ObjectHashAggregate
  Window,                       // Window execution
  JoinSelection,                // SortMergeJoin, BroadcastHashJoin, ShuffledHashJoin
  InMemoryScans,                // InMemoryTableScanExec (cached data)
  BasicOperators,               // Project, Filter, Sort, Generate, etc.
  EventTimeWatermarkStrategy    // Event-time watermarking
)
```

**QueryPlanner.plan()** ([QueryPlanner.scala:59](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/planning/QueryPlanner.scala#L59)):
```scala
def plan(plan: LogicalPlan): Iterator[PhysicalPlan] = {
  // For each strategy, try to apply it. If a strategy can handle the plan,
  // it returns a physical plan with PlanLater placeholders for children.
  // Then recursively fill in those placeholders.
  val candidates = strategies.iterator.flatMap(_(plan))
  for {
    candidate <- candidates
    childPlans = candidate.collect { case PlanLater(lp) => lp }
    childSubPlans <- childPlans.map(plan).sequence
  } yield candidate.transformDown {
    case PlanLater(lp) => childSubPlans.next()
  }
}
```

**Join Selection example:**
```
Logical Join
  → JoinSelection strategy
    → If statistics say "small table < 10MB": BroadcastHashJoinExec
    → If sorted by join keys: SortMergeJoinExec
    → Otherwise: ShuffledHashJoinExec
```

---

## 6. Phase 5: Preparation — Making the Plan Executable

**Key file:** [QueryExecution.scala:595-627](sql/core/src/main/scala/org/apache/spark/sql/execution/QueryExecution.scala#L595-L627)

```scala
private[execution] def preparations(...): Seq[Rule[SparkPlan]] = {
  adaptiveExecutionRule.toSeq ++
  Seq(
    CoalesceBucketsInJoin,
    PlanDynamicPruningFilters(sparkSession),
    PlanSubqueries(sparkSession),
    RemoveRedundantProjects,
    EnsureRequirements(),          // ★ Critical: adds exchanges for partitioning/sorting
    InsertSortForLimitAndOffset,
    ReplaceHashWithSortAgg,
    RemoveRedundantSorts,
    RemoveRedundantWindowGroupLimits,
    DisableUnnecessaryBucketedScan,
    ApplyColumnarRulesAndInsertTransitions(...),
    CollapseCodegenStages(),       // ★ Wraps codegen-capable chains into WholeStageCodegenExec
  ) ++ (if (subquery) Nil else Seq(ReuseExchangeAndSubquery))
}
```

**`EnsureRequirements`** is the most important preparation rule — it:
- Checks if each operator's `requiredChildDistribution` is satisfied
- Inserts `ShuffleExchangeExec` when data needs to be repartitioned (e.g., before SortMergeJoin, both sides must be hash-partitioned by join keys)
- Checks if each operator's `requiredChildOrdering` is satisfied
- Inserts `SortExec` when data needs to be sorted

---

## 7. Phase 6: Physical Execution

### 7.1 The Execute Contract

**Key file:** [SparkPlan.scala:197](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkPlan.scala#L197)

```scala
// SparkPlan.scala:197 — the execute() entry point
final def execute(): RDD[InternalRow] = executeQuery {
  executeRDD.get  // lazily calls doExecute()
}

// SparkPlan.scala:332 — subclasses override this
protected def doExecute(): RDD[InternalRow]
```

Every physical operator implements `doExecute()`, transforming its child's `RDD[InternalRow]` into a new `RDD[InternalRow]`. The execution follows the **Volcano iterator model** (pull-based): operators pull rows from their children, transform them, and push results to their parent.

### 7.2 How Actions Trigger Execution

**Key file:** [Dataset.scala:2261](sql/core/src/main/scala/org/apache/spark/sql/classic/Dataset.scala#L2261)

```scala
// Dataset.scala:1504
def collect(): Array[T] = withAction("collect", queryExecution)(collectFromPlan)

// Dataset.scala:2261
protected def withAction[T](name: String, qe: QueryExecution)(body: SparkPlan => T) = {
  SQLExecution.withNewExecutionId(qe) {
    // Accessing executedPlan triggers the entire lazy pipeline
    body(qe.executedPlan)
  }
}

// SparkPlan.scala:458
def executeCollect(): Array[InternalRow] = {
  executeQuery {
    getByteArrayRdd().collect().flatMap(decodeUnsafeRows)
  }
}
```

### 7.3 Key Physical Operators

#### FileSourceScanExec — Reading Data from Storage

**File:** [DataSourceScanExec.scala:743](sql/core/src/main/scala/org/apache/spark/sql/execution/DataSourceScanExec.scala#L743)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  val outputRow = if (needsUnsafeRowConversion) {
    val toUnsafe = UnsafeProjection.create(schema)
    inputRDD.mapPartitionsWithIndexInternal { (index, iter) =>
      val proj = toUnsafe.clone()
      iter.map(row => proj(row))
    }
  } else {
    inputRDD
  }
  ...
}

// inputRDD (line 717) — builds the FileScanRDD
lazy val inputRDD: RDD[InternalRow] = {
  readFile = relation.fileFormat.buildReaderWithPartitionValues(...)
  if (bucketedScan) createBucketedReadRDD(...) else createReadRDD(...)
}

// createReadRDD (line 849)
private def createReadRDD(...): RDD[InternalRow] = {
  val filePartitions = ... // split files into partitions
  FileScanRDD(sc, readCallback, filePartitions, ...)
}
```

**FileScanRDD.compute()** ([FileScanRDD.scala:93](sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/FileScanRDD.scala#L93)):
```scala
override def compute(split: Partition, context: TaskContext): Iterator[InternalRow] = {
  // For each file in the partition:
  val iterator = readFunction(currentFile)  // reads rows from the file
  // Returns an Iterator[InternalRow] that pulls from the file reader
}
```

#### ProjectExec — Column Transformations

**File:** [basicPhysicalOperators.scala:93](sql/core/src/main/scala/org/apache/spark/sql/execution/basicPhysicalOperators.scala#L93)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  val projectEvalFactory = ProjectEvaluatorFactory(projectList, child.output)
  child.execute().mapPartitionsWithIndexInternal { (index, iter) =>
    projectEvalFactory.eval(index, iter)
  }
}
```

#### FilterExec — Predicate Filtering

**File:** [basicPhysicalOperators.scala:274](sql/core/src/main/scala/org/apache/spark/sql/execution/basicPhysicalOperators.scala#L274)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  val filterEvalFactory = FilterEvaluatorFactory(condition, child.output)
  child.execute().mapPartitionsWithIndexInternal { (index, iter) =>
    filterEvalFactory.eval(index, iter)
  }
}
```

#### HashAggregateExec — Hash-Based Aggregation

**File:** [HashAggregateExec.scala:91](sql/core/src/main/scala/org/apache/spark/sql/execution/aggregate/HashAggregateExec.scala#L91)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  child.execute().mapPartitionsWithIndex { (index, iter) =>
    new TungstenAggregationIterator(
      context = context,
      groupingExpressions = groupingExpressions,
      aggregateExpressions = aggregateExpressions,
      initialInputBufferOffset = 0,
      inputIter = iter,
      unsafeRowAgg = true)
  }
}
```

`TungstenAggregationIterator` uses an `UnsafeFixedWidthAggregationMap` (off-heap hash map) for aggregation. When memory is exceeded, it spills to disk via `UnsafeKVExternalSorter`.

#### BroadcastHashJoinExec — Broadcast Join

**File:** [BroadcastHashJoinExec.scala:123](sql/core/src/main/scala/org/apache/spark/sql/execution/joins/BroadcastHashJoinExec.scala#L123)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  // 1. Broadcast the build side
  val buildFuture = buildPlan.executeBroadcast[HashedRelation]()
  // 2. Stream the other side, joining against the broadcast hash table
  streamedPlan.execute().mapPartitionsInternal { streamedIter =>
    val hashed = buildFuture.value.asReadOnlyCopy()
    join(streamedIter, hashedRelation, numOutputRows)
  }
}
```

The build side is serialized and broadcast to all executors. Each executor builds a local `HashedRelation` (in-memory hash table) and joins streamed rows against it — **no shuffle required**.

#### SortMergeJoinExec — Sort-Merge Join

**File:** [SortMergeJoinExec.scala:124](sql/core/src/main/scala/org/apache/spark/sql/execution/joins/SortMergeJoinExec.scala#L124)

```scala
override protected def doExecute(): RDD[InternalRow] = {
  left.execute().zipPartitions(right.execute()) { (leftIter, rightIter) =>
    join(leftIter, rightIter)  // merge-sort style join
  }
}

// requiredChildOrdering (line 94) — both sides must be sorted
override def requiredChildOrdering: Seq[Seq[SortOrder]] =
  Seq(leftKeys.map(SortOrder(_, Ascending)), rightKeys.map(SortOrder(_, Ascending)))
```

Both sides must be **hash-partitioned** and **sorted** by join keys, which is why `EnsureRequirements` inserts `ShuffleExchangeExec` + `SortExec` before this operator.

### 7.4 Shuffle & Exchange

**Key file:** [ShuffleExchangeExec.scala:257](sql/core/src/main/scala/org/apache/spark/sql/execution/exchange/ShuffleExchangeExec.scala#L257)

```scala
override def doExecute(): RDD[InternalRow] = {
  executeQuery {
    shuffleDependency = prepareShuffleDependency()
    ShuffledRowRDD(shuffleDependency, readMetrics)
  }
}

// prepareShuffleDependency (line 334)
private def prepareShuffleDependency(): ShuffleDependency[Int, InternalRow, InternalRow] = {
  // 1. Choose partitioner based on output partitioning
  val part: Partitioner = outputPartitioning match {
    case HashPartitioning(keys, numParts) => PartitionIdPassthrough(numParts)
    case RangePartitioning(sortingExprs, numParts) => new RangePartitioner(...)
    case SinglePartition => ConstantPartitioner
    case KeyGroupedPartitioning(_, _) => new KeyGroupedPartitioner(...)
  }
  // 2. Map each row to (partitionId, row)
  val rdd = child.execute().mapPartitionsInternal { iter =>
    iter.map { row => (partitioner.getPartition(keyExtractor(row)), row) }
  }
  // 3. Create shuffle dependency
  new ShuffleDependency(rdd, part, serializer, keyOrdering, aggregator, mapSideCombine)
}
```

Shuffle has two phases:
1. **Map side** — each executor writes rows to local files, partitioned by target partition
2. **Reduce side** — `ShuffledRowRDD` reads data via `ShuffleManager.getReader()`, fetching map outputs from remote executors

### 7.5 Whole-Stage Code Generation

**Key file:** [WholeStageCodegenExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/WholeStageCodegenExec.scala)

Spark compiles chains of operators into a **single Java function** at runtime, eliminating virtual function calls and enabling CPU-level optimizations.

#### The Produce/Consume Model

Operators implement `CodegenSupport` with two methods:
- **`produce()`** — generates code that iterates input and calls `consume()`
- **`consume()`** — generates code to pass processed rows to the parent

```scala
// CodegenSupport trait (line 47)
trait CodegenSupport extends SparkPlan {
  def produce(ctx: CodegenContext, parent: CodegenSupport): String
  def consume(ctx: CodegenContext, input: Seq[ExprCode]): String
}
```

#### Generated Code Structure

```java
// WholeStageCodegenExec.scala:680
final class GeneratedIteratorForCodegenStageN extends BufferedRowIterator {
    public void init(int index, scala.collection.Iterator[] inputs) { ... }
    protected void processNext() {
        // All operator logic fused into a single tight loop
        while (input.hasNext()) {
            InternalRow row = input.next();
            // Filter: if (!condition) continue;
            // Project: output.set(0, expr1); output.set(1, expr2);
            // Aggregation: hashMap.update(row);
            output.append(outputRow);
        }
    }
}
```

#### Compilation at Runtime

```scala
// CodeGenerator.scala:1488
val evaluator = new ClassBodyEvaluator()
val clazz = evaluator.compile(source)  // Janino compiles Java source
```

Uses **Janino** to compile generated Java source into bytecode at runtime. The compiled class is cached by `(classLoader, source)` to avoid recompilation.

#### CollapseCodegenStages

During preparation, this rule wraps chains of codegen-capable operators:

```
Before:
  Project → Filter → HashAggregate → Sort

After:
  WholeStageCodegenExec(
    Project → Filter → HashAggregate  // fused into one generated class
  )
  → SortExec  // codegen boundary (sort breaks the pipeline)
```

---

## 8. Data Flow at Runtime

```
Driver: executedPlan.executeCollect()
         │
         ▼
    ShuffleExchangeExec.doExecute()
         │ → creates ShuffledRowRDD(shuffleDependency)
         │
         ▼
    SparkContext submits tasks to executors
         │
         ▼
┌─────────────────────────────────────────────┐
│ Executor 1 (Partition 0)                    │
│                                              │
│ WholeStageCodegenStage1.processNext():       │
│   while (input.hasNext()) {                  │
│     row = fileScan.next()    ← reads Parquet │
│     if (!filter(row)) continue               │
│     project(row)                             │
│     output.append(resultRow)                 │
│   }                                          │
│                                              │
│ → writes to shuffle map output files         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Executor 2 (Partition 0)                     │
│                                              │
│ ShuffleExchange reads map outputs:           │
│   fetches from Executor 1, 3, 5...           │
│                                              │
│ WholeStageCodegenStage2.processNext():       │
│   while (input.hasNext()) {                  │
│     row = shuffleReader.next()               │
│     hashAggregate.update(row)                │
│     output.append(resultRow)                 │
│   }                                          │
│                                              │
│ → final RDD[InternalRow]                     │
└─────────────────────────────────────────────┘
         │
         ▼
    Collect to driver → Array[Row]
```

---

## 9. Columnar / Vectorized Execution

For file formats that support batch reading (Parquet, ORC), Spark can use columnar execution:

**ColumnarBatch** ([ColumnarBatch.java](sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ColumnarBatch.java)):
```java
// Wraps ColumnVector[] as a table
public class ColumnarBatch implements AutoCloseable {
    private ColumnVector[] columns;
    private int numRows;

    public ColumnarBatchRow getRow(int rowId) {
        // Returns a lightweight view — no data copy
        row.pointTo(columns, rowId);
        return row;
    }
}
```

**FileSourceScanExec** switches to columnar mode when:
- Whole-stage codegen is enabled
- Schema has manageable number of fields (< 1000)
- File format supports batch reading (`fileFormat.supportBatch()`)

In columnar mode, `doExecuteColumnar()` returns `RDD[ColumnarBatch]` instead of `RDD[InternalRow]`, processing rows in batches for better CPU cache utilization.

---

## 10. Adaptive Query Execution (AQE)

When AQE is enabled (`spark.sql.adaptive.enabled=true`), the `InsertAdaptiveSparkPlan` preparation rule wraps the physical plan in `AdaptiveSparkPlanExec`:

```scala
// AQE runs the query first, then re-optimizes based on runtime stats:
// 1. Coalesce partitions (merge small shuffle partitions)
// 2. Optimize join strategy (switch SHJ → BHJ if one side turns out small)
// 3. Optimize skew join (split skewed partitions)
```

AQE collects runtime statistics during the first execution pass and re-optimizes the plan dynamically.

---

## 11. Key Files Reference

| Component | File |
|-----------|------|
| Entry point | [SparkSession.scala](sql/core/src/main/scala/org/apache/spark/sql/classic/SparkSession.scala) |
| Pipeline orchestration | [QueryExecution.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/QueryExecution.scala) |
| SQL Parser | [SparkSqlParser.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkSqlParser.scala) |
| ANTLR Grammar | [SqlBaseParser.g4](sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser/SqlBaseParser.g4) |
| Analyzer | [Analyzer.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/analysis/Analyzer.scala) |
| Optimizer | [Optimizer.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/Optimizer.scala) |
| Spark Optimizer | [SparkOptimizer.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkOptimizer.scala) |
| Planner | [SparkPlanner.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkPlanner.scala) |
| QueryPlanner | [QueryPlanner.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/planning/QueryPlanner.scala) |
| SparkPlan (base) | [SparkPlan.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/SparkPlan.scala) |
| File scan | [DataSourceScanExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/DataSourceScanExec.scala) |
| Basic operators | [basicPhysicalOperators.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/basicPhysicalOperators.scala) |
| Hash aggregate | [HashAggregateExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/aggregate/HashAggregateExec.scala) |
| Broadcast join | [BroadcastHashJoinExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/joins/BroadcastHashJoinExec.scala) |
| Sort-merge join | [SortMergeJoinExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/joins/SortMergeJoinExec.scala) |
| Shuffled hash join | [ShuffledHashJoinExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/joins/ShuffledHashJoinExec.scala) |
| Shuffle exchange | [ShuffleExchangeExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/exchange/ShuffleExchangeExec.scala) |
| Whole-stage codegen | [WholeStageCodegenExec.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/WholeStageCodegenExec.scala) |
| Rule executor | [RuleExecutor.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/rules/RuleExecutor.scala) |
| FileScanRDD | [FileScanRDD.scala](sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/FileScanRDD.scala) |
| Internal row | [InternalRow.scala](sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/InternalRow.scala) |
| UnsafeRow | [UnsafeRow.java](sql/catalyst/src/main/java/org/apache/spark/sql/catalyst/expressions/UnsafeRow.java) |
| Columnar batch | [ColumnarBatch.java](sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ColumnarBatch.java) |
| Session state wiring | [BaseSessionStateBuilder.scala](sql/core/src/main/scala/org/apache/spark/sql/internal/BaseSessionStateBuilder.scala) |

---

## 12. Summary: The 6-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: PARSE     SQL text → UnresolvedLogicalPlan                 │
│                            ANTLR grammar + AstBuilder               │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 2: ANALYZE   UnresolvedLogicalPlan → AnalyzedLogicalPlan      │
│                            Resolve tables, columns, types, functions│
├─────────────────────────────────────────────────────────────────────┤
│ Stage 3: OPTIMIZE  AnalyzedLogicalPlan → OptimizedLogicalPlan       │
│                            Push predicates, prune columns,          │
│                            fold constants, eliminate joins          │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 4: PLAN      OptimizedLogicalPlan → SparkPlan (physical)      │
│                            Choose join strategies, scan types,      │
│                            aggregation methods                      │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 5: PREPARE   SparkPlan → ExecutableSparkPlan                  │
│                            Add exchanges, sort, codegen collapse    │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 6: EXECUTE   RDD[InternalRow] → Result                        │
│                            Volcano iterator model, whole-stage      │
│                            codegen, Tungsten Unsafe memory,         │
│                            shuffle for data redistribution          │
└───────────────────────────────────────────────────/n──────────────────┘
```

All stages 1-5 are **lazy** — they are only triggered when an action (collect, show, write, etc.) accesses `QueryExecution.executedPlan`.
