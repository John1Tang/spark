---
title: Spark SQL Implementation Guide
---

# Spark SQL Implementation Guide

This document traces the implementation of Spark SQL from user-facing APIs down to internal execution, covering key function call chains, data flow, and internal mechanisms.

## 1. Source File Organization

```
sql/
├── catalyst/src/main/scala/org/apache/spark/sql/catalyst/
│   ├── trees/
│   │   ├── TreeNode.scala              # Base tree abstraction
│   │   ├── TreePatternBits.scala       # Bit-based pattern matching
│   │   └── TreePatterns.scala          # Pattern definitions
│   ├── rules/
│   │   ├── Rule.scala                  # Abstract rule definition
│   │   ├── RuleExecutor.scala          # Batch/fixed-point execution engine
│   │   ├── RuleIdCollection.scala      # Unique rule identifiers
│   │   └── QueryExecutionMetering.scala # Rule execution metrics
│   ├── plans/
│   │   ├── QueryPlan.scala             # Base for all query plans
│   │   ├── logical/
│   │   │   ├── LogicalPlan.scala       # Logical plan base class
│   │   │   ├── basicLogicalOperators.scala # Project, Filter, Join, etc.
│   │   │   ├── hints.scala             # Join and partitioning hints
│   │   │   ├── v2Commands.scala        # V2 table commands
│   │   │   └── v2AlterTableCommands.scala
│   │   └── physical/
│   │       ├── partitioning.scala      # Partitioning abstractions
│   │       └── broadcastMode.scala     # Broadcast strategies
│   ├── expressions/
│   │   ├── Expression.scala            # Base expression class
│   │   ├── namedExpressions.scala      # NamedExpression, Attribute, Alias
│   │   ├── literals.scala              # Literal values
│   │   ├── predicates.scala            # And, Or, Not, EqualTo
│   │   ├── arithmetic.scala            # Add, Subtract, Multiply, Divide
│   │   ├── Cast.scala                  # Type casting
│   │   ├── conditionalExpressions.scala # If, CaseWhen
│   │   ├── SortOrder.scala             # Sort ordering
│   │   ├── aggregate/                  # Aggregate functions
│   │   ├── codegen/                    # Code generation utilities
│   │   │   ├── CodegenContext.scala    # Code generation context
│   │   │   └── ExprCode.scala          # Generated expression code
│   │   └── ...                         # (100+ expression files)
│   ├── analysis/
│   │   ├── Analyzer.scala              # Main analyzer with rule batches
│   │   ├── ResolveReferences.scala     # Column reference resolution
│   │   ├── ResolveRelations.scala      # Table reference resolution
│   │   ├── TypeCoercion.scala          # Type coercion rules
│   │   ├── CheckAnalysis.scala         # Plan validation
│   │   ├── CTESubstitution.scala       # CTE inlining
│   │   ├── ResolveAliases.scala        # Alias resolution
│   │   ├── ResolveSubquery.scala       # Subquery resolution
│   │   └── ...                         # (90+ analysis rule files)
│   ├── optimizer/
│   │   ├── Optimizer.scala             # Main optimizer with rule batches
│   │   ├── PushDownPredicates.scala    # Predicate pushdown
│   │   ├── ColumnPruning.scala         # Column pruning
│   │   ├── ConstantFolding.scala       # Constant evaluation
│   │   ├── NullPropagation.scala       # Null-aware optimization
│   │   └── ...                         # (40+ optimizer rule files)
│   ├── parser/
│   │   ├── ParserInterface.scala       # Parser abstraction
│   │   ├── AstBuilder.scala            # ANTLR visitor -> LogicalPlan
│   │   └── CatalystSqlParser.scala     # Parser implementation
│   ├── planning/
│   │   └── QueryPlanner.scala          # Logical -> Physical planning
│   ├── catalog/                        # Catalog management
│   ├── types/                          # DataType definitions
│   └── util/                           # Utilities
│
├── core/src/main/scala/org/apache/spark/sql/
│   ├── classic/
│   │   ├── SparkSession.scala          # Session entry point
│   │   ├── Dataset.scala               # Dataset[Row] API
│   │   ├── DataFrameNaFunctions.scala  # NaN handling
│   │   ├── DataFrameReader.scala       # Data source reading
│   │   ├── DataFrameWriter.scala       # Data source writing
│   │   ├── SQLContext.scala            # Legacy SQL context
│   │   └── UDFRegistration.scala       # UDF registry
│   ├── execution/
│   │   ├── SparkPlan.scala             # Physical plan base
│   │   ├── SparkPlanner.scala          # Physical planner
│   │   ├── SparkStrategies.scala       # Planning strategies
│   │   ├── SparkOptimizer.scala        # Core-specific optimizer rules
│   │   ├── QueryExecution.scala        # Pipeline orchestration
│   │   ├── basicPhysicalOperators.scala # ProjectExec, FilterExec, etc.
│   │   ├── WholeStageCodegenExec.scala # Whole-stage codegen
│   │   ├── SortExec.scala              # Sort execution
│   │   ├── limit.scala                 # Limit/offset execution
│   │   ├── aggregate/
│   │   │   ├── HashAggregateExec.scala # Hash aggregation
│   │   │   └── ObjectHashAggregateExec.scala # Generic aggregation
│   │   ├── joins/
│   │   │   ├── BroadcastHashJoinExec.scala
│   │   │   ├── ShuffledHashJoinExec.scala
│   │   │   └── SortMergeJoinExec.scala
│   │   ├── exchange/
│   │   │   ├── ShuffleExchangeExec.scala
│   │   │   └── BroadcastExchangeExec.scala
│   │   ├── datasources/
│   │   │   ├── DataSourceScanExec.scala # File scans
│   │   │   └── v2/                      # V2 strategy
│   │   ├── adaptive/                    # AQE
│   │   ├── command/                     # DDL/DML commands
│   │   ├── streaming/                   # Streaming execution
│   │   └── ui/                          # SQL UI
│   ├── internal/
│   │   ├── SessionState.scala          # Per-session state
│   │   ├── SharedState.scala           # Shared state
│   │   └── BaseSessionStateBuilder.scala # State construction
│   └── sources/                         # BaseRelation interfaces
```

## 2. Session Initialization Chain

```
SparkSession.builder.getOrCreate()
  -> SparkSession() [SparkSession.scala]
     -> new SparkContext(...)           // or reuse existing
     -> sharedState = new SharedState(sc)
     -> sessionState = BaseSessionStateBuilder.create(...)
        -> new SparkSqlParser()         // ANTLR parser
        -> new Analyzer(catalogManager, relationCache, sessionConf)
        -> new Optimizer(catalogManager)
        -> new SparkPlanner(session, experimentalMethods)
        -> new CacheManager(session)
        -> new StreamingQueryManager(session)
```

## 3. DataFrame API to LogicalPlan

Every DataFrame operation creates a new `LogicalPlan`:

```scala
df.select(col("a"), col("b"))
  -> Dataset.scala: select()
     -> select(columns.map(_.expr): _*)
        -> Dataset.ofRows(sparkSession,
             Project(projectList, queryExecution.analyzed))

df.filter(col("a") > 10)
  -> Dataset.scala: filter()
     -> Dataset.ofRows(sparkSession,
          Filter(GreaterThan('a, Literal(10)), queryExecution.analyzed))

df.groupBy("key").agg(sum("value"))
  -> RelationalGroupedDataset.scala: agg()
     -> Dataset.ofRows(sparkSession,
          Aggregate(groupExprs, aggExprs, queryExecution.analyzed))

df.join(other, Seq("key"))
  -> Dataset.scala: join()
     -> Dataset.ofRows(sparkSession,
          Join(left, right, JoinType.inner, Some(condition), hint=JoinHint.NONE))
```

Key mechanism: `Dataset.ofRows(sparkSession, logicalPlan)` creates a new `Dataset` wrapping a `QueryExecution` without triggering execution.

## 4. SQL Text to Execution — Full Chain

### 4.1 Parse

```
spark.sql("SELECT a, b FROM t WHERE c > 10")
  -> SparkSession.sql() [SparkSession.scala:680]
     -> sessionState.sqlParser.parsePlan(sqlText)
        -> SparkSqlParser.parsePlanWithParameters()
           -> parseInternal(sqlText) { parser =>
                astBuilder.visitCompoundOrSingleStatement(parser.compoundOrSingleStatement())
              }
           -> SparkSqlAstBuilder.visitSelectClause()
              -> SelectClause -> Project
              -> FromClause -> UnresolvedRelation
              -> WhereClause -> Filter
           -> Returns: Unresolved LogicalPlan
     -> Dataset.ofRows(sparkSession, logicalPlan)
        -> Creates Dataset[Row] with lazy QueryExecution
```

### 4.2 Analyze

```
df.collect()  // Action triggers the pipeline
  -> Dataset.withAction("collect", queryExecution)
     -> SQLExecution.withNewExecutionId(qe)
        -> qe.executedPlan  // Accessing triggers the lazy chain
           -> sparkPlan     // Triggers analyzed -> optimized -> planned
              -> analyzed   // Triggers analysis
                 -> analyzer.executeAndCheck(logical, tracker)
                    [Analyzer.scala]

Inside Analyzer.executeAndCheck():
  // The analyzer extends RuleExecutor[LogicalPlan]
  // Batches run sequentially:
  // 1. Substitution (FixedPoint): CTE, window substitution
  // 2. Hints resolution (FixedPoint)
  // 3. Function lookup (Once)
  // 4. Resolution (FixedPoint) — THE MAIN BATCH:
  //    - ResolveCatalogs: resolve database/table references
  //    - ResolveRelations: UnresolvedRelation -> LogicalRelation
  //    - ResolveReferences: UnresolvedAttribute -> AttributeReference
  //    - ResolveAliases: resolve named expressions
  //    - ResolveFunctions: resolve function names
  //    - ResolveSubquery: resolve subquery references
  //    - ResolveWindowOrder, ResolveWindowFrame
  //    - ResolveAggregateFunctions
  //    - TypeCoercion rules
  //    - ... (50+ rules total)
  // 5. Post-hoc resolution (Once)
  // 6. Cleanup (FixedPoint)

  After all batches: CheckAnalysis.validate(analyzed)
    // Validates plan correctness
```

### 4.3 Optimize

```
analyzed plan -> optimizer.executeAndTrack()
  [SparkOptimizer extends Optimizer]
  -> RuleExecutor.execute(analyzed)

Inside Optimizer.execute():
  // Batches run sequentially:
  // 1. Finish Analysis (FixedPoint(1))
  // 2. Rewrite With expression (FixedPoint)
  // 3. Eliminate Distinct (Once)
  // 4. Inline CTE (Once)
  // 5. Union (FixedPoint): combine unions, remove no-ops
  // 6. LocalRelation early (FixedPoint): convert to local, propagate empty
  // 7. Pullup Correlated Expressions (Once)
  // 8. Subquery (FixedPoint(1))
  // 9. Replace Operators (FixedPoint)
  // 10. Operator Optimization (FixedPoint) — THE BIG BATCH:
  //     Before Inferring Filters:
  //       PushProjectionThroughUnion, CollapseRepartition,
  //       CollapseProject, CollapseWindow
  //     After Inferring Filters:
  //       PushDownPredicates, LimitPushDown, ColumnPruning
  //       ConstantFolding, NullPropagation, ConstantPropagation
  //       FoldablePropagation, BooleanSimplification
  //       PruneFilters, CombineFilters
  // 11. Infer Filters (Once)
  // 12. Join Reorder (FixedPoint(1)) — cost-based
  // 13. Distinct Aggregate Rewrite (Once)
  // 14. LocalRelation (FixedPoint) — second pass
  // 15. Optimize One Row Plan (FixedPoint)
```

### 4.4 Plan

```
optimizedLogicalPlan -> createSparkPlan(planner, optimizedPlan)
  -> planner.plan(ReturnAnswer(optimizedPlan)).next()
  [QueryPlanner.plan()]

Inside QueryPlanner.plan():
  // For each strategy in order (first match wins):
  // strategies = [
  //   LogicalQueryStageStrategy,  // subquery stage
  //   PythonEvals,                // Python UDFs
  //   DataSourceV2Strategy,       // V2 scans
  //   V2CommandStrategy,          // V2 commands
  //   FileSourceStrategy,         // file scans
  //   DataSourceStrategy,         // generic scans
  //   SpecialLimits,              // take-ordered
  //   Aggregation,                // hash/object aggregation
  //   Window,                     // window functions
  //   JoinSelection,              // join type selection
  //   InMemoryScans,              // cached data
  //   BasicOperators,             // project, filter, sort
  //   EventTimeWatermarkStrategy  // watermarking
  // ]
  //
  // For the matching strategy:
  //   physicalPlan = strategy(logicalPlan)
  //   // physicalPlan contains PlanLater placeholders for children
  //   // Recursively fill placeholders:
  //   fillPlaceholders(physicalPlan)
```

### 4.5 Prepare

```
sparkPlan -> prepareForExecution(sparkPlan)
  [QueryExecution.preparations()]

Applied sequentially:
  1. InsertAdaptiveSparkPlan (if AQE enabled)
  2. CoalesceBucketsInJoin
  3. PlanDynamicPruningFilters
  4. PlanSubqueries
  5. RemoveRedundantProjects
  6. EnsureRequirements          // INSERTS EXCHANGES AND SORTS
  7. InsertSortForLimitAndOffset
  8. ReplaceHashWithSortAgg
  9. RemoveRedundantSorts
  10. DisableUnnecessaryBucketedScan
  11. ApplyColumnarRulesAndInsertTransitions
  12. CollapseCodegenStages      // INSERTS WholeStageCodegenExec
  13. ReuseExchangeAndSubquery
```

### 4.6 Execute

```
executedPlan.execute() -> RDD[InternalRow]
  -> executedPlan.doExecute()   // Implemented by each operator
     // Triggers Spark job submission
     // Returns RDD
```

## 5. TreeNode Transformation — Deep Dive

### 5.1 transformDown (Pre-order)

```scala
def transformDown(rule: PartialFunction[BaseType, BaseType]): BaseType = {
  val afterApply = rule.applyOrElse(this, identity)
  afterApply.transformChildrenDown(rule).map(_.transformDown(rule))
}

// Flow:
// 1. Apply rule to current node
// 2. If rule matched and changed node, apply to new children
// 3. Recurse into children (top-down)
```

### 5.2 transformUp (Post-order)

```scala
def transformUp(rule: PartialFunction[BaseType, BaseType]): BaseType = {
  transformChildrenUp(rule).map(_.transformUp(rule))
    .pipe(after => rule.applyOrElse(after, identity))
}

// Flow:
// 1. Recurse into children first (bottom-up)
// 2. Apply rule to current node after children are transformed
```

### 5.3 Pruning Optimization

```scala
def transformDownWithPruning(cond, ruleId)(rule):
  // 1. Check if rule is already ineffective for this subtree
  //    if (isRuleIneffective(ruleId)) return this
  // 2. Check if subtree matches the pattern
  //    if (!cond(this.treePatternBits)) skip transform, just apply rule to root
  // 3. Apply rule normally
  // 4. Mark rule as ineffective if no change was made
```

This avoids re-traversing subtrees where a rule has already been proven useless.

## 6. Join Selection — Deep Dive

File: `execution/SparkStrategies.scala`, `JoinSelection` strategy

```
Logical Join(left, right, joinType, condition, hint)
  -> JoinSelection.apply()

Decision tree:
  1. Check for hints:
     - BROADCAST(left) -> BroadcastHashJoinExec(left as build)
     - BROADCAST(right) -> BroadcastHashJoinExec(right as build)
     - MERGE -> SortMergeJoinExec
     - SHUFFLE_HASH -> ShuffledHashJoinExec
     - SHUFFLE_REPLICATE_NL -> CartesianProduct with replication

  2. If equi-join (has equality conditions):
     - If one side is small (< broadcastThreshold, default 10MB):
         -> BroadcastHashJoinExec
     - If statistics available and both small enough:
         -> BroadcastHashJoinExec
     - If AQE enabled (runtime size check):
         -> Could switch to BHJ at runtime
     - Default:
         -> SortMergeJoinExec (requires shuffle + sort on both sides)

  3. If non-equi-join:
     - If one side is small:
         -> BroadcastNestedLoopJoinExec
     - If cartesian:
         -> CartesianProductExec
     - Otherwise:
         -> BroadcastNestedLoopJoinExec

  4. Special join types:
     - LeftSemi / LeftAnti -> specialized join execution
     - ExistenceJoin -> mark existence of matching rows
```

## 7. Catalyst Expression Evaluation

### 7.1 Interpreted Evaluation

```scala
Expression.eval(input: InternalRow): Any

// Example: Add.eval()
// Add(left: Expression, right: Expression)
//   -> left.eval(input)  // returns Int
//   -> right.eval(input) // returns Int
//   -> result = leftResult + rightResult

// Example: FilterExec
//   condition.eval(row) match {
//     case true => yield row
//     case false | null => skip
//   }
```

### 7.2 Code-Generated Evaluation

```scala
Expression.genCode(ctx: CodegenContext): ExprCode

// Generates Java code that evaluates the expression
// Example: Add.genCode()
//   -> generates: "result = leftValue + rightValue;"

// CodegenContext manages:
// - Variable naming and scope
// - Null checking and propagation
// - Fresh name generation
// - Added imports and extra code
```

### 7.3 Fallback Mechanism

Some expressions cannot be code-generated and fall back to interpreted evaluation:

```scala
trait CodegenFallback { self: Expression =>
  override def genCode(ctx: CodegenContext): ExprCode = {
    // Generate code that calls eval() at runtime
    val evalObj = ctx.addReferenceObj("this", this)
    // ctx.fallbackFunction(...)
  }
}
```

## 8. Shuffle — Deep Dive

### 8.1 Exchange Insertion

```
EnsureRequirements.apply(sparkPlan)
  -> For each operator, check requiredChildDistribution:
     - BroadcastDistribution(n) -> insert BroadcastExchangeExec
     - HashClusteredDistribution(keys) -> insert ShuffleExchangeExec(HashPartitioning)
     - OrderedDistribution(ordering) -> may require shuffle + sort
     - UnspecifiedDistribution -> no exchange needed

Example (SortMergeJoin):
  Before EnsureRequirements:
    SortMergeJoin(leftKeys, rightKeys)
      ├─ Scan orders
      └─ Scan users

  After EnsureRequirements:
    SortMergeJoin(leftKeys, rightKeys)
      ├─ SortExec(leftKeys)
      │   └─ ShuffleExchangeExec(HashPartitioning(leftKeys))
      │       └─ Scan orders
      └─ SortExec(rightKeys)
          └─ ShuffleExchangeExec(HashPartitioning(rightKeys))
              └─ Scan users
```

### 8.2 Shuffle Execution

```
ShuffleExchangeExec.doExecute()
  -> prepareShuffleDependency()
     -> child.execute().mapPartitions { iter =>
          for each row:
            key = partitioning.keyExtractor(row)
            partitionId = partitioner.getPartition(key)
            emit (partitionId, row)
        }
     -> new ShuffleDependency(rdd, partitioner, serializer, ...)
  -> ShuffledRowRDD(shuffleDependency)

ShuffledRowRDD.compute(partitionId, context)
  -> shuffleManager.getReader()
     .read(partitionId)
  // Fetches map outputs from all executors
```

## 9. HashAggregate — Deep Dive

```
HashAggregateExec.doExecute()
  -> TungstenAggregationIterator

TungstenAggregationIterator:
  1. Create UnsafeFixedWidthAggregationMap (off-heap hash map)
  2. For each input row:
     a. Extract grouping keys
     b. Look up or create aggregation buffer in hash map
     c. Update buffer with input values
  3. If memory exceeded:
     a. Spill sorted data to disk via UnsafeKVExternalSorter
     b. Merge spill files on completion
  4. Return aggregated results

Two-phase aggregation (for distributed queries):
  Phase 1 (Partial): Aggregate locally within each partition
    -> ShuffleExchangeExec(HashPartitioning(groupingKeys))
  Phase 2 (Final): Aggregate across partitions
    -> Merge partial results
```

## 10. BroadcastHashJoin — Deep Dive

```
BroadcastHashJoinExec.doExecute()
  -> streamedPlan.execute().mapPartitions { streamedIter =>
       val broadcastRelation = buildPlan.executeBroadcast[HashedRelation]()
       val hashedRelation = broadcastRelation.value.asReadOnlyCopy()
       // Join each streamed row against the local hash table
       join(streamedIter, hashedRelation)
     }

BroadcastExchangeExec.executeBroadcast[T]()
  -> Creates a FutureAction that:
     1. Collects build-side data to driver
     2. Serializes and creates Broadcast[T]
     3. Broadcasts to all executors
  -> Cached: subsequent calls reuse the same broadcast
```

## 11. File Source Scan — Deep Dive

```
FileSourceScanExec.doExecute()
  -> lazy val inputRDD: RDD[InternalRow] = {
       val readFile = relation.fileFormat.buildReaderWithPartitionValues(
         dataSchema, partitionSchema, requiredSchema, filters, ...)
       createReadRDD(readFile, partitionedFiles, ...)
     }
  -> inputRDD

createReadRDD():
  -> For each file partition:
     -> For each file in partition:
        -> readFile(file, hadoopConf)  // returns Iterator[InternalRow]
        -> Apply remaining filters

createBucketedReadRDD() (for bucketed tables):
  -> Co-locates buckets with the same hash value
  -> Avoids shuffle for bucketed joins
```

## 12. QueryExecution Lifecycle

```
QueryExecution (lazy vals, triggered on first access):

1. logical: LogicalPlan           // From parser or DataFrame API
   -> Immediate: just wraps the input plan

2. analyzed: LogicalPlan          // Resolved references
   -> Triggers: Analyzer.executeAndCheck()
   -> Side effect: CheckAnalysis.validate()

3. withCachedData: LogicalPlan    // Substituted cached scans
   -> Triggers: cacheManager.useCachedData()

4. optimizedPlan: LogicalPlan     // Optimized
   -> Triggers: optimizer.executeAndTrack()

5. sparkPlan: SparkPlan           // Physical plan (before preparation)
   -> Triggers: planner.plan()

6. executedPlan: SparkPlan        // Ready for execution
   -> Triggers: prepareForExecution()
   -> Side effect: Inserts AQE wrapper if enabled

7. toRdd: RDD[InternalRow]        // Executable RDD
   -> Triggers: executedPlan.execute()

8. executedPlan.executeCollect()  // Actually runs the job
   -> Submits Spark job
   -> Returns Array[InternalRow]
```

Each stage is a **lazy val** — accessing a later stage triggers all prior stages in the chain.

## 13. DataFrame Caching

```
df.cache()
  -> cacheManager.cacheQuery(df, tableName=None)
     -> Creates InMemoryRelation(logicalPlan, storageLevel)
     -> Wraps the existing logical plan

df.collect()  // Triggers caching:
  -> cacheManager.getOrCompute(df)
     -> If cached: return cached RDD
     -> If not cached: compute and cache
        -> executedPlan.execute()
        -> Store result in memory (or disk based on storageLevel)

Subsequent queries:
  -> cacheManager.useCachedData(analyzed)
     -> Substitutes InMemoryRelation for cached scans
```

## 14. UDF Registration and Execution

```
spark.udf.register("myUdf", (x: Int) => x * 2)
  -> udfRegistration.register(name, func)
     -> FunctionRegistry.registerTemporaryFunction(name, udf)

In SQL: SELECT myUdf(age) FROM t
  -> Analyzer.ResolveFunctions resolves 'myUdf' to ScalaUDF
  -> ScalaUDF.eval(): calls the registered function

Code generation:
  -> ScalaUDF.genCode(): generates function call in Java bytecode
     -> If function is deterministic and simple: inline code
     -> Otherwise: CodegenFallback to eval()
```

## 15. Key Performance Mechanisms

| Mechanism | File | Purpose |
|-----------|------|---------|
| Whole-stage codegen | `WholeStageCodegenExec.scala` | Compile operator chains into single Java function |
| Tungsten Unsafe memory | `expressions/SpecificInternalRow.scala` | Off-heap binary row format |
| UnsafeRow | `UnsafeRow.java` | Compact binary representation |
| ColumnarBatch | `ColumnarBatch.java` | Batch row processing |
| UnsafeFixedWidthAggregationMap | `aggregate/` | Off-heap hash aggregation |
| HashedRelation | `joins/` | In-memory hash table for joins |
| Broadcast variable | `exchange/BroadcastExchangeExec.scala` | Small table distribution |
| AQE | `adaptive/` | Runtime plan re-optimization |
| Predicate pushdown | `FileSourceStrategy.scala` | Push filters to data source |
| Column pruning | `optimizer/ColumnPruning.scala` | Read only needed columns |
