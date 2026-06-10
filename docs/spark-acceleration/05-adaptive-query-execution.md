# Adaptive Query Execution (AQE)

## What It Is

Adaptive Query Execution (AQE) is a runtime query re-optimization framework in Spark SQL that breaks the traditional "plan once, execute once" model. Instead of committing to a single physical plan determined by static statistics at compile time, AQE executes the query in stages, collects real runtime statistics after each stage materializes, and uses those statistics to re-optimize the remaining plan.

AQE is controlled by `spark.sql.adaptive.enabled` (default: `true` since Spark 3.2). It is implemented primarily in:

- **`AdaptiveSparkPlanExec`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/AdaptiveSparkPlanExec.scala`) -- the root execution node that orchestrates stage creation, materialization, and re-optimization.
- **`InsertAdaptiveSparkPlan`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/InsertAdaptiveSparkPlan.scala`) -- the planner rule that wraps eligible query plans with `AdaptiveSparkPlanExec`.
- **`AQEOptimizer`** (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/AQEOptimizer.scala`) -- the logical plan re-optimizer that runs after each stage completes.

## Why It Works: The Query Stage Model

### Stage Boundaries at Exchange Nodes

The fundamental insight behind AQE is that **shuffle boundaries are natural synchronization points**. When a shuffle completes, Spark already knows the exact size, row count, and partition distribution of its output via `MapOutputStatistics`. AQE exploits this by treating every `Exchange` node (shuffle or broadcast) as a **query stage boundary**.

From `AdaptiveSparkPlanExec` (lines 54-66):

```scala
/**
 * A root node to execute the query plan adaptively. It splits the query plan into independent
 * stages and executes them in order according to their dependencies. The query stage
 * materializes its output at the end. When one stage completes, the data statistics of the
 * materialized output will be used to optimize the remainder of the query.
 */
```

### The Stage Materialization Loop

The execution flow in `AdaptiveSparkPlanExec.withFinalPlanUpdate` follows this cycle:

1. **Preparation**: The input plan runs through `queryStagePreparationRules` (including `EnsureRequirements`, `OptimizeSkewedJoin`, `CoalesceBucketsInJoin`, etc.) to reach a stable state.

2. **Stage Creation**: `createQueryStages` traverses the plan tree bottom-up. When it encounters an `Exchange` whose children are all materialized, it wraps the exchange in a `ShuffleQueryStageExec` or `BroadcastQueryStageExec`.

3. **Materialization**: New stages are submitted for asynchronous execution via a cached thread pool (`QueryStageCreator`, max 16 threads). Broadcast stages are submitted first to avoid broadcast timeout (SPARK-33933).

4. **Event-Driven Re-Optimization**: When a stage completes, a `StageMaterializationEvent` is posted to a `LinkedBlockingQueue`. The main loop drains completed events, then:
   - Updates the logical plan by replacing executed subtrees with `LogicalQueryStage` nodes (line 357: `replaceWithQueryStagesInLogicalPlan`).
   - Calls `reOptimize()` which invokes `AQEOptimizer.execute()` on the updated logical plan (line 364).
   - Compares the cost of the new plan vs. the current plan using a `CostEvaluator` (lines 367-379).
   - If the new plan is cheaper (or equal cost but different), it replaces the current plan.

5. **Repeat** until all stages are materialized and the final result stage is created.

### Exchange Reuse

`AdaptiveExecutionContext` maintains two caches shared across the entire query including subqueries:

- **`stageCache: TrieMap[SparkPlan, ExchangeQueryStageExec]`** -- canonicalized exchange plans map to already-created stages, enabling cross-subquery shuffle reuse.
- **`subqueryCache: TrieMap[SparkPlan, BaseSubqueryExec]`** -- enables subquery result reuse.

### Cost Evaluation

From `simpleCosting.scala`, the default `SimpleCostEvaluator` counts shuffle exchanges:

```scala
case class SimpleCostEvaluator(forceOptimizeSkewedJoin: Boolean) extends CostEvaluator {
  override def evaluateCost(plan: SparkPlan): Cost = {
    val numShuffles = plan.collect { case s: ShuffleExchangeLike => s }.size
    if (forceOptimizeSkewedJoin) {
      val numSkewJoins = plan.collect {
        case j: ShuffledJoin if j.isSkewJoin => j
        case j: BroadcastHashJoinExec if j.isSkewJoin => j
      }.size
      // Negative skew joins in high bits = lower cost when more skew joins resolved
      SimpleCost(-numSkewJoins.toLong << 32 | numShuffles)
    } else {
      SimpleCost(numShuffles)
    }
  }
}
```

The cost model is simple but effective: fewer shuffles = cheaper plan. When `forceOptimizeSkewedJoin` is true, plans that resolve more skew joins are preferred even at the cost of extra shuffles.

### When AQE Does NOT Activate

From `InsertAdaptiveSparkPlan.shouldApplyAQE` (lines 110-124), AQE is skipped unless one of these holds:

- `spark.sql.adaptive.forceApply` is `true`.
- The query is a subquery (AQE already decided for the main query).
- The plan contains `Exchange` nodes.
- Any operator requires non-`UnspecifiedDistribution` from its children (implying exchanges will be added later).
- The plan contains `InMemoryTableScanExec` whose cached plan is AQE-wrapped.
- The plan contains subquery expressions.

Additionally, AQE is **explicitly disabled** for:
- `ExecutedCommandExec` and `CommandResultExec` (DDL/DML commands).
- Stateful streaming operators (`StatefulOperator`) -- see SPARK-53941.
- Stateless streaming when `spark.sql.adaptiveExecution.enabledInStatelessStreaming` is `false`.

## What Problem It Solves

Traditional Spark SQL suffers from three fundamental issues that AQE addresses:

### 1. Static Statistics Are Wrong

The Catalyst optimizer estimates cardinalities using column statistics (NDV, null count, min/max) from the metastore. These estimates become stale after data changes, are absent for intermediate results, and are impossible to compute for complex expressions. AQE replaces guesses with **actual runtime measurements**.

### 2. One-Size-Fits-All Shuffle Partitions

`spark.sql.shuffle.partitions` defaults to 200. For small queries this creates 200 tiny tasks with scheduling overhead dominating compute time. For large queries, 200 partitions may cause OOM. AQE's coalesce rule dynamically adjusts partition counts based on actual data volume.

### 3. Skewed Data Kills Parallelism

A single skewed partition can make one task run 100x longer than all others, leaving the cluster idle while waiting for the straggler. AQE detects skew from actual partition sizes and splits skewed partitions into sub-partitions that run in parallel.

## The Five AQE Features

### 1. Coalesce Partitions

**Source**: `CoalesceShufflePartitions` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/CoalesceShufflePartitions.scala`)

**Mechanism**: After a shuffle stage materializes, `CoalesceShufflePartitions` reads the `MapOutputStatistics` from each `ShuffleQueryStageExec`. It then groups contiguous partitions into larger partitions using `ShufflePartitionsUtil.coalescePartitions`.

The target size is determined by `spark.sql.adaptive.advisoryPartitionSizeInBytes` (default 64MB). However, to avoid performance regressions compared to non-AQE execution, the rule also respects a minimum partition count:

```scala
val minNumPartitions = conf.getConf(SQLConf.COALESCE_PARTITIONS_MIN_PARTITION_NUM).getOrElse {
  if (conf.getConf(SQLConf.COALESCE_PARTITIONS_PARALLELISM_FIRST)) {
    session.sparkContext.defaultParallelism  // fall back to default parallelism
  } else {
    1  // respect advisory partition size fully
  }
}
```

**Coalesce Groups**: The rule identifies independent sub-plans separated by `UnionExec`, `CartesianProductExec`, `BroadcastHashJoinExec`, or `BroadcastNestedLoopJoinExec`. Each group can be coalesced independently, and minimum parallelism is distributed proportionally by data size (lines 76-91).

The result is wrapped in `AQEShuffleReadExec` with `ShufflePartitionSpec` arrays that tell each reader which contiguous range of mapper output to read.

**Configuration**:

| Config | Default | Recommended | Description |
|--------|---------|-------------|-------------|
| `spark.sql.adaptive.enabled` | `true` | `true` | Master AQE switch |
| `spark.sql.adaptive.coalescePartitions.enabled` | `true` | `true` | Enable partition coalescing |
| `spark.sql.adaptive.advisoryPartitionSizeInBytes` | `64MB` | `64MB`--`128MB` | Target size for coalesced partitions |
| `spark.sql.adaptive.coalescePartitions.minPartitionSize` | `1MB` | `1MB` | Minimum size of a coalesced partition |
| `spark.sql.adaptive.coalescePartitions.minPartitionNum` | unset | unset | Minimum number of partitions (falls back to `defaultParallelism` if unset and `parallelismFirst` is true) |
| `spark.sql.adaptive.coalescePartitions.parallelismFirst` | `true` | `true` | Whether to preserve parallelism at the cost of smaller partition sizes |

**Production Strategy**: Set `spark.sql.shuffle.partitions` to **500--2000** (higher than the default 200). AQE will coalesce down to the appropriate number based on actual data. This gives AQE finer granularity to work with -- starting from 2000 small partitions lets coalesce produce any count from 1 to 2000, while starting from 200 limits you to at most 200.

```scala
// Production-ready AQE configuration
val spark = SparkSession.builder()
  .config("spark.sql.adaptive.enabled", "true")
  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
  .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "4MB")
  .config("spark.sql.shuffle.partitions", "1000")  // start high, let AQE coalesce down
  .getOrCreate()
```

### 2. Dynamic Join Selection (SMJ to BHJ / SHJ)

**Source**: `DynamicJoinSelection` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/DynamicJoinSelection.scala`) and `OptimizeShuffleWithLocalRead` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/OptimizeShuffleWithLocalRead.scala`)

**Mechanism**: At plan time, Spark chooses between BroadcastHashJoin (BHJ), ShuffledHashJoin (SHJ), and SortMergeJoin (SMJ) based on size estimates. At runtime, AQE knows the actual sizes and can change the join strategy.

`DynamicJoinSelection` adds **hints** to the logical plan based on actual `MapOutputStatistics`:

**Demoting BHJ when many partitions are empty** (lines 40-45):
```scala
private def hasManyEmptyPartitions(mapStats: MapOutputStatistics): Boolean = {
  val partitionCnt = mapStats.bytesByPartitionId.length
  val nonZeroCnt = mapStats.bytesByPartitionId.count(_ > 0)
  partitionCnt > 0 && nonZeroCnt > 0 &&
    (nonZeroCnt * 1.0 / partitionCnt) < conf.nonEmptyPartitionRatioForBroadcastJoin
}
```

When a join child has a high ratio of empty partitions, BHJ becomes slower than shuffle join because many broadcast tasks complete immediately (short-circuit) but the broadcast overhead was already paid. AQE adds a `NO_BROADCAST_HASH` hint.

**Promoting SHJ when all partitions are small** (lines 47-53):
```scala
private def preferShuffledHashJoin(mapStats: MapOutputStatistics): Boolean = {
  val maxShuffledHashJoinLocalMapThreshold =
    conf.getConf(SQLConf.ADAPTIVE_MAX_SHUFFLE_HASH_JOIN_LOCAL_MAP_THRESHOLD)
  mapStats.bytesByPartitionId.forall(_ <= maxShuffledHashJoinLocalMapThreshold)
}
```

When every partition fits in the local map threshold (default 16MB), AQE adds a `PREFER_SHUFFLE_HASH` hint, avoiding the sort overhead of SMJ.

**Local Shuffle Reader** (`OptimizeShuffleWithLocalRead`): When a shuffle exchange's output is consumed by a broadcast hash join's probe side, AQE wraps the shuffle read in `AQEShuffleReadExec` with per-partition specs that read only the local mapper's output. This avoids network transfer for data already colocated on the executor.

**Configuration**:

| Config | Default | Recommended |
|--------|---------|-------------|
| `spark.sql.adaptive.localShuffleReader.enabled` | `true` | `true` |
| `spark.sql.adaptive.maxShuffleHashJoinLocalMapThreshold` | `16MB` | `16MB` |
| `spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin` | `0.3` | `0.3` |
| `spark.sql.autoBroadcastJoinThreshold` | `10MB` | `10MB`--`100MB` (BHJ size threshold) |

### 3. Skew Join Optimization

**Source**: `OptimizeSkewedJoin` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/OptimizeSkewedJoin.scala`)

**Mechanism**: After shuffle stages materialize, AQE examines `bytesByPartitionId` from both join sides. A partition is considered skewed if its size exceeds:

```scala
def getSkewThreshold(medianSize: Long): Long = {
  conf.getConf(SQLConf.SKEW_JOIN_SKEWED_PARTITION_THRESHOLD).max(
    (medianSize * conf.getConf(SQLConf.SKEW_JOIN_SKEWED_PARTITION_FACTOR)).toLong)
}
```

That is: `max(SKEW_JOIN_SKEWED_PARTITION_THRESHOLD, medianSize * SKEW_JOIN_SKEWED_PARTITION_FACTOR)`.

**Default thresholds**:
- `spark.sql.adaptive.skewJoin.skewedPartitionFactor` = `10.0`
- `spark.sql.adaptive.skewJoin.skewedPartitionThreshold` = `256MB`

A partition must exceed **both** the factor-based threshold (10x median) **and** the absolute threshold (256MB) to be considered skewed.

**Split Strategy**: For each skewed partition, `ShufflePartitionsUtil.createSkewPartitionSpecs` splits the mappers into smaller ranges. The target size for each split is the average size of non-skewed partitions (or the advisory partition size, whichever is larger):

```scala
private def targetSize(sizes: Array[Long], skewThreshold: Long): Long = {
  val advisorySize = conf.getConf(SQLConf.ADVISORY_PARTITION_SIZE_IN_BYTES)
  val nonSkewSizes = sizes.filter(_ <= skewThreshold)
  if (nonSkewSizes.isEmpty) advisorySize
  else math.max(advisorySize, nonSkewSizes.sum / nonSkewSizes.length)
}
```

The matching partition on the other side of the join is **replicated** for each split, creating a cartesian product of sub-partitions. For example, from the source code comments:

```
Original 4 tasks: (L1,R1), (L2,R2), (L3,R3), (L4,R4)
If L2,L4 and R3,R4 are skewed, split each into 2:
Result 9 tasks: (L1,R1), (L2-1,R2), (L2-2,R2), (L3,R3-1), (L3,R3-2),
                (L4-1,R4-1), (L4-2,R4-1), (L4-1,R4-2), (L4-2,R4-2)
```

**Split-side restrictions** by join type (lines 85-92):
- **Left side splittable**: Inner, Cross, LeftSemi, LeftAnti, LeftOuter
- **Right side splittable**: Inner, Cross, RightOuter

Full outer joins cannot be skew-optimized because splitting either side would lose unmatched rows.

**Configuration**:

| Config | Default | Recommended |
|--------|---------|-------------|
| `spark.sql.adaptive.skewJoin.enabled` | `true` | `true` |
| `spark.sql.adaptive.skewJoin.skewedPartitionFactor` | `10.0` | `5.0`--`10.0` (lower = more aggressive) |
| `spark.sql.adaptive.skewJoin.skewedPartitionThreshold` | `256MB` | `128MB`--`256MB` |
| `spark.sql.adaptive.forceOptimizeSkewedJoin` | `false` | `false` (only set if willing to accept extra shuffles) |

### 4. Empty Relation Propagation

**Source**: `AQEPropagateEmptyRelation` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/AQEPropagateEmptyRelation.scala`)

**Mechanism**: After a stage materializes, `AQEOptimizer` runs the `Propagate Empty Relations` batch which includes `AQEPropagateEmptyRelation`. This rule checks whether any `LogicalQueryStage` produced zero rows:

```scala
override protected def isEmpty(plan: LogicalPlan): Boolean =
  super.isEmpty(plan) || (!isRootRepartition(plan) && getEstimatedRowCount(plan).contains(0))
```

When an empty relation is detected, it propagates upward through the plan tree:
- **Join with empty side**: If one side of a join is empty (and the join type allows short-circuiting), the entire join result becomes empty.
- **Filter over empty**: Eliminates to empty.
- **Aggregate over empty with grouping**: Returns empty (no groups to produce).
- **NULL-aware anti-join (NAAJ)**: When the broadcast side contains all null keys (`HashedRelationWithAllNullKeys`), the NAAJ result is empty (lines 72-80).

The key difference from the compile-time `PropagateEmptyRelationBase` is that AQE uses **actual runtime row counts** from materialized stages rather than estimated statistics:

```scala
private def getEstimatedRowCount(plan: LogicalPlan): Option[BigInt] = plan match {
  case LogicalQueryStage(_, stage: QueryStageExec) if stage.isMaterialized =>
    stage.getRuntimeStatistics.rowCount
  // ... other cases
}
```

This can short-circuit entire downstream computations that would otherwise process empty data through expensive shuffles and joins.

**Special case: NULL-aware anti-join elimination** (SPARK-56313 area): When a broadcast hash join's build side produces a `HashedRelationWithAllNullKeys` and the join is a single-column NAAJ, AQE eliminates the entire join to an empty relation, avoiding the expensive NAAJ execution path.

### 5. Custom Cost Evaluator

**Source**: `costing.scala` and `AdaptiveSparkPlanExec` (lines 100-105)

**Mechanism**: AQE uses a `CostEvaluator` to decide whether a re-optimized plan should replace the current one:

```scala
@transient private val costEvaluator =
  conf.getConf(SQLConf.ADAPTIVE_CUSTOM_COST_EVALUATOR_CLASS) match {
    case Some(className) =>
      CostEvaluator.instantiate(className, context.session.sparkContext.getConf)
    case _ => SimpleCostEvaluator(conf.getConf(SQLConf.ADAPTIVE_FORCE_OPTIMIZE_SKEWED_JOIN))
  }
```

The `SimpleCostEvaluator` (shown above) counts shuffles and skew joins. You can provide a custom implementation:

```scala
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{Cost, CostEvaluator, SimpleCost}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike

class SizeAwareCostEvaluator extends CostEvaluator {
  override def evaluateCost(plan: SparkPlan): Cost = {
    // Count shuffles
    val numShuffles = plan.collect { case _: ShuffleExchangeLike => 1 }.size

    // Also consider estimated data size from runtime stats
    var totalSize: Long = 0
    plan.foreach {
      case stage: org.apache.spark.sql.execution.adaptive.ShuffleQueryStageExec =>
        stage.mapStats.foreach { stats =>
          totalSize += stats.bytesByPartitionId.sum
        }
      case _ =>
    }

    // Cost = shuffles weighted by data size
    SimpleCost(numShuffles * 1000 + totalSize / (1024 * 1024))
  }
}
```

Configure it via:
```
spark.sql.adaptive.customCostEvaluatorClass=com.yourcompany.SizeAwareCostEvaluator
```

The `CostEvaluator.instantiate` method uses `Utils.loadExtensions` to load the class via Spark's extension mechanism.

**Configuration**:

| Config | Default | Description |
|--------|---------|-------------|
| `spark.sql.adaptive.customCostEvaluatorClass` | unset | Fully qualified class name extending `CostEvaluator` |
| `spark.sql.adaptive.optimizer.excludedRules` | unset | Comma-separated rule names to exclude from AQEOptimizer |

## How to Use AQE in Production

### Full Configuration Block

```scala
val spark = SparkSession.builder()
  .appName("ProductionAQE")
  .config("spark.sql.adaptive.enabled", "true")
  // Coalesce partitions
  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
  .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "4MB")
  .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "true")
  // Skew join optimization
  .config("spark.sql.adaptive.skewJoin.enabled", "true")
  .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "10.0")
  .config("spark.sql.adaptive.skewJoin.skewedPartitionThreshold", "256MB")
  // Dynamic join selection
  .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
  .config("spark.sql.adaptive.maxShuffleHashJoinLocalMapThreshold", "16MB")
  // Start with high shuffle partitions, let AQE coalesce down
  .config("spark.sql.shuffle.partitions", "1000")
  // Auto broadcast threshold (BHJ candidate size)
  .config("spark.sql.autoBroadcastJoinThreshold", "50MB")
  .getOrCreate()
```

### EXPLAIN Output Showing AQE

Without AQE:
```
== Physical Plan ==
*(5) SortMergeJoin [id], [id], Inner
:- *(2) Sort [id ASC], 0, false
:  +- Exchange hashpartitioning(id, 200), ENSURE_REQUIREMENTS
:     +- *(1) FileScan parquet [id, name] ...
+- *(4) Sort [id ASC], 0, false
   +- Exchange hashpartitioning(id, 200), ENSURE_REQUIREMENTS
      +- *(3) Filter (date# > 2024-01-01)
         +- FileScan parquet [id, date, value] ...
```

With AQE enabled, the initial plan shows `AdaptiveSparkPlanExec`:
```
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- == Initial Plan ==
   SortMergeJoin [id], [id], Inner
   :- Sort [id ASC], 0, false
   :  +- Exchange hashpartitioning(id, 200), ENSURE_REQUIREMENTS
   :     +- FileScan parquet [id, name] ...
   +- Sort [id ASC], 0, false
      +- Exchange hashpartitioning(id, 200), ENSURE_REQUIREMENTS
         +- Filter (date# > 2024-01-01)
            +- FileScan parquet [id, date, value] ...
```

After materialization (`isFinalPlan=true`), the plan reflects AQE transformations:
```
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=true
+- == Final Plan ==
   *(3) SortMergeJoin [id], [id], Inner
   :- *(1) Sort [id ASC], 0, false
   :  +- AQEShuffleRead coalesced 12 partitions, [id]
   :     +- ShuffleQueryStage (id=0)
   :        +- Exchange hashpartitioning(id, 1000), ENSURE_REQUIREMENTS
   :           +- FileScan parquet [id, name] ...
   +- *(2) Sort [id ASC], 0, false
      +- AQEShuffleRead coalesced 8 partitions, isSkewed, [id]
         +- ShuffleQueryStage (id=1)
            +- Exchange hashpartitioning(id, 1000), ENSURE_REQUIREMENTS
               +- Filter (date# > 2024-01-01)
                  +- FileScan parquet [id, date, value] ...

   -- Note: "coalesced 12 partitions" shows original 1000 reduced to 12
   -- Note: "isSkewed" indicates skew join optimization was applied
```

Key indicators in the EXPLAIN output:
- `AdaptiveSparkPlan isFinalPlan=true` confirms AQE completed re-optimization.
- `AQEShuffleRead coalesced N partitions` shows partition reduction.
- `isSkewed` on a shuffle read indicates skew splitting was applied.
- `ShuffleQueryStage (id=N)` marks materialized stage boundaries.
- The **Initial Plan** vs **Final Plan** sections show what changed.

### Reading AQE Changes in the Spark UI SQL Tab

1. **SQL Tab -> Query Details**: Look for "Adaptive Execution" in the execution summary. The plan display shows the final optimized plan.

2. **Plan Graph**: During execution, the plan graph updates dynamically as stages complete. Each stage boundary appears as a separate node in the DAG.

3. **Stage Details**:
   - Materialized stages show actual bytes read/written vs. estimates.
   - Compare the "Input" metric of a shuffle read stage against the "Output" of the corresponding shuffle write to see coalescing ratios.
   - Skewed stages show extreme task duration variance; after optimization, re-examine the same stage for balanced task times.

4. **Adaptive Execution Events**: The event log contains `SparkListenerSQLAdaptiveExecutionUpdate` events that record each plan change. Tools like Spark Event Log Analyzer can replay these to see the plan evolution timeline.

5. **Metrics Comparison**: For a shuffle coalesced from 1000 to 12 partitions, the stage details will show 12 tasks instead of 1000, with significantly less scheduling overhead.

## Limitations and Gotchas

### Streaming Queries

AQE is **disabled by default for stateful Structured Streaming** (`StatefulOperator` detection in `InsertAdaptiveSparkPlan` lines 67-70). For stateless streaming, it is controlled by `spark.sql.adaptiveExecution.enabledInStatelessStreaming` (default: `false`). This is because:
- Streaming queries run continuously; re-optimization overhead compounds across micro-batches.
- Stateful operators maintain per-partition state; changing partition counts mid-stream would corrupt state.

### Deterministic Pipelines

Queries with **deterministic partitioning requirements** (e.g., `repartition(col)` followed by operations that depend on that exact partitioning) may not benefit from AQE. The `CoalesceShufflePartitions` rule respects `REPARTITION_BY_COL` and `REPARTITION_BY_NUM` origins and will not coalesce them.

### AQE Does Not Trigger on Small/Simple Queries

If a query has no `Exchange` nodes (no shuffles, no broadcasts), `shouldApplyAQE` returns `false` unless `forceApply` is set. Simple filter-and-project queries over a single table do not benefit from AQE.

### Stale Statistics Interactions

AQE's `AQEOptimizer` re-optimizes the **logical plan** using runtime statistics. However, the initial plan is still generated using compile-time statistics. If stale statistics cause a fundamentally wrong plan shape (e.g., choosing the wrong join order), AQE cannot fix it -- it only optimizes within the chosen plan structure. The `DynamicJoinSelection` rule can change join strategies but not join orders.

### Skew Join Optimization May Add Extra Shuffles

When skew join optimization introduces additional shuffles (e.g., to re-partition after splitting skewed partitions), it is rejected unless `spark.sql.adaptive.forceOptimizeSkewedJoin` is `true`. This trade-off means some heavily skewed joins may not be optimized to avoid the cost of extra shuffles.

### Small Queries Overhead

AQE adds latency from stage materialization and re-optimization. For queries that complete in under a few seconds, the overhead may outweigh the benefits. Consider disabling AQE for interactive/low-latency workloads:

```scala
spark.conf.set("spark.sql.adaptive.enabled", "false")  // for this session only
```

### Memory Pressure from Stage Caching

`AdaptiveExecutionContext.stageCache` retains all materialized stages for exchange reuse. In queries with many unique shuffles (no reuse opportunities), this can increase memory pressure. Monitor executor memory usage when running complex AQE queries.

### Configuration Must Be Set Before SparkSession Creation

AQE-related configurations (especially `spark.sql.adaptive.enabled`) must be set at `SparkSession.builder()` time. Setting them via `spark.conf.set()` after session creation may not take effect because `InsertAdaptiveSparkPlan` checks `conf.adaptiveExecutionEnabled` during plan construction.

### Coalesce Can Over-Coalesce

With `parallelismFirst=true` (default), the minimum partition count falls back to `spark.default.parallelism`. On small clusters, this can result in too few partitions for large data. Monitor the coalesced partition count in EXPLAIN output and adjust `minPartitionSize` or `advisoryPartitionSizeInBytes` if needed.

### Local Shuffle Reader Requires ENSURE_REQUIREMENTS Origin

`OptimizeShuffleWithLocalRead` only applies to exchanges with `ENSURE_REQUIREMENTS` or `REBALANCE_PARTITIONS_BY_NONE` origins. Manually repartitioned data (`repartition(col)`) uses `REPARTITION_BY_COL` and will not benefit from local shuffle read optimization.

---

## Production-Ready Python (PySpark) Code

### Session Factory with AQE Configuration

```python
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

logger = logging.getLogger(__name__)


def create_aqe_session(app_name: str = "AQEProduction") -> SparkSession:
    """Create a SparkSession with all AQE features enabled and tuned.

    Enables partition coalescing, dynamic join selection, skew join
    optimization, local shuffle reader, and empty relation propagation.

    Sets ``spark.sql.shuffle.partitions`` high (1000) so AQE has fine
    granularity to coalesce down from.

    Args:
        app_name: The Spark application name.

    Returns:
        Configured SparkSession instance.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        # Master switch
        .config("spark.sql.adaptive.enabled", "true")
        # Partition coalescing
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "4MB")
        .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "true")
        # Dynamic join selection
        .config("spark.sql.adaptive.joinEnabled", "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        .config("spark.sql.adaptive.maxShuffleHashJoinLocalMapThreshold", "16MB")
        .config("spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin", "0.3")
        # Skew join optimization
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "10.0")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThreshold", "256MB")
        # Broadcast threshold for dynamic join switching
        .config("spark.sql.autoBroadcastJoinThreshold", "52428800")  # 50 MB
        # Shuffle partitions -- start high, let AQE coalesce down
        .config("spark.sql.shuffle.partitions", "1000")
        .getOrCreate()
    )
```

### AQE Verification

```python
@dataclass
class AQEVerificationResult:
    """Result of verifying AQE is active for a query."""
    aqe_active: bool
    is_final_plan: bool = False
    stages_found: int = 0
    coalesced_partitions: list[str] = field(default_factory=list)
    skew_detected: bool = False
    join_strategy_changes: list[str] = field(default_factory=list)
    explain_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify_aqe_active(spark: SparkSession, sql_query: str) -> AQEVerificationResult:
    """Parse EXPLAIN output to confirm AQE is wrapping the plan.

    Looks for ``AdaptiveSparkPlanExec`` (shown as ``AdaptiveSparkPlan``),
    stage IDs, and final-plan markers.

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to analyze.

    Returns:
        AQEVerificationResult with stage and plan details.
    """
    try:
        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
    except Exception as exc:
        logger.error("EXPLAIN failed: %s", exc)
        return AQEVerificationResult(
            aqe_active=False, notes=[f"EXPLAIN failed: {exc}"]
        )

    full_plan = "\n".join(lines)
    aqe_active = "AdaptiveSparkPlan" in full_plan
    is_final = "isFinalPlan=true" in full_plan

    # Count ShuffleQueryStage entries
    stages = re.findall(r"ShuffleQueryStage\s*\(id=(\d+)\)", full_plan)
    stages_found = len(stages)

    # Coalesced partition annotations
    coalesced = re.findall(
        r"AQEShuffleRead\s+coalesced\s+(\d+)\s+partition", full_plan
    )

    # Skew indicators
    skew_detected = "isSkewed" in full_plan

    notes: list[str] = []
    if aqe_active:
        notes.append("AQE is active (AdaptiveSparkPlan found)")
        if is_final:
            notes.append("Final plan is materialized")
    if coalesced:
        notes.append(
            f"Partition coalescing found: {coalesced} partitions"
        )
    if skew_detected:
        notes.append("Skew join optimization detected (isSkewed)")

    return AQEVerificationResult(
        aqe_active=aqe_active,
        is_final_plan=is_final,
        stages_found=stages_found,
        coalesced_partitions=coalesced,
        skew_detected=skew_detected,
        explain_lines=lines,
        notes=notes,
    )
```

### AQE Optimization Inspector

```python
@dataclass
class AQEOptimizationReport:
    """Detailed report of AQE optimizations detected in a plan."""
    partition_coalescing: list[dict] = field(default_factory=list)
    join_type_changes: list[dict] = field(default_factory=list)
    skew_join_handling: list[dict] = field(default_factory=list)
    local_shuffle_reader: bool = False
    empty_relation_propagation: bool = False
    total_stages: int = 0
    notes: list[str] = field(default_factory=list)


def inspect_aqe_optimizations(
    spark: SparkSession, sql_query: str
) -> AQEOptimizationReport:
    """Parse EXPLAIN output for specific AQE optimization markers.

    Examines the physical plan for:
    - ``AQEShuffleRead coalesced`` (partition coalescing)
    - ``BroadcastHashJoin`` vs. ``SortMergeJoin`` changes
    - ``isSkewed`` (skew join splitting)
    - ``AQEShuffleRead`` without coalesce (local shuffle reader)
    - ``LocalLimit`` / ``Empty`` patterns (empty relation propagation)

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to analyze.

    Returns:
        AQEOptimizationReport with detected optimizations.
    """
    try:
        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
    except Exception as exc:
        logger.error("EXPLAIN failed: %s", exc)
        return AQEOptimizationReport(notes=[f"EXPLAIN failed: {exc}"])

    full_plan = "\n".join(lines)
    report = AQEOptimizationReport()

    # --- Partition coalescing ---
    coalesce_matches = re.findall(
        r"AQEShuffleRead\s+coalesced\s+(\d+)\s+partition",
        full_plan,
    )
    for count in coalesce_matches:
        report.partition_coalescing.append({
            "coalesced_to": int(count),
        })

    # --- Join strategy detection ---
    bhj = re.findall(r"BroadcastHashJoin", full_plan)
    smj = re.findall(r"SortMergeJoin", full_plan)
    shj = re.findall(r"ShuffleHashJoin", full_plan)
    if bhj:
        report.join_type_changes.append({
            "strategy": "BroadcastHashJoin",
            "count": len(bhj),
        })
    if smj:
        report.join_type_changes.append({
            "strategy": "SortMergeJoin",
            "count": len(smj),
        })
    if shj:
        report.join_type_changes.append({
            "strategy": "ShuffleHashJoin",
            "count": len(shj),
        })

    # --- Skew join handling ---
    skew_matches = re.findall(
        r"AQEShuffleRead[^,]*,\s*isSkewed", full_plan
    )
    for match in skew_matches:
        report.skew_join_handling.append({"detected": True})

    # --- Local shuffle reader ---
    # AQEShuffleRead that is NOT coalesced and NOT isSkewed often
    # indicates local shuffle reader wrapping
    all_aqe_read = re.findall(r"AQEShuffleRead", full_plan)
    coalesced_count = len(coalesce_matches)
    skewed_count = len(skew_matches)
    if len(all_aqe_read) > coalesced_count + skewed_count:
        report.local_shuffle_reader = True

    # --- Empty relation propagation ---
    if re.search(r"LocalLimit\s+0|EmptyRelation", full_plan):
        report.empty_relation_propagation = True

    # --- Stage count ---
    stages = re.findall(r"ShuffleQueryStage\s*\(id=(\d+)\)", full_plan)
    report.total_stages = len(stages)

    # --- Summary notes ---
    if report.partition_coalescing:
        report.notes.append(
            f"Partition coalescing active: "
            f"{len(report.partition_coalescing)} shuffle(s) coalesced"
        )
    if report.join_type_changes:
        strategies = ", ".join(
            f"{j['strategy']}({j['count']})"
            for j in report.join_type_changes
        )
        report.notes.append(f"Join strategies: {strategies}")
    if report.skew_join_handling:
        report.notes.append(
            f"Skew join optimization: "
            f"{len(report.skew_join_handling)} partition(s) split"
        )
    if report.local_shuffle_reader:
        report.notes.append("Local shuffle reader detected")
    if report.empty_relation_propagation:
        report.notes.append("Empty relation propagation detected")

    return report
```

### AQE Benchmark

```python
@dataclass
class AQEBenchmarkQueryResult:
    """Benchmark result for a single query."""
    query_name: str
    aqe_enabled_ms: float = 0.0
    aqe_disabled_ms: float = 0.0
    speedup_ratio: float = 1.0
    aqe_partitions: int = 0
    non_aqe_partitions: int = 0
    skew_resolved_aqe: bool = False
    skew_resolved_non_aqe: bool = False
    error: str = ""


@dataclass
class AQEBenchmarkReport:
    """Aggregate AQE benchmark results."""
    queries: list[AQEBenchmarkQueryResult] = field(default_factory=list)
    avg_speedup: float = 0.0
    timestamp: str = ""


def aqe_benchmark(
    spark: SparkSession,
    queries: dict[str, str],
    output_path: str = "/tmp/aqe_benchmark.json",
) -> AQEBenchmarkReport:
    """Run queries with AQE ON and OFF, comparing performance and plans.

    For each query:
    1. Execute with AQE enabled, measure wall-clock time, inspect plan.
    2. Disable AQE, repeat.
    3. Compare partition counts, skew resolution, and execution times.

    Args:
        spark: Active SparkSession.
        queries: Dict of query name -> SQL string.
        output_path: JSON output path.

    Returns:
        AQEBenchmarkReport with per-query results.
    """
    report = AQEBenchmarkReport(timestamp=datetime.utcnow().isoformat())
    results: list[AQEBenchmarkQueryResult] = []

    original_aqe = spark.conf.get("spark.sql.adaptive.enabled", "true")

    def _run_query(sql: str, aqe_on: bool) -> tuple[float, int, bool]:
        spark.conf.set("spark.sql.adaptive.enabled", str(aqe_on).lower())
        spark.conf.set(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(aqe_on).lower(),
        )

        try:
            start = time.perf_counter()
            df = spark.sql(sql)
            df.count()
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Inspect plan for partition info
            explain_df = spark.sql(f"EXPLAIN EXTENDED {sql}")
            plan = "\n".join(r["plan"] for r in explain_df.collect())

            # Extract coalesced partition count
            coalesced = re.findall(
                r"AQEShuffleRead\s+coalesced\s+(\d+)\s+partition", plan
            )
            partitions = (
                int(coalesced[0]) if coalesced else 0
            )

            skew_resolved = "isSkewed" in plan

            return elapsed_ms, partitions, skew_resolved

        except Exception as exc:
            logger.error(
                "Query with AQE=%s failed: %s", aqe_on, exc
            )
            return -1.0, 0, False

    for name, sql in queries.items():
        bq = AQEBenchmarkQueryResult(query_name=name)

        bq.aqe_enabled_ms, bq.aqe_partitions, bq.skew_resolved_aqe = (
            _run_query(sql, aqe_on=True)
        )
        bq.aqe_disabled_ms, bq.aqe_non_aqe_partitions, bq.skew_resolved_non_aqe = (
            _run_query(sql, aqe_on=False)
        )

        if bq.aqe_disabled_ms > 0 and bq.aqe_enabled_ms > 0:
            bq.speedup_ratio = round(
                bq.aqe_disabled_ms / bq.aqe_enabled_ms, 2
            )

        results.append(bq)

    # Restore original setting
    spark.conf.set("spark.sql.adaptive.enabled", original_aqe)
    spark.conf.set(
        "spark.sql.adaptive.coalescePartitions.enabled",
        original_aqe,
    )

    report.queries = results
    valid = [
        q
        for q in results
        if not q.error and q.aqe_enabled_ms > 0 and q.aqe_disabled_ms > 0
    ]
    if valid:
        report.avg_speedup = round(
            sum(q.speedup_ratio for q in valid) / len(valid), 2
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": report.timestamp,
                "avg_speedup": report.avg_speedup,
                "queries": [asdict(q) for q in report.queries],
            },
            fh,
            indent=2,
        )
    logger.info("AQE benchmark report written to %s", output_path)
    return report
```

### Skew Join Monitor

```python
@dataclass
class SkewPartitionInfo:
    """Information about a skewed partition detected by AQE."""
    partition_id: int = -1
    size_bytes: int = 0
    median_size_bytes: int = 0
    factor_above_median: float = 0.0
    split_into: int = 0


@dataclass
class SkewJoinReport:
    """Report on skew join detection and AQE resolution."""
    query_name: str
    skew_detected: bool = False
    skewed_partitions: list[SkewPartitionInfo] = field(default_factory=list)
    aqe_split_applied: bool = False
    notes: list[str] = field(default_factory=list)


def monitor_skew_join_resolution(
    spark: SparkSession,
    sql_query: str,
    query_name: str = "unnamed",
) -> SkewJoinReport:
    """Detect skewed partitions and report AQE's split behavior.

    Parses the EXPLAIN output for ``isSkewed`` markers and extracts
    partition-level skew information from the plan text.

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to monitor.
        query_name: Human-readable name for the report.

    Returns:
        SkewJoinReport with detected skew details.
    """
    report = SkewJoinReport(query_name=query_name)

    try:
        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
    except Exception as exc:
        logger.error("EXPLAIN failed: %s", exc)
        report.notes.append(f"EXPLAIN failed: {exc}")
        return report

    full_plan = "\n".join(lines)

    # Check for isSkewed marker
    if "isSkewed" in full_plan:
        report.skew_detected = True
        report.aqe_split_applied = True
        report.notes.append("AQE skew join optimization is active (isSkewed)")

        # Extract partition info from nearby context
        skew_blocks = re.findall(
            r"AQEShuffleRead[^,]*,\s*isSkewed.*?partition[s]?\s*(\d+)?",
            full_plan,
            re.DOTALL,
        )
        for idx, match in enumerate(skew_blocks):
            split_count = int(match) if match else 0
            report.skewed_partitions.append(
                SkewPartitionInfo(
                    partition_id=idx,
                    split_into=split_count,
                )
            )
    else:
        report.notes.append("No skew detected in this query plan")

    # Look for AQEShuffleRead coalesced counts that may hint at partition size distribution
    coalesced = re.findall(
        r"AQEShuffleRead\s+coalesced\s+(\d+)\s+partition", full_plan
    )
    if coalesced and not report.skew_detected:
        report.notes.append(
            f"AQE coalesced shuffles to {coalesced} partition(s) -- "
            "no skew splitting needed"
        )

    return report
```

### AQE Analysis Pipeline

```python
@dataclass
class AQEAnalysisResult:
    """End-to-end AQE analysis result."""
    aqe_active: bool
    optimizations: dict = field(default_factory=dict)
    skew_report: SkewJoinReport | None = None
    benchmark: AQEBenchmarkReport | None = None
    summary: str = ""
    timestamp: str = ""


class AQEAnalysisPipeline:
    """End-to-end AQE analysis with JSON reporting.

    Usage:
        pipeline = AQEAnalysisPipeline(spark)
        result = pipeline.analyze(
            query=my_sql,
            query_name="star_schema_join",
            run_benchmark=True,
            benchmark_queries={"q1": sql1, "q2": sql2},
        )
        pipeline.save_report("/tmp/aqe_analysis.json")
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._last_result: AQEAnalysisResult | None = None

    def analyze(
        self,
        query: str,
        query_name: str = "unnamed",
        run_benchmark: bool = False,
        benchmark_queries: dict[str, str] | None = None,
    ) -> AQEAnalysisResult:
        """Run full AQE analysis on a query.

        Args:
            query: SQL query to analyze.
            query_name: Human-readable query identifier.
            run_benchmark: If True, also run AQE ON/OFF benchmark.
            benchmark_queries: Queries for the benchmark (required if
                run_benchmark is True).

        Returns:
            AQEAnalysisResult with all analysis sections populated.
        """
        result = AQEAnalysisResult(
            timestamp=datetime.utcnow().isoformat(),
        )

        # Step 1: Verify AQE is active
        verification = verify_aqe_active(self.spark, query)
        result.aqe_active = verification.aqe_active

        # Step 2: Inspect optimizations
        opt_report = inspect_aqe_optimizations(self.spark, query)
        result.optimizations = {
            "partition_coalescing": opt_report.partition_coalescing,
            "join_type_changes": opt_report.join_type_changes,
            "skew_join_handling": opt_report.skew_join_handling,
            "local_shuffle_reader": opt_report.local_shuffle_reader,
            "empty_relation_propagation": opt_report.empty_relation_propagation,
            "total_stages": opt_report.total_stages,
        }

        # Step 3: Monitor skew joins
        skew_report = monitor_skew_join_resolution(
            self.spark, query, query_name
        )
        result.skew_report = skew_report

        # Step 4: Optional benchmark
        if run_benchmark and benchmark_queries:
            result.benchmark = aqe_benchmark(
                self.spark, benchmark_queries
            )

        # Build summary
        summary_parts: list[str] = []
        if result.aqe_active:
            summary_parts.append("AQE is active")
        else:
            summary_parts.append("AQE is NOT active")

        if opt_report.partition_coalescing:
            summary_parts.append(
                f"coalescing {len(opt_report.partition_coalescing)} shuffle(s)"
            )
        if skew_report.skew_detected:
            summary_parts.append("skew joins detected and resolved")

        if result.benchmark:
            summary_parts.append(
                f"benchmark avg speedup: {result.benchmark.avg_speedup}x"
            )

        result.summary = "; ".join(summary_parts)
        self._last_result = result
        return result

    def save_report(
        self, path: str = "/tmp/aqe_analysis.json"
    ) -> str:
        """Save the last analysis result as JSON.

        Args:
            path: Output file path.

        Returns:
            The written file path.
        """
        if self._last_result is None:
            raise RuntimeError("No analysis result available. Call analyze() first.")

        data = asdict(self._last_result)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("AQE analysis report written to %s", path)
        return path
```

### Production ETL with AQE Monitoring

```python
@dataclass
class ETLStepResult:
    """Result of a single ETL step."""
    step_name: str
    duration_ms: float = 0.0
    aqe_active: bool = False
    stages_count: int = 0
    partitions_coalesced: int = 0
    status: str = ""


@dataclass
class ETLPipelineResult:
    """Full ETL pipeline result with AQE monitoring."""
    pipeline_name: str
    steps: list[ETLStepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    aqe_enabled: bool = False
    timestamp: str = ""


def run_production_etl_with_aqe(
    spark: SparkSession,
    pipeline_name: str,
    etl_steps: dict[str, str],
    output_path: str = "/tmp/etl_aqe_report.json",
) -> ETLPipelineResult:
    """Run a production ETL pipeline with AQE monitoring at each step.

    Executes each SQL step, measures execution time, and inspects the
    AQE plan for coalescing, skew handling, and stage counts.

    Args:
        spark: Active SparkSession with AQE enabled.
        pipeline_name: Name of the ETL pipeline.
        etl_steps: Ordered dict mapping step name to SQL string.
        output_path: JSON report output path.

    Returns:
        ETLPipelineResult with per-step AQE analysis.
    """
    result = ETLPipelineResult(
        pipeline_name=pipeline_name,
        timestamp=datetime.utcnow().isoformat(),
        aqe_enabled=spark.conf.get("spark.sql.adaptive.enabled", "false") == "true",
    )

    total_start = time.perf_counter()

    for step_name, sql in etl_steps.items():
        step = ETLStepResult(step_name=step_name)
        try:
            start = time.perf_counter()
            df = spark.sql(sql)
            df.count()
            step.duration_ms = (time.perf_counter() - start) * 1000

            # AQE verification
            verification = verify_aqe_active(spark, sql)
            step.aqe_active = verification.aqe_active
            step.stages_count = verification.stages_found
            if verification.coalesced_partitions:
                step.partitions_coalesced = sum(
                    int(p) for p in verification.coalesced_partitions
                )
            step.status = "success"

            logger.info(
                "Step '%s': %.0f ms, AQE=%s, stages=%d, coalesced=%d",
                step_name,
                step.duration_ms,
                step.aqe_active,
                step.stages_count,
                step.partitions_coalesced,
            )

        except Exception as exc:
            step.status = f"error: {exc}"
            logger.error("ETL step '%s' failed: %s", step_name, exc)

        result.steps.append(step)

    result.total_duration_ms = (time.perf_counter() - total_start) * 1000

    # Write report
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pipeline_name": result.pipeline_name,
                "timestamp": result.timestamp,
                "aqe_enabled": result.aqe_enabled,
                "total_duration_ms": round(result.total_duration_ms, 2),
                "steps": [asdict(s) for s in result.steps],
            },
            fh,
            indent=2,
        )
    logger.info("ETL AQE report written to %s", output_path)
    return result
```
