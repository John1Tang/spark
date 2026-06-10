# Catalyst Optimizer: Rule-Based Query Optimization

## What It Is

The Catalyst Optimizer is Apache Spark's extensible, rule-based query optimizer for both logical and physical query plans. It is the core of Spark SQL's ability to automatically transform user-written queries into efficient execution plans. Catalyst operates on a tree representation of the query (a `LogicalPlan`), applying a sequence of optimization rules that rewrite the tree into a more efficient form.

**Source references (Spark 4.2):**
- `Optimizer` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/Optimizer.scala`
- `RuleExecutor` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/rules/RuleExecutor.scala`
- `TreeNode` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/trees/TreeNode.scala`

## Why It Works: The Mechanism

### The Batch/Fixed-Point Execution Model

Catalyst runs optimization rules inside `RuleExecutor[LogicalPlan]`. Rules are grouped into **batches**, and batches are executed serially. Each batch has a **strategy** that controls how many times it runs:

```scala
// RuleExecutor.scala (lines 149-159)
case object Once extends Strategy { val maxIterations = 1 }

case class FixedPoint(
  override val maxIterations: Int,
  override val errorOnTransit: Boolean = false,
  override val maxIterationsSetting: String = null) extends Strategy
```

- **`Once`**: Runs the batch exactly one time. Used for idempotent rules (e.g., `InlineCTE`, `EliminateDistinct`).
- **`FixedPoint(n)`**: Runs up to `n` iterations, stopping early if the plan converges (no rule in the batch makes a change). Used for rules that may enable each other.

The execution loop in `RuleExecutor.execute` (lines 215-284):

```scala
def execute(plan: TreeType): TreeType = {
  var curPlan = plan
  batches.foreach { batch =>
    var iteration = 1
    var continue = true
    while (continue) {
      curPlan = batch.rules.foldLeft(curPlan) {
        case (plan, rule) =>
          val result = rule(plan)
          val effective = !result.fastEquals(plan)
          // ... logging, metrics, validation ...
          result
      }
      iteration += 1
      continue = iteration <= batch.strategy.maxIterations && planChanged
    }
  }
  curPlan
}
```

**Key insight:** Rules within a batch are ordered. A rule that runs earlier can create patterns that later rules match. The fixed-point loop allows rules to trigger each other across iterations. For example, `ColumnPruning` might remove unused columns, which then enables `PushDownPredicates` to push a filter deeper on the next iteration.

### TreeNode Transformation: transformDown vs transformUp

Every optimizer rule is a `Rule[LogicalPlan]` that transforms the plan tree. Two fundamental traversal patterns:

**`transformDown` (pre-order):** Apply the rule to the current node first, then recurse into children.

```scala
// TreeNode.scala (lines 488-512)
def transformDownWithPruning(cond: TreePatternBits => Boolean, ruleId: RuleId)
  (rule: PartialFunction[BaseType, BaseType]): BaseType = {
  if (!cond.apply(this) || isRuleIneffective(ruleId)) return this
  val afterRule = rule.applyOrElse(this, identity[BaseType])
  if (this fastEquals afterRule) {
    mapChildren(_.transformDownWithPruning(cond, ruleId)(rule))
  } else {
    afterRule.copyTagsFrom(this)
    afterRule.mapChildren(_.transformDownWithPruning(cond, ruleId)(rule))
  }
}
```

Used when a rule needs to act on a parent before its children (e.g., `PushDownPredicates` pushes filters down through the tree from root to leaves).

**`transformUp` (post-order):** Recurse into children first, then apply the rule to the current node.

```scala
// TreeNode.scala (lines 540-557)
def transformUpWithPruning(cond: TreePatternBits => Boolean, ruleId: RuleId)
  (rule: PartialFunction[BaseType, BaseType]): BaseType = {
  if (!cond.apply(this) || isRuleIneffective(ruleId)) return this
  val afterRuleOnChildren = mapChildren(_.transformUpWithPruning(cond, ruleId)(rule))
  rule.applyOrElse(afterRuleOnChildren, identity[BaseType])
}
```

Used when a rule needs information from already-optimized children (e.g., `ColumnPruning` needs to know what columns the child actually produces).

**Pruning:** Both methods accept a `cond` predicate based on `TreePattern` bits. If the plan subtree does not contain the relevant pattern, the traversal is skipped entirely. This is a critical performance optimization -- e.g., `BooleanSimplification` only visits nodes containing `AND`, `OR`, or `NOT`:

```scala
// expressions.scala (line 393-394)
def apply(plan: LogicalPlan): LogicalPlan = plan.transformWithPruning(
  _.containsAnyPattern(AND, OR, NOT), ruleId) { ... }
```

### The Optimizer Batch Order

The `Optimizer.defaultBatches` (lines 100-287 in Optimizer.scala) defines the complete sequence. Key batches in order:

1. **Finish Analysis** (FixedPoint(1)) -- `FinishAnalysis`
2. **Eliminate Distinct** (Once) -- `EliminateDistinct`
3. **Inline CTE** (Once) -- `InlineCTE`
4. **Union** (fixedPoint) -- `CombineUnions`
5. **LocalRelation early** (fixedPoint) -- `ConvertToLocalRelation`, `PropagateEmptyRelation`
6. **Subquery** (FixedPoint(1)) -- `OptimizeSubqueries`
7. **Replace Operators** (fixedPoint) -- `ReplaceIntersectWithSemiJoin`, `ReplaceExceptWithFilter`
8. **Operator Optimization before Inferring Filters** (fixedPoint) -- the main rule set (~50 rules)
9. **Infer Filters** (Once) -- `InferFiltersFromGenerate`, `InferFiltersFromConstraints`
10. **Operator Optimization after Inferring Filters** (fixedPoint) -- the same rule set again
11. **Push extra predicate through join** (fixedPoint) -- `PushExtraPredicateThroughJoin`, `PushDownPredicates`
12. **Early Filter and Projection Push-Down** (Once) -- scan-level pushdowns
13. **Join Reorder** (FixedPoint(1)) -- `CostBasedJoinReorder`
14. **LocalRelation** (fixedPoint) -- second pass for empty relation elimination

`SparkOptimizer` (in `sql/core/src/main/scala/org/apache/spark/sql/execution/SparkOptimizer.scala`) extends `Optimizer` with Spark-specific batches like `PartitionPruning`, `Extract Python UDFs`, and user-provided optimizers.

---

## What Problem It Solves

Without an optimizer, a SQL query would execute exactly as written:

- Scanning all columns even when only a few are needed
- Evaluating constant expressions on every row
- Filtering after joins instead of before
- Running redundant aggregations
- Keeping unnecessary sorts and limits

Catalyst eliminates these inefficiencies automatically, so users can write declarative SQL without manually tuning the execution plan. The result is typically **10x-100x performance improvement** for complex analytical queries.

---

## All Major Optimizer Rules

### 1. Predicate Pushdown (PushDownPredicates)

**Source:** `Optimizer.scala` line 2066 -- `PushDownPredicates`

**What it does:** Moves `Filter` operators as close to the data source as possible. The filter is pushed through projections, joins, unions, aggregates, windows, and other permissible operators.

**Before optimization:**
```
Filter(state = "CA")
  +- Project(name, state, salary)
     +- Join (employees DEPARTMENTS)
        +- Scan employees  // 10M rows
        +- Scan departments
```

**After optimization:**
```
Project(name, state, salary)
  +- Join (Filter(state = "CA") DEPARTMENTS)
     +- Scan employees  // Filter applied before join
     +- Scan departments
```

**Implementation details:**
```scala
// Optimizer.scala (lines 2066-2073)
object PushDownPredicates extends Rule[LogicalPlan] {
  def apply(plan: LogicalPlan): LogicalPlan = plan.transformWithPruning(
    _.containsAnyPattern(FILTER, JOIN)) {
    CombineFilters.applyLocally
      .orElse(PushPredicateThroughNonJoin.applyLocally)
      .orElse(PushPredicateThroughJoin.applyLocally)
  }
}
```

`PushPredicateThroughNonJoin` handles pushing filters through specific operators with nuanced logic:

- **Through Project:** Only pushes deterministic predicates. If the user has `spark.sql.avoidDoubleFilterEval` enabled (the default), it tracks which projected aliases are "expensive" vs "cheap" and only pushes cheap ones (lines 2109-2164).
- **Through Aggregate:** Only pushes filters that reference grouping keys or constants, since the aggregate cannot reduce distinct grouping key values (lines 2170-2192).
- **Through Window:** Only pushes predicates on the window partitioning key (lines 2199-2218).
- **Through Union:** Splits filters by `&&`, pushes deterministic ones to each union child (lines 2220-2249).
- **Through EventTimeWatermark:** Pushes filters that don't reference the event time column (lines 2251-2264).

**Performance impact:** The single most impactful optimization. By filtering before joins and aggregations, it can reduce intermediate data by orders of magnitude.

**Gotcha:** Non-deterministic predicates (e.g., `rand()`, `current_timestamp`) are never pushed. Filters referencing UDF results in a projection may also stay above the projection if the UDF is expensive.

---

### 2. Column Pruning

**Source:** `Optimizer.scala` line 1072 -- `ColumnPruning`

**What it does:** Eliminates columns that are not needed by any ancestor operator. This reduces memory, network I/O, and compute throughout the pipeline.

**Before:**
```
Project(name)
  +- Project(id, name, email, phone, address, ...)
     +- Scan employees  // 50 columns
```

**After:**
```
Project(name)
  +- Project(name)
     +- Scan employees  // Only name column read
```

**Implementation:**
```scala
// Optimizer.scala (lines 1072-1191)
object ColumnPruning extends Rule[LogicalPlan] {
  def apply(plan: LogicalPlan): LogicalPlan = removeProjectBeforeFilter(
    plan.transformWithPruning(AlwaysProcess.fn, ruleId) {
    // Prune Project over Project
    case p @ Project(_, p2: Project) if !p2.outputSet.subsetOf(p.references) =>
      p.copy(child = p2.copy(projectList = p2.projectList.filter(p.references.contains)))
    // Prune Project over Aggregate
    case p @ Project(_, a: Aggregate) if !a.outputSet.subsetOf(p.references) =>
      p.copy(child = a.copy(aggregateExpressions = a.aggregateExpressions.filter(p.references.contains)))
    // Prune LeftSemi/LeftAnti join right side (no right columns needed)
    case j @ Join(_, right, LeftExistence(_), _, _) =>
      j.copy(right = prunedChild(right, j.references))
    // ... many more cases
  })
}
```

The rule also inserts a `Project` when a child produces more columns than needed:

```scala
private def prunedChild(c: LogicalPlan, allReferences: AttributeSet) =
  if (!c.outputSet.subsetOf(allReferences)) {
    Project(c.output.filter(allReferences.contains), c)
  } else {
    c
  }
```

**Performance impact:** For wide tables (100+ columns), column pruning can reduce scan I/O by 10-100x. With Parquet/ORC, only the required column chunks are read from disk.

**Gotcha:** Column pruning interacts with `PushDownPredicates`. The optimizer runs `removeProjectBeforeFilter` (a `transformUp` pass) after the main column pruning to eliminate unnecessary `Project` nodes that block predicate pushdown.

---

### 3. Constant Folding

**Source:** `expressions.scala` line 50 -- `ConstantFolding`

**What it does:** Evaluates foldable (constant) expressions at compile time rather than per row.

**Before:**
```
Filter(year > 2020 + 3 * 5)
  +- Scan sales
// Evaluates "2020 + 3 * 5" on every row
```

**After:**
```
Filter(year > 2035)
  +- Scan sales
// Constant computed once at compile time
```

**Implementation:**
```scala
// expressions.scala (lines 50-119)
object ConstantFolding extends Rule[LogicalPlan] {
  private[sql] def constantFolding(e: Expression, isConditionalBranch: Boolean = false): Expression = e match {
    case c: ConditionalExpression if !c.foldable =>
      c.mapChildren(constantFolding(_, isConditionalBranch = true))
    case l: Literal => l  // Skip redundant folding
    case e if e.containsTag(FAILED_TO_EVALUATE) => e
    case e if e.foldable => tryFold(e, isConditionalBranch)
    // Replace ScalarSubquery with null if its maxRows is 0
    case s: ScalarSubquery if s.plan.maxRows.contains(0) => Literal(null, s.dataType)
    case other =>
      val newOther = other.mapChildren(constantFolding(_, isConditionalBranch))
      if (newOther.foldable) tryFold(newOther, isConditionalBranch) else newOther
  }
}
```

**Performance impact:** Eliminates per-row computation for any deterministic expression. Particularly valuable inside loops (e.g., UDFs, vectorized operations).

**Gotcha:** Expressions inside conditional branches (e.g., `CASE WHEN`) are folded with error suppression -- if a branch would throw at compile time, it is left unfolded since it might not be reached at runtime.

---

### 4. Null Propagation

**Source:** `expressions.scala` line 900 -- `NullPropagation`

**What it does:** Simplifies expressions involving null values based on nullability information.

**Transformations:**

| Before | After |
|--------|-------|
| `IS NULL` on non-nullable column | `false` |
| `IS NOT NULL` on non-nullable column | `true` |
| `<=>` with null literal | `IS NULL(other)` |
| `COALESCE(a, NULL, b)` | `COALESCE(a, b)` |
| `COALESCE(a, b)` where `a` is NOT NULL | `a` |
| `COUNT(null, null, null)` | `CAST(0 AS BIGINT)` |
| `IN(value, [])` | `false` |
| `func(null, ...)` where func is NullIntolerant | `null` |

**Implementation:**
```scala
// expressions.scala (lines 900-968)
object NullPropagation extends Rule[LogicalPlan] {
  def apply(plan: LogicalPlan): LogicalPlan = plan.transformWithPruning(
    t => t.containsAnyPattern(NULL_CHECK, NULL_LITERAL, COUNT, COALESCE)
      || t.containsAllPatterns(WINDOW_EXPRESSION, CAST, LITERAL), ruleId) {
    case q: LogicalPlan => q.transformExpressionsUpWithPruning(...) {
      case IsNull(c) if !c.nullable => Literal.create(false, BooleanType)
      case IsNotNull(c) if !c.nullable => Literal.create(true, BooleanType)
      case EqualNullSafe(Literal(null, _), r) => IsNull(r)
      case Coalesce(children) =>
        val newChildren = children.filterNot(isNullLiteral)
        // ... trimming trailing nulls, truncating at first non-nullable
      case In(_, list) if list.isEmpty => Literal.create(false, BooleanType)
      // ... more cases
    }
  }
}
```

**Performance impact:** Eliminates null-check branches and simplifies expressions. The `COUNT(null...)` to `0` transformation avoids unnecessary aggregation.

---

### 5. Boolean Simplification

**Source:** `expressions.scala` line 392 -- `BooleanSimplification`

**What it does:** Simplifies boolean expressions using algebraic identities and De Morgan's laws.

**Transformations:**

| Before | After |
|--------|-------|
| `true AND x` | `x` |
| `false OR x` | `x` |
| `false AND x` | `false` |
| `true OR x` | `true` |
| `a AND (NOT a)` | `IF(IS_NULL(a), NULL, false)` |
| `a OR (NOT a)` | `IF(IS_NULL(a), NULL, true)` |
| `a AND a` | `a` |
| `NOT(NOT(a))` | `a` |
| `NOT(a > b)` | `a <= b` |
| `NOT(a AND b)` | `NOT(a) OR NOT(b)` (De Morgan) |
| `(a OR b OR c) AND (a OR b)` | `a OR b` (common factor elimination) |
| `(a AND b) AND a AND (a AND c)` | `a AND b AND c` (dedup) |

**Implementation:**
```scala
// expressions.scala (lines 392-563)
object BooleanSimplification extends Rule[LogicalPlan] with PredicateHelper {
  val actualExprTransformer: PartialFunction[Expression, Expression] = {
    case TrueLiteral And e => e
    case FalseLiteral And _ => FalseLiteral
    case a And b if Not(a).semanticEquals(b) =>
      If(IsNull(a), Literal.create(null, a.dataType), FalseLiteral)
    case a And b if a.semanticEquals(b) => a
    // Common factor elimination for conjunction and disjunction
    case and @ (left And right) =>
      val lhs = splitDisjunctivePredicates(left)
      val rhs = splitDisjunctivePredicates(right)
      val common = lhs.filter(e => rhs.exists(e.semanticEquals))
      // ... apply distributive law to eliminate common factors
    case not: Not => simplifyNot(not)  // De Morgan's laws, double negation
  }
}
```

**Performance impact:** Reduces the number of boolean evaluations. Common factor elimination is particularly valuable for complex WHERE clauses generated by ORMs or query builders.

---

### 6. Limit PushDown

**Source:** `Optimizer.scala` line 890 -- `LimitPushDown`

**What it does:** Pushes `LocalLimit` operators through unions, joins, and other operators to reduce the amount of data processed.

**Before:**
```
GlobalLimit(10)
  +- LocalLimit(10)
     +- Union
        +- Scan t1  // 1M rows
        +- Scan t2  // 1M rows
```

**After:**
```
GlobalLimit(10)
  +- LocalLimit(10)
     +- Union
        +- LocalLimit(10)     // Pushed down
           +- Scan t1
        +- LocalLimit(10)     // Pushed down
           +- Scan t2
```

**Implementation:**
```scala
// Optimizer.scala (lines 890-988)
object LimitPushDown extends Rule[LogicalPlan] {
  def apply(plan: LogicalPlan): LogicalPlan = plan.transformWithPruning(
    _.containsAnyPattern(LIMIT, LEFT_SEMI_OR_ANTI_JOIN), ruleId) {
    // Push through Union
    case LocalLimit(exp, u: Union) =>
      LocalLimit(exp, u.copy(children = u.children.map(maybePushLocalLimit(exp, _))))
    // Push through Join (careful: outer joins need special handling)
    case LocalLimit(exp, join: Join) =>
      LocalLimit(exp, pushLocalLimitThroughJoin(exp, join))
    // Push through GroupOnly Aggregate
    case Limit(le @ IntegerLiteral(1), a: Aggregate) if a.groupOnly =>
      // Turn into Project after pushing limit
  }
}
```

**Performance impact:** Dramatic for UNION ALL queries -- instead of scanning all inputs and then taking the top N, each union branch stops after producing N rows.

**Gotcha:** Limits are NOT pushed through joins with conditions (because you cannot know how many rows the join will produce). They ARE pushed through cross joins and outer joins without conditions.

---

### 7. Constant Propagation / Foldable Propagation

**Source:** `expressions.scala` lines 135 (ConstantPropagation) and 1018 (FoldablePropagation)

**ConstantPropagation:** Replaces attributes with their known constant values in conjunctive predicates.

**Before:**
```
Filter(i = 5 AND j = i + 3)
  +- Scan t
```

**After:**
```
Filter(i = 5 AND j = 8)
  +- Scan t
```

```scala
// expressions.scala (lines 135-267)
object ConstantPropagation extends Rule[LogicalPlan] {
  private def traverse(condition, replaceChildren, nullIsFalse)
    : (Option[Expression], AttributeMap[(Literal, BinaryComparison)]) =
    condition match {
      case e @ EqualTo(left: AttributeReference, right: Literal) if safeToReplace(left, nullIsFalse) =>
        (None, AttributeMap(Map(left -> (right, e))))
      case a: And =>
        val (newLeft, eqPredsLeft) = traverse(a.left, replaceChildren = false, nullIsFalse)
        val (newRight, eqPredsRight) = traverse(a.right, replaceChildren = false, nullIsFalse)
        val eqPreds = eqPredsLeft ++ eqPredsRight
        // Replace attributes with constants across the And tree
    }
}
```

**FoldablePropagation:** Replaces attributes that alias foldable expressions with the original foldable expression, enabling further optimization.

**Before:**
```
Sort(x ASC, y ASC, z ASC)
  +- Project(1.0 AS x, 'abc' AS y, now() AS z)
     +- Scan t
```

**After:**
```
// ORDER BY eliminated because x=1.0, y='abc', z=now() are all constant
Project(1.0 AS x, 'abc' AS y, now() AS z)
  +- Scan t
```

---

### 8. Combine Filters / Prune Filters

**CombineFilters** (`Optimizer.scala` line 1913): Merges adjacent `Filter` nodes into one.

```scala
// Optimizer.scala (lines 1913-1932)
object CombineFilters extends Rule[LogicalPlan] with PredicateHelper {
  val applyLocally: PartialFunction[LogicalPlan, LogicalPlan] = {
    case Filter(fc, nf @ Filter(nc, grandChild)) if nc.deterministic =>
      val (combineCandidates, rest) =
        splitConjunctivePredicates(fc).partition(p => p.deterministic && !p.throwable)
      val mergedFilter = (ExpressionSet(combineCandidates) --
        ExpressionSet(splitConjunctivePredicates(nc))).reduceOption(And) match {
        case Some(ac) => Filter(And(nc, ac), grandChild)
        case None => nf
      }
      rest.reduceOption(And).map(c => Filter(c, mergedFilter)).getOrElse(mergedFilter)
  }
}
```

**PruneFilters** (`Optimizer.scala` line 2029): Removes filters that always evaluate to `true` or replaces the entire plan with an empty relation for filters that always evaluate to `false`. Also eliminates conditions that are guaranteed true by child constraints.

**Before:**
```
Filter(true)
  +- Filter(amount > 0 AND amount > 0)
     +- Filter(childKey IS NOT NULL)
        +- Filter(childKey > 0)   // childKey is NOT NULL from above
           +- Scan t
```

**After:**
```
Filter(amount > 0)
  +- Scan t
```

---

### 9. Eliminate Outer Join

**Source:** `joins.scala` line 158 -- `EliminateOuterJoin`

**What it does:** Converts outer joins to inner joins (or simpler outer joins) when the filter on the result would eliminate null-supplemented rows anyway.

**Transformations:**

| Pattern | Result |
|---------|--------|
| `Filter(leftCol IS NOT NULL) <- LeftOuter` | Inner join |
| `Filter(rightCol IS NOT NULL) <- RightOuter` | Inner join |
| `Filter(leftCol IS NOT NULL AND rightCol IS NOT NULL) <- FullOuter` | Inner join |
| `Filter(leftCol IS NOT NULL) <- FullOuter` | LeftOuter |
| `Aggregate(leftColsOnly) <- LeftOuter` | Eliminate join, just use left |

**Implementation:**
```scala
// joins.scala (lines 158-238)
object EliminateOuterJoin extends Rule[LogicalPlan] with PredicateHelper {
  private def canFilterOutNull(e: Expression): Boolean = {
    // Evaluates expression with all-null inputs; returns true if result is null or false
  }
  private def buildNewJoinType(filter: Filter, join: Join): JoinType = {
    val leftConditions = conditions.filter(_.references.subsetOf(join.left.outputSet))
    val rightConditions = conditions.filter(_.references.subsetOf(join.right.outputSet))
    join.joinType match {
      case RightOuter if leftHasNonNullPredicate => Inner
      case LeftOuter if rightHasNonNullPredicate => Inner
      case FullOuter if leftHasNonNullPredicate && rightHasNonNullPredicate => Inner
      case FullOuter if leftHasNonNullPredicate => LeftOuter
      case FullOuter if rightHasNonNullPredicate => RightOuter
    }
  }
}
```

**Performance impact:** Inner joins can use more join strategies (e.g., SortMergeJoin) and avoid the overhead of null-padding. Complete elimination of unnecessary outer joins avoids the join entirely.

---

### 10. PushDownLeftSemiAntiJoin

**Source:** `PushDownLeftSemiAntiJoin.scala` line 35

**What it does:** Pushes Left Semi and Left Anti joins below projections, aggregates, windows, and unions -- similar to predicate pushdown but for existence joins.

**Before:**
```
Project(name, dept)
  +- LeftSemiJoin (employees DEPARTMENTS ON emp.deptId = dept.id)
     +- Project(id, name, deptId, salary, ...)  // wide projection
     +- Scan departments
```

**After:**
```
LeftSemiJoin (Project(id, name, deptId) DEPARTMENTS ON emp.deptId = dept.id)
  +- Project(id, name, deptId, salary, ...)  // Semi join pushed below
  +- Scan departments
```

This is useful because the semi join then enables column pruning on the right side.

---

### 11. RemoveRedundantAggregates

**Source:** `RemoveRedundantAggregates.scala` line 30

**What it does:** Eliminates aggregate nodes that are subsumed by a parent aggregate.

**Before:**
```
Aggregate(GROUP BY a, AGG(sum(b)))
  +- Aggregate(GROUP BY a, b, AGG(count(*)))
     +- Scan t
```

**After:**
```
Aggregate(GROUP BY a, AGG(sum(b)))
  +- Project(a, b)
     +- Scan t
// The inner Aggregate is removed -- parent already groups by a
```

```scala
// RemoveRedundantAggregates.scala (lines 30-68)
object RemoveRedundantAggregates extends Rule[LogicalPlan] with AliasHelper {
  def apply(plan: LogicalPlan): LogicalPlan = plan.transformUpWithPruning(
    _.containsPattern(AGGREGATE), ruleId) {
    case upper @ Aggregate(_, _, lower: Aggregate, _) if isLowerRedundant(upper, lower) =>
      upper.copy(child = Project(lower.aggregateExpressions.filter(upper.references.contains), lower.child))
    case agg @ Aggregate(groupingExps, _, child, _)
        if agg.groupOnly && child.distinctKeys.exists(_.subsetOf(ExpressionSet(groupingExps))) =>
      Project(agg.aggregateExpressions, child)
  }
}
```

---

### 12. Inline CTE

**Source:** `InlineCTE.scala` line 43 -- `InlineCTE`

**What it does:** Replaces CTE (`WITH` clause) references with the CTE definition body, avoiding materialization overhead.

**Inlining decision:** A CTE is inlined if:
1. It is deterministic (no non-deterministic expressions), OR
2. It is referenced only once, OR
3. It references an outer query (correlated)

CTEs that are non-deterministic AND referenced multiple times are NOT inlined (they would produce different results on each reference).

**Before:**
```
WithCTE(
  CTERelationDef(id=1, SELECT * FROM t WHERE x > 0),
  Project(*)
    +- CTERelationRef(id=1)
)
```

**After (single reference):**
```
Project(*)
  +- Filter(x > 0)
     +- Scan t
// CTE body inlined directly
```

**Implementation:** The rule builds a map of all CTEs and their reference counts, then decides whether to inline each one. For CTEs referenced in subqueries, it pulls them up to the main query level.

---

### 13. Propagate Empty Relation

**Source:** `PropagateEmptyRelation.scala` line 219 -- `PropagateEmptyRelation`

**What it does:** When one side of a join, union, or filter is known to be empty (zero rows), eliminates the entire operation or simplifies it.

**Transformations:**

| Pattern | Result |
|---------|--------|
| `EmptyRelation JOIN X` (Inner) | `EmptyRelation` |
| `EmptyRelation LEFT OUTER JOIN X` | `EmptyRelation` |
| `X LEFT OUTER JOIN EmptyRelation` | `Project(null-padded, X)` |
| `Filter(any, EmptyRelation)` | `EmptyRelation` |
| `EmptyRelation UNION ALL X` | `X` |

```scala
// PropagateEmptyRelation.scala (lines 219-223)
object PropagateEmptyRelation extends PropagateEmptyRelationBase {
  override protected def applyInternal(p: LogicalPlan): LogicalPlan = p.transformUpWithPruning(
    _.containsAnyPattern(LOCAL_RELATION, TRUE_OR_FALSE_LITERAL), ruleId) {
    commonApplyFunc
  }
}
```

**Performance impact:** Can reduce entire query branches to zero work. Particularly effective for partitioned tables where partition pruning produces an empty scan.

---

## How to See the Analyzed vs Optimized Plan

### Using EXPLAIN EXTENDED

```sql
EXPLAIN EXTENDED
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE e.salary > 50000 AND d.location = 'NY';
```

Output shows all four plan phases:

```
== Parsed Logical Plan ==
'Project ['e.name, 'd.dept_name]
+- 'Join Inner, ('e.dept_id = 'd.id)
   :- 'UnresolvedRelation `employees`
   +- 'UnresolvedRelation `departments`

== Analyzed Logical Plan ==
name: string, dept_name: string
Project [name#1, dept_name#5]
+- Join Inner, (dept_id#2 = id#4)
   :- SubqueryAlias e, Relation [id#0,name#1,dept_id#2,salary#3] parquet
   +- SubqueryAlias d, Relation [id#4,dept_name#5,location#6] parquet

== Optimized Logical Plan ==    <-- THIS is after Catalyst
Project [name#1, dept_name#5]
+- Join Inner, (dept_id#2 = id#4)
   :- Filter (isnotnull(salary#3) AND (salary#3 > 50000))
   :  +- Relation [id#0,name#1,dept_id#2,salary#3] parquet
   +- Filter (location#6 = NY)
      +- Relation [id#4,dept_name#5,location#6] parquet

== Physical Plan ==
...
```

### Programmatic Access in Scala

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().appName("CatalystDemo").getOrCreate()

val df = spark.sql("""
  SELECT e.name, d.dept_name
  FROM employees e
  JOIN departments d ON e.dept_id = d.id
  WHERE e.salary > 50000
""")

// Analyzed plan (before optimization)
val analyzedPlan = df.queryExecution.analyzed
println("=== ANALYZED ===")
println(analyzedPlan.numberedTreeString)

// Optimized plan (after Catalyst)
val optimizedPlan = df.queryExecution.optimizedPlan
println("=== OPTIMIZED ===")
println(optimizedPlan.numberedTreeString)

// Physical plan
val physicalPlan = df.queryExecution.executedPlan
println("=== PHYSICAL ===")
println(physicalPlan.numberedTreeString)
```

### Seeing Individual Rule Applications

Enable the plan change logger to see every rule's effect:

```scala
spark.conf.set("spark.sql.log.level", "DEBUG")
spark.conf.set("spark.sql.optimizer.excludedRules", "")  // Ensure no rules excluded

// Or at the session level for all queries:
spark.conf.set("spark.sql.planChangeLogLevel", "INFO")
```

When enabled, the executor logs each rule that changes the plan:

```
=== Applying Rule org.apache.spark.sql.catalyst.optimizer.PushDownPredicates ===
 Filter (salary#3 > 50000)              Project [name#1, dept_name#5]
   Project [name#1, dept_name#5]          +- Join Inner, ...
     +- Join Inner, ...                      :- Filter (salary#3 > 50000)
        :- Scan employees                       +- Scan employees
        +- Scan departments                     +- Scan departments
```

---

## How to See Rule Executor Metrics

```scala
import org.apache.spark.sql.catalyst.rules.RuleExecutor

// After running a query
println(RuleExecutor.dumpTimeSpent())
// Output:
// === Time Spent in Optimizer ===
// Total time: 1234 ms
// Top rules:
//   PushDownPredicates: 150 ms (45 runs, 12 effective)
//   ColumnPruning: 80 ms (30 runs, 8 effective)
//   ...
```

---

## Key Configurations

### Excluding Specific Rules

```scala
// Exclude specific optimizer rules (comma-separated rule names)
spark.conf.set("spark.sql.optimizer.excludedRules",
  "org.apache.spark.sql.catalyst.optimizer.PushDownPredicates")

// Reset to defaults
spark.conf.set("spark.sql.optimizer.excludedRules", "")
```

### Optimizer Iterations

```scala
// Maximum iterations for fixed-point batches (default: 100)
spark.conf.set("spark.sql.optimizer.maxIterations", "100")
```

### Plan Change Logging

```scala
// Log level for plan changes: OFF, DEBUG, INFO, WARN, ERROR
spark.conf.set("spark.sql.planChangeLogLevel", "INFO")

// Only log specific batches/rules (comma-separated)
spark.conf.set("spark.sql.planChangeBatches", "Operator Optimization")
spark.conf.set("spark.sql.planChangeRules",
  "org.apache.spark.sql.catalyst.optimizer.PushDownPredicates")
```

### Plan Validation

```scala
// Validate plan after each rule (catches bugs in rules)
spark.conf.set("spark.sql.planChangeValidation", "true")

// Lightweight validation (enabled by default)
spark.conf.set("spark.sql.lightweightPlanChangeValidation", "true")
```

---

## Limitations and Gotchas

### 1. Rules Are Order-Dependent

Catalyst rules are not commutative. The order in `defaultBatches` matters. If you write a custom rule, placement in the batch sequence determines what patterns it can match.

### 2. Non-Deterministic Expressions Block Pushdown

Functions like `rand()`, `current_timestamp()`, `input_file_name()` prevent predicate pushdown because their values change per-row. A filter referencing these stays at its original position.

### 3. UDFs Are Opaque to the Optimizer

User-defined functions are black boxes. Catalyst cannot see inside a UDF to push predicates through it. Use built-in functions whenever possible.

### 4. Fixed-Point May Not Converge

If you write conflicting custom rules (rule A enables rule B which enables rule A), the optimizer hits `maxIterations` and logs a warning. Set `spark.sql.optimizer.maxIterations` higher if legitimate rules need more iterations.

### 5. Statistics Are Not Used in Rule-Based Optimization

The Catalyst optimizer (without CBO) does not use table statistics. Rules like `PushDownPredicates` and `ColumnPruning` are purely structural. Join order is left-to-right as written in the query.

### 6. Transform Pruning Can Miss Patterns

`transformWithPruning` skips subtrees that don't match the condition. If a rule adds a new pattern (e.g., creates a `Filter` node), subsequent rules in the same pass might not see it if the pruning condition was already evaluated. The fixed-point loop handles this on the next iteration.

### 7. Some Rules May Increase Plan Size Temporarily

Rules like `BooleanSimplification` with De Morgan's laws can expand the tree (one `NOT` becomes two `NOT`s). The tree shrinks again after `ConstantFolding` or other rules clean up. This is why the optimizer runs multiple fixed-point passes.

### 8. Excluded Rules Can Break Optimization

Excluding critical rules (like `PushDownPredicates`) can cause severe performance degradation. Only exclude rules for debugging or specific edge cases.

### 9. Rule Idempotence

Rules marked with `Once` strategy must be idempotent -- running them twice produces the same result. Spark validates this in testing. If a `Once` rule's idempotence is broken, it throws an error.

---

## Production-Ready Python (PySpark) Code

### 1. PySpark Session Setup with Optimizer Configurations

```python
"""
PySpark session setup demonstrating all Catalyst optimizer configurations.
Run: spark-submit --master local[*] catalyst_session_setup.py
"""
import logging
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("catalyst_setup")


def create_spark_session(
    *,
    excluded_rules: list[str] | None = None,
    max_iterations: int = 100,
    plan_change_log_level: str = "OFF",
    plan_change_validation: bool = False,
    app_name: str = "CatalystOptimizerDemo",
) -> SparkSession:
    """Create a SparkSession with Catalyst optimizer configurations.

    Args:
        excluded_rules: Rule class names to exclude from optimization.
            Pass empty list to exclude nothing. Default is None (Spark defaults).
        max_iterations: Maximum fixed-point iterations for optimizer batches.
            Default is 100.
        plan_change_log_level: Log level for plan changes (OFF, DEBUG, INFO, WARN, ERROR).
            Default is "OFF". Set to "INFO" for production debugging.
        plan_change_validation: Whether to validate plan after each rule application.
            Expensive; use only for debugging custom rules.
        app_name: Application name shown in Spark UI.

    Returns:
        Configured SparkSession.
    """
    try:
        builder = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )

        if excluded_rules:
            rule_str = ",".join(excluded_rules)
            builder = builder.config("spark.sql.optimizer.excludedRules", rule_str)
            logger.info("Excluding optimizer rules: %s", rule_str)

        builder = builder.config("spark.sql.optimizer.maxIterations", str(max_iterations))

        if plan_change_log_level != "OFF":
            builder = builder.config(
                "spark.sql.planChangeLogLevel", plan_change_log_level,
            )
            logger.info(
                "Plan change logging enabled at level: %s", plan_change_log_level,
            )

        if plan_change_validation:
            builder = builder.config("spark.sql.planChangeValidation", "true")
            logger.warning("Plan validation enabled -- significant overhead expected")

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession created successfully (version: %s)", spark.version)
        return spark

    except Exception as exc:
        logger.error("Failed to create SparkSession: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Default session
    spark = create_spark_session(
        plan_change_log_level="INFO",
        app_name="CatalystDemo",
    )

    # Session with specific rules excluded (for debugging)
    debug_spark = create_spark_session(
        excluded_rules=[
            "org.apache.spark.sql.catalyst.optimizer.PushDownPredicates",
        ],
        max_iterations=50,
        plan_change_log_level="DEBUG",
        plan_change_validation=True,
        app_name="CatalystDebug",
    )

    spark.stop()
    debug_spark.stop()
```

### 2. Rule Execution Metrics (PySpark Equivalent of RuleExecutor Metrics)

```python
"""
Read optimizer metrics from Spark's internal metrics after query execution.

Spark does not expose RuleExecutor.dumpTimeSpent() via PySpark directly,
so we parse the Spark UI event log and the plan-change log output instead.
This script demonstrates both approaches.

Run: spark-submit --master local[*] catalyst_metrics.py
"""
import json
import logging
import re
import sys
from pathlib import Path
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("catalyst_metrics")


def capture_plan_changes(
    spark: SparkSession, df, query_name: str = "unknown",
) -> list[dict]:
    """Capture individual rule applications from the optimized plan comparison.

    By comparing analyzed vs optimized plans, we can identify which rule
    categories produced transformations.

    Args:
        spark: Active SparkSession.
        df: DataFrame whose plan to analyze.
        query_name: Identifier for this query in the output.

    Returns:
        List of dicts with rule category, before node count, after node count.
    """
    try:
        analyzed = df.queryExecution.analyzed
        optimized = df.queryExecution.optimizedPlan

        analyzed_nodes = analyzed.numberedTreeString.strip().split("\n")
        optimized_nodes = optimized.numberedTreeString.strip().split("\n")

        plan_diff = {
            "query": query_name,
            "analyzed_nodes": len(analyzed_nodes),
            "optimized_nodes": len(optimized_nodes),
            "nodes_eliminated": len(analyzed_nodes) - len(optimized_nodes),
            "analyzed_plan": analyzed.numberedTreeString,
            "optimized_plan": optimized.numberedTreeString,
        }

        logger.info(
            "Query '%s': %d analyzed nodes -> %d optimized nodes (%d eliminated)",
            query_name,
            len(analyzed_nodes),
            len(optimized_nodes),
            plan_diff["nodes_eliminated"],
        )

        return [plan_diff]

    except Exception as exc:
        logger.error("Failed to capture plan changes: %s", exc, exc_info=True)
        return []


def parse_optimizer_logs(log_text: str) -> list[dict]:
    """Parse plan-change log output to extract per-rule metrics.

    Spark's plan-change log (when planChangeLogLevel=INFO) emits blocks like:

        === Applying Rule org.apache.spark.sql.catalyst.optimizer.PushDownPredicates ===
        [before plan] [after plan]

    This function extracts rule names and counts applications.

    Args:
        log_text: Raw log text from Spark executor or driver log.

    Returns:
        List of dicts with rule_name and apply_count.
    """
    try:
        pattern = re.compile(
            r"=== Applying Rule (org\.apache\.spark\.sql\.catalyst\.\S+) ==="
        )
        rule_counts: dict[str, int] = {}

        for match in pattern.finditer(log_text):
            rule_name = match.group(1)
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        results = [
            {"rule_name": name, "apply_count": count}
            for name, count in sorted(
                rule_counts.items(), key=lambda x: x[1], reverse=True,
            )
        ]

        logger.info("Parsed %d distinct optimizer rules from log", len(results))
        return results

    except Exception as exc:
        logger.error("Failed to parse optimizer logs: %s", exc, exc_info=True)
        return []


def explain_rule_summary(df) -> dict:
    """Generate a summary of transformations by comparing EXPLAIN output phases.

    Uses EXPLAIN EXTENDED to get parsed, analyzed, optimized, and physical plans,
    then reports node counts per phase.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dict with plan phase node counts.
    """
    try:
        explain_text = df._jdf.queryExecution().toString()

        phases = {
            "parsed": 0,
            "analyzed": 0,
            "optimized": 0,
        }

        current_phase = None
        for line in explain_text.split("\n"):
            stripped = line.strip()
            if "== Parsed Logical Plan ==" in stripped:
                current_phase = "parsed"
                continue
            elif "== Analyzed Logical Plan ==" in stripped:
                current_phase = "analyzed"
                continue
            elif "== Optimized Logical Plan ==" in stripped:
                current_phase = "optimized"
                continue
            elif stripped.startswith("==") or stripped == "":
                if current_phase is not None and stripped == "":
                    current_phase = None
                continue

            if current_phase in phases and stripped:
                phases[current_phase] += 1

        phases["nodes_saved_by_optimizer"] = phases["analyzed"] - phases["optimized"]
        return phases

    except Exception as exc:
        logger.error("Failed to generate rule summary: %s", exc, exc_info=True)
        return {}


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CatalystMetrics").getOrCreate()

    try:
        # Create sample data
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW employees AS "
                  "SELECT * FROM VALUES (1, 'Alice', 1, 60000), (2, 'Bob', 2, 45000) "
                  "AS t(id, name, dept_id, salary)")
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW departments AS "
                  "SELECT * FROM VALUES (1, 'Engineering'), (2, 'Sales') "
                  "AS t(id, dept_name)")

        df = spark.sql("""
            SELECT e.name, d.dept_name
            FROM employees e
            JOIN departments d ON e.dept_id = d.id
            WHERE e.salary > 50000
        """)

        # Capture plan changes
        metrics = capture_plan_changes(spark, df, "salary_join_query")
        for m in metrics:
            logger.info("Plan diff: %s", json.dumps(m, indent=2))

        # Summarize optimization effect
        summary = explain_rule_summary(df)
        logger.info("Plan phase summary: %s", json.dumps(summary, indent=2))

    except Exception as exc:
        logger.error("Metrics collection failed: %s", exc, exc_info=True)
    finally:
        spark.stop()
```

### 3. Predicate Pushdown Verification via EXPLAIN Parsing

```python
"""
Verify that predicate pushdown actually occurs by creating a DataFrame with
filters, capturing the EXPLAIN output, and parsing it for pushdown evidence.

Run: spark-submit --master local[*] predicate_pushdown_verify.py
"""
import logging
import re
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("pushdown_verify")


def verify_predicate_pushdown(spark: SparkSession, parquet_path: str) -> dict:
    """Verify predicate pushdown on a Parquet scan with filter conditions.

    Creates a DataFrame with WHERE clauses, runs EXPLAIN, and checks that:
    1. The Filter node appears in the optimized plan (not just physical)
    2. Pushed filters appear in the physical plan's Scan operation
    3. The plan structure confirms pushdown through projections/joins

    Args:
        spark: Active SparkSession with Hive/Parquet support.
        parquet_path: Path to the Parquet data to query.

    Returns:
        Dict with pushdown verification results.
    """
    try:
        df = (
            spark.read.parquet(parquet_path)
            .filter("salary > 50000")
            .filter("dept_id = 1")
            .select("name", "salary")
        )

        # Get the optimized logical plan
        optimized_plan = df.queryExecution.optimizedPlan
        optimized_str = optimized_plan.numberedTreeString.strip().split("\n")

        # Get the physical plan
        physical_plan = df.queryExecution.executedPlan
        physical_str = physical_plan.numberedTreeString.strip().split("\n")

        result = {
            "optimized_plan_lines": optimized_str,
            "physical_plan_lines": physical_str,
            "pushdown_verified": False,
            "details": [],
        }

        # Check 1: Filter appears in optimized plan (pushed below Project)
        has_filter_in_optimized = any(
            "Filter" in line for line in optimized_str
        )
        if has_filter_in_optimized:
            result["details"].append(
                "PASS: Filter node present in optimized logical plan"
            )
        else:
            result["details"].append(
                "WARN: No Filter in optimized plan -- may have been "
                "pushed to data source directly"
            )

        # Check 2: Check that Filter is below Project (not above)
        filter_idx = None
        project_idx = None
        for i, line in enumerate(optimized_str):
            if "Filter" in line and filter_idx is None:
                filter_idx = i
            if "Project" in line and project_idx is None:
                project_idx = i

        if filter_idx is not None and project_idx is not None:
            # In tree string, children are indented more (appear after parent)
            # Filter should appear at same or deeper indentation than Project
            # if it was pushed below
            filter_indent = len(optimized_str[filter_idx]) - len(
                optimized_str[filter_idx].lstrip()
            )
            project_indent = len(optimized_str[project_idx]) - len(
                optimized_str[project_idx].lstrip()
            )
            if filter_indent >= project_indent:
                result["details"].append(
                    "PASS: Filter is at or below Project level (pushed down)"
                )
            else:
                result["details"].append(
                    "WARN: Filter is above Project in optimized plan"
                )

        # Check 3: Physical plan shows pushed filters in scan
        physical_text = "\n".join(physical_str)
        pushed_filter_pattern = re.compile(
            r"PushedFilters:.*\[.*\]", re.IGNORECASE,
        )
        pushed_match = pushed_filter_pattern.search(physical_text)
        if pushed_match:
            result["pushdown_verified"] = True
            result["details"].append(
                f"PASS: PushedFilters in physical plan: {pushed_match.group()}"
            )
        else:
            # Also check for Filter condition in FileScan
            filescan_filter = re.findall(
                r"Filter.*\([^)]+\)", physical_text,
            )
            if filescan_filter:
                result["pushdown_verified"] = True
                result["details"].append(
                    f"PASS: Filter in FileScan: {filescan_filter[0]}"
                )
            else:
                result["details"].append(
                    "WARN: No explicit pushed filter signature in physical plan"
                )

        for detail in result["details"]:
            logger.info(detail)

        return result

    except Exception as exc:
        logger.error("Pushdown verification failed: %s", exc, exc_info=True)
        return {"error": str(exc), "pushdown_verified": False}


def verify_pushdown_through_join(spark: SparkSession) -> dict:
    """Verify predicate pushdown through a join operation.

    Confirms that WHERE clause filters are applied to individual scan
    operators before the join, not on the join result.

    Returns:
        Dict with join pushdown verification results.
    """
    try:
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW t1 AS "
                  "SELECT * FROM VALUES (1, 'A', 100), (2, 'B', 200), (3, 'C', 300) "
                  "AS t1(id, name, value)")
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW t2 AS "
                  "SELECT * FROM VALUES (1, 'X', 10), (2, 'Y', 20), (4, 'Z', 40) "
                  "AS t2(id, category, score)")

        df = spark.sql("""
            SELECT t1.name, t2.category
            FROM t1 JOIN t2 ON t1.id = t2.id
            WHERE t1.value > 150
              AND t2.score < 30
        """)

        optimized = df.queryExecution.optimizedPlan
        plan_str = optimized.numberedTreeString

        # Count Filter nodes in optimized plan -- should see at least 2
        # (one per side of the join)
        filter_count = plan_str.count("Filter")

        result = {
            "filter_count_in_optimized_plan": filter_count,
            "pushdown_through_join_verified": filter_count >= 2,
            "optimized_plan": plan_str,
        }

        if filter_count >= 2:
            logger.info(
                "PASS: %d Filter nodes in optimized plan -- "
                "predicates pushed to both sides of join",
                filter_count,
            )
        else:
            logger.info(
                "WARN: Only %d Filter(s) -- some predicates may not have "
                "been pushed down",
                filter_count,
            )

        return result

    except Exception as exc:
        logger.error("Join pushdown verification failed: %s", exc, exc_info=True)
        return {"error": str(exc), "pushdown_through_join_verified": False}


if __name__ == "__main__":
    spark = SparkSession.builder.appName("PushdownVerify").getOrCreate()

    try:
        # Create test Parquet data
        test_path = "/tmp/catalyst_pushdown_test"
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW test_data AS "
                  "SELECT * FROM VALUES "
                  "(1, 'Alice', 1, 60000), "
                  "(2, 'Bob', 2, 45000), "
                  "(3, 'Carol', 1, 75000), "
                  "(4, 'Dave', 3, 52000) "
                  "AS t(id, name, dept_id, salary)")
        spark.read.table("test_data").write.mode("overwrite").parquet(test_path)

        # Verify pushdown on Parquet scan
        logger.info("=== Predicate Pushdown on Parquet Scan ===")
        result = verify_predicate_pushdown(spark, test_path)
        logger.info("Result: %s", result)

        # Verify pushdown through join
        logger.info("=== Predicate Pushdown Through Join ===")
        join_result = verify_pushdown_through_join(spark)
        logger.info("Result: %s", join_result)

    except Exception as exc:
        logger.error("Verification failed: %s", exc, exc_info=True)
    finally:
        spark.stop()
```

### 4. Column Pruning Verification

```python
"""
Verify column pruning by checking which columns are actually read from the
data source versus which columns are projected in the final output.

Run: spark-submit --master local[*] column_pruning_verify.py
"""
import logging
import re
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("column_pruning")


def verify_column_pruning(spark: SparkSession, parquet_path: str) -> dict:
    """Verify that unused columns are pruned from the scan.

    Creates a wide table, selects only a few columns, and verifies:
    1. The optimized plan's scan only lists selected columns
    2. The physical plan's FileScan shows only required columns
    3. Columns not in the select list are absent from the read schema

    Args:
        spark: Active SparkSession.
        parquet_path: Path to wide Parquet data.

    Returns:
        Dict with column pruning verification results.
    """
    try:
        df = spark.read.parquet(parquet_path).select("name", "salary")

        optimized = df.queryExecution.optimizedPlan
        physical = df.queryExecution.executedPlan

        # Get all columns available in the source
        source_columns = spark.read.parquet(parquet_path).columns
        selected_columns = set(df.columns)

        # Extract columns from optimized plan scan node
        optimized_str = optimized.numberedTreeString
        optimized_scan_cols = set()

        # Look for Relation or FileScan lines with column lists
        for line in optimized_str.split("\n"):
            if "Relation" in line or "FileSourceScan" in line:
                # Pattern: Relation [col1#0, col2#1, ...] format
                col_match = re.findall(r"(\w+)#\d+", line)
                optimized_scan_cols.update(col_match)

        # Extract columns from physical plan scan
        physical_str = physical.numberedTreeString
        physical_scan_cols = set()
        for line in physical_str.split("\n"):
            if "FileScan" in line:
                col_match = re.findall(r"(\w+)#\d+", line)
                physical_scan_cols.update(col_match)

        pruned_columns = source_columns - selected_columns

        result = {
            "source_columns": sorted(source_columns),
            "selected_columns": sorted(selected_columns),
            "pruned_columns": sorted(pruned_columns),
            "optimized_plan_scan_columns": sorted(optimized_scan_cols),
            "physical_plan_scan_columns": sorted(physical_scan_cols),
            "pruning_verified": False,
            "details": [],
        }

        # Verify: physical plan should only show selected columns
        if physical_scan_cols:
            non_selected_in_physical = physical_scan_cols - selected_columns
            if not non_selected_in_physical:
                result["pruning_verified"] = True
                result["details"].append(
                    "PASS: Physical scan only reads selected columns"
                )
            else:
                result["details"].append(
                    f"WARN: Non-selected columns in physical scan: "
                    f"{non_selected_in_physical}"
                )
        else:
            result["details"].append(
                "INFO: Could not extract column list from physical plan -- "
                "check plan text format"
            )

        # Verify: optimized plan scan should show fewer columns than source
        if optimized_scan_cols and len(optimized_scan_cols) < len(source_columns):
            result["details"].append(
                f"PASS: Optimized plan scan columns ({len(optimized_scan_cols)}) "
                f"< source columns ({len(source_columns)})"
            )
        elif optimized_scan_cols:
            result["details"].append(
                f"WARN: Scan column count ({len(optimized_scan_cols)}) "
                f"== source ({len(source_columns)}) -- no pruning detected"
            )

        for detail in result["details"]:
            logger.info(detail)

        logger.info(
            "Columns: %d source -> %d selected, %d pruned",
            len(source_columns),
            len(selected_columns),
            len(pruned_columns),
        )

        return result

    except Exception as exc:
        logger.error("Column pruning verification failed: %s", exc, exc_info=True)
        return {"error": str(exc), "pruning_verified": False}


def verify_column_pruning_through_join(spark: SparkSession) -> dict:
    """Verify column pruning on both sides of a join.

    When joining two wide tables but selecting few columns, Catalyst should
    prune columns from both sides before the join.

    Returns:
        Dict with join column pruning results.
    """
    try:
        # Create wide tables
        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW wide_left AS
            SELECT * FROM VALUES
              (1, 'Alice', 'alice@example.com', '555-0001', 'NY', 60000, 'Eng'),
              (2, 'Bob', 'bob@example.com', '555-0002', 'CA', 45000, 'Sales')
            AS t(id, name, email, phone, location, salary, department)
        """)
        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW wide_right AS
            SELECT * FROM VALUES
              (1, 'Eng', 'Engineering', 'Building A', 100),
              (2, 'Sales', 'Sales', 'Building B', 50)
            AS t(id, code, full_name, building, headcount)
        """)

        df = spark.sql("""
            SELECT l.name, r.full_name
            FROM wide_left l
            JOIN wide_right r ON l.department = r.code
            WHERE l.salary > 50000
        """)

        optimized = df.queryExecution.optimizedPlan
        plan_str = optimized.numberedTreeString

        # Count Project nodes -- column pruning adds Project nodes to
        # restrict columns
        project_count = plan_str.count("Project")

        # Check for Filter nodes (column pruning removes project before filter)
        filter_count = plan_str.count("Filter")

        result = {
            "project_nodes_in_optimized": project_count,
            "filter_nodes_in_optimized": filter_count,
            "output_columns": df.columns,
            "optimized_plan": plan_str,
        }

        if project_count >= 2:
            logger.info(
                "PASS: %d Project nodes suggest column pruning active on "
                "multiple plan branches",
                project_count,
            )
        else:
            logger.info(
                "INFO: %d Project node(s) in optimized plan", project_count,
            )

        return result

    except Exception as exc:
        logger.error(
            "Join column pruning verification failed: %s", exc, exc_info=True,
        )
        return {"error": str(exc)}


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ColumnPruningVerify").getOrCreate()

    try:
        # Create wide test data
        test_path = "/tmp/catalyst_column_pruning_test"
        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW wide_data AS
            SELECT * FROM VALUES
              (1, 'Alice', 60000, 'alice@example.com', '555-0001', 'NY',
               'Engineering', '2020-01-15', 4.5, True),
              (2, 'Bob', 45000, 'bob@example.com', '555-0002', 'CA',
               'Sales', '2019-06-01', 3.8, False),
              (3, 'Carol', 75000, 'carol@example.com', '555-0003', 'NY',
               'Engineering', '2018-03-20', 4.9, True)
            AS t(id, name, salary, email, phone, location, department,
                 hire_date, rating, is_active)
        """)
        spark.read.table("wide_data").write.mode("overwrite").parquet(test_path)

        logger.info("=== Column Pruning on Wide Table ===")
        result = verify_column_pruning(spark, test_path)
        logger.info("Result: %s", result)

        logger.info("=== Column Pruning Through Join ===")
        join_result = verify_column_pruning_through_join(spark)
        logger.info("Result: %s", join_result)

    except Exception as exc:
        logger.error("Verification failed: %s", exc, exc_info=True)
    finally:
        spark.stop()
```

### 5. Custom Optimizer Rule Exclusion for Debugging

```python
"""
Demonstrate how to selectively disable Catalyst optimizer rules for debugging.

Use cases:
- Isolating which rule causes incorrect results
- Measuring the performance impact of individual rules
- Workaround for a rule that produces suboptimal plans for specific data

Run: spark-submit --master local[*] rule_exclusion_debug.py
"""
import json
import logging
import sys
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rule_exclusion")

# Commonly debugged optimizer rules
OPTIMIZER_RULES = [
    "org.apache.spark.sql.catalyst.optimizer.PushDownPredicates",
    "org.apache.spark.sql.catalyst.optimizer.ColumnPruning",
    "org.apache.spark.sql.catalyst.optimizer.ConstantFolding",
    "org.apache.spark.sql.catalyst.optimizer.NullPropagation",
    "org.apache.spark.sql.catalyst.optimizer.BooleanSimplification",
    "org.apache.spark.sql.catalyst.optimizer.LimitPushDown",
    "org.apache.spark.sql.catalyst.optimizer.ConstantPropagation",
    "org.apache.spark.sql.catalyst.optimizer.CombineFilters",
    "org.apache.spark.sql.catalyst.optimizer.PruneFilters",
    "org.apache.spark.sql.catalyst.optimizer.EliminateOuterJoin",
    "org.apache.spark.sql.catalyst.optimizer.PushDownLeftSemiAntiJoin",
    "org.apache.spark.sql.catalyst.optimizer.RemoveRedundantAggregates",
    "org.apache.spark.sql.catalyst.optimizer.InlineCTE",
    "org.apache.spark.sql.catalyst.optimizer.PropagateEmptyRelation",
]


def run_query_with_exclusions(
    spark: SparkSession,
    sql_query: str,
    excluded_rules: list[str],
    label: str,
) -> dict:
    """Run a query with specific optimizer rules excluded and capture results.

    Args:
        spark: Active SparkSession (will have config modified).
        sql_query: SQL query to execute.
        excluded_rules: List of fully-qualified rule class names to exclude.
        label: Human-readable label for this run.

    Returns:
        Dict with query plan and row count.
    """
    try:
        if excluded_rules:
            rule_str = ",".join(excluded_rules)
            spark.conf.set("spark.sql.optimizer.excludedRules", rule_str)
            spark.conf.set("spark.sql.planChangeLogLevel", "INFO")
        else:
            spark.conf.set("spark.sql.optimizer.excludedRules", "")
            spark.conf.set("spark.sql.planChangeLogLevel", "OFF")

        df = spark.sql(sql_query)
        optimized = df.queryExecution.optimizedPlan
        physical = df.queryExecution.executedPlan

        row_count = df.count()

        result = {
            "label": label,
            "excluded_rules": excluded_rules,
            "row_count": row_count,
            "optimized_plan": optimized.numberedTreeString,
            "physical_plan": physical.numberedTreeString,
        }

        logger.info(
            "Run '%s': %d rows, %d optimized plan nodes",
            label,
            row_count,
            len(optimized.numberedTreeString.strip().split("\n")),
        )
        return result

    except Exception as exc:
        logger.error("Query with exclusions failed (%s): %s", label, exc)
        return {"label": label, "error": str(exc)}


def compare_rule_impact(spark: SparkSession, sql_query: str) -> list[dict]:
    """Compare query plans with and without each major optimizer rule.

    Iterates through each rule, excluding it individually, and compares
    the resulting plan against the baseline (all rules enabled).

    Args:
        spark: Active SparkSession.
        sql_query: SQL query to test.

    Returns:
        List of comparison results per rule.
    """
    results = []

    # Baseline: all rules enabled
    baseline = run_query_with_exclusions(
        spark, sql_query, [], "baseline_all_rules",
    )
    baseline_node_count = len(
        baseline.get("optimized_plan", "").strip().split("\n"),
    )
    results.append(baseline)

    # Test each rule individually
    for rule in OPTIMIZER_RULES:
        label = f"without_{rule.split('.')[-1]}"
        run_result = run_query_with_exclusions(
            spark, sql_query, [rule], label,
        )
        run_node_count = len(
            run_result.get("optimized_plan", "").strip().split("\n"),
        )

        if "optimized_plan" in run_result:
            plan_changed = run_result["optimized_plan"] != baseline.get(
                "optimized_plan", ""
            )
            run_result["plan_changed"] = plan_changed
            run_result["node_count_diff"] = run_node_count - baseline_node_count

            if plan_changed:
                logger.info(
                    "RULE IMPACT: Excluding %s changed the plan "
                    "(%+d nodes)",
                    rule.split(".")[-1],
                    run_result["node_count_diff"],
                )
            else:
                logger.debug(
                    "No plan change when excluding %s", rule.split(".")[-1],
                )

        results.append(run_result)

    # Reset config
    spark.conf.set("spark.sql.optimizer.excludedRules", "")
    spark.conf.set("spark.sql.planChangeLogLevel", "OFF")

    return results


if __name__ == "__main__":
    spark = SparkSession.builder.appName("RuleExclusionDebug").getOrCreate()

    try:
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW orders AS "
                  "SELECT * FROM VALUES "
                  "(1, 100, 500.0, True), "
                  "(2, 100, 750.0, False), "
                  "(3, 200, 300.0, True), "
                  "(4, 200, 0.0, False), "
                  "(5, 300, 1200.0, True) "
                  "AS t(order_id, customer_id, amount, is_active)")
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW customers AS "
                  "SELECT * FROM VALUES "
                  "(100, 'Alice', 'premium'), "
                  "(200, 'Bob', 'standard'), "
                  "(300, 'Carol', 'premium') "
                  "AS t(id, name, tier)")

        test_query = """
            SELECT c.name, c.tier, SUM(o.amount) AS total_spent
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE o.is_active = true AND o.amount > 0
            GROUP BY c.name, c.tier
            HAVING SUM(o.amount) > 400
            ORDER BY total_spent DESC
        """

        logger.info("=== Baseline vs Individual Rule Exclusion ===")
        comparisons = compare_rule_impact(spark, test_query)

        # Print summary
        changed_rules = [
            r for r in comparisons
            if r.get("plan_changed") is True
        ]
        if changed_rules:
            logger.info("Rules that changed the plan when excluded:")
            for r in changed_rules:
                logger.info(
                    "  - %s (%+d plan nodes)",
                    r["label"],
                    r.get("node_count_diff", 0),
                )
        else:
            logger.info("No individual rule exclusion changed the plan")

    except Exception as exc:
        logger.error("Rule exclusion debug failed: %s", exc, exc_info=True)
    finally:
        spark.stop()
```

### 6. Production Analytics Pipeline (Multiple Catalyst Optimizations)

```python
"""
Production analytics pipeline demonstrating multiple Catalyst optimizations
in action: Parquet scan with filters, join, aggregation -- with verification
of plan optimization at each stage.

Run: spark-submit --master local[*] analytics_pipeline.py
"""
import logging
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, when

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("analytics_pipeline")


class CatalystOptimizationVerifier:
    """Tracks and verifies Catalyst optimizations across a query pipeline."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.verifications: list[dict] = []

    def verify_query(self, df, query_name: str, expected_optimizations: list[str]) -> dict:
        """Verify that expected optimizations are present in the plan.

        Args:
            df: DataFrame representing the query.
            query_name: Label for this query stage.
            expected_optimizations: List of optimization keywords to look for.

        Returns:
            Dict with verification results.
        """
        try:
            optimized = df.queryExecution.optimizedPlan
            physical = df.queryExecution.executedPlan

            opt_text = optimized.numberedTreeString
            phys_text = physical.numberedTreeString

            result = {
                "query": query_name,
                "timestamp": datetime.utcnow().isoformat(),
                "optimized_node_count": len(opt_text.strip().split("\n")),
                "physical_node_count": len(phys_text.strip().split("\n")),
                "optimizations": {},
                "all_verified": True,
            }

            for opt in expected_optimizations:
                found_in_opt = opt in opt_text
                found_in_phys = opt in phys_text
                found = found_in_opt or found_in_phys

                result["optimizations"][opt] = {
                    "found": found,
                    "in_optimized": found_in_opt,
                    "in_physical": found_in_phys,
                }

                if not found:
                    result["all_verified"] = False
                    logger.warning(
                        "Query '%s': expected optimization '%s' not found",
                        query_name,
                        opt,
                    )
                else:
                    logger.info(
                        "Query '%s': optimization '%s' verified",
                        query_name,
                        opt,
                    )

            self.verifications.append(result)
            return result

        except Exception as exc:
            logger.error("Verification failed for '%s': %s", query_name, exc)
            return {"query": query_name, "error": str(exc)}

    def print_summary(self) -> None:
        """Print a summary of all verifications."""
        total = len(self.verifications)
        passed = sum(1 for v in self.verifications if v.get("all_verified"))
        logger.info(
            "Catalyst optimization summary: %d/%d queries fully verified",
            passed,
            total,
        )


def run_analytics_pipeline(spark: SparkSession, data_path: str) -> None:
    """Run a production analytics pipeline with Catalyst verification.

    Pipeline stages:
    1. Filtered Parquet scan (predicate pushdown + column pruning)
    2. Join with dimension table (predicate pushdown through join)
    3. Aggregation (constant folding, null propagation)
    4. Final filter + sort (boolean simplification, limit pushdown)

    Args:
        spark: Active SparkSession with optimizer logging enabled.
        data_path: Path for writing intermediate Parquet data.
    """
    verifier = CatalystOptimizationVerifier(spark)

    try:
        # --- Stage 1: Create test data ---
        logger.info("=== Stage 1: Creating test data ===")

        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW sales AS
            SELECT * FROM VALUES
              (1, 100, '2024-01-15', 500.0, 'electronics', 'CA'),
              (2, 100, '2024-02-20', 750.0, 'electronics', 'CA'),
              (3, 200, '2024-01-10', 300.0, 'clothing', 'NY'),
              (4, 200, '2024-03-05', 0.0, 'clothing', 'NY'),
              (5, 300, '2024-01-25', 1200.0, 'electronics', 'TX'),
              (6, 300, '2024-02-28', 450.0, 'furniture', 'TX'),
              (7, 400, '2024-03-10', 200.0, 'clothing', 'CA'),
              (8, 400, '2024-04-01', 800.0, 'electronics', 'CA')
            AS t(sale_id, customer_id, sale_date, amount, category, state)
        """)
        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW customers AS
            SELECT * FROM VALUES
              (100, 'Alice', 'premium', True),
              (200, 'Bob', 'standard', True),
              (300, 'Carol', 'premium', False),
              (400, 'Dave', 'standard', True)
            AS t(id, name, tier, is_active)
        """)

        # --- Stage 2: Filtered scan with predicate pushdown ---
        logger.info("=== Stage 2: Filtered Parquet scan ===")

        filtered_df = (
            spark.table("sales")
            .filter(col("amount") > 0)
            .filter(col("category") == "electronics")
            .select("sale_id", "customer_id", "amount", "state")
        )

        verifier.verify_query(
            filtered_df,
            "filtered_scan",
            expected_optimizations=["Filter", "Project"],
        )

        filtered_count = filtered_df.count()
        logger.info("Filtered result: %d rows", filtered_count)

        # --- Stage 3: Join with predicate pushdown ---
        logger.info("=== Stage 3: Join with dimension table ===")

        joined_df = (
            filtered_df
            .join(
                spark.table("customers").filter(col("is_active") == True),
                filtered_df["customer_id"] == col("id"),
                "inner",
            )
            .select("sale_id", "name", "tier", "amount", "state")
        )

        verifier.verify_query(
            joined_df,
            "join_with_filter",
            expected_optimizations=["Filter", "Join", "Project"],
        )

        joined_count = joined_df.count()
        logger.info("Joined result: %d rows", joined_count)

        # --- Stage 4: Aggregation ---
        logger.info("=== Stage 4: Aggregation ===")

        aggregated_df = (
            joined_df
            .groupBy("name", "tier", "state")
            .agg(
                spark_sum("amount").alias("total_revenue"),
                count("sale_id").alias("order_count"),
            )
            .filter(col("total_revenue") > 400)
        )

        verifier.verify_query(
            aggregated_df,
            "aggregation_with_filter",
            expected_optimizations=["Aggregate", "Filter", "Project"],
        )

        aggregated_df.show(truncate=False)

        # --- Stage 5: Final sort with limit ---
        logger.info("=== Stage 5: Final sort with limit ===")

        final_df = (
            aggregated_df
            .orderBy(col("total_revenue").desc())
            .limit(3)
        )

        verifier.verify_query(
            final_df,
            "sort_and_limit",
            expected_optimizations=["Sort", "GlobalLimit", "LocalLimit"],
        )

        final_df.show(truncate=False)

        # --- Print summary ---
        verifier.print_summary()

        # --- Show final optimized plan ---
        logger.info("=== Final Optimized Plan ===")
        logger.info(final_df.queryExecution.optimizedPlan.numberedTreeString)

        logger.info("=== Final Physical Plan ===")
        logger.info(final_df.queryExecution.executedPlan.numberedTreeString)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("ProductionAnalyticsPipeline")
        .config("spark.sql.planChangeLogLevel", "INFO")
        .getOrCreate()
    )

    try:
        run_analytics_pipeline(spark, "/tmp/analytics_pipeline_data")
    except Exception:
        sys.exit(1)
    finally:
        spark.stop()
```

### 7. Plan Change Logging Setup

```python
"""
Configure Spark to capture and log Catalyst rule applications during query
optimization. This setup enables production observability of optimizer
behavior.

Three logging modes are demonstrated:
1. Basic: Log only when the plan changes (INFO level)
2. Detailed: Log each rule application (DEBUG level)
3. Targeted: Log only specific batches or rules

Run: spark-submit --master local[*] plan_change_logging.py
"""
import io
import json
import logging
import re
import sys
from collections import defaultdict
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("plan_change_logging")


class CatalystPlanChangeCollector:
    """Collects and analyzes Catalyst plan change logs from Spark output.

    This class wraps a SparkSession and provides methods to:
    - Configure plan change logging at various levels
    - Capture rule applications from the JVM log output
    - Generate optimization reports per query
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.rule_applications: list[dict] = []
        self.query_plans: dict[str, dict] = {}

    def configure_logging(
        self,
        *,
        level: str = "INFO",
        batches: list[str] | None = None,
        rules: list[str] | None = None,
    ) -> None:
        """Configure Catalyst plan change logging.

        Args:
            level: Log level (OFF, DEBUG, INFO, WARN, ERROR).
                INFO logs before/after when plan changes.
                DEBUG logs every rule application including no-op.
            batches: If set, only log plan changes from these batch names.
                Example: ["Operator Optimization", "Join Reorder"].
            rules: If set, only log plan changes from these rule classes.
                Example: ["org.apache.spark.sql.catalyst.optimizer.PushDownPredicates"].
        """
        self.spark.conf.set("spark.sql.planChangeLogLevel", level)

        if batches:
            batch_str = ",".join(batches)
            self.spark.conf.set("spark.sql.planChangeBatches", batch_str)
            logger.info("Filtering plan change logs to batches: %s", batch_str)

        if rules:
            rule_str = ",".join(rules)
            self.spark.conf.set("spark.sql.planChangeRules", rule_str)
            logger.info("Filtering plan change logs to rules: %s", rule_str)

        # Also set the general SQL log level to capture the output
        if level != "OFF":
            self.spark.sparkContext.setLogLevel(level)

        logger.info("Plan change logging configured: level=%s", level)

    def analyze_query(self, df, query_name: str) -> dict:
        """Analyze a single query's optimization and capture plan details.

        Args:
            df: DataFrame to analyze.
            query_name: Human-readable label for this query.

        Returns:
            Dict with full optimization analysis.
        """
        try:
            qe = df.queryExecution

            analysis = {
                "query_name": query_name,
                "parsed_plan": qe.logical.numberedTreeString,
                "analyzed_plan": qe.analyzed.numberedTreeString,
                "optimized_plan": qe.optimizedPlan.numberedTreeString,
                "physical_plan": qe.executedPlan.numberedTreeString,
                "plan_transformation": {
                    "parsed_nodes": len(qe.logical.numberedTreeString.strip().split("\n")),
                    "analyzed_nodes": len(qe.analyzed.numberedTreeString.strip().split("\n")),
                    "optimized_nodes": len(qe.optimizedPlan.numberedTreeString.strip().split("\n")),
                    "physical_nodes": len(qe.executedPlan.numberedTreeString.strip().split("\n")),
                },
            }

            # Compute optimization metrics
            t = analysis["plan_transformation"]
            t["nodes_eliminated_by_optimizer"] = (
                t["analyzed_nodes"] - t["optimized_nodes"]
            )
            t["optimization_ratio"] = (
                round(
                    t["nodes_eliminated_by_optimizer"] / max(t["analyzed_nodes"], 1) * 100,
                    1,
                )
                if t["analyzed_nodes"] > 0
                else 0
            )

            self.query_plans[query_name] = analysis
            logger.info(
                "Query '%s': %d analyzed -> %d optimized nodes (%.1f%% reduction)",
                query_name,
                t["analyzed_nodes"],
                t["optimized_nodes"],
                t["optimization_ratio"],
            )

            return analysis

        except Exception as exc:
            logger.error("Failed to analyze query '%s': %s", query_name, exc)
            return {"query_name": query_name, "error": str(exc)}

    def generate_report(self) -> str:
        """Generate a JSON report of all captured plan changes.

        Returns:
            JSON string with full optimization report.
        """
        report = {
            "total_queries_analyzed": len(self.query_plans),
            "queries": {},
            "summary": {},
        }

        total_analyzed = 0
        total_optimized = 0
        queries_with_optimization = 0

        for name, analysis in self.query_plans.items():
            report["queries"][name] = {
                "plan_transformation": analysis.get("plan_transformation", {}),
            }

            t = analysis.get("plan_transformation", {})
            total_analyzed += t.get("analyzed_nodes", 0)
            total_optimized += t.get("optimized_nodes", 0)
            if t.get("nodes_eliminated_by_optimizer", 0) > 0:
                queries_with_optimization += 1

        report["summary"] = {
            "total_plan_nodes_analyzed": total_analyzed,
            "total_plan_nodes_optimized": total_optimized,
            "total_nodes_eliminated": total_analyzed - total_optimized,
            "queries_with_optimization": queries_with_optimization,
            "total_queries": len(self.query_plans),
        }

        return json.dumps(report, indent=2)


def main() -> None:
    """Demonstrate all three plan change logging modes."""
    spark = SparkSession.builder.appName("PlanChangeLogging").getOrCreate()

    try:
        # Setup test data
        spark.sql("CREATE OR REPLACE TEMPORARY VIEW t AS "
                  "SELECT * FROM VALUES "
                  "(1, 'A', 100, True), "
                  "(2, 'B', 200, False), "
                  "(3, 'A', 300, True), "
                  "(4, 'C', 0, True), "
                  "(5, 'B', 150, False) "
                  "AS t(id, category, value, active)")

        collector = CatalystPlanChangeCollector(spark)

        # --- Mode 1: Basic INFO-level logging ---
        logger.info("=== Mode 1: Basic plan change logging ===")
        collector.configure_logging(level="INFO")

        df1 = spark.sql("""
            SELECT category, SUM(value) AS total
            FROM t
            WHERE active = true AND value > 0
            GROUP BY category
            ORDER BY total DESC
        """)
        collector.analyze_query(df1, "basic_agg_query")

        # --- Mode 2: DEBUG-level (every rule) ---
        logger.info("=== Mode 2: Detailed DEBUG-level logging ===")
        collector.configure_logging(level="DEBUG")

        df2 = spark.sql("""
            SELECT t1.category, t2.category AS cat2
            FROM t t1
            JOIN t t2 ON t1.id = t2.id
            WHERE t1.value > 50 AND t2.active = true
        """)
        collector.analyze_query(df2, "self_join_query")

        # --- Mode 3: Targeted (specific rules only) ---
        logger.info("=== Mode 3: Targeted rule logging ===")
        collector.configure_logging(
            level="INFO",
            rules=[
                "org.apache.spark.sql.catalyst.optimizer.PushDownPredicates",
                "org.apache.spark.sql.catalyst.optimizer.ColumnPruning",
            ],
        )

        df3 = spark.sql("""
            SELECT category, COUNT(*) AS cnt, AVG(value) AS avg_val
            FROM t
            WHERE active = true
            GROUP BY category
            HAVING COUNT(*) > 1
        """)
        collector.analyze_query(df3, "targeted_having_query")

        # --- Reset and print report ---
        collector.configure_logging(level="OFF")
        spark.sparkContext.setLogLevel("WARN")

        logger.info("=== Full Optimization Report ===")
        print(collector.generate_report())

    except Exception as exc:
        logger.error("Plan change logging demo failed: %s", exc, exc_info=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
```

