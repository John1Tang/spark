# Cost-Based Optimizer (CBO): Join Reordering

## What It Is

The Cost-Based Optimizer (CBO) in Apache Spark is a statistics-driven optimization layer that reorders join operations in a multi-table query to minimize estimated execution cost. Unlike the rule-based Catalyst optimizer (which applies fixed transformations), CBO uses table and column statistics -- row counts, data sizes, null counts, distinct value counts (NDV), and histograms -- to estimate the selectivity of join predicates and choose the join order with the lowest total cost.

**Source references (Spark 4.2):**
- `CostBasedJoinReorder` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/CostBasedJoinReorder.scala`
- `SparkOptimizer` at `sql/core/src/main/scala/org/apache/spark/sql/execution/SparkOptimizer.scala`
- `JoinEstimation` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/statsEstimation/JoinEstimation.scala`
- `JoinReorderDP` at `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/CostBasedJoinReorder.scala` (line 143)
- SQL configurations in `sql/catalyst/src/main/scala/org/apache/spark/sql/internal/SQLConf.scala` (lines 3986-4045)

## Why It Works: The Mechanism

### The Selectivity Model

CBO estimates the number of rows produced by each join using **selectivity**. For an equi-join `A.key = B.key`, the estimated output cardinality is:

```
outputRows = (rowsA * rowsB) / max(NDV(A.key), NDV(B.key))
```

Where:
- `rowsA`, `rowsB` are the row counts of each input
- `NDV` is the number of distinct values in the join key

This formula assumes **uniform distribution** and **value inclusion** (the join keys from both sides share the same domain). In practice, Spark's `JoinEstimation` (in `JoinEstimation.scala`) handles several cases:

```scala
// JoinEstimation.scala (lines 55-80)
private def estimateInnerOuterJoin(): Option[Statistics] = join match {
  case _ if !rowCountsExist(join.left, join.right) => None  // Need stats

  case ExtractEquiJoinKeys(joinType, leftKeys, rightKeys, _, _, left, right, _) =>
    val joinKeyPairs = extractJoinKeysWithColStats(leftKeys, rightKeys)
    val leftSideUniqueness = left.distinctKeys.exists(_.subsetOf(ExpressionSet(leftKeys)))
    val rightSideUniqueness = right.distinctKeys.exists(_.subsetOf(ExpressionSet(rightKeys)))
    val (numInnerJoinedRows, keyStatsAfterJoin) =
      computeCardinalityAndStats(joinKeyPairs, leftSideUniqueness, rightSideUniqueness)
    // ...
}
```

**Key selectivity adjustments:**

| Scenario | Selectivity Adjustment |
|----------|----------------------|
| Left side has unique key (e.g., primary key) | `outputRows = rowsB` (each right row matches at most one left) |
| Right side has unique key | `outputRows = rowsA` |
| Both sides have unique keys | `outputRows = min(rowsA, rowsB)` |
| No stats available | Cannot estimate -- CBO skips |
| Non-equijoin | Uses default selectivity factor (1/10 per predicate) |

For non-equijoin conditions (range predicates, inequalities), Spark uses a default selectivity factor:

```scala
// Default selectivity: assumes 10% of rows match each non-equality predicate
val defaultSelectivity = 0.1
```

### Dynamic Programming Join Reorder Algorithm

The join reorder algorithm (`JoinReorderDP`, lines 143-380 in `CostBasedJoinReorder.scala`) is based on the Selinger system's dynamic programming approach ("Access Path Selection in a Relational Database Management System", System R, 1979).

**Algorithm overview:**

```
Level 0: {A}, {B}, {C}, {D}          -- single tables (cost = 0)
Level 1: {A,B}, {B,C}, {C,D}         -- best 2-way joins
Level 2: {A,B,C}, {B,C,D}            -- best 3-way joins
Level 3: {A,B,C,D}                   -- best 4-way join (FINAL)
```

At each level, the algorithm:
1. Combines plans from lower levels (one from level `k`, one from level `lev-k`)
2. For each combination, checks if there is a valid join condition between them
3. Computes the cost using `Cost(rowCount, sizeInBytes)`
4. Keeps only the **lowest-cost plan** for each set of items
5. Prunes cartesian products (no join condition = skipped)

```scala
// CostBasedJoinReorder.scala (lines 143-193)
object JoinReorderDP extends PredicateHelper with Logging {
  def search(conf: SQLConf, items: Seq[LogicalPlan],
             conditions: ExpressionSet, output: Seq[Attribute]): LogicalPlan = {

    // Level i maintains all found plans for i + 1 items
    val foundPlans = mutable.Buffer[JoinPlanMap]({
      val joinPlanMap = new JoinPlanMap
      itemIndex.foreach { case (item, id) =>
        joinPlanMap.put(Set(id), JoinPlan(Set(id), item, ExpressionSet(), Cost(0, 0)))
      }
      joinPlanMap
    })

    while (foundPlans.size < items.length) {
      foundPlans += searchLevel(foundPlans.toSeq, conf, conditions, topOutputSet, filters)
    }

    foundPlans.last.head._2.plan  // Best plan for all items
  }
}
```

**Cost comparison function** (lines 370-378):

```scala
def betterThan(other: JoinPlan, conf: SQLConf): Boolean = {
  val relativeRows = BigDecimal(this.planCost.card) / BigDecimal(other.planCost.card)
  val relativeSize = BigDecimal(this.planCost.size) / BigDecimal(other.planCost.size)
  Math.pow(relativeRows.doubleValue, conf.joinReorderCardWeight) *
    Math.pow(relativeSize.doubleValue, 1 - conf.joinReorderCardWeight) < 1
}
```

The cost is a **weighted geometric mean** of the row count ratio and size ratio, with default weight 0.7 for cardinality and 0.3 for size.

**Star-join filter:** When `spark.sql.cbo.joinReorder.dp.star.filter` is enabled, the algorithm detects star schemas (fact table + dimension tables) and plans star joins together before mixing with non-star tables, reducing the search space.

---

## What Problem It Solves

For a query joining N tables, there are N! possible join orderings (for left-deep trees). With 5 tables, that is 120 orderings; with 10 tables, 3.6 million.

Without CBO, Spark uses the join order as written in the query (rule-based). A bad join order can cause:

1. **Exploding intermediate results:** Joining two large tables first produces a massive intermediate result, even though joining one of them with a small filtered table first would have reduced the data dramatically.

2. **Suboptimal join strategy:** The physical planner chooses join strategies (BroadcastHashJoin, SortMergeJoin, ShuffleHashJoin) based on estimated sizes. Wrong size estimates lead to wrong strategy choices -- e.g., broadcasting a 50GB table instead of a 100MB one.

3. **Memory pressure and spilling:** Large intermediate results may exceed executor memory, causing disk spills that slow queries by 10-100x.

**Example:** Consider a query joining `customers` (10M rows), `orders` (100M rows), `order_items` (500M rows), and `products` (10K rows):

- **Bad order (left-to-right as written):**
  ```
  ((customers JOIN orders) JOIN order_items) JOIN products
  = ((10M x 100M -> ~500M intermediate) x 500M -> ~2B intermediate) x 10K -> ~20B
  ```

- **CBO-optimized order (small tables first):**
  ```
  customers JOIN (orders JOIN (order_items JOIN products))
  = 10K products filtered first, then 500M x 10K -> ~50M, etc.
  = Total intermediate: ~60M rows (300x less)
  ```

---

## Rule-Based vs Cost-Based Join Selection

| Aspect | Rule-Based (CBO disabled) | Cost-Based (CBO enabled) |
|--------|--------------------------|--------------------------|
| Join order | As written in the query | Optimized via dynamic programming |
| Statistics used | Only `sizeInBytes` from file metadata | Row counts, NDV, histograms, null counts |
| Join strategy | Based on file sizes | Based on estimated join output sizes |
| Broadcast threshold | `spark.sql.autoBroadcastJoinThreshold` vs file size | `autoBroadcastJoinThreshold` vs **estimated output** size |
| Source file | Catalyst optimizer batches | `CostBasedJoinReorder` in Join Reorder batch |
| Required setup | None | `ANALYZE TABLE` for accurate stats |

**Critical difference:** With CBO, Spark can reorder joins even when the user wrote them in a specific order. Without CBO, the user must manually write the optimal join order.

---

## How to Use CBO

### Step 1: Enable CBO

```scala
// Spark session configuration
spark.conf.set("spark.sql.cbo.enabled", "true")
spark.conf.set("spark.sql.cbo.planStats.enabled", "true")  // Fetch stats from catalog
spark.conf.set("spark.sql.cbo.joinReorder.enabled", "true")  // Enable join reordering
```

SQL:
```sql
SET spark.sql.cbo.enabled = true;
SET spark.sql.cbo.planStats.enabled = true;
SET spark.sql.cbo.joinReorder.enabled = true;
```

### Step 2: Collect Statistics

#### Basic Statistics (row count and file size)

```sql
-- For a single table
ANALYZE TABLE customers COMPUTE STATISTICS;

-- For a partitioned table (all partitions)
ANALYZE TABLE orders COMPUTE STATISTICS;

-- For a specific partition
ANALYZE TABLE orders PARTITION (order_date = '2024-01') COMPUTE STATISTICS;
```

This computes `rowCount` and `sizeInBytes` and stores them in the Hive metastore. For Parquet tables, Spark can also read these from file metadata without a full scan.

#### Column Statistics (for selectivity estimation)

```sql
-- Compute statistics for all columns (includes NDV, null count, avg length, max, min)
ANALYZE TABLE customers COMPUTE STATISTICS FOR COLUMNS;

-- For specific columns only
ANALYZE TABLE orders COMPUTE STATISTICS FOR COLUMNS customer_id, order_date, status;

-- For partitioned tables (all columns, all partitions)
ANALYZE TABLE orders COMPUTE STATISTICS FOR COLUMNS;
```

Column statistics include:
- **`distinctCount`** (NDV): Used for join selectivity estimation
- **`nullCount`**: Used for null-aware predicate estimation
- **`avgLen` / `maxLen`**: Used for size estimation
- **`min` / `max`**: Used for range predicate selectivity
- **`histogram`** (optional): Used for skewed data distribution

#### Programmatic Statistics Collection (Scala)

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("CBO Setup")
  .config("spark.sql.cbo.enabled", "true")
  .config("spark.sql.cbo.joinReorder.enabled", "true")
  .getOrCreate()

// Create tables
spark.sql("CREATE TABLE customers (id INT, name STRING, region STRING) USING parquet")
spark.sql("CREATE TABLE orders (id INT, customer_id INT, amount DOUBLE, status STRING) USING parquet")
spark.sql("CREATE TABLE order_items (order_id INT, product_id INT, quantity INT) USING parquet")
spark.sql("CREATE TABLE products (id INT, name STRING, price DOUBLE) USING parquet")

// Load data
customersDF.write.mode("overwrite").saveAsTable("customers")
ordersDF.write.mode("overwrite").saveAsTable("orders")
orderItemsDF.write.mode("overwrite").saveAsTable("order_items")
productsDF.write.mode("overwrite").saveAsTable("products")

// Collect statistics
spark.sql("ANALYZE TABLE customers COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE orders COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE order_items COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE products COMPUTE STATISTICS FOR COLUMNS")
```

#### PySpark

```python
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("CBO Setup")
    .config("spark.sql.cbo.enabled", "true")
    .config("spark.sql.cbo.joinReorder.enabled", "true")
    .getOrCreate())

# Analyze all tables
for table in ["customers", "orders", "order_items", "products"]:
    spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS FOR COLUMNS")
```

### Step 3: Verify CBO Is Working

#### Using EXPLAIN COST

```sql
EXPLAIN COST
SELECT c.name, SUM(oi.quantity * p.price) AS total
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE c.region = 'US'
GROUP BY c.name;
```

Output shows estimated statistics at each plan node:

```
== Optimized Logical Plan ==
Aggregate [name#1], [name#1, sum((quantity#10 * price#13)) AS total#15]
+- Project [name#1, quantity#10, price#13]
   +- SortMergeJoin [product_id#11], [id#12], Inner
      :- SortMergeJoin [id#0], [customer_id#7], Inner
      :  :- Sort [id#0 ASC]
      :  :  +- Filter (region#3 = US)
      :  :     +- Relation [id#0,name#1,region#3] customers
      :  :        Statistics: 2.5M rows, 150 MB, NDV(id)=2.5M
      :  +- Sort [customer_id#7 ASC]
      :     +- Filter (status#9 = COMPLETED)
      :        +- Relation [id#6,customer_id#7,status#9] orders
      :           Statistics: 50M rows, 2.1 GB, NDV(customer_id)=1.8M
      +- Sort [product_id#11 ASC]
         +- Relation [order_id#10,product_id#11,quantity#12] order_items
            Statistics: 200M rows, 8.5 GB, NDV(product_id)=50K
+- Sort [id#12 ASC]
   +- Relation [id#12,name#13,price#14] products
      Statistics: 10K rows, 500 KB, NDV(id)=10K
```

Note the **Statistics** lines showing row counts, sizes, and distinct value counts used by CBO.

#### Using EXPLAIN EXTENDED to See Join Reorder

```sql
EXPLAIN EXTENDED
SELECT c.name, o.amount, p.name
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN products p ON o.product_id = p.id;
```

**Without CBO** (join order as written: customers -> orders -> products):
```
== Optimized Logical Plan ==
Project [name#1, amount#8, name#13]
+- Join Inner, (product_id#9 = id#12)
   +- Join Inner, (id#0 = customer_id#7)
   :  :- Relation customers
   :  +- Relation orders
   +- Relation products
```

**With CBO enabled** (reordered: products first (smallest), then orders, then customers):
```
== Optimized Logical Plan ==
Project [name#1, amount#8, name#13]
+- Join Inner, (id#0 = customer_id#7)
   :- Relation products         <-- Small table moved to left
   :  Statistics: 10K rows
   +- Join Inner, (product_id#9 = id#12)
      :- Relation orders        <-- Medium table in middle
      :  Statistics: 50M rows
      +- Relation customers     <-- Large table last (but filtered)
         Statistics: 2.5M rows
```

#### Checking Statistics in the Catalog

```sql
-- Show table statistics
DESCRIBE EXTENDED customers;
-- Look for: Statistics: 2500000 bytes, 100000 rows

-- Show column statistics
SHOW TBLPROPERTIES customers;

-- Query the metastore directly (Hive)
SELECT * FROM hive.TAB_COL_STATS WHERE TBL_NAME = 'customers';
SELECT * FROM hive.TAB_COL_STATS WHERE TBL_NAME = 'orders' AND COL_NAME = 'customer_id';
```

#### Programmatic Statistics Check (Scala)

```scala
// Check if stats exist for a table
val stats = spark.sessionState.catalog.getTableStats(
  org.apache.spark.sql.catalyst.TableIdentifier("customers"))
println(s"Row count: ${stats.rowCount}")
println(s"Size in bytes: ${stats.sizeInBytes}")

// Check column statistics
val colStats = spark.sessionState.catalog.getColumnStats(
  org.apache.spark.sql.catalyst.TableIdentifier("orders"), "customer_id")
colStats.foreach { case (colName, colStat) =>
  println(s"Column: $colName")
  println(s"  NDV: ${colStat.distinctCount}")
  println(s"  Nulls: ${colStat.nullCount}")
  println(s"  Min: ${colStat.min}")
  println(s"  Max: ${colStat.max}")
}
```

---

## Key Configurations

### CBO Core Settings

| Configuration | Default | Recommended | Description |
|--------------|---------|-------------|-------------|
| `spark.sql.cbo.enabled` | `false` | `true` | Master switch for CBO. Enables plan stats and join cost estimation. |
| `spark.sql.cbo.planStats.enabled` | `false` | `true` | When true, the logical plan fetches row counts and column statistics from the catalog. |
| `spark.sql.cbo.joinReorder.enabled` | `false` | `true` | Enables join reordering via dynamic programming. |

### Join Reorder Tuning

| Configuration | Default | Recommended | Description |
|--------------|---------|-------------|-------------|
| `spark.sql.cbo.joinReorder.dp.threshold` | `12` | `12` (default) | Maximum number of joined tables for DP algorithm. Higher values increase planning time exponentially (O(3^n)). |
| `spark.sql.cbo.joinReorder.card.weight` | `0.7` | `0.7` (default) | Weight of cardinality vs size in cost function. Range [0, 1]. Higher = more emphasis on row count. |
| `spark.sql.cbo.joinReorder.dp.star.filter` | `false` | `true` for star schemas | Enables star-join filter heuristics to reduce search space for star schemas. |
| `spark.sql.cbo.starSchemaDetection` | `false` | `true` for star schemas | Enables automatic detection of star schema relationships. |
| `spark.sql.autoBroadcastJoinThreshold` | `10485760` (10MB) | `10485760` or higher | Tables below this size are broadcast. With CBO, this compares against **estimated** size, not file size. |

### Applying All CBO Settings (production-ready)

```scala
val spark = SparkSession.builder()
  .appName("CBO Production")
  .config("spark.sql.cbo.enabled", "true")
  .config("spark.sql.cbo.planStats.enabled", "true")
  .config("spark.sql.cbo.joinReorder.enabled", "true")
  .config("spark.sql.cbo.joinReorder.dp.threshold", "12")
  .config("spark.sql.cbo.joinReorder.card.weight", "0.7")
  .config("spark.sql.cbo.joinReorder.dp.star.filter", "true")
  .config("spark.sql.cbo.starSchemaDetection", "true")
  .config("spark.sql.autoBroadcastJoinThreshold", "100MB")  // 100MB threshold
  .getOrCreate()
```

---

## Complete Production Example: CBO in Action

### Scenario: E-commerce Analytics Query

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("CBO Demo")
  .config("spark.sql.cbo.enabled", "true")
  .config("spark.sql.cbo.planStats.enabled", "true")
  .config("spark.sql.cbo.joinReorder.enabled", "true")
  .config("spark.sql.cbo.joinReorder.dp.star.filter", "true")
  .config("spark.sql.autoBroadcastJoinThreshold", "104857600") // 100MB
  .getOrCreate()

import spark.implicits._

// ===== Schema =====
case class Product(id: Int, name: String, category: String, price: Double)
case class Customer(id: Int, name: String, region: String, tier: String)
case class Order(id: Int, customerId: Int, date: String, status: String)
case class OrderItem(orderId: Int, productId: Int, quantity: Int)

// ===== Create and populate tables =====
// (In production, these would be loaded from data lakes)
val products = (1 to 10000).map(i => Product(i, s"Product-$i", s"Cat-${i % 100}", i * 1.5))
val customers = (1 to 2500000).map(i => Customer(i, s"Cust-$i", s"Region-${i % 10}", Seq("Gold","Silver","Bronze")(i % 3)))
val orders = (1 to 50000000).map(i => Order(i, (i % 2500000) + 1, s"2024-${(i % 12) + 1:02d}", Seq("COMPLETED","PENDING","CANCELLED")(i % 3)))
val orderItems = (1 to 200000000).map(i => OrderItem(i / 4 + 1, (i % 10000) + 1, (i % 10) + 1))

Seq(products: _*).toDF().write.mode("overwrite").saveAsTable("products")
Seq(customers: _*).toDF().write.mode("overwrite").saveAsTable("customers")
Seq(orders: _*).toDF().write.mode("overwrite").saveAsTable("orders")
Seq(orderItems: _*).toDF().write.mode("overwrite").saveAsTable("order_items")

// ===== Collect statistics =====
spark.sql("ANALYZE TABLE products COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE customers COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE orders COMPUTE STATISTICS FOR COLUMNS")
spark.sql("ANALYZE TABLE order_items COMPUTE STATISTICS FOR COLUMNS")

// ===== Run the analytics query =====
val query = """
  SELECT c.name, c.region, p.name AS product_name,
         SUM(oi.quantity * p.price) AS total_spent
  FROM customers c
  JOIN orders o ON c.id = o.customer_id
  JOIN order_items oi ON o.id = oi.order_id
  JOIN products p ON oi.product_id = p.id
  WHERE c.region = 'Region-1'
    AND o.status = 'COMPLETED'
    AND o.date BETWEEN '2024-01-01' AND '2024-12-31'
  GROUP BY c.name, c.region, p.name
  ORDER BY total_spent DESC
  LIMIT 100
"""

// Show the optimized plan
spark.sql(s"EXPLAIN EXTENDED $query").show(false)

// Execute
val result = spark.sql(query)
result.show()
```

### Expected Plan with CBO

```
== Optimized Logical Plan ==
GlobalLimit 101
+- LocalLimit 101
   +- Sort [total_spent#20 DESC]
      +- Project [name#1, region#3, name#13 AS product_name#21,
                   sum((quantity#10 * price#14)) AS total_spent#20]
         +- Join Inner, (id#12 = product_id#11)
            :- Join Inner, (id#0 = customer_id#7)
            :  :- Join Inner, (id#6 = order_id#10)
            :  :  :- Filter (id#12 IN (1-10000))
            :  :  :  +- Relation products
            :  :  :     Statistics: 10K rows, 500 KB
            :  :  +- Filter ((status#9 = COMPLETED) AND (date#8 BETWEEN ...))
            :  :     +- Relation orders
            :  :        Statistics: 50M rows, 2.1 GB
            :  +- Filter ((region#3 = Region-1) AND (id#0 IN (subset)))
            :     +- Relation customers
            :        Statistics: 2.5M rows, 150 MB
            +- Relation order_items
               Statistics: 200M rows, 8.5 GB
```

Note how CBO reordered the joins to process the smallest table (`products`, 10K rows) first, then progressively larger tables.

---

## When CBO Is NOT Effective

### 1. Stale Statistics

If data has changed significantly since `ANALYZE TABLE` was last run, CBO's estimates are wrong. This can lead to worse plans than rule-based.

```sql
-- Check when stats were last updated (Hive metastore)
SELECT LAST_ANALYZED FROM hive.TABLE_PARAMS
WHERE TBL_NAME = 'orders' AND PARAM_KEY = 'transient_lastDdlTime';

-- Re-analyze if data has changed > 10%
ANALYZE TABLE orders COMPUTE STATISTICS FOR COLUMNS;
```

**Mitigation:** Schedule regular `ANALYZE TABLE` jobs as part of ETL pipelines. For tables that change frequently, consider running `ANALYZE TABLE` after each data load.

### 2. No Statistics

If `ANALYZE TABLE` has never been run, CBO falls back to file-size-only estimates (which do not include row counts). Join reorder will be skipped entirely if row counts are missing:

```scala
// From CostBasedJoinReorder.scala line 64:
if (items.size > 2 && items.size <= conf.joinReorderDPThreshold && conditions.nonEmpty &&
    items.forall(_.stats.rowCount.isDefined)) {
  JoinReorderDP.search(conf, items, conditions, output)
} else {
  plan  // Skip reorder if any item lacks row count
}
```

### 3. UDFs in Join Conditions

Join conditions that contain UDFs prevent accurate selectivity estimation because the optimizer cannot analyze the UDF's filtering behavior:

```sql
-- CBO cannot estimate this:
SELECT * FROM a JOIN b ON my_udf(a.key) = b.key

-- CBO works fine with this:
SELECT * FROM a JOIN b ON a.key = b.key WHERE my_udf(a.key) > 0
```

### 4. Fewer Than 3 Tables

Join reorder is only triggered when there are more than 2 tables to join (`items.size > 2`). A 2-table join is not reordered.

### 5. Join Hints Override CBO

If the query uses join hints (`BROADCAST`, `MERGE`, `SHUFFLE_HASH`, `SHUFFLE_REPLICATE_NL`), CBO respects the hint and does not reorder:

```scala
// From CostBasedJoinReorder.scala line 45:
case j @ Join(_, _, _: InnerLike, Some(cond), JoinHint.NONE) =>
  reorder(j, j.output)
// JoinHint.NONE is required -- hints skip reordering
```

```sql
-- This prevents CBO from reordering:
SELECT /*+ BROADCAST(products) */ *
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN products p ON o.product_id = p.id;
```

### 6. More Than `joinReorderDPThreshold` Tables

By default, CBO only reorders up to 12 tables. Beyond that, the DP algorithm's exponential complexity makes planning too slow. The query uses the original join order.

### 7. Non-Inner Joins

CBO currently only reorders inner-like joins (`Inner`, `Cross`). Left semi and left anti joins are NOT reordered. Outer joins are excluded from the DP algorithm.

### 8. Skewed Data with Default Statistics

CBO assumes uniform distribution. If a join key is heavily skewed (e.g., 90% of rows have the same key value), the selectivity formula `rowsA * rowsB / max(NDV)` can be off by orders of magnitude. Histograms help but are not always collected.

**Mitigation:** Use `ANALYZE TABLE ... COMPUTE STATISTICS FOR COLUMNS` to collect histograms for skewed columns. Alternatively, use `spark.sql.adaptive.skewJoin.enabled` for runtime skew handling.

---

## Comparison: Plan Without CBO vs With CBO

### Setup

```sql
-- Four-table join: customers (2.5M) -> orders (50M) -> order_items (200M) -> products (10K)

-- Without CBO
SET spark.sql.cbo.enabled = false;
SET spark.sql.cbo.joinReorder.enabled = false;

-- Query
SELECT c.name, p.name AS product, SUM(oi.quantity * p.price) AS total
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE c.region = 'US' AND o.status = 'COMPLETED'
GROUP BY c.name, p.name;
```

### Without CBO (left-to-right as written)

```
== Optimized Logical Plan ==
Aggregate [name, product_name], [name, product_name, sum(...)]
+- Join [product_id = id]
   +- Join [id = order_id]
   :  +- Join [id = customer_id]
   :  :  +- Filter(region = 'US')
   :  :  |  +- Scan customers        -- 250K rows after filter
   :  :  +- Scan orders              -- 50M rows (FULL scan)
   :  +- Scan order_items            -- 200M rows (FULL scan)
   +- Scan products                  -- 10K rows

Estimated intermediate:
  customers(250K) JOIN orders(50M) -> ~12.5M rows
  12.5M JOIN order_items(200M) -> ~250M rows
  250M JOIN products(10K) -> ~25M rows
```

### With CBO (optimal order)

```sql
SET spark.sql.cbo.enabled = true;
SET spark.sql.cbo.joinReorder.enabled = true;
SET spark.sql.cbo.planStats.enabled = true;
```

```
== Optimized Logical Plan ==
Aggregate [name, product_name], [name, product_name, sum(...)]
+- Join [id = customer_id]          -- Large join LAST
   +- Join [product_id = id]        -- Medium join
   :  +- Join [id = order_id]       -- Smallest join FIRST
   :  :  +- Scan products          -- 10K rows (smallest, starts here)
   :  :  +- Scan order_items       -- 200M rows, but join key reduces to ~2M
   :  +- Filter(status = 'COMPLETED')
   :     +- Scan orders            -- ~16.7M after filter
   +- Filter(region = 'US')
      +- Scan customers            -- ~250K after filter

Estimated intermediate:
  products(10K) JOIN order_items(200M) -> ~2M rows (NDV(product_id)=10K)
  2M JOIN filtered_orders(16.7M) -> ~5M rows
  5M JOIN filtered_customers(250K) -> ~1M rows
```

**Result:** CBO reduces estimated intermediate data from ~287.5M rows to ~7M rows -- a **40x reduction**.

---

## Limitations and Gotchas

### 1. CBO Is Off by Default

`spark.sql.cbo.enabled` defaults to `false`. Many production clusters run without CBO, missing significant optimization opportunities. Enable it explicitly.

### 2. Statistics Must Be Manually Maintained

Unlike some databases (e.g., PostgreSQL, Oracle), Spark does not auto-analyze tables. You must run `ANALYZE TABLE` as part of your data pipeline. Stale stats are worse than no stats -- CBO will confidently choose a bad plan.

### 3. Partition Statistics Are Per-Partition

For partitioned tables, `ANALYZE TABLE ... COMPUTE STATISTICS` without a partition clause analyzes all partitions. If you add new partitions frequently, either:
- Analyze the specific new partition: `ANALYZE TABLE t PARTITION (dt='2024-01') COMPUTE STATISTICS`
- Or re-analyze the entire table periodically

### 4. File Metadata Stats vs Catalog Stats

For Parquet/ORC files, Spark can read basic stats (row count, size) from file footers. However, column-level stats (NDV, histograms) require `ANALYZE TABLE ... FOR COLUMNS` and are stored only in the Hive metastore. Without column stats, join selectivity uses default values.

### 5. Dynamic Programming Complexity

The DP algorithm has O(3^n) complexity for n tables. At the default threshold of 12, planning can take several seconds for complex queries. If your queries have many more than 12 joins, consider:
- Breaking the query into CTEs
- Using join hints to guide the planner
- Increasing `spark.sql.cbo.joinReorder.dp.threshold` (with caution)

### 6. CBO Does Not Handle Subquery Reordering

CBO only reorders joins in the main query. Subquery placement is handled by other Catalyst rules (e.g., `DecorrelateInnerQuery`).

### 7. AQE and CBO Complement Each Other

Adaptive Query Execution (AQE) and CBO are complementary:
- **CBO** optimizes the logical plan before execution (join order, broadcast decisions)
- **AQE** optimizes the physical plan during execution (join strategy switches, skew handling, coalescing partitions)

For best results, enable both:
```scala
spark.conf.set("spark.sql.cbo.enabled", "true")
spark.conf.set("spark.sql.cbo.joinReorder.enabled", "true")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### 8. EXPLAIN COST Shows Estimates, Not Actuals

The statistics shown in `EXPLAIN COST` are estimates. To see actual row counts, check the Spark UI after execution:
- **SQL tab**: Look at the "Metrics" column for each stage
- **Stages tab**: Check "Input Size" and "Output Size" for shuffle operations
- **Event log**: Use `org.apache.spark.sql.execution.SparkPlanInfo` to get actual metrics

```scala
// After query execution, compare estimated vs actual
val planInfo = spark.sql(query).queryExecution.executedPlan
println(planInfo.treeString(verbose = true))
```

### 9. CBO Can Interact Poorly with Join Hints

If you use `/*+ BROADCAST(table) */` hint but CBO estimates the table is too large to broadcast, the hint wins. However, the estimated sizes used for other join decisions may still be wrong, leading to suboptimal plans for the non-hinted joins.

### 10. Row Count Estimation for Complex Joins

For non-equi joins, CBO uses a default selectivity of 0.1 per predicate. For joins with multiple non-equality conditions, the estimate can be significantly off. Equi-joins with column statistics produce the most accurate estimates.

---

## Production-Ready Python (PySpark) Code

### Session Factory with CBO Configuration

```python
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


def create_cbo_session(app_name: str = "CBOProduction") -> SparkSession:
    """Create a SparkSession with all CBO-related configs enabled.

    Returns a session with CBO, join reorder, statistics correlation,
    histogram support, and star schema detection all enabled.

    Args:
        app_name: The Spark application name.

    Returns:
        Configured SparkSession instance.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.cbo.enabled", "true")
        .config("spark.sql.cbo.planStats.enabled", "true")
        .config("spark.sql.cbo.joinReorder.enabled", "true")
        .config("spark.sql.cbo.joinReorder.dp.threshold", "12")
        .config("spark.sql.cbo.joinReorder.card.weight", "0.7")
        .config("spark.sql.cbo.joinReorder.dp.star.filter", "true")
        .config("spark.sql.cbo.starSchemaDetection", "true")
        .config("spark.sql.statistics.colStatsCorrelation", "true")
        .config("spark.sql.statistics.histogram.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600")  # 100 MB
        .getOrCreate()
    )
```

### Statistics Collection

```python
import logging

logger = logging.getLogger(__name__)


def collect_table_statistics(
    spark: SparkSession,
    database: str,
    table: str,
    columns: bool = True,
) -> dict:
    """Run ANALYZE TABLE COMPUTE STATISTICS and verify the results.

    Args:
        spark: Active SparkSession.
        database: Target database name.
        table: Target table name.
        columns: If True, compute column-level statistics (FOR COLUMNS).

    Returns:
        Dict with table name, command issued, and verification status.
    """
    fqtn = f"{database}.{table}" if database else table
    stmt = (
        f"ANALYZE TABLE {fqtn} COMPUTE STATISTICS FOR COLUMNS"
        if columns
        else f"ANALYZE TABLE {fqtn} COMPUTE STATISTICS"
    )

    try:
        logger.info("Collecting statistics for %s", fqtn)
        spark.sql(stmt)

        # Verify by reading extended table description
        desc_df = spark.sql(f"DESCRIBE EXTENDED {fqtn}")
        stats_row = desc_df.filter(
            desc_df["col_name"].rlike("(?i)statistics")
        ).collect()
        stats_info = (
            stats_row[0]["data_type"] if stats_row else "not available"
        )

        logger.info("Statistics for %s: %s", fqtn, stats_info)
        return {
            "table": fqtn,
            "command": stmt,
            "stats_info": stats_info,
            "verified": stats_info != "not available",
        }
    except Exception as exc:
        logger.error("Failed to collect statistics for %s: %s", fqtn, exc)
        raise


def collect_column_statistics(
    spark: SparkSession,
    database: str,
    table: str,
    columns: list[str],
) -> dict:
    """Collect statistics for specific columns only.

    Args:
        spark: Active SparkSession.
        database: Target database name.
        table: Target table name.
        columns: List of column names to analyze.

    Returns:
        Dict with table name, columns analyzed, and command issued.
    """
    fqtn = f"{database}.{table}" if database else table
    col_list = ", ".join(columns)
    stmt = (
        f"ANALYZE TABLE {fqtn} COMPUTE STATISTICS FOR COLUMNS {col_list}"
    )

    try:
        logger.info("Collecting column statistics for %s (%s)", fqtn, col_list)
        spark.sql(stmt)
        return {
            "table": fqtn,
            "columns": columns,
            "command": stmt,
            "status": "success",
        }
    except Exception as exc:
        logger.error(
            "Failed to collect column statistics for %s: %s", fqtn, exc
        )
        raise
```

### CBO Verification Utilities

```python
import re
from dataclasses import dataclass, field


@dataclass
class CBOVerificationResult:
    """Result of verifying whether CBO is active for a query."""
    cbo_active: bool
    join_type: str = ""
    join_order_tables: list[str] = field(default_factory=list)
    explain_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify_cbo_active(spark: SparkSession, sql_query: str) -> CBOVerificationResult:
    """Parse EXPLAIN output to determine whether CBO influenced the plan.

    Looks for SortMergeJoin vs. BroadcastHashJoin patterns, statistics
    annotations, and join ordering evidence in the extended plan.

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to analyze.

    Returns:
        CBOVerificationResult with join type, detected tables, and notes.
    """
    try:
        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
        full_plan = "\n".join(lines)
    except Exception as exc:
        logger.error("Failed to get EXPLAIN output: %s", exc)
        return CBOVerificationResult(
            cbo_active=False, notes=[f"EXPLAIN failed: {exc}"]
        )

    # Detect join strategies present in the plan
    smj = re.findall(r"SortMergeJoin", full_plan)
    bhj = re.findall(r"BroadcastHashJoin", full_plan)
    shj = re.findall(r"ShuffleHashJoin", full_plan)

    join_type = "SortMergeJoin" if smj else ("BroadcastHashJoin" if bhj else "Unknown")

    # Detect Statistics lines (CBO evidence)
    stats_lines = [
        line.strip()
        for line in lines
        if re.search(r"Statistics:", line, re.IGNORECASE)
    ]

    # Detect Relation entries to infer join order
    relations = re.findall(
        r"Relation\s+\[.*?\]\s+(\w+)", full_plan
    )

    notes: list[str] = []
    if stats_lines:
        notes.append(f"CBO statistics found in {len(stats_lines)} plan node(s)")
    if bhj:
        notes.append(f"CBO chose BroadcastHashJoin ({len(bhj)} occurrence(s))")
    if smj:
        notes.append(f"Plan uses SortMergeJoin ({len(smj)} occurrence(s))")

    return CBOVerificationResult(
        cbo_active=len(stats_lines) > 0 or len(bhj) > 0,
        join_type=join_type,
        join_order_tables=relations,
        explain_lines=lines,
        notes=notes,
    )
```

### CBO Impact Comparison

```python
from dataclasses import dataclass, field


@dataclass
class CBOComparisonResult:
    """Comparison of a query plan with CBO ON vs. OFF."""
    cbo_enabled_join_type: str = ""
    cbo_disabled_join_type: str = ""
    cbo_enabled_tables: list[str] = field(default_factory=list)
    cbo_disabled_tables: list[str] = field(default_factory=list)
    join_order_changed: bool = False
    cbo_enabled_plan: list[str] = field(default_factory=list)
    cbo_disabled_plan: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def compare_cbo_impact(
    spark: SparkSession,
    sql_query: str,
) -> CBOComparisonResult:
    """Run a query with CBO ON and OFF, comparing the resulting plans.

    Args:
        spark: Active SparkSession.
        sql_query: The SQL query to compare.

    Returns:
        CBOComparisonResult with plan differences.
    """
    original_cbo = spark.conf.get("spark.sql.cbo.enabled", "false")
    original_plan_stats = spark.conf.get("spark.sql.cbo.planStats.enabled", "false")
    original_reorder = spark.conf.get("spark.sql.cbo.joinReorder.enabled", "false")

    def _get_explain(cbo_on: bool) -> tuple[list[str], str, list[str]]:
        spark.conf.set("spark.sql.cbo.enabled", str(cbo_on).lower())
        spark.conf.set("spark.sql.cbo.planStats.enabled", str(cbo_on).lower())
        spark.conf.set("spark.sql.cbo.joinReorder.enabled", str(cbo_on).lower())

        explain_df = spark.sql(f"EXPLAIN EXTENDED {sql_query}")
        lines = [row["plan"] for row in explain_df.collect()]
        full_plan = "\n".join(lines)

        bhj = re.findall(r"BroadcastHashJoin", full_plan)
        smj = re.findall(r"SortMergeJoin", full_plan)
        join_type = "BroadcastHashJoin" if bhj else ("SortMergeJoin" if smj else "Unknown")
        relations = re.findall(r"Relation\s+\[.*?\]\s+(\w+)", full_plan)
        return lines, join_type, relations

    try:
        cbo_lines, cbo_join, cbo_tables = _get_explain(cbo_on=True)
        nocbo_lines, nocbo_join, nocbo_tables = _get_explain(cbo_on=False)
    except Exception as exc:
        logger.error("CBO comparison failed: %s", exc)
        return CBOComparisonResult(notes=[f"Error: {exc}"])
    finally:
        # Restore original settings
        spark.conf.set("spark.sql.cbo.enabled", original_cbo)
        spark.conf.set("spark.sql.cbo.planStats.enabled", original_plan_stats)
        spark.conf.set("spark.sql.cbo.joinReorder.enabled", original_reorder)

    join_order_changed = cbo_tables != nocbo_tables
    notes: list[str] = []
    if join_order_changed:
        notes.append(
            f"Join order changed: CBO={cbo_tables} vs. no-CBO={nocbo_tables}"
        )
    if cbo_join != nocbo_join:
        notes.append(
            f"Join strategy changed: CBO={cbo_join} vs. no-CBO={nocbo_join}"
        )

    return CBOComparisonResult(
        cbo_enabled_join_type=cbo_join,
        cbo_disabled_join_type=nocbo_join,
        cbo_enabled_tables=cbo_tables,
        cbo_disabled_tables=nocbo_tables,
        join_order_changed=join_order_changed,
        cbo_enabled_plan=cbo_lines,
        cbo_disabled_plan=nocbo_lines,
        notes=notes,
    )
```

### Statistics Collector Class

```python
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TableStats:
    """Statistics for a single table."""
    table_name: str
    row_count: int = -1
    size_bytes: int = -1
    column_stats: dict = field(default_factory=dict)
    analyzed_at: str = ""
    status: str = ""


@dataclass
class CBOStatisticsReport:
    """JSON-serializable report for CBO statistics collection."""
    database: str
    tables: list[TableStats] = field(default_factory=list)
    collected_at: str = ""
    total_tables: int = 0
    total_rows: int = 0
    total_size_bytes: int = 0


class CBOStatisticsCollector:
    """Collect statistics for multiple tables and generate a JSON report.

    Usage:
        collector = CBOStatisticsCollector(spark, "my_database")
        collector.add_table("customers")
        collector.add_table("orders", columns=["customer_id", "status", "order_date"])
        report = collector.collect_all()
        collector.save_report("/tmp/cbo_stats_report.json")
    """

    def __init__(self, spark: SparkSession, database: str):
        self.spark = spark
        self.database = database
        self._tables: list[tuple[str, list[str] | None]] = []

    def add_table(self, table: str, columns: list[str] | None = None) -> None:
        """Register a table for statistics collection.

        Args:
            table: Table name (without database prefix).
            columns: If provided, collect stats for these columns only.
        """
        self._tables.append((table, columns))

    def collect_all(self) -> CBOStatisticsReport:
        """Collect statistics for all registered tables.

        Returns:
            CBOStatisticsReport with results for each table.
        """
        report = CBOStatisticsReport(
            database=self.database,
            collected_at=datetime.utcnow().isoformat(),
        )

        for table, columns in self._tables:
            fqtn = f"{self.database}.{table}"
            ts = TableStats(table_name=fqtn)

            try:
                # Run ANALYZE
                if columns:
                    col_list = ", ".join(columns)
                    self.spark.sql(
                        f"ANALYZE TABLE {fqtn} COMPUTE STATISTICS FOR COLUMNS {col_list}"
                    )
                else:
                    self.spark.sql(
                        f"ANALYZE TABLE {fqtn} COMPUTE STATISTICS FOR COLUMNS"
                    )

                # Read stats from DESCRIBE EXTENDED
                desc_df = self.spark.sql(f"DESCRIBE EXTENDED {fqtn}")
                rows = desc_df.collect()
                for row in rows:
                    col_name = row["col_name"]
                    if re.search(r"(?i)numRows", col_name):
                        try:
                            ts.row_count = int(row["data_type"])
                        except (ValueError, TypeError):
                            pass
                    elif re.search(r"(?i)totalSize|sizeInBytes", col_name):
                        try:
                            ts.size_bytes = int(row["data_type"])
                        except (ValueError, TypeError):
                            pass

                ts.analyzed_at = datetime.utcnow().isoformat()
                ts.status = "success"
                logger.info(
                    "Collected stats for %s: %d rows, %d bytes",
                    fqtn, ts.row_count, ts.size_bytes,
                )

            except Exception as exc:
                ts.status = f"error: {exc}"
                logger.error("Failed to collect stats for %s: %s", fqtn, exc)

            report.tables.append(ts)

        report.total_tables = len(report.tables)
        report.total_rows = sum(
            t.row_count for t in report.tables if t.row_count > 0
        )
        report.total_size_bytes = sum(
            t.size_bytes for t in report.tables if t.size_bytes > 0
        )
        return report

    def save_report(self, path: str = "/tmp/cbo_statistics_report.json") -> str:
        """Collect stats and write a JSON report to disk.

        Args:
            path: Output file path.

        Returns:
            The path the report was written to.
        """
        report = self.collect_all()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info("CBO statistics report written to %s", path)
        return path
```

### CBO Benchmark

```python
import time
from dataclasses import dataclass, field


@dataclass
class CBOBenchmarkQuery:
    """Result for a single benchmarked query."""
    query_name: str
    sql: str
    cbo_enabled_ms: float = 0.0
    cbo_disabled_ms: float = 0.0
    speedup_ratio: float = 1.0
    cbo_join_type: str = ""
    nocbo_join_type: str = ""
    error: str = ""


@dataclass
class CBOBenchmarkReport:
    """Aggregate benchmark results."""
    queries: list[CBOBenchmarkQuery] = field(default_factory=list)
    avg_speedup: float = 0.0
    total_cbo_ms: float = 0.0
    total_nocbo_ms: float = 0.0
    timestamp: str = ""


def run_cbo_benchmark(
    spark: SparkSession,
    queries: dict[str, str],
    output_path: str = "/tmp/cbo_benchmark.json",
) -> CBOBenchmarkReport:
    """Benchmark star-schema queries with CBO ON vs. OFF.

    For each query, executes it twice (CBO enabled/disabled) and records
    wall-clock time plus join strategy differences.

    Args:
        spark: Active SparkSession.
        queries: Dict mapping query name to SQL string.
        output_path: Where to write the JSON report.

    Returns:
        CBOBenchmarkReport with per-query and aggregate results.
    """
    report = CBOBenchmarkReport(timestamp=datetime.utcnow().isoformat())
    results: list[CBOBenchmarkQuery] = []

    original_cbo = spark.conf.get("spark.sql.cbo.enabled", "false")
    original_plan_stats = spark.conf.get("spark.sql.cbo.planStats.enabled", "false")
    original_reorder = spark.conf.get("spark.sql.cbo.joinReorder.enabled", "false")

    for name, sql in queries.items():
        bq = CBOBenchmarkQuery(query_name=name, sql=sql)

        for cbo_on in (True, False):
            spark.conf.set("spark.sql.cbo.enabled", str(cbo_on).lower())
            spark.conf.set("spark.sql.cbo.planStats.enabled", str(cbo_on).lower())
            spark.conf.set("spark.sql.cbo.joinReorder.enabled", str(cbo_on).lower())

            try:
                start = time.perf_counter()
                df = spark.sql(sql)
                df.count()  # force materialization
                elapsed_ms = (time.perf_counter() - start) * 1000

                if cbo_on:
                    bq.cbo_enabled_ms = elapsed_ms
                    # Capture join type
                    explain_df = spark.sql(f"EXPLAIN EXTENDED {sql}")
                    plan = "\n".join(r["plan"] for r in explain_df.collect())
                    bq.cbo_join_type = (
                        "BroadcastHashJoin" if "BroadcastHashJoin" in plan
                        else "SortMergeJoin" if "SortMergeJoin" in plan
                        else "Unknown"
                    )
                else:
                    bq.cbo_disabled_ms = elapsed_ms
                    explain_df = spark.sql(f"EXPLAIN EXTENDED {sql}")
                    plan = "\n".join(r["plan"] for r in explain_df.collect())
                    bq.nocbo_join_type = (
                        "BroadcastHashJoin" if "BroadcastHashJoin" in plan
                        else "SortMergeJoin" if "SortMergeJoin" in plan
                        else "Unknown"
                    )
            except Exception as exc:
                bq.error = str(exc)
                logger.error("Benchmark query '%s' (CBO=%s) failed: %s", name, cbo_on, exc)

        # Compute speedup ratio
        if bq.cbo_disabled_ms > 0 and bq.cbo_enabled_ms > 0:
            bq.speedup_ratio = round(
                bq.cbo_disabled_ms / bq.cbo_enabled_ms, 2
            )

        results.append(bq)

    # Restore original settings
    spark.conf.set("spark.sql.cbo.enabled", original_cbo)
    spark.conf.set("spark.sql.cbo.planStats.enabled", original_plan_stats)
    spark.conf.set("spark.sql.cbo.joinReorder.enabled", original_reorder)

    report.queries = results
    valid = [q for q in results if not q.error and q.cbo_enabled_ms > 0 and q.cbo_disabled_ms > 0]
    if valid:
        report.avg_speedup = round(
            sum(q.speedup_ratio for q in valid) / len(valid), 2
        )
        report.total_cbo_ms = round(sum(q.cbo_enabled_ms for q in valid), 2)
        report.total_nocbo_ms = round(sum(q.cbo_disabled_ms for q in valid), 2)

    # Write report
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": report.timestamp,
                "avg_speedup": report.avg_speedup,
                "total_cbo_ms": report.total_cbo_ms,
                "total_nocbo_ms": report.total_nocbo_ms,
                "queries": [asdict(q) for q in report.queries],
            },
            fh,
            indent=2,
        )
    logger.info("CBO benchmark report written to %s", output_path)
    return report
```
