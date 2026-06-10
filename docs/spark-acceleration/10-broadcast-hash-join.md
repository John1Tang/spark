# Broadcast Hash Join (BHJ)

## What It Is

Broadcast Hash Join (BHJ) is Spark SQL's most efficient equi-join strategy. It eliminates shuffle entirely by sending a complete copy of the smaller table to every executor, building an in-memory hash table from it, and then streaming the larger table through it for local probe lookups.

Unlike SortMergeJoin, which requires shuffling **both** sides of the join (two full data shuffles across the network), BHJ requires **zero** shuffles. The only network cost is broadcasting the smaller table once to all executors.

## Why It Works: The Two-Phase Mechanism

BHJ operates in two distinct phases:

### Phase 1: Broadcast Build

The smaller table (the "build side") is collected, transformed into a hash table, and distributed to every executor. This is handled by `BroadcastExchangeExec` (`sql/core/src/main/scala/org/apache/spark/sql/execution/exchange/BroadcastExchangeExec.scala`):

1. **Collect**: The build-side plan executes and all rows are collected to the driver via `child.executeCollectIterator()`.
2. **Build**: The collected rows are transformed into a `HashedRelation` -- an in-memory hash table.
3. **Broadcast**: The serialized hash table is broadcast to all executors via Spark's `Broadcast` mechanism with `serializedOnly = true`.

The broadcast is asynchronous. When `doPrepare()` is called on `BroadcastExchangeExec`, it triggers `relationFuture`, which starts the collection/build/broadcast pipeline in a background thread pool (`broadcast-exchange` thread pool, sized by `BROADCAST_EXCHANGE_MAX_THREAD_THRESHOLD`).

```scala
// BroadcastExchangeExec.scala (lines 176-246)
@transient
private lazy val relationFuture: Future[broadcast.Broadcast[Any]] = {
  // Collect rows to driver
  val (numRows, input) = child.executeCollectIterator()

  // Build the HashedRelation (hash table)
  val relation = mode.transform(input, Some(numRows))

  // Validate size limits
  if (dataSize >= maxBroadcastTableSizeInBytes) {
    throw cannotBroadcastTableOverMaxTableBytesError(...)
  }

  // Broadcast serialized-only
  val broadcasted = sparkContext.broadcastInternal(relation, serializedOnly = true)
  broadcasted
}
```

### Phase 2: Stream and Probe

The larger table (the "streamed side") is processed partition-by-partition. Each executor:

1. Reads its local partition of the streamed data.
2. Retrieves the broadcasted `HashedRelation` from its local broadcast store.
3. For each streamed row, computes the join key and probes the hash table.

This is implemented in `BroadcastHashJoinExec.doExecute()` (`sql/core/src/main/scala/org/apache/spark/sql/execution/joins/BroadcastHashJoinExec.scala`):

```scala
// BroadcastHashJoinExec.scala (lines 159-165)
streamedPlan.execute().mapPartitions { streamedIter =>
  val hashed = broadcastRelation.value.asReadOnlyCopy()
  TaskContext.get().taskMetrics().incPeakExecutionMemory(hashed.estimatedSize)
  join(streamedIter, hashed, numOutputRows)
}
```

The `asReadOnlyCopy()` call ensures thread-safe access to the hash table within the task.

## HashedRelation: The In-Memory Hash Table

`HashedRelation` (`sql/core/src/main/scala/org/apache/spark/sql/execution/joins/HashedRelation.scala`) is the core data structure. There are three concrete implementations:

| Implementation | When Used | Backing Store |
|---|---|---|
| `EmptyHashedRelation` | Build side has zero rows (no null keys allowed) | Singleton, zero memory |
| `LongHashedRelation` | Single join key of `LongType` | Optimized long-only hash map |
| `UnsafeHashedRelation` | All other cases | `BytesToBytesMap` (off-heap Unsafe memory) |

The factory method at lines 136-163 selects the implementation:

```scala
// HashedRelation.scala (lines 154-162)
if (!input.hasNext && !allowsNullKey) {
  EmptyHashedRelation                     // Fast path for empty build side
} else if (key.length == 1 && key.head.dataType == LongType && !allowsNullKey) {
  LongHashedRelation(...)                 // Optimized for single long key
} else {
  UnsafeHashedRelation(...)               // General-purpose, uses BytesToBytesMap
}
```

`UnsafeHashedRelation` uses `BytesToBytesMap`, an open-addressed hash map that stores raw binary key-value pairs in off-heap memory. This avoids Java object overhead and enables zero-copy access during probe.

### Row Limit Enforcement

`BroadcastExchangeExec` enforces a hard cap on the number of rows that can be broadcast (lines 164-174):

- For `HashedRelationBroadcastMode` with non-long keys: `(BytesToBytesMap.MAX_CAPACITY / 1.5).toLong` -- approximately 341 million rows, limited by the 70% load factor before map growth.
- For other modes: 512 million rows.

## What Problem It Solves

In a SortMergeJoin, both sides of the join are shuffled on the join key, sorted, and then merged. For a join between a 10 TB fact table and a 500 MB dimension table:

| Strategy | Network Shuffle | Sort Required | Executor Work |
|---|---|---|---|
| SortMergeJoin | 20 TB (both sides) | Both sides | Sort + Merge |
| Broadcast Hash Join | 500 MB (broadcast) | Neither | Hash probe (O(1)) |

BHJ eliminates the expensive shuffle and sort phases entirely, reducing a 20 TB network transfer to a single 500 MB broadcast, plus replacing sort+merge with O(1) hash lookups.

## How Spark Chooses BHJ: The JoinSelection Decision Tree

The join strategy selection logic lives in `JoinSelection` (`sql/core/src/main/scala/org/apache/spark/sql/execution/SparkStrategies.scala`, lines 217-329). The decision tree is:

### Step 1: Check Hints First

If hints are present, they are evaluated in priority order:
1. `BROADCAST` hint -> attempt BroadcastHashJoin
2. `MERGE` (sort-merge) hint -> attempt SortMergeJoin
3. `SHUFFLE_HASH` hint -> attempt ShuffledHashJoin
4. `SHUFFLE_REPLICATE_NL` hint -> attempt CartesianProduct

### Step 2: No Hints -- Size-Based Selection

If no hints are present (or hints don't apply), the fallback chain is:

1. **BroadcastHashJoin** -- if one side's `sizeInBytes <= spark.sql.autoBroadcastJoinThreshold` (default 10 MB) and the join type supports it.
2. **ShuffledHashJoin** -- if one side is much smaller than the other, small enough to build a local hash map, and `spark.sql.join.preferSortMergeJoin = false`.
3. **SortMergeJoin** -- if join keys are sortable (the default fallback).
4. **CartesianProduct** -- for inner joins with no equi-keys.
5. **BroadcastNestedLoopJoin** -- as last resort (can OOM).

### Build Side Selection

The build side is determined by `getBroadcastBuildSide()` in `JoinSelectionHelper` (`sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/optimizer/joins.scala`, lines 292-316):

```scala
def getBroadcastBuildSide(join: Join, hintOnly: Boolean, conf: SQLConf): Option[BuildSide] = {
  // Check join type compatibility
  val canBuildLeft = canBuildBroadcastLeft(join.joinType) && shouldBuildLeft()
  val canBuildRight = canBuildBroadcastRight(join.joinType) && shouldBuildRight()
  getBuildSide(canBuildLeft, canBuildRight, join.left, join.right)
}
```

Build side compatibility by join type (`canBuildBroadcastLeft` / `canBuildBroadcastRight`):

| Join Type | Can Build Left | Can Build Right |
|---|---|---|
| Inner | Yes | Yes |
| Left Outer | No | Yes |
| Right Outer | Yes | No |
| Left Semi | No | Yes |
| Left Anti | No | Yes |
| Full Outer | No | No |
| ExistenceJoin | No | Yes |

**Key limitation**: Full outer joins cannot use BHJ because both sides may produce unmatched rows that need to be preserved.

## Production Code Examples

### Automatic BHJ Selection (Star Schema)

When the dimension table's statistics are below the broadcast threshold, Spark automatically selects BHJ:

```scala
// Scala
val orders = spark.read.parquet("/data/warehouse/orders")     // 500 GB fact table
val customers = spark.read.parquet("/data/warehouse/customers")  // 50 MB dimension

val result = orders.join(customers, "customer_id")
// Spark automatically picks BroadcastHashJoin because customers < 10 MB threshold
// (assuming stats show it's small enough)

result.explain("extended")
// == Physical Plan ==
// *(3) Project ...
// +- *(3) BroadcastHashJoin [customer_id], [customer_id], Inner, BuildRight
//    :- *(3) FileScan parquet ...orders...
//    +- BroadcastExchange HashedRelationBroadcastMode(...), [plan_id=...]
//       +- *(2) FileScan parquet ...customers...
```

```python
# Python
orders = spark.read.parquet("/data/warehouse/orders")
customers = spark.read.parquet("/data/warehouse/customers")

result = orders.join(customers, "customer_id")
result.explain("extended")
```

```sql
-- SQL
SELECT o.order_id, o.amount, c.name, c.region
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
-- Automatic BHJ if customer table size < spark.sql.autoBroadcastJoinThreshold
```

### Forcing BHJ with Broadcast Hints

When the table exceeds the default 10 MB threshold but you know it fits in memory:

```sql
-- SQL: All three hint variants are equivalent
SELECT /*+ BROADCAST(c) */ *
FROM orders o JOIN customers c ON o.customer_id = c.customer_id

SELECT /*+ BROADCASTJOIN(c) */ *
FROM orders o JOIN customers c ON o.customer_id = c.customer_id

SELECT /*+ MAPJOIN(c) */ *
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
```

```scala
// Scala: DataFrame API
val orders = spark.read.parquet("/data/orders")
val customers = spark.read.parquet("/data/customers")

// Hint on one side
orders.join(customers.hint("broadcast"), "customer_id")

// Hint on both sides -- Spark picks the smaller as build side
orders.hint("broadcast").join(customers.hint("broadcast"), "customer_id")

// Explicit broadcast function
import org.apache.spark.sql.functions.broadcast
orders.join(broadcast(customers), "customer_id")
```

```python
# Python: DataFrame API
from pyspark.sql.functions import broadcast

orders = spark.read.parquet("/data/orders")
customers = spark.read.parquet("/data/customers")

# Hint approach
orders.join(customers.hint("broadcast"), "customer_id")

# Explicit broadcast function
orders.join(broadcast(customers), "customer_id")
```

### Reading EXPLAIN Output

```sql
EXPLAIN EXTENDED
SELECT o.order_id, c.name
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
```

BroadcastHashJoin physical plan:
```
== Physical Plan ==
*(2) Project [order_id#1, name#10]
+- *(2) BroadcastHashJoin [customer_id#2], [customer_id#9], Inner, BuildRight
   :- *(2) FileScan parquet [order_id#1, customer_id#2]
   :     +- BatchScan ...
   +- BroadcastExchange HashedRelationBroadcastMode(
        cast(input[0, bigint, false] as bigint), false), [plan_id=x]
      +- *(1) FileScan parquet [customer_id#9, name#10]
            +- BatchScan ...
```

Key elements to look for:
- `BroadcastHashJoin` node shows the join type (`Inner`), build side (`BuildRight`)
- `BroadcastExchange` shows the broadcast mode (`HashedRelationBroadcastMode`)
- `HashedRelationBroadcastMode` includes the build key expression and null-awareness flag

Contrast with SortMergeJoin (when BHJ is not selected):
```
== Physical Plan ==
*(4) Project [order_id#1, name#10]
+- *(4) SortMergeJoin [customer_id#2], [customer_id#9], Inner
   :- *(2) Sort [customer_id#2 ASC]
   :  +- Exchange hashpartitioning(customer_id#2, 200), ENSURE_REQUIREMENTS
   :     +- FileScan parquet [order_id#1, customer_id#2]
   +- *(3) Sort [customer_id#9 ASC]
      +- Exchange hashpartitioning(customer_id#9, 200), ENSURE_REQUIREMENTS
         +- FileScan parquet [customer_id#9, name#10]
```

Notice the two `Exchange` nodes -- this is the double shuffle that BHJ eliminates.

### Verifying BHJ in Spark UI

1. **SQL Tab > Details**: Look for `BroadcastHashJoin` in the executed plan.
2. **Broadcast metrics**: Click the `BroadcastExchange` node to see:
   - `data size`: serialized size of the broadcasted hash table
   - `number of output rows`: row count of the build side
   - `time to collect`: time to collect rows to the driver
   - `time to build`: time to construct the HashedRelation
   - `time to broadcast`: time to distribute to all executors
3. **Storage Tab**: Verify the broadcast variable is cached on each executor (shows as `broadcast_X` in the storage table).
4. **Task metrics**: The streamed-side tasks should show `Peak Execution Memory` reflecting the HashedRelation size.

## Key Configurations

| Configuration | Default | Recommended | Description |
|---|---|---|---|
| `spark.sql.autoBroadcastJoinThreshold` | `10485760` (10 MB) | `104857600` (100 MB) for memory-rich clusters | Maximum table size for automatic BHJ. Set to `-1` to disable. |
| `spark.sql.broadcastTimeout` | `300` (seconds) | `600` for large broadcasts | Timeout waiting for broadcast to complete. Increase if broadcasts are slow due to large tables or many executors. |
| `spark.sql.adaptive.autoBroadcastJoinThreshold` | Same as `autoBroadcastJoinThreshold` | Same or higher | AQE-specific threshold for runtime SMJ-to-BHJ conversion. |
| `spark.sql.join.preferSortMergeJoin` | `true` | `true` (default) | When `false`, allows ShuffledHashJoin as fallback. Keep `true` unless you have specific workloads that benefit from shuffle hash. |
| `spark.sql.maxBroadcastTableSizeInBytes` | Derived from `BytesToBytesMap.MAX_CAPACITY` | N/A | Hard limit on serialized broadcast size. Do not override. |

### Recommended Tuning Strategy

```scala
val conf = new SparkConf()
  .set("spark.sql.autoBroadcastJoinThreshold", "104857600") // 100 MB
  .set("spark.sql.broadcastTimeout", "600")
  .set("spark.sql.adaptive.autoBroadcastJoinThreshold", "209715200") // 200 MB for AQE
```

For data warehouse star schemas with dimension tables in the 50-200 MB range, raising the threshold to 100-200 MB is the single highest-impact configuration change.

## AQE: Runtime SMJ-to-BHJ Conversion

One of AQE's most powerful features is converting a planned SortMergeJoin into a BroadcastHashJoin **at runtime**, even if the planner originally chose SMJ.

This happens when:
1. AQE is enabled (`spark.sql.adaptive.enabled = true`).
2. After a shuffle stage completes, AQE computes accurate runtime statistics.
3. The runtime `sizeInBytes` of one join side falls below `spark.sql.adaptive.autoBroadcastJoinThreshold`.
4. The join type supports BHJ.

The optimization is triggered by the AQE framework and uses `DynamicJoinSelection` (`sql/core/src/main/scala/org/apache/spark/sql/execution/adaptive/DynamicJoinSelection.scala`) which may inject hints based on runtime statistics.

```
-- Planned as SMJ (because catalog stats show both sides are large)
== Optimized Logical Plan ==
Join Inner, (o.customer_id = c.customer_id)
  :- Filter o.date > '2024-01-01'     -- Filter reduces rows significantly at runtime
  :  +- SubqueryAlias o
  :     +- Relation orders
  +- SubqueryAlias c
     +- Relation customers

-- AQE re-plans at runtime:
== Physical Plan (after AQE) ==
*(2) BroadcastHashJoin [customer_id], [customer_id], Inner, BuildRight
   :- *(2) LocalTableScan   -- Local shuffle read, no more exchange
   +- BroadcastExchange HashedRelationBroadcastMode(...)
      +- *(1) FileScan ...customers...
```

In the Spark UI, you will see the AQE "Optimized Plan" differ from the "Initial Plan", with `BroadcastHashJoin` replacing `SortMergeJoin`.

## When BHJ Is NOT Possible

| Scenario | Reason | Workaround |
|---|---|---|
| **Full outer join** | Both sides can produce unmatched rows; BHJ cannot handle this | Use SortMergeJoin; cannot avoid shuffle |
| **Both sides exceed threshold** | Neither side can be broadcast | Increase threshold, or use AQE to see if filtering reduces one side |
| **Non-equi join** (no equality keys) | Hash join requires equality comparison on keys | Falls back to BroadcastNestedLoopJoin (slow) or CartesianProduct |
| **Join keys not binary-stable** | `UnsafeRowUtils.isBinaryStable()` returns false for key types | BHJ not supported; SMJ is used instead |
| **Memory pressure** | Broadcast data must fit in executor memory | Reduce threshold, or increase executor memory |
| **Cross join** (no join condition) | No keys for hash table | CartesianProduct; use hints to control |

## Limitations and Gotchas

### 1. Executor Memory Pressure

The broadcasted `HashedRelation` is deserialized into each executor's memory. For a 200 MB broadcast table:
- **Serialized**: ~200 MB in broadcast storage
- **Deserialized**: Can expand to 400-800 MB in `BytesToBytesMap` due to the hash table structure

With 100 executors, the total network cost is 200 MB (driver-to-executors), but each executor must hold the full hash table in memory.

**Mitigation**: Monitor executor memory usage. If you see `OutOfMemoryError` with broadcast-related stack traces, reduce `spark.sql.autoBroadcastJoinThreshold` or increase executor memory.

### 2. Broadcast Timeout

The default 300-second timeout may be insufficient for:
- Very large broadcast tables (> 100 MB)
- Clusters with many executors (broadcast is fan-out)
- Network-congested environments

When a broadcast times out, the error message is:
```
Could not execute broadcast in 300 secs.
```

**Mitigation**: Increase `spark.sql.broadcastTimeout`. Monitor the `broadcastTime` metric to understand actual broadcast duration.

### 3. Driver Memory Pressure

`BroadcastExchangeExec` collects all build-side rows to the driver before broadcasting. For a 200 MB table, the driver temporarily holds all rows in memory during the build phase.

**Mitigation**: Ensure driver memory is sufficient. Monitor `collectTime` and `buildTime` metrics.

### 4. Data Skew Is Not a BHJ Problem

Unlike SortMergeJoin, BHJ is not affected by data skew on the streamed side, since each executor processes its local partition independently against the full hash table. However, skew on the **build side** (duplicate join keys) can cause multiple output rows per streamed row:

```scala
// If build side has duplicate keys, one streamed row produces multiple outputs
// This is handled correctly but may inflate output size
```

### 5. Stale Statistics Lead to Wrong Decisions

If table statistics are outdated, Spark may:
- Choose SMJ when BHJ would be better (table is actually small but stats say it's large)
- Choose BHJ when the table has grown beyond what fits in memory (stats are stale)

**Mitigation**: Run `ANALYZE TABLE` regularly. Enable AQE so runtime statistics can correct planning decisions.

### 6. NULL-Aware Anti-Join Optimization

For single-column `LEFT ANTI` joins where the build side is broadcastable, Spark uses a specialized `isNullAwareAntiJoin` mode (`BroadcastHashJoinExec`, lines 56-62, 132-158). This handles the three-valued logic of anti-joins with NULL keys efficiently:

- If the build-side hash table is empty (`EmptyHashedRelation`), all streamed rows pass through.
- If the build side contains any all-NULL key (`HashedRelationWithAllNullKeys`), no streamed rows pass through (three-valued logic semantics).

### 7. Codegen Path

`BroadcastHashJoinExec` implements `CodegenSupport`. The generated code inlines the hash table access for maximum performance (see `prepareBroadcast` and `prepareRelation` methods, lines 192-212). This avoids virtual method calls during the hot probe loop.

## Summary

Broadcast Hash Join is Spark SQL's fastest join strategy because it eliminates shuffle entirely. It works by broadcasting the smaller table to all executors, building a `HashedRelation` hash table, and streaming the larger table through it. Key configurations are `spark.sql.autoBroadcastJoinThreshold` (controls automatic selection) and `spark.sql.broadcastTimeout` (controls wait time). Use broadcast hints to force BHJ when statistics are stale, and enable AQE to benefit from runtime SMJ-to-BHJ conversion.

## Production-Ready Python (PySpark) Code

### Factory: Create a BHJ-Optimized Session

```python
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from pyspark.sql.types import StructType


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bhj")


def create_bhj_session(
    auto_broadcast_threshold_mb: int = 100,
    broadcast_timeout_s: int = 600,
    prefer_sort_merge: bool = True,
    aqe_broadcast_threshold_mb: Optional[int] = None,
    app_name: str = "BHJOptimized",
    **extra_conf: str,
) -> SparkSession:
    """Build a SparkSession tuned for Broadcast Hash Join performance.

    Parameters
    ----------
    auto_broadcast_threshold_mb :
        Size in MB below which Spark auto-selects BHJ (default 100 MB).
    broadcast_timeout_s :
        Seconds before broadcast exchange times out (default 600).
    prefer_sort_merge :
        When ``True`` (the default), Spark prefers SMJ over
        ShuffledHashJoin.  Set ``False`` only if you know the build
        side fits comfortably in memory and want to avoid sort overhead.
    aqe_broadcast_threshold_mb :
        AQE runtime SMJ-to-BHJ conversion threshold.  Defaults to
        ``auto_broadcast_threshold_mb * 2`` if not supplied.
    extra_conf :
        Additional key=value config pairs passed through to the builder.

    Returns
    -------
    SparkSession
    """
    aqe_threshold = aqe_broadcast_threshold_mb or auto_broadcast_threshold_mb * 2

    conf_map = {
        "spark.sql.autoBroadcastJoinThreshold": str(auto_broadcast_threshold_mb * 1024 * 1024),
        "spark.sql.broadcastTimeout": str(broadcast_timeout_s),
        "spark.sql.join.preferSortMergeJoin": str(prefer_sort_merge),
        "spark.sql.adaptive.autoBroadcastJoinThreshold": str(aqe_threshold * 1024 * 1024),
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
    }
    conf_map.update({k: v for k, v in extra_conf.items()})

    builder = SparkSession.builder.appName(app_name)
    for k, v in conf_map.items():
        builder = builder.config(k, v)

    session = builder.getOrCreate()
    logger.info(
        "BHJ session created: threshold=%d MB, timeout=%ds, "
        "preferSMJ=%s, aqeThreshold=%d MB",
        auto_broadcast_threshold_mb,
        broadcast_timeout_s,
        prefer_sort_merge,
        aqe_threshold,
    )
    return session


### EXPLAIN Parser: Verify BHJ Is Active

@dataclass
class BHJVerificationResult:
    """Structured result from parsing an EXPLAIN plan for BHJ indicators."""
    uses_bhj: bool = False
    broadcast_table_name: Optional[str] = None
    build_side: Optional[str] = None
    join_type: Optional[str] = None
    broadcast_mode: Optional[str] = None
    estimated_broadcast_size_bytes: Optional[int] = None
    raw_explain: str = ""
    notes: list[str] = field(default_factory=list)


def verify_bhj_active(spark: SparkSession, query: str) -> BHJVerificationResult:
    """Run EXPLAIN on *query* and parse the physical plan for BHJ markers.

    Looks for ``BroadcastHashJoin``, ``BroadcastExchange``, and extracts
    build side, join type, and estimated broadcast table size.

    Parameters
    ----------
    spark :
        Active SparkSession.
    query :
        SQL string to explain (do NOT execute; only plan).

    Returns
    -------
    BHJVerificationResult
    """
    plan = spark.sql(f"EXPLAIN EXTENDED {query}").collect()[0][0]
    result = BHJVerificationResult(raw_explain=plan)

    bhj_match = re.search(
        r"BroadcastHashJoin\s+\[([^\]]*)\],\s*\[([^\]]*)\],\s*(\w+),\s*Build(Right|Left)",
        plan,
    )
    if bhj_match:
        result.uses_bhj = True
        result.join_type = bhj_match.group(3)
        result.build_side = f"Build{bhj_match.group(4)}"

    exchange_match = re.search(
        r"BroadcastExchange\s+(\w+)",
        plan,
    )
    if exchange_match:
        result.broadcast_mode = exchange_match.group(1)

    # Try to detect the broadcast table from the scan under BroadcastExchange
    table_match = re.search(
        r"BroadcastExchange.*?FileScan\s+(?:parquet|orc|csv)\s+\[.*?\]\s+(?:BatchScan\s+)?(\S+)",
        plan,
        re.DOTALL,
    )
    if table_match:
        result.broadcast_table_name = table_match.group(1)

    # Estimate broadcast size from scan stats if present
    size_match = re.search(
        r"BroadcastExchange.*?sizeInBytes=(\d+)",
        plan,
        re.DOTALL,
    )
    if size_match:
        result.estimated_broadcast_size_bytes = int(size_match.group(1))

    if not result.uses_bhj:
        smj_match = re.search(r"SortMergeJoin", plan)
        if smj_match:
            result.notes.append("Plan fell back to SortMergeJoin; "
                                "check table size vs autoBroadcastJoinThreshold")
        else:
            result.notes.append("Neither BHJ nor SMJ found; review plan structure")
    else:
        size_mb = (result.estimated_broadcast_size_bytes or 0) / (1024 * 1024)
        result.notes.append(f"Broadcast build side estimated at {size_mb:.1f} MB")

    return result


### Benchmark: BHJ vs SMJ

@dataclass
class BHJBenchmarkResult:
    join_type: str = ""
    bhj_duration_s: float = 0.0
    smj_duration_s: float = 0.0
    bhj_rows_written: int = 0
    smj_rows_written: int = 0
    network_bytes_saved_mb: float = 0.0
    speedup_ratio: float = 0.0
    config_snapshot: dict[str, str] = field(default_factory=dict)


def bhj_benchmark(
    spark: SparkSession,
    fact_table_path: str,
    dim_table_path: str,
    join_column: str,
    select_expr: str = "*",
) -> BHJBenchmarkResult:
    """Compare Broadcast Hash Join vs SortMergeJoin on the same data.

    The function temporarily toggles ``spark.sql.autoBroadcastJoinThreshold``
    and ``spark.sql.join.preferSortMergeJoin`` to force each plan, then
    measures wall-clock execution time.

    Parameters
    ----------
    spark :
        Active SparkSession.
    fact_table_path :
        Path to the large (fact) table.
    dim_table_path :
        Path to the small (dimension) table.
    join_column :
        Equi-join column name present in both tables.
    select_expr :
        Column expression for the SELECT clause.

    Returns
    -------
    BHJBenchmarkResult
    """
    fact = spark.read.parquet(fact_table_path)
    dim = spark.read.parquet(dim_table_path)
    fact.createOrReplaceTempView("fact")
    dim.createOrReplaceTempView("dim")
    query = f"SELECT {select_expr} FROM fact JOIN dim ON fact.{join_column} = dim.{join_column}"

    snapshot = dict(spark.conf.getAll())

    # -- BHJ run --
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(1024 * 1024 * 1024))  # 1 GB
    spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
    start = time.monotonic()
    bhj_df = spark.sql(query)
    bhj_rows = bhj_df.count()  # materialize
    bhj_dur = time.monotonic() - start

    # -- SMJ run --
    original_threshold = snapshot.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")  # disable BHJ
    start = time.monotonic()
    smj_df = spark.sql(query)
    smj_rows = smj_df.count()
    smj_dur = time.monotonic() - start

    # Restore
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", original_threshold)

    # Estimate shuffle bytes SMJ would have moved (fact * 2 for both-side shuffle)
    fact_size_bytes = spark.sql(
        f"SELECT size_in_bytes FROM spark_catalog.system.tables "
        f"WHERE table_name = 'fact'"
    ).collect()
    # Fallback: approximate from count * average row size
    estimated_fact_mb = fact.count() * 0.0001  # rough heuristic
    network_saved = estimated_fact_mb * 2  # SMJ shuffles both sides

    speedup = smj_dur / bhj_dur if bhj_dur > 0 else 0.0

    result = BHJBenchmarkResult(
        join_type=join_column,
        bhj_duration_s=round(bhj_dur, 3),
        smj_duration_s=round(smj_dur, 3),
        bhj_rows_written=bhj_rows,
        smj_rows_written=smj_rows,
        network_bytes_saved_mb=round(network_saved, 2),
        speedup_ratio=round(speedup, 2),
        config_snapshot={k: snapshot[k] for k in [
            "spark.sql.autoBroadcastJoinThreshold",
            "spark.sql.join.preferSortMergeJoin",
        ] if k in snapshot},
    )

    report_path = Path("/tmp/bhj_benchmark.json")
    report_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info("BHJ benchmark written to %s", report_path)
    logger.info(
        "BHJ=%.3fs  SMJ=%.3fs  speedup=%.2fx  network_saved=%.1f MB",
        bhj_dur, smj_dur, speedup, network_saved,
    )
    return result


### Tuning: Find Optimal Broadcast Threshold

@dataclass
class ThresholdTuningResult:
    """Result from iterative broadcast threshold tuning."""
    threshold_mb: int
    avg_join_duration_s: float
    uses_bhj: bool
    notes: str


def tune_broadcast_threshold(
    spark: SparkSession,
    fact_table_path: str,
    dim_table_path: str,
    join_column: str,
    thresholds_mb: Optional[list[int]] = None,
) -> list[ThresholdTuningResult]:
    """Iteratively adjust ``autoBroadcastJoinThreshold`` and measure join
    performance at each level.

    Parameters
    ----------
    spark : Active SparkSession.
    fact_table_path : Fact table path.
    dim_table_path : Dimension table path.
    join_column : Join column name.
    thresholds_mb : List of threshold values to test (MB).

    Returns
    -------
    list[ThresholdTuningResult]
    """
    if thresholds_mb is None:
        thresholds_mb = [1, 10, 50, 100, 200, 500, 1024]

    fact = spark.read.parquet(fact_table_path)
    dim = spark.read.parquet(dim_table_path)
    fact.createOrReplaceTempView("tune_fact")
    dim.createOrReplaceTempView("tune_dim")

    original = spark.conf.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    results: list[ThresholdTuningResult] = []

    for mb in thresholds_mb:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(mb * 1024 * 1024))
        plan = spark.sql(
            f"EXPLAIN SELECT * FROM tune_fact f JOIN tune_dim d "
            f"ON f.{join_column} = d.{join_column}"
        ).collect()[0][0]
        uses_bhj = "BroadcastHashJoin" in plan

        start = time.monotonic()
        spark.sql(
            f"SELECT count(*) FROM tune_fact f JOIN tune_dim d "
            f"ON f.{join_column} = d.{join_column}"
        ).collect()
        dur = time.monotonic() - start

        note = "BHJ selected" if uses_bhj else "SMJ fallback"
        results.append(ThresholdTuningResult(
            threshold_mb=mb,
            avg_join_duration_s=round(dur, 3),
            uses_bhj=uses_bhj,
            notes=note,
        ))
        logger.info("threshold=%4d MB -> %s (%.3fs)", mb, note, dur)

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", original)

    report_path = Path("/tmp/bhj_threshold_tuning.json")
    report_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    logger.info("Threshold tuning results written to %s", report_path)
    return results


### Broadcast Hint Helper

def force_broadcast_join(
    spark: SparkSession,
    table_name: str,
    left_path: str,
    right_path: str,
    join_column: str,
    file_format: str = "parquet",
) -> BHJVerificationResult:
    """Force a Broadcast Hash Join using the BROADCAST hint on *table_name*.

    The hint is applied to whichever side's path matches *table_name*.
    Returns a verification result confirming BHJ selection.

    Parameters
    ----------
    spark : Active SparkSession.
    table_name : Which table alias to broadcast (``"left"`` or ``"right"``).
    left_path : Path for the left DataFrame.
    right_path : Path for the right DataFrame.
    join_column : Equi-join column.
    file_format : Underlying file format (default ``parquet``).

    Returns
    -------
    BHJVerificationResult
    """
    left_df = spark.read.format(file_format).load(left_path).alias("left_tbl")
    right_df = spark.read.format(file_format).load(right_path).alias("right_tbl")

    if table_name.lower() == "left":
        joined = left_df.hint("broadcast").join(right_df, on=join_column)
    elif table_name.lower() == "right":
        joined = right_df.hint("broadcast").join(left_df, on=join_column)
    else:
        raise ValueError(f"table_name must be 'left' or 'right', got '{table_name}'")

    joined.createOrReplaceTempView("forced_bhj_result")
    plan = joined._jdf.queryExecution().executedPlan().toString()

    result = BHJVerificationResult(raw_explain=plan)
    if "BroadcastHashJoin" in plan:
        result.uses_bhj = True
        result.notes.append(f"Forced BHJ via BROADCAST({table_name})")
    else:
        result.notes.append(f"BROADCAST({table_name}) hint did not produce BHJ")

    logger.info("force_broadcast_join: %s", result.notes[0])
    return result


### BHJMonitor Class

@dataclass
class JoinPlanAnalysis:
    query_id: str = ""
    join_strategy: str = ""
    is_bhj: bool = False
    broadcast_size_bytes: Optional[int] = None
    build_side: Optional[str] = None
    efficiency_score: float = 0.0  # 1.0 = optimal BHJ
    warnings: list[str] = field(default_factory=list)


class BHJMonitor:
    """Analyzes join query plans, detects BHJ vs SMJ selection, and tracks
    broadcast efficiency across multiple queries.

    Usage::

        monitor = BHJMonitor(spark)
        result = monitor.analyze("q1", "SELECT * FROM orders o JOIN customers c ON ...")
        print(monitor.summary())
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.analyses: list[JoinPlanAnalysis] = []

    def analyze(self, query_id: str, query: str) -> JoinPlanAnalysis:
        plan = self.spark.sql(f"EXPLAIN EXTENDED {query}").collect()[0][0]

        is_bhj = "BroadcastHashJoin" in plan
        strategy = "BroadcastHashJoin" if is_bhj else (
            "SortMergeJoin" if "SortMergeJoin" in plan else
            "ShuffledHashJoin" if "ShuffledHashJoin" in plan else
            "CartesianProduct" if "CartesianProduct" in plan else
            "BroadcastNestedLoopJoin" if "BroadcastNestedLoopJoin" in plan else
            "Unknown"
        )

        build_side: Optional[str] = None
        bs_match = re.search(r"Build(Right|Left)", plan)
        if bs_match:
            build_side = f"Build{bs_match.group(1)}"

        size_bytes: Optional[int] = None
        sz_match = re.search(r"BroadcastExchange.*?sizeInBytes=(\d+)", plan, re.DOTALL)
        if sz_match:
            size_bytes = int(sz_match.group(1))

        # Efficiency score: 1.0 for BHJ, 0.5 for SHJ, 0.0 for SMJ/Cartesian
        score = 1.0 if is_bhj else (
            0.5 if strategy == "ShuffledHashJoin" else 0.0
        )

        warnings: list[str] = []
        if not is_bhj and size_bytes is None:
            warnings.append("No broadcast detected; consider raising "
                            "autoBroadcastJoinThreshold or adding BROADCAST hint")
        if strategy == "SortMergeJoin" and "Exchange" in plan:
            exchange_count = plan.count("Exchange")
            warnings.append(f"SMJ with {exchange_count} exchange(s) = full shuffle on both sides")

        analysis = JoinPlanAnalysis(
            query_id=query_id,
            join_strategy=strategy,
            is_bhj=is_bhj,
            broadcast_size_bytes=size_bytes,
            build_side=build_side,
            efficiency_score=score,
            warnings=warnings,
        )
        self.analyses.append(analysis)
        return analysis

    def summary(self) -> dict:
        total = len(self.analyses)
        bhj_count = sum(1 for a in self.analyses if a.is_bhj)
        avg_score = (
            sum(a.efficiency_score for a in self.analyses) / total
            if total > 0 else 0.0
        )
        return {
            "total_queries_analyzed": total,
            "bhj_selected": bhj_count,
            "smj_fallback": total - bhj_count,
            "broadcast_efficiency": round(avg_score, 2),
            "analyses": [asdict(a) for a in self.analyses],
        }

    def write_report(self, path: str = "/tmp/bhj_monitor_report.json") -> str:
        Path(path).write_text(json.dumps(self.summary(), indent=2))
        logger.info("BHJ monitor report written to %s", path)
        return path


### Production ETL with BHJ

def run_production_bhj_etl(
    spark: SparkSession,
    fact_paths: dict[str, str],
    dim_paths: dict[str, str],
    output_path: str,
    broadcast_threshold_mb: int = 200,
) -> dict:
    """Run a star-schema ETL with intelligent broadcast hints and verification.

    Parameters
    ----------
    spark : Active SparkSession.
    fact_paths :
        Mapping of fact table alias to filesystem path.
    dim_paths :
        Mapping of dimension table alias to filesystem path.
    output_path :
        Destination directory for the materialized result.
    broadcast_threshold_mb :
        Per-query broadcast threshold override.

    Returns
    -------
    dict
        ETL run metadata including join strategy for each step and
        total execution time.
    """
    original_threshold = spark.conf.get(
        "spark.sql.autoBroadcastJoinThreshold", "10485760"
    )
    spark.conf.set(
        "spark.sql.autoBroadcastJoinThreshold",
        str(broadcast_threshold_mb * 1024 * 1024),
    )

    monitor = BHJMonitor(spark)
    start = time.monotonic()
    steps: list[dict] = []
    errors: list[str] = []

    try:
        # Register all tables
        for alias, path in {**fact_paths, **dim_paths}.items():
            logger.info("Registering table %s from %s", alias, path)
            spark.read.parquet(path).createOrReplaceTempView(alias)

        # Example: star-schema join (fact + all dimensions)
        fact_alias = next(iter(fact_paths.keys()))
        dim_aliases = list(dim_paths.keys())

        if dim_aliases:
            join_cols = ", ".join(f"{fact_alias}.{c} = {dim_aliases[0]}.{c}" for c in ["id"])
            query = f"SELECT {fact_alias}.* FROM {fact_alias}"
            for dim_alias in dim_aliases:
                query += f" LEFT JOIN {dim_alias} ON {fact_alias}.id = {dim_alias}.id"
            query += f" WHERE {fact_alias}.id IS NOT NULL"

            logger.info("Running star-schema ETL join...")
            result_df = spark.sql(query)

            analysis = monitor.analyze("star_schema_join", query)
            steps.append({
                "step": "star_schema_join",
                "query_id": "star_schema_join",
                "strategy": analysis.join_strategy,
                "uses_bhj": analysis.is_bhj,
                "warnings": analysis.warnings,
            })

            result_df.write.mode("overwrite").parquet(output_path)
            logger.info("ETL output written to %s", output_path)
        else:
            errors.append("No dimension tables provided; skipping join")

    except Exception as exc:
        logger.error("ETL failed: %s", exc)
        errors.append(str(exc))
    finally:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", original_threshold)

    total_dur = round(time.monotonic() - start, 3)
    report_path = Path("/tmp/bhj_etl_report.json")
    report = {
        "etl_duration_s": total_dur,
        "fact_tables": fact_paths,
        "dim_tables": dim_paths,
        "broadcast_threshold_mb": broadcast_threshold_mb,
        "steps": steps,
        "monitor_summary": monitor.summary(),
        "errors": errors,
        "output_path": output_path,
    }
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("ETL report written to %s (duration=%.3fs)", report_path, total_dur)
    return report
```

### Quick Usage Example

```python
# 1. Create optimized session
spark = create_bhj_session(
    auto_broadcast_threshold_mb=200,
    broadcast_timeout_s=600,
)

# 2. Verify BHJ selection
result = verify_bhj_active(spark, """
    SELECT o.order_id, c.name
    FROM parquet.`/data/orders` o
    JOIN parquet.`/data/customers` c
    ON o.customer_id = c.customer_id
""")
print(f"Uses BHJ: {result.uses_bhj}, Build: {result.build_side}")

# 3. Monitor multiple queries
monitor = BHJMonitor(spark)
monitor.analyze("q1", "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id")
monitor.analyze("q2", "SELECT * FROM orders o JOIN products p ON o.product_id = p.product_id")
monitor.write_report("/tmp/bhj_analysis.json")

# 4. Run benchmark
benchmark = bhj_benchmark(
    spark,
    fact_table_path="/data/warehouse/orders",
    dim_table_path="/data/warehouse/customers",
    join_column="customer_id",
)
print(f"BHJ speedup: {benchmark.speedup_ratio}x")
```
