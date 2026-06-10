# Spark Declarative Pipelines + Delta Lake: Research Report

> **Date:** 2026-05-16
> **Scope:** Can open-source Apache Spark Declarative Pipelines (SDP) perform Delta Lake operations (MERGE INTO, UPDATE, DELETE) via SQL? Is this production-ready with Spark 4.2?

---

## Executive Summary

| Question | Answer |
|----------|--------|
| Delta Lake version compatible with Spark 4.2? | **No released version yet.** Latest is Delta Lake 4.2.0, built for Spark 4.1.0 and Spark 4.0.1. Delta master branch compiles against Spark 4.2.0-SNAPSHOT but is unreleased. |
| Can SDP SQL run Delta MERGE INTO / UPDATE / DELETE? | **Yes, but indirectly.** Open-source SDP does not support imperative DML inside flow query functions. Instead, the declarative model (`CREATE STREAMING TABLE AS`, `CREATE MATERIALIZED VIEW AS`) writes to Delta-backed tables natively. Delta's MERGE/UPDATE/DELETE works as standalone Spark SQL, but not as first-class SDP pipeline operations. |
| Production-ready? | **Delta Lake: Yes. SDP (open-source): Early-stage.** Delta Lake is mature and battle-tested. Apache Spark SDP is newly donated (DAIS 2025), shipped in Spark 4.1, and is under active development. Databricks Lakeflow SDP (commercial extension) is production-ready. |

---

## 1. Delta Lake Version Compatibility

### 1.1 Official Compatibility Matrix

| Delta Lake | Apache Spark | Released | Maven Artifact |
|------------|-------------|----------|----------------|
| 4.2.0 | 4.1.0, 4.0.1 | 2026-04-16 | `io.delta:delta-spark_4.1_2.13:4.2.0` |
| 4.1.0 | 4.1.0, 4.0.1 | 2026-02-26 | `io.delta:delta-spark_4.1_2.13:4.1.0` |
| 4.0.1 | 4.0.1 | 2026-01-16 | `io.delta:delta-spark_2.13:4.0.1` |
| 4.0.0 | 4.0.0 | 2025-06-09 | `io.delta:delta-spark_2.13:4.0.0` |
| 3.3.x | 3.5.x | 2025 | `io.delta:delta-spark_2.12:3.3.x` |

Source: [docs.delta.io/releases](https://docs.delta.io/releases/)

### 1.2 Spark 4.2 Status

- **Apache Spark 4.2.0** is currently in **preview** (preview2, preview3, preview4). Not yet GA.
- Delta Lake master branch has added **Spark 4.2.0-SNAPSHOT compilation support** (PR #6256, #6319 in delta-io/delta), but **no Delta release targets Spark 4.2**.
- Delta Lake tracking issue [#5326](https://github.com/delta-io/delta/issues/5326) tracks multi-Spark-version support. Several items remain open.

### 1.3 Key Finding: No Delta Lake for Spark 4.2 Yet

If you need Spark 4.2 today, you must either:
1. Use Delta Lake master branch (unreleased, build from source against Spark 4.2.0-SNAPSHOT)
2. Drop to Spark 4.1.0 + Delta Lake 4.2.0 (latest stable combination)

**Recommendation:** Use **Spark 4.1.0 + Delta Lake 4.2.0** for the most stable open-source stack that supports both SDP and Delta Lake.

---

## 2. Spark Declarative Pipelines (SDP)

### 2.1 What Is SDP?

Spark Declarative Pipelines is a declarative framework for building batch and streaming data pipelines in Python or SQL. Originally created by Databricks as "Delta Live Tables (DLT)", it was **contributed to Apache Spark at DAIS 2025** and shipped in **Spark 4.1**.

- **Installation:** `pip install pyspark[pipelines]`
- **CLI:** `spark-pipelines init`, `spark-pipelines run`, `spark-pipelines dry-run`
- **Docs:** [spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html](https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html)

### 2.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Pipeline** | Top-level unit grouping datasets and flows. Orchestrates dependency resolution and execution order. |
| **Streaming Table** | Incremental target for streaming data. Created with `@dp.table` (Python) or `CREATE STREAMING TABLE` (SQL). |
| **Materialized View** | Batch target that stays aligned with source data. Created with `@dp.materialized_view` (Python) or `CREATE MATERIALIZED VIEW` (SQL). |
| **Flow** | A query + target pair. Append flows are the default. |

### 2.3 SQL Syntax

```sql
-- Streaming table (incremental)
CREATE STREAMING TABLE orders
AS SELECT * FROM STREAM read_files("/data/orders", format => "json");

-- Materialized view (batch)
CREATE MATERIALIZED VIEW customer_orders
AS SELECT c.*, o.order_id FROM customers c JOIN orders o ON c.id = o.customer_id;

-- Append flow to existing table
CREATE FLOW backfill_orders AS INSERT INTO orders BY NAME
SELECT * FROM historical_orders;
```

### 2.4 Open-Source vs Databricks Lakeflow SDP

| Feature | Apache Spark SDP | Databricks Lakeflow SDP |
|---------|-----------------|------------------------|
| `CREATE STREAMING TABLE` | Yes | Yes |
| `CREATE MATERIALIZED VIEW` | Yes | Yes |
| `CREATE FLOW` | Yes | Yes |
| `AUTO CDC` | **No** (planned) | Yes |
| `@dp.expect()` (data quality) | **No** | Yes |
| `@dp.temporary_view` | **No** | Yes |
| Scheduling (`SCHEDULE REFRESH`) | **No** | Yes |
| Continuous mode | **No** (planned) | Yes |
| Unity Catalog integration | No | Yes |
| Auto Loader (`read_files`) | No | Yes |

Source: [Databricks Python reference](https://docs.databricks.com/en/delta-live-tables/python-ref.html)

---

## 3. Delta Lake SQL Operations (MERGE, UPDATE, DELETE)

### 3.1 Standard Delta SQL DML

Delta Lake provides full SQL DML support **independent of SDP**:

```sql
-- MERGE (upsert)
MERGE INTO target USING source ON target.key = source.key
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE;

-- UPDATE
UPDATE events SET deleted = true WHERE date < '2024-01-01';

-- DELETE
DELETE FROM events WHERE event_id = 123;
```

These work with any Delta table via `spark.sql()` in a standard Spark session. They are **production-ready and battle-tested** since Delta Lake 0.7.0 (Spark 3.0).

### 3.2 Delta Table API (Programmatic)

```python
from delta.tables import DeltaTable

deltaTable = DeltaTable.forPath(spark, "/data/target")
deltaTable.alias("target").merge(
    source.alias("source"),
    "target.key = source.key"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```

---

## 4. The Critical Question: SDP + Delta DML via SQL

### 4.1 How SDP Handles SQL Inside Query Functions

When `spark.sql()` is called **inside an SDP query function** (Python flow function or SQL view definition):

- **Inside query functions:** Spark Connect treats `spark.sql()` as a **no-op** that returns the raw logical plan to SDP for deferred analysis as part of the Dataflow Graph. (PR #53024, SPARK-54020, merged for Spark 4.1.0)
- **Outside query functions:** Spark Connect eagerly executes the command, but only **SDP-allowlisted commands** are permitted.

This means:
- `SELECT`, `JOIN`, aggregations inside a flow function work fine (they return DataFrame plans that SDP orchestrates)
- **Imperative DML commands** (`MERGE INTO`, `UPDATE`, `DELETE`, `INSERT INTO` as standalone) are **not the intended pattern** inside SDP flow functions

### 4.2 The SDP Way vs The Delta Way

**SDP's intended pattern** (declarative):
```sql
-- You declare what the table SHOULD look like
CREATE MATERIALIZED VIEW silver_users
AS SELECT id, name, email, updated_at
   FROM bronze_users
   WHERE status = 'active';
```
SDP handles incremental updates, dependency tracking, and writes the result to a Delta table automatically.

**Delta's DML pattern** (imperative):
```sql
-- You explicitly say how to modify the table
MERGE INTO silver_users USING updates ON silver_users.id = updates.id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
```
This is a standalone command that executes immediately.

### 4.3 Can You Combine Them?

**Yes, but not inside flow definitions.** The working pattern is:

1. **SDP manages the pipeline** -- defines tables/views, handles orchestration, incremental processing
2. **Delta tables are the storage layer** -- all SDP tables are backed by Delta format
3. **Delta DML runs outside SDP flows** -- use `spark.sql("MERGE INTO ...")` in standard Spark code, or via `foreachBatch` in structured streaming

**Practical approach for a CDC pipeline:**
```python
# SDP handles ingestion
@dp.table
def bronze_events():
    return spark.readStream.format("kafka").load()

# Delta MERGE handles the upsert (outside SDP, in foreachBatch)
def upsert_to_silver(micro_batch_df, batch_id):
    micro_batch_df.createOrReplaceTempView("batch_updates")
    spark.sql("""
        MERGE INTO silver_events target
        USING batch_updates source ON target.event_id = source.event_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

# Use structured streaming with foreachBatch
bronze_df.writeStream.foreachBatch(upsert_to_silver).start()
```

### 4.4 Databricks Lakeflow AUTO CDC (Not in Open Source)

Databricks has a declarative CDC that replaces manual MERGE:

```sql
CREATE OR REFRESH STREAMING TABLE products_history
SCHEDULE REFRESH EVERY 1 DAY
FLOW AUTO CDC
FROM STREAM products WITH (readChangeFeed = true)
KEYS (product_id)
APPLY AS DELETE WHEN _change_type = 'delete'
SEQUENCE BY _commit_timestamp
STORED AS SCD TYPE 2;
```

This is **NOT available in open-source Apache Spark SDP**. It's a Databricks-only feature. Auto CDC is listed as "planned" for Apache Spark.

---

## 5. Real-World Evidence: Spark 4.1 + Delta Lake 4.2

### 5.1 Official Delta RC Testing

Delta Lake 4.2.0 release candidate (issue [#6345](https://github.com/delta-io/delta/issues/6345)) includes explicit testing instructions that run:

```
io.delta:delta-spark_4.1_2.13:4.2.0-SNAPSHOT
```

against **Spark 4.1.1**. This is concrete, official proof that the Delta 4.2.0 artifact is built and tested against the Spark 4.1.x line.

### 5.2 Community Blog Tutorials

**Sandipan Ghosh — "Spark Declarative Pipelines (SDP) in Spark 4.1+"** (Medium, Jan 2026):
- Full walkthrough of SDP setup with Spark 4.1
- Section 8.2 explicitly covers "Delta Lake (Spark extensions + Delta catalog)" with YAML configuration showing the required `spark.sql.extensions` and `spark.sql.catalog.spark_catalog` settings for SDP + Delta Lake combination
- Demonstrates `pip install pyspark[pipelines]` alongside Delta Lake dependencies

**Christian Hansen — "What Developers Need to Know About Delta Lake 4.1"** (Medium, April 2026):
- Confirms Delta Lake 4.1.0 was explicitly built for Spark 4.1.0
- Covers the new multi-Spark-version artifact naming convention (`delta-spark_4.1_2.13`)

**Ivan Kurchenko — "Spark 4 by example: Declarative pipelines"** (Level Up Coding, March 2026):
- Working batch and streaming pipeline examples using Spark 4.1
- Companion GitHub repo: `IvannKurchenko/blog-spark-4.1-declarative-pipelines`

### 5.3 GitHub Example Repositories

**shauryashaurya/rocket-ship** — "Spark Declarative Pipelines (Spark 4.1.1)":
- Reinsurance pipeline examples using `@sdp.materializedview(name="treatiesmv")` decorator
- Reads data in Delta format, demonstrating SDP tables backed by Delta storage

**Jacek Laskowski's comment on delta-io/delta#5326**:
- Asked about working on Delta Lake with Spark 4.1.0 specifically for Spark Declarative Pipelines use cases
- Signals community awareness of the SDP + Delta Lake pairing

### 5.4 Known Compatibility Issues

**NVIDIA spark-rapids PR [#14120](https://github.com/NVIDIA/spark-rapids/pull/14120)**:
- Delta Lake is **excluded for Spark 4.1.1 specifically** (not 4.1.0)
- Reason: `CheckpointFileManager` moved packages between 4.1.0 and 4.1.1
- Impact: If using GPU acceleration via spark-rapids, prefer **Spark 4.1.0** over 4.1.1

**SPARK-54730**:
- DataFrame column resolution conflicts with Delta on Spark 4.1+
- Fixed in Spark 4.2.0 and 4.1.1
- Impact: If you hit column resolution ambiguity errors with Delta tables, upgrade to Spark 4.1.1+ (but note spark-rapids caveat above)

### 5.5 Reddit and Community Forums

No dedicated Reddit discussions were found for the specific "Spark 4.1 + Delta Lake 4.2" combination. These versions are too new (Delta 4.2.0 released April 2026, Spark 4.1.0 late 2025/early 2026). The community evidence currently lives in blog tutorials, GitHub repos, and official release testing instructions rather than forum discussions.

---

## 6. Production Readiness Assessment

### 6.1 Delta Lake

| Aspect | Status | Notes |
|--------|--------|-------|
| ACID transactions | Production-ready | Core feature since 1.0 |
| MERGE INTO | Production-ready | Since 0.7.0, extensive optimizations |
| UPDATE / DELETE | Production-ready | Since 0.7.0 |
| Change Data Feed | Production-ready | Since 2.0.0 |
| Schema evolution | Production-ready | Including MERGE WITH SCHEMA EVOLUTION |
| Catalog-managed tables | Preview | Requires Unity Catalog |
| Delta Connect (Spark Connect) | Preview | Since 4.0.0 |
| Spark 4.2 support | Not released | Master branch compiles, no release |

### 6.2 Apache Spark Declarative Pipelines

| Aspect | Status | Notes |
|--------|--------|-------|
| Basic pipeline execution | Production-ready | Shipped in Spark 4.1 |
| Python API (`@dp.*`) | Production-ready | Core decorators work |
| SQL API (`CREATE STREAMING TABLE`) | Production-ready | Parser + execution work |
| `spark.sql()` in flow functions | Production-ready | Since SPARK-54020 (Spark 4.1) |
| Incremental processing | Production-ready | Core streaming support |
| Dependency resolution | Production-ready | Automatic DAG building |
| AUTO CDC | Not available | Planned for future |
| Data quality expectations | Not available | Databricks-only |
| Continuous execution | Not available | Planned for future |
| `spark-pipelines` CLI | Production-ready | init/run/dry-run |
| Community maturity | Early | Very new, limited real-world reports |

### 6.3 Key Risks

1. **No Spark 4.2 + Delta Lake pairing:** You cannot use the latest Spark preview with a released Delta version today.
2. **SDP is young:** Donated in mid-2025, shipped in early 2026. Limited community battle-testing outside Databricks.
3. **Feature gap vs Lakeflow:** AUTO CDC, expectations, scheduling, and Auto Loader are Databricks-only. If you need these, you either build them yourself or use Databricks.
4. **Delta DML not native to SDP:** MERGE INTO / UPDATE / DELETE work as standalone Spark SQL but are not first-class SDP pipeline constructs. You must orchestrate them outside SDP flow definitions (e.g., foreachBatch, scheduled jobs).
5. **Spark 4.1.1 + spark-rapids:** If using NVIDIA GPU acceleration, Delta Lake is excluded for Spark 4.1.1 specifically due to `CheckpointFileManager` package move. Use Spark 4.1.0 instead.
6. **SPARK-54730 column resolution bug:** DataFrame column resolution conflicts with Delta on 4.1+. Fixed in 4.1.1 and 4.2.0. If you hit ambiguity errors, upgrade — but balance against the spark-rapids caveat above.

---

## 7. Recommended Stack

For the most stable open-source combination supporting both SDP and Delta Lake SQL operations:

```
Apache Spark:   4.1.0 (latest GA with SDP)
Delta Lake:     4.2.0 (latest GA, built for Spark 4.1.0)
Java:           17+
Scala:          2.13
Python:         3.10+
```

Maven coordinates:
```xml
<dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-sql_2.13</artifactId>
    <version>4.1.0</version>
</dependency>
<dependency>
    <groupId>io.delta</groupId>
    <artifactId>delta-spark_4.1_2.13</artifactId>
    <version>4.2.0</version>
</dependency>
```

PySpark install:
```bash
pip install "pyspark[pipelines]==4.1.0" "delta-spark==4.2.0"
```

Spark session configuration:
```python
SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
```

---

## 8. Key GitHub References

### Delta Lake
- [delta-io/delta releases](https://github.com/delta-io/delta/releases) -- All Delta versions
- [Delta Lake 4.2.0 release](https://github.com/delta-io/delta/releases/tag/v4.2.0) -- Latest release
- [delta-io/delta#5326](https://github.com/delta-io/delta/issues/5326) -- Spark 4.0+ upgrade tracking
- [delta-io/delta#6345](https://github.com/delta-io/delta/issues/6345) -- Delta 4.2.0 release cut

### Apache Spark SDP
- [apache/spark PR #50875](https://github.com/apache/spark/pull/50875) -- SQL syntax for pipelines
- [apache/spark PR #53024](https://github.com/apache/spark/pull/53024) -- spark.sql() in flow functions
- [SDP Programming Guide](https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html)
- [SPARK-54730](https://issues.apache.org/jira/browse/SPARK-54730) -- DataFrame column resolution conflict with Delta (fixed in 4.1.1/4.2.0)

### Community Evidence
- [Delta 4.2.0 RC testing (issue #6345)](https://github.com/delta-io/delta/issues/6345) -- Official RC testing with Spark 4.1.1
- [Sandipan Ghosh: SDP in Spark 4.1+](https://medium.com) -- Full SDP + Delta Lake walkthrough
- [Christian Hansen: Delta Lake 4.1](https://medium.com) -- Delta 4.1.0 built for Spark 4.1.0
- [Ivan Kurchenko: Spark 4 by example](https://levelup.gitconnected.com) -- Working SDP batch/streaming examples
- [IvannKurchenko/blog-spark-4.1-declarative-pipelines](https://github.com/IvannKurchenko/blog-spark-4.1-declarative-pipelines) -- Companion GitHub repo
- [shauryashaurya/rocket-ship](https://github.com/shauryashaurya/rocket-ship) -- SDP reinsurance pipeline with Delta format
- [NVIDIA spark-rapids PR #14120](https://github.com/NVIDIA/spark-rapids/pull/14120) -- Delta Lake excluded for Spark 4.1.1

### Documentation
- [Delta Lake docs](https://docs.delta.io/latest/delta-update.html) -- MERGE/UPDATE/DELETE
- [Delta Lake versioning](https://docs.delta.io/versioning) -- Feature compatibility
- [Databricks Lakeflow SDP](https://docs.databricks.com/en/delta-live-tables/index.html) -- Commercial extension
- [Auto CDC blog](https://community.databricks.com/t5/technical-blog/from-150-lines-of-merge-into-to-7-lines-of-sql-auto-cdc-comes-to/ba-p/155355) -- AUTO CDC overview

---

## 10. PySpark + Delta Lake: Real Working Examples

### 10.1 Case 1 — Pure PySpark + Delta Lake (Standard Approach)

This is the most common pattern. Delta Lake SQL DML works via `spark.sql()` or the `DeltaTable` API in any PySpark session.

**Setup:**

```bash
pip install "pyspark==4.1.0" "delta-spark==4.2.0"
```

**SparkSession configuration:**

```python
import pyspark
from pyspark.sql import SparkSession

builder = SparkSession.builder \
    .appName("DeltaLakeExample") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

# If using delta-spark package (recommended for PySpark):
from delta import configure_spark_with_delta_pip
spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

**MERGE INTO via PySpark SQL:**

```python
# Create target table
data = spark.createDataFrame([
    (1, "Alice", 30), (2, "Bob", 25), (3, "Charlie", 35)
], ["id", "name", "age"])
data.write.format("delta").mode("overwrite").save("/tmp/delta/people")

# Create updates DataFrame
updates = spark.createDataFrame([
    (1, "Alice", 31),   # matched — update age
    (4, "Diana", 28)    # not matched — insert
], ["id", "name", "age"])
updates.write.format("delta").mode("append").save("/tmp/delta/people_updates")

# MERGE INTO via SQL
spark.sql("""
    MERGE INTO delta.`/tmp/delta/people` target
    USING delta.`/tmp/delta/people_updates` source
    ON target.id = source.id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

spark.read.format("delta").load("/tmp/delta/people").show()
```

**UPDATE via PySpark SQL:**

```python
spark.sql("UPDATE delta.`/tmp/delta/people` SET age = age + 1 WHERE name = 'Alice'")
```

**DELETE via PySpark SQL:**

```python
spark.sql("DELETE FROM delta.`/tmp/delta/people` WHERE age < 30")
```

**DeltaTable API (programmatic, no SQL strings):**

```python
from delta.tables import DeltaTable
from pyspark.sql.functions import col, lit

deltaTable = DeltaTable.forPath(spark, "/tmp/delta/people")

# Update with condition
deltaTable.update(
    condition=col("name") == "Bob",
    set={"age": lit(26)}
)

# Merge with full control
deltaTable.alias("target") \
    .merge(updates.alias("source"), "target.id = source.id") \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()

# Delete
deltaTable.delete(col("age") < 30)
```

### 10.2 Case 2 — PySpark SDP + Delta Lake (Declarative Pipelines)

SDP manages pipeline orchestration. Tables are Delta-backed by default when Delta extensions are configured.

**Pipeline file (`spark-pipeline.yml`):**

```yaml
name: etl_pipeline
libraries:
  - glob:
      include: transformations/**
configuration:
  spark.sql.extensions: io.delta.sql.DeltaSparkSessionExtension
  spark.sql.catalog.spark_catalog: org.apache.spark.sql.delta.catalog.DeltaCatalog
  spark.sql.shuffle.partitions: "1000"
```

**Pipeline code (`transformations/etl.py`):**

```python
from pyspark import pipelines as dp

# Bronze layer: ingest streaming data
@dp.table
def bronze_events():
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "events")
        .load()
    )

# Silver layer: materialized view (batch full-refresh)
@dp.materialized_view(name="silver_events")
def silver_events():
    return spark.sql("""
        SELECT
            CAST(value AS STRING) AS payload,
            timestamp,
            topic
        FROM bronze_events
        WHERE topic = 'events'
    """)

# Gold layer: aggregated materialized view
@dp.materialized_view(name="gold_event_counts")
def gold_event_counts():
    return spark.sql("""
        SELECT topic, COUNT(*) AS event_count
        FROM silver_events
        GROUP BY topic
    """)

# Multiple append flows to a single target
dp.create_streaming_table("consolidated_events")

@dp.append_flow(target="consolidated_events")
def append_web():
    return spark.readStream.table("web_events")

@dp.append_flow(target="consolidated_events")
def append_mobile():
    return spark.readStream.table("mobile_events")
```

**Run the pipeline:**

```bash
spark-pipelines init --name etl_pipeline
spark-pipelines run
```

### 10.3 Case 3 — SDP + Delta MERGE INTO via foreachBatch

This is the practical pattern for CDC/upsert within SDP-managed streaming pipelines.

```python
from pyspark import pipelines as dp
from delta.tables import DeltaTable

# SDP manages ingestion
@dp.table
def bronze_changes():
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "cdc_changes")
        .load()
    )

# Delta MERGE handles the upsert (outside SDP, via foreachBatch)
def upsert_to_silver(micro_batch_df, batch_id):
    micro_batch_df.createOrReplaceTempView("batch_updates")
    spark.sql("""
        MERGE INTO silver_changes target
        USING batch_updates source
        ON target.change_id = source.change_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

# Wire foreachBatch into the streaming query
(
    bronze_changes()
    .writeStream
    .foreachBatch(upsert_to_silver)
    .option("checkpointLocation", "/tmp/checkpoints/silver_changes")
    .start()
)
```

### 10.4 Case 4 — Pure PySpark SQL (No SDP, No Pipeline)

For simple ETL jobs that don't need pipeline orchestration:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Read source, transform, write to Delta
(spark.read.format("json").load("/data/raw/events")
    .withColumnRenamed("event_type", "type")
    .write.format("delta").mode("overwrite").save("/data/silver/events"))

# MERGE INTO
spark.sql("""
    MERGE INTO silver.events target
    USING golden.events source
    ON target.event_id = source.event_id
    WHEN MATCHED AND source.is_deleted THEN DELETE
    WHEN MATCHED THEN UPDATE SET target.type = source.type
    WHEN NOT MATCHED THEN INSERT *
""")

# Time travel
spark.read.format("delta").option("timestampAsOf", "2026-05-15").load("/data/silver/events").show()
```

### 10.5 Summary: Which Approach When

| Scenario | Approach | MERGE INTO | Pipeline Orchestration |
|----------|----------|------------|----------------------|
| Simple one-off ETL | Case 4 (Pure PySpark SQL) | Yes, via `spark.sql()` | No |
| Full CDC streaming pipeline | Case 3 (SDP + foreachBatch) | Yes, in foreachBatch | SDP handles ingestion |
| Declarative batch transforms | Case 2 (SDP) | Not inside flows | Full DAG orchestration |
| Standard Delta operations | Case 1 (Pure PySpark) | Yes, both SQL + API | No |

---

## 9. Conclusion

**Delta Lake + Spark Declarative Pipelines can work together, but with important caveats:**

1. **Delta Lake provides** the storage layer with full SQL DML (MERGE INTO, UPDATE, DELETE). This is production-ready and works with any Spark session.

2. **SDP provides** the pipeline orchestration layer with declarative table definitions and automatic dependency tracking. Tables are Delta-backed by default.

3. **The gap:** SDP does not natively incorporate Delta DML as pipeline constructs. You run MERGE/UPDATE/DELETE as standard Spark SQL, not as SDP flow definitions. For CDC use cases, Databricks has AUTO CDC (not open-source).

4. **Spark 4.2 is not yet an option** for Delta Lake -- use Spark 4.1.0 + Delta 4.2.0 for the best available combination.

5. **Production readiness:** Delta Lake is fully production-ready. SDP is functional but early-stage -- expect rough edges and missing features compared to Databricks Lakeflow.
