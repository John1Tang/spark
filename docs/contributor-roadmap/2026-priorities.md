---
layout: page
title: "Top 10 Open JIRA Tickets for 2026"
parent: "Spark Contributor Roadmap 2026"
nav_order: 1
---

# Top 10 Open JIRA Tickets for 2026

These are the most critical and community-demanding open JIRA tickets for Apache Spark in 2026.
They represent areas where the project needs contributors the most.

To find the current vote-sorted list, visit:
[SPARK JIRA - Open issues sorted by votes](https://issues.apache.org/jira/projects/SPARK/issues?jql=project%20%3D%20SPARK%20AND%20status%20in%20(Open%2C%20Reopened)%20ORDER%20BY%20votes%20DESC)

---

## 1. Full Python Type Hint Coverage for PySpark

**JIRA**: Search for unresolved `PYTHON` issues related to type hints
**Votes**: Very High
**Impact**: PySpark is the most popular Spark API, but type hint coverage remains incomplete.
Completing type hints across all modules enables IDE autocomplete, mypy checking, and better
developer experience.

**What to study**: Recent work like SPARK-54993 (Add type hints to NameTypeHolder classes).
Look at commits that incrementally add type hints to existing PySpark modules.

**Good starting point**: Pick a module with missing type hints and add them with tests.

---

## 2. Spark Connect Feature Parity with Classic Spark

**JIRA**: Search for unresolved `CONNECT` issues
**Votes**: High
**Impact**: Spark Connect is the future of how users interact with Spark. Many classic DataFrame
and SQL APIs are not yet available through Connect. Closing this gap is critical for adoption.

**What to study**: Recent additions like SPARK-56395 (Add NEAREST BY DataFrame API for Connect),
SPARK-52215 (Implement Scalar Arrow UDF for Connect). Study how Connect proto definitions
map to Spark Connect planner implementations.

**Good starting point**: Pick a DataFrame API that exists in classic Spark but not Connect,
and implement it following existing patterns.

---

## 3. Parquet Vectorized Reader Optimizations

**JIRA**: Search for unresolved `SQL` issues related to vectorized/Parquet
**Votes**: High
**Impact**: Parquet is the dominant storage format. The vectorized reader is the hottest
code path for SQL workloads. Bulk read paths, type-specific updaters, and dictionary
encoding improvements deliver immediate performance wins.

**What to study**: SPARK-56803 (Add bulk read+narrow path for INT64 DECIMAL to 32-bit Decimal),
SPARK-56802 (Add bulk read+widen path for FLOAT to Double). These show the pattern for
adding type-specific vectorized read paths.

**Good starting point**: Benchmark existing vectorized paths, identify missing type
combinations, and add optimized bulk readers.

---

## 4. Single-Pass Analyzer Completion

**JIRA**: Search for unresolved issues related to analyzer/optimization
**Votes**: High
**Impact**: The single-pass analyzer is a major architectural improvement that can
significantly reduce SQL compilation time. Completing this work touches the core of
Catalyst optimizer.

**What to study**: SPARK-56875 (Refactor time/session window resolution code for single-pass),
SPARK-52203 (New iteration of single-pass Analyzer functionality).

**Good starting point**: Identify remaining rules that cannot run in single-pass mode
and refactor them.

---

## 5. Arrow UDF and Arrow Table Function Enhancements

**JIRA**: Search for unresolved `PYTHON` issues related to Arrow
**Votes**: Medium-High
**Impact**: Arrow UDFs are the primary mechanism for Python-side computation in Spark SQL.
Improving performance, adding new UDF types, and fixing edge cases benefits all PySpark users
doing data science work.

**What to study**: SPARK-56717 (ASV microbenchmark for SQL_ARROW_TABLE_UDF),
SPARK-56312 (Refactor SQL_COGROUPED_MAP_ARROW_UDF), SPARK-52227 (Scalar Arrow UDF
named arguments).

**Good starting point**: Add benchmarks, fix edge cases in existing Arrow UDF types,
or add new Arrow-based DataFrame operations.

---

## 6. Advanced Window Function Support

**JIRA**: Search for unresolved `SQL` issues related to window functions
**Votes**: Medium-High
**Impact**: Window functions are essential for analytics. Completing support for non-invertible
aggregates, custom frame definitions, and new window functions (like `counter_diff`)
expands Spark's analytical capabilities.

**What to study**: SPARK-56546 (Block-chunked segment-tree window frame for non-invertible
sliding aggregates), SPARK-56820 (Add `counter_diff` window function).

**Good starting point**: Add new window functions or improve existing frame handling
for edge cases.

---

## 7. Data Source V2 MERGE and DELETE Improvements

**JIRA**: Search for unresolved `SQL` issues related to MERGE/DELETE/DSv2
**Votes**: Medium-High
**Impact**: MERGE INTO and DELETE FROM are critical for table maintenance workflows.
DSv2 improvements to these operations (group filtering, metrics, pushdown) directly
benefit lakehouse workloads.

**What to study**: SPARK-56680 (DSv2 INSERT and Insert-Only MERGE Metrics),
SPARK-56669 (Implement group filtering for WriteDelta row level operations).

**Good starting point**: Add metrics, improve error handling, or optimize execution
plans for row-level operations.

---

## 8. Geospatial Function Completeness

**JIRA**: Search for unresolved `GEO` issues
**Votes**: Medium
**Impact**: Geospatial support was recently enabled by default (SPARK-56771), but
the ST_* function library is not yet complete. This is a growing area with high
community interest from GIS and spatial analytics users.

**What to study**: SPARK-56682 (Extend ST_AsBinary with endianness),
SPARK-56813 (Refine geospatial documentation), SPARK-56770 (Update PROJ version).

**Good starting point**: Add missing ST_* functions from the OGC standard or fix
edge cases in existing ones.

---

## 9. Declarative Pipeline / SQL Scripting

**JIRA**: Search for unresolved `SDP` issues
**Votes**: Medium
**Impact**: SQL scripting and declarative pipelines enable users to write complex
multi-statement workflows directly in SQL. This is a newer area with room for
substantial contributions.

**What to study**: SPARK-52110 (SQL syntax support for pipelines),
SPARK-52129 (Improve ParserUtils for scripting constructs).

**Good starting point**: Add new SQL scripting constructs, improve error messages,
or add Connect support for pipeline features.

---

## 10. IO_URING and Transport-Level Optimizations

**JIRA**: Search for unresolved `CORE` issues related to IO/transport
**Votes**: Medium
**Impact**: IO_URING support for Linux can dramatically improve disk I/O throughput.
This is a newer initiative that touches Spark's core file handling and shuffle.

**What to study**: Look for recent PRs adding IO_URING transport mode (e.g.,
SPARK-XXXXX in recent open PRs). Study the existing transport abstraction.

**Good starting point**: This is an advanced area requiring Linux kernel I/O knowledge.
Start by reviewing existing transport implementations and benchmarking.

---

## How to Use This List

1. Pick the area that matches your expertise and interest
2. Search JIRA for related open issues
3. Study recent merged PRs in that area to understand patterns
4. Start with a small improvement (bug fix, test, benchmark) before tackling a large feature
5. Discuss your approach on the dev mailing list before starting substantial work
