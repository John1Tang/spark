---
layout: page
title: "Top 10 Notable Past Contributions to Study"
parent: "Spark Contributor Roadmap 2026"
nav_order: 2
---

# Top 10 Notable Past Contributions to Study

These are exemplary past contributions that showcase the quality, scope, and impact of
successful Spark PRs. Study these to understand what makes a great contribution:
clear problem definition, thorough testing, and community-aligned implementation.

---

## 1. NEAREST BY Top-K Ranking Join

**JIRA**: SPARK-56395
**Component**: SQL, Connect, Python
**Why study this**: This is a brand-new join type that adds top-K nearest-neighbor
matching to Spark SQL. It touches Catalyst optimizer (new logical/physical nodes),
Connect protocol, and PySpark API. A great example of a multi-layer feature
implemented across all Spark tiers.

**What to learn**:
- How to add a new SQL syntax and plan node
- How to thread the feature through Connect proto and planner
- How to expose it in the PySpark DataFrame API
- How to write golden file tests for a new query construct

---

## 2. Segment-Tree Window Frame for Non-Invertible Aggregates

**JIRA**: SPARK-56546
**Component**: SQL
**Why study this**: Window functions are among the most complex parts of Catalyst.
This PR implements a block-chunked segment-tree approach for sliding aggregates
that cannot be computed with simple subtract-and-add (e.g., COUNT DISTINCT, median).

**What to learn**:
- Advanced algorithm design within Spark's execution framework
- How to handle complex state management in window executors
- Performance optimization with chunked processing

---

## 3. Scalar Arrow UDF for Spark Connect

**JIRA**: SPARK-52215
**Component**: Python, Connect
**Why study this**: Arrow UDFs are critical for Python performance. This PR
implements scalar Arrow UDF support in Connect, requiring changes to both the
server-side execution and the client-side serialization.

**What to learn**:
- How Arrow serialization works in Spark Connect
- How to implement a UDF type from scratch
- How to write cross-language tests (Python client, Scala server)

---

## 4. Declarative Pipeline SQL Syntax

**JIRA**: SPARK-52110
**Component**: SQL, SDP
**Why study this**: This adds native SQL syntax for pipeline execution, extending
Spark's parser and execution engine. It demonstrates how to add new language
constructs to Spark SQL.

**What to learn**:
- Extending the SqlBase.g4 grammar file
- Implementing new parser rules and AST nodes
- Threading new constructs through analysis and execution

---

## 5. Schema-Level Collation Support

**JIRA**: SPARK-52219 (and related collation work)
**Component**: SQL
**Why study this**: Collation support was a long-standing gap in Spark SQL.
These PRs add schema-level collation inheritance, affecting type coercion,
comparison operations, and string functions.

**What to learn**:
- How to implement a feature that touches dozens of existing code paths
- Backward compatibility considerations
- How to handle ANSI vs. legacy mode differences

---

## 6. Pandas API on Spark — Rolling/Expanding Functions

**JIRA**: SPARK-40340, SPARK-40341, SPARK-40337
**Component**: PySpark (PS)
**Why study this**: These PRs incrementally add pandas-compatible rolling and
expanding window functions (`.sem()`, `.median()`, `.describe()`) to the
pandas API on Spark layer.

**What to learn**:
- How to mirror pandas API behavior in Spark
- The pattern for incrementally building out a pandas-compatible surface
- How to test against pandas reference implementations

---

## 7. Geospatial Support Enablement

**JIRA**: SPARK-56771 (enable by default) and related GEO issues
**Component**: SQL, GEO, Python, Connect
**Why study this**: Geospatial support is one of Spark's newest major features.
These PRs enable geospatial types and ST_* functions by default, and extend
functions like ST_AsBinary with additional options.

**What to learn**:
- How to add a new data type family to Spark SQL
- How to implement spatial functions using PROJ library
- Cross-component coordination (SQL, Python, Connect, docs)

---

## 8. DSv2 INSERT and MERGE Metrics

**JIRA**: SPARK-56680
**Component**: SQL
**Why study this**: Adding observability metrics to DSv2 write operations.
This PR demonstrates how to add telemetry and metrics to existing executors
without changing the execution logic.

**What to learn**:
- How Spark's metrics system works
- How to add new metrics to existing executors
- How to expose metrics in the UI

---

## 9. Parquet Bulk Read Path Optimizations

**JIRA**: SPARK-56803, SPARK-56802
**Component**: SQL
**Why study this**: These PRs add type-specific bulk read paths for Parquet
vectorized reader (INT64 DECIMAL narrowing, FLOAT to Double widening). They
show the pattern for hot-path optimization in Spark's most performance-sensitive
code.

**What to learn**:
- How to identify and optimize hot paths in Spark SQL
- The VectorizedReader and ColumnarBatch abstractions
- How to benchmark and measure the impact of optimizations

---

## 10. SQL Tab UI Enhancements

**JIRA**: SPARK-56809, SPARK-56811
**Component**: UI
**Why study this**: The Spark UI is the primary debugging tool for users.
These PRs add SQL description/metadata display on the execution detail page
and restore sub-execution grouping on the SQL tab listing.

**What to learn**:
- How the Spark UI is built (Scala + HTML/JS)
- How to add new UI elements that display plan metadata
- Frontend improvements to an existing Scala-based UI

---

## How to Use This List

1. **Find the PR on GitHub**: Search for the JIRA ID on [github.com/apache/spark/pulls](https://github.com/apache/spark/pulls)
2. **Read the diff**: Look at the files changed, the test additions, and the reviewer comments
3. **Understand the pattern**: Note how each contribution follows Spark's established patterns for:
   - Adding new code in the right module
   - Writing tests (unit, integration, golden file)
   - Updating documentation
   - Adding Connect support when applicable
4. **Emulate the approach**: When making your own contribution, follow the same structure and
   attention to detail that these PRs demonstrate.
