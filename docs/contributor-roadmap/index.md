---
layout: page
title: "Spark Contributor Roadmap 2026"
nav_order: 100
has_children: true
---

# Spark Contributor Roadmap 2026

This guide is for developers who want to start contributing to Apache Spark but don't know where
to begin. It provides an overview of the project's current priorities, open opportunities for
impactful contributions, and notable past contributions to study as examples.

## Quick Start

1. **Set up your environment** — Follow the [building Spark guide](building-spark.md) to get
   SBT, Java, and Scala configured.
2. **Read the development workflow** — See [CLAUDE.md](../CLAUDE.md) for git remote setup,
   fork conventions, and testing guidelines.
3. **Pick a component** — Focus on one area (SQL, PySpark, Connect, Streaming, etc.) and study
   existing PRs in that area.
4. **Start small** — Fix a bug, add a test, or improve documentation before tackling large features.
5. **Join the community** — Subscribe to the [dev mailing list](https://lists.apache.org/list.html?dev@spark.apache.org)
   and attend community meetups.

## Where to Find Work

- **JIRA**: [issues.apache.org/jira/browse/SPARK](https://issues.apache.org/jira/projects/SPARK/issues)
  - Filter by `status = Open AND resolution = Unresolved ORDER BY votes DESC` for community-prioritized issues
  - Look for issues labeled `good-first-issue` or `starter`
- **GitHub PRs**: [github.com/apache/spark/pulls](https://github.com/apache/spark/pulls)
  - Study recently merged PRs to understand review expectations
- **Mailing list**: [dev@spark.apache.org](https://lists.apache.org/list.html?dev@spark.apache.org)
  - Design discussions, RFCs, and community direction-setting

## Current Development Themes (2026)

The Spark community is actively working on several major areas:

| Theme | Description | Key Components |
|-------|-------------|----------------|
| **Spark Connect** | Decoupled client-server architecture for unified Spark access | `CONNECT`, `SQL` |
| **Python Modernization** | Type hints, Arrow integration, pandas API parity | `PYTHON`, `PS` |
| **SQL Acceleration** | Vectorized execution, columnar processing, Parquet optimizations | `SQL` |
| **Structured Streaming** | Error handling, state management, new APIs | `SS` |
| **Data Source V2** | Table APIs, catalog improvements, MERGE/DELETE support | `SQL` |
| **Geospatial** | Native geospatial types and ST_* functions | `SQL`, `GEO` |
| **Declarative Pipelines** | SQL scripting, pipeline execution | `SDP`, `SQL` |
| **Performance** | Window frames, single-pass analyzer, IO_URING transport | `SQL`, `CORE` |

## Databricks vs Open-Source Spark

Understanding the relationship between Databricks (the commercial platform) and Apache Spark
(the open-source project) is essential for contributors. See [Databricks vs Open-Source Spark](databricks-vs-opensource-spark.md)
for a detailed comparison of proprietary features, performance differences, and where your
contributions can have the most impact.
