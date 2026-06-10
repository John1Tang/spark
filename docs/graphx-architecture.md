---
title: GraphX Architecture Guide
---

# GraphX Architecture Guide

GraphX is Apache Spark's component for graph-parallel computation. It provides a unified API for graph construction, transformation, and algorithm execution on top of Spark RDDs.

## 1. Overview

GraphX extends the Spark RDD abstraction with a **Resilient Distributed Property Graph** — a directed multigraph where each vertex and edge carries a user-defined property (attribute). The design principle is **edge-cut partitioning**: edges are partitioned across workers, and vertices are replicated as needed.

Key design goals:
- **Composability**: graphs are immutable; transformations produce new graphs
- **Colocation**: edges sharing a source vertex are colocated for efficient local computation
- **Indexing**: clustered indexes enable fast vertex-to-edge lookups within partitions
- **Lazy replication**: vertex attributes are shipped to edge partitions only when needed

## 2. Core Data Structures

### 2.1 Type Aliases

Defined in `package.scala`:

```scala
type VertexId = Long        // 64-bit unique vertex identifier
type PartitionID = Int      // Edge partition identifier (must be < 2^30)
```

### 2.2 Edge

```scala
case class Edge[ED](srcId: VertexId, dstId: VertexId, attr: ED)
```

A directed edge from `srcId` to `dstId` with an attribute of type `ED`. Self-edges (src == dst) and parallel edges (multiple edges between same vertices) are permitted.

### 2.3 EdgeTriplet

```scala
class EdgeTriplet[VD, ED] extends Edge[ED] {
  var srcId: VertexId        // inherited from Edge
  var dstId: VertexId        // inherited from Edge
  var srcAttr: VD            // source vertex attribute
  var dstAttr: VD            // destination vertex attribute
  var attr: ED               // edge attribute (inherited)
}
```

Joins vertex attributes onto an edge, providing a denormalized view for computation. Used in `mapTriplets` and `aggregateMessages`.

### 2.4 EdgeContext

```scala
abstract class EdgeContext[VD, ED, A] {
  def srcId: VertexId
  def dstId: VertexId
  def srcAttr: VD
  def dstAttr: VD
  def attr: ED
  def sendToSrc(msg: A): Unit
  def sendToDst(msg: A): Unit
}
```

Used in `aggregateMessages` to send messages to neighboring vertices. The concrete implementation `AggregatingEdgeContext[VD, ED, A]` uses arrays and `BitSet` for efficient in-partition message aggregation.

### 2.5 VertexRDD

```scala
abstract class VertexRDD[VD] extends RDD[(VertexId, VD)]
```

A specialized RDD that guarantees:
- Each entry has a **unique** `VertexId` (no duplicates)
- Vertices are **pre-indexed** for fast joins with edge partitions
- Co-partitioned with their associated `EdgeRDD` for efficient zip-partition joins

Key methods: `mapValues`, `leftJoin`, `innerJoin`, `aggregateUsingIndex`, `diff`, `minus`, `reindex`, `withEdges`.

### 2.6 EdgeRDD

```scala
abstract class EdgeRDD[ED] extends RDD[Edge[ED]]
```

A specialized RDD backed by `RDD[(PartitionID, EdgePartition[ED, VD])]`. Provides columnar storage and efficient per-partition operations.

Key methods: `mapValues`, `reverse`, `innerJoin`.

### 2.7 Graph

```scala
abstract class Graph[VD: ClassTag, ED: ClassTag] {
  def vertices: VertexRDD[VD]
  def edges: EdgeRDD[ED]
  def triplets: RDD[EdgeTriplet[VD, ED]]
  // ... transformations and operations
}
```

The top-level abstraction. A `Graph` encapsulates both vertex and edge RDDs, along with operations for transformation, computation, and persistence.

## 3. Inheritance Hierarchy

```
                    org.apache.spark.rdd.RDD[Edge[ED]]
                                    |
                          abstract class EdgeRDD[ED]
                                    |
                          class EdgeRDDImpl[ED, VD]
                          (wraps RDD[(PartitionID, EdgePartition[ED, VD])])


                org.apache.spark.rdd.RDD[(VertexId, VD)]
                                    |
                       abstract class VertexRDD[VD]
                                    |
                       class VertexRDDImpl[VD]
                       (wraps RDD[ShippableVertexPartition[VD]])


                          abstract class Graph[VD, ED]
                                    |
                          class GraphImpl[VD, ED]
                          (holds vertices + ReplicatedVertexView)


           GraphOps[VD, ED] (implicit wrapper via graphToGraphOps)
           Provides convenience methods: numEdges, degrees, algorithms
```

## 4. Internal Storage

### 4.1 EdgePartition

The fundamental unit of edge storage in `impl/EdgePartition.scala`:

```scala
class EdgePartition[ED, VD](
  localSrcIds:    Array[Int],            // compact local src indices
  localDstIds:    Array[Int],            // compact local dst indices
  data:           Array[ED],             // edge attributes
  index:          GraphXPrimitiveKeyOpenHashMap[VertexId, Int], // clustered index
  vertexAttrs:    VertexAttributeBlock[VD], // replicated vertex attributes
  activeSet:      Option[BitSet],        // optional active vertex set
  global2local:   GraphXPrimitiveKeyOpenHashMap[VertexId, Int],
  local2global:   Array[VertexId]
)
```

Key design decisions:
- **Columnar format**: three parallel arrays (`localSrcIds`, `localDstIds`, `data`) instead of `Edge` objects. Improves cache locality and reduces GC overhead.
- **Local indices**: global `VertexId` (Long) compressed to compact local integer indices within the partition. Saves memory and speeds array lookups.
- **Clustered index**: `index` maps global source vertex ID to the starting offset in the edge arrays. All edges from the same source are contiguous.
- **Vertex attribute block**: optional replicated vertex attributes, shipped on demand via `ReplicatedVertexView`.

### 4.2 VertexPartition

```scala
class VertexPartition[VD](
  index:  VertexIdToIndexMap,   // OpenHashSet[VertexId] -> array position
  values: Array[VD],            // vertex attributes
  mask:   BitSet                // active/inactive vertices
)
```

`ShippableVertexPartition[VD]` extends this with a `routingTable: RoutingTablePartition` that tracks which edge partitions reference which vertices.

### 4.3 RoutingTablePartition

Encodes vertex-to-edge-partition dependencies:

```scala
// For each vertex partition, per edge partition:
(Array[VertexId], BitSet srcFlags, BitSet dstFlags)
```

Enables **targeted shipping**: a vertex attribute is only sent to the edge partitions that actually need it, in the position (src or dst) where it's referenced.

### 4.4 ReplicatedVertexView

```scala
class ReplicatedVertexView[VD, ED](
  edges: EdgeRDDImpl[ED, VD],
  hasSrcId: Boolean,
  hasDstId: Boolean
)
```

Manages lazy replication of vertex attributes into edge partitions:
- `hasSrcId` / `hasDstId` track whether source/destination vertex attributes have been shipped
- `upgrade(vertices, toSrc, toDst)` ships attributes on demand
- `updateVertices(changedVertices)` ships only deltas when vertex attributes change incrementally

## 5. Partition Strategies

Defined in `PartitionStrategy.scala`:

```scala
trait PartitionStrategy {
  def getPartition(src: VertexId, dst: VertexId, numParts: PartitionID): PartitionID
}
```

| Strategy | Algorithm | Vertex Replication Bound |
|----------|-----------|-------------------------|
| **EdgePartition2D** | 2D hash of sparse adjacency matrix into `sqrt(N) x sqrt(N)` grid. Uses prime mixing hash `1125899906842597L` | At most `2 * sqrt(N)` |
| **EdgePartition1D** | `abs(src * mixingPrime) % numParts` | Unbounded (depends on degree distribution) |
| **RandomVertexCut** | `abs((src, dst).hashCode()) % numParts` | Unbounded |
| **CanonicalRandomVertexCut** | `abs(min(src,dst), max(src,dst)).hashCode() % numParts` | Unbounded |

**EdgePartition2D** is the recommended default — it bounds vertex replication regardless of graph topology.

## 6. Integration with Spark RDDs

GraphX is built entirely on Spark RDDs:

1. **VertexRDD** is an `RDD[(VertexId, VD)]` partitioned by `HashPartitioner`. Each partition is a `ShippableVertexPartition`.
2. **EdgeRDD** is an `RDD[(PartitionID, EdgePartition[ED, VD])]`. Each partition is an `EdgePartition`.
3. **Co-partitioning**: when a `VertexRDD` and `EdgeRDD` share the same partitioner, joins use `zipPartitions` (no shuffle).
4. **Columnar execution**: within each partition, data is stored in parallel arrays rather than as objects for cache efficiency.

## 7. Graph Construction Paths

| Method | Description |
|--------|-------------|
| `Graph(vertices, edges)` | Construct from existing vertex and edge RDDs |
| `Graph.fromEdges(edges, defaultValue)` | Build from edge RDD; create vertices with default attribute |
| `Graph.fromEdgeTuples(rawTuples, defaultValue)` | Build from `(srcId, dstId)` tuples |
| `GraphLoader.edgeListFile(sc, path)` | Parse edge-list text files directly |
| `graph.partitionBy(strategy, numParts)` | Re-partition edges using a different strategy |

## 8. Computation Primitives

### 8.1 aggregateMessages

The core computation primitive for neighbor communication:

```scala
def aggregateMessages[A: ClassTag](
  sendMsg: EdgeContext[VD, ED, A] => Unit,
  mergeMsg: (A, A) => A,
  tripletFields: TripletFields = TripletFields.All
): VertexRDD[A]
```

Flow:
1. `replicatedVertexView.upgrade()` ships vertex attributes to edge partitions
2. Each `EdgePartition` scans edges, calling `sendMsg` with an `EdgeContext`
3. Messages are aggregated in-place using arrays + BitSet
4. Results are aggregated using `vertices.aggregateUsingIndex()`

### 8.2 Pregel

Bulk-synchronous parallel message-passing API:

```scala
def Pregel[VD](
  graph: Graph[VD, _],
  initialMsg: VD,
  maxIter: Int = Int.MaxValue,
  activeDirection: EdgeDirection = EdgeDirection.Either
)(
  vprog: (VertexId, VD, VD) => VD,
  sendMsg: EdgeTriplet[VD, _] => Iterator[(VertexId, VD)],
  mergeMsg: (VD, VD) => VD
): Graph[VD, _]
```

Each superstep:
1. Apply `vprog` to vertices that received messages
2. Run `aggregateMessages` with `sendMsg` to compute new messages
3. Repeat until no messages remain or max iterations reached

Active direction optimization: only processes edges connected to vertices that received messages in the previous round.
