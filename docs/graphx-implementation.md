---
title: GraphX Implementation Guide
---

# GraphX Implementation Guide

This document traces the implementation of GraphX from user-facing APIs down to internal execution, covering key function call chains and data flow.

## 1. Source File Organization

```
graphx/src/main/scala/org/apache/spark/graphx/
├── Edge.scala                        # Edge case class
├── EdgeContext.scala                 # EdgeContext abstract class + AggregatingEdgeContext impl
├── EdgeDirection.scala               # In, Out, Either, Both enum
├── EdgeRDD.scala                     # EdgeRDD abstract class
├── EdgeTriplet.scala                 # EdgeTriplet class (Edge + srcAttr + dstAttr)
├── Graph.scala                       # Graph abstract class + companion object
├── GraphLoader.scala                 # Load graphs from edge-list files
├── GraphOps.scala                    # Convenience ops + algorithm wrappers
├── GraphXUtils.scala                 # Kryo registration + mapReduceTriplets proxy
├── package.scala                     # VertexId, PartitionID type aliases
├── PartitionStrategy.scala           # Edge partitioning strategies
├── Pregel.scala                      # Bulk-synchronous message-passing API
├── VertexRDD.scala                   # VertexRDD abstract class
├── impl/
│   ├── EdgePartition.scala           # Columnar edge storage + message aggregation
│   ├── EdgePartitionBuilder.scala    # Builds EdgePartition from raw edges
│   ├── EdgeRDDImpl.scala             # Concrete EdgeRDD implementation
│   ├── GraphImpl.scala               # Concrete Graph implementation
│   ├── ReplicatedVertexView.scala    # Manages vertex attribute shipping
│   ├── RoutingTablePartition.scala   # Tracks vertex-to-edge-partition dependencies
│   ├── ShippableVertexPartition.scala # VertexPartition with routing tables
│   ├── VertexPartition.scala         # Basic vertex partition
│   ├── VertexPartitionBase.scala     # Abstract base for vertex partitions
│   ├── VertexPartitionBaseOps.scala  # Operations on vertex partitions
│   └── VertexRDDImpl.scala           # Concrete VertexRDD implementation
├── lib/
│   ├── ConnectedComponents.scala     # Connected components algorithm
│   ├── LabelPropagation.scala        # Community detection
│   ├── PageRank.scala                # PageRank variants
│   ├── ShortestPaths.scala           # BFS to landmarks
│   ├── StronglyConnectedComponents.scala # SCC detection
│   ├── SVDPlusPlus.scala             # Matrix factorization
│   └── TriangleCount.scala           # Triangle counting
└── util/
    ├── GraphGenerators.scala         # Synthetic graph generators
    └── PeriodicGraphCheckpointer.scala # Checkpointing for iterative algorithms
```

## 2. Graph Construction

### 2.1 From Existing RDDs

```
Graph(vertices: RDD[(VertexId, VD)], edges: RDD[Edge[ED]])
  -> Graph.apply() [Graph.scala companion]
     -> GraphImpl.fromExistingRDDs(verticesRDD, edgeRDD)
        -> new GraphImpl(vertices, ReplicatedVertexView(edges))
```

### 2.2 From Edge RDD

```
Graph.fromEdges(edges: RDD[Edge[ED]], defaultValue: VD, ...)
  -> GraphImpl(edges: RDD[Edge[ED]], defaultVertexAttr: VD, ...)
     -> EdgeRDD.fromEdges(edges)
        -> edges.mapPartitionsWithIndex { (pid, iter) =>
             val builder = new EdgePartitionBuilder[ED, VD](pid)
             iter.foreach(e => builder.add(e.srcId, e.dstId, e.attr))
             (pid, builder.toEdgePartition)  // sorts edges, builds columnar arrays
           }
        -> EdgeRDD.fromEdgePartitions(edgeParts)
           -> new EdgeRDDImpl(edgePartitions)
     -> VertexRDD.fromRDDs(vertices, edgeRDD, defaultVal)
        -> creates routing tables
        -> builds ShippableVertexPartition
     -> new GraphImpl(vertices, ReplicatedVertexView(edges))
```

### 2.3 From Edge Tuples

```
Graph.fromEdgeTuples(rawEdges: RDD[(VertexId, VertexId)], defaultValue: VD, ...)
  -> Count edge degrees via aggregate
  -> Build EdgePartition with degree computation
  -> Optionally partition and merge duplicate edges
  -> Create vertices from edge endpoints with default value
```

### 2.4 From Edge-List File

```
GraphLoader.edgeListFile(sc, path, canonicalOrientation, ...)
  -> Reads text file directly
  -> Parses (srcId, dstId) lines into EdgePartitionBuilder
  -> Builds EdgePartition in a single pass
  -> Creates GraphImpl.fromEdgePartitions(...)
```

This is the fastest construction path — bypasses `Edge` object creation entirely.

## 3. Core Transformation Chain

### 3.1 mapVertices

```
graph.mapVertices((vid, vdata) => newAttr)
  -> GraphImpl.mapVertices()
     -> if type unchanged:
          val newVerts = vertices.mapValues(...)
          // Use diff() to find changed vertices
          val changed = vertices.diff(newVerts)
          // Ship only deltas to edge partitions
          replicatedVertexView.updateVertices(changed)
        else:
          // Full re-replication needed
          replicatedVertexView = ReplicatedVertexView(replicatedVertexView.edges, false, false)
          replicatedVertexView.upgrade(newVerts, true, true)
     -> new GraphImpl(newVerts, replicatedVertexView)
```

### 3.2 mapEdges / mapTriplets

```
graph.mapEdges((edge) => newAttr)
  -> GraphImpl.mapEdges()
     -> val newEdges = edges.mapValues(edgeFunc)
     -> GraphImpl.fromExistingRDDs(vertices, newEdges)

graph.mapTriplets((triplet) => newAttr)
  -> GraphImpl.mapTriplets()
     -> replicatedVertexView.upgrade(vertices, toSrc=true, toDst=true)
        // Ship both src and dst attributes
     -> val newEdges = edges.mapTriplets(tripletFunc)
     -> GraphImpl.fromExistingRDDs(vertices, newEdges)
```

### 3.3 subgraph

```
graph.subgraph(epred, vpred)
  -> val newVerts = vertices.filter(vpred)
  -> val newEdges = edges.filter(epred)
  -> mask(newVerts)  // removes edges with missing endpoints
```

### 3.4 outerJoinVertices

```
graph.outerJoinVertices(otherVerts)(func)
  -> GraphImpl.outerJoinVertices()
     -> val newVerts = vertices.leftJoin(otherVerts)(func)
     -> if incremental replication possible:
          replicatedVertexView.updateVertices(vertices.diff(newVerts))
        else:
          replicatedVertexView = ReplicatedVertexView(replicatedVertexView.edges, false, false)
          replicatedVertexView.upgrade(newVerts, true, true)
     -> new GraphImpl(newVerts, replicatedVertexView)
```

## 4. aggregateMessages — Deep Dive

This is the core computation primitive. All graph algorithms are built on top of it.

### 4.1 Call Chain

```
graph.aggregateMessages(sendMsg, mergeMsg, tripletFields)
  -> Graph.aggregateMessages()
     -> GraphImpl.aggregateMessagesWithActiveSet(sendMsg, mergeMsg, tripletFields, activeSet=None)

Inside GraphImpl.aggregateMessagesWithActiveSet:
  1. vertices.cache()
  2. replicatedVertexView.upgrade(vertices, useSrc, useDst)
     -> vertices.shipVertexAttributes(shipSrc, shipDst)
        -> For each vertex partition:
           -> routingTable.foreachWithinEdgePartition(pid, shipSrc, shipDst) { vid =>
                emit (pid, VertexAttributeBlock(vids, attrs))
              }
        -> zipPartitions(edges, shippedAttrs):
           -> edgePartition.updateVertices(shippedAttrs)

  3. view.edges.partitionsRDD.mapPartitions { (pid, edgePart) =>
       // Choose scan method based on active fraction:
       if (activeFraction < 0.8) && (direction != Both):
         edgePart.aggregateMessagesIndexScan(...)  // indexed by source
       else:
         edgePart.aggregateMessagesEdgeScan(...)   // sequential scan
     }

  4. vertices.aggregateUsingIndex(preAgg, mergeMsg)
     -> Final reduction using vertex partition index
```

### 4.2 EdgePartition.aggregateMessagesEdgeScan

```scala
def aggregateMessagesEdgeScan(sendMsg, mergeMsg, tripletFields, activeness):
  // Create AggregatingEdgeContext with arrays + BitSet
  val context = new AggregatingEdgeContext[VD, ED, A](...)

  // Iterate all edges
  for each edge i:
    if activeness is not active for this edge: continue
    if tripletFields need srcAttr: context.setSrcAttr(vertexAttrs(srcLocalId))
    if tripletFields need dstAttr: context.setDstAttr(vertexAttrs(dstLocalId))
    context.setEdgeFields(localSrcIds(i), localDstIds(i), data(i))
    sendMsg(context)
    // context.sendToSrc/sendToDst aggregate into local arrays

  // Return (globalVertexId, aggregatedMessage) for each touched vertex
```

### 4.3 EdgePartition.aggregateMessagesIndexScan

```scala
def aggregateMessagesIndexScan(sendMsg, mergeMsg, tripletFields, activeness):
  // Uses the clustered index to skip inactive edges
  // Only iterates edges whose source vertex is in the active set
  index.foreachActiveVertex { srcId =>
    val startOffset = index(srcId)
    val endOffset = index.next(srcId)
    for i from startOffset to endOffset:
      // Same as edgeScan, but only for active edges
  }
```

## 5. Pregel — Deep Dive

File: `Pregel.scala`

```
Pregel.apply(graph, initialMsg, maxIter, activeDirection)(vprog, sendMsg, mergeMsg)
  -> g = graph.mapVertices((vid, vdata) => vprog(vid, vdata, initialMsg))
     // First superstep: apply vprog to ALL vertices with initialMsg

  -> messages = GraphXUtils.mapReduceTriplets(
       g,
       triplet => { sendMsg(triplet).map { case (vid, msg) => (vid, mergeMsg(initialMsg, msg)) } },
       mergeMsg
     )
     // Send initial messages from all edges

  -> var prevG = g
  -> var i = 0
  -> while (messages.nonEmpty && i < maxIter):
       g = g.joinVertices(messages)(vprog)
         // Update vertices that received messages
         // g.vertices.leftJoin(messages)(vprog)
       messages = GraphXUtils.mapReduceTriplets(
         g, sendMsg, mergeMsg,
         Some((oldMessages, activeDirection))
       )
         // activeDirection optimization:
         //   EdgeDirection.Out  -> only send from vertices that received messages
         //   EdgeDirection.In   -> only send to vertices that received messages
         //   EdgeDirection.Either -> send from/to active vertices
         //   EdgeDirection.Both -> send both directions
       prevG.unpersist()
       messages.cache()
       i += 1

  -> return g
```

**Key optimization**: The `activeSet` parameter in `aggregateMessagesWithActiveSet` restricts computation to only edges connected to vertices that received messages in the previous superstep. This avoids scanning the entire graph when only a subset is active.

## 6. Algorithm Implementations

### 6.1 PageRank

File: `lib/PageRank.scala`

#### Static PageRank (Fixed Iterations)

```
graph.staticPageRank(numIter, resetProb)
  -> PageRank.run(graph, numIter, resetProb)
     -> PageRank.runWithOptions(graph, numIter, resetProb, srcId=None)

     1. Initialize:
        rankGraph = graph
          .outerJoinVertices(graph.outDegrees) { (vid, _, optOutDeg) => optOutDeg.getOrElse(1) }
          .mapTriplets(e => 1.0 / e.srcAttr)  // edge weight = 1 / source out-degree
          .mapVertices { (vid, _) => 1.0 }     // initial rank = 1.0

     2. Loop numIter times:
        rankGraph = runUpdate(rankGraph):
          -> rankUpdates = rankGraph.aggregateMessages(
               ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr),  // send weighted rank
               _ + _                                          // merge by addition
             )
          -> rankGraph.outerJoinVertices(rankUpdates) { (vid, oldRank, msgSum) =>
               resetProb + (1.0 - resetProb) * msgSum.getOrElse(0.0)
             }

     3. normalizeRankSum(rankGraph)
        // Adjust for dangling nodes (nodes with no outgoing edges)
```

#### Convergent PageRank

```
graph.pageRank(tol, resetProb)
  -> PageRank.runUntilConvergence(graph, tol, resetProb)
     -> Uses Pregel with delta threshold:
          vertexProgram: (vid, (oldPR, lastDelta), msgSum) =>
            newPR = oldPR + (1.0 - resetProb) * msgSum
            delta = newPR - oldPR
            (newPR, delta)
          sendMessage: if (srcAttr._2 > tol) emit (dstId, srcAttr._2 * edgeWeight)
          mergeMsg: (a, b) => a + b
     // Only vertices whose delta exceeds the threshold trigger further computation
```

### 6.2 Connected Components

File: `lib/ConnectedComponents.scala`

```
graph.connectedComponents()
  -> ConnectedComponents.run(graph)
     1. ccGraph = graph.mapVertices { (vid, _) => vid }
        // Each vertex starts with its own ID as component ID

     2. Pregel(ccGraph, initialMsg=Long.MaxValue, activeDirection=Either)(
          vprog: (vid, attr, msg) => math.min(attr, msg),
            // Accept the smaller component ID
          sendMsg: edge =>
            if (srcAttr < dstAttr) send (dstId, srcAttr)
            else if (srcAttr > dstAttr) send (srcId, dstAttr)
            else Iterator.empty,
            // Propagate smaller ID in both directions
          mergeMsg: (a, b) => math.min(a, b)
        )
        // Converges when all vertices in a component share the minimum vertex ID

     3. Return graph where vertex attribute = connected component ID
```

### 6.3 Triangle Counting

File: `lib/TriangleCount.scala`

```
graph.triangleCount()
  -> TriangleCount.run(graph)
     1. Canonicalize the graph:
        canonicalGraph = graph
          .mapEdges(e => true)
          .removeSelfEdges()
          .convertToCanonicalEdges()
        // Ensures: srcId > dstId for all edges, no self-loops, no duplicates

     2. Collect neighbor sets:
        nbrSets = canonicalGraph.collectNeighborIds(EdgeDirection.Either)
          .mapValues { (vid, nbrs) => new VertexSet(nbrs) }
        // Build hash set of neighbors per vertex

     3. Attach neighbor sets to graph:
        setGraph = canonicalGraph.outerJoinVertices(nbrSets) { ... }

     4. Count triangles via aggregateMessages:
        counters = setGraph.aggregateMessages(
          // For each edge (src, dst):
          //   Iterate through the smaller neighbor set
          //   Count vertices that appear in BOTH neighbor sets (intersection)
          //   Send count to both src and dst
          edgeFunc: iterate smaller set, check membership in larger,
          mergeMsg: _ + _
        )

     5. Attach counts:
        graph.outerJoinVertices(counters) { (_, _, opt) => opt.getOrElse(0) }
        // Note: each triangle counted twice (once per edge), divide by 2
```

### 6.4 Strongly Connected Components

File: `lib/StronglyConnectedComponents.scala`

Three-phase iterative algorithm:

```
graph.stronglyConnectedComponents(numIter)
  1. Initialize:
     sccGraph = graph.mapVertices { (vid, _) => vid }
     sccWorkGraph = graph.mapVertices { (vid, _) => (vid, false) }.cache()

  2. Iterate up to numIter times:
     Phase 1 - Sink/Source Removal:
       Find vertices with in-degree=0 or out-degree=0
       Mark them as final, write SCC IDs to sccGraph
       Remove finalized vertices from sccWorkGraph

     Phase 2 - Forward Pregel (EdgeDirection.Out):
       Propagate minimum SCC ID forward through remaining graph
       Pregel(sccWorkGraph, Long.MaxValue, Out)(
         vprog: min(myId, receivedMin),
         sendMsg: if srcAttr < dstAttr, send srcAttr forward
       )

     Phase 3 - Reverse Pregel (EdgeDirection.In):
       Identify SCC roots by propagating backward
       Pregel(sccWorkGraph, false, In)(
         vprog: mark as final if root of color or has final same-color neighbor
       )

  3. Return sccGraph with SCC IDs
```

### 6.5 Label Propagation

File: `lib/LabelPropagation.scala`

```
graph.labelPropagation(maxIter)
  -> Initialize each vertex with its own label
  -> Pregel with message = Map[LabelId, VoteCount]:
       vprog: pick label with highest vote count
       sendMsg: emit (neighborId, Map(myLabel -> 1))
       mergeMsg: merge vote count maps
  -> Each vertex adopts the most common label among neighbors
```

## 7. Periodic Checkpointing

File: `util/PeriodicGraphCheckpointer.scala`

Iterative algorithms benefit from periodic checkpointing to truncate lineage:

```scala
val checkpointer = new PeriodicGraphCheckpointer(sc, interval=10)
for (i <- 1 to maxIter) {
  graph = runIteration(graph)
  checkpointer.update(graph)
}
checkpointer.shutdown()
```

Maintains a queue of at most 3 persisted graphs. Old checkpoint files are automatically cleaned up.

## 8. Graph Generators

File: `util/GraphGenerators.scala`

Synthetic graph generators for testing and benchmarking:

| Generator | Description |
|-----------|-------------|
| `starGraph` | One central node connected to N leaves |
| `gridGraph` | 2D grid topology |
| `logNormalGraph` | Log-normal degree distribution |
| `RMATGraph` | Recursive MATrix model (power-law graphs) |

## 9. Performance Considerations

### 9.1 When to Use Index Scan vs Edge Scan

- **Index scan**: preferred when < 80% of vertices are active. Uses the clustered index to skip inactive edges.
- **Edge scan**: preferred when > 80% active or when both directions are needed. Sequential scan is more efficient.

### 9.2 Vertex Replication

- Vertex attributes are **NOT** stored in edge partitions by default.
- `ReplicatedVertexView` ships them lazily when operations need them.
- `hasSrcId` / `hasDstId` flags track what has been shipped.
- Incremental updates (`updateVertices`) ship only changed vertices.

### 9.3 Storage Level

Both `VertexRDD` and `EdgeRDD` track a `targetStorageLevel`. Transformations preserve this target across the RDD lineage, ensuring consistent caching behavior.
