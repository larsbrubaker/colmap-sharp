// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MinSTGraphCut: port of the MinSTGraphCut class template in colmap/math/graph_cut.h, the
// minimum S-T cut of a directed graph by max-flow. Its neighbor GraphCut.cs holds the free
// functions of the same header. Delaunay meshing (mvs/delaunay_meshing) labels cells
// inside/outside with it. Tests: ColmapSharp.Tests/Mathematics/GraphCutTests.cs, plus the
// C#-only GraphCutCrossCheckTests.cs and MinSTGraphCutScalingTests.cs.
//
// COLMAP calls boost::boykov_kolmogorov_max_flow. Neither Boost's implementation nor
// Kolmogorov's own maxflow library (GPL / research-only) was read or transcribed; this is the
// algorithm written from its paper: Y. Boykov and V. Kolmogorov, "An Experimental Comparison
// of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision", IEEE PAMI 26(9), 2004 -
// search trees S and T grown from the terminals, augmentation along the path where they meet,
// and adoption of orphans, with the paper's timestamp/distance heuristic for choosing parents.
//
// Terminal links are not stored as edges. As the paper's implementation section describes,
// each node keeps one residual terminal capacity (positive: to the source, negative: to the
// sink). The flow s -> v -> t that a node can carry on its own is pushed up front, and every
// node left with terminal capacity starts in its tree and active. This matters for scale:
// Delaunay meshing gives nearly every cell a terminal link, and treating those as out-edges of
// a terminal made the terminal rescan all of them after each augmentation (quadratic).
// Growth also keeps a current arc per active node, so a node that found a path resumes its
// scan where it stopped instead of rescanning arcs it already classified.
//
// Tier A for integer capacities: the flow value is the unique max-flow, and the labels are
// too, because at termination the S tree is exactly the set of nodes reachable from the
// source in the residual graph and the T tree the set that can reach the sink; both sets are
// the same for every maximum flow. Boost colors free nodes (in neither tree) gray, which
// COLMAP reports as connected to the source; so does this. With float capacities it is
// Tier B: see docs/CPP_DIVERGENCES.md, entry 5.
//
// Translation notes:
// - The C++ template parameter node_t becomes int (node indices address arrays here).
// - value_t becomes .NET generic math, INumber<TValue>.
// - Non-terminal edges are stored in pairs, so the reverse of edge e is e ^ 1; Boost's
//   explicit reverse edge map is not needed.
// - COLMAP's index checks admit the two terminal indices (num_nodes and num_nodes + 1), so an
//   edge may touch a terminal. Such an edge becomes terminal capacity (source -> v, v -> sink)
//   or direct flow (source -> sink); edges into the source or out of the sink can carry no
//   s-t flow and are only counted.

using System.Numerics;

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of colmap::MinSTGraphCut: the minimum cut of a directed S-T graph using the
/// Boykov-Kolmogorov max-flow min-cut algorithm.
/// </summary>
public sealed class MinSTGraphCut<TValue>
	where TValue : INumber<TValue>
{
	private const byte FreeNode = 0;
	private const byte SourceTree = 1;
	private const byte SinkTree = 2;

	// Parent-edge markers for nodes that have no parent edge.
	private const int TerminalParent = -1;
	private const int OrphanParent = -2;
	private const int NoParent = -3;

	private readonly int numNodes;

	// Non-terminal edge e runs from tail[e] to head[e]; its reverse is e ^ 1.
	private readonly List<int> tail = [];
	private readonly List<int> head = [];
	private readonly List<TValue> capacity = [];

	// Terminal link capacities per node, and the capacity of edges straight from source to sink.
	private readonly TValue[] sourceCapacity;
	private readonly TValue[] sinkCapacity;
	private TValue directFlowCapacity = TValue.Zero;

	// Edges as COLMAP's Boost graph counts them, terminal edges and their reverses included.
	private int numEdges;

	// Filled by Compute: tree membership per node (FreeNode, SourceTree, SinkTree).
	private byte[] trees = [];

	/// <summary>
	/// Diagnostic for tests: the steps the last <see cref="Compute"/> took, counting every arc
	/// examined while growing trees and adopting orphans and every parent link followed while
	/// augmenting or measuring distances. Unlike wall-clock time it does not depend on machine
	/// load, so a scaling test can bound it to catch quadratic behavior.
	/// </summary>
	internal long LastComputeWork { get; private set; }

	/// <summary>Creates a graph with <paramref name="numNodes"/> nodes plus the two terminals.</summary>
	public MinSTGraphCut(int numNodes)
	{
		this.numNodes = numNodes;
		sourceCapacity = new TValue[numNodes];
		sinkCapacity = new TValue[numNodes];
		Array.Fill(sourceCapacity, TValue.Zero);
		Array.Fill(sinkCapacity, TValue.Zero);
	}

	/// <summary>The number of nodes, not counting the terminals.</summary>
	public int NumNodes => numNodes;

	/// <summary>The number of directed edges, each reverse edge counted.</summary>
	public int NumEdges => numEdges;

	private int SourceNode => numNodes;

	private int SinkNode => numNodes + 1;

	/// <summary>
	/// Connects node <paramref name="nodeIdx"/> to the source and sink terminals. A zero
	/// capacity adds no edge.
	/// </summary>
	public void AddNode(int nodeIdx, TValue sourceCapacity, TValue sinkCapacity)
	{
		Check.Ge(nodeIdx, 0);
		Check.Le(nodeIdx, numNodes + 2);
		Check.Ge(sourceCapacity, TValue.Zero);
		Check.Ge(sinkCapacity, TValue.Zero);

		if (sourceCapacity > TValue.Zero)
		{
			AddEdgePair(SourceNode, nodeIdx, sourceCapacity, TValue.Zero);
		}

		if (sinkCapacity > TValue.Zero)
		{
			AddEdgePair(nodeIdx, SinkNode, sinkCapacity, TValue.Zero);
		}
	}

	/// <summary>
	/// Adds an edge from <paramref name="nodeIdx1"/> to <paramref name="nodeIdx2"/> with
	/// <paramref name="edgeCapacity"/>, and the reverse edge with <paramref name="reverseCapacity"/>.
	/// </summary>
	public void AddEdge(int nodeIdx1, int nodeIdx2, TValue edgeCapacity, TValue reverseCapacity)
	{
		Check.Ge(nodeIdx1, 0);
		Check.Le(nodeIdx1, numNodes + 2);
		Check.Ge(nodeIdx2, 0);
		Check.Le(nodeIdx2, numNodes + 2);
		Check.Ge(edgeCapacity, TValue.Zero);
		Check.Ge(reverseCapacity, TValue.Zero);

		AddEdgePair(nodeIdx1, nodeIdx2, edgeCapacity, reverseCapacity);
	}

	/// <summary>Computes the min-cut with the max-flow algorithm and returns the flow.</summary>
	public TValue Compute()
	{
		var solver = new Solver(this);
		var flow = solver.Run(out trees);
		LastComputeWork = solver.Work;
		return flow;
	}

	/// <summary>
	/// After <see cref="Compute"/>: whether the node is on the source side of the cut. Nodes in
	/// neither search tree count as source side, as in COLMAP.
	/// </summary>
	public bool IsConnectedToSource(int nodeIdx)
	{
		return TreeOf(nodeIdx) != SinkTree;
	}

	/// <summary>After <see cref="Compute"/>: whether the node is on the sink side of the cut.</summary>
	public bool IsConnectedToSink(int nodeIdx)
	{
		return TreeOf(nodeIdx) == SinkTree;
	}

	private byte TreeOf(int nodeIdx)
	{
		if (nodeIdx == SourceNode)
		{
			return SourceTree;
		}

		if (nodeIdx == SinkNode)
		{
			return SinkTree;
		}

		return trees[nodeIdx];
	}

	private void AddEdgePair(int from, int to, TValue forwardCapacity, TValue backwardCapacity)
	{
		// The Check.Le calls admit the index one past the last terminal, as COLMAP's
		// THROW_CHECK_LE does; Boost would grow the graph there. Refuse it rather than corrupt.
		Check.That(from < numNodes + 2 && to < numNodes + 2);
		numEdges += 2;

		if (from < numNodes && to < numNodes && from != to)
		{
			tail.Add(from);
			head.Add(to);
			capacity.Add(forwardCapacity);
			tail.Add(to);
			head.Add(from);
			capacity.Add(backwardCapacity);
			return;
		}

		AddTerminalArc(from, to, forwardCapacity);
		AddTerminalArc(to, from, backwardCapacity);
	}

	// One direction of an edge that touches a terminal, or a self-loop (which carries nothing).
	private void AddTerminalArc(int from, int to, TValue arcCapacity)
	{
		if (from == SourceNode && to == SinkNode)
		{
			directFlowCapacity += arcCapacity;
		}
		else if (from == SourceNode && to < numNodes)
		{
			sourceCapacity[to] += arcCapacity;
		}
		else if (from < numNodes && to == SinkNode)
		{
			sinkCapacity[from] += arcCapacity;
		}
	}

	// One max-flow run over a snapshot of the graph, so Compute can be called again.
	private sealed class Solver
	{
		private readonly int[] head;
		private readonly TValue[] residual;
		private readonly int[] arcStart; // node p's arcs are arcs[arcStart[p] .. arcStart[p + 1])
		private readonly int[] arcs;
		private readonly TValue[] terminal; // residual terminal capacity: > 0 source, < 0 sink
		private readonly byte[] tree;
		private readonly int[] parentEdge; // edge from the node to its parent in its tree
		private readonly int[] currentArc;
		private readonly int[] timestamp;
		private readonly int[] distance;
		private readonly bool[] isActive;
		private readonly Queue<int> active = new();
		private readonly Queue<int> orphans = new();
		private TValue flow;
		private int time;

		// Steps taken, for MinSTGraphCut.LastComputeWork.
		public long Work { get; private set; }

		public Solver(MinSTGraphCut<TValue> graph)
		{
			var n = graph.numNodes;
			head = [.. graph.head];
			residual = [.. graph.capacity];

			// Compressed adjacency, each node's arcs in insertion order.
			arcStart = new int[n + 1];
			foreach (var from in graph.tail)
			{
				arcStart[from + 1]++;
			}

			for (var v = 0; v < n; v++)
			{
				arcStart[v + 1] += arcStart[v];
			}

			arcs = new int[head.Length];
			var fill = arcStart[..n];
			for (var e = 0; e < head.Length; e++)
			{
				arcs[fill[graph.tail[e]]++] = e;
			}

			terminal = new TValue[n];
			tree = new byte[n];
			parentEdge = new int[n];
			Array.Fill(parentEdge, NoParent);
			currentArc = new int[n];
			timestamp = new int[n];
			distance = new int[n];
			isActive = new bool[n];

			// Push s -> v -> t directly, then seed both trees with the nodes that still have
			// terminal capacity.
			flow = graph.directFlowCapacity;
			for (var v = 0; v < n; v++)
			{
				var toSource = graph.sourceCapacity[v];
				var toSink = graph.sinkCapacity[v];
				flow += TValue.Min(toSource, toSink);
				terminal[v] = toSource - toSink;
				if (terminal[v] > TValue.Zero)
				{
					AddToTree(v, SourceTree, TerminalParent);
				}
				else if (terminal[v] < TValue.Zero)
				{
					AddToTree(v, SinkTree, TerminalParent);
				}
			}
		}

		public TValue Run(out byte[] trees)
		{
			while (true)
			{
				var meetingEdge = Grow();
				if (meetingEdge < 0)
				{
					break;
				}

				flow += Augment(meetingEdge);
				Adopt();
			}

			trees = tree;
			return flow;
		}

		private void AddToTree(int node, byte whichTree, int parent)
		{
			tree[node] = whichTree;
			parentEdge[node] = parent;
			distance[node] = 1;
			Activate(node);
		}

		// Queues the node if it is not queued, and restarts its arc scan in any case: arcs it
		// already passed may lead somewhere new (a neighbor that was freed, for example).
		private void Activate(int node)
		{
			currentArc[node] = arcStart[node];
			if (!isActive[node])
			{
				isActive[node] = true;
				active.Enqueue(node);
			}
		}

		// Growth stage: expand the trees from active nodes until they touch. Returns the edge
		// from the source-tree side to the sink-tree side where they met, or -1 when no active
		// node is left, which means the flow is maximal.
		private int Grow()
		{
			while (active.Count > 0)
			{
				Work++;
				var p = active.Peek();
				if (tree[p] == FreeNode)
				{
					// Became free during adoption after it was queued.
					active.Dequeue();
					isActive[p] = false;
					continue;
				}

				var end = arcStart[p + 1];
				for (var k = currentArc[p]; k < end; k++)
				{
					Work++;
					var e = arcs[k];
					var q = head[e];
					// Tree capacity: parent-to-child residual in S, child-to-parent in T.
					var treeEdge = tree[p] == SourceTree ? e : e ^ 1;
					if (residual[treeEdge] <= TValue.Zero)
					{
						continue;
					}

					if (tree[q] == FreeNode)
					{
						AddToTree(q, tree[p], e ^ 1);
						distance[q] = distance[p] + 1;
						timestamp[q] = timestamp[p];
					}
					else if (tree[q] != tree[p])
					{
						// p stays active and resumes at this arc: it may still have capacity
						// once this path is saturated.
						currentArc[p] = k;
						return treeEdge;
					}
				}

				active.Dequeue();
				isActive[p] = false;
			}

			return -1;
		}

		// Augmentation stage: push the bottleneck along source -> ... -> meeting edge -> ... ->
		// sink, and make orphans of the nodes whose parent link saturates.
		private TValue Augment(int meetingEdge)
		{
			var bottleneck = residual[meetingEdge];
			var v = head[meetingEdge ^ 1];
			for (; parentEdge[v] != TerminalParent; v = head[parentEdge[v]])
			{
				Work++;
				bottleneck = TValue.Min(bottleneck, residual[parentEdge[v] ^ 1]);
			}

			bottleneck = TValue.Min(bottleneck, terminal[v]);

			for (v = head[meetingEdge]; parentEdge[v] != TerminalParent; v = head[parentEdge[v]])
			{
				Work++;
				bottleneck = TValue.Min(bottleneck, residual[parentEdge[v]]);
			}

			bottleneck = TValue.Min(bottleneck, -terminal[v]);

			Push(meetingEdge, bottleneck);

			for (v = head[meetingEdge ^ 1]; parentEdge[v] != TerminalParent;)
			{
				var toParent = parentEdge[v];
				Push(toParent ^ 1, bottleneck);
				if (residual[toParent ^ 1] <= TValue.Zero)
				{
					MakeOrphan(v);
				}

				v = head[toParent];
			}

			terminal[v] -= bottleneck;
			if (terminal[v] <= TValue.Zero)
			{
				MakeOrphan(v);
			}

			for (v = head[meetingEdge]; parentEdge[v] != TerminalParent;)
			{
				var toParent = parentEdge[v];
				Push(toParent, bottleneck);
				if (residual[toParent] <= TValue.Zero)
				{
					MakeOrphan(v);
				}

				v = head[toParent];
			}

			terminal[v] += bottleneck;
			if (terminal[v] >= TValue.Zero)
			{
				MakeOrphan(v);
			}

			return bottleneck;
		}

		private void Push(int edge, TValue amount)
		{
			residual[edge] -= amount;
			residual[edge ^ 1] += amount;
		}

		private void MakeOrphan(int node)
		{
			parentEdge[node] = OrphanParent;
			orphans.Enqueue(node);
		}

		// Adoption stage: give each orphan a new parent in its own tree whose path reaches the
		// terminal, or free it and orphan its children.
		private void Adopt()
		{
			time++;
			while (orphans.Count > 0)
			{
				Work++;
				var p = orphans.Dequeue();
				var ownTree = tree[p];

				// Terminal capacity in the tree's direction reconnects it directly. (Terminal
				// capacity never changes sign, so an orphan normally has none left.)
				if (ownTree == SourceTree ? terminal[p] > TValue.Zero : terminal[p] < TValue.Zero)
				{
					parentEdge[p] = TerminalParent;
					timestamp[p] = time;
					distance[p] = 1;
					continue;
				}

				var bestEdge = -1;
				var bestDistance = int.MaxValue;
				var end = arcStart[p + 1];
				for (var k = arcStart[p]; k < end; k++)
				{
					Work++;
					var e = arcs[k];
					var q = head[e];
					if (tree[q] != ownTree || residual[ownTree == SourceTree ? e ^ 1 : e] <= TValue.Zero)
					{
						continue;
					}

					var d = DistanceToTerminal(q);
					if (d >= 0 && d < bestDistance)
					{
						bestEdge = e;
						bestDistance = d;
					}
				}

				if (bestEdge >= 0)
				{
					parentEdge[p] = bestEdge;
					timestamp[p] = time;
					distance[p] = bestDistance + 1;
					continue;
				}

				for (var k = arcStart[p]; k < end; k++)
				{
					Work++;
					var e = arcs[k];
					var q = head[e];
					if (tree[q] != ownTree)
					{
						continue;
					}

					if (residual[ownTree == SourceTree ? e ^ 1 : e] > TValue.Zero)
					{
						Activate(q);
					}

					if (parentEdge[q] >= 0 && head[parentEdge[q]] == p)
					{
						MakeOrphan(q);
					}
				}

				tree[p] = FreeNode;
				parentEdge[p] = NoParent;
			}
		}

		// Follows parent edges from q. Returns the distance of q from the terminal (a node with
		// a terminal link is at 1), or -1 if the path runs into an orphan. Nodes verified during
		// this adoption stage are timestamped with their distance, so later walks stop there
		// (the paper's heuristic).
		private int DistanceToTerminal(int q)
		{
			var steps = 0;
			var v = q;
			int total;
			while (true)
			{
				Work++;
				if (timestamp[v] == time)
				{
					total = steps + distance[v];
					break;
				}

				var toParent = parentEdge[v];
				if (toParent == TerminalParent)
				{
					timestamp[v] = time;
					distance[v] = 1;
					total = steps + 1;
					break;
				}

				if (toParent < 0)
				{
					return -1;
				}

				steps++;
				v = head[toParent];
			}

			var remaining = total;
			for (v = q; timestamp[v] != time; v = head[parentEdge[v]])
			{
				timestamp[v] = time;
				distance[v] = remaining;
				remaining--;
			}

			return total;
		}
	}
}
