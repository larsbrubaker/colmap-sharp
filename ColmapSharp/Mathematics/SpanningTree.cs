// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SpanningTree: port of colmap/math/spanning_tree.h and spanning_tree.cc, minimum and maximum
// spanning trees of an undirected weighted graph, returned as parent pointers rooted at a
// chosen node. Rotation averaging (estimators/rotation_averaging) initializes from the
// maximum spanning tree of the pose graph. Tests:
// ColmapSharp.Tests/Mathematics/SpanningTreeTests.cs (spanning_tree_test.cc 1:1).
//
// COLMAP runs boost::kruskal_minimum_spanning_tree. Boost is not ported (docs/LICENSE_AUDIT.md);
// this is Kruskal's algorithm written from its published description (J. B. Kruskal, "On the
// shortest spanning subtree of a graph and the traveling salesman problem", Proc. AMS 7(1),
// 1956) with UnionFind.cs as the disjoint-set forest.
//
// Tier A when the edge weights (after the max-weight negation below) are distinct: the
// minimum spanning tree is then unique, so the parents match COLMAP exactly. Among equal
// weights boost's priority queue picks in heap order; here the earlier edge in the input
// wins. See divergence 3.

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of colmap::SpanningTree: a rooted spanning tree as a parent map. For each node index
/// i, Parents[i] is its parent; the root has Parents[root] == root, and nodes outside the
/// root's component have -1.
/// </summary>
public sealed class SpanningTree
{
	/// <summary>The root node, or -1 for an empty tree.</summary>
	public int Root { get; init; } = -1;

	/// <summary>Parent of each node (see the class summary).</summary>
	public int[] Parents { get; init; } = [];

	/// <summary>True when the tree has a root and at least one node.</summary>
	public bool IsValid => Root >= 0 && Parents.Length > 0;

	/// <summary>The number of nodes in the tree.</summary>
	public int NumNodes => Parents.Length;
}

/// <summary>
/// Port of colmap::ComputeMaximumSpanningTree and colmap::ComputeMinimumSpanningTree.
/// </summary>
public static class SpanningTrees
{
	/// <summary>
	/// Computes the maximum spanning tree of an undirected weighted graph with nodes
	/// 0..numNodes-1 (higher weight is preferred). If the graph is disconnected, only the
	/// component containing <paramref name="root"/> gets parents.
	/// </summary>
	public static SpanningTree ComputeMaximumSpanningTree(
		int numNodes,
		IReadOnlyList<(int, int)> edges,
		IReadOnlyList<float> weights,
		int root = 0)
	{
		return ComputeSpanningTree(numNodes, edges, weights, root, maximize: true);
	}

	/// <summary>
	/// Computes the minimum spanning tree; same interface as
	/// <see cref="ComputeMaximumSpanningTree"/>.
	/// </summary>
	public static SpanningTree ComputeMinimumSpanningTree(
		int numNodes,
		IReadOnlyList<(int, int)> edges,
		IReadOnlyList<float> weights,
		int root = 0)
	{
		return ComputeSpanningTree(numNodes, edges, weights, root, maximize: false);
	}

	private static SpanningTree ComputeSpanningTree(
		int numNodes,
		IReadOnlyList<(int, int)> edges,
		IReadOnlyList<float> weights,
		int root,
		bool maximize)
	{
		if (numNodes <= 0)
		{
			return new SpanningTree();
		}

		// boost::add_edge on a vecS graph would silently grow the vertex set for an
		// out-of-range endpoint, and COLMAP then indexes past its adjacency list; fail loudly.
		Check.Eq(edges.Count, weights.Count);
		Check.That(root >= 0 && root < numNodes);

		// For the maximum spanning tree COLMAP maps w to (max_weight - w) in float, with
		// max_weight starting at 0, and finds the minimum. Doing the same float arithmetic keeps
		// the same ties.
		var maxWeight = 0.0f;
		if (maximize)
		{
			for (var i = 0; i < weights.Count; i++)
			{
				maxWeight = Math.Max(maxWeight, weights[i]);
			}
		}

		var costs = new float[edges.Count];
		var order = new int[edges.Count];
		for (var i = 0; i < edges.Count; i++)
		{
			var (node1, node2) = edges[i];
			Check.That(node1 >= 0 && node1 < numNodes && node2 >= 0 && node2 < numNodes);
			costs[i] = maximize ? maxWeight - weights[i] : weights[i];
			order[i] = i;
		}

		// Kruskal: take edges by increasing cost, keeping each that joins two components.
		// The index tie-break makes the order total, so the unstable Array.Sort is deterministic.
		Array.Sort(order, (a, b) =>
		{
			var byCost = costs[a].CompareTo(costs[b]);
			return byCost != 0 ? byCost : a.CompareTo(b);
		});

		var components = new UnionFind<int>();
		components.Reserve(numNodes);
		var adjacency = new List<int>[numNodes];
		for (var i = 0; i < numNodes; i++)
		{
			adjacency[i] = [];
		}

		foreach (var edgeIdx in order)
		{
			var (source, target) = edges[edgeIdx];
			if (components.Find(source) != components.Find(target))
			{
				components.Union(source, target);
				adjacency[source].Add(target);
				adjacency[target].Add(source);
			}
		}

		return new SpanningTree
		{
			Root = root,
			Parents = BuildParentsFromAdjacencyList(adjacency, root),
		};
	}

	// Breadth-first search from the root turns the undirected tree into parent pointers.
	// Nodes the search never reaches (other components) keep -1.
	private static int[] BuildParentsFromAdjacencyList(List<int>[] adjacency, int root)
	{
		var parents = new int[adjacency.Length];
		Array.Fill(parents, -1);
		parents[root] = root;

		var visited = new bool[adjacency.Length];
		visited[root] = true;

		var queue = new Queue<int>();
		queue.Enqueue(root);
		while (queue.Count > 0)
		{
			var current = queue.Dequeue();
			foreach (var neighbor in adjacency[current])
			{
				if (!visited[neighbor])
				{
					visited[neighbor] = true;
					parents[neighbor] = current;
					queue.Enqueue(neighbor);
				}
			}
		}

		return parents;
	}
}
