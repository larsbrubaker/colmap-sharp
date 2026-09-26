// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ConnectedComponents: port of colmap/math/connected_components.h, the connected components
// of an undirected graph given as a node set and an edge list, built on UnionFind.cs.
// scene/pose_graph uses both functions. Tests:
// ColmapSharp.Tests/Mathematics/ConnectedComponentsTests.cs (connected_components_test.cc 1:1).
//
// Tier A for the component *sets*. The order of the components, the order of the nodes
// inside each, and which of two equally large components FindLargestConnectedComponent
// returns follow COLMAP's hash-container iteration there; here they follow the order in
// which the caller's node collection enumerates. See docs/CPP_DIVERGENCES.md, entry 2.

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of colmap::FindConnectedComponents and colmap::FindLargestConnectedComponent.
/// </summary>
public static class ConnectedComponents
{
	/// <summary>
	/// Finds all connected components in a graph. Each component lists its nodes in the order
	/// <paramref name="nodes"/> enumerates them, and components are ordered by their first
	/// node in that order. Nodes that appear only in <paramref name="edges"/> are not reported.
	/// </summary>
	/// <param name="nodes">All nodes in the graph (COLMAP takes a FlatHashSet).</param>
	/// <param name="edges">Edges as (node1, node2) pairs.</param>
	public static List<List<T>> FindConnectedComponents<T>(IEnumerable<T> nodes, IReadOnlyList<(T, T)> edges)
		where T : notnull
	{
		return [.. GroupByRoot(nodes, edges).Values];
	}

	/// <summary>
	/// Finds the largest connected component in a graph. Of several equally large
	/// components, returns the one whose first node comes first in <paramref name="nodes"/>.
	/// Returns an empty list for an empty graph.
	/// </summary>
	public static List<T> FindLargestConnectedComponent<T>(IEnumerable<T> nodes, IReadOnlyList<(T, T)> edges)
		where T : notnull
	{
		// COLMAP starts from an empty component and keeps the first strictly larger one.
		List<T> largest = [];
		foreach (var members in GroupByRoot(nodes, edges).Values)
		{
			if (members.Count > largest.Count)
			{
				largest = members;
			}
		}

		return largest;
	}

	// Unites every edge, then buckets each node under its root. The dictionary never removes,
	// so its enumeration is insertion order: components by first node seen.
	private static Dictionary<T, List<T>> GroupByRoot<T>(IEnumerable<T> nodes, IReadOnlyList<(T, T)> edges)
		where T : notnull
	{
		var unionFind = new UnionFind<T>();
		if (nodes.TryGetNonEnumeratedCount(out var count))
		{
			unionFind.Reserve(count);
		}

		foreach (var (node1, node2) in edges)
		{
			unionFind.Union(node1, node2);
		}

		var components = new Dictionary<T, List<T>>();
		foreach (var node in nodes)
		{
			var root = unionFind.Find(node);
			if (!components.TryGetValue(root, out var members))
			{
				members = [];
				components.Add(root, members);
			}

			members.Add(node);
		}

		return components;
	}
}
