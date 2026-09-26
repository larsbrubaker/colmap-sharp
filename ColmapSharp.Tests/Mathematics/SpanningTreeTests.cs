// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SpanningTreeTests: colmap/math/spanning_tree_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/SpanningTree.cs.
// Tier A (exact): these graphs have distinct weights, so the trees are unique.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class SpanningTreeTests
{
	[Test]
	public async Task SpanningTree_Nominal()
	{
		// Triangle: edges with weights 1, 2, 3.
		// Max spanning tree uses edges 2+3=5, min uses 1+2=3.
		var edges = new List<(int, int)> { (0, 1), (1, 2), (0, 2) };
		var weights = new List<float> { 1.0f, 2.0f, 3.0f };

		var maxTree = SpanningTrees.ComputeMaximumSpanningTree(3, edges, weights);
		var minTree = SpanningTrees.ComputeMinimumSpanningTree(3, edges, weights);

		using (Assert.Multiple())
		{
			await Assert.That(ComputeTreeWeight(maxTree, edges, weights)).IsEqualTo(5.0f);
			await Assert.That(ComputeTreeWeight(minTree, edges, weights)).IsEqualTo(3.0f);
		}
	}

	[Test]
	public async Task SpanningTree_DisconnectedGraph()
	{
		// Two components: {0,1} and {2,3}. Only component containing root is included.
		var edges = new List<(int, int)> { (0, 1), (2, 3) };
		var weights = new List<float> { 1.0f, 2.0f };

		// Root at 0: includes {0,1}, excludes {2,3}.
		var tree0 = SpanningTrees.ComputeMaximumSpanningTree(4, edges, weights, 0);
		using (Assert.Multiple())
		{
			await Assert.That(tree0.Root).IsEqualTo(0);
			await Assert.That(tree0.Parents[0]).IsEqualTo(0);
			await Assert.That(tree0.Parents[1]).IsEqualTo(0);
			await Assert.That(tree0.Parents[2]).IsEqualTo(-1);
			await Assert.That(tree0.Parents[3]).IsEqualTo(-1);
		}

		// Root at 2: includes {2,3}, excludes {0,1}.
		var tree2 = SpanningTrees.ComputeMaximumSpanningTree(4, edges, weights, 2);
		using (Assert.Multiple())
		{
			await Assert.That(tree2.Root).IsEqualTo(2);
			await Assert.That(tree2.Parents[0]).IsEqualTo(-1);
			await Assert.That(tree2.Parents[1]).IsEqualTo(-1);
			await Assert.That(tree2.Parents[2]).IsEqualTo(2);
			await Assert.That(tree2.Parents[3]).IsEqualTo(2);
		}
	}

	[Test]
	public async Task SpanningTree_EmptyGraph()
	{
		var tree = SpanningTrees.ComputeMaximumSpanningTree(0, [], []);
		await Assert.That(tree.IsValid).IsFalse();
	}

	// Helper to compute total weight of edges in the spanning tree.
	private static float ComputeTreeWeight(SpanningTree tree, List<(int, int)> edges, List<float> weights)
	{
		var total = 0.0f;
		for (var i = 0; i < edges.Count; ++i)
		{
			var (u, v) = edges[i];
			// Check if this edge is in the tree (either direction).
			if (tree.Parents[u] == v || tree.Parents[v] == u)
			{
				total += weights[i];
			}
		}

		return total;
	}
}
