// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GraphCutTests: colmap/math/graph_cut_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/GraphCut.cs and
// MinSTGraphCut.cs. Tier A: cut weights, flows and S-T labels are exact. The
// ComputeNormalizedMinGraphCut* cases are Tier C (MultilevelPartitioner stands in for METIS,
// docs/CPP_DIVERGENCES.md entry 77); like COLMAP's, they pin the label count and range, that
// both parts are used, and the component split of the disconnected graph.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class GraphCutTests
{
	[Test]
	public async Task GraphCut_ComputeMinGraphCutStoerWagner()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 4), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2), (3, 4),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4 };
		GraphCut.ComputeMinGraphCutStoerWagner(edges, weights, out var cutWeight, out var cutLabels);
		await AssertCut(cutWeight, cutLabels, 7, 8);
	}

	[Test]
	public async Task GraphCut_ComputeMinGraphCutStoerWagnerDuplicateEdge()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 4), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2), (3, 4), (3, 4),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4, 4 };
		GraphCut.ComputeMinGraphCutStoerWagner(edges, weights, out var cutWeight, out var cutLabels);
		await AssertCut(cutWeight, cutLabels, 7, 8);
	}

	[Test]
	public async Task GraphCut_ComputeMinGraphCutStoerWagnerMissingVertex()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1 };
		GraphCut.ComputeMinGraphCutStoerWagner(edges, weights, out var cutWeight, out var cutLabels);
		await AssertCut(cutWeight, cutLabels, 2, 8);
	}

	[Test]
	public async Task GraphCut_ComputeMinGraphCutStoerWagnerDisconnected()
	{
		var edges = new List<(int, int)> { (0, 1), (1, 2), (3, 4) };
		var weights = new List<int> { 1, 3, 1 };
		GraphCut.ComputeMinGraphCutStoerWagner(edges, weights, out var cutWeight, out var cutLabels);
		await AssertCut(cutWeight, cutLabels, 0, 5);
	}

	[Test]
	public async Task GraphCut_ComputeNormalizedMinGraphCut()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 4), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2), (3, 4),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4 };
		var cutLabels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, 2);
		await AssertTwoPartLabels(cutLabels, 8);
	}

	[Test]
	public async Task GraphCut_ComputeNormalizedMinGraphCutDuplicateEdge()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 4), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2), (3, 4), (3, 4),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4, 4 };
		var cutLabels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, 2);
		await AssertTwoPartLabels(cutLabels, 8);
	}

	[Test]
	public async Task GraphCut_ComputeNormalizedMinGraphCutMissingVertex()
	{
		var edges = new List<(int, int)>
		{
			(3, 4), (3, 6), (3, 5), (0, 1), (0, 6), (0, 7), (0, 5),
			(0, 2), (4, 1), (1, 6), (1, 5), (6, 7), (7, 5), (5, 2),
		};
		var weights = new List<int> { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1 };
		var cutLabels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, 2);
		await AssertTwoPartLabels(cutLabels, 8);
	}

	[Test]
	public async Task GraphCut_ComputeNormalizedMinGraphCutDisconnected()
	{
		var edges = new List<(int, int)> { (0, 1), (1, 2), (3, 4) };
		var weights = new List<int> { 1, 3, 1 };
		var cutLabels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, 2);
		using (Assert.Multiple())
		{
			await Assert.That(cutLabels.Count).IsEqualTo(5);
			await Assert.That(cutLabels[0]).IsEqualTo(cutLabels[1]);
			await Assert.That(cutLabels[1]).IsEqualTo(cutLabels[2]);
			await Assert.That(cutLabels[2]).IsNotEqualTo(cutLabels[3]);
			await Assert.That(cutLabels[3]).IsEqualTo(cutLabels[4]);
		}
	}

	[Test]
	public async Task GraphCut_MinSTGraphCut1()
	{
		var graph = new MinSTGraphCut<int>(2);
		using (Assert.Multiple())
		{
			await Assert.That(graph.NumNodes).IsEqualTo(2);
			await Assert.That(graph.NumEdges).IsEqualTo(0);
		}

		graph.AddNode(0, 5, 1);
		graph.AddNode(1, 2, 6);
		graph.AddEdge(0, 1, 3, 4);
		using (Assert.Multiple())
		{
			await Assert.That(graph.NumEdges).IsEqualTo(10);
			await Assert.That(graph.Compute()).IsEqualTo(6);
			await Assert.That(graph.IsConnectedToSource(0)).IsTrue();
			await Assert.That(graph.IsConnectedToSink(1)).IsTrue();
		}
	}

	[Test]
	public async Task GraphCut_MinSTGraphCut2()
	{
		var graph = new MinSTGraphCut<int>(2);
		graph.AddNode(0, 1, 5);
		graph.AddNode(1, 2, 6);
		graph.AddEdge(0, 1, 3, 4);
		using (Assert.Multiple())
		{
			await Assert.That(graph.NumEdges).IsEqualTo(10);
			await Assert.That(graph.Compute()).IsEqualTo(3);
			await Assert.That(graph.IsConnectedToSink(0)).IsTrue();
			await Assert.That(graph.IsConnectedToSink(1)).IsTrue();
		}
	}

	[Test]
	public async Task GraphCut_MinSTGraphCut3()
	{
		var graph = new MinSTGraphCut<int>(3);
		graph.AddNode(0, 6, 4);
		graph.AddNode(2, 3, 6);
		graph.AddEdge(0, 1, 2, 4);
		graph.AddEdge(1, 2, 3, 5);
		using (Assert.Multiple())
		{
			await Assert.That(graph.NumEdges).IsEqualTo(12);
			await Assert.That(graph.Compute()).IsEqualTo(9);
			await Assert.That(graph.IsConnectedToSource(0)).IsTrue();
			await Assert.That(graph.IsConnectedToSink(1)).IsTrue();
			await Assert.That(graph.IsConnectedToSink(2)).IsTrue();
		}
	}

	private static async Task AssertCut(int cutWeight, byte[] cutLabels, int expectedWeight, int expectedCount)
	{
		using (Assert.Multiple())
		{
			await Assert.That(cutWeight).IsEqualTo(expectedWeight);
			await Assert.That(cutLabels.Length).IsEqualTo(expectedCount);
			foreach (var label in cutLabels)
			{
				await Assert.That(label).IsGreaterThanOrEqualTo((byte)0);
				await Assert.That(label).IsLessThan((byte)2);
			}
		}
	}

	private static async Task AssertTwoPartLabels(Dictionary<int, int> cutLabels, int expectedCount)
	{
		var numLabels = new int[2];
		using (Assert.Multiple())
		{
			await Assert.That(cutLabels.Count).IsEqualTo(expectedCount);
			foreach (var (_, label) in cutLabels)
			{
				await Assert.That(label).IsGreaterThanOrEqualTo(0);
				await Assert.That(label).IsLessThan(2);
				if (label is >= 0 and < 2)
				{
					numLabels[label]++;
				}
			}

			await Assert.That(numLabels[0]).IsGreaterThan(0);
			await Assert.That(numLabels[1]).IsGreaterThan(0);
		}
	}
}
