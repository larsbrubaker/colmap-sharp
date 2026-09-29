// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// NormalizedMinGraphCutTests: C#-only tests, not a port of any COLMAP case.
// GraphCut.ComputeNormalizedMinGraphCut runs on MultilevelPartitioner.cs, written here in
// place of METIS (divergence 77), and graph_cut_test.cc's graphs have at
// most eight vertices, which never reach coarsening. These pin the outcome (Tier C) on larger
// graphs: planted clusters are recovered, parts stay balanced, a grid is cut near its
// optimum, the result is deterministic, and the work grows near-linearly on the graph shapes
// (stars, hub images, many components) that stall plain heavy-edge matching.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class NormalizedMinGraphCutTests
{
	[Test]
	[Arguments(2)]
	[Arguments(3)]
	[Arguments(4)]
	[Arguments(5)]
	public async Task PlantedClusters_AreEachOnePart(int numClusters)
	{
		// Ten random draws per cluster count.
		for (var seedOffset = 0; seedOffset < 10; seedOffset++)
		{
			await PlantedClustersCase(numClusters, seedOffset);
		}
	}

	private static async Task PlantedClustersCase(int numClusters, int seedOffset)
	{
		// Dense clusters of 30 vertices with heavy edges, joined in a ring by single light
		// edges. The balanced min cut is exactly the ring edges.
		const int clusterSize = 30;
		var random = new Random(7 + numClusters + (1000 * seedOffset));
		var edges = new List<(int, int)>();
		var weights = new List<int>();
		for (var c = 0; c < numClusters; c++)
		{
			var offset = c * clusterSize;
			for (var i = 0; i < clusterSize; i++)
			{
				for (var j = i + 1; j < clusterSize; j++)
				{
					if (random.Next(3) == 0 || j == i + 1)
					{
						edges.Add((offset + i, offset + j));
						weights.Add(random.Next(20, 100));
					}
				}
			}

			edges.Add((offset, ((c + 1) % numClusters) * clusterSize + 1));
			weights.Add(1);
		}

		var labels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, numClusters);
		using (Assert.Multiple())
		{
			await Assert.That(labels.Count).IsEqualTo(numClusters * clusterSize);
			var clusterLabels = new HashSet<int>();
			for (var c = 0; c < numClusters; c++)
			{
				var label = labels[c * clusterSize];
				clusterLabels.Add(label);
				for (var i = 0; i < clusterSize; i++)
				{
					await Assert.That(labels[(c * clusterSize) + i]).IsEqualTo(label);
				}
			}

			await Assert.That(clusterLabels.Count).IsEqualTo(numClusters);
		}
	}

	[Test]
	public async Task RandomGraphs_PartsAreBalancedAndDeterministic()
	{
		var random = new Random(4321);
		foreach (var numParts in new[] { 1, 2, 3, 5, 8 })
		{
			for (var trial = 0; trial < 5; trial++)
			{
				var numVertices = random.Next(100, 400);
				var edges = new List<(int, int)>();
				var weights = new List<int>();
				for (var v = 1; v < numVertices; v++)
				{
					edges.Add((random.Next(v), v));
					weights.Add(random.Next(0, 50));
				}

				for (var i = 0; i < 3 * numVertices; i++)
				{
					edges.Add((random.Next(numVertices), random.Next(numVertices)));
					weights.Add(random.Next(0, 50));
				}

				var labels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, numParts);
				var again = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, numParts);
				var sizes = new int[numParts];
				foreach (var (_, label) in labels)
				{
					sizes[label]++;
				}

				// Each bisection allows 3% over its target (plus rounding up to a whole
				// vertex), compounded over at most three levels for eight parts.
				var maxSize = Math.Ceiling((double)numVertices / numParts * 1.1) + 3;
				using (Assert.Multiple())
				{
					await Assert.That(labels.Count).IsEqualTo(numVertices);
					await Assert.That(again).IsEquivalentTo(labels);
					foreach (var size in sizes)
					{
						await Assert.That(size).IsGreaterThan(0);
						await Assert.That(size).IsLessThanOrEqualTo((int)maxSize);
					}
				}
			}
		}
	}

	[Test]
	public async Task Grid_IsBisectedNearTheOptimalCut()
	{
		// A 100 x 40 grid of unit edges: the best balanced bisection cuts 40 edges.
		const int width = 100;
		const int height = 40;
		var edges = new List<(int, int)>();
		var weights = new List<int>();
		for (var y = 0; y < height; y++)
		{
			for (var x = 0; x < width; x++)
			{
				var v = (y * width) + x;
				if (x + 1 < width)
				{
					edges.Add((v, v + 1));
					weights.Add(1);
				}

				if (y + 1 < height)
				{
					edges.Add((v, v + width));
					weights.Add(1);
				}
			}
		}

		var labels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, 2);
		var cut = 0;
		for (var i = 0; i < edges.Count; i++)
		{
			if (labels[edges[i].Item1] != labels[edges[i].Item2])
			{
				cut += weights[i];
			}
		}

		// This partitioner cuts 43 edges here; allow 20% over the optimum.
		await Assert.That(cut).IsLessThanOrEqualTo(48);
	}

	[Test]
	[Arguments("star")]
	[Arguments("hubs")]
	[Arguments("pairs")]
	public async Task Work_GrowsNearLinearly(string shape)
	{
		// Counts elementary steps rather than timing, so machine load cannot make it flaky.
		// Quadrupling n must cost well under the 16x a quadratic step would.
		var small = ToCsr(MakeGraph(shape, 5000));
		MultilevelPartitioner.Partition(small.Xadj, small.Adjncy, small.Adjwgt, 2, out var smallWork);
		var large = ToCsr(MakeGraph(shape, 20000));
		MultilevelPartitioner.Partition(large.Xadj, large.Adjncy, large.Adjwgt, 2, out var largeWork);
		await Assert.That((double)largeWork / smallWork).IsLessThan(6.0);
	}

	[Test]
	public async Task GrowRegion_SkipsAVertexThatDoesNotFit()
	{
		// Vertex 0 joins vertex 1 by a heavy edge and vertices 2 and 3 by light ones. Vertex 1
		// weighs 5, more than the grown side may hold, so growth must pass over it and take
		// the lighter vertices instead of stopping.
		var (xadj, adjncy, adjwgt) = ToCsr((4, [(0, 1, 10), (0, 2, 1), (0, 3, 1)]));
		var side = MultilevelPartitioner.GrowRegion(xadj, adjncy, adjwgt, [1, 5, 1, 1], 0, 3.0, 3);
		await Assert.That(side).IsEquivalentTo(new[] { 0, 1, 0, 0 });
	}

	// Graph shapes whose heavy-edge matching stalls. "star": one hub and n leaves. "hubs":
	// n leaves, each linked to 3 of 20 hubs. "pairs": n/2 disconnected edges.
	private static (int NumVertices, List<(int, int, int)> Edges) MakeGraph(string shape, int n)
	{
		var edges = new List<(int, int, int)>();
		switch (shape)
		{
			case "star":
				for (var i = 1; i <= n; i++)
				{
					edges.Add((0, i, 1 + (i % 7)));
				}

				return (n + 1, edges);
			case "hubs":
				for (var i = 0; i < n; i++)
				{
					var leaf = 20 + i;
					edges.Add((i % 20, leaf, 1 + (i % 5)));
					edges.Add(((i + 7) % 20, leaf, 1 + (i % 3)));
					edges.Add(((i + 13) % 20, leaf, 2));
				}

				return (n + 20, edges);
			default:
				for (var i = 0; i + 1 < n; i += 2)
				{
					edges.Add((i, i + 1, 1 + (i % 4)));
				}

				return (n, edges);
		}
	}

	private static (int[] Xadj, int[] Adjncy, int[] Adjwgt) ToCsr((int NumVertices, List<(int, int, int)> Edges) graph)
	{
		var adjacency = new List<(int, int)>[graph.NumVertices];
		for (var v = 0; v < graph.NumVertices; v++)
		{
			adjacency[v] = [];
		}

		foreach (var (a, b, w) in graph.Edges)
		{
			adjacency[a].Add((b, w));
			adjacency[b].Add((a, w));
		}

		var xadj = new int[graph.NumVertices + 1];
		var adjncy = new List<int>();
		var adjwgt = new List<int>();
		for (var v = 0; v < graph.NumVertices; v++)
		{
			foreach (var (u, w) in adjacency[v])
			{
				adjncy.Add(u);
				adjwgt.Add(w);
			}

			xadj[v + 1] = adjncy.Count;
		}

		return (xadj, [.. adjncy], [.. adjwgt]);
	}
}
