// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GraphCutCrossCheckTests: C#-only tests, not a port of any COLMAP case. GraphCut.cs
// (Stoer-Wagner) and MinSTGraphCut.cs (Boykov-Kolmogorov) are written from their papers in
// place of Boost, so graph_cut_test.cc's three tiny graphs are not enough to trust them. These
// compare both against an exhaustive search over every cut of small random graphs.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class GraphCutCrossCheckTests
{
	[Test]
	public async Task StoerWagner_MatchesExhaustiveMinCut()
	{
		var random = new Random(1234);
		for (var trial = 0; trial < 200; trial++)
		{
			var numVertices = random.Next(2, 9);
			var edges = new List<(int, int)>();
			var weights = new List<int>();
			var numEdges = random.Next(2, 20);
			// The first edge pins the vertex count (Stoer-Wagner needs at least two vertices).
			edges.Add((0, numVertices - 1));
			weights.Add(random.Next(0, 10));
			for (var i = 1; i < numEdges; i++)
			{
				edges.Add((random.Next(numVertices), random.Next(numVertices)));
				weights.Add(random.Next(0, 10));
			}

			GraphCut.ComputeMinGraphCutStoerWagner(edges, weights, out var cutWeight, out var cutLabels);
			var n = cutLabels.Length;

			var best = int.MaxValue;
			for (var mask = 1; mask < (1 << n) - 1; mask++)
			{
				best = Math.Min(best, CutWeight(edges, weights, v => ((mask >> v) & 1) == 1));
			}

			var labelsOneSide = cutLabels.Count(l => l == 1);
			using (Assert.Multiple())
			{
				await Assert.That(cutWeight).IsEqualTo(best);
				await Assert.That(CutWeight(edges, weights, v => cutLabels[v] == 1)).IsEqualTo(cutWeight);
				await Assert.That(labelsOneSide).IsGreaterThan(0);
				await Assert.That(labelsOneSide).IsLessThan(n);
			}
		}
	}

	[Test]
	public async Task MinSTGraphCut_MatchesExhaustiveMinCut()
	{
		var random = new Random(4321);
		for (var trial = 0; trial < 300; trial++)
		{
			var numNodes = random.Next(1, 9);
			var graph = new MinSTGraphCut<int>(numNodes);
			var sourceCapacities = new int[numNodes];
			var sinkCapacities = new int[numNodes];
			var arcs = new List<(int From, int To, int Capacity)>();
			for (var v = 0; v < numNodes; v++)
			{
				sourceCapacities[v] = random.Next(3) == 0 ? 0 : random.Next(0, 10);
				sinkCapacities[v] = random.Next(3) == 0 ? 0 : random.Next(0, 10);
				graph.AddNode(v, sourceCapacities[v], sinkCapacities[v]);
			}

			var numEdges = random.Next(0, 16);
			for (var i = 0; i < numEdges; i++)
			{
				// Occasionally an endpoint is a terminal (index numNodes or numNodes + 1), which
				// COLMAP's index checks allow.
				var a = random.Next(10) == 0 ? numNodes + random.Next(2) : random.Next(numNodes);
				var b = random.Next(10) == 0 ? numNodes + random.Next(2) : random.Next(numNodes);
				var forward = random.Next(0, 10);
				var backward = random.Next(0, 10);
				graph.AddEdge(a, b, forward, backward);
				arcs.Add((a, b, forward));
				arcs.Add((b, a, backward));
			}

			// Capacity of the cut whose source side is `onSourceSide`.
			int CutCapacity(Func<int, bool> onSourceSide)
			{
				var total = 0;
				for (var v = 0; v < numNodes; v++)
				{
					total += onSourceSide(v) ? sinkCapacities[v] : sourceCapacities[v];
				}

				foreach (var (from, to, capacity) in arcs)
				{
					var fromSource = from == numNodes || (from < numNodes && onSourceSide(from));
					var toSource = to == numNodes || (to < numNodes && onSourceSide(to));
					if (fromSource && !toSource)
					{
						total += capacity;
					}
				}

				return total;
			}

			var best = int.MaxValue;
			var minimalSinkSide = 0;
			for (var mask = 0; mask < (1 << numNodes); mask++)
			{
				var capacity = CutCapacity(v => ((mask >> v) & 1) == 1);
				// Among min cuts, the one with the largest source side has the smallest sink
				// side: the nodes that can reach the sink in every max flow's residual graph.
				if (capacity < best || (capacity == best && BitCount(mask) > BitCount(minimalSinkSide)))
				{
					best = capacity;
					minimalSinkSide = mask;
				}
			}

			var flow = graph.Compute();
			using (Assert.Multiple())
			{
				await Assert.That(flow).IsEqualTo(best);
				await Assert.That(CutCapacity(graph.IsConnectedToSource)).IsEqualTo(best);
				for (var v = 0; v < numNodes; v++)
				{
					// Tier A: the sink side is the unique minimal one, whatever the flow.
					await Assert.That(graph.IsConnectedToSink(v)).IsEqualTo(((minimalSinkSide >> v) & 1) == 0);
				}
			}
		}
	}

	private static int CutWeight(List<(int, int)> edges, List<int> weights, Func<int, bool> side)
	{
		var total = 0;
		for (var i = 0; i < edges.Count; i++)
		{
			if (side(edges[i].Item1) != side(edges[i].Item2))
			{
				total += weights[i];
			}
		}

		return total;
	}

	private static int BitCount(int mask)
	{
		return System.Numerics.BitOperations.PopCount((uint)mask);
	}
}
