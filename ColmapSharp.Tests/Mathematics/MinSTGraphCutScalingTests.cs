// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MinSTGraphCutScalingTests: C#-only test, not a port of any COLMAP case. Delaunay meshing
// hands MinSTGraphCut millions of cells, each with terminal capacities. A version that stored
// those as ordinary edges out of the terminal rescanned all of them after every augmentation
// and went quadratic (160k nodes took over two minutes). This pins the fix: a 200k-node grid
// with terminal capacities on every node must finish in work linear in its size.
//
// Work is counted, not timed: a wall-clock bound can fail under heavy machine load while the
// code is fine. MinSTGraphCut.LastComputeWork (an internal diagnostic) counts every arc
// examined while growing trees and adopting orphans and every parent link followed, so the
// quadratic rescan shows up in it directly.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class MinSTGraphCutScalingTests
{
	[Test]
	public async Task MinSTGraphCut_LargeGridWithTerminalsOnEveryNodeIsNotQuadratic()
	{
		const int side = 448; // 200,704 nodes
		var random = new Random(1);
		var graph = new MinSTGraphCut<float>(side * side);
		var sourceCapacities = new float[side * side];
		var sinkCapacities = new float[side * side];
		var arcs = new List<(int From, int To, float Capacity)>();
		for (var i = 0; i < side * side; i++)
		{
			sourceCapacities[i] = (float)random.NextDouble();
			sinkCapacities[i] = (float)random.NextDouble();
			graph.AddNode(i, sourceCapacities[i], sinkCapacities[i]);
		}

		for (var y = 0; y < side; y++)
		{
			for (var x = 0; x < side; x++)
			{
				var i = (y * side) + x;
				if (x + 1 < side)
				{
					AddEdge(i, i + 1);
				}

				if (y + 1 < side)
				{
					AddEdge(i, i + side);
				}
			}
		}

		void AddEdge(int a, int b)
		{
			var forward = (float)random.NextDouble() * 0.5f;
			var backward = (float)random.NextDouble() * 0.5f;
			graph.AddEdge(a, b, forward, backward);
			arcs.Add((a, b, forward));
			arcs.Add((b, a, backward));
		}

		var flow = graph.Compute();

		// The labels must describe a cut whose capacity is the flow (max-flow = min-cut).
		double cutCapacity = 0;
		for (var i = 0; i < side * side; i++)
		{
			cutCapacity += graph.IsConnectedToSource(i) ? sinkCapacities[i] : sourceCapacities[i];
		}

		foreach (var (from, to, capacity) in arcs)
		{
			if (graph.IsConnectedToSource(from) && graph.IsConnectedToSink(to))
			{
				cutCapacity += capacity;
			}
		}

		using (Assert.Multiple())
		{
			// 6,210,545 steps measured, about 3.4 per node and edge (200,704 nodes; NumEdges is
			// 1,603,840, counting reverses and terminal links), the same ratio as at a quarter
			// and a sixteenth of the size. The quadratic version rescanned ~200k terminal links
			// per augmentation, some 1e10 steps.
			await Assert.That(graph.LastComputeWork).IsLessThan(10L * (graph.NumNodes + graph.NumEdges));
			await Assert.That(Math.Abs(cutCapacity - flow)).IsLessThan(1e-3 * Math.Max(1.0, flow));
		}
	}
}
