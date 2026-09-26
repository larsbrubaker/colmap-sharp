// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GraphCut: the free functions of colmap/math/graph_cut.h and graph_cut.cc. This file holds
// ComputeMinGraphCutStoerWagner, the global min-cut of an undirected graph; its neighbor
// MinSTGraphCut.cs holds the S-T min-cut class from the same header. Tests:
// ColmapSharp.Tests/Mathematics/GraphCutTests.cs (graph_cut_test.cc).
//
// COLMAP calls boost::stoer_wagner_min_cut. Boost is not ported (docs/LICENSE_AUDIT.md); this
// is the algorithm written from its paper: M. Stoer and F. Wagner, "A Simple Min-Cut
// Algorithm", Journal of the ACM 44(4), 1997.
//
// Tier A for the cut weight (the minimum is unique). When several cuts share that weight, the
// side reported can differ from boost's, and which side is labeled 1 is this code's choice
// (the vertices merged into the last-added vertex of the best phase). graph_cut_test.cc only
// pins the weight and the label range. See docs/CPP_DIVERGENCES.md, entry 4.
//
// Not here: ComputeNormalizedMinGraphCut, which wraps METIS's k-way partitioner. It lands
// with its replacement partitioner (scene clustering needs it); see PORTING_PLAN.md.

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of the free functions in colmap/math/graph_cut.h.
/// </summary>
public static class GraphCut
{
	/// <summary>
	/// Port of colmap::ComputeMinGraphCutStoerWagner: the minimum cut of an undirected graph
	/// with vertices 0..max vertex id. <paramref name="cutLabels"/> has one entry per vertex,
	/// 0 or 1 by side of the cut. Parallel edges add up; self-loops never cross a cut.
	/// </summary>
	public static void ComputeMinGraphCutStoerWagner(
		IReadOnlyList<(int, int)> edges,
		IReadOnlyList<int> weights,
		out int cutWeight,
		out byte[] cutLabels)
	{
		Check.Eq(edges.Count, weights.Count);
		Check.Ge(edges.Count, 2);

		var maxVertexIndex = 0;
		foreach (var (v1, v2) in edges)
		{
			Check.Ge(v1, 0);
			Check.Ge(v2, 0);
			maxVertexIndex = Math.Max(maxVertexIndex, v1);
			maxVertexIndex = Math.Max(maxVertexIndex, v2);
		}

		var numVertices = maxVertexIndex + 1;

		// boost::stoer_wagner_min_cut throws bad_graph for fewer than two vertices.
		Check.Ge(numVertices, 2);

		// Merged-graph adjacency: neighbor -> total weight. Sums stay int, as boost's do.
		var adjacency = new Dictionary<int, int>[numVertices];
		var members = new List<int>[numVertices];
		for (var v = 0; v < numVertices; v++)
		{
			adjacency[v] = [];
			members[v] = [v];
		}

		for (var i = 0; i < edges.Count; i++)
		{
			var (v1, v2) = edges[i];
			if (v1 == v2)
			{
				continue;
			}

			adjacency[v1][v2] = adjacency[v1].GetValueOrDefault(v2) + weights[i];
			adjacency[v2][v1] = adjacency[v2].GetValueOrDefault(v1) + weights[i];
		}

		var active = new List<int>(numVertices);
		for (var v = 0; v < numVertices; v++)
		{
			active.Add(v);
		}

		var bestWeight = int.MaxValue;
		List<int> bestSide = [];

		var key = new int[numVertices];
		var inA = new bool[numVertices];

		// Each phase orders the remaining (merged) vertices by "most tightly connected to the
		// set so far". The weight between the last vertex and the rest is a minimum cut
		// separating the last two, which are then merged. The best such cut is the global one.
		var queue = new PriorityQueue<int, (int NegKey, int Vertex)>();
		while (active.Count > 1)
		{
			queue.Clear();
			foreach (var v in active)
			{
				key[v] = 0;
				inA[v] = false;
				queue.Enqueue(v, (0, v));
			}

			var previous = -1;
			var last = -1;
			for (var added = 0; added < active.Count; added++)
			{
				int v;
				(int NegKey, int Vertex) priority;
				// Lazy deletion: skip entries for vertices already added or with a stale key.
				while (queue.TryDequeue(out v, out priority) && (inA[v] || -priority.NegKey != key[v]))
				{
				}

				inA[v] = true;
				previous = last;
				last = v;
				foreach (var (neighbor, weight) in adjacency[v])
				{
					if (!inA[neighbor])
					{
						key[neighbor] += weight;
						queue.Enqueue(neighbor, (-key[neighbor], neighbor));
					}
				}
			}

			var cutOfThePhase = key[last];
			if (cutOfThePhase < bestWeight)
			{
				bestWeight = cutOfThePhase;
				bestSide = [.. members[last]];
			}

			Merge(adjacency, members, previous, last);
			active.Remove(last);
		}

		cutWeight = bestWeight;
		cutLabels = new byte[numVertices];
		foreach (var v in bestSide)
		{
			cutLabels[v] = 1;
		}
	}

	// Contracts vertex `from` into vertex `into`, summing parallel edges and dropping the edge
	// between them.
	private static void Merge(Dictionary<int, int>[] adjacency, List<int>[] members, int into, int from)
	{
		foreach (var (neighbor, weight) in adjacency[from])
		{
			adjacency[neighbor].Remove(from);
			if (neighbor == into)
			{
				continue;
			}

			adjacency[into][neighbor] = adjacency[into].GetValueOrDefault(neighbor) + weight;
			adjacency[neighbor][into] = adjacency[neighbor].GetValueOrDefault(into) + weight;
		}

		adjacency[from].Clear();
		members[into].AddRange(members[from]);
		members[from].Clear();
	}
}
