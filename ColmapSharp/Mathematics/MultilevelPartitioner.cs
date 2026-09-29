// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MultilevelPartitioner: the k-way graph partitioner behind
// GraphCut.ComputeNormalizedMinGraphCut (GraphCut.cs). COLMAP calls METIS_PartGraphKway
// there; METIS is not ported (divergence 77). This is written here from
// the published multilevel scheme, not from METIS's code:
//
//   - B. Hendrickson and R. Leland, "A Multilevel Algorithm for Partitioning Graphs",
//     Supercomputing 1995 (coarsen, partition the coarsest graph, project and refine).
//   - G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning
//     Irregular Graphs", SIAM J. Sci. Comput. 20(1), 1998 (heavy-edge matching, greedy graph
//     growing for the initial bisection, recursive bisection for k parts).
//   - C. M. Fiduccia and R. M. Mattheyses, "A Linear-Time Heuristic for Improving Network
//     Partitions", DAC 1982 (the boundary refinement with rollback to the best prefix).
//
// k parts come from recursive bisection: floor(k/2) parts on one side, the rest on the other,
// with the target weights split in the same ratio (as METIS_PartGraphRecursive splits them).
// There is no randomness: every choice breaks ties by vertex index, so the output is a pure
// function of the input graph. Tier C: COLMAP's contract is "a balanced partition with a
// small cut", and graph_cut_test.cc checks the label range, that both parts are used, and
// that the disconnected graph is split along its components.

namespace ColmapSharp.Mathematics;

/// <summary>
/// Deterministic multilevel recursive-bisection partitioner with Fiduccia-Mattheyses
/// refinement; the replacement for METIS_PartGraphKway.
/// </summary>
internal sealed class MultilevelPartitioner
{
	// Allowed imbalance per side: METIS's k-way default ufactor is 30, i.e. 1.03.
	private const double ImbalanceTolerance = 1.03;

	// Graphs this small are bisected directly instead of coarsened further.
	private const int CoarsenTo = 20;

	// Coarsening stops when a level shrinks the graph by less than this.
	private const double MinCoarseningRatio = 0.95;

	private const int MaxRefinementPasses = 10;

	private const int MaxInitialSeeds = 8;

	// Fraction of vertices heavy-edge matching may leave unmatched before two-hop matching
	// pairs up the rest.
	private const double UnmatchedForTwoHop = 0.1;

	/// <summary>
	/// Partitions a weighted undirected graph given in CSR form (<paramref name="xadj"/>,
	/// <paramref name="adjncy"/>, <paramref name="adjwgt"/>, each edge stored in both
	/// directions, unit vertex weights) into <paramref name="numParts"/> parts. Returns one
	/// label in [0, numParts) per vertex. Parallel edges add up and self-loops are ignored.
	/// </summary>
	public static int[] Partition(int[] xadj, int[] adjncy, int[] adjwgt, int numParts) =>
		Partition(xadj, adjncy, adjwgt, numParts, out _);

	/// <summary>
	/// <see cref="Partition(int[], int[], int[], int)"/>, also returning a count of the
	/// elementary steps taken (adjacency entries, vertices and queue entries visited). Tests
	/// use it to pin the running time's growth without timing anything.
	/// </summary>
	public static int[] Partition(int[] xadj, int[] adjncy, int[] adjwgt, int numParts, out long work)
	{
		var partitioner = new MultilevelPartitioner();
		var labels = partitioner.Run(xadj, adjncy, adjwgt, numParts);
		work = partitioner.work;
		return labels;
	}

	// Elementary steps taken so far; see Partition(..., out long work).
	private long work;

	private int[] Run(int[] xadj, int[] adjncy, int[] adjwgt, int numParts)
	{
		var numVertices = xadj.Length - 1;
		var vertexWeights = new int[numVertices];
		Array.Fill(vertexWeights, 1);
		var identity = new int[numVertices];
		for (var v = 0; v < numVertices; v++)
		{
			identity[v] = v;
		}

		// Contracting by the identity map merges parallel edges and drops self-loops.
		var graph = Contract(new Graph(xadj, adjncy, adjwgt, vertexWeights), identity, numVertices);
		var labels = new int[numVertices];
		PartitionRecursive(graph, identity, numParts, 0, labels);
		return labels;
	}

	private sealed class Graph(int[] xadj, int[] adjncy, int[] adjwgt, int[] vertexWeights)
	{
		public readonly int[] XAdj = xadj;
		public readonly int[] AdjNcy = adjncy;
		public readonly int[] AdjWgt = adjwgt;
		public readonly int[] VertexWeights = vertexWeights;

		public int NumVertices => VertexWeights.Length;

		public long TotalWeight()
		{
			long total = 0;
			foreach (var w in VertexWeights)
			{
				total += w;
			}

			return total;
		}
	}

	private void PartitionRecursive(Graph graph, int[] originalIds, int numParts, int labelOffset, int[] labels)
	{
		if (graph.NumVertices == 0)
		{
			return;
		}

		if (numParts == 1)
		{
			foreach (var id in originalIds)
			{
				labels[id] = labelOffset;
			}

			return;
		}

		var leftParts = numParts / 2;
		var side = Bisect(graph, (double)leftParts / numParts);
		for (var s = 0; s < 2; s++)
		{
			var (subgraph, subIds) = ExtractSide(graph, originalIds, side, s);
			PartitionRecursive(
				subgraph,
				subIds,
				s == 0 ? leftParts : numParts - leftParts,
				s == 0 ? labelOffset : labelOffset + leftParts,
				labels);
		}
	}

	// Multilevel bisection: coarsen by heavy-edge matching, bisect the coarsest graph, then
	// project back level by level with FM refinement at each one.
	private int[] Bisect(Graph graph, double leftFraction)
	{
		var total = graph.TotalWeight();
		var maxWeights = new long[2];
		var targets = new[] { leftFraction * total, (1 - leftFraction) * total };
		for (var s = 0; s < 2; s++)
		{
			// The ceiling keeps an exact split feasible for small graphs of unit weights.
			maxWeights[s] = (long)Math.Max(Math.Floor(targets[s] * ImbalanceTolerance), Math.Ceiling(targets[s]));
		}

		var levels = new List<Graph> { graph };
		var coarseMaps = new List<int[]>();
		var maxCoarseVertexWeight = Math.Max(1, (int)(1.5 * total / CoarsenTo));
		while (levels[^1].NumVertices > CoarsenTo)
		{
			var current = levels[^1];
			var (coarseMap, numCoarse) = HeavyEdgeMatching(current, maxCoarseVertexWeight);
			if (numCoarse > MinCoarseningRatio * current.NumVertices)
			{
				break;
			}

			coarseMaps.Add(coarseMap);
			levels.Add(Contract(current, coarseMap, numCoarse));
		}

		var side = InitialBisection(levels[^1], targets[0], maxWeights);
		for (var level = levels.Count - 2; level >= 0; level--)
		{
			var coarseMap = coarseMaps[level];
			var fine = new int[levels[level].NumVertices];
			for (var v = 0; v < fine.Length; v++)
			{
				work++;
				fine[v] = side[coarseMap[v]];
			}

			side = fine;
			RefineFm(levels[level], side, maxWeights);
		}

		return side;
	}

	// Visits vertices by increasing degree (index breaks ties) and matches each unmatched one
	// with the unmatched neighbor across its heaviest edge, as long as the merged weight stays
	// under the cap that keeps coarse vertices from dominating the balance.
	private (int[] CoarseMap, int NumCoarse) HeavyEdgeMatching(Graph graph, int maxVertexWeight)
	{
		var n = graph.NumVertices;
		var order = new int[n];
		var degrees = new int[n];
		for (var v = 0; v < n; v++)
		{
			work++;
			order[v] = v;
			degrees[v] = graph.XAdj[v + 1] - graph.XAdj[v];
		}

		Array.Sort(order, (a, b) => degrees[a] != degrees[b] ? degrees[a].CompareTo(degrees[b]) : a.CompareTo(b));

		var match = new int[n];
		Array.Fill(match, -1);
		foreach (var v in order)
		{
			work++;
			if (match[v] >= 0)
			{
				continue;
			}

			var best = v;
			var bestWeight = int.MinValue;
			for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
			{
				work++;
				var u = graph.AdjNcy[e];
				var w = graph.AdjWgt[e];
				if (match[u] >= 0 || graph.VertexWeights[u] + graph.VertexWeights[v] > maxVertexWeight)
				{
					continue;
				}

				if (w > bestWeight || (w == bestWeight && u < best))
				{
					best = u;
					bestWeight = w;
				}
			}

			match[v] = best;
			match[best] = v;
		}

		var numUnmatched = 0;
		for (var v = 0; v < n; v++)
		{
			work++;
			if (match[v] == v)
			{
				numUnmatched++;
			}
		}

		// Heavy-edge matching stalls on stars and hub-dominated graphs (one hub matches, its
		// leaves have no one left) and on isolated vertices, which would leave the coarsest
		// graph nearly as large as the input and everything after it quadratic. When over
		// UnmatchedForTwoHop of the vertices are left unmatched, pair them up further.
		if (numUnmatched > UnmatchedForTwoHop * n)
		{
			MatchTwoHop(graph, match, maxVertexWeight);
		}

		var coarseMap = new int[n];
		Array.Fill(coarseMap, -1);
		var numCoarse = 0;
		for (var v = 0; v < n; v++)
		{
			work++;
			if (coarseMap[v] < 0)
			{
				coarseMap[v] = numCoarse;
				coarseMap[match[v]] = numCoarse;
				numCoarse++;
			}
		}

		return (coarseMap, numCoarse);
	}

	// Pairs still-unmatched vertices (match[v] == v) that share a neighbor: for each vertex h
	// in index order, its unmatched neighbors pair off in adjacency order. Then unmatched
	// isolated vertices, which share the empty neighborhood, pair off in index order. Both
	// respect the coarse weight cap. Linear in the number of edges.
	private void MatchTwoHop(Graph graph, int[] match, int maxVertexWeight)
	{
		var n = graph.NumVertices;
		for (var h = 0; h < n; h++)
		{
			work++;
			var pending = -1;
			for (var e = graph.XAdj[h]; e < graph.XAdj[h + 1]; e++)
			{
				work++;
				var u = graph.AdjNcy[e];
				if (match[u] != u)
				{
					continue;
				}

				if (pending >= 0 && graph.VertexWeights[pending] + graph.VertexWeights[u] <= maxVertexWeight)
				{
					match[pending] = u;
					match[u] = pending;
					pending = -1;
				}
				else if (pending < 0 || graph.VertexWeights[u] < graph.VertexWeights[pending])
				{
					pending = u;
				}
			}
		}

		var pendingIsolated = -1;
		for (var v = 0; v < n; v++)
		{
			work++;
			if (match[v] != v || graph.XAdj[v + 1] != graph.XAdj[v])
			{
				continue;
			}

			if (pendingIsolated >= 0 && graph.VertexWeights[pendingIsolated] + graph.VertexWeights[v] <= maxVertexWeight)
			{
				match[pendingIsolated] = v;
				match[v] = pendingIsolated;
				pendingIsolated = -1;
			}
			else if (pendingIsolated < 0 || graph.VertexWeights[v] < graph.VertexWeights[pendingIsolated])
			{
				pendingIsolated = v;
			}
		}
	}

	// Builds the graph whose vertex c is the union of the fine vertices mapped to c. Edge
	// weights between merged vertices add up; edges inside a merged vertex disappear.
	private Graph Contract(Graph graph, int[] coarseMap, int numCoarse)
	{
		var members = new List<int>[numCoarse];
		for (var c = 0; c < numCoarse; c++)
		{
			members[c] = [];
		}

		for (var v = 0; v < graph.NumVertices; v++)
		{
			work++;
			members[coarseMap[v]].Add(v);
		}

		var xadj = new int[numCoarse + 1];
		var adjncy = new List<int>();
		var adjwgt = new List<int>();
		var vertexWeights = new int[numCoarse];
		var slot = new int[numCoarse];
		Array.Fill(slot, -1);
		for (var c = 0; c < numCoarse; c++)
		{
			var start = adjncy.Count;
			foreach (var v in members[c])
			{
				work++;
				vertexWeights[c] += graph.VertexWeights[v];
				for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
				{
					work++;
					var cu = coarseMap[graph.AdjNcy[e]];
					if (cu == c)
					{
						continue;
					}

					if (slot[cu] < 0)
					{
						slot[cu] = adjncy.Count;
						adjncy.Add(cu);
						adjwgt.Add(graph.AdjWgt[e]);
					}
					else
					{
						adjwgt[slot[cu]] += graph.AdjWgt[e];
					}
				}
			}

			for (var e = start; e < adjncy.Count; e++)
			{
				slot[adjncy[e]] = -1;
			}

			xadj[c + 1] = adjncy.Count;
		}

		return new Graph(xadj, [.. adjncy], [.. adjwgt], vertexWeights);
	}

	// Greedy graph growing from several evenly spaced seeds: side 0 grows by the vertex that
	// most reduces the cut among those adjacent to it (the lowest-index remaining vertex when
	// none is adjacent, so disconnected graphs still grow) until it reaches its target
	// weight. A vertex too heavy to fit is skipped, not an end to growth. Each candidate is
	// FM-refined and the best one kept.
	private int[] InitialBisection(Graph graph, double target0, long[] maxWeights)
	{
		var n = graph.NumVertices;
		var numSeeds = Math.Min(n, MaxInitialSeeds);
		int[]? best = null;
		(long Excess, long Cut) bestScore = default;
		for (var t = 0; t < numSeeds; t++)
		{
			var side = GrowRegion(graph, t * n / numSeeds, target0, maxWeights[0]);
			RefineFm(graph, side, maxWeights);
			var score = Score(graph, side, maxWeights);
			if (best == null || score.CompareTo(bestScore) < 0)
			{
				best = side;
				bestScore = score;
			}
		}

		return best ?? [];
	}

	/// <summary>
	/// Greedy graph growing from <paramref name="seed"/> for a graph in CSR form with vertex
	/// weights; returns 0 for the grown side and 1 for the rest. Exposed for tests.
	/// </summary>
	internal static int[] GrowRegion(
		int[] xadj, int[] adjncy, int[] adjwgt, int[] vertexWeights, int seed, double target0, long maxWeight0) =>
		new MultilevelPartitioner().GrowRegion(new Graph(xadj, adjncy, adjwgt, vertexWeights), seed, target0, maxWeight0);

	private int[] GrowRegion(Graph graph, int seed, double target0, long maxWeight0)
	{
		var n = graph.NumVertices;
		var side = new int[n];
		Array.Fill(side, 1);
		var connection = new long[n];
		var degree = new long[n];
		var touched = new bool[n];

		// A vertex too heavy for side 0 now never fits later, since side 0 only grows.
		var rejected = new bool[n];
		for (var v = 0; v < n; v++)
		{
			work++;
			for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
			{
				work++;
				degree[v] += graph.AdjWgt[e];
			}
		}

		// Touched candidates by largest gain, then lowest index. Gains only rise as side 0
		// grows, so an entry whose gain no longer matches is a stale duplicate.
		var queue = new PriorityQueue<int, (long NegGain, int Vertex)>();
		var cursor = 0;
		long weight0 = 0;
		var next = seed;
		while (weight0 < target0)
		{
			if (next < 0)
			{
				while (queue.TryDequeue(out var v, out var priority))
				{
					work++;
					if (side[v] == 0 || rejected[v] || -priority.NegGain != (2 * connection[v]) - degree[v])
					{
						continue;
					}

					if (weight0 + graph.VertexWeights[v] > maxWeight0)
					{
						rejected[v] = true;
						continue;
					}

					next = v;
					break;
				}
			}

			// No touched vertex fits: continue from the lowest-index untouched vertex, so
			// disconnected graphs still grow.
			while (next < 0 && cursor < n)
			{
				work++;
				var v = cursor++;
				if (side[v] == 0 || touched[v] || rejected[v])
				{
					continue;
				}

				if (weight0 + graph.VertexWeights[v] > maxWeight0)
				{
					rejected[v] = true;
					continue;
				}

				next = v;
			}

			if (next < 0)
			{
				break;
			}

			if (weight0 + graph.VertexWeights[next] > maxWeight0)
			{
				rejected[next] = true;
				next = -1;
				continue;
			}

			side[next] = 0;
			weight0 += graph.VertexWeights[next];
			for (var e = graph.XAdj[next]; e < graph.XAdj[next + 1]; e++)
			{
				work++;
				var u = graph.AdjNcy[e];
				connection[u] += graph.AdjWgt[e];
				touched[u] = true;
				if (side[u] == 1 && !rejected[u])
				{
					queue.Enqueue(u, (-((2 * connection[u]) - degree[u]), u));
				}
			}

			next = -1;
		}

		return side;
	}

	private (long Excess, long Cut) Score(Graph graph, int[] side, long[] maxWeights)
	{
		var weights = new long[2];
		long cut = 0;
		for (var v = 0; v < graph.NumVertices; v++)
		{
			work++;
			weights[side[v]] += graph.VertexWeights[v];
			for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
			{
				work++;
				if (side[graph.AdjNcy[e]] != side[v])
				{
					cut += graph.AdjWgt[e];
				}
			}
		}

		return (Excess(weights, maxWeights), cut / 2);
	}

	private long Excess(long[] weights, long[] maxWeights) =>
		Math.Max(0, weights[0] - maxWeights[0]) + Math.Max(0, weights[1] - maxWeights[1]);

	// Fiduccia-Mattheyses passes. Each pass moves every vertex at most once, always taking the
	// allowed move with the largest cut reduction (even a negative one, to climb out of local
	// minima), then rolls back to the best state seen. States compare by excess weight over
	// the balance limits first and cut second, so an unbalanced projection is repaired. A move
	// is allowed when the source is over its limit, or when the destination stays within its
	// limit plus one heaviest vertex. That slack lets a pass swap vertices when both sides sit
	// exactly at their limits (an exact split of unit weights leaves no other move); the
	// rollback only keeps a state that is back within the limits, or as close as it got.
	private void RefineFm(Graph graph, int[] side, long[] maxWeights)
	{
		var n = graph.NumVertices;
		var gains = new long[n];
		var locked = new bool[n];
		var moves = new List<int>(n);
		var maxNonImproving = Math.Clamp(n / 100, 15, 100);
		var minVertexWeight = n == 0 ? 0 : graph.VertexWeights.Min();
		var slack = n == 0 ? 0 : graph.VertexWeights.Max();
		for (var pass = 0; pass < MaxRefinementPasses; pass++)
		{
			var (excess, cut) = Score(graph, side, maxWeights);
			var weights = new long[2];
			var queues = new[] { new SortedSet<(long NegGain, int Vertex)>(), new SortedSet<(long NegGain, int Vertex)>() };
			for (var v = 0; v < n; v++)
			{
				work++;
				weights[side[v]] += graph.VertexWeights[v];
				gains[v] = Gain(graph, side, v);
				locked[v] = false;
				queues[side[v]].Add((-gains[v], v));
			}

			var bestScore = (excess, cut);
			var bestCount = 0;
			var nonImproving = 0;
			moves.Clear();
			while (true)
			{
				var v = SelectMove(graph, queues, weights, maxWeights, minVertexWeight, slack);
				if (v < 0)
				{
					break;
				}

				var from = side[v];
				queues[from].Remove((-gains[v], v));
				locked[v] = true;
				side[v] = 1 - from;
				weights[from] -= graph.VertexWeights[v];
				weights[1 - from] += graph.VertexWeights[v];
				cut -= gains[v];
				for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
				{
					work++;
					var u = graph.AdjNcy[e];
					if (locked[u])
					{
						continue;
					}

					queues[side[u]].Remove((-gains[u], u));
					gains[u] += side[u] == from ? 2L * graph.AdjWgt[e] : -2L * graph.AdjWgt[e];
					queues[side[u]].Add((-gains[u], u));
				}

				moves.Add(v);
				var score = (Excess(weights, maxWeights), cut);
				if (score.CompareTo(bestScore) < 0)
				{
					bestScore = score;
					bestCount = moves.Count;
					nonImproving = 0;
				}
				else if (++nonImproving > maxNonImproving)
				{
					break;
				}
			}

			for (var i = moves.Count - 1; i >= bestCount; i--)
			{
				side[moves[i]] = 1 - side[moves[i]];
			}

			if (bestCount == 0)
			{
				break;
			}
		}
	}

	// The cut reduction from moving v to the other side.
	private long Gain(Graph graph, int[] side, int v)
	{
		long gain = 0;
		for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
		{
			work++;
			gain += side[graph.AdjNcy[e]] != side[v] ? graph.AdjWgt[e] : -graph.AdjWgt[e];
		}

		return gain;
	}

	// The best allowed move from each side's queue; between the two, the larger gain wins,
	// then the move out of the side further over its limit, then the lower vertex index.
	private int SelectMove(
		Graph graph,
		SortedSet<(long NegGain, int Vertex)>[] queues,
		long[] weights,
		long[] maxWeights,
		int minVertexWeight,
		int slack)
	{
		var candidates = new (long NegGain, int Vertex)?[2];
		for (var from = 0; from < 2; from++)
		{
			var to = 1 - from;
			var sourceOver = weights[from] > maxWeights[from];
			if (!sourceOver && weights[to] + minVertexWeight > maxWeights[to] + slack)
			{
				// Nothing fits; skip the scan that would reject every queued vertex.
				continue;
			}

			foreach (var entry in queues[from])
			{
				work++;
				if (sourceOver || weights[to] + graph.VertexWeights[entry.Vertex] <= maxWeights[to] + slack)
				{
					candidates[from] = entry;
					break;
				}
			}
		}

		if (candidates[0] is not { } c0)
		{
			return candidates[1]?.Vertex ?? -1;
		}

		if (candidates[1] is not { } c1)
		{
			return c0.Vertex;
		}

		if (c0.NegGain != c1.NegGain)
		{
			return c0.NegGain < c1.NegGain ? c0.Vertex : c1.Vertex;
		}

		var over0 = weights[0] - maxWeights[0];
		var over1 = weights[1] - maxWeights[1];
		if (over0 != over1)
		{
			return over0 > over1 ? c0.Vertex : c1.Vertex;
		}

		return Math.Min(c0.Vertex, c1.Vertex);
	}

	// The subgraph induced by the vertices on side s, renumbered in increasing order.
	private (Graph Subgraph, int[] OriginalIds) ExtractSide(Graph graph, int[] originalIds, int[] side, int s)
	{
		var newIndex = new int[graph.NumVertices];
		var ids = new List<int>();
		var vertexWeights = new List<int>();
		for (var v = 0; v < graph.NumVertices; v++)
		{
			work++;
			newIndex[v] = -1;
			if (side[v] == s)
			{
				newIndex[v] = ids.Count;
				ids.Add(originalIds[v]);
				vertexWeights.Add(graph.VertexWeights[v]);
			}
		}

		var xadj = new int[ids.Count + 1];
		var adjncy = new List<int>();
		var adjwgt = new List<int>();
		var index = 0;
		for (var v = 0; v < graph.NumVertices; v++)
		{
			work++;
			if (side[v] != s)
			{
				continue;
			}

			for (var e = graph.XAdj[v]; e < graph.XAdj[v + 1]; e++)
			{
				work++;
				var u = graph.AdjNcy[e];
				if (side[u] == s)
				{
					adjncy.Add(newIndex[u]);
					adjwgt.Add(graph.AdjWgt[e]);
				}
			}

			xadj[++index] = adjncy.Count;
		}

		return (new Graph(xadj, [.. adjncy], [.. adjwgt], [.. vertexWeights]), [.. ids]);
	}
}
