// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/parameter_block_ordering.cc
// (ComputeStableSchurOrdering, CreateHessianGraph), internal/ceres/graph_algorithms.h
// (StableIndependentSetOrdering) and internal/ceres/reorder_program.cc
// (ReorderProgramForSchurTypeLinearSolver, ApplyOrdering, LexicographicallyOrderResidualBlocks)
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The ordering the Schur solvers need. With a user ParameterBlockOrdering of several groups
// (COLMAP's global positioner), the first group must be an independent set and is
// eliminated; the program takes the groups' order (ApplyOrdering). Otherwise (no ordering, as
// in COLMAP's bundle adjuster, or a single group) it is chosen automatically: the E blocks are a
// large independent set of the Hessian graph (no two share a residual block), found greedily
// in order of increasing degree, which in bundle adjustment picks the points. The reduced
// program's parameter blocks become [E blocks, F blocks], and its residual blocks are
// grouped by their E block (the "chunks" SchurEliminator.cs walks), rows without an E block
// last. LeastSquaresSolver.cs calls this before building the evaluator, whose Jacobian
// layout (BlockSparseMatrix.cs) then puts the E cells first.
//
// Not ported: the fill-reducing pre-ordering of the F blocks for SPARSE_SCHUR, which the
// sparse Cholesky's own AMD replaces (docs/CPP_DIVERGENCES.md entry 35).

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>The automatic Schur elimination ordering of a reduced program.</summary>
internal static class SchurOrdering
{
	/// <summary>
	/// ReorderProgramForSchurTypeLinearSolver: reorders <paramref name="program"/> (the
	/// reduced one) for <paramref name="ordering"/>, or automatically when it is null or has a
	/// single group, and gives the number of E blocks. False with <paramref name="error"/> set
	/// if the ordering does not cover the program or its first group is not independent.
	/// </summary>
	public static bool ReorderProgramForSchurTypeLinearSolver(
		Problem problem, ParameterBlockOrdering? ordering, Program program, out int sizeOfFirstEliminationGroup, out string error)
	{
		error = string.Empty;
		sizeOfFirstEliminationGroup = 0;
		if (ordering is not null && ordering.NumElements != program.ParameterBlocks.Count)
		{
			error = $"The program has {program.ParameterBlocks.Count} parameter blocks, but the parameter block ordering has {ordering.NumElements} parameter blocks.";
			return false;
		}

		if (ordering is null || ordering.NumGroups == 1)
		{
			// Ceres is completely free to choose the parameter block ordering: the e_blocks are
			// a maximal independent set.
			sizeOfFirstEliminationGroup = ReorderProgramForSchurTypeLinearSolver(program);
			return true;
		}

		// The user provided an ordering with more than one elimination group. Verify that the
		// first elimination group is an independent set.
		var firstEliminationGroup = new HashSet<ParameterBlock>(ReferenceEqualityComparer.Instance);
		int firstGroupSize = 0;
		foreach ((double[] array, int offset) in ordering.Groups().First().Elements)
		{
			firstGroupSize++;
			if (problem.Find(array, offset) is ParameterBlock block)
			{
				firstEliminationGroup.Add(block);
			}
		}

		if (!program.IsParameterBlockSetIndependent(firstEliminationGroup))
		{
			error = $"The first elimination group in the parameter block ordering of size {firstGroupSize} is not an independent set";
			return false;
		}

		if (!ApplyOrdering(problem, ordering, program, out error))
		{
			return false;
		}

		program.SetParameterOffsetsAndIndex();
		sizeOfFirstEliminationGroup = firstGroupSize;

		// Schur type solvers also require that their residual blocks be lexicographically
		// ordered.
		LexicographicallyOrderResidualBlocks(sizeOfFirstEliminationGroup, program);
		program.SetParameterOffsetsAndIndex();
		return true;
	}

	/// <summary>
	/// ApplyOrdering: the program's parameter blocks become the ordering's, group by group
	/// (each in insertion order, ParameterBlockOrdering.cs). False with an error if the sizes
	/// differ or an element is not a block of <paramref name="problem"/>.
	/// </summary>
	public static bool ApplyOrdering(Problem problem, ParameterBlockOrdering ordering, Program program, out string error)
	{
		error = string.Empty;
		int numParameterBlocks = program.ParameterBlocks.Count;
		if (ordering.NumElements != numParameterBlocks)
		{
			error = "User specified ordering does not have the same number of parameters as the problem. The problem"
				+ $"has {numParameterBlocks} blocks while the ordering has {ordering.NumElements} blocks.";
			return false;
		}

		var parameterBlocks = new List<ParameterBlock>(numParameterBlocks);
		foreach ((int group, IEnumerable<(double[] Array, int Offset)> elements) in ordering.Groups())
		{
			foreach ((double[] array, int offset) in elements)
			{
				ParameterBlock? block = problem.Find(array, offset);
				if (block is null)
				{
					error = "User specified ordering contains a pointer to a double that is not a parameter block in "
						+ $"the problem. The invalid double is in group: {group}";
					return false;
				}

				parameterBlocks.Add(block);
			}
		}

		program.ParameterBlocks.Clear();
		program.ParameterBlocks.AddRange(parameterBlocks);
		return true;
	}

	// The automatic (single group) case: reorders the program and returns the number of E
	// blocks.
	private static int ReorderProgramForSchurTypeLinearSolver(Program program)
	{
		int sizeOfFirstEliminationGroup = ComputeStableSchurOrdering(program, out List<ParameterBlock> schurOrdering);
		Check.Eq(schurOrdering.Count, program.ParameterBlocks.Count, "Congratulations, you found a Ceres bug! Please report this error to the developers.");
		program.ParameterBlocks.Clear();
		program.ParameterBlocks.AddRange(schurOrdering);
		program.SetParameterOffsetsAndIndex();

		// Schur type solvers also require that their residual blocks be lexicographically
		// ordered.
		LexicographicallyOrderResidualBlocks(sizeOfFirstEliminationGroup, program);
		program.SetParameterOffsetsAndIndex();
		return sizeOfFirstEliminationGroup;
	}

	/// <summary>
	/// ComputeStableSchurOrdering: the variable blocks as an independent set (returned count)
	/// followed by the rest, each part by increasing degree with ties in program order,
	/// then the constant blocks.
	/// </summary>
	public static int ComputeStableSchurOrdering(Program program, out List<ParameterBlock> ordering)
	{
		Graph<ParameterBlock> graph = CreateHessianGraph(program);
		List<ParameterBlock> parameterBlocks = program.ParameterBlocks;
		ordering = [.. parameterBlocks.Where(graph.HasVertex)];
		int independentSetSize = StableIndependentSetOrdering(graph, ordering);

		// Add the excluded blocks to back of the ordering vector.
		foreach (ParameterBlock parameterBlock in parameterBlocks)
		{
			if (parameterBlock.IsConstant)
			{
				ordering.Add(parameterBlock);
			}
		}

		return independentSetSize;
	}

	/// <summary>
	/// CreateHessianGraph: a vertex per variable block, an edge between variable blocks that
	/// share a residual block.
	/// </summary>
	public static Graph<ParameterBlock> CreateHessianGraph(Program program)
	{
		var graph = new Graph<ParameterBlock>();
		foreach (ParameterBlock parameterBlock in program.ParameterBlocks)
		{
			if (!parameterBlock.IsConstant)
			{
				graph.AddVertex(parameterBlock);
			}
		}

		foreach (ResidualBlock residualBlock in program.ResidualBlocks)
		{
			ParameterBlock[] blocks = residualBlock.ParameterBlocks;
			for (int j = 0; j < blocks.Length; j++)
			{
				if (blocks[j].IsConstant)
				{
					continue;
				}

				for (int k = j + 1; k < blocks.Length; k++)
				{
					if (blocks[k].IsConstant)
					{
						continue;
					}

					graph.AddEdge(blocks[j], blocks[k]);
				}
			}
		}

		return graph;
	}

	/// <summary>
	/// StableIndependentSetOrdering (graph_algorithms.h): reorders <paramref name="ordering"/>
	/// (which must hold every vertex of the graph) into a large independent set followed by
	/// the remaining vertices, each part in increasing order of degree with ties kept in the
	/// given order (a greedy search in that order), and returns the size of the set.
	/// </summary>
	public static int StableIndependentSetOrdering<T>(Graph<T> graph, List<T> ordering)
		where T : notnull
	{
		Check.Eq(graph.VertexCount, ordering.Count);

		// Colors for labeling the graph during the BFS.
		const byte White = 0;
		const byte Grey = 1;
		const byte Black = 2;

		// Stable sort by degree (LINQ OrderBy is stable, as std::stable_sort).
		T[] vertexQueue = [.. ordering.OrderBy(v => graph.Neighbors(v).Count)];

		// Mark all vertices white.
		var vertexColor = new Dictionary<T, byte>(graph.VertexCount);
		foreach (T vertex in vertexQueue)
		{
			vertexColor[vertex] = White;
		}

		ordering.Clear();

		// Iterate over vertex_queue. Pick the first white vertex, add it to the independent
		// set. Mark it black and its neighbors grey.
		foreach (T vertex in vertexQueue)
		{
			if (vertexColor[vertex] != White)
			{
				continue;
			}

			ordering.Add(vertex);
			vertexColor[vertex] = Black;
			foreach (T neighbor in graph.Neighbors(vertex))
			{
				vertexColor[neighbor] = Grey;
			}
		}

		int independentSetSize = ordering.Count;

		// Iterate over the vertices and add all the grey vertices to the ordering. At this
		// stage there should only be black or grey vertices in the graph.
		foreach (T vertex in vertexQueue)
		{
			if (vertexColor[vertex] != Black)
			{
				ordering.Add(vertex);
			}
		}

		Check.Eq(ordering.Count, graph.VertexCount);
		return independentSetSize;
	}

	/// <summary>
	/// LexicographicallyOrderResidualBlocks: residual blocks bucketed by their lowest E
	/// block (rows without one last). Each bucket is filled from its back, so within a bucket
	/// the residual blocks end up in reverse program order, exactly as in Ceres.
	/// </summary>
	public static void LexicographicallyOrderResidualBlocks(int sizeOfFirstEliminationGroup, Program program)
	{
		Check.Ge(sizeOfFirstEliminationGroup, 1, "Congratulations, you found a Ceres bug! Please report this error to the developers.");

		// Create a histogram of the number of residuals for each E block. There is an extra
		// bucket at the end to catch all non-eliminated F blocks.
		List<ResidualBlock> residualBlocks = program.ResidualBlocks;
		var residualBlocksPerEBlock = new int[sizeOfFirstEliminationGroup + 1];
		var minPositionPerResidual = new int[residualBlocks.Count];
		for (int i = 0; i < residualBlocks.Count; i++)
		{
			int position = MinParameterBlock(residualBlocks[i], sizeOfFirstEliminationGroup);
			minPositionPerResidual[i] = position;
			residualBlocksPerEBlock[position]++;
		}

		// Run a cumulative sum on the histogram, to obtain offsets to the start of each
		// histogram bucket (where each bucket is for the residuals for that E-block).
		var offsets = new int[sizeOfFirstEliminationGroup + 1];
		for (int i = 0, sum = 0; i < offsets.Length; i++)
		{
			sum += residualBlocksPerEBlock[i];
			offsets[i] = sum;
		}

		Check.Eq(offsets[^1], residualBlocks.Count);
		for (int i = 0; i < sizeOfFirstEliminationGroup; i++)
		{
			Check.That(residualBlocksPerEBlock[i] != 0, "Congratulations, you found a Ceres bug! Please report this error to the developers.");
		}

		// Fill in each bucket with the residual blocks for its corresponding E block. Each
		// bucket is individually filled from the back of the bucket to the front of the
		// bucket. The filling order among the buckets is dictated by the residual blocks.
		// This loop uses the offsets as counters; subtracting one from each offset as a
		// residual block is placed in the bucket.
		var reordered = new ResidualBlock[residualBlocks.Count];
		for (int i = 0; i < residualBlocks.Count; i++)
		{
			int bucket = minPositionPerResidual[i];
			offsets[bucket]--;
			Check.That(reordered[offsets[bucket]] is null);
			reordered[offsets[bucket]] = residualBlocks[i];
		}

		residualBlocks.Clear();
		residualBlocks.AddRange(reordered);
	}

	// The lowest index of a variable block of the residual block, where indices at or past
	// the first elimination group count as its size.
	private static int MinParameterBlock(ResidualBlock residualBlock, int sizeOfFirstEliminationGroup)
	{
		int minParameterBlockPosition = sizeOfFirstEliminationGroup;
		foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
		{
			if (!parameterBlock.IsConstant)
			{
				Check.Ne(parameterBlock.Index, -1, "Did you forget to call Program::SetParameterOffsetsAndIndex()?");
				minParameterBlockPosition = Math.Min(parameterBlock.Index, minParameterBlockPosition);
			}
		}

		return minParameterBlockPosition;
	}
}

/// <summary>
/// ceres::internal::Graph: an undirected graph without weights. Neighbor sets are hash sets
/// as in Ceres; nothing here depends on their iteration order (the independent set search
/// only colors neighbors).
/// </summary>
internal sealed class Graph<T>
	where T : notnull
{
	private readonly Dictionary<T, HashSet<T>> edges = [];

	/// <summary>Number of vertices.</summary>
	public int VertexCount => edges.Count;

	/// <summary>The vertices.</summary>
	public IEnumerable<T> Vertices => edges.Keys;

	/// <summary>Adds a vertex (no-op if present).</summary>
	public void AddVertex(T vertex) => edges.TryAdd(vertex, []);

	/// <summary>True if <paramref name="vertex"/> is in the graph.</summary>
	public bool HasVertex(T vertex) => edges.ContainsKey(vertex);

	/// <summary>Adds the undirected edge; both vertices must already be in the graph.</summary>
	public void AddEdge(T vertex1, T vertex2)
	{
		Check.That(edges.ContainsKey(vertex1) && edges.ContainsKey(vertex2), "AddEdge needs both vertices in the graph.");
		edges[vertex1].Add(vertex2);
		edges[vertex2].Add(vertex1);
	}

	/// <summary>The neighbors of <paramref name="vertex"/>.</summary>
	public IReadOnlySet<T> Neighbors(T vertex) => edges[vertex];
}
