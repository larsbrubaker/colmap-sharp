// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/parameter_block_ordering_test.cc,
// internal/ceres/graph_algorithms_test.cc (StableIndependentSet.BreakTies) and
// internal/ceres/reorder_program_test.cc (ReorderResidualBlockNormalFunction,
// ApplyOrderingOrderingTooSmall, ApplyOrderingNormal)
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SchurOrderingTests, StableIndependentSetTests and ReorderProgramTests (Ceres' tests, not
// COLMAP's): ColmapSharp/Solver/SchurOrdering.cs, same problems and expectations.
// - ApplyOrderingOrderingTooSmall applies the ordering to a copy of the problem's program,
//   as Ceres does; ApplyOrderingNormal to the problem's own.
// - SchurOrderingTest.OneFixed ends by checking that ComputeSchurOrdering puts the constant
//   block last. Only the stable variant (ComputeStableSchurOrdering, the one Ceres' Schur
//   solvers use) is ported, so the check runs on it; it appends constant blocks the same way.
// Not ported: graph_algorithms_test's IndependentSetOrdering, Degree2MaximumSpanningForest and
//   VertexTotalOrdering cases (those algorithms are not ported: unstable ordering, and the
//   visibility-clustering preconditioners); reorder_program_test's
//   ReorderProgramForSparseCholeskyUsingSuiteSparse cases (SuiteSparse is not ported) and
//   ReorderResidualBlocksbyPartition (the SUBSET preconditioner and CGNR are not ported).

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

/// <summary>A cost function with the given sizes that is never evaluated (Ceres' DummyCostFunction).</summary>
internal sealed class DummyCostFunction(int numResiduals, params int[] parameterBlockSizes)
	: CostFunction(numResiduals, parameterBlockSizes)
{
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians) => true;
}

public class SchurOrderingTests
{
	private readonly double[] x = new double[3];
	private readonly double[] y = new double[4];
	private readonly double[] z = new double[5];
	private readonly double[] w = new double[6];

	private Problem SetUp()
	{
		// The explicit calls to AddParameterBlock are necessary because the below tests
		// depend on the specific numbering of the parameter blocks.
		var problem = new Problem();
		problem.AddParameterBlock(x);
		problem.AddParameterBlock(y);
		problem.AddParameterBlock(z);
		problem.AddParameterBlock(w);

		problem.AddResidualBlock(new DummyCostFunction(2, 3), null, x);
		problem.AddResidualBlock(new DummyCostFunction(6, 5, 4), null, z, y);
		problem.AddResidualBlock(new DummyCostFunction(3, 3, 5), null, x, z);
		problem.AddResidualBlock(new DummyCostFunction(7, 5, 3), null, z, x);
		problem.AddResidualBlock(new DummyCostFunction(1, 5, 3, 6), null, z, x, w);
		return problem;
	}

	private static async Task ExpectNeighbors(Graph<ParameterBlock> graph, ParameterBlock vertex, params ParameterBlock[] expected)
	{
		IReadOnlySet<ParameterBlock> neighbors = graph.Neighbors(vertex);
		await Assert.That(neighbors.Count).IsEqualTo(expected.Length);
		foreach (ParameterBlock block in expected)
		{
			await Assert.That(neighbors.Contains(block)).IsTrue();
		}
	}

	[Test]
	public async Task NoFixed()
	{
		Program program = SetUp().Program;
		List<ParameterBlock> parameterBlocks = program.ParameterBlocks;
		Graph<ParameterBlock> graph = SchurOrdering.CreateHessianGraph(program);

		await Assert.That(graph.VertexCount).IsEqualTo(4);
		for (int i = 0; i < 4; ++i)
		{
			await Assert.That(graph.HasVertex(parameterBlocks[i])).IsTrue();
		}

		await ExpectNeighbors(graph, parameterBlocks[0], parameterBlocks[2], parameterBlocks[3]);
		await ExpectNeighbors(graph, parameterBlocks[1], parameterBlocks[2]);
		await ExpectNeighbors(graph, parameterBlocks[2], parameterBlocks[0], parameterBlocks[1], parameterBlocks[3]);
		await ExpectNeighbors(graph, parameterBlocks[3], parameterBlocks[0], parameterBlocks[2]);
	}

	[Test]
	public async Task AllFixed()
	{
		Problem problem = SetUp();
		problem.SetParameterBlockConstant(x);
		problem.SetParameterBlockConstant(y);
		problem.SetParameterBlockConstant(z);
		problem.SetParameterBlockConstant(w);

		Graph<ParameterBlock> graph = SchurOrdering.CreateHessianGraph(problem.Program);
		await Assert.That(graph.VertexCount).IsEqualTo(0);
	}

	[Test]
	public async Task OneFixed()
	{
		Problem problem = SetUp();
		problem.SetParameterBlockConstant(x);

		Program program = problem.Program;
		List<ParameterBlock> parameterBlocks = program.ParameterBlocks;
		Graph<ParameterBlock> graph = SchurOrdering.CreateHessianGraph(program);

		await Assert.That(graph.VertexCount).IsEqualTo(3);
		await Assert.That(graph.HasVertex(parameterBlocks[0])).IsFalse();
		for (int i = 1; i < 3; ++i)
		{
			await Assert.That(graph.HasVertex(parameterBlocks[i])).IsTrue();
		}

		await ExpectNeighbors(graph, parameterBlocks[1], parameterBlocks[2]);
		await ExpectNeighbors(graph, parameterBlocks[2], parameterBlocks[1], parameterBlocks[3]);
		await ExpectNeighbors(graph, parameterBlocks[3], parameterBlocks[2]);

		// The constant parameter block is at the end.
		SchurOrdering.ComputeStableSchurOrdering(program, out List<ParameterBlock> ordering);
		await Assert.That(ReferenceEquals(ordering[^1], parameterBlocks[0])).IsTrue();
	}
}

public class StableIndependentSetTests
{
	[Test]
	public async Task BreakTies()
	{
		var graph = new Graph<int>();
		graph.AddVertex(0);
		graph.AddVertex(1);
		graph.AddVertex(2);
		graph.AddVertex(3);

		graph.AddEdge(0, 1);
		graph.AddEdge(0, 2);
		graph.AddEdge(0, 3);
		graph.AddEdge(1, 2);
		graph.AddEdge(1, 3);
		graph.AddEdge(2, 3);

		// Since this is a completely connected graph, the independent set contains exactly
		// one vertex. StableIndependentSetOrdering guarantees that it will always be the first
		// vertex in the ordering vector.
		{
			List<int> ordering = [0, 1, 2, 3];
			int independentSetSize = SchurOrdering.StableIndependentSetOrdering(graph, ordering);
			await Assert.That(independentSetSize).IsEqualTo(1);
			await Assert.That(ordering[0]).IsEqualTo(0);
		}

		{
			List<int> ordering = [1, 0, 2, 3];
			int independentSetSize = SchurOrdering.StableIndependentSetOrdering(graph, ordering);
			await Assert.That(independentSetSize).IsEqualTo(1);
			await Assert.That(ordering[0]).IsEqualTo(1);
		}
	}
}

public class ReorderProgramTests
{
	[Test]
	public async Task ReorderResidualBlockNormalFunction()
	{
		var problem = new Problem();
		double[] x = new double[1];
		double[] y = new double[1];
		double[] z = new double[1];

		problem.AddParameterBlock(x);
		problem.AddParameterBlock(y);
		problem.AddParameterBlock(z);

		problem.AddResidualBlock(new DummyCostFunction(2, 1), null, x);
		problem.AddResidualBlock(new DummyCostFunction(2, 1, 1), null, z, x);
		problem.AddResidualBlock(new DummyCostFunction(2, 1, 1), null, z, y);
		problem.AddResidualBlock(new DummyCostFunction(2, 1), null, z);
		problem.AddResidualBlock(new DummyCostFunction(2, 1, 1), null, x, y);
		problem.AddResidualBlock(new DummyCostFunction(2, 1), null, y);

		// Ceres builds a linear_solver_ordering {x, y} -> 0, {z} -> 1 here that the call below
		// does not read; the elimination group size (2) is passed directly.
		List<ResidualBlock> residualBlocks = problem.Program.ResidualBlocks;

		// This is a bit fragile, but it serves the purpose. We know the bucketing algorithm
		// that the reordering function uses, so we expect the order for residual blocks for
		// each e_block to be filled in reverse.
		ResidualBlock[] expectedResidualBlocks =
		[
			residualBlocks[4],
			residualBlocks[1],
			residualBlocks[0],
			residualBlocks[5],
			residualBlocks[2],
			residualBlocks[3],
		];

		Program program = problem.Program;
		program.SetParameterOffsetsAndIndex();

		SchurOrdering.LexicographicallyOrderResidualBlocks(2, program);
		await Assert.That(residualBlocks.Count).IsEqualTo(expectedResidualBlocks.Length);
		for (int i = 0; i < expectedResidualBlocks.Length; ++i)
		{
			await Assert.That(ReferenceEquals(residualBlocks[i], expectedResidualBlocks[i])).IsTrue();
		}
	}

	[Test]
	public async Task ApplyOrderingOrderingTooSmall()
	{
		var problem = new Problem();
		double[] x = new double[1];
		double[] y = new double[1];
		double[] z = new double[1];

		problem.AddParameterBlock(x);
		problem.AddParameterBlock(y);
		problem.AddParameterBlock(z);

		var linearSolverOrdering = new ParameterBlockOrdering();
		linearSolverOrdering.AddElementToGroup(x, 0);
		linearSolverOrdering.AddElementToGroup(y, 1);

		var program = new Program(problem.Program);
		await Assert.That(SchurOrdering.ApplyOrdering(problem, linearSolverOrdering, program, out _)).IsFalse();
	}

	[Test]
	public async Task ApplyOrderingNormal()
	{
		var problem = new Problem();
		double[] x = new double[1];
		double[] y = new double[1];
		double[] z = new double[1];

		problem.AddParameterBlock(x);
		problem.AddParameterBlock(y);
		problem.AddParameterBlock(z);

		var linearSolverOrdering = new ParameterBlockOrdering();
		linearSolverOrdering.AddElementToGroup(x, 0);
		linearSolverOrdering.AddElementToGroup(y, 2);
		linearSolverOrdering.AddElementToGroup(z, 1);

		Program program = problem.Program;
		await Assert.That(SchurOrdering.ApplyOrdering(problem, linearSolverOrdering, program, out _)).IsTrue();
		List<ParameterBlock> parameterBlocks = program.ParameterBlocks;

		await Assert.That(parameterBlocks.Count).IsEqualTo(3);
		await Assert.That(ReferenceEquals(parameterBlocks[0].UserState.Array, x)).IsTrue();
		await Assert.That(ReferenceEquals(parameterBlocks[1].UserState.Array, z)).IsTrue();
		await Assert.That(ReferenceEquals(parameterBlocks[2].UserState.Array, y)).IsTrue();
	}
}
