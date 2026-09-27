// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SchurEliminatorParallelTests (C#-only; no Ceres counterpart): the two-phase parallel
// elimination (ColmapSharp/Solver/SchurEliminator.Parallel.cs) gives the same bits as the
// sequential loop for the reduced matrix, the reduced right-hand side and the back
// substitution, on Ceres' test problems 2 and 4 (the ones SchurEliminatorTests uses), with
// and without regularization, into a dense matrix (whose lower triangle unsorted rows may
// write) and a block diagonal one with no right-hand side (the SCHUR_JACOBI preconditioner).
// A batch size of 1 puts every chunk in its own batch, so the order across batches is
// covered too. SchurDeterminismTests covers the same property end to end on bundle
// adjustment.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class SchurEliminatorParallelTests
{
	private static bool SameBits(double[] x, double[] y) =>
		x.Length == y.Length && x.Zip(y).All(p => BitConverter.DoubleToInt64Bits(p.First) == BitConverter.DoubleToInt64Bits(p.Second));

	private static double[] Regularization(int n, bool regularize)
	{
		var d = new double[n];
		for (int i = 0; i < n; i++)
		{
			d[i] = regularize ? 1.0 + (0.37 * i) : 0.0;
		}

		return d;
	}

	// The dense lhs, rhs and back-substituted solution, concatenated.
	private static double[] EliminateDense(LinearLeastSquaresProblem problem, double[] d, int numThreads, int maxBatchBufferSize)
	{
		CompressedRowBlockStructure bs = problem.A.Structure;
		var lhs = new BlockRandomAccessDenseMatrix(SchurEliminator.ReducedBlocks(bs, problem.NumEliminateBlocks));
		var rhs = new double[lhs.NumRows];
		var eliminator = new SchurEliminator(numThreads) { MaxBatchBufferSize = maxBatchBufferSize };
		eliminator.Init(problem.NumEliminateBlocks, bs);
		eliminator.Eliminate(problem.A, problem.B, d, lhs, rhs);

		// Any z will do: back substitution must agree for the same z.
		int numCols = problem.A.NumCols;
		var z = new double[lhs.NumRows];
		for (int i = 0; i < z.Length; i++)
		{
			z[i] = 0.5 - (0.25 * i);
		}

		var y = new double[numCols];
		eliminator.BackSubstitute(problem.A, problem.B, d, z, y);
		return [.. lhs.Values, .. rhs, .. y];
	}

	private static double[] EliminateBlockDiagonal(LinearLeastSquaresProblem problem, double[] d, int numThreads, int maxBatchBufferSize)
	{
		CompressedRowBlockStructure bs = problem.A.Structure;
		var lhs = new BlockRandomAccessDiagonalMatrix(SchurEliminator.ReducedBlocks(bs, problem.NumEliminateBlocks));
		var eliminator = new SchurEliminator(numThreads) { MaxBatchBufferSize = maxBatchBufferSize };
		eliminator.Init(problem.NumEliminateBlocks, bs);
		eliminator.Eliminate(problem.A, [], d, lhs, []);
		return lhs.Values;
	}

	[Test]
	[Arguments(2, false)]
	[Arguments(2, true)]
	[Arguments(4, false)]
	[Arguments(4, true)]
	public async Task ParallelEliminationMatchesSequential(int problemId, bool regularize)
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(problemId);
		double[] d = Regularization(problem.A.NumCols, regularize);
		double[] sequential = EliminateDense(problem, d, 1, 1 << 22);
		double[] sequentialDiagonal = EliminateBlockDiagonal(problem, d, 1, 1 << 22);
		foreach (int batch in new[] { 1, 1 << 22 })
		{
			foreach (int threads in new[] { 2, 4 })
			{
				await Assert.That(SameBits(EliminateDense(problem, d, threads, batch), sequential)).IsTrue();
				await Assert.That(SameBits(EliminateBlockDiagonal(problem, d, threads, batch), sequentialDiagonal)).IsTrue();
			}
		}
	}
}
