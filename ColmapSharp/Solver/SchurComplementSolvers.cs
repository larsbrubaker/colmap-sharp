// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/schur_complement_solver.h/.cc and
// internal/ceres/dense_cholesky.cc (the Eigen back end) (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// DENSE_SCHUR and SPARSE_SCHUR: eliminate the E blocks with SchurEliminator.cs, solve the
// reduced camera system S z = rhs by Cholesky, then back-substitute the E variables.
// - DENSE_SCHUR stores S in a BlockRandomAccessDenseMatrix and factors it with
//   LinearAlgebra/LLT.cs (Eigen's LLT in Ceres).
// - SPARSE_SCHUR stores only the cells two F blocks share (through a chunk or a row without
//   an E block) and factors the lower triangle with LinearAlgebra/SimplicialCholesky.cs
//   (LLT, AMD on the scalar pattern). Ceres with EIGEN_SPARSE instead pre-orders the F blocks
//   by AMD on the block pattern and factors with Eigen's SimplicialLLT in natural order
//   (divergence 35).
// The pattern (chunks, S's cells, the Cholesky's symbolic analysis) is built on the first
// solve and whenever the Jacobian's block structure changes.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::SchurComplementSolver: the part shared by DENSE_SCHUR and SPARSE_SCHUR.</summary>
internal abstract class SchurComplementSolver(int numEliminateBlocks, int numThreads) : LinearSolver
{
	private readonly SchurEliminator eliminator = new(numThreads);
	private CompressedRowBlockStructure? structure;

	/// <summary>Number of leading column blocks eliminated (Ceres' elimination_groups[0]).</summary>
	protected int NumEliminateBlocks { get; } = numEliminateBlocks;

	/// <summary>The reduced camera matrix.</summary>
	protected BlockRandomAccessMatrix? Lhs { get; set; }

	/// <summary>The reduced right-hand side.</summary>
	protected double[] Rhs { get; set; } = [];

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance)
	{
		var matrix = (BlockSparseMatrix)a;
		CompressedRowBlockStructure bs = matrix.Structure;
		if (!ReferenceEquals(structure, bs))
		{
			structure = bs;
			InitStorage(bs);
			eliminator.Init(NumEliminateBlocks, bs);
		}

		int numCols = matrix.NumCols;
		x[..numCols].Clear();
		eliminator.Eliminate(matrix, b, d, Lhs!, Rhs);

		Span<double> reducedSolution = x[(numCols - Lhs!.NumRows)..numCols];
		LinearSolverSummary summary = SolveReducedLinearSystem(reducedSolution);
		if (summary.TerminationType == LinearSolverTerminationType.Success)
		{
			eliminator.BackSubstitute(matrix, b, d, reducedSolution, x);
		}

		return summary;
	}

	/// <summary>Creates the reduced system's storage for <paramref name="bs"/>.</summary>
	protected abstract void InitStorage(CompressedRowBlockStructure bs);

	/// <summary>Solves S z = rhs into <paramref name="solution"/>.</summary>
	protected abstract LinearSolverSummary SolveReducedLinearSystem(Span<double> solution);
}

/// <summary>ceres::internal::DenseSchurComplementSolver.</summary>
internal sealed class DenseSchurComplementSolver(int numEliminateBlocks, int numThreads = 1) : SchurComplementSolver(numEliminateBlocks, numThreads)
{
	/// <inheritdoc/>
	protected override void InitStorage(CompressedRowBlockStructure bs)
	{
		Lhs = new BlockRandomAccessDenseMatrix(SchurEliminator.ReducedBlocks(bs, NumEliminateBlocks));
		Rhs = new double[Lhs.NumRows];
	}

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveReducedLinearSystem(Span<double> solution)
	{
		var m = (BlockRandomAccessDenseMatrix)Lhs!;
		int numRows = m.NumRows;

		// The case where there are no f blocks, and the system is block diagonal.
		if (numRows == 0)
		{
			return new LinearSolverSummary(LinearSolverTerminationType.Success, 0, "Success.");
		}

		// The row-major upper triangle, read column-major, is the lower triangle LLT reads.
		var llt = new LLT(MatrixXd.FromColumnMajor(numRows, numRows, m.Values));
		if (llt.Info != ComputationInfo.Success)
		{
			return new LinearSolverSummary(
				LinearSolverTerminationType.Failure, 1, "Eigen failure. Unable to perform dense Cholesky factorization.");
		}

		llt.Solve(new VectorXd(Rhs)).AsSpan().CopyTo(solution);
		return new LinearSolverSummary(LinearSolverTerminationType.Success, 1, "Success.");
	}
}

/// <summary>ceres::internal::SparseSchurComplementSolver (the direct factorization; ITERATIVE_SCHUR is IterativeSchurSolver.cs).</summary>
internal sealed class SparseSchurComplementSolver(int numEliminateBlocks, int numThreads = 1) : SchurComplementSolver(numEliminateBlocks, numThreads)
{
	private readonly SimplicialCholesky cholesky = new(SimplicialCholeskyKind.LLT, SparseOrdering.Amd, SymmetricPart.Lower);
	private SparseMatrixCsc? crsLhs;
	private int[] valueSources = [];

	/// <summary>
	/// Determines the non-zero blocks of the Schur complement: every diagonal block, every
	/// pair of F blocks in one chunk, and every pair in one row without an E block.
	/// </summary>
	protected override void InitStorage(CompressedRowBlockStructure bs)
	{
		int numEliminate = NumEliminateBlocks;
		Block[] blocks = SchurEliminator.ReducedBlocks(bs, numEliminate);
		var blockPairs = new SortedSet<(int, int)>();
		for (int i = 0; i < blocks.Length; i++)
		{
			blockPairs.Add((i, i));
		}

		CompressedRow[] rows = bs.Rows;
		int r = 0;
		while (r < rows.Length)
		{
			int eBlockId = rows[r].Cells[0].BlockId;
			if (eBlockId >= numEliminate)
			{
				break;
			}

			var fBlocks = new SortedSet<int>();

			// Add to the chunk until the first block in the row is different than the one in
			// the first row for the chunk.
			for (; r < rows.Length && rows[r].Cells[0].BlockId == eBlockId; r++)
			{
				// Iterate over the blocks in the row, ignoring the first block since it is the
				// one to be eliminated.
				Cell[] cells = rows[r].Cells;
				for (int c = 1; c < cells.Length; c++)
				{
					fBlocks.Add(cells[c].BlockId - numEliminate);
				}
			}

			int[] sorted = [.. fBlocks];
			for (int i = 0; i < sorted.Length; i++)
			{
				for (int j = i + 1; j < sorted.Length; j++)
				{
					blockPairs.Add((sorted[i], sorted[j]));
				}
			}
		}

		// Remaining rows do not contribute to the chunks and directly go into the schur
		// complement via an outer product.
		for (; r < rows.Length; r++)
		{
			Cell[] cells = rows[r].Cells;
			Util.Check.Ge(cells[0].BlockId, numEliminate);
			foreach (Cell cell1 in cells)
			{
				foreach (Cell cell2 in cells)
				{
					if (cell1.BlockId <= cell2.BlockId)
					{
						blockPairs.Add((cell1.BlockId - numEliminate, cell2.BlockId - numEliminate));
					}
				}
			}
		}

		var lhs = new BlockRandomAccessSparseMatrix(blocks, blockPairs);
		Lhs = lhs;
		Rhs = new double[lhs.NumRows];
		crsLhs = null;
		if (lhs.NumRows > 0)
		{
			crsLhs = lhs.ToLowerCsc(out valueSources);
			cholesky.AnalyzePattern(crsLhs);
		}
	}

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveReducedLinearSystem(Span<double> solution)
	{
		if (crsLhs is null)
		{
			return new LinearSolverSummary(LinearSolverTerminationType.Success, 0, "Success.");
		}

		double[] cellValues = ((BlockRandomAccessSparseMatrix)Lhs!).Values;
		Span<double> values = crsLhs.Values;
		for (int i = 0; i < valueSources.Length; i++)
		{
			values[i] = cellValues[valueSources[i]];
		}

		if (!cholesky.Factorize(crsLhs))
		{
			return new LinearSolverSummary(
				LinearSolverTerminationType.Failure, 1, "Eigen failure. Unable to find numeric factorization.");
		}

		cholesky.Solve(new VectorXd(Rhs)).AsSpan().CopyTo(solution);
		return new LinearSolverSummary(LinearSolverTerminationType.Success, 1, "Success.");
	}
}
