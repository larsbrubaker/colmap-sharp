// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/linear_least_squares_problems.cc (problems 2,
// 3 and 4, the BlockSparseMatrix versions) (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The small block-sparse least-squares problems Ceres' Schur solver tests use
// (SchurEliminatorTests, ImplicitSchurComplementTests, SchurComplementSolverTests,
// IterativeSchurComplementSolverTests), with the same values, plus the dense copy of A
// the reference solutions are computed from.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Tests.Solver;

/// <summary>ceres::internal::LinearLeastSquaresProblem: min |A x - b|^2 + |D x|^2.</summary>
internal sealed record LinearLeastSquaresProblem(BlockSparseMatrix A, double[] B, double[] D, int NumEliminateBlocks)
{
	/// <summary>A as a dense matrix (BlockSparseMatrix::ToDenseMatrix).</summary>
	public MatrixXd DenseA()
	{
		var dense = new MatrixXd(A.NumRows, A.NumCols);
		CompressedRowBlockStructure bs = A.Structure;
		foreach (CompressedRow row in bs.Rows)
		{
			foreach (Cell cell in row.Cells)
			{
				Block col = bs.Cols[cell.BlockId];
				for (int r = 0; r < row.Block.Size; r++)
				{
					for (int c = 0; c < col.Size; c++)
					{
						dense[row.Block.Position + r, col.Position + c] += A.Values[cell.Position + (r * col.Size) + c];
					}
				}
			}
		}

		return dense;
	}

	/// <summary>A as the DenseSparseMatrix the DENSE_QR reference solver takes.</summary>
	public DenseSparseMatrix DenseSparseA()
	{
		var dense = new DenseSparseMatrix(A.NumRows, A.NumCols);
		DenseA().AsSpan().CopyTo(dense.Matrix.AsSpan());
		return dense;
	}
}

/// <summary>ceres::internal::CreateLinearLeastSquaresProblemFromId (the ids the Schur tests use).</summary>
internal static class LinearLeastSquaresProblems
{
	public static LinearLeastSquaresProblem FromId(int id) => id switch
	{
		2 => Problem2(),
		3 => Problem3(),
		4 => Problem4(),
		_ => throw new ArgumentOutOfRangeException(nameof(id), id, "Unknown problem id."),
	};

	// A = [1 0 | 2 0 0
	//      3 0 | 0 4 0
	//      0 5 | 0 0 6
	//      0 7 | 8 0 0
	//      0 9 | 1 0 0
	//      0 0 | 1 1 1]
	// b = [0 1 2 3 4 5], D = [1 1 1 1 1], two E blocks.
	private static LinearLeastSquaresProblem Problem2()
	{
		Block[] cols = [.. Enumerable.Range(0, 5).Select(c => new Block(1, c))];
		CompressedRow[] rows =
		[
			new(new Block(1, 0), [new Cell(0, 0), new Cell(2, 1)]),
			new(new Block(1, 1), [new Cell(0, 2), new Cell(3, 3)]),
			new(new Block(1, 2), [new Cell(1, 4), new Cell(4, 5)]),
			new(new Block(1, 3), [new Cell(1, 6), new Cell(2, 7)]),
			new(new Block(1, 4), [new Cell(1, 8), new Cell(2, 9)]),
			new(new Block(1, 5), [new Cell(2, 10), new Cell(3, 11), new Cell(4, 12)]),
		];
		double[] values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 1];
		return Build(cols, rows, values, [1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], 2);
	}

	// A = [1 0; 3 0; 0 5; 0 7; 0 9], b = [0 1 2 3 4], D = [1 1], both blocks eliminated
	// (no F blocks).
	private static LinearLeastSquaresProblem Problem3()
	{
		Block[] cols = [new Block(1, 0), new Block(1, 1)];
		CompressedRow[] rows =
		[
			new(new Block(1, 0), [new Cell(0, 0)]),
			new(new Block(1, 1), [new Cell(0, 1)]),
			new(new Block(1, 2), [new Cell(1, 2)]),
			new(new Block(1, 3), [new Cell(1, 3)]),
			new(new Block(1, 4), [new Cell(1, 4)]),
		];
		return Build(cols, rows, [1, 3, 5, 7, 9], [1, 1], [0, 1, 2, 3, 4], 2);
	}

	// A = [1 2 0 0 0 1 1
	//      1 4 0 0 0 5 6
	//      0 0 9 0 0 3 1], b = [0 1 2], D = [100 200 ... 700], one E block.
	// Two different sized f-blocks, but only one of them occurs in the rows involving the
	// one e-block, which tests the handling of non-e-block rows whose structure does not
	// conform to the static structure. Too small and rank deficient to be solved without the
	// diagonal regularization.
	private static LinearLeastSquaresProblem Problem4()
	{
		Block[] cols = [new Block(2, 0), new Block(3, 2), new Block(2, 5)];
		CompressedRow[] rows =
		[
			new(new Block(2, 0), [new Cell(0, 0), new Cell(2, 4)]),
			new(new Block(1, 2), [new Cell(1, 8), new Cell(2, 11)]),
		];
		double[] values = [1, 2, 1, 4, 1, 1, 5, 6, 9, 0, 0, 3, 1];
		double[] d = [.. Enumerable.Range(0, 7).Select(i => (i + 1) * 100.0)];
		return Build(cols, rows, values, d, [0, 1, 2], 1);
	}

	private static LinearLeastSquaresProblem Build(
		Block[] cols, CompressedRow[] rows, double[] values, double[] d, double[] b, int numEliminateBlocks)
	{
		var a = new BlockSparseMatrix(new CompressedRowBlockStructure(cols, rows));
		values.CopyTo(a.Values, 0);
		return new LinearLeastSquaresProblem(a, b, d, numEliminateBlocks);
	}
}
