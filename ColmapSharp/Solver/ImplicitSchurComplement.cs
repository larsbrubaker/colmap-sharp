// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/implicit_schur_complement.h/.cc and
// internal/ceres/partitioned_matrix_view.h / partitioned_matrix_view_impl.h (the
// single-threaded paths) (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The Schur complement S = F'F + D_f^2 - F'E (E'E + D_e^2)^-1 E'F of a block-sparse
// Jacobian A = [E F] as a linear operator, never formed: S x is computed from products with
// E, F and the block-diagonal (E'E + D_e^2)^-1. ITERATIVE_SCHUR (IterativeSchurSolver.cs)
// runs conjugate gradients on it. The rows must be ordered as for SchurEliminator.cs: the
// rows of each E block together, before the rows with no E block. Every loop walks the row
// blocks in order, as Ceres' single-threaded code does, so the result does not depend on the
// thread count.

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::PartitionedMatrixView: products with the E and F column parts of A.</summary>
internal sealed class PartitionedMatrixView
{
	private readonly BlockSparseMatrix matrix;
	private readonly int numColBlocksE;

	/// <summary>Views <paramref name="matrix"/> with its first <paramref name="numColBlocksE"/> column blocks as E.</summary>
	public PartitionedMatrixView(BlockSparseMatrix matrix, int numColBlocksE)
	{
		this.matrix = matrix;
		this.numColBlocksE = numColBlocksE;
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int c = 0; c < numColBlocksE; c++)
		{
			NumColsE += bs.Cols[c].Size;
		}

		for (int c = numColBlocksE; c < bs.Cols.Length; c++)
		{
			NumColsF += bs.Cols[c].Size;
		}

		// Compute the number of row blocks in E. The number of row blocks in E may be less
		// than the number of row blocks in the input matrix as some of the row blocks at the
		// bottom may not have any e_blocks.
		while (NumRowBlocksE < bs.Rows.Length && bs.Rows[NumRowBlocksE].Cells[0].BlockId < numColBlocksE)
		{
			NumRowBlocksE++;
		}
	}

	/// <summary>Scalar columns of E.</summary>
	public int NumColsE { get; }

	/// <summary>Scalar columns of F.</summary>
	public int NumColsF { get; }

	/// <summary>Leading row blocks that have an E cell.</summary>
	public int NumRowBlocksE { get; }

	/// <summary>Rows of A.</summary>
	public int NumRows => matrix.NumRows;

	/// <summary>The E column blocks, positioned from 0.</summary>
	public Block[] EBlocks()
	{
		var blocks = new Block[numColBlocksE];
		Array.Copy(matrix.Structure.Cols, blocks, numColBlocksE);
		return blocks;
	}

	/// <summary>y += E x.</summary>
	public void RightMultiplyAndAccumulateE(ReadOnlySpan<double> x, Span<double> y)
	{
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < NumRowBlocksE; r++)
		{
			CompressedRow row = bs.Rows[r];
			Cell cell = row.Cells[0];
			Block col = bs.Cols[cell.BlockId];
			SmallBlas.MatrixVectorMultiply(
				matrix.Values.AsSpan(cell.Position), row.Block.Size, col.Size, x.Slice(col.Position, col.Size), y.Slice(row.Block.Position, row.Block.Size), 1);
		}
	}

	/// <summary>y += F x.</summary>
	public void RightMultiplyAndAccumulateF(ReadOnlySpan<double> x, Span<double> y)
	{
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < bs.Rows.Length; r++)
		{
			CompressedRow row = bs.Rows[r];
			for (int c = r < NumRowBlocksE ? 1 : 0; c < row.Cells.Length; c++)
			{
				Cell cell = row.Cells[c];
				Block col = bs.Cols[cell.BlockId];
				SmallBlas.MatrixVectorMultiply(
					matrix.Values.AsSpan(cell.Position),
					row.Block.Size,
					col.Size,
					x.Slice(col.Position - NumColsE, col.Size),
					y.Slice(row.Block.Position, row.Block.Size),
					1);
			}
		}
	}

	/// <summary>y += E' x.</summary>
	public void LeftMultiplyAndAccumulateE(ReadOnlySpan<double> x, Span<double> y)
	{
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < NumRowBlocksE; r++)
		{
			CompressedRow row = bs.Rows[r];
			Cell cell = row.Cells[0];
			Block col = bs.Cols[cell.BlockId];
			SmallBlas.MatrixTransposeVectorMultiply(
				matrix.Values.AsSpan(cell.Position), row.Block.Size, col.Size, x.Slice(row.Block.Position, row.Block.Size), y.Slice(col.Position, col.Size), 1);
		}
	}

	/// <summary>y += F' x.</summary>
	public void LeftMultiplyAndAccumulateF(ReadOnlySpan<double> x, Span<double> y)
	{
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < bs.Rows.Length; r++)
		{
			CompressedRow row = bs.Rows[r];
			for (int c = r < NumRowBlocksE ? 1 : 0; c < row.Cells.Length; c++)
			{
				Cell cell = row.Cells[c];
				Block col = bs.Cols[cell.BlockId];
				SmallBlas.MatrixTransposeVectorMultiply(
					matrix.Values.AsSpan(cell.Position),
					row.Block.Size,
					col.Size,
					x.Slice(row.Block.Position, row.Block.Size),
					y.Slice(col.Position - NumColsE, col.Size),
					1);
			}
		}
	}

	/// <summary>UpdateBlockDiagonalEtE: the diagonal blocks of E'E into <paramref name="diagonal"/>.</summary>
	public void UpdateBlockDiagonalEtE(BlockRandomAccessDiagonalMatrix diagonal)
	{
		diagonal.SetZero();
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < NumRowBlocksE; r++)
		{
			CompressedRow row = bs.Rows[r];
			Cell cell = row.Cells[0];
			AddOuterProduct(diagonal, cell.BlockId, row.Block.Size, cell.Position, bs.Cols[cell.BlockId].Size);
		}
	}

	/// <summary>UpdateBlockDiagonalFtF: the diagonal blocks of F'F into <paramref name="diagonal"/>.</summary>
	public void UpdateBlockDiagonalFtF(BlockRandomAccessDiagonalMatrix diagonal)
	{
		diagonal.SetZero();
		CompressedRowBlockStructure bs = matrix.Structure;
		for (int r = 0; r < bs.Rows.Length; r++)
		{
			CompressedRow row = bs.Rows[r];
			for (int c = r < NumRowBlocksE ? 1 : 0; c < row.Cells.Length; c++)
			{
				Cell cell = row.Cells[c];
				AddOuterProduct(diagonal, cell.BlockId - numColBlocksE, row.Block.Size, cell.Position, bs.Cols[cell.BlockId].Size);
			}
		}
	}

	private void AddOuterProduct(BlockRandomAccessDiagonalMatrix diagonal, int block, int rowSize, int position, int size)
	{
		diagonal.TryGetCell(block, block, out Span<double> cell, out int stride);
		ReadOnlySpan<double> values = matrix.Values.AsSpan(position);
		SmallBlas.MatrixTransposeMatrixMultiply(values, rowSize, size, values, size, cell, 0, 0, stride, 1);
	}
}

/// <summary>ceres::internal::ImplicitSchurComplement.</summary>
internal sealed class ImplicitSchurComplement
{
	private readonly int numEliminateBlocks;
	private readonly bool computeFtfInverse;
	private BlockSparseMatrix? matrix;
	private PartitionedMatrixView? view;
	private BlockRandomAccessDiagonalMatrix? blockDiagonalEtEInverse;
	private double[] d = [];
	private double[] b = [];
	private double[] tmpRows = [];
	private double[] tmpECols = [];
	private double[] tmpECols2 = [];

	/// <summary>
	/// Creates the operator for Jacobians whose first <paramref name="numEliminateBlocks"/>
	/// column blocks are eliminated; <paramref name="computeFtfInverse"/> also keeps
	/// (F'F + D_f^2)^-1 for the JACOBI preconditioner.
	/// </summary>
	public ImplicitSchurComplement(int numEliminateBlocks, bool computeFtfInverse)
	{
		this.numEliminateBlocks = numEliminateBlocks;
		this.computeFtfInverse = computeFtfInverse;
	}

	/// <summary>Size of the reduced system.</summary>
	public int NumRows => view!.NumColsF;

	/// <summary>The reduced right-hand side F'(b - E (E'E + D_e^2)^-1 E'b).</summary>
	public double[] Rhs { get; private set; } = [];

	/// <summary>(F'F + D_f^2)^-1, block diagonal, if requested.</summary>
	public BlockRandomAccessDiagonalMatrix? BlockDiagonalFtFInverse { get; private set; }

	/// <summary>
	/// ImplicitSchurComplement::Init: takes A, D (empty for none) and b, and computes the
	/// block-diagonal inverses and the right-hand side. The spans are copied, as Ceres keeps
	/// the pointers for the operator's later use.
	/// </summary>
	public void Init(BlockSparseMatrix a, ReadOnlySpan<double> d, ReadOnlySpan<double> b)
	{
		if (!ReferenceEquals(matrix, a) || view is null)
		{
			matrix = a;
			view = new PartitionedMatrixView(a, numEliminateBlocks);
			blockDiagonalEtEInverse = new BlockRandomAccessDiagonalMatrix(view.EBlocks());
			if (computeFtfInverse)
			{
				BlockDiagonalFtFInverse = new BlockRandomAccessDiagonalMatrix(
					SchurEliminator.ReducedBlocks(a.Structure, numEliminateBlocks));
			}

			Rhs = new double[view.NumColsF];
			tmpRows = new double[view.NumRows];
			tmpECols = new double[view.NumColsE];
			tmpECols2 = new double[view.NumColsE];
		}

		this.d = d.ToArray();
		this.b = b.ToArray();

		// Compute the block diagonals and invert them.
		view.UpdateBlockDiagonalEtE(blockDiagonalEtEInverse!);
		AddDiagonalAndInvert(this.d.AsSpan(), 0, blockDiagonalEtEInverse!);
		if (computeFtfInverse)
		{
			view.UpdateBlockDiagonalFtF(BlockDiagonalFtFInverse!);
			AddDiagonalAndInvert(this.d.AsSpan(), view.NumColsE, BlockDiagonalFtFInverse!);
		}

		// Compute the right hand side for the Schur complement system.
		UpdateRhs();
	}

	/// <summary>
	/// y = S x: y = F'F x - F'E (E'E)^-1 E'F x + D_f^2 x, computed as
	/// y = F'(F x - E (E'E)^-1 E'F x) + D_f^2 x.
	/// </summary>
	public void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		PartitionedMatrixView v = view!;

		// y1 = F x
		Array.Clear(tmpRows);
		v.RightMultiplyAndAccumulateF(x, tmpRows);

		// y2 = E' y1
		Array.Clear(tmpECols);
		v.LeftMultiplyAndAccumulateE(tmpRows, tmpECols);

		// y3 = -(E'E)^-1 y2
		Array.Clear(tmpECols2);
		blockDiagonalEtEInverse!.RightMultiplyAndAccumulate(tmpECols, tmpECols2);
		for (int i = 0; i < tmpECols2.Length; i++)
		{
			tmpECols2[i] = -tmpECols2[i];
		}

		// y1 = y1 + E y3
		v.RightMultiplyAndAccumulateE(tmpECols2, tmpRows);

		// y5 = D_f^2 x (or 0)
		int numCols = v.NumColsF;
		if (d.Length > 0)
		{
			for (int i = 0; i < numCols; i++)
			{
				double di = d[v.NumColsE + i];
				y[i] = di * di * x[i];
			}
		}
		else
		{
			y[..numCols].Clear();
		}

		// y = y5 + F' y1
		v.LeftMultiplyAndAccumulateF(tmpRows, y);
	}

	/// <summary>
	/// y += Z x with Z = (F'F + D_f^2)^-1 F'E (E'E + D_e^2)^-1 E'F, the operator whose power
	/// series approximates the inverse of the Schur complement (Ceres' SPSE preconditioner
	/// and initialization build on it). Needs the F'F inverse (computeFtfInverse).
	/// </summary>
	public void InversePowerSeriesOperatorRightMultiplyAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		Util.Check.That(computeFtfInverse, "The F'F inverse was not requested.");
		PartitionedMatrixView v = view!;

		// y1 = F x
		Array.Clear(tmpRows);
		v.RightMultiplyAndAccumulateF(x, tmpRows);

		// y2 = E' y1
		Array.Clear(tmpECols);
		v.LeftMultiplyAndAccumulateE(tmpRows, tmpECols);

		// y3 = (E'E)^-1 y2
		Array.Clear(tmpECols2);
		blockDiagonalEtEInverse!.RightMultiplyAndAccumulate(tmpECols, tmpECols2);

		// y1 = E y3
		Array.Clear(tmpRows);
		v.RightMultiplyAndAccumulateE(tmpECols2, tmpRows);

		// y4 = F' y1
		var tmpFCols = new double[v.NumColsF];
		v.LeftMultiplyAndAccumulateF(tmpRows, tmpFCols);

		// y += (F'F)^-1 y4
		BlockDiagonalFtFInverse!.RightMultiplyAndAccumulate(tmpFCols, y);
	}

	/// <summary>
	/// Given a solution <paramref name="x"/> of the reduced system, the full solution y:
	/// y_e = (E'E)^-1 E'(b - F x), y_f = x.
	/// </summary>
	public void BackSubstitute(ReadOnlySpan<double> x, Span<double> y)
	{
		PartitionedMatrixView v = view!;
		int numColsE = v.NumColsE;
		int numColsF = v.NumColsF;

		// y1 = F x
		Array.Clear(tmpRows);
		v.RightMultiplyAndAccumulateF(x, tmpRows);

		// y2 = b - y1
		for (int i = 0; i < tmpRows.Length; i++)
		{
			tmpRows[i] = b[i] - tmpRows[i];
		}

		// y3 = E' y2
		Array.Clear(tmpECols);
		v.LeftMultiplyAndAccumulateE(tmpRows, tmpECols);

		// y = (E'E)^-1 y3
		y[..(numColsE + numColsF)].Clear();
		blockDiagonalEtEInverse!.RightMultiplyAndAccumulate(tmpECols, y);

		// The full solution vector y has two blocks. The first block of variables
		// corresponds to the eliminated variables, which we just computed via back
		// substitution. The second block of variables corresponds to the Schur complement
		// system, so we just copy those values from the solution to the Schur complement.
		x[..numColsF].CopyTo(y[numColsE..]);
	}

	// m = (m + diag(D)^2)^-1 for every diagonal block, D read from dOffset on.
	private static void AddDiagonalAndInvert(ReadOnlySpan<double> d, int dOffset, BlockRandomAccessDiagonalMatrix blockDiagonal)
	{
		if (!d.IsEmpty)
		{
			for (int i = 0, position = 0; position < blockDiagonal.NumRows; i++)
			{
				blockDiagonal.TryGetCell(i, i, out Span<double> cell, out int size);
				for (int k = 0; k < size; k++)
				{
					double dk = d[dOffset + position + k];
					cell[(k * size) + k] += dk * dk;
				}

				position += size;
			}
		}

		blockDiagonal.Invert();
	}

	// rhs = F'b - F'E (E'E)^-1 E'b
	private void UpdateRhs()
	{
		PartitionedMatrixView v = view!;

		// y1 = E'b
		Array.Clear(tmpECols);
		v.LeftMultiplyAndAccumulateE(b, tmpECols);

		// y2 = (E'E)^-1 y1
		Array.Clear(tmpECols2);
		blockDiagonalEtEInverse!.RightMultiplyAndAccumulate(tmpECols, tmpECols2);

		// y3 = E y2
		Array.Clear(tmpRows);
		v.RightMultiplyAndAccumulateE(tmpECols2, tmpRows);

		// y3 = b - y3
		for (int i = 0; i < tmpRows.Length; i++)
		{
			tmpRows[i] = b[i] - tmpRows[i];
		}

		// rhs = F' y3
		Array.Clear(Rhs);
		v.LeftMultiplyAndAccumulateF(tmpRows, Rhs);
	}
}
