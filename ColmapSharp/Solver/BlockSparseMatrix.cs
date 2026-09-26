// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/block_structure.h,
// internal/ceres/block_sparse_matrix.cc and internal/ceres/block_jacobian_writer.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The block-sparse Jacobian: one row block per residual block, one column block per
// (variable) parameter block, and a dense row-major cell wherever a residual block depends
// on a parameter block. The evaluator writes each cell in place (no copy), which is why the
// layout is computed here from the program. Ceres puts the cells of the first
// num_eliminate_blocks column blocks (the Schur "E" blocks) before all the others in the
// value array; the Schur solvers (next phase) rely on that, and with no eliminated blocks
// the layout is simply residual-block order. SparseNormalCholeskySolver (LinearSolvers.cs)
// forms J'J from this structure.

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::Block: a run of rows or columns.</summary>
internal readonly record struct Block(int Size, int Position);

/// <summary>ceres::internal::Cell: a stored block in a row, by column block and value offset.</summary>
internal readonly record struct Cell(int BlockId, int Position);

/// <summary>ceres::internal::CompressedRow: a row block and its cells, sorted by column block.</summary>
internal sealed class CompressedRow(Block block, Cell[] cells)
{
	/// <summary>The rows.</summary>
	public Block Block { get; } = block;

	/// <summary>The stored cells, by increasing column block.</summary>
	public Cell[] Cells { get; } = cells;
}

/// <summary>ceres::internal::CompressedRowBlockStructure.</summary>
internal sealed class CompressedRowBlockStructure(Block[] cols, CompressedRow[] rows)
{
	/// <summary>The column blocks.</summary>
	public Block[] Cols { get; } = cols;

	/// <summary>The row blocks.</summary>
	public CompressedRow[] Rows { get; } = rows;
}

/// <summary>ceres::internal::BlockSparseMatrix.</summary>
internal sealed class BlockSparseMatrix : SparseMatrix
{
	private readonly int numRows;
	private readonly int numCols;

	/// <summary>Creates a zero matrix with the given structure.</summary>
	public BlockSparseMatrix(CompressedRowBlockStructure structure)
	{
		Structure = structure;
		int numNonzeros = 0;
		foreach (CompressedRow row in structure.Rows)
		{
			numRows += row.Block.Size;
			foreach (Cell cell in row.Cells)
			{
				numNonzeros += row.Block.Size * structure.Cols[cell.BlockId].Size;
			}
		}

		foreach (Block col in structure.Cols)
		{
			numCols += col.Size;
		}

		Values = new double[numNonzeros];
	}

	/// <summary>The block structure.</summary>
	public CompressedRowBlockStructure Structure { get; }

	/// <summary>The cell values; each cell is row-major, row block size x column block size.</summary>
	public double[] Values { get; }

	/// <inheritdoc/>
	public override int NumRows => numRows;

	/// <inheritdoc/>
	public override int NumCols => numCols;

	/// <inheritdoc/>
	public override void SetZero() => Array.Clear(Values);

	/// <inheritdoc/>
	public override void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		Block[] cols = Structure.Cols;
		foreach (CompressedRow row in Structure.Rows)
		{
			int rowSize = row.Block.Size;
			int rowPos = row.Block.Position;
			foreach (Cell cell in row.Cells)
			{
				Block col = cols[cell.BlockId];
				for (int r = 0; r < rowSize; r++)
				{
					double sum = 0.0;
					int start = cell.Position + r * col.Size;
					for (int c = 0; c < col.Size; c++)
					{
						sum += Values[start + c] * x[col.Position + c];
					}

					y[rowPos + r] += sum;
				}
			}
		}
	}

	/// <inheritdoc/>
	public override void LeftMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		Block[] cols = Structure.Cols;
		foreach (CompressedRow row in Structure.Rows)
		{
			int rowSize = row.Block.Size;
			int rowPos = row.Block.Position;
			foreach (Cell cell in row.Cells)
			{
				Block col = cols[cell.BlockId];
				for (int c = 0; c < col.Size; c++)
				{
					double sum = 0.0;
					for (int r = 0; r < rowSize; r++)
					{
						sum += Values[cell.Position + r * col.Size + c] * x[rowPos + r];
					}

					y[col.Position + c] += sum;
				}
			}
		}
	}

	/// <inheritdoc/>
	public override void SquaredColumnNorm(Span<double> x)
	{
		x[..numCols].Clear();
		Block[] cols = Structure.Cols;
		foreach (CompressedRow row in Structure.Rows)
		{
			int rowSize = row.Block.Size;
			foreach (Cell cell in row.Cells)
			{
				Block col = cols[cell.BlockId];
				for (int c = 0; c < col.Size; c++)
				{
					double sum = 0.0;
					for (int r = 0; r < rowSize; r++)
					{
						double v = Values[cell.Position + r * col.Size + c];
						sum += v * v;
					}

					x[col.Position + c] += sum;
				}
			}
		}
	}

	/// <inheritdoc/>
	public override void ScaleColumns(ReadOnlySpan<double> scale)
	{
		Block[] cols = Structure.Cols;
		foreach (CompressedRow row in Structure.Rows)
		{
			int rowSize = row.Block.Size;
			foreach (Cell cell in row.Cells)
			{
				Block col = cols[cell.BlockId];
				for (int r = 0; r < rowSize; r++)
				{
					int start = cell.Position + r * col.Size;
					for (int c = 0; c < col.Size; c++)
					{
						Values[start + c] *= scale[col.Position + c];
					}
				}
			}
		}
	}
}

/// <summary>
/// ceres::internal::BlockJacobianWriter: where each residual block's Jacobian cells live in a
/// <see cref="BlockSparseMatrix"/>'s values, and the matrix's block structure.
/// </summary>
internal sealed class BlockJacobianLayout
{
	private readonly Program program;

	// layout[i][k]: value offset of the k-th variable parameter block (in residual block
	// order) of residual block i.
	private readonly int[][] layout;

	/// <summary>Computes the layout; the first <paramref name="numEliminateBlocks"/> column blocks' cells go first.</summary>
	public BlockJacobianLayout(Program program, int numEliminateBlocks)
	{
		this.program = program;
		List<ResidualBlock> residualBlocks = program.ResidualBlocks;
		int fBlockPos = 0;
		foreach (ResidualBlock residualBlock in residualBlocks)
		{
			foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
			{
				if (!parameterBlock.IsConstant && parameterBlock.Index < numEliminateBlocks)
				{
					fBlockPos += residualBlock.NumResiduals * parameterBlock.TangentSize;
				}
			}
		}

		layout = new int[residualBlocks.Count][];
		int eBlockPos = 0;
		var active = new List<(int K, int J)>();
		for (int i = 0; i < residualBlocks.Count; i++)
		{
			ResidualBlock residualBlock = residualBlocks[i];
			ParameterBlock[] blocks = residualBlock.ParameterBlocks;
			active.Clear();
			for (int j = 0; j < blocks.Length; j++)
			{
				if (!blocks[j].IsConstant)
				{
					active.Add((active.Count, j));
				}
			}

			// Cells go in column-block order, so the value array walks each row block's
			// columns left to right. Parameter block indices within one residual block are
			// distinct, so this order has no ties.
			active.Sort((a, b) => blocks[a.J].Index.CompareTo(blocks[b.J].Index));
			var positions = new int[active.Count];
			foreach ((int k, int j) in active)
			{
				int size = residualBlock.NumResiduals * blocks[j].TangentSize;
				if (blocks[j].Index < numEliminateBlocks)
				{
					positions[k] = eBlockPos;
					eBlockPos += size;
				}
				else
				{
					positions[k] = fBlockPos;
					fBlockPos += size;
				}
			}

			layout[i] = positions;
		}
	}

	/// <summary>Value offsets of residual block <paramref name="i"/>'s variable parameter blocks, in its order.</summary>
	public int[] this[int i] => layout[i];

	/// <summary>Builds the (zero) block-sparse Jacobian.</summary>
	public BlockSparseMatrix CreateJacobian()
	{
		List<ParameterBlock> parameterBlocks = program.ParameterBlocks;
		var cols = new Block[parameterBlocks.Count];
		for (int i = 0, cursor = 0; i < parameterBlocks.Count; i++)
		{
			cols[i] = new Block(parameterBlocks[i].TangentSize, cursor);
			cursor += cols[i].Size;
		}

		List<ResidualBlock> residualBlocks = program.ResidualBlocks;
		var rows = new CompressedRow[residualBlocks.Count];
		int rowBlockPosition = 0;
		for (int i = 0; i < residualBlocks.Count; i++)
		{
			ResidualBlock residualBlock = residualBlocks[i];
			var cells = new List<Cell>(residualBlock.NumParameterBlocks);
			int k = 0;
			foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
			{
				if (!parameterBlock.IsConstant)
				{
					cells.Add(new Cell(parameterBlock.Index, layout[i][k]));
					k++;
				}
			}

			cells.Sort((a, b) => a.BlockId.CompareTo(b.BlockId));
			rows[i] = new CompressedRow(new Block(residualBlock.NumResiduals, rowBlockPosition), [.. cells]);
			rowBlockPosition += residualBlock.NumResiduals;
		}

		return new BlockSparseMatrix(new CompressedRowBlockStructure(cols, rows));
	}
}
