// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSystemMatrix: FEMTree's SystemMatrixType (thirdparty/PoissonRecon/FEMTree.h:
// SparseMatrix< Real , matrix_index_type , WindowSize< OverlapSizes >::Size >, SparseMatrix.h),
// the per-slice system matrix the multigrid solver relaxes: each row holds at most one entry per
// neighbor in the 3x3x3 overlap window, so the rows live in one flat array with a fixed stride
// and a row size, and resizing allocates only when the row count outgrows the storage.
// PoissonSystem fills it; the solver (a later slice) multiplies and relaxes with it.
//
// Translation notes: MatrixEntry< float , int > is split into a column array (N, the node
// index minus the slice's first index) and a value array; entry 0 of a set row is the
// diagonal, as the C++'s row assembly guarantees.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A sparse matrix with a bounded row length in flat storage. Port of PoissonRecon's
/// <c>SparseMatrix&lt; Real , matrix_index_type , MaxRowSize &gt;</c>.
/// </summary>
public sealed class PoissonSystemMatrix
{
	private int[] columns = [];
	private float[] values = [];
	private int[] rowSizes = [];

	/// <summary>An empty matrix whose rows hold at most <paramref name="stride"/> entries.</summary>
	public PoissonSystemMatrix(int stride)
	{
		Stride = stride;
	}

	/// <summary>The maximum entries per row.</summary>
	public int Stride { get; }

	/// <summary>The number of rows.</summary>
	public int Rows { get; private set; }

	/// <summary>Sets the row count and empties every row. Port of <c>resize( rows )</c>.</summary>
	public void Resize(int rows)
	{
		if (rows > rowSizes.Length)
		{
			rowSizes = new int[rows];
			columns = new int[rows * Stride];
			values = new float[rows * Stride];
		}

		Rows = rows;
		Array.Clear(rowSizes, 0, rows);
	}

	/// <summary>The number of entries in a row. Port of <c>rowSize( row )</c>.</summary>
	public int RowSize(int row) => rowSizes[row];

	/// <summary>Sets a row's entry count. Port of <c>setRowSize( row , count )</c>.</summary>
	public void SetRowSize(int row, int count)
	{
		if (count > Stride)
		{
			throw new InvalidOperationException($"Row size {count} exceeds the maximum {Stride}.");
		}

		rowSizes[row] = count;
	}

	/// <summary>The column (index relative to the slice start) of entry j of a row.</summary>
	public ref int Column(int row, int j) => ref columns[row * Stride + j];

	/// <summary>The value of entry j of a row.</summary>
	public ref float Value(int row, int j) => ref values[row * Stride + j];
}
