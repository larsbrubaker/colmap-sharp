// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SparseMatrixCsc: a compressed sparse column matrix, the replacement for
// Eigen::SparseMatrix<double> (column-major, the Eigen default COLMAP uses everywhere:
// optim/sparse_cholesky, optim/least_absolute_deviations, rotation averaging, and later the
// bundle-adjustment normal equations). The sparse Cholesky (SimplicialCholesky.cs, with
// AmdOrdering.cs) factors these.
//
// Storage is the textbook CSC layout (Davis, "Direct Methods for Sparse Linear Systems",
// SIAM 2006, ch. 2, used as a description only; no CSparse code is ported, it is LGPL):
// ColPtr has Cols + 1 offsets, and column j's entries are RowIndices/Values[ColPtr[j] ..
// ColPtr[j+1]). Every matrix built here keeps row indices strictly increasing within a
// column, with no duplicates, which Coeff's binary search and the Cholesky's pattern
// check rely on. Explicitly stored zeros stay stored, as they do in Eigen when a caller
// inserts a 0 (least_absolute_deviations_test.cc does), so they are part of the pattern.
//
// Unlike Eigen there is no random-access insert: CoeffRef only reaches stored entries,
// because every COLMAP call site updates an existing pattern (the diagonal of A^T A).
// AddToDiagonal covers the one case that may need new entries.

namespace ColmapSharp.LinearAlgebra;

/// <summary>One (row, col, value) entry used to build a sparse matrix, Eigen::Triplet&lt;double&gt;.</summary>
public readonly record struct SparseTriplet(int Row, int Col, double Value);

/// <summary>Which triangle of a symmetric matrix is stored or read.</summary>
public enum SymmetricPart
{
	/// <summary>Entries with row &gt;= col (Eigen::Lower, the default of Eigen's sparse Cholesky).</summary>
	Lower,

	/// <summary>Entries with row &lt;= col (Eigen::Upper).</summary>
	Upper,
}

/// <summary>
/// Compressed sparse column matrix of doubles. Replacement for Eigen::SparseMatrix&lt;double&gt;.
/// </summary>
public sealed class SparseMatrixCsc
{
	private readonly int[] _colPtr;
	private readonly int[] _rowIdx;
	private readonly double[] _values;

	private SparseMatrixCsc(int rows, int cols, int[] colPtr, int[] rowIdx, double[] values)
	{
		Rows = rows;
		Cols = cols;
		_colPtr = colPtr;
		_rowIdx = rowIdx;
		_values = values;
	}

	/// <summary>Number of rows.</summary>
	public int Rows { get; }

	/// <summary>Number of columns.</summary>
	public int Cols { get; }

	/// <summary>Number of stored entries (Eigen's nonZeros(), explicit zeros included).</summary>
	public int NonZeros => _colPtr[Cols];

	/// <summary>Column offsets, length Cols + 1.</summary>
	public ReadOnlySpan<int> ColPtr => _colPtr;

	/// <summary>Row index of each stored entry, increasing within each column.</summary>
	public ReadOnlySpan<int> RowIndices => _rowIdx;

	/// <summary>Value of each stored entry. Writable: changing values keeps the pattern.</summary>
	public Span<double> Values => _values;

	/// <summary>An all-zero matrix with no stored entries.</summary>
	public static SparseMatrixCsc Zero(int rows, int cols)
	{
		CheckShape(rows, cols);
		return new SparseMatrixCsc(rows, cols, new int[cols + 1], [], []);
	}

	/// <summary>The n x n identity.</summary>
	public static SparseMatrixCsc Identity(int n)
	{
		var triplets = new SparseTriplet[n];
		for (int i = 0; i < n; i++)
		{
			triplets[i] = new SparseTriplet(i, i, 1.0);
		}

		return FromTriplets(n, n, triplets);
	}

	/// <summary>
	/// Builds a matrix from triplets, summing duplicates (Eigen's setFromTriplets). Duplicates
	/// are summed in the order they appear in <paramref name="triplets"/>.
	/// </summary>
	public static SparseMatrixCsc FromTriplets(int rows, int cols, IEnumerable<SparseTriplet> triplets)
	{
		CheckShape(rows, cols);
		SparseTriplet[] list = triplets as SparseTriplet[] ?? triplets.ToArray();

		// Counting sort by column, then by row within each column. Both passes are stable,
		// so duplicates keep their input order and are summed in it.
		var count = new int[cols + 1];
		foreach (SparseTriplet t in list)
		{
			if ((uint)t.Row >= (uint)rows || (uint)t.Col >= (uint)cols)
			{
				throw new ArgumentOutOfRangeException(
					nameof(triplets), $"Triplet ({t.Row}, {t.Col}) is outside a {rows}x{cols} matrix.");
			}

			count[t.Col + 1]++;
		}

		for (int j = 0; j < cols; j++)
		{
			count[j + 1] += count[j];
		}

		var byCol = new SparseTriplet[list.Length];
		var next = (int[])count.Clone();
		foreach (SparseTriplet t in list)
		{
			byCol[next[t.Col]++] = t;
		}

		var colPtr = new int[cols + 1];
		var rowIdx = new List<int>(list.Length);
		var values = new List<double>(list.Length);
		var lastSlot = new int[rows];
		Array.Fill(lastSlot, -1);
		for (int j = 0; j < cols; j++)
		{
			int start = rowIdx.Count;
			for (int k = count[j]; k < count[j + 1]; k++)
			{
				SparseTriplet t = byCol[k];
				int slot = lastSlot[t.Row];
				if (slot >= start)
				{
					values[slot] += t.Value;
				}
				else
				{
					lastSlot[t.Row] = rowIdx.Count;
					rowIdx.Add(t.Row);
					values.Add(t.Value);
				}
			}

			SortColumn(rowIdx, values, start, rowIdx.Count - start);
			colPtr[j + 1] = rowIdx.Count;
		}

		return new SparseMatrixCsc(rows, cols, colPtr, [.. rowIdx], [.. values]);
	}

	/// <summary>
	/// Builds a matrix from raw CSC arrays (copied). Row indices must be in range and strictly
	/// increasing within each column.
	/// </summary>
	public static SparseMatrixCsc FromCsc(
		int rows, int cols, ReadOnlySpan<int> colPtr, ReadOnlySpan<int> rowIndices, ReadOnlySpan<double> values)
	{
		CheckShape(rows, cols);
		if (colPtr.Length != cols + 1 || colPtr[0] != 0 || rowIndices.Length != colPtr[cols]
			|| values.Length != colPtr[cols])
		{
			throw new ArgumentException("CSC arrays have inconsistent lengths.");
		}

		for (int j = 0; j < cols; j++)
		{
			if (colPtr[j + 1] < colPtr[j])
			{
				throw new ArgumentException("CSC column offsets must be non-decreasing.");
			}

			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				int i = rowIndices[p];
				if ((uint)i >= (uint)rows || (p > colPtr[j] && rowIndices[p - 1] >= i))
				{
					throw new ArgumentException("CSC row indices must be in range and strictly increasing per column.");
				}
			}
		}

		return new SparseMatrixCsc(rows, cols, colPtr.ToArray(), rowIndices.ToArray(), values.ToArray());
	}

	/// <summary>Converts a dense matrix, storing only its nonzero entries.</summary>
	public static SparseMatrixCsc FromDense(MatrixXd dense)
	{
		var triplets = new List<SparseTriplet>();
		for (int j = 0; j < dense.Cols; j++)
		{
			for (int i = 0; i < dense.Rows; i++)
			{
				if (dense[i, j] != 0.0)
				{
					triplets.Add(new SparseTriplet(i, j, dense[i, j]));
				}
			}
		}

		return FromTriplets(dense.Rows, dense.Cols, triplets);
	}

	/// <summary>Converts to a dense matrix.</summary>
	public MatrixXd ToDense()
	{
		var dense = new MatrixXd(Rows, Cols);
		for (int j = 0; j < Cols; j++)
		{
			for (int p = _colPtr[j]; p < _colPtr[j + 1]; p++)
			{
				dense[_rowIdx[p], j] = _values[p];
			}
		}

		return dense;
	}

	/// <summary>A deep copy.</summary>
	public SparseMatrixCsc Clone() =>
		new(Rows, Cols, (int[])_colPtr.Clone(), (int[])_rowIdx.Clone(), (double[])_values.Clone());

	/// <summary>Entry (i, j), zero if it is not stored (Eigen's coeff()).</summary>
	public double this[int i, int j]
	{
		get
		{
			int p = Find(i, j);
			return p >= 0 ? _values[p] : 0.0;
		}
	}

	/// <summary>
	/// Reference to the stored entry (i, j). Eigen's coeffRef inserts a missing entry; this
	/// type has a fixed pattern, so a structural zero throws instead.
	/// </summary>
	public ref double CoeffRef(int i, int j)
	{
		int p = Find(i, j);
		if (p < 0)
		{
			throw new ArgumentException($"Entry ({i}, {j}) is not stored in the sparse pattern.");
		}

		return ref _values[p];
	}

	/// <summary>
	/// Returns this matrix plus <paramref name="value"/> times the identity, adding diagonal
	/// entries to the pattern where they are missing.
	/// </summary>
	public SparseMatrixCsc AddToDiagonal(double value)
	{
		int n = Math.Min(Rows, Cols);
		var triplets = new List<SparseTriplet>(NonZeros + n);
		ForEach((i, j, v) => triplets.Add(new SparseTriplet(i, j, v)));
		for (int i = 0; i < n; i++)
		{
			triplets.Add(new SparseTriplet(i, i, value));
		}

		return FromTriplets(Rows, Cols, triplets);
	}

	/// <summary>Calls <paramref name="action"/>(row, col, value) for every stored entry, column by column.</summary>
	public void ForEach(Action<int, int, double> action)
	{
		for (int j = 0; j < Cols; j++)
		{
			for (int p = _colPtr[j]; p < _colPtr[j + 1]; p++)
			{
				action(_rowIdx[p], j, _values[p]);
			}
		}
	}

	/// <summary>The transpose, as a new CSC matrix.</summary>
	public SparseMatrixCsc Transpose()
	{
		var colPtr = new int[Rows + 1];
		for (int p = 0; p < NonZeros; p++)
		{
			colPtr[_rowIdx[p] + 1]++;
		}

		for (int i = 0; i < Rows; i++)
		{
			colPtr[i + 1] += colPtr[i];
		}

		var next = (int[])colPtr.Clone();
		var rowIdx = new int[NonZeros];
		var values = new double[NonZeros];

		// Visiting columns in increasing order writes each transposed column's rows in
		// increasing order, so no sort is needed.
		for (int j = 0; j < Cols; j++)
		{
			for (int p = _colPtr[j]; p < _colPtr[j + 1]; p++)
			{
				int q = next[_rowIdx[p]]++;
				rowIdx[q] = j;
				values[q] = _values[p];
			}
		}

		return new SparseMatrixCsc(Cols, Rows, colPtr, rowIdx, values);
	}

	/// <summary>
	/// The stored triangle of this square matrix that <paramref name="part"/> names (Eigen's
	/// triangularView&lt;Lower/Upper&gt;() materialized), diagonal included.
	/// </summary>
	public SparseMatrixCsc TriangularPart(SymmetricPart part)
	{
		var triplets = new List<SparseTriplet>(NonZeros);
		ForEach((i, j, v) =>
		{
			if (part == SymmetricPart.Lower ? i >= j : i <= j)
			{
				triplets.Add(new SparseTriplet(i, j, v));
			}
		});
		return FromTriplets(Rows, Cols, triplets);
	}

	/// <summary>y = A x.</summary>
	public VectorXd Multiply(VectorXd x)
	{
		if (x.Length != Cols)
		{
			throw new ArgumentException($"Cannot multiply a {Rows}x{Cols} matrix by a vector of length {x.Length}.");
		}

		var y = new VectorXd(Rows);
		Span<double> ys = y.AsSpan();
		ReadOnlySpan<double> xs = x.AsSpan();
		for (int j = 0; j < Cols; j++)
		{
			double xj = xs[j];
			for (int p = _colPtr[j]; p < _colPtr[j + 1]; p++)
			{
				ys[_rowIdx[p]] += _values[p] * xj;
			}
		}

		return y;
	}

	/// <summary>y = A^T x, without forming the transpose.</summary>
	public VectorXd TransposeMultiply(VectorXd x)
	{
		if (x.Length != Rows)
		{
			throw new ArgumentException($"Cannot multiply the transpose of a {Rows}x{Cols} matrix by a vector of length {x.Length}.");
		}

		var y = new VectorXd(Cols);
		Span<double> ys = y.AsSpan();
		ReadOnlySpan<double> xs = x.AsSpan();
		for (int j = 0; j < Cols; j++)
		{
			double sum = 0.0;
			for (int p = _colPtr[j]; p < _colPtr[j + 1]; p++)
			{
				sum += _values[p] * xs[_rowIdx[p]];
			}

			ys[j] = sum;
		}

		return y;
	}

	/// <summary>A x.</summary>
	public static VectorXd operator *(SparseMatrixCsc a, VectorXd x) => a.Multiply(x);

	/// <summary>
	/// Sparse product A B (Gustavson's column-by-column algorithm: column j of the product
	/// is the combination of A's columns weighted by column j of B). Every product term
	/// stays in the pattern, even one that cancels to zero.
	/// </summary>
	public static SparseMatrixCsc operator *(SparseMatrixCsc a, SparseMatrixCsc b)
	{
		if (a.Cols != b.Rows)
		{
			throw new ArgumentException($"Cannot multiply a {a.Rows}x{a.Cols} by a {b.Rows}x{b.Cols} sparse matrix.");
		}

		var colPtr = new int[b.Cols + 1];
		var rowIdx = new List<int>(a.NonZeros + b.NonZeros);
		var values = new List<double>(a.NonZeros + b.NonZeros);
		var slotOf = new int[a.Rows];
		Array.Fill(slotOf, -1);
		for (int j = 0; j < b.Cols; j++)
		{
			int start = rowIdx.Count;
			for (int pb = b._colPtr[j]; pb < b._colPtr[j + 1]; pb++)
			{
				int k = b._rowIdx[pb];
				double bkj = b._values[pb];
				for (int pa = a._colPtr[k]; pa < a._colPtr[k + 1]; pa++)
				{
					int i = a._rowIdx[pa];
					int slot = slotOf[i];
					if (slot >= start)
					{
						values[slot] += a._values[pa] * bkj;
					}
					else
					{
						slotOf[i] = rowIdx.Count;
						rowIdx.Add(i);
						values.Add(a._values[pa] * bkj);
					}
				}
			}

			SortColumn(rowIdx, values, start, rowIdx.Count - start);
			colPtr[j + 1] = rowIdx.Count;
		}

		return new SparseMatrixCsc(a.Rows, b.Cols, colPtr, [.. rowIdx], [.. values]);
	}

	/// <summary>A^T A with both triangles stored (the normal-equations matrix).</summary>
	public SparseMatrixCsc TransposeTimesSelf() => Transpose() * this;

	/// <summary>A^T A with only the <paramref name="part"/> triangle stored.</summary>
	public SparseMatrixCsc TransposeTimesSelf(SymmetricPart part) => TransposeTimesSelf().TriangularPart(part);

	/// <summary>
	/// The stored entries inside the <paramref name="rows"/> x <paramref name="cols"/> block at
	/// (<paramref name="row"/>, <paramref name="col"/>), as a new matrix (Eigen's
	/// SparseMatrix::block() assigned to a SparseMatrix).
	/// </summary>
	public SparseMatrixCsc Block(int row, int col, int rows, int cols)
	{
		if (row < 0 || col < 0 || rows < 0 || cols < 0 || row + rows > Rows || col + cols > Cols)
		{
			throw new ArgumentOutOfRangeException(
				nameof(row), $"Block ({row}, {col}, {rows}, {cols}) is outside a {Rows}x{Cols} matrix.");
		}

		var colPtr = new int[cols + 1];
		var rowIdx = new List<int>();
		var values = new List<double>();
		for (int j = 0; j < cols; j++)
		{
			for (int p = _colPtr[col + j]; p < _colPtr[col + j + 1]; p++)
			{
				int i = _rowIdx[p] - row;
				if (i >= 0 && i < rows)
				{
					rowIdx.Add(i);
					values.Add(_values[p]);
				}
			}

			colPtr[j + 1] = rowIdx.Count;
		}

		return new SparseMatrixCsc(rows, cols, colPtr, [.. rowIdx], [.. values]);
	}

	/// <summary>A - B over the union of both patterns (an entry only in B stores -b).</summary>
	public static SparseMatrixCsc operator -(SparseMatrixCsc a, SparseMatrixCsc b)
	{
		if (a.Rows != b.Rows || a.Cols != b.Cols)
		{
			throw new ArgumentException($"Cannot subtract a {b.Rows}x{b.Cols} from a {a.Rows}x{a.Cols} sparse matrix.");
		}

		var colPtr = new int[a.Cols + 1];
		var rowIdx = new List<int>(a.NonZeros + b.NonZeros);
		var values = new List<double>(a.NonZeros + b.NonZeros);
		for (int j = 0; j < a.Cols; j++)
		{
			// Merge the two sorted row lists of column j.
			int pa = a._colPtr[j];
			int pb = b._colPtr[j];
			int ea = a._colPtr[j + 1];
			int eb = b._colPtr[j + 1];
			while (pa < ea || pb < eb)
			{
				int ia = pa < ea ? a._rowIdx[pa] : int.MaxValue;
				int ib = pb < eb ? b._rowIdx[pb] : int.MaxValue;
				if (ia == ib)
				{
					rowIdx.Add(ia);
					values.Add(a._values[pa++] - b._values[pb++]);
				}
				else if (ia < ib)
				{
					rowIdx.Add(ia);
					values.Add(a._values[pa++]);
				}
				else
				{
					rowIdx.Add(ib);
					values.Add(-b._values[pb++]);
				}
			}

			colPtr[j + 1] = rowIdx.Count;
		}

		return new SparseMatrixCsc(a.Rows, a.Cols, colPtr, [.. rowIdx], [.. values]);
	}

	/// <summary>Position of entry (i, j) in the value array, or -1 when it is not stored.</summary>
	private int Find(int i, int j)
	{
		if ((uint)i >= (uint)Rows || (uint)j >= (uint)Cols)
		{
			throw new ArgumentOutOfRangeException(null, $"Entry ({i}, {j}) is outside a {Rows}x{Cols} matrix.");
		}

		int lo = _colPtr[j];
		int hi = _colPtr[j + 1] - 1;
		while (lo <= hi)
		{
			int mid = (lo + hi) >>> 1;
			int r = _rowIdx[mid];
			if (r == i)
			{
				return mid;
			}

			if (r < i)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid - 1;
			}
		}

		return -1;
	}

	/// <summary>Sorts one column's (row, value) pairs by row. Rows are unique, so order is total.</summary>
	private static void SortColumn(List<int> rowIdx, List<double> values, int start, int length)
	{
		if (length < 2)
		{
			return;
		}

		int[] keys = new int[length];
		double[] items = new double[length];
		rowIdx.CopyTo(start, keys, 0, length);
		values.CopyTo(start, items, 0, length);
		Array.Sort(keys, items);
		for (int k = 0; k < length; k++)
		{
			rowIdx[start + k] = keys[k];
			values[start + k] = items[k];
		}
	}

	private static void CheckShape(int rows, int cols)
	{
		if (rows < 0 || cols < 0)
		{
			throw new ArgumentOutOfRangeException(nameof(rows), $"Invalid sparse matrix shape {rows}x{cols}.");
		}
	}
}
