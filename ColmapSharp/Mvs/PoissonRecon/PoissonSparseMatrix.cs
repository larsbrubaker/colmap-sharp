// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSparseMatrix: SparseMatrix< float , int > with unbounded rows
// (thirdparty/PoissonRecon/SparseMatrix.h/.inl, SparseMatrixInterface.inl), the general sparse
// matrix of the base-depth multigrid (_solveRegularMG): the restriction matrices
// (PoissonMultigrid.DownSampleMatrix), their transposes (the prolongations), the Galerkin
// products R * M * P (whose row order follows libc++'s unordered_map, LibcxxUnorderedMap),
// matrix-vector products and the reciprocal diagonal. Each row keeps its
// entries in the order they were set, since that order reaches every float sum (products and
// Gauss-Seidel residuals). PoissonSystemMatrix is the fixed-stride counterpart for the
// per-slice system rows.
//
// Translation notes: vectors are float arrays with an offset, standing in for the C++'s
// pointer arithmetic (solution + nodesBegin( depth )). The C++ multiplies rows with
// ThreadPool::ParallelFor; each row writes only its own output, so the port runs them in order.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A row-major sparse matrix with per-row entry lists. Port of PoissonRecon's
/// <c>SparseMatrix&lt; Real , matrix_index_type , 0 &gt;</c>.
/// </summary>
public sealed class PoissonSparseMatrix
{
	private int[][] columns = [];
	private float[][] values = [];
	private int[] rowSizes = [];

	/// <summary>The number of rows (rowNum).</summary>
	public int Rows => rowSizes.Length;

	/// <summary>Sets the row count, every row empty. Port of <c>resize( rows )</c>.</summary>
	public void Resize(int rows)
	{
		columns = new int[rows][];
		values = new float[rows][];
		rowSizes = new int[rows];
		for (int i = 0; i < rows; i++)
		{
			columns[i] = [];
			values[i] = [];
		}
	}

	/// <summary>The number of entries in a row. Port of <c>rowSize( row )</c>.</summary>
	public int RowSize(int row) => rowSizes[row];

	/// <summary>Allocates <paramref name="count"/> entries for a row. Port of <c>setRowSize( row , count )</c>.</summary>
	public void SetRowSize(int row, int count)
	{
		columns[row] = new int[count];
		values[row] = new float[count];
		rowSizes[row] = count;
	}

	/// <summary>Sets the number of used entries of an allocated row (the C++'s rowSizes[row] = n).</summary>
	public void SetUsedSize(int row, int count) => rowSizes[row] = count;

	/// <summary>The column of entry j of a row (MatrixEntry::N).</summary>
	public ref int Column(int row, int j) => ref columns[row][j];

	/// <summary>The value of entry j of a row (MatrixEntry::Value).</summary>
	public ref float Value(int row, int j) => ref values[row][j];

	/// <summary>The rows of a fixed-stride system matrix, entry order kept.</summary>
	public static PoissonSparseMatrix From(PoissonSystemMatrix m)
	{
		var result = new PoissonSparseMatrix();
		result.Resize(m.Rows);
		for (int i = 0; i < m.Rows; i++)
		{
			result.SetRowSize(i, m.RowSize(i));
			for (int j = 0; j < m.RowSize(i); j++)
			{
				result.columns[i][j] = m.Column(i, j);
				result.values[i][j] = m.Value(i, j);
			}
		}

		return result;
	}

	/// <summary>
	/// The transpose with <paramref name="outRows"/> rows: row c lists (r, value) for every
	/// entry (r, c), in increasing r and then entry order. Port of <c>transpose( aRows )</c>.
	/// </summary>
	public PoissonSparseMatrix Transpose(int outRows)
	{
		// The C++'s dimension check, with its test as written (it only looks at entries whose
		// column is at least the row count).
		int requiredRows = 0;
		for (int i = 0; i < Rows; i++)
		{
			for (int j = 0; j < rowSizes[i]; j++)
			{
				if (Rows <= columns[i][j])
				{
					requiredRows = columns[i][j] + 1;
				}
			}
		}

		if (requiredRows > outRows)
		{
			throw new InvalidOperationException($"Prescribed output dimension too low: {outRows} < {requiredRows}");
		}

		var a = new PoissonSparseMatrix();
		a.Resize(outRows);
		var counts = new int[outRows];
		for (int i = 0; i < Rows; i++)
		{
			for (int j = 0; j < rowSizes[i]; j++)
			{
				counts[columns[i][j]]++;
			}
		}

		for (int i = 0; i < outRows; i++)
		{
			a.SetRowSize(i, counts[i]);
			a.rowSizes[i] = 0;
		}

		for (int i = 0; i < Rows; i++)
		{
			for (int j = 0; j < rowSizes[i]; j++)
			{
				int ii = columns[i][j];
				int k = a.rowSizes[ii]++;
				a.columns[ii][k] = i;
				a.values[ii][k] = values[i][j];
			}
		}

		return a;
	}

	/// <summary>
	/// The product this * <paramref name="b"/>: each output row sums a(i, k) * b(k, j) per
	/// column j in a's then b's entry order, and lists its columns in the iteration order of the
	/// libc++ std::unordered_map&lt;int, float&gt; it was gathered in (LibcxxUnorderedMap), which
	/// later sums depend on. Port of <c>SparseMatrix::operator*( const SparseMatrix&amp; )</c>.
	/// </summary>
	public PoissonSparseMatrix Multiply(PoissonSparseMatrix b)
	{
		int aCols = 0;
		for (int i = 0; i < Rows; i++)
		{
			for (int j = 0; j < rowSizes[i]; j++)
			{
				if (aCols <= columns[i][j])
				{
					aCols = columns[i][j] + 1;
				}
			}
		}

		if (b.Rows < aCols)
		{
			throw new InvalidOperationException($"Matrix sizes do not support multiplication {Rows} x {aCols} * {b.Rows}");
		}

		var result = new PoissonSparseMatrix();
		result.Resize(Rows);
		var row = new ColmapSharp.Util.LibcxxUnorderedMap<float>();
		for (int i = 0; i < Rows; i++)
		{
			row.Reset();
			for (int j = 0; j < rowSizes[i]; j++)
			{
				int idx1 = columns[i][j];
				float aValue = values[i][j];
				for (int k = 0; k < b.rowSizes[idx1]; k++)
				{
					int idx2 = b.columns[idx1][k];
					float bValue = b.values[idx1][k];
					int node = row.Find(idx2);
					if (node == -1)
					{
						row.Insert(idx2, aValue * bValue);
					}
					else
					{
						row.Value(node) += aValue * bValue;
					}
				}
			}

			result.SetRowSize(i, row.Count);
			int count = 0;
			for (int node = row.First; node != -1; node = row.Next(node))
			{
				result.columns[i][count] = row.Key(node);
				result.values[i][count++] = row.Value(node);
			}
		}

		return result;
	}

	/// <summary>
	/// out[i] = (or +=) the sum over row i of in[N] * value, accumulated in float in entry order.
	/// Port of <c>multiply( in , out , multiplyFlag )</c> (0 or MULTIPLY_ADD).
	/// </summary>
	public void Multiply(float[] input, int inOffset, float[] output, int outOffset, bool add = false)
	{
		for (int i = 0; i < Rows; i++)
		{
			float temp = 0;
			int[] cols = columns[i];
			float[] vals = values[i];
			for (int j = 0; j < rowSizes[i]; j++)
			{
				temp += input[inOffset + cols[j]] * vals[j];
			}

			if (add)
			{
				output[outOffset + i] += temp;
			}
			else
			{
				output[outOffset + i] = temp;
			}
		}
	}

	/// <summary>
	/// The reciprocal of each row's diagonal (the sum of its entries in column i; 0 stays 0),
	/// the reciprocal taken in double. Port of <c>setDiagonalR( diagonal )</c>.
	/// </summary>
	public void SetDiagonalR(float[] diagonal)
	{
		for (int i = 0; i < Rows; i++)
		{
			diagonal[i] = 0;
			for (int j = 0; j < rowSizes[i]; j++)
			{
				if (columns[i][j] == i)
				{
					diagonal[i] += values[i][j];
				}
			}

			if (diagonal[i] != 0)
			{
				diagonal[i] = (float)(1.0 / diagonal[i]);
			}
		}
	}

	/// <summary>
	/// One multi-colored Gauss-Seidel sweep with a reciprocal diagonal: per color (in order, or
	/// reversed), x[j] += ( b[j] - row j . x ) * diagonal[j]. Port of <c>gsIteration(
	/// multiColorIndices , diagonal , b , x , forward , dReciprocal = true )</c> (rows of one color
	/// never share an unknown, so the C++'s parallel order within a color does not matter).
	/// </summary>
	public void GsIteration(List<int>[] colors, float[] diagonal, float[] b, int bOffset, float[] x, int xOffset, bool forward)
	{
		for (int c = 0; c < colors.Length; c++)
		{
			List<int> indices = colors[forward ? c : colors.Length - 1 - c];
			for (int k = 0; k < indices.Count; k++)
			{
				int jj = indices[k];
				float residual = b[bOffset + jj];
				int[] cols = columns[jj];
				float[] vals = values[jj];
				for (int e = 0; e < rowSizes[jj]; e++)
				{
					residual -= x[xOffset + cols[e]] * vals[e];
				}

				x[xOffset + jj] += residual * diagonal[jj];
			}
		}
	}

	/// <summary>
	/// Conjugate gradients on this matrix (plus, with <paramref name="addDCTerm"/>, the mean of
	/// the input added to every output, pinning the constant), from x, for at most
	/// <paramref name="iters"/> iterations or until the squared residual falls below
	/// eps^2 times its start. Scalars are float as in the C++'s SolveCG&lt; SPDFunctor , float ,
	/// float &gt; (Dot is v * w), compared against the double eps. Every 50th iteration recomputes
	/// the residual from x. Returns the iterations run. Port of LinearSolvers' <c>SolveCG( M , dim
	/// , b , iters , x , eps , Dot )</c> (SparseMatrixInterface.inl) with FEMTree's SPDFunctor.
	/// </summary>
	public int SolveCG(bool addDCTerm, float[] b, int bOffset, int iters, float[] x, int xOffset, double eps)
	{
		int dim = Rows;
		eps *= eps;
		var r = new float[dim];
		var d = new float[dim];
		var q = new float[dim];
		float deltaNew = 0;
		Apply(addDCTerm, x, xOffset, r);
		float scratch = 0;
		for (int i = 0; i < dim; i++)
		{
			d[i] = r[i] = b[bOffset + i] - r[i];
			scratch += r[i] * r[i];
		}

		deltaNew += scratch;
		float delta0 = deltaNew;
		if (deltaNew <= eps)
		{
			return 0;
		}

		int ii;
		for (ii = 0; ii < iters && deltaNew > eps * delta0; ii++)
		{
			Apply(addDCTerm, d, 0, q);
			scratch = 0;
			for (int i = 0; i < dim; i++)
			{
				scratch += d[i] * q[i];
			}

			float dDotQ = 0;
			dDotQ += scratch;
			if (dDotQ == 0)
			{
				break;
			}

			float alpha = deltaNew / dDotQ;
			float deltaOld = deltaNew;
			deltaNew = 0;
			scratch = 0;
			if ((ii % 50) == (50 - 1))
			{
				for (int i = 0; i < dim; i++)
				{
					x[xOffset + i] += d[i] * alpha;
				}

				Apply(addDCTerm, x, xOffset, r);
				for (int i = 0; i < dim; i++)
				{
					r[i] = b[bOffset + i] - r[i];
					scratch += r[i] * r[i];
					x[xOffset + i] += d[i] * alpha;
				}
			}
			else
			{
				for (int i = 0; i < dim; i++)
				{
					r[i] -= q[i] * alpha;
					scratch += r[i] * r[i];
					x[xOffset + i] += d[i] * alpha;
				}
			}

			deltaNew += scratch;
			float beta = deltaNew / deltaOld;
			for (int i = 0; i < dim; i++)
			{
				d[i] = r[i] + d[i] * beta;
			}
		}

		return ii;
	}

	// FEMTree's SPDFunctor: out = M * in, plus the mean of in on every entry with the DC term.
	private void Apply(bool addDCTerm, float[] input, int inOffset, float[] output)
	{
		Multiply(input, inOffset, output, 0);
		if (addDCTerm)
		{
			float average = 0;
			for (int i = 0; i < Rows; i++)
			{
				average += input[inOffset + i];
			}

			average /= Rows;
			for (int i = 0; i < Rows; i++)
			{
				output[i] += average;
			}
		}
	}
}
