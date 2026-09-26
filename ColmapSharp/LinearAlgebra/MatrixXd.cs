// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MatrixXd: heap-backed, dynamically sized matrix of doubles, the replacement for
// Eigen::MatrixXd and the partly dynamic shapes COLMAP's estimators build
// (Eigen::Matrix<double, Dynamic, 9> for the 8-point and homography DLT systems,
// Dynamic x 6 for affine, 12 x 12 for DLT pose). Written here to Eigen's documented
// semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Siblings: VectorXd, the
// fixed-size types (conversions in MatrixXd.Conversions.cs) and the decompositions
// (PartialPivLU, LLT, LDLT, HouseholderQR, ColPivHouseholderQR). Tests:
// ColmapSharp.Tests/LinearAlgebra/DynamicMatrixTests.cs and DecompositionTests.cs.
//
// Storage and mutability: a sealed, mutable class over one column-major double[], the
// same memory order as Eigen's default, so a COLMAP walk over matrix.data() or a
// Map<RowMajor>(col.data()) maps to the same flat index here. It is a class rather than a
// struct-with-array so that assignment never silently aliases (a struct copy would share
// the array); copies are explicit (Clone, Block, Row, Col). Hot loops use the Span views:
// ColumnSpan(j) is a writable view of column j (contiguous), AsSpan() the whole buffer.
// Block/Row/Col return copies; SetBlock/SetRow/SetCol write back.
//
// Arithmetic order: the folder contract in Vector3d.cs. Every product coefficient is a
// left-to-right sum over k seeded with the k = 0 term, no FMA. Eigen evaluates dynamic
// products with a blocked, vectorized kernel, so dynamic products are Tier B against
// COLMAP (rounding may differ in the last bits).
//
// Inventory of Eigen decompositions COLMAP 4.2.0 calls (src/colmap outside ui/, exe/,
// CUDA/Caspar and tests), for the next step (SVD + eigen solvers). Provided here already:
// HouseholderQR (householderQ, matrixQR; math/matrix.h, geometry/pose.cc
// GravityAlignedRotation, 8-point F/E null space), ColPivHouseholderQR (rank, solve;
// absolute_pose.cc, bundle_adjustment_ceres.cc), PartialPivLU (solve; affine,
// homography, essential 5-pt, sensor/models.h), LLT (matrixL; cost_functions/utils.h),
// LDLT (solve, info; optim/tiny_solver.h). Still to provide:
//
// | Eigen type                         | Shape                 | Calls used                        | Callers                                           |
// |------------------------------------|-----------------------|-----------------------------------|---------------------------------------------------|
// | JacobiSVD                          | 3x3                   | matrixU, matrixV (Full), singularValues | fundamental/essential/homography/pose/p3p/calibration, generalized_relative_pose, triangulation |
// | JacobiSVD                          | Dynamic x 9, x 6      | matrixV (Full), singularValues    | 8-pt F/E, homography DLT, affine                  |
// | JacobiSVD                          | 4x4, 6x4, 6x3, 6x5, 12x12 | matrixV (Full)                | triangulation, absolute_pose (DLT, EPnP)          |
// | JacobiSVD                          | MatrixXd              | matrixU (Full)                    | geometry/pose.cc (ComputeClosestRotation...)      |
// | SelfAdjointEigenSolver             | 4x4                   | eigenvectors, eigenvalues (ascending) | geometry/triangulation.cc                     |
// | EigenSolver (general, real input)  | MatrixXd (companion)  | eigenvalues() complex, no vectors | math/polynomial.cc FindPolynomialRootsCompanionMatrix |
// | EigenSolver                        | 4x4                   | eigenvalues + eigenvectors (complex) | generalized_relative_pose.cc                   |
// | FullPivLU                          | 3 x N (Dynamic)       | rank()                            | estimators/solvers/similarity_transform.h         |
// Sparse SimplicialLLT/LDLT and CHOLMOD belong to Phase 5 (sparse_cholesky), not here;
// retrieval/ and coordinate_frame.cc (AGPL) are out of scope.

using System.Globalization;
using System.Text;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Dynamically sized, column-major matrix of doubles. Replacement for Eigen::MatrixXd and
/// Eigen::Matrix&lt;double, Dynamic, N&gt;. Mutable reference type; copies are explicit.
/// </summary>
public sealed partial class MatrixXd
{
	// Column-major: element (row, col) is at col * Rows + row.
	private readonly double[] _data;

	/// <summary>A zero matrix of the given shape, Eigen's MatrixXd::Zero(rows, cols).</summary>
	public MatrixXd(int rows, int cols)
	{
		ArgumentOutOfRangeException.ThrowIfNegative(rows);
		ArgumentOutOfRangeException.ThrowIfNegative(cols);
		Rows = rows;
		Cols = cols;
		_data = new double[rows * cols];
	}

	private MatrixXd(int rows, int cols, double[] data)
	{
		Rows = rows;
		Cols = cols;
		_data = data;
	}

	/// <summary>Number of rows.</summary>
	public int Rows { get; }

	/// <summary>Number of columns.</summary>
	public int Cols { get; }

	/// <summary>Number of coefficients, Rows * Cols (Eigen's size()).</summary>
	public int Size => _data.Length;

	/// <summary>Coefficient at (row, col), readable and writable.</summary>
	public double this[int row, int col]
	{
		get
		{
			CheckIndex(row, col);
			return _data[col * Rows + row];
		}

		set
		{
			CheckIndex(row, col);
			_data[col * Rows + row] = value;
		}
	}

	/// <summary>The whole column-major buffer as a writable span (Eigen's data()).</summary>
	public Span<double> AsSpan() => _data;

	/// <summary>Column j as a writable, contiguous span view.</summary>
	public Span<double> ColumnSpan(int j)
	{
		if ((uint)j >= (uint)Cols)
		{
			throw new ArgumentOutOfRangeException(nameof(j));
		}

		return _data.AsSpan(j * Rows, Rows);
	}

	/// <summary>Eigen's MatrixXd::Zero(rows, cols).</summary>
	public static MatrixXd Zero(int rows, int cols) => new(rows, cols);

	/// <summary>Eigen's MatrixXd::Identity(n, n).</summary>
	public static MatrixXd Identity(int n) => Identity(n, n);

	/// <summary>Eigen's MatrixXd::Identity(rows, cols): ones on the main diagonal.</summary>
	public static MatrixXd Identity(int rows, int cols)
	{
		var m = new MatrixXd(rows, cols);
		for (int i = 0; i < Math.Min(rows, cols); i++)
		{
			m._data[i * rows + i] = 1;
		}

		return m;
	}

	/// <summary>Eigen's MatrixXd::Constant(rows, cols, value).</summary>
	public static MatrixXd Constant(int rows, int cols, double value)
	{
		var m = new MatrixXd(rows, cols);
		m._data.AsSpan().Fill(value);
		return m;
	}

	/// <summary>Builds a matrix from values in column-major order (Eigen's memory layout).</summary>
	public static MatrixXd FromColumnMajor(int rows, int cols, ReadOnlySpan<double> values)
	{
		if (values.Length != rows * cols)
		{
			throw new ArgumentException($"Expected {rows * cols} values.", nameof(values));
		}

		return new MatrixXd(rows, cols, values.ToArray());
	}

	/// <summary>
	/// Builds a matrix from values in row-major reading order, like Eigen's comma
	/// initializer.
	/// </summary>
	public static MatrixXd FromRowMajor(int rows, int cols, ReadOnlySpan<double> values)
	{
		if (values.Length != rows * cols)
		{
			throw new ArgumentException($"Expected {rows * cols} values.", nameof(values));
		}

		var m = new MatrixXd(rows, cols);
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				m._data[c * rows + r] = values[r * cols + c];
			}
		}

		return m;
	}

	/// <summary>A diagonal matrix, Eigen's v.asDiagonal().</summary>
	public static MatrixXd FromDiagonal(VectorXd diagonal)
	{
		int n = diagonal.Length;
		var m = new MatrixXd(n, n);
		for (int i = 0; i < n; i++)
		{
			m._data[i * n + i] = diagonal[i];
		}

		return m;
	}

	/// <summary>A deep copy.</summary>
	public MatrixXd Clone() => new(Rows, Cols, (double[])_data.Clone());

	/// <summary>A copy of column j.</summary>
	public VectorXd Col(int j) => new(ColumnSpan(j));

	/// <summary>A copy of row i.</summary>
	public VectorXd Row(int i)
	{
		CheckIndex(i, 0, allowEmptyCols: true);
		var r = new VectorXd(Cols);
		for (int c = 0; c < Cols; c++)
		{
			r[c] = _data[c * Rows + i];
		}

		return r;
	}

	/// <summary>Overwrites column j.</summary>
	public void SetCol(int j, VectorXd values)
	{
		RequireSameLength(values.Length, Rows);
		values.AsSpan().CopyTo(ColumnSpan(j));
	}

	/// <summary>Overwrites row i.</summary>
	public void SetRow(int i, VectorXd values)
	{
		CheckIndex(i, 0, allowEmptyCols: true);
		RequireSameLength(values.Length, Cols);
		for (int c = 0; c < Cols; c++)
		{
			_data[c * Rows + i] = values[c];
		}
	}

	/// <summary>A copy of the rows x cols block starting at (row, col), Eigen's block().</summary>
	public MatrixXd Block(int row, int col, int rows, int cols)
	{
		CheckBlock(row, col, rows, cols);
		var b = new MatrixXd(rows, cols);
		for (int c = 0; c < cols; c++)
		{
			_data.AsSpan((col + c) * Rows + row, rows).CopyTo(b._data.AsSpan(c * rows, rows));
		}

		return b;
	}

	/// <summary>Writes a block into this matrix with its top-left corner at (row, col).</summary>
	public void SetBlock(int row, int col, MatrixXd block)
	{
		CheckBlock(row, col, block.Rows, block.Cols);
		for (int c = 0; c < block.Cols; c++)
		{
			block._data.AsSpan(c * block.Rows, block.Rows).CopyTo(_data.AsSpan((col + c) * Rows + row, block.Rows));
		}
	}

	/// <summary>A copy of the first n rows, Eigen's topRows(n).</summary>
	public MatrixXd TopRows(int n) => Block(0, 0, n, Cols);

	/// <summary>A copy of the last n rows, Eigen's bottomRows(n).</summary>
	public MatrixXd BottomRows(int n) => Block(Rows - n, 0, n, Cols);

	/// <summary>A copy of the first n columns, Eigen's leftCols(n).</summary>
	public MatrixXd LeftCols(int n) => Block(0, 0, Rows, n);

	/// <summary>A copy of the last n columns, Eigen's rightCols(n).</summary>
	public MatrixXd RightCols(int n) => Block(0, Cols - n, Rows, n);

	/// <summary>The transpose, as a new matrix.</summary>
	public MatrixXd Transpose()
	{
		var t = new MatrixXd(Cols, Rows);
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				t._data[r * Cols + c] = _data[c * Rows + r];
			}
		}

		return t;
	}

	/// <summary>Rows in reverse order, Eigen's colwise().reverse().</summary>
	public MatrixXd ReverseRows()
	{
		var m = new MatrixXd(Rows, Cols);
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r < Rows; r++)
			{
				m._data[c * Rows + r] = _data[c * Rows + (Rows - 1 - r)];
			}
		}

		return m;
	}

	/// <summary>Columns in reverse order, Eigen's rowwise().reverse().</summary>
	public MatrixXd ReverseCols()
	{
		var m = new MatrixXd(Rows, Cols);
		for (int c = 0; c < Cols; c++)
		{
			_data.AsSpan((Cols - 1 - c) * Rows, Rows).CopyTo(m._data.AsSpan(c * Rows, Rows));
		}

		return m;
	}

	/// <summary>Sum of the diagonal, left to right.</summary>
	public double Trace()
	{
		int n = Math.Min(Rows, Cols);
		if (n == 0)
		{
			return 0;
		}

		double sum = _data[0];
		for (int i = 1; i < n; i++)
		{
			sum += _data[i * Rows + i];
		}

		return sum;
	}

	/// <summary>Squared Frobenius norm, a left-to-right sum over the column-major buffer.</summary>
	public double SquaredNorm() => VectorXd.Dot(_data, _data);

	/// <summary>Frobenius norm, Eigen's norm() on a matrix.</summary>
	public double Norm() => Math.Sqrt(SquaredNorm());

	/// <summary>
	/// Determinant through PartialPivLU, like Eigen does for a dynamic square matrix.
	/// </summary>
	public double Determinant() => new PartialPivLU(this).Determinant();

	/// <summary>Inverse through PartialPivLU (no singularity check, like Eigen).</summary>
	public MatrixXd Inverse() => new PartialPivLU(this).Inverse();

	/// <summary>True when every coefficient strictly below the diagonal is within precision of 0 relative to the largest coefficient on or above it (Eigen's isUpperTriangular).</summary>
	public bool IsUpperTriangular(double precision = LinearAlgebraConstants.DummyPrecision)
	{
		double maxAbsOnUpperPart = 0;
		for (int c = 0; c < Cols; c++)
		{
			for (int r = 0; r <= Math.Min(c, Rows - 1); r++)
			{
				maxAbsOnUpperPart = Math.Max(maxAbsOnUpperPart, Math.Abs(_data[c * Rows + r]));
			}
		}

		double threshold = maxAbsOnUpperPart * precision;
		for (int c = 0; c < Cols; c++)
		{
			for (int r = c + 1; r < Rows; r++)
			{
				if (Math.Abs(_data[c * Rows + r]) > threshold)
				{
					return false;
				}
			}
		}

		return true;
	}

	/// <summary>
	/// True when the columns are orthonormal within precision (Eigen's isUnitary: every
	/// column dot product is approximately 0 or 1).
	/// </summary>
	public bool IsUnitary(double precision = LinearAlgebraConstants.DummyPrecision)
	{
		for (int i = 0; i < Cols; i++)
		{
			ReadOnlySpan<double> ci = ColumnSpan(i);
			if (Math.Abs(VectorXd.Dot(ci, ci) - 1) > precision)
			{
				return false;
			}

			for (int j = 0; j < i; j++)
			{
				if (Math.Abs(VectorXd.Dot(ci, ColumnSpan(j))) > precision)
				{
					return false;
				}
			}
		}

		return true;
	}

	/// <summary>Eigen's isApprox: ||a - b|| &lt;= precision * min(||a||, ||b||), Frobenius norms.</summary>
	public bool IsApprox(MatrixXd other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return Rows == other.Rows && Cols == other.Cols
			&& LinearAlgebraConstants.IsApprox(_data, other._data, precision);
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		var sb = new StringBuilder("[");
		for (int r = 0; r < Rows; r++)
		{
			sb.Append(r == 0 ? "[" : ", [");
			for (int c = 0; c < Cols; c++)
			{
				sb.Append(c == 0 ? string.Empty : ", ");
				sb.Append(_data[c * Rows + r].ToString("R", CultureInfo.InvariantCulture));
			}

			sb.Append(']');
		}

		return sb.Append(']').ToString();
	}

	private void CheckIndex(int row, int col, bool allowEmptyCols = false)
	{
		if ((uint)row >= (uint)Rows)
		{
			throw new ArgumentOutOfRangeException(nameof(row));
		}

		if (!(allowEmptyCols && col == 0) && (uint)col >= (uint)Cols)
		{
			throw new ArgumentOutOfRangeException(nameof(col));
		}
	}

	private void CheckBlock(int row, int col, int rows, int cols)
	{
		if (row < 0 || col < 0 || rows < 0 || cols < 0 || row + rows > Rows || col + cols > Cols)
		{
			throw new ArgumentOutOfRangeException(
				nameof(row), $"Block ({row}, {col}, {rows}x{cols}) is outside a {Rows}x{Cols} matrix.");
		}
	}

	private static void RequireSameLength(int a, int b)
	{
		if (a != b)
		{
			throw new ArgumentException($"Dimension mismatch: {a} vs {b}.");
		}
	}
}
