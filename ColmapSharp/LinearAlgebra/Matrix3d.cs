// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix3d: fixed-size 3x3 matrix of doubles, the replacement for Eigen::Matrix3d
// (rotations, calibration matrices, essential/fundamental/homography matrices). Written
// here to Eigen's documented semantics; Eigen (MPL-2.0) is not ported
// (docs/LICENSE_AUDIT.md). Siblings: Matrix2d, Matrix3x4d, Matrix4d, and Quaterniond /
// AngleAxisd, which convert to and from it. Tests:
// ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs.
//
// Layout: coefficients are stored column-major like Eigen, so FromColumnMajor/
// CopyToColumnMajor match matrix.data(). The 9-argument constructor takes the values in
// row-major reading order, like Eigen's comma initializer (m << a, b, c, ...).
//
// Arithmetic order: every product coefficient is a left-to-right sum over k of
// a(i,k)*b(k,j), no FMA (the contract is in Vector3d.cs). Determinant and Inverse use the
// closed-form cofactor expansion along the first row. Tier B and unpinned: Eigen's closed
// form may order the terms differently, and no pycolmap 4.2.0 binding reaches Eigen's 3x3
// inverse or determinant cleanly (FundamentalFromEssentialMatrix and
// DecomposeProjectionMatrix are not bound; pose_from_homography_matrix mixes K.inverse()
// into an SVD), so there is no oracle fixture for them yet. Like Eigen, Inverse does not check for singularity; a
// singular matrix yields infinities or NaN.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3x3 matrix of doubles. Replacement for Eigen::Matrix3d.
/// </summary>
public readonly struct Matrix3d : IEquatable<Matrix3d>
{
	private const int Size = 3;

	// Column-major: element (row, col) is at col * 3 + row.
	private readonly Buffer9 _m;

	private Matrix3d(in Buffer9 m)
	{
		_m = m;
	}

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix3d(
		double m00, double m01, double m02,
		double m10, double m11, double m12,
		double m20, double m21, double m22)
	{
		_m[0] = m00;
		_m[1] = m10;
		_m[2] = m20;
		_m[3] = m01;
		_m[4] = m11;
		_m[5] = m21;
		_m[6] = m02;
		_m[7] = m12;
		_m[8] = m22;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix3d Zero => default;

	/// <summary>The identity matrix.</summary>
	public static Matrix3d Identity => new(1, 0, 0, 0, 1, 0, 0, 0, 1);

	/// <summary>Eigen's Matrix3d::Ones().</summary>
	public static Matrix3d Ones => new(1, 1, 1, 1, 1, 1, 1, 1, 1);

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col]
	{
		get
		{
			if ((uint)row >= Size || (uint)col >= Size)
			{
				throw new ArgumentOutOfRangeException(row >= 0 && row < Size ? nameof(col) : nameof(row));
			}

			return _m[col * Size + row];
		}
	}

	/// <summary>Coefficient by its column-major index (col * 3 + row), unchecked beyond the inline array's own bounds.</summary>
	internal double AtColumnMajor(int index) => _m[index];

	/// <summary>Builds a matrix from 9 values in column-major order (Eigen's memory layout).</summary>
	public static Matrix3d FromColumnMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != 9)
		{
			throw new ArgumentException("Expected 9 values.", nameof(values));
		}

		var m = default(Buffer9);
		values.CopyTo(m);
		return new Matrix3d(m);
	}

	/// <summary>Builds a matrix whose rows are the given vectors.</summary>
	public static Matrix3d FromRows(Vector3d r0, Vector3d r1, Vector3d r2)
	{
		return new Matrix3d(r0.X, r0.Y, r0.Z, r1.X, r1.Y, r1.Z, r2.X, r2.Y, r2.Z);
	}

	/// <summary>Builds a matrix whose columns are the given vectors.</summary>
	public static Matrix3d FromColumns(Vector3d c0, Vector3d c1, Vector3d c2)
	{
		return new Matrix3d(c0.X, c1.X, c2.X, c0.Y, c1.Y, c2.Y, c0.Z, c1.Z, c2.Z);
	}

	/// <summary>Diagonal matrix, Eigen's v.asDiagonal().</summary>
	public static Matrix3d FromDiagonal(Vector3d diagonal)
	{
		return new Matrix3d(diagonal.X, 0, 0, 0, diagonal.Y, 0, 0, 0, diagonal.Z);
	}

	/// <summary>Copies the 9 coefficients out in column-major order.</summary>
	public void CopyToColumnMajor(Span<double> destination)
	{
		((ReadOnlySpan<double>)_m).CopyTo(destination);
	}

	/// <summary>Row i as a vector.</summary>
	public Vector3d Row(int i) => new(this[i, 0], this[i, 1], this[i, 2]);

	/// <summary>Column j as a vector.</summary>
	public Vector3d Col(int j) => new(this[0, j], this[1, j], this[2, j]);

	/// <summary>The diagonal (m00, m11, m22).</summary>
	public Vector3d Diagonal() => new(_m[0], _m[4], _m[8]);

	/// <summary>
	/// Sum of the diagonal, grouped m00 + (m11 + m22). That is how Eigen's unrolled
	/// reduction over the (strided, so not vectorized) diagonal splits it, as
	/// oracle/linear_algebra_rotations.py shows through Quaterniond.FromRotationMatrix:
	/// bit-identical on all fixture cases with this grouping, 4 mismatches left to right.
	/// </summary>
	public double Trace() => _m[0] + (_m[4] + _m[8]);

	/// <summary>The transpose.</summary>
	public Matrix3d Transpose()
	{
		return new Matrix3d(
			_m[0], _m[1], _m[2],
			_m[3], _m[4], _m[5],
			_m[6], _m[7], _m[8]);
	}

	/// <summary>Frobenius norm, Eigen's norm() on a matrix.</summary>
	public double Norm()
	{
		double sum = _m[0] * _m[0];
		for (int i = 1; i < 9; i++)
		{
			sum += _m[i] * _m[i];
		}

		return Math.Sqrt(sum);
	}

	/// <summary>Determinant by cofactor expansion along the first row.</summary>
	public double Determinant()
	{
		double m00 = _m[0], m01 = _m[3], m02 = _m[6];
		double m10 = _m[1], m11 = _m[4], m12 = _m[7];
		double m20 = _m[2], m21 = _m[5], m22 = _m[8];
		double c00 = m11 * m22 - m12 * m21;
		double c01 = m12 * m20 - m10 * m22;
		double c02 = m10 * m21 - m11 * m20;
		return m00 * c00 + m01 * c01 + m02 * c02;
	}

	/// <summary>
	/// The inverse, as the adjugate (transposed cofactor matrix) divided by the
	/// determinant. No singularity check, like Eigen's inverse().
	/// </summary>
	public Matrix3d Inverse()
	{
		double m00 = _m[0], m01 = _m[3], m02 = _m[6];
		double m10 = _m[1], m11 = _m[4], m12 = _m[7];
		double m20 = _m[2], m21 = _m[5], m22 = _m[8];

		// Cofactors c(i, j) of element (i, j).
		double c00 = m11 * m22 - m12 * m21;
		double c01 = m12 * m20 - m10 * m22;
		double c02 = m10 * m21 - m11 * m20;
		double c10 = m02 * m21 - m01 * m22;
		double c11 = m00 * m22 - m02 * m20;
		double c12 = m01 * m20 - m00 * m21;
		double c20 = m01 * m12 - m02 * m11;
		double c21 = m02 * m10 - m00 * m12;
		double c22 = m00 * m11 - m01 * m10;
		double det = m00 * c00 + m01 * c01 + m02 * c02;

		// inverse(i, j) = c(j, i) / det.
		return new Matrix3d(
			c00 / det, c10 / det, c20 / det,
			c01 / det, c11 / det, c21 / det,
			c02 / det, c12 / det, c22 / det);
	}

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix3d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return LinearAlgebraConstants.IsApprox(_m, other._m, precision);
	}

	/// <summary>Matrix sum.</summary>
	public static Matrix3d operator +(Matrix3d a, Matrix3d b)
	{
		var m = default(Buffer9);
		for (int i = 0; i < 9; i++)
		{
			m[i] = a._m[i] + b._m[i];
		}

		return new Matrix3d(m);
	}

	/// <summary>Matrix difference.</summary>
	public static Matrix3d operator -(Matrix3d a, Matrix3d b)
	{
		var m = default(Buffer9);
		for (int i = 0; i < 9; i++)
		{
			m[i] = a._m[i] - b._m[i];
		}

		return new Matrix3d(m);
	}

	/// <summary>Negation.</summary>
	public static Matrix3d operator -(Matrix3d a)
	{
		var m = default(Buffer9);
		for (int i = 0; i < 9; i++)
		{
			m[i] = -a._m[i];
		}

		return new Matrix3d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix3d operator *(Matrix3d a, double s)
	{
		var m = default(Buffer9);
		for (int i = 0; i < 9; i++)
		{
			m[i] = a._m[i] * s;
		}

		return new Matrix3d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix3d operator *(double s, Matrix3d a) => a * s;

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Matrix3d operator /(Matrix3d a, double s)
	{
		var m = default(Buffer9);
		for (int i = 0; i < 9; i++)
		{
			m[i] = a._m[i] / s;
		}

		return new Matrix3d(m);
	}

	/// <summary>Matrix-vector product, each row a left-to-right dot product.</summary>
	public static Vector3d operator *(Matrix3d a, Vector3d v)
	{
		return new Vector3d(
			a._m[0] * v.X + a._m[3] * v.Y + a._m[6] * v.Z,
			a._m[1] * v.X + a._m[4] * v.Y + a._m[7] * v.Z,
			a._m[2] * v.X + a._m[5] * v.Y + a._m[8] * v.Z);
	}

	/// <summary>Matrix product; (i, j) = a(i,0)*b(0,j) + a(i,1)*b(1,j) + a(i,2)*b(2,j).</summary>
	public static Matrix3d operator *(Matrix3d a, Matrix3d b)
	{
		var m = default(Buffer9);
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				m[j * 3 + i] = a._m[i] * b._m[j * 3] + a._m[3 + i] * b._m[j * 3 + 1] + a._m[6 + i] * b._m[j * 3 + 2];
			}
		}

		return new Matrix3d(m);
	}

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Matrix3d a, Matrix3d b)
	{
		for (int i = 0; i < 9; i++)
		{
			if (a._m[i] != b._m[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Matrix3d a, Matrix3d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix3d other) => ((ReadOnlySpan<double>)_m).SequenceEqual(other._m);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode()
	{
		var hash = default(HashCode);
		for (int i = 0; i < 9; i++)
		{
			hash.Add(_m[i]);
		}

		return hash.ToHashCode();
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		return string.Create(
			CultureInfo.InvariantCulture,
			$"[{Row(0)}, {Row(1)}, {Row(2)}]");
	}
}
