// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix2d: fixed-size 2x2 matrix of doubles, the replacement for Eigen::Matrix2d (2D
// Jacobians and covariances in the camera models and feature code). Written here to
// Eigen's documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md).
// Siblings: Matrix3d, Matrix3x4d, Matrix4d. Tests:
// ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs.
//
// Layout and arithmetic follow Matrix3d.cs: column-major storage, a row-major
// 4-argument constructor like Eigen's comma initializer, left-to-right product sums, no
// FMA. Inverse is the closed form [d -b; -c a] / det with no singularity check.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 2x2 matrix of doubles. Replacement for Eigen::Matrix2d.
/// </summary>
public readonly struct Matrix2d : IEquatable<Matrix2d>
{
	private readonly double _m00;
	private readonly double _m10;
	private readonly double _m01;
	private readonly double _m11;

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix2d(double m00, double m01, double m10, double m11)
	{
		_m00 = m00;
		_m01 = m01;
		_m10 = m10;
		_m11 = m11;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix2d Zero => default;

	/// <summary>The identity matrix.</summary>
	public static Matrix2d Identity => new(1, 0, 0, 1);

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col] => (row, col) switch
	{
		(0, 0) => _m00,
		(0, 1) => _m01,
		(1, 0) => _m10,
		(1, 1) => _m11,
		_ => throw new ArgumentOutOfRangeException(row is 0 or 1 ? nameof(col) : nameof(row)),
	};

	/// <summary>Builds a matrix from 4 values in column-major order (Eigen's memory layout).</summary>
	public static Matrix2d FromColumnMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != 4)
		{
			throw new ArgumentException("Expected 4 values.", nameof(values));
		}

		return new Matrix2d(values[0], values[2], values[1], values[3]);
	}

	/// <summary>Builds a matrix whose columns are the given vectors.</summary>
	public static Matrix2d FromColumns(Vector2d c0, Vector2d c1) => new(c0.X, c1.X, c0.Y, c1.Y);

	/// <summary>Builds a matrix whose rows are the given vectors.</summary>
	public static Matrix2d FromRows(Vector2d r0, Vector2d r1) => new(r0.X, r0.Y, r1.X, r1.Y);

	/// <summary>Row i as a vector.</summary>
	public Vector2d Row(int i) => new(this[i, 0], this[i, 1]);

	/// <summary>Column j as a vector.</summary>
	public Vector2d Col(int j) => new(this[0, j], this[1, j]);

	/// <summary>Sum of the diagonal.</summary>
	public double Trace() => _m00 + _m11;

	/// <summary>The transpose.</summary>
	public Matrix2d Transpose() => new(_m00, _m10, _m01, _m11);

	/// <summary>Determinant, m00*m11 - m01*m10.</summary>
	public double Determinant() => _m00 * _m11 - _m01 * _m10;

	/// <summary>The inverse, [m11 -m01; -m10 m00] / det. No singularity check.</summary>
	public Matrix2d Inverse()
	{
		double det = Determinant();
		return new Matrix2d(_m11 / det, -_m01 / det, -_m10 / det, _m00 / det);
	}

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix2d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		ReadOnlySpan<double> a = [_m00, _m10, _m01, _m11];
		ReadOnlySpan<double> b = [other._m00, other._m10, other._m01, other._m11];
		return LinearAlgebraConstants.IsApprox(a, b, precision);
	}

	/// <summary>Matrix sum.</summary>
	public static Matrix2d operator +(Matrix2d a, Matrix2d b)
	{
		return new Matrix2d(a._m00 + b._m00, a._m01 + b._m01, a._m10 + b._m10, a._m11 + b._m11);
	}

	/// <summary>Matrix difference.</summary>
	public static Matrix2d operator -(Matrix2d a, Matrix2d b)
	{
		return new Matrix2d(a._m00 - b._m00, a._m01 - b._m01, a._m10 - b._m10, a._m11 - b._m11);
	}

	/// <summary>Negation.</summary>
	public static Matrix2d operator -(Matrix2d a) => new(-a._m00, -a._m01, -a._m10, -a._m11);

	/// <summary>Scalar product.</summary>
	public static Matrix2d operator *(Matrix2d a, double s) => new(a._m00 * s, a._m01 * s, a._m10 * s, a._m11 * s);

	/// <summary>Scalar product.</summary>
	public static Matrix2d operator *(double s, Matrix2d a) => a * s;

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Matrix2d operator /(Matrix2d a, double s) => new(a._m00 / s, a._m01 / s, a._m10 / s, a._m11 / s);

	/// <summary>Matrix-vector product.</summary>
	public static Vector2d operator *(Matrix2d a, Vector2d v)
	{
		return new Vector2d(a._m00 * v.X + a._m01 * v.Y, a._m10 * v.X + a._m11 * v.Y);
	}

	/// <summary>Matrix product, left-to-right sums.</summary>
	public static Matrix2d operator *(Matrix2d a, Matrix2d b)
	{
		return new Matrix2d(
			a._m00 * b._m00 + a._m01 * b._m10,
			a._m00 * b._m01 + a._m01 * b._m11,
			a._m10 * b._m00 + a._m11 * b._m10,
			a._m10 * b._m01 + a._m11 * b._m11);
	}

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Matrix2d a, Matrix2d b)
	{
		return a._m00 == b._m00 && a._m01 == b._m01 && a._m10 == b._m10 && a._m11 == b._m11;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Matrix2d a, Matrix2d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix2d other)
	{
		return _m00.Equals(other._m00) && _m01.Equals(other._m01) && _m10.Equals(other._m10) && _m11.Equals(other._m11);
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix2d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(_m00, _m10, _m01, _m11);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"[{Row(0)}, {Row(1)}]");
}
