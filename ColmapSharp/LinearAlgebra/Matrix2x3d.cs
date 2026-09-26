// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix2x3d: fixed-size 2x3 matrix of doubles, the replacement for Eigen::Matrix2x3d
// (COLMAP's projection Jacobian d(x, y) / d(u, v, w), Sensor/CameraModels.cs). Written here
// to Eigen's documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md).
// Its transpose shape is Matrix3x2d; siblings Matrix2d, Matrix3d, Matrix3x4d.
// Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs.
//
// Only what the ported code uses: coefficient access, rows, the Frobenius norm, isApprox
// and the two products. Product sums run left to right with no FMA.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 2x3 matrix of doubles. Replacement for Eigen::Matrix2x3d.
/// </summary>
public readonly struct Matrix2x3d : IEquatable<Matrix2x3d>
{
	private readonly double _m00;
	private readonly double _m01;
	private readonly double _m02;
	private readonly double _m10;
	private readonly double _m11;
	private readonly double _m12;

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix2x3d(double m00, double m01, double m02, double m10, double m11, double m12)
	{
		_m00 = m00;
		_m01 = m01;
		_m02 = m02;
		_m10 = m10;
		_m11 = m11;
		_m12 = m12;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix2x3d Zero => default;

	/// <summary>
	/// Builds a matrix from 6 values in row-major order, like
	/// <c>Eigen::Map&lt;const Eigen::Matrix&lt;double, 2, 3, Eigen::RowMajor&gt;&gt;</c>.
	/// </summary>
	public static Matrix2x3d FromRowMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != 6)
		{
			throw new ArgumentException("Expected 6 values.", nameof(values));
		}

		return new Matrix2x3d(values[0], values[1], values[2], values[3], values[4], values[5]);
	}

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col] => (row, col) switch
	{
		(0, 0) => _m00,
		(0, 1) => _m01,
		(0, 2) => _m02,
		(1, 0) => _m10,
		(1, 1) => _m11,
		(1, 2) => _m12,
		_ => throw new ArgumentOutOfRangeException(row is 0 or 1 ? nameof(col) : nameof(row)),
	};

	/// <summary>Row i as a vector.</summary>
	public Vector3d Row(int i) => new(this[i, 0], this[i, 1], this[i, 2]);

	/// <summary>Frobenius norm, Eigen's norm() on a matrix.</summary>
	public double Norm() =>
		Math.Sqrt(_m00 * _m00 + _m10 * _m10 + _m01 * _m01 + _m11 * _m11 + _m02 * _m02 + _m12 * _m12);

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix2x3d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		ReadOnlySpan<double> a = [_m00, _m10, _m01, _m11, _m02, _m12];
		ReadOnlySpan<double> b = [other._m00, other._m10, other._m01, other._m11, other._m02, other._m12];
		return LinearAlgebraConstants.IsApprox(a, b, precision);
	}

	/// <summary>Matrix-vector product.</summary>
	public static Vector2d operator *(Matrix2x3d a, Vector3d v) =>
		new(a._m00 * v.X + a._m01 * v.Y + a._m02 * v.Z, a._m10 * v.X + a._m11 * v.Y + a._m12 * v.Z);

	/// <summary>Matrix product, left-to-right sums.</summary>
	public static Matrix2d operator *(Matrix2x3d a, Matrix3x2d b) => new(
		a._m00 * b[0, 0] + a._m01 * b[1, 0] + a._m02 * b[2, 0],
		a._m00 * b[0, 1] + a._m01 * b[1, 1] + a._m02 * b[2, 1],
		a._m10 * b[0, 0] + a._m11 * b[1, 0] + a._m12 * b[2, 0],
		a._m10 * b[0, 1] + a._m11 * b[1, 1] + a._m12 * b[2, 1]);

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Matrix2x3d a, Matrix2x3d b) =>
		a._m00 == b._m00 && a._m01 == b._m01 && a._m02 == b._m02
		&& a._m10 == b._m10 && a._m11 == b._m11 && a._m12 == b._m12;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Matrix2x3d a, Matrix2x3d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix2x3d other) =>
		_m00.Equals(other._m00) && _m01.Equals(other._m01) && _m02.Equals(other._m02)
		&& _m10.Equals(other._m10) && _m11.Equals(other._m11) && _m12.Equals(other._m12);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix2x3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(_m00, _m10, _m01, _m11, _m02, _m12);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"[{Row(0)}, {Row(1)}]");
}
