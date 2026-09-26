// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix3x2d: fixed-size 3x2 matrix of doubles, the replacement for Eigen::Matrix3x2d
// (COLMAP's unprojection Jacobian d(u, v, w) / d(x, y), CamRayFromImgJacobian in
// Sensor/CameraModels.cs). Written here to Eigen's documented semantics; Eigen (MPL-2.0) is
// not ported (docs/LICENSE_AUDIT.md). Its transpose shape is Matrix2x3d.
// Also the unit-ray Jacobian of Geometry/Pose.cs CamRayWithJac (tangent Sampson error).
// Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs.
//
// Only what the ported code uses: coefficient access, columns, transpose, the Frobenius
// norm, division by a scalar, and the left product with a Matrix3d (left-to-right sums,
// no FMA, as in Matrix3d.cs).

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3x2 matrix of doubles. Replacement for Eigen::Matrix3x2d.
/// </summary>
public readonly struct Matrix3x2d : IEquatable<Matrix3x2d>
{
	private readonly double _m00;
	private readonly double _m01;
	private readonly double _m10;
	private readonly double _m11;
	private readonly double _m20;
	private readonly double _m21;

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix3x2d(double m00, double m01, double m10, double m11, double m20, double m21)
	{
		_m00 = m00;
		_m01 = m01;
		_m10 = m10;
		_m11 = m11;
		_m20 = m20;
		_m21 = m21;
	}

	/// <summary>The zero matrix, Eigen's Matrix3x2d::Zero().</summary>
	public static Matrix3x2d Zero => default;

	/// <summary>Builds a matrix whose columns are the given vectors.</summary>
	public static Matrix3x2d FromColumns(Vector3d c0, Vector3d c1) => new(c0.X, c1.X, c0.Y, c1.Y, c0.Z, c1.Z);

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col] => (row, col) switch
	{
		(0, 0) => _m00,
		(0, 1) => _m01,
		(1, 0) => _m10,
		(1, 1) => _m11,
		(2, 0) => _m20,
		(2, 1) => _m21,
		_ => throw new ArgumentOutOfRangeException(row is >= 0 and <= 2 ? nameof(col) : nameof(row)),
	};

	/// <summary>Column j as a vector.</summary>
	public Vector3d Col(int j) => new(this[0, j], this[1, j], this[2, j]);

	/// <summary>The transpose.</summary>
	public Matrix2x3d Transpose() => new(_m00, _m10, _m20, _m01, _m11, _m21);

	/// <summary>Frobenius norm, Eigen's norm() on a matrix.</summary>
	public double Norm() =>
		Math.Sqrt(_m00 * _m00 + _m10 * _m10 + _m20 * _m20 + _m01 * _m01 + _m11 * _m11 + _m21 * _m21);

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Matrix3x2d operator /(Matrix3x2d a, double s) =>
		new(a._m00 / s, a._m01 / s, a._m10 / s, a._m11 / s, a._m20 / s, a._m21 / s);

	/// <summary>The product a * b (3x3 times 3x2), column by column.</summary>
	public static Matrix3x2d operator *(in Matrix3d a, in Matrix3x2d b) => FromColumns(a * b.Col(0), a * b.Col(1));

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Matrix3x2d a, Matrix3x2d b) =>
		a._m00 == b._m00 && a._m01 == b._m01 && a._m10 == b._m10
		&& a._m11 == b._m11 && a._m20 == b._m20 && a._m21 == b._m21;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Matrix3x2d a, Matrix3x2d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix3x2d other) =>
		_m00.Equals(other._m00) && _m01.Equals(other._m01) && _m10.Equals(other._m10)
		&& _m11.Equals(other._m11) && _m20.Equals(other._m20) && _m21.Equals(other._m21);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix3x2d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(_m00, _m10, _m20, _m01, _m11, _m21);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"[{Col(0)}, {Col(1)}]^T");
}
