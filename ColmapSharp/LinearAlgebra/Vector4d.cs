// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Vector4d: fixed-size 4-vector of doubles, the replacement for Eigen::Vector4d
// (homogeneous 3D points, quaternion coefficients). Written here to Eigen's documented
// semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Siblings: Vector2d,
// Vector3d and the matrices in this folder. Tests:
// ColmapSharp.Tests/LinearAlgebra/VectorTests.cs.
//
// Arithmetic order: no FMA (the contract is spelled out in Vector3d.cs), but the two
// reductions are NOT left to right. Eigen vectorizes a 4-coefficient reduction as two
// 2-lane packets, so it sums lanes {0, 2} and {1, 3} first: (x*x + z*z) + (y*y + w*w).
// oracle/linear_algebra_rotations.py pins this through Quaterniond.Norm and Inverse:
// bit-identical on all fixture cases with this pairing; 14 and 35 of the first 130 miss with
// the left-to-right sum. Dot uses the same pairing on the assumption that it is the same
// reduction; pycolmap exposes no Eigen dot, so that part is unverified.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 4-vector of doubles. Replacement for Eigen::Vector4d.
/// </summary>
public readonly struct Vector4d : IEquatable<Vector4d>
{
	/// <summary>First coefficient, Eigen's x() / [0].</summary>
	public readonly double X;

	/// <summary>Second coefficient, Eigen's y() / [1].</summary>
	public readonly double Y;

	/// <summary>Third coefficient, Eigen's z() / [2].</summary>
	public readonly double Z;

	/// <summary>Fourth coefficient, Eigen's w() / [3].</summary>
	public readonly double W;

	/// <summary>Creates the vector (x, y, z, w).</summary>
	public Vector4d(double x, double y, double z, double w)
	{
		X = x;
		Y = y;
		Z = z;
		W = w;
	}

	/// <summary>The zero vector, Eigen's Vector4d::Zero().</summary>
	public static Vector4d Zero => default;

	/// <summary>Eigen's Vector4d::Ones().</summary>
	public static Vector4d Ones => new(1, 1, 1, 1);

	/// <summary>Coefficient i (0 to 3).</summary>
	public double this[int i] => i switch
	{
		0 => X,
		1 => Y,
		2 => Z,
		3 => W,
		_ => throw new ArgumentOutOfRangeException(nameof(i)),
	};

	/// <summary>Squared Euclidean norm, paired as Eigen's packet reduction: (x*x + z*z) + (y*y + w*w).</summary>
	public double SquaredNorm => (X * X + Z * Z) + (Y * Y + W * W);

	/// <summary>Euclidean norm, sqrt(SquaredNorm).</summary>
	public double Norm => Math.Sqrt(SquaredNorm);

	/// <summary>Dot product, paired as Eigen's packet reduction (see the file header).</summary>
	public double Dot(Vector4d other) => (X * other.X + Z * other.Z) + (Y * other.Y + W * other.W);

	/// <summary>
	/// This vector divided by its norm. Like Eigen, a zero vector is returned unchanged
	/// rather than turned into NaN.
	/// </summary>
	public Vector4d Normalized()
	{
		double squaredNorm = SquaredNorm;
		if (squaredNorm > 0)
		{
			return this / Math.Sqrt(squaredNorm);
		}

		return this;
	}

	/// <summary>(x / w, y / w, z / w), Eigen's hnormalized().</summary>
	public Vector3d HNormalized() => new(X / W, Y / W, Z / W);

	/// <summary>(x, y, z), Eigen's head&lt;3&gt;().</summary>
	public Vector3d Head3() => new(X, Y, Z);

	/// <summary>
	/// Eigen's isApprox: ||a - b|| &lt;= precision * min(||a||, ||b||). Note that this
	/// is relative, so nothing but an exact zero is approximately equal to zero.
	/// </summary>
	public bool IsApprox(Vector4d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return (this - other).Norm <= precision * Math.Min(Norm, other.Norm);
	}

	/// <summary>Vector sum.</summary>
	public static Vector4d operator +(Vector4d a, Vector4d b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);

	/// <summary>Vector difference.</summary>
	public static Vector4d operator -(Vector4d a, Vector4d b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);

	/// <summary>Negation.</summary>
	public static Vector4d operator -(Vector4d a) => new(-a.X, -a.Y, -a.Z, -a.W);

	/// <summary>Scalar product.</summary>
	public static Vector4d operator *(Vector4d a, double s) => new(a.X * s, a.Y * s, a.Z * s, a.W * s);

	/// <summary>Scalar product.</summary>
	public static Vector4d operator *(double s, Vector4d a) => new(s * a.X, s * a.Y, s * a.Z, s * a.W);

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Vector4d operator /(Vector4d a, double s) => new(a.X / s, a.Y / s, a.Z / s, a.W / s);

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Vector4d a, Vector4d b) => a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Vector4d a, Vector4d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Vector4d other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z) && W.Equals(other.W);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Vector4d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(X, Y, Z, W);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"({X:R}, {Y:R}, {Z:R}, {W:R})");
}
