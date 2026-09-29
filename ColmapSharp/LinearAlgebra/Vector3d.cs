// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Vector3d: fixed-size 3-vector of doubles, the replacement for Eigen::Vector3d (points,
// translations, rays, normals). Written here to Eigen's documented semantics; Eigen
// (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Siblings: Vector2d, Vector4d, the
// matrices, Quaterniond and AngleAxisd in this folder. Tests:
// ColmapSharp.Tests/LinearAlgebra/VectorTests.cs.
//
// Arithmetic order contract for the whole LinearAlgebra folder. Tier A code downstream
// (pose/transform algebra, camera models) needs every sum to round the same way on every
// platform, so:
// - Dot products, norms and matrix products are written out as plain left-to-right sums
//   starting from the first term, a0*b0 + a1*b1 + a2*b2, never seeded with 0.0 (that
//   would turn a -0.0 result into +0.0).
// - No FMA (Math.FusedMultiplyAdd) and no System.Numerics.Vector<T>.
// Exception: where Eigen vectorizes a reduction or a product it pairs terms differently,
// and this folder follows what the pycolmap oracle shows rather than left to right.
// Vector4d's reductions and Quaterniond's product are such cases; each says so and cites
// its evidence (oracle/linear_algebra_rotations.py). The 3-vector norm here is plain
// left to right, which the same oracle confirms through AngleAxisd's angle.
// The macOS arm64 pycolmap wheel also contracts some a*b - c*d into an FMA (Cross, for
// one); we do not, see divergence 6.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3-vector of doubles. Replacement for Eigen::Vector3d.
/// </summary>
public readonly struct Vector3d : IEquatable<Vector3d>
{
	/// <summary>First coefficient, Eigen's x() / [0].</summary>
	public readonly double X;

	/// <summary>Second coefficient, Eigen's y() / [1].</summary>
	public readonly double Y;

	/// <summary>Third coefficient, Eigen's z() / [2].</summary>
	public readonly double Z;

	/// <summary>Creates the vector (x, y, z).</summary>
	public Vector3d(double x, double y, double z)
	{
		X = x;
		Y = y;
		Z = z;
	}

	/// <summary>The zero vector, Eigen's Vector3d::Zero().</summary>
	public static Vector3d Zero => default;

	/// <summary>Eigen's Vector3d::Ones().</summary>
	public static Vector3d Ones => new(1, 1, 1);

	/// <summary>(1, 0, 0), Eigen's UnitX().</summary>
	public static Vector3d UnitX => new(1, 0, 0);

	/// <summary>(0, 1, 0), Eigen's UnitY().</summary>
	public static Vector3d UnitY => new(0, 1, 0);

	/// <summary>(0, 0, 1), Eigen's UnitZ().</summary>
	public static Vector3d UnitZ => new(0, 0, 1);

	/// <summary>Coefficient i (0, 1 or 2).</summary>
	public double this[int i] => i switch
	{
		0 => X,
		1 => Y,
		2 => Z,
		_ => throw new ArgumentOutOfRangeException(nameof(i)),
	};

	/// <summary>Squared Euclidean norm, x*x + y*y + z*z.</summary>
	public double SquaredNorm => X * X + Y * Y + Z * Z;

	/// <summary>Euclidean norm, sqrt(SquaredNorm).</summary>
	public double Norm => Math.Sqrt(SquaredNorm);

	/// <summary>Dot product, x*o.x + y*o.y + z*o.z.</summary>
	public double Dot(Vector3d other) => X * other.X + Y * other.Y + Z * other.Z;

	/// <summary>Cross product this x other.</summary>
	public Vector3d Cross(Vector3d other)
	{
		return new Vector3d(
			Y * other.Z - Z * other.Y,
			Z * other.X - X * other.Z,
			X * other.Y - Y * other.X);
	}

	/// <summary>
	/// This vector divided by its norm. Like Eigen, a zero vector is returned unchanged
	/// rather than turned into NaN.
	/// </summary>
	public Vector3d Normalized()
	{
		double squaredNorm = SquaredNorm;
		if (squaredNorm > 0)
		{
			return this / Math.Sqrt(squaredNorm);
		}

		return this;
	}

	/// <summary>(x, y, z, 1), Eigen's homogeneous().</summary>
	public Vector4d Homogeneous() => new(X, Y, Z, 1);

	/// <summary>(x / z, y / z), Eigen's hnormalized().</summary>
	public Vector2d HNormalized() => new(X / Z, Y / Z);

	/// <summary>(x, y), Eigen's head&lt;2&gt;().</summary>
	public Vector2d Head2() => new(X, Y);

	/// <summary>Coefficient-wise product, Eigen's cwiseProduct.</summary>
	public Vector3d CwiseProduct(Vector3d other) => new(X * other.X, Y * other.Y, Z * other.Z);

	/// <summary>Coefficient-wise absolute value, Eigen's cwiseAbs.</summary>
	public Vector3d CwiseAbs() => new(Math.Abs(X), Math.Abs(Y), Math.Abs(Z));

	/// <summary>
	/// Eigen's isApprox: ||a - b|| &lt;= precision * min(||a||, ||b||). Note that this
	/// is relative, so nothing but an exact zero is approximately equal to zero.
	/// </summary>
	public bool IsApprox(Vector3d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return (this - other).Norm <= precision * Math.Min(Norm, other.Norm);
	}

	/// <summary>Vector sum.</summary>
	public static Vector3d operator +(Vector3d a, Vector3d b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

	/// <summary>Vector difference.</summary>
	public static Vector3d operator -(Vector3d a, Vector3d b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

	/// <summary>Negation.</summary>
	public static Vector3d operator -(Vector3d a) => new(-a.X, -a.Y, -a.Z);

	/// <summary>Scalar product.</summary>
	public static Vector3d operator *(Vector3d a, double s) => new(a.X * s, a.Y * s, a.Z * s);

	/// <summary>Scalar product.</summary>
	public static Vector3d operator *(double s, Vector3d a) => new(s * a.X, s * a.Y, s * a.Z);

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Vector3d operator /(Vector3d a, double s) => new(a.X / s, a.Y / s, a.Z / s);

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Vector3d a, Vector3d b) => a.X == b.X && a.Y == b.Y && a.Z == b.Z;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Vector3d a, Vector3d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Vector3d other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Vector3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(X, Y, Z);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"({X:R}, {Y:R}, {Z:R})");
}
