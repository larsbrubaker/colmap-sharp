// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Vector2d: fixed-size 2-vector of doubles, the replacement for Eigen::Vector2d (image
// points, 2D feature coordinates). Written here to Eigen's documented semantics; Eigen
// (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Siblings: Vector3d, Vector4d, and the
// matrices in this folder. Tests: ColmapSharp.Tests/LinearAlgebra/VectorTests.cs.
//
// Arithmetic order (Tier A code downstream depends on it, see Vector3d.cs): sums are
// evaluated left to right starting from the first term, a0*b0 + a1*b1, with no FMA.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 2-vector of doubles. Replacement for Eigen::Vector2d.
/// </summary>
public readonly struct Vector2d : IEquatable<Vector2d>
{
	/// <summary>First coefficient, Eigen's x() / [0].</summary>
	public readonly double X;

	/// <summary>Second coefficient, Eigen's y() / [1].</summary>
	public readonly double Y;

	/// <summary>Creates the vector (x, y).</summary>
	public Vector2d(double x, double y)
	{
		X = x;
		Y = y;
	}

	/// <summary>The zero vector, Eigen's Vector2d::Zero().</summary>
	public static Vector2d Zero => default;

	/// <summary>Eigen's Vector2d::Ones().</summary>
	public static Vector2d Ones => new(1, 1);

	/// <summary>Coefficient i (0 or 1).</summary>
	public double this[int i] => i switch
	{
		0 => X,
		1 => Y,
		_ => throw new ArgumentOutOfRangeException(nameof(i)),
	};

	/// <summary>Squared Euclidean norm, x*x + y*y.</summary>
	public double SquaredNorm => X * X + Y * Y;

	/// <summary>Euclidean norm, sqrt(SquaredNorm).</summary>
	public double Norm => Math.Sqrt(SquaredNorm);

	/// <summary>Dot product, x*o.x + y*o.y.</summary>
	public double Dot(Vector2d other) => X * other.X + Y * other.Y;

	/// <summary>
	/// This vector divided by its norm. Like Eigen, a zero vector is returned unchanged
	/// rather than turned into NaN.
	/// </summary>
	public Vector2d Normalized()
	{
		double squaredNorm = SquaredNorm;
		if (squaredNorm > 0)
		{
			return this / Math.Sqrt(squaredNorm);
		}

		return this;
	}

	/// <summary>(x, y, 1), Eigen's homogeneous().</summary>
	public Vector3d Homogeneous() => new(X, Y, 1);

	/// <summary>Coefficient-wise product, Eigen's cwiseProduct.</summary>
	public Vector2d CwiseProduct(Vector2d other) => new(X * other.X, Y * other.Y);

	/// <summary>
	/// Eigen's isApprox: ||a - b|| &lt;= precision * min(||a||, ||b||). Note that this
	/// is relative, so nothing but an exact zero is approximately equal to zero.
	/// </summary>
	public bool IsApprox(Vector2d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return (this - other).Norm <= precision * Math.Min(Norm, other.Norm);
	}

	/// <summary>Vector sum.</summary>
	public static Vector2d operator +(Vector2d a, Vector2d b) => new(a.X + b.X, a.Y + b.Y);

	/// <summary>Vector difference.</summary>
	public static Vector2d operator -(Vector2d a, Vector2d b) => new(a.X - b.X, a.Y - b.Y);

	/// <summary>Negation.</summary>
	public static Vector2d operator -(Vector2d a) => new(-a.X, -a.Y);

	/// <summary>Scalar product.</summary>
	public static Vector2d operator *(Vector2d a, double s) => new(a.X * s, a.Y * s);

	/// <summary>Scalar product.</summary>
	public static Vector2d operator *(double s, Vector2d a) => new(s * a.X, s * a.Y);

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Vector2d operator /(Vector2d a, double s) => new(a.X / s, a.Y / s);

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(Vector2d a, Vector2d b) => a.X == b.X && a.Y == b.Y;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Vector2d a, Vector2d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Vector2d other) => X.Equals(other.X) && Y.Equals(other.Y);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Vector2d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(X, Y);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"({X:R}, {Y:R})");
}
