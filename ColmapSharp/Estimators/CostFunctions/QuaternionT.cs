// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// QuaternionT<T> and Vector3T<T>: the scalar-generic stand-ins for Eigen::Quaternion<T> and
// Eigen::Matrix<T, 3, 1> that COLMAP's autodiff cost functors (ReprojectionError.cs,
// PosePriorCostFunctions.cs, AlignmentCostFunctions.cs, MotionAveragingCostFunctions.cs)
// build on through EigenQuaternionMap<T> / EigenVector3Map<T>. Not a port of a COLMAP file:
// Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md); these are written from the
// published algorithms to Eigen's documented semantics, as LinearAlgebra/Quaterniond.cs is.
//
// Why value types over T instead of spans: IAutoDiffFunctor only constrains T to a struct,
// so a functor cannot stackalloc a span of T; small readonly structs keep every
// intermediate in registers or on the stack, with no allocation per Evaluate.
//
// Conventions (as Quaterniond.cs): coefficients stored (x, y, z, w), Hamilton product,
// rotation of v as q v q* assuming a unit q.
// - The product is the textbook four-term sum per coefficient, left to right. Eigen uses
//   that form for Jets; for plain doubles it pairs the terms for SIMD (Quaterniond's
//   operator *), so on the residual-only path the two may differ in the last ulp (Tier B,
//   divergence 115).
// - The rotation is t = 2 (u x v), v' = v + w t + u x t (F. Giesen, "Rotating a vector by a
//   unit quaternion", 2015), the form Quaterniond uses, with t formed as uv + uv.
// - Inverse is conjugate / squared norm (0 for the zero quaternion), the squared norm summed
//   in pairs (x^2 + y^2) + (z^2 + w^2) as Vector4d does.
// Constant (double) operands meet T through IScalar's mixed operators, which give the same
// value and derivative as a constant Jet in C++'s `.cast<T>()` up to the sign of a zero.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>A 3-vector over a scalar type T: Eigen::Matrix&lt;T, 3, 1&gt;.</summary>
public readonly struct Vector3T<T>(T x, T y, T z)
	where T : struct, IScalar<T>
{
	/// <summary>Components.</summary>
	public readonly T X = x, Y = y, Z = z;

	/// <summary>EigenVector3Map: the three values at the start of <paramref name="values"/>.</summary>
	public static Vector3T<T> Map(ReadOnlySpan<T> values) => new(values[0], values[1], values[2]);

	/// <summary>A constant vector, C++'s <c>v.cast&lt;T&gt;()</c>.</summary>
	public static Vector3T<T> From(Vector3d v) => new(T.FromDouble(v.X), T.FromDouble(v.Y), T.FromDouble(v.Z));

	/// <summary>Sum.</summary>
	public static Vector3T<T> operator +(in Vector3T<T> a, in Vector3T<T> b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

	/// <summary>Sum with a constant vector.</summary>
	public static Vector3T<T> operator +(in Vector3T<T> a, Vector3d b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

	/// <summary>Sum with a constant vector.</summary>
	public static Vector3T<T> operator +(Vector3d a, in Vector3T<T> b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

	/// <summary>Difference.</summary>
	public static Vector3T<T> operator -(in Vector3T<T> a, in Vector3T<T> b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

	/// <summary>Difference with a constant vector.</summary>
	public static Vector3T<T> operator -(in Vector3T<T> a, Vector3d b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

	/// <summary>Difference from a constant vector.</summary>
	public static Vector3T<T> operator -(Vector3d a, in Vector3T<T> b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

	/// <summary>Scaling by a scalar.</summary>
	public static Vector3T<T> operator *(in Vector3T<T> a, in T s) => new(a.X * s, a.Y * s, a.Z * s);

	/// <summary>Scaling by a scalar.</summary>
	public static Vector3T<T> operator *(in T s, in Vector3T<T> a) => new(s * a.X, s * a.Y, s * a.Z);

	/// <summary>The cross product a x b.</summary>
	public static Vector3T<T> Cross(in Vector3T<T> a, in Vector3T<T> b) => new(
		a.Y * b.Z - a.Z * b.Y,
		a.Z * b.X - a.X * b.Z,
		a.X * b.Y - a.Y * b.X);

	/// <summary>Writes the components to the first three entries of <paramref name="destination"/>.</summary>
	public void CopyTo(Span<T> destination)
	{
		destination[0] = X;
		destination[1] = Y;
		destination[2] = Z;
	}
}

/// <summary>A quaternion over a scalar type T: Eigen::Quaternion&lt;T&gt;.</summary>
public readonly struct QuaternionT<T>(T x, T y, T z, T w)
	where T : struct, IScalar<T>
{
	/// <summary>Coefficients, stored (x, y, z, w) like Eigen.</summary>
	public readonly T X = x, Y = y, Z = z, W = w;

	/// <summary>EigenQuaternionMap: the (x, y, z, w) coefficients at the start of <paramref name="coeffs"/>.</summary>
	public static QuaternionT<T> Map(ReadOnlySpan<T> coeffs) => new(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);

	/// <summary>A constant quaternion, C++'s <c>q.cast&lt;T&gt;()</c>.</summary>
	public static QuaternionT<T> From(Quaterniond q) =>
		new(T.FromDouble(q.X), T.FromDouble(q.Y), T.FromDouble(q.Z), T.FromDouble(q.W));

	/// <summary>The vector part (x, y, z).</summary>
	public Vector3T<T> Vec => new(X, Y, Z);

	/// <summary>Hamilton product: the rotation b followed by a.</summary>
	public static QuaternionT<T> operator *(in QuaternionT<T> a, in QuaternionT<T> b) => new(
		a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
		a.W * b.Y + a.Y * b.W + a.Z * b.X - a.X * b.Z,
		a.W * b.Z + a.Z * b.W + a.X * b.Y - a.Y * b.X,
		a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z);

	/// <summary>Hamilton product with a constant quaternion on the right.</summary>
	public static QuaternionT<T> operator *(in QuaternionT<T> a, Quaterniond b) => new(
		a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
		a.W * b.Y + a.Y * b.W + a.Z * b.X - a.X * b.Z,
		a.W * b.Z + a.Z * b.W + a.X * b.Y - a.Y * b.X,
		a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z);

	/// <summary>Rotates v by this (unit) quaternion, q v q*.</summary>
	public static Vector3T<T> operator *(in QuaternionT<T> q, in Vector3T<T> v)
	{
		Vector3T<T> u = q.Vec;
		Vector3T<T> uv = Vector3T<T>.Cross(u, v);
		uv += uv;
		return v + q.W * uv + Vector3T<T>.Cross(u, uv);
	}

	/// <summary>Rotates a constant vector by this (unit) quaternion.</summary>
	public static Vector3T<T> operator *(in QuaternionT<T> q, Vector3d v)
	{
		Vector3T<T> u = q.Vec;
		var uv = new Vector3T<T>(
			q.Y * v.Z - q.Z * v.Y,
			q.Z * v.X - q.X * v.Z,
			q.X * v.Y - q.Y * v.X);
		uv += uv;
		return v + q.W * uv + Vector3T<T>.Cross(u, uv);
	}

	/// <summary>
	/// The multiplicative inverse, conjugate / squared norm; the zero quaternion for a zero
	/// input (Eigen's inverse()). The test compares the value part, as Jet's operator &gt; does.
	/// </summary>
	public QuaternionT<T> Inverse()
	{
		T n2 = (X * X + Y * Y) + (Z * Z + W * W);
		if (T.ScalarPart(n2) > 0)
		{
			return new QuaternionT<T>(-X / n2, -Y / n2, -Z / n2, W / n2);
		}

		T zero = T.FromDouble(0);
		return new QuaternionT<T>(zero, zero, zero, zero);
	}

	/// <summary>Writes the (x, y, z, w) coefficients to the start of <paramref name="destination"/>.</summary>
	public void CopyTo(Span<T> destination)
	{
		destination[0] = X;
		destination[1] = Y;
		destination[2] = Z;
		destination[3] = W;
	}
}
