// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Jet2: ceres::Jet<double, 2> (ceres/jet.h), a dual number a + v0*e0 + v1*e1 with
// e_i*e_j = 0, for forward-mode differentiation in two variables. Its first user is the
// camera models' IterativeUndistortion (Sensor/IterativeUndistortion.cs), which gets the
// 2x2 Jacobian of a model's distortion by evaluating it on Jet2, exactly as COLMAP does.
// Phase 7 brings the general Jet<N>; this one stays as the N = 2 case or folds into it.
// The rules follow Ceres 2.2.0, the version the pycolmap 4.2.0 wheel links (its binary
// reports "2.2.0-eigen-(3.5.0)-lapack-suitesparse-(7.13.0)-acceleratesparse"); COLMAP's
// CMake sets no minimum.
//
// Each rule is written with Ceres' own grouping, because IterativeUndistortion is Tier A
// (bit-exact) and the Jacobian feeds its Newton steps: in particular the quotient is
// f.a * (1 / g.a), not f.a / g.a, and sqrt divides by 2 * sqrt(a) through a reciprocal.
// The derivative part is computed per component with no FMA (CLAUDE.md).

using System.Globalization;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::Jet&lt;double, 2&gt;: a value and its gradient with respect to two variables.
/// </summary>
public readonly struct Jet2 : IScalar<Jet2>
{
	/// <summary>The value part, Ceres' <c>a</c>.</summary>
	public readonly double A;

	/// <summary>Derivative with respect to variable 0, Ceres' <c>v[0]</c>.</summary>
	public readonly double V0;

	/// <summary>Derivative with respect to variable 1, Ceres' <c>v[1]</c>.</summary>
	public readonly double V1;

	/// <summary>Creates a + v0*e0 + v1*e1.</summary>
	public Jet2(double a, double v0, double v1)
	{
		A = a;
		V0 = v0;
		V1 = v1;
	}

	/// <summary>Ceres' <c>Jet(value, k)</c>: the variable k (0 or 1) at the given value.</summary>
	public static Jet2 Variable(double value, int k) => k switch
	{
		0 => new Jet2(value, 1, 0),
		1 => new Jet2(value, 0, 1),
		_ => throw new ArgumentOutOfRangeException(nameof(k)),
	};

	/// <inheritdoc/>
	public static Jet2 FromDouble(double value) => new(value, 0, 0);

	/// <inheritdoc/>
	public static Jet2 operator +(Jet2 f, Jet2 g) => new(f.A + g.A, f.V0 + g.V0, f.V1 + g.V1);

	/// <inheritdoc/>
	public static Jet2 operator -(Jet2 f, Jet2 g) => new(f.A - g.A, f.V0 - g.V0, f.V1 - g.V1);

	/// <inheritdoc/>
	public static Jet2 operator *(Jet2 f, Jet2 g) =>
		new(f.A * g.A, f.A * g.V0 + f.V0 * g.A, f.A * g.V1 + f.V1 * g.A);

	/// <summary>
	/// Quotient. Ceres uses (a + u)/(b + v) = (a + u)(b - v)/b^2, which holds because
	/// v*v = 0, and evaluates it through the reciprocal of g.a.
	/// </summary>
	public static Jet2 operator /(Jet2 f, Jet2 g)
	{
		double gAInverse = 1.0 / g.A;
		double fAByGA = f.A * gAInverse;
		return new Jet2(fAByGA, (f.V0 - fAByGA * g.V0) * gAInverse, (f.V1 - fAByGA * g.V1) * gAInverse);
	}

	/// <inheritdoc/>
	public static Jet2 operator -(Jet2 f) => new(-f.A, -f.V0, -f.V1);

	/// <inheritdoc/>
	public static Jet2 operator -(double s, Jet2 f) => new(s - f.A, -f.V0, -f.V1);

	/// <inheritdoc/>
	public static bool operator <(Jet2 f, Jet2 g) => f.A < g.A;

	/// <inheritdoc/>
	public static bool operator >(Jet2 f, Jet2 g) => f.A > g.A;

	/// <inheritdoc/>
	public static bool operator <=(Jet2 f, Jet2 g) => f.A <= g.A;

	/// <inheritdoc/>
	public static bool operator >=(Jet2 f, Jet2 g) => f.A >= g.A;

	/// <summary>sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a)).</summary>
	public static Jet2 Sqrt(Jet2 f)
	{
		double tmp = Math.Sqrt(f.A);
		double twoAInverse = 1.0 / (2.0 * tmp);
		return new Jet2(tmp, f.V0 * twoAInverse, f.V1 * twoAInverse);
	}

	/// <summary>
	/// abs(a + h) ~= abs(a) + sgn(a) h, with Ceres 2.1+'s sgn = copysign(1, a): so at
	/// a = -0.0 the value is +0.0 and the derivative is -h (older Ceres returned f itself).
	/// </summary>
	public static Jet2 Abs(Jet2 f)
	{
		double sign = Math.CopySign(1.0, f.A);
		return new Jet2(Math.Abs(f.A), sign * f.V0, sign * f.V1);
	}

	/// <summary>sin(a + h) ~= sin(a) + cos(a) h.</summary>
	public static Jet2 Sin(Jet2 f)
	{
		double cosA = Math.Cos(f.A);
		return new Jet2(Math.Sin(f.A), cosA * f.V0, cosA * f.V1);
	}

	/// <summary>cos(a + h) ~= cos(a) - sin(a) h.</summary>
	public static Jet2 Cos(Jet2 f)
	{
		double minusSinA = -Math.Sin(f.A);
		return new Jet2(Math.Cos(f.A), minusSinA * f.V0, minusSinA * f.V1);
	}

	/// <summary>tan(a + h) ~= tan(a) + (1 + tan(a)^2) h.</summary>
	public static Jet2 Tan(Jet2 f)
	{
		double tanA = Math.Tan(f.A);
		double tmp = 1.0 + tanA * tanA;
		return new Jet2(tanA, tmp * f.V0, tmp * f.V1);
	}

	/// <summary>atan(a + h) ~= atan(a) + 1 / (1 + a^2) h.</summary>
	public static Jet2 Atan(Jet2 f)
	{
		double tmp = 1.0 / (1.0 + f.A * f.A);
		return new Jet2(Math.Atan(f.A), tmp * f.V0, tmp * f.V1);
	}

	/// <summary>
	/// atan2(g + dg, f + df) ~= atan2(g, f) + (f dg - g df) / (f^2 + g^2), with Ceres'
	/// argument order: the first argument is y.
	/// </summary>
	public static Jet2 Atan2(Jet2 g, Jet2 f)
	{
		double tmp = 1.0 / (f.A * f.A + g.A * g.A);
		return new Jet2(
			Math.Atan2(g.A, f.A),
			tmp * (-g.A * f.V0 + f.A * g.V0),
			tmp * (-g.A * f.V1 + f.A * g.V1));
	}

	/// <inheritdoc/>
	public override string ToString() =>
		string.Create(CultureInfo.InvariantCulture, $"[{A} ; {V0}, {V1}]");
}
