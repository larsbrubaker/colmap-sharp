// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/jet.h (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Jet<TGrad>: ceres::Jet<double, N>, a dual number a + sum_i v_i e_i with e_i e_j = 0, for
// forward-mode automatic differentiation in N = TGrad.Length variables (JetGradients.cs).
// It implements IScalar (Scalar.cs), so the camera models (Sensor/), the rotation helpers
// (Rotation.cs) and every cost functor evaluated by AutoDiffCostFunction differentiate
// through it. IterativeUndistortion uses Jet<Grad2>, which replaced the hand-written Jet2.
// The rules follow Ceres 2.2.0, the version the pycolmap 4.2.0 wheel links (its binary
// reports "2.2.0-eigen-(3.5.0)-lapack-suitesparse-(7.13.0)-acceleratesparse"), compiled as
// C++17 (COLMAP's CMAKE_CXX_STANDARD), which picks fmax/fmin's (x + y) * 0.5 tie rule over
// C++20's midpoint.
//
// Each rule keeps Ceres' own grouping, because Jets feed Tier A code (IterativeUndistortion's
// Newton steps): the quotient is f.a * (1 / g.a), sqrt divides by 2 * sqrt(a) through a
// reciprocal, and so on. Every derivative component is computed from the value parts and
// that component alone, with no FMA (CLAUDE.md), which is what lets AutoDiffCostFunction
// evaluate a wide gradient in narrower chunks and still get the same bits.
// Performance: results are built in place (Unsafe.SkipInit on the result Jet, then writes
// through its fields) and the slot loops walk refs with Unsafe.Add up to the JIT-time
// constant TGrad.Length, so there is no zeroing pass, no copy of the gradient into the
// result and no bounds check in the arithmetic. No SIMD: Vector128 would change nothing
// numerically, but it is deferred until the scalar path is profiled in bundle adjustment.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::Jet&lt;double, N&gt;: a value and its gradient with respect to
/// N = <c>TGrad.Length</c> variables.
/// </summary>
public readonly struct Jet<TGrad> : IScalar<Jet<TGrad>>
	where TGrad : unmanaged, IJetGradient
{
	/// <summary>The value part, Ceres' <c>a</c>.</summary>
	public readonly double A;

	/// <summary>The derivative part, Ceres' <c>v</c>.</summary>
	public readonly TGrad V;

	/// <summary>Creates a + v.</summary>
	public Jet(double a, in TGrad v)
	{
		A = a;
		V = v;
	}

	/// <summary>The number of derivative slots, Ceres' <c>N</c>.</summary>
	public static int Size => TGrad.Length;

	/// <summary>Ceres' <c>Jet(value, k)</c>: the k-th variable at the given value.</summary>
	public static Jet<TGrad> Variable(double value, int k)
	{
		if ((uint)k >= (uint)TGrad.Length)
		{
			throw new ArgumentOutOfRangeException(nameof(k));
		}

		TGrad v = default;
		Unsafe.Add(ref Slot0(in v), k) = 1.0;
		return new Jet<TGrad>(value, v);
	}

	/// <summary>Creates a Jet from a value and explicit derivative components.</summary>
	public static Jet<TGrad> Create(double a, ReadOnlySpan<double> derivatives)
	{
		Check.Eq(derivatives.Length, TGrad.Length);
		TGrad v = default;
		derivatives.CopyTo(WritableSlots(ref v));
		return new Jet<TGrad>(a, v);
	}

	/// <summary>The derivative with respect to variable i, Ceres' <c>v[i]</c>.</summary>
	public double Derivative(int i)
	{
		if ((uint)i >= (uint)TGrad.Length)
		{
			throw new ArgumentOutOfRangeException(nameof(i));
		}

		return Unsafe.Add(ref Slot0(in V), i);
	}

	/// <summary>The derivative part as a span over the Jet's own storage.</summary>
	public static ReadOnlySpan<double> Gradient(in Jet<TGrad> jet) => Slots(in jet.V);

	// The first derivative slot, for loops that walk the slots with Unsafe.Add up to the
	// JIT-time constant TGrad.Length: no span construction and no bounds checks in the
	// arithmetic, which is where autodiff spends its time. Writing through it is only done
	// on fresh locals.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ref double Slot0(in TGrad v) => ref Unsafe.As<TGrad, double>(ref Unsafe.AsRef(in v));

	private static Span<double> WritableSlots(ref TGrad v) =>
		MemoryMarshal.CreateSpan(ref Unsafe.As<TGrad, double>(ref v), TGrad.Length);

	private static ReadOnlySpan<double> Slots(in TGrad v) =>
		MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<TGrad, double>(ref Unsafe.AsRef(in v)), TGrad.Length);

	// a + s * v, the shape of most unary rules.
	private static Jet<TGrad> Scaled(double a, double s, in TGrad v)
	{
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in v);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = s * Unsafe.Add(ref x, i);
		}

		Unsafe.AsRef(in result.A) = a;
		return result;
	}

	// a - v: a true negation of every slot, as Eigen's unary minus.
	private static Jet<TGrad> Negated(double a, in TGrad v)
	{
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in v);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = -Unsafe.Add(ref x, i);
		}

		Unsafe.AsRef(in result.A) = a;
		return result;
	}

	// a + (s0 * v0 + s1 * v1).
	private static Jet<TGrad> Combined(double a, double s0, in TGrad v0, double s1, in TGrad v1)
	{
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in v0);
		ref double y = ref Slot0(in v1);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = s0 * Unsafe.Add(ref x, i) + s1 * Unsafe.Add(ref y, i);
		}

		Unsafe.AsRef(in result.A) = a;
		return result;
	}

	/// <inheritdoc/>
	public static Jet<TGrad> FromDouble(double value) => new(value, default);

	/// <inheritdoc/>
	public static double ScalarPart(in Jet<TGrad> a) => a.A;

	/// <inheritdoc/>
	public static Jet<TGrad> operator +(in Jet<TGrad> f, in Jet<TGrad> g)
	{
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in f.V);
		ref double y = ref Slot0(in g.V);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = Unsafe.Add(ref x, i) + Unsafe.Add(ref y, i);
		}

		Unsafe.AsRef(in result.A) = f.A + g.A;
		return result;
	}

	/// <summary>Jet plus a double: (a + s, v).</summary>
	public static Jet<TGrad> operator +(in Jet<TGrad> f, double s) => new(f.A + s, f.V);

	/// <summary>Double plus a Jet: (a + s, v).</summary>
	public static Jet<TGrad> operator +(double s, in Jet<TGrad> f) => new(f.A + s, f.V);

	/// <inheritdoc/>
	public static Jet<TGrad> operator -(in Jet<TGrad> f, in Jet<TGrad> g)
	{
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in f.V);
		ref double y = ref Slot0(in g.V);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = Unsafe.Add(ref x, i) - Unsafe.Add(ref y, i);
		}

		Unsafe.AsRef(in result.A) = f.A - g.A;
		return result;
	}

	/// <summary>Jet minus a double: (a - s, v).</summary>
	public static Jet<TGrad> operator -(in Jet<TGrad> f, double s) => new(f.A - s, f.V);

	/// <summary>
	/// C++ <c>1.0 - alpha</c>: Ceres gives (s - a, -v), a true negation, so a +0.0 derivative
	/// becomes -0.0 (Jet(s) - alpha would compute 0 - v = +0.0).
	/// </summary>
	public static Jet<TGrad> operator -(double s, in Jet<TGrad> f) => Negated(s - f.A, f.V);

	/// <summary>Negation: (-a, -v).</summary>
	public static Jet<TGrad> operator -(in Jet<TGrad> f) => Negated(-f.A, f.V);

	/// <summary>Product: (f.a g.a, f.a g.v + f.v g.a).</summary>
	public static Jet<TGrad> operator *(in Jet<TGrad> f, in Jet<TGrad> g) =>
		Combined(f.A * g.A, f.A, g.V, g.A, f.V);

	/// <summary>Jet times a double: (a s, v s).</summary>
	public static Jet<TGrad> operator *(in Jet<TGrad> f, double s) => Scaled(f.A * s, s, f.V);

	/// <summary>Double times a Jet: (a s, v s).</summary>
	public static Jet<TGrad> operator *(double s, in Jet<TGrad> f) => Scaled(f.A * s, s, f.V);

	/// <summary>
	/// Quotient. Ceres uses (a + u)/(b + v) = (a + u)(b - v)/b^2, which holds because
	/// v*v = 0, and evaluates it through the reciprocal of g.a.
	/// </summary>
	public static Jet<TGrad> operator /(in Jet<TGrad> f, in Jet<TGrad> g)
	{
		double gAInverse = 1.0 / g.A;
		double fAByGA = f.A * gAInverse;
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double x = ref Slot0(in f.V);
		ref double y = ref Slot0(in g.V);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = (Unsafe.Add(ref x, i) - fAByGA * Unsafe.Add(ref y, i)) * gAInverse;
		}

		Unsafe.AsRef(in result.A) = fAByGA;
		return result;
	}

	/// <summary>Jet over a double: through the reciprocal, (a / s, v / s) = (a s^-1, v s^-1).</summary>
	public static Jet<TGrad> operator /(in Jet<TGrad> f, double s)
	{
		double sInverse = 1.0 / s;
		return Scaled(f.A * sInverse, sInverse, f.V);
	}

	/// <summary>Double over a Jet: (s / g.a, g.v * (-s / g.a^2)).</summary>
	public static Jet<TGrad> operator /(double s, in Jet<TGrad> g)
	{
		double minusSGAInverse2 = -s / (g.A * g.A);
		return Scaled(s / g.A, minusSGAInverse2, g.V);
	}

	/// <inheritdoc/>
	public static bool operator <(in Jet<TGrad> f, in Jet<TGrad> g) => f.A < g.A;

	/// <inheritdoc/>
	public static bool operator >(in Jet<TGrad> f, in Jet<TGrad> g) => f.A > g.A;

	/// <inheritdoc/>
	public static bool operator <=(in Jet<TGrad> f, in Jet<TGrad> g) => f.A <= g.A;

	/// <inheritdoc/>
	public static bool operator >=(in Jet<TGrad> f, in Jet<TGrad> g) => f.A >= g.A;

	/// <summary>sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a)).</summary>
	public static Jet<TGrad> Sqrt(in Jet<TGrad> f)
	{
		double tmp = Math.Sqrt(f.A);
		double twoAInverse = 1.0 / (2.0 * tmp);
		return Scaled(tmp, twoAInverse, f.V);
	}

	/// <summary>
	/// abs(a + h) ~= abs(a) + sgn(a) h, with Ceres 2.1+'s sgn = copysign(1, a): so at
	/// a = -0.0 the value is +0.0 and the derivative is -h (older Ceres returned f itself).
	/// </summary>
	public static Jet<TGrad> Abs(in Jet<TGrad> f) => Scaled(Math.Abs(f.A), Math.CopySign(1.0, f.A), f.V);

	/// <summary>sin(a + h) ~= sin(a) + cos(a) h.</summary>
	public static Jet<TGrad> Sin(in Jet<TGrad> f) => Scaled(Math.Sin(f.A), Math.Cos(f.A), f.V);

	/// <summary>cos(a + h) ~= cos(a) - sin(a) h.</summary>
	public static Jet<TGrad> Cos(in Jet<TGrad> f) => Scaled(Math.Cos(f.A), -Math.Sin(f.A), f.V);

	/// <summary>tan(a + h) ~= tan(a) + (1 + tan(a)^2) h.</summary>
	public static Jet<TGrad> Tan(in Jet<TGrad> f)
	{
		double tanA = Math.Tan(f.A);
		double tmp = 1.0 + tanA * tanA;
		return Scaled(tanA, tmp, f.V);
	}

	/// <summary>asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h.</summary>
	public static Jet<TGrad> Asin(in Jet<TGrad> f)
	{
		double tmp = 1.0 / Math.Sqrt(1.0 - f.A * f.A);
		return Scaled(Math.Asin(f.A), tmp, f.V);
	}

	/// <summary>acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h.</summary>
	public static Jet<TGrad> Acos(in Jet<TGrad> f)
	{
		double tmp = -1.0 / Math.Sqrt(1.0 - f.A * f.A);
		return Scaled(Math.Acos(f.A), tmp, f.V);
	}

	/// <summary>atan(a + h) ~= atan(a) + 1 / (1 + a^2) h.</summary>
	public static Jet<TGrad> Atan(in Jet<TGrad> f)
	{
		double tmp = 1.0 / (1.0 + f.A * f.A);
		return Scaled(Math.Atan(f.A), tmp, f.V);
	}

	/// <summary>
	/// atan2(g + dg, f + df) ~= atan2(g, f) + (f dg - g df) / (f^2 + g^2), with Ceres'
	/// argument order: the first argument is y. Ceres evaluates tmp * (-g.a * f.v + f.a * g.v).
	/// </summary>
	public static Jet<TGrad> Atan2(in Jet<TGrad> g, in Jet<TGrad> f)
	{
		double tmp = 1.0 / (f.A * f.A + g.A * g.A);
		double minusGA = -g.A;
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double fv = ref Slot0(in f.V);
		ref double gv = ref Slot0(in g.V);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = tmp * (minusGA * Unsafe.Add(ref fv, i) + f.A * Unsafe.Add(ref gv, i));
		}

		Unsafe.AsRef(in result.A) = Math.Atan2(g.A, f.A);
		return result;
	}

	/// <summary>exp(a + h) ~= exp(a) + exp(a) h.</summary>
	public static Jet<TGrad> Exp(in Jet<TGrad> f)
	{
		double tmp = Math.Exp(f.A);
		return Scaled(tmp, tmp, f.V);
	}

	/// <summary>log(a + h) ~= log(a) + h / a, through the reciprocal of a.</summary>
	public static Jet<TGrad> Log(in Jet<TGrad> f)
	{
		double aInverse = 1.0 / f.A;
		return Scaled(Math.Log(f.A), aInverse, f.V);
	}

	/// <summary>(a + da)^p ~= a^p + p a^(p - 1) da, for a constant exponent.</summary>
	public static Jet<TGrad> Pow(in Jet<TGrad> f, double g)
	{
		double tmp = g * Math.Pow(f.A, g - 1.0);
		return Scaled(Math.Pow(f.A, g), tmp, f.V);
	}

	/// <summary>
	/// pow with both base and exponent differentiable, with Ceres' special cases: f == 0 and
	/// g &gt; 1 gives zero; f == 0 and g == 1 gives f; f &lt; 0 with an integral g
	/// differentiates the base only and makes every slot where g has a derivative NaN
	/// (moving g leaves the reals); everything else uses
	/// f^g + g f^(g - 1) df + f^g log(f) dg and lets log produce the non-finite cases.
	/// </summary>
	public static Jet<TGrad> Pow(in Jet<TGrad> f, in Jet<TGrad> g)
	{
		if (f.A == 0.0 && g.A >= 1)
		{
			return g.A > 1 ? FromDouble(0.0) : f;
		}

		if (f.A < 0 && g.A == Math.Floor(g.A))
		{
			double tmp = g.A * Math.Pow(f.A, g.A - 1.0);
			Jet<TGrad> result = Scaled(Math.Pow(f.A, g.A), tmp, f.V);
			TGrad v = result.V;
			ref double r = ref Slot0(in v);
			ref double gv = ref Slot0(in g.V);
			for (int i = 0; i < TGrad.Length; i++)
			{
				if (Unsafe.Add(ref gv, i) != 0.0)
				{
					Unsafe.Add(ref r, i) = double.NaN;
				}
			}

			return new Jet<TGrad>(result.A, v);
		}

		double tmp1 = Math.Pow(f.A, g.A);
		double tmp2 = g.A * Math.Pow(f.A, g.A - 1.0);
		double tmp3 = tmp1 * Math.Log(f.A);
		return Combined(tmp1, tmp2, f.V, tmp3, g.V);
	}

	/// <summary>floor: piecewise constant, so the derivative is zero.</summary>
	public static Jet<TGrad> Floor(in Jet<TGrad> f) => FromDouble(Math.Floor(f.A));

	/// <summary>
	/// hypot(x, y) = sqrt(x^2 + y^2) without intermediate overflow; derivative
	/// x / hypot dx + y / hypot dy.
	/// </summary>
	public static Jet<TGrad> Hypot(in Jet<TGrad> x, in Jet<TGrad> y)
	{
		double tmp = double.Hypot(x.A, y.A);
		return Combined(tmp, x.A / tmp, x.V, y.A / tmp, y.V);
	}

	/// <summary>
	/// hypot(x, y, z); derivative x / hypot dx + y / hypot dy + z / hypot dz, summed left
	/// to right as Eigen's expression evaluates it.
	/// </summary>
	public static Jet<TGrad> Hypot(in Jet<TGrad> x, in Jet<TGrad> y, in Jet<TGrad> z)
	{
		double tmp = ScalarMath.Hypot(x.A, y.A, z.A);
		double sx = x.A / tmp;
		double sy = y.A / tmp;
		double sz = z.A / tmp;
		Unsafe.SkipInit(out Jet<TGrad> result);
		ref double r = ref Slot0(in result.V);
		ref double xv = ref Slot0(in x.V);
		ref double yv = ref Slot0(in y.V);
		ref double zv = ref Slot0(in z.V);
		for (int i = 0; i < TGrad.Length; i++)
		{
			Unsafe.Add(ref r, i) = sx * Unsafe.Add(ref xv, i) + sy * Unsafe.Add(ref yv, i) + sz * Unsafe.Add(ref zv, i);
		}

		Unsafe.AsRef(in result.A) = tmp;
		return result;
	}

	/// <summary>
	/// ceres::fmin: returns the smaller by value, the other one when either is NaN, and on a
	/// value tie the average (x + y) * 0.5 of the two Jets, so the result keeps a
	/// derivative and does not depend on argument order.
	/// </summary>
	public static Jet<TGrad> Min(in Jet<TGrad> x, in Jet<TGrad> y)
	{
		if (double.IsNaN(x.A) || double.IsNaN(y.A) || x.A < y.A || x.A > y.A)
		{
			return double.IsNaN(x.A) || x.A > y.A ? y : x;
		}

		return (x + y) * 0.5;
	}

	/// <summary>ceres::fmax, with the same NaN and tie rules as <see cref="Min"/>.</summary>
	public static Jet<TGrad> Max(in Jet<TGrad> x, in Jet<TGrad> y)
	{
		if (double.IsNaN(x.A) || double.IsNaN(y.A) || x.A < y.A || x.A > y.A)
		{
			return double.IsNaN(x.A) || x.A < y.A ? y : x;
		}

		return (x + y) * 0.5;
	}

	/// <inheritdoc/>
	public static bool IsFinite(in Jet<TGrad> f) => double.IsFinite(f.A);

	/// <inheritdoc/>
	public static bool IsNaN(in Jet<TGrad> f) => double.IsNaN(f.A);

	/// <summary>Ceres' stream format, <c>[a ; v0, v1, ...]</c>.</summary>
	public override string ToString()
	{
		var builder = new StringBuilder();
		builder.Append(CultureInfo.InvariantCulture, $"[{A} ; ");
		ReadOnlySpan<double> v = Slots(in V);
		for (int i = 0; i < v.Length; i++)
		{
			builder.Append(CultureInfo.InvariantCulture, $"{v[i]}");
			if (i != v.Length - 1)
			{
				builder.Append(", ");
			}
		}

		builder.Append(']');
		return builder.ToString();
	}
}
