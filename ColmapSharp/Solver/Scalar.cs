// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IScalar<T> and Real: the numeric abstraction that COLMAP's `template <typename T>` code is
// written against here. In C++ the camera models (sensor/models.h) and every cost function
// are templates instantiated with `double` and with `ceres::Jet<double, N>`, so one body
// serves both plain evaluation and automatic differentiation. C# has no duck-typed
// templates, so that code is generic over `T : struct, IScalar<T>` instead:
// - Real is `double` behind the interface (a single-field struct the JIT keeps in a register).
//   Spans of double are reinterpreted as spans of Real with MemoryMarshal.Cast, so the double
//   path allocates nothing and rounds exactly like plain double arithmetic.
// - Jet<TGrad> (Jet.cs) is ceres::Jet<double, N>, with N carried by an [InlineArray] gradient
//   struct (JetGradients.cs). The camera models, the rotation helpers and every cost functor
//   written against IScalar differentiate without being rewritten.
// The interface holds what ceres::Jet offers COLMAP's templates: the four binary operators,
// negation, the mixed scalar/Jet operators (`t * 2.0`, `1.0 / t`, ...), comparisons on the
// value part, T(double) construction, and the elementary functions COLMAP's templates call.
// The mixed operators are not the same as `T.FromDouble(s) op x` for a Jet: Ceres gives them
// their own rules (s / g is s / g.a with derivative -s / g.a^2, f - s leaves v untouched
// where Jet(s) subtraction would compute v - 0), and those reach the last bit, so they are
// separate members here.
// Operands are `in`: a Jet with N = 15 is 128 bytes, and readonly structs passed by `in` are
// neither copied nor defensively copied; for Real the JIT inlines them away.
// It deliberately does not implement the BCL's INumberBase: a Jet cannot honestly implement
// parsing, formatting and the rest, and "no stubs" (CLAUDE.md) rules out faking them.

using System.Runtime.InteropServices;

namespace ColmapSharp.Solver;

/// <summary>
/// A scalar type COLMAP's templated numeric code can run on: <see cref="Real"/> for plain
/// double evaluation, <see cref="Jet{TGrad}"/> for forward-mode automatic differentiation.
/// </summary>
public interface IScalar<TSelf>
	where TSelf : struct, IScalar<TSelf>
{
	/// <summary>C++ <c>T(value)</c>: a constant (a Jet with zero derivative).</summary>
	static abstract TSelf FromDouble(double value);

	/// <summary>The value part: the double itself, or a Jet's <c>a</c>.</summary>
	static abstract double ScalarPart(in TSelf a);

	/// <summary>Addition.</summary>
	static abstract TSelf operator +(in TSelf a, in TSelf b);

	/// <summary>Scalar plus a plain double.</summary>
	static abstract TSelf operator +(in TSelf a, double s);

	/// <summary>A plain double plus a scalar.</summary>
	static abstract TSelf operator +(double s, in TSelf a);

	/// <summary>Subtraction.</summary>
	static abstract TSelf operator -(in TSelf a, in TSelf b);

	/// <summary>Scalar minus a plain double.</summary>
	static abstract TSelf operator -(in TSelf a, double s);

	/// <summary>A plain double minus a scalar, C++ <c>1.0 - alpha</c> with alpha a T.</summary>
	static abstract TSelf operator -(double s, in TSelf a);

	/// <summary>Multiplication.</summary>
	static abstract TSelf operator *(in TSelf a, in TSelf b);

	/// <summary>Scalar times a plain double.</summary>
	static abstract TSelf operator *(in TSelf a, double s);

	/// <summary>A plain double times a scalar.</summary>
	static abstract TSelf operator *(double s, in TSelf a);

	/// <summary>Division.</summary>
	static abstract TSelf operator /(in TSelf a, in TSelf b);

	/// <summary>Scalar divided by a plain double.</summary>
	static abstract TSelf operator /(in TSelf a, double s);

	/// <summary>A plain double divided by a scalar.</summary>
	static abstract TSelf operator /(double s, in TSelf a);

	/// <summary>Negation.</summary>
	static abstract TSelf operator -(in TSelf a);

	/// <summary>Less than, on the value part (as ceres::Jet compares).</summary>
	static abstract bool operator <(in TSelf a, in TSelf b);

	/// <summary>Greater than, on the value part.</summary>
	static abstract bool operator >(in TSelf a, in TSelf b);

	/// <summary>Less than or equal, on the value part.</summary>
	static abstract bool operator <=(in TSelf a, in TSelf b);

	/// <summary>Greater than or equal, on the value part.</summary>
	static abstract bool operator >=(in TSelf a, in TSelf b);

	/// <summary>ceres::sqrt.</summary>
	static abstract TSelf Sqrt(in TSelf a);

	/// <summary>ceres::abs.</summary>
	static abstract TSelf Abs(in TSelf a);

	/// <summary>ceres::sin.</summary>
	static abstract TSelf Sin(in TSelf a);

	/// <summary>ceres::cos.</summary>
	static abstract TSelf Cos(in TSelf a);

	/// <summary>ceres::tan.</summary>
	static abstract TSelf Tan(in TSelf a);

	/// <summary>ceres::asin.</summary>
	static abstract TSelf Asin(in TSelf a);

	/// <summary>ceres::acos.</summary>
	static abstract TSelf Acos(in TSelf a);

	/// <summary>ceres::atan.</summary>
	static abstract TSelf Atan(in TSelf a);

	/// <summary>ceres::atan2(y, x).</summary>
	static abstract TSelf Atan2(in TSelf y, in TSelf x);

	/// <summary>ceres::exp.</summary>
	static abstract TSelf Exp(in TSelf a);

	/// <summary>ceres::log (natural logarithm).</summary>
	static abstract TSelf Log(in TSelf a);

	/// <summary>ceres::pow with a constant exponent.</summary>
	static abstract TSelf Pow(in TSelf a, double exponent);

	/// <summary>ceres::pow with a differentiable exponent.</summary>
	static abstract TSelf Pow(in TSelf a, in TSelf exponent);

	/// <summary>ceres::floor (a Jet's derivative is zero).</summary>
	static abstract TSelf Floor(in TSelf a);

	/// <summary>ceres::hypot(x, y).</summary>
	static abstract TSelf Hypot(in TSelf x, in TSelf y);

	/// <summary>C++17 three-argument hypot(x, y, z), as ceres/rotation.h calls it.</summary>
	static abstract TSelf Hypot(in TSelf x, in TSelf y, in TSelf z);

	/// <summary>ceres::fmin: NaN is missing data; a Jet tie averages the two Jets.</summary>
	static abstract TSelf Min(in TSelf a, in TSelf b);

	/// <summary>ceres::fmax: NaN is missing data; a Jet tie averages the two Jets.</summary>
	static abstract TSelf Max(in TSelf a, in TSelf b);

	/// <summary>ceres::isfinite, on the value part.</summary>
	static abstract bool IsFinite(in TSelf a);

	/// <summary>ceres::isnan, on the value part.</summary>
	static abstract bool IsNaN(in TSelf a);
}

/// <summary>
/// A double behind <see cref="IScalar{TSelf}"/>: the <c>T = double</c> instantiation of
/// COLMAP's templates. Same layout as double, so double spans reinterpret as Real spans.
/// Every member is the plain double operation (the C++ <c>std::</c> function), so this path
/// rounds exactly like double code.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Real : IScalar<Real>
{
	/// <summary>The wrapped double.</summary>
	public readonly double Value;

	/// <summary>Wraps a double.</summary>
	public Real(double value)
	{
		Value = value;
	}

	/// <summary>Wraps a double.</summary>
	public static implicit operator Real(double value) => new(value);

	/// <summary>Unwraps the double.</summary>
	public static implicit operator double(Real value) => value.Value;

	/// <summary>Reinterprets a span of doubles as Reals (no copy).</summary>
	public static ReadOnlySpan<Real> Cast(ReadOnlySpan<double> values) => MemoryMarshal.Cast<double, Real>(values);

	/// <summary>Reinterprets a writable span of doubles as Reals (no copy).</summary>
	public static Span<Real> CastWritable(Span<double> values) => MemoryMarshal.Cast<double, Real>(values);

	/// <inheritdoc/>
	public static Real FromDouble(double value) => new(value);

	/// <inheritdoc/>
	public static double ScalarPart(in Real a) => a.Value;

	/// <inheritdoc/>
	public static Real operator +(in Real a, in Real b) => new(a.Value + b.Value);

	/// <inheritdoc/>
	public static Real operator +(in Real a, double s) => new(a.Value + s);

	/// <inheritdoc/>
	public static Real operator +(double s, in Real a) => new(s + a.Value);

	/// <inheritdoc/>
	public static Real operator -(in Real a, in Real b) => new(a.Value - b.Value);

	/// <inheritdoc/>
	public static Real operator -(in Real a, double s) => new(a.Value - s);

	/// <inheritdoc/>
	public static Real operator -(double s, in Real a) => new(s - a.Value);

	/// <inheritdoc/>
	public static Real operator *(in Real a, in Real b) => new(a.Value * b.Value);

	/// <inheritdoc/>
	public static Real operator *(in Real a, double s) => new(a.Value * s);

	/// <inheritdoc/>
	public static Real operator *(double s, in Real a) => new(s * a.Value);

	/// <inheritdoc/>
	public static Real operator /(in Real a, in Real b) => new(a.Value / b.Value);

	/// <inheritdoc/>
	public static Real operator /(in Real a, double s) => new(a.Value / s);

	/// <inheritdoc/>
	public static Real operator /(double s, in Real a) => new(s / a.Value);

	/// <inheritdoc/>
	public static Real operator -(in Real a) => new(-a.Value);

	/// <inheritdoc/>
	public static bool operator <(in Real a, in Real b) => a.Value < b.Value;

	/// <inheritdoc/>
	public static bool operator >(in Real a, in Real b) => a.Value > b.Value;

	/// <inheritdoc/>
	public static bool operator <=(in Real a, in Real b) => a.Value <= b.Value;

	/// <inheritdoc/>
	public static bool operator >=(in Real a, in Real b) => a.Value >= b.Value;

	/// <inheritdoc/>
	public static Real Sqrt(in Real a) => new(Math.Sqrt(a.Value));

	/// <inheritdoc/>
	public static Real Abs(in Real a) => new(Math.Abs(a.Value));

	/// <inheritdoc/>
	public static Real Sin(in Real a) => new(Math.Sin(a.Value));

	/// <inheritdoc/>
	public static Real Cos(in Real a) => new(Math.Cos(a.Value));

	/// <inheritdoc/>
	public static Real Tan(in Real a) => new(Math.Tan(a.Value));

	/// <inheritdoc/>
	public static Real Asin(in Real a) => new(Math.Asin(a.Value));

	/// <inheritdoc/>
	public static Real Acos(in Real a) => new(Math.Acos(a.Value));

	/// <inheritdoc/>
	public static Real Atan(in Real a) => new(Math.Atan(a.Value));

	/// <inheritdoc/>
	public static Real Atan2(in Real y, in Real x) => new(Math.Atan2(y.Value, x.Value));

	/// <inheritdoc/>
	public static Real Exp(in Real a) => new(Math.Exp(a.Value));

	/// <inheritdoc/>
	public static Real Log(in Real a) => new(Math.Log(a.Value));

	/// <inheritdoc/>
	public static Real Pow(in Real a, double exponent) => new(Math.Pow(a.Value, exponent));

	/// <inheritdoc/>
	public static Real Pow(in Real a, in Real exponent) => new(Math.Pow(a.Value, exponent.Value));

	/// <inheritdoc/>
	public static Real Floor(in Real a) => new(Math.Floor(a.Value));

	/// <inheritdoc/>
	public static Real Hypot(in Real x, in Real y) => new(double.Hypot(x.Value, y.Value));

	/// <inheritdoc/>
	public static Real Hypot(in Real x, in Real y, in Real z) => new(ScalarMath.Hypot(x.Value, y.Value, z.Value));

	/// <inheritdoc/>
	public static Real Min(in Real a, in Real b) => new(ScalarMath.FMin(a.Value, b.Value));

	/// <inheritdoc/>
	public static Real Max(in Real a, in Real b) => new(ScalarMath.FMax(a.Value, b.Value));

	/// <inheritdoc/>
	public static bool IsFinite(in Real a) => double.IsFinite(a.Value);

	/// <inheritdoc/>
	public static bool IsNaN(in Real a) => double.IsNaN(a.Value);

	/// <inheritdoc/>
	public override string ToString() => Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
}

/// <summary>
/// C/C++ standard-library double functions that .NET has no same-semantics equivalent for;
/// <see cref="Real"/>, <see cref="Jet{TGrad}"/> and the manifolds share them.
/// </summary>
public static class ScalarMath
{
	/// <summary>
	/// C++17 <c>std::hypot(x, y, z)</c> as libc++ computes it (the library the macOS pycolmap
	/// wheel links): sqrt(x^2 + y^2 + z^2) on arguments pre-scaled by a power of two when the
	/// largest magnitude could overflow (above 2^512) or underflow (below 2^-512) the squares.
	/// Power-of-two scaling is exact, so in range this is the plain sqrt of the sum of squares.
	/// </summary>
	public static double Hypot(double x, double y, double z)
	{
		// numeric_limits<double>::max_exponent / 2 = 1024 / 2.
		const int HalfMaxExponent = 512;
		double overflowThreshold = Math.ScaleB(1.0, HalfMaxExponent);
		double overflowScale = Math.ScaleB(1.0, -(HalfMaxExponent + 20));
		double maxAbs = Math.Max(Math.Abs(x), Math.Max(Math.Abs(y), Math.Abs(z)));
		double scale;
		if (maxAbs > overflowThreshold)
		{
			scale = overflowScale;
		}
		else if (maxAbs < 1 / overflowThreshold)
		{
			scale = 1 / overflowScale;
		}
		else
		{
			scale = 1;
		}

		x *= scale;
		y *= scale;
		z *= scale;
		return Math.Sqrt(x * x + y * y + z * z) / scale;
	}

	/// <summary>
	/// <c>std::fmin</c>: a NaN argument counts as missing, so the other one is returned
	/// (Math.Min would propagate the NaN).
	/// </summary>
	public static double FMin(double x, double y)
	{
		if (double.IsNaN(x))
		{
			return y;
		}

		if (double.IsNaN(y))
		{
			return x;
		}

		return y < x ? y : x;
	}

	/// <summary><c>std::fmax</c>: a NaN argument counts as missing.</summary>
	public static double FMax(double x, double y)
	{
		if (double.IsNaN(x))
		{
			return y;
		}

		if (double.IsNaN(y))
		{
			return x;
		}

		return x < y ? y : x;
	}
}
