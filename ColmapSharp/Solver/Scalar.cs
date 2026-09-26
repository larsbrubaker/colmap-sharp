// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IScalar<T> and Real: the numeric abstraction that COLMAP's `template <typename T>` code is
// written against here. In C++ the camera models (sensor/models.h) and, later, every cost
// function are templates instantiated with `double` and with `ceres::Jet<double, N>`, so one
// body serves both plain evaluation and automatic differentiation. C# has no duck-typed
// templates, so that code is generic over `T : struct, IScalar<T>` instead:
// - Real is `double` behind the interface (a single-field struct the JIT keeps in a register).
//   Spans of double are reinterpreted as spans of Real with MemoryMarshal.Cast, so the double
//   path allocates nothing and rounds exactly like plain double arithmetic.
// - Jet2 (this folder) is ceres::Jet<double, 2>, which IterativeUndistortion needs now.
//   Phase 7's general Jet<N> implements this same interface, so the camera models, and
//   everything else written against IScalar, differentiate without being rewritten.
// The interface holds exactly what ceres::Jet offers COLMAP's templates: the four binary
// operators, negation, the scalar-minus-jet mix that `1.0 - alpha` compiles to, comparisons
// on the value part, T(double) construction, and the elementary functions the models call.
// It deliberately does not implement the BCL's INumberBase: a Jet cannot honestly implement
// parsing, formatting and the rest, and "no stubs" (CLAUDE.md) rules out faking them.

using System.Runtime.InteropServices;

namespace ColmapSharp.Solver;

/// <summary>
/// A scalar type COLMAP's templated numeric code can run on: <see cref="Real"/> for plain
/// double evaluation, Jets for forward-mode automatic differentiation.
/// </summary>
public interface IScalar<TSelf>
	where TSelf : struct, IScalar<TSelf>
{
	/// <summary>C++ <c>T(value)</c>: a constant (a Jet with zero derivative).</summary>
	static abstract TSelf FromDouble(double value);

	/// <summary>Addition.</summary>
	static abstract TSelf operator +(TSelf a, TSelf b);

	/// <summary>Subtraction.</summary>
	static abstract TSelf operator -(TSelf a, TSelf b);

	/// <summary>Multiplication.</summary>
	static abstract TSelf operator *(TSelf a, TSelf b);

	/// <summary>Division.</summary>
	static abstract TSelf operator /(TSelf a, TSelf b);

	/// <summary>Negation.</summary>
	static abstract TSelf operator -(TSelf a);

	/// <summary>A plain double minus a scalar, C++ <c>1.0 - alpha</c> with alpha a T.</summary>
	static abstract TSelf operator -(double s, TSelf a);

	/// <summary>Less than, on the value part (as ceres::Jet compares).</summary>
	static abstract bool operator <(TSelf a, TSelf b);

	/// <summary>Greater than, on the value part.</summary>
	static abstract bool operator >(TSelf a, TSelf b);

	/// <summary>Less than or equal, on the value part.</summary>
	static abstract bool operator <=(TSelf a, TSelf b);

	/// <summary>Greater than or equal, on the value part.</summary>
	static abstract bool operator >=(TSelf a, TSelf b);

	/// <summary>ceres::sqrt.</summary>
	static abstract TSelf Sqrt(TSelf a);

	/// <summary>ceres::abs.</summary>
	static abstract TSelf Abs(TSelf a);

	/// <summary>ceres::sin.</summary>
	static abstract TSelf Sin(TSelf a);

	/// <summary>ceres::cos.</summary>
	static abstract TSelf Cos(TSelf a);

	/// <summary>ceres::tan.</summary>
	static abstract TSelf Tan(TSelf a);

	/// <summary>ceres::atan.</summary>
	static abstract TSelf Atan(TSelf a);

	/// <summary>ceres::atan2(y, x).</summary>
	static abstract TSelf Atan2(TSelf y, TSelf x);
}

/// <summary>
/// A double behind <see cref="IScalar{TSelf}"/>: the <c>T = double</c> instantiation of
/// COLMAP's templates. Same layout as double, so double spans reinterpret as Real spans.
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

	/// <inheritdoc/>
	public static Real FromDouble(double value) => new(value);

	/// <inheritdoc/>
	public static Real operator +(Real a, Real b) => new(a.Value + b.Value);

	/// <inheritdoc/>
	public static Real operator -(Real a, Real b) => new(a.Value - b.Value);

	/// <inheritdoc/>
	public static Real operator *(Real a, Real b) => new(a.Value * b.Value);

	/// <inheritdoc/>
	public static Real operator /(Real a, Real b) => new(a.Value / b.Value);

	/// <inheritdoc/>
	public static Real operator -(Real a) => new(-a.Value);

	/// <inheritdoc/>
	public static Real operator -(double s, Real a) => new(s - a.Value);

	/// <inheritdoc/>
	public static bool operator <(Real a, Real b) => a.Value < b.Value;

	/// <inheritdoc/>
	public static bool operator >(Real a, Real b) => a.Value > b.Value;

	/// <inheritdoc/>
	public static bool operator <=(Real a, Real b) => a.Value <= b.Value;

	/// <inheritdoc/>
	public static bool operator >=(Real a, Real b) => a.Value >= b.Value;

	/// <inheritdoc/>
	public static Real Sqrt(Real a) => new(Math.Sqrt(a.Value));

	/// <inheritdoc/>
	public static Real Abs(Real a) => new(Math.Abs(a.Value));

	/// <inheritdoc/>
	public static Real Sin(Real a) => new(Math.Sin(a.Value));

	/// <inheritdoc/>
	public static Real Cos(Real a) => new(Math.Cos(a.Value));

	/// <inheritdoc/>
	public static Real Tan(Real a) => new(Math.Tan(a.Value));

	/// <inheritdoc/>
	public static Real Atan(Real a) => new(Math.Atan(a.Value));

	/// <inheritdoc/>
	public static Real Atan2(Real y, Real x) => new(Math.Atan2(y.Value, x.Value));

	/// <inheritdoc/>
	public override string ToString() => Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
}
