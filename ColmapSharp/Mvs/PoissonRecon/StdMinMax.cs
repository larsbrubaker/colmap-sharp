// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// StdMinMax: std::min / std::max with the C++ standard library's exact semantics, for the
// PoissonRecon port. std::max(a, b) is `a < b ? b : a` and std::min(a, b) is `b < a ? b : a`:
// with a NaN they return the first argument, and for -0.0 vs +0.0 (which compare equal) they
// also return the first argument. .NET's Math.Max/Math.Min instead propagate NaN and order
// -0.0 below +0.0, so they can return a different value. Every std::max/std::min in ported
// PoissonRecon code goes through these helpers (integer ones too, for uniformity).

using System.Numerics;

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>std::min / std::max with the C++ semantics (first argument wins on ties and NaN).</summary>
public static class StdMinMax
{
	/// <summary>Port of <c>std::max( a , b )</c>: <c>a &lt; b ? b : a</c>.</summary>
	public static T StdMax<T>(T a, T b)
		where T : IComparisonOperators<T, T, bool> => a < b ? b : a;

	/// <summary>Port of <c>std::min( a , b )</c>: <c>b &lt; a ? b : a</c>.</summary>
	public static T StdMin<T>(T a, T b)
		where T : IComparisonOperators<T, T, bool> => b < a ? b : a;
}
