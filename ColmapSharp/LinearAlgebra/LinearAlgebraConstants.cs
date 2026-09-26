// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LinearAlgebraConstants: the numeric constants and the shared isApprox rule used by the
// fixed-size vectors, matrices and quaternions in this folder. Not a port; the values are
// Eigen's documented defaults, which COLMAP's tests rely on through isApprox.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Numeric constants shared by the LinearAlgebra types.
/// </summary>
public static class LinearAlgebraConstants
{
	/// <summary>
	/// Eigen's NumTraits&lt;double&gt;::dummy_precision(), the default precision of
	/// isApprox for double.
	/// </summary>
	public const double DummyPrecision = 1e-12;

	/// <summary>
	/// std::numeric_limits&lt;double&gt;::epsilon(). Not double.Epsilon, which is the
	/// smallest subnormal.
	/// </summary>
	public const double MachineEpsilon = 2.220446049250313E-16;

	/// <summary>
	/// Eigen's documented isApprox rule on flat coefficient spans:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||), with Frobenius (l2) norms.
	/// </summary>
	internal static bool IsApprox(ReadOnlySpan<double> a, ReadOnlySpan<double> b, double precision)
	{
		double differenceSquared = 0;
		double aSquared = 0;
		double bSquared = 0;
		for (int i = 0; i < a.Length; i++)
		{
			double difference = a[i] - b[i];
			differenceSquared += difference * difference;
			aSquared += a[i] * a[i];
			bSquared += b[i] * b[i];
		}

		return Math.Sqrt(differenceSquared) <= precision * Math.Min(Math.Sqrt(aSquared), Math.Sqrt(bSquared));
	}
}
