// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SupportMeasurement: colmap/optim/support_measurement.h and support_measurement.cc - the
// ways RANSAC scores a model from its residuals: inlier count plus residual sum
// (InlierSupportMeasurer), unique inlier ids first (UniqueInlierSupportMeasurer, used by
// the generalized absolute pose), and the MSAC truncated score (MEstimatorSupportMeasurer).
// The consumer is the RANSAC port (optim/ransac.h, loransac.h), which is templated on the
// measurer; the samplers it pairs with are in Sampler.cs. Tests:
// ColmapSharp.Tests/Optim/SupportMeasurementTests.cs (support_measurement_test.cc 1:1).
//
// Tier A (exact): counting and a left-to-right double sum, as in COLMAP.
//
// Translation notes:
// - C++ templates reach `typename SupportMeasurer::Support` and `support.num_inliers`
//   directly. C# has no associated types, so ISupportMeasurer<TSupport> names the support
//   type and IRansacSupport exposes NumInliers; RANSAC is generic over both
//   `TSupportMeasurer : ISupportMeasurer<TSupport>` and `TSupport`.
// - Each Support is a mutable struct with COLMAP's default member values. Those defaults
//   come from the parameterless constructor, so generic code must create an empty support
//   with `new TSupport()`: `default(TSupport)` would give a residual sum of 0 instead of
//   double.MaxValue and so beat every real support.
// - size_t counts become int (they are bounded by the residual count). Unique sample ids
//   are point3D_t-style ids, so they stay 64-bit (ulong).
// - residuals are a ReadOnlySpan<double>, so arrays and lists both work without copying.

using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>The part of a RANSAC support that RANSAC itself reads (<c>support.num_inliers</c>).</summary>
public interface IRansacSupport
{
	/// <summary>The number of inliers, used for the dynamic number of RANSAC trials.</summary>
	int NumInliers { get; }
}

/// <summary>
/// A way to measure how well a model is supported by its residuals, and to compare two
/// supports. The C# face of COLMAP's SupportMeasurer template parameter.
/// </summary>
public interface ISupportMeasurer<TSupport>
	where TSupport : struct, IRansacSupport
{
	/// <summary>Compute the support of the residuals.</summary>
	TSupport Evaluate(ReadOnlySpan<double> residuals, double maxResidual);

	/// <summary>Compare the two supports.</summary>
	bool IsLeftBetter(in TSupport left, in TSupport right);
}

/// <summary>
/// Measure the support of a model by counting the number of inliers and summing all inlier
/// residuals. The support is better if it has more inliers and a smaller residual sum.
/// Port of colmap::InlierSupportMeasurer.
/// </summary>
public sealed class InlierSupportMeasurer : ISupportMeasurer<InlierSupportMeasurer.Support>
{
	/// <summary>Port of colmap::InlierSupportMeasurer::Support.</summary>
	public struct Support : IRansacSupport
	{
		/// <summary>The number of inliers.</summary>
		public int NumInliers = 0;

		/// <summary>The sum of all inlier residuals.</summary>
		public double ResidualSum = double.MaxValue;

		/// <summary>An empty support with COLMAP's default values.</summary>
		public Support()
		{
		}

		readonly int IRansacSupport.NumInliers => NumInliers;
	}

	/// <inheritdoc/>
	public Support Evaluate(ReadOnlySpan<double> residuals, double maxResidual)
	{
		var support = new Support();
		support.NumInliers = 0;
		support.ResidualSum = 0;

		foreach (double residual in residuals)
		{
			if (residual <= maxResidual)
			{
				support.NumInliers += 1;
				support.ResidualSum += residual;
			}
		}

		return support;
	}

	/// <inheritdoc/>
	public bool IsLeftBetter(in Support left, in Support right)
	{
		if (left.NumInliers > right.NumInliers)
		{
			return true;
		}

		return left.NumInliers == right.NumInliers && left.ResidualSum < right.ResidualSum;
	}
}

/// <summary>
/// Measure the support of a model by counting the number of unique inliers (e.g., visible
/// 3D points), number of inliers, and summing all inlier residuals. Each sample should have
/// an associated id. Samples with the same id are only counted once in num_unique_inliers.
/// The support is better if it has more unique inliers, more inliers, and a smaller
/// residual sum. Port of colmap::UniqueInlierSupportMeasurer.
/// </summary>
public sealed class UniqueInlierSupportMeasurer : ISupportMeasurer<UniqueInlierSupportMeasurer.Support>
{
	/// <summary>Port of colmap::UniqueInlierSupportMeasurer::Support.</summary>
	public struct Support : IRansacSupport
	{
		/// <summary>The number of unique inliers.</summary>
		public int NumUniqueInliers = 0;

		/// <summary>
		/// The number of inliers. This is still needed for determining the dynamic number of
		/// iterations.
		/// </summary>
		public int NumInliers = 0;

		/// <summary>The sum of all inlier residuals.</summary>
		public double ResidualSum = double.MaxValue;

		/// <summary>An empty support with COLMAP's default values.</summary>
		public Support()
		{
		}

		readonly int IRansacSupport.NumInliers => NumInliers;
	}

	private readonly ulong[] uniqueSampleIds;

	/// <summary>
	/// Create a measurer for samples whose ids are <paramref name="uniqueSampleIds"/>, one per
	/// residual. The array is taken over, not copied (COLMAP moves the vector in), and is
	/// only ever read, so copies of this measurer can share it across threads.
	/// </summary>
	public UniqueInlierSupportMeasurer(ulong[] uniqueSampleIds)
	{
		this.uniqueSampleIds = uniqueSampleIds;
	}

	/// <inheritdoc/>
	public Support Evaluate(ReadOnlySpan<double> residuals, double maxResidual)
	{
		Check.Eq(residuals.Length, uniqueSampleIds.Length);
		var support = new Support();
		support.NumInliers = 0;
		support.NumUniqueInliers = 0;
		support.ResidualSum = 0;

		var inlierPointIds = new HashSet<ulong>();
		for (int idx = 0; idx < residuals.Length; ++idx)
		{
			if (residuals[idx] <= maxResidual)
			{
				support.NumInliers += 1;
				inlierPointIds.Add(uniqueSampleIds[idx]);
				support.ResidualSum += residuals[idx];
			}
		}

		support.NumUniqueInliers = inlierPointIds.Count;
		return support;
	}

	/// <inheritdoc/>
	public bool IsLeftBetter(in Support left, in Support right)
	{
		if (left.NumUniqueInliers > right.NumUniqueInliers)
		{
			return true;
		}
		else if (left.NumUniqueInliers == right.NumUniqueInliers)
		{
			if (left.NumInliers > right.NumInliers)
			{
				return true;
			}

			return left.NumInliers == right.NumInliers && left.ResidualSum < right.ResidualSum;
		}

		return false;
	}
}

/// <summary>
/// Measure the support of a model by its fitness to the data as used in MSAC. A support is
/// better if it has a smaller MSAC score. Port of colmap::MEstimatorSupportMeasurer.
/// </summary>
public sealed class MEstimatorSupportMeasurer : ISupportMeasurer<MEstimatorSupportMeasurer.Support>
{
	/// <summary>Port of colmap::MEstimatorSupportMeasurer::Support.</summary>
	public struct Support : IRansacSupport
	{
		/// <summary>The number of inliers.</summary>
		public int NumInliers = 0;

		/// <summary>The MSAC score, defined as the truncated sum of residuals.</summary>
		public double Score = double.MaxValue;

		/// <summary>An empty support with COLMAP's default values.</summary>
		public Support()
		{
		}

		readonly int IRansacSupport.NumInliers => NumInliers;
	}

	/// <inheritdoc/>
	public Support Evaluate(ReadOnlySpan<double> residuals, double maxResidual)
	{
		var support = new Support();
		support.NumInliers = 0;
		support.Score = 0;

		foreach (double residual in residuals)
		{
			if (residual <= maxResidual)
			{
				support.NumInliers += 1;
				support.Score += residual;
			}
			else
			{
				support.Score += maxResidual;
			}
		}

		return support;
	}

	/// <inheritdoc/>
	public bool IsLeftBetter(in Support left, in Support right)
	{
		return left.Score < right.Score;
	}
}
