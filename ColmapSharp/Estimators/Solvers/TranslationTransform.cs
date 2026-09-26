// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TranslationTransform: colmap/estimators/solvers/translation_transform.h - the N-D
// translation estimator (difference of the point means), an Optim/Estimator.cs
// implementation. COLMAP instantiates only kDim = 2 (TranslationTransformEstimator<2>,
// run by LO-RANSAC in estimators/two_view_geometry.cc), so that is the one ported, as
// TranslationTransformEstimator2d; SimilarityTransform.cs made the same choice for kDim = 3.
// Tests: ColmapSharp.Tests/Estimators/Solvers/TranslationTransformTests.cs
// (translation_transform_test.cc 1:1).
//
// Tier A (exact): sums and one division per coordinate, in COLMAP's order.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Estimate a 2D translation between point pairs; the model is the translation vector.
/// Port of colmap::TranslationTransformEstimator&lt;2&gt;.
/// </summary>
public readonly struct TranslationTransformEstimator2d
	: IEstimator<Vector2d, Vector2d, Vector2d>, ILocalEstimator<Vector2d, Vector2d, Vector2d>
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 1;

	/// <summary>Estimate the translation as mean(points2) - mean(points1).</summary>
	public void Estimate(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, List<Vector2d> models)
	{
		Check.Eq(points1.Length, points2.Length);

		models.Clear();

		Vector2d meanSrc = Vector2d.Zero;
		Vector2d meanDst = Vector2d.Zero;

		for (int i = 0; i < points1.Length; ++i)
		{
			meanSrc += points1[i];
			meanDst += points2[i];
		}

		// Eigen divides a vector by the size_t count converted to double, coefficient-wise.
		meanSrc /= points1.Length;
		meanDst /= points2.Length;

		models.Add(meanDst - meanSrc);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Vector2d initialModel, List<Vector2d> models)
	{
		Estimate(points1, points2, models);
	}

	/// <summary>The squared translation error of each pair.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Vector2d translation, Span<double> residuals)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Eq(residuals.Length, points1.Length);

		for (int i = 0; i < points1.Length; ++i)
		{
			Vector2d diff = points2[i] - points1[i] - translation;
			residuals[i] = diff.SquaredNorm;
		}
	}
}
