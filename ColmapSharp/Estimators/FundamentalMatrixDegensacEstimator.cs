// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FundamentalMatrixDegensacEstimator: the FundamentalMatrixDegensacEstimator class of
// colmap/estimators/fundamental_matrix_degensac.h and .cc - the DEGENSAC fundamental matrix
// estimator that RANSAC runs as both the hypothesis and the LO-RANSAC local estimator
// (Optim/Estimator.cs). It fits the 7-point or 8-point model
// (Estimators/Solvers/FundamentalMatrixEstimators.cs) and replaces H-degenerate hypotheses by
// the plane-and-parallax completion of FundamentalMatrixDegensac.cs, next to this file,
// which also holds EstimateFundamentalMatrixDegensac.
// Tests: ColmapSharp.Tests/Estimators/FundamentalMatrixDegensacTests.cs.
//
// Tier C (outcome): the hypotheses depend on PRNG draws (Tier A) and solvers (Tier B).
//
// Translation notes:
// - COLMAP holds `const std::vector<Vector2d>*` to the full correspondence set, which must
//   outlive the estimator. A struct cannot hold a span, so it holds ReadOnlyMemory, which
//   arrays convert to implicitly.
// - The class has Refine(X, Y, M_t*), so LO-RANSAC refines the current best model; in C#
//   that is EstimateLocal, which refines a copy and appends it on success (Estimator.cs).

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// DEGENSAC-based fundamental matrix estimation, robust to a dominant scene plane. Based on
/// Ondrej Chum, Tomas Werner, Jiri Matas, "Two-View Geometry Estimation Unaffected by a
/// Dominant Plane", CVPR 2005. https://cmp.felk.cvut.cz/~werner/papers/chum-degen-cvpr05.pdf
/// </summary>
/// <remarks>
/// When the scene contains a dominant plane, a random minimal sample frequently contains a
/// majority of coplanar correspondences. The fundamental matrix computed from such an
/// H-degenerate sample is compatible with the plane homography but its epipole is
/// essentially unconstrained, yet it can still accrue high inlier support (all plane points
/// fit it). The recovered epipolar geometry is then wrong even though RANSAC reports a
/// confident result.
///
/// This is implemented as a solver that detects H-degeneracy of a sample and, instead of
/// returning the plane-corrupted model, completes it via plane-and-parallax: the dominant
/// plane homography is refit on the H-consistent data and the epipole (hence the fundamental
/// matrix) is recovered from the off-plane parallax. If no usable off-plane parallax exists,
/// the sample is rejected.
///
/// The solver handles both minimal (7-point) and non-minimal (&gt;= 8-point) samples, so it
/// can be used as BOTH the hypothesis and the local-optimization estimator in LO-RANSAC.
/// This is important: a plain non-minimal solver would re-fit a plane-corrupted model on the
/// (plane-dominated) inlier set during local optimization, undoing the completion.
///
/// The estimator needs the full correspondence set (not just the minimal sample) for the
/// plane-and-parallax completion, so it holds the data passed to RANSAC.
/// Port of colmap::FundamentalMatrixDegensacEstimator.
/// </remarks>
public readonly struct FundamentalMatrixDegensacEstimator
	: IEstimator<Vector2d, Vector2d, Matrix3d>, ILocalEstimator<Vector2d, Vector2d, Matrix3d>
{
	private readonly ReadOnlyMemory<Vector2d> points1;
	private readonly ReadOnlyMemory<Vector2d> points2;
	private readonly double sampsonMaxResidual;
	private readonly double planeMaxResidual;
	private readonly double offPlaneMinResidual;
	private readonly double minSampleHInlierRatio;
	private readonly int maxPlaneParallaxTrials;
	private readonly bool useSampsonRefinement;

	/// <summary>Create the estimator.</summary>
	/// <param name="points1">Full first-image correspondences (the same data passed to RANSAC).</param>
	/// <param name="points2">Full second-image correspondences.</param>
	/// <param name="sampsonMaxResidual">Squared max Sampson error used when scoring
	/// completion candidates (typically the RANSAC max_error squared).</param>
	/// <param name="planeMaxResidual">Squared max transfer error for a correspondence to
	/// count as lying on the dominant plane (degeneracy detection and homography refit).</param>
	/// <param name="offPlaneMinResidual">Squared min transfer error for a correspondence to
	/// be used as an off-plane parallax source when recovering the epipole.</param>
	/// <param name="minSampleHInlierRatio">Fraction of a sample that must be consistent with
	/// a single plane homography for the sample to be H-degenerate.</param>
	/// <param name="maxPlaneParallaxTrials">Off-plane pairs sampled during completion.</param>
	/// <param name="useSampsonRefinement">Polish local-optimization refits by minimizing the
	/// Sampson error when they are not plane-degenerate.</param>
	public FundamentalMatrixDegensacEstimator(
		ReadOnlyMemory<Vector2d> points1,
		ReadOnlyMemory<Vector2d> points2,
		double sampsonMaxResidual,
		double planeMaxResidual,
		double offPlaneMinResidual,
		double minSampleHInlierRatio,
		int maxPlaneParallaxTrials,
		bool useSampsonRefinement)
	{
		this.points1 = points1;
		this.points2 = points2;
		this.sampsonMaxResidual = sampsonMaxResidual;
		this.planeMaxResidual = planeMaxResidual;
		this.offPlaneMinResidual = offPlaneMinResidual;
		this.minSampleHInlierRatio = minSampleHInlierRatio;
		this.maxPlaneParallaxTrials = maxPlaneParallaxTrials;
		this.useSampsonRefinement = useSampsonRefinement;
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 7;

	/// <summary>
	/// Estimate fundamental matrix hypotheses from a sample (minimal 7-point or non-minimal
	/// &gt;= 8-point). For each hypothesis whose sample is H-degenerate, the plane-corrupted
	/// model is replaced by a plane-and-parallax completion, or dropped if no usable
	/// off-plane parallax exists.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> samplePoints1, ReadOnlySpan<Vector2d> samplePoints2, List<Matrix3d> models)
	{
		Check.Ge(samplePoints1.Length, MinNumSamples);

		// Fit the fundamental matrix from the sample: the 7-point solver for a minimal sample
		// (yielding up to three roots), the 8-point solver otherwise (e.g. the
		// local-optimization inlier set).
		var sampleModels = new List<Matrix3d>(3);
		if (samplePoints1.Length == MinNumSamples)
		{
			new FundamentalMatrixSevenPointEstimator().Estimate(samplePoints1, samplePoints2, sampleModels);
		}
		else
		{
			new FundamentalMatrixEightPointEstimator().Estimate(samplePoints1, samplePoints2, sampleModels);
		}

		models.Clear();
		foreach (Matrix3d sampleModel in sampleModels)
		{
			Matrix3d? planeH = FundamentalMatrixDegensac.DetectSampleHDegeneracy(
				sampleModel,
				samplePoints1,
				samplePoints2,
				planeMaxResidual,
				minSampleHInlierRatio);
			if (!planeH.HasValue)
			{
				// Non-degenerate sample: keep the fitted hypothesis as-is.
				models.Add(sampleModel);
				continue;
			}

			// H-degenerate sample: replace the plane-corrupted hypothesis by a
			// plane-and-parallax completion, or drop it if there is no usable parallax.
			Matrix3d? completedModel = FundamentalMatrixDegensac.FundamentalFromPlaneAndParallax(
				planeH.Value,
				points1.Span,
				points2.Span,
				sampsonMaxResidual,
				planeMaxResidual,
				offPlaneMinResidual,
				maxPlaneParallaxTrials);
			if (completedModel.HasValue)
			{
				models.Add(completedModel.Value);
			}
		}
	}

	/// <summary>
	/// Local optimization over an inlier set, used by LO-RANSAC. Refits via Estimate, then
	/// polishes by minimizing the Sampson error, unless the refit is plane-degenerate: the
	/// Sampson cost is then nearly flat along the epipole, so refining it trades a large
	/// epipole excursion for a negligible cost reduction. Returns false (leaving
	/// <paramref name="f"/> unchanged) when the refit yields no model.
	/// Port of colmap::FundamentalMatrixDegensacEstimator::Refine.
	/// </summary>
	public bool Refine(ReadOnlySpan<Vector2d> inlierPoints1, ReadOnlySpan<Vector2d> inlierPoints2, ref Matrix3d f)
	{
		Check.Eq(inlierPoints1.Length, inlierPoints2.Length);

		// Refit the inlier set with the full degeneracy handling.
		var models = new List<Matrix3d>(3);
		Estimate(inlierPoints1, inlierPoints2, models);
		if (models.Count == 0)
		{
			return false;
		}

		f = models[0];

		// Test the refit rather than the incoming model, so the gate applies to the model
		// actually being polished.
		if (useSampsonRefinement
			&& !FundamentalMatrixDegensac.IsSampleHDegenerate(
				f,
				inlierPoints1,
				inlierPoints2,
				planeMaxResidual,
				minSampleHInlierRatio))
		{
			FundamentalMatrix.RefineFundamentalMatrixSampson(inlierPoints1, inlierPoints2, ref f);
		}

		return true;
	}

	/// <summary>
	/// LO-RANSAC's local step: refines a copy of the current best model with
	/// <see cref="Refine"/> and appends it on success, as loransac.h does for an estimator
	/// with a Refine method.
	/// </summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> inlierPoints1, ReadOnlySpan<Vector2d> inlierPoints2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Matrix3d refinedModel = initialModel;
		if (Refine(inlierPoints1, inlierPoints2, ref refinedModel))
		{
			models.Add(refinedModel);
		}
	}

	/// <summary>Squared Sampson error residuals over the given correspondences.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d f, Span<double> residuals)
	{
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, f, residuals);
	}
}
