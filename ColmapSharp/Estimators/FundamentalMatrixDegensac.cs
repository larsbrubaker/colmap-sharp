// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FundamentalMatrixDegensac: colmap/estimators/fundamental_matrix_degensac.h and .cc - the
// free functions of DEGENSAC (Chum, Werner and Matas, "Two-View Geometry Estimation
// Unaffected by a Dominant Plane", CVPR 2005): the epipole of F, the plane homography
// compatible with F through three correspondences (H&Z Result 13.6), the H-degeneracy test
// of a sample, the plane-and-parallax completion, and EstimateFundamentalMatrixDegensac,
// which runs FundamentalMatrixDegensacEstimator.cs (the estimator struct next to this file)
// inside Optim/LoRansac.cs as both the hypothesis and the local-optimization estimator.
// Built on Estimators/Solvers/FundamentalMatrixEstimators.cs (7-point, 8-point, Sampson
// refiner) and Estimators/Solvers/HomographyMatrixEstimator.cs. Consumer: the two-view
// geometry estimator's `use_degensac` branch of EstimateFundamentalMatrix.
// Tests: ColmapSharp.Tests/Estimators/FundamentalMatrixDegensacTests.cs
// (fundamental_matrix_degensac_test.cc 1:1).
//
// Tier B (tolerance) for the closed-form pieces (epipole, compatible homography); Tier C
// (outcome) for the degeneracy test on non-minimal samples, the completion and the robust
// estimate, which draw from the PRNG. The draws themselves are Tier A: they go through
// RandomUtils in COLMAP's order with COLMAP's integer types (int for the triplet indices,
// size_t -> ulong for the Fisher-Yates and parallax-pair draws), so a seeded run consumes
// the same sequence as COLMAP.
//
// Translation notes:
// - std::vector inputs become ReadOnlySpan; std::optional<Matrix3d> becomes Matrix3d?.
// - The `evaluate_triplet` lambda of DetectSampleHDegeneracy captures the sample vectors,
//   which a C# lambda cannot do with spans, so it is a private static method that updates
//   the running best through ref parameters.
// - std::llround rounds halves away from zero: Math.Round(x, MidpointRounding.AwayFromZero).

using System.Runtime.InteropServices;

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Options of <see cref="FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac"/>.
/// Port of colmap::FundamentalMatrixDegensacOptions.
/// </summary>
public struct FundamentalMatrixDegensacOptions
{
	/// <summary>
	/// RANSAC options that control sampling, scoring, and termination. As in
	/// RANSAC/LO-RANSAC, <c>MaxError</c> is a pixel error that is squared internally, since
	/// Sampson residuals are squared errors.
	/// </summary>
	public RansacOptions Ransac = new();

	/// <summary>
	/// Maximum pixel error for a correspondence to count as lying on the dominant plane, used
	/// for the degeneracy test and the homography refit. Squared internally. Deliberately
	/// looser than the inlier threshold so noisy plane points are still recognized as
	/// coplanar. If &lt;= 0, derived as sqrt(3) * Ransac.MaxError (i.e. 3x in squared-pixel
	/// units).
	/// </summary>
	public double PlaneMaxError = -1;

	/// <summary>
	/// Minimum pixel error for a correspondence to be used as an off-plane parallax source
	/// when recovering the epipole. Squared internally. Deliberately much larger than the
	/// inlier threshold so only clean, high-parallax points constrain the epipole (near-plane,
	/// low-parallax points are ignored). If &lt;= 0, derived as 10 * Ransac.MaxError (100x
	/// squared).
	/// </summary>
	public double OffPlaneMinError = -1;

	/// <summary>
	/// Fraction of a sample that must be consistent with a single plane homography for the
	/// sample to be considered H-degenerate. The default corresponds to the paper's "at least
	/// 5 of 7" criterion for a minimal sample.
	/// </summary>
	public double MinSampleHInlierRatio = 5.0 / 7.0;

	/// <summary>
	/// Maximum number of off-plane correspondence pairs sampled during the plane-and-parallax
	/// model completion to recover the epipole. Only a couple of off-plane inliers are needed,
	/// so a small budget suffices; the completed model is scored and locally optimized by the
	/// surrounding RANSAC anyway. The value 25 sits at the knee of a runtime/accuracy sweep:
	/// fewer trials erode success on the hardest >=98%-plane scenes (t=5: -3pts) with no
	/// runtime win (the inner pair loop is not the bottleneck), while more trials (or dynamic
	/// termination) yield no accuracy gain at added cost.
	/// </summary>
	public int MaxPlaneParallaxTrials = 25;

	/// <summary>
	/// Polish the local-optimization refit by minimizing the Sampson error, on inlier sets
	/// that are not plane-degenerate.
	/// </summary>
	public bool UseSampsonRefinement = true;

	/// <summary>Options with COLMAP's defaults.</summary>
	public FundamentalMatrixDegensacOptions()
	{
	}
}

/// <summary>The free functions of fundamental_matrix_degensac.h.</summary>
public static class FundamentalMatrixDegensac
{
	// All C(7,3) = 35 triplets of the seven minimal-sample indices, in COLMAP's
	// lexicographic order (the order decides which of equally good triplets wins).
	private static readonly int[] SampleTriplets = BuildSampleTriplets();

	// Number of triplets randomly sampled from a non-minimal sample when searching for the
	// dominant plane (a minimal sample enumerates all 35 exhaustively).
	private const int NumSampledTriplets = 50;

	/// <summary>
	/// Robustly estimate the fundamental matrix from corresponding image points using
	/// DEGENSAC inside LO-RANSAC (the DEGENSAC estimator is used as both the hypothesis and
	/// the local-optimization solver). The estimator keeps the point memory for its
	/// plane-and-parallax completion, so pass the same data it will score.
	/// Port of colmap::EstimateFundamentalMatrixDegensac.
	/// </summary>
	public static RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support> EstimateFundamentalMatrixDegensac(
		ReadOnlyMemory<Vector2d> points1,
		ReadOnlyMemory<Vector2d> points2,
		in FundamentalMatrixDegensacOptions options)
	{
		double sampsonMaxResidual = options.Ransac.MaxError * options.Ransac.MaxError;
		// A correspondence counts as on the dominant plane within a looser margin than the
		// inlier threshold; only points well off the plane serve as parallax.
		double planeMaxError = options.PlaneMaxError > 0
			? options.PlaneMaxError
			: Math.Sqrt(3.0) * options.Ransac.MaxError;
		double offPlaneMinError = options.OffPlaneMinError > 0
			? options.OffPlaneMinError
			: 10.0 * options.Ransac.MaxError;
		double planeMaxResidual = planeMaxError * planeMaxError;
		double offPlaneMinResidual = offPlaneMinError * offPlaneMinError;

		// The DEGENSAC estimator is used as BOTH the hypothesis and the local-optimization
		// solver, so the local optimization also applies the degeneracy handling instead of
		// re-fitting a plane-corrupted model.
		var estimator = new FundamentalMatrixDegensacEstimator(
			points1,
			points2,
			sampsonMaxResidual,
			planeMaxResidual,
			offPlaneMinResidual,
			options.MinSampleHInlierRatio,
			options.MaxPlaneParallaxTrials,
			options.UseSampsonRefinement);
		var ransac = new LoRansac<FundamentalMatrixDegensacEstimator, FundamentalMatrixDegensacEstimator,
			Vector2d, Vector2d, Matrix3d, MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			options.Ransac, estimator, estimator, new MEstimatorSupportMeasurer());
		return ransac.Estimate(points1.Span, points2.Span);
	}

	/// <summary>
	/// Compute the epipole in the second image, i.e. the left null vector e2 of the
	/// fundamental matrix with F^T e2 = 0. Returned in homogeneous coordinates and normalized
	/// to unit length. Port of colmap::EpipoleFromFundamentalMatrix.
	/// </summary>
	public static Vector3d EpipoleFromFundamentalMatrix(in Matrix3d f)
	{
		// The epipole e2 is the left null vector (F^T e2 = 0), i.e. it is orthogonal to the
		// column space of the rank-2 matrix F and thus parallel to the cross product of two
		// of its columns. Using the column pair with the largest cross product is numerically
		// stable and avoids a full SVD.
		Vector3d e01 = f.Col(0).Cross(f.Col(1));
		Vector3d e02 = f.Col(0).Cross(f.Col(2));
		Vector3d e12 = f.Col(1).Cross(f.Col(2));
		double n01 = e01.SquaredNorm;
		double n02 = e02.SquaredNorm;
		double n12 = e12.SquaredNorm;
		if (n01 >= n02 && n01 >= n12)
		{
			return e01.Normalized();
		}

		return n02 >= n12 ? e02.Normalized() : e12.Normalized();
	}

	/// <summary>
	/// Compute the plane-induced homography compatible with the epipolar geometry F from
	/// three point correspondences, following Hartley and Zisserman, "Multiple View
	/// Geometry", Result 13.6: H = [e2]_x F - e2 (M^{-1} b)^T, where the rows of M are the
	/// homogeneous first-image points x1_i, and
	/// b_i = (x2_i x ([e2]_x F x1_i)) . (x2_i x e2) / ||x2_i x e2||^2.
	/// Returns null on degenerate input (collinear first-image points, or a point at the
	/// epipole). <paramref name="points1"/> and <paramref name="points2"/> hold exactly three
	/// points each (std::array&lt;Vector2d, 3&gt;).
	/// Port of colmap::HomographyFromFundamentalAndPoints.
	/// </summary>
	public static Matrix3d? HomographyFromFundamentalAndPoints(
		in Matrix3d f,
		Vector3d epipole2,
		ReadOnlySpan<Vector2d> points1,
		ReadOnlySpan<Vector2d> points2)
	{
		Check.Eq(points1.Length, 3);
		Check.Eq(points2.Length, 3);
		// A = [e2]_x F is a particular homography compatible with F (H&Z Result 13.6).
		return HomographyFromCompatible(
			Rigid3d.CrossProductMatrix(epipole2) * f,
			epipole2,
			points1[0], points1[1], points1[2],
			points2[0], points2[1], points2[2]);
	}

	/// <summary>
	/// Test a sample for H-degeneracy, i.e. whether a fraction of at least
	/// <paramref name="minSampleHInlierRatio"/> of the correspondences lie on a common scene
	/// plane. Plane homographies compatible with F are constructed from triplets of the
	/// sample (all C(7,3) triplets for a minimal sample, otherwise a fixed number of randomly
	/// sampled triplets) and the number of sample correspondences consistent with each
	/// (squared forward transfer error &lt;= <paramref name="hMaxResidual"/>) is counted.
	/// Returns the homography of the triplet with the most consistent correspondences if that
	/// count reaches the threshold, otherwise null. Port of colmap::DetectSampleHDegeneracy.
	/// </summary>
	public static Matrix3d? DetectSampleHDegeneracy(
		in Matrix3d f,
		ReadOnlySpan<Vector2d> samplePoints1,
		ReadOnlySpan<Vector2d> samplePoints2,
		double hMaxResidual,
		double minSampleHInlierRatio)
	{
		int numSamples = samplePoints1.Length;
		Check.Eq(samplePoints1.Length, samplePoints2.Length);
		Check.Ge(numSamples, 3);

		Vector3d epipole2 = EpipoleFromFundamentalMatrix(f);
		// A = [e2]_x F is shared by all triplets; compute it once.
		Matrix3d a = Rigid3d.CrossProductMatrix(epipole2) * f;
		int minConsistent = Math.Max(
			3,
			(int)(long)Math.Round(minSampleHInlierRatio * numSamples, MidpointRounding.AwayFromZero));

		int bestNumConsistent = minConsistent - 1;
		Matrix3d? bestH = null;

		if (numSamples == FundamentalMatrixDegensacEstimator.MinNumSamples)
		{
			// Minimal sample: exhaustively enumerate all triplets.
			for (int t = 0; t < SampleTriplets.Length; t += 3)
			{
				if (EvaluateTriplet(
					a, epipole2, samplePoints1, samplePoints2, hMaxResidual,
					SampleTriplets[t], SampleTriplets[t + 1], SampleTriplets[t + 2],
					ref bestNumConsistent, ref bestH))
				{
					break;
				}
			}
		}
		else
		{
			// Non-minimal sample (e.g. a local-optimization inlier set): sample triplets. On a
			// plane-dominated sample most triplets lie on the plane, so a modest number
			// reliably discovers it.
			int last = numSamples - 1;
			for (int t = 0; t < NumSampledTriplets; ++t)
			{
				int i = RandomUtils.RandomUniformInteger(0, last);
				int j = RandomUtils.RandomUniformInteger(0, last);
				int k = RandomUtils.RandomUniformInteger(0, last);
				if (i == j || j == k || i == k)
				{
					continue;
				}

				if (EvaluateTriplet(
					a, epipole2, samplePoints1, samplePoints2, hMaxResidual,
					i, j, k,
					ref bestNumConsistent, ref bestH))
				{
					break;
				}
			}
		}

		return bestH;
	}

	/// <summary>
	/// Convenience predicate wrapping <see cref="DetectSampleHDegeneracy"/>.
	/// Port of colmap::IsSampleHDegenerate.
	/// </summary>
	public static bool IsSampleHDegenerate(
		in Matrix3d f,
		ReadOnlySpan<Vector2d> samplePoints1,
		ReadOnlySpan<Vector2d> samplePoints2,
		double hMaxResidual,
		double minSampleHInlierRatio)
	{
		return DetectSampleHDegeneracy(f, samplePoints1, samplePoints2, hMaxResidual, minSampleHInlierRatio).HasValue;
	}

	/// <summary>
	/// Recover the fundamental matrix from a dominant plane homography and the off-plane
	/// parallax (plane-and-parallax model completion). The seed homography is refit on its
	/// plane inliers (squared transfer error &lt;= <paramref name="planeMaxResidual"/>); then
	/// correspondences whose squared transfer error exceeds
	/// <paramref name="offPlaneMinResidual"/> are treated as clean off-plane parallax sources,
	/// and the epipole is recovered from a pair of them as
	/// e2 = (x2_a x H x1_a) x (x2_b x H x1_b), giving F = [e2]_x H. Pairs are sampled robustly
	/// and the best F is then refined by fitting the 8-point algorithm to mixed samples of
	/// plane and off-plane inliers (so the epipole is constrained by more than two off-plane
	/// points). The F with the largest squared-Sampson support (threshold
	/// <paramref name="sampsonMaxResidual"/>) over all correspondences is returned, or null
	/// if there are too few off-plane correspondences.
	/// Port of colmap::FundamentalFromPlaneAndParallax.
	/// </summary>
	public static Matrix3d? FundamentalFromPlaneAndParallax(
		in Matrix3d seedH,
		ReadOnlySpan<Vector2d> points1,
		ReadOnlySpan<Vector2d> points2,
		double sampsonMaxResidual,
		double planeMaxResidual,
		double offPlaneMinResidual,
		int maxTrials)
	{
		Check.Eq(points1.Length, points2.Length);
		int numPoints = points1.Length;

		// The seed homography is built from only three sample correspondences (via the
		// plane-corrupted sample fundamental matrix), so it is only approximate. Refit it on
		// all of its plane inliers to obtain an accurate dominant-plane homography; otherwise
		// the off-plane classification below is polluted with plane points and the epipole
		// recovery becomes unreliable.
		Matrix3d h = seedH;
		var homographyEstimator = new HomographyMatrixEstimator();
		var planeIdxs = new List<ulong>();
		var planePoints1 = new List<Vector2d>();
		var planePoints2 = new List<Vector2d>();
		var homographies = new List<Matrix3d>();
		const int NumRefitIters = 2;
		// A homography is well determined by a modest, spatially spread subset, so cap the
		// number of correspondences used for the DLT to avoid an O(N) SVD when the dominant
		// plane has many inliers.
		const int MaxHomographyFitPoints = 64;
		for (int iter = 0; iter < NumRefitIters; ++iter)
		{
			planeIdxs.Clear();
			for (int i = 0; i < numPoints; ++i)
			{
				if (HomographyMatrix.ComputeSquaredHomographyError(points1[i], points2[i], h) <= planeMaxResidual)
				{
					planeIdxs.Add((ulong)i);
				}
			}

			if (planeIdxs.Count < HomographyMatrixEstimator.MinNumSamples)
			{
				break;
			}

			int numFit = Math.Min(MaxHomographyFitPoints, planeIdxs.Count);
			if (planeIdxs.Count > numFit)
			{
				SampleDistinct(numFit, planeIdxs);
			}

			planePoints1.Clear();
			planePoints2.Clear();
			for (int i = 0; i < numFit; ++i)
			{
				planePoints1.Add(points1[(int)planeIdxs[i]]);
				planePoints2.Add(points2[(int)planeIdxs[i]]);
			}

			homographies.Clear();
			homographyEstimator.Estimate(
				CollectionsMarshal.AsSpan(planePoints1), CollectionsMarshal.AsSpan(planePoints2), homographies);
			if (homographies.Count == 0)
			{
				break;
			}

			h = homographies[0];
		}

		// The dominant-plane homography is now fixed, so the transfer error of every
		// correspondence against it is reused below (off-plane classification here and the
		// plane/off-plane split for the mixed-sample refit) instead of recomputed.
		var planeTransferErrors = new double[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			planeTransferErrors[i] = HomographyMatrix.ComputeSquaredHomographyError(points1[i], points2[i], h);
		}

		// Only correspondences well off the plane carry reliable parallax: near-plane points
		// have tiny, noise-dominated parallax that destabilizes the epipole.
		var offPlaneIdxs = new List<int>(numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			if (planeTransferErrors[i] > offPlaneMinResidual)
			{
				offPlaneIdxs.Add(i);
			}
		}

		int numOffPlane = offPlaneIdxs.Count;
		if (numOffPlane < 2)
		{
			return null;
		}

		// Precompute the transferred plane points H * x1 and the off-plane points, so the
		// epipole search below scores over only the off-plane subset.
		var lines = new Vector3d[numOffPlane];
		var offPoints1 = new Vector2d[numOffPlane];
		var offPoints2 = new Vector2d[numOffPlane];
		for (int i = 0; i < numOffPlane; ++i)
		{
			int idx = offPlaneIdxs[i];
			lines[i] = points2[idx].Homogeneous().Cross(h * points1[idx].Homogeneous());
			offPoints1[i] = points1[idx];
			offPoints2[i] = points2[idx];
		}

		var supportMeasurer = new InlierSupportMeasurer();
		var offResiduals = new double[numOffPlane];

		// Plane correspondences are consistent with F = [e2]_x H for ANY epipole e2
		// (x2^T [e2]_x H x1 ~ x2^T [e2]_x x2 = 0 when H x1 ~ x2), so among the F = [e2]_x H
		// candidates only the off-plane correspondences discriminate. Scoring epipole
		// candidates over just the off-plane subset ranks them correctly at a fraction of the
		// cost of scoring all correspondences.
		var bestOffSupport = new InlierSupportMeasurer.Support();
		Matrix3d? bestF = null;
		ulong numOffPlaneU = (ulong)numOffPlane;
		ulong numPairs = numOffPlaneU * (numOffPlaneU - 1) / 2;
		ulong numTrials = Math.Min((ulong)Math.Max(maxTrials, 1), numPairs);
		for (ulong trial = 0; trial < numTrials; ++trial)
		{
			ulong a = RandomUtils.RandomUniformInteger<ulong>(0, numOffPlaneU - 1);
			ulong b = RandomUtils.RandomUniformInteger<ulong>(0, numOffPlaneU - 1);
			if (a == b)
			{
				b = (b + 1 == numOffPlaneU) ? 0 : b + 1;
			}

			// The epipole is the intersection of the two parallax lines.
			Vector3d epipole2 = lines[(int)a].Cross(lines[(int)b]);
			if (epipole2.Norm < 1e-9)
			{
				continue;
			}

			Matrix3d f = Rigid3d.CrossProductMatrix(epipole2.Normalized()) * h;

			FundamentalMatrixResiduals.SquaredSampsonError(offPoints1, offPoints2, f, offResiduals);
			InlierSupportMeasurer.Support support = supportMeasurer.Evaluate(offResiduals, sampsonMaxResidual);
			if (supportMeasurer.IsLeftBetter(support, bestOffSupport))
			{
				bestOffSupport = support;
				bestF = f;
			}
		}

		if (!bestF.HasValue)
		{
			return null;
		}

		// Seed the full-data support of the best epipole candidate, against which the
		// mixed-sample refinement below is compared.
		var residuals = new double[numPoints];
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, bestF.Value, residuals);
		InlierSupportMeasurer.Support bestSupport = supportMeasurer.Evaluate(residuals, sampsonMaxResidual);

		RefineWithMixedSamples(
			points1, points2, planeTransferErrors, residuals, supportMeasurer,
			sampsonMaxResidual, planeMaxResidual, offPlaneMinResidual,
			ref bestSupport, ref bestF);

		return bestF;
	}

	// The tail of FundamentalFromPlaneAndParallax: refine the recovered fundamental matrix by
	// fitting the 8-point algorithm to mixed samples of plane and off-plane inliers, so the
	// epipole is constrained by several off-plane points rather than just the two used for
	// the epipole candidate. Fitting on a balanced sample avoids the plane bias that a
	// full-inlier fit would incur. `residuals` holds the Sampson residuals of the incoming
	// best F over all points (COLMAP recomputes them here; they are identical).
	private static void RefineWithMixedSamples(
		ReadOnlySpan<Vector2d> points1,
		ReadOnlySpan<Vector2d> points2,
		double[] planeTransferErrors,
		double[] residuals,
		InlierSupportMeasurer supportMeasurer,
		double sampsonMaxResidual,
		double planeMaxResidual,
		double offPlaneMinResidual,
		ref InlierSupportMeasurer.Support bestSupport,
		ref Matrix3d? bestF)
	{
		int numPoints = points1.Length;
		var planeInliers = new List<ulong>();
		var offPlaneInliers = new List<ulong>();
		for (int i = 0; i < numPoints; ++i)
		{
			if (planeTransferErrors[i] <= planeMaxResidual)
			{
				planeInliers.Add((ulong)i);
			}
		}

		for (int i = 0; i < numPoints; ++i)
		{
			if (residuals[i] <= sampsonMaxResidual && planeTransferErrors[i] > offPlaneMinResidual)
			{
				offPlaneInliers.Add((ulong)i);
			}
		}

		const int NumPlaneSample = 6;
		const int MaxNumOffPlaneSample = 4;
		if (planeInliers.Count < NumPlaneSample || offPlaneInliers.Count < 2)
		{
			return;
		}

		int numOffSample = Math.Min(MaxNumOffPlaneSample, offPlaneInliers.Count);
		var eightPoint = new FundamentalMatrixEightPointEstimator();
		var samplePoints1 = new Vector2d[NumPlaneSample + numOffSample];
		var samplePoints2 = new Vector2d[NumPlaneSample + numOffSample];
		var refinedModels = new List<Matrix3d>();
		const int NumRefineTrials = 15;
		for (int trial = 0; trial < NumRefineTrials; ++trial)
		{
			SampleDistinct(NumPlaneSample, planeInliers);
			SampleDistinct(numOffSample, offPlaneInliers);
			for (int i = 0; i < NumPlaneSample; ++i)
			{
				samplePoints1[i] = points1[(int)planeInliers[i]];
				samplePoints2[i] = points2[(int)planeInliers[i]];
			}

			for (int i = 0; i < numOffSample; ++i)
			{
				samplePoints1[NumPlaneSample + i] = points1[(int)offPlaneInliers[i]];
				samplePoints2[NumPlaneSample + i] = points2[(int)offPlaneInliers[i]];
			}

			refinedModels.Clear();
			eightPoint.Estimate(samplePoints1, samplePoints2, refinedModels);
			if (refinedModels.Count == 0)
			{
				continue;
			}

			FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, refinedModels[0], residuals);
			InlierSupportMeasurer.Support support = supportMeasurer.Evaluate(residuals, sampsonMaxResidual);
			if (supportMeasurer.IsLeftBetter(support, bestSupport))
			{
				bestSupport = support;
				bestF = refinedModels[0];
			}
		}
	}

	// Plane-induced homography compatible with the epipolar geometry from three
	// correspondences (H&Z Result 13.6), given the precomputed A = [e2]_x F and the epipole
	// e2. Splitting this out lets the degeneracy test reuse A across all triplets of a sample
	// instead of recomputing it each time. Port of the anonymous HomographyFromCompatible.
	private static Matrix3d? HomographyFromCompatible(
		in Matrix3d a,
		Vector3d epipole2,
		Vector2d p10, Vector2d p11, Vector2d p12,
		Vector2d p20, Vector2d p21, Vector2d p22)
	{
		if (!TransferCoefficient(a, epipole2, p10, p20, out double b0)
			|| !TransferCoefficient(a, epipole2, p11, p21, out double b1)
			|| !TransferCoefficient(a, epipole2, p12, p22, out double b2))
		{
			return null;
		}

		var m = Matrix3d.FromRows(p10.Homogeneous(), p11.Homogeneous(), p12.Homogeneous());

		// Reject collinear first-image points (scale-invariant singularity check), then solve
		// with the closed-form 3x3 inverse.
		double det = m.Determinant();
		double scale = m.Row(0).Norm * m.Row(1).Norm * m.Row(2).Norm;
		if (scale < 1e-12 || Math.Abs(det) < 1e-9 * scale)
		{
			return null;
		}

		Vector3d minvB = m.Inverse() * new Vector3d(b0, b1, b2);
		// A - e2 * (M^{-1} b)^T.
		return a - Matrix3d.FromColumns(epipole2 * minvB.X, epipole2 * minvB.Y, epipole2 * minvB.Z);
	}

	// b_i of H&Z Result 13.6 for one correspondence; false when the point lies (near) the
	// epipole, so the transfer is undefined.
	private static bool TransferCoefficient(in Matrix3d a, Vector3d epipole2, Vector2d point1, Vector2d point2, out double b)
	{
		Vector3d x1 = point1.Homogeneous();
		Vector3d x2 = point2.Homogeneous();
		Vector3d x2CrossE2 = x2.Cross(epipole2);
		double denom = x2CrossE2.SquaredNorm;
		if (denom < 1e-12)
		{
			b = 0;
			return false;
		}

		b = x2.Cross(a * x1).Dot(x2CrossE2) / denom;
		return true;
	}

	// The `evaluate_triplet` lambda of DetectSampleHDegeneracy. Returns true if all sample
	// points are consistent (nothing left to improve).
	private static bool EvaluateTriplet(
		in Matrix3d a,
		Vector3d epipole2,
		ReadOnlySpan<Vector2d> samplePoints1,
		ReadOnlySpan<Vector2d> samplePoints2,
		double hMaxResidual,
		int i,
		int j,
		int k,
		ref int bestNumConsistent,
		ref Matrix3d? bestH)
	{
		Matrix3d? h = HomographyFromCompatible(
			a, epipole2,
			samplePoints1[i], samplePoints1[j], samplePoints1[k],
			samplePoints2[i], samplePoints2[j], samplePoints2[k]);
		if (!h.HasValue)
		{
			return false;
		}

		int numSamples = samplePoints1.Length;
		int numConsistent = 0;
		for (int p = 0; p < numSamples; ++p)
		{
			if (HomographyMatrix.ComputeSquaredHomographyError(samplePoints1[p], samplePoints2[p], h.Value) <= hMaxResidual)
			{
				++numConsistent;
			}
		}

		if (numConsistent > bestNumConsistent)
		{
			bestNumConsistent = numConsistent;
			bestH = h;
		}

		return bestNumConsistent >= numSamples;
	}

	// Draws `num` distinct entries of `scratch` into its front via a partial Fisher-Yates
	// shuffle, with COLMAP's size_t draws (ulong here).
	private static void SampleDistinct(int num, List<ulong> scratch)
	{
		ulong n = (ulong)scratch.Count;
		for (int i = 0; i < num; ++i)
		{
			ulong j = (ulong)i + RandomUtils.RandomUniformInteger<ulong>(0, n - 1 - (ulong)i);
			(scratch[i], scratch[(int)j]) = (scratch[(int)j], scratch[i]);
		}
	}

	private static int[] BuildSampleTriplets()
	{
		// C++ spells the 35 triplets out; the nested loops produce the same lexicographic list.
		var triplets = new List<int>(35 * 3);
		for (int i = 0; i < 7; ++i)
		{
			for (int j = i + 1; j < 7; ++j)
			{
				for (int k = j + 1; k < 7; ++k)
				{
					triplets.Add(i);
					triplets.Add(j);
					triplets.Add(k);
				}
			}
		}

		return triplets.ToArray();
	}
}
