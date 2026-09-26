// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AffineTransform: colmap/estimators/solvers/affine_transform.h and .cc - the 2D affine
// transform estimator (least squares from at least 3 correspondences, an
// Optim/Estimator.cs implementation) and EstimateAffine2d / EstimateAffine2dRobust (its
// LO-RANSAC wrapper). COLMAP's consumer is retrieval/vote_and_verify.cc; its 3D siblings
// live in SimilarityTransform.cs.
// Tests: ColmapSharp.Tests/Estimators/Solvers/AffineTransformTests.cs
// (affine_transform_test.cc 1:1).
//
// Tier B (tolerance): an LU solve (minimal case) or an SVD least-squares solve.
//
// Hot path: RANSAC calls Estimate once per hypothesis with 3 correspondences, so that case
// solves its 6 x 6 system with PartialPivLU's span kernel on stackalloc buffers; only the
// over-determined (local optimization) case allocates.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// 2D affine transform estimator; the model is the 2x3 matrix [A | t] mapping source to
/// target. Port of colmap::AffineTransformEstimator.
/// </summary>
public readonly struct AffineTransformEstimator
	: IEstimator<Vector2d, Vector2d, Matrix2x3d>, ILocalEstimator<Vector2d, Vector2d, Matrix2x3d>
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 3;

	/// <summary>
	/// Estimate the affine transformation from at least 3 correspondences. Appends no model
	/// for a degenerate configuration.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> src, ReadOnlySpan<Vector2d> tgt, List<Matrix2x3d> tgtFromSrc)
	{
		int numPoints = src.Length;
		Check.Eq(numPoints, tgt.Length);
		Check.Ge(numPoints, 3);

		tgtFromSrc.Clear();

		// Sets up the linear system that we solve to obtain a least squared solution
		// for the affine transformation. Row 2i is [x y 1 0 0 0] -> tgt.x, row 2i+1 is
		// [0 0 0 x y 1] -> tgt.y, so the solution holds the model row by row.
		Span<double> sol = stackalloc double[6];
		if (numPoints == 3)
		{
			const int n = 6;
			Span<double> a = stackalloc double[n * n];
			Span<double> b = stackalloc double[n];
			for (int i = 0; i < numPoints; ++i)
			{
				SetRows(a, n, b, i, src[i], tgt[i]);
			}

			Span<int> permutation = stackalloc int[n];
			PartialPivLU.FactorInPlace(a, n, permutation);
			PartialPivLU.SolveInPlace(a, n, permutation, b, sol);
			foreach (double value in sol)
			{
				if (double.IsNaN(value))
				{
					return;
				}
			}
		}
		else
		{
			int rows = 2 * numPoints;
			var a = new MatrixXd(rows, 6);
			var b = new VectorXd(rows);
			for (int i = 0; i < numPoints; ++i)
			{
				SetRows(a.AsSpan(), rows, b.AsSpan(), i, src[i], tgt[i]);
			}

			var svd = new JacobiSVD(a, SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
			if (svd.Rank() < 6)
			{
				return;
			}

			svd.Solve(b).AsSpan().CopyTo(sol);
		}

		// Eigen::Map<const Matrix<double, 3, 2>>(sol.data()).transpose(): sol is row-major 2x3.
		tgtFromSrc.Add(new Matrix2x3d(sol[0], sol[1], sol[2], sol[3], sol[4], sol[5]));
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> src, ReadOnlySpan<Vector2d> tgt, in Matrix2x3d initialModel, List<Matrix2x3d> models)
	{
		Estimate(src, tgt, models);
	}

	/// <summary>The squared transformation error of each correspondence.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> src, ReadOnlySpan<Vector2d> tgt, in Matrix2x3d tgtFromSrc, Span<double> residuals)
	{
		int numPoints = src.Length;
		Check.Eq(numPoints, tgt.Length);
		Check.Eq(residuals.Length, numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			residuals[i] = (tgt[i] - tgtFromSrc * src[i].Homogeneous()).SquaredNorm;
		}
	}

	/// <summary>Writes rows 2i and 2i+1 of the column-major system A (rows x 6) and b.</summary>
	private static void SetRows(Span<double> a, int rows, Span<double> b, int i, Vector2d src, Vector2d tgt)
	{
		int r0 = 2 * i;
		int r1 = r0 + 1;
		Span<double> h = [src.X, src.Y, 1];
		for (int k = 0; k < 3; ++k)
		{
			a[k * rows + r0] = h[k];
			a[(3 + k) * rows + r0] = 0;
			a[k * rows + r1] = 0;
			a[(3 + k) * rows + r1] = h[k];
		}

		b[r0] = tgt.X;
		b[r1] = tgt.Y;
	}
}

/// <summary>EstimateAffine2d and EstimateAffine2dRobust (affine_transform.cc).</summary>
public static class AffineTransform
{
	/// <summary>
	/// Least-squares affine transform from at least 3 point pairs. Returns false (and leaves
	/// <paramref name="tgtFromSrc"/> unchanged) for degenerate input. Port of
	/// colmap::EstimateAffine2d.
	/// </summary>
	public static bool EstimateAffine2d(ReadOnlySpan<Vector2d> src, ReadOnlySpan<Vector2d> tgt, ref Matrix2x3d tgtFromSrc)
	{
		var models = new List<Matrix2x3d>();
		new AffineTransformEstimator().Estimate(src, tgt, models);
		if (models.Count == 0)
		{
			return false;
		}

		Check.Eq(models.Count, 1);
		tgtFromSrc = models[0];
		return true;
	}

	/// <summary>
	/// LO-RANSAC affine transform; <paramref name="tgtFromSrc"/> is assigned only on success.
	/// Port of colmap::EstimateAffine2dRobust.
	/// </summary>
	public static RansacReport<Matrix2x3d, InlierSupportMeasurer.Support> EstimateAffine2dRobust(
		ReadOnlySpan<Vector2d> src,
		ReadOnlySpan<Vector2d> tgt,
		RansacOptions options,
		ref Matrix2x3d tgtFromSrc)
	{
		var ransac = new LoRansac<AffineTransformEstimator, AffineTransformEstimator, Vector2d, Vector2d, Matrix2x3d>(
			options, new AffineTransformEstimator(), new AffineTransformEstimator());
		var report = ransac.Estimate(src, tgt);
		if (report.Success)
		{
			tgtFromSrc = report.Model;
		}

		return report;
	}
}
