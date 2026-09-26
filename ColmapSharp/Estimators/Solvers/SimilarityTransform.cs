// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SimilarityTransform: colmap/estimators/solvers/similarity_transform.h and .cc (plus the
// EuclideanTransformEstimator alias of euclidean_transform.h) - the 3D similarity and rigid
// transform estimators (least-squares alignment of corresponding points) and
// EstimateRigid3d/EstimateSim3d with their LO-RANSAC robust variants. The first
// IEstimator implementation (Optim/Estimator.cs), and the estimator that ransac_test.cc and
// loransac_test.cc run RANSAC with. Results feed Geometry/Rigid3d.cs and Sim3d.cs.
// Tests: ColmapSharp.Tests/Estimators/Solvers/SimilarityTransformTests.cs
// (similarity_transform_test.cc 1:1).
//
// Tier B (tolerance): the solution goes through a 3x3 SVD.
//
// The least-squares solution is written from the paper, not from Eigen::umeyama (Eigen is
// MPL-2.0 and is not ported): S. Umeyama, "Least-Squares Estimation of Transformation
// Parameters Between Two Point Patterns", IEEE TPAMI 13(4), 1991, Theorem and eqs. (34)-(42).
// With means mu_x, mu_y, source variance sigma_x^2 = 1/n sum |x - mu_x|^2 and covariance
// Sigma_xy = 1/n sum (y - mu_y)(x - mu_x)^T = U D V^T: R = U S V^T, c = tr(D S) / sigma_x^2,
// t = mu_y - c R mu_x, where S = diag(1, 1, -1) when det(U) det(V) < 0 and I otherwise. That
// sign test is the paper's rule (det(Sigma_xy) < 0, or det(U) det(V) = -1 when Sigma_xy has
// rank m - 1) stated so it also covers the rank-deficient case, as COLMAP's Eigen call does.
//
// Translation notes:
// - Only kDim = 3 is ported: it is the only dimension COLMAP instantiates (the
//   EuclideanTransformEstimator alias is instantiated nowhere). The kEstimateScale template
//   flag becomes two structs, SimilarityTransformEstimator3d (<3, true>) and
//   EuclideanTransformEstimator3d (<3, false>), sharing one solver.
// - The rank test of the 3 x N point matrices uses FullPivLU, as COLMAP does.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// 3D similarity transform estimator from corresponding point pairs in the source and
/// destination coordinate systems (Umeyama 1991). Port of
/// colmap::SimilarityTransformEstimator&lt;3, true&gt;. The model is the 3x4 matrix
/// [c R | t] mapping source to target.
/// </summary>
public readonly struct SimilarityTransformEstimator3d
	: IEstimator<Vector3d, Vector3d, Matrix3x4d>, ILocalEstimator<Vector3d, Vector3d, Matrix3x4d>
{
	/// <summary>
	/// The minimum number of samples needed to estimate a model. Note that this only returns
	/// the true minimal sample in the two-dimensional case. For higher dimensions, the system
	/// will always be over-determined.
	/// </summary>
	public static int MinNumSamples => 3;

	/// <summary>Estimate the similarity transform tgt_from_src.</summary>
	public void Estimate(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, List<Matrix3x4d> models)
	{
		SimilarityTransformSolver.Estimate(src, tgt, estimateScale: true, models);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, in Matrix3x4d initialModel, List<Matrix3x4d> models)
	{
		Estimate(src, tgt, models);
	}

	/// <summary>
	/// The squared transformation error of each pair when transforming the source to the
	/// destination coordinates.
	/// </summary>
	public void Residuals(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, in Matrix3x4d tgtFromSrc, Span<double> residuals)
	{
		SimilarityTransformSolver.Residuals(src, tgt, tgtFromSrc, residuals);
	}
}

/// <summary>
/// 3D rigid transform estimator (no scale). Port of colmap::SimilarityTransformEstimator&lt;3,
/// false&gt;, which COLMAP also names EuclideanTransformEstimator&lt;3&gt;. The model is the
/// 3x4 matrix [R | t] mapping source to target.
/// </summary>
public readonly struct EuclideanTransformEstimator3d
	: IEstimator<Vector3d, Vector3d, Matrix3x4d>, ILocalEstimator<Vector3d, Vector3d, Matrix3x4d>
{
	/// <inheritdoc cref="SimilarityTransformEstimator3d.MinNumSamples"/>
	public static int MinNumSamples => 3;

	/// <summary>Estimate the rigid transform tgt_from_src.</summary>
	public void Estimate(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, List<Matrix3x4d> models)
	{
		SimilarityTransformSolver.Estimate(src, tgt, estimateScale: false, models);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, in Matrix3x4d initialModel, List<Matrix3x4d> models)
	{
		Estimate(src, tgt, models);
	}

	/// <inheritdoc cref="SimilarityTransformEstimator3d.Residuals"/>
	public void Residuals(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, in Matrix3x4d tgtFromSrc, Span<double> residuals)
	{
		SimilarityTransformSolver.Residuals(src, tgt, tgtFromSrc, residuals);
	}
}

/// <summary>
/// EstimateRigid3d / EstimateSim3d and their robust variants
/// (colmap/estimators/solvers/similarity_transform.cc).
/// </summary>
public static class SimilarityTransform
{
	/// <summary>
	/// Least-squares rigid transform from at least 3 point pairs. Returns false (and leaves
	/// <paramref name="tgtFromSrc"/> unchanged) for degenerate input. Port of
	/// colmap::EstimateRigid3d.
	/// </summary>
	public static bool EstimateRigid3d(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, ref Rigid3d tgtFromSrc)
	{
		if (!EstimateRigidOrSim3d<EuclideanTransformEstimator3d>(src, tgt, out Matrix3x4d matrix))
		{
			return false;
		}

		tgtFromSrc = Rigid3d.FromMatrix(matrix);
		return true;
	}

	/// <summary>
	/// LO-RANSAC rigid transform. As in COLMAP, <paramref name="tgtFromSrc"/> is assigned
	/// even when the estimation fails (from a zero matrix). Port of
	/// colmap::EstimateRigid3dRobust.
	/// </summary>
	public static RansacReport<Matrix3x4d, InlierSupportMeasurer.Support> EstimateRigid3dRobust(
		ReadOnlySpan<Vector3d> src,
		ReadOnlySpan<Vector3d> tgt,
		RansacOptions options,
		out Rigid3d tgtFromSrc)
	{
		var report = EstimateRigidOrSim3dRobust<EuclideanTransformEstimator3d>(src, tgt, options, out Matrix3x4d matrix);
		tgtFromSrc = Rigid3d.FromMatrix(matrix);
		return report;
	}

	/// <summary>
	/// Least-squares similarity transform from at least 3 point pairs. Returns false (and
	/// leaves <paramref name="tgtFromSrc"/> unchanged) for degenerate input. Port of
	/// colmap::EstimateSim3d.
	/// </summary>
	public static bool EstimateSim3d(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, ref Sim3d tgtFromSrc)
	{
		if (!EstimateRigidOrSim3d<SimilarityTransformEstimator3d>(src, tgt, out Matrix3x4d matrix))
		{
			return false;
		}

		tgtFromSrc = Sim3d.FromMatrix(matrix);
		return true;
	}

	/// <summary>
	/// LO-RANSAC similarity transform; <paramref name="tgtFromSrc"/> is assigned only on
	/// success. Port of colmap::EstimateSim3dRobust.
	/// </summary>
	public static RansacReport<Matrix3x4d, InlierSupportMeasurer.Support> EstimateSim3dRobust(
		ReadOnlySpan<Vector3d> src,
		ReadOnlySpan<Vector3d> tgt,
		RansacOptions options,
		ref Sim3d tgtFromSrc)
	{
		var report = EstimateRigidOrSim3dRobust<SimilarityTransformEstimator3d>(src, tgt, options, out Matrix3x4d matrix);
		if (report.Success)
		{
			tgtFromSrc = Sim3d.FromMatrix(matrix);
		}

		return report;
	}

	private static bool EstimateRigidOrSim3d<TEstimator>(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, out Matrix3x4d tgtFromSrc)
		where TEstimator : struct, IEstimator<Vector3d, Vector3d, Matrix3x4d>
	{
		tgtFromSrc = Matrix3x4d.Zero;
		var models = new List<Matrix3x4d>();
		new TEstimator().Estimate(src, tgt, models);
		if (models.Count == 0)
		{
			return false;
		}

		Check.Eq(models.Count, 1);
		tgtFromSrc = models[0];
		return true;
	}

	private static RansacReport<Matrix3x4d, InlierSupportMeasurer.Support> EstimateRigidOrSim3dRobust<TEstimator>(
		ReadOnlySpan<Vector3d> src,
		ReadOnlySpan<Vector3d> tgt,
		RansacOptions options,
		out Matrix3x4d tgtFromSrc)
		where TEstimator : struct, IEstimator<Vector3d, Vector3d, Matrix3x4d>, ILocalEstimator<Vector3d, Vector3d, Matrix3x4d>
	{
		tgtFromSrc = Matrix3x4d.Zero;
		var ransac = new LoRansac<TEstimator, TEstimator, Vector3d, Vector3d, Matrix3x4d>(options, new TEstimator(), new TEstimator());
		var report = ransac.Estimate(src, tgt);
		if (report.Success)
		{
			tgtFromSrc = report.Model;
		}

		return report;
	}
}

/// <summary>The Umeyama least-squares solver shared by the two estimator structs.</summary>
internal static class SimilarityTransformSolver
{
	private const int Dim = 3;

	/// <summary>
	/// Port of SimilarityTransformEstimator&lt;3, kEstimateScale&gt;::Estimate: no model when
	/// either point set has rank below 3 or the solution has a NaN.
	/// </summary>
	public static void Estimate(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, bool estimateScale, List<Matrix3x4d> models)
	{
		Check.Eq(src.Length, tgt.Length);
		Check.Ge(src.Length, Dim);

		models.Clear();

		if (new FullPivLU(PointMatrix(src)).Rank() < Dim || new FullPivLU(PointMatrix(tgt)).Rank() < Dim)
		{
			return;
		}

		Matrix3x4d sol = Umeyama(src, tgt, estimateScale);

		if (HasNaN(sol))
		{
			return;
		}

		models.Add(sol);
	}

	/// <summary>Port of SimilarityTransformEstimator::Residuals: |tgt - M [src; 1]|^2.</summary>
	public static void Residuals(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, in Matrix3x4d tgtFromSrc, Span<double> residuals)
	{
		int numPoints = src.Length;
		Check.Eq(numPoints, tgt.Length);
		Check.Eq(residuals.Length, numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			residuals[i] = (tgt[i] - tgtFromSrc * src[i].Homogeneous()).SquaredNorm;
		}
	}

	/// <summary>
	/// Umeyama's least-squares [c R | t] with tgt ~ c R src + t (c = 1 without scale); see
	/// the file header for the equations.
	/// </summary>
	private static Matrix3x4d Umeyama(ReadOnlySpan<Vector3d> src, ReadOnlySpan<Vector3d> tgt, bool estimateScale)
	{
		int n = src.Length;
		double oneOverN = 1.0 / n;

		Vector3d srcSum = Vector3d.Zero;
		Vector3d tgtSum = Vector3d.Zero;
		for (int i = 0; i < n; ++i)
		{
			srcSum += src[i];
			tgtSum += tgt[i];
		}

		Vector3d srcMean = srcSum * oneOverN;
		Vector3d tgtMean = tgtSum * oneOverN;

		// Sigma_xy = 1/n sum (y - mu_y)(x - mu_x)^T, accumulated column-major.
		Span<double> sigma = stackalloc double[9];
		double srcVarianceSum = 0;
		for (int i = 0; i < n; ++i)
		{
			Vector3d xs = src[i] - srcMean;
			Vector3d ys = tgt[i] - tgtMean;
			srcVarianceSum += xs.SquaredNorm;
			for (int col = 0; col < Dim; ++col)
			{
				for (int row = 0; row < Dim; ++row)
				{
					sigma[col * Dim + row] += ys[row] * xs[col];
				}
			}
		}

		for (int k = 0; k < 9; ++k)
		{
			sigma[k] *= oneOverN;
		}

		double srcVariance = srcVarianceSum * oneOverN;

		Svd3d svd = Svd3d.Compute(Matrix3d.FromColumnMajor(sigma));
		Matrix3d u = svd.MatrixU;
		Matrix3d v = svd.MatrixV;

		// S = diag(1, 1, -1) when U V^T would be a reflection.
		Vector3d s = new Vector3d(1, 1, u.Determinant() * v.Determinant() < 0 ? -1 : 1);

		Matrix3d rotation = u * Matrix3d.FromDiagonal(s) * v.Transpose();

		double scale = 1;
		if (estimateScale)
		{
			scale = svd.SingularValues.Dot(s) / srcVariance;
		}

		Vector3d translation = tgtMean - scale * (rotation * srcMean);
		return Matrix3x4d.FromBlocks(rotation * scale, translation);
	}

	/// <summary>The 3 x N matrix whose columns are the points.</summary>
	private static MatrixXd PointMatrix(ReadOnlySpan<Vector3d> points)
	{
		var matrix = new MatrixXd(Dim, points.Length);
		for (int j = 0; j < points.Length; ++j)
		{
			matrix[0, j] = points[j].X;
			matrix[1, j] = points[j].Y;
			matrix[2, j] = points[j].Z;
		}

		return matrix;
	}

	private static bool HasNaN(in Matrix3x4d m)
	{
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				if (double.IsNaN(m[row, col]))
				{
					return true;
				}
			}
		}

		return false;
	}
}
