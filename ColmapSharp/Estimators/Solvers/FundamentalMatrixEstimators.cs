// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FundamentalMatrixEstimators: colmap/estimators/solvers/fundamental_matrix.h and .cc - the
// 7-point and 8-point fundamental matrix estimators (Optim/Estimator.cs), whose residual is
// the squared Sampson error of Geometry/EssentialMatrix.cs. The 8-point estimator normalizes
// with Geometry/Normalization.cs; the 7-point one finds the rank-2 members of the null-space
// pencil with Mathematics/Polynomial.cs's cubic solver. Consumers: the two-view geometry
// and DEGENSAC estimators.
// Tests: ColmapSharp.Tests/Estimators/Solvers/FundamentalMatrixTests.cs
// (fundamental_matrix_test.cc).
//
// Also here: RefineFundamentalMatrixSampson and FundamentalMatrixSampsonEstimator (the
// LO-RANSAC refiner), which run Optim/TinySolver.cs on
// Estimators/CostFunctions/TinySampsonError.cs over the factorized manifold of
// Estimators/CostFunctions/TinyManifold.cs.
//
// Tier B (tolerance): QR, SVD and a cubic. The refiner is Tier C (iterative).
//
// Signs of null vectors: every null vector here (7-point QR basis, 8-point QR or SVD vector,
// and the U, V of the rank-2 projection) has an arbitrary sign. The results are F up to
// scale: the 7-point models are normalized to unit norm with a sign that depends on the
// basis, the 8-point model's sign follows the null vector's. The squared Sampson error is
// invariant to F -> -F (its numerator and denominator are both quadratic in F), so RANSAC
// scores are sign-independent, and COLMAP's tests compare F / F(2, 2).
//
// Hot path: RANSAC calls the 7-point Estimate once per hypothesis, so it works on stackalloc
// buffers with the span kernels of LinearAlgebra/Householder.cs; so does the minimal
// 8-point case. The over-determined 8-point case allocates its N x 9 system.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Fundamental matrix estimator from corresponding point pairs. This algorithm solves the
/// 7-Point problem and is based on: Zhengyou Zhang and T. Kanade, Determining the Epipolar
/// Geometry and its Uncertainty: A Review, International Journal of Computer Vision, 1998.
/// Port of colmap::FundamentalMatrixSevenPointEstimator.
/// </summary>
/// <remarks>
/// COLMAP takes the 2D null space from Eigen's FullPivHouseholderQR; this uses unpivoted
/// Householder QR, which spans the same null space with a different basis. The solution set
/// is the same; see divergence 24, for the one degenerate edge where the
/// basis shows.
/// </remarks>
public readonly struct FundamentalMatrixSevenPointEstimator
	: IEstimator<Vector2d, Vector2d, Matrix3d>, ILocalEstimator<Vector2d, Vector2d, Matrix3d>
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 7;

	/// <summary>
	/// Estimate either 1 or 3 possible fundamental matrix solutions from exactly 7
	/// corresponding points.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, List<Matrix3d> models)
	{
		Check.Eq(points1.Length, 7);
		Check.Eq(points2.Length, 7);

		models.Clear();

		// Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'. A is 9 x 7,
		// column-major, one correspondence per column; entry 3c + r multiplies F(r, c).
		Span<double> a = stackalloc double[9 * 7];
		for (int i = 0; i < 7; ++i)
		{
			Span<double> col = a.Slice(9 * i, 9);
			Vector2d p1 = points1[i];
			Vector2d p2 = points2[i];
			col[0] = p1.X * p2.X;
			col[1] = p1.X * p2.Y;
			col[2] = p1.X;
			col[3] = p1.Y * p2.X;
			col[4] = p1.Y * p2.Y;
			col[5] = p1.Y;
			col[6] = p2.X;
			col[7] = p2.Y;
			col[8] = 1;
		}

		// 9 unknowns with 7 equations, so we have 2D null space: the last two columns of the
		// full Q of A = Q R are orthogonal to A's columns.
		Span<double> tau = stackalloc double[7];
		Householder.FactorInPlace(a, 9, 7, tau);
		Span<double> f1 = stackalloc double[9];
		Span<double> f2 = stackalloc double[9];
		f1.Clear();
		f2.Clear();
		f1[7] = 1;
		f2[8] = 1;
		Householder.ApplyQ(a, 9, tau, f1);
		Householder.ApplyQ(a, 9, tau, f2);

		// Normalize, such that lambda + mu = 1
		// and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).
		for (int k = 0; k < 9; ++k)
		{
			f1[k] -= f2[k];
		}

		double t0 = f1[4] * f1[8] - f1[5] * f1[7];
		double t1 = f1[3] * f1[8] - f1[5] * f1[6];
		double t2 = f1[3] * f1[7] - f1[4] * f1[6];
		double t3 = f2[4] * f2[8] - f2[5] * f2[7];
		double t4 = f2[3] * f2[8] - f2[5] * f2[6];
		double t5 = f2[3] * f2[7] - f2[4] * f2[6];

		double coeffs0 = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
		if (Math.Abs(coeffs0) < 1e-16)
		{
			return;
		}

		double coeffs1 = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
			f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
			f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
			f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
			f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
			f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
			f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);
		double coeffs2 = f1[0] * t3 - f1[1] * t4 + f1[2] * t5 -
			f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
			f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
			f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
			f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
			f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
			f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);
		double coeffs3 = f2[0] * t3 - f2[1] * t4 + f2[2] * t5;

		coeffs1 /= coeffs0;
		coeffs2 /= coeffs0;
		coeffs3 /= coeffs0;

		int numRoots = Polynomial.FindCubicPolynomialRoots(coeffs1, coeffs2, coeffs3, out Vector3d roots);

		Span<double> f = stackalloc double[9];
		for (int i = 0; i < numRoots; ++i)
		{
			double squaredNorm = 0;
			for (int k = 0; k < 9; ++k)
			{
				f[k] = f1[k] * roots[i] + f2[k];
				squaredNorm += f[k] * f[k];
			}

			// Eigen's normalized(): a zero vector is returned unchanged.
			if (squaredNorm > 0)
			{
				double norm = Math.Sqrt(squaredNorm);
				for (int k = 0; k < 9; ++k)
				{
					f[k] /= norm;
				}
			}

			// Eigen::Map<const Matrix3d>(F.data()): column-major.
			models.Add(Matrix3d.FromColumnMajor(f));
		}
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Estimate(points1, points2, models);
	}

	/// <summary>The squared Sampson error of each correspondence.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d f, Span<double> residuals)
	{
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, f, residuals);
	}
}

/// <summary>
/// Fundamental matrix estimator from corresponding point pairs. This algorithm solves the
/// 8-Point problem based on Hartley and Zisserman, Multiple View Geometry, algorithm 11.1,
/// page 282. Port of colmap::FundamentalMatrixEightPointEstimator.
/// </summary>
public readonly struct FundamentalMatrixEightPointEstimator
	: IEstimator<Vector2d, Vector2d, Matrix3d>, ILocalEstimator<Vector2d, Vector2d, Matrix3d>
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 8;

	/// <summary>
	/// Estimate the fundamental matrix from at least 8 corresponding points (one model).
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, List<Matrix3d> models)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Ge(points1.Length, 8);

		models.Clear();

		int numPoints = points1.Length;

		// Center and normalize image points for better numerical stability.
		Span<Vector2d> normedPoints1 = numPoints <= 64 ? stackalloc Vector2d[numPoints] : new Vector2d[numPoints];
		Span<Vector2d> normedPoints2 = numPoints <= 64 ? stackalloc Vector2d[numPoints] : new Vector2d[numPoints];
		Matrix3d normedFromOrig1 = Normalization.CenterAndNormalizeImagePoints(points1, normedPoints1);
		Matrix3d normedFromOrig2 = Normalization.CenterAndNormalizeImagePoints(points2, normedPoints2);

		// Setup homogeneous linear equation as x2' * F * x1 = 0; row i of A is
		// [x2 x1^T, y2 x1^T, x1^T] (x1 homogeneous), so entry 3r + c multiplies F(r, c).
		// Solve for the nullspace of the constraint matrix.
		Span<double> nullVector = stackalloc double[9];
		if (numPoints == 8)
		{
			// A^T is 9 x 8, column-major: column i is row i of A.
			Span<double> at = stackalloc double[9 * 8];
			for (int i = 0; i < 8; ++i)
			{
				SetRow(at.Slice(9 * i, 9), normedPoints1[i], normedPoints2[i]);
			}

			Span<double> tau = stackalloc double[8];
			Householder.FactorInPlace(at, 9, 8, tau);
			nullVector.Clear();
			nullVector[8] = 1;
			Householder.ApplyQ(at, 9, tau, nullVector);
		}
		else
		{
			var a = new MatrixXd(numPoints, 9);
			Span<double> row = stackalloc double[9];
			for (int i = 0; i < numPoints; ++i)
			{
				SetRow(row, normedPoints1[i], normedPoints2[i]);
				for (int c = 0; c < 9; ++c)
				{
					a[i, c] = row[c];
				}
			}

			var svd = new JacobiSVD(a, SvdOptions.ComputeFullV);
			svd.MatrixV().ColumnSpan(8).CopyTo(nullVector);
		}

		// Eigen::Map<const Matrix<double, 3, 3, RowMajor>>: the null vector holds Q row by row.
		var q = new Matrix3d(
			nullVector[0], nullVector[1], nullVector[2],
			nullVector[3], nullVector[4], nullVector[5],
			nullVector[6], nullVector[7], nullVector[8]);

		// Enforcing the internal constraint that two singular values must non-zero
		// and one must be zero.
		Svd3d svd3 = Svd3d.Compute(q);
		Vector3d singularValues = svd3.SingularValues;
		singularValues = new Vector3d(singularValues.X, singularValues.Y, 0.0);
		Matrix3d f = svd3.MatrixU * Matrix3d.FromDiagonal(singularValues) * svd3.MatrixV.Transpose();

		models.Add(normedFromOrig2.Transpose() * f * normedFromOrig1);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Estimate(points1, points2, models);
	}

	/// <summary>The squared Sampson error of each correspondence.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d f, Span<double> residuals)
	{
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, f, residuals);
	}

	private static void SetRow(Span<double> row, Vector2d p1, Vector2d p2)
	{
		row[0] = p2.X * p1.X;
		row[1] = p2.X * p1.Y;
		row[2] = p2.X;
		row[3] = p2.Y * p1.X;
		row[4] = p2.Y * p1.Y;
		row[5] = p2.Y;
		row[6] = p1.X;
		row[7] = p1.Y;
		row[8] = 1;
	}
}

/// <summary>The residual shared by the fundamental matrix estimators.</summary>
internal static class FundamentalMatrixResiduals
{
	/// <summary>
	/// Span form of colmap::ComputeSquaredSampsonError (Vector2d overload): the same
	/// per-point call on the homogeneous points.
	/// </summary>
	public static void SquaredSampsonError(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d f, Span<double> residuals)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Eq(residuals.Length, points1.Length);
		for (int i = 0; i < points1.Length; ++i)
		{
			residuals[i] = EssentialMatrix.ComputeSquaredSampsonError(points1[i].Homogeneous(), points2[i].Homogeneous(), f);
		}
	}
}

/// <summary>
/// Fundamental matrix refiner for use as the local estimator of LO-RANSAC. Provides Refine
/// rather than Estimate (C#: EstimateLocal refines a copy of the current best model), so
/// LO-RANSAC passes the current best model as the initial value, which a non-minimal solver
/// cannot use. Port of colmap::FundamentalMatrixSampsonEstimator.
/// </summary>
public readonly struct FundamentalMatrixSampsonEstimator : ILocalEstimator<Vector2d, Vector2d, Matrix3d>
{
	/// <summary>
	/// The minimum number of samples needed to refine a model. Refining an already
	/// determined model is only meaningful on an over-determined set.
	/// </summary>
	public static int MinNumSamples => 8;

	/// <summary>Refine f in place, see <see cref="FundamentalMatrix.RefineFundamentalMatrixSampson"/>.</summary>
	public static bool Refine(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, ref Matrix3d f) =>
		FundamentalMatrix.RefineFundamentalMatrixSampson(points1, points2, ref f);

	/// <summary>Refines a copy of the current best model and appends it on success.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Matrix3d f = initialModel;
		if (Refine(points1, points2, ref f))
		{
			models.Add(f);
		}
	}

	/// <summary>Squared Sampson error residuals, matching the estimators above.</summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d f, Span<double> residuals)
	{
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, f, residuals);
	}
}

/// <summary>The free functions of fundamental_matrix.cc.</summary>
public static class FundamentalMatrix
{
	// The 7-DoF manifold of a factorized fundamental matrix: the two rotations of its SVD on
	// SO(3), plus the singular value ratio as a 1-D Euclidean parameter. The ambient layout
	// matches TinyFundamentalSampsonErrorCostFunctor: [qU (xyzw), qV (xyzw), sigma].
	private static readonly TinyProductManifold<TinyEigenQuaternionManifold, TinyEigenQuaternionManifold, TinyEuclideanManifold1>
		FactorizedFundamentalMatrixManifold = new(default, default, default);

	/// <summary>
	/// Refine a fundamental matrix in place by minimizing the Sampson error over the given
	/// correspondences, starting from <paramref name="f"/>. Optimizes the SVD factorization
	/// of Bartoli and Sturm, "Non-Linear Estimation of the Fundamental Matrix With Minimal
	/// Parameters", PAMI 2004, which keeps rank 2 and the scale gauge exact at every iterate,
	/// unlike the 8-point algorithm, which truncates the smallest singular value after the
	/// fact. Points are centered and normalized internally, with a scale shared by both views
	/// to keep the cost in pixel space. The fit is a plain least squares, so the points are
	/// expected to be an inlier set; robustness comes from the surrounding RANSAC.
	/// Returns false and leaves <paramref name="f"/> unchanged if it cannot be factorized
	/// (zero or numerically rank 1) or if the solve leaves non-finite parameters.
	/// Port of colmap::RefineFundamentalMatrixSampson.
	/// </summary>
	public static bool RefineFundamentalMatrixSampson(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, ref Matrix3d f)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Ge(points1.Length, FundamentalMatrixSampsonEstimator.MinNumSamples);

		// Center and normalize image points for better numerical stability, as in the
		// eight-point estimator. On raw pixel coordinates the 7x7 normal equations
		// of the refinement are severely ill-conditioned.
		var normedPoints1 = new Vector2d[points1.Length];
		var normedPoints2 = new Vector2d[points2.Length];
		Matrix3d normedFromOrig1 = Normalization.CenterAndNormalizeImagePoints(points1, normedPoints1);
		Matrix3d normedFromOrig2 = Normalization.CenterAndNormalizeImagePoints(points2, normedPoints2);

		// Rescale both views by a common factor. The Sampson error is invariant to a
		// common scale, but not to two different ones, which would weight the two
		// terms of its denominator differently.
		double scale1 = normedFromOrig1[0, 0];
		double scale2 = normedFromOrig2[0, 0];
		double scale = Math.Sqrt(scale1 * scale2);
		normedFromOrig1 = RescaleNormalizedImagePoints(scale / scale1, normedPoints1, normedFromOrig1);
		normedFromOrig2 = RescaleNormalizedImagePoints(scale / scale2, normedPoints2, normedFromOrig2);

		// Inverse of the map that the eight-point estimator applies to its solution.
		Span<double> parameters = stackalloc double[9];
		if (!FactorizeFundamentalMatrix(normedFromOrig2.Transpose().Inverse() * f * normedFromOrig1.Inverse(), parameters))
		{
			return false;
		}

		// Plain least squares: the points are assumed to be the inlier set, so
		// robustness comes from the RANSAC inlier selection.
		var functor = new TinyFundamentalSampsonErrorCostFunctor(normedPoints1, normedPoints2);
		var solver = new TinySolver<TinyFundamentalSampsonErrorCostFunctor,
			TinyProductManifold<TinyEigenQuaternionManifold, TinyEigenQuaternionManifold, TinyEuclideanManifold1>>(
			FactorizedFundamentalMatrixManifold);
		var options = new TinySolverOptions { MaxNumIterations = 25 };
		solver.Solve(functor, parameters, options);

		if (!AllFinite(parameters))
		{
			return false;
		}

		f = normedFromOrig2.Transpose() *
			TinyFundamentalSampsonErrorCostFunctor.FundamentalFromParams(parameters) *
			normedFromOrig1;
		return true;
	}

	/// <summary>
	/// Factorize F = U diag(1, sigma, 0) V^T into [qU, qV, sigma], dropping any rank-3
	/// component as the eight-point estimator does. U and V are sign corrected to proper
	/// rotations so they can be represented by quaternions, which at most negates F and
	/// leaves the Sampson error unchanged. Port of the anonymous FactorizeFundamentalMatrix.
	/// </summary>
	internal static bool FactorizeFundamentalMatrix(in Matrix3d f, Span<double> parameters)
	{
		Svd3d svd = Svd3d.Compute(f);
		Vector3d singularValues = svd.SingularValues;

		// A zero or numerically rank-1 matrix has no meaningful factorization, as
		// U's second column is arbitrary once the second singular value vanishes.
		const double kMinSingularValueRatio = 1e-12;
		if (!(singularValues.X > 0) || singularValues.Y <= kMinSingularValueRatio * singularValues.X)
		{
			return false;
		}

		Matrix3d u = svd.MatrixU;
		Matrix3d v = svd.MatrixV;
		if (u.Determinant() < 0)
		{
			u = -u;
		}

		if (v.Determinant() < 0)
		{
			v = -v;
		}

		Vector4d qU = Quaterniond.FromRotationMatrix(u).Normalized().Coeffs;
		Vector4d qV = Quaterniond.FromRotationMatrix(v).Normalized().Coeffs;
		parameters[0] = qU.X;
		parameters[1] = qU.Y;
		parameters[2] = qU.Z;
		parameters[3] = qU.W;
		parameters[4] = qV.X;
		parameters[5] = qV.Y;
		parameters[6] = qV.Z;
		parameters[7] = qV.W;
		parameters[8] = singularValues.Y / singularValues.X;
		return AllFinite(parameters);
	}

	/// <summary>
	/// Rescale a normalizing transform and its points, keeping the centroid. Returns the
	/// rescaled transform (its top two rows times <paramref name="factor"/>).
	/// </summary>
	private static Matrix3d RescaleNormalizedImagePoints(double factor, Span<Vector2d> normedPoints, in Matrix3d normedFromOrig)
	{
		for (int i = 0; i < normedPoints.Length; ++i)
		{
			normedPoints[i] *= factor;
		}

		Matrix3d m = normedFromOrig;
		return new Matrix3d(
			m[0, 0] * factor, m[0, 1] * factor, m[0, 2] * factor,
			m[1, 0] * factor, m[1, 1] * factor, m[1, 2] * factor,
			m[2, 0], m[2, 1], m[2, 2]);
	}

	private static bool AllFinite(ReadOnlySpan<double> values)
	{
		foreach (double value in values)
		{
			if (!double.IsFinite(value))
			{
				return false;
			}
		}

		return true;
	}
}
