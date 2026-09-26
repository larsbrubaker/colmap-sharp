// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EssentialMatrixEstimators: colmap/estimators/solvers/essential_matrix.h and .cc - the
// five-point and eight-point essential matrix estimators on camera rays, and
// EssentialMatrixTangentSampsonEstimator, which scores in pixels through the rays'
// unprojection Jacobians and refines with Optim/TinySolver.cs on
// Estimators/CostFunctions/TinyRelativePoseSampsonError.cs (the LO-RANSAC local optimizer).
// The minimal five-point case is PoseLib's relpose_5pt (PoseLib/Relpose5pt.cs); the
// over-determined case is COLMAP's own Nister solver, whose generated polynomial code lives
// in EssentialMatrixPolynomials.*.cs (scripts/generate-essential-matrix-poly.py). Consumers:
// the two-view geometry estimators. Tests:
// ColmapSharp.Tests/Estimators/Solvers/EssentialMatrixTests.cs (essential_matrix_test.cc).
//
// Tier B (QR, SVD, LU, a companion-matrix eigen solve); the refiner is Tier C.
//
// Signs: every null vector (QR, SVD) has an arbitrary sign, so E is determined up to sign,
// which the (tangent) Sampson errors and COLMAP's tests (min of |E - E*| and |E + E*|)
// ignore.
//
// Hot path: RANSAC calls the five-point Estimate once per hypothesis. The minimal case runs
// on stackalloc buffers; the cheirality filter allocates two 5-element ray arrays and an
// index list per call, because Geometry/EssentialMatrix.PoseFromEssentialMatrix takes
// lists. The over-determined and eight-point cases allocate their N x 9 system.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Essential matrix estimator from corresponding normalized camera ray pairs. This algorithm
/// solves the 5-Point problem based on D. Nister, An efficient solution to the five-point
/// relative pose problem, IEEE-T-PAMI, 26(6), 2004.
/// Port of colmap::EssentialMatrixFivePointEstimator.
/// </summary>
/// <remarks>
/// This minimal solver provides no residual in COLMAP (the former Sampson-on-unit-bearings
/// residual was retired); inlier scoring is done in pixel units by
/// <see cref="EssentialMatrixTangentSampsonEstimator"/>. So it is not an IEstimator.
/// </remarks>
public readonly struct EssentialMatrixFivePointEstimator
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 5;

	/// <summary>
	/// Estimate up to 10 possible essential matrix solutions from a set of corresponding
	/// camera rays (at least 5). Clears <paramref name="models"/> first.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<Vector3d> camRays2, List<Matrix3d> models)
	{
		Check.Eq(camRays1.Length, camRays2.Length);
		Check.Ge(camRays1.Length, MinNumSamples);
		models.Clear();

		// PoseLib's 5-point solver only supports the minimal case. The non-minimal case falls
		// through to the SVD-based solver below.
		if (camRays1.Length == MinNumSamples)
		{
			EstimateMinimal(camRays1, camRays2, models);
			return;
		}

		EstimateNonMinimal(camRays1, camRays2, models);
	}

	private static void EstimateMinimal(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<Vector3d> camRays2, List<Matrix3d> models)
	{
		var candidateModels = new List<Matrix3d>(10);
		Relpose5pt.Solve(camRays1, camRays2, candidateModels);

		// Keep only hypotheses whose minimal sample is in front of both cameras, pruning
		// geometrically invalid essential matrices before they are scored.
		Vector3d[] rays1 = camRays1.ToArray();
		Vector3d[] rays2 = camRays2.ToArray();
		var validIndices = new List<int>(MinNumSamples);
		foreach (Matrix3d candidateModel in candidateModels)
		{
			EssentialMatrix.PoseFromEssentialMatrix(candidateModel, rays1, rays2, out _, validIndices);
			if (validIndices.Count == MinNumSamples)
			{
				models.Add(candidateModel);
			}
		}
	}

	private static void EstimateNonMinimal(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<Vector3d> camRays2, List<Matrix3d> models)
	{
		// Setup system of equations: cam_rays2(i)' * E * cam_rays1(i) = 0.
		MatrixXd q = EssentialMatrixSystem.ConstraintMatrix(camRays1, camRays2);

		// Step 1: Extraction of the nullspace. The minimal case is handled by PoseLib above,
		// so we always reach this with an over-determined system. E (9 x 4, column-major)
		// holds the right singular vectors of the four smallest singular values.
		var svd = new JacobiSVD(q, SvdOptions.ComputeFullV);
		MatrixXd v = svd.MatrixV();
		Span<double> e = stackalloc double[36];
		for (int col = 0; col < 4; ++col)
		{
			v.ColumnSpan(5 + col).CopyTo(e.Slice(9 * col, 9));
		}

		// Step 2: Gauss-Jordan elimination with partial pivoting on A.
		Span<double> e2 = stackalloc double[36];
		Span<double> e3 = stackalloc double[36];
		for (int i = 0; i < 36; ++i)
		{
			e2[i] = e[i] * e[i];
			e3[i] = e2[i] * e[i];
		}

		Span<double> a = stackalloc double[10 * 20];
		EssentialMatrixPolynomials.FillPolynomialMatrixPart1(a, e, e2, e3);
		EssentialMatrixPolynomials.FillPolynomialMatrixPart2(a, e, e2, e3);
		EssentialMatrixPolynomials.FillPolynomialMatrixPart3(a, e, e2, e3);

		// AA = A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10)); both
		// blocks are contiguous column-major spans.
		Span<double> lu = stackalloc double[100];
		a[..100].CopyTo(lu);
		Span<int> permutation = stackalloc int[10];
		PartialPivLU.FactorInPlace(lu, 10, permutation);
		Span<double> aa = stackalloc double[100];
		for (int col = 0; col < 10; ++col)
		{
			PartialPivLU.SolveInPlace(lu, 10, permutation, a.Slice(100 + 10 * col, 10), aa.Slice(10 * col, 10));
		}

		// Step 3: Expansion of the determinant polynomial of the 3x3 polynomial matrix B to
		// obtain the tenth degree polynomial. B is 13 x 3, column-major.
		Span<double> b = stackalloc double[13 * 3];
		for (int i = 0; i < 3; ++i)
		{
			Span<double> bCol = b.Slice(13 * i, 13);
			bCol[0] = 0;
			bCol[4] = 0;
			bCol[8] = 0;
			for (int k = 0; k < 3; ++k)
			{
				bCol[1 + k] = aa[(i * 2 + 4) + 10 * k];
				bCol[5 + k] = aa[(i * 2 + 4) + 10 * (3 + k)];
			}

			for (int k = 0; k < 4; ++k)
			{
				bCol[9 + k] = aa[(i * 2 + 4) + 10 * (6 + k)];
			}

			for (int k = 0; k < 3; ++k)
			{
				bCol[k] -= aa[(i * 2 + 5) + 10 * k];
				bCol[4 + k] -= aa[(i * 2 + 5) + 10 * (3 + k)];
			}

			for (int k = 0; k < 4; ++k)
			{
				bCol[8 + k] -= aa[(i * 2 + 5) + 10 * (6 + k)];
			}
		}

		// Step 4: Extraction of roots from the degree 10 polynomial.
		var coeffs = new VectorXd(11);
		Span<double> coeffValues = stackalloc double[11];
		EssentialMatrixPolynomials.ComputeDeterminantCoefficients(b, coeffValues);
		for (int i = 0; i < 11; ++i)
		{
			coeffs[i] = coeffValues[i];
		}

		if (!Polynomial.FindPolynomialRootsCompanionMatrix(coeffs, out VectorXd rootsReal, out VectorXd rootsImag))
		{
			return;
		}

		int numRoots = rootsReal.Length;
		Span<double> model = stackalloc double[9];
		Span<double> bz = stackalloc double[9];
		for (int i = 0; i < numRoots; ++i)
		{
			const double kMaxRootImag = 1e-10;
			if (Math.Abs(rootsImag[i]) > kMaxRootImag)
			{
				continue;
			}

			double z1 = rootsReal[i];
			double z2 = z1 * z1;
			double z3 = z2 * z1;
			double z4 = z3 * z1;

			// Bz(j, c) from column j of B.
			for (int j = 0; j < 3; ++j)
			{
				ReadOnlySpan<double> bj = b.Slice(13 * j, 13);
				bz[3 * j] = bj[0] * z3 + bj[1] * z2 + bj[2] * z1 + bj[3];
				bz[3 * j + 1] = bj[4] * z3 + bj[5] * z2 + bj[6] * z1 + bj[7];
				bz[3 * j + 2] = bj[8] * z4 + bj[9] * z3 + bj[10] * z2 + bj[11] * z1 + bj[12];
			}

			// bz holds Bz row by row.
			var bzMatrix = new Matrix3d(bz[0], bz[1], bz[2], bz[3], bz[4], bz[5], bz[6], bz[7], bz[8]);
			Vector3d x = Svd3d.Compute(bzMatrix).MatrixV.Col(2);

			const double kMaxX3 = 1e-10;
			if (Math.Abs(x.Z) < kMaxX3)
			{
				continue;
			}

			double squaredNorm = 0;
			for (int k = 0; k < 9; ++k)
			{
				model[k] = e[k] * (x.X / x.Z) + e[9 + k] * (x.Y / x.Z) + e[18 + k] * z1 + e[27 + k];
				squaredNorm += model[k] * model[k];
			}

			// Eigen's normalized(): a zero vector is returned unchanged.
			if (squaredNorm > 0)
			{
				double norm = Math.Sqrt(squaredNorm);
				for (int k = 0; k < 9; ++k)
				{
					model[k] /= norm;
				}
			}

			// Eigen::Map<const Matrix<double, 3, 3, RowMajor>>(e.data()).
			models.Add(new Matrix3d(model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7], model[8]));
		}
	}
}

/// <summary>
/// Essential matrix estimator from corresponding normalized camera ray pairs. This algorithm
/// solves the 8-Point problem based on Hartley and Zisserman, Multiple View Geometry,
/// algorithm 11.1, page 282. Port of colmap::EssentialMatrixEightPointEstimator.
/// </summary>
/// <remarks>No residual, as for <see cref="EssentialMatrixFivePointEstimator"/>.</remarks>
public readonly struct EssentialMatrixEightPointEstimator
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 8;

	/// <summary>
	/// Estimate the essential matrix from at least 8 corresponding camera rays (one model).
	/// Clears <paramref name="models"/> first.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<Vector3d> camRays2, List<Matrix3d> models)
	{
		Check.Eq(camRays1.Length, camRays2.Length);
		Check.Ge(camRays1.Length, 8);
		models.Clear();

		// Setup homogeneous linear equation as x2' * E * x1 = 0. Row i, entry 3r + c,
		// multiplies E(r, c). Solve for the nullspace of the constraint matrix.
		Span<double> nullVector = stackalloc double[9];
		if (camRays1.Length == 8)
		{
			// A^T is 9 x 8, column-major: column i is row i of A. The last column of the full
			// Q of A^T = Q R is orthogonal to every constraint.
			Span<double> at = stackalloc double[9 * 8];
			for (int i = 0; i < 8; ++i)
			{
				EssentialMatrixSystem.SetRow(at.Slice(9 * i, 9), camRays1[i], camRays2[i]);
			}

			Span<double> tau = stackalloc double[8];
			Householder.FactorInPlace(at, 9, 8, tau);
			nullVector.Clear();
			nullVector[8] = 1;
			Householder.ApplyQ(at, 9, tau, nullVector);
		}
		else
		{
			var svd = new JacobiSVD(EssentialMatrixSystem.ConstraintMatrix(camRays1, camRays2), SvdOptions.ComputeFullV);
			svd.MatrixV().ColumnSpan(8).CopyTo(nullVector);
		}

		// Eigen::Map<const Matrix<double, 3, 3, RowMajor>>: the null vector holds Q row by row.
		var q = new Matrix3d(
			nullVector[0], nullVector[1], nullVector[2],
			nullVector[3], nullVector[4], nullVector[5],
			nullVector[6], nullVector[7], nullVector[8]);

		// Enforcing the internal constraint that two singular values must be non-zero and
		// one must be zero.
		Svd3d svd3 = Svd3d.Compute(q);
		Vector3d singularValues = svd3.SingularValues;
		singularValues = new Vector3d(singularValues.X, singularValues.Y, 0.0);
		models.Add(svd3.MatrixU * Matrix3d.FromDiagonal(singularValues) * svd3.MatrixV.Transpose());
	}
}

/// <summary>
/// Essential matrix estimator scoring correspondences by the tangent Sampson error, i.e. in
/// pixel units, for arbitrary central camera models. The model is estimated by the same
/// five-point solver as <see cref="EssentialMatrixFivePointEstimator"/>; only the residual
/// differs. Because the residual is measured in pixels rather than radians, the RANSAC
/// threshold is the plain pixel threshold and needs no per-camera conversion via
/// Camera.CamFromImgThreshold. See EssentialMatrix.ComputeSquaredTangentSampsonError.
/// Port of colmap::EssentialMatrixTangentSampsonEstimator.
/// </summary>
public readonly struct EssentialMatrixTangentSampsonEstimator
	: IEstimator<CamRayWithJac, CamRayWithJac, Matrix3d>, ILocalEstimator<CamRayWithJac, CamRayWithJac, Matrix3d>
{
	// The 5-DoF manifold of a relative pose (rotation on SO(3), translation on the unit
	// sphere), matching the block layout of Rigid3d::params ([qx, qy, qz, qw, tx, ty, tz]).
	private static readonly TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold>
		RelativePoseManifold = new(default, default);

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 5;

	/// <summary>
	/// Estimate up to 10 possible essential matrix solutions from a set of corresponding
	/// camera rays. The Jacobians are ignored here. They only affect scoring.
	/// </summary>
	public void Estimate(ReadOnlySpan<CamRayWithJac> camRays1WithJac, ReadOnlySpan<CamRayWithJac> camRays2WithJac, List<Matrix3d> models)
	{
		int n = camRays1WithJac.Length;
		Span<Vector3d> rays1 = n <= 64 ? stackalloc Vector3d[n] : new Vector3d[n];
		Span<Vector3d> rays2 = camRays2WithJac.Length <= 64 ? stackalloc Vector3d[camRays2WithJac.Length] : new Vector3d[camRays2WithJac.Length];
		UnpackCamRaysWithJac(camRays1WithJac, rays1);
		UnpackCamRaysWithJac(camRays2WithJac, rays2);
		new EssentialMatrixFivePointEstimator().Estimate(rays1, rays2, models);
	}

	/// <summary>The local estimate refines a copy of the current best model (C++'s Refine hook).</summary>
	public void EstimateLocal(ReadOnlySpan<CamRayWithJac> camRays1WithJac, ReadOnlySpan<CamRayWithJac> camRays2WithJac, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Matrix3d e = initialModel;
		if (Refine(camRays1WithJac, camRays2WithJac, ref e))
		{
			models.Add(e);
		}
	}

	/// <summary>
	/// Refine E in place by nonlinearly minimizing the pixel-unit tangent Sampson error over
	/// the given correspondences, starting from <paramref name="e"/>. This is the local
	/// optimizer used by LO-RANSAC. Returns false and leaves E unchanged on a degenerate
	/// decomposition.
	/// </summary>
	public static bool Refine(ReadOnlySpan<CamRayWithJac> camRays1WithJac, ReadOnlySpan<CamRayWithJac> camRays2WithJac, ref Matrix3d e)
	{
		Check.Eq(camRays1WithJac.Length, camRays2WithJac.Length);
		Check.Ge(camRays1WithJac.Length, MinNumSamples);

		// Decompose the initial E into a relative pose (resolving the four-fold ambiguity via
		// cheirality over the bearings).
		var rays1 = new Vector3d[camRays1WithJac.Length];
		var rays2 = new Vector3d[camRays2WithJac.Length];
		UnpackCamRaysWithJac(camRays1WithJac, rays1);
		UnpackCamRaysWithJac(camRays2WithJac, rays2);
		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(e, rays1, rays2, out Rigid3d cam2FromCam1, validIndices);
		if (validIndices.Count == 0)
		{
			return false;
		}

		// Nonlinear pixel-space tangent Sampson refinement of the full 7-parameter pose via
		// TinySolver, applying the relative pose manifold. Plain least squares: robustness
		// comes from the RANSAC inlier selection.
		var f = new TinyTangentSampsonErrorCostFunctor(camRays1WithJac.ToArray(), camRays2WithJac.ToArray());
		var solver = new TinySolver<TinyTangentSampsonErrorCostFunctor,
			TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold>>(RelativePoseManifold);
		var options = new TinySolverOptions { MaxNumIterations = 25 };

		Vector4d q = cam2FromCam1.Rotation.Normalized().Coeffs;
		Vector3d t = cam2FromCam1.Translation.Normalized();
		Span<double> x = [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z];
		solver.Solve(f, x, options);

		// Keep the refined pose only if the solve stayed finite.
		bool allFinite = true;
		foreach (double value in x)
		{
			allFinite &= double.IsFinite(value);
		}

		if (allFinite)
		{
			cam2FromCam1 = new Rigid3d(
				new Quaterniond(x[3], x[0], x[1], x[2]).Normalized(), new Vector3d(x[4], x[5], x[6]));
		}

		e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
		return true;
	}

	/// <summary>
	/// The residuals of a set of corresponding rays under E, as the squared tangent Sampson
	/// error in squared pixels, additionally enforcing the cheirality constraint.
	/// </summary>
	public void Residuals(ReadOnlySpan<CamRayWithJac> camRays1WithJac, ReadOnlySpan<CamRayWithJac> camRays2WithJac, in Matrix3d e, Span<double> residuals)
	{
		EssentialMatrix.ComputeSquaredTangentSampsonErrorWithCheirality(camRays1WithJac, camRays2WithJac, e, residuals);
	}

	// Extract the bearings into the contiguous array the five-point solver expects. The
	// Jacobians play no part in estimation. They only affect scoring.
	private static void UnpackCamRaysWithJac(ReadOnlySpan<CamRayWithJac> camRaysWithJac, Span<Vector3d> rays)
	{
		for (int i = 0; i < camRaysWithJac.Length; ++i)
		{
			rays[i] = camRaysWithJac[i].Ray;
		}
	}
}

/// <summary>The N x 9 epipolar constraint system shared by the ray estimators.</summary>
internal static class EssentialMatrixSystem
{
	/// <summary>Row i is [x2 x1^T, y2 x1^T, z2 x1^T]: entry 3r + c multiplies E(r, c).</summary>
	public static MatrixXd ConstraintMatrix(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<Vector3d> camRays2)
	{
		var q = new MatrixXd(camRays1.Length, 9);
		Span<double> row = stackalloc double[9];
		for (int i = 0; i < camRays1.Length; ++i)
		{
			SetRow(row, camRays1[i], camRays2[i]);
			for (int c = 0; c < 9; ++c)
			{
				q[i, c] = row[c];
			}
		}

		return q;
	}

	/// <summary>One constraint row, see <see cref="ConstraintMatrix"/>.</summary>
	public static void SetRow(Span<double> row, Vector3d ray1, Vector3d ray2)
	{
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				row[3 * r + c] = ray2[r] * ray1[c];
			}
		}
	}
}
