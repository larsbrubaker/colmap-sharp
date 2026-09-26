// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08 as pinned by
// COLMAP's src/thirdparty/CMakeLists.txt.
//
// Relpose6ptOnesidedFocal: PoseLib's solvers/relpose_6pt_onesided_focal.h/.cc - the minimal
// relative pose solver from 6 correspondences when the first view has one unknown focal
// length and the second view is calibrated, as an automatically generated elimination-
// template (action matrix) solver. COLMAP's RelativePoseOneSidedFocalEstimator
// (Estimators/Solvers/RelativePoseOneSidedFocal.cs) is its only caller. The generated parts
// are split out: Relpose6ptOnesidedFocal.Coeffs1.cs / .Coeffs2.cs (the 190 coefficients)
// and .Template.cs (the elimination template fill).
//
// Only the full template (use_elim = false) is ported: it is the only one COLMAP calls, as
// it solves for the focal directly. The compact template (solver_..._elim) and the Kruppa
// focal recovery it needs (onesided_focal_from_fundamental) are left for the caller that
// first needs them.
//
// Pipeline, as in PoseLib: the view-1 points are rescaled by their mean magnitude; the 3D
// null space N of the six epipolar constraints parametrizes F = N0 + x N1 + y N2; the
// coefficients fill the template [C0 | C1], C2 = C0^-1 C1 (partial-pivot LU), and the 9 x 9
// action matrix is read off C2. Its eigenvalues give x; y and q = 1/f^2 are eigenvector
// ratios, so the eigenvector phase freedom of LinearAlgebra/EigenSolver.cs cancels. Each
// real solution becomes E = F diag(1, 1, 1/f) and is decomposed by Essential.cs.
//
// The null space comes from an unpivoted Householder QR instead of PoseLib's
// fullPivHouseholderQr (docs/CPP_DIVERGENCES.md, entry 30). The order of the eigenvalues
// (Schur order) sets the order of the returned poses, which may differ from Eigen's; COLMAP
// treats the output as a set. Tier B.

using System.Numerics;

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>
/// Minimal relative pose solver with one unknown focal length (first view). Port of
/// poselib::relpose_6pt_onesided_focal with use_elim = false.
/// </summary>
internal static partial class Relpose6ptOnesidedFocal
{
	private const int NumCoeffs = 190;

	// PoseLib's AM_rows: the rows of RR = [-C2.bottomRows(5); I9] forming the action matrix.
	private static ReadOnlySpan<int> ActionMatrixRows => [9, 7, 1, 0, 10, 4, 3, 11, 2];

	/// <summary>
	/// Estimates the relative pose and the first view's focal from the first 6
	/// correspondences: <paramref name="x1"/> homogeneous, principal-point-centered pixels of
	/// the uncalibrated view, <paramref name="x2"/> bearings of the calibrated one. Clears
	/// <paramref name="outImagePairs"/>, appends one ImagePair per cheirality-consistent pose
	/// (camera 1 SIMPLE_PINHOLE with the focal in the pixel units of x1, camera 2 the unit
	/// SIMPLE_PINHOLE) and returns their count.
	/// </summary>
	public static int Solve(ReadOnlySpan<Vector3d> x1, ReadOnlySpan<Vector3d> x2, List<ImagePair> outImagePairs)
	{
		// The solver works on rescaled pixel coordinates in the uncalibrated image: dividing
		// by the average magnitude brings the effective focal length close to one, which
		// keeps the polynomial system well-conditioned for pixel-scale input. The focal is
		// scaled back when constructing the output cameras.
		double scale = 0.0;
		Span<Vector3d> x1s = stackalloc Vector3d[6];
		for (int i = 0; i < 6; ++i)
		{
			x1s[i] = x1[i] / x1[i].Z;
			scale += Math.Sqrt(x1s[i].X * x1s[i].X + x1s[i].Y * x1s[i].Y);
		}

		scale /= 6.0;
		for (int i = 0; i < 6; ++i)
		{
			x1s[i] = new Vector3d(x1s[i].X / scale, x1s[i].Y / scale, x1s[i].Z);
		}

		// Compute nullspace to the epipolar constraints (3-dimensional). The fundamental
		// matrix is parameterized as F = N.col(0) + x1 * N.col(1) + x2 * N.col(2).
		Span<double> epipolarConstraints = stackalloc double[9 * 6];
		for (int i = 0; i < 6; ++i)
		{
			Span<double> col = epipolarConstraints.Slice(9 * i, 9);
			for (int k = 0; k < 3; ++k)
			{
				for (int r = 0; r < 3; ++r)
				{
					col[3 * k + r] = x1s[i][k] * x2[i][r];
				}
			}
		}

		Span<double> b = stackalloc double[27];
		Relpose6ptSharedFocal.NullSpace3(epipolarConstraints, b);

		var sols = new Complex[3, 9];
		int nSols = SolveFullTemplate(b, sols);

		outImagePairs.Clear();

		int nPoses = 0;
		var poses = new List<CameraPose>(4);
		Span<Vector3d> x1U = stackalloc Vector3d[6];
		Span<Vector3d> x2U = stackalloc Vector3d[6];
		Span<double> fVector = stackalloc double[9];
		for (int i = 0; i < nSols; i++)
		{
			// x1 and x2 (rows 0 and 1) must be real.
			if (Math.Abs(sols[0, i].Imaginary) > 1e-8 || Math.Abs(sols[1, i].Imaginary) > 1e-8)
			{
				continue;
			}

			double x1Val = sols[0, i].Real;
			double x2Val = sols[1, i].Real;

			double squaredNorm = 0.0;
			for (int k = 0; k < 9; ++k)
			{
				fVector[k] = b[k] + x1Val * b[9 + k] + x2Val * b[18 + k];
				squaredNorm += fVector[k] * fVector[k];
			}

			if (squaredNorm > 0)
			{
				double norm = Math.Sqrt(squaredNorm);
				for (int k = 0; k < 9; ++k)
				{
					fVector[k] /= norm;
				}
			}

			// Eigen::Matrix3d(F_vector.data()) reads column-major.
			var f = new Matrix3d(
				fVector[0], fVector[3], fVector[6],
				fVector[1], fVector[4], fVector[7],
				fVector[2], fVector[5], fVector[8]);

			// The full template solves for q = 1/f^2 (monomial halving) as the third unknown.
			if (Math.Abs(sols[2, i].Imaginary) > 1e-8 || sols[2, i].Real <= 0.0)
			{
				continue;
			}

			double focal = 1.0 / Math.Sqrt(sols[2, i].Real);
			if (!double.IsFinite(focal) || focal <= 0.0)
			{
				continue;
			}

			// Camera 1 is the uncalibrated camera (K1 = diag(f, f, 1), x1 are pixel points, so
			// diag(1, 1, f) * x1 is the calibrated bearing); camera 2 is calibrated. The solved
			// focal lives in the rescaled coordinates; undo the rescaling.
			double focalPx = focal * scale;

			Matrix3d e = f * Matrix3d.FromDiagonal(new Vector3d(1.0, 1.0, 1.0 / focal));

			for (int j = 0; j < 6; j++)
			{
				x1U[j] = new Vector3d(x1s[j].X, x1s[j].Y, x1s[j].Z * focal).Normalized();
				x2U[j] = x2[j].Normalized();
			}

			poses.Clear();
			Essential.MotionFromEssential(e, x1U, x2U, poses);

			foreach (CameraPose pose in poses)
			{
				// PoseLib copies the cameras into each ImagePair; PoseLibCamera is a mutable
				// class, so each pair gets its own instances rather than shared references.
				outImagePairs.Add(new ImagePair(
					pose, new PoseLibCamera(0, [focalPx, 0.0, 0.0], -1, -1), new PoseLibCamera(0, [1.0, 0.0, 0.0], -1, -1)));
				nPoses++;
			}
		}

		return nPoses;
	}

	// PoseLib's solver_relpose_6pt_onesided_focal_full: full action-matrix template (190
	// coeffs, 19 x 19 elimination, 9 solutions). Fills sols (3 x 9) and returns 9.
	private static int SolveFullTemplate(ReadOnlySpan<double> data, Complex[,] sols)
	{
		Span<double> coeffs = stackalloc double[NumCoeffs];
		ComputeCoefficients1(data, coeffs);
		ComputeCoefficients2(data, coeffs);

		Span<double> c0 = stackalloc double[19 * 19];
		Span<double> c1 = stackalloc double[19 * 9];
		c0.Clear();
		c1.Clear();
		ReadOnlySpan<int> c0Ind = C0Indices;
		ReadOnlySpan<int> c0Coeffs = C0Coeffs;
		for (int i = 0; i < c0Ind.Length; i++)
		{
			c0[c0Ind[i]] = coeffs[c0Coeffs[i]];
		}

		ReadOnlySpan<int> c1Ind = C1Indices;
		ReadOnlySpan<int> c1Coeffs = C1Coeffs;
		for (int i = 0; i < c1Ind.Length; i++)
		{
			c1[c1Ind[i]] = coeffs[c1Coeffs[i]];
		}

		// C2 = C0.partialPivLu().solve(C1), column by column into c1.
		Span<int> permutation = stackalloc int[19];
		PartialPivLU.FactorInPlace(c0, 19, permutation);
		Span<double> rhs = stackalloc double[19];
		for (int col = 0; col < 9; ++col)
		{
			Span<double> column = c1.Slice(19 * col, 19);
			column.CopyTo(rhs);
			PartialPivLU.SolveInPlace(c0, 19, permutation, rhs, column);
		}

		// RR = [-C2.bottomRows(5); I9] (14 x 9), AM.row(i) = RR.row(AM_rows[i]).
		var am = new MatrixXd(9, 9);
		ReadOnlySpan<int> amRows = ActionMatrixRows;
		for (int i = 0; i < 9; i++)
		{
			int rr = amRows[i];
			for (int c = 0; c < 9; c++)
			{
				am[i, c] = rr < 5 ? -c1[(14 + rr) + 19 * c] : (rr - 5 == c ? 1.0 : 0.0);
			}
		}

		var es = new EigenSolver(am);
		Complex[] eigs = es.Eigenvalues();
		Complex[,] v = es.Eigenvectors();

		for (int j = 0; j < 9; j++)
		{
			Complex scale = v[0, j];
			sols[0, j] = eigs[j];         // x1
			sols[1, j] = v[7, j] / scale; // x2
			sols[2, j] = v[1, j] / scale; // focal variable
		}

		return 9;
	}
}
