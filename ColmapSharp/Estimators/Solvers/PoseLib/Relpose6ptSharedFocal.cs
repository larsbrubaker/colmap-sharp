// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08 as pinned by
// COLMAP's src/thirdparty/CMakeLists.txt.
//
// Relpose6ptSharedFocal: PoseLib's solvers/relpose_6pt_focal.h/.cc - the minimal relative
// pose solver for two views sharing one unknown focal length, from 6 correspondences, as an
// automatically generated elimination-template (action matrix) solver. COLMAP's
// RelativePoseSharedFocalEstimator (Estimators/Solvers/RelativePoseSharedFocal.cs) is its
// only caller. The generated parts are split out: Relpose6ptSharedFocal.Coeffs1.cs /
// .Coeffs2.cs (the 280 coefficients) and .Template.cs (the elimination template indices).
//
// Pipeline, as in PoseLib: the 3D null space N of the six epipolar constraints parametrizes
// F = N0 + x N1 + y N2; the coefficients are scattered into the template [C0 | C1],
// C12 = C0^-1 C1 (partial-pivot LU), the 15 x 15 action matrix is read off C12, its
// characteristic polynomial (Danilevsky, Sturm.cs) has its real roots bracketed by Sturm
// sequences, and the other unknowns come from the structured back-substitution of
// shared_focal_relpose_fast_eigenvector_solver (a 7 x 7 Householder QR solve per root). Each
// F with a positive q = 1/f^2 becomes E = K F K and is decomposed by Essential.cs.
//
// The null space comes from an unpivoted Householder QR instead of PoseLib's
// fullPivHouseholderQr (divergence 30): same space, different basis,
// the same solution set in exact arithmetic, though spurious real roots of the ill-conditioned
// degree-15 polynomial can appear or vanish with the basis. Tier B.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>
/// Minimal relative pose solver with a shared unknown focal length. Port of
/// poselib::relpose_6pt_shared_focal.
/// </summary>
internal static partial class Relpose6ptSharedFocal
{
	private const int NumCoeffs = 280;

	// PoseLib's AM_ind: the rows of RR = [-C12.bottomRows(8); I] forming the action matrix.
	private static ReadOnlySpan<int> ActionMatrixRows => [15, 11, 0, 1, 2, 12, 3, 16, 4, 5, 17, 6, 18, 19, 7];

	// The non-trivial rows of the action matrix (fast eigenvector solver's ind).
	private static ReadOnlySpan<int> NonTrivialRows => [2, 3, 4, 6, 8, 9, 11, 14];

	/// <summary>
	/// Estimates the relative pose and shared focal from the first 6 correspondences (x2^T F
	/// x1 = 0, homogeneous image points of both views, principal point at the origin). Clears
	/// <paramref name="outImagePairs"/>, appends one ImagePair per cheirality-consistent pose
	/// (both cameras SIMPLE_PINHOLE with the recovered focal) and returns their count.
	/// </summary>
	public static int Solve(ReadOnlySpan<Vector3d> x1, ReadOnlySpan<Vector3d> x2, List<ImagePair> outImagePairs)
	{
		// Compute nullspace to epipolar constraints. Column i holds x1[i](k) * x2[i](r) at
		// 3k + r, the coefficient of F(r, k) in column-major order.
		Span<double> epipolarConstraints = stackalloc double[9 * 6];
		for (int i = 0; i < 6; ++i)
		{
			Span<double> col = epipolarConstraints.Slice(9 * i, 9);
			for (int k = 0; k < 3; ++k)
			{
				for (int r = 0; r < 3; ++r)
				{
					col[3 * k + r] = x1[i][k] * x2[i][r];
				}
			}
		}

		// B = N (9 x 3, column-major): the last three columns of the full Q.
		Span<double> b = stackalloc double[27];
		NullSpace3(epipolarConstraints, b);

		Span<double> sols = stackalloc double[3 * 15];
		int nSols = SolveTemplate(b, sols);

		outImagePairs.Clear();

		int nPoses = 0;
		var poses = new List<CameraPose>(4);
		Span<Vector3d> x1U = stackalloc Vector3d[6];
		Span<Vector3d> x2U = stackalloc Vector3d[6];
		Span<double> fVector = stackalloc double[9];
		for (int i = 0; i < nSols; i++)
		{
			// The real-root path only produces real solutions, so PoseLib's imaginary-part
			// test (sols.col(i).imag().norm() > 1e-8) never rejects one.
			if (sols[3 * i + 2] < 1e-8)
			{
				continue;
			}

			double focal = Math.Sqrt(1.0 / sols[3 * i + 2]);

			double squaredNorm = 0.0;
			for (int k = 0; k < 9; ++k)
			{
				fVector[k] = b[k] + sols[3 * i] * b[9 + k] + sols[3 * i + 1] * b[18 + k];
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


			var kMat = new Matrix3d(focal, 0.0, 0.0, 0.0, focal, 0.0, 0.0, 0.0, 1.0);
			Matrix3d e = kMat * (f * kMat);

			for (int j = 0; j < 6; j++)
			{
				x1U[j] = new Vector3d(x1[j].X / focal, x1[j].Y / focal, x1[j].Z).Normalized();
				x2U[j] = new Vector3d(x2[j].X / focal, x2[j].Y / focal, x2[j].Z).Normalized();
			}

			poses.Clear();
			Essential.MotionFromEssential(e, x1U, x2U, poses);

			foreach (CameraPose pose in poses)
			{
				// PoseLib copies calib into each ImagePair; PoseLibCamera is a mutable class, so
				// each pair gets its own two instances rather than shared references.
				outImagePairs.Add(new ImagePair(
					pose, new PoseLibCamera(0, [focal, 0.0, 0.0], -1, -1), new PoseLibCamera(0, [focal, 0.0, 0.0], -1, -1)));
				nPoses++;
			}
		}

		return nPoses;
	}

	/// <summary>
	/// The last three columns of the full Q of the 9 x 6 column-major matrix
	/// <paramref name="a"/> (destroyed), written column-major to <paramref name="n"/>: an
	/// orthonormal basis of the space orthogonal to a's columns.
	/// </summary>
	internal static void NullSpace3(Span<double> a, Span<double> n)
	{
		Span<double> tau = stackalloc double[6];
		Householder.FactorInPlace(a, 9, 6, tau);
		for (int j = 0; j < 3; ++j)
		{
			Span<double> q = n.Slice(9 * j, 9);
			q.Clear();
			q[6 + j] = 1;
			Householder.ApplyQ(a, 9, tau, q);
		}
	}

	// PoseLib's solver_shared_focal_relpose_6pt: fills sols (3 x 15, column-major; real
	// parts only, see Solve) and returns the number of real roots.
	private static int SolveTemplate(ReadOnlySpan<double> data, Span<double> sols)
	{
		// Compute coefficients
		Span<double> coeffs = stackalloc double[NumCoeffs];
		ComputeCoefficients1(data, coeffs);
		ComputeCoefficients2(data, coeffs);

		// Setup elimination template
		Span<double> c0 = stackalloc double[31 * 31];
		Span<double> c1 = stackalloc double[31 * 15];
		c0.Clear();
		c1.Clear();
		ReadOnlySpan<int> c0Ind = C0Indices;
		ReadOnlySpan<int> coeffs0Ind = Coeffs0Indices;
		for (int i = 0; i < 556; i++)
		{
			c0[c0Ind[i]] = coeffs[coeffs0Ind[i]];
		}

		ReadOnlySpan<int> c1Ind = C1Indices;
		ReadOnlySpan<int> coeffs1Ind = Coeffs1Indices;
		for (int i = 0; i < 258; i++)
		{
			c1[c1Ind[i]] = coeffs[coeffs1Ind[i]];
		}

		// C12 = C0.partialPivLu().solve(C1), column by column into c1.
		Span<int> permutation = stackalloc int[31];
		PartialPivLU.FactorInPlace(c0, 31, permutation);
		Span<double> rhs = stackalloc double[31];
		for (int col = 0; col < 15; ++col)
		{
			Span<double> column = c1.Slice(31 * col, 31);
			column.CopyTo(rhs);
			PartialPivLU.SolveInPlace(c0, 31, permutation, rhs, column);
		}

		// Setup action matrix: RR = [-C12.bottomRows(8); I15] (23 x 15), AM.row(i) =
		// RR.row(AM_ind[i]). am is column-major 15 x 15.
		Span<double> am = stackalloc double[15 * 15];
		ReadOnlySpan<int> amInd = ActionMatrixRows;
		for (int i = 0; i < 15; i++)
		{
			int rr = amInd[i];
			for (int c = 0; c < 15; c++)
			{
				am[i + 15 * c] = rr < 8 ? -c1[(23 + rr) + 31 * c] : (rr - 8 == c ? 1.0 : 0.0);
			}
		}

		sols.Clear();

		// Solve eigenvalue problem: the real eigenvalues are the real roots of the
		// characteristic polynomial.
		Span<double> p = stackalloc double[1 + 15];
		Span<double> amp = stackalloc double[15 * 15];
		am.CopyTo(amp);
		Sturm.CharpolyDanilevskyPiv(amp, 15, p);
		Span<double> eigv = stackalloc double[15];
		int nroots = Sturm.BisectSturm(15, p, eigv, 1e-12);

		FastEigenvectorSolver(eigv[..nroots], am, sols);

		return nroots;
	}

	// PoseLib's shared_focal_relpose_fast_eigenvector_solver.
	private static void FastEigenvectorSolver(ReadOnlySpan<double> eigv, ReadOnlySpan<double> am, Span<double> sols)
	{
		// Truncated action matrix containing non-trivial rows (8 x 15, row accessor).
		ReadOnlySpan<int> ind = NonTrivialRows;

		Span<double> zi = stackalloc double[3];
		var aa = new MatrixXd(7, 7);
		var rhs = new VectorXd(7);
		Span<double> col = stackalloc double[8];
		for (int i = 0; i < eigv.Length; i++)
		{
			zi[0] = eigv[i];
			for (int j = 1; j < 3; j++)
			{
				zi[j] = zi[j - 1] * eigv[i];
			}

			// AA (8 x 8) column by column; only its top 7 rows are used.
			for (int c = 0; c < 8; c++)
			{
				for (int r = 0; r < 8; r++)
				{
					col[r] = c switch
					{
						0 => am[ind[r] + 15 * 2],
						1 => am[ind[r] + 15 * 6],
						2 => zi[0] * am[ind[r] + 15 * 4] + am[ind[r] + 15 * 5],
						3 => am[ind[r] + 15 * 1] + zi[0] * am[ind[r] + 15 * 3],
						4 => am[ind[r] + 15 * 14],
						5 => zi[0] * am[ind[r] + 15 * 11] + am[ind[r] + 15 * 13],
						6 => zi[1] * am[ind[r] + 15 * 9] + zi[0] * am[ind[r] + 15 * 10] + am[ind[r] + 15 * 12],
						_ => am[ind[r] + 15 * 0] + zi[0] * am[ind[r] + 15 * 7] + zi[1] * am[ind[r] + 15 * 8],
					};
				}

				switch (c)
				{
					case 0: col[0] -= zi[0]; break;
					case 1: col[3] -= zi[0]; break;
					case 2: col[2] -= zi[1]; break;
					case 3: col[1] -= zi[1]; break;
					case 4: col[7] -= zi[0]; break;
					case 5: col[6] -= zi[1]; break;
					case 6: col[5] -= zi[2]; break;
					default: col[4] -= zi[2]; break;
				}

				for (int r = 0; r < 7; r++)
				{
					if (c < 7)
					{
						aa[r, c] = col[r];
					}
					else
					{
						rhs[r] = -col[r];
					}
				}
			}

			// Using column pivoting leads to unstable numerics
			VectorXd s = new HouseholderQR(aa).Solve(rhs);

			sols[3 * i] = s[3];
			sols[3 * i + 1] = zi[0];
			sols[3 * i + 2] = s[6];
		}
	}
}
