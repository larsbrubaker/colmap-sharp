// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08 as pinned by
// COLMAP's src/thirdparty/CMakeLists.txt.
//
// GenRelpose6pt: PoseLib's solvers/gen_relpose_6pt.h/.cc - the minimal generalized relative
// pose solver from 6 correspondences between two generalized (multi-camera) cameras,
// generated with the method of Larsson et al., "Efficient Solvers for Minimal Problems by
// Syzygy-based Reduction", CVPR 2017. COLMAP's GR6PEstimator
// (Estimators/Solvers/GeneralizedRelativePose.cs) is its only caller. The generated parts
// are split out: GenRelpose6pt.CoeffMatrix.cs (the 15 equations), .Polynomials.cs (their
// polynomial products) and .Template0.cs / .Template1.cs (the elimination template indices).
//
// Pipeline, as in PoseLib: the 84 x 15 coefficients are scattered into the elimination
// template [C0 | C1], C12 = C0^-1 C1 (partial-pivot LU), the 64 x 64 action matrix is read
// off C12, its real eigenvalues (|imag| < 1e-6) give the third Cayley parameter, and the
// other two come from the structured back-substitution of fast_eigenvector_solver
// (PoseLib's USE_FAST_EIGENVECTOR_SOLVER path, which it defines). Each rotation's
// translation is the least-squares solution of the epipolar constraints, poses failing
// cheirality are dropped, and the survivors get 5 Newton steps (root_refinement).
//
// Only eigenvalues of the action matrix are used, never its eigenvectors, so the
// eigenvector-phase freedom of LinearAlgebra/EigenSolver.cs cannot reach the result. The
// order of the real eigenvalues (Schur order) sets the order of the returned poses, which
// may differ from Eigen's; RANSAC treats the output as a set. Tier B.
//
// PoseLib's CameraPose and the quaternion.h / essential.h helpers this file needs are kept
// private here (Pose, QuatRotate, QuatStepPre, CheckCheirality) so the file stands alone next
// to the other PoseLib ports; PoseLib stores quaternions (w, x, y, z).

using System.Numerics;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>
/// Minimal generalized relative pose solver from 6 point correspondences. Port of
/// poselib::gen_relpose_6pt.
/// </summary>
internal static partial class GenRelpose6pt
{
	/// <summary>
	/// Solves for the generalized relative pose (R, t) with
	/// <c>lambda2 * x2 + p2 = R * (lambda1 * x1 + p1) + t</c> from 6 correspondences, where
	/// p are the ray origins and x the unit ray directions in each rig. Clears
	/// <paramref name="output"/>, appends the solutions (rig2_from_rig1, converted as COLMAP's
	/// ConvertPoseLibPoseToRigid3d does) and returns their count.
	/// </summary>
	public static int Solve(
		ReadOnlySpan<Vector3d> p1,
		ReadOnlySpan<Vector3d> x1,
		ReadOnlySpan<Vector3d> p2,
		ReadOnlySpan<Vector3d> x2,
		List<Rigid3d> output)
	{
		Span<double> m = stackalloc double[84 * 15];
		SetupCoeffMatrix(p1, x1, p2, x2, m);

		// The elimination template [C0 | C1] as one column-major 99 x 163 per-thread buffer
		// (C1's columns follow C0's): at 130 KB it would otherwise be a fresh allocation on
		// every RANSAC hypothesis.
		double[] template = t_template ??= new double[99 * (99 + 64)];
		Span<double> c0Data = template.AsSpan(0, 99 * 99);
		Span<double> c1Data = template.AsSpan(99 * 99, 99 * 64);
		template.AsSpan().Clear();
		ReadOnlySpan<int> coeffs0Ind = Coeffs0Indices;
		ReadOnlySpan<int> c0Ind = C0Indices;
		for (int i = 0; i < 4655; i++)
		{
			c0Data[c0Ind[i]] = m[coeffs0Ind[i]];
		}

		ReadOnlySpan<int> coeffs1Ind = Coeffs1Indices;
		ReadOnlySpan<int> c1Ind = C1Indices;
		for (int i = 0; i < 3661; i++)
		{
			c1Data[c1Ind[i]] = m[coeffs1Ind[i]];
		}

		// C12 = C0.partialPivLu().solve(C1). Eliminating [C0 | C1] together does the
		// forward substitution of all 64 columns in the same pass (bit-identical to solving
		// column by column). The action matrix reads only C12's rows 78..98, and back
		// substitution of a row needs only the rows below it, so the other 78 rows are never
		// finished.
		Span<int> permutation = stackalloc int[99];
		PartialPivLU.FactorAugmentedInPlace(template, 99, 64, permutation);
		var c12 = new double[21, 64];
		for (int j = 0; j < 64; j++)
		{
			Span<double> column = c1Data.Slice(99 * j, 99);
			PartialPivLU.BackSubstituteTrailingRows(c0Data, 99, column, 78);
			for (int r = 0; r < 21; r++)
			{
				c12[r, j] = column[78 + r];
			}
		}

		// Setup action matrix
		var am = MatrixXd.Zero(64, 64);
		am[0, 57] = 1.0;
		am[1, 34] = 1.0;
		am[2, 19] = 1.0;
		am[3, 11] = 1.0;
		am[4, 7] = 1.0;
		SetNegatedRow(am, 5, c12, 78);
		SetNegatedRow(am, 6, c12, 79);
		SetNegatedRow(am, 7, c12, 80);
		am[8, 10] = 1.0;
		SetNegatedRow(am, 9, c12, 81);
		SetNegatedRow(am, 10, c12, 82);
		am[11, 12] = 1.0;
		SetNegatedRow(am, 12, c12, 83);
		am[13, 17] = 1.0;
		am[14, 16] = 1.0;
		SetNegatedRow(am, 15, c12, 84);
		SetNegatedRow(am, 16, c12, 85);
		am[17, 18] = 1.0;
		SetNegatedRow(am, 18, c12, 86);
		am[19, 20] = 1.0;
		am[20, 21] = 1.0;
		am[21, 22] = 1.0;
		SetNegatedRow(am, 22, c12, 87);
		am[23, 30] = 1.0;
		am[24, 28] = 1.0;
		am[25, 27] = 1.0;
		SetNegatedRow(am, 26, c12, 88);
		SetNegatedRow(am, 27, c12, 89);
		am[28, 29] = 1.0;
		SetNegatedRow(am, 29, c12, 90);
		am[30, 31] = 1.0;
		am[31, 32] = 1.0;
		am[32, 33] = 1.0;
		SetNegatedRow(am, 33, c12, 91);
		am[34, 35] = 1.0;
		am[35, 36] = 1.0;
		am[36, 37] = 1.0;
		am[37, 38] = 1.0;
		SetNegatedRow(am, 38, c12, 92);
		am[39, 52] = 1.0;
		am[40, 48] = 1.0;
		am[41, 45] = 1.0;
		am[42, 44] = 1.0;
		SetNegatedRow(am, 43, c12, 93);
		SetNegatedRow(am, 44, c12, 94);
		am[45, 46] = 1.0;
		am[46, 47] = 1.0;
		SetNegatedRow(am, 47, c12, 95);
		am[48, 49] = 1.0;
		am[49, 50] = 1.0;
		am[50, 51] = 1.0;
		SetNegatedRow(am, 51, c12, 96);
		am[52, 53] = 1.0;
		am[53, 54] = 1.0;
		am[54, 55] = 1.0;
		am[55, 56] = 1.0;
		SetNegatedRow(am, 56, c12, 97);
		am[57, 58] = 1.0;
		am[58, 59] = 1.0;
		am[59, 60] = 1.0;
		am[60, 61] = 1.0;
		am[61, 62] = 1.0;
		am[62, 63] = 1.0;
		SetNegatedRow(am, 63, c12, 98);

		// Here we only compute eigenvalues and we use the structured backsubsitution to
		// solve for the eigenvectors
		Complex[] d = new EigenSolver(am, computeEigenvectors: false).Eigenvalues();
		Span<double> eigv = stackalloc double[64];
		int nRoots = 0;
		for (int i = 0; i < 64; i++)
		{
			if (Math.Abs(d[i].Imaginary) < 1e-6)
			{
				eigv[nRoots++] = d[i].Real;
			}
		}

		Span<Vector3d> sols = stackalloc Vector3d[64];
		FastEigenvectorSolver(eigv[..nRoots], am, sols);

		var poses = new List<Pose>(nRoots);
		for (int solK = 0; solK < nRoots; ++solK)
		{
			// From each solution we compute the rotation and solve for the translation
			Vector3d w = sols[solK];
			double qNorm = Math.Sqrt(1.0 * 1.0 + w.X * w.X + w.Y * w.Y + w.Z * w.Z);
			var pose = new Pose(1.0 / qNorm, w.X / qNorm, w.Y / qNorm, w.Z / qNorm, Vector3d.Zero);

			Matrix3d r = new Quaterniond(pose.Qw, pose.Qx, pose.Qy, pose.Qz).ToRotationMatrix();

			// Solve for the translation
			Matrix3d a = Matrix3d.Zero;
			Vector3d b = Vector3d.Zero;
			for (int i = 0; i < 6; ++i)
			{
				Vector3d u = (r * x1[i]).Cross(x2[i]);
				Vector3d v = p2[i] - r * p1[i];
				a += OuterProduct(u, u);
				b += u * u.Dot(v);
			}

			pose = pose with { T = SolveLlt3(a, b) };

			// Filter solution using cheirality
			bool cheiralOk = true;
			for (int ptK = 0; ptK < 6; ++ptK)
			{
				if (!CheckCheirality(pose, p1[ptK], x1[ptK], p2[ptK], x2[ptK]))
				{
					cheiralOk = false;
					break;
				}
			}

			if (!cheiralOk)
			{
				continue;
			}

			poses.Add(pose);
		}

		RootRefinement(p1, x1, p2, x2, poses);

		output.Clear();
		foreach (Pose pose in poses)
		{
			// COLMAP's ConvertPoseLibPoseToRigid3d: PoseLib's quaternion is (w, x, y, z).
			output.Add(new Rigid3d(new Quaterniond(pose.Qw, pose.Qx, pose.Qy, pose.Qz), pose.T));
		}

		return output.Count;
	}

	/// <summary>
	/// Solves for the eigenvector by using structured backsubstitution (i.e. substituting
	/// the eigenvalue into the eigenvector using the known structure to get a reduced linear
	/// system). Port of PoseLib's fast_eigenvector_solver; writes (s(14), s(19), eigenvalue)
	/// for each eigenvalue into <paramref name="sols"/>.
	/// </summary>
	private static void FastEigenvectorSolver(ReadOnlySpan<double> eigv, MatrixXd am, Span<Vector3d> sols)
	{
		ReadOnlySpan<int> ind = [5, 6, 7, 9, 10, 12, 15, 16, 18, 22, 26, 27, 29, 33, 38, 43, 44, 47, 51, 56, 63];

		// Truncated action matrix containing non-trivial rows (row-major, ams[i * 64 + j]).
		Span<double> ams = stackalloc double[21 * 64];
		for (int i = 0; i < 21; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				ams[i * 64 + j] = am[ind[i], j];
			}
		}

		Span<double> zi = stackalloc double[8];

		// AA.leftCols(20) (column-major, 21 rows) and the right-hand side -AA.col(20),
		// refilled for each eigenvalue and solved in place by HouseholderQR's kernels.
		Span<double> aaData = stackalloc double[21 * 20];
		Span<double> rhs = stackalloc double[21];
		Span<double> tau = stackalloc double[20];
		for (int i = 0; i < eigv.Length; i++)
		{
			zi[0] = eigv[i];
			for (int j = 1; j < 8; j++)
			{
				zi[j] = zi[j - 1] * eigv[i];
			}

			for (int k = 0; k < 21; k++)
			{
				aaData[0 * 21 + k] = ams[k * 64 + 5];
				aaData[1 * 21 + k] = ams[k * 64 + 6];
				aaData[2 * 21 + k] = ams[k * 64 + 4] + zi[0] * ams[k * 64 + 7];
				aaData[3 * 21 + k] = ams[k * 64 + 9];
				aaData[4 * 21 + k] = ams[k * 64 + 8] + zi[0] * ams[k * 64 + 10];
				aaData[5 * 21 + k] = ams[k * 64 + 3] + zi[0] * ams[k * 64 + 11] + zi[1] * ams[k * 64 + 12];
				aaData[6 * 21 + k] = ams[k * 64 + 15];
				aaData[7 * 21 + k] = ams[k * 64 + 14] + zi[0] * ams[k * 64 + 16];
				aaData[8 * 21 + k] = ams[k * 64 + 13] + zi[0] * ams[k * 64 + 17] + zi[1] * ams[k * 64 + 18];
				aaData[9 * 21 + k] = ams[k * 64 + 2] + zi[0] * ams[k * 64 + 19] + zi[1] * ams[k * 64 + 20] + zi[2] * ams[k * 64 + 21] + zi[3] * ams[k * 64 + 22];
				aaData[10 * 21 + k] = ams[k * 64 + 26];
				aaData[11 * 21 + k] = ams[k * 64 + 25] + zi[0] * ams[k * 64 + 27];
				aaData[12 * 21 + k] = ams[k * 64 + 24] + zi[0] * ams[k * 64 + 28] + zi[1] * ams[k * 64 + 29];
				aaData[13 * 21 + k] = ams[k * 64 + 23] + zi[0] * ams[k * 64 + 30] + zi[1] * ams[k * 64 + 31] + zi[2] * ams[k * 64 + 32] + zi[3] * ams[k * 64 + 33];
				aaData[14 * 21 + k] = ams[k * 64 + 1] + zi[0] * ams[k * 64 + 34] + zi[1] * ams[k * 64 + 35] + zi[2] * ams[k * 64 + 36]
					+ zi[3] * ams[k * 64 + 37] + zi[4] * ams[k * 64 + 38];
				aaData[15 * 21 + k] = ams[k * 64 + 43];
				aaData[16 * 21 + k] = ams[k * 64 + 42] + zi[0] * ams[k * 64 + 44];
				aaData[17 * 21 + k] = ams[k * 64 + 41] + zi[0] * ams[k * 64 + 45] + zi[1] * ams[k * 64 + 46] + zi[2] * ams[k * 64 + 47];
				aaData[18 * 21 + k] = ams[k * 64 + 40] + zi[0] * ams[k * 64 + 48] + zi[1] * ams[k * 64 + 49] + zi[2] * ams[k * 64 + 50] + zi[3] * ams[k * 64 + 51];
				aaData[19 * 21 + k] = ams[k * 64 + 39] + zi[0] * ams[k * 64 + 52] + zi[1] * ams[k * 64 + 53] + zi[2] * ams[k * 64 + 54]
					+ zi[3] * ams[k * 64 + 55] + zi[4] * ams[k * 64 + 56];
				double col20 = ams[k * 64 + 0] + zi[0] * ams[k * 64 + 57] + zi[1] * ams[k * 64 + 58] + zi[2] * ams[k * 64 + 59]
					+ zi[3] * ams[k * 64 + 60] + zi[4] * ams[k * 64 + 61] + zi[5] * ams[k * 64 + 62] + zi[6] * ams[k * 64 + 63];

				// AA(20, 20) = AA(20, 20) - zi[7] lands in the right-hand-side column.
				if (k == 20)
				{
					col20 = col20 - zi[7];
				}

				rhs[k] = -col20;
			}

			// AA(i, i) is aaData[i * 22] (column-major, 21 rows).
			aaData[0 * 22] = aaData[0 * 22] - zi[0];
			aaData[1 * 22] = aaData[1 * 22] - zi[0];
			aaData[2 * 22] = aaData[2 * 22] - zi[1];
			aaData[3 * 22] = aaData[3 * 22] - zi[0];
			aaData[4 * 22] = aaData[4 * 22] - zi[1];
			aaData[5 * 22] = aaData[5 * 22] - zi[2];
			aaData[6 * 22] = aaData[6 * 22] - zi[0];
			aaData[7 * 22] = aaData[7 * 22] - zi[1];
			aaData[8 * 22] = aaData[8 * 22] - zi[2];
			aaData[9 * 22] = aaData[9 * 22] - zi[4];
			aaData[10 * 22] = aaData[10 * 22] - zi[0];
			aaData[11 * 22] = aaData[11 * 22] - zi[1];
			aaData[12 * 22] = aaData[12 * 22] - zi[2];
			aaData[13 * 22] = aaData[13 * 22] - zi[4];
			aaData[14 * 22] = aaData[14 * 22] - zi[5];
			aaData[15 * 22] = aaData[15 * 22] - zi[0];
			aaData[16 * 22] = aaData[16 * 22] - zi[1];
			aaData[17 * 22] = aaData[17 * 22] - zi[3];
			aaData[18 * 22] = aaData[18 * 22] - zi[4];
			aaData[19 * 22] = aaData[19 * 22] - zi[5];

			// AA.leftCols(20).householderQr().solve(-AA.col(20)): the factorization, Q^T b
			// and the back substitution of HouseholderQR.Solve, on the stack.
			Householder.FactorInPlace(aaData, 21, 20, tau);
			for (int j = 0; j < 20; j++)
			{
				Householder.ApplyLeft(aaData.Slice(j * 21 + j + 1, 20 - j), tau[j], rhs[j..]);
			}

			for (int r = 19; r >= 0; r--)
			{
				double sum = rhs[r];
				for (int c = r + 1; c < 20; c++)
				{
					sum -= aaData[c * 21 + r] * rhs[c];
				}

				rhs[r] = sum / aaData[r * 21 + r];
			}

			sols[i] = new Vector3d(rhs[14], rhs[19], zi[0]);
		}
	}

	/// <summary>
	/// Performs Newton iterations on the epipolar constraints. Port of PoseLib's
	/// root_refinement (at most 5 steps, stopping once the residual norm is below 1e-12).
	/// </summary>
	private static void RootRefinement(
		ReadOnlySpan<Vector3d> p1,
		ReadOnlySpan<Vector3d> x1,
		ReadOnlySpan<Vector3d> p2,
		ReadOnlySpan<Vector3d> x2,
		List<Pose> output)
	{
		Span<Vector3d> qq1 = stackalloc Vector3d[6];
		Span<Vector3d> qq2 = stackalloc Vector3d[6];
		for (int ptK = 0; ptK < 6; ++ptK)
		{
			qq1[ptK] = x1[ptK].Cross(p1[ptK]);
			qq2[ptK] = x2[ptK].Cross(p2[ptK]);
		}

		var jacobian = new MatrixXd(6, 6);
		var res = new VectorXd(6);
		for (int poseK = 0; poseK < output.Count; ++poseK)
		{
			Pose pose = output[poseK];

			for (int iter = 0; iter < 5; ++iter)
			{
				// compute residual and jacobian
				for (int ptK = 0; ptK < 6; ++ptK)
				{
					Vector3d x2t = x2[ptK].Cross(pose.T);
					Vector3d rx1 = QuatRotate(pose, x1[ptK]);
					Vector3d rqq1 = QuatRotate(pose, qq1[ptK]);

					res[ptK] = (x2t - qq2[ptK]).Dot(rx1) - x2[ptK].Dot(rqq1);
					Vector3d jRotation = -x2t.Cross(rx1) + qq2[ptK].Cross(rx1) + x2[ptK].Cross(rqq1);
					Vector3d jTranslation = -x2[ptK].Cross(rx1);
					jacobian[ptK, 0] = jRotation.X;
					jacobian[ptK, 1] = jRotation.Y;
					jacobian[ptK, 2] = jRotation.Z;
					jacobian[ptK, 3] = jTranslation.X;
					jacobian[ptK, 4] = jTranslation.Y;
					jacobian[ptK, 5] = jTranslation.Z;
				}

				if (res.Norm() < 1e-12)
				{
					break;
				}

				VectorXd dp = new PartialPivLU(jacobian).Solve(res);

				var w = new Vector3d(-dp[0], -dp[1], -dp[2]);
				pose = QuatStepPre(pose, w) with { T = pose.T - new Vector3d(dp[3], dp[4], dp[5]) };
			}

			output[poseK] = pose;
		}
	}

	/// <summary>AM.row(row) = -C12.row(c12Row), with c12 holding C12's rows 78..98.</summary>
	private static void SetNegatedRow(MatrixXd am, int row, double[,] c12, int c12Row)
	{
		for (int j = 0; j < 64; j++)
		{
			am[row, j] = -c12[c12Row - 78, j];
		}
	}

	private static Matrix3d OuterProduct(Vector3d a, Vector3d b) => new(
		a.X * b.X, a.X * b.Y, a.X * b.Z,
		a.Y * b.X, a.Y * b.Y, a.Y * b.Z,
		a.Z * b.X, a.Z * b.Y, a.Z * b.Z);

	/// <summary>A.llt().solve(b) for the 3x3 normal equations of the translation.</summary>
	private static Vector3d SolveLlt3(in Matrix3d a, Vector3d b)
	{
		var m = new MatrixXd(3, 3);
		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				m[r, c] = a[r, c];
			}
		}

		return new LLT(m).Solve(VectorXd.From(b)).ToVector3d();
	}

	/// <summary>
	/// The generalized cheirality test of PoseLib's misc/essential.cc with min_depth = 0:
	/// both depths along the rays at their closest points are positive.
	/// </summary>
	private static bool CheckCheirality(in Pose pose, Vector3d p1, Vector3d x1, Vector3d p2, Vector3d x2)
	{
		// This code assumes that x1 and x2 are unit vectors
		Vector3d rx1 = QuatRotate(pose, x1);

		// [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
		// [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
		Vector3d rhs = pose.T + QuatRotate(pose, p1) - p2;
		double a = -rx1.Dot(x2);
		double b1 = -rx1.Dot(rhs);
		double b2 = x2.Dot(rhs);

		// Note that we drop the factor 1.0/(1-a*a) since it is always positive.
		double lambda1 = b1 - a * b2;
		double lambda2 = -a * b1 + b2;

		double minDepth = 0.0 * (1 - a * a);
		return lambda1 > minDepth && lambda2 > minDepth;
	}

	/// <summary>PoseLib's quat_rotate (misc/quaternion.h): rotates p by the pose's quaternion.</summary>
	private static Vector3d QuatRotate(in Pose q, Vector3d p)
	{
		double q1 = q.Qw, q2 = q.Qx, q3 = q.Qy, q4 = q.Qz;
		double p1 = p.X, p2 = p.Y, p3 = p.Z;
		double px1 = -p1 * q2 - p2 * q3 - p3 * q4;
		double px2 = p1 * q1 - p2 * q4 + p3 * q3;
		double px3 = p2 * q1 + p1 * q4 - p3 * q2;
		double px4 = p2 * q2 - p1 * q3 + p3 * q1;
		return new Vector3d(
			px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3,
			px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
			px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1);
	}

	/// <summary>
	/// PoseLib's quat_step_pre (misc/quaternion.h), quat_multiply(quat_exp(w), q), with
	/// quat_exp's Taylor branch at theta &lt;= 1e-6. Returns the pose with the new rotation.
	/// </summary>
	private static Pose QuatStepPre(in Pose q, Vector3d w)
	{
		double theta2 = w.SquaredNorm;
		double theta = Math.Sqrt(theta2);
		double thetaHalf = 0.5 * theta;

		double re, im;
		if (theta > 1e-6)
		{
			re = Math.Cos(thetaHalf);
			im = Math.Sin(thetaHalf) / theta;
		}
		else
		{
			// we are close to zero, use taylor expansion to avoid problems
			// with zero divisors in sin(theta/2)/theta
			double theta4 = theta2 * theta2;
			re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
			im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

			// for the linearized part we re-normalize to ensure unit length
			// here s should be roughly 1.0 anyways, so no problem with zero div
			double s = Math.Sqrt(re * re + im * im * theta2);
			re /= s;
			im /= s;
		}

		// quat_multiply(qa = quat_exp(w), qb = q)
		double qa1 = re, qa2 = im * w.X, qa3 = im * w.Y, qa4 = im * w.Z;
		double qb1 = q.Qw, qb2 = q.Qx, qb3 = q.Qy, qb4 = q.Qz;
		return q with
		{
			Qw = qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4,
			Qx = qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
			Qy = qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
			Qz = qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1,
		};
	}

	/// <summary>Per-thread elimination template buffer, column-major [C0 | C1].</summary>
	[ThreadStatic]
	private static double[]? t_template;

	/// <summary>PoseLib's CameraPose: rotation quaternion (w, x, y, z) and translation.</summary>
	private readonly record struct Pose(double Qw, double Qx, double Qy, double Qz, Vector3d T);
}
