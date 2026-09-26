// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// P4pf: PoseLib/solvers/p4pf.cc (both poselib::p4pf overloads) - absolute pose with unknown
// focal length from four 2D-3D correspondences, via a nullspace of the projection
// constraints and the three-quadratics solver Re3q3.cs. Behind COLMAP's P4PFEstimator
// (Estimators/Solvers/AbsolutePose.cs).
//
// The method: V. Larsson, Z. Kukelova, Y. Zheng, "Making Minimal Solvers for Absolute Pose
// Estimation Compact and Robust", ICCV 2017.
//
// Translation note: the 8x8 Householder Q is formed through LinearAlgebra/HouseholderQR.cs
// on a MatrixXd, the one heap allocation per call besides the output lists.
//
// Tier B.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>The P4Pf minimal solver. Port of poselib::p4pf.</summary>
public static class P4pf
{
	/// <summary>
	/// Shared-focal variant: solves with separate fx, fy and reports (fx + fy) / 2. With
	/// <paramref name="filterSolutions"/>, keeps only the solution with the aspect ratio
	/// closest to 1 (and only when it is within a factor of 2). Clears the outputs and
	/// returns the number of poses.
	/// </summary>
	public static int Solve(
		ReadOnlySpan<Vector2d> x, ReadOnlySpan<Vector3d> bigX, List<CameraPose> output, List<double> outputFocal,
		bool filterSolutions)
	{
		var poses = new List<CameraPose>(8);
		var fx = new List<double>(8);
		var fy = new List<double>(8);
		int n = Solve(x, bigX, poses, fx, fy, filterSolutions);

		// Note: unlike the overload below, PoseLib does not clear the outputs here; COLMAP
		// always passes fresh vectors, and so do the callers here.
		if (filterSolutions)
		{
			int bestInd = -1;
			double bestErr = 1.0;

			for (int i = 0; i < n; ++i)
			{
				double a = fx[i] / fy[i];
				double err = Math.Max(Math.Abs(a - 1.0), Math.Abs(1 / a - 1.0));
				if (err < bestErr)
				{
					bestErr = err;
					bestInd = i;
				}
			}

			if (bestErr < 1.0 && bestInd > -1)
			{
				double focal = (fx[bestInd] + fy[bestInd]) / 2.0;
				outputFocal.Add(focal);
				output.Add(poses[bestInd]);
			}
		}
		else
		{
			output.Clear();
			output.AddRange(poses);
			outputFocal.Clear();
			for (int i = 0; i < n; ++i)
			{
				outputFocal.Add((fx[i] + fy[i]) / 2.0);
			}
		}

		return output.Count;
	}

	/// <summary>
	/// Separate-focal variant: estimates the pose and the focal lengths fx, fy. With
	/// <paramref name="filterSolutions"/>, drops solutions with a point behind the camera.
	/// Clears the outputs and returns the number of poses.
	/// </summary>
	public static int Solve(
		ReadOnlySpan<Vector2d> x, ReadOnlySpan<Vector3d> bigX, List<CameraPose> output, List<double> outputFx,
		List<double> outputFy, bool filterSolutions)
	{
		double f0 = (x[0].Norm + x[1].Norm + x[2].Norm + x[3].Norm) / 4;
		Span<Vector2d> points2d = stackalloc Vector2d[4];
		for (int i = 0; i < 4; ++i)
		{
			points2d[i] = x[i] / f0;
		}

		// Setup nullspace
		var m = new MatrixXd(8, 4);
		for (int i = 0; i < 4; i++)
		{
			m[0, i] = -points2d[i].Y * bigX[i].X;
			m[2, i] = -points2d[i].Y * bigX[i].Y;
			m[4, i] = -points2d[i].Y * bigX[i].Z;
			m[6, i] = -points2d[i].Y;
			m[1, i] = points2d[i].X * bigX[i].X;
			m[3, i] = points2d[i].X * bigX[i].Y;
			m[5, i] = points2d[i].X * bigX[i].Z;
			m[7, i] = points2d[i].X;
		}

		// d[0..31] is N (8 x 4) and d[32..47] is B (4 x 4), both column-major, as in PoseLib.
		Span<double> d = stackalloc double[48];

		// Compute nullspace using QR
		MatrixXd q = new HouseholderQR(m).HouseholderQ();
		for (int c = 0; c < 4; ++c)
		{
			for (int r = 0; r < 8; ++r)
			{
				d[r + 8 * c] = q[r, 4 + c];
			}
		}

		// Setup matrices A and B (see paper for definition)
		Span<double> a = stackalloc double[16];
		Span<double> b = stackalloc double[16];
		for (int i = 0; i < 4; ++i)
		{
			Vector3d xi = bigX[i];
			// Row offset 1 reads the y constraint rows of N (1, 3, 5, 7), offset 0 the x rows.
			bool useY = Math.Abs(points2d[i].X) < Math.Abs(points2d[i].Y);
			double s = useY ? points2d[i].Y : points2d[i].X;
			int o = useY ? 1 : 0;
			a[i] = s * xi.X;
			a[i + 4] = s * xi.Y;
			a[i + 8] = s * xi.Z;
			a[i + 12] = s;
			for (int k = 0; k < 4; ++k)
			{
				// alpha1, alpha2, alpha3, 1
				b[i + 4 * k] = xi.X * d[o + 8 * k] + xi.Y * d[o + 2 + 8 * k] + xi.Z * d[o + 4 + 8 * k] + d[o + 6 + 8 * k];
			}
		}

		// [p31,p32,p33,p34] = B * [alpha; 1]
		Matrix4d bMat = Matrix4d.FromColumnMajor(a).Inverse() * Matrix4d.FromColumnMajor(b);
		bMat.CopyToColumnMajor(d[32..]);

		Span<double> coeffs = stackalloc double[30];
		Span<double> solutions = stackalloc double[24];
		SetupCoefficients(d, coeffs);

		int nSols = Re3q3.Solve(coeffs, solutions);

		output.Clear();
		outputFx.Clear();
		outputFy.Clear();

		Span<double> pm = stackalloc double[12];
		for (int i = 0; i < nSols; ++i)
		{
			// P is 3 x 4, column-major: P(r, c) = pm[r + 3 c].
			Vector4d alpha = new(solutions[3 * i], solutions[3 * i + 1], solutions[3 * i + 2], 1.0);
			for (int c = 0; c < 4; ++c)
			{
				// P.block<2,4>(0,0) = Map<2x4>(N * alpha).
				pm[3 * c] = NTimesAlpha(d, 2 * c, alpha);
				pm[3 * c + 1] = NTimesAlpha(d, 2 * c + 1, alpha);
				// P.row(2) = B * alpha.
				pm[3 * c + 2] = bMat[c, 0] * alpha.X + bMat[c, 1] * alpha.Y + bMat[c, 2] * alpha.Z + bMat[c, 3] * alpha.W;
			}

			Matrix3x4d p = Matrix3x4d.FromColumnMajor(pm);
			if (p.LeftCols3().Determinant() < 0)
			{
				p = -p;
			}

			p /= new Vector3d(p[2, 0], p[2, 1], p[2, 2]).Norm;
			double fx = new Vector3d(p[0, 0], p[0, 1], p[0, 2]).Norm;
			double fy = new Vector3d(p[1, 0], p[1, 1], p[1, 2]).Norm;
			p.CopyToColumnMajor(pm);
			for (int c = 0; c < 4; ++c)
			{
				pm[3 * c] /= fx;
				pm[3 * c + 1] /= fy;
			}

			p = Matrix3x4d.FromColumnMajor(pm);
			Matrix3d r = p.LeftCols3();
			Vector3d t = p.Col(3);
			fx *= f0;
			fy *= f0;

			var pose = new CameraPose(r, t);

			if (filterSolutions)
			{
				// Check cheirality
				bool ok = true;
				for (int k = 0; k < 4; ++k)
				{
					if (r.Row(2).Dot(bigX[k]) + t.Z < 0.0)
					{
						ok = false;
						break;
					}
				}

				if (!ok)
				{
					continue;
				}
			}

			output.Add(pose);
			outputFx.Add(fx);
			outputFy.Add(fy);
		}

		return output.Count;
	}

	// Row `row` of N (8 x 4, column-major in d[0..31]) times alpha.
	private static double NTimesAlpha(ReadOnlySpan<double> d, int row, Vector4d alpha) =>
		d[row] * alpha.X + d[row + 8] * alpha.Y + d[row + 16] * alpha.Z + d[row + 24] * alpha.W;

	// Orthogonality constraints
	private static void SetupCoefficients(ReadOnlySpan<double> d, Span<double> coeffs)
	{
		coeffs[0] = d[0] * d[1] + d[2] * d[3] + d[4] * d[5];
		coeffs[3] = d[0] * d[9] + d[1] * d[8] + d[2] * d[11] + d[3] * d[10] + d[4] * d[13] + d[5] * d[12];
		coeffs[6] = d[0] * d[17] + d[1] * d[16] + d[2] * d[19] + d[3] * d[18] + d[4] * d[21] + d[5] * d[20];
		coeffs[9] = d[8] * d[9] + d[10] * d[11] + d[12] * d[13];
		coeffs[12] = d[8] * d[17] + d[9] * d[16] + d[10] * d[19] + d[11] * d[18] + d[12] * d[21] + d[13] * d[20];
		coeffs[15] = d[16] * d[17] + d[18] * d[19] + d[20] * d[21];
		coeffs[18] = d[0] * d[25] + d[1] * d[24] + d[2] * d[27] + d[3] * d[26] + d[4] * d[29] + d[5] * d[28];
		coeffs[21] = d[8] * d[25] + d[9] * d[24] + d[10] * d[27] + d[11] * d[26] + d[12] * d[29] + d[13] * d[28];
		coeffs[24] = d[16] * d[25] + d[17] * d[24] + d[18] * d[27] + d[19] * d[26] + d[20] * d[29] + d[21] * d[28];
		coeffs[27] = d[24] * d[25] + d[26] * d[27] + d[28] * d[29];
		coeffs[1] = d[0] * d[32] + d[2] * d[33] + d[4] * d[34];
		coeffs[4] = d[0] * d[36] + d[2] * d[37] + d[8] * d[32] + d[4] * d[38] + d[10] * d[33] + d[12] * d[34];
		coeffs[7] = d[0] * d[40] + d[2] * d[41] + d[4] * d[42] + d[16] * d[32] + d[18] * d[33] + d[20] * d[34];
		coeffs[10] = d[8] * d[36] + d[10] * d[37] + d[12] * d[38];
		coeffs[13] = d[8] * d[40] + d[10] * d[41] + d[16] * d[36] + d[12] * d[42] + d[18] * d[37] + d[20] * d[38];
		coeffs[16] = d[16] * d[40] + d[18] * d[41] + d[20] * d[42];
		coeffs[19] = d[0] * d[44] + d[2] * d[45] + d[4] * d[46] + d[24] * d[32] + d[26] * d[33] + d[28] * d[34];
		coeffs[22] = d[8] * d[44] + d[10] * d[45] + d[12] * d[46] + d[24] * d[36] + d[26] * d[37] + d[28] * d[38];
		coeffs[25] = d[16] * d[44] + d[18] * d[45] + d[24] * d[40] + d[20] * d[46] + d[26] * d[41] + d[28] * d[42];
		coeffs[28] = d[24] * d[44] + d[26] * d[45] + d[28] * d[46];
		coeffs[2] = d[1] * d[32] + d[3] * d[33] + d[5] * d[34];
		coeffs[5] = d[1] * d[36] + d[3] * d[37] + d[9] * d[32] + d[5] * d[38] + d[11] * d[33] + d[13] * d[34];
		coeffs[8] = d[1] * d[40] + d[3] * d[41] + d[5] * d[42] + d[17] * d[32] + d[19] * d[33] + d[21] * d[34];
		coeffs[11] = d[9] * d[36] + d[11] * d[37] + d[13] * d[38];
		coeffs[14] = d[9] * d[40] + d[11] * d[41] + d[17] * d[36] + d[13] * d[42] + d[19] * d[37] + d[21] * d[38];
		coeffs[17] = d[17] * d[40] + d[19] * d[41] + d[21] * d[42];
		coeffs[20] = d[1] * d[44] + d[3] * d[45] + d[5] * d[46] + d[25] * d[32] + d[27] * d[33] + d[29] * d[34];
		coeffs[23] = d[9] * d[44] + d[11] * d[45] + d[13] * d[46] + d[25] * d[36] + d[27] * d[37] + d[29] * d[38];
		coeffs[26] = d[17] * d[44] + d[19] * d[45] + d[25] * d[40] + d[21] * d[46] + d[27] * d[41] + d[29] * d[42];
		coeffs[29] = d[25] * d[44] + d[27] * d[45] + d[29] * d[46];
	}
}
