// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// Gp3p: PoseLib/solvers/gp3p.cc - the minimal generalized absolute pose solver (three rays
// with individual origins p and directions x, and their three world points X). The six
// projection equations are linear in [t; vec(R); 1]; the translation is eliminated with the
// first three and the remaining three constrain the rotation alone, which
// Re3q3.SolveRotation solves. Called by GP3PEstimator
// (Estimators/Solvers/GeneralizedAbsolutePose.cs).
//
// Translation notes:
// - The 6 x 13 system A and the 3 x 10 rotation system AR are column-major stack spans.
// - Re3q3.SolveRotation draws its random pre-rotation from a fixed-seed mt19937 rather than
//   std::rand (divergence 29).
//
// Tier B.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>Generalized P3P. Port of poselib::gp3p.</summary>
public static class Gp3p
{
	/// <summary>
	/// Solve for the poses (x_cam = R X + t in the generalized camera frame) with
	/// lambda_i x_i = R X_i + t - p_i. Clears <paramref name="output"/>, appends up to eight
	/// poses and returns their count.
	/// </summary>
	/// <param name="p">The ray origins in the generalized camera frame.</param>
	/// <param name="x">The ray directions in the generalized camera frame.</param>
	/// <param name="bigX">The world points.</param>
	/// <param name="output">Receives the poses.</param>
	public static int Solve(ReadOnlySpan<Vector3d> p, ReadOnlySpan<Vector3d> x, ReadOnlySpan<Vector3d> bigX, List<CameraPose> output)
	{
		// A(r, c) = a[r + 6 c].
		Span<double> a = stackalloc double[6 * 13];

		for (int i = 0; i < 3; ++i)
		{
			// xx = [x3 0 -x1; 0 x3 -x2]
			// eqs = [xx kron(X',xx), -xx*p] * [t; vec(R); 1]
			ReadOnlySpan<double> row0 =
			[
				x[i].Z, 0.0, -x[i].X, bigX[i].X * x[i].Z, 0.0, -bigX[i].X * x[i].X, bigX[i].Y * x[i].Z, 0.0,
				-bigX[i].Y * x[i].X, bigX[i].Z * x[i].Z, 0.0, -bigX[i].Z * x[i].X, -p[i].X * x[i].Z + p[i].Z * x[i].X,
			];
			ReadOnlySpan<double> row1 =
			[
				0.0, x[i].Z, -x[i].Y, 0.0, bigX[i].X * x[i].Z, -bigX[i].X * x[i].Y, 0.0, bigX[i].Y * x[i].Z,
				-bigX[i].Y * x[i].Y, 0.0, bigX[i].Z * x[i].Z, -bigX[i].Z * x[i].Y, -p[i].Y * x[i].Z + p[i].Z * x[i].Y,
			];
			for (int c = 0; c < 13; ++c)
			{
				a[2 * i + 6 * c] = row0[c];
				a[2 * i + 1 + 6 * c] = row1[c];
			}
		}

		Matrix3d b = new Matrix3d(
			a[0], a[6], a[12],
			a[1], a[7], a[13],
			a[2], a[8], a[14]).Inverse();

		// Top = A.block<3, 3>(3, 0) * B (3 x 3), then
		// AR = A.block<3, 10>(3, 3) - Top * A.block<3, 10>(0, 3).
		Matrix3d bottomLeft = new Matrix3d(
			a[3], a[9], a[15],
			a[4], a[10], a[16],
			a[5], a[11], a[17]);
		Matrix3d top = bottomLeft * b;

		// AR(r, c) = ar[r + 3 c].
		Span<double> ar = stackalloc double[30];
		for (int c = 0; c < 10; ++c)
		{
			int col = c + 3;
			for (int r = 0; r < 3; ++r)
			{
				double sum = top[r, 0] * a[0 + 6 * col] + top[r, 1] * a[1 + 6 * col] + top[r, 2] * a[2 + 6 * col];
				ar[r + 3 * c] = a[3 + r + 6 * col] - sum;
			}
		}

		Span<Vector4d> solutions = stackalloc Vector4d[8];
		int nSols = Re3q3.SolveRotation(ar, solutions);

		output.Clear();
		for (int i = 0; i < nSols; ++i)
		{
			Vector4d q = solutions[i];
			Matrix3d rot = CameraPose.QuatToRotmat(q);

			// A.block<3, 9>(0, 3) * quat_to_rotmatvec(q) + A.block<3, 1>(0, 12), where
			// quat_to_rotmatvec is R stacked column-major.
			Vector3d rhs = new Vector3d(
				RotationRow(a, 0, rot) + a[0 + 6 * 12],
				RotationRow(a, 1, rot) + a[1 + 6 * 12],
				RotationRow(a, 2, rot) + a[2 + 6 * 12]);

			Vector3d t = -(b * rhs);
			output.Add(new CameraPose(q, t));
		}

		return nSols;
	}

	// Row r of A.block<3, 9>(0, 3) times vec(R) (column-major).
	private static double RotationRow(ReadOnlySpan<double> a, int r, Matrix3d rot)
	{
		double sum = 0.0;
		for (int k = 0; k < 9; ++k)
		{
			sum += a[r + 6 * (3 + k)] * rot[k % 3, k / 3];
		}

		return sum;
	}
}
