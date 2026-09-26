// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08 as pinned by
// COLMAP's src/thirdparty/CMakeLists.txt.
//
// Essential: the parts of PoseLib/misc/essential.cc the focal-length relative pose solvers
// use - motion_from_essential (the SVD-free decomposition of an essential matrix into the
// four candidate poses, kept only if all points pass cheirality) and the central
// check_cheirality - plus quat_rotate from misc/quaternion.h, which check_cheirality rotates
// with. Callers: Relpose6ptSharedFocal.cs and Relpose6ptOnesidedFocal.cs. GenRelpose6pt.cs
// keeps its own generalized cheirality test.
//
// Tier B: every step is scalar and follows PoseLib's operation order; the rotation goes
// through CameraPose.RotmatToQuat (Eigen's matrix-to-quaternion), which Tier A does not cover.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>Essential matrix decomposition helpers. Port of poselib's misc/essential.cc.</summary>
internal static class Essential
{
	/// <summary>
	/// Decomposes <paramref name="e"/> into the (up to four) relative poses whose
	/// correspondences (unit bearings <paramref name="x1"/>, <paramref name="x2"/>) all lie in
	/// front of both cameras, and appends them to <paramref name="relativePoses"/>.
	/// Port of poselib::motion_from_essential.
	/// </summary>
	public static void MotionFromEssential(in Matrix3d e, ReadOnlySpan<Vector3d> x1, ReadOnlySpan<Vector3d> x2, List<CameraPose> relativePoses)
	{
		// Compute the necessary cross products
		Vector3d e0 = e.Col(0), e1 = e.Col(1), e2 = e.Col(2);
		Vector3d u12 = e0.Cross(e1);
		Vector3d u13 = e0.Cross(e2);
		Vector3d u23 = e1.Cross(e2);
		double n12 = u12.SquaredNorm;
		double n13 = u13.SquaredNorm;
		double n23 = u23.SquaredNorm;

		// Compute the U*W factor
		Vector3d uw1, uw2;
		if (n12 > n13)
		{
			if (n12 > n23)
			{
				uw1 = e0.Normalized();
				uw2 = u12 / Math.Sqrt(n12);
			}
			else
			{
				uw1 = e1.Normalized();
				uw2 = u23 / Math.Sqrt(n23);
			}
		}
		else
		{
			if (n13 > n23)
			{
				uw1 = e0.Normalized();
				uw2 = u13 / Math.Sqrt(n13);
			}
			else
			{
				uw1 = e1.Normalized();
				uw2 = u23 / Math.Sqrt(n23);
			}
		}

		Vector3d uw0 = -uw2.Cross(uw1);

		// Compute the V factor: row0 = UW.col(1)^T E, row1 = -UW.col(0)^T E.
		Matrix3d et = e.Transpose();
		Vector3d vt0 = (et * uw1).Normalized();
		Vector3d vt1 = et * -uw0;

		// Here v1 and v2 should be orthogonal. However, if E is not exactly an essential matrix
		// they might not be. To ensure we end up with a rotation matrix we orthogonalize them
		// again here, this should be a nop for good data
		vt1 -= vt0.Dot(vt1) * vt0;

		vt1 = vt1.Normalized();
		Vector3d vt2 = vt0.Cross(vt1);
		Matrix3d vt = Matrix3d.FromRows(vt0, vt1, vt2);

		var pose = new CameraPose(Matrix3d.FromColumns(uw0, uw1, uw2) * vt, uw2);
		AddIfCheiral(pose, x1, x2, relativePoses);
		pose = new CameraPose(pose.Q, -pose.T);
		AddIfCheiral(pose, x1, x2, relativePoses);

		// U * W.transpose()
		pose = new CameraPose(CameraPose.RotmatToQuat(Matrix3d.FromColumns(-uw0, -uw1, uw2) * vt), pose.T);
		AddIfCheiral(pose, x1, x2, relativePoses);
		pose = new CameraPose(pose.Q, -pose.T);
		AddIfCheiral(pose, x1, x2, relativePoses);
	}

	/// <summary>
	/// Whether the triangulated depths of the unit bearings x1, x2 are both above
	/// <paramref name="minDepth"/>. Port of poselib::check_cheirality (central form).
	/// </summary>
	public static bool CheckCheirality(in CameraPose pose, Vector3d x1, Vector3d x2, double minDepth = 0.0)
	{
		// This code assumes that x1 and x2 are unit vectors
		Vector3d rx1 = QuatRotate(pose.Q, x1);

		// [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
		// [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
		double a = -rx1.Dot(x2);
		double b1 = -rx1.Dot(pose.T);
		double b2 = x2.Dot(pose.T);

		// Note that we drop the factor 1.0/(1-a*a) since it is always positive.
		double lambda1 = b1 - a * b2;
		double lambda2 = -a * b1 + b2;

		minDepth *= 1 - a * a;
		return lambda1 > minDepth && lambda2 > minDepth;
	}

	/// <summary>PoseLib's quat_rotate (misc/quaternion.h), q = (w, x, y, z).</summary>
	public static Vector3d QuatRotate(Vector4d q, Vector3d p)
	{
		double q1 = q.X, q2 = q.Y, q3 = q.Z, q4 = q.W;
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

	// The vector wrapper of check_cheirality: every correspondence must pass.
	private static void AddIfCheiral(in CameraPose pose, ReadOnlySpan<Vector3d> x1, ReadOnlySpan<Vector3d> x2, List<CameraPose> relativePoses)
	{
		for (int i = 0; i < x1.Length; ++i)
		{
			if (!CheckCheirality(pose, x1[i], x2[i]))
			{
				return;
			}
		}

		relativePoses.Add(pose);
	}
}
