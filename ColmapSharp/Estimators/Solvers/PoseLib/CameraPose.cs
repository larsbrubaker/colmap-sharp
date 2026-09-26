// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08, the version
// COLMAP 4.2.0 fetches.
//
// CameraPose: PoseLib/camera_pose.h (struct CameraPose) and the two quaternion helpers of
// PoseLib/misc/quaternion.h it is built on (rotmat_to_quat, quat_to_rotmat). The pose type
// every PoseLib solver returns (P3p.cs, P4pf.cs); Estimators/Solvers/PoseLibUtils.cs converts
// it to and from COLMAP's Rigid3d. Also ImagePair (camera_pose.h), the pose-plus-cameras
// output of the focal relative pose solvers (Relpose6ptSharedFocal.cs, Relpose6ptOnesidedFocal.cs).
//
// Only the members COLMAP's callers use are ported (construction from R and t, R(), Rt());
// PoseLib's rotate/compose/center helpers are used only inside PoseLib's own refinement
// code, which COLMAP does not call.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>
/// A camera pose x_cam = R x_world + t with the rotation as a unit quaternion. Port of
/// poselib::CameraPose.
/// </summary>
public readonly struct CameraPose
{
	/// <summary>
	/// The rotation quaternion in PoseLib's order, real part first: Q.X = qw, Q.Y = qx,
	/// Q.Z = qy, Q.W = qz. (A Vector4d rather than a Quaterniond because PoseLib stores and
	/// normalizes it as a plain 4-vector in that order.)
	/// </summary>
	public Vector4d Q { get; }

	/// <summary>The translation.</summary>
	public Vector3d T { get; }

	/// <summary>The identity pose.</summary>
	public CameraPose()
	{
		Q = new Vector4d(1, 0, 0, 0);
		T = Vector3d.Zero;
	}

	/// <summary>A pose from a (qw, qx, qy, qz) quaternion and a translation.</summary>
	public CameraPose(Vector4d q, Vector3d t)
	{
		Q = q;
		T = t;
	}

	/// <summary>A pose from a rotation matrix and a translation (rotmat_to_quat).</summary>
	public CameraPose(Matrix3d r, Vector3d t)
	{
		Q = RotmatToQuat(r);
		T = t;
	}

	/// <summary>The rotation matrix.</summary>
	public Matrix3d R() => QuatToRotmat(Q);

	/// <summary>The 3x4 matrix [R | t].</summary>
	public Matrix3x4d Rt() => Matrix3x4d.FromBlocks(QuatToRotmat(Q), T);

	/// <summary>Port of poselib::quat_to_rotmat.</summary>
	public static Matrix3d QuatToRotmat(Vector4d q) => new Quaterniond(q.X, q.Y, q.Z, q.W).ToRotationMatrix();

	/// <summary>Port of poselib::rotmat_to_quat: Eigen's matrix-to-quaternion, reordered and normalized.</summary>
	public static Vector4d RotmatToQuat(Matrix3d r)
	{
		Quaterniond qFlip = Quaterniond.FromRotationMatrix(r);
		return new Vector4d(qFlip.W, qFlip.X, qFlip.Y, qFlip.Z).Normalized();
	}
}

/// <summary>
/// Two cameras and their relative pose, the output of PoseLib's relative pose solvers with
/// unknown focal lengths (Relpose6ptSharedFocal.cs, Relpose6ptOnesidedFocal.cs). Port of
/// poselib::ImagePair (camera_pose.h).
/// </summary>
public readonly record struct ImagePair(CameraPose Pose, PoseLibCamera Camera1, PoseLibCamera Camera2);
