// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseLibUtils: colmap/estimators/solvers/poselib_utils.h and .cc - conversions between
// COLMAP's Camera and Rigid3d and PoseLib's Camera and CameraPose
// (Estimators/Solvers/PoseLib/PoseLibCamera.cs, CameraPose.cs). Used by the solvers that
// wrap PoseLib (AbsolutePose.cs and the generalized / focal-length solvers).
// Tests: ColmapSharp.Tests/Estimators/Solvers/PoseLibUtilsTests.cs (poselib_utils_test.cc).
//
// Tier A: plain copies; the quaternion only changes order.

using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>COLMAP / PoseLib type conversions. Port of colmap/estimators/solvers/poselib_utils.</summary>
public static class PoseLibUtils
{
	/// <summary>Convert COLMAP Camera to PoseLib Camera.</summary>
	public static PoseLibCamera ConvertCameraToPoseLibCamera(Camera camera) =>
		new(camera.ModelName, camera.Params, camera.Width, camera.Height);

	/// <summary>Convert PoseLib Camera to COLMAP Camera.</summary>
	public static Camera ConvertPoseLibCameraToCamera(PoseLibCamera camera) => new()
	{
		ModelId = CameraModels.CameraModelNameToId(camera.ModelName),
		Width = camera.Width,
		Height = camera.Height,
		Params = [.. camera.Params],
	};

	/// <summary>Convert COLMAP Rigid3d to PoseLib CameraPose.</summary>
	public static CameraPose ConvertRigid3dToPoseLibPose(Rigid3d rigid)
	{
		// PoseLib uses (w, x, y, z) ordering for quaternion
		Quaterniond r = rigid.Rotation;
		return new CameraPose(new Vector4d(r.W, r.X, r.Y, r.Z), rigid.Translation);
	}

	/// <summary>Convert PoseLib CameraPose to COLMAP Rigid3d.</summary>
	public static Rigid3d ConvertPoseLibPoseToRigid3d(CameraPose pose)
	{
		// PoseLib quaternion is (w, x, y, z), Eigen::Quaterniond constructor is also
		// (w, x, y, z)
		return new Rigid3d(new Quaterniond(pose.Q.X, pose.Q.Y, pose.Q.Z, pose.Q.W), pose.T);
	}
}
