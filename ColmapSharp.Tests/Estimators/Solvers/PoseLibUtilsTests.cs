// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseLibUtilsTests: colmap/estimators/solvers/poselib_utils_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name. Tests
// ColmapSharp/Estimators/Solvers/PoseLibUtils.cs. Tier A (plain copies). The CSharpOnly_
// tests are not from COLMAP; they pin details the ported cases cannot see.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class PoseLibUtilsTests
{
	[Test]
	public async Task PoseLibUtils_CameraRoundTrip()
	{
		var camera = new Camera
		{
			ModelId = CameraModelId.SimpleRadial,
			Width = 1920,
			Height = 1080,
			Params = [1000.0, 960.0, 540.0, 0.1],
		};

		PoseLibCamera poselibCamera = PoseLibUtils.ConvertCameraToPoseLibCamera(camera);
		Camera cameraBack = PoseLibUtils.ConvertPoseLibCameraToCamera(poselibCamera);

		await Assert.That(cameraBack.ModelId).IsEqualTo(camera.ModelId);
		await Assert.That(cameraBack.Width).IsEqualTo(camera.Width);
		await Assert.That(cameraBack.Height).IsEqualTo(camera.Height);
		await Assert.That(cameraBack.Params.SequenceEqual(camera.Params)).IsTrue();
	}

	[Test]
	public async Task PoseLibUtils_Rigid3dRoundTrip()
	{
		Quaterniond rotation = new Quaterniond(0.5, 0.5, 0.5, 0.5).Normalized();
		var translation = new Vector3d(1.0, 2.0, 3.0);
		var rigid = new Rigid3d(rotation, translation);

		CameraPose poselibPose = PoseLibUtils.ConvertRigid3dToPoseLibPose(rigid);
		Rigid3d rigidBack = PoseLibUtils.ConvertPoseLibPoseToRigid3d(poselibPose);

		await Assert.That(Rigid3dNear(rigidBack, rigid, 1e-10, 1e-10)).IsTrue();
	}

	/// <summary>
	/// C#-only: with an asymmetric quaternion, the PoseLib pose stores (w, x, y, z) in that
	/// order, describes the same rotation, and converts back exactly. (COLMAP's round trip uses
	/// (0.5, 0.5, 0.5, 0.5), which cannot tell the orders apart.)
	/// </summary>
	[Test]
	public async Task CSharpOnly_Rigid3dRoundTripAsymmetricQuaternion()
	{
		Quaterniond rotation = new Quaterniond(0.9, 0.1, -0.2, 0.3).Normalized();
		var rigid = new Rigid3d(rotation, new Vector3d(-1.5, 0.25, 4.0));

		CameraPose pose = PoseLibUtils.ConvertRigid3dToPoseLibPose(rigid);
		await Assert.That(pose.Q).IsEqualTo(new Vector4d(rotation.W, rotation.X, rotation.Y, rotation.Z));
		await Assert.That(pose.T).IsEqualTo(rigid.Translation);
		await Assert.That(pose.R().IsApprox(rotation.ToRotationMatrix(), 1e-15)).IsTrue();

		Rigid3d rigidBack = PoseLibUtils.ConvertPoseLibPoseToRigid3d(pose);
		await Assert.That(Rigid3dEq(rigidBack, rigid)).IsTrue();
	}
}
