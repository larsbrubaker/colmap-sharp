// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FrameTests: colmap/scene/frame_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Scene/Frame.cs. frame_test.cc names one of its
// cases TEST(Image, Equals) although it tests Frame; the name is kept. C++ copy
// construction/assignment is Frame.Clone(); `frame.SetRigPtr(&rig)` passes the reference;
// operator<< is ToString(); UnorderedElementsAre compares as sets.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test that draws
// starts with RandomUtils.SetPRNGSeed(0) and draws everything before its first await (the
// PRNG is per thread and an await may resume elsewhere).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Tests.EigenMatchers;
using static ColmapSharp.Util.Types;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class FrameTests
{
	private static Rigid3d TestRigid3d() =>
		new(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());

	[Test]
	public async Task Frame_Default()
	{
		var frame = new Frame();
		using (Assert.Multiple())
		{
			await Assert.That(frame.FrameId).IsEqualTo(InvalidFrameId);
			await Assert.That(frame.HasPose).IsFalse();
			await Assert.That(frame.RigId).IsEqualTo(InvalidRigId);
			await Assert.That(frame.HasRigId).IsFalse();
			await Assert.That(frame.HasRigPtr).IsFalse();
			await Assert.That(frame.NumDataIds).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Frame_Copy()
	{
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(2);
		var dataId1 = new DataId(new SensorId(SensorType.Camera, 1), 1);
		var dataId2 = new DataId(new SensorId(SensorType.Camera, 1), 2);
		frame.AddDataId(dataId1);
		frame.FinalizeDataIds();

		using (Assert.Multiple())
		{
			// Copy constructor: copy should not be finalized.
			Frame copy = frame.Clone();
			await Assert.That(copy.HasFinalDataIds).IsFalse();
			await Assert.That(copy.FrameId).IsEqualTo(1u);
			await Assert.That(copy.RigId).IsEqualTo(2u);
			await Assert.That(copy.HasDataId(dataId1)).IsTrue();
			await Assert.That(() => copy.AddDataId(dataId2)).ThrowsNothing();
			await Assert.That(copy.HasDataId(dataId2)).IsTrue();
			await Assert.That(() => copy.ClearDataIds()).ThrowsNothing();
			await Assert.That(copy.NumDataIds).IsEqualTo(0);

			// Copy assignment: target should not be finalized.
			var assigned = new Frame();
			assigned = frame.Clone();
			await Assert.That(assigned.HasFinalDataIds).IsFalse();
			await Assert.That(assigned.FrameId).IsEqualTo(1u);
			await Assert.That(assigned.HasDataId(dataId1)).IsTrue();
			await Assert.That(() => assigned.AddDataId(dataId2)).ThrowsNothing();

			// Original remains finalized.
			await Assert.That(frame.HasFinalDataIds).IsTrue();
			await Assert.That(() => frame.AddDataId(dataId2)).ThrowsException();
		}
	}

	[Test]
	public async Task Frame_SetUp()
	{
		RandomUtils.SetPRNGSeed(0);
		var frame = new Frame();
		var rig = new Rig { RigId = 1 };

		await Assert.That(frame.HasRigId).IsFalse();
		await Assert.That(frame.HasRigPtr).IsFalse();

		frame.SetRigId(1);
		await Assert.That(frame.HasRigId).IsTrue();
		await Assert.That(frame.HasRigPtr).IsFalse();

		var sensorId1 = new SensorId(SensorType.Imu, 0);
		rig.AddRefSensor(sensorId1);
		var sensorId2 = new SensorId(SensorType.Camera, 0);
		rig.AddSensor(sensorId2, TestRigid3d());
		frame.SetRigPtr(rig);
		await Assert.That(frame.HasRigPtr).IsTrue();

		await Assert.That(frame.DataIds).IsEmpty();
		var dataId1 = new DataId(sensorId1, 2);
		await Assert.That(frame.HasDataId(dataId1)).IsFalse();
		frame.AddDataId(dataId1);
		await Assert.That(frame.NumDataIds).IsEqualTo(1);
		await Assert.That(frame.HasDataId(dataId1)).IsTrue();
		await Assert.That(frame.DataIds.SetEquals([dataId1])).IsTrue();
		var dataId2 = new DataId(sensorId2, 5);
		await Assert.That(frame.HasDataId(dataId2)).IsFalse();
		frame.AddDataId(dataId2);
		await Assert.That(frame.NumDataIds).IsEqualTo(2);
		await Assert.That(frame.HasDataId(dataId2)).IsTrue();
		await Assert.That(frame.DataIds.SetEquals([dataId1, dataId2])).IsTrue();
		frame.ClearDataIds();
		await Assert.That(frame.NumDataIds).IsEqualTo(0);
		await Assert.That(frame.DataIds).IsEmpty();
		await Assert.That(frame.HasPose).IsFalse();
	}

	[Test]
	public async Task Frame_SetResetRigPtr()
	{
		RandomUtils.SetPRNGSeed(0);
		Rigid3d sensor2FromRig = TestRigid3d();
		var frame = new Frame();
		var rig = new Rig();

		await Assert.That(frame.HasRigId).IsFalse();
		await Assert.That(frame.HasRigPtr).IsFalse();

		frame.SetRigId(1);
		await Assert.That(frame.HasRigId).IsTrue();
		await Assert.That(frame.HasRigPtr).IsFalse();

		await Assert.That(() => frame.SetRigPtr(rig)).ThrowsException();
		rig.RigId = 1;
		frame.SetRigPtr(rig);
		await Assert.That(frame.HasRigPtr).IsTrue();
		frame.ResetRigPtr();
		await Assert.That(frame.HasRigPtr).IsFalse();

		var sensorId1 = new SensorId(SensorType.Imu, 0);
		var sensorId2 = new SensorId(SensorType.Camera, 0);
		frame.AddDataId(new DataId(sensorId1, 1));
		frame.AddDataId(new DataId(sensorId2, 1));
		await Assert.That(() => frame.SetRigPtr(rig)).ThrowsException();

		rig.AddRefSensor(sensorId1);
		rig.AddSensor(sensorId2, sensor2FromRig);
		frame.SetRigPtr(rig);
		await Assert.That(frame.HasRigPtr).IsTrue();
	}

	[Test]
	public async Task Frame_AddDataId()
	{
		RandomUtils.SetPRNGSeed(0);
		var rig = new Rig { RigId = 1 };
		var sensorId1 = new SensorId(SensorType.Imu, 0);
		var sensorId2 = new SensorId(SensorType.Camera, 0);
		var sensorId3 = new SensorId(SensorType.Camera, 1);
		rig.AddRefSensor(sensorId1);
		rig.AddSensor(sensorId2, TestRigid3d());

		var frame = new Frame();
		frame.SetRigId(1);
		frame.SetRigPtr(rig);
		frame.AddDataId(new DataId(sensorId1, 1));
		frame.AddDataId(new DataId(sensorId2, 1));
		await Assert.That(() => frame.AddDataId(new DataId(sensorId3, 2))).ThrowsException();
	}

	[Test]
	public async Task Frame_FilteredDataIds()
	{
		var frame = new Frame();
		var dataId1 = new DataId(new SensorId(SensorType.Imu, 0), 2);
		frame.AddDataId(dataId1);
		var dataId2 = new DataId(new SensorId(SensorType.Camera, 0), 2);
		frame.AddDataId(dataId2);
		var dataId3 = new DataId(new SensorId(SensorType.Camera, 1), 1);
		frame.AddDataId(dataId3);
		using (Assert.Multiple())
		{
			await Assert.That(frame.DataIdsOfType(SensorType.Imu).ToHashSet().SetEquals([dataId1])).IsTrue();
			await Assert.That(frame.DataIdsOfType(SensorType.Camera).ToHashSet().SetEquals([dataId2, dataId3])).IsTrue();
			await Assert.That(frame.DataIdsOfType(SensorType.Invalid)).IsEmpty();
		}
	}

	[Test]
	public async Task Frame_ImageIds()
	{
		var frame = new Frame();
		var dataId1 = new DataId(new SensorId(SensorType.Imu, 0), 2);
		frame.AddDataId(dataId1);
		var dataId2 = new DataId(new SensorId(SensorType.Camera, 0), 2);
		frame.AddDataId(dataId2);
		var dataId3 = new DataId(new SensorId(SensorType.Camera, 1), 1);
		frame.AddDataId(dataId3);
		await Assert.That(frame.ImageIds().ToHashSet().SetEquals([dataId2, dataId3])).IsTrue();
	}

	[Test]
	public async Task Frame_SetResetPose()
	{
		var frame = new Frame();
		using (Assert.Multiple())
		{
			await Assert.That(frame.HasPose).IsFalse();
			await Assert.That(() => frame.RigFromWorld()).ThrowsException();
			await Assert.That(frame.MaybeRigFromWorld.HasValue).IsFalse();
			frame.SetRigFromWorld(new Rigid3d());
			await Assert.That(frame.HasPose).IsTrue();
			await Assert.That(frame.RigFromWorld() == new Rigid3d()).IsTrue();
			await Assert.That(frame.MaybeRigFromWorld!.Value == new Rigid3d()).IsTrue();
			frame.ResetPose();
			await Assert.That(frame.HasPose).IsFalse();
			await Assert.That(() => frame.RigFromWorld()).ThrowsException();
			await Assert.That(frame.MaybeRigFromWorld.HasValue).IsFalse();
		}
	}

	[Test]
	public async Task Frame_SetCamFromWorld()
	{
		RandomUtils.SetPRNGSeed(0);
		var frame = new Frame();
		var rig = new Rig { RigId = 1 };
		frame.SetRigId(1);
		var sensorId1 = new SensorId(SensorType.Camera, 0);
		rig.AddRefSensor(sensorId1);
		var sensorId2 = new SensorId(SensorType.Camera, 1);
		rig.AddSensor(sensorId2, TestRigid3d());
		frame.SetRigPtr(rig);

		Rigid3d cam1FromWorld = TestRigid3d();
		frame.SetCamFromWorld(sensorId1.Id, cam1FromWorld);
		Rigid3d rigFromWorld1 = frame.RigFromWorld();
		Rigid3d sensor1FromWorld = frame.SensorFromWorld(sensorId1);

		Rigid3d cam2FromWorld = TestRigid3d();
		frame.SetCamFromWorld(sensorId2.Id, cam2FromWorld);
		Rigid3d sensor2FromWorld = frame.SensorFromWorld(sensorId2);

		using (Assert.Multiple())
		{
			await Assert.That(rigFromWorld1 == cam1FromWorld).IsTrue();
			await Assert.That(sensor1FromWorld == cam1FromWorld).IsTrue();
			await Assert.That(EigenMatrixNear(cam2FromWorld.Translation, sensor2FromWorld.Translation, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(cam2FromWorld.Rotation.Coeffs, sensor2FromWorld.Rotation.Coeffs, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task Image_Equals()
	{
		var frame = new Frame();
		Frame other = frame.Clone();
		await Assert.That(frame == other).IsTrue();
		frame.FrameId = 2;
		await Assert.That(frame != other).IsTrue();
		other.FrameId = 2;
		await Assert.That(frame == other).IsTrue();
	}

	[Test]
	public async Task Frame_Print()
	{
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(2);
		frame.AddDataId(new DataId(new SensorId(SensorType.Imu, 0), 2));
		frame.AddDataId(new DataId(new SensorId(SensorType.Camera, 1), 3));
		await Assert.That(frame.ToString()).IsEqualTo(
			"Frame(frame_id=1, rig_id=2, has_pose=0, data_ids=[(CAMERA, 1, 3), (IMU, 0, 2)])");
	}

	[Test]
	public async Task Frame_FinalizeDataIds()
	{
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(2);
		var dataId1 = new DataId(new SensorId(SensorType.Camera, 1), 1);
		var dataId2 = new DataId(new SensorId(SensorType.Camera, 1), 2);
		frame.AddDataId(dataId1);
		using (Assert.Multiple())
		{
			await Assert.That(frame.HasFinalDataIds).IsFalse();
			frame.FinalizeDataIds();
			await Assert.That(frame.HasFinalDataIds).IsTrue();
			await Assert.That(() => frame.AddDataId(dataId2)).ThrowsException();
			await Assert.That(() => frame.ClearDataIds()).ThrowsException();
			await Assert.That(frame.NumDataIds).IsEqualTo(1);
			await Assert.That(frame.HasDataId(dataId1)).IsTrue();
		}
	}

	// C#-only (no frame_test.cc counterpart): pins the in-place pose storage that Phase 8's
	// bundle adjustment writes through (Frame.cs header). Params is one 7-value array in
	// colmap::Rigid3d::params order that stays the same object across pose resets, and a
	// write into it is the frame's new pose.
	[Test]
	public async Task RigFromWorldStorage_IsStableAndWritable()
	{
		var frame = new Frame();
		await Assert.That(() => frame.RigFromWorldStorage).ThrowsException();
		frame.SetRigFromWorld(new Rigid3d());
		Rigid3dStorage storage = frame.RigFromWorldStorage;
		double[] parameters = storage.Params;

		// [qx, qy, qz, qw, tx, ty, tz].
		await Assert.That(parameters.SequenceEqual([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])).IsTrue();
		parameters[6] = 5;
		await Assert.That(frame.RigFromWorld().Translation).IsEqualTo(new Vector3d(0, 0, 5));

		// Rotation and Translation are views of the same array.
		ArraySegment<double> translation = storage.Translation;
		translation[0] = 7;
		await Assert.That(parameters[4]).IsEqualTo(7);
		await Assert.That(ReferenceEquals(storage.Rotation.Array, parameters)).IsTrue();

		frame.ResetPose();
		frame.SetRigFromWorld(new Rigid3d(Quaterniond.Identity, new Vector3d(1, 2, 3)));
		using (Assert.Multiple())
		{
			await Assert.That(ReferenceEquals(frame.RigFromWorldStorage.Params, parameters)).IsTrue();
			await Assert.That(parameters.SequenceEqual([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0])).IsTrue();
		}
	}
}
