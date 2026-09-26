// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RigTests: colmap/sensor/rig_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, same checks. Tests ColmapSharp/Sensor/Rig.cs. COLMAP's operator<< is
// Rig.ToString(); UnorderedElementsAre compares as sets.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and draws everything before its first await (the PRNG is per
// thread and an await may resume elsewhere).

using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class RigTests
{
	private static Rigid3d TestRigid3d() =>
		new(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());

	[Test]
	public async Task Rig_Default()
	{
		var rig = new Rig();
		using (Assert.Multiple())
		{
			await Assert.That(rig.RigId).IsEqualTo(InvalidRigId);
			await Assert.That(rig.RefSensorId).IsEqualTo(InvalidSensorId);
			await Assert.That(rig.NumSensors).IsEqualTo(0);
			await Assert.That(rig.NonRefSensors.Count).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Rig_SetUp()
	{
		RandomUtils.SetPRNGSeed(0);
		var rig = new Rig();
		var sensorId0 = new SensorId(SensorType.Imu, 0);
		rig.AddRefSensor(sensorId0);
		var sensorId1 = new SensorId(SensorType.Imu, 1);
		Rigid3d sensor1FromRig = TestRigid3d();
		rig.AddSensor(sensorId1, sensor1FromRig);
		var sensorId2 = new SensorId(SensorType.Camera, 0);
		Rigid3d sensor2FromRig = TestRigid3d();
		rig.AddSensor(sensorId2, sensor2FromRig);
		var sensorId3 = new SensorId(SensorType.Camera, 1);
		rig.AddSensor(sensorId3); // no input sensor_from_rig
		Rigid3d sensor3FromRig = TestRigid3d();

		using (Assert.Multiple())
		{
			await Assert.That(rig.NumSensors).IsEqualTo(4);
			await Assert.That(rig.NonRefSensors.Count).IsEqualTo(3);
			await Assert.That(rig.SensorIds().SetEquals([sensorId0, sensorId1, sensorId2, sensorId3])).IsTrue();

			await Assert.That(rig.RefSensorId.Type).IsEqualTo(SensorType.Imu);
			await Assert.That(rig.RefSensorId.Id).IsEqualTo(0u);

			await Assert.That(rig.IsRefSensor(sensorId0)).IsTrue();
			await Assert.That(rig.IsRefSensor(sensorId1)).IsFalse();
			await Assert.That(rig.IsRefSensor(sensorId2)).IsFalse();
			await Assert.That(rig.IsRefSensor(sensorId3)).IsFalse();

			await Assert.That(rig.HasSensorFromRig(sensorId0)).IsFalse();
			await Assert.That(rig.HasSensorFromRig(sensorId1)).IsTrue();
			await Assert.That(rig.HasSensorFromRig(sensorId2)).IsTrue();
			await Assert.That(rig.HasSensorFromRig(sensorId3)).IsFalse(); // no sensor_from_rig

			await Assert.That(rig.SensorFromRig(sensorId1) == sensor1FromRig).IsTrue();
			await Assert.That(rig.MaybeSensorFromRig(sensorId1)!.Value == sensor1FromRig).IsTrue();

			await Assert.That(rig.SensorFromRig(sensorId2) == sensor2FromRig).IsTrue();
			await Assert.That(rig.MaybeSensorFromRig(sensorId2)!.Value == sensor2FromRig).IsTrue();

			await Assert.That(rig.HasSensor(sensorId3)).IsTrue();
			await Assert.That(() => rig.SensorFromRig(sensorId3)).ThrowsException();
			await Assert.That(rig.MaybeSensorFromRig(sensorId3).HasValue).IsFalse();
		}

		rig.SetSensorFromRig(sensorId3, sensor3FromRig);
		using (Assert.Multiple())
		{
			await Assert.That(rig.SensorFromRig(sensorId3) == sensor3FromRig).IsTrue();
			await Assert.That(rig.MaybeSensorFromRig(sensorId3)!.Value == sensor3FromRig).IsTrue();
		}
	}

	[Test]
	public async Task Rig_Print()
	{
		var rig = new Rig { RigId = 0 };
		rig.AddRefSensor(new SensorId(SensorType.Imu, 0));
		rig.AddSensor(new SensorId(SensorType.Camera, 1), new Rigid3d());
		rig.AddSensor(new SensorId(SensorType.Camera, 2), new Rigid3d());
		await Assert.That(rig.ToString()).IsEqualTo("Rig(rig_id=0, ref_sensor_id=(IMU, 0), sensors=[(CAMERA, 1), (CAMERA, 2)])");
	}

	// C#-only (no rig_test.cc counterpart): pins the in-place sensor_from_rig storage that
	// Phase 8's bundle adjustment registers as one 7-value block (Rig.cs header). The Params
	// array is the same object across set, reset and set again, and a write into it is the
	// rig's new transform.
	[Test]
	public async Task SensorFromRigStorage_IsStableAndWritable()
	{
		var rig = new Rig();
		var refSensor = new SensorId(SensorType.Camera, 0);
		var sensor = new SensorId(SensorType.Camera, 1);
		rig.AddRefSensor(refSensor);
		rig.AddSensor(sensor);
		await Assert.That(() => rig.SensorFromRigStorage(sensor)).ThrowsException();
		await Assert.That(() => rig.SensorFromRigStorage(refSensor)).ThrowsException();

		rig.SetSensorFromRig(sensor, new Rigid3d(ColmapSharp.LinearAlgebra.Quaterniond.Identity, new ColmapSharp.LinearAlgebra.Vector3d(1, 2, 3)));
		double[] parameters = rig.SensorFromRigStorage(sensor).Params;
		await Assert.That(parameters.SequenceEqual([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0])).IsTrue();
		parameters[4] = 9;
		await Assert.That(rig.SensorFromRig(sensor).Translation).IsEqualTo(new ColmapSharp.LinearAlgebra.Vector3d(9, 2, 3));
		await Assert.That(rig.NonRefSensors[sensor]!.Value.Translation.X).IsEqualTo(9);

		rig.ResetSensorFromRig(sensor);
		await Assert.That(rig.HasSensorFromRig(sensor)).IsFalse();
		await Assert.That(rig.NonRefSensors[sensor]).IsNull();
		rig.SetSensorFromRig(sensor, new Rigid3d());
		await Assert.That(ReferenceEquals(rig.SensorFromRigStorage(sensor).Params, parameters)).IsTrue();
		await Assert.That(parameters.SequenceEqual([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])).IsTrue();
	}
}
