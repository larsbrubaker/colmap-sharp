// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraDatabaseTests: colmap/sensor/database_test.cc ported 1:1 (TEST(Suite, Name) ->
// Suite_Name). Tests ColmapSharp/Sensor/CameraDatabase.cs and, through it, the generated
// sensor-width table of CameraSpecs.cs (COLMAP has no specs_test.cc).
//
// Tier A: EXPECT_EQ compares the double result with the float literal, as in the C++.

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class CameraDatabaseTests
{
	[Test]
	public async Task CameraDatabase_Initialization()
	{
		var database = new CameraDatabase();
		var specs = CameraSpecs.InitializeCameraSpecs();
		await Assert.That(database.NumEntries).IsEqualTo(specs.Count);
	}

	[Test]
	public async Task CameraDatabase_ExactMatch()
	{
		var database = new CameraDatabase();
		double sensorWidth = 0;
		using (Assert.Multiple())
		{
			await Assert.That(database.QuerySensorWidth("canon", "digitalixus100is", ref sensorWidth)).IsTrue();
			await Assert.That(sensorWidth).IsEqualTo((double)6.1600f);
		}
	}

	[Test]
	public async Task CameraDatabase_AmbiguousMatch()
	{
		var database = new CameraDatabase();
		double sensorWidth = 0;
		using (Assert.Multiple())
		{
			await Assert.That(!database.QuerySensorWidth("canon", "digitalixus", ref sensorWidth)).IsTrue();
			await Assert.That(sensorWidth).IsEqualTo((double)6.1600f);
		}
	}
}
