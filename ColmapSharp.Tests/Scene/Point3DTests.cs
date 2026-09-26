// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Point3DTests: colmap/scene/point3d_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/Point3D.cs.
// `stream << p` is p.ToString(); `Point3D other = point3D` is point3D.Clone().

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public class Point3DTests
{
	[Test]
	public async Task Point3D_Default()
	{
		var point3D = new Point3D();
		using (Assert.Multiple())
		{
			await Assert.That(point3D.Xyz == Vector3d.Zero).IsTrue();
			await Assert.That(point3D.Color).IsEqualTo(Vector3ub.Zero);
			await Assert.That(point3D.Error).IsEqualTo(-1.0);
			await Assert.That(point3D.HasError).IsFalse();
			await Assert.That(point3D.Track.Length).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Point3D_Equals()
	{
		var point3D = new Point3D();
		Point3D other = point3D.Clone();
		bool equalAtStart = point3D == other;
		point3D.Xyz = new Vector3d(point3D.Xyz.X + 1, point3D.Xyz.Y, point3D.Xyz.Z);
		bool notEqualAfterChange = point3D != other;
		other.Xyz = new Vector3d(other.Xyz.X + 1, other.Xyz.Y, other.Xyz.Z);
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(point3D == other).IsTrue();
		}
	}

	[Test]
	public async Task Point3D_Print()
	{
		var point3D = new Point3D { Xyz = new Vector3d(1, 2, 3) };
		await Assert.That(point3D.ToString()).IsEqualTo("Point3D(xyz=[1, 2, 3], track_len=0)");
	}

	[Test]
	public async Task Point3D_Error()
	{
		var point3D = new Point3D();
		double initialError = point3D.Error;
		bool initialHas = point3D.HasError;
		point3D.Error = 1.0;
		using (Assert.Multiple())
		{
			await Assert.That(initialError).IsEqualTo(-1.0);
			await Assert.That(initialHas).IsFalse();
			await Assert.That(point3D.Error).IsEqualTo(1.0);
			await Assert.That(point3D.HasError).IsTrue();
		}
	}
}
