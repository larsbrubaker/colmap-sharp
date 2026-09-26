// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Point2DTests: colmap/scene/point2d_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/Point2D.cs.
// `stream << p` is p.ToString(); Point2D is a struct, so `Point2D other = point2D` copies.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public class Point2DTests
{
	[Test]
	public async Task Point2D_Default()
	{
		var point2D = new Point2D();
		using (Assert.Multiple())
		{
			await Assert.That(point2D.Xy == Vector2d.Zero).IsTrue();
			await Assert.That(point2D.Point3DId).IsEqualTo(InvalidPoint3DId);
			await Assert.That(point2D.HasPoint3D).IsFalse();
		}
	}

	[Test]
	public async Task Point2D_Equals()
	{
		var point2D = new Point2D();
		Point2D other = point2D;
		bool equalAtStart = point2D == other;
		point2D.Xy = new Vector2d(point2D.Xy.X + 1, point2D.Xy.Y);
		bool notEqualAfterChange = point2D != other;
		other.Xy = new Vector2d(other.Xy.X + 1, other.Xy.Y);
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(point2D == other).IsTrue();
		}
	}

	[Test]
	public async Task Point2D_Print()
	{
		var point2D = new Point2D { Xy = new Vector2d(1, 2) };
		await Assert.That(point2D.ToString()).IsEqualTo("Point2D(xy=[1, 2], point3D_id=-1)");
	}

	[Test]
	public async Task Point2D_Point3DId()
	{
		var point2D = new Point2D();
		ulong initialId = point2D.Point3DId;
		bool initialHas = point2D.HasPoint3D;
		point2D.Point3DId = 1;
		ulong setId = point2D.Point3DId;
		bool setHas = point2D.HasPoint3D;
		point2D.Point3DId = InvalidPoint3DId;
		using (Assert.Multiple())
		{
			await Assert.That(initialId).IsEqualTo(InvalidPoint3DId);
			await Assert.That(initialHas).IsFalse();
			await Assert.That(setId).IsEqualTo(1UL);
			await Assert.That(setHas).IsTrue();
			await Assert.That(point2D.Point3DId).IsEqualTo(InvalidPoint3DId);
			await Assert.That(point2D.HasPoint3D).IsFalse();
		}
	}
}
