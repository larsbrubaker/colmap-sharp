// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TriangulationTests: colmap/geometry/triangulation_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/Triangulation.cs. Tier B (SVD / eigen null vectors); the pycolmap
// comparisons are in GeometryTwoViewOracleTests (C#-only).
//
// The poses built from Eigen::Quaterniond(0.2, 0.3, 0.4, qz) use a non-unit quaternion,
// as COLMAP's test does: q * v and ToRotationMatrix are the same polynomial in q, so the
// projections and the projection matrices agree anyway.
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class TriangulationTests
{
	private static readonly Vector3d[] NominalPoints =
	[
		new Vector3d(0, 0.1, 0.1),
		new Vector3d(0, 1, 3),
		new Vector3d(0, 1, 2),
		new Vector3d(0.01, 0.2, 3),
		new Vector3d(-1, 0.1, 1),
		new Vector3d(0.1, 0.1, 0.2),
	];

	private static readonly Vector3d[] BearingPoints =
	[
		new Vector3d(0, 0.1, 0.1),
		new Vector3d(0, 1, 3),
		new Vector3d(-1, 0.1, 1),
		new Vector3d(0.1, 0.1, -0.5), // behind cam1 (negative Z)
		new Vector3d(0.2, -0.3, -2), // behind cam1
	];

	[Test]
	public async Task TriangulatePoint_Nominal()
	{
		var cam1FromWorld = new Rigid3d();
		var failures = new List<string>();
		for (int z = 0; z < 5; ++z)
		{
			double qz = z / 5.0;
			for (int tx = 0; tx < 10; tx += 2)
			{
				var cam2FromWorld = new Rigid3d(new Quaterniond(0.2, 0.3, 0.4, qz), new Vector3d(tx, 2, 3));
				foreach (Vector3d point3D in NominalPoints)
				{
					Vector2d point1 = (cam1FromWorld * point3D).HNormalized();
					Vector2d point2 = (cam2FromWorld * point3D).HNormalized();
					bool ok = Triangulation.TriangulatePoint(
						cam1FromWorld.ToMatrix(), cam2FromWorld.ToMatrix(), point1, point2, out Vector3d triPoint3D);
					if (!ok || !EigenMatrixNear(point3D, triPoint3D, 1e-10))
					{
						failures.Add($"z={z} tx={tx} point={point3D} ok={ok} got={triPoint3D}");
					}
				}
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task TriangulatePoint_ParallelRays()
	{
		bool ok = Triangulation.TriangulatePoint(
			new Rigid3d().ToMatrix(),
			new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)).ToMatrix(),
			new Vector2d(0, 0),
			new Vector2d(0, 0),
			out _);
		await Assert.That(ok).IsFalse();
	}

	[Test]
	public async Task TriangulatePoint_Bearings()
	{
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(new Quaterniond(0.21, 0.31, 0.41, 0.1), new Vector3d(1, 2, 3));
		var failures = new List<string>();
		foreach (Vector3d point3D in BearingPoints)
		{
			Vector3d camRay1 = (cam1FromWorld * point3D).Normalized();
			Vector3d camRay2 = (cam2FromWorld * point3D).Normalized();
			bool ok = Triangulation.TriangulatePoint(
				cam1FromWorld.ToMatrix(), cam2FromWorld.ToMatrix(), camRay1, camRay2, out Vector3d triPoint3D);
			if (!ok || !EigenMatrixNear(point3D, triPoint3D, 1e-9))
			{
				failures.Add($"point={point3D} ok={ok} got={triPoint3D}");
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task TriangulatePoint_BearingsParallelRays()
	{
		bool ok = Triangulation.TriangulatePoint(
			new Rigid3d().ToMatrix(),
			new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)).ToMatrix(),
			new Vector3d(0, 0, 1),
			new Vector3d(0, 0, 1),
			out _);
		await Assert.That(ok).IsFalse();
	}

	[Test]
	public async Task TriangulateMidPoint_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		const int NumTrials = 10;
		var failures = new List<string>();
		for (int i = 0; i < NumTrials; ++i)
		{
			Quaterniond q1 = RandomEigen.RandomEigenQuaterniond();
			Vector3d t1 = RandomEigen.RandomEigenVector3d();
			Quaterniond q2 = RandomEigen.RandomEigenQuaterniond();
			Vector3d t2 = RandomEigen.RandomEigenVector3d();
			var cam1FromWorld = new Rigid3d(q1, t1);
			var cam2FromWorld = new Rigid3d(q2, t2);
			Vector3d point3D = RandomEigen.RandomEigenVector3d();
			Vector3d camRay1 = (cam1FromWorld * point3D).Normalized();
			Vector3d camRay2 = (cam2FromWorld * point3D).Normalized();

			bool ok = Triangulation.TriangulateMidPoint(
				cam2FromWorld * cam1FromWorld.Inverse(), camRay1, camRay2, out Vector3d point3DInCam1);
			if (!ok)
			{
				// ASSERT_TRUE: stop at the first failure.
				failures.Add($"trial {i}: not triangulated");
				break;
			}

			Vector3d point3DInWorld = cam1FromWorld.Inverse() * point3DInCam1;
			if (!EigenMatrixNear(point3D, point3DInWorld, 1e-10))
			{
				failures.Add($"trial {i}: {point3D} vs {point3DInWorld}");
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task TriangulateMidPoint_NonPerfectIntersection()
	{
		var cam1FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(-1, 0, 0));
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));
		var expectedPoint3D = new Vector3d(0, 0, 5);
		bool ok = Triangulation.TriangulateMidPoint(
			cam2FromWorld * cam1FromWorld.Inverse(),
			(cam1FromWorld * (expectedPoint3D + new Vector3d(0, 0.1, 0))).Normalized(),
			(cam2FromWorld * (expectedPoint3D + new Vector3d(0, -0.1, 0))).Normalized(),
			out Vector3d point3DInCam1);
		using (Assert.Multiple())
		{
			await Assert.That(ok).IsTrue();
			await Assert.That(EigenMatrixNear(point3DInCam1, cam1FromWorld * expectedPoint3D, 1e-3)).IsTrue();
		}
	}

	[Test]
	public async Task TriangulateMidPoint_ParallelRays()
	{
		bool ok = Triangulation.TriangulateMidPoint(
			new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)),
			new Vector3d(0, 0, 1),
			new Vector3d(0, 0, 1),
			out _);
		await Assert.That(ok).IsFalse();
	}

	[Test]
	public async Task TriangulateMidPoint_BehindCameras()
	{
		bool ok = Triangulation.TriangulateMidPoint(
			new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)),
			new Vector3d(0.1, 0, 1).Normalized(),
			new Vector3d(-0.1, 0, 1).Normalized(),
			out _);
		await Assert.That(ok).IsFalse();
	}

	[Test]
	public async Task TriangulateMultiViewPoint_Nominal()
	{
		var cam1FromWorld = new Rigid3d();
		var failures = new List<string>();
		for (int z = 0; z < 5; ++z)
		{
			double qz = z / 5.0;
			for (int tx = 0; tx < 10; tx += 2)
			{
				var cam2FromWorld = new Rigid3d(new Quaterniond(0.21, 0.31, 0.41, qz), new Vector3d(tx, 2, 3));
				var cam3FromWorld = new Rigid3d(new Quaterniond(0.2, 0.3, 0.4, qz), new Vector3d(tx, 2.1, 3.1));
				foreach (Vector3d point3D in NominalPoints)
				{
					Matrix3x4d[] camsFromWorld = [cam1FromWorld.ToMatrix(), cam2FromWorld.ToMatrix(), cam3FromWorld.ToMatrix()];
					Vector2d[] points =
					[
						(cam1FromWorld * point3D).HNormalized(),
						(cam2FromWorld * point3D).HNormalized(),
						(cam3FromWorld * point3D).HNormalized(),
					];
					bool ok = Triangulation.TriangulateMultiViewPoint(camsFromWorld, points, out Vector3d triPoint3D);
					if (!ok || !EigenMatrixNear(point3D, triPoint3D, 1e-10))
					{
						failures.Add($"z={z} tx={tx} point={point3D} ok={ok} got={triPoint3D}");
					}
				}
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task TriangulateMultiViewPoint_Bearings()
	{
		// The 3D bearing overload recovers the same points as the 2D overload, and
		// additionally handles back-hemisphere rays (negative Z in the camera frame)
		// that the 2D (u, v, 1) representation cannot encode -- as produced by
		// omnidirectional (e.g. EQUIRECTANGULAR) cameras.
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(new Quaterniond(0.21, 0.31, 0.41, 0.1), new Vector3d(1, 2, 3));
		var cam3FromWorld = new Rigid3d(new Quaterniond(0.2, 0.3, 0.4, 0.05), new Vector3d(2, 2.1, 3.1));
		Matrix3x4d[] camsFromWorld = [cam1FromWorld.ToMatrix(), cam2FromWorld.ToMatrix(), cam3FromWorld.ToMatrix()];

		var failures = new List<string>();
		foreach (Vector3d point3D in BearingPoints)
		{
			Vector3d[] camRays =
			[
				(cam1FromWorld * point3D).Normalized(),
				(cam2FromWorld * point3D).Normalized(),
				(cam3FromWorld * point3D).Normalized(),
			];
			bool ok = Triangulation.TriangulateMultiViewPoint(camsFromWorld, camRays, out Vector3d triPoint3D);
			if (!ok || !EigenMatrixNear(point3D, triPoint3D, 1e-9))
			{
				failures.Add($"point={point3D} ok={ok} got={triPoint3D}");
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task CalculateTriangulationAngle_Nominal()
	{
		var tvec1 = new Vector3d(0, 0, 0);
		var tvec2 = new Vector3d(0, 1, 0);

		double[] parallel = Triangulation.CalculateTriangulationAngles(
			Vector3d.Zero,
			Vector3d.Zero,
			[new Vector3d(0, 0, 0), new Vector3d(50, 0, 0), new Vector3d(0, 50, 0), new Vector3d(0, 0, 50)]);
		double[] orthogonal = Triangulation.CalculateTriangulationAngles(
			Vector3d.Zero, new Vector3d(50, 0, 50), [new Vector3d(50, 0, 0), new Vector3d(0, 0, 50)]);
		double[] opposing = Triangulation.CalculateTriangulationAngles(
			Vector3d.Zero,
			new Vector3d(0, 0, 50),
			[new Vector3d(0, 0, 0), new Vector3d(0, 0, 50), new Vector3d(0, 0, 25), new Vector3d(0, 0, -25), new Vector3d(0, 0, 75)]);

		using (Assert.Multiple())
		{
			await Assert.That(Triangulation.CalculateTriangulationAngle(tvec1, tvec2, new Vector3d(0, 0, 100)))
				.IsEqualTo(0.009999666687).Within(1e-8);
			await Assert.That(Triangulation.CalculateTriangulationAngle(tvec1, tvec2, new Vector3d(0, 0, 50)))
				.IsEqualTo(0.019997333973).Within(1e-8);
			await Assert.That(Triangulation.CalculateTriangulationAngles(tvec1, tvec2, [new Vector3d(0, 0, 100)])[0])
				.IsEqualTo(0.009999666687).Within(1e-8);
			await Assert.That(Triangulation.CalculateTriangulationAngles(tvec1, tvec2, [new Vector3d(0, 0, 50)])[0])
				.IsEqualTo(0.019997333973).Within(1e-8);

			// Parallel rays.
			await Assert.That(parallel.All(a => Math.Abs(a) <= 1e-6)).IsTrue();

			// Orthogonal rays.
			await Assert.That(orthogonal.All(a => Math.Abs(a - Math.PI / 2) <= 1e-6)).IsTrue();

			// Opposing rays.
			await Assert.That(opposing.All(a => Math.Abs(a) <= 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task CalculateAngleBetweenVectors_ParallelVectors() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(1, 0, 0), new Vector3d(2, 0, 0)))
			.IsEqualTo(0.0).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_OppositeVectors() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(1, 0, 0), new Vector3d(-1, 0, 0)))
			.IsEqualTo(Math.PI).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_PerpendicularVectors() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(1, 0, 0), new Vector3d(0, 1, 0)))
			.IsEqualTo(Math.PI / 2).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_PerpendicularVectorsDifferentMagnitudes() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(3, 0, 0), new Vector3d(0, 5, 0)))
			.IsEqualTo(Math.PI / 2).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_ZeroVector()
	{
		var v1 = new Vector3d(1, 0, 0);
		var v2 = new Vector3d(0, 0, 0);
		using (Assert.Multiple())
		{
			await Assert.That(Triangulation.CalculateAngleBetweenVectors(v1, v2)).IsEqualTo(0.0).Within(1e-10);
			await Assert.That(Triangulation.CalculateAngleBetweenVectors(v2, v1)).IsEqualTo(0.0).Within(1e-10);
		}
	}

	[Test]
	public async Task CalculateAngleBetweenVectors_BothZeroVectors() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(Vector3d.Zero, Vector3d.Zero))
			.IsEqualTo(0.0).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_FortyFiveDegrees() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(1, 0, 0), new Vector3d(1, 1, 0)))
			.IsEqualTo(Math.PI / 4).Within(1e-10);

	[Test]
	public async Task CalculateAngleBetweenVectors_IdenticalVectors() =>
		await Assert.That(Triangulation.CalculateAngleBetweenVectors(new Vector3d(1, 2, 3), new Vector3d(1, 2, 3)))
			.IsEqualTo(0.0).Within(1e-10);
}
