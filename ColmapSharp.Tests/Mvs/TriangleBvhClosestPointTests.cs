// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TriangleBvhClosestPointTests: C#-only tests (not ports; COLMAP has no closest-point query).
// They pin ColmapSharp/Mvs/TriangleBvh.ClosestPoint.cs: every Voronoi region of a triangle
// (face, edges, corners) gives the right point, and on a closed cube the distance matches the
// analytic distance to a box surface for points inside and outside, so the tree's pruning never
// loses the nearest triangle.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing.Benchmark;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class TriangleBvhClosestPointTests
{
	[Test]
	public async Task CSharpOnly_ClosestPointInEveryRegionOfATriangle()
	{
		// (0,0,0), (1,0,0), (0,1,0) with id 7.
		var bvh = new TriangleBvh([0, 0, 0, 1, 0, 0, 0, 1, 0], [7]);
		(Vector3d Query, Vector3d Expected)[] cases =
		[
			(new(0.25, 0.25, 2), new(0.25, 0.25, 0)), // face
			(new(-1, -1, 1), new(0, 0, 0)), // corner A
			(new(3, -1, 0), new(1, 0, 0)), // corner B
			(new(-1, 3, 0), new(0, 1, 0)), // corner C
			(new(0.5, -2, 0), new(0.5, 0, 0)), // edge AB
			(new(-2, 0.5, 1), new(0, 0.5, 0)), // edge AC
			(new(1, 1, 0), new(0.5, 0.5, 0)), // edge BC
		];
		foreach ((Vector3d query, Vector3d expected) in cases)
		{
			double distance = bvh.ClosestPoint(query, out Vector3d closest, out int id);
			await Assert.That((closest - expected).Norm).IsLessThan(1e-12);
			await Assert.That(Math.Abs(distance - (query - expected).Norm)).IsLessThan(1e-12);
			await Assert.That(id).IsEqualTo(7);
		}
	}

	[Test]
	public async Task CSharpOnly_EmptyTreeHasNoClosestPoint()
	{
		var bvh = new TriangleBvh([], []);
		double distance = bvh.ClosestPoint(new Vector3d(1, 2, 3), out _, out int id);
		await Assert.That(double.IsPositiveInfinity(distance)).IsTrue();
		await Assert.That(id).IsEqualTo(-1);
	}

	[Test]
	public async Task CSharpOnly_DistanceToACubeMatchesTheAnalyticBoxDistance()
	{
		BenchmarkMesh cube = BenchmarkMeshTestShapes.Cube(halfSide: 0.5);
		TriangleBvh bvh = cube.BuildBvh();
		var random = new Mt19937(3);
		double worst = 0;
		for (int i = 0; i < 2000; i++)
		{
			var p = new Vector3d(Uniform(random), Uniform(random), Uniform(random));
			double expected = BoxSurfaceDistance(p, 0.5);
			double distance = bvh.ClosestPoint(p, out Vector3d closest, out _);
			worst = Math.Max(worst, Math.Abs(distance - expected));
			worst = Math.Max(worst, Math.Abs((closest - p).Norm - distance));
		}

		await Assert.That(worst).IsLessThan(1e-9);
	}

	// Uniform in [-1.5, 1.5).
	private static double Uniform(Mt19937 random) => random.Next() / 4294967296.0 * 3 - 1.5;

	// Distance from p to the surface of the box [-h, h]^3.
	private static double BoxSurfaceDistance(Vector3d p, double h)
	{
		double dx = Math.Abs(p.X) - h, dy = Math.Abs(p.Y) - h, dz = Math.Abs(p.Z) - h;
		if (dx <= 0 && dy <= 0 && dz <= 0)
		{
			return -Math.Max(dx, Math.Max(dy, dz));
		}

		double ox = Math.Max(dx, 0), oy = Math.Max(dy, 0), oz = Math.Max(dz, 0);
		return Math.Sqrt(ox * ox + oy * oy + oz * oz);
	}
}

/// <summary>Small meshes with known geometry for the benchmark metric tests.</summary>
internal static class BenchmarkMeshTestShapes
{
	/// <summary>The closed, outward-wound cube [-h, h]^3 as 12 triangles.</summary>
	public static BenchmarkMesh Cube(double halfSide) => Box(halfSide, halfSide, halfSide);

	/// <summary>The closed, outward-wound box [-hx, hx] x [-hy, hy] x [-hz, hz] as 12 triangles.</summary>
	public static BenchmarkMesh Box(double hx, double hy, double hz)
	{
		Vector3d[] v =
		[
			new(-hx, -hy, -hz), new(hx, -hy, -hz), new(hx, hy, -hz), new(-hx, hy, -hz),
			new(-hx, -hy, hz), new(hx, -hy, hz), new(hx, hy, hz), new(-hx, hy, hz),
		];
		int[] f =
		[
			0, 2, 1, 0, 3, 2, // z = -h
			4, 5, 6, 4, 6, 7, // z = +h
			0, 1, 5, 0, 5, 4, // y = -h
			3, 7, 6, 3, 6, 2, // y = +h
			0, 4, 7, 0, 7, 3, // x = -h
			1, 2, 6, 1, 6, 5, // x = +h
		];
		var faces = new PlyMeshFace[12];
		for (int t = 0; t < 12; t++)
		{
			faces[t] = new PlyMeshFace(f[3 * t], f[3 * t + 1], f[3 * t + 2]);
		}

		return new BenchmarkMesh(v, faces);
	}

	/// <summary>The square [0, side]^2 in the plane z = <paramref name="z"/>, as two triangles.</summary>
	public static BenchmarkMesh Square(double side, double z) =>
		new(
			[new(0, 0, z), new(side, 0, z), new(side, side, z), new(0, side, z)],
			[new PlyMeshFace(0, 1, 2), new PlyMeshFace(0, 2, 3)]);
}
