// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayMeshingSceneCSharpOnlyTests (C#-only): end-to-end properties of
// ColmapSharp/Mvs/DelaunayMeshing.cs on a scene with a known answer - points on a unit sphere
// seen by cameras around it, inside a background "room" or on its own (object-only, the
// MatterCAD case). The mesh must be closed and oriented outward (faces are taken from the
// empty, source side), independent of the thread count (entry 110), report progress up to 1
// and stop on cancellation; a scene with no surface must say so readably (entry 112). Tier C: properties of the outcome, not COLMAP numbers.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class DelaunayMeshingSceneCSharpOnlyTests
{
	[Test]
	public async Task CSharpOnly_SphereMeshIsOrientedOutward()
	{
		var mesh = DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 2 }, CreateSphereScene());

		int outward = 0;
		double maxRadiusError = 0;
		int objectFaces = 0;
		foreach (var face in mesh.Faces)
		{
			var a = mesh.Vertices[face.VertexIdx1];
			var b = mesh.Vertices[face.VertexIdx2];
			var c = mesh.Vertices[face.VertexIdx3];
			if (Radius(a) > 1.5 || Radius(b) > 1.5 || Radius(c) > 1.5)
			{
				continue;
			}

			++objectFaces;
			double e1x = b.X - a.X, e1y = b.Y - a.Y, e1z = b.Z - a.Z;
			double e2x = c.X - a.X, e2y = c.Y - a.Y, e2z = c.Z - a.Z;
			double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
			double cx = (a.X + b.X + c.X) / 3, cy = (a.Y + b.Y + c.Y) / 3, cz = (a.Z + b.Z + c.Z) / 3;
			if (nx * cx + ny * cy + nz * cz > 0)
			{
				++outward;
			}
		}

		foreach (var v in mesh.Vertices)
		{
			double radius = Radius(v);
			maxRadiusError = Math.Max(maxRadiusError, Math.Min(Math.Abs(radius - 1), Math.Abs(radius - 8)));
		}

		using (Assert.Multiple())
		{
			// A closed sphere of 1500 points has about 3000 faces.
			await Assert.That(objectFaces).IsGreaterThan(2500);
			await Assert.That(outward).IsEqualTo(objectFaces);
			await Assert.That(maxRadiusError).IsLessThan(1e-5);
		}
	}

	[Test]
	public async Task CSharpOnly_ObjectOnlyNoisySphereGivesAClosedOutwardSurface()
	{
		// MatterCAD's main case: an object photographed from outside, with no background.
		// Noise puts points inside the hull, so viewing rays cross cells there.
		var mesh = DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 2 },
			CreateScene(3000, 0.01, withRoom: false));

		int outward = 0;
		var edgeUses = new Dictionary<(int, int), int>();
		foreach (var face in mesh.Faces)
		{
			var a = mesh.Vertices[face.VertexIdx1];
			var b = mesh.Vertices[face.VertexIdx2];
			var c = mesh.Vertices[face.VertexIdx3];
			if (IsOutward(a, b, c))
			{
				++outward;
			}

			foreach (var (u, v) in new[] { (face.VertexIdx1, face.VertexIdx2), (face.VertexIdx2, face.VertexIdx3), (face.VertexIdx3, face.VertexIdx1) })
			{
				var key = u < v ? (u, v) : (v, u);
				edgeUses[key] = edgeUses.GetValueOrDefault(key) + 1;
			}
		}

		double boundaryRatio = (double)edgeUses.Values.Count(uses => uses == 1) / edgeUses.Count;

		// Sparser, the surface has holes, but grazing rays must not eat into it: giving the
		// infinite cell behind every point a sink vote, entered hull or not, left 1733 faces
		// here; the entry-111 gate keeps 2054, the same as skipping those observations.
		var sparse = DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 2 },
			CreateScene(1500, 0.01, withRoom: false));
		using (Assert.Multiple())
		{
			// A closed surface through 3000 points has about 6000 faces.
			await Assert.That(mesh.Faces.Count).IsGreaterThan(4500);
			await Assert.That((double)outward / mesh.Faces.Count).IsGreaterThan(0.95);
			await Assert.That(boundaryRatio).IsLessThan(0.03);
			await Assert.That(sparse.Faces.Count).IsGreaterThanOrEqualTo(2000);
		}
	}

	[Test]
	public async Task CSharpOnly_NoSurfaceThrowsAReadableError()
	{
		// An exact sphere seen only from outside: every point is a hull vertex, no viewing
		// ray enters the hull, and the cut has no surface (COLMAP fails a Check here).
		var input = CreateScene(1500, 0.0, withRoom: false);
		await Assert.That(() => DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 2 }, input))
			.Throws<InvalidOperationException>()
			.WithMessageContaining("Delaunay meshing found no surface");
	}

	[Test]
	public async Task CSharpOnly_ThreadCountDoesNotChangeTheMesh()
	{
		var input = CreateSphereScene();
		var one = DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 1 }, input);
		var many = DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 5 }, input);
		using (Assert.Multiple())
		{
			await Assert.That(Describe(many)).IsEqualTo(Describe(one));
			await Assert.That(one.Faces.Count).IsGreaterThan(0);
		}
	}

	[Test]
	public async Task CSharpOnly_ReportsProgressAndHonorsCancellation()
	{
		var input = CreateSphereScene();
		var reports = new List<double>();
		DelaunayMeshing.Run(new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 2 }, input,
			new SynchronousProgress(reports.Add));

		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		using (Assert.Multiple())
		{
			await Assert.That(reports.Count).IsEqualTo(input.Images.Count);
			await Assert.That(reports[^1]).IsEqualTo(1.0);
			await Assert.That(reports).IsInOrder();
			await Assert.That(() => DelaunayMeshing.Run(new DelaunayMeshingOptions(), input, null, cancellation.Token))
				.Throws<OperationCanceledException>();
		}
	}

	private static bool IsOutward(PlyMeshVertex a, PlyMeshVertex b, PlyMeshVertex c)
	{
		double e1x = b.X - a.X, e1y = b.Y - a.Y, e1z = b.Z - a.Z;
		double e2x = c.X - a.X, e2y = c.Y - a.Y, e2z = c.Z - a.Z;
		double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
		return nx * (a.X + b.X + c.X) + ny * (a.Y + b.Y + c.Y) + nz * (a.Z + b.Z + c.Z) > 0;
	}

	private static double Radius(PlyMeshVertex v) => Math.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);

	private static string Describe(PlyMesh mesh) =>
		string.Join(";", mesh.Vertices.Select(v => $"{v.X:R},{v.Y:R},{v.Z:R}"))
		+ "|" + string.Join(";", mesh.Faces.Select(f => $"{f.VertexIdx1},{f.VertexIdx2},{f.VertexIdx3}"));

	// 1500 points on a unit sphere inside a "room" of 600 points on a radius-8 sphere, and 10
	// pinhole cameras at distance 4 looking at the center; a point is visible from a camera
	// when it faces it (occlusion is ignored).
	private static DelaunayMeshingInput CreateSphereScene() => CreateScene(1500, 0.0, withRoom: true);

	// numObjectPoints on a unit sphere, radially perturbed by Gaussian noise of the given
	// standard deviation, optionally inside the room.
	private static DelaunayMeshingInput CreateScene(int numObjectPoints, double noise, bool withRoom)
	{
		var input = new DelaunayMeshingInput();
		input.Cameras.Add(1, Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 500, 640, 480));

		// The object: a unit sphere. The room: a sphere of radius 8 around it, facing in. On an
		// exact sphere every object point is a hull vertex, so without the room each camera ray
		// ends outside the hull (just before a hull vertex), crosses no cell and casts no
		// source vote - COLMAP behaves the same. The room tetrahedralizes the free space; the
		// object-only test below uses noise instead, which puts points inside the hull.
		int numRoomPoints = withRoom ? 600 : 0;
		int numPoints = numObjectPoints + numRoomPoints;
		var random = new Random(7);
		var positions = new Vector3d[numPoints];
		var normals = new Vector3d[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			bool room = i >= numObjectPoints;
			int k = room ? i - numObjectPoints : i;
			int count = room ? numRoomPoints : numObjectPoints;
			var unit = FibonacciSphere(k, count);
			double radius = room ? 8 : 1 + noise * Gaussian(random);
			positions[i] = new Vector3d(radius * unit.X, radius * unit.Y, radius * unit.Z);
			normals[i] = room ? new Vector3d(-unit.X, -unit.Y, -unit.Z) : unit;
		}

		var centers = new List<Vector3d>();
		for (int k = 0; k < 8; ++k)
		{
			double angle = 2 * Math.PI * k / 8;
			centers.Add(new Vector3d(4 * Math.Cos(angle), 0.5 * (k % 2 == 0 ? 1 : -1), 4 * Math.Sin(angle)));
		}

		centers.Add(new Vector3d(0.1, 4, 0.2));
		centers.Add(new Vector3d(-0.2, -4, 0.1));

		var visibleCount = new uint[numPoints];
		foreach (var center in centers)
		{
			var image = new DelaunayMeshingInput.InputImage
			{
				CameraId = 1,
				CamFromWorld = LookAtOrigin(center),
				CamInWorld = new Vector3f((float)center.X, (float)center.Y, (float)center.Z),
			};
			for (int i = 0; i < numPoints; ++i)
			{
				var p = positions[i];
				var n = normals[i];
				double facing = n.X * (center.X - p.X) + n.Y * (center.Y - p.Y) + n.Z * (center.Z - p.Z);
				if (facing > 0.1)
				{
					image.PointIdxs.Add(i);
					++visibleCount[i];
				}
			}

			input.Images.Add(image);
		}

		for (int i = 0; i < numPoints; ++i)
		{
			var p = positions[i];
			input.Points.Add(new DelaunayMeshingInput.InputPoint(new Vector3f((float)p.X, (float)p.Y, (float)p.Z), visibleCount[i]));
		}

		return input;
	}

	private static double Gaussian(Random random) =>
		Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());

	private static Vector3d FibonacciSphere(int i, int count)
	{
		double golden = Math.PI * (3 - Math.Sqrt(5));
		double y = 1 - 2 * (i + 0.5) / count;
		double r = Math.Sqrt(1 - y * y);
		return new Vector3d(r * Math.Cos(golden * i), y, r * Math.Sin(golden * i));
	}

	// Row-major cam_from_world for a camera at center whose +z axis points at the origin.
	private static float[] LookAtOrigin(Vector3d center)
	{
		double length = Math.Sqrt(center.X * center.X + center.Y * center.Y + center.Z * center.Z);
		double zx = -center.X / length, zy = -center.Y / length, zz = -center.Z / length;

		// x = up x z with up = (0, 0, 1) when the view is near vertical, else (0, 1, 0).
		double ux = 0, uy = Math.Abs(zy) > 0.9 ? 0 : 1, uz = Math.Abs(zy) > 0.9 ? 1 : 0;
		double xx = uy * zz - uz * zy, xy = uz * zx - ux * zz, xz = ux * zy - uy * zx;
		double xl = Math.Sqrt(xx * xx + xy * xy + xz * xz);
		xx /= xl;
		xy /= xl;
		xz /= xl;
		double yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;

		double tx = -(xx * center.X + xy * center.Y + xz * center.Z);
		double ty = -(yx * center.X + yy * center.Y + yz * center.Z);
		double tz = -(zx * center.X + zy * center.Y + zz * center.Z);
		return
		[
			(float)xx, (float)xy, (float)xz, (float)tx,
			(float)yx, (float)yy, (float)yz, (float)ty,
			(float)zx, (float)zy, (float)zz, (float)tz,
		];
	}

	// Progress<T> posts to the thread pool; this one calls back inline.
	private sealed class SynchronousProgress(Action<double> report) : IProgress<double>
	{
		public void Report(double value) => report(value);
	}
}
