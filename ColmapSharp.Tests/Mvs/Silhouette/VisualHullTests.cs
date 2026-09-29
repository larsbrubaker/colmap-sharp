// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullTests: C#-only tests (not ports; COLMAP has no visual hull) of
// ColmapSharp/Mvs/Silhouette on analytic silhouettes (VisualHullTestRig.cs):
// - a sphere seen from a ring of 12 views plus 2 from above is contained by its hull, and the
//   hull stays within a stated distance of it;
// - a box seen from 6 axis-aligned, nearly orthographic views is recovered to within a voxel;
// - with k = 1 one bitten mask does not bite the hull, with k = 0 it does;
// - the mesh is closed and consistently wound outward;
// - sequential and parallel carving give identical results.
// The rendered DarkObject check is in VisualHullSceneTests.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.Silhouette;

public class VisualHullTests
{
	private const double Radius = 1;
	private static readonly AlignedBox3d SphereBox = new(new Vector3d(-1.5, -1.5, -1.5), new Vector3d(1.5, 1.5, 1.5));
	private static readonly Lazy<List<VisualHullView>> SphereViews = new(() => VisualHullTestRig.RingViews(4, VisualHullTestRig.Sphere(Radius)));

	private static VisualHull BuildSphere(int tolerance, IReadOnlyList<VisualHullView>? views = null, bool parallel = true) =>
		VisualHull.Build(
			views ?? SphereViews.Value,
			SphereBox,
			new VisualHullOptions { Resolution = 64, DisagreementTolerance = tolerance, Parallel = parallel });

	// Points on the true sphere that are neither inside the hull nor within one voxel of its mesh.
	private static List<Vector3d> Uncontained(VisualHull hull, int samples)
	{
		TriangleBvh bvh = VisualHullTestRig.Bvh(hull.Mesh);
		double voxel = hull.Grid.VoxelSize;
		var missed = new List<Vector3d>();
		for (int s = 0; s < samples; s++)
		{
			// Fibonacci sphere: near-uniform samples.
			double z = 1 - (2 * s + 1.0) / samples;
			double r = Math.Sqrt(1 - z * z);
			double phi = s * Math.PI * (3 - Math.Sqrt(5));
			Vector3d p = new Vector3d(r * Math.Cos(phi), r * Math.Sin(phi), z) * Radius;
			if (hull.Grid.Sample(p) < 0.5 && bvh.ClosestPoint(p, out _, out _) > voxel)
			{
				missed.Add(p);
			}
		}

		return missed;
	}

	[Test]
	public async Task CSharpOnly_SphereHullContainsTheSphereAndStaysClose()
	{
		VisualHull hull = BuildSphere(0);
		double voxel = hull.Grid.VoxelSize;
		await Assert.That(Uncontained(hull, 4000).Count).IsEqualTo(0);

		double maxOut = 0, maxIn = 0;
		for (int v = 0; v < hull.Mesh.Vertices.Count; v++)
		{
			double d = VisualHullTestRig.Position(hull.Mesh, v).Norm - Radius;
			maxOut = Math.Max(maxOut, d);
			maxIn = Math.Max(maxIn, -d);
		}

		Console.WriteLine($"sphere hull: voxel {voxel:F4}, max outside {maxOut:F4}, max inside {maxIn:F4}, {hull.Mesh.Faces.Count} faces");

		// Never more than a voxel inside the sphere (the [1 2 1] smoothing rounds by less than that).
		await Assert.That(maxIn).IsLessThan(voxel);

		// Outside: the ring's 12 perspective cones bound the equator by a 24-gon around the tangent
		// circle, whose corners stand R (1/cos(pi/24) - 1) = 0.0086 R out; away from the ring the
		// two 60-degree views and the ring's top and bottom tangents cut the rest. Add one voxel for
		// sampling the occupancy at voxel centers and the marching-cubes interpolation. Measured:
		// 0.030 at 64 voxels (0.64 voxel).
		await Assert.That(maxOut).IsLessThan((1 / Math.Cos(Math.PI / 24) - 1) * Radius + voxel);
	}

	[Test]
	public async Task CSharpOnly_BoxFromSixAxisViewsIsRecoveredToAVoxel()
	{
		var half = new Vector3d(0.5, 0.3, 0.2);
		Func<Vector3d, Vector3d, bool> box = VisualHullTestRig.Box(half);

		// Telephoto cameras far away: nearly orthographic, so the six cones meet in the box itself.
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 6000, 320, 320);
		var views = new List<VisualHullView>();
		foreach (Vector3d axis in new[] { Vector3d.UnitX, -Vector3d.UnitX, Vector3d.UnitY, -Vector3d.UnitY, Vector3d.UnitZ, -Vector3d.UnitZ })
		{
			Vector3d up = Math.Abs(axis.Z) > 0.5 ? Vector3d.UnitY : Vector3d.UnitZ;
			var pose = VisualHullTestRig.LookAt(axis * 50, Vector3d.Zero, up);
			views.Add(new VisualHullView(camera, pose, VisualHullTestRig.RenderMask(camera, pose, box)));
		}

		var bounds = new AlignedBox3d(new Vector3d(-0.7, -0.7, -0.7), new Vector3d(0.7, 0.7, 0.7));
		VisualHull hull = VisualHull.Build(views, bounds, new VisualHullOptions { Resolution = 64 });
		double voxel = hull.Grid.VoxelSize;

		// Every mesh vertex lies within a voxel of the box surface.
		double worst = 0;
		for (int v = 0; v < hull.Mesh.Vertices.Count; v++)
		{
			Vector3d p = VisualHullTestRig.Position(hull.Mesh, v);
			Vector3d q = p.CwiseAbs() - half;
			double outside = new Vector3d(Math.Max(q.X, 0), Math.Max(q.Y, 0), Math.Max(q.Z, 0)).Norm;
			double inside = Math.Min(Math.Max(q.X, Math.Max(q.Y, q.Z)), 0);
			worst = Math.Max(worst, Math.Abs(outside + inside));
		}

		Console.WriteLine($"box hull: voxel {voxel:F4}, worst vertex distance to the box {worst:F4}");
		await Assert.That(worst).IsLessThan(voxel);

		// And the box's extent is recovered to a voxel on every side.
		double[] min = [double.MaxValue, double.MaxValue, double.MaxValue], max = [double.MinValue, double.MinValue, double.MinValue];
		for (int v = 0; v < hull.Mesh.Vertices.Count; v++)
		{
			Vector3d p = VisualHullTestRig.Position(hull.Mesh, v);
			for (int a = 0; a < 3; a++)
			{
				min[a] = Math.Min(min[a], p[a]);
				max[a] = Math.Max(max[a], p[a]);
			}
		}

		for (int a = 0; a < 3; a++)
		{
			await Assert.That(Math.Abs(max[a] - half[a])).IsLessThan(voxel);
			await Assert.That(Math.Abs(min[a] + half[a])).IsLessThan(voxel);
		}
	}

	[Test]
	public async Task CSharpOnly_ToleranceOfOneViewIgnoresABittenMask()
	{
		// Bite a 40-pixel disc out of the silhouette's right edge in view 0.
		var views = new List<VisualHullView>(SphereViews.Value);
		Bitmap bitten = views[0].Mask.Clone();
		int rightEdge = 0;
		for (int x = 0; x < bitten.Width; x++)
		{
			rightEdge = bitten.RowMajorData[120 * bitten.Width + x] != 0 ? x : rightEdge;
		}

		for (int y = 0; y < bitten.Height; y++)
		{
			for (int x = 0; x < bitten.Width; x++)
			{
				if ((x - rightEdge) * (x - rightEdge) + (y - 120) * (y - 120) < 40 * 40)
				{
					bitten.RowMajorData[y * bitten.Width + x] = 0;
				}
			}
		}

		views[0] = views[0] with { Mask = bitten };
		List<Vector3d> strict = Uncontained(BuildSphere(0, views), 4000);
		List<Vector3d> tolerant = Uncontained(BuildSphere(1, views), 4000);
		Console.WriteLine($"bitten mask: {strict.Count} sphere samples lost with k = 0, {tolerant.Count} with k = 1");
		await Assert.That(strict.Count).IsGreaterThan(20);
		await Assert.That(tolerant.Count).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_HullMeshIsClosedAndWoundOutward()
	{
		foreach (int tolerance in new[] { 0, 1 })
		{
			VisualHull hull = BuildSphere(tolerance);
			(bool closed, bool consistent, double volume) = VisualHullTestRig.Topology(hull.Mesh);
			await Assert.That(hull.Mesh.Faces.Count).IsGreaterThan(1000);
			await Assert.That(closed).IsTrue();
			await Assert.That(consistent).IsTrue();

			// Outward winding gives a positive volume, here about the sphere's (4.19).
			await Assert.That(volume).IsGreaterThan(4 * Math.PI / 3);
			await Assert.That(volume).IsLessThan(2 * 4 * Math.PI / 3);
		}
	}

	[Test]
	public async Task CSharpOnly_SequentialEqualsParallel()
	{
		VisualHull sequential = BuildSphere(1, parallel: false);
		VisualHull parallel = BuildSphere(1, parallel: true);
		await Assert.That(parallel.Grid.Values.SequenceEqual(sequential.Grid.Values)).IsTrue();
		await Assert.That(parallel.Mesh.Vertices.Count).IsEqualTo(sequential.Mesh.Vertices.Count);
		await Assert.That(parallel.Mesh.Faces.Count).IsEqualTo(sequential.Mesh.Faces.Count);
		for (int v = 0; v < sequential.Mesh.Vertices.Count; v++)
		{
			PlyMeshVertex a = sequential.Mesh.Vertices[v], b = parallel.Mesh.Vertices[v];
			await Assert.That(a.X == b.X && a.Y == b.Y && a.Z == b.Z).IsTrue();
		}
	}
}
