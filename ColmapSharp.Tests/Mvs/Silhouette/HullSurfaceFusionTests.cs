// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullSurfaceFusionTests: C#-only (not a port; COLMAP has no visual hull). Pins the two rules of
// Mvs/Silhouette/HullSurfaceFusion.cs on the analytic sphere of VisualHullTestRig: hull samples
// are added only where the fused cloud has no point nearby; pulling a mesh inside the hull and
// removing pieces wholly outside the silhouettes (more than k views) keep a closed mesh closed.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.Silhouette;

public class HullSurfaceFusionTests
{
	private static readonly AlignedBox3d SphereBox = new(new Vector3d(-1.5, -1.5, -1.5), new Vector3d(1.5, 1.5, 1.5));
	private static readonly Lazy<List<VisualHullView>> SphereViews = new(() => VisualHullTestRig.RingViews(4, VisualHullTestRig.Sphere(1)));
	private static readonly Lazy<VisualHull> SphereHull = new(() =>
		VisualHull.Build(SphereViews.Value, SphereBox, new VisualHullOptions { Resolution = 48 }));

	[Test]
	public async Task CSharpOnly_GapSamplesOnlyWhereTheCloudIsMissing()
	{
		VisualHull hull = SphereHull.Value;
		PlyMesh mesh = hull.Mesh;
		double distance = 3 * hull.Grid.VoxelSize;

		// A "fused cloud" covering the upper half of the sphere only (a dense Fibonacci sampling
		// of the true surface, z > 0): the lower half is the gap.
		var cloud = new List<Vector3d>();
		const int n = 20000;
		for (int s = 0; s < n; s++)
		{
			double z = 1 - (2 * s + 1.0) / n;
			double r = Math.Sqrt(1 - z * z);
			double phi = s * Math.PI * (3 - Math.Sqrt(5));
			if (z > 0)
			{
				cloud.Add(new Vector3d(r * Math.Cos(phi), r * Math.Sin(phi), z));
			}
		}

		List<PlyPoint> samples = HullSurfaceFusion.GapSamples(mesh, cloud, distance);
		var tree = new ColmapSharp.Mvs.PointKdTree(cloud);

		// Samples fill the lower half and never sit beside the cloud.
		await Assert.That(samples.Count).IsGreaterThan(mesh.Vertices.Count / 3);
		await Assert.That(samples.All(p => p.Z < 0.2)).IsTrue();
		await Assert.That(samples.All(p => tree.NearestDistance(new Vector3d(p.X, p.Y, p.Z)) > distance)).IsTrue();

		// Every hull vertex well inside the gap is sampled.
		int deepGap = mesh.Vertices.Count(v => v.Z < -0.3);
		await Assert.That(samples.Count(p => p.Z < -0.3)).IsEqualTo(deepGap);

		// Normals are unit and point out of the sphere.
		await Assert.That(samples.All(p => Math.Abs(Math.Sqrt(p.Nx * p.Nx + p.Ny * p.Ny + p.Nz * p.Nz) - 1) < 1e-5)).IsTrue();
		await Assert.That(samples.All(p => p.X * p.Nx + p.Y * p.Ny + p.Z * p.Nz > 0.5)).IsTrue();

		// No cloud: every vertex is a sample. The whole cloud: none is.
		await Assert.That(HullSurfaceFusion.GapSamples(mesh, [], distance).Count).IsEqualTo(mesh.Vertices.Count);
		List<Vector3d> everywhere = [.. mesh.Vertices.Select(v => new Vector3d(v.X, v.Y, v.Z))];
		await Assert.That(HullSurfaceFusion.GapSamples(mesh, everywhere, distance).Count).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_OnlyComponentsWhollyOutsideTheSilhouettesAreRemoved()
	{
		PlyMesh hull = SphereHull.Value.Mesh;

		// Three closed pieces: the hull shrunk to 0.9 (inside every silhouette), a small sphere
		// straddling the silhouette (radius 0.5 at x = 0.8), and the hull grown to 1.6 (outside all).
		var mesh = new PlyMesh();
		Append(mesh, hull, 0.9, 0);
		int innerVertices = mesh.Vertices.Count, innerFaces = mesh.Faces.Count;
		Append(mesh, hull, 0.5, 0.8);
		int keptVertices = mesh.Vertices.Count, keptFaces = mesh.Faces.Count;
		Append(mesh, hull, 1.6, 0);

		PlyMesh trimmed = HullSurfaceFusion.RemoveOutsideComponents(mesh, SphereViews.Value, tolerance: 1);
		await Assert.That(trimmed.Faces.Count).IsEqualTo(keptFaces);
		await Assert.That(trimmed.Vertices.Count).IsEqualTo(keptVertices);
		await Assert.That(HullSurfaceFusion.IsClosed(trimmed)).IsTrue();
		await Assert.That(trimmed.Vertices.SequenceEqual(mesh.Vertices.Take(keptVertices))).IsTrue();

		// One view that sees nothing: outvoted with k = 1; with k = 0 every face is outside, so
		// every piece goes.
		VisualHullView first = SphereViews.Value[0];
		var empty = new Bitmap(first.Mask.Width, first.Mask.Height, asRgb: false);
		List<VisualHullView> withBad = [.. SphereViews.Value, new VisualHullView(first.Camera, first.CamFromWorld, empty)];
		await Assert.That(HullSurfaceFusion.RemoveOutsideComponents(mesh, withBad, tolerance: 1).Faces.Count).IsEqualTo(keptFaces);
		await Assert.That(HullSurfaceFusion.RemoveOutsideComponents(mesh, withBad, tolerance: 0).Faces.Count).IsEqualTo(0);
		await Assert.That(innerFaces).IsLessThan(keptFaces);
		await Assert.That(innerVertices).IsLessThan(keptVertices);
	}

	[Test]
	public async Task CSharpOnly_PullInsideHullKeepsAClosedMeshClosed()
	{
		VisualHull sphere = SphereHull.Value;
		OccupancyGrid surfaceGrid = sphere.Grid.Smoothed(new VisualHullOptions().SmoothingPasses);

		// The hull stretched to 1.15 along x (up to about 2.4 voxels out, like a Poisson surface
		// that overshoots the silhouettes): closed, and partly outside the hull.
		var mesh = new PlyMesh();
		foreach (PlyMeshVertex v in sphere.Mesh.Vertices)
		{
			mesh.Vertices.Add(new PlyMeshVertex(v.X * 1.15f, v.Y * 0.95f, v.Z * 0.95f));
		}

		mesh.Faces.AddRange(sphere.Mesh.Faces);
		await Assert.That(HullSurfaceFusion.IsClosed(mesh)).IsTrue();
		int outside = mesh.Vertices.Count(v => surfaceGrid.Sample(new Vector3d(v.X, v.Y, v.Z)) < 0.5);
		await Assert.That(outside).IsGreaterThan(mesh.Vertices.Count / 10);

		PlyMesh pulled = HullSurfaceFusion.PullInsideHull(mesh, sphere.Mesh, surfaceGrid);
		await Assert.That(pulled.Faces.SequenceEqual(mesh.Faces)).IsTrue();
		await Assert.That(HullSurfaceFusion.IsClosed(pulled)).IsTrue();

		// Inside vertices stay put; outside ones land on the hull surface (within float rounding),
		// except the few left in place because moving them would flip a face.
		double voxel = sphere.Grid.VoxelSize;
		TriangleBvh bvh = VisualHullTestRig.Bvh(sphere.Mesh);
		int moved = 0;
		for (int v = 0; v < mesh.Vertices.Count; v++)
		{
			var before = new Vector3d(mesh.Vertices[v].X, mesh.Vertices[v].Y, mesh.Vertices[v].Z);
			var after = new Vector3d(pulled.Vertices[v].X, pulled.Vertices[v].Y, pulled.Vertices[v].Z);
			if (after != before)
			{
				moved++;
				await Assert.That(surfaceGrid.Sample(before)).IsLessThan(0.5);
				await Assert.That(bvh.ClosestPoint(after, out _, out _)).IsLessThan(1e-4 * voxel + 1e-6);
			}
		}

		Console.WriteLine($"moved {moved} of {outside} outside vertices");
		await Assert.That(moved).IsGreaterThanOrEqualTo(outside * 9 / 10);
	}

	[Test]
	public async Task CSharpOnly_CleanUpRemovesAFloaterBeforePullingTheRestInside()
	{
		VisualHull sphere = SphereHull.Value;
		OccupancyGrid surfaceGrid = sphere.Grid.Smoothed(new VisualHullOptions().SmoothingPasses);

		// The object (the hull grown to 1.1: outside every silhouette, but within 3 voxels of the
		// hull, so it is kept and pulled in) and a separate
		// floater at x = 2.5 that every silhouette rules out. Pulled first, the floater would be
		// flattened onto the hull and survive as a sliver.
		var mesh = new PlyMesh();
		Append(mesh, sphere.Mesh, 1.1, 0);
		int objectVertices = mesh.Vertices.Count, objectFaces = mesh.Faces.Count;
		Append(mesh, sphere.Mesh, 0.3, 2.5);

		PlyMesh cleaned = HullSurfaceFusion.CleanUp(mesh, sphere.Mesh, surfaceGrid, 0.5, SphereViews.Value, tolerance: 1, keepWithin: 3 * sphere.Grid.VoxelSize);
		await Assert.That(cleaned.Faces.Count).IsEqualTo(objectFaces);
		await Assert.That(cleaned.Vertices.Count).IsEqualTo(objectVertices);
		await Assert.That(HullSurfaceFusion.IsClosed(cleaned)).IsTrue();
		await Assert.That(cleaned.Vertices.All(v => v.X < 1.6)).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_PullInsideAConcaveHullFlipsNoFace()
	{
		// A dumbbell: two spheres of radius 0.6 at x = -0.7 and 0.7, with a concave neck between.
		Func<Vector3d, Vector3d, bool> left = VisualHullTestRig.Sphere(0.6), right = VisualHullTestRig.Sphere(0.6);
		var offset = new Vector3d(0.7, 0, 0);
		List<VisualHullView> views = VisualHullTestRig.RingViews(4, (o, d) => left(o + offset, d) || right(o - offset, d));
		VisualHull dumbbell = VisualHull.Build(views, SphereBox, new VisualHullOptions { Resolution = 48 });
		OccupancyGrid surfaceGrid = dumbbell.Grid.Smoothed(new VisualHullOptions().SmoothingPasses);

		// The hull pushed out 2 voxels along its vertex normals, like a Poisson surface that
		// overshoots: closed, outside the hull, and around the neck the nearest hull points of
		// neighbouring vertices lie on opposite sides of the crease, so pulling them all would fold faces.
		PlyMesh hull = dumbbell.Mesh;
		double push = 2 * dumbbell.Grid.VoxelSize;
		var normals = new Vector3d[hull.Vertices.Count];
		foreach (PlyMeshFace f in hull.Faces)
		{
			Vector3d n = Normal(hull, f);
			normals[f.VertexIdx1] += n;
			normals[f.VertexIdx2] += n;
			normals[f.VertexIdx3] += n;
		}

		var mesh = new PlyMesh { Faces = [.. hull.Faces] };
		for (int v = 0; v < hull.Vertices.Count; v++)
		{
			Vector3d p = VisualHullTestRig.Position(hull, v) + normals[v].Normalized() * push;
			mesh.Vertices.Add(new PlyMeshVertex((float)p.X, (float)p.Y, (float)p.Z));
		}

		PlyMesh pulled = HullSurfaceFusion.PullInsideHull(mesh, hull, surfaceGrid);

		await Assert.That(HullSurfaceFusion.IsClosed(pulled)).IsTrue();
		int moved = Enumerable.Range(0, mesh.Vertices.Count).Count(v => !mesh.Vertices[v].Equals(pulled.Vertices[v]));
		Console.WriteLine($"moved {moved} of {mesh.Vertices.Count}");
		await Assert.That(moved).IsGreaterThan(mesh.Vertices.Count * 3 / 4);
		// No face turns over. Faces that collapse to zero area (vertices snapped to one hull point)
		// are the documented limit (PullInsideHull); they are counted to keep them rare.
		int flipped = 0, collapsed = 0;
		foreach (PlyMeshFace f in mesh.Faces)
		{
			Vector3d before = Normal(mesh, f), after = Normal(pulled, f);
			if (before.Dot(after) < 0)
			{
				flipped++;
			}
			else if (after.SquaredNorm == 0)
			{
				collapsed++;
			}
		}

		Console.WriteLine($"collapsed {collapsed} of {mesh.Faces.Count}");
		await Assert.That(flipped).IsEqualTo(0);
		await Assert.That(collapsed).IsLessThan(mesh.Faces.Count / 100);
	}

	private static Vector3d Normal(PlyMesh mesh, PlyMeshFace f)
	{
		Vector3d a = VisualHullTestRig.Position(mesh, f.VertexIdx1), b = VisualHullTestRig.Position(mesh, f.VertexIdx2);
		Vector3d c = VisualHullTestRig.Position(mesh, f.VertexIdx3);
		return (b - a).Cross(c - a);
	}

	private static void Append(PlyMesh target, PlyMesh source, double scale, double shiftX)
	{
		int offset = target.Vertices.Count;
		foreach (PlyMeshVertex v in source.Vertices)
		{
			target.Vertices.Add(new PlyMeshVertex((float)(v.X * scale + shiftX), (float)(v.Y * scale), (float)(v.Z * scale)));
		}

		foreach (PlyMeshFace f in source.Faces)
		{
			target.Faces.Add(new PlyMeshFace(f.VertexIdx1 + offset, f.VertexIdx2 + offset, f.VertexIdx3 + offset));
		}
	}
}
