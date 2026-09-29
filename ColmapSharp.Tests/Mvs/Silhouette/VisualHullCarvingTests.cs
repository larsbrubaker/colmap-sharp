// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullCarvingTests: C#-only tests (not ports; COLMAP has no visual hull) of the octree
// carving in ColmapSharp/Mvs/Silhouette/VisualHull.cs:
// - the octree gives the same grid as evaluating every voxel center on its own
//   (VisualHull.CarveByPoints), for a pinhole and a radially distorted camera, k = 0 and 1, with
//   one camera inside the box so cells straddle its image plane;
// - both off-image behaviours (VisualHullOptions.OffImageIsOutside): by default a view that
//   crops the object leaves it whole but the box outside the frusta stays filled; counted as
//   outside, the box is carved but the cropped part is lost.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.Silhouette;

public class VisualHullCarvingTests
{
	// A lumpy union of spheres, so the silhouettes are not convex.
	private static readonly (Vector3d Center, double Radius)[] Lumps =
	[
		(new(0, 0, 0), 0.8), (new(0.7, 0.3, 0.1), 0.45), (new(-0.5, -0.6, 0.4), 0.35), (new(0.1, 0.5, -0.7), 0.3),
	];

	private static bool HitsLumps(Vector3d origin, Vector3d direction)
	{
		foreach ((Vector3d center, double radius) in Lumps)
		{
			Vector3d oc = origin - center;
			double b = oc.Dot(direction), c = oc.SquaredNorm - radius * radius;
			if (b * b - c >= 0 && -b + Math.Sqrt(b * b - c) > 0)
			{
				return true;
			}
		}

		return false;
	}

	// Six views around the lumps plus one from inside the box, at 96x72.
	private static List<VisualHullView> Views(Camera camera)
	{
		var centers = new List<Vector3d>();
		for (int v = 0; v < 6; v++)
		{
			double azimuth = v * Math.PI / 3, elevation = v % 2 == 0 ? 0.3 : -0.4;
			centers.Add(new Vector3d(
				4 * Math.Cos(elevation) * Math.Cos(azimuth), 4 * Math.Cos(elevation) * Math.Sin(azimuth), 4 * Math.Sin(elevation)));
		}

		centers.Add(new Vector3d(1.3, -1.2, 0.9));
		var views = new List<VisualHullView>();
		foreach (Vector3d center in centers)
		{
			Rigid3d pose = VisualHullTestRig.LookAt(center, Vector3d.Zero, Vector3d.UnitZ);
			views.Add(new VisualHullView(camera, pose, VisualHullTestRig.RenderMask(camera, pose, HitsLumps)));
		}

		return views;
	}

	[Test]
	[Arguments(0, 0)]
	[Arguments(0, 1)]
	[Arguments(1, 0)]
	[Arguments(1, 1)]
	public async Task CSharpOnly_OctreeEqualsEveryVoxelCenter(int distorted, int tolerance)
	{
		Camera camera = distorted == 0
			? Camera.CreateFromModelId(1, CameraModelId.Pinhole, 90, 96, 72)
			: Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 90, 96, 72);
		if (distorted == 1)
		{
			camera.Params[3] = 0.15;
		}

		List<VisualHullView> views = Views(camera);
		var box = new AlignedBox3d(new Vector3d(-1.5, -1.5, -1.5), new Vector3d(1.5, 1.5, 1.5));
		foreach (bool offImageIsOutside in new[] { false, true })
		{
			var options = new VisualHullOptions { Resolution = 24, DisagreementTolerance = tolerance, OffImageIsOutside = offImageIsOutside };
			OccupancyGrid octree = VisualHull.Carve(views, box, options);
			OccupancyGrid reference = VisualHull.CarveByPoints(views, box, options);
			int differ = 0;
			for (int i = 0; i < octree.Values.Length; i++)
			{
				differ += octree.Values[i] == reference.Values[i] ? 0 : 1;
			}

			await Assert.That(octree.Values.Count(v => v >= 0.5f)).IsGreaterThan(100);
			await Assert.That(differ).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_OffImageBehaviours()
	{
		const double radius = 1;
		Func<Vector3d, Vector3d, bool> sphere = VisualHullTestRig.Sphere(radius);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 300, 320, 240);

		// Four ring views that keep the sphere in frame, in a box much larger than their frusta.
		var views = new List<VisualHullView>();
		for (int v = 0; v < 4; v++)
		{
			double azimuth = v * Math.PI / 2;
			Rigid3d pose = VisualHullTestRig.LookAt(new Vector3d(4 * Math.Cos(azimuth), 4 * Math.Sin(azimuth), 0.5), Vector3d.Zero, Vector3d.UnitZ);
			views.Add(new VisualHullView(camera, pose, VisualHullTestRig.RenderMask(camera, pose, sphere)));
		}

		var box = new AlignedBox3d(new Vector3d(-4, -4, -4), new Vector3d(4, 4, 4));
		int Occupied(IReadOnlyList<VisualHullView> vs, bool offImageIsOutside) =>
			VisualHull.Carve(vs, box, new VisualHullOptions { Resolution = 48, OffImageIsOutside = offImageIsOutside })
				.Values.Count(x => x >= 0.5f);
		int unseen = Occupied(views, false), outside = Occupied(views, true);

		// Voxels of the true sphere, for scale: (4/3) pi / (8/48)^3 = 905.
		Console.WriteLine($"in-frame views: {unseen} voxels with off-image unseen, {outside} with off-image outside");
		await Assert.That(outside).IsLessThan(2000);
		await Assert.That(unseen).IsGreaterThan(5 * outside);

		// Add a close view that crops the sphere: counting off-image as outside cuts the sphere.
		Rigid3d close = VisualHullTestRig.LookAt(new Vector3d(0, -1.6, 0), new Vector3d(0.3, 0, 0), Vector3d.UnitZ);
		var cropped = new List<VisualHullView>(views)
		{
			new(camera, close, VisualHullTestRig.RenderMask(camera, close, sphere)),
		};
		foreach (bool offImageIsOutside in new[] { false, true })
		{
			OccupancyGrid grid = VisualHull.Carve(cropped, box, new VisualHullOptions { Resolution = 48, OffImageIsOutside = offImageIsOutside });
			int lost = 0;
			for (int s = 0; s < 500; s++)
			{
				double z = 1 - (2 * s + 1.0) / 500, r = Math.Sqrt(1 - z * z), phi = s * Math.PI * (3 - Math.Sqrt(5));
				Vector3d p = new Vector3d(r * Math.Cos(phi), r * Math.Sin(phi), z) * (radius - 1.5 * grid.VoxelSize);
				lost += grid.Sample(p) < 0.5 ? 1 : 0;
			}

			Console.WriteLine($"cropping view, off-image {(offImageIsOutside ? "outside" : "unseen")}: {lost} of 500 sphere samples lost");
			if (offImageIsOutside)
			{
				await Assert.That(lost).IsGreaterThan(50);
			}
			else
			{
				await Assert.That(lost).IsEqualTo(0);
			}
		}
	}
}
