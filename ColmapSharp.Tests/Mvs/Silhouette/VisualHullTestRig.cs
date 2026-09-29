// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullTestRig: shared scaffolding for the C#-only visual hull tests (VisualHullTests.cs,
// VisualHullSceneTests.cs). Not a port; COLMAP has no visual hull. It places look-at pinhole
// cameras, renders analytic silhouettes (a pixel is object when the ray through its center hits
// the shape), and checks mesh topology and containment.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Tests.Mvs.Silhouette;

internal static class VisualHullTestRig
{
	/// <summary>The pose of a camera at <paramref name="center"/> looking at <paramref name="target"/>.</summary>
	public static Rigid3d LookAt(Vector3d center, Vector3d target, Vector3d up)
	{
		Vector3d forward = (target - center).Normalized();
		Vector3d right = forward.Cross(up).Normalized();
		Vector3d down = forward.Cross(right);
		Matrix3d rotation = Matrix3d.FromRows(right, down, forward);
		Quaterniond q = Quaterniond.FromRotationMatrix(rotation);
		return new Rigid3d(q, -(q * center));
	}

	/// <summary>The mask of the shape <paramref name="hits"/> (ray origin, unit direction) as seen by the camera.</summary>
	public static Bitmap RenderMask(Camera camera, Rigid3d camFromWorld, Func<Vector3d, Vector3d, bool> hits)
	{
		var mask = new Bitmap(camera.Width, camera.Height, asRgb: false);
		byte[] data = mask.RowMajorData;
		Rigid3d worldFromCam = camFromWorld.Inverse();
		Vector3d origin = worldFromCam.Translation;
		for (int y = 0; y < camera.Height; y++)
		{
			for (int x = 0; x < camera.Width; x++)
			{
				Vector2d cam = camera.CamFromImg(new Vector2d(x + 0.5, y + 0.5))!.Value;
				Vector3d direction = (worldFromCam.Rotation * new Vector3d(cam.X, cam.Y, 1)).Normalized();
				data[y * camera.Width + x] = hits(origin, direction) ? (byte)255 : (byte)0;
			}
		}

		return mask;
	}

	/// <summary>Ray against the sphere of radius <paramref name="radius"/> at the origin.</summary>
	public static Func<Vector3d, Vector3d, bool> Sphere(double radius) => (o, d) =>
	{
		double b = o.Dot(d);
		double c = o.SquaredNorm - radius * radius;
		return b * b - c >= 0 && -b + Math.Sqrt(b * b - c) > 0;
	};

	/// <summary>Ray against the box [-half, half] (slab test).</summary>
	public static Func<Vector3d, Vector3d, bool> Box(Vector3d half) => (o, d) =>
	{
		double near = double.NegativeInfinity, far = double.PositiveInfinity;
		for (int axis = 0; axis < 3; axis++)
		{
			double t0 = (-half[axis] - o[axis]) / d[axis], t1 = (half[axis] - o[axis]) / d[axis];
			near = Math.Max(near, Math.Min(t0, t1));
			far = Math.Min(far, Math.Max(t0, t1));
		}

		return near <= far && far > 0;
	};

	/// <summary>
	/// Twelve views on a horizontal ring of radius <paramref name="distance"/> plus two from 60
	/// degrees above, all looking at the origin, each with the silhouette of <paramref name="hits"/>.
	/// </summary>
	public static List<VisualHullView> RingViews(double distance, Func<Vector3d, Vector3d, bool> hits)
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 300, 320, 240);
		var views = new List<VisualHullView>();
		for (int v = 0; v < 14; v++)
		{
			double azimuth = v < 12 ? v * Math.PI / 6 : (v - 12) * Math.PI + 0.3;
			double elevation = v < 12 ? 0 : Math.PI / 3;
			var center = new Vector3d(
				distance * Math.Cos(elevation) * Math.Cos(azimuth),
				distance * Math.Cos(elevation) * Math.Sin(azimuth),
				distance * Math.Sin(elevation));
			Rigid3d pose = LookAt(center, Vector3d.Zero, Vector3d.UnitZ);
			views.Add(new VisualHullView(camera, pose, RenderMask(camera, pose, hits)));
		}

		return views;
	}

	/// <summary>The mesh's triangles as a BVH (float storage, like the benchmark's).</summary>
	public static TriangleBvh Bvh(PlyMesh mesh) =>
		ColmapSharp.Mvs.Testing.Benchmark.BenchmarkMesh.FromPly(mesh).BuildBvh();

	/// <summary>Positions of a mesh's vertices.</summary>
	public static Vector3d Position(PlyMesh mesh, int index) =>
		new(mesh.Vertices[index].X, mesh.Vertices[index].Y, mesh.Vertices[index].Z);

	/// <summary>
	/// Whether every undirected edge has exactly two faces and every directed edge exactly one
	/// (a closed, consistently wound 2-manifold edge structure), and the signed volume.
	/// </summary>
	public static (bool Closed, bool Consistent, double Volume) Topology(PlyMesh mesh)
	{
		var directed = new Dictionary<(int, int), int>();
		double volume = 0;
		foreach (PlyMeshFace f in mesh.Faces)
		{
			int[] v = [f.VertexIdx1, f.VertexIdx2, f.VertexIdx3];
			for (int e = 0; e < 3; e++)
			{
				(int, int) key = (v[e], v[(e + 1) % 3]);
				directed[key] = directed.GetValueOrDefault(key) + 1;
			}

			Vector3d a = Position(mesh, v[0]), b = Position(mesh, v[1]), c = Position(mesh, v[2]);
			volume += a.Dot(b.Cross(c)) / 6;
		}

		bool consistent = directed.Values.All(count => count == 1);
		bool closed = directed.Keys.All(key => directed.GetValueOrDefault((key.Item2, key.Item1)) == directed[key]);
		return (closed, consistent, volume);
	}
}
