// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkMesh: a triangle mesh in double precision as the reconstruction benchmark scores it
// (docs/QUALITY_PLAN.md, stage 0b). Not a COLMAP port. It carries the truth mesh of a
// SyntheticObjectScene and a reconstructed mesh read from a PLY, maps one into the other's frame
// with a Sim3, and samples points uniformly by area with a seeded mt19937 (so the same mesh and
// seed always give the same samples). Neighbors: SurfaceMetrics.cs measures distances between
// meshes and clouds, SilhouetteMetrics.cs renders meshes into masks.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>An indexed triangle mesh with double-precision vertices.</summary>
public sealed class BenchmarkMesh
{
	/// <summary>Creates a mesh from its vertices and triangles (indices into the vertices).</summary>
	public BenchmarkMesh(IReadOnlyList<Vector3d> vertices, IReadOnlyList<PlyMeshFace> triangles)
	{
		Vertices = [.. vertices];
		Triangles = [.. triangles];
		foreach (PlyMeshFace face in Triangles)
		{
			Check.That(IsIndex(face.VertexIdx1) && IsIndex(face.VertexIdx2) && IsIndex(face.VertexIdx3), "Triangle index out of range");
		}
	}

	/// <summary>The vertices.</summary>
	public Vector3d[] Vertices { get; }

	/// <summary>The triangles, as indices into <see cref="Vertices"/>.</summary>
	public PlyMeshFace[] Triangles { get; }

	/// <summary>The truth mesh of a synthetic scene (in its world frame, the object's).</summary>
	public static BenchmarkMesh FromScene(SyntheticObjectScene scene) => new(scene.MeshVertices, scene.MeshTriangles);

	/// <summary>A mesh read from PLY (e.g. dense/0/meshed-poisson.ply); texture data is ignored.</summary>
	public static BenchmarkMesh FromPly(PlyMesh mesh) =>
		new(mesh.Vertices.Select(v => new Vector3d(v.X, v.Y, v.Z)).ToArray(), mesh.Faces);

	/// <summary>This mesh with every vertex mapped by <paramref name="transform"/>.</summary>
	public BenchmarkMesh Transformed(Sim3d transform) =>
		new(Vertices.Select(v => transform * v).ToArray(), Triangles);

	/// <summary>The axis-aligned bounds (min, max) of the vertices that triangles use.</summary>
	public (Vector3d Min, Vector3d Max) Bounds()
	{
		double minX = double.PositiveInfinity, minY = double.PositiveInfinity, minZ = double.PositiveInfinity;
		double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity, maxZ = double.NegativeInfinity;
		foreach (PlyMeshFace face in Triangles)
		{
			foreach (int i in (ReadOnlySpan<int>)[face.VertexIdx1, face.VertexIdx2, face.VertexIdx3])
			{
				Vector3d v = Vertices[i];
				minX = Math.Min(minX, v.X);
				minY = Math.Min(minY, v.Y);
				minZ = Math.Min(minZ, v.Z);
				maxX = Math.Max(maxX, v.X);
				maxY = Math.Max(maxY, v.Y);
				maxZ = Math.Max(maxZ, v.Z);
			}
		}

		return (new Vector3d(minX, minY, minZ), new Vector3d(maxX, maxY, maxZ));
	}

	/// <summary>The length of the bounding box's diagonal, the benchmark's unit of distance.</summary>
	public double Diagonal()
	{
		(Vector3d min, Vector3d max) = Bounds();
		return Triangles.Length == 0 ? 0 : (max - min).Norm;
	}

	/// <summary>
	/// A bounding volume hierarchy over the triangles (ids are triangle indices), for
	/// point-to-mesh distances. The tree stores float corners, as TriangleBvh does.
	/// </summary>
	public TriangleBvh BuildBvh()
	{
		var corners = new float[9 * Triangles.Length];
		var ids = new int[Triangles.Length];
		for (int t = 0; t < Triangles.Length; t++)
		{
			ids[t] = t;
			PlyMeshFace face = Triangles[t];
			Store(corners, 9 * t, Vertices[face.VertexIdx1]);
			Store(corners, 9 * t + 3, Vertices[face.VertexIdx2]);
			Store(corners, 9 * t + 6, Vertices[face.VertexIdx3]);
		}

		return new TriangleBvh(corners, ids);
	}

	/// <summary>
	/// <paramref name="count"/> points uniformly distributed over the surface by area: a
	/// triangle is drawn with probability proportional to its area, then a point uniformly
	/// inside it (the square-root barycentric map). Deterministic in <paramref name="seed"/>.
	/// A mesh with no area gives no points.
	/// </summary>
	public Vector3d[] SampleSurface(int count, uint seed)
	{
		Check.That(count >= 0);
		var cumulative = new double[Triangles.Length];
		double total = 0;
		for (int t = 0; t < Triangles.Length; t++)
		{
			(Vector3d a, Vector3d b, Vector3d c) = Corners(t);
			total += 0.5 * (b - a).Cross(c - a).Norm;
			cumulative[t] = total;
		}

		if (!(total > 0))
		{
			return [];
		}

		var random = new Mt19937(seed);
		var samples = new Vector3d[count];
		for (int i = 0; i < count; i++)
		{
			double pick = Uniform(random) * total;
			int t = Array.BinarySearch(cumulative, pick);
			t = t < 0 ? ~t : t;
			t = Math.Min(t, Triangles.Length - 1);

			(Vector3d a, Vector3d b, Vector3d c) = Corners(t);
			double r1 = Math.Sqrt(Uniform(random));
			double r2 = Uniform(random);
			samples[i] = (1 - r1) * a + (r1 * (1 - r2)) * b + (r1 * r2) * c;
		}

		return samples;
	}

	private (Vector3d A, Vector3d B, Vector3d C) Corners(int t)
	{
		PlyMeshFace face = Triangles[t];
		return (Vertices[face.VertexIdx1], Vertices[face.VertexIdx2], Vertices[face.VertexIdx3]);
	}

	private bool IsIndex(int i) => i >= 0 && i < Vertices.Length;

	private static void Store(float[] corners, int offset, Vector3d v)
	{
		corners[offset] = (float)v.X;
		corners[offset + 1] = (float)v.Y;
		corners[offset + 2] = (float)v.Z;
	}

	// Uniform in [0, 1) from one mt19937 draw (32 bits are plenty for sampling).
	private static double Uniform(Mt19937 random) => random.Next() / 4294967296.0;
}
