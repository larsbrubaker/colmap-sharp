// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticObjectScene (continued): the object meshes and their surface properties (albedo,
// specular strength, flat or smooth shading). Not a COLMAP port. Each mesh is closed and
// outward-oriented in the object's frame (y up, the mouse's nose along +x), and the albedo is a
// function of the 3D point in that frame, so the texture sticks to the surface in every view.
// The rasterizer that draws them is in SyntheticObjectScene.Render.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class SyntheticObjectScene
{
	// Latitude rings and longitude segments of the smooth meshes (about 4000 triangles).
	private const int MeshRings = 32;
	private const int MeshSegments = 64;

	// The mouse's half-extents: along x (nose), along z, and above and below the seam y = 0.
	internal const double MouseHalfLength = 0.6;
	internal const double MouseHalfWidth = 0.33;
	internal const double MouseTopHeight = 0.24;
	internal const double MouseBottomHeight = 0.12;

	// The underside label's albedo spans [LabelMinAlbedo, LabelMinAlbedo + LabelAlbedoRange].
	internal const double LabelMinAlbedo = 0.1;
	internal const double LabelAlbedoRange = 0.15;

	// DarkObject's highlight weight (SpecularStrength).
	internal const double DarkObjectSpecular = 0.08;

	private sealed class ObjectShape
	{
		private readonly SyntheticObjectKind kind;

		private ObjectShape(SyntheticObjectKind kind, Vector3d[] vertices, PlyMeshFace[] triangles, bool flatShading)
		{
			this.kind = kind;
			Vertices = vertices;
			Triangles = triangles;
			FlatShading = flatShading;
			OrientOutward();
			FaceNormals = new Vector3d[triangles.Length];
			VertexNormals = new Vector3d[vertices.Length];
			for (int f = 0; f < triangles.Length; ++f)
			{
				PlyMeshFace tri = triangles[f];
				// Unnormalized, so its length weights the vertex normals by area.
				Vector3d normal = (vertices[tri.VertexIdx2] - vertices[tri.VertexIdx1])
					.Cross(vertices[tri.VertexIdx3] - vertices[tri.VertexIdx1]);
				FaceNormals[f] = normal.Normalized();
				VertexNormals[tri.VertexIdx1] += normal;
				VertexNormals[tri.VertexIdx2] += normal;
				VertexNormals[tri.VertexIdx3] += normal;
			}

			for (int v = 0; v < vertices.Length; ++v)
			{
				VertexNormals[v] = VertexNormals[v].Normalized();
			}
		}

		public Vector3d[] Vertices { get; }

		public PlyMeshFace[] Triangles { get; }

		public Vector3d[] FaceNormals { get; }

		public Vector3d[] VertexNormals { get; }

		// Box faces are flat: interpolating across its corners would round the edges' shading.
		public bool FlatShading { get; }

		// Weight of the Blinn-Phong highlight on top of the diffuse term.
		public double SpecularStrength => kind switch
		{
			SyntheticObjectKind.DarkObject => DarkObjectSpecular,
			SyntheticObjectKind.TexturedSphere => 0.05,
			_ => 0,
		};

		public static ObjectShape Create(SyntheticObjectKind kind) => kind switch
		{
			SyntheticObjectKind.DarkObject => Grid(kind, MouseSurface),
			SyntheticObjectKind.TexturedSphere => Grid(kind, (eta, omega) => 0.5 * new Vector3d(
				Math.Cos(eta) * Math.Cos(omega), Math.Sin(eta), Math.Cos(eta) * Math.Sin(omega))),
			SyntheticObjectKind.TexturelessBox => Box(kind, new Vector3d(0.45, 0.3, 0.35)),
			_ => throw new ArgumentOutOfRangeException(nameof(kind)),
		};

		/// <summary>The surface's reflectance at object-frame point <paramref name="p"/>, in [0, 1].</summary>
		public double Albedo(Vector3d p) => kind switch
		{
			SyntheticObjectKind.DarkObject => MouseAlbedo(p),
			SyntheticObjectKind.TexturedSphere => SolidNoiseAlbedo(p.X, p.Y, p.Z),
			_ => 0.7,
		};

		// A superellipsoid, half-length 0.6 along x, half-width 0.33 along z, 0.24 tall above
		// the seam plane y = 0 and 0.12 below it (a mouse's flatter base). Exponents below 1
		// square it off; both are at most 2, so it stays convex.
		private static Vector3d MouseSurface(double eta, double omega)
		{
			const double Vertical = 0.5, Horizontal = 0.8;
			double ring = SignedPow(Math.Cos(eta), Vertical);
			double up = SignedPow(Math.Sin(eta), Vertical);
			return new Vector3d(
				MouseHalfLength * ring * SignedPow(Math.Cos(omega), Horizontal),
				(up >= 0 ? MouseTopHeight : MouseBottomHeight) * up,
				MouseHalfWidth * ring * SignedPow(Math.Sin(omega), Horizontal));
		}

		private static double SignedPow(double value, double exponent) =>
			Math.Sign(value) * Math.Pow(Math.Abs(value), exponent);

		// Black plastic: 0.06 with +-1.5 grey levels of smooth noise, darker grooves along the
		// seam between shell and base (y = 0), between the two buttons (z = 0, front top) and
		// behind the buttons (x = 0.1, top), and a printed label on the underside.
		private static double MouseAlbedo(Vector3d p)
		{
			const double Base = 0.06;
			if (p.Y < -0.03 && Math.Abs(p.X + 0.1) < 0.2 && Math.Abs(p.Z) < 0.13)
			{
				// The label: a lighter, textured patch, kept below the wall's brightness.
				return LabelMinAlbedo + LabelAlbedoRange * SolidNoiseAlbedo(2 * p.X, 2 * p.Y, 2 * p.Z);
			}

			double albedo = Base + (ValueNoise(6 * p.X, 6 * p.Y, 6 * p.Z) - 0.5) * 2 * (1.5 / 255);
			double groove = Groove(p.Y, 0.012);
			if (p.Y > 0 && p.X > 0.1)
			{
				groove = Math.Max(groove, Groove(p.Z, 0.01));
			}

			if (p.Y > 0.05)
			{
				groove = Math.Max(groove, Groove(p.X - 0.1, 0.01));
			}

			return albedo * (1 - 0.55 * groove);
		}

		// 1 on the seam, falling off smoothly over about halfWidth.
		private static double Groove(double distance, double halfWidth) =>
			Math.Exp(-(distance / halfWidth) * (distance / halfWidth));

		// A closed latitude-longitude mesh: a pole vertex at each end and MeshRings - 1 rings of
		// MeshSegments vertices between them.
		private static ObjectShape Grid(SyntheticObjectKind kind, Func<double, double, Vector3d> surface)
		{
			var vertices = new List<Vector3d> { surface(-Math.PI / 2, 0) };
			for (int i = 1; i < MeshRings; ++i)
			{
				double eta = -Math.PI / 2 + Math.PI * i / MeshRings;
				for (int j = 0; j < MeshSegments; ++j)
				{
					vertices.Add(surface(eta, -Math.PI + 2 * Math.PI * j / MeshSegments));
				}
			}

			vertices.Add(surface(Math.PI / 2, 0));
			int top = vertices.Count - 1;
			int Ring(int i, int j) => 1 + (i - 1) * MeshSegments + (j % MeshSegments);

			var triangles = new List<PlyMeshFace>();
			for (int j = 0; j < MeshSegments; ++j)
			{
				triangles.Add(new PlyMeshFace(0, Ring(1, j), Ring(1, j + 1)));
				triangles.Add(new PlyMeshFace(top, Ring(MeshRings - 1, j + 1), Ring(MeshRings - 1, j)));
				for (int i = 1; i < MeshRings - 1; ++i)
				{
					triangles.Add(new PlyMeshFace(Ring(i, j), Ring(i + 1, j), Ring(i + 1, j + 1)));
					triangles.Add(new PlyMeshFace(Ring(i, j), Ring(i + 1, j + 1), Ring(i, j + 1)));
				}
			}

			return new ObjectShape(kind, [.. vertices], [.. triangles], flatShading: false);
		}

		// An axis-aligned box centered at the origin: 8 shared corners, 2 triangles per face.
		private static ObjectShape Box(SyntheticObjectKind kind, Vector3d half)
		{
			var vertices = new Vector3d[8];
			for (int c = 0; c < 8; ++c)
			{
				vertices[c] = new Vector3d(
					(c & 1) != 0 ? half.X : -half.X, (c & 2) != 0 ? half.Y : -half.Y, (c & 4) != 0 ? half.Z : -half.Z);
			}

			// Corner bits: 1 = +x, 2 = +y, 4 = +z. Each quad lists its corners around the face.
			int[][] quads = [[0, 2, 6, 4], [1, 5, 7, 3], [0, 4, 5, 1], [2, 3, 7, 6], [0, 1, 3, 2], [4, 6, 7, 5]];
			var triangles = new List<PlyMeshFace>();
			foreach (int[] q in quads)
			{
				triangles.Add(new PlyMeshFace(q[0], q[1], q[2]));
				triangles.Add(new PlyMeshFace(q[0], q[2], q[3]));
			}

			return new ObjectShape(kind, vertices, [.. triangles], flatShading: true);
		}

		// Every shape is convex around the origin, so a triangle faces outward exactly when its
		// normal points away from the origin; flip the ones that don't.
		private void OrientOutward()
		{
			for (int f = 0; f < Triangles.Length; ++f)
			{
				PlyMeshFace tri = Triangles[f];
				Vector3d a = Vertices[tri.VertexIdx1], b = Vertices[tri.VertexIdx2], c = Vertices[tri.VertexIdx3];
				if ((b - a).Cross(c - a).Dot(a + b + c) < 0)
				{
					Triangles[f] = new PlyMeshFace(tri.VertexIdx1, tri.VertexIdx3, tri.VertexIdx2);
				}
			}
		}
	}
}
