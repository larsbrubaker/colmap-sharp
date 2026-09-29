// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteHullModel: the hull surface that silhouette pose refinement projects
// (docs/QUALITY_PLAN.md, stage 4b). Not a COLMAP port. It is the visual hull's mesh
// (Mvs/Silhouette/VisualHull) flattened into arrays, so SilhouettePoseRefiner.cs can project and
// rasterize it many times per frame without touching the PLY types.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>A closed triangle mesh (world coordinates) whose projection is the predicted silhouette.</summary>
public sealed class SilhouetteHullModel
{
	/// <summary>A model over <paramref name="vertices"/> and <paramref name="triangles"/> (three indices per triangle).</summary>
	public SilhouetteHullModel(Vector3d[] vertices, int[] triangles)
	{
		Check.That(triangles.Length % 3 == 0);
		Check.That(vertices.Length > 0 && triangles.Length > 0, "The hull is empty");
		Vertices = vertices;
		Triangles = triangles;
		var sum = new Vector3d(0, 0, 0);
		foreach (Vector3d v in vertices)
		{
			sum += v;
		}

		Centroid = sum / vertices.Length;
	}

	/// <summary>The vertices.</summary>
	public Vector3d[] Vertices { get; }

	/// <summary>Three vertex indices per triangle.</summary>
	public int[] Triangles { get; }

	/// <summary>The mean vertex, which sets the refinement's translation scale.</summary>
	public Vector3d Centroid { get; }

	/// <summary>The model of a hull's surface mesh.</summary>
	public static SilhouetteHullModel FromMesh(PlyMesh mesh)
	{
		var vertices = new Vector3d[mesh.Vertices.Count];
		for (int i = 0; i < vertices.Length; i++)
		{
			vertices[i] = new Vector3d(mesh.Vertices[i].X, mesh.Vertices[i].Y, mesh.Vertices[i].Z);
		}

		var triangles = new int[mesh.Faces.Count * 3];
		for (int f = 0; f < mesh.Faces.Count; f++)
		{
			triangles[3 * f] = mesh.Faces[f].VertexIdx1;
			triangles[3 * f + 1] = mesh.Faces[f].VertexIdx2;
			triangles[3 * f + 2] = mesh.Faces[f].VertexIdx3;
		}

		return new SilhouetteHullModel(vertices, triangles);
	}

	/// <summary>The model of <paramref name="hull"/>'s surface.</summary>
	public static SilhouetteHullModel FromHull(VisualHull hull) => FromMesh(hull.Mesh);
}
