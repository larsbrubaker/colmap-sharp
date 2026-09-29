// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MeshBridge: converts the library's results into what agg-sharp draws - a PlyMesh (with or
// without its texture atlas) into a PolygonMesh.Mesh, and a sparse model's points into a list of
// colored points. The viewport (ModelViewport.cs) draws the results; the session
// (ReconstructionSession.cs) calls these on its worker thread so the UI thread only swaps in a
// finished mesh. Tests: demo/ColmapDemo.Tests/MeshBridgeTests.cs.
//
// agg's Mesh has no per-vertex colors, only per-face ones, so an untextured mesh gets each
// face's color as the mean of its three vertices' colors. A textured mesh gets one
// FaceTextureData per face sharing the one atlas ImageBuffer; the library's UVs already have
// (0, 0) at the atlas's bottom-left (ObjWriter.cs), which is where agg's bottom-up ImageBuffer
// keeps it too, so they pass through unchanged.

using System.Collections.Generic;
using ColmapSharp.Scene;
using ColmapSharp.Util;
using MatterHackers.Agg;
using MatterHackers.Agg.Image;
using MatterHackers.PolygonMesh;
using MatterHackers.VectorMath;

namespace ColmapDemo
{
	/// <summary>A sparse point with its color, ready to draw.</summary>
	public readonly record struct ColoredPoint(Vector3 Position, Color Color);

	/// <summary>Library mesh and point results to agg drawables.</summary>
	public static class MeshBridge
	{
		/// <summary>
		/// Converts <paramref name="plyMesh"/> into an agg mesh. With a non-empty
		/// <paramref name="atlas"/> and one UV pair per face corner in <paramref name="faceUvs"/>
		/// (6 floats per face), every face is textured from the atlas; otherwise each face is
		/// colored with the mean of its vertices' colors.
		/// </summary>
		public static Mesh ToAggMesh(PlyMesh plyMesh, ImageBuffer atlas = null, IReadOnlyList<float> faceUvs = null)
		{
			var vertices = new List<Vector3Float>(plyMesh.Vertices.Count);
			foreach (PlyMeshVertex vertex in plyMesh.Vertices)
			{
				vertices.Add(new Vector3Float(vertex.X, vertex.Y, vertex.Z));
			}

			var faces = new FaceList();
			foreach (PlyMeshFace face in plyMesh.Faces)
			{
				faces.Add(face.VertexIdx1, face.VertexIdx2, face.VertexIdx3, vertices);
			}

			var mesh = new Mesh(vertices, faces);
			bool textured = atlas != null && atlas.Width > 0 && faceUvs != null && faceUvs.Count == plyMesh.Faces.Count * 6;
			if (textured)
			{
				for (int i = 0; i < plyMesh.Faces.Count; i++)
				{
					int uv = i * 6;
					mesh.FaceTextures[i] = new FaceTextureData(
						atlas,
						new Vector2Float(faceUvs[uv], faceUvs[uv + 1]),
						new Vector2Float(faceUvs[uv + 2], faceUvs[uv + 3]),
						new Vector2Float(faceUvs[uv + 4], faceUvs[uv + 5]));
				}
			}
			else
			{
				var colors = new Color[plyMesh.Faces.Count];
				for (int i = 0; i < colors.Length; i++)
				{
					PlyMeshFace face = plyMesh.Faces[i];
					PlyMeshVertex a = plyMesh.Vertices[face.VertexIdx1];
					PlyMeshVertex b = plyMesh.Vertices[face.VertexIdx2];
					PlyMeshVertex c = plyMesh.Vertices[face.VertexIdx3];
					colors[i] = new Color((a.R + b.R + c.R) / 3, (a.G + b.G + c.G) / 3, (a.B + b.B + c.B) / 3);
				}

				mesh.FaceColors = colors;
			}

			return mesh;
		}

		/// <summary>
		/// The 3D points of model <paramref name="modelIndex"/> of <paramref name="models"/>, with
		/// their colors; none when the index is -1 (no model). Only one model: separate models have
		/// unrelated coordinate frames, so drawing them together would overlay unrelated clouds.
		/// </summary>
		public static List<ColoredPoint> SparsePoints(ReconstructionManager models, int modelIndex)
		{
			var points = new List<ColoredPoint>();
			if (modelIndex < 0)
			{
				return points;
			}

			foreach (Point3D point in models.Get(modelIndex).Points3D.Values)
			{
				var xyz = point.Xyz;
				points.Add(new ColoredPoint(
					new Vector3(xyz.X, xyz.Y, xyz.Z),
					new Color(point.Color.X, point.Color.Y, point.Color.Z)));
			}

			return points;
		}
	}
}
