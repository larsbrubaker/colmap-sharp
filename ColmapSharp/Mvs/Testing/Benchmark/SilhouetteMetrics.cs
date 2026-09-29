// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteMetrics: silhouette intersection-over-union of a reconstructed mesh
// (docs/QUALITY_PLAN.md, stage 0b). Not a COLMAP port. The mesh is projected through each
// registered frame's *estimated* camera and pose (so no truth pose is needed) and its covered
// pixels compared with that frame's foreground mask. On synthetic scenes the mask is the true
// one; on real captures it will come from segmentation (stage 1a), which makes this the one
// surface metric a real capture can have.
//
// Rendering: each triangle's corners are projected with Camera.ImgFromCam (so a distortion
// model bends the corners but not the edges between them, which is fine at benchmark mesh
// resolutions), and a pixel is covered when its center (x + 0.5, y + 0.5), COLMAP's pixel
// convention and the one SyntheticObjectScene's masks use, is inside or on a projected
// triangle. A triangle with a corner behind the camera is skipped. No depth test is needed:
// a silhouette is the union of all triangles.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>Mesh silhouettes and their IoU with foreground masks.</summary>
public static class SilhouetteMetrics
{
	/// <summary>
	/// The mask (8-bit grey, 255 covered, 0 not) of <paramref name="mesh"/> seen by
	/// <paramref name="camera"/> at <paramref name="camFromWorld"/>, the camera's size.
	/// </summary>
	public static Bitmap RenderMask(Camera camera, Rigid3d camFromWorld, BenchmarkMesh mesh)
	{
		int width = camera.Width, height = camera.Height;
		var mask = new Bitmap(width, height, asRgb: false);
		byte[] data = mask.RowMajorData;
		var projected = new Vector2d?[mesh.Vertices.Length];
		for (int i = 0; i < projected.Length; i++)
		{
			projected[i] = camera.ImgFromCam(camFromWorld * mesh.Vertices[i]);
		}

		foreach (PlyMeshFace face in mesh.Triangles)
		{
			if (projected[face.VertexIdx1] is not Vector2d a
				|| projected[face.VertexIdx2] is not Vector2d b
				|| projected[face.VertexIdx3] is not Vector2d c)
			{
				continue;
			}

			FillTriangle(data, width, height, a, b, c);
		}

		return mask;
	}

	/// <summary>
	/// |A and B| / |A or B| over pixels, a pixel being foreground where its value is nonzero.
	/// Two empty masks give 1. The masks must be the same size and single channel.
	/// </summary>
	public static double Iou(Bitmap a, Bitmap b)
	{
		Check.That(a.Width == b.Width && a.Height == b.Height, "Mask sizes differ");
		Check.That(a.IsGrey && b.IsGrey, "Masks must be single channel");
		byte[] da = a.RowMajorData, db = b.RowMajorData;
		long intersection = 0, union = 0;
		for (int i = 0; i < da.Length; i++)
		{
			bool fa = da[i] != 0, fb = db[i] != 0;
			intersection += fa && fb ? 1 : 0;
			union += fa || fb ? 1 : 0;
		}

		return union == 0 ? 1.0 : intersection / (double)union;
	}

	// Covers every pixel whose center is inside or on the triangle (edge functions, either
	// winding), within the image.
	private static void FillTriangle(byte[] data, int width, int height, Vector2d a, Vector2d b, Vector2d c)
	{
		double area = Edge(a, b, c);
		if (area == 0 || double.IsNaN(area))
		{
			return;
		}

		// Clamped in double before the cast: a corner far outside the image must not overflow int.
		int x0 = (int)Math.Clamp(Math.Floor(Math.Min(a.X, Math.Min(b.X, c.X)) - 0.5), 0, width);
		int x1 = (int)Math.Clamp(Math.Ceiling(Math.Max(a.X, Math.Max(b.X, c.X)) - 0.5), -1, width - 1);
		int y0 = (int)Math.Clamp(Math.Floor(Math.Min(a.Y, Math.Min(b.Y, c.Y)) - 0.5), 0, height);
		int y1 = (int)Math.Clamp(Math.Ceiling(Math.Max(a.Y, Math.Max(b.Y, c.Y)) - 0.5), -1, height - 1);
		double sign = area > 0 ? 1 : -1;
		for (int y = y0; y <= y1; y++)
		{
			for (int x = x0; x <= x1; x++)
			{
				var p = new Vector2d(x + 0.5, y + 0.5);
				if (sign * Edge(a, b, p) >= 0 && sign * Edge(b, c, p) >= 0 && sign * Edge(c, a, p) >= 0)
				{
					data[y * width + x] = 255;
				}
			}
		}
	}

	private static double Edge(Vector2d a, Vector2d b, Vector2d p) =>
		(b.X - a.X) * (p.Y - a.Y) - (b.Y - a.Y) * (p.X - a.X);
}
