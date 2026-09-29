// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticObjectScene (continued): renders one frame. Not a COLMAP port. The object mesh is
// drawn by a z-buffer rasterizer over a 3x3 grid of samples per pixel (the sample grid of
// RenderTexturedSphereOnWall: pixel x covers [x, x + 1) and its samples sit at
// x + (s + 0.5) / 3, so the middle one is COLMAP's pixel center x + 0.5). A sample is covered
// when its point lies inside the projected triangle (edges included), and the nearest cover
// wins; depth and surface attributes are interpolated perspective-correctly (by 1 / z), which
// is the same as intersecting the sample's ray with the mesh. So the rendered object is exactly
// MeshVertices / MeshTriangles, and the mask is the middle sample's coverage. Samples the
// object does not cover see the wall, a lab-fixed plane behind the object. The z-buffer holds
// one band of rows at a time, so a 1920x1080 frame needs ~16 MB with its outputs, not ~500.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class SyntheticObjectScene
{
	private const int Supersampling = 3;

	// Pixel rows per z-buffer band (RenderFrame).
	private const int BandRows = 16;

	// Bytes of z-buffer per sample: 1 / z, the two stored weights, and the triangle id.
	private const int ZBufferBytesPerSample = 8 + 8 + 8 + 4;

	// The wall is the lab plane z = WallZ (the camera is at z ~ -2.7, the object within 0.6 of
	// the origin). Its brightness is WallValue plus a faint gradient across the view.
	private const double WallZ = 2.0;
	private const double WallValue = 0.5;

	// The wall's gradient per lab unit along x and y (WallRadiance).
	private const double WallGradientX = 0.016;
	private const double WallGradientY = 0.008;

	// Light direction in the lab (toward the light): from the camera's side, above and to the
	// left, so the underside the camera sees is lit too.
	private static readonly Vector3d LightInLab = new Vector3d(-0.3, 0.3, -1).Normalized();

	internal const double Ambient = 0.25;
	internal const double Diffuse = 0.75;
	private const double SpecularExponent = 40;

	private static (Bitmap Frame, Bitmap Mask) RenderFrame(
		ObjectShape shape,
		Camera camera,
		Rigid3d camFromWorld,
		Rigid3d camFromLab,
		Mt19937 noise,
		double sensorNoiseSigma)
	{
		int width = camera.Width, height = camera.Height;
		int sampleWidth = width * Supersampling;
		double fx = camera.FocalLengthX(), fy = camera.FocalLengthY();
		double cx = camera.PrincipalPointX(), cy = camera.PrincipalPointY();

		var camVertices = new Vector3d[shape.Vertices.Length];
		var screen = new (double X, double Y)[shape.Vertices.Length];
		for (int v = 0; v < camVertices.Length; ++v)
		{
			Vector3d p = camFromWorld * shape.Vertices[v];
			camVertices[v] = p;

			// Sample-grid coordinates: sample i sits at image coordinate (i + 0.5) / 3.
			screen[v] = (
				(fx * p.X / p.Z + cx) * Supersampling - 0.5,
				(fy * p.Y / p.Z + cy) * Supersampling - 0.5);
		}

		// Lighting and the viewer, in the object's (world) frame.
		Rigid3d worldFromCam = camFromWorld.Inverse();
		Rigid3d labFromCam = camFromLab.Inverse();
		Rigid3d worldFromLab = worldFromCam * camFromLab;
		Vector3d light = worldFromLab.Rotation * LightInLab;
		Vector3d eye = worldFromCam.Translation;

		var frame = new Bitmap(width, height, asRgb: true);
		var mask = new Bitmap(width, height, asRgb: false);
		byte[] pixels = frame.RowMajorData;
		byte[] maskPixels = mask.RowMajorData;
		// The z-buffer covers one band of BandRows pixel rows at a time, so a frame needs
		// ~28 bytes per sample of only one band, not of the whole image. Every sample's result
		// depends only on the triangles over it (tested in index order) and noise is drawn in
		// row-major pixel order, so banding does not change a single pixel.
		int bandSamples = BandRows * Supersampling * sampleWidth;
		var inverseDepth = new double[bandSamples];
		var triangleOf = new int[bandSamples];
		var weight1 = new double[bandSamples];
		var weight2 = new double[bandSamples];
		for (int bandStart = 0; bandStart < height; bandStart += BandRows)
		{
			int bandEnd = Math.Min(height, bandStart + BandRows);
			int sampleRowStart = bandStart * Supersampling;
			Array.Clear(inverseDepth);
			Array.Fill(triangleOf, -1);
			for (int f = 0; f < shape.Triangles.Length; ++f)
			{
				PlyMeshFace tri = shape.Triangles[f];
				RasterizeTriangle(
					f, screen[tri.VertexIdx1], screen[tri.VertexIdx2], screen[tri.VertexIdx3],
					camVertices[tri.VertexIdx1].Z, camVertices[tri.VertexIdx2].Z, camVertices[tri.VertexIdx3].Z,
					sampleWidth, sampleRowStart, bandEnd * Supersampling, inverseDepth, triangleOf, weight1, weight2);
			}

			for (int y = bandStart; y < bandEnd; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					double sum = 0;
					for (int sy = 0; sy < Supersampling; ++sy)
					{
						for (int sx = 0; sx < Supersampling; ++sx)
						{
							int sample = (y * Supersampling + sy - sampleRowStart) * sampleWidth + x * Supersampling + sx;
							int f = triangleOf[sample];
							if (f < 0)
							{
								double u = (x + (sx + 0.5) / Supersampling - cx) / fx;
								double v = (y + (sy + 0.5) / Supersampling - cy) / fy;
								sum += WallRadiance(labFromCam, new Vector3d(u, v, 1));
							}
							else
							{
								sum += ShadeObject(shape, f, sample, camVertices, inverseDepth, weight1, weight2, light, eye);
							}
						}
					}

					int center = (y * Supersampling + Supersampling / 2 - sampleRowStart) * sampleWidth
						+ x * Supersampling + Supersampling / 2;
					maskPixels[y * width + x] = triangleOf[center] >= 0 ? (byte)255 : (byte)0;
					double value = sum / (Supersampling * Supersampling) * 255;
					for (int channel = 0; channel < 3; ++channel)
					{
						double noisy = value + sensorNoiseSigma * Gaussian(noise);
						pixels[(y * width + x) * 3 + channel] = (byte)Math.Clamp(Math.Round(noisy), 0, 255);
					}
				}
			}
		}

		return (frame, mask);
	}

	// Writes triangle f into the band's z-buffer (sample rows [sampleRowStart, sampleRowEnd))
	// wherever it covers a sample (edges included) nearer than what is there. a, b, c are
	// sample-grid positions of the whole image; za, zb, zc camera-space depths.
	private static void RasterizeTriangle(
		int f,
		(double X, double Y) a,
		(double X, double Y) b,
		(double X, double Y) c,
		double za,
		double zb,
		double zc,
		int sampleWidth,
		int sampleRowStart,
		int sampleRowEnd,
		double[] inverseDepth,
		int[] triangleOf,
		double[] weight1,
		double[] weight2)
	{
		// The object stays well in front of the camera; this only guards the division.
		if (za <= 0 || zb <= 0 || zc <= 0)
		{
			return;
		}

		double area = (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
		if (area == 0)
		{
			return;
		}

		int minX = Math.Max(0, (int)Math.Ceiling(Math.Min(a.X, Math.Min(b.X, c.X))));
		int maxX = Math.Min(sampleWidth - 1, (int)Math.Floor(Math.Max(a.X, Math.Max(b.X, c.X))));
		int minY = Math.Max(sampleRowStart, (int)Math.Ceiling(Math.Min(a.Y, Math.Min(b.Y, c.Y))));
		int maxY = Math.Min(sampleRowEnd - 1, (int)Math.Floor(Math.Max(a.Y, Math.Max(b.Y, c.Y))));
		for (int sy = minY; sy <= maxY; ++sy)
		{
			for (int sx = minX; sx <= maxX; ++sx)
			{
				// Barycentric weights from the signed sub-areas; dividing by the signed total makes
				// "inside" mean all three are >= 0 for either winding.
				double w1 = ((sx - a.X) * (c.Y - a.Y) - (sy - a.Y) * (c.X - a.X)) / area;
				double w2 = ((b.X - a.X) * (sy - a.Y) - (b.Y - a.Y) * (sx - a.X)) / area;
				double w0 = 1 - w1 - w2;
				if (w0 < 0 || w1 < 0 || w2 < 0)
				{
					continue;
				}

				double invZ = w0 / za + w1 / zb + w2 / zc;
				int sample = (sy - sampleRowStart) * sampleWidth + sx;
				if (invZ > inverseDepth[sample])
				{
					inverseDepth[sample] = invZ;
					triangleOf[sample] = f;
					weight1[sample] = w1;
					weight2[sample] = w2;
				}
			}
		}
	}

	// Ambient plus Lambert plus a Blinn-Phong highlight at the sample's surface point.
	private static double ShadeObject(
		ObjectShape shape,
		int f,
		int sample,
		Vector3d[] camVertices,
		double[] inverseDepth,
		double[] weight1,
		double[] weight2,
		Vector3d light,
		Vector3d eye)
	{
		PlyMeshFace tri = shape.Triangles[f];
		double invZ = inverseDepth[sample];
		double w1 = weight1[sample], w2 = weight2[sample], w0 = 1 - w1 - w2;

		// Perspective-correct weights: screen weights over depth, renormalized by 1 / z.
		double p0 = w0 / camVertices[tri.VertexIdx1].Z / invZ;
		double p1 = w1 / camVertices[tri.VertexIdx2].Z / invZ;
		double p2 = w2 / camVertices[tri.VertexIdx3].Z / invZ;
		Vector3d point = p0 * shape.Vertices[tri.VertexIdx1] + p1 * shape.Vertices[tri.VertexIdx2]
			+ p2 * shape.Vertices[tri.VertexIdx3];
		Vector3d normal = shape.FlatShading
			? shape.FaceNormals[f]
			: (p0 * shape.VertexNormals[tri.VertexIdx1] + p1 * shape.VertexNormals[tri.VertexIdx2]
				+ p2 * shape.VertexNormals[tri.VertexIdx3]).Normalized();

		double lambert = Math.Max(0, normal.Dot(light));
		double shade = shape.Albedo(point) * (Ambient + Diffuse * lambert);
		if (shape.SpecularStrength > 0 && lambert > 0)
		{
			Vector3d halfway = (light + (eye - point).Normalized()).Normalized();
			shade += shape.SpecularStrength * Math.Pow(Math.Max(0, normal.Dot(halfway)), SpecularExponent);
		}

		return shade;
	}

	// The wall's brightness where the camera ray (camera-frame direction) meets it: 0.5 with a
	// faint left-to-right and top-to-bottom gradient. Linear in the hit point, so over an image
	// its extremes lie at the corners (tests bound the wall's darkest value that way).
	internal static double WallRadiance(Rigid3d labFromCam, Vector3d rayInCam)
	{
		Vector3d origin = labFromCam.Translation;
		Vector3d direction = labFromCam.Rotation * rayInCam;
		double t = (WallZ - origin.Z) / direction.Z;
		Vector3d hit = origin + t * direction;
		return WallValue + WallGradientX * hit.X + WallGradientY * hit.Y;
	}
}
