// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteRaster: the binary-image pieces of silhouette pose registration
// (docs/QUALITY_PLAN.md, stage 4b). Not a COLMAP port; COLMAP places images only through
// feature correspondences. SilhouettePoseRefiner.cs uses these per pyramid level:
// - Downsample: a mask at 1/factor of full size (box average, foreground where at least half
//   the block is), the coarse levels that widen the refinement's basin;
// - SignedDistance: the exact Euclidean distance transform of Felzenszwalb and Huttenlocher,
//   "Distance Transforms of Sampled Functions" (Theory of Computing 2012), written from the
//   paper, signed so it is positive outside the foreground and negative inside, with the
//   zero level half a pixel from the boundary pixels on either side;
// - Contour: foreground pixels with a background 4-neighbour, excluding pixels on the image
//   border (a silhouette cut by the image edge is not an object contour there);
// - Fill: triangle rasterization by edge functions at pixel centers (x + 0.5, y + 0.5),
//   COLMAP's pixel convention and the one Mvs/Testing/Benchmark/SilhouetteMetrics uses.
// Everything is sequential and allocation-per-call; the images here are at most full frame.

using ColmapSharp.Sensor;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>A binary image: <c>true</c> is foreground. Row-major, <see cref="Width"/> fastest.</summary>
internal sealed class BinaryImage
{
	public BinaryImage(int width, int height)
	{
		Width = width;
		Height = height;
		Data = new bool[width * height];
	}

	public int Width { get; }

	public int Height { get; }

	public bool[] Data { get; }

	public bool this[int x, int y] => Data[y * Width + x];
}

internal static class SilhouetteRaster
{
	/// <summary>
	/// The mask at 1/<paramref name="factor"/> size: a pixel is foreground when at least half of
	/// the full-size pixels of its block that exist are (value &gt;= 128).
	/// </summary>
	public static BinaryImage Downsample(Bitmap mask, int factor)
	{
		byte[] data = mask.RowMajorData;
		int channels = mask.Channels, fullWidth = mask.Width, fullHeight = mask.Height;
		int width = (fullWidth + factor - 1) / factor, height = (fullHeight + factor - 1) / factor;
		var result = new BinaryImage(width, height);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int count = 0, on = 0;
				for (int yy = y * factor; yy < Math.Min(fullHeight, (y + 1) * factor); yy++)
				{
					for (int xx = x * factor; xx < Math.Min(fullWidth, (x + 1) * factor); xx++)
					{
						count++;
						on += data[(yy * fullWidth + xx) * channels] >= 128 ? 1 : 0;
					}
				}

				result.Data[y * width + x] = 2 * on >= count;
			}
		}

		return result;
	}

	/// <summary>
	/// Signed Euclidean distance to the silhouette boundary, in pixels: for a background pixel its
	/// distance to the nearest foreground pixel minus a half, for a foreground pixel minus (its
	/// distance to the nearest background pixel minus a half). An image with no foreground (or
	/// no background) gives a large constant of the right sign.
	/// </summary>
	public static float[] SignedDistance(BinaryImage image)
	{
		float[] toForeground = SquaredDistanceTo(image, true);
		float[] toBackground = SquaredDistanceTo(image, false);
		var result = new float[image.Data.Length];
		for (int i = 0; i < result.Length; i++)
		{
			result[i] = image.Data[i]
				? -(MathF.Sqrt(toBackground[i]) - 0.5f)
				: MathF.Sqrt(toForeground[i]) - 0.5f;
		}

		return result;
	}

	/// <summary>
	/// The centers (level pixel coordinates) of the contour pixels, row by row: foreground
	/// pixels with a background 4-neighbour, not on the image border.
	/// </summary>
	public static List<(double X, double Y)> Contour(BinaryImage image)
	{
		var points = new List<(double X, double Y)>();
		int w = image.Width, h = image.Height;
		for (int y = 1; y < h - 1; y++)
		{
			for (int x = 1; x < w - 1; x++)
			{
				if (image[x, y] && (!image[x - 1, y] || !image[x + 1, y] || !image[x, y - 1] || !image[x, y + 1]))
				{
					points.Add((x + 0.5, y + 0.5));
				}
			}
		}

		return points;
	}

	/// <summary>
	/// Covers every pixel whose center is inside or on a projected triangle. A triangle with a
	/// corner that did not project (NaN) is skipped: no depth test is needed for a silhouette.
	/// </summary>
	public static BinaryImage Fill(int width, int height, double[] px, double[] py, int[] triangles)
	{
		var image = new BinaryImage(width, height);
		bool[] data = image.Data;
		for (int t = 0; t < triangles.Length; t += 3)
		{
			int a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
			double ax = px[a], ay = py[a], bx = px[b], by = py[b], cx = px[c], cy = py[c];
			if (double.IsNaN(ax) || double.IsNaN(bx) || double.IsNaN(cx))
			{
				continue;
			}

			double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
			if (area == 0)
			{
				continue;
			}

			double sign = area > 0 ? 1 : -1;
			int x0 = Math.Max(0, (int)Math.Floor(Math.Min(ax, Math.Min(bx, cx)) - 0.5));
			int x1 = Math.Min(width - 1, (int)Math.Ceiling(Math.Max(ax, Math.Max(bx, cx)) - 0.5));
			int y0 = Math.Max(0, (int)Math.Floor(Math.Min(ay, Math.Min(by, cy)) - 0.5));
			int y1 = Math.Min(height - 1, (int)Math.Ceiling(Math.Max(ay, Math.Max(by, cy)) - 0.5));
			for (int y = y0; y <= y1; y++)
			{
				double sy = y + 0.5;
				for (int x = x0; x <= x1; x++)
				{
					double sx = x + 0.5;
					double e0 = ((bx - ax) * (sy - ay) - (by - ay) * (sx - ax)) * sign;
					double e1 = ((cx - bx) * (sy - by) - (cy - by) * (sx - bx)) * sign;
					double e2 = ((ax - cx) * (sy - cy) - (ay - cy) * (sx - cx)) * sign;
					if (e0 >= 0 && e1 >= 0 && e2 >= 0)
					{
						data[y * width + x] = true;
					}
				}
			}
		}

		return image;
	}

	/// <summary>|A and B| / |A or B|; two empty images give 1.</summary>
	public static double Iou(BinaryImage a, BinaryImage b)
	{
		long intersection = 0, union = 0;
		for (int i = 0; i < a.Data.Length; i++)
		{
			intersection += a.Data[i] && b.Data[i] ? 1 : 0;
			union += a.Data[i] || b.Data[i] ? 1 : 0;
		}

		return union == 0 ? 1.0 : intersection / (double)union;
	}

	/// <summary>
	/// <paramref name="values"/> (level pixel grid, centers at +0.5) sampled bilinearly at
	/// (x, y). Beyond the image the nearest edge value grows by the distance past the edge, so
	/// the signed distance keeps pointing back toward the image.
	/// </summary>
	public static double SampleBilinear(float[] values, int width, int height, double x, double y)
	{
		double fx = x - 0.5, fy = y - 0.5;
		double cx = Math.Clamp(fx, 0, width - 1), cy = Math.Clamp(fy, 0, height - 1);
		double excess = Math.Sqrt((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy));
		int i = Math.Min((int)cx, width - 2 < 0 ? 0 : width - 2), j = Math.Min((int)cy, height - 2 < 0 ? 0 : height - 2);
		int i1 = Math.Min(i + 1, width - 1), j1 = Math.Min(j + 1, height - 1);
		double tx = cx - i, ty = cy - j;
		double top = values[j * width + i] * (1 - tx) + values[j * width + i1] * tx;
		double bottom = values[j1 * width + i] * (1 - tx) + values[j1 * width + i1] * tx;
		return top * (1 - ty) + bottom * ty + excess;
	}

	// Squared distance from each pixel to the nearest pixel whose value is `target`
	// (Felzenszwalb-Huttenlocher: a 1D lower-envelope pass over columns, then over rows).
	private static float[] SquaredDistanceTo(BinaryImage image, bool target)
	{
		int w = image.Width, h = image.Height;
		const float Infinity = 1e20f;
		var grid = new float[w * h];
		for (int i = 0; i < grid.Length; i++)
		{
			grid[i] = image.Data[i] == target ? 0f : Infinity;
		}

		int n = Math.Max(w, h);
		var f = new float[n];
		var d = new float[n];
		var v = new int[n];
		var z = new float[n + 1];
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				f[y] = grid[y * w + x];
			}

			Transform1D(f, h, d, v, z);
			for (int y = 0; y < h; y++)
			{
				grid[y * w + x] = d[y];
			}
		}

		for (int y = 0; y < h; y++)
		{
			Array.Copy(grid, y * w, f, 0, w);
			Transform1D(f, w, d, v, z);
			Array.Copy(d, 0, grid, y * w, w);
		}

		return grid;
	}

	private static void Transform1D(float[] f, int n, float[] d, int[] v, float[] z)
	{
		int k = 0;
		v[0] = 0;
		z[0] = float.NegativeInfinity;
		z[1] = float.PositiveInfinity;
		for (int q = 1; q < n; q++)
		{
			float s = (f[q] + q * (float)q - (f[v[k]] + v[k] * (float)v[k])) / (2f * q - 2f * v[k]);
			while (s <= z[k])
			{
				k--;
				s = (f[q] + q * (float)q - (f[v[k]] + v[k] * (float)v[k])) / (2f * q - 2f * v[k]);
			}

			k++;
			v[k] = q;
			z[k] = s;
			z[k + 1] = float.PositiveInfinity;
		}

		k = 0;
		for (int q = 0; q < n; q++)
		{
			while (z[k + 1] < q)
			{
				k++;
			}

			float delta = q - v[k];
			d[q] = delta * delta + f[v[k]];
		}
	}
}
