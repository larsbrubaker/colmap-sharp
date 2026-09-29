// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GoodFeaturesToTrack: Shi-Tomasi corner selection for the KLT tracker (docs/QUALITY_PLAN.md
// stage 2a). Not a COLMAP port. Written from the papers only (no code was read):
// - J. Shi and C. Tomasi, "Good Features to Track", CVPR 1994: a window is trackable when the
//   smaller eigenvalue of its gradient structure tensor is large.
// - C. Tomasi and T. Kanade, "Detection and Tracking of Point Features", CMU-CS-91-132, 1991.
//
// Selection: the minimum eigenvalue of the structure tensor summed over a BlockSize window, per
// pixel of an ImagePyramid's level 0; keep 3x3 local maxima at or above QualityLevel times the
// strongest response (inside the mask); then take them strongest first, skipping any closer
// than MinDistance to one already taken or to an existing point. Ties in response are broken
// by pixel index (row-major), so the result is deterministic.
//
// Output coordinates are COLMAP's continuous image coordinates: a corner at pixel (i, j) is
// returned as (i + 0.5, j + 0.5).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Feature.Tracking;

/// <summary>Options for <see cref="GoodFeaturesToTrack.Detect"/>.</summary>
public sealed class GoodFeaturesOptions
{
	/// <summary>Most corners to return.</summary>
	public int MaxCorners { get; set; } = 500;

	/// <summary>Weakest accepted response, as a fraction of the strongest in the (masked) image.</summary>
	public double QualityLevel { get; set; } = 0.01;

	/// <summary>Minimum distance in pixels between two returned corners (and to existing points).</summary>
	public double MinDistance { get; set; } = 8;

	/// <summary>Side of the window the structure tensor is summed over (odd).</summary>
	public int BlockSize { get; set; } = 5;

	/// <summary>Corners closer than this many pixels to the image border are skipped.</summary>
	public int BorderMargin { get; set; } = 4;
}

/// <summary>Shi-Tomasi minimum-eigenvalue corner detection. See the file header.</summary>
public static class GoodFeaturesToTrack
{
	/// <summary>
	/// Detects corners on level 0 of <paramref name="pyramid"/>. <paramref name="mask"/>, if
	/// given, is 8-bit grey of the same size: corners are only placed where it is at least 128
	/// (255 = keep). Corners within MinDistance of any of <paramref name="existing"/> are
	/// skipped, so a tracker can top up its live points. Returns continuous image coordinates,
	/// strongest first.
	/// </summary>
	public static List<Vector2d> Detect(
		ImagePyramid pyramid,
		GoodFeaturesOptions options,
		Bitmap? mask = null,
		IReadOnlyList<Vector2d>? existing = null)
	{
		Check.That(options.BlockSize >= 1 && options.BlockSize % 2 == 1);
		Check.That(options.QualityLevel > 0 && options.QualityLevel <= 1);
		PyramidLevel level = pyramid.Levels[0];
		int w = level.Width, h = level.Height;
		if (mask != null)
		{
			Check.That(mask.IsGrey && mask.Width == w && mask.Height == h);
		}

		float[] response = MinEigenResponse(level, options.BlockSize);
		byte[]? maskData = mask?.RowMajorData;
		int margin = Math.Max(options.BorderMargin, 1);

		float maxResponse = 0;
		for (int y = margin; y < h - margin; ++y)
		{
			for (int x = margin; x < w - margin; ++x)
			{
				int i = y * w + x;
				if ((maskData == null || maskData[i] >= 128) && response[i] > maxResponse)
				{
					maxResponse = response[i];
				}
			}
		}

		var result = new List<Vector2d>();
		if (maxResponse <= 0)
		{
			return result;
		}

		float threshold = (float)(options.QualityLevel * maxResponse);
		var candidates = new List<(float Response, int Index)>();
		for (int y = margin; y < h - margin; ++y)
		{
			for (int x = margin; x < w - margin; ++x)
			{
				int i = y * w + x;
				float r = response[i];
				if (r < threshold || (maskData != null && maskData[i] < 128) || !IsLocalMax(response, w, x, y))
				{
					continue;
				}

				candidates.Add((r, i));
			}
		}

		// Strongest first; equal responses in row-major order.
		candidates.Sort((a, b) => a.Response != b.Response ? b.Response.CompareTo(a.Response) : a.Index.CompareTo(b.Index));

		var grid = new SpatialGrid(w, h, options.MinDistance);
		if (existing != null)
		{
			foreach (Vector2d p in existing)
			{
				grid.Add(p);
			}
		}

		foreach ((_, int index) in candidates)
		{
			if (result.Count >= options.MaxCorners)
			{
				break;
			}

			var p = new Vector2d(index % w + 0.5, index / w + 0.5);
			if (grid.HasNeighbour(p))
			{
				continue;
			}

			grid.Add(p);
			result.Add(p);
		}

		return result;
	}

	/// <summary>
	/// Per-pixel smaller eigenvalue of the structure tensor summed over a
	/// <paramref name="blockSize"/> window (reflected borders), in grey levels squared.
	/// </summary>
	public static float[] MinEigenResponse(PyramidLevel level, int blockSize)
	{
		int w = level.Width, h = level.Height, r = blockSize / 2;
		var xx = new float[w * h];
		var xy = new float[w * h];
		var yy = new float[w * h];
		for (int i = 0; i < xx.Length; ++i)
		{
			float gx = level.GradX[i], gy = level.GradY[i];
			xx[i] = gx * gx;
			xy[i] = gx * gy;
			yy[i] = gy * gy;
		}

		xx = BoxSum(xx, w, h, r);
		xy = BoxSum(xy, w, h, r);
		yy = BoxSum(yy, w, h, r);
		var response = new float[w * h];
		for (int i = 0; i < response.Length; ++i)
		{
			double a = xx[i], b = xy[i], c = yy[i];
			double half = (a - c) * 0.5;
			response[i] = (float)((a + c) * 0.5 - Math.Sqrt(half * half + b * b));
		}

		return response;
	}

	private static bool IsLocalMax(float[] response, int w, int x, int y)
	{
		float v = response[y * w + x];
		for (int dy = -1; dy <= 1; ++dy)
		{
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (response[(y + dy) * w + x + dx] > v)
				{
					return false;
				}
			}
		}

		return true;
	}

	// Separable box sum with reflected borders; sequential sums in a fixed order.
	private static float[] BoxSum(float[] src, int w, int h, int r)
	{
		var tmp = new float[w * h];
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				float s = 0;
				for (int d = -r; d <= r; ++d)
				{
					s += src[y * w + PyramidLevel.Reflect(x + d, w)];
				}

				tmp[y * w + x] = s;
			}
		}

		var dst = new float[w * h];
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				float s = 0;
				for (int d = -r; d <= r; ++d)
				{
					s += tmp[PyramidLevel.Reflect(y + d, h) * w + x];
				}

				dst[y * w + x] = s;
			}
		}

		return dst;
	}

	// Buckets of MinDistance-sized cells, so the distance test looks at 3x3 cells only.
	private sealed class SpatialGrid
	{
		private readonly double minDistance;
		private readonly double cell;
		private readonly int cols;
		private readonly int rows;
		private readonly List<Vector2d>[] cells;

		public SpatialGrid(int width, int height, double minDistance)
		{
			this.minDistance = minDistance;
			cell = Math.Max(minDistance, 1);
			cols = (int)Math.Ceiling(width / cell) + 1;
			rows = (int)Math.Ceiling(height / cell) + 1;
			cells = new List<Vector2d>[cols * rows];
		}

		public void Add(Vector2d p)
		{
			int i = CellOf(p.Y, rows) * cols + CellOf(p.X, cols);
			(cells[i] ??= []).Add(p);
		}

		public bool HasNeighbour(Vector2d p)
		{
			if (minDistance <= 0)
			{
				return false;
			}

			int cx = CellOf(p.X, cols), cy = CellOf(p.Y, rows);
			double minSq = minDistance * minDistance;
			for (int y = Math.Max(cy - 1, 0); y <= Math.Min(cy + 1, rows - 1); ++y)
			{
				for (int x = Math.Max(cx - 1, 0); x <= Math.Min(cx + 1, cols - 1); ++x)
				{
					List<Vector2d>? list = cells[y * cols + x];
					if (list == null)
					{
						continue;
					}

					foreach (Vector2d q in list)
					{
						double dx = q.X - p.X, dy = q.Y - p.Y;
						if (dx * dx + dy * dy < minSq)
						{
							return true;
						}
					}
				}
			}

			return false;
		}

		private int CellOf(double v, int count) => Math.Clamp((int)Math.Floor(v / cell), 0, count - 1);
	}
}
