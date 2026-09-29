// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BinaryMorphology: the binary-mask operations of the segmentation pipeline - exact Euclidean
// distance transform, disk erosion/dilation/opening, largest connected component, morphological
// reconstruction, hole filling and the Otsu threshold. Not a COLMAP port. Sources (papers only, no code read):
// - P. Felzenszwalb and D. Huttenlocher, "Distance Transforms of Sampled Functions", Theory of
//   Computing 8, 2012: the squared Euclidean distance transform as two passes of the 1D lower
//   envelope of parabolas. Erosion and dilation by a disk of radius r are then thresholds of
//   that distance, so the structuring element's size costs nothing.
// - A. M. Andrew, "Another Efficient Algorithm for Convex Hulls in Two Dimensions", Inf. Proc.
//   Letters 9(5), 1979: the monotone-chain hull behind ConvexHull.
// - N. Otsu, "A Threshold Selection Method from Gray-Level Histograms", IEEE SMC 9(1), 1979.
// Masks are row-major bool arrays on SilhouetteSegmenter's working grid (LabImage).

namespace ColmapSharp.Segmentation;

internal static class BinaryMorphology
{
	private const double Infinity = 1e20;

	/// <summary>
	/// Squared Euclidean distance from every pixel to the nearest pixel where
	/// <paramref name="target"/> is true (1e20 everywhere when there is none).
	/// </summary>
	public static double[] SquaredDistanceTo(bool[] target, int width, int height)
	{
		var result = new double[width * height];
		for (int i = 0; i < result.Length; ++i)
		{
			result[i] = target[i] ? 0.0 : Infinity;
		}

		int n = Math.Max(width, height);
		var f = new double[n];
		var d = new double[n];
		var v = new int[n];
		var z = new double[n + 1];
		for (int x = 0; x < width; ++x)
		{
			for (int y = 0; y < height; ++y)
			{
				f[y] = result[y * width + x];
			}

			Transform1D(f, height, d, v, z);
			for (int y = 0; y < height; ++y)
			{
				result[y * width + x] = d[y];
			}
		}

		for (int y = 0; y < height; ++y)
		{
			Array.Copy(result, y * width, f, 0, width);
			Transform1D(f, width, d, v, z);
			Array.Copy(d, 0, result, y * width, width);
		}

		return result;
	}

	/// <summary>Erosion by a disk: pixels whose every neighbour within <paramref name="radius"/> is set.</summary>
	/// <remarks>Outside the image counts as neither set nor unset, so the border does not erode.</remarks>
	public static bool[] Erode(bool[] mask, int width, int height, double radius)
	{
		double[] toBackground = SquaredDistanceTo(Not(mask), width, height);
		double r2 = radius * radius;
		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = toBackground[i] > r2;
		}

		return result;
	}

	/// <summary>Dilation by a disk: pixels within <paramref name="radius"/> of a set pixel.</summary>
	public static bool[] Dilate(bool[] mask, int width, int height, double radius)
	{
		double[] toForeground = SquaredDistanceTo(mask, width, height);
		double r2 = radius * radius;
		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = toForeground[i] <= r2;
		}

		return result;
	}

	/// <summary>Opening (erosion, then dilation) by a disk: removes parts thinner than about 2r.</summary>
	public static bool[] Open(bool[] mask, int width, int height, double radius) =>
		Dilate(Erode(mask, width, height, radius), width, height, radius);

	/// <summary>
	/// Closing (dilation, then erosion) by a disk: seals channels and notches narrower than
	/// about 2r. The mask is padded with unset pixels by more than r first, so the erosion sees
	/// the unset outside and an object touching the frame edge keeps its border pixels instead
	/// of gaining (or losing) strips there.
	/// </summary>
	public static bool[] Close(bool[] mask, int width, int height, double radius)
	{
		int pad = (int)Math.Ceiling(radius) + 1;
		int pw = width + 2 * pad, ph = height + 2 * pad;
		var padded = new bool[pw * ph];
		for (int y = 0; y < height; ++y)
		{
			Array.Copy(mask, y * width, padded, (y + pad) * pw + pad, width);
		}

		bool[] closed = Erode(Dilate(padded, pw, ph, radius), pw, ph, radius);
		var result = new bool[mask.Length];
		for (int y = 0; y < height; ++y)
		{
			Array.Copy(closed, (y + pad) * pw + pad, result, y * width, width);
		}

		return result;
	}

	/// <summary>
	/// The pixels inside the convex hull of the set pixels' centres (monotone chain, A. M.
	/// Andrew, Inf. Proc. Letters 9(5), 1979), filled row by row.
	/// </summary>
	public static bool[] ConvexHull(bool[] mask, int width, int height)
	{
		// Only the leftmost and rightmost pixel of each row can be hull vertices.
		var points = new List<(long X, long Y)>();
		for (int y = 0; y < height; ++y)
		{
			int left = -1, right = -1;
			for (int x = 0; x < width; ++x)
			{
				if (mask[y * width + x])
				{
					left = left < 0 ? x : left;
					right = x;
				}
			}

			if (left >= 0)
			{
				points.Add((left, y));
				if (right != left)
				{
					points.Add((right, y));
				}
			}
		}

		var result = new bool[mask.Length];
		if (points.Count == 0)
		{
			return result;
		}

		points.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));
		var hull = new List<(long X, long Y)>();
		for (int pass = 0; pass < 2; ++pass)
		{
			int start = hull.Count;
			for (int i = 0; i < points.Count; ++i)
			{
				(long X, long Y) p = pass == 0 ? points[i] : points[points.Count - 1 - i];
				while (hull.Count >= start + 2 && Cross(hull[^2], hull[^1], p) <= 0)
				{
					hull.RemoveAt(hull.Count - 1);
				}

				hull.Add(p);
			}

			hull.RemoveAt(hull.Count - 1);
		}

		if (hull.Count == 0)
		{
			hull.Add(points[0]);
		}

		// Each row's span is the extent of the hull edges (and vertices) that cross it.
		for (int y = 0; y < height; ++y)
		{
			double min = double.PositiveInfinity, max = double.NegativeInfinity;
			for (int i = 0; i < hull.Count; ++i)
			{
				(long X, long Y) a = hull[i];
				(long X, long Y) b = hull[(i + 1) % hull.Count];
				if (y < Math.Min(a.Y, b.Y) || y > Math.Max(a.Y, b.Y))
				{
					continue;
				}

				if (a.Y == b.Y)
				{
					min = Math.Min(min, Math.Min(a.X, b.X));
					max = Math.Max(max, Math.Max(a.X, b.X));
				}
				else
				{
					double x = a.X + (double)(y - a.Y) * (b.X - a.X) / (b.Y - a.Y);
					min = Math.Min(min, x);
					max = Math.Max(max, x);
				}
			}

			if (min > max)
			{
				continue;
			}

			int from = Math.Max(0, (int)Math.Ceiling(min - 1e-9));
			int to = Math.Min(width - 1, (int)Math.Floor(max + 1e-9));
			for (int x = from; x <= to; ++x)
			{
				result[y * width + x] = true;
			}
		}

		return result;
	}

	/// <summary>The radius of the largest disk that fits inside the mask (0 for an empty mask).</summary>
	public static double MaxInscribedRadius(bool[] mask, int width, int height)
	{
		bool[] background = Not(mask);
		if (Array.IndexOf(background, true) < 0)
		{
			return Math.Min(width, height) / 2.0;
		}

		double[] toBackground = SquaredDistanceTo(background, width, height);
		double max = 0.0;
		for (int i = 0; i < mask.Length; ++i)
		{
			if (mask[i] && toBackground[i] > max)
			{
				max = toBackground[i];
			}
		}

		return Math.Sqrt(max);
	}

	/// <summary>
	/// The largest 8-connected component of the mask. Ties go to the component whose first
	/// pixel in row-major order comes first.
	/// </summary>
	public static bool[] LargestComponent(bool[] mask, int width, int height)
	{
		var label = new int[mask.Length];
		var stack = new Stack<int>();
		int bestLabel = 0, bestSize = 0, nextLabel = 0;
		for (int seed = 0; seed < mask.Length; ++seed)
		{
			if (!mask[seed] || label[seed] != 0)
			{
				continue;
			}

			++nextLabel;
			int size = Flood(mask, true, label, nextLabel, seed, width, height, eightConnected: true, stack);
			if (size > bestSize)
			{
				bestSize = size;
				bestLabel = nextLabel;
			}
		}

		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = bestLabel != 0 && label[i] == bestLabel;
		}

		return result;
	}

	/// <summary>
	/// Fills the holes of the mask: unset pixels that no 4-connected path of unset pixels joins
	/// to the image border (4-connected background is the dual of 8-connected foreground).
	/// </summary>
	public static bool[] FillHoles(bool[] mask, int width, int height)
	{
		var label = new int[mask.Length];
		var stack = new Stack<int>();
		for (int x = 0; x < width; ++x)
		{
			SeedBackground(mask, label, x, width, height, stack);
			SeedBackground(mask, label, (height - 1) * width + x, width, height, stack);
		}

		for (int y = 0; y < height; ++y)
		{
			SeedBackground(mask, label, y * width, width, height, stack);
			SeedBackground(mask, label, y * width + width - 1, width, height, stack);
		}

		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = mask[i] || label[i] == 0;
		}

		return result;
	}

	/// <summary>
	/// Otsu's threshold of <paramref name="values"/> over a 256-bin histogram of
	/// [<paramref name="min"/>, <paramref name="max"/>]. Values above the returned threshold are
	/// the upper class. Ties in between-class variance go to the lowest bin.
	/// </summary>
	public static double OtsuThreshold(float[] values, double min, double max)
	{
		const int Bins = 256;
		var histogram = new long[Bins];
		double scale = Bins / (max - min);
		foreach (float value in values)
		{
			int bin = (int)((value - min) * scale);
			histogram[Math.Clamp(bin, 0, Bins - 1)]++;
		}

		long total = values.Length;
		double sumAll = 0.0;
		for (int i = 0; i < Bins; ++i)
		{
			sumAll += i * (double)histogram[i];
		}

		double sumLow = 0.0;
		long countLow = 0;
		double bestVariance = -1.0;
		int bestBin = 0;
		for (int t = 0; t < Bins - 1; ++t)
		{
			countLow += histogram[t];
			sumLow += t * (double)histogram[t];
			long countHigh = total - countLow;
			if (countLow == 0 || countHigh == 0)
			{
				continue;
			}

			double meanLow = sumLow / countLow;
			double meanHigh = (sumAll - sumLow) / countHigh;
			double diff = meanLow - meanHigh;
			double variance = (double)countLow * countHigh * diff * diff;
			if (variance > bestVariance)
			{
				bestVariance = variance;
				bestBin = t;
			}
		}

		// Values in bins 0..bestBin are the lower class.
		return min + (bestBin + 1) / scale;
	}

	/// <summary>The fraction of the image's border pixels that are set in the mask.</summary>
	public static double BorderFraction(bool[] mask, int width, int height)
	{
		int set = 0, count = 0;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
				{
					++count;
					if (mask[y * width + x])
					{
						++set;
					}
				}
			}
		}

		return count == 0 ? 0.0 : (double)set / count;
	}

	/// <summary>
	/// Morphological reconstruction: the 8-connected components of <paramref name="mask"/> that
	/// hold at least one pixel of <paramref name="marker"/>.
	/// </summary>
	public static bool[] Reconstruct(bool[] mask, bool[] marker, int width, int height)
	{
		var label = new int[mask.Length];
		var stack = new Stack<int>();
		for (int seed = 0; seed < mask.Length; ++seed)
		{
			if (mask[seed] && marker[seed] && label[seed] == 0)
			{
				Flood(mask, true, label, 1, seed, width, height, eightConnected: true, stack);
			}
		}

		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = label[i] != 0;
		}

		return result;
	}

	/// <summary>The pixels of the mask that lie on the image border.</summary>
	public static bool[] BorderPixels(bool[] mask, int width, int height)
	{
		var result = new bool[mask.Length];
		for (int p = 0; p < mask.Length; ++p)
		{
			int x = p % width, y = p / width;
			result[p] = mask[p] && (x == 0 || y == 0 || x == width - 1 || y == height - 1);
		}

		return result;
	}

	public static bool[] Not(bool[] mask)
	{
		var result = new bool[mask.Length];
		for (int i = 0; i < mask.Length; ++i)
		{
			result[i] = !mask[i];
		}

		return result;
	}

	private static void SeedBackground(bool[] mask, int[] label, int seed, int width, int height, Stack<int> stack)
	{
		if (!mask[seed] && label[seed] == 0)
		{
			Flood(mask, false, label, 1, seed, width, height, eightConnected: false, stack);
		}
	}

	// Labels the component of pixels equal to `value` that contains `seed` and returns its size.
	private static int Flood(bool[] mask, bool value, int[] label, int id, int seed, int width, int height, bool eightConnected, Stack<int> stack)
	{
		int size = 0;
		label[seed] = id;
		stack.Push(seed);
		while (stack.Count > 0)
		{
			int p = stack.Pop();
			++size;
			int px = p % width;
			int py = p / width;
			for (int dy = -1; dy <= 1; ++dy)
			{
				for (int dx = -1; dx <= 1; ++dx)
				{
					if ((dx == 0 && dy == 0) || (!eightConnected && dx != 0 && dy != 0))
					{
						continue;
					}

					int nx = px + dx, ny = py + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height)
					{
						continue;
					}

					int q = ny * width + nx;
					if (mask[q] == value && label[q] == 0)
					{
						label[q] = id;
						stack.Push(q);
					}
				}
			}
		}

		return size;
	}

	private static long Cross((long X, long Y) o, (long X, long Y) a, (long X, long Y) b) =>
		(a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

	// Where the parabola rooted at q overtakes the one rooted at p (q > p).
	private static double Intersection(double[] f, int q, int p) =>
		((f[q] + (double)q * q) - (f[p] + (double)p * p)) / (2.0 * q - 2.0 * p);

	// Felzenszwalb-Huttenlocher: the lower envelope of the parabolas (q - v)^2 + f(v).
	private static void Transform1D(double[] f, int n, double[] d, int[] v, double[] z)
	{
		int k = 0;
		v[0] = 0;
		z[0] = double.NegativeInfinity;
		z[1] = double.PositiveInfinity;
		for (int q = 1; q < n; ++q)
		{
			// z[0] is -infinity, so the search always stops at k = 0.
			double s = Intersection(f, q, v[k]);
			while (s <= z[k])
			{
				--k;
				s = Intersection(f, q, v[k]);
			}

			++k;
			v[k] = q;
			z[k] = s;
			z[k + 1] = double.PositiveInfinity;
		}

		k = 0;
		for (int q = 0; q < n; ++q)
		{
			while (z[k + 1] < q)
			{
				++k;
			}

			double diff = q - v[k];
			d[q] = diff * diff + f[v[k]];
		}
	}
}
