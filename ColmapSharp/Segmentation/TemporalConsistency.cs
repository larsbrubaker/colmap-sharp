// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TemporalConsistency: repairs single-frame silhouette errors in a video using the frames around
// it (docs/QUALITY_PLAN.md stage 1a+). Not a COLMAP port. Runs after SilhouetteSegmenter has
// produced every frame's cleaned working-resolution mask, when SegmentationOptions.TemporalWindow
// is above 0. Frames are taken as time-ordered in the order given.
// 1. Alignment: each neighbour j within the window is warped onto frame k by the similarity
//    transform that maps j's mask moments onto k's (centroid, principal axis, scale from the area
//    ratio). This is only a first-order alignment: a turning 3D object changes its outline, not
//    just its pose in the image, and a near-round outline has no reliable axis (the rotation is
//    then taken as zero). So neighbours are weighted by time distance, and by how well they agree
//    with their own neighbours, and consensus only overrides a frame where it is strong.
// 2. Shape distance: the symmetric mean squared distance between the two contours (the Euclidean
//    distance transform of BinaryMorphology). A frame's score is the median of its distances to
//    its aligned neighbours, its bridge the median distance between frames k - i and k + i. It
//    is an outlier when its score is above TemporalMinOutlierDistance and above
//    TemporalOutlierRatio times both its neighbours' median score and its bridge. A one-frame
//    bite makes its own score large but only one of each neighbour's distances, and the frames
//    either side still agree with each other; when the object turns fast between samples every
//    frame differs from the next, the bridge is large too, and the frame is left alone (on the
//    f40 mouse clip, without the bridge a correct frame got a wall wedge added at a turning end).
//    Only outlier frames are changed. A mask touching the image border takes no part: its
//    moments are those of the visible part and do not follow the object. The bridge needs a
//    neighbour on both sides, so the first and last frame of a sequence (and a frame next to
//    one without a mask) are never repaired.
// 3. Fusion, for an outlier frame: P = w * own + (1 - w) * consensus, with own the frame's mask
//    (0/1) and consensus the weighted mean of the aligned neighbour masks. The result is P
//    thresholded with hysteresis (Canny-style, on the region): a pixel is foreground when
//    P >= TemporalHighThreshold, or P >= TemporalLowThreshold and it connects (8-neighbours)
//    through such pixels to one above the high threshold. Rule for not inventing geometry: with
//    the defaults (own weight 0.3, thresholds 0.55 / 0.45) a pixel the frame calls background
//    becomes foreground on its own when at least 79% (0.55 / 0.7) of the weighted neighbours
//    agree, and when it connects to such foreground at 64% (0.45 / 0.7); a pixel the frame calls
//    foreground is dropped only when fewer than 21% (0.15 / 0.7) agree, or fewer than 36% where
//    it is cut off from the object. A real notch that persists stays: at its first frame half
//    the symmetric window has it, 50% < 64%. Only a change seen in (nearly) one frame alone is
//    undone. Raising the low threshold toward 80% was not needed to keep the persistent notch,
//    and the mouse repairs (f120 098, f40 033) hold at these values. No
//    colour or edge veto is used: on the driving real case (a lit grey underside matching the
//    wall) the frame's colour evidence is exactly what is wrong, so a colour veto would keep the
//    bite. The bite itself shifted the frame's centroid and area, so the neighbours are aligned
//    once more to the moments of the frame with the first pass's additions (no removals), and
//    fusion repeats.
// 4. The fused mask goes through the segmenter's cleanup (open, largest component, close, fill).
// Every frame's result is written to its own slot from the unchanged inputs, so sequential and
// parallel runs agree.

namespace ColmapSharp.Segmentation;

internal static class TemporalConsistency
{
	// Axis ratio (minor / major eigenvalue of the second moments) above which an outline counts as
	// round and its principal axis is ignored.
	private const double RoundAxisRatio = 0.9;

	/// <summary>
	/// Replaces the masks of outlier frames with their temporally fused version. Masks that are
	/// null (no frame or no object) or of another working size take no part.
	/// </summary>
	public static void Apply(WorkingMask?[] masks, SegmentationOptions options, CancellationToken cancellationToken)
	{
		int window = options.TemporalWindow;
		int count = masks.Length;
		if (window <= 0 || count < 2)
		{
			return;
		}

		var parallelOptions = new ParallelOptions
		{
			CancellationToken = cancellationToken,
			MaxDegreeOfParallelism = options.MaxDegreeOfParallelism,
		};

		var moments = new Moments?[count];
		Parallel.For(0, count, parallelOptions, k =>
		{
			// An object cut by the image border has the moments of its visible part only, which
			// do not follow the object; such frames take no part.
			if (masks[k] is { Mask: { } mask } working && BinaryMorphology.BorderFraction(mask, working.Width, working.Height) == 0.0)
			{
				moments[k] = Moments.Of(mask, working.Width, working.Height);
			}
		});

		// score[k]: the median distance from frame k to its aligned neighbours (NaN: none).
		var score = new double[count];
		// bridge[k]: the median distance between frames k - i and k + i, aligned (NaN: none).
		var bridge = new double[count];
		Parallel.For(0, count, parallelOptions, k =>
		{
			var distances = new List<double>();
			foreach (int j in Neighbours(masks, moments, k, window))
			{
				WorkingMask target = masks[k]!;
				bool[] aligned = Threshold(Warp(masks[j]!, moments[j]!.Value, moments[k]!.Value));
				distances.Add(ContourDistance(target.Mask!, aligned, target.Width, target.Height));
			}

			score[k] = Median(distances);

			// The bridge: how well the neighbours on either side agree with each other across k.
			var across = new List<double>();
			for (int i = 1; i <= window; ++i)
			{
				int before = k - i, after = k + i;
				if (before >= 0 && after < count && IsCompatible(masks, moments, k, before) && IsCompatible(masks, moments, k, after))
				{
					WorkingMask target = masks[after]!;
					bool[] aligned = Threshold(Warp(masks[before]!, moments[before]!.Value, moments[after]!.Value));
					across.Add(ContourDistance(target.Mask!, aligned, target.Width, target.Height));
				}
			}

			bridge[k] = Median(across);
		});

		var result = new WorkingMask?[count];
		Parallel.For(0, count, parallelOptions, k =>
		{
			if (IsOutlier(score, bridge, k, window, options))
			{
				result[k] = Fuse(masks, moments, score, bridge, k, options);
			}
		});

		for (int k = 0; k < count; ++k)
		{
			if (result[k] is { } fused)
			{
				masks[k] = fused;
			}
		}
	}

	/// <summary>
	/// The symmetric mean squared distance between the contours of two masks of one size, in
	/// squared pixels (0 when either has no contour).
	/// </summary>
	internal static double ContourDistance(bool[] a, bool[] b, int width, int height)
	{
		bool[] contourA = Contour(a, width, height), contourB = Contour(b, width, height);
		double[] toA = BinaryMorphology.SquaredDistanceTo(contourA, width, height);
		double[] toB = BinaryMorphology.SquaredDistanceTo(contourB, width, height);
		double sumAB = 0.0, sumBA = 0.0;
		int countA = 0, countB = 0;
		for (int p = 0; p < a.Length; ++p)
		{
			if (contourA[p])
			{
				sumAB += toB[p];
				++countA;
			}

			if (contourB[p])
			{
				sumBA += toA[p];
				++countB;
			}
		}

		return countA == 0 || countB == 0 ? 0.0 : 0.5 * (sumAB / countA + sumBA / countB);
	}

	private static bool IsOutlier(double[] score, double[] bridge, int k, int window, SegmentationOptions options)
	{
		if (double.IsNaN(score[k]) || score[k] <= options.TemporalMinOutlierDistance)
		{
			return false;
		}

		var others = new List<double>();
		for (int j = Math.Max(0, k - window); j <= Math.Min(score.Length - 1, k + window); ++j)
		{
			if (j != k && !double.IsNaN(score[j]))
			{
				others.Add(score[j]);
			}
		}

		// The neighbours must also agree with each other across k: when the object turns fast
		// between samples the moment alignment is poor, every frame differs from the next, and
		// a consensus would invent geometry (a wall wedge at a turning end) rather than repair it.
		return others.Count > 0
			&& score[k] > options.TemporalOutlierRatio * Median(others)
			&& !double.IsNaN(bridge[k])
			&& score[k] > options.TemporalOutlierRatio * bridge[k];
	}

	private static WorkingMask Fuse(WorkingMask?[] masks, Moments?[] moments, double[] score, double[] bridge, int k, SegmentationOptions options)
	{
		WorkingMask own = masks[k]!;
		int width = own.Width, height = own.Height, window = options.TemporalWindow;
		// A neighbour's agreement weight falls with its own score, relative to the typical score
		// in the window; neighbours that are outliers themselves take no part.
		var typical = new List<double>();
		foreach (int j in Neighbours(masks, moments, k, window))
		{
			typical.Add(score[j]);
		}

		double scale = Math.Max(options.TemporalMinOutlierDistance, Median(typical));
		bool[] fused = own.Mask!;
		Moments target = moments[k]!.Value;
		for (int pass = 0; pass < 2; ++pass)
		{
			var consensus = new double[fused.Length];
			double weightSum = 0.0;
			foreach (int j in Neighbours(masks, moments, k, window))
			{
				if (IsOutlier(score, bridge, j, window, options))
				{
					continue;
				}

				double timeWeight = (window + 1.0 - Math.Abs(j - k)) / (window + 1.0);
				double weight = timeWeight / (1.0 + score[j] / scale);
				float[] aligned = Warp(masks[j]!, moments[j]!.Value, target);
				for (int p = 0; p < consensus.Length; ++p)
				{
					consensus[p] += weight * aligned[p];
				}

				weightSum += weight;
			}

			if (weightSum <= 0.0)
			{
				return own;
			}

			var probability = new double[fused.Length];
			double ownWeight = options.TemporalOwnWeight;
			for (int p = 0; p < probability.Length; ++p)
			{
				probability[p] = ownWeight * (own.Mask![p] ? 1.0 : 0.0) + (1.0 - ownWeight) * consensus[p] / weightSum;
			}

			fused = Hysteresis(probability, width, height, options.TemporalHighThreshold, options.TemporalLowThreshold);
			if (Array.IndexOf(fused, true) < 0)
			{
				return own;
			}

			// Re-centre on the frame with the repairs added but nothing removed: the bite biased
			// the first alignment (centroid away from it, area too small), which misplaces the
			// neighbours' rims, and removals made from that alignment would carry the same bias.
			var repaired = new bool[fused.Length];
			for (int p = 0; p < repaired.Length; ++p)
			{
				repaired[p] = fused[p] || own.Mask![p];
			}

			target = Moments.Of(repaired, width, height);
		}

		bool[]? cleaned = SilhouetteSegmenter.Cleanup(fused, width, height, options);
		return cleaned is null ? own : own with { Mask = cleaned };
	}

	// Foreground where p >= high, grown through 8-connected pixels with p >= low.
	private static bool[] Hysteresis(double[] probability, int width, int height, double high, double low)
	{
		var result = new bool[probability.Length];
		var stack = new Stack<int>();
		for (int p = 0; p < probability.Length; ++p)
		{
			if (probability[p] >= high && !result[p])
			{
				result[p] = true;
				stack.Push(p);
				while (stack.Count > 0)
				{
					int q = stack.Pop();
					int qx = q % width, qy = q / width;
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dx = -1; dx <= 1; ++dx)
						{
							int nx = qx + dx, ny = qy + dy;
							if (nx < 0 || ny < 0 || nx >= width || ny >= height)
							{
								continue;
							}

							int n = ny * width + nx;
							if (!result[n] && probability[n] >= low)
							{
								result[n] = true;
								stack.Push(n);
							}
						}
					}
				}
			}
		}

		return result;
	}

	// Whether frames k and j both take part (a mask, moments) and share a working size.
	private static bool IsCompatible(WorkingMask?[] masks, Moments?[] moments, int k, int j) =>
		moments[k] is not null && moments[j] is not null
		&& masks[k]!.Width == masks[j]!.Width && masks[k]!.Height == masks[j]!.Height;

	// The frames within the window of k that take part with it.
	private static IEnumerable<int> Neighbours(WorkingMask?[] masks, Moments?[] moments, int k, int window)
	{
		for (int j = Math.Max(0, k - window); j <= Math.Min(masks.Length - 1, k + window); ++j)
		{
			if (j != k && IsCompatible(masks, moments, k, j))
			{
				yield return j;
			}
		}
	}

	// Samples mask `from` (bilinear, 0 outside) at the similarity transform of every pixel of the
	// target grid that maps the target's moments onto the source's.
	private static float[] Warp(WorkingMask from, Moments source, Moments target)
	{
		int width = from.Width, height = from.Height;
		bool[] mask = from.Mask!;
		double scale = Math.Sqrt(source.Area / target.Area);
		double rotation = source.IsRound || target.IsRound ? 0.0 : WrapHalfTurn(source.Angle - target.Angle);
		double cos = Math.Cos(rotation) * scale, sin = Math.Sin(rotation) * scale;
		var result = new float[width * height];
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				double dx = x - target.CenterX, dy = y - target.CenterY;
				double sx = source.CenterX + cos * dx - sin * dy;
				double sy = source.CenterY + sin * dx + cos * dy;
				result[y * width + x] = (float)Bilinear(mask, width, height, sx, sy);
			}
		}

		return result;
	}

	private static double Bilinear(bool[] mask, int width, int height, double x, double y)
	{
		int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y);
		double fx = x - x0, fy = y - y0;
		double At(int px, int py) => px < 0 || py < 0 || px >= width || py >= height || !mask[py * width + px] ? 0.0 : 1.0;
		return (1.0 - fy) * ((1.0 - fx) * At(x0, y0) + fx * At(x0 + 1, y0))
			+ fy * ((1.0 - fx) * At(x0, y0 + 1) + fx * At(x0 + 1, y0 + 1));
	}

	// An angle difference of undirected axes, wrapped to (-pi/2, pi/2].
	private static double WrapHalfTurn(double angle)
	{
		while (angle > Math.PI / 2.0)
		{
			angle -= Math.PI;
		}

		while (angle <= -Math.PI / 2.0)
		{
			angle += Math.PI;
		}

		return angle;
	}

	private static bool[] Threshold(float[] soft)
	{
		var result = new bool[soft.Length];
		for (int p = 0; p < soft.Length; ++p)
		{
			result[p] = soft[p] >= 0.5f;
		}

		return result;
	}

	// Foreground pixels with a background 4-neighbour inside the image (the image border is not
	// part of an object's contour: a frame-filling object is cut there, not outlined).
	private static bool[] Contour(bool[] mask, int width, int height)
	{
		var result = new bool[mask.Length];
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int p = y * width + x;
				result[p] = mask[p]
					&& ((x > 0 && !mask[p - 1]) || (x + 1 < width && !mask[p + 1])
						|| (y > 0 && !mask[p - width]) || (y + 1 < height && !mask[p + width]));
			}
		}

		return result;
	}

	private static double Median(List<double> values)
	{
		if (values.Count == 0)
		{
			return double.NaN;
		}

		values.Sort();
		int mid = values.Count / 2;
		return values.Count % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
	}

	/// <summary>Area, centroid and principal-axis angle of a mask (pixel coordinates).</summary>
	private readonly record struct Moments(double Area, double CenterX, double CenterY, double Angle, bool IsRound)
	{
		public static Moments Of(bool[] mask, int width, int height)
		{
			double area = 0.0, sx = 0.0, sy = 0.0;
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					if (mask[y * width + x])
					{
						area += 1.0;
						sx += x;
						sy += y;
					}
				}
			}

			double cx = sx / area, cy = sy / area;
			double xx = 0.0, yy = 0.0, xy = 0.0;
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					if (mask[y * width + x])
					{
						double dx = x - cx, dy = y - cy;
						xx += dx * dx;
						yy += dy * dy;
						xy += dx * dy;
					}
				}
			}

			double half = 0.5 * (xx + yy);
			double spread = Math.Sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
			double major = half + spread, minor = half - spread;
			bool round = major <= 0.0 || minor / major > RoundAxisRatio;
			return new Moments(area, cx, cy, 0.5 * Math.Atan2(2.0 * xy, xx - yy), round);
		}
	}
}
