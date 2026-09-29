// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BackgroundModel: step 1 of SilhouetteSegmenter - the per-pixel temporal median background and
// the initial foreground masks (difference from the median, or N. Otsu's threshold on L*, IEEE
// SMC 9(1), 1979, as the fallback). Not a COLMAP port. The median is built from an evenly spaced
// sample of the frames only, so a long video never needs every frame in memory at once.

namespace ColmapSharp.Segmentation;

/// <summary>Which background model <see cref="SilhouetteSegmenter"/> used, and why.</summary>
public enum BackgroundModelKind
{
	/// <summary>The per-pixel temporal median of the frames.</summary>
	TemporalMedian,

	/// <summary>Otsu on L*: fewer frames than <see cref="SegmentationOptions.MinMedianFrames"/>.</summary>
	OtsuTooFewFrames,

	/// <summary>Otsu on L*: the sampled frames differ in size.</summary>
	OtsuMixedSizes,

	/// <summary>Otsu on L*: the camera moved (too little of each frame agrees with the median).</summary>
	OtsuMovingCamera,

	/// <summary>Otsu on L*: an object that barely moved survived into the median.</summary>
	OtsuPersistentObject,
}

internal static class BackgroundModel
{
	// The most of the image border an Otsu foreground class may cover. An object cut by the
	// frame edge covers a few percent; a class covering a quarter is part of the wall.
	private const double MaxObjectBorderFraction = 0.25;

	/// <summary>
	/// The indices of the frames that go into the median: at most MaxMedianFrames, evenly spaced.
	/// </summary>
	public static int[] SampleIndices(int count, SegmentationOptions options)
	{
		int m = Math.Min(count, options.MaxMedianFrames);
		var indices = new int[m];
		for (int j = 0; j < m; ++j)
		{
			indices[j] = (int)((long)j * count / m);
		}

		return indices;
	}

	/// <summary>
	/// The temporal median of <paramref name="sampled"/> and the model kind; the median is null
	/// unless the kind is <see cref="BackgroundModelKind.TemporalMedian"/>.
	/// </summary>
	public static (LabImage? Median, BackgroundModelKind Kind) Build(IReadOnlyList<LabImage> sampled, int frameCount, SegmentationOptions options, CancellationToken cancellationToken)
	{
		if (frameCount < options.MinMedianFrames || sampled.Count == 0)
		{
			return (null, BackgroundModelKind.OtsuTooFewFrames);
		}

		LabImage first = sampled[0];
		foreach (LabImage frame in sampled)
		{
			if (frame.Width != first.Width || frame.Height != first.Height || frame.Factor != first.Factor)
			{
				return (null, BackgroundModelKind.OtsuMixedSizes);
			}
		}

		int m = sampled.Count;
		int width = first.Width, height = first.Height;
		var median = new LabImage(width, height, first.Factor, first.SourceWidth, first.SourceHeight);
		var parallelOptions = new ParallelOptions
		{
			CancellationToken = cancellationToken,
			MaxDegreeOfParallelism = options.MaxDegreeOfParallelism,
		};
		Parallel.For(0, height, parallelOptions, y =>
		{
			var values = new float[m];
			for (int x = 0; x < width; ++x)
			{
				int p = y * width + x;
				median.L[p] = Median(sampled, frame => frame.L[p], values);
				median.A[p] = Median(sampled, frame => frame.A[p], values);
				median.B[p] = Median(sampled, frame => frame.B[p], values);
			}
		});

		// Near-static camera: most of each frame agrees with the median (the median over frames
		// of the agreeing fraction must reach StaticAgreement).
		var agreement = new double[m];
		for (int j = 0; j < m; ++j)
		{
			bool[] different = DifferenceMask(sampled[j], median, options.BackgroundDistance);
			agreement[j] = 1.0 - (double)different.Count(d => d) / different.Length;
		}

		Array.Sort(agreement);
		if (agreement[(m - 1) / 2] < options.StaticAgreement)
		{
			return (null, BackgroundModelKind.OtsuMovingCamera);
		}

		return HasPersistentObject(median, options)
			? (null, BackgroundModelKind.OtsuPersistentObject)
			: (median, BackgroundModelKind.TemporalMedian);
	}

	/// <summary>
	/// The initial foreground of a frame: far from the median in Lab when a median fits the
	/// frame's grid, otherwise the Otsu class of L* that covers less of the image border.
	/// </summary>
	public static bool[] InitialMask(LabImage frame, LabImage? median, SegmentationOptions options)
	{
		bool fits = median is not null && median.Width == frame.Width && median.Height == frame.Height;
		return fits ? DifferenceMask(frame, median!, options.BackgroundDistance) : OtsuMask(frame);
	}

	// Whether the median background holds a sizable blob that does not reach the border: then
	// an object stayed where it was in most frames and the difference would miss it.
	private static bool HasPersistentObject(LabImage median, SegmentationOptions options)
	{
		int width = median.Width, height = median.Height;
		bool[] upper = UpperClass(median);
		foreach (bool[] cls in new[] { upper, BinaryMorphology.Not(upper) })
		{
			// Erode first, so a thin cable joining the object to the border does not count.
			bool[] core = BinaryMorphology.Erode(cls, width, height, 2.0);
			// Pixels of the class not 4-connected to the border through the class.
			bool[] interior = BinaryMorphology.FillHoles(BinaryMorphology.Not(core), width, height);
			for (int p = 0; p < interior.Length; ++p)
			{
				interior[p] = interior[p] && core[p];
			}

			bool[] blob = BinaryMorphology.LargestComponent(interior, width, height);
			double area = (double)blob.Count(b => b) / blob.Length;
			if (area > 0.0 && area >= options.PersistentObjectMinArea)
			{
				return true;
			}
		}

		return false;
	}

	private static bool[] OtsuMask(LabImage frame)
	{
		bool[] upper = UpperClass(frame);
		bool[] lower = BinaryMorphology.Not(upper);
		double upperBorder = BinaryMorphology.BorderFraction(upper, frame.Width, frame.Height);
		double lowerBorder = BinaryMorphology.BorderFraction(lower, frame.Width, frame.Height);
		bool[] foreground = upperBorder < lowerBorder ? upper : lower;
		// Both classes run along the border: Otsu split the wall (a gradient, no object).
		return Math.Min(upperBorder, lowerBorder) > MaxObjectBorderFraction ? new bool[foreground.Length] : foreground;
	}

	private static bool[] UpperClass(LabImage image)
	{
		double threshold = BinaryMorphology.OtsuThreshold(image.L, 0.0, 100.0);
		var upper = new bool[image.L.Length];
		for (int p = 0; p < upper.Length; ++p)
		{
			upper[p] = image.L[p] > threshold;
		}

		return upper;
	}

	private static bool[] DifferenceMask(LabImage frame, LabImage background, double distance)
	{
		double d2 = distance * distance;
		var mask = new bool[frame.L.Length];
		for (int p = 0; p < mask.Length; ++p)
		{
			double dl = frame.L[p] - background.L[p];
			double da = frame.A[p] - background.A[p];
			double db = frame.B[p] - background.B[p];
			mask[p] = dl * dl + da * da + db * db > d2;
		}

		return mask;
	}

	// The lower median, so an even count picks a sample rather than averaging two.
	private static float Median(IReadOnlyList<LabImage> frames, Func<LabImage, float> value, float[] buffer)
	{
		for (int j = 0; j < frames.Count; ++j)
		{
			buffer[j] = value(frames[j]);
		}

		Array.Sort(buffer);
		return buffer[(frames.Count - 1) / 2];
	}
}
