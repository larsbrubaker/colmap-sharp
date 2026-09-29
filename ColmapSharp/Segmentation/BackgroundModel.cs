// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BackgroundModel: step 1 of SilhouetteSegmenter - the per-pixel temporal median background and
// the initial foreground masks (difference from the median, or N. Otsu's threshold on L*, IEEE
// SMC 9(1), 1979, as the fallback, grown by enclosed textured regions: GrowByTexture). Not a COLMAP port. The median is built from an evenly spaced
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
		if (Math.Min(upperBorder, lowerBorder) > MaxObjectBorderFraction)
		{
			return new bool[foreground.Length];
		}

		return GrowByTexture(frame, foreground);
	}

	// Otsu on L* splits a textured object's own light and dark parts as readily as it splits
	// object from wall: on a mid-grey wall the object's lighter half (and its wall-grey texture)
	// joins the wall's class. Found on the TexturedSphere scene, where that half was cut away:
	// outside the hull of the dark half it was sure background, which GrabCut can never give
	// back. Colour cannot tell it from the wall, but texture can: the wall is smooth. Textured
	// regions (Textured) that do not reach the image border and touch the Otsu object are added
	// to it. A matte object on a plain wall (the mouse) gains little or nothing. Distance from a
	// plane fitted to the wall's colour was tried as a second test; it added nothing on the
	// sphere and trimmed the rim of a dark blob whose white label channel it added.
	private static bool[] GrowByTexture(LabImage frame, bool[] foreground)
	{
		int width = frame.Width, height = frame.Height;
		// The wall class: the Otsu class that is not the object, where it reaches the border.
		bool[] wall = BinaryMorphology.Not(BinaryMorphology.FillHoles(foreground, width, height));
		bool[] textured = Textured(frame, wall, foreground, width, height);
		for (int p = 0; p < textured.Length; ++p)
		{
			textured[p] &= !foreground[p];
		}

		bool[] reachesBorder = BinaryMorphology.Reconstruct(textured, BinaryMorphology.BorderPixels(textured, width, height), width, height);
		var union = new bool[textured.Length];
		for (int p = 0; p < textured.Length; ++p)
		{
			union[p] = foreground[p] || (textured[p] && !reachesBorder[p]);
		}

		// Additions count only where they connect to the Otsu object.
		bool[] grown = BinaryMorphology.Reconstruct(union, foreground, width, height);
		for (int p = 0; p < grown.Length; ++p)
		{
			grown[p] |= foreground[p];
		}

		return grown;
	}

	// Texture test of GrowByTexture: window radius, and the factor over the wall's own energy
	// (its TextureWallQuantile quantile in the border band). On the TexturedSphere scene the
	// sphere's weakest texture is about 2.5 times the wall's typical energy, and the masks held
	// from quantile 0.5 to 0.9 and factor 2 to 4. The high quantile is the margin for a real
	// wall, whose energy varies more than sensor noise: with the median (and before the rim
	// was left out, below) the mouse captures grew bumps into the wall.
	private const int TextureRadius = 3;
	private const double TextureRatio = 3.0;
	private const double TextureWallQuantile = 0.9;

	// Pixels whose local gradient energy of L* (mean squared central difference over a
	// (2r+1)^2 window) is well above the wall's: the wall is smooth, a textured object is not.
	// The wall's energy is measured only in a band along the image border (a tenth of the
	// shorter side): the object's own wall-coloured texture sits in the wall class too, and
	// measured over the whole class it raised the threshold enough to lose the sphere again.
	// Eroded by the window radius plus the central difference's reach, so the rim's edge
	// response does not spill onto the wall.
	private static bool[] Textured(LabImage frame, bool[] wall, bool[] foreground, int width, int height)
	{
		// The Otsu object's own rim is an edge, not texture: counted, it made every narrow
		// wall-coloured notch in a dark object read as textured and filled it. Differences that
		// touch the object are left out, and each window averages only over what is left.
		bool[] nearObject = BinaryMorphology.Dilate(foreground, width, height, 1.5);
		var energy = new double[width * height];
		var counted = new double[width * height];
		for (int y = 1; y < height - 1; ++y)
		{
			for (int x = 1; x < width - 1; ++x)
			{
				int p = y * width + x;
				if (nearObject[p])
				{
					continue;
				}

				counted[p] = 1.0;
				double gx = frame.L[p + 1] - frame.L[p - 1];
				double gy = frame.L[p + width] - frame.L[p - width];
				energy[p] = gx * gx + gy * gy;
			}
		}

		double[] integral = Integral(energy, width, height);
		double[] integralCount = Integral(counted, width, height);

		var local = new double[width * height];
		var wallValues = new List<double>();
		int band = Math.Max(1, Math.Min(width, height) / 10);
		for (int y = 0; y < height; ++y)
		{
			int y0 = Math.Max(0, y - TextureRadius), y1 = Math.Min(height, y + TextureRadius + 1);
			for (int x = 0; x < width; ++x)
			{
				int x0 = Math.Max(0, x - TextureRadius), x1 = Math.Min(width, x + TextureRadius + 1);
				double sum = BoxSum(integral, width, x0, y0, x1, y1);
				double n = BoxSum(integralCount, width, x0, y0, x1, y1);
				int p = y * width + x;
				local[p] = n > 0.0 ? sum / n : 0.0;
				if (wall[p] && (x < band || y < band || x >= width - band || y >= height - band))
				{
					wallValues.Add(local[p]);
				}
			}
		}

		if (wallValues.Count == 0)
		{
			return new bool[local.Length];
		}

		wallValues.Sort();
		double threshold = TextureRatio * wallValues[(int)((wallValues.Count - 1) * TextureWallQuantile)];
		var textured = new bool[local.Length];
		for (int p = 0; p < local.Length; ++p)
		{
			textured[p] = local[p] > threshold;
		}

		return BinaryMorphology.Erode(textured, width, height, TextureRadius + 1);
	}

	// Summed-area table with a zero first row and column: (width + 1) x (height + 1).
	private static double[] Integral(double[] values, int width, int height)
	{
		var integral = new double[(width + 1) * (height + 1)];
		for (int y = 0; y < height; ++y)
		{
			double row = 0.0;
			for (int x = 0; x < width; ++x)
			{
				row += values[y * width + x];
				integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + row;
			}
		}

		return integral;
	}

	// The sum over [x0, x1) x [y0, y1) of a summed-area table.
	private static double BoxSum(double[] integral, int width, int x0, int y0, int x1, int y1) =>
		integral[y1 * (width + 1) + x1] - integral[y0 * (width + 1) + x1]
		- integral[y1 * (width + 1) + x0] + integral[y0 * (width + 1) + x0];

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
