// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteSegmenter: automatic foreground masks (silhouettes) for a photo set or video of one
// object in front of a plain background, handed to the reconstruction as the mask source of
// AutomaticReconstructionOptions.Masks. Not a COLMAP port; COLMAP only consumes masks. Stage 1a
// of docs/QUALITY_PLAN.md. Pipeline, per frame at working resolution (LabImage):
// 1. Background model (BackgroundModel): the per-pixel temporal median of a sample of the frames
//    when there are enough of one size and the camera is near-static; otherwise (or when an
//    object that barely moves survived into the median) N. Otsu's threshold on L* (IEEE SMC 1979).
//    The initial mask is the pixels far from the background.
// 2. Trimap: the eroded initial mask is sure-foreground; the dilated initial mask and the convex
//    hull of the initial object (opened, largest component) are unknown; the rest is sure-
//    background. The hull matters on real footage: a lit grey underside matches the wall, and
//    as sure-background it would be a bite GrabCut could never give back.
// 3. GrabCut refinement (Rother, Kolmogorov and Blake, SIGGRAPH 2004) in GrabCutRefiner/ColorGmm,
//    with the background colour model learnt only outside the hull, from pixels that reach the
//    border.
// 4. Cleanup (BinaryMorphology): opening by a disk sized to the object's width (drops a cable or
//    a thin stand), the largest connected component, closing sized the same way (seals channels
//    that white label text cuts in from the rim), then hole filling (a specular highlight inside
//    the object stays foreground).
// Masks are upsampled with the downsampling factor to each frame's size: grey, 255 = keep,
// 0 = drop, named "<image name>.png" - the convention ImageReader (mask lookup),
// FeatureExtraction.MaskFeatures and StereoFusion (MaskPath) share. Frames are independent after
// the background model, so they run in parallel, each writing only its own slot. With
// SegmentationOptions.TemporalWindow above 0, TemporalConsistency then repairs single-frame
// errors of a video from the frames around each one (stage 1a+).

using ColmapSharp.Controllers;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Segmentation;

/// <summary>The masks of <see cref="SilhouetteSegmenter.Segment"/>.</summary>
public sealed class SegmentationResult
{
	internal SegmentationResult(Bitmap[] masks, bool[] foregroundFound, BackgroundModelKind background)
	{
		Masks = masks;
		ForegroundFound = foregroundFound;
		Background = background;
	}

	/// <summary>
	/// One grey mask per frame, the frame's size: 255 where the object is (keep), 0 elsewhere.
	/// A frame where no object was found gets an all-255 mask, so it is used unmasked rather
	/// than dropped.
	/// </summary>
	public IReadOnlyList<Bitmap> Masks { get; }

	/// <summary>Whether an object was found in each frame.</summary>
	public IReadOnlyList<bool> ForegroundFound { get; }

	/// <summary>Which background model was used, and why.</summary>
	public BackgroundModelKind Background { get; }

	/// <summary>Whether the temporal-median background was used (false: the Otsu fallback).</summary>
	public bool UsedTemporalMedian => Background == BackgroundModelKind.TemporalMedian;
}

/// <summary>
/// Automatic silhouette masks for a single object in front of a plain background. See the file
/// header for the pipeline and its sources.
/// </summary>
public static class SilhouetteSegmenter
{
	/// <summary>The name under which the mask of <paramref name="imageName"/> is looked up.</summary>
	public static string MaskName(string imageName) => imageName + ".png";

	/// <summary>
	/// Segments a sequence of frames (grey or RGB). Frames of different sizes are allowed; the
	/// temporal median is then not used. Reports the number of frames finished.
	/// </summary>
	public static SegmentationResult Segment(
		IReadOnlyList<Bitmap> frames,
		SegmentationOptions? options = null,
		IProgress<int>? progress = null,
		CancellationToken cancellationToken = default)
	{
		options ??= new SegmentationOptions();
		options.Validate();
		var masks = new Bitmap[frames.Count];
		var found = new bool[frames.Count];
		BackgroundModelKind kind = Run(
			frames.Count,
			i => LabImage.FromBitmap(frames[i], options.WorkingSize),
			(i, mask) =>
			{
				masks[i] = mask.Upsample();
				found[i] = mask.Mask is not null;
			},
			options,
			progress,
			cancellationToken);
		return new SegmentationResult(masks, found, kind);
	}

	/// <summary>
	/// Segments every image of <paramref name="images"/> (in ordinal name order, as ImageReader
	/// reads them) and returns the masks as a mask source for
	/// <see cref="AutomaticReconstructionOptions.Masks"/>. Images that fail to decode get no mask.
	/// Only the median's sample of frames is held at once (downsampled); the masks are kept at
	/// working resolution and upsampled when read. Reads from <paramref name="images"/> are
	/// serialised, so the host source need not be thread-safe.
	/// </summary>
	public static SilhouetteMaskSource SegmentToMaskSource(
		IImageSource images,
		SegmentationOptions? options = null,
		IProgress<int>? progress = null,
		CancellationToken cancellationToken = default)
	{
		options ??= new SegmentationOptions();
		options.Validate();
		string[] names = [.. images.ListNames().OrderBy(n => n, StringComparer.Ordinal)];
		var gate = new object();
		var masks = new WorkingMask?[names.Length];
		BackgroundModelKind kind = Run(
			names.Length,
			i =>
			{
				Bitmap? bitmap;
				lock (gate)
				{
					bitmap = images.Read(names[i]);
				}

				return bitmap is null || bitmap.IsEmpty ? null : LabImage.FromBitmap(bitmap, options.WorkingSize);
			},
			(i, mask) => masks[i] = mask,
			options,
			progress,
			cancellationToken);

		var source = new SilhouetteMaskSource(kind);
		for (int i = 0; i < names.Length; ++i)
		{
			if (masks[i] is { } mask)
			{
				source.Add(names[i], mask);
			}
		}

		return source;
	}

	/// <summary>
	/// Wraps full-size masks as a mask source for <see cref="AutomaticReconstructionOptions.Masks"/>
	/// (and ImageReaderOptions.Masks): mask i under <see cref="MaskName"/> of image i.
	/// </summary>
	public static InMemoryImageSource CreateMaskSource(IReadOnlyList<string> imageNames, IReadOnlyList<Bitmap> masks)
	{
		Check.Eq(imageNames.Count, masks.Count);
		var source = new InMemoryImageSource();
		for (int i = 0; i < imageNames.Count; ++i)
		{
			source.Add(MaskName(imageNames[i]), masks[i]);
		}

		return source;
	}

	// Builds the background from the sampled frames, then loads and segments every frame
	// (loading again the sampled ones, so only the sample is ever held). A null load is a frame
	// that cannot be decoded; it gets no mask.
	private static BackgroundModelKind Run(
		int count,
		Func<int, LabImage?> load,
		Action<int, WorkingMask> store,
		SegmentationOptions options,
		IProgress<int>? progress,
		CancellationToken cancellationToken)
	{
		var sampled = new List<LabImage>();
		if (count >= options.MinMedianFrames)
		{
			foreach (int i in BackgroundModel.SampleIndices(count, options))
			{
				cancellationToken.ThrowIfCancellationRequested();
				if (load(i) is { } frame)
				{
					sampled.Add(frame);
				}
			}
		}

		(LabImage? median, BackgroundModelKind kind) = BackgroundModel.Build(sampled, count, options, cancellationToken);
		sampled.Clear();

		int done = 0;
		var parallelOptions = new ParallelOptions
		{
			CancellationToken = cancellationToken,
			MaxDegreeOfParallelism = options.MaxDegreeOfParallelism,
		};
		if (options.TemporalWindow == 0)
		{
			// Frames are independent: each is segmented and stored (upsampled) by its own worker.
			Parallel.For(0, count, parallelOptions, i =>
			{
				if (load(i) is { } frame)
				{
					store(i, WorkingMask.Of(frame, SegmentFrame(frame, median, options, cancellationToken)));
				}

				progress?.Report(Interlocked.Increment(ref done));
			});

			return kind;
		}

		// Temporal consistency needs every frame's mask first. The last frame's progress step is
		// held back until the temporal pass and the stores are done, so 100% means finished.
		var masks = new WorkingMask?[count];
		Parallel.For(0, count, parallelOptions, i =>
		{
			if (load(i) is { } frame)
			{
				masks[i] = WorkingMask.Of(frame, SegmentFrame(frame, median, options, cancellationToken));
			}

			int finished = Interlocked.Increment(ref done);
			if (finished < count)
			{
				progress?.Report(finished);
			}
		});

		TemporalConsistency.Apply(masks, options, cancellationToken);
		Parallel.For(0, count, parallelOptions, i =>
		{
			if (masks[i] is { } mask)
			{
				store(i, mask);
			}
		});
		progress?.Report(count);

		return kind;
	}

	// The mask at working resolution, or null when no object was found.
	private static bool[]? SegmentFrame(LabImage frame, LabImage? median, SegmentationOptions options, CancellationToken cancellationToken)
	{
		int width = frame.Width, height = frame.Height;
		bool[] initial = BackgroundModel.InitialMask(frame, median, options);

		bool[] sure = BinaryMorphology.Erode(initial, width, height, options.TrimapErodeRadius);
		if (Array.IndexOf(sure, true) < 0)
		{
			return null;
		}

		(byte[] trimap, bool[] backgroundModelAllowed) = BuildTrimap(initial, sure, width, height, options);
		bool[] mask = GrabCutRefiner.Refine(frame, trimap, initial, backgroundModelAllowed, options, cancellationToken);
		return Cleanup(mask, width, height, options);
	}

	/// <summary>
	/// The cleanup after GrabCut (and after temporal fusion): opening sized to the object, the
	/// largest component, closing, hole filling. Null when nothing is left.
	/// </summary>
	internal static bool[]? Cleanup(bool[] mask, int width, int height, SegmentationOptions options)
	{
		mask = BinaryMorphology.Open(mask, width, height, ObjectRadius(mask, width, height, options.OpeningFraction));
		if (options.KeepLargestComponent)
		{
			mask = BinaryMorphology.LargestComponent(mask, width, height);
		}

		if (options.CloseFraction > 0.0)
		{
			mask = BinaryMorphology.Close(mask, width, height, ObjectRadius(mask, width, height, options.CloseFraction));
		}

		if (options.FillHoles)
		{
			mask = BinaryMorphology.FillHoles(mask, width, height);
		}

		return Array.IndexOf(mask, true) < 0 ? null : mask;
	}


	/// <summary>
	/// The GrabCut trimap of an initial mask (<paramref name="sure"/> is its erosion), and the
	/// pixels the background colour model may learn from.
	/// </summary>
	internal static (byte[] Trimap, bool[] BackgroundModelAllowed) BuildTrimap(bool[] initial, bool[] sure, int width, int height, SegmentationOptions options)
	{
		// The object's hull, from the initial mask without its cable and specks. Closing first:
		// where lit rim breaks the initial mask into streaks, opening alone would eat the whole
		// region and the hull would cut a bite across it.
		double coreRadius = ObjectRadius(initial, width, height, options.OpeningFraction);
		bool[] core = BinaryMorphology.LargestComponent(
			BinaryMorphology.Open(BinaryMorphology.Close(initial, width, height, coreRadius), width, height, coreRadius),
			width,
			height);
		bool[] hull = BinaryMorphology.ConvexHull(core, width, height);
		bool[] possible = BinaryMorphology.Dilate(initial, width, height, options.TrimapDilateRadius);
		var trimap = new byte[initial.Length];
		// The background colour model learns only from sure background: outside the hull and
		// outside the unknown band. Letting the band train it (standard GrabCut learns from all
		// of T_B, which the band was not) taught it the lit grey rim of the mouse as wall.
		var backgroundModelAllowed = new bool[trimap.Length];
		for (int p = 0; p < trimap.Length; ++p)
		{
			backgroundModelAllowed[p] = !hull[p] && !possible[p];
			trimap[p] = sure[p] ? GrabCutRefiner.SureForeground
				: possible[p] || hull[p] ? GrabCutRefiner.Unknown
				: GrabCutRefiner.SureBackground;
		}

		return (trimap, backgroundModelAllowed);
	}

	// A disk radius as a fraction of the object's width (twice its largest inscribed radius),
	// at least one pixel.
	private static double ObjectRadius(bool[] mask, int width, int height, double fraction) =>
		Math.Max(1.0, fraction * 2.0 * BinaryMorphology.MaxInscribedRadius(mask, width, height));
}
