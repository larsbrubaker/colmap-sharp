// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteMaskSource: the masks of SilhouetteSegmenter.SegmentToMaskSource as the mask
// IImageSource that AutomaticReconstructionOptions.Masks (ImageReader, StereoFusion) reads. Not
// a COLMAP port. Masks are held at working resolution and upsampled to the image's size on each
// Read, so a long video costs a few kilobytes per frame rather than a full-size bitmap.

using ColmapSharp.Controllers;
using ColmapSharp.Sensor;

namespace ColmapSharp.Segmentation;

/// <summary>
/// Silhouette masks under "&lt;image name&gt;.png" (<see cref="SilhouetteSegmenter.MaskName"/>):
/// grey, 255 = keep, 0 = drop. An image where no object was found reads as all 255.
/// </summary>
public sealed class SilhouetteMaskSource : IImageSource
{
	private readonly Dictionary<string, WorkingMask> masks = new(StringComparer.Ordinal);
	private readonly Dictionary<string, bool> found = new(StringComparer.Ordinal);

	internal SilhouetteMaskSource(BackgroundModelKind background)
	{
		Background = background;
	}

	/// <summary>Which background model the segmentation used.</summary>
	public BackgroundModelKind Background { get; }

	/// <summary>The names of the images that got a mask, in segmentation order.</summary>
	public IReadOnlyList<string> ImageNames => [.. found.Keys];

	/// <summary>
	/// Whether an object was found in the image (false: its mask keeps everything). Throws for
	/// a name that was not segmented (not in the source, or it failed to decode).
	/// </summary>
	public bool ForegroundFound(string imageName) =>
		found.TryGetValue(imageName, out bool value)
			? value
			: throw new ArgumentException($"No image named '{imageName}' was segmented; see ImageNames.", nameof(imageName));

	/// <inheritdoc/>
	public IReadOnlyList<string> ListNames() => [.. masks.Keys];

	/// <inheritdoc/>
	public bool Exists(string name) => masks.ContainsKey(name);

	/// <inheritdoc/>
	public Bitmap? Read(string name) => masks.TryGetValue(name, out WorkingMask? mask) ? mask.Upsample() : null;

	internal void Add(string imageName, WorkingMask mask)
	{
		masks[SilhouetteSegmenter.MaskName(imageName)] = mask;
		found[imageName] = mask.Mask is not null;
	}
}

/// <summary>A mask at working resolution with the grid it upsamples from; null Mask = keep all.</summary>
internal sealed record WorkingMask(bool[]? Mask, int Width, int Height, int Factor, int SourceWidth, int SourceHeight)
{
	public static WorkingMask Of(LabImage frame, bool[]? mask) =>
		new(mask, frame.Width, frame.Height, frame.Factor, frame.SourceWidth, frame.SourceHeight);

	public Bitmap Upsample()
	{
		if (Mask is null)
		{
			var all = new Bitmap(SourceWidth, SourceHeight, asRgb: false);
			all.Fill(new BitmapColor<byte>(255));
			return all;
		}

		return LabImage.UpsampleMask(Mask, Width, Height, Factor, SourceWidth, SourceHeight);
	}
}
