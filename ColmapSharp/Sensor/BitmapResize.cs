// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BitmapResize: Bitmap::Rescale and Bitmap::Thumbnail of colmap/sensor/bitmap.cc. Part of
// Bitmap (Bitmap.cs).
//
// COLMAP resamples with OpenImageIO's ImageBufAlgo::resize (a "triangle" filter for
// kBilinear, "box" for kBox), which is native and not ported. This is a managed resampler
// written here to the model OIIO's output follows, found by probing pycolmap 4.2.0
// (oracle/fixture_bitmap_rescale.py): each destination pixel center maps to source
// coordinates by the size ratio; the filter (triangle of radius 1, box of radius 1/2, a
// box including samples exactly on its edge) is widened by the ratio when downsampling, so
// downsampling antialiases; source samples beyond the border repeat the edge pixel (clamp)
// and still count toward the weight total; the filter is separable; the result is rounded
// half away from zero to a byte. Accumulation is in double.
//
// Not Tier A (divergence 9). Bilinear (the default, used by Thumbnail) is
// Tier B: OIIO accumulates in float, so a sum within float rounding of a half-integer can
// round to the neighboring gray level; BitmapRescaleOracleTests pins it against pycolmap
// within one gray level. Box matches OIIO on every probe without a tie, but when a source
// pixel center lies exactly on the box edge at a non-integer ratio (23 -> 10 pixels), OIIO
// includes or excludes it by rules not reproduced here, so a destination pixel can average
// one more or one fewer source pixel. bitmap_test.cc pins only the output dimensions and
// that the two filters differ.
// Thumbnail's size arithmetic is exact.

using ColmapSharp.Util;

namespace ColmapSharp.Sensor;

public sealed partial class Bitmap
{
	/// <summary>Port of Bitmap::RescaleFilter.</summary>
	public enum RescaleFilter
	{
		/// <summary>Triangle (tent) filter; OIIO's "triangle".</summary>
		Bilinear,

		/// <summary>Box filter; OIIO's "box".</summary>
		Box,
	}

	/// <summary>
	/// Port of Bitmap::Rescale: resample the image to the new dimensions in place.
	/// </summary>
	public void Rescale(int newWidth, int newHeight, RescaleFilter filter = RescaleFilter.Bilinear)
	{
		var newData = new byte[newWidth * newHeight * channels];
		if (newData.Length > 0 && data.Length > 0)
		{
			var columnWeights = ComputeResampleWeights(width, newWidth, filter);
			var rowWeights = ComputeResampleWeights(height, newHeight, filter);

			// Horizontal pass: height x newWidth intermediate, kept in double.
			var horizontal = new double[height * newWidth * channels];
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < newWidth; x++)
				{
					var (first, weights) = columnWeights[x];
					for (int c = 0; c < channels; c++)
					{
						double sum = 0;
						for (int i = 0; i < weights.Length; i++)
						{
							sum += weights[i] * data[(y * width + first + i) * channels + c];
						}
						horizontal[(y * newWidth + x) * channels + c] = sum;
					}
				}
			}

			// Vertical pass.
			for (int y = 0; y < newHeight; y++)
			{
				var (first, weights) = rowWeights[y];
				for (int x = 0; x < newWidth; x++)
				{
					for (int c = 0; c < channels; c++)
					{
						double sum = 0;
						for (int i = 0; i < weights.Length; i++)
						{
							sum += weights[i] * horizontal[((first + i) * newWidth + x) * channels + c];
						}
						double rounded = Math.Round(sum, MidpointRounding.AwayFromZero);
						newData[(y * newWidth + x) * channels + c] = (byte)Math.Clamp(rounded, 0, 255);
					}
				}
			}
		}

		width = newWidth;
		height = newHeight;
		data = newData;
		// COLMAP updates the metadata's image spec here, which fails on a bitmap without one.
		Check.NotNull(metaData);
	}

	/// <summary>
	/// Port of Bitmap::Thumbnail: downscale in place so that neither dimension exceeds
	/// <paramref name="maxImageSize"/>, preserving the aspect ratio. Returns the scale factor
	/// applied (1 if the image already fits).
	/// </summary>
	public double Thumbnail(int maxImageSize, RescaleFilter filter = RescaleFilter.Bilinear)
	{
		Check.Gt(maxImageSize, 0);
		if (width <= maxImageSize && height <= maxImageSize)
		{
			return 1.0;
		}
		// Fit the down-sampled version exactly into the max dimensions.
		double scale = (double)maxImageSize / Math.Max(width, height);
		Rescale(
			(int)Math.Round(width * scale, MidpointRounding.AwayFromZero),
			(int)Math.Round(height * scale, MidpointRounding.AwayFromZero),
			filter);
		return scale;
	}

	/// <summary>
	/// Normalized filter weights for each destination index along one axis: the first
	/// contributing source index and the weights of the consecutive source indices from it.
	/// </summary>
	private static (int First, double[] Weights)[] ComputeResampleWeights(int sourceSize, int destSize, RescaleFilter filter)
	{
		double ratio = (double)sourceSize / destSize;
		// Filter widths in destination pixels (triangle 2, box 1), widened to source pixels
		// when downsampling so every source pixel contributes.
		double scale = Math.Max(1.0, ratio);
		double radius = (filter == RescaleFilter.Bilinear ? 1.0 : 0.5) * scale;

		var result = new (int, double[])[destSize];
		for (int d = 0; d < destSize; d++)
		{
			double center = (d + 0.5) * ratio;
			// Source pixels whose centers can fall inside the filter, including virtual ones
			// beyond the border, which fold onto the edge pixel.
			int first = (int)Math.Floor(center - radius - 0.5);
			int last = (int)Math.Ceiling(center + radius - 0.5);
			int foldedFirst = Math.Clamp(first, 0, sourceSize - 1);
			int foldedLast = Math.Clamp(last, 0, sourceSize - 1);
			var weights = new double[foldedLast - foldedFirst + 1];
			double total = 0;
			for (int s = first; s <= last; s++)
			{
				double distance = Math.Abs(s + 0.5 - center) / scale;
				double weight = filter == RescaleFilter.Bilinear
					? Math.Max(0.0, 1.0 - distance)
					: (distance <= 0.5 ? 1.0 : 0.0);
				weights[Math.Clamp(s, 0, sourceSize - 1) - foldedFirst] += weight;
				total += weight;
			}
			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] /= total;
			}
			result[d] = (foldedFirst, weights);
		}
		return result;
	}
}
