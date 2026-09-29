// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LabImage: a frame box-downsampled to the segmentation working size and converted to CIE Lab,
// plus the upsampling of a working-resolution mask back to the frame's size. Not a COLMAP port.
// Sources: the sRGB transfer function and D65 matrix of IEC 61966-2-1 (1999), and the CIE 1976
// L*a*b* definition (CIE 15:2004). Averaging happens in linear light, before conversion, so a
// block that straddles an edge gets the colour a smaller camera would have seen. Used by
// SilhouetteSegmenter; the grid it defines is the one BinaryMorphology and GrabCutRefiner use.

using ColmapSharp.Sensor;

namespace ColmapSharp.Segmentation;

/// <summary>A frame at working resolution in CIE Lab (D65), one float plane per channel.</summary>
internal sealed class LabImage
{
	// sRGB byte to linear light, built once with the IEC 61966-2-1 transfer function.
	private static readonly double[] SrgbToLinear = BuildSrgbTable();

	public LabImage(int width, int height, int factor, int sourceWidth, int sourceHeight)
	{
		Factor = factor;
		Width = width;
		Height = height;
		SourceWidth = sourceWidth;
		SourceHeight = sourceHeight;
		L = new float[width * height];
		A = new float[width * height];
		B = new float[width * height];
	}

	/// <summary>The downsampling factor: working pixel (x, y) averages the source block at (x, y) * Factor.</summary>
	public int Factor { get; }

	public int Width { get; }

	public int Height { get; }

	/// <summary>The size of the frame this was made from, which the mask is upsampled to.</summary>
	public int SourceWidth { get; }

	public int SourceHeight { get; }

	public float[] L { get; }

	public float[] A { get; }

	public float[] B { get; }

	/// <summary>
	/// Box-downsamples <paramref name="bitmap"/> by the integer factor that brings its longest
	/// side to at most <paramref name="workingSize"/>, and converts it to Lab.
	/// </summary>
	public static LabImage FromBitmap(Bitmap bitmap, int workingSize)
	{
		int w = bitmap.Width;
		int h = bitmap.Height;
		int factor = Math.Max(1, (Math.Max(w, h) + workingSize - 1) / workingSize);
		int ww = Math.Max(1, w / factor);
		int wh = Math.Max(1, h / factor);
		int fx = Math.Min(factor, w);
		int fy = Math.Min(factor, h);
		var image = new LabImage(ww, wh, factor, w, h);
		byte[] data = bitmap.RowMajorData;
		int channels = bitmap.Channels;
		double inverseCount = 1.0 / (fx * fy);
		for (int y = 0; y < wh; ++y)
		{
			for (int x = 0; x < ww; ++x)
			{
				double r = 0.0, g = 0.0, b = 0.0;
				for (int dy = 0; dy < fy; ++dy)
				{
					int row = (y * factor + dy) * w;
					for (int dx = 0; dx < fx; ++dx)
					{
						int pixel = (row + x * factor + dx) * channels;
						if (channels >= 3)
						{
							r += SrgbToLinear[data[pixel]];
							g += SrgbToLinear[data[pixel + 1]];
							b += SrgbToLinear[data[pixel + 2]];
						}
						else
						{
							double v = SrgbToLinear[data[pixel]];
							r += v;
							g += v;
							b += v;
						}
					}
				}

				int i = y * ww + x;
				LinearToLab(r * inverseCount, g * inverseCount, b * inverseCount, out image.L[i], out image.A[i], out image.B[i]);
			}
		}

		return image;
	}

	/// <summary>Upsamples a working-resolution mask of this image to its source size.</summary>
	public Bitmap UpsampleMask(bool[] mask) => UpsampleMask(mask, Width, Height, Factor, SourceWidth, SourceHeight);

	/// <summary>
	/// Upsamples a working-resolution mask to the source size by bilinear interpolation of the
	/// 0/1 labels, thresholded at one half. Source pixel x lies at working coordinate
	/// (x + 0.5) / factor - 0.5, the inverse of FromBitmap's block average (source columns past
	/// the last whole block clamp to the last working column). The result is a grey bitmap,
	/// 255 = keep, 0 = drop (the mask convention of ImageReader, MaskFeatures and StereoFusion).
	/// </summary>
	public static Bitmap UpsampleMask(bool[] mask, int width, int height, int factor, int sourceWidth, int sourceHeight)
	{
		var result = new Bitmap(sourceWidth, sourceHeight, asRgb: false);
		byte[] output = result.RowMajorData;
		double inverseFactor = 1.0 / factor;
		for (int y = 0; y < sourceHeight; ++y)
		{
			double sy = Math.Clamp((y + 0.5) * inverseFactor - 0.5, 0.0, height - 1);
			int y0 = (int)sy;
			int y1 = Math.Min(y0 + 1, height - 1);
			double ty = sy - y0;
			for (int x = 0; x < sourceWidth; ++x)
			{
				double sx = Math.Clamp((x + 0.5) * inverseFactor - 0.5, 0.0, width - 1);
				int x0 = (int)sx;
				int x1 = Math.Min(x0 + 1, width - 1);
				double tx = sx - x0;
				double top = (mask[y0 * width + x0] ? 1.0 - tx : 0.0) + (mask[y0 * width + x1] ? tx : 0.0);
				double bottom = (mask[y1 * width + x0] ? 1.0 - tx : 0.0) + (mask[y1 * width + x1] ? tx : 0.0);
				double value = top * (1.0 - ty) + bottom * ty;
				output[y * sourceWidth + x] = value >= 0.5 ? (byte)255 : (byte)0;
			}
		}

		return result;
	}

	private static void LinearToLab(double r, double g, double b, out float l, out float a, out float bb)
	{
		// IEC 61966-2-1 linear sRGB to XYZ, divided by the D65 white point.
		double x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
		double y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
		double z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
		double fx = LabF(x);
		double fy = LabF(y);
		double fz = LabF(z);
		l = (float)(116.0 * fy - 16.0);
		a = (float)(500.0 * (fx - fy));
		bb = (float)(200.0 * (fy - fz));
	}

	private static double LabF(double t)
	{
		const double Delta = 6.0 / 29.0;
		return t > Delta * Delta * Delta ? Math.Cbrt(t) : t / (3.0 * Delta * Delta) + 4.0 / 29.0;
	}

	private static double[] BuildSrgbTable()
	{
		var table = new double[256];
		for (int i = 0; i < 256; ++i)
		{
			double c = i / 255.0;
			table[i] = c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
		}

		return table;
	}
}
