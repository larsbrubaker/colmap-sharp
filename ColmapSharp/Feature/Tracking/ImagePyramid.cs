// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ImagePyramid: the float greyscale image pyramid (with per-level gradients) that the KLT tracker
// in this folder works on (docs/QUALITY_PLAN.md stage 2a). Not a COLMAP port; COLMAP has no
// feature tracker. Written from the papers only (no code was read):
// - J.-Y. Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature Tracker: Description of
//   the Algorithm", Intel Corporation, 2000 (the 1/16 [1 4 6 4 1] low-pass and decimation).
// - H. Scharr, "Optimal Operators in Digital Image Processing", PhD thesis, 2000 (the 3x3
//   [3 10 3] x [-1 0 1] derivative, used here divided by 32 so it is a per-pixel derivative).
//
// Coordinates: every array here is indexed by pixel, so pixel (i, j) has index coordinates
// (i, j). The tracker's public API uses COLMAP's continuous image coordinates, where the
// center of pixel (i, j) is (i + 0.5, j + 0.5); PyramidalLucasKanade converts between the two.
// Level L+1 pixel j is the low-passed level-L pixel 2j, so in index coordinates the levels are
// related by a pure scale of 2 (u_{L+1} = u_L / 2) - the reason the tracker works in index
// coordinates internally.
//
// Borders reflect without repeating the edge pixel (... 2 1 | 0 1 2 ... ), for the low-pass and
// for the gradients. Determinism: all loops are sequential, scalar and FMA-free.

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Feature.Tracking;

/// <summary>One pyramid level: intensity (0..255) and its x and y derivatives, row-major.</summary>
public sealed class PyramidLevel
{
	internal PyramidLevel(int width, int height, float[] intensity)
	{
		Width = width;
		Height = height;
		Intensity = intensity;
		(GradX, GradY) = ScharrGradients(width, height, intensity);
	}

	/// <summary>Width in pixels.</summary>
	public int Width { get; }

	/// <summary>Height in pixels.</summary>
	public int Height { get; }

	/// <summary>Grey level per pixel (0..255), row-major.</summary>
	public float[] Intensity { get; }

	/// <summary>d(Intensity)/dx per pixel (Scharr / 32).</summary>
	public float[] GradX { get; }

	/// <summary>d(Intensity)/dy per pixel (Scharr / 32).</summary>
	public float[] GradY { get; }

	/// <summary>
	/// Bilinear sample of <paramref name="image"/> (one of this level's arrays) at index
	/// coordinates (x, y); outside the image the nearest edge value is used.
	/// </summary>
	public float Sample(float[] image, double x, double y)
	{
		x = Math.Clamp(x, 0, Width - 1);
		y = Math.Clamp(y, 0, Height - 1);
		int x0 = Math.Min((int)x, Width - 2 < 0 ? 0 : Width - 2);
		int y0 = Math.Min((int)y, Height - 2 < 0 ? 0 : Height - 2);
		int x1 = Math.Min(x0 + 1, Width - 1);
		int y1 = Math.Min(y0 + 1, Height - 1);
		float fx = (float)(x - x0);
		float fy = (float)(y - y0);
		float top = image[y0 * Width + x0] * (1 - fx) + image[y0 * Width + x1] * fx;
		float bottom = image[y1 * Width + x0] * (1 - fx) + image[y1 * Width + x1] * fx;
		return top * (1 - fy) + bottom * fy;
	}

	internal static int Reflect(int i, int n)
	{
		if (n == 1)
		{
			return 0;
		}

		while (i < 0 || i >= n)
		{
			i = i < 0 ? -i : 2 * (n - 1) - i;
		}

		return i;
	}

	private static (float[] Gx, float[] Gy) ScharrGradients(int width, int height, float[] img)
	{
		var gx = new float[width * height];
		var gy = new float[width * height];
		for (int y = 0; y < height; ++y)
		{
			int ym = Reflect(y - 1, height) * width;
			int y0 = y * width;
			int yp = Reflect(y + 1, height) * width;
			for (int x = 0; x < width; ++x)
			{
				int xm = Reflect(x - 1, width);
				int xp = Reflect(x + 1, width);
				float dxm = img[ym + xp] - img[ym + xm];
				float dx0 = img[y0 + xp] - img[y0 + xm];
				float dxp = img[yp + xp] - img[yp + xm];
				gx[y0 + x] = (3 * dxm + 10 * dx0 + 3 * dxp) / 32f;
				float dym = img[yp + xm] - img[ym + xm];
				float dy0 = img[yp + x] - img[ym + x];
				float dyp = img[yp + xp] - img[ym + xp];
				gy[y0 + x] = (3 * dym + 10 * dy0 + 3 * dyp) / 32f;
			}
		}

		return (gx, gy);
	}
}

/// <summary>
/// A greyscale float pyramid of one frame: level 0 is the frame, each further level half the
/// size (rounded up) after the 5-tap Gaussian low-pass. See the file header for coordinates.
/// </summary>
public sealed class ImagePyramid
{
	private ImagePyramid(PyramidLevel[] levels) => Levels = levels;

	/// <summary>The levels, finest (the frame itself) first.</summary>
	public IReadOnlyList<PyramidLevel> Levels { get; }

	/// <summary>Width of level 0.</summary>
	public int Width => Levels[0].Width;

	/// <summary>Height of level 0.</summary>
	public int Height => Levels[0].Height;

	/// <summary>
	/// Builds the pyramid of <paramref name="frame"/> (8-bit grey or RGB; RGB goes through
	/// COLMAP's Rec. 709 luma) with <paramref name="numLevels"/> levels above level 0. Levels
	/// stop early once an image side would drop below 8 pixels.
	/// </summary>
	public static ImagePyramid Build(Bitmap frame, int numLevels)
	{
		Check.That(!frame.IsEmpty);
		Check.That(numLevels >= 0);
		return Build(frame.Width, frame.Height, ToGrey(frame), numLevels);
	}

	/// <summary>Builds the pyramid of a row-major grey image (values 0..255).</summary>
	public static ImagePyramid Build(int width, int height, float[] grey, int numLevels)
	{
		Check.Eq(grey.Length, width * height);
		var levels = new List<PyramidLevel> { new(width, height, grey) };
		for (int l = 1; l <= numLevels; ++l)
		{
			PyramidLevel prev = levels[^1];
			int w = (prev.Width + 1) / 2;
			int h = (prev.Height + 1) / 2;
			if (w < 8 || h < 8)
			{
				break;
			}

			levels.Add(new PyramidLevel(w, h, Downsample(prev.Width, prev.Height, prev.Intensity, w, h)));
		}

		return new ImagePyramid([.. levels]);
	}

	/// <summary>
	/// Grey levels of an 8-bit grey or RGB bitmap as floats, RGB weighted like
	/// Bitmap.CloneAsGrey but without its rounding to bytes.
	/// </summary>
	public static float[] ToGrey(Bitmap frame)
	{
		byte[] data = frame.RowMajorData;
		var grey = new float[frame.Width * frame.Height];
		if (frame.IsGrey)
		{
			for (int i = 0; i < grey.Length; ++i)
			{
				grey[i] = data[i];
			}
		}
		else
		{
			Check.That(frame.IsRGB);
			for (int i = 0; i < grey.Length; ++i)
			{
				grey[i] = .2126f * data[3 * i] + .7152f * data[3 * i + 1] + .0722f * data[3 * i + 2];
			}
		}

		return grey;
	}

	// Separable [1 4 6 4 1] / 16 low-pass (Bouguet 2000, eq. 2), then keep every other pixel.
	private static float[] Downsample(int width, int height, float[] src, int w, int h)
	{
		// Horizontal pass only at the kept columns, over every row.
		var tmp = new float[w * height];
		for (int y = 0; y < height; ++y)
		{
			int row = y * width;
			for (int x = 0; x < w; ++x)
			{
				int c = 2 * x;
				tmp[y * w + x] = (src[row + PyramidLevel.Reflect(c - 2, width)]
					+ 4 * src[row + PyramidLevel.Reflect(c - 1, width)]
					+ 6 * src[row + c]
					+ 4 * src[row + PyramidLevel.Reflect(c + 1, width)]
					+ src[row + PyramidLevel.Reflect(c + 2, width)]) / 16f;
			}
		}

		var dst = new float[w * h];
		for (int y = 0; y < h; ++y)
		{
			int r = 2 * y;
			int rm2 = PyramidLevel.Reflect(r - 2, height) * w;
			int rm1 = PyramidLevel.Reflect(r - 1, height) * w;
			int r0 = r * w;
			int rp1 = PyramidLevel.Reflect(r + 1, height) * w;
			int rp2 = PyramidLevel.Reflect(r + 2, height) * w;
			for (int x = 0; x < w; ++x)
			{
				dst[y * w + x] = (tmp[rm2 + x] + 4 * tmp[rm1 + x] + 6 * tmp[r0 + x]
					+ 4 * tmp[rp1 + x] + tmp[rp2 + x]) / 16f;
			}
		}

		return dst;
	}
}
