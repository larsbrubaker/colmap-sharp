// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlImOpv.Smooth: vl_imsmooth_f (with _vl_new_gaussian_fitler_f), vl_imgradient_f and
// vl_imgradient_polar_f from thirdparty/VLFeat/imopv.c. The covariant detector smooths its
// scale space and orientation patches with the first (VlScaleSpace.cs, VlCovDet.Shape.cs),
// and takes second-moment and orientation gradients with the others; COLMAP's covariant
// extractor (Feature/CovariantSift.cs) runs the polar gradient on each descriptor patch.
// Part of VlImOpv (VlImOpv.cs has the column convolution these build on).
//
// Tier A (exact). Pixels and filter taps are float; the filter's mass accumulates in float from
// double taps, as the C's `T mass; mass += g + g` does. The gradients are VLFeat's per-pixel
// one-sided (border) / central (inside) differences; the traversal is written as a plain
// double loop because each output pixel depends only on its input neighbours.

namespace ColmapSharp.Feature.VLFeat;

/// <content>Gaussian smoothing and image gradients.</content>
public static partial class VlImOpv
{
	/// <summary>
	/// Port of vl_imsmooth_f: separable Gaussian smoothing (standard deviations
	/// <paramref name="sigmax"/>, <paramref name="sigmay"/>) with padding by continuity.
	/// <paramref name="smoothed"/> may be the same buffer as <paramref name="image"/>.
	/// </summary>
	public static void ImSmoothF(
		float[] smoothed,
		int smoothedOffset,
		int smoothedStride,
		float[] image,
		int imageOffset,
		int width,
		int height,
		int stride,
		double sigmax,
		double sigmay)
	{
		float[] filterx = NewGaussianFilter(sigmax);
		float[] filtery = sigmax == sigmay ? filterx : NewGaussianFilter(sigmay);
		int sizex = filterx.Length;
		int sizey = filtery.Length;
		var buffer = new float[width * height];

		ConvColVF(
			buffer, 0, height,
			image, imageOffset, width, height, stride,
			filtery, 0, -(sizey - 1) / 2, (sizey - 1) / 2,
			1, PadByContinuity | Transpose);

		ConvColVF(
			smoothed, smoothedOffset, smoothedStride,
			buffer, 0, height, width, height,
			filterx, 0, -(sizex - 1) / 2, (sizex - 1) / 2,
			1, PadByContinuity | Transpose);
	}

	/// <summary>
	/// Port of vl_imgradient_f: x and y derivatives, one-sided on the border and central
	/// inside. Output pixel (x, y) goes to <c>y * gradHeightStride + x * gradWidthStride</c>
	/// past each offset.
	/// </summary>
	public static void ImGradientF(
		float[] xGradient,
		int xOffset,
		float[] yGradient,
		int yOffset,
		int gradWidthStride,
		int gradHeightStride,
		float[] image,
		int imageOffset,
		int imageWidth,
		int imageHeight,
		int imageStride)
	{
		for (int y = 0; y < imageHeight; ++y)
		{
			for (int x = 0; x < imageWidth; ++x)
			{
				Gradient(image, imageOffset, imageWidth, imageHeight, imageStride, x, y, out float gx, out float gy);
				int o = (y * gradHeightStride) + (x * gradWidthStride);
				xGradient[xOffset + o] = gx;
				yGradient[yOffset + o] = gy;
			}
		}
	}

	/// <summary>
	/// Port of vl_imgradient_polar_f: the gradient's modulus (vl_fast_sqrt_f) and angle
	/// (vl_fast_atan2_f, wrapped to [0, 2*pi]), laid out like <see cref="ImGradientF"/>.
	/// </summary>
	public static void ImGradientPolarF(
		float[] gradientModulus,
		int modulusOffset,
		float[] gradientAngle,
		int angleOffset,
		int gradientHorizontalStride,
		int gradHeightStride,
		float[] image,
		int imageOffset,
		int imageWidth,
		int imageHeight,
		int imageStride)
	{
		for (int y = 0; y < imageHeight; ++y)
		{
			for (int x = 0; x < imageWidth; ++x)
			{
				Gradient(image, imageOffset, imageWidth, imageHeight, imageStride, x, y, out float gx, out float gy);
				int o = (y * gradHeightStride) + (x * gradientHorizontalStride);
				gradientModulus[modulusOffset + o] = VlMathOp.FastSqrtF((gx * gx) + (gy * gy));

				// `vl_fast_atan2_f(gy, gx) + 2*VL_PI` is a double, rounded to float by the call.
				gradientAngle[angleOffset + o] = VlMathOp.Mod2PiF((float)(VlMathOp.FastAtan2F(gy, gx) + (2 * VlMathOp.Pi)));
			}
		}
	}

	// Port of _vl_new_gaussian_fitler_f: a normalized float Gaussian of half-width
	// ceil(3 * sigma).
	private static float[] NewGaussianFilter(double sigma)
	{
		int width = (int)VlMathOp.CeilD(sigma * 3.0);
		var filter = new float[(2 * width) + 1];
		float mass = 1.0f;
		filter[width] = 1.0f;
		for (int i = 1; i <= width; ++i)
		{
			double x = (double)i / sigma;
			double g = Math.Exp(-0.5 * x * x);
			mass = (float)(mass + (g + g));
			filter[width - i] = (float)g;
			filter[width + i] = (float)g;
		}

		for (int i = 0; i < filter.Length; ++i)
		{
			filter[i] /= mass;
		}

		return filter;
	}

	// The difference VLFeat's gradient functions take at (x, y): (next - previous) / 2 inside,
	// one-sided on the first and last row/column. "0.5 * (a - b)" promotes the float
	// difference to double; halving is exact, so the float result is the same.
	private static void Gradient(float[] image, int offset, int w, int h, int stride, int x, int y, out float gx, out float gy)
	{
		int p = offset + (y * stride) + x;
		if (x == 0)
		{
			gx = image[p + 1] - image[p];
		}
		else if (x == w - 1)
		{
			gx = image[p] - image[p - 1];
		}
		else
		{
			gx = (float)(0.5 * (image[p + 1] - image[p - 1]));
		}

		if (y == 0)
		{
			gy = image[p + stride] - image[p];
		}
		else if (y == h - 1)
		{
			gy = image[p] - image[p - stride];
		}
		else
		{
			gy = (float)(0.5 * (image[p + stride] - image[p - stride]));
		}
	}
}
