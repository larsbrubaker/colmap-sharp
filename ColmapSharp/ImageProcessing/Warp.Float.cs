// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Warp.Float: the row-major float-image helpers of colmap/image/warp.h/.cc -
// ResampleImageBilinear, SmoothImage and DownsampleImage. Part of Warp (Warp.cs); the MVS
// depth and normal maps downsample through these.
//
// Tier A (exact). The arithmetic stays in single precision as in COLMAP. SmoothImage is
// VLFeat's vl_imsmooth_f, already ported for SIFT (Feature/VLFeat/VlImOpv.Smooth.cs).
// One caveat: Apple clang contracts `a * b + c * d` into an FMA by default, so the macOS
// pycolmap wheel may differ in the last float ulp on the interpolation sums; the ported
// tests compare these with exact expected values that are unaffected.
//
// Translation notes:
// - A C++ `const float*` into a larger buffer (normal_map.cc downsamples each channel plane
//   at an offset) becomes an array plus an offset.

using ColmapSharp.Feature.VLFeat;
using ColmapSharp.Util;

namespace ColmapSharp.ImageProcessing;

public static partial class Warp
{
	/// <summary>std::numeric_limits&lt;float&gt;::epsilon() (not float.Epsilon, the smallest subnormal).</summary>
	private const float FloatEpsilon = 1.1920929E-07f;

	/// <summary>
	/// Port of colmap::ResampleImageBilinear: resample the rows x cols row-major image at
	/// <paramref name="dataOffset"/> to new_rows x new_cols with bilinear interpolation (pixel
	/// centers aligned, zero outside the image), written at <paramref name="resampledOffset"/>.
	/// </summary>
	public static void ResampleImageBilinear(
		float[] data,
		int rows,
		int cols,
		int newRows,
		int newCols,
		float[] resampled,
		int dataOffset = 0,
		int resampledOffset = 0)
	{
		Check.Gt(rows, 0);
		Check.Gt(cols, 0);
		Check.Gt(newRows, 0);
		Check.Gt(newCols, 0);

		ReadOnlySpan<float> source = data.AsSpan(dataOffset, rows * cols);
		Span<float> target = resampled.AsSpan(resampledOffset, newRows * newCols);

		float scaleR = (float)rows / (float)newRows;
		float scaleC = (float)cols / (float)newCols;

		for (int r = 0; r < newRows; r++)
		{
			float rI = (r + 0.5f) * scaleR - 0.5f;
			int rIMin = (int)MathF.Floor(rI);
			int rIMax = rIMin + 1;
			float dRMin = rI - rIMin;
			float dRMax = rIMax - rI;

			for (int c = 0; c < newCols; c++)
			{
				float cI = (c + 0.5f) * scaleC - 0.5f;
				int cIMin = (int)MathF.Floor(cI);
				int cIMax = cIMin + 1;
				float dCMin = cI - cIMin;
				float dCMax = cIMax - cI;

				// Interpolation in column direction.
				float value1 =
					dCMax * GetPixelConstantBorder(source, rows, cols, rIMin, cIMin) +
					dCMin * GetPixelConstantBorder(source, rows, cols, rIMin, cIMax);
				float value2 =
					dCMax * GetPixelConstantBorder(source, rows, cols, rIMax, cIMin) +
					dCMin * GetPixelConstantBorder(source, rows, cols, rIMax, cIMax);

				// Interpolation in row direction.
				target[r * newCols + c] = dRMax * value1 + dRMin * value2;
			}
		}
	}

	/// <summary>
	/// Port of colmap::SmoothImage: Gaussian smoothing of the rows x cols row-major image with
	/// standard deviations <paramref name="sigmaR"/> (along rows) and <paramref name="sigmaC"/>
	/// (along columns), borders padded by continuity (vl_imsmooth_f).
	/// </summary>
	public static void SmoothImage(
		float[] data,
		int rows,
		int cols,
		float sigmaR,
		float sigmaC,
		float[] smoothed,
		int dataOffset = 0,
		int smoothedOffset = 0)
	{
		Check.Gt(rows, 0);
		Check.Gt(cols, 0);
		Check.Gt(sigmaR, 0);
		Check.Gt(sigmaC, 0);
		VlImOpv.ImSmoothF(smoothed, smoothedOffset, cols, data, dataOffset, cols, rows, cols, sigmaC, sigmaR);
	}

	/// <summary>
	/// Port of colmap::DownsampleImage: smooth with a Gaussian sized to the scale change, then
	/// resample bilinearly to new_rows x new_cols (no larger than the input).
	/// </summary>
	public static void DownsampleImage(
		float[] data,
		int rows,
		int cols,
		int newRows,
		int newCols,
		float[] downsampled,
		int dataOffset = 0,
		int downsampledOffset = 0)
	{
		Check.Le(newRows, rows);
		Check.Le(newCols, cols);
		Check.Gt(rows, 0);
		Check.Gt(cols, 0);
		Check.Gt(newRows, 0);
		Check.Gt(newCols, 0);

		float scaleC = (float)cols / (float)newCols;
		float scaleR = (float)rows / (float)newRows;

		const float kSigmaScale = 0.5f;
		// std::max(a, b) is (a < b) ? b : a.
		float sigmaC = FloatEpsilon < kSigmaScale * (scaleC - 1) ? kSigmaScale * (scaleC - 1) : FloatEpsilon;
		float sigmaR = FloatEpsilon < kSigmaScale * (scaleR - 1) ? kSigmaScale * (scaleR - 1) : FloatEpsilon;

		var smoothed = new float[rows * cols];
		SmoothImage(data, rows, cols, sigmaR, sigmaC, smoothed, dataOffset);

		ResampleImageBilinear(smoothed, rows, cols, newRows, newCols, downsampled, 0, downsampledOffset);
	}

	/// <summary>Port of the anonymous GetPixelConstantBorder: zero outside the image.</summary>
	private static float GetPixelConstantBorder(ReadOnlySpan<float> data, int rows, int cols, int row, int col)
	{
		if (row >= 0 && col >= 0 && row < rows && col < cols)
		{
			return data[row * cols + col];
		}

		return 0;
	}
}
