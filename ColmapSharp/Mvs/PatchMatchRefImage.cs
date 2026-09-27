// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchRefImage: colmap/mvs/gpu_mat_ref_image.h and .cu - the reference image of a
// PatchMatch problem with, per pixel, the bilaterally weighted mean and mean square of its
// window (the reference half of the NCC, computed once instead of per hypothesis), and
// BilateralWeightComputer, the weight both this prefilter and the NCC use. PatchMatch keeps
// these as Mat planes and rotates them with Mat.Rotate (Mat.Transforms.cs) after every
// sweep. Tests: ColmapSharp.Tests/Mvs/PatchMatchInputsTests.cs (C#-only; COLMAP has no
// test for it).
//
// COLMAP's float arithmetic in COLMAP's operation order (PatchMatch is Tier C); the exponential is MathF.Exp
// rather than CUDA's expf, so the sums can differ from a GPU run in the last bits
// (docs/CPP_DIVERGENCES.md, entry 96).
//
// Texture reads: COLMAP samples the image through a point-filtered CUDA texture with border
// addressing and normalized-float reads, so a pixel outside the image is 0 and a pixel
// inside is its byte value / 255. Integer coordinates select exactly that texel.
// Parallel over rows (at most numThreads, COLMAP's num_threads); each row writes only its
// own outputs.

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::BilateralWeightComputer: exp(-spatial distance² / (2 sigma_spatial²)
/// - color distance² / (2 sigma_color²)).
/// </summary>
public readonly struct BilateralWeightComputer
{
	private readonly float spatialNormalization;
	private readonly float colorNormalization;

	/// <summary>The weight for the given spatial and color sigmas.</summary>
	public BilateralWeightComputer(float sigmaSpatial, float sigmaColor)
	{
		spatialNormalization = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
		colorNormalization = 1.0f / (2.0f * sigmaColor * sigmaColor);
	}

	// 1 / (2 sigma_spatial²) and 1 / (2 sigma_color²), which the GPU path (PatchMatchGpu.cs)
	// packs into its problem uniform.
	internal float SpatialNormalization => spatialNormalization;

	internal float ColorNormalization => colorNormalization;

	// The weight from its two derived values (FromNormalizations).
	private BilateralWeightComputer((float Spatial, float Color) normalizations)
	{
		spatialNormalization = normalizations.Spatial;
		colorNormalization = normalizations.Color;
	}

	/// <summary>
	/// The weight from the values the public constructor derives: 1 / (2 sigma_spatial²) and
	/// 1 / (2 sigma_color²) (the GPU problem uniform carries these).
	/// </summary>
	internal static BilateralWeightComputer FromNormalizations(float spatialNormalization, float colorNormalization)
		=> new((spatialNormalization, colorNormalization));

	/// <summary>The weight of a pixel (rowDiff, colDiff) away with color2, relative to color1.</summary>
	public float Compute(float rowDiff, float colDiff, float color1, float color2)
	{
		float spatialDistSquared = rowDiff * rowDiff + colDiff * colDiff;
		float colorDist = color1 - color2;
		return MathF.Exp(-spatialDistSquared * spatialNormalization - colorDist * colorDist * colorNormalization);
	}
}

/// <summary>
/// Port of colmap::mvs::GpuMatRefImage: the reference image and its bilaterally weighted
/// window sums.
/// </summary>
public sealed class PatchMatchRefImage
{
	/// <summary>An unfilled width x height reference image.</summary>
	public PatchMatchRefImage(int width, int height)
	{
		Image = new Mat<byte>(width, height, 1);
		SumImage = new Mat<float>(width, height, 1);
		SquaredSumImage = new Mat<float>(width, height, 1);
	}

	/// <summary>The grey reference image.</summary>
	public Mat<byte> Image { get; }

	/// <summary>Per pixel, the bilaterally weighted mean of its window, in [0, 1].</summary>
	public Mat<float> SumImage { get; }

	/// <summary>Per pixel, the bilaterally weighted mean of its window's squared values.</summary>
	public Mat<float> SquaredSumImage { get; }

	/// <summary>
	/// Fills the images from the row-major grey pixels <paramref name="imageData"/>, with a
	/// (2 windowRadius + 1)² window sampled every windowStep pixels, on at most
	/// <paramref name="numThreads"/> threads (-1: all cores). Port of
	/// GpuMatRefImage::Filter (FilterKernel).
	/// </summary>
	public void Filter(ReadOnlySpan<byte> imageData, int windowRadius, int windowStep, float sigmaSpatial, float sigmaColor, int numThreads = -1)
	{
		int width = Image.GetWidth();
		int height = Image.GetHeight();
		Util.Check.Eq(imageData.Length, width * height);
		Util.Check.Gt(windowStep, 0);

		byte[] input = imageData.ToArray();
		byte[] image = Image.Data;
		float[] sumImage = SumImage.Data;
		float[] squaredSumImage = SquaredSumImage.Data;
		var bilateralWeightComputer = new BilateralWeightComputer(sigmaSpatial, sigmaColor);

		Parallel.For(0, height, Mat<byte>.ParallelOptionsFor(numThreads), row =>
		{
			for (int col = 0; col < width; ++col)
			{
				float centerColor = Texel(input, width, height, row, col);

				float colorSum = 0.0f;
				float colorSquaredSum = 0.0f;
				float bilateralWeightSum = 0.0f;

				for (int windowRow = -windowRadius; windowRow <= windowRadius; windowRow += windowStep)
				{
					for (int windowCol = -windowRadius; windowCol <= windowRadius; windowCol += windowStep)
					{
						float color = Texel(input, width, height, row + windowRow, col + windowCol);
						float bilateralWeight = bilateralWeightComputer.Compute(windowRow, windowCol, centerColor, color);
						colorSum += bilateralWeight * color;
						colorSquaredSum += bilateralWeight * color * color;
						bilateralWeightSum += bilateralWeight;
					}
				}

				colorSum /= bilateralWeightSum;
				colorSquaredSum /= bilateralWeightSum;

				int idx = row * width + col;

				// static_cast<uint8_t>(255.0f * center_color) truncates, as COLMAP does.
				image[idx] = (byte)(255.0f * centerColor);
				sumImage[idx] = colorSum;
				squaredSumImage[idx] = colorSquaredSum;
			}
		});
	}

	/// <summary>
	/// A point-filtered, border-addressed, normalized-float texture read: the byte at
	/// (row, col) / 255, or 0 outside the image.
	/// </summary>
	internal static float Texel(byte[] image, int width, int height, int row, int col)
	{
		if ((uint)row >= (uint)height || (uint)col >= (uint)width)
		{
			return 0.0f;
		}

		return image[row * width + col] / 255.0f;
	}
}
