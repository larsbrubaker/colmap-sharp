// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GpuMatTests: colmap/mvs/gpu_mat_test.cu ported 1:1 (GpuMat_Name), testing the CPU
// replacements of GpuMat's operations in ColmapSharp/Mvs/Mat.Transforms.cs. COLMAP's
// GpuMat is a CUDA buffer; here the same operations run on Mat<T>, so the cases run without
// a GPU. GpuMatPRNG / FillWithRandomNumbers (cuRAND) becomes a fill from PatchMatchRandom
// (docs/CPP_DIVERGENCES.md, entry 86): the values only need to differ, since each case
// compares an input with its transformed copy. The expected index formulas, including the
// rotation test's rounded trigonometry, are COLMAP's. Tier A.

using System.Numerics;

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class GpuMatTests
{
	// FillWithRandomNumbers(T(0.0), T(100.0), prng): curand_uniform * (max - min) + min,
	// converted to T.
	private static void FillWithRandomNumbers<T>(Mat<T> array, T minValue, T maxValue)
		where T : unmanaged, INumber<T>
	{
		float min = float.CreateTruncating(minValue);
		float max = float.CreateTruncating(maxValue);
		T[] data = array.Data;
		int width = array.GetWidth();
		int height = array.GetHeight();
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				var random = new PatchMatchRandom(PatchMatchRandom.DefaultSeed, row, col, PatchMatchRandom.InitDepthPhase);
				for (int slice = 0; slice < array.GetDepth(); ++slice)
				{
					data[(slice * height + row) * width + col] = T.CreateTruncating(random.NextUniform() * (max - min) + min);
				}
			}
		}
	}

	[Test]
	public async Task GpuMat_FillWithVector()
	{
		var array = new Mat<float>(100, 100, 2);
		float[] vector = [1.0f, 2.0f];
		array.FillWithVector(vector);

		float[] arrayHost = array.Data;
		for (int r = 0; r < 100; ++r)
		{
			for (int c = 0; c < 100; ++c)
			{
				await Assert.That(arrayHost[0 * 100 * 100 + r * 100 + c]).IsEqualTo(1.0f);
				await Assert.That(arrayHost[1 * 100 * 100 + r * 100 + c]).IsEqualTo(2.0f);
			}
		}
	}

	private static int TestTransposeImage<T>(int width, int height, int depth)
		where T : unmanaged, INumber<T>
	{
		var array = new Mat<T>(width, height, depth);
		FillWithRandomNumbers(array, T.CreateTruncating(0.0), T.CreateTruncating(100.0));

		var arrayTransposed = new Mat<T>(height, width, depth);
		array.Transpose(arrayTransposed);

		T[] arrayHost = array.Data;
		T[] arrayTransposedHost = arrayTransposed.Data;
		int failures = 0;
		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				for (int d = 0; d < depth; ++d)
				{
					if (arrayHost[d * width * height + r * width + c] != arrayTransposedHost[d * width * height + c * height + r])
					{
						failures++;
					}
				}
			}
		}

		return failures;
	}

	[Test]
	public async Task GpuMat_Transpose()
	{
		for (int w = 1; w <= 5; ++w)
		{
			for (int h = 1; h <= 5; ++h)
			{
				for (int d = 1; d <= 3; ++d)
				{
					int width = 20 * w;
					int height = 20 * h;
					await Assert.That(TestTransposeImage<sbyte>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestTransposeImage<short>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestTransposeImage<int>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestTransposeImage<long>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestTransposeImage<float>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestTransposeImage<double>(width, height, d)).IsEqualTo(0);
				}
			}
		}
	}

	private static int TestFlipHorizontalImage<T>(int width, int height, int depth)
		where T : unmanaged, INumber<T>
	{
		var array = new Mat<T>(width, height, depth);
		FillWithRandomNumbers(array, T.CreateTruncating(0.0), T.CreateTruncating(100.0));

		var arrayFlipped = new Mat<T>(width, height, depth);
		array.FlipHorizontal(arrayFlipped);

		T[] arrayHost = array.Data;
		T[] arrayFlippedHost = arrayFlipped.Data;
		int failures = 0;
		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				for (int d = 0; d < depth; ++d)
				{
					if (arrayHost[d * width * height + r * width + c] != arrayFlippedHost[d * width * height + r * width + width - 1 - c])
					{
						failures++;
					}
				}
			}
		}

		return failures;
	}

	[Test]
	public async Task GpuMat_FlipHorizontal()
	{
		for (int w = 1; w <= 5; ++w)
		{
			for (int h = 1; h <= 5; ++h)
			{
				for (int d = 1; d <= 3; ++d)
				{
					int width = 20 * w;
					int height = 20 * h;
					await Assert.That(TestFlipHorizontalImage<sbyte>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestFlipHorizontalImage<short>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestFlipHorizontalImage<int>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestFlipHorizontalImage<long>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestFlipHorizontalImage<float>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestFlipHorizontalImage<double>(width, height, d)).IsEqualTo(0);
				}
			}
		}
	}

	private static int TestRotateImage<T>(int width, int height, int depth)
		where T : unmanaged, INumber<T>
	{
		var array = new Mat<T>(width, height, depth);
		FillWithRandomNumbers(array, T.CreateTruncating(0.0), T.CreateTruncating(100.0));

		var arrayRotated = new Mat<T>(height, width, depth);
		array.Rotate(arrayRotated);

		T[] arrayHost = array.Data;
		T[] arrayRotatedHost = arrayRotated.Data;

		double arrayCenterH = width / 2.0 - 0.5;
		double arrayCenterV = height / 2.0 - 0.5;
		double angle = -Math.PI / 2;
		int failures = 0;
		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				for (int d = 0; d < depth; ++d)
				{
					// std::round rounds halves away from zero.
					long rotc = (long)Math.Round(
						Math.Cos(angle) * (c - arrayCenterH) - Math.Sin(angle) * (r - arrayCenterV) + arrayCenterV,
						MidpointRounding.AwayFromZero);
					long rotr = (long)Math.Round(
						Math.Sin(angle) * (c - arrayCenterH) + Math.Cos(angle) * (r - arrayCenterV) + arrayCenterH,
						MidpointRounding.AwayFromZero);
					if (arrayHost[d * width * height + r * width + c] != arrayRotatedHost[d * width * height + rotr * height + rotc])
					{
						failures++;
					}
				}
			}
		}

		return failures;
	}

	[Test]
	public async Task GpuMat_Rotate()
	{
		for (int w = 1; w <= 5; ++w)
		{
			for (int h = 1; h <= 5; ++h)
			{
				for (int d = 1; d <= 3; ++d)
				{
					int width = 20 * w;
					int height = 20 * h;
					await Assert.That(TestRotateImage<sbyte>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestRotateImage<short>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestRotateImage<int>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestRotateImage<long>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestRotateImage<float>(width, height, d)).IsEqualTo(0);
					await Assert.That(TestRotateImage<double>(width, height, d)).IsEqualTo(0);
				}
			}
		}
	}
}
