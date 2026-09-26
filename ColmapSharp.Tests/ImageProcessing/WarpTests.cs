// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// WarpTests: colmap/image/warp_test.cc ported 1:1 (Suite_Name). Tests
// ColmapSharp/ImageProcessing/Warp.cs and Warp.Float.cs.
//
// Tier A: the expected values and tolerances are COLMAP's. The random test bitmaps come from
// the default-seeded PRNG like COLMAP's; no expectation depends on their content. The
// per-pixel EXPECT_EQs inside the helpers count mismatches and assert the count is zero, so a
// 100x100 comparison is one assertion rather than ten thousand.

using ColmapSharp.ImageProcessing;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.ImageProcessing;

public class WarpTests
{
	private static Bitmap GenerateRandomBitmap(int width, int height, bool asRgb)
	{
		var bitmap = new Bitmap(width, height, asRgb);
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				byte r = (byte)RandomUtils.RandomUniformInteger(0, 255);
				byte g = (byte)RandomUtils.RandomUniformInteger(0, 255);
				byte b = (byte)RandomUtils.RandomUniformInteger(0, 255);
				bitmap.SetPixel(x, y, new BitmapColor<byte>(r, g, b));
			}
		}
		return bitmap;
	}

	private static async Task CheckBitmapsExactlyEqual(Bitmap bitmap1, Bitmap bitmap2)
	{
		using (Assert.Multiple())
		{
			await Assert.That(bitmap1.Width).IsEqualTo(bitmap2.Width);
			await Assert.That(bitmap1.Height).IsEqualTo(bitmap2.Height);
			await Assert.That(bitmap1.IsRGB).IsEqualTo(bitmap2.IsRGB);
			await Assert.That(bitmap1.RowMajorData.AsSpan().SequenceEqual(bitmap2.RowMajorData)).IsTrue();
		}
	}

	private static WarpImageOptions DirectWarpMinScaleOptions(double minScale) =>
		new() { DirectWarpMinScale = minScale };

	// Check that the two bitmaps are equal, ignoring a 1px boundary. With transposed set, pixel
	// (x, y) of the first is compared to (y, x) of the second (CheckBitmapsTransposed).
	private static async Task CheckBitmapsEqual(Bitmap bitmap1, Bitmap bitmap2, bool transposed = false)
	{
		await Assert.That(bitmap1.IsGrey).IsEqualTo(bitmap2.IsGrey);
		await Assert.That(bitmap1.IsRGB).IsEqualTo(bitmap2.IsRGB);
		await Assert.That(bitmap1.Width).IsEqualTo(bitmap2.Width);
		await Assert.That(bitmap1.Height).IsEqualTo(bitmap2.Height);
		int numMissing = 0;
		int numDifferent = 0;
		for (int x = 1; x < bitmap1.Width - 1; x++)
		{
			for (int y = 1; y < bitmap1.Height - 1; y++)
			{
				BitmapColor<byte>? color1 = bitmap1.GetPixel(x, y);
				BitmapColor<byte>? color2 = transposed ? bitmap2.GetPixel(y, x) : bitmap2.GetPixel(x, y);
				if (color1 is null || color2 is null)
				{
					numMissing++;
				}
				else if (color1.Value != color2.Value)
				{
					numDifferent++;
				}
			}
		}
		await Assert.That(numMissing).IsEqualTo(0);
		await Assert.That(numDifferent).IsEqualTo(0);
	}

	private static Task CheckBitmapsTransposed(Bitmap bitmap1, Bitmap bitmap2) =>
		CheckBitmapsEqual(bitmap1, bitmap2, transposed: true);

	[Test]
	public async Task Warp_IdenticalCameras()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 100, 100);
		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, false);
		Bitmap targetImageGray = Warp.WarpImageBetweenCameras(new WarpImageOptions(), camera, camera, sourceImageGray);
		await CheckBitmapsEqual(sourceImageGray, targetImageGray);
		Bitmap sourceImageRgb = GenerateRandomBitmap(100, 100, true);
		Bitmap targetImageRgb = Warp.WarpImageBetweenCameras(new WarpImageOptions(), camera, camera, sourceImageRgb);
		await CheckBitmapsEqual(sourceImageRgb, targetImageRgb);
	}

	[Test]
	public async Task Warp_DirectWarpMinScale()
	{
		Camera sourceCamera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 100, 100, 100);
		Bitmap sourceImage = GenerateRandomBitmap(100, 100, true);

		Camera targetCamera = sourceCamera.Clone();
		targetCamera.Rescale(50, 50);
		Bitmap defaultImage = Warp.WarpImageBetweenCameras(new WarpImageOptions(), sourceCamera, targetCamera, sourceImage);
		Bitmap directImage = Warp.WarpImageBetweenCameras(DirectWarpMinScaleOptions(0.0), sourceCamera, targetCamera, sourceImage);
		Bitmap resizedImage = Warp.WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0), sourceCamera, targetCamera, sourceImage);
		await CheckBitmapsExactlyEqual(defaultImage, directImage);
		await Assert.That(defaultImage.RowMajorData.AsSpan().SequenceEqual(resizedImage.RowMajorData)).IsFalse();

		targetCamera = sourceCamera.Clone();
		targetCamera.Rescale(49, 49);
		defaultImage = Warp.WarpImageBetweenCameras(new WarpImageOptions(), sourceCamera, targetCamera, sourceImage);
		resizedImage = Warp.WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0), sourceCamera, targetCamera, sourceImage);
		await CheckBitmapsExactlyEqual(defaultImage, resizedImage);

		targetCamera = sourceCamera.Clone();
		targetCamera.Rescale(95, 49);
		defaultImage = Warp.WarpImageBetweenCameras(new WarpImageOptions(), sourceCamera, targetCamera, sourceImage);
		resizedImage = Warp.WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0), sourceCamera, targetCamera, sourceImage);
		await CheckBitmapsExactlyEqual(defaultImage, resizedImage);

		await Assert.That(() => Warp.WarpImageBetweenCameras(
			DirectWarpMinScaleOptions(-0.1), sourceCamera, targetCamera, sourceImage)).ThrowsException();
	}

	[Test]
	public async Task Warp_Interpolation()
	{
		Camera sourceCamera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 4, 4, 4);
		Camera targetCamera = sourceCamera.Clone();
		targetCamera.SetPrincipalPointX(1.5);

		var sourceImage = new Bitmap(4, 4, asRgb: false);
		for (int y = 0; y < sourceImage.Height; y++)
		{
			sourceImage.SetPixel(0, y, new BitmapColor<byte>(0));
			sourceImage.SetPixel(1, y, new BitmapColor<byte>(100));
		}

		var options = new WarpImageOptions();
		Bitmap bilinearImage = Warp.WarpImageBetweenCameras(options, sourceCamera, targetCamera, sourceImage);
		await Assert.That(bilinearImage.GetPixel(0, 0).HasValue).IsTrue();
		await Assert.That(bilinearImage.GetPixel(0, 0)!.Value.R).IsEqualTo((byte)50);

		options.Interpolation = WarpInterpolation.NearestNeighbor;
		Bitmap nearestImage = Warp.WarpImageBetweenCameras(options, sourceCamera, targetCamera, sourceImage);
		await Assert.That(nearestImage.GetPixel(0, 0).HasValue).IsTrue();
		await Assert.That(nearestImage.GetPixel(0, 0)!.Value.R).IsEqualTo((byte)100);

		options.Interpolation = (WarpInterpolation)(-1);
		await Assert.That(() => Warp.WarpImageBetweenCameras(options, sourceCamera, targetCamera, sourceImage)).ThrowsException();
	}

	[Test]
	public async Task Warp_ShiftedCameras()
	{
		Camera sourceCamera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 100, 100);
		Camera targetCamera = sourceCamera.Clone();
		targetCamera.SetPrincipalPointX(0.0);
		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, true);
		Bitmap targetImageGray = Warp.WarpImageBetweenCameras(new WarpImageOptions(), sourceCamera, targetCamera, sourceImageGray);
		int numMissing = 0;
		int numNonBlackRight = 0;
		int numDifferent = 0;
		for (int x = 0; x < targetImageGray.Width; x++)
		{
			for (int y = 0; y < targetImageGray.Height; y++)
			{
				BitmapColor<byte>? color = targetImageGray.GetPixel(x, y);
				if (color is null)
				{
					numMissing++;
					continue;
				}
				if (x >= 50)
				{
					if (color.Value != new BitmapColor<byte>(0))
					{
						numNonBlackRight++;
					}
				}
				else
				{
					BitmapColor<byte>? sourceColor = sourceImageGray.GetPixel(x + 50, y);
					if (sourceColor is not null && color.Value != new BitmapColor<byte>(0) && color.Value != sourceColor.Value)
					{
						numDifferent++;
					}
				}
			}
		}
		await Assert.That(numMissing).IsEqualTo(0);
		await Assert.That(numNonBlackRight).IsEqualTo(0);
		await Assert.That(numDifferent).IsEqualTo(0);
	}

	[Test]
	public async Task Warp_WarpImageWithHomographyIdentity()
	{
		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, false);
		var targetImageGray = new Bitmap(100, 100, false);
		Warp.WarpImageWithHomography(Matrix3d.Identity, sourceImageGray, targetImageGray);
		await CheckBitmapsEqual(sourceImageGray, targetImageGray);

		Bitmap sourceImageRgb = GenerateRandomBitmap(100, 100, true);
		var targetImageRgb = new Bitmap(100, 100, true);
		Warp.WarpImageWithHomography(Matrix3d.Identity, sourceImageRgb, targetImageRgb);
		await CheckBitmapsEqual(sourceImageRgb, targetImageRgb);
	}

	[Test]
	public async Task Warp_WarpImageWithHomographyTransposed()
	{
		var h = new Matrix3d(0, 1, 0, 1, 0, 0, 0, 0, 1);

		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, false);
		var targetImageGray = new Bitmap(100, 100, false);
		Warp.WarpImageWithHomography(h, sourceImageGray, targetImageGray);
		await CheckBitmapsTransposed(sourceImageGray, targetImageGray);

		Bitmap sourceImageRgb = GenerateRandomBitmap(100, 100, true);
		var targetImageRgb = new Bitmap(100, 100, true);
		Warp.WarpImageWithHomography(h, sourceImageRgb, targetImageRgb);
		await CheckBitmapsTransposed(sourceImageRgb, targetImageRgb);
	}

	[Test]
	public async Task Warp_WarpImageWithHomographyBetweenCamerasIdentity()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 100, 100);
		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, false);
		Bitmap targetImageGray = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), Matrix3d.Identity, camera, camera, sourceImageGray);
		await CheckBitmapsEqual(sourceImageGray, targetImageGray);

		Bitmap sourceImageRgb = GenerateRandomBitmap(100, 100, true);
		Bitmap targetImageRgb = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), Matrix3d.Identity, camera, camera, sourceImageRgb);
		await CheckBitmapsEqual(sourceImageRgb, targetImageRgb);
	}

	[Test]
	public async Task Warp_HomographyDirectWarpMinScale()
	{
		Camera sourceCamera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 100, 100, 100);
		Bitmap sourceImage = GenerateRandomBitmap(100, 100, false);

		Camera targetCamera = sourceCamera.Clone();
		targetCamera.Rescale(50, 50);
		Bitmap defaultImage = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		Bitmap directImage = Warp.WarpImageWithHomographyBetweenCameras(
			DirectWarpMinScaleOptions(0.0), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		Bitmap resizedImage = Warp.WarpImageWithHomographyBetweenCameras(
			DirectWarpMinScaleOptions(1.0), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		await CheckBitmapsExactlyEqual(defaultImage, directImage);
		await Assert.That(defaultImage.RowMajorData.AsSpan().SequenceEqual(resizedImage.RowMajorData)).IsFalse();

		targetCamera = sourceCamera.Clone();
		targetCamera.Rescale(49, 49);
		defaultImage = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		resizedImage = Warp.WarpImageWithHomographyBetweenCameras(
			DirectWarpMinScaleOptions(1.0), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		await CheckBitmapsExactlyEqual(defaultImage, resizedImage);

		await Assert.That(() => Warp.WarpImageWithHomographyBetweenCameras(
			DirectWarpMinScaleOptions(-0.1), Matrix3d.Identity, sourceCamera, targetCamera, sourceImage)).ThrowsException();
	}

	[Test]
	public async Task Warp_HomographyInterpolation()
	{
		Camera sourceCamera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 4, 4, 4);
		Camera targetCamera = sourceCamera.Clone();
		targetCamera.SetPrincipalPointX(1.5);

		var sourceImage = new Bitmap(4, 4, asRgb: false);
		for (int y = 0; y < sourceImage.Height; y++)
		{
			sourceImage.SetPixel(0, y, new BitmapColor<byte>(0));
			sourceImage.SetPixel(1, y, new BitmapColor<byte>(100));
		}

		var options = new WarpImageOptions();
		Bitmap bilinearImage = Warp.WarpImageWithHomographyBetweenCameras(
			options, Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		await Assert.That(bilinearImage.GetPixel(0, 0).HasValue).IsTrue();
		await Assert.That(bilinearImage.GetPixel(0, 0)!.Value.R).IsEqualTo((byte)50);

		options.Interpolation = WarpInterpolation.NearestNeighbor;
		Bitmap nearestImage = Warp.WarpImageWithHomographyBetweenCameras(
			options, Matrix3d.Identity, sourceCamera, targetCamera, sourceImage);
		await Assert.That(nearestImage.GetPixel(0, 0).HasValue).IsTrue();
		await Assert.That(nearestImage.GetPixel(0, 0)!.Value.R).IsEqualTo((byte)100);

		options.Interpolation = (WarpInterpolation)(-1);
		await Assert.That(() => Warp.WarpImageWithHomographyBetweenCameras(
			options, Matrix3d.Identity, sourceCamera, targetCamera, sourceImage)).ThrowsException();
	}

	[Test]
	public async Task Warp_WarpImageWithHomographyBetweenCamerasTransposed()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 100, 100);

		var h = new Matrix3d(0, 1, 0, 1, 0, 0, 0, 0, 1);

		Bitmap sourceImageGray = GenerateRandomBitmap(100, 100, false);
		Bitmap targetImageGray = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), h, camera, camera, sourceImageGray);
		await CheckBitmapsTransposed(sourceImageGray, targetImageGray);

		Bitmap sourceImageRgb = GenerateRandomBitmap(100, 100, true);
		Bitmap targetImageRgb = Warp.WarpImageWithHomographyBetweenCameras(
			new WarpImageOptions(), h, camera, camera, sourceImageRgb);
		await CheckBitmapsTransposed(sourceImageRgb, targetImageRgb);
	}

	private static float[] Ramp16()
	{
		var image = new float[16];
		for (int i = 0; i < image.Length; i++)
		{
			image[i] = i;
		}
		return image;
	}

	[Test]
	public async Task Warp_ResampleImageBilinear()
	{
		float[] image = Ramp16();

		var resampled = new float[4];
		Warp.ResampleImageBilinear(image, 4, 4, 2, 2, resampled);

		using (Assert.Multiple())
		{
			await Assert.That(resampled[0]).IsEqualTo(2.5f);
			await Assert.That(resampled[1]).IsEqualTo(4.5f);
			await Assert.That(resampled[2]).IsEqualTo(10.5f);
			await Assert.That(resampled[3]).IsEqualTo(12.5f);
		}
	}

	[Test]
	public async Task Warp_SmoothImage()
	{
		float[] image = Ramp16();

		var smoothed = new float[16];
		Warp.SmoothImage(image, 4, 4, 1, 1, smoothed);

		using (Assert.Multiple())
		{
			await Assert.That((double)smoothed[0]).IsEqualTo(1.81673253).Within(1e-3);
			await Assert.That((double)smoothed[1]).IsEqualTo(2.51182437).Within(1e-3);
			await Assert.That((double)smoothed[2]).IsEqualTo(3.39494729).Within(1e-3);
			await Assert.That((double)smoothed[3]).IsEqualTo(4.09003973).Within(1e-3);
			await Assert.That((double)smoothed[4]).IsEqualTo(4.59710073).Within(1e-3);
			await Assert.That((double)smoothed[5]).IsEqualTo(5.29219341).Within(1e-3);
			await Assert.That((double)smoothed[6]).IsEqualTo(6.17531633).Within(1e-3);
			await Assert.That((double)smoothed[7]).IsEqualTo(6.87040806).Within(1e-3);
		}
	}

	[Test]
	public async Task Warp_DownsampleImage()
	{
		float[] image = Ramp16();

		var downsampled = new float[4];
		Warp.DownsampleImage(image, 4, 4, 2, 2, downsampled);

		using (Assert.Multiple())
		{
			await Assert.That((double)downsampled[0]).IsEqualTo(2.76810598).Within(1e-3);
			await Assert.That((double)downsampled[1]).IsEqualTo(4.66086388).Within(1e-3);
			await Assert.That((double)downsampled[2]).IsEqualTo(10.3391361).Within(1e-3);
			await Assert.That((double)downsampled[3]).IsEqualTo(12.2318935).Within(1e-3);
		}
	}
}
