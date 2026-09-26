// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Warp: the Bitmap warps of colmap/image/warp.h/.cc - WarpImageBetweenCameras,
// WarpImageWithHomography and WarpImageWithHomographyBetweenCameras. The float-buffer
// helpers of the same file (ResampleImageBilinear, SmoothImage, DownsampleImage) are in
// Warp.Float.cs. Undistortion (Undistortion.cs) is the main caller.
//
// The module is COLMAP's image/, named ImageProcessing (folder and namespace) because a
// ColmapSharp.Image namespace would shadow the ColmapSharp.Scene.Image class in every
// other ColmapSharp.* namespace - the same reason math/ is Mathematics.
//
// Tier A (exact) when the image is warped directly: the per-pixel camera mapping and the
// bilinear lookup (Bitmap.InterpolateBilinear) are scalar double arithmetic. When the target
// is much smaller than the source, COLMAP warps at source resolution and then resizes with
// Bitmap::Rescale, which is OIIO's resize in COLMAP and Tier B here (docs/CPP_DIVERGENCES.md,
// entry 9), so that path inherits Rescale's tier.
//
// Translation notes:
// - The C++ output parameter `Bitmap* target_image` that the function reallocates becomes the
//   return value; WarpImageWithHomography writes into a caller-allocated target, as in C++.
// - The template over the interpolation mode becomes a runtime switch; an invalid mode
//   throws before anything else is checked, like COLMAP's LOG(FATAL_THROW) after its switch.
// - Rows are warped in parallel (each row writes only its own pixels), so sequential and
//   parallel runs give identical output. The CancellationToken is observed between rows.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.ImageProcessing;

/// <summary>Port of colmap::WarpImageOptions::Interpolation.</summary>
public enum WarpInterpolation
{
	/// <summary>Nearest-neighbor sampling of the source image.</summary>
	NearestNeighbor,

	/// <summary>Bilinear sampling of the source image.</summary>
	Bilinear,
}

/// <summary>Port of colmap::WarpImageOptions.</summary>
public sealed class WarpImageOptions
{
	/// <summary>Interpolation method for sampling the source image.</summary>
	public WarpInterpolation Interpolation { get; set; } = WarpInterpolation.Bilinear;

	/// <summary>
	/// Minimum target-to-source dimension ratio for warping directly at target resolution.
	/// Smaller ratios warp at source resolution and resize the result to avoid aliasing.
	/// </summary>
	public double DirectWarpMinScale { get; set; } = 0.5;

	/// <summary>A copy (C++ copy construction).</summary>
	public WarpImageOptions Clone() => new()
	{
		Interpolation = Interpolation,
		DirectWarpMinScale = DirectWarpMinScale,
	};
}

/// <summary>Port of the free functions in colmap/image/warp.h.</summary>
public static partial class Warp
{
	/// <summary>
	/// Port of colmap::WarpImageBetweenCameras: warp the source image to the target camera by
	/// projecting each target pixel up to infinity and down into the source image (an inverse
	/// mapping). Pixels without a source sample are black. If the minimum target-to-source
	/// dimension ratio is at least <see cref="WarpImageOptions.DirectWarpMinScale"/> the image
	/// is warped at target resolution; otherwise at source resolution and then resized, to
	/// avoid aliasing.
	/// </summary>
	public static Bitmap WarpImageBetweenCameras(
		WarpImageOptions options,
		Camera sourceCamera,
		Camera targetCamera,
		Bitmap sourceImage,
		CancellationToken cancellationToken = default)
	{
		CheckInterpolation(options.Interpolation);
		Check.Eq(sourceCamera.Width, sourceImage.Width);
		Check.Eq(sourceCamera.Height, sourceImage.Height);

		bool warpDirectly = ShouldWarpDirectly(sourceCamera, targetCamera, options);
		Camera warpTargetCamera = targetCamera.Clone();
		if (!warpDirectly)
		{
			warpTargetCamera.Rescale(sourceCamera.Width, sourceCamera.Height);
		}

		var targetImage = new Bitmap(
			warpDirectly ? targetCamera.Width : sourceCamera.Width,
			warpDirectly ? targetCamera.Height : sourceCamera.Height,
			sourceImage.IsRGB);

		WarpInterpolation interpolation = options.Interpolation;
		ForEachRow(targetImage.Height, cancellationToken, y =>
		{
			double imageY = y + 0.5;
			for (int x = 0; x < targetImage.Width; x++)
			{
				// Camera models assume that the upper left pixel center is (0.5, 0.5).
				Vector2d? camPoint = warpTargetCamera.CamFromImg(new Vector2d(x + 0.5, imageY));
				SetWarpedPixel(targetImage, x, y, camPoint, sourceCamera, sourceImage, interpolation);
			}
		});

		if (!warpDirectly)
		{
			targetImage.Rescale(targetCamera.Width, targetCamera.Height);
		}

		return targetImage;
	}

	/// <summary>
	/// Port of colmap::WarpImageWithHomography: warp the source image into the caller-allocated
	/// <paramref name="targetImage"/>, where <paramref name="h"/> maps target pixels to source
	/// pixels. Pixel centers are at (0.5, 0.5). Pixels without a bilinear source sample are
	/// black.
	/// </summary>
	public static void WarpImageWithHomography(
		Matrix3d h,
		Bitmap sourceImage,
		Bitmap targetImage,
		CancellationToken cancellationToken = default)
	{
		Check.Gt(targetImage.Width, 0);
		Check.Gt(targetImage.Height, 0);
		Check.That(sourceImage.IsRGB == targetImage.IsRGB, null, "source_image.IsRGB() == target_image->IsRGB()");

		ForEachRow(targetImage.Height, cancellationToken, y =>
		{
			double targetY = y + 0.5;
			for (int x = 0; x < targetImage.Width; x++)
			{
				Vector2d sourcePixel = (h * new Vector3d(x + 0.5, targetY, 1)).HNormalized();
				BitmapColor<float>? color = sourceImage.InterpolateBilinear(sourcePixel.X - 0.5, sourcePixel.Y - 0.5);
				targetImage.SetPixel(x, y, color is { } c ? c.Cast<byte>() : new BitmapColor<byte>(0));
			}
		});
	}

	/// <summary>
	/// Port of colmap::WarpImageWithHomographyBetweenCameras: first map each target pixel
	/// through the homography <paramref name="h"/>, then warp between the cameras as
	/// <see cref="WarpImageBetweenCameras"/> does, with the same direct-warp threshold.
	/// </summary>
	public static Bitmap WarpImageWithHomographyBetweenCameras(
		WarpImageOptions options,
		Matrix3d h,
		Camera sourceCamera,
		Camera targetCamera,
		Bitmap sourceImage,
		CancellationToken cancellationToken = default)
	{
		CheckInterpolation(options.Interpolation);
		Check.Eq(sourceCamera.Width, sourceImage.Width);
		Check.Eq(sourceCamera.Height, sourceImage.Height);

		bool warpDirectly = ShouldWarpDirectly(sourceCamera, targetCamera, options);
		var targetImage = new Bitmap(
			warpDirectly ? targetCamera.Width : sourceCamera.Width,
			warpDirectly ? targetCamera.Height : sourceCamera.Height,
			sourceImage.IsRGB);

		// Unlike WarpImageBetweenCameras, COLMAP maps through the unrescaled target camera
		// even when warping at source resolution; kept as is.
		WarpInterpolation interpolation = options.Interpolation;
		ForEachRow(targetImage.Height, cancellationToken, y =>
		{
			double imageY = y + 0.5;
			for (int x = 0; x < targetImage.Width; x++)
			{
				// Camera models assume that the upper left pixel center is (0.5, 0.5).
				Vector3d warpedPoint = h * new Vector3d(x + 0.5, imageY, 1);
				if (warpedPoint.Z == 0)
				{
					// Left as allocated (black), like COLMAP's `continue`.
					continue;
				}

				Vector2d? camPoint = targetCamera.CamFromImg(warpedPoint.HNormalized());
				SetWarpedPixel(targetImage, x, y, camPoint, sourceCamera, sourceImage, interpolation);
			}
		});

		if (!warpDirectly)
		{
			targetImage.Rescale(targetCamera.Width, targetCamera.Height);
		}

		return targetImage;
	}

	/// <summary>
	/// The shared tail of the camera warps: project the camera-plane point into the source
	/// camera, sample the source image there, and write black when any step fails.
	/// </summary>
	private static void SetWarpedPixel(
		Bitmap targetImage,
		int x,
		int y,
		Vector2d? camPoint,
		Camera sourceCamera,
		Bitmap sourceImage,
		WarpInterpolation interpolation)
	{
		if (camPoint is not { } cam)
		{
			targetImage.SetPixel(x, y, new BitmapColor<byte>(0));
			return;
		}

		Vector2d? sourcePoint = sourceCamera.ImgFromCam(cam.Homogeneous());
		BitmapColor<byte>? color = sourcePoint is { } source ? InterpolatePixel(interpolation, sourceImage, source) : null;
		targetImage.SetPixel(x, y, color ?? new BitmapColor<byte>(0));
	}

	/// <summary>Port of the anonymous InterpolatePixel: sample at a pixel-center-(0.5, 0.5) point.</summary>
	private static BitmapColor<byte>? InterpolatePixel(WarpInterpolation interpolation, Bitmap image, Vector2d point)
	{
		double x = point.X - 0.5;
		double y = point.Y - 0.5;
		if (interpolation == WarpInterpolation.NearestNeighbor)
		{
			return image.InterpolateNearestNeighbor(x, y);
		}

		BitmapColor<float>? color = image.InterpolateBilinear(x, y);
		return color is { } c ? c.Cast<byte>() : null;
	}

	/// <summary>Port of the anonymous ShouldWarpDirectly.</summary>
	private static bool ShouldWarpDirectly(Camera sourceCamera, Camera targetCamera, WarpImageOptions options)
	{
		Check.Ge(options.DirectWarpMinScale, 0);
		Check.Gt(sourceCamera.Width, 0);
		Check.Gt(sourceCamera.Height, 0);
		Check.Gt(targetCamera.Width, 0);
		Check.Gt(targetCamera.Height, 0);
		if (targetCamera.Width == sourceCamera.Width && targetCamera.Height == sourceCamera.Height)
		{
			return true;
		}

		double scaleX = (double)targetCamera.Width / sourceCamera.Width;
		double scaleY = (double)targetCamera.Height / sourceCamera.Height;
		return Math.Min(scaleX, scaleY) >= options.DirectWarpMinScale;
	}

	/// <summary>COLMAP's LOG(FATAL_THROW) for an interpolation mode outside the enum.</summary>
	private static void CheckInterpolation(WarpInterpolation interpolation)
	{
		if (interpolation is not (WarpInterpolation.NearestNeighbor or WarpInterpolation.Bilinear))
		{
			throw new InvalidOperationException($"Invalid warp image interpolation mode: {(int)interpolation}");
		}
	}

	/// <summary>
	/// Runs <paramref name="rowBody"/> for every row in parallel. Each row writes only its own
	/// pixels, so the result does not depend on scheduling.
	/// </summary>
	private static void ForEachRow(int rows, CancellationToken cancellationToken, Action<int> rowBody)
	{
		var parallelOptions = new ParallelOptions { CancellationToken = cancellationToken };
		Parallel.For(0, rows, parallelOptions, rowBody);
	}
}
