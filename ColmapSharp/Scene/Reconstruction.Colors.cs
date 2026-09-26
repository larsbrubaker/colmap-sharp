// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction.Colors: port of Reconstruction::ExtractColorsForImage and
// ExtractColorsForAllImages from colmap/scene/reconstruction.cc - point coloring from the
// images. The incremental pipeline (Controllers/IncrementalPipeline*.cs) colors per image
// after each registration and averages over all images after triangulating an existing
// model. The storage lives in Reconstruction.cs. Tests: ExtractColorsForAllImages in
// ColmapSharp.Tests/Scene/ReconstructionTests.Colors.cs (reconstruction_test.cc 1:1);
// ExtractColorsForImage has no COLMAP test and is covered by the C#-only
// IncrementalPipeline_ExtractsColorsFromReadImage (Controllers/IncrementalPipelineTests.Host.cs).
//
// Translation notes:
// - COLMAP reads `path / image.Name()` with Bitmap::Read(as_rgb=true). The library does not
//   decode image files (the host does), so the caller passes the decoded bitmap, or an image
//   provider from image name to bitmap (null = the read failed: COLMAP logs a warning and
//   skips the image). A grey bitmap is converted to RGB as Bitmap::Read(as_rgb=true) would
//   (docs/CPP_DIVERGENCES.md, entry 68).
// - ExtractColorsForAllImages: COLMAP accumulates per-thread sums in thread-pool order, so its
//   floating-point sums depend on scheduling. Here each image fills its own partial sums (in
//   parallel, one slot per image) and they are reduced in ascending image-id order, so the
//   result is the same for any thread count (docs/CPP_DIVERGENCES.md, entry 69).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class Reconstruction
{
	/// <summary>
	/// Port of Reconstruction::ExtractColorsForImage: colors the still-black 3D points the
	/// image observes by bilinear interpolation of <paramref name="bitmap"/>, the decoded
	/// image. Points that already have a color keep it.
	/// </summary>
	public void ExtractColorsForImage(uint imageId, Bitmap bitmap)
	{
		Image image = Image(imageId);
		Bitmap rgb = bitmap.IsRGB ? bitmap : bitmap.CloneAsRGB();

		var blackColor = new Vector3ub(0, 0, 0);
		foreach (Point2D point2D in image.Points2D)
		{
			if (!point2D.HasPoint3D)
			{
				continue;
			}

			Point3D point3D = Point3D(point2D.Point3DId);
			if (point3D.Color == blackColor)
			{
				// COLMAP assumes that the upper left pixel center is (0.5, 0.5).
				BitmapColor<float>? color = rgb.InterpolateBilinear(point2D.Xy.X - 0.5, point2D.Xy.Y - 0.5);
				if (color is { } value)
				{
					BitmapColor<byte> colorUb = value.Cast<byte>();
					point3D.Color = new Vector3ub(colorUb.R, colorUb.G, colorUb.B);
				}
			}
		}
	}

	/// <summary>
	/// Port of Reconstruction::ExtractColorsForAllImages: sets every 3D point's color to the
	/// rounded mean of its bilinearly interpolated colors over all registered images;
	/// points seen in no readable image become black. <paramref name="readImage"/> decodes
	/// an image by name (COLMAP's path / name), returning null when it cannot.
	/// </summary>
	public void ExtractColorsForAllImages(Func<string, Bitmap?> readImage, int numThreads)
	{
		ArgumentNullException.ThrowIfNull(readImage);
		List<uint> regImageIds = RegImageIds();
		regImageIds.Sort();

		// One slot per image, filled in parallel; null when the image could not be read.
		var imageData = new Dictionary<ulong, ColorData>?[regImageIds.Count];
		Parallel.For(
			0,
			regImageIds.Count,
			new ParallelOptions { MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(numThreads) },
			i => imageData[i] = ExtractImageColorSums(readImage, Image(regImageIds[i])));

		// Merge per-image results in image-id order.
		var mergedData = new Dictionary<ulong, ColorData>();
		foreach (Dictionary<ulong, ColorData>? data in imageData)
		{
			if (data is null)
			{
				continue;
			}

			foreach ((ulong point3DId, ColorData imageColorData) in data)
			{
				mergedData.TryGetValue(point3DId, out ColorData merged);
				mergedData[point3DId] = new ColorData(
					merged.SumR + imageColorData.SumR,
					merged.SumG + imageColorData.SumG,
					merged.SumB + imageColorData.SumB,
					merged.Count + imageColorData.Count);
			}
		}

		foreach ((ulong point3DId, Point3D point3D) in _points3D)
		{
			if (mergedData.TryGetValue(point3DId, out ColorData colorData))
			{
				// Eigen's cast<uint8_t> after std::round: a plain conversion of the rounded mean.
				point3D.Color = new Vector3ub(
					(byte)Math.Round(colorData.SumR / colorData.Count, MidpointRounding.AwayFromZero),
					(byte)Math.Round(colorData.SumG / colorData.Count, MidpointRounding.AwayFromZero),
					(byte)Math.Round(colorData.SumB / colorData.Count, MidpointRounding.AwayFromZero));
			}
			else
			{
				point3D.Color = Vector3ub.Zero;
			}
		}
	}

	// The color sums of one image's observations, or null when the image cannot be read.
	private static Dictionary<ulong, ColorData>? ExtractImageColorSums(Func<string, Bitmap?> readImage, Image image)
	{
		Bitmap? bitmap = readImage(image.Name);
		if (bitmap is null)
		{
			Log.Warning($"Could not read image {image.Name}");
			return null;
		}

		Bitmap rgb = bitmap.IsRGB ? bitmap : bitmap.CloneAsRGB();
		var data = new Dictionary<ulong, ColorData>();
		foreach (Point2D point2D in image.Points2D)
		{
			if (!point2D.HasPoint3D)
			{
				continue;
			}

			// COLMAP assumes that the upper left pixel center is (0.5, 0.5).
			if (rgb.InterpolateBilinear(point2D.Xy.X - 0.5, point2D.Xy.Y - 0.5) is { } color)
			{
				data.TryGetValue(point2D.Point3DId, out ColorData colorData);
				data[point2D.Point3DId] = new ColorData(
					colorData.SumR + color.R, colorData.SumG + color.G, colorData.SumB + color.B, colorData.Count + 1);
			}
		}

		return data;
	}

	// COLMAP's ColorData: the summed colors (in double) and their count.
	private readonly record struct ColorData(double SumR, double SumG, double SumB, int Count);
}
