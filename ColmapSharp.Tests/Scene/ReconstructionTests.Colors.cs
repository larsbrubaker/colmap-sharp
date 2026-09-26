// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests.Colors: the ExtractColorsForAllImages case of
// colmap/scene/reconstruction_test.cc, testing ColmapSharp/Scene/Reconstruction.Colors.cs.
// The rest of the file is ReconstructionTests.cs and its partials.
//
// Tier A (exact): the colors of a solid image are exact after rounding.
//
// Translation notes: WriteSolidColorImages writes a solid bitmap per image into a folder and
// ExtractColorsForAllImages reads them back; here the image provider returns the same
// solid bitmaps from memory (the library does not decode files), and "deleting" the first
// image's file makes the provider return null for it, like the failed read.
// ExtractColorsForAllImagesIndependentOfThreadCount is C#-only (divergence 69).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	// WriteSolidColorImages: a solid bitmap of each image's camera size, by image name.
	private static Dictionary<string, Bitmap> SolidColorImages(Reconstruction reconstruction, BitmapColor<byte> color)
	{
		var images = new Dictionary<string, Bitmap>();
		foreach (Image image in reconstruction.Images.Values)
		{
			Camera camera = reconstruction.Camera(image.CameraId);
			var bitmap = new Bitmap((int)camera.Width, (int)camera.Height, asRgb: true);
			bitmap.Fill(color);
			images[image.Name] = bitmap;
		}

		return images;
	}

	[Test]
	public async Task Reconstruction_ExtractColorsForAllImages()
	{
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 3,
			NumPoints3D = 10,
			NumPoints2DWithoutPoint3D = 2,
		};
		Synthetic.SynthesizeDataset(options, reconstruction);

		var color = new BitmapColor<byte>(20, 40, 220);
		Dictionary<string, Bitmap> images = SolidColorImages(reconstruction, color);

		// Delete one image file so extraction must handle the missing file.
		uint firstImageId = reconstruction.RegImageIds()[0];
		images.Remove(reconstruction.Image(firstImageId).Name);

		reconstruction.ExtractColorsForAllImages(name => images.GetValueOrDefault(name), numThreads: 2);

		var colors = reconstruction.Point3DIds().Select(id => reconstruction.Point3D(id).Color).ToList();
		await Assert.That(colors.Count).IsEqualTo(10);
		foreach (Vector3ub pointColor in colors)
		{
			await Assert.That(pointColor).IsEqualTo(new Vector3ub(color.R, color.G, color.B));
		}
	}

	// C#-only: the averaged colors do not depend on the thread count (divergence 69).
	[Test]
	public async Task Reconstruction_ExtractColorsForAllImagesIndependentOfThreadCount()
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(new SyntheticDatasetOptions { NumFramesPerRig = 4, NumPoints3D = 50 }, reconstruction);

		// A gradient, so points take different, non-integer averages.
		var images = new Dictionary<string, Bitmap>();
		int shade = 0;
		foreach (Image image in reconstruction.Images.Values)
		{
			Camera camera = reconstruction.Camera(image.CameraId);
			var bitmap = new Bitmap((int)camera.Width, (int)camera.Height, asRgb: true);
			for (int y = 0; y < bitmap.Height; ++y)
			{
				for (int x = 0; x < bitmap.Width; ++x)
				{
					bitmap.SetPixel(x, y, new BitmapColor<byte>((byte)((x + shade) % 256), (byte)(y % 256), (byte)((x * y + shade) % 256)));
				}
			}

			shade += 37;
			images[image.Name] = bitmap;
		}

		reconstruction.ExtractColorsForAllImages(name => images.GetValueOrDefault(name), numThreads: 1);
		var colors1 = reconstruction.Points3D.ToDictionary(p => p.Key, p => p.Value.Color);
		reconstruction.ExtractColorsForAllImages(name => images.GetValueOrDefault(name), numThreads: 8);
		var colors8 = reconstruction.Points3D.ToDictionary(p => p.Key, p => p.Value.Color);

		await Assert.That(colors8).IsEquivalentTo(colors1);
		await Assert.That(colors1.Values.Distinct().Count()).IsGreaterThan(1);
	}
}
