// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// UndistortersTests: colmap/controllers/undistorters_test.cc ported 1:1, one method per gtest
// case named Suite_Name, plus C#-only tests (labeled) for the MatterCAD additions. Tests
// ColmapSharp/Controllers/{ColmapUndistorter,PmvsUndistorter,StandaloneImageUndistorter}.cs.
// Translation of the fixtures:
// - The input image files COLMAP writes with Bitmap::Write become an InMemoryImageSource of
//   the same bitmaps under the same names (decoding is the host's job,
//   docs/CPP_DIVERGENCES.md entry 82).
// - The undistorted images go to an InMemoryBitmapStore (entry 98), so ExistsFile /
//   GetRecursiveFileList on an image path ask the store; text files and directories are on
//   disk, as in COLMAP.
// Tier A (exact) for what these tests pin (which files exist).

using ColmapSharp.Controllers;
using ColmapSharp.ImageProcessing;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class UndistortersTests
{
	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-undistorters-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	private static (Reconstruction Reconstruction, InMemoryImageSource Images) CreateSyntheticReconstructionWithBitmaps(
		int numImages = 2,
		int imageWidth = 100,
		int imageHeight = 100,
		string imageExtension = ".png")
	{
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numImages,
			CameraWidth = imageWidth,
			CameraHeight = imageHeight,
			ImageExtension = imageExtension,
		};

		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		// Create dummy images.
		var images = new InMemoryImageSource();
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			var bitmap = new Bitmap(imageWidth, imageHeight, true);
			bitmap.Fill(new BitmapColor<byte>(128, 128, 128));
			images.Add(image.Name, bitmap);
		}

		return (reconstruction, images);
	}

	[Test]
	public async Task COLMAPUndistorter_Integration()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();

		// Run COLMAP undistorter.
		var sink = new InMemoryBitmapStore();
		var undistorter = new ColmapUndistorter(
			new ColmapUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		undistorter.Run();

		// Verify output directories were created.
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "images"))).IsTrue();
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "sparse"))).IsTrue();
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "stereo"))).IsTrue();

		// Verify undistorted images were written.
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			await Assert.That(sink.Exists(Path.Combine(outputPath, "images", image.Name))).IsTrue();
		}

		// Expect dense reconstruction files to be written.
		await Assert.That(File.Exists(Path.Combine(outputPath, "stereo/patch-match.cfg"))).IsTrue();
		await Assert.That(File.Exists(Path.Combine(outputPath, "stereo/fusion.cfg"))).IsTrue();
	}

	[Test]
	public async Task COLMAPUndistorter_StopsPendingWork()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "output");
		Directory.CreateDirectory(outputPath);

		(Reconstruction reconstruction, InMemoryImageSource images) =
			CreateSyntheticReconstructionWithBitmaps(numImages: 10);
		var undistorter = new ColmapUndistorter(
			new ColmapUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath,
			new InMemoryBitmapStore());
		bool stopChecked = false;
		undistorter.SetCheckIfStoppedFunc(() =>
		{
			stopChecked = true;
			return true;
		});
		undistorter.Run();

		await Assert.That(stopChecked).IsTrue();
		await Assert.That(File.Exists(Path.Combine(outputPath, "stereo/patch-match.cfg"))).IsFalse();
		await Assert.That(File.Exists(Path.Combine(outputPath, "stereo/fusion.cfg"))).IsFalse();
	}

	[Test]
	public async Task COLMAPUndistorter_SpecificImages()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps(
			numImages: 2, imageWidth: 100, imageHeight: 100, imageExtension: ".jpg");

		ColmapSharp.Scene.Image image = reconstruction.Image(reconstruction.RegImageIds()[0]);

		// Run COLMAP undistorter.
		var options = new ColmapUndistorter.Options { ImageIds = [image.ImageId] };
		var sink = new InMemoryBitmapStore();
		var undistorter = new ColmapUndistorter(
			options, new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		undistorter.Run();

		// Verify that only the specified image was written.
		await Assert.That(sink.Paths).IsEquivalentTo([Path.Combine(outputPath, "images", image.Name)]);
	}

	[Test]
	public async Task COLMAPUndistorter_JpegQuality()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps(
			numImages: 1, imageWidth: 100, imageHeight: 100, imageExtension: ".jpg");

		// Run COLMAP undistorter.
		var options = new ColmapUndistorter.Options { JpegQuality = 50 };
		var sink = new InMemoryBitmapStore();
		var undistorter = new ColmapUndistorter(
			options, new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		undistorter.Run();

		// Verify undistorted images were written.
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			await Assert.That(sink.Exists(Path.Combine(outputPath, "images", image.Name))).IsTrue();
		}
	}

	[Test]
	public async Task PMVSUndistorter_Integration()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "pmvs_output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();

		// Run PMVS undistorter.
		var sink = new InMemoryBitmapStore();
		var undistorter = new PmvsUndistorter(
			new PmvsUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		undistorter.Run();

		// Verify PMVS output structure was created (under pmvs/ subdirectory).
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "pmvs"))).IsTrue();
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "pmvs", "models"))).IsTrue();
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "pmvs", "txt"))).IsTrue();
		await Assert.That(Directory.Exists(Path.Combine(outputPath, "pmvs", "visualize"))).IsTrue();

		// Verify undistorted images were written with numbered names.
		// PMVS writes images as 00000000.jpg, 00000001.jpg, etc.
		int numImages = reconstruction.NumRegImages;
		for (int i = 0; i < numImages; ++i)
		{
			string imageName = $"{i:D8}.jpg";
			await Assert.That(sink.Exists(Path.Combine(outputPath, "pmvs", "visualize", imageName))).IsTrue();
		}
	}

	[Test]
	public async Task CMPMVSUndistorter_Integration()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "cmpmvs_output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();

		// Run CMP-MVS undistorter.
		var sink = new InMemoryBitmapStore();
		var undistorter = new CmpMvsUndistorter(
			new CmpMvsUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		undistorter.Run();

		// Verify CMP-MVS output structure was created.
		await Assert.That(Directory.Exists(outputPath)).IsTrue();

		// Verify undistorted images were written with sequential numbering.
		// CMP-MVS writes images as 00001.jpg, 00002.jpg, etc.
		int numImages = reconstruction.NumRegImages;
		for (int i = 1; i <= numImages; ++i)
		{
			string imageName = $"{i:D5}.jpg";
			await Assert.That(sink.Exists(Path.Combine(outputPath, imageName))).IsTrue();
		}
	}

	[Test]
	public async Task StandaloneImageUndistorter_Integration()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "pure_output");

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();

		var options = new StandaloneImageUndistorter.Options();
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			options.ImageNamesAndCameras.Add((image.Name, image.CameraPtr));
		}

		// Run standalone image undistorter.
		var sink = new InMemoryBitmapStore();
		var undistorter = new StandaloneImageUndistorter(options, new UndistortCameraOptions(), images, outputPath, sink);
		undistorter.Run();

		// Verify output directory was created.
		await Assert.That(Directory.Exists(outputPath)).IsTrue();

		// Verify undistorted images were written.
		foreach ((string imageName, Camera _) in options.ImageNamesAndCameras)
		{
			await Assert.That(sink.Exists(Path.Combine(outputPath, imageName))).IsTrue();
		}
	}

	[Test]
	public async Task StereoImageRectifier_Integration()
	{
		string tempDir = CreateTestDir();
		string outputPath = Path.Combine(tempDir, "stereo_output");
		Directory.CreateDirectory(outputPath);

		// Create synthetic reconstruction with dummy images.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();

		// Create stereo pair from first two images.
		var options = new StereoImageRectifier.Options();
		List<uint> imageIds = reconstruction.RegImageIds();
		await Assert.That(imageIds.Count).IsGreaterThanOrEqualTo(2);
		options.StereoPairs.Add((imageIds[0], imageIds[1]));

		// Run stereo image rectifier.
		var sink = new InMemoryBitmapStore();
		var rectifier = new StereoImageRectifier(
			options, new UndistortCameraOptions(), reconstruction, images, outputPath, sink);
		rectifier.Run();

		// Verify output directory was created.
		await Assert.That(Directory.Exists(outputPath)).IsTrue();

		// Verify rectified images were written.
		// StereoImageRectifier creates a subdirectory for each stereo pair.
		ColmapSharp.Scene.Image image1 = reconstruction.Image(options.StereoPairs[0].ImageId1);
		ColmapSharp.Scene.Image image2 = reconstruction.Image(options.StereoPairs[0].ImageId2);
		string stereoPairName = $"{image1.Name}-{image2.Name}";
		await Assert.That(Directory.Exists(Path.Combine(outputPath, stereoPairName))).IsTrue();
		await Assert.That(sink.Exists(Path.Combine(outputPath, stereoPairName, image1.Name))).IsTrue();
		await Assert.That(sink.Exists(Path.Combine(outputPath, stereoPairName, image2.Name))).IsTrue();
	}
}
