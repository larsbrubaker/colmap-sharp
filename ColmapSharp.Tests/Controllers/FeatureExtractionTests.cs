// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureExtractionTests: colmap/controllers/feature_extraction_test.cc ported 1:1, one
// method per gtest case named Suite_Name. Tests ColmapSharp/Controllers/FeatureExtraction.cs.
// Database files become InMemoryDatabases, and the written image files an
// InMemoryImageSource (divergence 82); the camera mask file becomes
// ImageReaderOptions.CameraMask. `use_gpu = false` has no counterpart (there is no GPU path).
// The importer's feature files are written to a temporary directory (the importer reads a
// directory, like COLMAP), with an ostream's default float formatting.
// The C#-only test at the end is labeled as such.

using ColmapSharp.Controllers;
using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class FeatureExtractionTests
{
	private static Bitmap CreateTestBitmap()
	{
		var bitmap = new Bitmap(100, 100, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		for (int y = 30; y < 70; ++y)
		{
			for (int x = 30; x < 70; ++x)
			{
				bitmap.SetPixel(x, y, new BitmapColor<byte>(255));
			}
		}

		return bitmap;
	}

	[Test]
	public async Task CreateFeatureExtractorController_Nominal()
	{
		using var database = new InMemoryDatabase();

		// Create test images
		const int kNumImages = 2;
		Bitmap testBitmap = CreateTestBitmap();
		var images = new InMemoryImageSource();
		for (int i = 0; i < kNumImages; ++i)
		{
			images.Add($"{i}.png", testBitmap);
		}

		// Set up options
		var readerOptions = new ImageReaderOptions { Images = images };
		var extractionOptions = new FeatureExtractionOptions { NumThreads = kNumImages };

		// Create and run the controller
		FeatureExtraction.ExtractFeatures(database, readerOptions, extractionOptions);

		// Verify results in database
		List<Image> dbImages = database.ReadAllImages();
		await Assert.That(dbImages.Count).IsEqualTo(kNumImages);

		foreach (Image image in dbImages)
		{
			await Assert.That(database.ExistsKeypoints(image.ImageId)).IsTrue();
			await Assert.That(database.ExistsDescriptors(image.ImageId)).IsTrue();

			List<FeatureKeypoint> keypoints = database.ReadKeypoints(image.ImageId);
			FeatureDescriptors descriptors = database.ReadDescriptors(image.ImageId);

			// Check that features were extracted
			await Assert.That(keypoints.Count).IsGreaterThan(0);
			await Assert.That(keypoints.Count).IsEqualTo(descriptors.Data.Rows);
			await Assert.That(descriptors.Type).IsEqualTo(FeatureExtractorType.Sift);
			await Assert.That(descriptors.Data.Cols).IsEqualTo(128);
		}
	}

	[Test]
	public async Task CreateFeatureExtractorController_WithCameraMask()
	{
		// Create test image with features
		Bitmap testBitmap = CreateTestBitmap();
		var images = new InMemoryImageSource();
		images.Add("test.png", testBitmap);

		// Create a mask that only allows the center region (white = keep, black = mask).
		// The test bitmap has a white square from (30,30) to (70,70). We'll create a mask
		// that only keeps a smaller region.
		var maskBitmap = new Bitmap(100, 100, asRgb: false);
		maskBitmap.Fill(new BitmapColor<byte>(0)); // Start with all black (masked)

		// Only keep center region (40,40) to (60,60)
		for (int y = 40; y < 60; ++y)
		{
			for (int x = 40; x < 60; ++x)
			{
				maskBitmap.SetPixel(x, y, new BitmapColor<byte>(255)); // White = keep
			}
		}

		// Extract features without mask first to get baseline
		var readerOptionsNoMask = new ImageReaderOptions { Images = images };
		var extractionOptions = new FeatureExtractionOptions { NumThreads = 1 };

		using var database = new InMemoryDatabase();
		FeatureExtraction.ExtractFeatures(database, readerOptionsNoMask, extractionOptions);

		List<Image> dbImages = database.ReadAllImages();
		await Assert.That(dbImages.Count).IsEqualTo(1);

		int numFeaturesNoMask = database.ReadKeypoints(dbImages[0].ImageId).Count;
		await Assert.That(numFeaturesNoMask).IsGreaterThan(0);

		// Now extract with mask
		var readerOptionsMasked = new ImageReaderOptions { Images = images, CameraMask = maskBitmap };

		using var databaseMasked = new InMemoryDatabase();
		FeatureExtraction.ExtractFeatures(databaseMasked, readerOptionsMasked, extractionOptions);

		dbImages = databaseMasked.ReadAllImages();
		await Assert.That(dbImages.Count).IsEqualTo(1);

		List<FeatureKeypoint> keypointsMasked = databaseMasked.ReadKeypoints(dbImages[0].ImageId);
		FeatureDescriptors descriptorsMasked = databaseMasked.ReadDescriptors(dbImages[0].ImageId);
		int numFeaturesMasked = keypointsMasked.Count;

		// With mask, should have fewer features
		await Assert.That(numFeaturesMasked).IsLessThan(numFeaturesNoMask);
		await Assert.That(numFeaturesMasked).IsGreaterThan(0); // But should still have some features

		// All remaining keypoints should be within the unmasked region (40-60, 40-60)
		foreach (FeatureKeypoint kp in keypointsMasked)
		{
			await Assert.That(kp.X).IsGreaterThanOrEqualTo(40.0f);
			await Assert.That(kp.X).IsLessThan(60.0f);
			await Assert.That(kp.Y).IsGreaterThanOrEqualTo(40.0f);
			await Assert.That(kp.Y).IsLessThan(60.0f);
		}

		// Descriptors should match keypoints count
		await Assert.That(descriptorsMasked.Data.Rows).IsEqualTo(keypointsMasked.Count);
		await Assert.That(descriptorsMasked.Type).IsEqualTo(FeatureExtractorType.Sift);
		await Assert.That(descriptorsMasked.Data.Cols).IsEqualTo(128);
	}

	[Test]
	public async Task CreateFeatureImporterController_Nominal()
	{
		string importPath = Path.Combine(Path.GetTempPath(), "colmapsharp-import-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(importPath);
		try
		{
			const int kNumImages = 2;
			const int kNumFeatures = 3;

			// Create test images
			Bitmap testBitmap = CreateTestBitmap();
			var images = new InMemoryImageSource();
			for (int i = 0; i < kNumImages; ++i)
			{
				images.Add($"{i}.png", testBitmap);
			}

			// Create feature text files for each image
			for (int i = 0; i < kNumImages; ++i)
			{
				var file = new System.Text.StringBuilder();

				// Write header: num_features dimension
				const int kDimension = 128;
				file.Append($"{kNumFeatures} {kDimension}\n");

				// Write features: x y scale orientation descriptor[0..127]
				for (int j = 0; j < kNumFeatures; ++j)
				{
					// Keypoint data
					file.Append(Format(10.0f + j * 5.0f)).Append(' ')  // x
						.Append(Format(20.0f + j * 5.0f)).Append(' ')     // y
						.Append(Format(1.5f + j * 0.1f)).Append(' ')      // scale
						.Append(Format(0.5f + j * 0.2f));                 // orientation

					// Descriptor data (128 values)
					for (int k = 0; k < kDimension; ++k)
					{
						file.Append(' ').Append((j * kDimension + k) % 256);
					}

					file.Append('\n');
				}

				File.WriteAllText(Path.Combine(importPath, $"{i}.png.txt"), file.ToString());
			}

			// Set up options
			var readerOptions = new ImageReaderOptions { Images = images };

			// Create and run the controller
			using var database = new InMemoryDatabase();
			FeatureExtraction.ImportFeatures(database, readerOptions, importPath);

			// Verify results in database
			List<Image> dbImages = database.ReadAllImages();
			await Assert.That(dbImages.Count).IsEqualTo(kNumImages);

			foreach (Image image in dbImages)
			{
				await Assert.That(database.ExistsKeypoints(image.ImageId)).IsTrue();
				await Assert.That(database.ExistsDescriptors(image.ImageId)).IsTrue();

				List<FeatureKeypoint> keypoints = database.ReadKeypoints(image.ImageId);
				FeatureDescriptors descriptors = database.ReadDescriptors(image.ImageId);

				// Check that features were imported correctly
				await Assert.That(keypoints.Count).IsEqualTo(kNumFeatures);
				await Assert.That(descriptors.Type).IsEqualTo(FeatureExtractorType.Sift);
				await Assert.That(descriptors.Data.Rows).IsEqualTo(kNumFeatures);
				await Assert.That(descriptors.Data.Cols).IsEqualTo(128);

				// Verify some keypoint values
				await Assert.That(keypoints[0].X).IsEqualTo(10.0f);
				await Assert.That(keypoints[0].Y).IsEqualTo(20.0f);
			}
		}
		finally
		{
			Directory.Delete(importPath, recursive: true);
		}
	}

	// `ostream << float`: default precision 6.
	private static string Format(float value) => ColmapSharp.Util.CppStreamFormat.FormatDouble(value);

	// C#-only: the database contents (ids, keypoints, descriptors) do not depend on the thread
	// count, since images are committed in reader order (divergence 83),
	// and progress reports one step per image.
	[Test]
	public async Task CSharpOnly_ThreadCountIndependentAndReportsProgress()
	{
		var images = new InMemoryImageSource();
		for (int i = 0; i < 5; ++i)
		{
			Bitmap bitmap = CreateTestBitmap();
			bitmap.SetPixel(10 + i * 5, 10, new BitmapColor<byte>(255));
			images.Add($"{i}.png", bitmap);
		}

		using var serial = new InMemoryDatabase();
		var steps = new List<ControllerProgress>();
		FeatureExtraction.ExtractFeatures(
			serial,
			new ImageReaderOptions { Images = images },
			new FeatureExtractionOptions { NumThreads = 1 },
			new SynchronousProgress(steps));

		using var parallel = new InMemoryDatabase();
		FeatureExtraction.ExtractFeatures(
			parallel, new ImageReaderOptions { Images = images }, new FeatureExtractionOptions { NumThreads = 3 });

		await Assert.That(steps.Count).IsEqualTo(5);
		await Assert.That(steps[4].Done).IsEqualTo(5);
		await Assert.That(steps[4].Total).IsEqualTo(5);

		List<Image> serialImages = serial.ReadAllImages();
		List<Image> parallelImages = parallel.ReadAllImages();
		await Assert.That(parallelImages.Count).IsEqualTo(serialImages.Count);
		for (int i = 0; i < serialImages.Count; ++i)
		{
			uint imageId = serialImages[i].ImageId;
			await Assert.That(parallelImages[i].ImageId).IsEqualTo(imageId);
			await Assert.That(parallelImages[i].Name).IsEqualTo(serialImages[i].Name);
			await Assert.That(parallel.ReadKeypoints(imageId).SequenceEqual(serial.ReadKeypoints(imageId))).IsTrue();
			await Assert.That(parallel.ReadDescriptors(imageId).Data == serial.ReadDescriptors(imageId).Data).IsTrue();
		}

		await Assert.That(parallel.NumFrames()).IsEqualTo(serial.NumFrames());
	}

	// Progress<T> posts to the thread pool; this one records in call order.
	private sealed class SynchronousProgress(List<ControllerProgress> steps) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value) => steps.Add(value);
	}
}
