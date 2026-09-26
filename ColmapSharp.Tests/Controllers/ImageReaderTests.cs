// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ImageReaderTests: colmap/controllers/image_reader_test.cc ported 1:1, one method per gtest
// case named Suite_Name (the parameterized suite's INSTANTIATE values become Arguments).
// Tests ColmapSharp/Controllers/ImageReader.cs. Translation of the fixtures:
// - Database::Open(kInMemorySqliteDatabasePath) becomes an InMemoryDatabase.
// - The test directory of written image files becomes an InMemoryImageSource of the same
//   bitmaps under the same names (decoding is the host's job, docs/CPP_DIVERGENCES.md entry
//   82); "a file that is not a valid image" is a name mapped to null.
// Tier A (exact).

using ColmapSharp.Controllers;
using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class ImageReaderTests
{
	private static Bitmap CreateTestBitmap(bool asRgb)
	{
		var bitmap = new Bitmap(1, 3, asRgb);
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(1));
		bitmap.SetPixel(1, 0, new BitmapColor<byte>(2));
		bitmap.SetPixel(2, 0, new BitmapColor<byte>(3));
		return bitmap;
	}

	[Test]
	[Arguments(0, false, true, true, ".png")]
	[Arguments(5, false, false, true, ".png")]
	[Arguments(5, true, false, true, ".png")]
	[Arguments(5, true, false, true, ".bmp")]
	[Arguments(5, true, false, false, ".png")]
	[Arguments(5, false, true, true, ".png")]
	public async Task ParameterizedImageReaderTests_Nominal(
		int kNumImages, bool kWithMasks, bool kWithExistingImages, bool kAsRGB, string kExtension)
	{
		using var database = new InMemoryDatabase();

		var images = new InMemoryImageSource();
		var masks = new InMemoryImageSource();
		var options = new ImageReaderOptions { Images = images, AsRgb = kAsRGB };
		if (kWithMasks)
		{
			options.Masks = masks;
		}

		Bitmap testBitmap = CreateTestBitmap(kAsRGB);
		for (int i = 0; i < kNumImages; ++i)
		{
			string stem = i.ToString(System.Globalization.CultureInfo.InvariantCulture);
			string imageName = stem + kExtension;
			images.Add(imageName, testBitmap);
			if (kWithMasks)
			{
				if (i == 0)
				{
					// append .png to image_name
					masks.Add(imageName + ".png", testBitmap);
				}
				else
				{
					// replace mask extension by .png
					masks.Add(stem + ".png", testBitmap);
				}
			}

			if (kWithExistingImages)
			{
				var image = new Image { Name = imageName };
				image.SetCameraId(database.WriteCamera(Camera.CreateFromModelName(
					(uint)(i + 1), options.CameraModel, focalLength: 1, testBitmap.Width, testBitmap.Height)));
				image.ImageId = database.WriteImage(image);
				database.WriteKeypoints(image.ImageId, new List<FeatureKeypoint>());
				database.WriteDescriptors(image.ImageId, new FeatureDescriptors());
				var rig = new Rig();
				rig.AddRefSensor(new SensorId(SensorType.Camera, image.CameraId));
				database.WriteRig(rig);
			}
		}

		var imageReader = new ImageReader(options, database);
		await Assert.That(imageReader.NumImages).IsEqualTo(kNumImages);

		for (int i = 0; i < kNumImages; ++i)
		{
			await Assert.That(imageReader.NextIndex).IsEqualTo(i);
			ImageReader.Status status = imageReader.Next(out ImageReaderData data);
			if (kWithExistingImages)
			{
				await Assert.That(status).IsEqualTo(ImageReader.Status.ImageExists);
				continue;
			}

			await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
			await Assert.That(data.Rig.RigId).IsEqualTo((uint)(i + 1));
			await Assert.That(data.Camera.CameraId).IsEqualTo((uint)(i + 1));
			await Assert.That(data.Camera.ModelName).IsEqualTo(options.CameraModel);
			await Assert.That(data.Camera.Width).IsEqualTo(testBitmap.Width);
			await Assert.That(data.Camera.Height).IsEqualTo(testBitmap.Height);
			await Assert.That(data.Image.Name).IsEqualTo(i.ToString(System.Globalization.CultureInfo.InvariantCulture) + kExtension);
			await Assert.That(data.Bitmap.IsRGB).IsEqualTo(kAsRGB);
			await Assert.That(data.Bitmap.RowMajorData.SequenceEqual(testBitmap.RowMajorData)).IsTrue();
			if (kWithExistingImages)
			{
				await Assert.That(database.NumRigs()).IsEqualTo(kNumImages);
				await Assert.That(database.NumCameras()).IsEqualTo(kNumImages);
			}
			else
			{
				await Assert.That(database.NumRigs()).IsEqualTo(i + 1);
				await Assert.That(database.NumCameras()).IsEqualTo(i + 1);
			}
		}

		await Assert.That(() => imageReader.Next(out _)).Throws<ArgumentException>();
		await Assert.That(database.NumRigs()).IsEqualTo(kNumImages);
		await Assert.That(database.NumCameras()).IsEqualTo(kNumImages);
	}

	// Writes the same bitmap under each name (test_bitmap.Write(image_path / name)).
	private static InMemoryImageSource CreateImages(Bitmap bitmap, params string[] names)
	{
		var images = new InMemoryImageSource();
		foreach (string name in names)
		{
			images.Add(name, bitmap);
		}

		return images;
	}

	// Next for tests that only need the status.
	private static ImageReader.Status Next(ImageReader imageReader) => imageReader.Next(out _);

	[Test]
	public async Task ImageReaderTest_SingleCamera()
	{
		using var database = new InMemoryDatabase();

		// Create 3 test images with same dimensions
		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "0.png", "1.png", "2.png"),
			SingleCamera = true,
		};

		var imageReader = new ImageReader(options, database);
		await Assert.That(imageReader.NumImages).IsEqualTo(3);

		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.Success);
		}

		await Assert.That(database.NumRigs()).IsEqualTo(1);
		await Assert.That(database.NumCameras()).IsEqualTo(1);
	}

	[Test]
	public async Task ImageReaderTest_SingleCameraDimensionError()
	{
		using var database = new InMemoryDatabase();

		// Create images with different dimensions
		var images = new InMemoryImageSource();
		images.Add("0.png", new Bitmap(10, 20, true));
		images.Add("1.png", new Bitmap(30, 40, true));
		var options = new ImageReaderOptions { Images = images, SingleCamera = true };

		var imageReader = new ImageReader(options, database);

		// First image succeeds
		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.Success);

		// Second image fails due to dimension mismatch
		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.CameraSingleDimError);
	}

	[Test]
	public async Task ImageReaderTest_SingleCameraPerFolder()
	{
		using var database = new InMemoryDatabase();

		// Create 2 images in each folder
		var options = new ImageReaderOptions
		{
			Images = CreateImages(
				new Bitmap(10, 20, true), "folder1/0.png", "folder1/1.png", "folder2/0.png", "folder2/1.png"),
			SingleCameraPerFolder = true,
		};

		var imageReader = new ImageReader(options, database);
		await Assert.That(imageReader.NumImages).IsEqualTo(4);

		var folderCameras = new Dictionary<string, uint>();
		for (int i = 0; i < 4; ++i)
		{
			ImageReader.Status status = imageReader.Next(out ImageReaderData data);
			await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
			string folder = data.Image.Name[..data.Image.Name.LastIndexOf('/')];
			if (!folderCameras.TryGetValue(folder, out uint cameraId))
			{
				folderCameras[folder] = data.Camera.CameraId;
			}
			else
			{
				await Assert.That(data.Camera.CameraId).IsEqualTo(cameraId);
			}
		}

		// Should have 2 cameras (one per folder)
		await Assert.That(database.NumCameras()).IsEqualTo(2);
	}

	[Test]
	public async Task ImageReaderTest_SingleCameraPerImage()
	{
		using var database = new InMemoryDatabase();

		// Create 3 images with same dimensions
		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "0.png", "1.png", "2.png"),
			SingleCameraPerImage = true,
		};

		var imageReader = new ImageReader(options, database);

		for (int i = 0; i < 3; ++i)
		{
			ImageReader.Status status = imageReader.Next(out ImageReaderData data);
			await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
			await Assert.That(data.Camera.CameraId).IsEqualTo((uint)(i + 1)); // Each image gets its own camera
		}

		// Should have 3 cameras (one per image)
		await Assert.That(database.NumCameras()).IsEqualTo(3);
	}

	[Test]
	public async Task ImageReaderTest_ExistingCameraId()
	{
		using var database = new InMemoryDatabase();

		// Create an existing camera in the database
		var existingCamera = new Camera
		{
			ModelId = CameraModels.CameraModelNameToId("SIMPLE_RADIAL"),
			Width = 10,
			Height = 20,
			Params = [1.0, 5.0, 10.0, 0.0],
		};
		existingCamera.CameraId = database.WriteCamera(existingCamera);
		var existingRig = new Rig();
		existingRig.AddRefSensor(existingCamera.SensorId);
		database.WriteRig(existingRig);

		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "0.png", "1.png"),
			ExistingCameraId = existingCamera.CameraId,
		};

		var imageReader = new ImageReader(options, database);

		for (int i = 0; i < 2; ++i)
		{
			ImageReader.Status status = imageReader.Next(out ImageReaderData data);
			await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
			await Assert.That(data.Camera.CameraId).IsEqualTo(existingCamera.CameraId);
			await Assert.That(data.Camera.Params.SequenceEqual(existingCamera.Params)).IsTrue();
		}

		// No new cameras created
		await Assert.That(database.NumCameras()).IsEqualTo(1);
	}

	[Test]
	public async Task ImageReaderTest_ManualCameraParams()
	{
		using var database = new InMemoryDatabase();

		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(640, 480, true), "test.png"),
			CameraModel = "PINHOLE",
			CameraParams = "500.0, 500.0, 320.0, 240.0",
		};

		var imageReader = new ImageReader(options, database);

		ImageReader.Status status = imageReader.Next(out ImageReaderData data);
		await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
		await Assert.That(data.Camera.ModelId).IsEqualTo(CameraModelId.Pinhole);
		await Assert.That(data.Camera.Params[0]).IsEqualTo(500.0);
		await Assert.That(data.Camera.Params[1]).IsEqualTo(500.0);
		await Assert.That(data.Camera.Params[2]).IsEqualTo(320.0);
		await Assert.That(data.Camera.Params[3]).IsEqualTo(240.0);
		await Assert.That(data.Camera.HasPriorFocalLength).IsTrue();
	}

	[Test]
	public async Task ImageReaderTest_ExplicitImageNames()
	{
		using var database = new InMemoryDatabase();

		// Create 5 images
		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "0.png", "1.png", "2.png", "3.png", "4.png"),
			// Only select a subset of images
			ImageNames = ["1.png", "3.png"],
		};

		var imageReader = new ImageReader(options, database);
		await Assert.That(imageReader.NumImages).IsEqualTo(2);

		ImageReader.Status status = imageReader.Next(out ImageReaderData data);
		await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
		await Assert.That(data.Image.Name).IsEqualTo("1.png");

		status = imageReader.Next(out data);
		await Assert.That(status).IsEqualTo(ImageReader.Status.Success);
		await Assert.That(data.Image.Name).IsEqualTo("3.png");
	}

	[Test]
	public async Task ImageReaderTest_BitmapError()
	{
		using var database = new InMemoryDatabase();

		// Create a file that is not a valid image
		var images = new InMemoryImageSource();
		images.Add("invalid.png", null);
		var options = new ImageReaderOptions { Images = images };

		var imageReader = new ImageReader(options, database);

		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.BitmapError);
	}

	[Test]
	public async Task ImageReaderTest_MaskErrorMissing()
	{
		using var database = new InMemoryDatabase();

		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "test.png"),
			// Don't create mask file
			Masks = new InMemoryImageSource(),
		};

		var imageReader = new ImageReader(options, database);

		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.MaskError);
	}

	[Test]
	public async Task ImageReaderTest_MaskErrorInvalid()
	{
		using var database = new InMemoryDatabase();

		// Create invalid mask file
		var masks = new InMemoryImageSource();
		masks.Add("test.png.png", null);
		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "test.png"),
			Masks = masks,
		};

		var imageReader = new ImageReader(options, database);

		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.MaskError);
	}

	[Test]
	public async Task ImageReaderTest_ImageExistsWithKeypoints()
	{
		using var database = new InMemoryDatabase();

		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "test.png"),
		};

		// Add existing image with keypoints and descriptors
		var existingCamera = new Camera
		{
			ModelId = CameraModels.CameraModelNameToId("SIMPLE_RADIAL"),
			Width = 10,
			Height = 20,
			Params = [1.0, 5.0, 10.0, 0.0],
		};
		existingCamera.CameraId = database.WriteCamera(existingCamera);
		var existingRig = new Rig();
		existingRig.AddRefSensor(existingCamera.SensorId);
		database.WriteRig(existingRig);

		var existingImage = new Image { Name = "test.png" };
		existingImage.SetCameraId(existingCamera.CameraId);
		existingImage.ImageId = database.WriteImage(existingImage);
		database.WriteKeypoints(existingImage.ImageId, new List<FeatureKeypoint>());
		database.WriteDescriptors(existingImage.ImageId, new FeatureDescriptors());

		var imageReader = new ImageReader(options, database);

		await Assert.That(Next(imageReader)).IsEqualTo(ImageReader.Status.ImageExists);
	}

	// C#-only: image names sort by their UTF-8 bytes, like std::sort on std::string. UTF-16
	// ordinal order would put the supplementary-plane name (surrogate 0xD83D) before U+FFFD;
	// in UTF-8, U+FFFD (EF BF BD) comes before it (F0 9F 98 80).
	[Test]
	public async Task CSharpOnly_ImageNamesSortByUtf8Bytes()
	{
		using var database = new InMemoryDatabase();
		var options = new ImageReaderOptions
		{
			Images = CreateImages(new Bitmap(10, 20, true), "\U0001F600.png", "\uFFFD.png"),
		};

		var imageReader = new ImageReader(options, database);

		imageReader.Next(out ImageReaderData data);
		await Assert.That(data.Image.Name).IsEqualTo("\uFFFD.png");
		imageReader.Next(out data);
		await Assert.That(data.Image.Name).IsEqualTo("\U0001F600.png");
	}

	// C#-only: names listed by the source sort like COLMAP's sorted std::filesystem::path list,
	// element by element ("set" < "set-2", "img" < "img.jpg"), not as whole strings, where
	// '-' (0x2D) and '.' (0x2E) sort before '/' (0x2F). Explicit ImageNames keep the plain
	// byte-wise string sort.
	[Test]
	public async Task CSharpOnly_ListedImageNamesSortByPathElements()
	{
		string[] names = ["set-2/0.jpg", "set/0.jpg", "img.jpg", "img/1.jpg"];

		using var database = new InMemoryDatabase();
		var imageReader = new ImageReader(
			new ImageReaderOptions { Images = CreateImages(new Bitmap(10, 20, true), names) }, database);
		var order = new List<string>();
		for (int i = 0; i < names.Length; ++i)
		{
			imageReader.Next(out ImageReaderData data);
			order.Add(data.Image.Name);
		}

		await Assert.That(order.SequenceEqual(["img/1.jpg", "img.jpg", "set/0.jpg", "set-2/0.jpg"])).IsTrue();

		using var explicitDatabase = new InMemoryDatabase();
		var explicitReader = new ImageReader(
			new ImageReaderOptions { Images = CreateImages(new Bitmap(10, 20, true), names), ImageNames = [.. names] },
			explicitDatabase);
		order.Clear();
		for (int i = 0; i < names.Length; ++i)
		{
			explicitReader.Next(out ImageReaderData data);
			order.Add(data.Image.Name);
		}

		await Assert.That(order.SequenceEqual(["img.jpg", "img/1.jpg", "set-2/0.jpg", "set/0.jpg"])).IsTrue();
	}
}
