// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RigConfigTests: colmap/scene/rig_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Scene/RigConfig.cs and RigConfig.Apply.cs. The
// SQLite in-memory database is InMemoryDatabase; EXPECT_ANY_THROW is Assert.Throws<Exception>.
// All cases are Tier A (bookkeeping; the only arithmetic is copied through). C#-only cases
// are in RigConfigTests.CSharpOnly.cs.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test that draws does so before its first await (the PRNG is per thread and an
// await may resume elsewhere).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class RigConfigTests
{
	private static string WriteTestConfig(string config)
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-rig-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		string filePath = Path.Combine(dir, "config.json");
		File.WriteAllText(filePath, config + "\n");
		return filePath;
	}

	[Test]
	public async Task ReadRigConfig_Empty()
	{
		await Assert.That(RigConfig.ReadRigConfig(WriteTestConfig("[]"))).IsEmpty();
	}

	[Test]
	public async Task ReadRigConfig_InvalidJson()
	{
		await Assert.That(() => RigConfig.ReadRigConfig(WriteTestConfig(""))).Throws<Exception>();
		await Assert.That(() => RigConfig.ReadRigConfig(WriteTestConfig("[{"))).Throws<Exception>();
	}

	[Test]
	public async Task ReadRigConfig_MissingImagePrefix()
	{
		await Assert.That(() => RigConfig.ReadRigConfig(WriteTestConfig("""

          [
            {
              "cameras": [
                {
                    "ref_sensor": true,
                }
              ]
            }
          ]

""")))
			.Throws<Exception>();
	}

	[Test]
	public async Task ReadRigConfig_InvalidRefSensor()
	{
		await Assert.That(() => RigConfig.ReadRigConfig(WriteTestConfig("""

          [
            {
              "cameras": [
                {
                    "image_prefix": "rig1/camera1/",
                    "ref_sensor": true,
                },
                {
                    "image_prefix": "rig1/camera2/",
                    "ref_sensor": true,
                }
              ]
            }
          ]

""")))
			.Throws<Exception>();
		await Assert.That(() => RigConfig.ReadRigConfig(WriteTestConfig("""

              [
                {
                  "cameras": [
                    {
                        "image_prefix": "rig1/camera1/",
                    },
                    {
                        "image_prefix": "rig1/camera2/",
                    }
                  ]
                }
              ]

""")))
			.Throws<Exception>();
	}

	[Test]
	public async Task ReadRigConfig_Nominal()
	{
		List<RigConfig> configs = RigConfig.ReadRigConfig(WriteTestConfig("""

[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0, 1, 0, 0],
          "cam_from_rig_translation": [1, 2, 3]
      }
    ]
  },
  {
    "cameras": [
      {
          "image_prefix": "rig2/camera1/",
          "ref_sensor": true
      },
      {
          "image_prefix": "rig2/camera2/"
      }
    ]
  }
]

"""));
		await Assert.That(configs.Count).IsEqualTo(2);
		await Assert.That(configs[0].Cameras.Count).IsEqualTo(2);
		await Assert.That(configs[1].Cameras.Count).IsEqualTo(2);

		using (Assert.Multiple())
		{
			await Assert.That(configs[0].Cameras[0].ImagePrefix).IsEqualTo("rig1/camera1/");
			await Assert.That(configs[0].Cameras[0].RefSensor).IsTrue();
			await Assert.That(configs[0].Cameras[0].CamFromRig.HasValue).IsFalse();
			await Assert.That(configs[0].Cameras[0].Camera).IsNotNull();
			await Assert.That(configs[0].Cameras[0].Camera!.ModelId).IsEqualTo(CameraModelId.OpenCV);
			await Assert.That(configs[0].Cameras[0].Camera!.HasPriorFocalLength).IsTrue();
			await Assert.That(configs[0].Cameras[0].Camera!.Params)
				.IsEquivalentTo(new double[] { 640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4 }, CollectionOrdering.Matching);

			await Assert.That(configs[0].Cameras[1].ImagePrefix).IsEqualTo("rig1/camera2/");
			await Assert.That(configs[0].Cameras[1].RefSensor).IsFalse();
			await Assert.That(configs[0].Cameras[1].CamFromRig.HasValue).IsTrue();
			await Assert.That(configs[0].Cameras[1].CamFromRig!.Value.Rotation.Coeffs).IsEqualTo(new Vector4d(1, 0, 0, 0));
			await Assert.That(configs[0].Cameras[1].CamFromRig!.Value.Translation).IsEqualTo(new Vector3d(1, 2, 3));
			await Assert.That(configs[0].Cameras[1].Camera).IsNull();

			await Assert.That(configs[1].Cameras[0].ImagePrefix).IsEqualTo("rig2/camera1/");
			await Assert.That(configs[1].Cameras[0].RefSensor).IsTrue();
			await Assert.That(configs[1].Cameras[0].CamFromRig.HasValue).IsFalse();
			await Assert.That(configs[1].Cameras[0].Camera).IsNull();

			await Assert.That(configs[1].Cameras[1].ImagePrefix).IsEqualTo("rig2/camera2/");
			await Assert.That(configs[1].Cameras[1].RefSensor).IsFalse();
			await Assert.That(configs[1].Cameras[1].CamFromRig.HasValue).IsFalse();
			await Assert.That(configs[1].Cameras[1].Camera).IsNull();
		}
	}

	private static void CreateTestData(int numFrames, int numCamerasPerRig, Database database, Reconstruction reconstruction)
	{
		// Create a synthetic dataset with a single rig and trivial frames.
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFrames * numCamerasPerRig,
		};
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		// Manually overwrite the image names to match the expected rig config.
		int imageIdx = 0;
		int frameIdx = 0;
		foreach (Image image in database.ReadAllImages())
		{
			if (imageIdx++ % numCamerasPerRig == 0)
			{
				++frameIdx;
			}

			image.Name = "camera" + (imageIdx % numCamerasPerRig) + "_frame" + frameIdx;
			reconstruction.Image(image.ImageId).Name = image.Name;
			database.UpdateImage(image);
		}
	}

	private static List<RigConfig> TwoCameraConfig(out RigConfig.RigCamera camera2)
	{
		var configs = new List<RigConfig>();
		var config = new RigConfig();
		configs.Add(config);
		config.Cameras.Add(new RigConfig.RigCamera { ImagePrefix = "camera0_", RefSensor = true });
		camera2 = new RigConfig.RigCamera { ImagePrefix = "camera1_" };
		config.Cameras.Add(camera2);
		return configs;
	}

	[Test]
	public async Task ApplyRigConfig_WithReconstruction()
	{
		using var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		CreateTestData(numFrames: 5, numCamerasPerRig: 2, database, reconstruction);

		List<RigConfig> configs = TwoCameraConfig(out _);

		RigConfig.ApplyRigConfig(configs, database, reconstruction);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumRigs()).IsEqualTo(1);
			await Assert.That(database.NumFrames()).IsEqualTo(5);
			await Assert.That(reconstruction.NumRigs).IsEqualTo(1);
			await Assert.That(reconstruction.NumFrames).IsEqualTo(5);
			await Assert.That(reconstruction.NumRegFrames).IsEqualTo(5);
		}
	}

	[Test]
	public async Task ApplyRigConfig_WithDifferingDatabaseAndReconstructionIds()
	{
		using var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		CreateTestData(numFrames: 5, numCamerasPerRig: 2, database, reconstruction);

		// Create a reconstruction with differing rig/camera/frame/image ids.
		var differingReconstruction = new Reconstruction();
		foreach (Camera camera in reconstruction.Cameras.Values)
		{
			Camera differingCamera = camera.Clone();
			differingCamera.CameraId = camera.CameraId + 1;
			differingReconstruction.AddCamera(differingCamera);
		}

		foreach (Rig rig in reconstruction.Rigs.Values)
		{
			var differingRig = new Rig { RigId = rig.RigId + 1 };
			differingRig.AddRefSensor(new SensorId(rig.RefSensorId.Type, rig.RefSensorId.Id + 1));
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				differingRig.AddSensor(new SensorId(sensorId.Type, sensorId.Id + 1), sensorFromRig);
			}

			differingReconstruction.AddRig(differingRig);
		}

		foreach (Frame frame in reconstruction.Frames.Values)
		{
			Frame differingFrame = frame.Clone();
			differingFrame.ResetRigPtr();
			differingFrame.FrameId = frame.FrameId + 1;
			differingFrame.SetRigId(frame.RigId + 1);
			differingFrame.ClearDataIds();
			foreach (DataId dataId in frame.DataIds)
			{
				differingFrame.AddDataId(new DataId(
					new SensorId(dataId.SensorId.Type, dataId.SensorId.Id + 1), (uint)dataId.Id + 1));
			}

			differingReconstruction.AddFrame(differingFrame);
		}

		foreach (Image image in reconstruction.Images.Values)
		{
			Image differingImage = image.Clone();
			differingImage.ResetCameraPtr();
			differingImage.ResetFramePtr();
			differingImage.ImageId = image.ImageId + 1;
			differingImage.SetCameraId(image.CameraId + 1);
			differingImage.SetFrameId(image.FrameId + 1);
			differingReconstruction.AddImage(differingImage);
		}

		List<RigConfig> configs = TwoCameraConfig(out _);

		RigConfig.ApplyRigConfig(configs, database, differingReconstruction);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumRigs()).IsEqualTo(1);
			await Assert.That(database.NumFrames()).IsEqualTo(5);
			await Assert.That(differingReconstruction.NumRigs).IsEqualTo(1);
			await Assert.That(differingReconstruction.NumFrames).IsEqualTo(5);
			await Assert.That(differingReconstruction.NumRegFrames).IsEqualTo(5);
		}
	}

	[Test]
	public async Task ApplyRigConfig_WithPartialReconstruction()
	{
		using var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		CreateTestData(numFrames: 5, numCamerasPerRig: 2, database, reconstruction);

		reconstruction.DeRegisterFrame(1);
		reconstruction.DeRegisterFrame(2);
		// De-register only one image in the frame.
		// The other image allows us to register the frame.
		reconstruction.DeRegisterFrame(3);

		List<RigConfig> configs = TwoCameraConfig(out _);

		RigConfig.ApplyRigConfig(configs, database, reconstruction);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumRigs()).IsEqualTo(1);
			await Assert.That(database.NumFrames()).IsEqualTo(5);
			await Assert.That(reconstruction.NumRigs).IsEqualTo(1);
			await Assert.That(reconstruction.NumFrames).IsEqualTo(5);
			await Assert.That(reconstruction.NumRegFrames).IsEqualTo(4);
		}
	}

	[Test]
	public async Task ApplyRigConfig_WithoutReconstruction()
	{
		using var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		CreateTestData(numFrames: 5, numCamerasPerRig: 2, database, reconstruction);

		List<RigConfig> configs = TwoCameraConfig(out RigConfig.RigCamera camera2);
		camera2.Camera = Camera.CreateFromModelId(2, CameraModelId.OpenCV, 2.0, 1024, 768);
		camera2.CamFromRig = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());

		RigConfig.ApplyRigConfig(configs, database);
		(SensorId sensorId2, Rigid3d? sensor2FromRig) = database.ReadAllRigs()[0].NonRefSensors.First();
		using (Assert.Multiple())
		{
			await Assert.That(database.NumRigs()).IsEqualTo(1);
			await Assert.That(database.NumFrames()).IsEqualTo(5);
			await Assert.That(sensor2FromRig!.Value.Rotation.Coeffs).IsEqualTo(camera2.CamFromRig!.Value.Rotation.Coeffs);
			await Assert.That(sensor2FromRig!.Value.Translation).IsEqualTo(camera2.CamFromRig!.Value.Translation);
			await Assert.That(database.ReadCamera((uint)sensorId2.Id)).IsEqualTo(camera2.Camera);
		}
	}

	[Test]
	public async Task ApplyRigConfig_WithUnconfiguredSingleAndConfiguredMultiCameraRigs()
	{
		using var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 3,
			NumFramesPerRig = 5,
		};
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		options.NumRigs = 1;
		options.NumCamerasPerRig = 1;
		options.NumFramesPerRig = 3;
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		var configs = new List<RigConfig>();
		var config = new RigConfig();
		configs.Add(config);
		config.Cameras.Add(new RigConfig.RigCamera { ImagePrefix = "camera000001_", RefSensor = true });
		config.Cameras.Add(new RigConfig.RigCamera { ImagePrefix = "camera000002_" });

		RigConfig.ApplyRigConfig(configs, database, reconstruction);

		int numNonTrivialRigs = database.ReadAllRigs().Count(rig => rig.NumSensors > 1);
		int numNonTrivialFrames = database.ReadAllFrames().Count(frame => frame.NumDataIds > 1);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumRigs()).IsEqualTo(3);
			await Assert.That(database.NumFrames()).IsEqualTo(13);
			await Assert.That(reconstruction.NumRigs).IsEqualTo(3);
			await Assert.That(reconstruction.NumFrames).IsEqualTo(13);
			await Assert.That(reconstruction.NumRegFrames).IsEqualTo(13);
			await Assert.That(numNonTrivialRigs).IsEqualTo(1);
			await Assert.That(numNonTrivialFrames).IsEqualTo(5);
		}
	}
}
