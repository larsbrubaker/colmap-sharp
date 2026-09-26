// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DatabaseTests.Pairs: the second half of colmap/scene/database_test.cc - the two-view
// geometry, Merge and LoadRandomDatabaseDescriptorsTest cases. See DatabaseTests.cs for the
// conventions. `EXPECT_EQ(a.F, b.F)` on std::optional<Matrix3d> is the lifted == of the
// nullable Matrix3d; isApprox is IsApprox with Eigen's dummy precision.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Tests.Feature;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public partial class DatabaseTests
{
	[Test]
	public async Task ParameterizedDatabaseTests_TwoViewGeometry()
	{
		using var database = new InMemoryDatabase();
		const uint ImageId1 = 1;
		const uint ImageId2 = 2;
		var twoViewGeometry = new TwoViewGeometry
		{
			InlierMatches = [.. new FeatureMatch[1000]],
			Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic,
			F = RandomEigen.RandomEigenMatrix3d(),
			E = RandomEigen.RandomEigenMatrix3d(),
			H = RandomEigen.RandomEigenMatrix3d(),
			Cam2FromCam1 = RandomRigid3d(),
			// Distinct cameras so the swap performed on inverse reads is observable.
			Camera1 = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100.0, 640, 480),
			Camera2 = Camera.CreateFromModelId(2, CameraModelId.Pinhole, 200.0, 800, 600),
		};
		// Drawn now, in COLMAP's order, since the PRNG is per thread and awaits may resume
		// elsewhere.
		Matrix3d noHE = RandomEigen.RandomEigenMatrix3d();
		Matrix3d noHF = RandomEigen.RandomEigenMatrix3d();
		database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometry);
		TwoViewGeometry read = database.ReadTwoViewGeometry(ImageId1, ImageId2);
		TwoViewGeometry readInv = database.ReadTwoViewGeometry(ImageId2, ImageId1);
		using (Assert.Multiple())
		{
			await Assert.That(read.InlierMatches.SequenceEqual(twoViewGeometry.InlierMatches)).IsTrue();
			await Assert.That(read.Config).IsEqualTo(twoViewGeometry.Config);
			await Assert.That(read.F == twoViewGeometry.F).IsTrue();
			await Assert.That(read.E == twoViewGeometry.E).IsTrue();
			await Assert.That(read.H == twoViewGeometry.H).IsTrue();
			await Assert.That(twoViewGeometry.Cam2FromCam1.HasValue).IsTrue();
			await Assert.That(read.Cam2FromCam1.HasValue).IsTrue();
			await Assert.That(read.Cam2FromCam1!.Value.Rotation.Coeffs == twoViewGeometry.Cam2FromCam1!.Value.Rotation.Coeffs).IsTrue();
			await Assert.That(read.Cam2FromCam1!.Value.Translation == twoViewGeometry.Cam2FromCam1!.Value.Translation).IsTrue();
			await Assert.That(read.Camera1 == twoViewGeometry.Camera1).IsTrue();
			await Assert.That(read.Camera2 == twoViewGeometry.Camera2).IsTrue();

			await Assert.That(readInv.InlierMatches.Count).IsEqualTo(read.InlierMatches.Count);
			for (int i = 0; i < read.InlierMatches.Count; ++i)
			{
				await Assert.That(readInv.InlierMatches[i].Point2DIdx2).IsEqualTo(read.InlierMatches[i].Point2DIdx1);
				await Assert.That(readInv.InlierMatches[i].Point2DIdx1).IsEqualTo(read.InlierMatches[i].Point2DIdx2);
			}

			await Assert.That(readInv.Config).IsEqualTo(read.Config);
			await Assert.That(readInv.F!.Value.Transpose() == read.F!.Value).IsTrue();
			await Assert.That(readInv.E!.Value.Transpose() == read.E!.Value).IsTrue();
			await Assert.That(readInv.H!.Value.Inverse().IsApprox(read.H!.Value)).IsTrue();
			await Assert.That(readInv.Cam2FromCam1.HasValue).IsTrue();
			Rigid3d inverse = read.Cam2FromCam1!.Value.Inverse();
			await Assert.That(readInv.Cam2FromCam1!.Value.Rotation.Coeffs.IsApprox(inverse.Rotation.Coeffs)).IsTrue();
			await Assert.That(readInv.Cam2FromCam1!.Value.Translation.IsApprox(inverse.Translation)).IsTrue();
			// The inverse read swaps the two cameras.
			await Assert.That(readInv.Camera1 == twoViewGeometry.Camera2).IsTrue();
			await Assert.That(readInv.Camera2 == twoViewGeometry.Camera1).IsTrue();

			var twoViewGeometries = database.ReadTwoViewGeometries();
			await Assert.That(twoViewGeometries.Count).IsEqualTo(1);
			await Assert.That(twoViewGeometries[0].PairId).IsEqualTo(ImagePairToPairId(ImageId1, ImageId2));
			TwoViewGeometry all0 = twoViewGeometries[0].TwoViewGeometry;
			await Assert.That(all0.Config).IsEqualTo(twoViewGeometry.Config);
			await Assert.That(all0.F == twoViewGeometry.F).IsTrue();
			await Assert.That(all0.E == twoViewGeometry.E).IsTrue();
			await Assert.That(all0.H == twoViewGeometry.H).IsTrue();
			await Assert.That(all0.Cam2FromCam1.HasValue).IsTrue();
			await Assert.That(all0.Cam2FromCam1!.Value.Rotation.Coeffs == twoViewGeometry.Cam2FromCam1!.Value.Rotation.Coeffs).IsTrue();
			await Assert.That(all0.Cam2FromCam1!.Value.Translation == twoViewGeometry.Cam2FromCam1!.Value.Translation).IsTrue();
			await Assert.That(all0.Camera1 == twoViewGeometry.Camera1).IsTrue();
			await Assert.That(all0.Camera2 == twoViewGeometry.Camera2).IsTrue();
			await Assert.That(all0.InlierMatches.Count).IsEqualTo(twoViewGeometry.InlierMatches.Count);
			var pairIdsAndNumInliers = database.ReadTwoViewGeometryNumInliers();
			await Assert.That(pairIdsAndNumInliers.Count).IsEqualTo(1);
			await Assert.That(pairIdsAndNumInliers[0].PairId).IsEqualTo(ImagePairToPairId(ImageId1, ImageId2));
			await Assert.That(pairIdsAndNumInliers[0].NumInliers).IsEqualTo(twoViewGeometry.InlierMatches.Count);
			await Assert.That(database.NumInlierMatches()).IsEqualTo(1000);
			database.DeleteInlierMatches(ImageId1, ImageId2);
			await Assert.That(database.ExistsTwoViewGeometry(ImageId1, ImageId2)).IsTrue();
			await Assert.That(database.NumInlierMatches()).IsEqualTo(0);
			database.DeleteTwoViewGeometry(ImageId1, ImageId2);
			await Assert.That(database.ExistsTwoViewGeometry(ImageId1, ImageId2)).IsFalse();
			await Assert.That(database.NumInlierMatches()).IsEqualTo(0);
			database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometry);
			await Assert.That(() => database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometry)).ThrowsException();
			await Assert.That(database.NumInlierMatches()).IsEqualTo(1000);
			database.ClearTwoViewGeometries();
			await Assert.That(database.NumInlierMatches()).IsEqualTo(0);
			twoViewGeometry.InlierMatches.Clear();
			database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometry);
			await Assert.That(database.ReadTwoViewGeometry(ImageId1, ImageId2).Cam2FromCam1 == twoViewGeometry.Cam2FromCam1).IsTrue();
		}

		// Test with E and F set, but H missing.
		database.ClearTwoViewGeometries();
		var twoViewGeometryNoH = new TwoViewGeometry
		{
			InlierMatches = [.. new FeatureMatch[10]],
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			E = noHE,
			F = noHF,
		};
		database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometryNoH);
		TwoViewGeometry noHRead = database.ReadTwoViewGeometry(ImageId1, ImageId2);
		using (Assert.Multiple())
		{
			await Assert.That(noHRead.E.HasValue).IsTrue();
			await Assert.That(noHRead.F.HasValue).IsTrue();
			await Assert.That(noHRead.E == twoViewGeometryNoH.E).IsTrue();
			await Assert.That(noHRead.F == twoViewGeometryNoH.F).IsTrue();
			await Assert.That(noHRead.H.HasValue).IsFalse();
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_TwoViewGeometryWithoutCameras()
	{
		using var database = new InMemoryDatabase();
		const uint ImageId1 = 1;
		const uint ImageId2 = 2;

		// A geometry that leaves the estimated cameras unset (as for configurations that
		// consume fixed intrinsics) round-trips them as null.
		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			E = RandomEigen.RandomEigenMatrix3d(),
		};
		await Assert.That(twoViewGeometry.Camera1).IsNull();
		await Assert.That(twoViewGeometry.Camera2).IsNull();
		database.WriteTwoViewGeometry(ImageId1, ImageId2, twoViewGeometry);

		TwoViewGeometry read = database.ReadTwoViewGeometry(ImageId1, ImageId2);
		var all = database.ReadTwoViewGeometries();
		using (Assert.Multiple())
		{
			await Assert.That(read.Camera1).IsNull();
			await Assert.That(read.Camera2).IsNull();
		}

		await Assert.That(all.Count).IsEqualTo(1);
		using (Assert.Multiple())
		{
			await Assert.That(all[0].TwoViewGeometry.Camera1).IsNull();
			await Assert.That(all[0].TwoViewGeometry.Camera2).IsNull();
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Merge()
	{
		using var database1 = new InMemoryDatabase();
		using var database2 = new InMemoryDatabase();

		// This test intentionally uses custom, large, partially overlapping IDs from
		// rigs/frames/images/cameras which then require remapping of the IDs. This is to
		// ensure that the database can handle this case.

		Camera camera1 = SimplePinhole();
		camera1.CameraId = 50;
		database1.WriteCamera(camera1, useCameraId: true);
		Camera camera2 = SimplePinhole();
		camera2.CameraId = 60;
		database1.WriteCamera(camera2, useCameraId: true);
		Camera camera3 = SimplePinhole();
		camera3.CameraId = 55;
		database2.WriteCamera(camera3, useCameraId: true);
		Camera camera4 = SimplePinhole();
		camera4.CameraId = 60;
		database2.WriteCamera(camera4, useCameraId: true);

		var rig1 = new Rig { RigId = 100 };
		rig1.AddRefSensor(camera1.SensorId);
		rig1.AddSensor(camera2.SensorId, new Rigid3d());
		database1.WriteRig(rig1, useRigId: true);

		var rig2 = new Rig { RigId = 200 };
		rig2.AddRefSensor(camera3.SensorId);
		rig2.AddSensor(camera4.SensorId, new Rigid3d());
		database2.WriteRig(rig2, useRigId: true);

		const uint ImageId1 = 300;
		const uint ImageId2 = 400;
		const uint ImageId3 = 350;
		const uint ImageId4 = 400;

		var image = new Image { ImageId = ImageId1, Name = "test1" };
		image.SetCameraId(camera1.CameraId);
		database1.WriteImage(image, useImageId: true);
		image.ImageId = ImageId2;
		image.SetCameraId(camera2.CameraId);
		image.Name = "test2";
		database1.WriteImage(image, useImageId: true);

		image.ImageId = ImageId3;
		image.SetCameraId(camera3.CameraId);
		image.Name = "test3";
		database2.WriteImage(image, useImageId: true);
		image.ImageId = ImageId4;
		image.SetCameraId(camera4.CameraId);
		image.Name = "test4";
		database2.WriteImage(image, useImageId: true);

		var frame1 = new Frame();
		frame1.SetRigId(rig1.RigId);
		frame1.AddDataId(new DataId(camera1.SensorId, ImageId1));
		frame1.AddDataId(new DataId(camera2.SensorId, ImageId2));
		frame1.FrameId = 1000;
		database1.WriteFrame(frame1, useFrameId: true);
		var frame2 = new Frame();
		frame2.SetRigId(rig2.RigId);
		frame2.AddDataId(new DataId(camera3.SensorId, ImageId3));
		frame2.AddDataId(new DataId(camera4.SensorId, ImageId4));
		frame2.FrameId = 2000;
		database2.WriteFrame(frame2, useFrameId: true);

		var posePrior1 = new PosePrior
		{
			CorrDataId = new DataId(camera1.SensorId, ImageId1),
			Position = RandomEigen.RandomEigenVector3d(),
		};
		posePrior1.PosePriorId = database1.WritePosePrior(posePrior1);

		var posePrior2 = new PosePrior
		{
			CorrDataId = new DataId(camera3.SensorId, ImageId3),
			Position = RandomEigen.RandomEigenVector3d(),
		};
		posePrior2.PosePriorId = database2.WritePosePrior(posePrior2);

		List<FeatureKeypoint> keypoints1 = KeypointsWithFirstX(10, 100);
		List<FeatureKeypoint> keypoints2 = KeypointsWithFirstX(20, 200);
		List<FeatureKeypoint> keypoints3 = KeypointsWithFirstX(30, 300);
		List<FeatureKeypoint> keypoints4 = KeypointsWithFirstX(40, 400);

		var descriptors1 = new FeatureDescriptors(FeatureExtractorType.Undefined, FeatureTypesTests.RandomBytes(10, 128, 1));
		var descriptors2 = new FeatureDescriptors(FeatureExtractorType.Undefined, FeatureTypesTests.RandomBytes(20, 128, 2));
		var descriptors3 = new FeatureDescriptors(FeatureExtractorType.Undefined, FeatureTypesTests.RandomBytes(30, 128, 3));
		var descriptors4 = new FeatureDescriptors(FeatureExtractorType.Undefined, FeatureTypesTests.RandomBytes(40, 128, 4));

		database1.WriteKeypoints(ImageId1, keypoints1);
		database1.WriteKeypoints(ImageId2, keypoints2);
		database2.WriteKeypoints(ImageId3, keypoints3);
		database2.WriteKeypoints(ImageId4, keypoints4);
		database1.WriteDescriptors(ImageId1, descriptors1);
		database1.WriteDescriptors(ImageId2, descriptors2);
		database2.WriteDescriptors(ImageId3, descriptors3);
		database2.WriteDescriptors(ImageId4, descriptors4);
		database1.WriteMatches(ImageId1, ImageId2, new FeatureMatch[10]);
		database2.WriteMatches(ImageId3, ImageId4, new FeatureMatch[10]);
		database1.WriteTwoViewGeometry(ImageId1, ImageId2, new TwoViewGeometry());
		database2.WriteTwoViewGeometry(ImageId3, ImageId4, new TwoViewGeometry());

		using var mergedDatabase = new InMemoryDatabase();
		Database.Merge(database1, database2, mergedDatabase);
		using (Assert.Multiple())
		{
			await Assert.That(mergedDatabase.NumRigs()).IsEqualTo(2);
			await Assert.That(mergedDatabase.NumCameras()).IsEqualTo(4);
			await Assert.That(mergedDatabase.NumFrames()).IsEqualTo(2);
			await Assert.That(mergedDatabase.NumImages()).IsEqualTo(4);
			await Assert.That(mergedDatabase.NumPosePriors()).IsEqualTo(2);
			await Assert.That(mergedDatabase.NumKeypoints()).IsEqualTo(100);
			await Assert.That(mergedDatabase.NumDescriptors()).IsEqualTo(100);
			await Assert.That(mergedDatabase.NumMatches()).IsEqualTo(20);
			await Assert.That(mergedDatabase.NumInlierMatches()).IsEqualTo(0);
			await Assert.That(mergedDatabase.ReadAllFrames()[0].NumDataIds).IsEqualTo(frame1.NumDataIds);
			await Assert.That(mergedDatabase.ReadAllFrames()[1].NumDataIds).IsEqualTo(frame2.NumDataIds);
			foreach (Frame frame in mergedDatabase.ReadAllFrames())
			{
				foreach (DataId dataId in frame.DataIds)
				{
					await Assert.That(dataId.SensorId.Type).IsEqualTo(SensorType.Camera);
					await Assert.That(mergedDatabase.ExistsCamera(dataId.SensorId.Id)).IsTrue();
					await Assert.That(mergedDatabase.ExistsImage((uint)dataId.Id)).IsTrue();
				}
			}

			foreach (PosePrior posePrior in mergedDatabase.ReadAllPosePriors())
			{
				await Assert.That(posePrior.CorrDataId.SensorId.Type).IsEqualTo(SensorType.Camera);
				await Assert.That(mergedDatabase.ExistsCamera(posePrior.CorrDataId.SensorId.Id)).IsTrue();
				await Assert.That(mergedDatabase.ExistsImage((uint)posePrior.CorrDataId.Id)).IsTrue();
			}

			List<Image> mergedImages = mergedDatabase.ReadAllImages();
			await Assert.That(mergedImages[0].CameraId).IsEqualTo(1u);
			await Assert.That(mergedImages[1].CameraId).IsEqualTo(2u);
			await Assert.That(mergedImages[2].CameraId).IsEqualTo(3u);
			await Assert.That(mergedImages[3].CameraId).IsEqualTo(4u);
			await Assert.That(mergedDatabase.ReadKeypoints(1).Count).IsEqualTo(10);
			await Assert.That(mergedDatabase.ReadKeypoints(2).Count).IsEqualTo(20);
			await Assert.That(mergedDatabase.ReadKeypoints(3).Count).IsEqualTo(30);
			await Assert.That(mergedDatabase.ReadKeypoints(4).Count).IsEqualTo(40);
			await Assert.That(mergedDatabase.ReadKeypoints(1)[0].X).IsEqualTo(100f);
			await Assert.That(mergedDatabase.ReadKeypoints(2)[0].X).IsEqualTo(200f);
			await Assert.That(mergedDatabase.ReadKeypoints(3)[0].X).IsEqualTo(300f);
			await Assert.That(mergedDatabase.ReadKeypoints(4)[0].X).IsEqualTo(400f);
			await Assert.That(mergedDatabase.ReadDescriptors(1).Type).IsEqualTo(descriptors1.Type);
			await Assert.That(mergedDatabase.ReadDescriptors(1).Data.Size).IsEqualTo(descriptors1.Data.Size);
			await Assert.That(mergedDatabase.ReadDescriptors(2).Type).IsEqualTo(descriptors2.Type);
			await Assert.That(mergedDatabase.ReadDescriptors(2).Data.Size).IsEqualTo(descriptors2.Data.Size);
			await Assert.That(mergedDatabase.ReadDescriptors(3).Type).IsEqualTo(descriptors3.Type);
			await Assert.That(mergedDatabase.ReadDescriptors(3).Data.Size).IsEqualTo(descriptors3.Data.Size);
			await Assert.That(mergedDatabase.ReadDescriptors(4).Type).IsEqualTo(descriptors4.Type);
			await Assert.That(mergedDatabase.ReadDescriptors(4).Data.Size).IsEqualTo(descriptors4.Data.Size);
			await Assert.That(mergedDatabase.ExistsMatches(1, 2)).IsTrue();
			await Assert.That(mergedDatabase.ExistsMatches(2, 3)).IsFalse();
			await Assert.That(mergedDatabase.ExistsMatches(2, 4)).IsFalse();
			await Assert.That(mergedDatabase.ExistsMatches(3, 4)).IsTrue();

			mergedDatabase.ClearAllTables();
			await Assert.That(mergedDatabase.NumRigs()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumCameras()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumFrames()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumImages()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumPosePriors()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumKeypoints()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumDescriptors()).IsEqualTo(0);
			await Assert.That(mergedDatabase.NumMatches()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadEmpty()
	{
		using var database = new InMemoryDatabase();
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, -1);
		using (Assert.Multiple())
		{
			await Assert.That(result.Data.Rows).IsEqualTo(0);
			await Assert.That(result.Data.Cols).IsEqualTo(0);
			await Assert.That(result.Type).IsEqualTo(FeatureExtractorType.Undefined);
		}
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadAll()
	{
		using Database database = CreateDatabaseWithRandomDescriptors([10, 20, 30]);
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, -1);
		await ExpectDescriptors(result, 60, FeatureExtractorType.Sift);
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadAllWithLargeMax()
	{
		using Database database = CreateDatabaseWithRandomDescriptors([15, 15]);
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, 1000);
		using (Assert.Multiple())
		{
			await Assert.That(result.Data.Rows).IsEqualTo(30);
			await Assert.That(result.Data.Cols).IsEqualTo(128);
		}
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadSubset()
	{
		using Database database = CreateDatabaseWithRandomDescriptors([10, 20, 30]);
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, 10);
		await ExpectDescriptors(result, 10, FeatureExtractorType.Sift);
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadSubsetWithSomeEmpty()
	{
		using Database database = CreateDatabaseWithRandomDescriptors([0, 10, 0, 15, 0, 20, 0]);
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, 15);
		await ExpectDescriptors(result, 15, FeatureExtractorType.Sift);
	}

	[Test]
	public async Task LoadRandomDatabaseDescriptorsTest_LoadExactTotal()
	{
		using Database database = CreateDatabaseWithRandomDescriptors([0, 10, 0, 10, 0]);
		FeatureDescriptorsFloat result = Database.LoadRandomDatabaseDescriptors(database, 20);
		using (Assert.Multiple())
		{
			await Assert.That(result.Data.Rows).IsEqualTo(20);
			await Assert.That(result.Data.Cols).IsEqualTo(128);
		}
	}

	// Helper to create a database with images and descriptors.
	private static InMemoryDatabase CreateDatabaseWithRandomDescriptors(int[] numDescriptorsPerImage)
	{
		var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		for (int i = 0; i < numDescriptorsPerImage.Length; ++i)
		{
			var image = new Image { Name = "image" + i };
			image.SetCameraId(camera.CameraId);
			image.ImageId = database.WriteImage(image);
			database.WriteDescriptors(
				image.ImageId,
				new FeatureDescriptors(FeatureExtractorType.Sift, FeatureTypesTests.RandomBytes(numDescriptorsPerImage[i], 128, i)));
		}

		return database;
	}

	private static async Task ExpectDescriptors(FeatureDescriptorsFloat result, int rows, FeatureExtractorType type)
	{
		using (Assert.Multiple())
		{
			await Assert.That(result.Data.Rows).IsEqualTo(rows);
			await Assert.That(result.Data.Cols).IsEqualTo(128);
			await Assert.That(result.Type).IsEqualTo(type);
		}
	}

	private static List<FeatureKeypoint> KeypointsWithFirstX(int count, float x)
	{
		List<FeatureKeypoint> keypoints = FeatureKeypoints.Create(count);
		FeatureKeypoint first = keypoints[0];
		first.X = x;
		keypoints[0] = first;
		return keypoints;
	}
}
