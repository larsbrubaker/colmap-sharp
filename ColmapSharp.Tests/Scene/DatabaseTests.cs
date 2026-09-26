// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DatabaseTests: colmap/scene/database_test.cc ported 1:1 against InMemoryDatabase, one
// method per gtest case named Suite_Name (the ParameterizedDatabaseTests cases run on the
// single in-memory implementation). Tests ColmapSharp/Scene/Database.cs,
// Database.Merge.cs and InMemoryDatabase*.cs.
//
// Skipped (SQLite database files are out of scope, PORTING_PLAN.md): OpenFile,
// OpenCloseFile, OpenFileWithNonASCIIPath. OpenInMemory / OpenCloseInMemory construct an
// InMemoryDatabase.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and tests drawing random poses/matrices draw before their first await.
// FeatureDescriptorsData::Random is FeatureTypesTests.RandomBytes (the values don't matter).
// EXPECT_ANY_THROW is ThrowsException().

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
	private static Rigid3d RandomRigid3d() =>
		new(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());

	private static Camera SimplePinhole() =>
		Camera.CreateFromModelId(InvalidCameraId, CameraModelId.SimplePinhole, 1.0, 1, 1);

	[Test]
	public async Task ParameterizedDatabaseTests_OpenInMemory()
	{
		using var database = new InMemoryDatabase();
		await Assert.That(database.NumImages()).IsEqualTo(0);
	}

	[Test]
	public async Task ParameterizedDatabaseTests_OpenCloseInMemory()
	{
		var database = new InMemoryDatabase();
		database.Close();
		// Any database operation after closing the database should fail.
		await Assert.That(() => database.ExistsCamera(42)).ThrowsException();
		database.Close();
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Transaction()
	{
		using var database = new InMemoryDatabase();
		using (new DatabaseTransaction(database))
		{
		}

		await Assert.That(database.NumImages()).IsEqualTo(0);
	}

	[Test]
	public async Task ParameterizedDatabaseTests_TransactionMultiThreaded()
	{
		using var database = new InMemoryDatabase();
		const int NumThreads = 3;
		var threads = new List<Thread>(NumThreads);
		for (int i = 0; i < NumThreads; ++i)
		{
			threads.Add(new Thread(() =>
			{
				using var transaction = new DatabaseTransaction(database);
				Thread.Sleep(10);
			}));
		}

		threads.ForEach(thread => thread.Start());
		threads.ForEach(thread => thread.Join());
		await Assert.That(database.NumImages()).IsEqualTo(0);
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Empty()
	{
		using var database = new InMemoryDatabase();
		using (Assert.Multiple())
		{
			await Assert.That(database.NumCameras()).IsEqualTo(0);
			await Assert.That(database.NumFrames()).IsEqualTo(0);
			await Assert.That(database.NumImages()).IsEqualTo(0);
			await Assert.That(database.NumKeypoints()).IsEqualTo(0);
			await Assert.That(database.MaxNumKeypoints()).IsEqualTo(0);
			await Assert.That(database.NumDescriptors()).IsEqualTo(0);
			await Assert.That(database.MaxNumDescriptors()).IsEqualTo(0);
			await Assert.That(database.NumMatches()).IsEqualTo(0);
			await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(0);
			await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Rig()
	{
		using var database = new InMemoryDatabase();
		long numRigs0 = database.NumRigs();
		var rig = new Rig();
		rig.AddRefSensor(new SensorId(SensorType.Camera, 1));
		rig.RigId = database.WriteRig(rig);
		long numRigs1 = database.NumRigs();
		bool exists1 = database.ExistsRig(rig.RigId);
		bool readEqual1 = database.ReadRig(rig.RigId) == rig;

		database.ClearRigs();
		long numRigsCleared = database.NumRigs();

		rig.AddSensor(new SensorId(SensorType.Camera, 2), RandomRigid3d());
		rig.AddSensor(new SensorId(SensorType.Imu, 3));
		rig.AddSensor(new SensorId(SensorType.Imu, 4), RandomRigid3d());
		rig.RigId = database.WriteRig(rig);
		long numRigs2 = database.NumRigs();
		bool exists2 = database.ExistsRig(rig.RigId);
		bool readEqual2 = database.ReadRig(rig.RigId) == rig;
		bool withCamera1 = database.ReadRigWithSensor(new SensorId(SensorType.Camera, 1)) == rig;
		bool withImu4 = database.ReadRigWithSensor(new SensorId(SensorType.Imu, 4)) == rig;
		bool withImu42 = database.ReadRigWithSensor(new SensorId(SensorType.Imu, 42)) is null;
		rig.SetSensorFromRig(new SensorId(SensorType.Camera, 2), RandomRigid3d());
		database.UpdateRig(rig);
		bool readEqual3 = database.ReadRig(rig.RigId) == rig;
		var rig2 = new Rig();
		rig2.AddRefSensor(new SensorId(SensorType.Imu, 10));
		rig2.RigId = rig.RigId + 1;
		database.WriteRig(rig2, useRigId: true);
		long numRigs3 = database.NumRigs();
		List<Rig> allRigs = database.ReadAllRigs();

		using (Assert.Multiple())
		{
			await Assert.That(numRigs0).IsEqualTo(0);
			await Assert.That(numRigs1).IsEqualTo(1);
			await Assert.That(exists1).IsTrue();
			await Assert.That(readEqual1).IsTrue();
			await Assert.That(numRigsCleared).IsEqualTo(0);
			await Assert.That(numRigs2).IsEqualTo(1);
			await Assert.That(exists2).IsTrue();
			await Assert.That(readEqual2).IsTrue();
			await Assert.That(withCamera1).IsTrue();
			await Assert.That(withImu4).IsTrue();
			await Assert.That(withImu42).IsTrue();
			await Assert.That(readEqual3).IsTrue();
			await Assert.That(numRigs3).IsEqualTo(2);
			await Assert.That(database.ExistsRig(rig.RigId)).IsTrue();
			await Assert.That(database.ExistsRig(rig2.RigId)).IsTrue();
			await Assert.That(allRigs.Count).IsEqualTo(2);
			await Assert.That(allRigs[0].RigId).IsEqualTo(rig.RigId);
			await Assert.That(allRigs[1].RigId).IsEqualTo(rig2.RigId);
			database.ClearRigs();
			await Assert.That(database.NumRigs()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Camera()
	{
		using var database = new InMemoryDatabase();
		using (Assert.Multiple())
		{
			await Assert.That(database.NumCameras()).IsEqualTo(0);
			Camera camera = SimplePinhole();
			camera.CameraId = database.WriteCamera(camera);
			await Assert.That(database.NumCameras()).IsEqualTo(1);
			await Assert.That(database.ExistsCamera(camera.CameraId)).IsTrue();
			await Assert.That(database.ReadCamera(camera.CameraId).CameraId).IsEqualTo(camera.CameraId);
			await Assert.That(database.ReadCamera(camera.CameraId).ModelId).IsEqualTo(camera.ModelId);
			await Assert.That(database.ReadCamera(camera.CameraId) == camera).IsTrue();
			camera.SetFocalLength(2 * camera.FocalLength());
			database.UpdateCamera(camera);
			await Assert.That(database.ReadCamera(camera.CameraId) == camera).IsTrue();
			Camera camera2 = camera.Clone();
			camera2.CameraId = camera.CameraId + 1;
			database.WriteCamera(camera2, true);
			await Assert.That(database.NumCameras()).IsEqualTo(2);
			await Assert.That(database.ExistsCamera(camera.CameraId)).IsTrue();
			await Assert.That(database.ExistsCamera(camera2.CameraId)).IsTrue();
			List<Camera> allCameras = database.ReadAllCameras();
			await Assert.That(allCameras.Count).IsEqualTo(2);
			await Assert.That(allCameras[0] == camera && allCameras[1] == camera2).IsTrue();
			database.ClearCameras();
			await Assert.That(database.NumCameras()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Frame()
	{
		using var database = new InMemoryDatabase();
		var rig = new Rig();
		rig.AddRefSensor(new SensorId(SensorType.Camera, 1));
		rig.RigId = database.WriteRig(rig);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumFrames()).IsEqualTo(0);

			var frame = new Frame();
			frame.SetRigId(rig.RigId);
			frame.FrameId = database.WriteFrame(frame);
			await Assert.That(database.NumFrames()).IsEqualTo(1);
			await Assert.That(database.ExistsFrame(frame.FrameId)).IsTrue();
			await Assert.That(database.ReadFrame(frame.FrameId) == frame).IsTrue();

			database.ClearFrames();
			await Assert.That(database.NumFrames()).IsEqualTo(0);

			frame.AddDataId(new DataId(new SensorId(SensorType.Imu, 1), 2));
			frame.AddDataId(new DataId(new SensorId(SensorType.Camera, 1), 3));
			frame.FrameId = database.WriteFrame(frame);
			await Assert.That(database.NumFrames()).IsEqualTo(1);
			await Assert.That(database.ExistsFrame(frame.FrameId)).IsTrue();
			await Assert.That(database.ReadFrame(frame.FrameId) == frame).IsTrue();

			frame.AddDataId(new DataId(new SensorId(SensorType.Camera, 2), 4));
			database.UpdateFrame(frame);
			await Assert.That(database.ReadFrame(frame.FrameId) == frame).IsTrue();
			var frame2 = new Frame();
			frame2.SetRigId(rig.RigId);
			frame2.AddDataId(new DataId(new SensorId(SensorType.Camera, 2), 5));
			frame2.FrameId = frame.FrameId + 1;
			database.WriteFrame(frame2, useFrameId: true);
			await Assert.That(database.NumFrames()).IsEqualTo(2);
			await Assert.That(database.ExistsFrame(frame.FrameId)).IsTrue();
			await Assert.That(database.ExistsFrame(frame2.FrameId)).IsTrue();
			List<Frame> allFrames = database.ReadAllFrames();
			await Assert.That(allFrames.Count).IsEqualTo(2);
			await Assert.That(allFrames[0].FrameId).IsEqualTo(frame.FrameId);
			await Assert.That(allFrames[1].FrameId).IsEqualTo(frame2.FrameId);

			database.ClearFrames();
			await Assert.That(database.NumFrames()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Image()
	{
		using var database = new InMemoryDatabase();
		Camera camera = SimplePinhole();
		camera.CameraId = database.WriteCamera(camera);
		var rig = new Rig();
		rig.AddRefSensor(new SensorId(SensorType.Camera, camera.CameraId));
		rig.RigId = database.WriteRig(rig);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumImages()).IsEqualTo(0);
			var image = new Image { Name = "test" };
			image.SetCameraId(camera.CameraId);
			image.ImageId = database.WriteImage(image);
			var frame = new Frame();
			frame.SetRigId(rig.RigId);
			frame.AddDataId(image.DataId);
			frame.FrameId = database.WriteFrame(frame);
			image.SetFrameId(frame.FrameId);
			await Assert.That(database.NumImages()).IsEqualTo(1);
			await Assert.That(database.ExistsImage(image.ImageId)).IsTrue();
			await Assert.That(database.ReadImage(image.ImageId) == image).IsTrue();
			await Assert.That(database.ReadImageWithName(image.Name) == image).IsTrue();
			await Assert.That(database.ReadImageWithName("foobar")).IsNull();
			image.Name = "test_changed";
			database.UpdateImage(image);
			await Assert.That(database.ReadImage(image.ImageId) == image).IsTrue();
			Image image2 = image.Clone();
			image2.Name = "test2";
			image2.ImageId = image.ImageId + 1;
			frame.AddDataId(image2.DataId);
			database.UpdateFrame(frame);
			await Assert.That(database.WriteImage(image2, useImageId: true)).IsEqualTo(image2.ImageId);
			await Assert.That(database.NumImages()).IsEqualTo(2);
			await Assert.That(database.ExistsImage(image.ImageId)).IsTrue();
			await Assert.That(database.ExistsImage(image2.ImageId)).IsTrue();
			List<Image> allImages = database.ReadAllImages();
			await Assert.That(allImages.Count).IsEqualTo(2);
			await Assert.That(allImages[0] == image && allImages[1] == image2).IsTrue();
			database.ClearImages();
			await Assert.That(database.NumImages()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_PosePrior()
	{
		using var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		var image = new Image();
		image.SetCameraId(camera.CameraId);
		long numPriors0 = database.NumPosePriors();
		var posePrior = new PosePrior
		{
			CorrDataId = image.DataId,
			Position = new Vector3d(0.1, 0.2, 0.3),
			PositionCovariance = RandomEigen.RandomEigenMatrix3d(),
			CoordinateSystem = PosePriorCoordinateSystem.Cartesian,
			Gravity = RandomEigen.RandomEigenVector3d(),
		};
		posePrior.PosePriorId = database.WritePosePrior(posePrior);
		using (Assert.Multiple())
		{
			await Assert.That(numPriors0).IsEqualTo(0);
			await Assert.That(() => database.WritePosePrior(posePrior)).ThrowsException();
			await Assert.That(database.NumPosePriors()).IsEqualTo(1);
			await Assert.That(database.ReadPosePrior(posePrior.PosePriorId, isDeprecatedImagePrior: false) == posePrior).IsTrue();
			posePrior.PositionCovariance = Matrix3d.Identity;
			database.UpdatePosePrior(posePrior);
			await Assert.That(database.ReadPosePrior(posePrior.PosePriorId, isDeprecatedImagePrior: false) == posePrior).IsTrue();
			List<PosePrior> allPriors = database.ReadAllPosePriors();
			await Assert.That(allPriors.Count).IsEqualTo(1);
			await Assert.That(allPriors[0] == posePrior).IsTrue();
			database.ClearPosePriors();
			await Assert.That(database.NumPosePriors()).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Keypoints()
	{
		using var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		var image = new Image { Name = "test" };
		image.SetCameraId(camera.CameraId);
		image.ImageId = database.WriteImage(image);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumKeypoints()).IsEqualTo(0);
			await Assert.That(database.NumKeypointsForImage(image.ImageId)).IsEqualTo(0);
			List<FeatureKeypoint> keypoints = FeatureKeypoints.Create(10);
			database.WriteKeypoints(image.ImageId, keypoints);
			await Assert.That(keypoints.SequenceEqual(database.ReadKeypoints(image.ImageId))).IsTrue();
			await Assert.That(database.NumKeypoints()).IsEqualTo(10);
			await Assert.That(database.MaxNumKeypoints()).IsEqualTo(10);
			await Assert.That(database.NumKeypointsForImage(image.ImageId)).IsEqualTo(10);
			List<FeatureKeypoint> keypoints2 = FeatureKeypoints.Create(20);
			image.Name = "test2";
			image.ImageId = database.WriteImage(image);
			database.WriteKeypoints(image.ImageId, keypoints2);
			await Assert.That(keypoints2.SequenceEqual(database.ReadKeypoints(image.ImageId))).IsTrue();
			await Assert.That(database.NumKeypoints()).IsEqualTo(30);
			await Assert.That(database.MaxNumKeypoints()).IsEqualTo(20);
			await Assert.That(database.NumKeypointsForImage(image.ImageId)).IsEqualTo(20);
			FeatureKeypoint first = keypoints2[0];
			first.X += 1;
			keypoints2[0] = first;
			database.UpdateKeypoints(image.ImageId, keypoints2);
			await Assert.That(keypoints2.SequenceEqual(database.ReadKeypoints(image.ImageId))).IsTrue();
			database.ClearKeypoints();
			await Assert.That(database.NumKeypoints()).IsEqualTo(0);
			await Assert.That(database.MaxNumKeypoints()).IsEqualTo(0);
			await Assert.That(database.NumKeypointsForImage(image.ImageId)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_ReadKeypointsEmpty()
	{
		using var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		var image = new Image { Name = "test" };
		image.SetCameraId(camera.CameraId);
		image.ImageId = database.WriteImage(image);
		// Reading keypoints for an image with no keypoints should return empty.
		List<FeatureKeypoint> keypoints = database.ReadKeypoints(image.ImageId);
		await Assert.That(keypoints.Count).IsEqualTo(0);
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Descriptors()
	{
		using var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		var image = new Image { Name = "test" };
		image.SetCameraId(camera.CameraId);
		image.ImageId = database.WriteImage(image);
		using (Assert.Multiple())
		{
			await Assert.That(database.NumDescriptors()).IsEqualTo(0);
			await Assert.That(database.NumDescriptorsForImage(image.ImageId)).IsEqualTo(0);
			var descriptors = new FeatureDescriptors(FeatureExtractorType.Sift, FeatureTypesTests.RandomBytes(10, 128));
			database.WriteDescriptors(image.ImageId, descriptors);
			FeatureDescriptors descriptorsRead = database.ReadDescriptors(image.ImageId);
			await Assert.That(descriptorsRead.Data.Rows).IsEqualTo(descriptors.Data.Rows);
			await Assert.That(descriptorsRead.Data.Cols).IsEqualTo(descriptors.Data.Cols);
			await Assert.That(descriptorsRead.Type).IsEqualTo(descriptors.Type);
			await Assert.That(descriptorsRead.Data == descriptors.Data).IsTrue();
			await Assert.That(database.NumDescriptors()).IsEqualTo(10);
			await Assert.That(database.MaxNumDescriptors()).IsEqualTo(10);
			await Assert.That(database.NumDescriptorsForImage(image.ImageId)).IsEqualTo(10);
			var descriptors2 = new FeatureDescriptors(FeatureExtractorType.Undefined, new RowMajorMatrix<byte>(20, 128));
			image.Name = "test2";
			image.ImageId = database.WriteImage(image);
			database.WriteDescriptors(image.ImageId, descriptors2);
			FeatureDescriptors descriptors2Read = database.ReadDescriptors(image.ImageId);
			await Assert.That(descriptors2Read.Type).IsEqualTo(descriptors2.Type);
			await Assert.That(database.NumDescriptors()).IsEqualTo(30);
			await Assert.That(database.MaxNumDescriptors()).IsEqualTo(20);
			await Assert.That(database.NumDescriptorsForImage(image.ImageId)).IsEqualTo(20);
			database.ClearDescriptors();
			await Assert.That(database.NumDescriptors()).IsEqualTo(0);
			await Assert.That(database.MaxNumDescriptors()).IsEqualTo(0);
			await Assert.That(database.NumDescriptorsForImage(image.ImageId)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_DescriptorFeatureTypeDefault()
	{
		// Descriptors written with UNDEFINED type read back as UNDEFINED. (COLMAP's comment on
		// the SIFT migration default concerns old SQLite files, which do not exist here.)
		using var database = new InMemoryDatabase();
		var camera = new Camera();
		camera.CameraId = database.WriteCamera(camera);
		var image = new Image { Name = "test" };
		image.SetCameraId(camera.CameraId);
		image.ImageId = database.WriteImage(image);

		// Write descriptors with UNDEFINED type (default).
		var descriptors = new FeatureDescriptors { Data = new RowMajorMatrix<byte>(5, 128) };
		FeatureExtractorType writtenType = descriptors.Type;
		database.WriteDescriptors(image.ImageId, descriptors);

		// Read back and verify the type is preserved.
		FeatureDescriptors descriptorsRead = database.ReadDescriptors(image.ImageId);

		// Write another image with SIFT type.
		image.Name = "test2";
		image.ImageId = database.WriteImage(image);
		var descriptorsSift = new FeatureDescriptors(FeatureExtractorType.Sift, new RowMajorMatrix<byte>(5, 128));
		database.WriteDescriptors(image.ImageId, descriptorsSift);
		FeatureDescriptors descriptorsSiftRead = database.ReadDescriptors(image.ImageId);

		using (Assert.Multiple())
		{
			await Assert.That(writtenType).IsEqualTo(FeatureExtractorType.Undefined);
			await Assert.That(descriptorsRead.Type).IsEqualTo(FeatureExtractorType.Undefined);
			await Assert.That(descriptorsSiftRead.Type).IsEqualTo(FeatureExtractorType.Sift);
		}
	}

	[Test]
	public async Task ParameterizedDatabaseTests_Matches()
	{
		using var database = new InMemoryDatabase();
		const uint ImageId1 = 1;
		const uint ImageId2 = 2;
		const int NumMatches = 1000;
		var matches12 = new List<FeatureMatch>(NumMatches);
		var matches21 = new List<FeatureMatch>(NumMatches);
		for (uint i = 0; i < NumMatches; ++i)
		{
			matches12.Add(new FeatureMatch(i, 10000 + i));
			matches21.Add(new FeatureMatch(10000 + i, i));
		}

		async Task ExpectValidMatches()
		{
			await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(1);
			List<FeatureMatch> matchesRead12 = database.ReadMatches(ImageId1, ImageId2);
			await Assert.That(matchesRead12.Count).IsEqualTo(matches12.Count);
			await Assert.That(matchesRead12.SequenceEqual(matches12)).IsTrue();
			List<FeatureMatch> matchesRead21 = database.ReadMatches(ImageId2, ImageId1);
			await Assert.That(matchesRead21.Count).IsEqualTo(matches12.Count);
			await Assert.That(matchesRead21.SequenceEqual(matches21)).IsTrue();
		}

		using (Assert.Multiple())
		{
			await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(0);
			database.WriteMatches(ImageId1, ImageId2, matches12);
			await ExpectValidMatches();
			database.DeleteMatches(ImageId1, ImageId2);
			await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(0);
			database.WriteMatches(ImageId2, ImageId1, matches21);
			await ExpectValidMatches();

			await Assert.That(database.ReadAllMatchesBlob().Count).IsEqualTo(1);
			await Assert.That(database.ReadAllMatchesBlob()[0].PairId).IsEqualTo(ImagePairToPairId(ImageId1, ImageId2));
			var matches = database.ReadAllMatches();
			await Assert.That(matches.Count).IsEqualTo(1);
			await Assert.That(matches[0].PairId).IsEqualTo(ImagePairToPairId(ImageId1, ImageId2));
			var pairIdsAndNumMatches = database.ReadNumMatches();
			await Assert.That(pairIdsAndNumMatches.Count).IsEqualTo(1);
			await Assert.That(pairIdsAndNumMatches[0].PairId).IsEqualTo(ImagePairToPairId(ImageId1, ImageId2));
			await Assert.That(pairIdsAndNumMatches[0].NumMatches).IsEqualTo(matches[0].Matches.Count);
			await Assert.That(database.NumMatches()).IsEqualTo(NumMatches);
			database.DeleteMatches(ImageId1, ImageId2);
			await Assert.That(database.NumMatches()).IsEqualTo(0);
			database.WriteMatches(ImageId1, ImageId2, matches12);
			await Assert.That(database.NumMatches()).IsEqualTo(NumMatches);
			database.ClearMatches();
			await Assert.That(database.NumMatches()).IsEqualTo(0);
		}
	}
}
