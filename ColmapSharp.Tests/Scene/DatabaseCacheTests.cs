// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DatabaseCacheTests: colmap/scene/database_cache_test.cc ported 1:1, one method per gtest
// case named Suite_Name, against InMemoryDatabase (kInMemorySqliteDatabasePath). Tests
// ColmapSharp/Scene/DatabaseCache.cs.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so tests that draw call
// RandomUtils.SetPRNGSeed(0) and draw before their first await. The C++ test database sets
// pose_prior1.corr_data_id twice (the second time where pose_prior2 was meant); that is
// kept, since it is the input COLMAP's expectations were written against.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public class DatabaseCacheTests
{
	private static List<FeatureKeypoint> Keypoints(int count) => Enumerable.Repeat(new FeatureKeypoint(), count).ToList();

	private static TwoViewGeometry RandomTwoViewGeometry() => new()
	{
		InlierMatches = [new FeatureMatch(0, 1)],
		Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic,
		F = RandomEigen.RandomEigenMatrix3d(),
		E = RandomEigen.RandomEigenMatrix3d(),
		H = RandomEigen.RandomEigenMatrix3d(),
		Cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d()),
	};

	internal static InMemoryDatabase CreateTestDatabase()
	{
		var database = new InMemoryDatabase();

		Camera camera1 = Camera.CreateFromModelId(InvalidCameraId, CameraModelId.SimplePinhole, 1, 1, 1);
		camera1.CameraId = database.WriteCamera(camera1);
		Camera camera2 = Camera.CreateFromModelId(InvalidCameraId, CameraModelId.SimplePinhole, 2, 2, 2);
		camera2.CameraId = database.WriteCamera(camera2);

		var rig = new Rig();
		rig.AddRefSensor(camera1.SensorId);
		rig.AddSensor(camera2.SensorId);
		uint rigId = database.WriteRig(rig);

		var image1 = new Image { Name = "image1" };
		image1.SetCameraId(camera1.CameraId);
		image1.ImageId = database.WriteImage(image1);
		var image2 = new Image { Name = "image2" };
		image2.SetCameraId(camera2.CameraId);
		image2.ImageId = database.WriteImage(image2);
		var image3 = new Image { Name = "image3" };
		image3.SetCameraId(camera1.CameraId);
		image3.ImageId = database.WriteImage(image3);
		var image4 = new Image { Name = "image4" };
		image4.SetCameraId(camera2.CameraId);
		image4.ImageId = database.WriteImage(image4);

		var posePrior1 = new PosePrior { CorrDataId = image1.DataId, Position = RandomEigen.RandomEigenVector3d() };
		posePrior1.PosePriorId = database.WritePosePrior(posePrior1);
		var posePrior2 = new PosePrior();
		posePrior1.CorrDataId = image2.DataId;
		posePrior2.Position = RandomEigen.RandomEigenVector3d();
		posePrior2.PosePriorId = database.WritePosePrior(posePrior2);

		var frame1 = new Frame();
		frame1.SetRigId(rigId);
		frame1.AddDataId(image1.DataId);
		frame1.AddDataId(image2.DataId);
		frame1.FrameId = database.WriteFrame(frame1);
		var frame2 = new Frame();
		frame2.SetRigId(rigId);
		frame2.AddDataId(image3.DataId);
		frame2.AddDataId(image4.DataId);
		frame2.FrameId = database.WriteFrame(frame2);

		database.WriteKeypoints(image1.ImageId, Keypoints(10));
		database.WriteKeypoints(image2.ImageId, Keypoints(5));
		database.WriteKeypoints(image3.ImageId, Keypoints(6));
		database.WriteKeypoints(image4.ImageId, Keypoints(3));

		TwoViewGeometry twoViewGeometry = RandomTwoViewGeometry();
		database.WriteTwoViewGeometry(image1.ImageId, image2.ImageId, twoViewGeometry);
		database.WriteTwoViewGeometry(image2.ImageId, image3.ImageId, twoViewGeometry);
		database.WriteTwoViewGeometry(image3.ImageId, image4.ImageId, twoViewGeometry);

		return database;
	}

	[Test]
	public async Task DatabaseCache_Empty()
	{
		var cache = new DatabaseCache();
		using (Assert.Multiple())
		{
			await Assert.That(cache.NumRigs).IsEqualTo(0);
			await Assert.That(cache.NumCameras).IsEqualTo(0);
			await Assert.That(cache.NumFrames).IsEqualTo(0);
			await Assert.That(cache.NumImages).IsEqualTo(0);
			await Assert.That(cache.NumPosePriors).IsEqualTo(0);
		}
	}

	[Test]
	public async Task DatabaseCache_ConstructFromDatabase()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateTestDatabase();
		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());

		using (Assert.Multiple())
		{
			await Assert.That(cache.NumRigs).IsEqualTo(1);
			await Assert.That(cache.NumCameras).IsEqualTo(2);
			await Assert.That(cache.NumFrames).IsEqualTo(2);
			await Assert.That(cache.NumImages).IsEqualTo(4);
			await Assert.That(cache.NumPosePriors).IsEqualTo(2);

			foreach (Rig rig in database.ReadAllRigs())
			{
				await Assert.That(cache.ExistsRig(rig.RigId)).IsTrue();
				await Assert.That(cache.Rig(rig.RigId) == rig).IsTrue();
			}

			foreach (Camera camera in database.ReadAllCameras())
			{
				await Assert.That(cache.ExistsCamera(camera.CameraId)).IsTrue();
				await Assert.That(cache.Camera(camera.CameraId) == camera).IsTrue();
			}

			foreach (Frame frame in database.ReadAllFrames())
			{
				await Assert.That(cache.ExistsFrame(frame.FrameId)).IsTrue();
				await Assert.That(cache.Frame(frame.FrameId) == frame).IsTrue();
			}

			List<Image> images = database.ReadAllImages();
			await Assert.That(cache.ExistsImage(images[0].ImageId)).IsTrue();
			await Assert.That(cache.Image(images[0].ImageId).NumPoints2D).IsEqualTo(10u);
			await Assert.That(cache.ExistsImage(images[1].ImageId)).IsTrue();
			await Assert.That(cache.Image(images[1].ImageId).NumPoints2D).IsEqualTo(5u);
			await Assert.That(cache.ExistsImage(images[2].ImageId)).IsTrue();
			await Assert.That(cache.Image(images[2].ImageId).NumPoints2D).IsEqualTo(6u);
			await Assert.That(cache.ExistsImage(images[3].ImageId)).IsTrue();
			await Assert.That(cache.Image(images[3].ImageId).NumPoints2D).IsEqualTo(3u);

			await Assert.That(cache.PosePriors.SequenceEqual(database.ReadAllPosePriors())).IsTrue();

			CorrespondenceGraph correspondenceGraph = cache.CorrespondenceGraph;
			await Assert.That(correspondenceGraph.ExistsImage(images[0].ImageId)).IsTrue();
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[0].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[0].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.ExistsImage(images[1].ImageId)).IsTrue();
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[1].ImageId)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[1].ImageId)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[2].ImageId)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[2].ImageId)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[3].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[3].ImageId)).IsEqualTo(1u);
		}
	}

	[Test]
	public async Task DatabaseCache_ConstructFromDatabaseWithCustomImages()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateTestDatabase();

		// Note that the first two images are part of the same frame.
		List<Image> images = database.ReadAllImages();
		var options = new DatabaseCache.Options { ImageNames = [images[0].Name, images[1].Name] };
		DatabaseCache cache = DatabaseCache.Create(database, options);

		using (Assert.Multiple())
		{
			await Assert.That(cache.NumRigs).IsEqualTo(1);
			await Assert.That(cache.NumCameras).IsEqualTo(2);
			await Assert.That(cache.NumFrames).IsEqualTo(1);
			await Assert.That(cache.NumImages).IsEqualTo(2);
			await Assert.That(cache.NumPosePriors).IsEqualTo(2);

			CorrespondenceGraph correspondenceGraph = cache.CorrespondenceGraph;
			await Assert.That(correspondenceGraph.ExistsImage(images[0].ImageId)).IsTrue();
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[0].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[0].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.ExistsImage(images[1].ImageId)).IsTrue();
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(images[1].ImageId)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(images[1].ImageId)).IsEqualTo(1u);
		}
	}

	private static InMemoryDatabase CreateLegacyTestDatabase()
	{
		var database = new InMemoryDatabase();

		Camera camera = Camera.CreateFromModelId(InvalidCameraId, CameraModelId.SimplePinhole, 1, 1, 1);
		uint cameraId = database.WriteCamera(camera);
		var image1 = new Image { Name = "image1" };
		image1.SetCameraId(cameraId);
		var image2 = new Image { Name = "image2" };
		image2.SetCameraId(cameraId);
		var image3 = new Image { Name = "image3" };
		image3.SetCameraId(cameraId);
		uint imageId1 = database.WriteImage(image1);
		uint imageId2 = database.WriteImage(image2);
		uint imageId3 = database.WriteImage(image3);
		database.WriteKeypoints(imageId1, Keypoints(10));
		database.WriteKeypoints(imageId2, Keypoints(5));
		database.WriteKeypoints(imageId3, Keypoints(7));
		TwoViewGeometry twoViewGeometry = RandomTwoViewGeometry();
		database.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		database.WriteTwoViewGeometry(imageId1, imageId3, twoViewGeometry);

		return database;
	}

	[Test]
	public async Task DatabaseCache_ConstructFromLegacyDatabaseWithoutRigsAndFrames()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateLegacyTestDatabase();
		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());
		using (Assert.Multiple())
		{
			await Assert.That(cache.NumCameras).IsEqualTo(1);
			await Assert.That(cache.NumImages).IsEqualTo(3);
			await Assert.That(cache.NumPosePriors).IsEqualTo(0);
			await Assert.That(cache.ExistsCamera(1)).IsTrue();
			await Assert.That(cache.Camera(1).ModelId).IsEqualTo(CameraModelId.SimplePinhole);
			await Assert.That(cache.ExistsImage(1)).IsTrue();
			await Assert.That(cache.ExistsImage(2)).IsTrue();
			await Assert.That(cache.ExistsImage(3)).IsTrue();
			await Assert.That(cache.Image(1).NumPoints2D).IsEqualTo(10u);
			await Assert.That(cache.Image(2).NumPoints2D).IsEqualTo(5u);
			await Assert.That(cache.Image(3).NumPoints2D).IsEqualTo(7u);
			await Assert.That(cache.CorrespondenceGraph.ExistsImage(1)).IsTrue();
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(2u);
			await Assert.That(cache.CorrespondenceGraph.NumObservationsForImage(1)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.ExistsImage(2)).IsTrue();
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(2)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.NumObservationsForImage(2)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.ExistsImage(3)).IsTrue();
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(3)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.NumObservationsForImage(3)).IsEqualTo(1u);
		}
	}

	[Test]
	public async Task DatabaseCache_ConstructFromLegacyDatabaseWithCustomImages()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateLegacyTestDatabase();
		List<Image> images = database.ReadAllImages();
		var options = new DatabaseCache.Options { ImageNames = [images[0].Name, images[2].Name] };
		DatabaseCache cache = DatabaseCache.Create(database, options);
		using (Assert.Multiple())
		{
			await Assert.That(cache.NumCameras).IsEqualTo(1);
			await Assert.That(cache.NumImages).IsEqualTo(2);
			await Assert.That(cache.NumPosePriors).IsEqualTo(0);
			await Assert.That(cache.ExistsCamera(1)).IsTrue();
			await Assert.That(cache.Camera(1).ModelId).IsEqualTo(CameraModelId.SimplePinhole);
			await Assert.That(cache.ExistsImage(1)).IsTrue();
			await Assert.That(cache.ExistsImage(3)).IsTrue();
			await Assert.That(cache.Image(1).NumPoints2D).IsEqualTo(10u);
			await Assert.That(cache.Image(3).NumPoints2D).IsEqualTo(7u);
			await Assert.That(cache.CorrespondenceGraph.ExistsImage(1)).IsTrue();
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.NumObservationsForImage(1)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.ExistsImage(3)).IsTrue();
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(3)).IsEqualTo(1u);
			await Assert.That(cache.CorrespondenceGraph.NumObservationsForImage(3)).IsEqualTo(1u);
		}
	}

	[Test]
	public async Task DatabaseCache_ConstructFromCustom()
	{
		RandomUtils.SetPRNGSeed(0);
		var cache = new DatabaseCache();
		int numRigs0 = cache.NumRigs;
		int numCameras0 = cache.NumCameras;
		int numFrames0 = cache.NumFrames;
		int numImages0 = cache.NumImages;
		int numPosePriors0 = cache.NumPosePriors;

		const uint RigId = 41;
		var rig = new Rig { RigId = RigId };
		cache.AddRig(rig);

		const uint CameraId = 42;
		cache.AddCamera(Camera.CreateFromModelId(CameraId, CameraModelId.SimplePinhole, 1, 1, 1));

		const uint FrameId = 43;
		var frame = new Frame { FrameId = FrameId };
		cache.AddFrame(frame);

		const uint ImageId = 44;
		var image = new Image { ImageId = ImageId, Name = "image" };
		image.SetCameraId(CameraId);
		cache.AddImage(image);

		const uint PosePriorId = 45;
		var posePrior = new PosePrior { Position = RandomEigen.RandomEigenVector3d(), PosePriorId = PosePriorId };
		cache.AddPosePrior(posePrior);

		using (Assert.Multiple())
		{
			await Assert.That(numRigs0).IsEqualTo(0);
			await Assert.That(numCameras0).IsEqualTo(0);
			await Assert.That(numFrames0).IsEqualTo(0);
			await Assert.That(numImages0).IsEqualTo(0);
			await Assert.That(numPosePriors0).IsEqualTo(0);

			await Assert.That(cache.NumCameras).IsEqualTo(1);
			await Assert.That(cache.NumImages).IsEqualTo(1);
			await Assert.That(cache.NumPosePriors).IsEqualTo(1);
			await Assert.That(cache.ExistsRig(RigId)).IsTrue();
			await Assert.That(cache.ExistsCamera(CameraId)).IsTrue();
			await Assert.That(cache.ExistsFrame(FrameId)).IsTrue();
			await Assert.That(cache.ExistsImage(ImageId)).IsTrue();
			await Assert.That(cache.PosePriors.SequenceEqual([posePrior])).IsTrue();
		}
	}

	/// <summary>
	/// C#-only (database_cache_test.cc has no CreateFromCache case). Unlike Load, the
	/// "connected" filter is per image: a pair counts only if both images are candidates.
	/// The kept images' whole frames come along, with their cameras and rig, and only the
	/// pairs between kept images.
	/// </summary>
	[Test]
	public async Task CSharpOnly_CreateFromCacheKeepsWholeFrames()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateTestDatabase();
		DatabaseCache source = DatabaseCache.Create(database, new DatabaseCache.Options());
		DatabaseCache cache = DatabaseCache.CreateFromCache(source, new DatabaseCache.Options { ImageNames = ["image3", "image4"] });

		using (Assert.Multiple())
		{
			await Assert.That(cache.NumRigs).IsEqualTo(1);
			await Assert.That(cache.NumCameras).IsEqualTo(2);
			await Assert.That(cache.NumFrames).IsEqualTo(1);
			await Assert.That(cache.Images.Keys.ToList()).IsEquivalentTo(new List<uint> { 3, 4 });
			await Assert.That(cache.NumPosePriors).IsEqualTo(2);
			await Assert.That(cache.Image(3).NumPoints2D).IsEqualTo(6u);
			await Assert.That(cache.CorrespondenceGraph.ImagePairs()).IsEquivalentTo(new List<ulong> { ImagePairToPairId(3, 4) });
			await Assert.That(cache.CorrespondenceGraph.NumCorrespondencesForImage(3)).IsEqualTo(1u);
			await Assert.That(ReferenceEquals(cache.Image(3), source.Image(3))).IsFalse();
		}

		// A single loaded image brings its frame partner (image2) and their pair.
		DatabaseCache wholeFrame = DatabaseCache.CreateFromCache(
			source, new DatabaseCache.Options { ImageNames = ["image1"], LoadAllImages = true });
		await Assert.That(wholeFrame.Images.Keys.ToList()).IsEquivalentTo(new List<uint> { 1, 2 });
		await Assert.That(wholeFrame.CorrespondenceGraph.ImagePairs()).IsEquivalentTo(new List<ulong> { ImagePairToPairId(1, 2) });

		// image1 and image4 share no pair, so nothing is loaded unless all candidates are
		// requested.
		DatabaseCache unconnected = DatabaseCache.CreateFromCache(source, new DatabaseCache.Options { ImageNames = ["image1", "image4"] });
		await Assert.That(unconnected.NumImages).IsEqualTo(0);
		DatabaseCache all = DatabaseCache.CreateFromCache(
			source, new DatabaseCache.Options { ImageNames = ["image1", "image4"], LoadAllImages = true });
		await Assert.That(all.NumImages).IsEqualTo(4);
		await Assert.That(all.CorrespondenceGraph.NumImagePairs).IsEqualTo(3);
	}

	[Test]
	public async Task DatabaseCache_NonConstCorrespondenceGraph()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = CreateTestDatabase();
		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());

		// Non-const overload returns a mutable shared_ptr.
		CorrespondenceGraph mutableGraph = cache.CorrespondenceGraph;
		await Assert.That(mutableGraph).IsNotNull();

		// Verify it can be used to update a two-view geometry.
		List<ulong> imagePairs = mutableGraph.ImagePairs();
		await Assert.That(imagePairs).IsNotEmpty();
		(uint imageId1, uint imageId2) = PairIdToImagePair(imagePairs[0]);

		TwoViewGeometry geom = mutableGraph.ExtractTwoViewGeometry(imageId1, imageId2, extractInlierMatches: false);
		var newPose = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		geom.Cam2FromCam1 = newPose;
		mutableGraph.UpdateTwoViewGeometry(imageId1, imageId2, geom);

		// Read back through the const accessor and verify the update is visible.
		DatabaseCache constCache = cache;
		TwoViewGeometry updated = constCache.CorrespondenceGraph.ExtractTwoViewGeometry(imageId1, imageId2, false);
		await Assert.That(updated.Cam2FromCam1.HasValue).IsTrue();
		await Assert.That(Rigid3dMatchers.Rigid3dNear(updated.Cam2FromCam1!.Value, newPose, 1e-6, 1e-6)).IsTrue();
	}
}
