// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests.Database: the reconstruction_test.cc case that needs the database,
// TranscribeImageIdsToDatabase, run against InMemoryDatabase (kInMemorySqliteDatabasePath).
// Tests ColmapSharp/Scene/Reconstruction.Database.cs. Helpers and the other cases are in
// ReconstructionTests.cs / ReconstructionTests.Queries.cs. CSharpOnly_LoadFromDatabaseCache
// covers Load, which reconstruction_test.cc does not test.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	[Test]
	public async Task Reconstruction_TranscribeImageIdsToDatabase()
	{
		RandomUtils.SetPRNGSeed(0);
		string[] imageNames = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"];

		using var database = new InMemoryDatabase();

		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		database.WriteCamera(camera, useCameraId: true);

		// Write images to database.
		var dbImage1 = new Image { Name = imageNames[0] };
		dbImage1.SetCameraId(camera.CameraId);
		dbImage1.ImageId = database.WriteImage(dbImage1);
		var dbImage2 = new Image { Name = imageNames[1] };
		dbImage2.SetCameraId(camera.CameraId);
		dbImage2.ImageId = database.WriteImage(dbImage2);
		var dbImage3 = new Image { Name = imageNames[2] };
		dbImage3.SetCameraId(camera.CameraId);
		dbImage3.ImageId = database.WriteImage(dbImage3);

		// Create a reconstruction with different image IDs but same names.
		var reconstruction = new Reconstruction();
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);

		uint[] reconImageIds = [100, 200, 300];
		for (int i = 0; i < 3; ++i)
		{
			uint imageId = reconImageIds[i];

			var frame = new Frame { FrameId = imageId };
			frame.SetRigId(rig.RigId);
			frame.AddDataId(new DataId(camera.SensorId, imageId));
			frame.SetRigFromWorld(new Rigid3d());
			reconstruction.AddFrame(frame);

			var image = new Image { ImageId = imageId, Name = imageNames[i] };
			image.SetCameraId(camera.CameraId);
			image.SetFrameId(frame.FrameId);
			image.SetPoints2D(Enumerable.Repeat(Vector2d.Zero, 10).ToList());
			reconstruction.AddImage(image);
		}

		// Add a 3D point with observations to test track updates.
		var track = new Track();
		track.AddElement(reconImageIds[0], 0);
		track.AddElement(reconImageIds[1], 1);
		track.AddElement(reconImageIds[2], 2);
		ulong point3DId = reconstruction.AddPoint3D(RandomEigen.RandomEigenVector3d(), track);

		reconstruction.TranscribeImageIdsToDatabase(database);

		using (Assert.Multiple())
		{
			// Verify image IDs were updated to match database.
			await Assert.That(reconstruction.ExistsImage(dbImage1.ImageId)).IsTrue();
			await Assert.That(reconstruction.ExistsImage(dbImage2.ImageId)).IsTrue();
			await Assert.That(reconstruction.ExistsImage(dbImage3.ImageId)).IsTrue();
			await Assert.That(reconstruction.ExistsImage(reconImageIds[0])).IsFalse();
			await Assert.That(reconstruction.ExistsImage(reconImageIds[1])).IsFalse();
			await Assert.That(reconstruction.ExistsImage(reconImageIds[2])).IsFalse();

			// Verify image names are preserved.
			await Assert.That(reconstruction.Image(dbImage1.ImageId).Name).IsEqualTo(imageNames[0]);
			await Assert.That(reconstruction.Image(dbImage2.ImageId).Name).IsEqualTo(imageNames[1]);
			await Assert.That(reconstruction.Image(dbImage3.ImageId).Name).IsEqualTo(imageNames[2]);

			// Verify frame data IDs were updated.
			await Assert.That(reconstruction.Frame(reconImageIds[0]).HasDataId(dbImage1.DataId)).IsTrue();
			await Assert.That(reconstruction.Frame(reconImageIds[1]).HasDataId(dbImage2.DataId)).IsTrue();
			await Assert.That(reconstruction.Frame(reconImageIds[2]).HasDataId(dbImage3.DataId)).IsTrue();

			// Verify track elements were updated.
			Track updatedTrack = reconstruction.Point3D(point3DId).Track;
			await Assert.That(updatedTrack.Length).IsEqualTo(3);
			List<uint> trackImageIds = updatedTrack.Elements.Select(trackEl => trackEl.ImageId).ToList();
			await Assert.That(trackImageIds).IsEquivalentTo(new List<uint> { dbImage1.ImageId, dbImage2.ImageId, dbImage3.ImageId });
		}

		await ExpectValidPtrs(reconstruction);
	}

	/// <summary>
	/// C#-only (reconstruction_test.cc has no Load case): Load adds every cached object with
	/// its references wired, and loading the same cache again only checks consistency.
	/// </summary>
	[Test]
	public async Task CSharpOnly_LoadFromDatabaseCache()
	{
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = DatabaseCacheTests.CreateTestDatabase();
		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());

		var reconstruction = new Reconstruction();
		reconstruction.Load(cache);
		reconstruction.Load(cache);

		using (Assert.Multiple())
		{
			await Assert.That(reconstruction.NumRigs).IsEqualTo(1);
			await Assert.That(reconstruction.NumCameras).IsEqualTo(2);
			await Assert.That(reconstruction.NumFrames).IsEqualTo(2);
			await Assert.That(reconstruction.NumImages).IsEqualTo(4);
			await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
			foreach ((uint imageId, Image image) in cache.Images)
			{
				await Assert.That(reconstruction.Image(imageId).Name).IsEqualTo(image.Name);
				await Assert.That(reconstruction.Image(imageId).NumPoints2D).IsEqualTo(image.NumPoints2D);
			}
		}

		await ExpectValidPtrs(reconstruction);
	}
}
