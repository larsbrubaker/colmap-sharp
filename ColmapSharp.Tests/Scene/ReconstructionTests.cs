// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests: colmap/scene/reconstruction_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/Reconstruction*.cs. This
// file holds the helpers and the construction, add, register and 3D point cases;
// ReconstructionTests.Queries.cs holds normalization, transforms, cropping, lookups,
// statistics and validation; ReconstructionTests.Database.cs holds
// TranscribeImageIdsToDatabase.
//
// Translation notes: pointer comparisons (`&reconstruction.Image(1)`) are reference
// equality; `frame.RigFromWorld().translation().z() = v` is SetRigFromWorld with that
// translation; try { ... } catch (std::exception& e) { EXPECT_THAT(e.what(), HasSubstr(s)) }
// stays a try/catch that only checks the message when something throws, as in C++.
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so tests that draw call
// RandomUtils.SetPRNGSeed(0) and draw before their first await.
//
// Not ported yet (it needs image decoding, which is not ported): ExtractColorsForAllImages.
// The SynthesizeDataset cases are ReconstructionTests.Synthetic.cs,
// TranscribeImageIdsToDatabase is ReconstructionTests.Database.cs, and ConvertToPLY and
// ImportPLYFromVector are ReconstructionTests.Ply.cs.

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
	private static async Task ExpectValidPtrs(Reconstruction reconstruction)
	{
		foreach (Frame frame in reconstruction.Frames.Values)
		{
			await Assert.That(frame.HasRigPtr).IsTrue();
			await Assert.That(ReferenceEquals(frame.RigPtr, reconstruction.Rig(frame.RigId))).IsTrue();
		}

		foreach (Image image in reconstruction.Images.Values)
		{
			await Assert.That(image.HasCameraPtr).IsTrue();
			await Assert.That(ReferenceEquals(image.CameraPtr, reconstruction.Camera(image.CameraId))).IsTrue();
			await Assert.That(image.HasFramePtr).IsTrue();
			await Assert.That(ReferenceEquals(image.FramePtr, reconstruction.Frame(image.FrameId))).IsTrue();
		}
	}

	private static Reconstruction GenerateReconstruction(uint numImages)
	{
		const int kNumPoints2D = 10;

		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 1, 1);
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);

		for (uint imageId = 1; imageId <= numImages; ++imageId)
		{
			var frame = new Frame { FrameId = imageId };
			frame.SetRigId(rig.RigId);
			frame.AddDataId(new DataId(camera.SensorId, imageId));
			frame.SetRigFromWorld(new Rigid3d());
			reconstruction.AddFrame(frame);
			var image = new Image { ImageId = imageId, Name = "image" + imageId };
			image.SetCameraId(camera.CameraId);
			image.SetFrameId(frame.FrameId);
			image.SetPoints2D(Enumerable.Repeat(Vector2d.Zero, kNumPoints2D).ToList());
			reconstruction.AddImage(image);
		}

		return reconstruction;
	}

	private static Rigid3d TranslatedZ(double z) => new(Quaterniond.Identity, new Vector3d(0, 0, z));

	private static async Task ExpectCounts(
		Reconstruction reconstruction, int rigs, int cameras, int frames, int regFrames, int images, int regImages, int points3D)
	{
		await Assert.That(reconstruction.NumRigs).IsEqualTo(rigs);
		await Assert.That(reconstruction.NumCameras).IsEqualTo(cameras);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(frames);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(regFrames);
		await Assert.That(reconstruction.NumImages).IsEqualTo(images);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(regImages);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(points3D);
	}

	[Test]
	public async Task Reconstruction_Empty()
	{
		var reconstruction = new Reconstruction();
		await ExpectCounts(reconstruction, 0, 0, 0, 0, 0, 0, 0);
	}

	[Test]
	public async Task Reconstruction_AddRig()
	{
		var reconstruction = new Reconstruction();
		var rig = new Rig { RigId = 1 };
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		rig.AddRefSensor(camera.SensorId);
		await Assert.That(() => reconstruction.AddRig(rig)).ThrowsException();
		reconstruction.AddCamera(camera);
		reconstruction.AddRig(rig);
		await Assert.That(reconstruction.ExistsRig(rig.RigId)).IsTrue();
		await Assert.That(reconstruction.Rig(rig.RigId).RigId).IsEqualTo(rig.RigId);
		await Assert.That(reconstruction.Rigs.ContainsKey(rig.RigId)).IsTrue();
		await Assert.That(reconstruction.Rigs.Count).IsEqualTo(1);
		await ExpectCounts(reconstruction, 1, 1, 0, 0, 0, 0, 0);
	}

	[Test]
	public async Task Reconstruction_AddCamera()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCamera(camera);
		await Assert.That(reconstruction.ExistsCamera(camera.CameraId)).IsTrue();
		await Assert.That(reconstruction.Camera(camera.CameraId).CameraId).IsEqualTo(camera.CameraId);
		await Assert.That(reconstruction.Cameras.ContainsKey(camera.CameraId)).IsTrue();
		await Assert.That(reconstruction.Cameras.Count).IsEqualTo(1);
		await ExpectCounts(reconstruction, 0, 1, 0, 0, 0, 0, 0);
	}

	[Test]
	public async Task Reconstruction_AddCameraWithTrivialRig()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCameraWithTrivialRig(camera);
		await Assert.That(reconstruction.ExistsCamera(camera.CameraId)).IsTrue();
		await Assert.That(reconstruction.Camera(camera.CameraId).CameraId).IsEqualTo(camera.CameraId);
		await Assert.That(reconstruction.Cameras.ContainsKey(camera.CameraId)).IsTrue();
		await Assert.That(reconstruction.Cameras.Count).IsEqualTo(1);
		await Assert.That(reconstruction.ExistsRig(camera.CameraId)).IsTrue();
		await Assert.That(reconstruction.Rig(camera.CameraId).RigId).IsEqualTo(camera.CameraId);
		await Assert.That(reconstruction.Rig(camera.CameraId).NumSensors).IsEqualTo(1);
		await ExpectCounts(reconstruction, 1, 1, 0, 0, 0, 0, 0);
	}

	[Test]
	public async Task Reconstruction_AddFrame()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		frame.AddDataId(new DataId(camera.SensorId, 1));
		try
		{
			reconstruction.AddFrame(frame);
		}
		catch (Exception e)
		{
			await Assert.That(e.Message).Contains("Rig with ID 1 does not exist");
		}

		reconstruction.AddRig(rig);
		try
		{
			reconstruction.AddFrame(frame);
		}
		catch (Exception e)
		{
			await Assert.That(e.Message).Contains("Check failed: rig.HasSensor(data_id.sensor_id)");
		}

		await Assert.That(() => reconstruction.AddFrame(frame)).ThrowsException();
		reconstruction.Rig(frame.RigId).AddRefSensor(camera.SensorId);
		reconstruction.AddFrame(frame);
		await Assert.That(reconstruction.ExistsFrame(1)).IsTrue();
		await Assert.That(reconstruction.Frame(1).FrameId).IsEqualTo(1u);
		await Assert.That(reconstruction.Frames.ContainsKey(1)).IsTrue();
		await Assert.That(reconstruction.Frames.Count).IsEqualTo(1);
		await Assert.That(reconstruction.NumRigs).IsEqualTo(1);
		await Assert.That(reconstruction.NumCameras).IsEqualTo(1);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.NumImages).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		reconstruction.Frame(1).SetRigFromWorld(new Rigid3d());
		reconstruction.RegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await ExpectValidPtrs(reconstruction);
	}

	[Test]
	public async Task Reconstruction_AddImageWrongFrameCorrespondence()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(camera.CameraId);
		image.SetFrameId(frame.FrameId);
		frame.AddDataId(image.DataId);
		reconstruction.AddFrame(frame);
		reconstruction.AddImage(image);
		image.ImageId = 2;
		await Assert.That(() => reconstruction.AddImage(image)).ThrowsException();
	}

	[Test]
	public async Task Reconstruction_AddImage()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(camera.CameraId);
		image.SetFrameId(1);

		// Verify that adding an image fails if the frame does not exist.
		{
			var reconstruction = new Reconstruction();
			reconstruction.AddCamera(camera);
			reconstruction.AddRig(rig);
			try
			{
				reconstruction.AddImage(image);
			}
			catch (Exception e)
			{
				await Assert.That(e.Message).Contains("Frame with ID 1 does not exist");
			}
		}

		// Verify that adding an image fails if the frame has no matching data id.
		{
			var reconstruction = new Reconstruction();
			reconstruction.AddCamera(camera);
			reconstruction.AddRig(rig);
			var frameWithoutData = new Frame { FrameId = 1 };
			frameWithoutData.SetRigId(rig.RigId);
			reconstruction.AddFrame(frameWithoutData);
			try
			{
				reconstruction.AddImage(image);
			}
			catch (Exception e)
			{
				await Assert.That(e.Message).Contains("Check failed: frame.HasDataId(image.DataId())");
			}
		}

		// Successfully add the image when the frame has the matching data id.
		{
			var reconstruction = new Reconstruction();
			reconstruction.AddCamera(camera);
			reconstruction.AddRig(rig);
			var frame = new Frame { FrameId = 1 };
			frame.SetRigId(rig.RigId);
			frame.AddDataId(image.DataId);
			reconstruction.AddFrame(frame);
			reconstruction.AddImage(image);
			await Assert.That(reconstruction.ExistsImage(1)).IsTrue();
			await Assert.That(reconstruction.Image(1).ImageId).IsEqualTo(1u);
			await Assert.That(reconstruction.Image(1).HasPose).IsFalse();
			await Assert.That(reconstruction.Images.ContainsKey(1)).IsTrue();
			await Assert.That(reconstruction.Images.Count).IsEqualTo(1);
			await ExpectCounts(reconstruction, 1, 1, 1, 0, 1, 0, 0);
			reconstruction.Image(1).FramePtr.SetRigFromWorld(new Rigid3d());
			reconstruction.RegisterFrame(1);
			await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
			await Assert.That(reconstruction.NumRegImages).IsEqualTo(1);
			await ExpectValidPtrs(reconstruction);
		}
	}

	[Test]
	public async Task Reconstruction_AddImageWithTrivialFrame()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCameraWithTrivialRig(camera);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(camera.CameraId);
		reconstruction.AddImageWithTrivialFrame(image);

		await Assert.That(reconstruction.ExistsImage(1)).IsTrue();
		await Assert.That(reconstruction.Image(1).ImageId).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(1).FrameId).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(1).HasPose).IsFalse();
		await Assert.That(reconstruction.Images.ContainsKey(1)).IsTrue();
		await Assert.That(reconstruction.Images.Count).IsEqualTo(1);
		await Assert.That(reconstruction.ExistsFrame(1)).IsTrue();
		await Assert.That(reconstruction.Frame(1).NumDataIds).IsEqualTo(1);
		await ExpectCounts(reconstruction, 1, 1, 1, 0, 1, 0, 0);
		reconstruction.Image(1).FramePtr.SetRigFromWorld(new Rigid3d());
		reconstruction.RegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(1);
		await ExpectValidPtrs(reconstruction);
	}

	[Test]
	public async Task Reconstruction_AddImageWithTrivialFrameExistsNonTrivialRig()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCameraWithTrivialRig(camera);
		camera.CameraId = 2;
		reconstruction.Rig(1).AddSensor(camera.SensorId);
		Check.Eq(reconstruction.Rig(1).NumSensors, 2);

		var image = new Image { ImageId = 1 };
		// The rig has multiple cameras
		image.SetCameraId(1);
		await Assert.That(() => reconstruction.AddImageWithTrivialFrame(image)).ThrowsException();
		// No rig with id 2 found in the reconstruction
		image.SetCameraId(2);
		await Assert.That(() => reconstruction.AddImageWithTrivialFrame(image)).ThrowsException();
	}

	[Test]
	public async Task Reconstruction_AddImageWithTrivialFrameSetCamFromWorld()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCameraWithTrivialRig(camera);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(camera.CameraId);
		reconstruction.AddImageWithTrivialFrame(image, new Rigid3d());
		await Assert.That(reconstruction.ExistsImage(1)).IsTrue();
		await Assert.That(reconstruction.Image(1).ImageId).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(1).FrameId).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(1).HasPose).IsTrue();
		await Assert.That(reconstruction.Images.ContainsKey(1)).IsTrue();
		await Assert.That(reconstruction.Images.Count).IsEqualTo(1);
		await Assert.That(reconstruction.ExistsFrame(1)).IsTrue();
		await Assert.That(reconstruction.Frame(1).NumDataIds).IsEqualTo(1);
		await Assert.That(reconstruction.Frame(1).HasPose).IsTrue();
		await ExpectCounts(reconstruction, 1, 1, 1, 1, 1, 1, 0);
		await ExpectValidPtrs(reconstruction);
	}

	[Test]
	public async Task Reconstruction_RegImageIds()
	{
		var reconstruction = new Reconstruction();

		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		reconstruction.AddCamera(camera);

		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);

		var image1 = new Image { ImageId = 1 };
		image1.SetCameraId(camera.CameraId);
		image1.SetFrameId(1);
		var image2 = new Image { ImageId = 2 };
		image2.SetCameraId(camera.CameraId);
		image2.SetFrameId(1);

		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		frame.AddDataId(image1.DataId);
		frame.AddDataId(image2.DataId);
		reconstruction.AddFrame(frame);
		reconstruction.Frame(frame.FrameId).SetRigFromWorld(new Rigid3d());

		reconstruction.RegisterFrame(frame.FrameId);

		// Throws because no image was added.
		await Assert.That(() => reconstruction.RegImageIds()).ThrowsException();
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(2);

		// Throws because second image was not added.
		reconstruction.AddImage(image1);
		await Assert.That(() => reconstruction.RegImageIds()).ThrowsException();
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(2);

		reconstruction.AddImage(image2);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(2);
		await Assert.That(reconstruction.RegImageIds()).IsEquivalentTo(new List<uint> { 1, 2 }, TUnit.Assertions.Enums.CollectionOrdering.Matching);

		// Registering a frame twice is a no-op.
		reconstruction.RegisterFrame(frame.FrameId);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(2);
		await Assert.That(reconstruction.RegImageIds()).IsEquivalentTo(new List<uint> { 1, 2 }, TUnit.Assertions.Enums.CollectionOrdering.Matching);

		reconstruction.DeRegisterFrame(frame.FrameId);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(0);
		await Assert.That(reconstruction.RegImageIds()).IsEmpty();

		// De-registering a frame twice is a no-op.
		reconstruction.DeRegisterFrame(frame.FrameId);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(0);
		await Assert.That(reconstruction.RegImageIds()).IsEmpty();
	}

	[Test]
	public async Task Reconstruction_AddPoint3D()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		var reconstruction = new Reconstruction();
		ulong point3DId = reconstruction.AddPoint3D(xyz, new Track());
		await Assert.That(reconstruction.ExistsPoint3D(point3DId)).IsTrue();
		await Assert.That(reconstruction.Point3D(point3DId).Track.Length).IsEqualTo(0);
		await Assert.That(reconstruction.Points3D.ContainsKey(point3DId)).IsTrue();
		await Assert.That(reconstruction.Points3D.Count).IsEqualTo(1);
		await Assert.That(reconstruction.NumRigs).IsEqualTo(0);
		await Assert.That(reconstruction.NumCameras).IsEqualTo(0);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(0);
		await Assert.That(reconstruction.NumImages).IsEqualTo(0);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(reconstruction.Point3DIds().Contains(point3DId)).IsTrue();

		Reconstruction reconstruction2 = GenerateReconstruction(2);
		var point3D = new Point3D { Xyz = new Vector3d(1.0, 2.0, 3.0) };
		point3D.Track.AddElement(1, 0);
		point3D.Track.AddElement(2, 1);
		reconstruction2.AddPoint3D(5, point3D);
		await Assert.That(reconstruction2.Point3D(5).Track.Length).IsEqualTo(2);
		await Assert.That(reconstruction2.Image(1).Point2DAt(0).HasPoint3D).IsTrue();
		await Assert.That(reconstruction2.Image(2).Point2DAt(1).HasPoint3D).IsTrue();
		await Assert.That(reconstruction2.NumRigs).IsEqualTo(1);
		await Assert.That(reconstruction2.NumCameras).IsEqualTo(1);
		await Assert.That(reconstruction2.NumFrames).IsEqualTo(2);
		await Assert.That(reconstruction2.NumImages).IsEqualTo(2);
		await Assert.That(reconstruction2.NumRegFrames).IsEqualTo(2);
		await Assert.That(reconstruction2.NumPoints3D).IsEqualTo(1);
		await Assert.That(reconstruction2.Point3DIds().Contains(5UL)).IsTrue();
	}

	[Test]
	public async Task Reconstruction_AddObservation()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(3);
		var track = new Track();
		track.AddElement(1, 0);
		track.AddElement(2, 1);
		ulong point3DId = reconstruction.AddPoint3D(xyz, track);
		await Assert.That(reconstruction.Image(1).NumPoints3D).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(1).Point2DAt(0).HasPoint3D).IsTrue();
		await Assert.That(reconstruction.Image(1).Point2DAt(1).HasPoint3D).IsFalse();
		await Assert.That(reconstruction.Image(2).NumPoints3D).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(2).Point2DAt(0).HasPoint3D).IsFalse();
		await Assert.That(reconstruction.Image(2).Point2DAt(1).HasPoint3D).IsTrue();
		await Assert.That(reconstruction.Point3D(point3DId).Track.Length).IsEqualTo(2);
		reconstruction.AddObservation(point3DId, new TrackElement(3, 2));
		await Assert.That(reconstruction.Image(3).NumPoints3D).IsEqualTo(1u);
		await Assert.That(reconstruction.Image(3).Point2DAt(2).HasPoint3D).IsTrue();
		await Assert.That(reconstruction.Point3D(point3DId).Track.Length).IsEqualTo(3);
	}

	[Test]
	public async Task Reconstruction_MergePoints3D()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		ulong point3DId1 = reconstruction.AddPoint3D(new Vector3d(0, 0, 0), new Track());
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		reconstruction.Point3D(point3DId1).Color = new Vector3ub(0, 0, 0);
		ulong point3DId2 = reconstruction.AddPoint3D(new Vector3d(1, 1, 1), new Track());
		reconstruction.AddObservation(point3DId2, new TrackElement(1, 1));
		reconstruction.AddObservation(point3DId2, new TrackElement(2, 1));
		reconstruction.Point3D(point3DId2).Color = new Vector3ub(20, 20, 20);
		ulong mergedPoint3DId = reconstruction.MergePoints3D(point3DId1, point3DId2);
		await Assert.That(reconstruction.ExistsPoint3D(point3DId1)).IsFalse();
		await Assert.That(reconstruction.ExistsPoint3D(point3DId2)).IsFalse();
		await Assert.That(reconstruction.ExistsPoint3D(mergedPoint3DId)).IsTrue();
		await Assert.That(reconstruction.Image(1).Point2DAt(0).Point3DId).IsEqualTo(mergedPoint3DId);
		await Assert.That(reconstruction.Image(1).Point2DAt(1).Point3DId).IsEqualTo(mergedPoint3DId);
		await Assert.That(reconstruction.Image(2).Point2DAt(0).Point3DId).IsEqualTo(mergedPoint3DId);
		await Assert.That(reconstruction.Image(2).Point2DAt(1).Point3DId).IsEqualTo(mergedPoint3DId);
		await Assert.That(reconstruction.Point3D(mergedPoint3DId).Xyz.IsApprox(new Vector3d(0.5, 0.5, 0.5))).IsTrue();
		await Assert.That(reconstruction.Point3D(mergedPoint3DId).Color).IsEqualTo(new Vector3ub(10, 10, 10));
	}

	[Test]
	public async Task Reconstruction_DeletePoint3D()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(1);
		ulong point3DId = reconstruction.AddPoint3D(xyz, new Track());
		reconstruction.AddObservation(point3DId, new TrackElement(1, 0));
		reconstruction.DeletePoint3D(point3DId);
		await Assert.That(reconstruction.ExistsPoint3D(point3DId)).IsFalse();
		await Assert.That(reconstruction.Image(1).NumPoints3D).IsEqualTo(0u);
	}

	[Test]
	public async Task Reconstruction_DeleteObservation()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		ulong point3DId = reconstruction.AddPoint3D(new Vector3d(0, 0, 0), new Track());
		reconstruction.AddObservation(point3DId, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId, new TrackElement(1, 1));
		reconstruction.AddObservation(point3DId, new TrackElement(1, 2));
		reconstruction.DeleteObservation(1, 0);
		await Assert.That(reconstruction.Point3D(point3DId).Track.Length).IsEqualTo(2);
		// COLMAP indexes the image by the point id here (both are 1).
		await Assert.That(reconstruction.Image((uint)point3DId).Point2DAt(0).HasPoint3D).IsFalse();
		reconstruction.DeleteObservation(1, 1);
		await Assert.That(reconstruction.ExistsPoint3D(point3DId)).IsFalse();
		await Assert.That(reconstruction.Image((uint)point3DId).Point2DAt(1).HasPoint3D).IsFalse();
		await Assert.That(reconstruction.Image((uint)point3DId).Point2DAt(2).HasPoint3D).IsFalse();
	}

	[Test]
	public async Task Reconstruction_RegisterFrame()
	{
		Reconstruction reconstruction = GenerateReconstruction(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.Image(1).HasPose).IsTrue();
		await Assert.That(reconstruction.Frame(1).HasPose).IsTrue();
		reconstruction.RegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.Image(1).HasPose).IsTrue();
		await Assert.That(reconstruction.Frame(1).HasPose).IsTrue();
		reconstruction.RegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(1);
		await Assert.That(reconstruction.Image(1).HasPose).IsTrue();
		await Assert.That(reconstruction.Frame(1).HasPose).IsTrue();
		reconstruction.DeRegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.Image(1).HasPose).IsFalse();
		await Assert.That(reconstruction.Frame(1).HasPose).IsFalse();
		reconstruction.DeRegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
		await Assert.That(reconstruction.Image(1).HasPose).IsFalse();
		await Assert.That(reconstruction.Frame(1).HasPose).IsFalse();
	}

	[Test]
	public async Task Reconstruction_DeRegisterFrame()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(3);
		var track = new Track();
		track.AddElement(1, 0);
		track.AddElement(2, 0);
		ulong point3DId = reconstruction.AddPoint3D(xyz, track);

		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(3);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(3);

		reconstruction.DeRegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(2);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(2);
		await Assert.That(reconstruction.ExistsFrame(1)).IsTrue();
		await Assert.That(reconstruction.Frame(1).HasPose).IsFalse();
		// The 3D point had observations in images 1 and 2; after de-registering frame 1, the
		// point should be deleted (track becomes too short).
		await Assert.That(reconstruction.ExistsPoint3D(point3DId)).IsFalse();

		// De-registering an already de-registered frame is a no-op (with warning)
		reconstruction.DeRegisterFrame(1);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(2);
	}

	[Test]
	public async Task Reconstruction_Point3DIds()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		ulong p1 = reconstruction.AddPoint3D(new Vector3d(1, 2, 3), new Track());
		ulong p2 = reconstruction.AddPoint3D(new Vector3d(4, 5, 6), new Track());

		HashSet<ulong> ids = reconstruction.Point3DIds();
		await Assert.That(ids.Count).IsEqualTo(2);
		await Assert.That(ids.Contains(p1)).IsTrue();
		await Assert.That(ids.Contains(p2)).IsTrue();
	}
}
