// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ObservationManagerTests: colmap/sfm/observation_manager_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Sfm/ObservationManager*.cs.
// This file holds the helper, Print and the filter cases; ObservationManagerTests.Stats.cs
// holds FilterFrames, the visibility/correspondence statistics and AddImage.
//
// Tier A (exact): integer counts and the filters' scalar thresholds, as in C++.
//
// Translation notes: FlatHashSet{...} arguments are HashSet<ulong>/HashSet<uint>. COLMAP's
// gtest_main seeds the PRNG with 0 before every test; the PRNG is per thread, so tests that
// draw call RandomUtils.SetPRNGSeed(0) and draw every RandomEigenVectord<3>() they need
// before their first await, in the C++ draw order. The drawn values only need to be
// "random" (any of them has a nonzero reprojection error), so drawing them up front changes
// nothing the test checks.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sfm;

public partial class ObservationManagerTests
{
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

	private static HashSet<ulong> Ids(params ulong[] ids) => [.. ids];

	private static HashSet<uint> ImageIds(params uint[] ids) => [.. ids];

	[Test]
	public async Task ObservationManager_Print()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		await Assert.That(obsManager.ToString()).IsEqualTo(
			"ObservationManager(reconstruction=Reconstruction(num_rigs=1, "
			+ "num_cameras=1, num_frames=2, num_reg_frames=2, num_images=2, "
			+ "num_points3D=0), correspondence_graph=null)");
	}

	[Test]
	public async Task ObservationManager_FilterPoints3D()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d random1 = RandomEigen.RandomEigenVector3d();
		Vector3d random2 = RandomEigen.RandomEigenVector3d();

		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		ulong point3DId1 = reconstruction.AddPoint3D(random1, new Track());
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3D(0.0, 0.0, Ids())).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3D(0.0, 0.0, Ids(point3DId1 + 1))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3D(0.0, 0.0, Ids(point3DId1))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId2 = reconstruction.AddPoint3D(random2, new Track());
		reconstruction.AddObservation(point3DId2, new TrackElement(1, 0));
		await Assert.That(obsManager.FilterPoints3D(0.0, 0.0, Ids(point3DId2))).IsEqualTo(1);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId3 = reconstruction.AddPoint3D(new Vector3d(-0.5, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId3, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId3, new TrackElement(2, 0));
		await Assert.That(obsManager.FilterPoints3D(0.0, 0.0, Ids(point3DId3))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3D(0.0, 1e-3, Ids(point3DId3))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId4 = reconstruction.AddPoint3D(new Vector3d(-0.6, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId4, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId4, new TrackElement(2, 0));
		await Assert.That(obsManager.FilterPoints3D(0.1, 0.0, Ids(point3DId4))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3D(0.09, 0.0, Ids(point3DId4))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_FilterPoints3DWithLargeReprojectionErrorTypes()
	{
		var reconstruction = new Reconstruction();
		const uint kCameraId = 1;
		// PINHOLE camera with f=100, image 100x100, so cx=cy=50
		// This gives: pixel_error = 100 * normalized_error
		Camera camera = Camera.CreateFromModelId(kCameraId, CameraModelId.Pinhole, 100, 100, 100);
		reconstruction.AddCamera(camera);
		AddTwoImagesInOneFrame(reconstruction, camera, new Vector2d(50, 50)); // Principal point

		var obsManager = new ObservationManager(reconstruction);

		// Point (0, 0, 2) projects exactly to (50, 50). Point (0.02, 0, 2) projects to
		// (51, 50), giving 1px pixel error and 0.01 normalized error.
		var kPoint3D = new Vector3d(0.02, 0, 2);

		// PIXEL: 1px error, passes at 1.0px, filtered at 0.9px
		ulong id1 = AddTwoViewPoint(reconstruction, kPoint3D);
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(1.0, Ids(id1), ReprojectionErrorType.Pixel)).IsEqualTo(0);
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(0.9, Ids(id1), ReprojectionErrorType.Pixel)).IsEqualTo(2);

		// NORMALIZED: 0.01 normalized error, passes at 0.01, filtered at 0.009
		ulong id2 = AddTwoViewPoint(reconstruction, kPoint3D);
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(0.01, Ids(id2), ReprojectionErrorType.Normalized)).IsEqualTo(0);
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(0.009, Ids(id2), ReprojectionErrorType.Normalized)).IsEqualTo(2);

		// ANGULAR: 0.57 degree error.
		ulong id3 = AddTwoViewPoint(reconstruction, kPoint3D);
		// Threshold 0.6deg does not filter 0.57deg error.
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(0.6, Ids(id3), ReprojectionErrorType.Angular)).IsEqualTo(0);
		// Threshold 0.5deg filters the 0.57deg error.
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(0.5, Ids(id3), ReprojectionErrorType.Angular)).IsEqualTo(2);
	}

	[Test]
	public async Task ObservationManager_FilterPoints3DSphericalSeam()
	{
		var reconstruction = new Reconstruction();
		const uint kCameraId = 1;
		Camera camera = Camera.CreateFromModelId(kCameraId, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		reconstruction.AddCamera(camera);

		// Both images observe the back direction (0, 0, -1) at the x = 0 seam
		// representation; the point projects to x = w. A raw pixel error would be ~width
		// across the seam, but the spherical reprojection error is seam-invariant and ~0.
		AddTwoImagesInOneFrame(reconstruction, camera, new Vector2d(0, 250));

		var obsManager = new ObservationManager(reconstruction);

		ulong id = AddTwoViewPoint(reconstruction, new Vector3d(0, 0, -2));

		// The seam-straddling exact match is not filtered (a pixel metric would wrongly drop
		// it with a ~width error).
		await Assert.That(obsManager.FilterPoints3DWithLargeReprojectionError(maxError: 1.0, Ids(id), ReprojectionErrorType.Pixel)).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_FilterPoints3DInImages()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d random1 = RandomEigen.RandomEigenVector3d();

		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		ulong point3DId1 = reconstruction.AddPoint3D(random1, new Track());
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 0.0, ImageIds())).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 0.0, ImageIds(1))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);

		ulong point3DId2 = reconstruction.AddPoint3D(new Vector3d(-0.4, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId2, new TrackElement(1, 0));
		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 0.0, ImageIds(2))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 0.0, ImageIds(1))).IsEqualTo(1);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId3 = reconstruction.AddPoint3D(new Vector3d(-0.5, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId3, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId3, new TrackElement(2, 0));
		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 0.0, ImageIds(1))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);

		await Assert.That(obsManager.FilterPoints3DInImages(0.0, 1e-3, ImageIds(1))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId4 = reconstruction.AddPoint3D(new Vector3d(-0.6, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId4, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId4, new TrackElement(2, 0));

		await Assert.That(obsManager.FilterPoints3DInImages(0.1, 0.0, ImageIds(1))).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterPoints3DInImages(0.09, 0.0, ImageIds(1))).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_FilterAllPoints()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d random1 = RandomEigen.RandomEigenVector3d();
		Vector3d random2 = RandomEigen.RandomEigenVector3d();

		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		ulong point3DId1 = reconstruction.AddPoint3D(random1, new Track());
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterAllPoints3D(0.0, 0.0)).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId2 = reconstruction.AddPoint3D(random2, new Track());
		reconstruction.AddObservation(point3DId2, new TrackElement(1, 0));
		await Assert.That(obsManager.FilterAllPoints3D(0.0, 0.0)).IsEqualTo(1);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId3 = reconstruction.AddPoint3D(new Vector3d(-0.5, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId3, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId3, new TrackElement(2, 0));
		await Assert.That(obsManager.FilterAllPoints3D(0.0, 0.0)).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterAllPoints3D(0.0, 1e-3)).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		ulong point3DId4 = reconstruction.AddPoint3D(new Vector3d(-0.6, -0.5, 1), new Track());
		reconstruction.AddObservation(point3DId4, new TrackElement(1, 0));
		reconstruction.AddObservation(point3DId4, new TrackElement(2, 0));
		await Assert.That(obsManager.FilterAllPoints3D(0.1, 0.0)).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterAllPoints3D(0.09, 0.0)).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_FilterPoints3DWithShortTracks()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d random1 = RandomEigen.RandomEigenVector3d();
		Vector3d random2 = RandomEigen.RandomEigenVector3d();
		Vector3d random3 = RandomEigen.RandomEigenVector3d();

		Reconstruction reconstruction = GenerateReconstruction(4);
		var obsManager = new ObservationManager(reconstruction);

		ulong point3DId1 = reconstruction.AddPoint3D(random1, new Track());
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));

		ulong point3DId2 = reconstruction.AddPoint3D(random2, new Track());
		reconstruction.AddObservation(point3DId2, new TrackElement(1, 1));
		reconstruction.AddObservation(point3DId2, new TrackElement(2, 1));

		ulong point3DId3 = reconstruction.AddPoint3D(random3, new Track());
		reconstruction.AddObservation(point3DId3, new TrackElement(1, 2));
		reconstruction.AddObservation(point3DId3, new TrackElement(2, 2));
		reconstruction.AddObservation(point3DId3, new TrackElement(3, 2));

		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(3);

		await Assert.That(obsManager.FilterPoints3DWithShortTracks(2)).IsEqualTo(1);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(2);

		await Assert.That(obsManager.FilterPoints3DWithShortTracks(3)).IsEqualTo(2);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);

		await Assert.That(obsManager.FilterPoints3DWithShortTracks(4)).IsEqualTo(3);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_FilterObservationsWithNegativeDepth()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		ulong point3DId1 = reconstruction.AddPoint3D(new Vector3d(0, 0, 1), new Track());
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		await Assert.That(obsManager.FilterObservationsWithNegativeDepth()).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		reconstruction.Point3D(point3DId1).XyzParams[2] = 0.001;
		await Assert.That(obsManager.FilterObservationsWithNegativeDepth()).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		reconstruction.Point3D(point3DId1).XyzParams[2] = 0.0;
		await Assert.That(obsManager.FilterObservationsWithNegativeDepth()).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		reconstruction.Point3D(point3DId1).XyzParams[2] = 0.001;
		await Assert.That(obsManager.FilterObservationsWithNegativeDepth()).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		reconstruction.Point3D(point3DId1).XyzParams[2] = 0.0;
		await Assert.That(obsManager.FilterObservationsWithNegativeDepth()).IsEqualTo(1);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	// The shared setup of the two single-frame filter cases: rig 1, frame 1 holding images 1
	// and 2, each with one point at xy.
	private static void AddTwoImagesInOneFrame(Reconstruction reconstruction, Camera camera, Vector2d xy)
	{
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);

		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		frame.AddDataId(new DataId(camera.SensorId, 1));
		frame.AddDataId(new DataId(camera.SensorId, 2));
		frame.SetRigFromWorld(new Rigid3d());
		reconstruction.AddFrame(frame);

		for (uint imageId = 1; imageId <= 2; ++imageId)
		{
			var image = new Image { ImageId = imageId };
			image.SetCameraId(camera.CameraId);
			image.SetFrameId(1);
			image.SetPoints2D([xy]);
			reconstruction.AddImage(image);
		}
	}

	private static ulong AddTwoViewPoint(Reconstruction reconstruction, Vector3d xyz)
	{
		ulong id = reconstruction.AddPoint3D(xyz, new Track());
		reconstruction.AddObservation(id, new TrackElement(1, 0));
		reconstruction.AddObservation(id, new TrackElement(2, 0));
		return id;
	}
}
