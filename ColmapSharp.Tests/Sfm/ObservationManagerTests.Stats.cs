// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ObservationManagerTests (statistics): the FilterFrames, NumVisiblePoints3D,
// NumVisibleCorrespondences(WithoutCorrespondenceGraph), Point3DVisibilityScore and AddImage
// cases of colmap/sfm/observation_manager_test.cc, ported 1:1. The helpers and the filter
// cases are in ObservationManagerTests.cs. Finalize() is FinalizeGraph(); the Eigen vector of
// per-level pyramid scores is a ulong array, scores.sum() its sum and
// scores.bottomRows(n - 1).sum() the sum over all but the first level.

using ColmapSharp.Feature;
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
	[Test]
	public async Task ObservationManager_FilterFrames()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d random1 = RandomEigen.RandomEigenVector3d();

		Reconstruction reconstruction = GenerateReconstruction(4);
		var obsManager = new ObservationManager(reconstruction);
		ulong point3DId1 = reconstruction.AddPoint3D(random1, new Track());
		obsManager.AddObservation(point3DId1, new TrackElement(1, 0));
		obsManager.AddObservation(point3DId1, new TrackElement(2, 0));
		obsManager.AddObservation(point3DId1, new TrackElement(3, 0));
		void FilterFrames(double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam)
		{
			foreach (uint frameId in obsManager.FindFramesToFilter(
				minFocalLengthRatio: minFocalLengthRatio,
				maxFocalLengthRatio: maxFocalLengthRatio,
				maxExtraParam: maxExtraParam,
				minNumObservations: 1))
			{
				obsManager.DeRegisterFrame(frameId);
			}
		}

		FilterFrames(0.0, 10.0, 1.0);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(3);
		reconstruction.DeleteObservation(3, 0);
		FilterFrames(0.0, 10.0, 1.0);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(2);
		FilterFrames(0.0, 0.9, 1.0);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(0);
	}

	[Test]
	public async Task ObservationManager_NumVisiblePoints3D()
	{
		const uint kImageId1 = 1;
		var (reconstruction, correspondenceGraph) = BuildTwoImageScene(focalLength: 10, size: 10, Enumerable.Repeat(Vector2d.Zero, 10).ToList());
		var obsManager = new ObservationManager(reconstruction, correspondenceGraph);

		await Assert.That(obsManager.NumObservations(kImageId1)).IsEqualTo(10u);
		await Assert.That(obsManager.NumCorrespondences(kImageId1)).IsEqualTo(10u);

		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(0u);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(1u);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(2u);
		obsManager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(2u);
		obsManager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(1u);
		obsManager.DecrementCorrespondenceHasPoint3D(kImageId1, 1);
		await Assert.That(obsManager.NumVisiblePoints3D(kImageId1)).IsEqualTo(0u);
	}

	[Test]
	public async Task ObservationManager_NumVisibleCorrespondences()
	{
		var reconstruction = new Reconstruction();
		const uint kImageId1 = 1;
		const uint kImageId2 = 2;
		const uint kImageId3 = 3;
		const uint kCameraId = 1;

		Camera camera = Camera.CreateFromModelId(kCameraId, CameraModelId.Pinhole, focalLength: 10, width: 10, height: 10);
		reconstruction.AddCamera(camera);

		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);

		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		frame.AddDataId(new DataId(camera.SensorId, kImageId1));
		frame.AddDataId(new DataId(camera.SensorId, kImageId2));
		frame.AddDataId(new DataId(camera.SensorId, kImageId3));
		reconstruction.AddFrame(frame);

		foreach (uint imageId in new[] { kImageId1, kImageId2, kImageId3 })
		{
			var image = new Image { ImageId = imageId };
			image.SetCameraId(kCameraId);
			image.SetFrameId(frame.FrameId);
			image.SetPoints2D(Enumerable.Repeat(Vector2d.Zero, 10).ToList());
			reconstruction.AddImage(image);
		}

		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(kImageId1, 10);
		correspondenceGraph.AddImage(kImageId2, 10);
		correspondenceGraph.AddImage(kImageId3, 10);

		var twoViewGeometry12 = new TwoViewGeometry();
		for (uint i = 0; i < 5; ++i)
		{
			twoViewGeometry12.InlierMatches.Add(new FeatureMatch(i, i));
		}

		correspondenceGraph.AddTwoViewGeometry(kImageId1, kImageId2, twoViewGeometry12);

		var twoViewGeometry13 = new TwoViewGeometry();
		for (uint i = 5; i < 8; ++i)
		{
			twoViewGeometry13.InlierMatches.Add(new FeatureMatch(i, i));
		}

		correspondenceGraph.AddTwoViewGeometry(kImageId1, kImageId3, twoViewGeometry13);

		var twoViewGeometry23 = new TwoViewGeometry();
		for (uint i = 0; i < 2; ++i)
		{
			twoViewGeometry23.InlierMatches.Add(new FeatureMatch(i, i + 5));
		}

		correspondenceGraph.AddTwoViewGeometry(kImageId2, kImageId3, twoViewGeometry23);

		correspondenceGraph.FinalizeGraph();

		var obsManager = new ObservationManager(reconstruction, correspondenceGraph);

		// Initially, frame is not registered, so visible correspondences should be 0
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId1)).IsEqualTo(0u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId2)).IsEqualTo(0u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId3)).IsEqualTo(0u);

		// Set pose and register the frame (contains all three images)
		reconstruction.Frame(1).SetRigFromWorld(new Rigid3d());
		obsManager.RegisterFrame(1);
		// All images now have visible correspondences with each other
		// Image 1: 5 (to image 2) + 3 (to image 3) = 8
		// Image 2: 5 (to image 1) + 2 (to image 3) = 7
		// Image 3: 3 (to image 1) + 2 (to image 2) = 5
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId1)).IsEqualTo(8u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId2)).IsEqualTo(7u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId3)).IsEqualTo(5u);

		// Deregister the frame
		obsManager.DeRegisterFrame(1);
		// All visible correspondences should drop back to 0
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId1)).IsEqualTo(0u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId2)).IsEqualTo(0u);
		await Assert.That(obsManager.NumVisibleCorrespondences(kImageId3)).IsEqualTo(0u);
	}

	[Test]
	public async Task ObservationManager_NumVisibleCorrespondencesWithoutCorrespondenceGraph()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction);
		await Assert.That(obsManager.NumVisibleCorrespondences(1)).IsEqualTo(0u);
		await Assert.That(obsManager.NumVisibleCorrespondences(2)).IsEqualTo(0u);
	}

	[Test]
	public async Task ObservationManager_Point3DVisibilityScore()
	{
		const uint kImageId1 = 1;
		var points2D = new List<Vector2d>();
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				points2D.Add(new Vector2d(i, j));
			}
		}

		var (reconstruction, correspondenceGraph) = BuildTwoImageScene(focalLength: 4, size: 4, points2D);
		var obsManager = new ObservationManager(reconstruction, correspondenceGraph);

		await Assert.That(obsManager.NumObservations(kImageId1)).IsEqualTo(16u);
		await Assert.That(obsManager.NumCorrespondences(kImageId1)).IsEqualTo(16u);

		ulong[] scores = new ulong[ObservationManager.kNumPoint3DVisibilityPyramidLevels];
		for (int i = 1; i <= ObservationManager.kNumPoint3DVisibilityPyramidLevels; ++i)
		{
			scores[i - 1] = (ulong)((1 << i) * (1 << i));
		}

		ulong sum = 0;
		foreach (ulong score in scores)
		{
			sum += score;
		}

		ulong tailSum = sum - scores[0];

		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(0UL);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum + tailSum);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 4);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum + (2 * tailSum));
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 4);
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 5);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum + (3 * tailSum));
		obsManager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum + (3 * tailSum));
		obsManager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo(sum + (2 * tailSum));
		obsManager.IncrementCorrespondenceHasPoint3D(kImageId1, 2);
		await Assert.That(obsManager.Point3DVisibilityScore(kImageId1)).IsEqualTo((2 * sum) + (2 * tailSum));
	}

	[Test]
	public async Task ObservationManager_AddImage()
	{
		// Images 1,2 start registered with a triangulated point. Image 3 arrives
		// incrementally with a match on a previously-unmatched point in image 1, which
		// increases image 1's num_observations in the correspondence graph. AddImage must
		// refresh the cached stats for existing images to stay in sync.

		var graph = new CorrespondenceGraph();
		graph.AddImage(1, 10);
		graph.AddImage(2, 10);
		graph.AddTwoViewGeometry(1, 2, new TwoViewGeometry { InlierMatches = [new FeatureMatch(0, 0), new FeatureMatch(1, 1)] });

		Reconstruction reconstruction = GenerateReconstruction(2);
		var obsManager = new ObservationManager(reconstruction, graph);

		// Triangulate a point connecting image1:0 and image2:0.
		var track = new Track();
		track.AddElement(1, 0);
		track.AddElement(2, 0);
		ulong point3DId = obsManager.AddPoint3D(new Vector3d(0, 0, 1), track);

		// Image 3 arrives. Match {2, 1} hits image1:2 which had no prior correspondences, so
		// image 1's num_observations in the correspondence graph goes 2 -> 3.
		graph.AddImage(3, 10);
		graph.AddTwoViewGeometry(1, 3, new TwoViewGeometry { InlierMatches = [new FeatureMatch(0, 0), new FeatureMatch(2, 1)] });
		graph.AddTwoViewGeometry(2, 3, new TwoViewGeometry { InlierMatches = [new FeatureMatch(0, 0)] });

		{
			Camera camera = reconstruction.Camera(1);
			var frame = new Frame { FrameId = 3 };
			frame.SetRigId(1);
			frame.AddDataId(new DataId(camera.SensorId, 3));
			reconstruction.AddFrame(frame);
			var image = new Image { ImageId = 3, Name = "image3" };
			image.SetCameraId(camera.CameraId);
			image.SetFrameId(3);
			image.SetPoints2D(Enumerable.Repeat(Vector2d.Zero, 10).ToList());
			reconstruction.AddImage(image);
		}

		obsManager.AddImage(3);

		// Image 3 sees the triangulated point via retroactive visibility.
		await Assert.That(obsManager.NumVisiblePoints3D(3)).IsEqualTo(1u);
		await Assert.That(obsManager.NumObservations(3)).IsEqualTo(2u);
		await Assert.That(obsManager.NumCorrespondences(3)).IsEqualTo(3u);

		// Verify AddImage refreshed image 1's cached stats (was 2 obs, now 3).
		await Assert.That(obsManager.NumObservations(1)).IsEqualTo(3u);
		await Assert.That(obsManager.NumCorrespondences(1)).IsEqualTo(4u);

		// Register image 3 and add it to the existing track.
		reconstruction.Frame(3).SetRigFromWorld(new Rigid3d());
		obsManager.RegisterFrame(3);
		obsManager.AddObservation(point3DId, new TrackElement(3, 0));
		await Assert.That(reconstruction.Point3D(point3DId).Track.Length).IsEqualTo(3);

		// Triangulate a new point on image1:2 <-> image3:1. This calls
		// IncrementCorrespondenceHasPoint3D on image 1, which would violate
		// THROW_CHECK_LE(num_visible_points3D, num_observations) if the cached stats were
		// stale.
		var track2 = new Track();
		track2.AddElement(1, 2);
		track2.AddElement(3, 1);
		obsManager.AddPoint3D(new Vector3d(1, 0, 1), track2);
	}

	// The shared setup of NumVisiblePoints3D and Point3DVisibilityScore: one PINHOLE camera,
	// frame 1 (no pose) holding images 1 and 2 with the same points, and a finalized graph
	// matching point i to point i.
	private static (Reconstruction Reconstruction, CorrespondenceGraph Graph) BuildTwoImageScene(
		double focalLength, int size, List<Vector2d> points2D)
	{
		const uint kImageId1 = 1;
		const uint kImageId2 = 2;
		const uint kCameraId = 1;
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(kCameraId, CameraModelId.Pinhole, focalLength, size, size);
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(camera.SensorId);
		reconstruction.AddRig(rig);
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(rig.RigId);
		frame.AddDataId(new DataId(camera.SensorId, kImageId1));
		frame.AddDataId(new DataId(camera.SensorId, kImageId2));
		reconstruction.AddFrame(frame);
		var image = new Image { ImageId = kImageId1 };
		image.SetCameraId(kCameraId);
		image.SetFrameId(frame.FrameId);
		image.SetPoints2D(points2D);
		reconstruction.AddImage(image);
		image.ImageId = kImageId2;
		reconstruction.AddImage(image);

		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(kImageId1, points2D.Count);
		correspondenceGraph.AddImage(kImageId2, points2D.Count);
		var twoViewGeometry = new TwoViewGeometry();
		for (uint i = 0; i < points2D.Count; ++i)
		{
			twoViewGeometry.InlierMatches.Add(new FeatureMatch(i, i));
		}

		correspondenceGraph.AddTwoViewGeometry(kImageId1, kImageId2, twoViewGeometry);
		correspondenceGraph.FinalizeGraph();
		return (reconstruction, correspondenceGraph);
	}
}
