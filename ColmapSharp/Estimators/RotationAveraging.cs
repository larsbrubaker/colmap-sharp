// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveraging: the free functions of colmap/estimators/rotation_averaging.h and .cc
// (RunRotationAveraging, RunRotationAveragingOnComponent, InitializeRigRotationsFromImages,
// FilterEdgesByRelativeRotation) and the .cc's anonymous-namespace helpers (gravity checks,
// largest component, rig expansion for cameras with unknown cam_from_rig). The estimator
// itself is RotationEstimator.cs. Tests: ColmapSharp.Tests/Estimators/RotationAveragingTests.cs
// (rotation_averaging_test.cc 1:1).
//
// Tier C (iterative).
//
// Translation notes:
// - Frames are walked in ascending id order where COLMAP walks Reconstruction::Frames() (a
//   hash map): the quaternion averages of InitializeRigRotationsFromImages sum their samples
//   in that order, and CreateExpandedReconstruction numbers the singleton rigs (ascending rig
//   id, then COLMAP's std::map sensor order) and the new frames (ascending frame id, then the
//   frame's ordered data ids) in it (divergence 44).
// - NodeHashMap<image_t, Rigid3d> is IReadOnlyDictionary<uint, Rigid3d>; FlatHashSet<image_t>
//   is HashSet<uint> / IReadOnlySet<uint>.
// - LOG(ERROR) messages go to Util/Log.cs; LOG(INFO) messages are dropped.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Rotation averaging entry points. Port of colmap/estimators/rotation_averaging.h.</summary>
public static class RotationAveraging
{
	private static readonly Vector3d KUnknownTranslation = new(double.NaN, double.NaN, double.NaN);

	internal static bool HasGravityPriors(IReadOnlyList<PosePrior> posePriors) =>
		posePriors.Any(posePrior => posePrior.HasGravity());

	internal static bool UseGravity(RotationEstimatorOptions options, IReadOnlyList<PosePrior> posePriors) =>
		options.UseGravity && HasGravityPriors(posePriors);

	/// <summary>
	/// False if any rig has a sensor with unknown sensor_from_rig (logged as an error for
	/// each, as in COLMAP).
	/// </summary>
	internal static bool AllSensorsFromRigKnown(Reconstruction reconstruction)
	{
		bool allKnown = true;
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				if (sensorFromRig is null)
				{
					Log.Error(
						$"Rig {rigId} with unknown sensor_from_rig for sensor {sensorId.Id}, but gravity aligned rotation is "
						+ "requested. Please specify the rig calibration.");
					allKnown = false;
				}
			}
		}

		return allKnown;
	}

	/// <summary>Computes the largest connected component and returns its image ids.</summary>
	internal static HashSet<uint> ComputeLargestConnectedComponentImageIds(
		PoseGraph poseGraph, Reconstruction reconstruction, bool filterUnregistered)
	{
		HashSet<uint> frameIds = poseGraph.LargestConnectedFrameComponent(reconstruction, filterUnregistered);
		var imageIds = new HashSet<uint>();
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			if (frameIds.Contains(image.FrameId))
			{
				imageIds.Add(imageId);
			}
		}

		return imageIds;
	}

	/// <summary>True if any camera in the reconstruction has unknown cam_from_rig.</summary>
	private static bool HasUnknownCamsFromRig(Reconstruction reconstruction)
	{
		foreach (Rig rig in reconstruction.Rigs.Values)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				if (sensorId.Type == SensorType.Camera && sensorFromRig is null)
				{
					return true;
				}
			}
		}

		return false;
	}

	/// <summary>
	/// Creates an expanded reconstruction where cameras with unknown cam_from_rig are split
	/// into separate singleton rigs (each such camera becomes its own rig). This allows
	/// rotation averaging to estimate their orientations independently.
	/// </summary>
	private static Reconstruction CreateExpandedReconstruction(Reconstruction reconstruction)
	{
		var reconExpanded = new Reconstruction();

		// Add all cameras first (required before adding rigs).
		foreach (Camera camera in reconstruction.Cameras.Values)
		{
			reconExpanded.AddCamera(camera);
		}

		// Create expanded rigs with known sensors only. Cameras with unknown cam_from_rig get
		// their own singleton rigs.
		var singletonRigIds = new Dictionary<uint, uint>();

		// First, find the max rig ID to avoid conflicts when creating singleton rigs.
		uint nextRigId = 0;
		foreach (uint rigId in reconstruction.Rigs.Keys)
		{
			nextRigId = Math.Max(nextRigId, rigId + 1);
		}

		foreach ((uint rigId, Rig rig) in reconstruction.Rigs.OrderBy(kv => kv.Key))
		{
			var rigExpanded = new Rig { RigId = rigId };
			rigExpanded.AddRefSensor(rig.RefSensorId);

			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				if (sensorId.Type != SensorType.Camera)
				{
					continue;
				}

				if (sensorFromRig is not null)
				{
					rigExpanded.AddSensor(sensorId, sensorFromRig);
				}
				else
				{
					// Create singleton rig for this camera.
					uint singletonRigId = nextRigId++;
					var rigSingleton = new Rig { RigId = singletonRigId };
					rigSingleton.AddRefSensor(sensorId);
					reconExpanded.AddRig(rigSingleton);
					singletonRigIds[sensorId.Id] = singletonRigId;
				}
			}

			reconExpanded.AddRig(rigExpanded);
		}

		uint nextFrameId = 0;
		foreach (uint frameId in reconstruction.Frames.Keys)
		{
			nextFrameId = Math.Max(nextFrameId, frameId + 1);
		}

		var kUnknownPose = new Rigid3d(
			new Quaterniond(double.NaN, double.NaN, double.NaN, double.NaN), KUnknownTranslation);

		// First pass: build expanded frames with their data ids, and collect images to add
		// afterwards.
		var expandedImages = new List<Image>();
		var expandedFrames = new List<Frame>();
		foreach ((uint frameId, Frame frame) in reconstruction.Frames.OrderBy(kv => kv.Key))
		{
			var frameExpanded = new Frame { FrameId = frameId };
			frameExpanded.SetRigId(frame.RigId);
			frameExpanded.SetRigFromWorld(frame.HasPose ? frame.RigFromWorld() : kUnknownPose);

			Rig originalRig = reconstruction.Rig(frame.RigId);
			foreach (DataId dataId in frame.ImageIds())
			{
				Image image = reconstruction.Image((uint)dataId.Id);
				var imageExpanded = new Image { ImageId = image.ImageId, Name = image.Name };
				imageExpanded.SetCameraId(image.CameraId);

				// Check if camera belongs to this frame's rig (ref sensor or known
				// cam_from_rig).
				SensorId cameraSensorId = image.CameraPtr.SensorId;
				bool belongsToFrameRig = originalRig.RefSensorId == cameraSensorId
					|| originalRig.MaybeSensorFromRig(cameraSensorId) is not null;

				if (belongsToFrameRig)
				{
					// Camera belongs to this frame's rig.
					frameExpanded.AddDataId(imageExpanded.DataId);
					imageExpanded.SetFrameId(frameId);
				}
				else
				{
					// Camera has its own singleton rig, create a new frame for it.
					uint newFrameId = nextFrameId++;
					var newFrame = new Frame { FrameId = newFrameId };
					newFrame.SetRigId(singletonRigIds[image.CameraId]);
					newFrame.AddDataId(imageExpanded.DataId);
					newFrame.SetRigFromWorld(kUnknownPose);
					expandedFrames.Add(newFrame);
					imageExpanded.SetFrameId(newFrameId);
				}

				expandedImages.Add(imageExpanded);
			}

			expandedFrames.Add(frameExpanded);
		}

		// Second pass: add all frames, then all images.
		foreach (Frame frame in expandedFrames)
		{
			reconExpanded.AddFrame(frame);
		}

		foreach (Image image in expandedImages)
		{
			reconExpanded.AddImage(image);
		}

		return reconExpanded;
	}

	/// <summary>
	/// Marks image pairs as invalid whose relative rotation disagrees with the reconstructed
	/// rotations by more than <paramref name="maxAngleDeg"/>. Pairs whose images do not both
	/// have a pose are left untouched. Port of colmap::FilterEdgesByRelativeRotation.
	/// </summary>
	public static void FilterEdgesByRelativeRotation(PoseGraph poseGraph, Reconstruction reconstruction, double maxAngleDeg)
	{
		double maxAngleRad = MathUtils.DegToRad(maxAngleDeg);
		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.ValidEdges().ToList())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			Image image1 = reconstruction.Image(imageId1);
			Image image2 = reconstruction.Image(imageId2);
			if (!image1.HasPose || !image2.HasPose)
			{
				continue;
			}

			Quaterniond cam2FromCam1 = image2.CamFromWorld().Rotation * image1.CamFromWorld().Rotation.Inverse();
			if (cam2FromCam1.AngularDistance(edge.Cam2FromCam1.Rotation) > maxAngleRad)
			{
				poseGraph.SetInvalidEdge(pairId);
			}
		}
	}

	/// <summary>
	/// Estimates rotations for the given connected component <paramref name="activeImageIds"/>
	/// and registers its frames. Handles rigs with unknown cam_from_rig by first solving on an
	/// expanded reconstruction. Does not perform outlier filtering or de-registration.
	/// Port of colmap::RunRotationAveragingOnComponent.
	/// </summary>
	public static bool RunRotationAveragingOnComponent(
		RotationEstimatorOptions options,
		PoseGraph poseGraph,
		IReadOnlySet<uint> activeImageIds,
		Reconstruction reconstruction,
		IReadOnlyList<PosePrior> posePriors)
	{
		if (activeImageIds.Count == 0)
		{
			Log.Error("No connected components found");
			return false;
		}

		if (!HasUnknownCamsFromRig(reconstruction))
		{
			poseGraph.InvalidatePairsOutsideActiveImageIds(activeImageIds);
			return new RotationEstimator(options).EstimateRotations(poseGraph, posePriors, activeImageIds, reconstruction);
		}

		// First solve on an expanded reconstruction where cameras with unknown cam_from_rig
		// are treated as independent rigs.
		Reconstruction reconExpanded = CreateExpandedReconstruction(reconstruction);
		HashSet<uint> expandedActiveImageIds = ComputeLargestConnectedComponentImageIds(
			poseGraph, reconExpanded, options.FilterUnregistered);
		if (expandedActiveImageIds.Count == 0)
		{
			Log.Error("No connected components found");
			return false;
		}

		poseGraph.InvalidatePairsOutsideActiveImageIds(expandedActiveImageIds);
		if (!new RotationEstimator(options).EstimateRotations(poseGraph, posePriors, expandedActiveImageIds, reconExpanded))
		{
			return false;
		}

		var expandedCamsFromWorld = new Dictionary<uint, Rigid3d>();
		foreach ((uint imageId, Image image) in reconExpanded.Images)
		{
			if (image.HasPose)
			{
				expandedCamsFromWorld[imageId] = image.CamFromWorld();
			}
		}

		// Initialize cam_from_rig from preliminary rotation estimates.
		InitializeRigRotationsFromImages(expandedCamsFromWorld, reconstruction, options.RefineSensorFromRig);

		// Expanding rigs changes frame connectivity, so recompute the active set on the
		// original reconstruction before the final solve.
		HashSet<uint> finalActiveImageIds = ComputeLargestConnectedComponentImageIds(
			poseGraph, reconstruction, options.FilterUnregistered);
		if (finalActiveImageIds.Count == 0)
		{
			Log.Error("No connected components found");
			return false;
		}

		poseGraph.InvalidatePairsOutsideActiveImageIds(finalActiveImageIds);
		RotationEstimatorOptions finalOptions = options.Clone();
		finalOptions.SkipInitialization = true;
		finalOptions.UseStratified = false;
		return new RotationEstimator(finalOptions).EstimateRotations(poseGraph, posePriors, finalActiveImageIds, reconstruction);
	}

	/// <summary>
	/// Initialize rig rotations by averaging per-image rotations: estimates cam_from_rig for
	/// cameras with unknown calibration, then computes rig_from_world for each frame. When
	/// <paramref name="refineSensorFromRig"/> is false, the per-sensor cam_from_rig values are
	/// left untouched. Port of colmap::InitializeRigRotationsFromImages.
	/// </summary>
	public static bool InitializeRigRotationsFromImages(
		IReadOnlyDictionary<uint, Rigid3d> camsFromWorld,
		Reconstruction reconstruction,
		bool refineSensorFromRig = true)
	{
		List<Frame> frames = [.. reconstruction.Frames.OrderBy(kv => kv.Key).Select(kv => kv.Value)];

		// Step 1: Estimate cam_from_rig for cameras with unknown calibration. Collect samples
		// across frames, then average.
		var camFromRigSamples = new SortedDictionary<uint, (uint RigId, List<Quaterniond> Rotations)>();
		foreach (Frame frame in frames)
		{
			// Find the rotation of the reference image.
			Quaterniond? refRotation = null;
			foreach (DataId dataId in frame.ImageIds())
			{
				Image image = reconstruction.Image((uint)dataId.Id);
				if (image.IsRefInFrame)
				{
					if (camsFromWorld.TryGetValue((uint)dataId.Id, out Rigid3d refCamFromWorld))
					{
						refRotation = refCamFromWorld.Rotation;
					}

					break;
				}
			}

			if (refRotation is not Quaterniond refRot)
			{
				continue;
			}

			// Collect cam_from_rig samples for non-reference cameras.
			foreach (DataId dataId in frame.ImageIds())
			{
				Image image = reconstruction.Image((uint)dataId.Id);
				if (image.IsRefInFrame || !camsFromWorld.TryGetValue((uint)dataId.Id, out Rigid3d camFromWorld))
				{
					continue;
				}

				if (!camFromRigSamples.TryGetValue(image.CameraId, out var samples))
				{
					samples = (frame.RigId, []);
				}

				samples.Rotations.Add(camFromWorld.Rotation * refRot.Inverse());
				camFromRigSamples[image.CameraId] = (frame.RigId, samples.Rotations);
			}
		}

		if (refineSensorFromRig)
		{
			foreach ((uint cameraId, (uint rigId, List<Quaterniond> samples)) in camFromRigSamples)
			{
				var sensorId = new SensorId(SensorType.Camera, cameraId);
				Rigid3d? existing = reconstruction.Rig(rigId).MaybeSensorFromRig(sensorId);

				// If sensor_from_rig is already fully calibrated, preserve it rather than
				// overwriting with MST-derived approximation.
				if (existing is Rigid3d known && !HasNaN(known.Translation))
				{
					continue;
				}

				Quaterniond camFromRig = Pose.AverageQuaternions(samples, Ones(samples.Count));
				reconstruction.Rig(rigId).SetSensorFromRig(sensorId, new Rigid3d(camFromRig, KUnknownTranslation));
			}
		}
		else
		{
			// Check if rotations are valid.
			foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
			{
				foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
				{
					Check.That(
						sensorFromRig is Rigid3d value && !HasNaN(value.Rotation),
						$"sensor_from_rig has NaN rotation but refine_sensor_from_rig=false (rig_id={rigId}, sensor_id={sensorId.Id})");
				}
			}
		}

		// Step 2: Compute rig_from_world for each frame by averaging across images.
		var rigFromWorldSamples = new List<Quaterniond>();
		foreach (Frame frame in frames)
		{
			rigFromWorldSamples.Clear();
			foreach (DataId dataId in frame.ImageIds())
			{
				uint imageId = (uint)dataId.Id;
				if (!reconstruction.ExistsImage(imageId) || !camsFromWorld.TryGetValue(imageId, out Rigid3d camFromWorld))
				{
					continue;
				}

				Image image = reconstruction.Image(imageId);
				if (image.IsRefInFrame)
				{
					rigFromWorldSamples.Add(camFromWorld.Rotation);
				}
				else
				{
					Rigid3d? maybeCamFromRig = reconstruction.Rig(frame.RigId).MaybeSensorFromRig(image.CameraPtr.SensorId);
					if (maybeCamFromRig is not Rigid3d camFromRig)
					{
						continue;
					}

					rigFromWorldSamples.Add(camFromRig.Rotation.Inverse() * camFromWorld.Rotation);
				}
			}

			if (rigFromWorldSamples.Count > 0)
			{
				Quaterniond rigFromWorld = Pose.AverageQuaternions(rigFromWorldSamples, Ones(rigFromWorldSamples.Count));
				frame.SetRigFromWorld(new Rigid3d(rigFromWorld, KUnknownTranslation));
			}
		}

		return true;
	}

	/// <summary>
	/// High-level rotation averaging solver that handles rig expansion: solves the largest
	/// connected component, then (if MaxRotationErrorDeg &gt; 0) invalidates pairs whose
	/// relative rotation disagrees with the result and de-registers frames outside the new
	/// largest component. Port of colmap::RunRotationAveraging.
	/// </summary>
	public static bool RunRotationAveraging(
		RotationEstimatorOptions options,
		PoseGraph poseGraph,
		Reconstruction reconstruction,
		IReadOnlyList<PosePrior> posePriors)
	{
		// Step 1: Compute the largest connected component and solve rotation averaging on it.
		HashSet<uint> activeImageIds = ComputeLargestConnectedComponentImageIds(
			poseGraph, reconstruction, options.FilterUnregistered);
		if (!RunRotationAveragingOnComponent(options, poseGraph, activeImageIds, reconstruction, posePriors))
		{
			return false;
		}

		// Step 2: Filter outlier pairs by rotation error and update the active set.
		if (options.MaxRotationErrorDeg > 0)
		{
			FilterEdgesByRelativeRotation(poseGraph, reconstruction, options.MaxRotationErrorDeg);

			// Recompute largest connected component among registered frames.
			HashSet<uint> filteredActiveImageIds = ComputeLargestConnectedComponentImageIds(
				poseGraph, reconstruction, filterUnregistered: true);
			if (filteredActiveImageIds.Count == 0)
			{
				Log.Error("No connected components found after filtering");
				return false;
			}

			poseGraph.InvalidatePairsOutsideActiveImageIds(filteredActiveImageIds);

			// De-register frames outside the new active set.
			var activeFrameIds = new HashSet<uint>();
			foreach (uint imageId in filteredActiveImageIds)
			{
				activeFrameIds.Add(reconstruction.Image(imageId).FrameId);
			}

			List<uint> regFrameIdsSnapshot = [.. reconstruction.RegFrameIds];
			foreach (uint frameId in regFrameIdsSnapshot)
			{
				Check.That(reconstruction.Frame(frameId).HasPose);
				if (!activeFrameIds.Contains(frameId))
				{
					reconstruction.DeRegisterFrame(frameId);
				}
			}
		}

		return true;
	}

	private static bool HasNaN(Vector3d v) => double.IsNaN(v.X) || double.IsNaN(v.Y) || double.IsNaN(v.Z);

	private static bool HasNaN(Quaterniond q) =>
		double.IsNaN(q.X) || double.IsNaN(q.Y) || double.IsNaN(q.Z) || double.IsNaN(q.W);

	private static double[] Ones(int count)
	{
		var ones = new double[count];
		Array.Fill(ones, 1.0);
		return ones;
	}
}
