// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapper.Register: the next-image registration half of
// colmap/sfm/incremental_mapper.cc: RegisterNextImage (central absolute pose from 2D-3D
// correspondences, with focal length / distortion estimation), RegisterNextGeneralFrame
// (generalized absolute pose for multi-camera rigs) and RegisterNextStructureLessImage
// (Zheng and Wu's structure-less resection from 2D-2D correspondences). The rest of the
// mapper is in IncrementalMapper.cs and IncrementalMapper.Adjust.cs.
//
// Translation notes:
// - Correspondences are collected in point2D order and, per point, in the correspondence
//   graph's order, as in C++; the per-point duplicate set only answers membership.
// - C++ copies the cameras it hands to the generalized and structure-less estimators into
//   vectors; they are cloned here for the same independence.
// - The structure-less refinement sets Ceres' logging to SILENT in C++; the port's solver
//   does not log, so only the loss function and print_summary are set.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm;

public sealed partial class IncrementalMapper
{
	/// <summary>
	/// Attempts to register an image to the existing model. This requires that a previous
	/// call to RegisterInitialImagePair was successful (or that the reconstruction was begun
	/// with registered frames).
	/// </summary>
	public bool RegisterNextImage(Options options, uint imageId)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		Check.Gt(reconstruction.NumRegFrames, 0);
		Check.That(options.Check());

		Image image = reconstruction.Image(imageId);
		Camera camera = image.CameraPtr;

		foreach (Rigid3d? sensorFromRig in image.FramePtr.RigPtr.NonRefSensors.Values)
		{
			Check.That(
				sensorFromRig.HasValue,
				"Registration only implemented for frames with known sensor_from_rig poses");
		}

		// Use central camera pose estimation for trivial frames and when we don't have a good
		// estimate of the camera's focal length, because we don't have a focal length
		// estimator for non-central/generalized cameras.
		if (image.FramePtr.RigPtr.NumSensors > 1)
		{
			bool allCamerasHaveGoodFocalLength = true;
			foreach (DataId dataId in image.FramePtr.ImageIds())
			{
				Image frameImage = reconstruction.Image((uint)dataId.Id);
				if ((!frameImage.CameraPtr.HasPriorFocalLength
						&& CountAt(_regStats.NumRegImagesPerCamera, frameImage.CameraId) == 0)
					|| frameImage.CameraPtr.HasBogusParams(
						options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
				{
					allCamerasHaveGoodFocalLength = false;
					break;
				}
			}

			if (allCamerasHaveGoodFocalLength)
			{
				return RegisterNextGeneralFrame(options, image.FramePtr);
			}
		}

		Increment(_regStats.NumRegTrials, imageId);

		// Check if enough 2D-3D correspondences.
		if (obsManager.NumVisiblePoints3D(imageId) < options.AbsPoseMinNumInliers)
		{
			return false;
		}

		// Search for 2D-3D correspondences.
		var triCorrs = new List<(uint Point2DIdx, ulong Point3DId)>();
		var triPoints2D = new List<Vector2d>();
		var triPoints3D = new List<Vector3d>();

		CorrespondenceGraph correspondenceGraph = _databaseCache.CorrespondenceGraph;

		var corrPoint3DIds = new HashSet<ulong>();
		for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
		{
			Point2D point2D = image.Points2D[(int)point2DIdx];

			corrPoint3DIds.Clear();
			foreach (CorrespondenceGraph.Correspondence corr in correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
			{
				Image corrImage = reconstruction.Image(corr.ImageId);
				if (!corrImage.HasPose)
				{
					continue;
				}

				Point2D corrPoint2D = corrImage.Points2D[(int)corr.Point2DIdx];
				if (!corrPoint2D.HasPoint3D)
				{
					continue;
				}

				// Avoid duplicate correspondences.
				if (corrPoint3DIds.Contains(corrPoint2D.Point3DId))
				{
					continue;
				}

				Camera corrCamera = corrImage.CameraPtr;

				// Avoid correspondences to images with bogus camera parameters.
				if (corrCamera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
				{
					continue;
				}

				Point3D point3D = reconstruction.Point3D(corrPoint2D.Point3DId);

				triCorrs.Add((point2DIdx, corrPoint2D.Point3DId));
				corrPoint3DIds.Add(corrPoint2D.Point3DId);
				triPoints2D.Add(point2D.Xy);
				triPoints3D.Add(point3D.Xyz);
			}
		}

		// The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()` can only
		// differ, when there are images with bogus camera parameters, and hence we skip some
		// of the 2D-3D correspondences.
		if (triPoints2D.Count < options.AbsPoseMinNumInliers)
		{
			return false;
		}

		// 2D-3D estimation.

		// Only refine / estimate focal length, if no focal length was specified (manually or
		// through EXIF) and if it was not already estimated previously from another image
		// (when multiple images share the same camera parameters).

		// Note that we use single-threaded RANSAC here, because benchmarking showed no
		// significant speedup for multi-threaded RANSAC here (as opposed to the generalized
		// absolute pose estimation).
		var absPoseOptions = new AbsolutePoseEstimationOptions();
		absPoseOptions.RansacOptions.MaxError = options.AbsPoseMaxError;
		absPoseOptions.RansacOptions.MinInlierRatio = options.AbsPoseMinInlierRatio;
		absPoseOptions.RansacOptions.RandomSeed = options.RandomSeed;

		var absPoseRefinementOptions = new AbsolutePoseRefinementOptions();
		if (options.ConstantCameras.Contains(image.CameraId))
		{
			absPoseOptions.EstimateFocalLength = false;
			absPoseRefinementOptions.RefineFocalLength = false;
			absPoseRefinementOptions.RefineExtraParams = false;
		}
		else
		{
			if (CountAt(_regStats.NumRegImagesPerCamera, image.CameraId) > 0)
			{
				// Camera already refined from another image with the same camera.
				if (camera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
				{
					absPoseOptions.EstimateFocalLength = !camera.HasPriorFocalLength;
					absPoseRefinementOptions.RefineFocalLength = true;
					absPoseRefinementOptions.RefineExtraParams = true;
				}
				else
				{
					absPoseOptions.EstimateFocalLength = false;
					absPoseRefinementOptions.RefineFocalLength = false;
					absPoseRefinementOptions.RefineExtraParams = false;
				}
			}
			else
			{
				// Camera not refined before. Note that the camera parameters might have been
				// changed before but the image was filtered, so we explicitly reset the camera
				// parameters and try to re-estimate them.
				ResetCameraParams(camera);
				absPoseOptions.EstimateFocalLength = !camera.HasPriorFocalLength;
				absPoseRefinementOptions.RefineFocalLength = true;
				absPoseRefinementOptions.RefineExtraParams = true;
			}

			if (!options.AbsPoseRefineFocalLength)
			{
				absPoseOptions.EstimateFocalLength = false;
				absPoseRefinementOptions.RefineFocalLength = false;
			}

			if (!options.AbsPoseRefineExtraParams)
			{
				absPoseRefinementOptions.RefineExtraParams = false;
			}

			// Omnidirectional cameras (e.g. EQUIRECTANGULAR) have no focal length, and their
			// parameters (e.g. image dimensions) are not distortion coefficients to be refined
			// during registration.
			if (!camera.IsPerspective)
			{
				absPoseOptions.EstimateFocalLength = false;
				absPoseRefinementOptions.RefineFocalLength = false;
				absPoseRefinementOptions.RefineExtraParams = false;
			}
		}

		// If any of the cameras in the same rig has bogus cameras, reset them to the original
		// values from the database, so we have a chance of recovering from previous failed
		// estimations. Notice that this function will be called for non-trivial frames, when
		// there is no good estimate for the focal length or one of the rig's cameras has bogus
		// parameters.
		foreach (DataId dataId in image.FramePtr.ImageIds())
		{
			Image frameImage = reconstruction.Image((uint)dataId.Id);
			if (frameImage.CameraPtr.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
			{
				ResetCameraParams(frameImage.CameraPtr);
			}
		}

		Rigid3d camFromWorld = Rigid3d.Identity;
		if (!PoseEstimation.EstimateAbsolutePose(
			absPoseOptions, triPoints2D, triPoints3D, ref camFromWorld, camera, out int numInliers, out bool[] inlierMask))
		{
			return false;
		}

		if (numInliers < options.AbsPoseMinNumInliers)
		{
			return false;
		}

		// Pose refinement.
		if (!PoseEstimation.RefineAbsolutePose(
			absPoseRefinementOptions, inlierMask, triPoints2D, triPoints3D, ref camFromWorld, camera))
		{
			return false;
		}

		// Continue tracks.
		image.FramePtr.SetCamFromWorld(image.CameraId, camFromWorld);

		obsManager.RegisterFrame(image.FrameId);
		RegisterFrameEvent(image.FrameId);

		IncrementalTriangulator triangulator = Check.NotNull(_triangulator);
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (inlierMask[i])
			{
				(uint point2DIdx, ulong point3DId) = triCorrs[i];
				if (!image.Points2D[(int)point2DIdx].HasPoint3D)
				{
					obsManager.AddObservation(point3DId, new TrackElement(imageId, point2DIdx));
					triangulator.AddModifiedPoint3D(point3DId);
				}
			}
		}

		return true;
	}

	// Registers a frame using generalized absolute pose estimation. Only called for frames
	// whose rig has more than one sensor.
	private bool RegisterNextGeneralFrame(Options options, Frame frame)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		Check.Gt(frame.RigPtr.NumSensors, 1);

		var triCorrs = new List<(uint Point2DIdx, uint ImageId, ulong Point3DId)>();
		var triPoints2D = new List<Vector2d>();
		var triPoints3D = new List<Vector3d>();
		var triCameraIdxs = new List<int>();

		var camsFromRig = new List<Rigid3d>(frame.RigPtr.NumSensors);
		var cameras = new List<Camera>(frame.RigPtr.NumSensors);

		CorrespondenceGraph correspondenceGraph = _databaseCache.CorrespondenceGraph;

		foreach (DataId dataId in frame.ImageIds())
		{
			uint imageId = (uint)dataId.Id;
			Image image = reconstruction.Image(imageId);
			Camera camera = image.CameraPtr;

			int cameraIdx = cameras.Count;
			camsFromRig.Add(frame.RigPtr.IsRefSensor(camera.SensorId) ? Rigid3d.Identity : frame.RigPtr.SensorFromRig(camera.SensorId));
			cameras.Add(camera.Clone());

			Increment(_regStats.NumRegTrials, imageId);

			var corrPoint3DIds = new HashSet<ulong>();
			for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				Point2D point2D = image.Points2D[(int)point2DIdx];

				corrPoint3DIds.Clear();
				foreach (CorrespondenceGraph.Correspondence corr in correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
				{
					Image corrImage = reconstruction.Image(corr.ImageId);
					if (!corrImage.HasPose)
					{
						continue;
					}

					Point2D corrPoint2D = corrImage.Points2D[(int)corr.Point2DIdx];
					if (!corrPoint2D.HasPoint3D)
					{
						continue;
					}

					// Avoid duplicate correspondences.
					if (corrPoint3DIds.Contains(corrPoint2D.Point3DId))
					{
						continue;
					}

					Camera corrCamera = corrImage.CameraPtr;

					// Avoid correspondences to images with bogus camera parameters.
					if (corrCamera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
					{
						continue;
					}

					Point3D point3D = reconstruction.Point3D(corrPoint2D.Point3DId);

					triCorrs.Add((point2DIdx, imageId, corrPoint2D.Point3DId));
					corrPoint3DIds.Add(corrPoint2D.Point3DId);
					triPoints2D.Add(point2D.Xy);
					triPoints3D.Add(point3D.Xyz);
					triCameraIdxs.Add(cameraIdx);
				}
			}
		}

		// The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()` can only
		// differ, when there are images with bogus camera parameters, and hence we skip some
		// of the 2D-3D correspondences.
		if (triPoints2D.Count < options.AbsPoseMinNumInliers)
		{
			return false;
		}

		// 2D-3D estimation.
		var absPoseOptions = new RansacOptions
		{
			MaxError = options.AbsPoseMaxError,
			MinInlierRatio = options.AbsPoseMinInlierRatio,
			RandomSeed = options.RandomSeed,
		};

		var absPoseRefinementOptions = new AbsolutePoseRefinementOptions
		{
			RefineFocalLength = false,
			RefineExtraParams = false,
		};

		Rigid3d rigFromWorld = Rigid3d.Identity;
		if (!GeneralizedPoseEstimation.EstimateGeneralizedAbsolutePose(
			absPoseOptions,
			triPoints2D,
			triPoints3D,
			triCameraIdxs,
			camsFromRig,
			cameras,
			ref rigFromWorld,
			out int numInliers,
			out bool[] inlierMask))
		{
			return false;
		}

		if (numInliers < options.AbsPoseMinNumInliers)
		{
			return false;
		}

		// Pose refinement.
		if (!GeneralizedPoseEstimation.RefineGeneralizedAbsolutePose(
			absPoseRefinementOptions,
			inlierMask,
			triPoints2D,
			triPoints3D,
			triCameraIdxs,
			camsFromRig,
			ref rigFromWorld,
			cameras))
		{
			return false;
		}

		// Continue tracks.
		frame.SetRigFromWorld(rigFromWorld);

		obsManager.RegisterFrame(frame.FrameId);
		RegisterFrameEvent(frame.FrameId);

		IncrementalTriangulator triangulator = Check.NotNull(_triangulator);
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (inlierMask[i])
			{
				(uint point2DIdx, uint imageId, ulong point3DId) = triCorrs[i];
				Image image = reconstruction.Image(imageId);
				if (!image.Points2D[(int)point2DIdx].HasPoint3D)
				{
					obsManager.AddObservation(point3DId, new TrackElement(imageId, point2DIdx));
					triangulator.AddModifiedPoint3D(point3DId);
				}
			}
		}

		return true;
	}

	/// <summary>
	/// Attempts to register an image using structure-less resectioning as proposed in
	/// "Structure from Motion Using Structure-less Resection" by Zheng and Wu: from 2D-2D
	/// correspondences to registered images, continuing or triangulating tracks afterwards.
	/// </summary>
	public bool RegisterNextStructureLessImage(Options options, uint imageId)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		if (reconstruction.NumRegImages < 2)
		{
			// Structure-less registration requires at least 2 registered images.
			return false;
		}

		Check.That(options.Check());

		Increment(_regStats.NumStructureLessRegTrials, imageId);

		Image image = reconstruction.Image(imageId);
		Camera camera = image.CameraPtr;

		// Search for structure-less correspondences.

		// Each 2D-2D correspondence contributes 1 geometric constraint, whereas each 2D-3D
		// correspondence contributes 2, so require 2x the number of inliers.
		int minNumInliers = 2 * options.AbsPoseMinNumInliers;

		// Check if enough 2D-2D correspondences.
		if (obsManager.NumVisibleCorrespondences(imageId) < minNumInliers)
		{
			return false;
		}

		CorrespondenceGraph correspondenceGraph = _databaseCache.CorrespondenceGraph;

		var point2DIdxs = new List<uint>();
		var corrs = new List<CorrespondenceGraph.Correspondence>();
		var points2D = new List<Vector2d>();
		var worldPoints2D = new List<Vector2d>();
		var worldCameraIdxs = new List<int>();
		var worldCamsFromWorld = new List<Rigid3d>();
		var worldCameras = new List<Camera>();
		var worldImageIdToCameraIdx = new Dictionary<uint, int>();

		uint numPoints2D = image.NumPoints2D;
		for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
		{
			Point2D point2D = image.Points2D[(int)point2DIdx];

			foreach (CorrespondenceGraph.Correspondence corr in correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
			{
				Image worldImage = reconstruction.Image(corr.ImageId);
				if (!worldImage.HasPose)
				{
					continue;
				}

				Camera worldCamera = worldImage.CameraPtr;

				// Avoid correspondences to images with bogus camera parameters.
				if (worldCamera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
				{
					continue;
				}

				worldPoints2D.Add(worldImage.Points2D[(int)corr.Point2DIdx].Xy);
				points2D.Add(point2D.Xy);

				if (!worldImageIdToCameraIdx.TryGetValue(corr.ImageId, out int worldCameraIdx))
				{
					worldCameraIdx = worldCameras.Count;
					worldImageIdToCameraIdx.Add(corr.ImageId, worldCameraIdx);
					worldCamsFromWorld.Add(worldImage.CamFromWorld());
					worldCameras.Add(worldCamera.Clone());
				}

				worldCameraIdxs.Add(worldCameraIdx);

				point2DIdxs.Add(point2DIdx);
				corrs.Add(corr);
			}
		}

		// Check if we pass the minimum number of inliers.
		if (worldPoints2D.Count < minNumInliers)
		{
			return false;
		}

		// Structure-less resectioning.
		var absPoseOptions = new StructureLessAbsolutePoseEstimationOptions();
		// Structure-less resectioning uses epipolar Sampson error, so we are stricter in
		// accepting 2D-2D correspondences inliers.
		absPoseOptions.RansacOptions.MaxError = 0.5 * options.AbsPoseMaxError;
		absPoseOptions.RansacOptions.MinInlierRatio = options.AbsPoseMinInlierRatio;
		// As opposed to structure-based resectioning, structure-less resectioning is based on
		// an expensive minimal solver, so we use multi-threading, which leads to a significant
		// speedup based on benchmarking.
		absPoseOptions.RansacOptions.NumThreads = options.NumThreads;

		var absPoseRefinementOptions = new BundleAdjustmentOptions();
		if (absPoseRefinementOptions.Ceres is not null)
		{
			absPoseRefinementOptions.Ceres.LossFunctionType = BundleAdjustmentLossFunctionType.Cauchy;
		}

		absPoseRefinementOptions.PrintSummary = false;
		if (CountAt(_regStats.NumRegImagesPerCamera, image.CameraId) > 0)
		{
			// Camera already refined from another image with the same camera.
			if (camera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam))
			{
				// Previously refined camera has bogus parameters, so reset parameters and try
				// to re-estimage.
				ResetCameraParams(camera);
				absPoseRefinementOptions.RefineFocalLength = true;
				absPoseRefinementOptions.RefineExtraParams = true;
			}
			else
			{
				absPoseRefinementOptions.RefineFocalLength = false;
				absPoseRefinementOptions.RefineExtraParams = false;
			}
		}
		else
		{
			// Camera not refined before. Note that the camera parameters might have been
			// changed before but the image was filtered, so we explicitly reset the camera
			// parameters and try to re-estimate them.
			ResetCameraParams(camera);
			absPoseRefinementOptions.RefineFocalLength = true;
			absPoseRefinementOptions.RefineExtraParams = true;
		}

		if (!options.AbsPoseRefineFocalLength)
		{
			absPoseRefinementOptions.RefineFocalLength = false;
		}

		if (!options.AbsPoseRefineExtraParams)
		{
			absPoseRefinementOptions.RefineExtraParams = false;
		}

		Rigid3d camFromWorld = Rigid3d.Identity;
		if (!GeneralizedPoseEstimation.EstimateStructureLessAbsolutePose(
			absPoseOptions,
			points2D,
			worldPoints2D,
			worldCameraIdxs,
			worldCamsFromWorld,
			worldCameras,
			camera,
			ref camFromWorld,
			out int numInliers,
			out bool[] inlierMask))
		{
			return false;
		}

		if (numInliers < minNumInliers)
		{
			return false;
		}

		// Continue or triangulate tracks.
		image.FramePtr.SetCamFromWorld(image.CameraId, camFromWorld);

		obsManager.RegisterFrame(image.FrameId);
		RegisterFrameEvent(image.FrameId);

		Check.Eq(point2DIdxs.Count, corrs.Count);
		Check.Eq(point2DIdxs.Count, inlierMask.Length);
		var inlierCorrs = new List<CorrespondenceGraph.Correspondence>[numPoints2D];
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (inlierMask[i])
			{
				(inlierCorrs[point2DIdxs[i]] ??= []).Add(corrs[i]);
			}
		}

		var absPoseRefinementConfig = new BundleAdjustmentConfig();
		absPoseRefinementConfig.AddImage(imageId);

		IncrementalTriangulator triangulator = Check.NotNull(_triangulator);
		var triPoints = new List<Vector2d>();
		var triCamsFromWorld = new List<Rigid3d>();
		var triCameras = new List<Camera>();
		for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
		{
			List<CorrespondenceGraph.Correspondence>? pointInlierCorrs = inlierCorrs[point2DIdx];
			if (pointInlierCorrs is null)
			{
				continue;
			}

			// Check if any of the corresponding inlier points is already triangulated. Simply
			// add the current 2D point to the first track we find.
			bool continuedTrack = false;
			foreach (CorrespondenceGraph.Correspondence corr in pointInlierCorrs)
			{
				Point2D corrPoint2D = reconstruction.Image(corr.ImageId).Points2D[(int)corr.Point2DIdx];
				if (corrPoint2D.HasPoint3D)
				{
					obsManager.AddObservation(corrPoint2D.Point3DId, new TrackElement(imageId, point2DIdx));
					triangulator.AddModifiedPoint3D(corrPoint2D.Point3DId);
					continuedTrack = true;
					break;
				}
			}

			if (continuedTrack)
			{
				continue;
			}

			// Otherwise, robustly triangulate a new point.
			triPoints.Clear();
			triCamsFromWorld.Clear();
			triCameras.Clear();
			foreach (CorrespondenceGraph.Correspondence corr in pointInlierCorrs)
			{
				Image corrImage = reconstruction.Image(corr.ImageId);
				triPoints.Add(corrImage.Points2D[(int)corr.Point2DIdx].Xy);
				triCamsFromWorld.Add(corrImage.CamFromWorld());
				triCameras.Add(reconstruction.Camera(corrImage.CameraId));
			}

			triPoints.Add(image.Points2D[(int)point2DIdx].Xy);
			triCamsFromWorld.Add(image.CamFromWorld());
			triCameras.Add(camera);

			var triOptions = new EstimateTriangulationOptions { MinTriAngle = options.FilterMinTriAngle };
			triOptions.RansacOptions.MaxError = options.AbsPoseMaxError;
			if (!TriangulationEstimation.EstimateTriangulation(
					triOptions, triPoints, triCamsFromWorld, triCameras, out bool[] triInlierMask, out Vector3d triXyz)
				|| !triInlierMask[^1])
			{
				// Skip this 2D point, if we failed to triangulate and if it is itself not in
				// the inlier set.
				continue;
			}

			var track = new Track();
			track.AddElement(imageId, point2DIdx);
			for (int i = 0; i < triInlierMask.Length - 1; ++i)
			{
				if (triInlierMask[i])
				{
					CorrespondenceGraph.Correspondence inlierCorr = pointInlierCorrs[i];
					track.AddElement(inlierCorr.ImageId, inlierCorr.Point2DIdx);
				}
			}

			ulong point3DId = obsManager.AddPoint3D(triXyz, track);
			triangulator.AddModifiedPoint3D(point3DId);
			absPoseRefinementConfig.AddVariablePoint(point3DId);
		}

		// Refine pose using triangulated 3D point structure.
		BundleAdjuster absPoseRefinement = BundleAdjusters.CreateDefaultBundleAdjuster(
			absPoseRefinementOptions, absPoseRefinementConfig, reconstruction);
		return absPoseRefinement.Solve().IsSolutionUsable();
	}
}
