// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapper.Adjust: the refinement half of colmap/sfm/incremental_mapper.cc:
// triangulation and track completion/merging (forwarded to the IncrementalTriangulator),
// local and global bundle adjustment, the iterative refinement loops, and point / frame
// filtering. The rest of the mapper is in IncrementalMapper.cs and
// IncrementalMapper.Register.cs.
//
// Translation notes:
// - Modified-point sets. AdjustLocalBundle hands the triangulator its own copy of the
//   variable points (C++'s local variable_point3D_ids), because the triangulator's Merge
//   rewrites its modified-point set while it runs. The final FilterPoints3D reads the
//   caller's point3DIds after the merge and completion: when that is the live set from
//   GetModifiedPoints3D (IterativeLocalRefinement), it has grown as in C++, where it is a
//   const reference to the same set.
// - The variable points are collected in the order of point3DIds (a HashSet, deterministic
//   for the same sequence of operations; docs/CPP_DIVERGENCES.md, entry 59).
// - AdjustGlobalBundle applies its stricter convergence criteria for small reconstructions
//   only to the pose-prior adjuster, exactly as COLMAP does (the default adjuster gets the
//   caller's options unchanged).
// - check_if_stopped is Func<bool>?; a null function never stops.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm;

public sealed partial class IncrementalMapper
{
	/// <summary>Triangulates observations of an image. Returns the number of added observations.</summary>
	public int TriangulateImage(IncrementalTriangulator.Options triOptions, uint imageId)
	{
		Check.NotNull(_reconstruction);
		return Check.NotNull(_triangulator).TriangulateImage(triOptions, imageId);
	}

	/// <summary>
	/// Retriangulates image pairs that should have common observations according to the
	/// scene graph but don't due to drift, etc. To handle drift, the employed reprojection
	/// error thresholds should be relatively large. If the thresholds are too large,
	/// non-robust bundle adjustment will break down; if the thresholds are too small, we
	/// cannot fix drift effectively.
	/// </summary>
	public int Retriangulate(IncrementalTriangulator.Options triOptions)
	{
		Check.NotNull(_reconstruction);
		return Check.NotNull(_triangulator).Retriangulate(triOptions);
	}

	/// <summary>
	/// Completes tracks by transitively following the scene graph correspondences. This is
	/// especially effective after bundle adjustment, since many cameras and point locations
	/// might have improved. Completion of tracks enables better subsequent registration of
	/// new images.
	/// </summary>
	public int CompleteTracks(IncrementalTriangulator.Options triOptions)
	{
		Check.NotNull(_reconstruction);
		return Check.NotNull(_triangulator).CompleteAllTracks(triOptions);
	}

	/// <summary>
	/// Merges tracks by using scene graph correspondences. Similar to CompleteTracks, this is
	/// effective after bundle adjustment and improves the redundancy in subsequent bundle
	/// adjustments.
	/// </summary>
	public int MergeTracks(IncrementalTriangulator.Options triOptions)
	{
		Check.NotNull(_reconstruction);
		return Check.NotNull(_triangulator).MergeAllTracks(triOptions);
	}

	/// <summary>Globally completes and merges tracks. Returns the number of changed observations.</summary>
	public int CompleteAndMergeTracks(IncrementalTriangulator.Options triOptions)
	{
		int numCompletedObservations = CompleteTracks(triOptions);
		int numMergedObservations = MergeTracks(triOptions);
		return numCompletedObservations + numMergedObservations;
	}

	/// <summary>
	/// Adjusts the locally connected images and points of a reference image. In addition,
	/// refines the provided 3D points. Only images connected to the reference image are
	/// optimized. If the provided 3D points are not locally connected to the reference image,
	/// their observing images are set as constant in the adjustment.
	/// </summary>
	public LocalBundleAdjustmentReport AdjustLocalBundle(
		Options options,
		BundleAdjustmentOptions baOptions,
		IncrementalTriangulator.Options triOptions,
		uint imageId,
		IReadOnlySet<ulong> point3DIds)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		IncrementalTriangulator triangulator = Check.NotNull(_triangulator);
		Check.That(options.Check());

		var report = new LocalBundleAdjustmentReport();

		// Find images that have most 3D points with given image in common.
		List<uint> localBundle = FindLocalBundle(options, imageId);

		// Do the bundle adjustment only if there is any connected images.
		var baConfig = new BundleAdjustmentConfig();
		var imageIds = new HashSet<uint>();
		if (localBundle.Count > 0)
		{
			baConfig.FixGauge(BundleAdjustmentGauge.ThreePoints);

			// Insert the images of all local frames.
			Image image = reconstruction.Image(imageId);
			var frameIds = new HashSet<uint> { image.FrameId };
			foreach (DataId dataId in image.FramePtr.ImageIds())
			{
				baConfig.AddImage((uint)dataId.Id);
			}

			foreach (uint localImageId in localBundle)
			{
				Image localImage = reconstruction.Image(localImageId);
				frameIds.Add(localImage.FrameId);
				foreach (DataId dataId in localImage.FramePtr.ImageIds())
				{
					baConfig.AddImage((uint)dataId.Id);
				}
			}

			// Fix the existing images, if option specified.
			if (options.FixExistingFrames)
			{
				foreach (uint frameId in frameIds)
				{
					if (_existingFrameIds.Contains(frameId))
					{
						baConfig.SetConstantRigFromWorldPose(frameId);
					}
				}
			}

			// Fix rig poses, if not all frames within the local bundle.
			var numFramesPerRig = new Dictionary<uint, int>(frameIds.Count);
			foreach (uint frameId in frameIds)
			{
				Increment(numFramesPerRig, reconstruction.Frame(frameId).RigId);
			}

			foreach ((uint rigId, int numFrames) in numFramesPerRig)
			{
				if (options.ConstantRigs.Contains(rigId) || numFrames < AtOrThrow(_regStats.NumRegFramesPerRig, rigId))
				{
					Rig rig = reconstruction.Rig(rigId);
					foreach (SensorId sensorId in rig.NonRefSensors.Keys)
					{
						baConfig.SetConstantSensorFromRigPose(sensorId);
					}
				}
			}

			// Fix camera intrinsics, if not all registered images within local bundle.
			var numImagesPerCamera = new Dictionary<uint, int>(baConfig.NumImages);
			foreach (uint baImageId in baConfig.Images)
			{
				Increment(numImagesPerCamera, reconstruction.Image(baImageId).CameraId);
			}

			foreach ((uint cameraId, int numImages) in numImagesPerCamera)
			{
				if (options.ConstantCameras.Contains(cameraId)
					|| numImages < AtOrThrow(_regStats.NumRegImagesPerCamera, cameraId))
				{
					baConfig.SetConstantCamIntrinsics(cameraId);
				}
			}

			// Make sure, we refine all new and short-track 3D points, no matter if they are
			// fully contained in the local image set or not. Do not include long track 3D
			// points as they are usually already very stable and adding to them to bundle
			// adjustment and track merging/completion would slow down the local bundle
			// adjustment significantly.
			var variablePoint3DIds = new HashSet<ulong>();
			foreach (ulong point3DId in point3DIds)
			{
				Point3D point3D = reconstruction.Point3D(point3DId);
				const int kMaxTrackLength = 15;
				if (!point3D.HasError || point3D.Track.Length <= kMaxTrackLength)
				{
					baConfig.AddVariablePoint(point3DId);
					variablePoint3DIds.Add(point3DId);
				}
			}

			// Adjust the local bundle.
			imageIds = [.. baConfig.Images];

			BundleAdjuster bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(baOptions, baConfig, reconstruction);
			BundleAdjustmentSummary summary = bundleAdjuster.Solve();

			report.NumAdjustedObservations = summary.NumResiduals / 2;

			bool stopped = baOptions.CheckIfStopped is not null && baOptions.CheckIfStopped();
			if (!stopped)
			{
				// Merge refined tracks with other existing points.
				report.NumMergedObservations = triangulator.MergeTracks(triOptions, variablePoint3DIds);
				// Complete tracks that may have failed to triangulate before refinement of
				// camera pose and calibration in bundle-adjustment. This may avoid that some
				// points are filtered and it helps for subsequent image registrations.
				report.NumCompletedObservations = triangulator.CompleteTracks(triOptions, variablePoint3DIds);
				report.NumCompletedObservations += triangulator.CompleteImage(triOptions, imageId);
			}
		}

		// Filter both the modified images and all changed 3D points to make sure there are
		// no outlier points in the model. This results in duplicate work as many of the
		// provided 3D points may also be contained in the adjusted images, but the filtering
		// is not a bottleneck at this point.
		report.NumFilteredObservations = obsManager.FilterPoints3DInImages(
			options.FilterMaxReprojError, options.FilterMinTriAngle, imageIds);
		report.NumFilteredObservations += obsManager.FilterPoints3D(
			options.FilterMaxReprojError, options.FilterMinTriAngle, point3DIds);

		return report;
	}

	/// <summary>Global bundle adjustment. Returns whether the solution is usable.</summary>
	public bool AdjustGlobalBundle(Options options, BundleAdjustmentOptions baOptions)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);

		BundleAdjustmentOptions customBaOptions = baOptions.Clone();
		// Use stricter convergence criteria for first registered images.
		const int kMinNumRegFramesForFastBA = 10;
		bool isSmallReconstruction = reconstruction.NumRegFrames < kMinNumRegFramesForFastBA;
		if (isSmallReconstruction && customBaOptions.Ceres is not null)
		{
			Solver.SolverOptions solverOptions = customBaOptions.Ceres.SolverOptions;
			solverOptions.FunctionTolerance /= 10;
			solverOptions.GradientTolerance /= 10;
			solverOptions.ParameterTolerance /= 10;
			solverOptions.MaxNumIterations *= 2;
			solverOptions.MaxLinearSolverIterations = 200;
		}

		// Avoid degeneracies in bundle adjustment.
		obsManager.FilterObservationsWithNegativeDepth();

		// Configure bundle adjustment.
		var baConfig = new BundleAdjustmentConfig();
		foreach (uint frameId in reconstruction.RegFrameIds)
		{
			foreach (DataId dataId in reconstruction.Frame(frameId).ImageIds())
			{
				baConfig.AddImage((uint)dataId.Id);
			}
		}

		// After filtering, the reconstruction may have fewer than 2 images, in which case
		// global bundle adjustment is not possible.
		if (baConfig.NumImages < 2)
		{
			return false;
		}

		// Fix the existing images, if option specified.
		if (options.FixExistingFrames)
		{
			foreach (uint frameId in reconstruction.RegFrameIds)
			{
				if (_existingFrameIds.Contains(frameId))
				{
					baConfig.SetConstantRigFromWorldPose(frameId);
				}
			}
		}

		foreach (uint rigId in options.ConstantRigs)
		{
			foreach (SensorId sensorId in reconstruction.Rig(rigId).NonRefSensors.Keys)
			{
				baConfig.SetConstantSensorFromRigPose(sensorId);
			}
		}

		foreach (uint cameraId in options.ConstantCameras)
		{
			baConfig.SetConstantCamIntrinsics(cameraId);
		}

		List<ulong> redundantPoint3DIds = [];
		if (!isSmallReconstruction && options.BaGlobalIgnoreRedundantPoints3D)
		{
			redundantPoint3DIds = ReconstructionPruning.FindRedundantPoints3D(
				options.BaGlobalIgnoreRedundantPoints3DMinCoverageGain, reconstruction);
			foreach (ulong point3DId in redundantPoint3DIds)
			{
				baConfig.IgnorePoint(point3DId);
			}
		}

		bool usePriorPosition =
			options.UsePriorPosition && NumRegisteredPosePriors(_databaseCache.PosePriors, baConfig) >= 3;

		BundleAdjuster bundleAdjuster;
		if (!usePriorPosition)
		{
			// Fixing the gauge with two cameras leads to a more stable optimization with fewer
			// steps as compared to fixing three points.
			// TODO(jsch): Investigate whether it is safe to not fix the gauge at all, as
			// initial experiments show that it is even faster.
			baConfig.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
			bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(baOptions, baConfig, reconstruction);
		}
		else
		{
			var priorOptions = new PosePriorBundleAdjustmentOptions();
			CeresPosePriorBundleAdjustmentOptions priorCeres = Check.NotNull(priorOptions.Ceres);
			if (options.UseRobustLossOnPriorPosition)
			{
				priorCeres.PriorPositionLossFunctionType = BundleAdjustmentLossFunctionType.Cauchy;
			}

			priorCeres.PriorPositionLossScale = options.PriorPositionLossScale;
			// RansacOptions is a struct held by a property: copy, set, store back.
			Optim.RansacOptions alignmentRansacOptions = priorOptions.AlignmentRansacOptions;
			alignmentRansacOptions.RandomSeed = options.RandomSeed;
			priorOptions.AlignmentRansacOptions = alignmentRansacOptions;
			bundleAdjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
				customBaOptions, priorOptions, baConfig, _databaseCache.PosePriors, reconstruction);
		}

		// Optimize the redundant 3D points with all other parameters fixed.
		if (!isSmallReconstruction && options.BaGlobalIgnoreRedundantPoints3D)
		{
			if (!bundleAdjuster.Solve().IsSolutionUsable())
			{
				return false;
			}

			if (baOptions.CheckIfStopped is not null && baOptions.CheckIfStopped())
			{
				return true;
			}

			baConfig = new BundleAdjustmentConfig();
			foreach (ulong point3DId in redundantPoint3DIds)
			{
				baConfig.AddVariablePoint(point3DId);
			}

			foreach (uint frameId in reconstruction.RegFrameIds)
			{
				baConfig.SetConstantRigFromWorldPose(frameId);
			}

			foreach (uint cameraId in reconstruction.Cameras.Keys)
			{
				baConfig.SetConstantCamIntrinsics(cameraId);
			}

			foreach (Rig rig in reconstruction.Rigs.Values)
			{
				foreach (SensorId sensorId in rig.NonRefSensors.Keys)
				{
					baConfig.SetConstantSensorFromRigPose(sensorId);
				}
			}

			bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(baOptions, baConfig, reconstruction);
		}

		return bundleAdjuster.Solve().IsSolutionUsable();
	}

	/// <summary>Performs multiple rounds of local bundle adjustment.</summary>
	public void IterativeLocalRefinement(
		int maxNumRefinements,
		double maxRefinementChange,
		Options options,
		BundleAdjustmentOptions baOptions,
		IncrementalTriangulator.Options triOptions,
		uint imageId)
	{
		BundleAdjustmentOptions customBaOptions = baOptions.Clone();
		for (int i = 0; i < maxNumRefinements; ++i)
		{
			if (customBaOptions.CheckIfStopped is not null && customBaOptions.CheckIfStopped())
			{
				break;
			}

			LocalBundleAdjustmentReport report =
				AdjustLocalBundle(options, customBaOptions, triOptions, imageId, GetModifiedPoints3D());
			double changed = report.NumAdjustedObservations == 0
				? 0
				: (report.NumMergedObservations + report.NumCompletedObservations + report.NumFilteredObservations)
					/ (double)report.NumAdjustedObservations;
			if (changed < maxRefinementChange)
			{
				break;
			}

			// Only use robust cost function for first iteration.
			if (customBaOptions.Ceres is not null)
			{
				customBaOptions.Ceres.LossFunctionType = BundleAdjustmentLossFunctionType.Trivial;
			}
		}

		ClearModifiedPoints3D();
	}

	/// <summary>Performs multiple rounds of global bundle adjustment.</summary>
	public void IterativeGlobalRefinement(
		int maxNumRefinements,
		double maxRefinementChange,
		Options options,
		BundleAdjustmentOptions baOptions,
		IncrementalTriangulator.Options triOptions,
		bool normalizeReconstruction = true)
	{
		if (baOptions.CheckIfStopped is not null && baOptions.CheckIfStopped())
		{
			return;
		}

		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		CompleteAndMergeTracks(triOptions);
		Retriangulate(triOptions);
		for (int i = 0; i < maxNumRefinements; ++i)
		{
			if (baOptions.CheckIfStopped is not null && baOptions.CheckIfStopped())
			{
				break;
			}

			long numObservations = reconstruction.ComputeNumObservations();
			AdjustGlobalBundle(options, baOptions);
			if (baOptions.CheckIfStopped is not null && baOptions.CheckIfStopped())
			{
				break;
			}

			if (normalizeReconstruction && !options.UsePriorPosition)
			{
				// Normalize scene for numerical stability and to avoid large scale changes in
				// the viewer.
				reconstruction.Normalize();
			}

			int numChangedObservations = CompleteAndMergeTracks(triOptions);
			numChangedObservations += FilterPoints(options);
			double changed = numObservations == 0 ? 0 : (double)numChangedObservations / numObservations;
			if (changed < maxRefinementChange)
			{
				break;
			}
		}

		ClearModifiedPoints3D();
	}

	/// <summary>
	/// Filters registered frames with too few observations or bogus camera parameters.
	/// Does nothing before 20 frames are registered. Returns the number of filtered frames.
	/// </summary>
	public int FilterFrames(Options options)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		Check.That(options.Check());

		// Do not filter frames in the early stage of the reconstruction, since the
		// calibration is often still refining a lot. Hence, the camera parameters are not
		// stable in the beginning.
		const int kMinNumFrames = 20;
		if (reconstruction.NumRegFrames < kMinNumFrames)
		{
			return 0;
		}

		List<uint> filterFrameIds = obsManager.FindFramesToFilter(
			options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam, minNumObservations: 1);

		int numFiltered = 0;
		foreach (uint frameId in filterFrameIds)
		{
			if (!options.FixExistingFrames || !_existingFrameIds.Contains(frameId))
			{
				obsManager.DeRegisterFrame(frameId);
				DeRegisterFrameEvent(frameId);
				_filteredFrames.Add(frameId);
				++numFiltered;
			}
		}

		return numFiltered;
	}

	/// <summary>Filters point observations. Returns the number of filtered observations.</summary>
	public int FilterPoints(Options options)
	{
		ObservationManager obsManager = Check.NotNull(_obsManager);
		Check.That(options.Check());
		return obsManager.FilterAllPoints3D(options.FilterMaxReprojError, options.FilterMinTriAngle);
	}

	// The number of position priors of camera images that take part in the adjustment.
	private static int NumRegisteredPosePriors(IEnumerable<PosePrior> posePriors, BundleAdjustmentConfig baConfig)
	{
		int numRegisteredPosePriors = 0;
		foreach (PosePrior posePrior in posePriors)
		{
			if (posePrior.HasPosition()
				&& posePrior.CorrDataId.SensorId.Type == SensorType.Camera
				&& baConfig.HasImage((uint)posePrior.CorrDataId.Id))
			{
				++numRegisteredPosePriors;
			}
		}

		return numRegisteredPosePriors;
	}

	// C++'s map .at(): the count, throwing for a missing key.
	private static int AtOrThrow(Dictionary<uint, int> counts, uint key)
	{
		Check.That(counts.TryGetValue(key, out int count), $"Key {key} not found");
		return count;
	}
}
