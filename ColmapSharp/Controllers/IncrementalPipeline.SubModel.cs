// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipeline.SubModel: the mapping of one model from colmap/controllers/
// incremental_pipeline.cc - InitializeReconstruction (seed from the initial pair),
// ReconstructSubModel (the next-image loop with local refinement, periodic global
// refinement, color extraction and snapshots) and the file-local helpers they use
// (IterativeGlobalRefinement, ExtractColors, WriteSnapshot, HasUnknownSensorFromRig). The
// controller shell and the model trial loop are in IncrementalPipeline.cs.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

public sealed partial class IncrementalPipeline
{
	/// <summary>
	/// Port of IncrementalPipeline::InitializeReconstruction: finds (or takes the configured)
	/// initial image pair, registers and triangulates it, and runs global bundle adjustment.
	/// </summary>
	public Status InitializeReconstruction(
		IncrementalMapper mapper, IncrementalMapper.Options mapperOptions, Reconstruction reconstruction)
	{
		// static_cast<image_t>(-1) is kInvalidImageId.
		uint imageId1 = unchecked((uint)_options.InitImageId1);
		uint imageId2 = unchecked((uint)_options.InitImageId2);

		// Try to find good initial pair.
		Rigid3d cam2FromCam1 = Rigid3d.Identity;
		if (!_options.IsInitialPairProvided)
		{
			bool findInitSuccess = mapper.FindInitialImagePair(mapperOptions, ref imageId1, ref imageId2, ref cam2FromCam1);
			if (CheckIfStopped() || CheckReachedMaxRuntime())
			{
				return Status.Interrupted;
			}

			if (!findInitSuccess)
			{
				return Status.NoInitialPair;
			}
		}
		else
		{
			if (!reconstruction.ExistsImage(imageId1) || !reconstruction.ExistsImage(imageId2))
			{
				return Status.NoInitialPair;
			}

			bool providedInitSuccess =
				mapper.EstimateInitialTwoViewGeometry(mapperOptions, imageId1, imageId2, ref cam2FromCam1);
			if (!providedInitSuccess)
			{
				return Status.BadInitialPair;
			}
		}

		mapper.RegisterInitialImagePair(mapperOptions, imageId1, imageId2, cam2FromCam1);

		IncrementalTriangulator.Options triOptions = _options.Triangulation();
		triOptions.MinAngle = mapperOptions.InitMinTriAngle;
		foreach (uint imageId in new[] { imageId1, imageId2 })
		{
			Image image = reconstruction.Image(imageId);
			foreach (DataId dataId in image.FramePtr.ImageIds())
			{
				mapper.TriangulateImage(triOptions, (uint)dataId.Id);
			}
		}

		if (reconstruction.NumPoints3D == 0)
		{
			return Status.BadInitialPair;
		}

		BundleAdjustmentOptions baOptions = _options.GlobalBundleAdjustment();
		baOptions.CheckIfStopped = CheckIfStopped;
		mapper.AdjustGlobalBundle(mapperOptions, baOptions);
		if (CheckIfStopped() || CheckReachedMaxRuntime())
		{
			return Status.Interrupted;
		}

		reconstruction.Normalize();
		mapper.FilterPoints(mapperOptions);
		mapper.FilterFrames(mapperOptions);

		// Initial image pair failed to register.
		if (reconstruction.NumRegFrames == 0 || reconstruction.NumPoints3D == 0)
		{
			return Status.BadInitialPair;
		}

		// Number of triangulated points not enough for registering future images.
		if (reconstruction.NumPoints3D < mapperOptions.AbsPoseMinNumInliers)
		{
			return Status.BadInitialPair;
		}

		if (_options.ExtractColors)
		{
			foreach (uint imageId in new[] { imageId1, imageId2 })
			{
				Image image = reconstruction.Image(imageId);
				foreach (DataId dataId in image.FramePtr.ImageIds())
				{
					ExtractColors(_options.ReadImage, (uint)dataId.Id, reconstruction);
				}
			}
		}

		return Status.Success;
	}

	/// <summary>
	/// Port of IncrementalPipeline::ReconstructSubModel: grows one model from its initial
	/// pair until no further image can be registered.
	/// </summary>
	public Status ReconstructSubModel(
		IncrementalMapper mapper, IncrementalMapper.Options mapperOptions, Reconstruction reconstruction)
	{
		mapper.BeginReconstruction(reconstruction);

		if (HasUnknownSensorFromRig(reconstruction))
		{
			return Status.UnknownSensorFromRig;
		}

		// Register initial pair.
		if (reconstruction.NumRegFrames == 0)
		{
			Status initStatus = InitializeReconstruction(mapper, mapperOptions, reconstruction);
			if (initStatus != Status.Success)
			{
				return initStatus;
			}
		}

		Callback((int)CallbackType.InitialImagePairRegCallback);
		ReportProgress(IncrementalPipelineStage.InitialPairRegistered, mapper);

		// Incremental mapping.
		int snapshotPrevNumRegFrames = reconstruction.NumRegFrames;
		int baPrevNumRegFrames = reconstruction.NumRegFrames;
		int baPrevNumPoints = reconstruction.NumPoints3D;

		bool[] structureLessFlags = _options.StructureLessRegistrationOnly
			? [true]
			: _options.StructureLessRegistrationFallback ? [false, true] : [false];

		bool regNextSuccess = true;
		bool prevRegNextSuccess;
		do
		{
			if (CheckIfStopped() || CheckReachedMaxRuntime())
			{
				break;
			}

			prevRegNextSuccess = regNextSuccess;
			regNextSuccess = false;
			uint nextImageId = InvalidImageId;

			// Try to register next image. Always prefer structure-based registration first,
			// and if that fails, try (less reliable) structure-less registration.
			foreach (bool structureLess in structureLessFlags)
			{
				List<uint> nextImages = mapper.FindNextImages(mapperOptions, structureLess);

				for (int regTrial = 0; regTrial < nextImages.Count; ++regTrial)
				{
					nextImageId = nextImages[regTrial];

					regNextSuccess = structureLess
						? mapper.RegisterNextStructureLessImage(mapperOptions, nextImageId)
						: mapper.RegisterNextImage(mapperOptions, nextImageId);

					if (regNextSuccess)
					{
						break;
					}

					// If initial model fails to continue for some time, abort and try different
					// initial pair.
					const int MinNumInitialRegTrials = 30;
					if (regTrial >= MinNumInitialRegTrials && reconstruction.NumRegImages < _options.MinModelSize)
					{
						break;
					}
				}

				if (regNextSuccess)
				{
					break;
				}
			}

			if (regNextSuccess)
			{
				Image image = reconstruction.Image(nextImageId);
				foreach (DataId dataId in image.FramePtr.ImageIds())
				{
					mapper.TriangulateImage(_options.Triangulation(), (uint)dataId.Id);
				}

				BundleAdjustmentOptions baOptions = _options.LocalBundleAdjustment();
				baOptions.CheckIfStopped = CheckIfStopped;
				mapper.IterativeLocalRefinement(
					_options.BaLocalMaxRefinements,
					_options.BaLocalMaxRefinementChange,
					mapperOptions,
					baOptions,
					_options.Triangulation(),
					nextImageId);

				if (CheckIfStopped() || CheckReachedMaxRuntime())
				{
					break;
				}

				if (CheckRunGlobalRefinement(reconstruction, baPrevNumRegFrames, baPrevNumPoints))
				{
					ReportProgress(IncrementalPipelineStage.GlobalRefinement, mapper);
					IterativeGlobalRefinement(_options, mapperOptions, mapper, CheckIfStopped);
					if (CheckIfStopped() || CheckReachedMaxRuntime())
					{
						break;
					}

					baPrevNumPoints = reconstruction.NumPoints3D;
					baPrevNumRegFrames = reconstruction.NumRegFrames;
				}

				if (_options.ExtractColors)
				{
					foreach (DataId dataId in image.FramePtr.ImageIds())
					{
						ExtractColors(_options.ReadImage, (uint)dataId.Id, reconstruction);
					}
				}

				if (_options.SnapshotFramesFreq > 0
					&& reconstruction.NumRegFrames >= _options.SnapshotFramesFreq + snapshotPrevNumRegFrames)
				{
					snapshotPrevNumRegFrames = reconstruction.NumRegFrames;
					WriteSnapshot(reconstruction, _options.SnapshotPath);
				}

				Callback((int)CallbackType.NextImageRegCallback);
				ReportProgress(IncrementalPipelineStage.ImageRegistered, mapper);
			}

			if (mapper.NumSharedRegImages >= _options.MaxModelOverlap)
			{
				break;
			}

			// If no image could be registered, try a single final global iterative bundle
			// adjustment and try again to register one image. If this fails once, then exit
			// the incremental mapping.
			if (!regNextSuccess && prevRegNextSuccess)
			{
				ReportProgress(IncrementalPipelineStage.GlobalRefinement, mapper);
				IterativeGlobalRefinement(_options, mapperOptions, mapper, CheckIfStopped);
			}
		}
		while (regNextSuccess || prevRegNextSuccess);

		if (CheckIfStopped() || CheckReachedMaxRuntime())
		{
			return Status.Interrupted;
		}

		// Only run final global BA, if last incremental BA was not global.
		if (reconstruction.NumRegFrames > 0
			&& reconstruction.NumRegFrames != baPrevNumRegFrames
			&& reconstruction.NumPoints3D != baPrevNumPoints)
		{
			ReportProgress(IncrementalPipelineStage.GlobalRefinement, mapper);
			IterativeGlobalRefinement(_options, mapperOptions, mapper, CheckIfStopped);
		}

		return Status.Success;
	}

	private static void IterativeGlobalRefinement(
		IncrementalPipelineOptions options,
		IncrementalMapper.Options mapperOptions,
		IncrementalMapper mapper,
		Func<bool> checkIfStopped)
	{
		BundleAdjustmentOptions baOptions = options.GlobalBundleAdjustment();
		baOptions.CheckIfStopped = checkIfStopped;
		mapper.IterativeGlobalRefinement(
			options.BaGlobalMaxRefinements,
			options.BaGlobalMaxRefinementChange,
			mapperOptions,
			baOptions,
			options.Triangulation());
		if (baOptions.CheckIfStopped is null || !baOptions.CheckIfStopped())
		{
			mapper.FilterFrames(mapperOptions);
		}
	}

	// COLMAP reads image_path / name; a failed read logs a warning and leaves the points black.
	private static void ExtractColors(Func<string, Bitmap?>? readImage, uint imageId, Reconstruction reconstruction)
	{
		string name = reconstruction.Image(imageId).Name;
		Bitmap? bitmap = readImage?.Invoke(name);
		if (bitmap is null)
		{
			Log.Warning($"Could not read image {name}.");
			return;
		}

		reconstruction.ExtractColorsForImage(imageId, bitmap);
	}

	private static void WriteSnapshot(Reconstruction reconstruction, string snapshotPath)
	{
		// Write reconstruction to unique path with the current timestamp in milliseconds.
		long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
		string path = Path.Combine(snapshotPath, timestamp.ToString("D10", System.Globalization.CultureInfo.InvariantCulture));
		Directory.CreateDirectory(path);
		reconstruction.Write(path);
	}

	private static bool HasUnknownSensorFromRig(Reconstruction reconstruction)
	{
		var parameterizedRigs = new HashSet<Rig>(ReferenceEqualityComparer.Instance);
		foreach (Image image in reconstruction.Images.Values)
		{
			parameterizedRigs.Add(image.FramePtr.RigPtr);
		}

		foreach (Rig rig in parameterizedRigs)
		{
			foreach (KeyValuePair<SensorId, Rigid3d?> sensor in rig.NonRefSensors)
			{
				if (sensor.Key.Type == SensorType.Camera && !sensor.Value.HasValue)
				{
					return true;
				}
			}
		}

		return false;
	}
}
