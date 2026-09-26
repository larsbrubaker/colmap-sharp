// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalMapper.Refine: the second half of colmap/sfm/global_mapper.cc - the anonymous
// RunBundleAdjustment helper, IterativeBundleAdjustment, IterativeRetriangulateAndRefine and
// Solve, which runs every stage of GlobalMapper.cs in order. See GlobalMapper.cs for the
// overview and tests.
//
// Translation notes:
// - The on_progress callback (std::function<bool()>) is a Func<bool>?: it is called after
//   global positioning, after each bundle adjustment iteration and after retriangulation,
//   and returning true stops the pipeline early with the current result. A controller wires
//   it to its CancellationToken; to stop inside a long bundle adjustment as well, set
//   BundleAdjustmentOptions.CheckIfStopped on the options.
// - onStageStarted (C#-only) replaces the LOG_HEADING1 stage headings, so a host can show
//   which stage is running.

using ColmapSharp.Estimators;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm;

/// <summary>The stages of <see cref="GlobalMapper.Solve"/>, in the order they run (C#-only).</summary>
public enum GlobalMapperStage
{
	/// <summary>Rotation averaging.</summary>
	RotationAveraging,

	/// <summary>Track establishment and selection.</summary>
	TrackEstablishment,

	/// <summary>Global positioning.</summary>
	GlobalPositioning,

	/// <summary>Iterative bundle adjustment.</summary>
	BundleAdjustment,

	/// <summary>Iterative retriangulation and refinement.</summary>
	Retriangulation,
}

public sealed partial class GlobalMapper
{
	// The anonymous-namespace RunBundleAdjustment of global_mapper.cc: bundle adjusts all
	// posed images with the gauge fixed by two cameras.
	private static bool RunBundleAdjustment(BundleAdjustmentOptions options, Reconstruction reconstruction)
	{
		if (reconstruction.NumImages == 0)
		{
			Log.Error("Cannot run bundle adjustment: no registered images");
			return false;
		}

		if (reconstruction.NumPoints3D == 0)
		{
			Log.Error("Cannot run bundle adjustment: no 3D points to optimize");
			return false;
		}

		var baConfig = new BundleAdjustmentConfig();
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			if (image.HasPose)
			{
				baConfig.AddImage(imageId);
			}
		}

		baConfig.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		BundleAdjuster ba = BundleAdjusters.CreateDefaultBundleAdjuster(options, baConfig, reconstruction);

		return ba.Solve().IsSolutionUsable();
	}

	/// <summary>
	/// Runs iterative bundle adjustment to refine poses and structure, tightening the outlier
	/// filter each round. <paramref name="onProgress"/> is invoked after each iteration and
	/// returns true if a stop has been requested, in which case the iteration ends early.
	/// </summary>
	public bool IterativeBundleAdjustment(
		BundleAdjustmentOptions options,
		double maxNormalizedReprojError,
		double minTriAngleDeg,
		int numIterations,
		bool skipFixedRotationStage = false,
		bool skipJointOptimizationStage = false,
		Func<bool>? onProgress = null)
	{
		Reconstruction recon = Check.NotNull(reconstruction);
		for (int ite = 0; ite < numIterations; ite++)
		{
			// Optional fixed-rotation stage: optimize positions only
			if (!skipFixedRotationStage)
			{
				BundleAdjustmentOptions optsPositionOnly = options.Clone();
				optsPositionOnly.ConstantRigFromWorldRotation = true;
				// Caspar's pose node is a single retracted Pose3 (rotation+translation
				// together) with no mechanism to hold rotation constant while translation is
				// free -- constant_rig_from_world_rotation is silently ignored when backend ==
				// CASPAR.
				if (optsPositionOnly.Backend == BundleAdjustmentBackend.Caspar)
				{
					optsPositionOnly.Backend = BundleAdjustmentBackend.Ceres;
				}

				if (!RunBundleAdjustment(optsPositionOnly, recon))
				{
					return false;
				}
			}

			// Joint optimization stage: default BA
			if (!skipJointOptimizationStage)
			{
				if (!RunBundleAdjustment(options, recon))
				{
					return false;
				}
			}

			// Normalize the structure for numerical stability.
			// TODO (COLMAP): Skip normalization when position priors are used (similar to
			// incremental mapper's !use_prior_position condition).
			recon.Normalize();

			// Report progress for this refinement iteration and stop early if requested. The
			// filter passes above leave point3D.error in normalized units, so recompute it in
			// pixels first to keep intermediate visualizations consistent with the final
			// reconstruction.
			if (onProgress is not null)
			{
				recon.UpdatePoint3DErrors();
				if (onProgress())
				{
					break;
				}
			}

			// Filter tracks based on the estimation. For the filtering, in each round, the
			// criteria for outlier is tightened. If only few tracks are changed, no need to
			// start bundle adjustment right away. Instead, use a more strict criteria to
			// filter.
			var obsManager = new ObservationManager(recon);
			bool status = true;
			long filteredNum = 0;
			while (status && ite < numIterations)
			{
				double scaling = Math.Max(3 - ite, 1);
				filteredNum += obsManager.FilterPoints3DWithLargeReprojectionError(
					scaling * maxNormalizedReprojError, recon.Point3DIds(), ReprojectionErrorType.Normalized);

				if (filteredNum > 1e-3 * recon.NumPoints3D)
				{
					status = false;
				}
				else
				{
					ite++;
				}
			}

			if (status)
			{
				// Fewer than 0.1% tracks are filtered, stop the iteration.
				break;
			}
		}

		// Filter tracks based on the estimation
		var finalObsManager = new ObservationManager(recon);
		finalObsManager.FilterPoints3DWithLargeReprojectionError(
			maxNormalizedReprojError, recon.Point3DIds(), ReprojectionErrorType.Normalized);
		finalObsManager.FilterPoints3DWithSmallTriangulationAngle(minTriAngleDeg, recon.Point3DIds());

		return true;
	}

	/// <summary>
	/// Deletes all 3D points, retriangulates every registered image with the incremental
	/// mapper's triangulator, runs its iterative global refinement, then filters and bundle
	/// adjusts once more.
	/// </summary>
	public bool IterativeRetriangulateAndRefine(
		IncrementalTriangulator.Options options,
		BundleAdjustmentOptions baOptions,
		double maxNormalizedReprojError,
		double minTriAngleDeg)
	{
		Reconstruction recon = Check.NotNull(reconstruction);

		// Delete all existing 3D points and re-establish 2D-3D correspondences.
		recon.DeleteAllPoints2DAndPoints3D();

		// Initialize mapper.
		var mapper = new IncrementalMapper(databaseCache);
		mapper.BeginReconstruction(recon);

		// Triangulate all registered images.
		foreach (uint imageId in recon.RegImageIds())
		{
			mapper.TriangulateImage(options, imageId);
		}

		// Set up bundle adjustment options for colmap's incremental mapper.
		BundleAdjustmentOptions customBaOptions = baOptions.Clone();
		customBaOptions.PrintSummary = false;
		if (customBaOptions.Ceres is not null && baOptions.Ceres is not null)
		{
			customBaOptions.Ceres.SolverOptions.NumThreads = baOptions.Ceres.SolverOptions.NumThreads;
			customBaOptions.Ceres.SolverOptions.MaxNumIterations = 50;
			customBaOptions.Ceres.SolverOptions.MaxLinearSolverIterations = 100;
		}

		// Iterative global refinement.
		var mapperOptions = new IncrementalMapper.Options { RandomSeed = options.RandomSeed };
		mapper.IterativeGlobalRefinement(
			maxNumRefinements: 5,
			maxRefinementChange: 0.0005,
			mapperOptions,
			customBaOptions,
			options,
			normalizeReconstruction: true);

		mapper.EndReconstruction(discard: false);

		// Final filtering and bundle adjustment.
		var obsManager = new ObservationManager(recon);
		obsManager.FilterPoints3DWithLargeReprojectionError(
			maxNormalizedReprojError, recon.Point3DIds(), ReprojectionErrorType.Normalized);

		if (!RunBundleAdjustment(baOptions, recon))
		{
			return false;
		}

		// Normalize the structure for numerical stability.
		// TODO (COLMAP): Skip normalization when position priors are used (similar to
		// incremental mapper's !use_prior_position condition).
		recon.Normalize();

		obsManager.FilterPoints3DWithLargeReprojectionError(
			maxNormalizedReprojError, recon.Point3DIds(), ReprojectionErrorType.Normalized);
		obsManager.FilterPoints3DWithSmallTriangulationAngle(minTriAngleDeg, recon.Point3DIds());

		return true;
	}

	/// <summary>
	/// Runs the global SfM pipeline: rotation averaging, track establishment, global
	/// positioning, iterative bundle adjustment and retriangulation, each unless skipped in
	/// <paramref name="options"/>. The optional <paramref name="onProgress"/> callback is
	/// invoked after global positioning, after each bundle-adjustment iteration, and after
	/// retriangulation/refinement; it returns true if a stop has been requested, in which case
	/// the pipeline terminates early and keeps the current result. The optional
	/// <paramref name="onStageStarted"/> (C#-only) is called as each stage starts, where COLMAP
	/// logs the stage heading.
	/// </summary>
	public bool Solve(
		GlobalMapperOptions options, Func<bool>? onProgress = null, Action<GlobalMapperStage>? onStageStarted = null)
	{
		Reconstruction recon = Check.NotNull(reconstruction);
		PoseGraph graph = Check.NotNull(poseGraph);

		if (graph.Empty)
		{
			Log.Error("Cannot continue with empty pose graph");
			return false;
		}

		// Reports the current reconstruction and returns whether a stop was requested. Point
		// errors are recomputed in pixels before reporting because the preceding filter passes
		// leave point3D.error in normalized units, which would otherwise make intermediate
		// visualizations inconsistent with the final reconstruction.
		bool ReportAndCheckStop()
		{
			if (onProgress is null)
			{
				return false;
			}

			recon.UpdatePoint3DErrors();
			return onProgress();
		}

		// Run rotation averaging
		if (!options.SkipRotationAveraging)
		{
			onStageStarted?.Invoke(GlobalMapperStage.RotationAveraging);
			if (!RotationAveraging(options.RotationAveraging()))
			{
				return false;
			}
		}

		// Track establishment and selection
		if (!options.SkipTrackEstablishment)
		{
			onStageStarted?.Invoke(GlobalMapperStage.TrackEstablishment);
			EstablishTracks(options);
		}

		// Global positioning
		if (!options.SkipGlobalPositioning)
		{
			onStageStarted?.Invoke(GlobalMapperStage.GlobalPositioning);
			if (!GlobalPositioning(
				options.GlobalPositioning(),
				options.MaxAngularReprojErrorDeg,
				options.MaxNormalizedReprojError,
				options.MinTriAngleDeg))
			{
				return false;
			}

			// Report the first 3D view after global positioning and stop early if requested.
			if (ReportAndCheckStop())
			{
				return true;
			}
		}

		// Bundle adjustment
		if (!options.SkipBundleAdjustment)
		{
			onStageStarted?.Invoke(GlobalMapperStage.BundleAdjustment);
			if (!IterativeBundleAdjustment(
				options.BundleAdjustment(),
				options.MaxNormalizedReprojError,
				options.MinTriAngleDeg,
				options.BaNumIterations,
				options.BaSkipFixedRotationStage,
				options.BaSkipJointOptimizationStage,
				onProgress))
			{
				return false;
			}
		}

		// Retriangulation
		if (!options.SkipRetriangulation)
		{
			onStageStarted?.Invoke(GlobalMapperStage.Retriangulation);
			if (!IterativeRetriangulateAndRefine(
				options.Retriangulation(),
				options.BundleAdjustment(),
				options.MaxNormalizedReprojError,
				options.MinTriAngleDeg))
			{
				return false;
			}

			// Report the result after retriangulation and stop early if requested.
			if (ReportAndCheckStop())
			{
				return true;
			}
		}

		// Filter passes here use NORMALIZED/ANGULAR error, so point3D.error is left in
		// non-pixel units. Recompute in pixels for consistent reporting in model_analyzer.
		recon.UpdatePoint3DErrors();

		return true;
	}
}
