// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPipeline: port of colmap/controllers/global_pipeline.h and .cc - the controller that
// runs global SfM (Sfm/GlobalMapper*.cs) over a database. With multiple_models it first
// splits the view graph into components with the same rotation averaging and relative
// rotation filtering the mapper uses, then maps each large enough component into its own
// reconstruction; otherwise it maps the database once. The kept reconstructions are sorted
// by registered frame count, aligned to the rigs' metric scale and colored. The incremental
// counterpart is IncrementalPipeline.cs. Tests:
// ColmapSharp.Tests/Controllers/GlobalPipelineTests*.cs (global_pipeline_test.cc 1:1).
//
// Tier C (outcome): reconstructions against the ground truth within COLMAP's bounds.
//
// Translation notes:
// - image_path becomes ReadImage (the IncrementalPipeline shape): the host decodes images by
//   name. Colors are extracted only when ReadImage is set, as COLMAP extracts them only for
//   a non-empty image_path.
// - Cancellation: BaseController.CancellationToken (and SetCheckIfStoppedFunc) is checked
//   where COLMAP checks CheckIfStopped (before each component and from the mapper's progress
//   callback), and inside every bundle adjustment through BundleAdjustmentOptions.CheckIfStopped.
// - Progress (C#-only): Progress, when set, receives a GlobalPipelineProgress as each mapper
//   stage starts, at each model update and when a component is finished.
// - The newly created reconstructions are sorted by registered frames with a stable sort
//   (docs/CPP_DIVERGENCES.md entry 102).
// - LOG(WARNING)/LOG(ERROR) go to Util/Log.cs; LOG(INFO) and the timers are dropped, and
//   with them ReconstructionStats (failed / too-small counts), which only fed the log.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::GlobalPipelineOptions.</summary>
public sealed class GlobalPipelineOptions
{
	/// <summary>The minimum number of matches for inlier matches to be considered.</summary>
	public int MinNumMatches { get; set; } = 15;

	/// <summary>Whether to ignore the inlier matches of watermark image pairs.</summary>
	public bool IgnoreWatermarks { get; set; }

	/// <summary>Names of images to reconstruct. If empty, all images are used.</summary>
	public List<string> ImageNames { get; set; } = [];

	/// <summary>
	/// Decodes an image by name to extract point colors (COLMAP's image_path). Null (the
	/// default) skips color extraction, like COLMAP's empty image_path.
	/// </summary>
	public Func<string, Bitmap?>? ReadImage { get; set; }

	/// <summary>Number of threads for parallel processing.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Random seed for reproducibility.</summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>Whether to decompose relative poses from two-view geometries.</summary>
	public bool DecomposeRelativePose { get; set; } = true;

	/// <summary>
	/// If true (default), reconstruct every connected component of the view graph (one model
	/// per component). If false, reconstruct only the largest connected component.
	/// </summary>
	public bool MultipleModels { get; set; } = true;

	/// <summary>
	/// Minimum number of registered frames for a reconstruction to be kept. Reconstructions
	/// with fewer registered frames are discarded.
	/// </summary>
	public int MinModelSize { get; set; } = 3;

	/// <summary>Options for the global mapper.</summary>
	public GlobalMapperOptions Mapper { get; set; } = new();
}

/// <summary>What a <see cref="GlobalPipelineProgress"/> report was sent for.</summary>
public enum GlobalPipelineStage
{
	/// <summary>Rotation averaging of a component started.</summary>
	RotationAveraging,

	/// <summary>Track establishment of a component started.</summary>
	TrackEstablishment,

	/// <summary>Global positioning of a component started.</summary>
	GlobalPositioning,

	/// <summary>Iterative bundle adjustment of a component started.</summary>
	BundleAdjustment,

	/// <summary>Retriangulation and refinement of a component started.</summary>
	Retriangulation,

	/// <summary>The in-progress reconstruction was updated (MODEL_UPDATE_CALLBACK).</summary>
	ModelUpdated,

	/// <summary>A component is finished (kept or discarded).</summary>
	ComponentFinished,
}

/// <summary>
/// Progress of a global reconstruction: component <paramref name="Component"/> (0-based) of
/// <paramref name="NumComponents"/> is at <paramref name="Stage"/>, and its reconstruction has
/// <paramref name="NumRegImages"/> registered images.
/// </summary>
public readonly record struct GlobalPipelineProgress(
	GlobalPipelineStage Stage, int Component, int NumComponents, int NumRegImages);

/// <summary>Port of colmap::GlobalPipeline: controls the global mapping procedure.</summary>
public sealed class GlobalPipeline : BaseController
{
	/// <summary>Port of GlobalPipeline::CallbackType.</summary>
	public enum CallbackType
	{
		/// <summary>
		/// Triggered after global positioning, after each global refinement iteration, and
		/// after retriangulation, so the in-progress reconstruction can be rendered.
		/// </summary>
		ModelUpdateCallback,
	}

	private const double MinPriorFocalLengthRatio = 0.5;

	private readonly GlobalPipelineOptions _options;
	private readonly DatabaseCache _databaseCache;
	private readonly ReconstructionManager _reconstructionManager;

	// The component being mapped, for progress reports.
	private int _currentComponent;
	private int _numComponents = 1;

	/// <summary>Loads the database into a cache (decomposing relative poses if the options say so).</summary>
	public GlobalPipeline(GlobalPipelineOptions options, Database database, ReconstructionManager reconstructionManager)
	{
		_options = Check.NotNull(options);
		_reconstructionManager = Check.NotNull(reconstructionManager);
		Check.NotNull(database);
		Check.Ge(_options.MinModelSize, 0);

		// Create database cache with relative poses for pose graph.
		var databaseCacheOptions = new DatabaseCache.Options
		{
			MinNumMatches = _options.MinNumMatches,
			IgnoreWatermarks = _options.IgnoreWatermarks,
			ImageNames = [.. _options.ImageNames],
		};
		_databaseCache = DatabaseCache.Create(database, databaseCacheOptions);
		if (_options.DecomposeRelativePose)
		{
			TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(_databaseCache);
		}

		RegisterCallback((int)CallbackType.ModelUpdateCallback);
	}

	/// <summary>The options (shared with the caller).</summary>
	public GlobalPipelineOptions Options => _options;

	/// <summary>Receives progress reports during <see cref="Run"/> (C#-only).</summary>
	public IProgress<GlobalPipelineProgress>? Progress { get; set; }

	/// <summary>Maps the database into one reconstruction per (large enough) component.</summary>
	public override void Run()
	{
		bool hasInsufficientPriorFocalLengths = HasInsufficientPriorFocalLengths(_databaseCache);
		if (hasInsufficientPriorFocalLengths)
		{
			WarnInsufficientPriorFocalLengths();
		}

		// Prepare mapper options with top-level options (a copy, as in C++, so the stop check
		// does not leak into the caller's options).
		GlobalMapperOptions mapperOptions = _options.Mapper.Clone();
		mapperOptions.NumThreads = _options.NumThreads;
		mapperOptions.RandomSeed = _options.RandomSeed;
		mapperOptions.BundleAdjustmentOptions.CheckIfStopped = CheckIfStopped;

		int firstReconstructionIdx = _reconstructionManager.Size;
		if (_options.MultipleModels)
		{
			ReconstructMultiComponents(mapperOptions);
		}
		else
		{
			_currentComponent = 0;
			_numComponents = 1;
			Reconstruction? reconstruction = ReconstructSingleComponent(_databaseCache, mapperOptions);
			if (reconstruction is null || reconstruction.NumRegFrames < _options.MinModelSize)
			{
				_reconstructionManager.Delete(_reconstructionManager.Size - 1);
			}

			ReportProgress(GlobalPipelineStage.ComponentFinished, reconstruction);
		}

		// Sort newly created reconstructions by registered frame count. Keep any
		// reconstructions that were already managed before this run untouched. Stable, so
		// equally large ones keep their component order (entry 102).
		var reconstructions = new List<Reconstruction>();
		for (int i = firstReconstructionIdx; i < _reconstructionManager.Size; ++i)
		{
			reconstructions.Add(_reconstructionManager.Get(i));
		}

		List<Reconstruction> sorted = [.. reconstructions.OrderByDescending(r => r.NumRegFrames)];
		for (int i = 0; i < sorted.Count; ++i)
		{
			_reconstructionManager.Set(firstReconstructionIdx + i, sorted[i]);
		}

		if (_options.ReadImage is not null)
		{
			for (int i = firstReconstructionIdx; i < _reconstructionManager.Size; ++i)
			{
				_reconstructionManager.Get(i).ExtractColorsForAllImages(_options.ReadImage, _options.NumThreads);
			}
		}

		if (hasInsufficientPriorFocalLengths)
		{
			// Intentionally logging this warning before and after the reconstruction to make
			// sure it is not missed.
			WarnInsufficientPriorFocalLengths();
		}
	}

	// Runs the full global SfM pipeline on the given database cache and returns the
	// resulting reconstruction, or null if mapping fails. The in-progress reconstruction is
	// added to the manager so callbacks can render it. The caller decides whether to keep it.
	private Reconstruction? ReconstructSingleComponent(DatabaseCache databaseCache, GlobalMapperOptions mapperOptions)
	{
		Reconstruction reconstruction = _reconstructionManager.Get(_reconstructionManager.Add());

		var globalMapper = new GlobalMapper(databaseCache);
		globalMapper.BeginReconstruction(reconstruction);

		bool success = globalMapper.Solve(
			mapperOptions,
			() =>
			{
				Callback((int)CallbackType.ModelUpdateCallback);
				ReportProgress(GlobalPipelineStage.ModelUpdated, reconstruction);
				return CheckIfStopped();
			},
			stage => ReportProgress((GlobalPipelineStage)stage, reconstruction));

		// A stop requested through the callback is reported as success, so false only
		// denotes a genuine mapping failure. The caller removes failed reconstructions from
		// the output manager.
		if (!success)
		{
			Log.Error("Global mapping failed");
			return null;
		}

		// Align reconstruction to the original metric scales in rig extrinsics.
		Alignment.AlignReconstructionToOrigRigScales(databaseCache.Rigs, reconstruction);

		return reconstruction;
	}

	// Partitions the input view graph once using rotation averaging and reconstructs each
	// resulting component at most once.
	private void ReconstructMultiComponents(GlobalMapperOptions mapperOptions)
	{
		// Build the base reconstruction, pose graph, and pose priors from the cache.
		var baseReconstruction = new Reconstruction();
		baseReconstruction.Load(_databaseCache);
		var poseGraph = new PoseGraph();
		poseGraph.Load(_databaseCache.CorrespondenceGraph);
		IReadOnlyList<PosePrior> posePriors = _databaseCache.PosePriors;
		if (poseGraph.Empty)
		{
			Log.Error("Cannot continue with empty pose graph");
			return;
		}

		// Decompose the view graph once after rotation filtering. The full mapper is then run
		// at most once per resulting component; any additional fragments rejected by its
		// refinement pass are not recursively retried.
		List<HashSet<uint>> components = ComputeComponentsByRotationAveraging(
			mapperOptions.RotationAveraging(), poseGraph, baseReconstruction, posePriors, _options.MinModelSize);
		_numComponents = components.Count;

		for (int componentIdx = 0; componentIdx < components.Count; ++componentIdx)
		{
			if (CheckIfStopped())
			{
				return;
			}

			_currentComponent = componentIdx;
			HashSet<uint> imageIds = components[componentIdx];

			if (NumFramesForImages(baseReconstruction, imageIds) < _options.MinModelSize)
			{
				ReportProgress(GlobalPipelineStage.ComponentFinished, null);
				continue;
			}

			var cacheOptions = new DatabaseCache.Options();
			foreach (uint imageId in imageIds)
			{
				cacheOptions.ImageNames.Add(baseReconstruction.Image(imageId).Name);
			}

			DatabaseCache componentCache = DatabaseCache.CreateFromCache(_databaseCache, cacheOptions);

			Reconstruction? reconstruction = ReconstructSingleComponent(componentCache, mapperOptions);
			if (reconstruction is null || reconstruction.NumRegFrames < _options.MinModelSize)
			{
				_reconstructionManager.Delete(_reconstructionManager.Size - 1);
			}

			ReportProgress(GlobalPipelineStage.ComponentFinished, reconstruction);
		}
	}

	private void ReportProgress(GlobalPipelineStage stage, Reconstruction? reconstruction) =>
		Progress?.Report(new GlobalPipelineProgress(
			stage, _currentComponent, _numComponents, reconstruction?.NumRegImages ?? 0));

	private static bool HasInsufficientPriorFocalLengths(DatabaseCache databaseCache)
	{
		IReadOnlyDictionary<uint, Camera> cameras = databaseCache.Cameras;
		if (cameras.Count == 0)
		{
			return false;
		}

		int numWithPrior = cameras.Values.Count(camera => camera.HasPriorFocalLength);
		return numWithPrior < MinPriorFocalLengthRatio * cameras.Count;
	}

	private static void WarnInsufficientPriorFocalLengths() =>
		Log.Warning(
			$"Less than {MinPriorFocalLengthRatio * 100}% of cameras have prior focal lengths. The global mapper "
			+ "depends on reasonably good focal length priors to perform well. Consider running "
			+ "'colmap view_graph_calibrator' before 'colmap global_mapper' or providing camera "
			+ "calibrations manually.");

	private static int NumFramesForImages(Reconstruction reconstruction, IReadOnlySet<uint> imageIds)
	{
		var frameIds = new HashSet<uint>();
		foreach (uint imageId in imageIds)
		{
			frameIds.Add(reconstruction.Image(imageId).FrameId);
		}

		return frameIds.Count;
	}

	// Splits every input view-graph component once using the same rotation averaging and
	// relative-rotation filtering as the global mapper. Components that are already too
	// small are discarded without running rotation averaging.
	private static List<HashSet<uint>> ComputeComponentsByRotationAveraging(
		RotationEstimatorOptions options,
		PoseGraph poseGraph,
		Reconstruction baseReconstruction,
		IReadOnlyList<PosePrior> posePriors,
		int minModelSize)
	{
		var result = new List<HashSet<uint>>();
		List<HashSet<uint>> inputComponents =
			poseGraph.ConnectedImageIdsForFrameComponents(baseReconstruction, filterUnregistered: false);

		foreach (HashSet<uint> inputComponent in inputComponents)
		{
			if (NumFramesForImages(baseReconstruction, inputComponent) < minModelSize)
			{
				continue;
			}

			Reconstruction reconstruction = baseReconstruction.Clone();
			PoseGraph componentPoseGraph = poseGraph.Clone();
			componentPoseGraph.InvalidatePairsOutsideActiveImageIds(inputComponent);

			RotationEstimatorOptions decompositionOptions = options.Clone();
			decompositionOptions.FilterUnregistered = false;
			if (!RotationAveraging.RunRotationAveragingOnComponent(
				decompositionOptions, componentPoseGraph, inputComponent, reconstruction, posePriors))
			{
				continue;
			}

			if (decompositionOptions.MaxRotationErrorDeg > 0)
			{
				RotationAveraging.FilterEdgesByRelativeRotation(
					componentPoseGraph, reconstruction, decompositionOptions.MaxRotationErrorDeg);
			}

			result.AddRange(componentPoseGraph.ConnectedImageIdsForFrameComponents(reconstruction, filterUnregistered: true));
		}

		return result;
	}
}
