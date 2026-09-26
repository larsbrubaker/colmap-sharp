// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipeline: port of IncrementalPipeline from colmap/controllers/
// incremental_pipeline.h and .cc - the controller that runs incremental SfM over a database:
// it seeds a model from an initial image pair, registers the next images one by one with
// local and periodic global bundle adjustment (Sfm/IncrementalMapper*.cs), and repeats for
// further sub-models, relaxing the initialization thresholds when no pair can be found. Its
// options are IncrementalPipelineOptions.cs; this file holds construction, Run, the model
// trial loop (Reconstruct) and TriangulateReconstruction (the point triangulator's entry);
// IncrementalPipeline.SubModel.cs holds one model's mapping (InitializeReconstruction,
// ReconstructSubModel) and the file-local helpers. Tests:
// ColmapSharp.Tests/Controllers/IncrementalPipelineTests*.cs (incremental_pipeline_test.cc).
//
// Tier C (outcome): registered images and poses after Sim3 alignment within COLMAP's bounds.
//
// Translation notes:
// - Cancellation: BaseController.CancellationToken (and SetCheckIfStoppedFunc) is checked
//   exactly where COLMAP checks CheckIfStopped: between image registrations, and inside
//   every bundle adjustment through BundleAdjustmentOptions.CheckIfStopped.
// - Progress (C#-only): Progress, when set, receives an IncrementalPipelineProgress at the
//   points COLMAP fires its callbacks and before each global refinement.
// - COLMAP's Timer is a Stopwatch. LOG(WARNING)/LOG(ERROR) go to Util/Log.cs; LOG(INFO)
//   lines are dropped (Progress reports the milestones instead).
// - The options object is shared with the caller, as COLMAP's shared_ptr is: construction
//   lowers MinModelSize for small image sets in the caller's object.

using System.Diagnostics;

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>The stage an <see cref="IncrementalPipelineProgress"/> report was sent from.</summary>
public enum IncrementalPipelineStage
{
	/// <summary>The initial image pair of a model was registered.</summary>
	InitialPairRegistered,

	/// <summary>The next image was registered and locally refined.</summary>
	ImageRegistered,

	/// <summary>Retriangulation and global bundle adjustment is about to run.</summary>
	GlobalRefinement,

	/// <summary>A model is finished (kept or discarded).</summary>
	ModelFinished,
}

/// <summary>
/// Progress of an incremental reconstruction: <paramref name="NumRegImages"/> of
/// <paramref name="NumImages"/> images are registered in at least one model, and
/// <paramref name="NumModels"/> models exist so far.
/// </summary>
public readonly record struct IncrementalPipelineProgress(
	IncrementalPipelineStage Stage, int NumRegImages, int NumImages, int NumModels);

/// <summary>
/// Port of colmap::IncrementalPipeline: controls the incremental mapping procedure by
/// iteratively initializing reconstructions from the same scene graph.
/// </summary>
public sealed partial class IncrementalPipeline : BaseController
{
	/// <summary>Port of IncrementalPipeline::CallbackType.</summary>
	public enum CallbackType
	{
		/// <summary>After the initial image pair of a model is registered.</summary>
		InitialImagePairRegCallback,

		/// <summary>After each further image is registered.</summary>
		NextImageRegCallback,

		/// <summary>After a model is finished.</summary>
		LastImageRegCallback,
	}

	/// <summary>Port of IncrementalPipeline::Status.</summary>
	public enum Status
	{
		/// <summary>The model was reconstructed.</summary>
		Success,

		/// <summary>The run was stopped or reached its maximum runtime.</summary>
		Interrupted,

		/// <summary>Try again with relaxed initialization thresholds.</summary>
		Continue,

		/// <summary>No further models should be attempted.</summary>
		Stop,

		/// <summary>No initial image pair was found.</summary>
		NoInitialPair,

		/// <summary>The initial image pair did not give a usable model.</summary>
		BadInitialPair,

		/// <summary>A rig used by the images has a sensor without sensor_from_rig.</summary>
		UnknownSensorFromRig,
	}

	private readonly IncrementalPipelineOptions _options;
	private readonly ReconstructionManager _reconstructionManager;
	private readonly DatabaseCache _databaseCache;
	private readonly Stopwatch _totalRunTimer = new();

	/// <summary>Loads the database into a cache and prepares the pipeline.</summary>
	public IncrementalPipeline(
		IncrementalPipelineOptions options, Database database, ReconstructionManager reconstructionManager)
	{
		_options = Check.NotNull(options);
		_reconstructionManager = Check.NotNull(reconstructionManager);
		Check.That(_options.Check());
		Check.NotNull(database);

		_databaseCache = DatabaseCache.Create(database, CreateDatabaseCacheOptions(_options, _reconstructionManager));

		CustomizeIncrementalPipelineOptions(_databaseCache, _options);

		RegisterCallbacks();
	}

	/// <summary>Prepares the pipeline from an already loaded database cache (filtered by the options).</summary>
	public IncrementalPipeline(
		IncrementalPipelineOptions options, DatabaseCache databaseCache, ReconstructionManager reconstructionManager)
	{
		_options = Check.NotNull(options);
		_reconstructionManager = Check.NotNull(reconstructionManager);
		Check.That(_options.Check());
		Check.NotNull(databaseCache);

		_databaseCache = DatabaseCache.CreateFromCache(
			databaseCache, CreateDatabaseCacheOptions(_options, _reconstructionManager));

		CustomizeIncrementalPipelineOptions(_databaseCache, _options);

		RegisterCallbacks();
	}

	/// <summary>The options (shared with the caller).</summary>
	public IncrementalPipelineOptions Options => _options;

	/// <summary>The models reconstructed so far.</summary>
	public ReconstructionManager ReconstructionManager => _reconstructionManager;

	/// <summary>The database cache the pipeline reads.</summary>
	public DatabaseCache DatabaseCache => _databaseCache;

	/// <summary>Receives progress reports during <see cref="Run"/> (C#-only).</summary>
	public IProgress<IncrementalPipelineProgress>? Progress { get; set; }

	/// <summary>Runs incremental mapping until every image is registered or no model can grow.</summary>
	public override void Run()
	{
		_totalRunTimer.Restart();

		if (_databaseCache.NumImages == 0)
		{
			Log.Warning("No images with matches");
			return;
		}

		if (_options.UsePriorPosition && _databaseCache.NumPosePriors == 0)
		{
			Log.Warning("No pose priors");
			return;
		}

		// Is there a sub-reconstruction before we start the reconstruction? I.e. the user has
		// imported an existing reconstruction.
		bool continueReconstruction = _reconstructionManager.Size > 0;
		Check.Le(
			_reconstructionManager.Size,
			1,
			"Can only continue from a single reconstruction, but multiple are given.");

		int numImages = _databaseCache.NumImages;

		IncrementalMapper.Options mapperOptions = _options.Mapper();
		var mapper = new IncrementalMapper(_databaseCache);
		if (Reconstruct(mapper, mapperOptions, continueReconstruction) == Status.Stop)
		{
			return;
		}

		bool ShouldStop() => mapper.NumTotalRegImages == numImages || CheckIfStopped() || CheckReachedMaxRuntime();

		const int NumInitRelaxations = 2;
		for (int i = 0; i < NumInitRelaxations; ++i)
		{
			if (ShouldStop())
			{
				break;
			}

			// Relaxing the initialization constraints.
			mapperOptions.InitMinNumInliers /= 2;
			mapper.ResetInitializationStats();
			if (Reconstruct(mapper, mapperOptions, continueReconstruction: false) == Status.Stop)
			{
				break;
			}

			if (ShouldStop())
			{
				break;
			}

			mapperOptions.InitMinTriAngle /= 2;
			mapper.ResetInitializationStats();
			if (Reconstruct(mapper, mapperOptions, continueReconstruction: false) == Status.Stop)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Port of IncrementalPipeline::Reconstruct: tries up to init_num_trials models, keeping
	/// the successful ones, until no further model should be attempted.
	/// </summary>
	public Status Reconstruct(IncrementalMapper mapper, IncrementalMapper.Options mapperOptions, bool continueReconstruction)
	{
		for (int numTrials = 0; numTrials < _options.InitNumTrials; ++numTrials)
		{
			if (CheckIfStopped() || CheckReachedMaxRuntime())
			{
				return Status.Stop;
			}

			int reconstructionIdx = (!continueReconstruction || numTrials > 0) ? _reconstructionManager.Add() : 0;
			Reconstruction reconstruction = _reconstructionManager.Get(reconstructionIdx);

			Status status = ReconstructSubModel(mapper, mapperOptions, reconstruction);
			switch (status)
			{
				case Status.Interrupted:
					if (reconstruction.NumRegFrames == 0)
					{
						mapper.EndReconstruction(discard: true);
						_reconstructionManager.Delete(reconstructionIdx);
						return Status.Stop;
					}

					reconstruction.UpdatePoint3DErrors();
					// Keeping reconstruction due to interrupt.
					mapper.EndReconstruction(discard: false);
					AlignReconstructionToPriorsOrRigScale(_options, _databaseCache, reconstruction);
					return Status.Stop;

				case Status.UnknownSensorFromRig:
					Log.Error(
						"Discarding reconstruction due to unknown sensor_from_rig poses. Either explicitly "
						+ "define the poses by configuring the rigs or first run reconstruction without "
						+ "configured rigs and then derive the poses from the initial reconstruction for a "
						+ "subsequent reconstruction with rig constraints. See documentation for detailed "
						+ "instructions.");
					mapper.EndReconstruction(discard: true);
					_reconstructionManager.Delete(reconstructionIdx);
					// If the reconstruction was discarded due to an unknown sensor from rig, we
					// can stop the outer trial loop, because all trials will fail.
					return Status.Stop;

				case Status.BadInitialPair:
					mapper.EndReconstruction(discard: true);
					_reconstructionManager.Delete(reconstructionIdx);
					// If an initial pair was found but it was bad, we discard and attempt to
					// initialize from any of the remaining pairs in the next trials.
					break;

				case Status.NoInitialPair:
					mapper.EndReconstruction(discard: true);
					_reconstructionManager.Delete(reconstructionIdx);
					// If no pair could be found, we can exit the trial loop, because the next
					// trials in this loop will not find anything unless the initialization
					// thresholds are relaxed. However, by relaxing the constraints in the outer
					// loop we can succeed.
					return Status.Continue;

				case Status.Success:
				{
					// Remember the total number of registered images before potentially
					// discarding it below due to small size, so we can exit out of the main
					// loop, if all images were registered.
					int numRegImages = reconstruction.NumRegImages;
					int totalNumRegImages = mapper.NumTotalRegImages;

					// Always keep the first reconstruction, independent of size.
					if ((_options.MultipleModels && _reconstructionManager.Size > 1 && numRegImages < _options.MinModelSize)
						|| numRegImages == 0)
					{
						Log.Warning("Discarding reconstruction due to insufficient size");
						mapper.EndReconstruction(discard: true);
						_reconstructionManager.Delete(reconstructionIdx);
					}
					else
					{
						reconstruction.UpdatePoint3DErrors();
						mapper.EndReconstruction(discard: false);
						AlignReconstructionToPriorsOrRigScale(_options, _databaseCache, reconstruction);
					}

					Callback((int)CallbackType.LastImageRegCallback);
					ReportProgress(IncrementalPipelineStage.ModelFinished, mapper);

					// Check if we should or can reconstruct another sub-model.
					if (!_options.MultipleModels
						|| _reconstructionManager.Size >= _options.MaxNumModels
						|| totalNumRegImages >= _databaseCache.NumImages - 1)
					{
						return Status.Stop;
					}

					// In case the reconstruction was successful and there are remaining images,
					// we try to reconstruct another sub-model in the next trial.
					break;
				}

				default:
					throw new InvalidOperationException("Unknown reconstruction status.");
			}
		}

		return Status.Continue;
	}

	/// <summary>
	/// Port of IncrementalPipeline::CheckRunGlobalRefinement: whether the model grew enough
	/// since the last global refinement to run another.
	/// </summary>
	public bool CheckRunGlobalRefinement(Reconstruction reconstruction, int baPrevNumRegFrames, int baPrevNumPoints)
	{
		// C++ compares size_t with double products, i.e. in double.
		return reconstruction.NumRegFrames >= _options.BaGlobalFramesRatio * baPrevNumRegFrames
			|| (long)reconstruction.NumRegFrames >= (long)_options.BaGlobalFramesFreq + baPrevNumRegFrames
			|| reconstruction.NumPoints3D >= _options.BaGlobalPointsRatio * baPrevNumPoints
			|| (long)reconstruction.NumPoints3D >= (long)_options.BaGlobalPointsFreq + baPrevNumPoints;
	}

	/// <summary>
	/// Port of IncrementalPipeline::TriangulateReconstruction: triangulates the observations of
	/// every registered image of an existing model (the point triangulator), then retriangulates
	/// with global bundle adjustment (without normalizing) and colors the points from all
	/// images (black without <see cref="IncrementalPipelineOptions.ReadImage"/>).
	/// </summary>
	public void TriangulateReconstruction(Reconstruction reconstruction)
	{
		Check.Gt(_databaseCache.NumImages, 0, "No images with matches found in the database");
		var mapper = new IncrementalMapper(_databaseCache);
		mapper.BeginReconstruction(reconstruction);

		// Iterative triangulation.
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			if (CheckIfStopped())
			{
				break;
			}

			mapper.TriangulateImage(_options.Triangulation(), imageId);
		}

		if (!CheckIfStopped())
		{
			// Retriangulation and Global bundle adjustment.
			BundleAdjustmentOptions baOptions = _options.GlobalBundleAdjustment();
			baOptions.CheckIfStopped = CheckIfStopped;
			mapper.IterativeGlobalRefinement(
				_options.BaGlobalMaxRefinements,
				_options.BaGlobalMaxRefinementChange,
				_options.Mapper(),
				baOptions,
				_options.Triangulation(),
				normalizeReconstruction: false);
		}

		mapper.EndReconstruction(discard: false);

		reconstruction.UpdatePoint3DErrors();

		if (!CheckIfStopped())
		{
			// Extracting colors. Without an image source every read fails, as with COLMAP's
			// empty image_path, and the points become black.
			reconstruction.ExtractColorsForAllImages(_options.ReadImage ?? (_ => null), _options.NumThreads);
		}
	}

	/// <summary>Port of IncrementalPipeline::CheckReachedMaxRuntime.</summary>
	public bool CheckReachedMaxRuntime() =>
		_options.MaxRuntimeSeconds > 0 && _totalRunTimer.Elapsed.TotalSeconds > _options.MaxRuntimeSeconds;

	private void RegisterCallbacks()
	{
		RegisterCallback((int)CallbackType.InitialImagePairRegCallback);
		RegisterCallback((int)CallbackType.NextImageRegCallback);
		RegisterCallback((int)CallbackType.LastImageRegCallback);
	}

	private void ReportProgress(IncrementalPipelineStage stage, IncrementalMapper mapper) =>
		Progress?.Report(new IncrementalPipelineProgress(
			stage, mapper.NumTotalRegImages, _databaseCache.NumImages, _reconstructionManager.Size));

	// If the total number of images is small then do not enforce the minimum model size so
	// that we can reconstruct small image collections, i.e., if the model is at least half of
	// the total number of images, we always keep it.
	private static void CustomizeIncrementalPipelineOptions(DatabaseCache databaseCache, IncrementalPipelineOptions options)
	{
		// std::min<size_t>(0.5 * NumImages(), min_model_size): the product truncates to size_t.
		options.MinModelSize = (int)Math.Min((ulong)(0.5 * databaseCache.NumImages), (ulong)options.MinModelSize);
	}

	private static DatabaseCache.Options CreateDatabaseCacheOptions(
		IncrementalPipelineOptions options, ReconstructionManager reconstructionManager)
	{
		var databaseCacheOptions = new DatabaseCache.Options
		{
			MinNumMatches = options.MinNumMatches,
			IgnoreWatermarks = options.IgnoreWatermarks,
			ImageNames = [.. options.ImageNames],
		};
		// Make sure images of the given reconstruction are also included when manually
		// specifying images for the reconstruction procedure.
		if (reconstructionManager.Size == 1 && options.ImageNames.Count > 0)
		{
			Reconstruction reconstruction = reconstructionManager.Get(0);
			foreach (uint imageId in reconstruction.RegImageIds())
			{
				databaseCacheOptions.ImageNames.Add(reconstruction.Image(imageId).Name);
			}
		}

		databaseCacheOptions.LoadAllImages = options.LoadAllImages;
		databaseCacheOptions.ConvertPosePriorsToEnu = options.UsePriorPosition;
		return databaseCacheOptions;
	}

	private static void AlignReconstructionToPriorsOrRigScale(
		IncrementalPipelineOptions options, DatabaseCache databaseCache, Reconstruction reconstruction)
	{
		if (options.UsePriorPosition)
		{
			var priorOptions = new PosePriorBundleAdjustmentOptions();
			RansacOptions ransacOptions = priorOptions.AlignmentRansacOptions;
			ransacOptions.RandomSeed = options.RandomSeed;

			Sim3d metricFromReconstruction = Sim3d.Identity;
			if (Alignment.AlignReconstructionToPosePriors(
				reconstruction,
				databaseCache.PosePriors,
				ransacOptions,
				priorOptions.PriorPositionFallbackStddev,
				ref metricFromReconstruction))
			{
				reconstruction.Transform(metricFromReconstruction);
				return;
			}

			Log.Warning("Final alignment w.r.t. prior positions failed; restoring the original rig scale instead");
		}

		Alignment.AlignReconstructionToOrigRigScales(databaseCache.Rigs, reconstruction);
	}
}
