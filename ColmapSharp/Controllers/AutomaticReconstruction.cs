// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionController: port of colmap/controllers/automatic_reconstruction.cc -
// photos in, sparse models, fused dense points and a mesh out. It runs feature extraction
// (FeatureExtraction.cs), matching (FeatureMatching.cs), sparse mapping (IncrementalPipeline,
// HierarchicalPipeline or GlobalPipeline) and, per model, the dense stages in
// AutomaticReconstruction.Dense.cs, the last of which (texturing, C#-only) is in
// AutomaticReconstruction.Texture.cs. The options and presets are in
// AutomaticReconstructionOptions.cs. Tests: ColmapSharp.Tests/Controllers/
// AutomaticReconstructionTests.cs (automatic_reconstruction_test.cc).
//
// Tier C (outcome): the result goes through RANSAC and bundle adjustment.
//
// Translation notes:
// - The Thread becomes BaseController: Stop() is the host's CancellationToken, and Run() runs
//   the stages synchronously. Like COLMAP, Run returns between stages once stopped; a stage
//   stopped part-way throws OperationCanceledException, as the stage controllers here do.
// - RunAsync (C#-only) runs the same stages and awaits PatchMatch on the host's compute device,
//   for a device that cannot be waited on synchronously (the browser). Both share RunStages,
//   an iterator that yields each PatchMatchController for its caller to run blocking or
//   awaited, so the two entries differ only in that step.
// - database.db is a Database (InMemoryDatabase unless the host passes one), so a re-run only
//   skips extraction and matching when the host passes the same database back.
// - COLMAP's LOG_HEADING1 lines become Progress reports whose Stage names the step. Every
//   sub-stage's progress is re-labelled with the controller's own stage (the sub-stage's name,
//   e.g. "PatchMatch geometric", moves into Message), so a host grouping by Stage sees only
//   the stages listed on Progress.
// - option_manager_.Write(sparse/project.ini) has no counterpart (no ini registry), and the
//   vocabulary-tree matcher is never chosen (docs/CPP_DIVERGENCES.md entry 134).

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::AutomaticReconstructionController.</summary>
public sealed partial class AutomaticReconstructionController : BaseController
{
	/// <summary>The Stage of the progress reports of the sparse mappers.</summary>
	public const string SparseStage = "Sparse reconstruction";

	private readonly AutomaticReconstructionOptions options;
	private readonly ReconstructionOptionSet optionManager = new();
	private readonly ReconstructionManager reconstructionManager;
	private readonly Database database;
	private ImageReaderOptions? readerOptions;

	/// <summary>
	/// Prepares a reconstruction of <paramref name="options"/>' images into
	/// <paramref name="reconstructionManager"/>. Features and matches go to
	/// <paramref name="database"/> (a new InMemoryDatabase when null).
	/// </summary>
	public AutomaticReconstructionController(
		AutomaticReconstructionOptions options,
		ReconstructionManager reconstructionManager,
		Database? database = null)
	{
		this.options = Check.NotNull(options);
		Check.That(Directory.Exists(options.WorkspacePath), $"Directory {options.WorkspacePath} does not exist");
		Check.NotNull(options.Images);
		this.reconstructionManager = Check.NotNull(reconstructionManager);
		this.database = database ?? new InMemoryDatabase();

		optionManager.ImageReader.Images = options.Images!;
		optionManager.ImageReader.ImageNames = [.. options.ImageNames];
		optionManager.Mapper.ImageNames = [.. options.ImageNames];

		switch (options.Data)
		{
			case AutomaticReconstructionOptions.DataType.Video:
				optionManager.ModifyForVideoData();
				// Deliberate fix of an upstream ordering bug (docs/CPP_DIVERGENCES.md entry 134):
				// COLMAP sets image_names above, then ModifyForVideoData's ResetOptions(false)
				// rebuilds image_reader and mapper and restores only the project/database/image
				// *paths*, so video data silently ignores the selection that individual and
				// internet data keep. The port restores it. The image source is restored too
				// because it is image_path in COLMAP.
				optionManager.ImageReader.Images = options.Images!;
				optionManager.ImageReader.ImageNames = [.. options.ImageNames];
				optionManager.Mapper.ImageNames = [.. options.ImageNames];
				break;
			case AutomaticReconstructionOptions.DataType.Individual:
				optionManager.ModifyForIndividualData();
				break;
			case AutomaticReconstructionOptions.DataType.Internet:
				optionManager.ModifyForInternetData();
				break;
			default:
				throw new InvalidOperationException("Data type not supported");
		}

		Check.That(CameraModels.ExistsCameraModelWithName(options.CameraModel));

		// Set feature type first so quality modifiers can query EffMaxImageSize().
		switch (options.Feature)
		{
			case AutomaticReconstructionOptions.FeatureType.Sift:
				optionManager.FeatureExtraction.Type = FeatureExtractorType.Sift;
				optionManager.FeatureMatching.Type = FeatureMatcherType.SiftBruteForce;
				break;
			case AutomaticReconstructionOptions.FeatureType.Aliked:
				optionManager.FeatureExtraction.Type = FeatureExtractorType.AlikedN16Rot;
				optionManager.FeatureMatching.Type = FeatureMatcherType.AlikedBruteForce;
				break;
			case AutomaticReconstructionOptions.FeatureType.Loma:
				optionManager.FeatureExtraction.Type = FeatureExtractorType.LomaB;
				optionManager.FeatureMatching.Type = FeatureMatcherType.LomaB;
				break;
			case AutomaticReconstructionOptions.FeatureType.Loma128:
				optionManager.FeatureExtraction.Type = FeatureExtractorType.LomaB128;
				optionManager.FeatureMatching.Type = FeatureMatcherType.LomaB128;
				break;
		}

		// Apply quality preset (scales max_image_size relative to extractor default).
		switch (options.Quality)
		{
			case AutomaticReconstructionOptions.QualityLevel.Low:
				optionManager.ModifyForLowQuality();
				break;
			case AutomaticReconstructionOptions.QualityLevel.Medium:
				optionManager.ModifyForMediumQuality();
				break;
			case AutomaticReconstructionOptions.QualityLevel.High:
				optionManager.ModifyForHighQuality();
				break;
			case AutomaticReconstructionOptions.QualityLevel.Extreme:
				optionManager.ModifyForExtremeQuality();
				break;
		}

		// Feature-specific overrides that must come after quality.
		if (options.Feature != AutomaticReconstructionOptions.FeatureType.Sift)
		{
			// Guided matching is not supported for ALIKED/LoMa
			optionManager.FeatureMatching.GuidedMatching = false;
		}

		optionManager.FeatureExtraction.NumThreads = options.NumThreads;
		optionManager.FeatureMatching.NumThreads = options.NumThreads;
		// sequential_pairing / vocab_tree_pairing num_threads only feed the vocabulary tree,
		// which is out of scope, so they have no counterpart.
		optionManager.Mapper.NumThreads = options.NumThreads;
		optionManager.PatchMatchStereo.NumThreads = options.NumThreads;
		optionManager.PoissonMeshing.NumThreads = options.NumThreads;
		optionManager.DelaunayMeshing.NumThreads = options.NumThreads;
		optionManager.MeshTextureMapping.NumThreads = options.NumThreads;

		// COLMAP turns on loop detection with its downloadable vocabulary tree; without
		// vocabulary-tree support sequential matching runs without loop detection
		// (docs/CPP_DIVERGENCES.md entry 134).
		optionManager.SequentialPairing.LoopDetection = false;

		// Apply mapper-appropriate two-view geometry defaults.
		// Global uses stricter thresholds; Incremental/Hierarchical use standard.
		TwoViewGeometryOptions twoViewGeometryOptions = optionManager.TwoViewGeometry;
		twoViewGeometryOptions.RansacOptions.RandomSeed = options.RandomSeed;
		if (options.Mapper == AutomaticReconstructionOptions.MapperType.Global)
		{
			twoViewGeometryOptions.RansacOptions.MaxError = 1.0;
			twoViewGeometryOptions.MinNumInliers = 30;
			twoViewGeometryOptions.MinInlierRatio = 0.25;
			// Disable guided matching for global mapper to avoid regression issues.
			// Currently the guided matching leads to significantly worse results of the
			// global pipeline.
			optionManager.FeatureMatching.GuidedMatching = false;
		}

		optionManager.Mapper.RandomSeed = options.RandomSeed;

		if (options.Masks is not null)
		{
			optionManager.StereoFusion.MaskPath = MaskRoot;
		}

		// use_gpu / gpu_index / ba_backend: no GPU stages and only the Ceres-style backend
		// exist here, so the mapper keeps its CPU defaults.
	}

	/// <summary>
	/// Receives a report at the start of every step (Done = Total = 0, Stage = the step's
	/// COLMAP heading) and the steps' own progress. Stage is always one of
	/// FeatureExtraction.ExtractionStage, FeatureMatching.MatchingStage, SparseStage,
	/// DenseStage (undistortion and PatchMatch), FusionStage, MeshingStage or TexturingStage; a sub-stage's
	/// own name (e.g. "Geometric verification", "Image undistortion") is prefixed to Message.
	/// Done/Total count the units of the sub-stage reporting, so they restart when it changes.
	/// </summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <summary>The database holding the features and matches.</summary>
	public Database Database => database;

	/// <summary>Whether any of the selected reconstruction stages requires OpenGL.</summary>
	public bool RequiresOpenGL() =>
		(options.Extraction && optionManager.FeatureExtraction.RequiresOpenGL())
		|| (options.Matching && optionManager.FeatureMatching.RequiresOpenGL());

	/// <summary>Port of AutomaticReconstructionController::Setup: prepares the image reader.</summary>
	public void Setup()
	{
		if (options.Extraction)
		{
			ImageReaderOptions reader = optionManager.ImageReader;
			reader.Masks = options.Masks;
			reader.SingleCamera = options.SingleCamera;
			reader.SingleCameraPerFolder = options.SingleCameraPerFolder;
			reader.CameraModel = options.CameraModel;
			reader.CameraParams = options.CameraParams;
			reader.Images = options.Images!;
			reader.AsRgb = optionManager.FeatureExtraction.RequiresRGB();
			readerOptions = reader;
		}
	}

	/// <summary>
	/// Runs the selected stages synchronously on the calling thread (a UI host should call it
	/// off its UI thread). PatchMatch uses <see cref="AutomaticReconstructionOptions.ComputeDevice"/>
	/// only when it reports <see cref="Compute.IComputeDevice.SupportsBlockingWait"/>; any other
	/// device (the browser's) is skipped with a warning and PatchMatch runs on the CPU. Hosts in
	/// the browser must call <see cref="RunAsync"/> to use their GPU.
	/// </summary>
	public override void Run()
	{
		// PatchMatchController.Run is the one place this blocks on the device, and
		// BlockingComputeDevice only hands it a device that allows that.
		foreach (PatchMatchController patchMatch in RunStages(deviceIsAwaited: false))
		{
			patchMatch.Run(CancellationToken, Under(DenseStage));
		}
	}

	/// <summary>
	/// <see cref="Run"/>, awaiting PatchMatch on
	/// <see cref="AutomaticReconstructionOptions.ComputeDevice"/> instead of blocking on it, so any
	/// device works, including one that cannot be waited on synchronously: the entry point for a
	/// host in the browser. The stages, their order, outputs, progress and stop semantics
	/// (<see cref="BaseController.CancellationToken"/> and the stop function) are those of Run.
	/// Every stage but PatchMatch on the device is CPU work that runs synchronously inside this
	/// call, so a UI host should start it off its UI thread where it has one.
	/// </summary>
	public async Task RunAsync()
	{
		foreach (PatchMatchController patchMatch in RunStages(deviceIsAwaited: true))
		{
			await patchMatch.RunAsync(CancellationToken, Under(DenseStage)).ConfigureAwait(false);
		}
	}

	// The body of COLMAP's Run, shared by Run and RunAsync: every stage runs here except
	// PatchMatch, whose controller is yielded for the caller to run (blocking or awaiting)
	// before the enumeration resumes with fusion. deviceIsAwaited says which the caller does,
	// and so whether the host's device may be used when it cannot be waited on synchronously.
	private IEnumerable<PatchMatchController> RunStages(bool deviceIsAwaited)
	{
		if (CheckIfStopped())
		{
			yield break;
		}

		if (options.Extraction)
		{
			RunFeatureExtraction();
		}

		if (CheckIfStopped())
		{
			yield break;
		}

		if (options.Matching)
		{
			RunFeatureMatching();
		}

		if (CheckIfStopped())
		{
			yield break;
		}

		if (options.Sparse)
		{
			RunSparseMapper();
		}

		if (CheckIfStopped())
		{
			yield break;
		}

		if (options.Dense)
		{
			foreach (PatchMatchController patchMatch in RunDenseMapper(deviceIsAwaited))
			{
				yield return patchMatch;
			}
		}
	}

	private void Heading(string stage) => Progress?.Report(new ControllerProgress(stage, 0, 0, ""));

	private void RunFeatureExtraction()
	{
		Heading(FeatureExtraction.ExtractionStage);

		// THROW_CHECK_NOTNULL(feature_extractor_): Setup must have run.
		Check.NotNull(readerOptions);
		FeatureExtraction.ExtractFeatures(
			database, readerOptions!, optionManager.FeatureExtraction, Under(FeatureExtraction.ExtractionStage), CancellationToken);
	}

	private void RunFeatureMatching()
	{
		Heading(FeatureMatching.MatchingStage);

		// Vocabulary-tree matching (vocab_tree_path, >= 200 images) is out of scope, so
		// individual and internet data always match exhaustively.
		if (options.Data == AutomaticReconstructionOptions.DataType.Video)
		{
			FeatureMatching.MatchSequential(database, optionManager.SequentialPairing,
				optionManager.FeatureMatching, optionManager.TwoViewGeometry, Under(FeatureMatching.MatchingStage), CancellationToken);
		}
		else
		{
			FeatureMatching.MatchExhaustive(database, optionManager.ExhaustivePairing,
				optionManager.FeatureMatching, optionManager.TwoViewGeometry, Under(FeatureMatching.MatchingStage), CancellationToken);
		}
	}

	private void RunSparseMapper()
	{
		Heading(SparseStage);

		string sparsePath = Path.Combine(options.WorkspacePath, "sparse");
		if (Directory.Exists(sparsePath))
		{
			string[] dirList = Directory.GetDirectories(sparsePath);
			Array.Sort(dirList, StringComparer.Ordinal);
			if (dirList.Length > 0)
			{
				// Skipping sparse reconstruction because it is already computed.
				foreach (string dir in dirList)
				{
					reconstructionManager.Read(dir);
				}

				return;
			}
		}

		Func<string, Bitmap?> readImage = options.Images!.Read;
		BaseController mapper;
		switch (options.Mapper)
		{
			case AutomaticReconstructionOptions.MapperType.Incremental:
			{
				IncrementalPipelineOptions mapperOptions = optionManager.Mapper.Clone();
				mapperOptions.ReadImage = readImage;
				mapper = new IncrementalPipeline(mapperOptions, database, reconstructionManager)
				{
					Progress = Forward<IncrementalPipelineProgress>(
						p => new ControllerProgress(SparseStage, p.NumRegImages, p.NumImages, p.Stage.ToString())),
				};
				break;
			}

			case AutomaticReconstructionOptions.MapperType.Hierarchical:
			{
				var mapperOptions = new HierarchicalPipelineOptions
				{
					ReadImage = readImage,
					IncrementalOptions = optionManager.Mapper.Clone(),
				};
				mapper = new HierarchicalPipeline(mapperOptions, database, reconstructionManager)
				{
					Progress = Under(SparseStage),
				};
				break;
			}

			case AutomaticReconstructionOptions.MapperType.Global:
			{
				var vgcOptions = new ViewGraphCalibrationOptions { RandomSeed = options.RandomSeed };
				vgcOptions.SolverOptions.NumThreads = options.NumThreads;
				ViewGraphCalibration.CalibrateViewGraph(vgcOptions, database);
				var globalOptions = new GlobalPipelineOptions
				{
					ReadImage = readImage,
					NumThreads = options.NumThreads,
					RandomSeed = options.RandomSeed,
				};
				mapper = new GlobalPipeline(globalOptions, database, reconstructionManager)
				{
					Progress = Forward<GlobalPipelineProgress>(
						p => new ControllerProgress(SparseStage, p.Component, p.NumComponents, p.Stage.ToString())),
				};
				break;
			}

			default:
				throw new InvalidOperationException("Mapper not supported");
		}

		mapper.CancellationToken = CancellationToken;
		mapper.SetCheckIfStoppedFunc(CheckIfStopped);
		mapper.Run();

		Directory.CreateDirectory(sparsePath);
		reconstructionManager.Write(sparsePath);
	}

	// Maps a stage's own progress type onto Progress, synchronously on the reporting thread.
	private IProgress<T>? Forward<T>(Func<T, ControllerProgress> map) =>
		Progress is null ? null : new MappedProgress<T>(Progress, map);

	// Re-labels a sub-stage's ControllerProgress with the controller's stage, keeping the
	// sub-stage's own name as the start of Message so the detail is not lost.
	private IProgress<ControllerProgress>? Under(string stage) =>
		Forward<ControllerProgress>(p => p.Stage == stage ? p : p with
		{
			Stage = stage,
			Message = p.Message.Length == 0 ? p.Stage : p.Stage + ": " + p.Message,
		});

	private sealed class MappedProgress<T>(IProgress<ControllerProgress> target, Func<T, ControllerProgress> map) : IProgress<T>
	{
		public void Report(T value) => target.Report(map(value));
	}
}
