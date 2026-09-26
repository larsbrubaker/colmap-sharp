// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipelineOptions: port of IncrementalPipelineOptions from
// colmap/controllers/incremental_pipeline.h and .cc - every knob of incremental SfM, and the
// builders that derive the mapper, triangulator and local/global bundle adjustment options
// from it. The pipeline that consumes them is IncrementalPipeline.cs. Tests:
// ColmapSharp.Tests/Controllers/IncrementalPipelineTests.Priors.cs (the Options cases).
//
// Translation notes:
// - image_path becomes ReadImage: the library does not decode image files, so the host
//   hands over a decoded bitmap per image name (null when it has none, which COLMAP treats
//   as a failed read: the points stay black). The default (null) behaves like COLMAP's
//   default empty image_path (docs/CPP_DIVERGENCES.md, entry 68).
// - ba_use_gpu / ba_gpu_index are forwarded to the Ceres options like COLMAP does; the
//   managed solver has no GPU path and ignores them.
// - The Caspar (GPU) backend is not available (docs/CPP_DIVERGENCES.md, entry 66): Check
//   rejects it exactly as a COLMAP build without CASPAR_ENABLED does, and the option builders
//   have no Caspar options to fill.
// - LoggingType / minimizer_progress_to_stdout have no counterpart (the solver does not log).

using ColmapSharp.Estimators;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::IncrementalPipelineOptions.</summary>
public sealed class IncrementalPipelineOptions
{
	// Default maximum number of bundle adjustment iterations for the Ceres backend, used when
	// ba_{local,global}_max_num_iterations is -1. The Caspar backend has different
	// convergence behavior and instead falls back to its own tuned default in
	// CasparBundleAdjustmentOptions::solver_iter_max.
	internal const int DefaultCeresLocalMaxNumIterations = 25;
	internal const int DefaultCeresGlobalMaxNumIterations = 50;

	/// <summary>
	/// COLMAP's CasparBundleAdjustmentOptions().solver_iter_max. There is no Caspar backend
	/// here, but EffBa*MaxNumIterations still reports it for a Caspar configuration, as COLMAP
	/// does.
	/// </summary>
	public const int CasparDefaultSolverIterMax = 200;

	/// <summary>The minimum number of matches for inlier matches to be considered.</summary>
	public int MinNumMatches { get; set; } = 15;

	/// <summary>Whether to ignore the inlier matches of watermark image pairs.</summary>
	public bool IgnoreWatermarks { get; set; }

	/// <summary>Whether to reconstruct multiple sub-models.</summary>
	public bool MultipleModels { get; set; } = true;

	/// <summary>The number of sub-models to reconstruct.</summary>
	public int MaxNumModels { get; set; } = 50;

	/// <summary>
	/// The maximum number of overlapping images between sub-models. If the current sub-model
	/// shares more than this number of images with another model, then the reconstruction is
	/// stopped.
	/// </summary>
	public int MaxModelOverlap { get; set; } = 20;

	/// <summary>
	/// The minimum number of registered images of a sub-model, otherwise the sub-model is
	/// discarded. Note that the first sub-model is always kept independent of size. If the
	/// model contains at least half of the total number of images, we also always keep it.
	/// </summary>
	public int MinModelSize { get; set; } = 10;

	/// <summary>
	/// The image identifiers used to initialize the reconstruction. Note that only one or both
	/// image identifiers can be specified. In the former case, the second image is
	/// automatically determined.
	/// </summary>
	public int InitImageId1 { get; set; } = -1;

	/// <summary>See <see cref="InitImageId1"/>.</summary>
	public int InitImageId2 { get; set; } = -1;

	/// <summary>The number of trials to initialize the reconstruction.</summary>
	public int InitNumTrials { get; set; } = 200;

	/// <summary>
	/// Enable fallback to structure-less image registration using 2D-2D correspondences, if
	/// structured-based registration fails using 2D-3D correspondences.
	/// </summary>
	public bool StructureLessRegistrationFallback { get; set; } = true;

	/// <summary>Only use structure-less and skip structure-based image registration.</summary>
	public bool StructureLessRegistrationOnly { get; set; }

	/// <summary>Whether to extract colors for reconstructed points.</summary>
	public bool ExtractColors { get; set; } = true;

	/// <summary>The number of threads to use during reconstruction.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>PRNG seed for all stochastic methods during reconstruction.</summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>Thresholds for filtering images with degenerate intrinsics.</summary>
	public double MinFocalLengthRatio { get; set; } = 0.1;

	/// <summary>See <see cref="MinFocalLengthRatio"/>.</summary>
	public double MaxFocalLengthRatio { get; set; } = 10.0;

	/// <summary>See <see cref="MinFocalLengthRatio"/>.</summary>
	public double MaxExtraParam { get; set; } = 1.0;

	/// <summary>Whether to optimize the focal length during the reconstruction.</summary>
	public bool BaRefineFocalLength { get; set; } = true;

	/// <summary>Whether to optimize the principal point during the reconstruction.</summary>
	public bool BaRefinePrincipalPoint { get; set; }

	/// <summary>Whether to optimize the extra camera parameters during the reconstruction.</summary>
	public bool BaRefineExtraParams { get; set; } = true;

	/// <summary>Whether to optimize rig poses during the reconstruction.</summary>
	public bool BaRefineSensorFromRig { get; set; } = true;

	/// <summary>
	/// The minimum number of residuals per bundle adjustment problem to enable
	/// multi-threading solving of the problems.
	/// </summary>
	public int BaMinNumResidualsForCpuMultiThreading { get; set; } = 50000;

	/// <summary>Solver function tolerance for local bundle adjustment.</summary>
	public double BaLocalFunctionTolerance { get; set; }

	/// <summary>
	/// The maximum number of local bundle adjustment iterations. If -1, the default of the
	/// configured bundle adjustment backend is used.
	/// </summary>
	public int BaLocalMaxNumIterations { get; set; } = -1;

	/// <summary>The growth rates after which to perform global bundle adjustment.</summary>
	public double BaGlobalFramesRatio { get; set; } = 1.1;

	/// <summary>See <see cref="BaGlobalFramesRatio"/>.</summary>
	public double BaGlobalPointsRatio { get; set; } = 1.1;

	/// <summary>See <see cref="BaGlobalFramesRatio"/>.</summary>
	public int BaGlobalFramesFreq { get; set; } = 500;

	/// <summary>See <see cref="BaGlobalFramesRatio"/>.</summary>
	public int BaGlobalPointsFreq { get; set; } = 250000;

	/// <summary>Solver function tolerance for global bundle adjustment.</summary>
	public double BaGlobalFunctionTolerance { get; set; }

	/// <summary>
	/// The maximum number of global bundle adjustment iterations. If -1, the default of the
	/// configured bundle adjustment backend is used.
	/// </summary>
	public int BaGlobalMaxNumIterations { get; set; } = -1;

	/// <summary>The thresholds for iterative bundle adjustment refinements.</summary>
	public int BaLocalMaxRefinements { get; set; } = 2;

	/// <summary>See <see cref="BaLocalMaxRefinements"/>.</summary>
	public double BaLocalMaxRefinementChange { get; set; } = 0.001;

	/// <summary>See <see cref="BaLocalMaxRefinements"/>.</summary>
	public int BaGlobalMaxRefinements { get; set; } = 5;

	/// <summary>See <see cref="BaLocalMaxRefinements"/>.</summary>
	public double BaGlobalMaxRefinementChange { get; set; } = 0.0005;

	/// <summary>Whether to use a GPU sparse linear algebra library (ignored: no GPU path).</summary>
	public bool BaUseGpu { get; set; }

	/// <summary>GPU device index for bundle adjustment (-1 = auto-select; ignored).</summary>
	public string BaGpuIndex { get; set; } = "-1";

	/// <summary>Bundle adjustment solver backend for local bundle adjustment.</summary>
	public BundleAdjustmentBackend BaLocalBackend { get; set; } = BundleAdjustmentBackend.Ceres;

	/// <summary>Bundle adjustment solver backend for global bundle adjustment.</summary>
	public BundleAdjustmentBackend BaGlobalBackend { get; set; } = BundleAdjustmentBackend.Ceres;

	/// <summary>Whether to use priors on the camera positions.</summary>
	public bool UsePriorPosition { get; set; }

	/// <summary>Whether to use a robust loss on prior camera positions.</summary>
	public bool UseRobustLossOnPriorPosition { get; set; }

	/// <summary>
	/// Threshold on the residual for the robust position prior loss (chi2 for 3DOF at 95% =
	/// 7.815).
	/// </summary>
	public double PriorPositionLossScale { get; set; } = 7.815;

	/// <summary>
	/// Path to a folder with reconstruction snapshots during incremental reconstruction.
	/// Snapshots will be saved according to the specified frequency of registered frames.
	/// </summary>
	public string SnapshotPath { get; set; } = "";

	/// <summary>See <see cref="SnapshotPath"/>; 0 disables snapshots.</summary>
	public int SnapshotFramesFreq { get; set; }

	/// <summary>
	/// Decodes the image with the given name (COLMAP's image_path / name) to extract point
	/// colors, or returns null when it cannot. If not set, all point colors will be black.
	/// </summary>
	public Func<string, Bitmap?>? ReadImage { get; set; }

	/// <summary>
	/// Optional list of image names to reconstruct. If no images are specified, all images
	/// will be reconstructed by default.
	/// </summary>
	public List<string> ImageNames { get; set; } = [];

	/// <summary>
	/// Whether to load all images from the database, including those without
	/// correspondences. Only useful for triangulation where all images are already registered
	/// and should retain their keypoints. Should not be enabled for incremental SfM.
	/// </summary>
	public bool LoadAllImages { get; set; }

	/// <summary>If reconstruction is provided as input, fix the existing frame poses.</summary>
	public bool FixExistingFrames { get; set; }

	/// <summary>
	/// Rigs for which to fix the sensor_from_rig transformation, independent of
	/// ba_refine_sensor_from_rig.
	/// </summary>
	public HashSet<uint> ConstantRigs { get; set; } = [];

	/// <summary>
	/// Cameras for which to fix the camera parameters independent of refine_focal_length,
	/// refine_principal_point, and refine_extra_params.
	/// </summary>
	public HashSet<uint> ConstantCameras { get; set; } = [];

	/// <summary>
	/// Maximum runtime in seconds for the reconstruction process. If set to a non-positive
	/// value, the process will run until completion.
	/// </summary>
	public int MaxRuntimeSeconds { get; set; } = -1;

	/// <summary>The mapper options the pipeline options override (see <see cref="Mapper"/>).</summary>
	public IncrementalMapper.Options MapperOptions { get; set; } = new();

	/// <summary>The triangulation options the pipeline options override (see <see cref="Triangulation"/>).</summary>
	public IncrementalTriangulator.Options TriangulationOptions { get; set; } = new();

	/// <summary>True when both initial image ids are given.</summary>
	public bool IsInitialPairProvided => InitImageId1 != -1 && InitImageId2 != -1;

	/// <summary>Port of IncrementalPipelineOptions::Mapper: the mapper options with the pipeline's overrides.</summary>
	public IncrementalMapper.Options Mapper()
	{
		IncrementalMapper.Options options = MapperOptions.Clone();
		options.AbsPoseRefineFocalLength = BaRefineFocalLength;
		options.AbsPoseRefineExtraParams = BaRefineExtraParams;
		options.MinFocalLengthRatio = MinFocalLengthRatio;
		options.MaxFocalLengthRatio = MaxFocalLengthRatio;
		options.MaxExtraParam = MaxExtraParam;
		options.NumThreads = NumThreads;
		options.FixExistingFrames = FixExistingFrames;
		options.ConstantRigs = [.. ConstantRigs];
		options.ConstantCameras = [.. ConstantCameras];
		options.UsePriorPosition = UsePriorPosition;
		options.UseRobustLossOnPriorPosition = UseRobustLossOnPriorPosition;
		options.PriorPositionLossScale = PriorPositionLossScale;
		options.RandomSeed = RandomSeed;
		return options;
	}

	/// <summary>Port of IncrementalPipelineOptions::Triangulation.</summary>
	public IncrementalTriangulator.Options Triangulation()
	{
		IncrementalTriangulator.Options options = TriangulationOptions.Clone();
		options.MinFocalLengthRatio = MinFocalLengthRatio;
		options.MaxFocalLengthRatio = MaxFocalLengthRatio;
		options.MaxExtraParam = MaxExtraParam;
		options.RandomSeed = RandomSeed;
		return options;
	}

	/// <summary>Port of IncrementalPipelineOptions::LocalBundleAdjustment.</summary>
	public BundleAdjustmentOptions LocalBundleAdjustment()
	{
		var options = new BundleAdjustmentOptions
		{
			PrintSummary = false,
			Backend = BaLocalBackend,
			RefineFocalLength = BaRefineFocalLength,
			RefinePrincipalPoint = BaRefinePrincipalPoint,
			RefineExtraParams = BaRefineExtraParams,
			RefineSensorFromRig = BaRefineSensorFromRig,
		};
		if (options.Ceres is { } ceres)
		{
			ceres.SolverOptions.FunctionTolerance = BaLocalFunctionTolerance;
			ceres.SolverOptions.GradientTolerance = 10.0;
			ceres.SolverOptions.ParameterTolerance = 0.0;
			ceres.SolverOptions.MaxNumIterations =
				BaLocalMaxNumIterations >= 0 ? BaLocalMaxNumIterations : DefaultCeresLocalMaxNumIterations;
			ceres.SolverOptions.MaxLinearSolverIterations = 100;
			ceres.SolverOptions.NumThreads = NumThreads;
			ceres.MinNumResidualsForCpuMultiThreading = BaMinNumResidualsForCpuMultiThreading;
			ceres.LossFunctionScale = 1.0;
			ceres.LossFunctionType = BundleAdjustmentLossFunctionType.SoftL1;
			ceres.UseGpu = BaUseGpu;
			ceres.GpuIndex = BaGpuIndex;
		}

		return options;
	}

	/// <summary>Port of IncrementalPipelineOptions::GlobalBundleAdjustment.</summary>
	public BundleAdjustmentOptions GlobalBundleAdjustment()
	{
		var options = new BundleAdjustmentOptions
		{
			PrintSummary = false,
			Backend = BaGlobalBackend,
			RefineFocalLength = BaRefineFocalLength,
			RefinePrincipalPoint = BaRefinePrincipalPoint,
			RefineExtraParams = BaRefineExtraParams,
			RefineSensorFromRig = BaRefineSensorFromRig,
		};
		if (options.Ceres is { } ceres)
		{
			ceres.SolverOptions.FunctionTolerance = BaGlobalFunctionTolerance;
			ceres.SolverOptions.GradientTolerance = 1.0;
			ceres.SolverOptions.ParameterTolerance = 0.0;
			ceres.SolverOptions.MaxNumIterations =
				BaGlobalMaxNumIterations >= 0 ? BaGlobalMaxNumIterations : DefaultCeresGlobalMaxNumIterations;
			ceres.SolverOptions.MaxLinearSolverIterations = 100;
			ceres.SolverOptions.NumThreads = NumThreads;
			ceres.MinNumResidualsForCpuMultiThreading = BaMinNumResidualsForCpuMultiThreading;
			ceres.LossFunctionType = BundleAdjustmentLossFunctionType.Trivial;
			ceres.UseGpu = BaUseGpu;
			ceres.GpuIndex = BaGpuIndex;
		}

		return options;
	}

	/// <summary>
	/// Port of EffBaLocalMaxNumIterations: the effective maximum number of local bundle
	/// adjustment iterations (the configured backend's default when the option is -1).
	/// </summary>
	public int EffBaLocalMaxNumIterations()
	{
		if (BaLocalMaxNumIterations >= 0)
		{
			return BaLocalMaxNumIterations;
		}

		return BaLocalBackend == BundleAdjustmentBackend.Caspar
			? CasparDefaultSolverIterMax
			: DefaultCeresLocalMaxNumIterations;
	}

	/// <summary>Port of EffBaGlobalMaxNumIterations (see <see cref="EffBaLocalMaxNumIterations"/>).</summary>
	public int EffBaGlobalMaxNumIterations()
	{
		if (BaGlobalMaxNumIterations >= 0)
		{
			return BaGlobalMaxNumIterations;
		}

		return BaGlobalBackend == BundleAdjustmentBackend.Caspar
			? CasparDefaultSolverIterMax
			: DefaultCeresGlobalMaxNumIterations;
	}

	/// <summary>Port of IncrementalPipelineOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check() =>
		MinNumMatches > 0
		&& MaxNumModels > 0
		&& MaxModelOverlap > 0
		&& MinModelSize >= 0
		&& InitNumTrials > 0
		&& MinFocalLengthRatio > 0
		&& MaxFocalLengthRatio > 0
		&& MaxExtraParam >= 0
		&& BaLocalMaxNumIterations >= -1
		&& BaGlobalFramesRatio > 1.0
		&& BaGlobalPointsRatio > 1.0
		&& BaGlobalFramesFreq > 0
		&& BaGlobalPointsFreq > 0
		&& BaGlobalMaxNumIterations >= -1
		&& BaGlobalMaxNumIterations != 0
		&& BaLocalMaxRefinements > 0
		&& BaLocalMaxRefinementChange >= 0
		&& BaGlobalMaxRefinements >= 0
		&& BaGlobalMaxRefinementChange >= 0
		&& SnapshotFramesFreq >= 0
		&& PriorPositionLossScale > 0.0
		&& NumThreads >= -1
		&& RandomSeed >= -1
		&& BaLocalBackend != BundleAdjustmentBackend.Caspar
		&& BaGlobalBackend != BundleAdjustmentBackend.Caspar
		&& Mapper().Check()
		&& Triangulation().Check();

	/// <summary>A deep copy (C++ copies the options struct by value).</summary>
	public IncrementalPipelineOptions Clone()
	{
		var clone = (IncrementalPipelineOptions)MemberwiseClone();
		clone.ImageNames = [.. ImageNames];
		clone.ConstantRigs = [.. ConstantRigs];
		clone.ConstantCameras = [.. ConstantCameras];
		clone.MapperOptions = MapperOptions.Clone();
		clone.TriangulationOptions = TriangulationOptions.Clone();
		return clone;
	}
}
