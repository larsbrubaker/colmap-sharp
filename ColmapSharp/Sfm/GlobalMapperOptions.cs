// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalMapperOptions: port of colmap::GlobalMapperOptions (colmap/sfm/global_mapper.h and
// the option accessors of global_mapper.cc), the settings of the global SfM pipeline in
// GlobalMapper.cs: the per-stage sub-options (rotation averaging, global positioning, bundle
// adjustment, retriangulation), track establishment limits, filter thresholds and the
// skip flags. The accessors RotationAveraging() ... Retriangulation() return copies of the
// sub-options with the top-level settings (seed, threads, refine_sensor_from_rig) applied.
// Tests: ColmapSharp.Tests/Sfm/GlobalMapperTests.cs (global_mapper_test.cc 1:1).
//
// Translation notes:
// - image_path is not ported: point colors are read through a host callback
//   (divergence 68), GlobalPipelineOptions.ReadImage.
// - As in COLMAP, the bundle_adjustment initializer sets the Ceres use_gpu, and
//   BundleAdjustment() forwards ba_gpu_index to the Ceres gpu_index; the managed solver has no
//   GPU path and ignores both. There are no Caspar (GPU) options, so BundleAdjustment() does
//   not set caspar->gpu_index (divergence 66).
// - The C++ accessors return by value; here each returns a fresh Clone().

using ColmapSharp.Estimators;
using ColmapSharp.Solver;

namespace ColmapSharp.Sfm;

/// <summary>Port of colmap::GlobalMapperOptions.</summary>
public sealed class GlobalMapperOptions
{
	/// <summary>Number of threads.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// PRNG seed for all stochastic methods during reconstruction. If -1 (default), the seed
	/// is derived from the current time (non-deterministic). If &gt;= 0, the pipeline is
	/// deterministic with the given seed.
	/// </summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>When false, treat each non-ref sensor's cam_from_rig as pre-calibrated.</summary>
	public bool RefineSensorFromRig { get; set; } = true;

	/// <summary>Options of the rotation averaging stage.</summary>
	public RotationEstimatorOptions RotationAveragingOptions { get; set; } = new();

	/// <summary>Options of the global positioning stage.</summary>
	public GlobalPositionerOptions GlobalPositioningOptions { get; set; } = new();

	/// <summary>Options of the bundle adjustment stage.</summary>
	public BundleAdjustmentOptions BundleAdjustmentOptions { get; set; } = DefaultBundleAdjustmentOptions();

	/// <summary>Options of the retriangulation stage.</summary>
	public IncrementalTriangulator.Options RetriangulationOptions { get; set; } = new()
	{
		CompleteMaxReprojError = 15.0,
		MergeMaxReprojError = 15.0,
		MinAngle = 1.0,
	};

	/// <summary>Max pixel distance between observations of the same track within one image.</summary>
	public double TrackIntraImageConsistencyThreshold { get; set; } = 10.0;

	/// <summary>Required number of tracks per view before early stopping.</summary>
	public int TrackRequiredTracksPerView { get; set; } = int.MaxValue;

	/// <summary>Minimum number of views per track.</summary>
	public int TrackMinNumViewsPerTrack { get; set; } = 3;

	/// <summary>
	/// Maximum total number of tracks to establish. Tracks are selected in order of
	/// decreasing length, so the longest tracks are kept. Use this to bound memory usage on
	/// large datasets. By default, there is no limit.
	/// </summary>
	public int KeepMaxNumTracks { get; set; } = int.MaxValue;

	/// <summary>Angular reprojection threshold (degrees) for global positioning.</summary>
	public double MaxAngularReprojErrorDeg { get; set; } = 1.0;

	/// <summary>Normalized reprojection threshold for bundle adjustment.</summary>
	public double MaxNormalizedReprojError { get; set; } = 1e-2;

	/// <summary>Minimum triangulation angle (degrees).</summary>
	public double MinTriAngleDeg { get; set; } = 1.0;

	/// <summary>
	/// GPU device index for bundle adjustment (-1 = auto-select). Kept for option
	/// compatibility; ColmapSharp solves on the CPU.
	/// </summary>
	public string BaGpuIndex { get; set; } = "-1";

	/// <summary>The number of iterations of bundle adjustment.</summary>
	public int BaNumIterations { get; set; } = 3;

	/// <summary>
	/// Whether to skip the fixed-rotation stage in bundle adjustment. By default, BA runs in
	/// two stages: first with fixed rotations (position only), then with full optimization.
	/// Setting this to true skips the first stage and runs full optimization directly.
	/// </summary>
	public bool BaSkipFixedRotationStage { get; set; }

	/// <summary>
	/// Whether to skip the joint optimization stage in bundle adjustment. When true, only
	/// the fixed-rotation stage is run (optimizing positions only). This is mutually
	/// exclusive with <see cref="BaSkipFixedRotationStage"/>.
	/// </summary>
	public bool BaSkipJointOptimizationStage { get; set; }

	/// <summary>Skips rotation averaging.</summary>
	public bool SkipRotationAveraging { get; set; }

	/// <summary>Skips track establishment.</summary>
	public bool SkipTrackEstablishment { get; set; }

	/// <summary>Skips global positioning.</summary>
	public bool SkipGlobalPositioning { get; set; }

	/// <summary>Skips bundle adjustment.</summary>
	public bool SkipBundleAdjustment { get; set; }

	/// <summary>Skips retriangulation.</summary>
	public bool SkipRetriangulation { get; set; }

	/// <summary>A deep copy (C++ copies the options struct by value).</summary>
	public GlobalMapperOptions Clone()
	{
		var copy = (GlobalMapperOptions)MemberwiseClone();
		copy.RotationAveragingOptions = RotationAveragingOptions.Clone();
		copy.GlobalPositioningOptions = GlobalPositioningOptions.Clone();
		copy.BundleAdjustmentOptions = BundleAdjustmentOptions.Clone();
		copy.RetriangulationOptions = RetriangulationOptions.Clone();
		return copy;
	}

	// The C++ member initializer lambda of bundle_adjustment.
	private static BundleAdjustmentOptions DefaultBundleAdjustmentOptions()
	{
		var options = new BundleAdjustmentOptions
		{
			MinTrackLength = 3,
			PrintSummary = false,
		};
		if (options.Ceres is not null)
		{
			options.Ceres.LossFunctionType = BundleAdjustmentLossFunctionType.Huber;
			options.Ceres.UseGpu = true;
			// TODO (COLMAP): Investigate whether disabling auto solver selection and using
			// explicit SPARSE_SCHUR is necessary for global SfM, or if we can just rely on
			// COLMAP's auto selection.
			options.Ceres.AutoSelectSolverType = false;
			options.Ceres.SolverOptions.FunctionTolerance = 1e-5;
			options.Ceres.SolverOptions.MaxNumIterations = 200;
			options.Ceres.SolverOptions.LinearSolverType = LinearSolverType.SparseSchur;
		}

		return options;
	}

	/// <summary>The rotation averaging options with the top-level settings applied.</summary>
	public RotationEstimatorOptions RotationAveraging()
	{
		RotationEstimatorOptions opts = RotationAveragingOptions.Clone();
		opts.RefineSensorFromRig = RefineSensorFromRig;
		if (RandomSeed >= 0)
		{
			opts.RandomSeed = RandomSeed;
		}

		return opts;
	}

	/// <summary>The global positioning options with the top-level settings applied.</summary>
	public GlobalPositionerOptions GlobalPositioning()
	{
		GlobalPositionerOptions opts = GlobalPositioningOptions.Clone();
		opts.RefineSensorFromRig = RefineSensorFromRig;
		opts.SolverOptions.NumThreads = NumThreads;
		if (RandomSeed >= 0)
		{
			opts.RandomSeed = RandomSeed;
			opts.UseParameterBlockOrdering = false;
		}

		return opts;
	}

	/// <summary>The bundle adjustment options with the top-level settings applied.</summary>
	public BundleAdjustmentOptions BundleAdjustment()
	{
		BundleAdjustmentOptions opts = BundleAdjustmentOptions.Clone();
		opts.RefineSensorFromRig = RefineSensorFromRig;
		if (opts.Ceres is not null)
		{
			opts.Ceres.SolverOptions.NumThreads = NumThreads;
			opts.Ceres.GpuIndex = BaGpuIndex;
		}

		// COLMAP also sets caspar->gpu_index here; there is no Caspar backend (entry 66).
		return opts;
	}

	/// <summary>The retriangulation options with the top-level seed applied.</summary>
	public IncrementalTriangulator.Options Retriangulation()
	{
		IncrementalTriangulator.Options opts = RetriangulationOptions.Clone();
		if (RandomSeed >= 0)
		{
			opts.RandomSeed = RandomSeed;
		}

		return opts;
	}
}
