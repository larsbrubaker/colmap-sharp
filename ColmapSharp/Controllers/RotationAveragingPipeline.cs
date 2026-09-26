// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingPipeline: port of colmap/controllers/rotation_averaging.h and .cc, the
// controller that estimates only the global rotations of a database's frames: it loads the
// database into a cache (optionally decomposing the relative poses of the two-view
// geometries), seeds frame rotations from gravity priors, optionally refines the gravity
// priors (Estimators/GravityRefinement.cs) and runs rotation averaging
// (Estimators/RotationAveraging.cs). Sfm/GlobalMapper.cs runs the same estimator as the
// first stage of full global SfM. Tests:
// ColmapSharp.Tests/Controllers/RotationAveragingPipelineTests.cs (rotation_averaging_test.cc
// of controllers/, 1:1).
//
// Tier C (outcome): rotations against the ground truth within COLMAP's tolerances.
//
// Translation notes:
// - Cancellation: COLMAP's Run never checks CheckIfStopped. Here BaseController's
//   CancellationToken (and SetCheckIfStoppedFunc) is checked before gravity seeding, gravity
//   refinement and rotation averaging; a stop after seeding un-poses the seeded frames, so a
//   stopped run never leaves half-posed frames (docs/CPP_DIVERGENCES.md entry 101).
// - Progress (C#-only): Progress, when set, receives a ControllerProgress as each stage
//   finishes, the way the incremental pipeline reports its milestones.
// - LOG(ERROR) goes to Util/Log.cs; LOG(INFO) and the timer are dropped.
// - The gravity prior initialization looks the image up by the prior's corr_data_id (camera
//   priors only), where COLMAP uses pose_prior_id (docs/CPP_DIVERGENCES.md entry 102).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::RotationAveragingPipelineOptions.</summary>
public sealed class RotationAveragingPipelineOptions
{
	/// <summary>The minimum number of matches for inlier matches to be considered.</summary>
	public int MinNumMatches { get; set; }

	/// <summary>Whether to ignore the inlier matches of watermark image pairs.</summary>
	public bool IgnoreWatermarks { get; set; }

	/// <summary>Names of images to reconstruct. If empty, all images are used.</summary>
	public List<string> ImageNames { get; set; } = [];

	/// <summary>Number of threads.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// PRNG seed for all stochastic methods during reconstruction. If -1 (default), the seed
	/// is derived from the current time (non-deterministic). If &gt;= 0, the pipeline is
	/// deterministic with the given seed.
	/// </summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>Whether to decompose relative poses from two-view geometries.</summary>
	public bool DecomposeRelativePose { get; set; } = true;

	/// <summary>Whether to refine gravity priors before rotation averaging.</summary>
	public bool RefineGravity { get; set; }

	/// <summary>Options for gravity refinement.</summary>
	public GravityRefinerOptions GravityRefiner { get; set; } = new();

	/// <summary>Options for rotation averaging.</summary>
	public RotationEstimatorOptions RotationEstimation { get; set; } = new();

	/// <summary>A deep copy (C++ copies the options struct by value).</summary>
	public RotationAveragingPipelineOptions Clone()
	{
		var copy = (RotationAveragingPipelineOptions)MemberwiseClone();
		copy.ImageNames = [.. ImageNames];
		copy.GravityRefiner = GravityRefiner.Clone();
		copy.RotationEstimation = RotationEstimation.Clone();
		return copy;
	}
}

/// <summary>
/// Port of colmap::RotationAveragingPipeline: estimates global frame rotations of a database
/// into a reconstruction.
/// </summary>
public sealed class RotationAveragingPipeline : BaseController
{
	/// <summary>The stage name reported after gravity refinement.</summary>
	public const string GravityRefinementStage = "Gravity refinement";

	/// <summary>The stage name reported after rotation averaging.</summary>
	public const string RotationAveragingStage = "Rotation averaging";

	private readonly RotationAveragingPipelineOptions _options;
	private readonly DatabaseCache _databaseCache;
	private readonly Reconstruction _reconstruction;

	/// <summary>
	/// Loads <paramref name="database"/> into a cache (decomposing relative poses if the
	/// options say so); <see cref="Run"/> writes the rotations into
	/// <paramref name="reconstruction"/>.
	/// </summary>
	public RotationAveragingPipeline(
		RotationAveragingPipelineOptions options, Database database, Reconstruction reconstruction)
	{
		_options = Check.NotNull(options).Clone();
		_reconstruction = Check.NotNull(reconstruction);
		Check.NotNull(database);
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
	}

	/// <summary>Receives a report as each stage finishes (C#-only).</summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <summary>Runs gravity initialization, optional gravity refinement and rotation averaging.</summary>
	public override void Run()
	{
		// Propagate options to component options.
		RotationAveragingPipelineOptions options = _options.Clone();
		options.RotationEstimation.RandomSeed = options.RandomSeed;
		options.GravityRefiner.SolverOptions.NumThreads = options.NumThreads;

		// Load reconstruction and pose graph from database cache.
		_reconstruction.Load(_databaseCache);
		var poseGraph = new PoseGraph();
		poseGraph.Load(_databaseCache.CorrespondenceGraph);

		if (poseGraph.Empty)
		{
			Log.Error("Cannot continue without image pairs");
			return;
		}

		// Get a mutable copy of pose priors.
		List<PosePrior> posePriors = [.. _databaseCache.PosePriors];

		// Stop before any frame is seeded; later stops un-pose the seeded frames
		// (StopAndClearSeeds), so a stopped run never leaves half-posed frames.
		if (CheckIfStopped())
		{
			return;
		}

		// Initialize frame rotations from gravity priors. COLMAP looks the image up by
		// pose_prior_id; the prior's image is its corr_data_id, as everywhere else (entry 102).
		var unknownTranslation = new Vector3d(double.NaN, double.NaN, double.NaN);
		var seededFrameIds = new List<uint>();
		foreach (PosePrior posePrior in posePriors)
		{
			if (!posePrior.HasGravity() || posePrior.CorrDataId.SensorId.Type != SensorType.Camera)
			{
				continue;
			}

			Image image = _reconstruction.Image((uint)posePrior.CorrDataId.Id);
			if (!image.IsRefInFrame)
			{
				continue;
			}

			_reconstruction.Frame(image.FrameId).SetRigFromWorld(new Rigid3d(
				Quaterniond.FromRotationMatrix(Pose.GravityAlignedRotation(posePrior.Gravity)),
				unknownTranslation));
			seededFrameIds.Add(image.FrameId);
		}

		// Optionally refine gravity priors (only if gravity priors exist).
		if (options.RefineGravity && posePriors.Count > 0)
		{
			if (StopAndClearSeeds(seededFrameIds))
			{
				return;
			}

			// Compute largest connected component and invalidate pairs before gravity
			// refinement.
			HashSet<uint> activeFrameIds = poseGraph.LargestConnectedFrameComponent(
				_reconstruction, filterUnregistered: false);
			var activeImageIds = new HashSet<uint>();
			foreach ((uint imageId, Image image) in _reconstruction.Images)
			{
				if (activeFrameIds.Contains(image.FrameId))
				{
					activeImageIds.Add(imageId);
				}
			}

			poseGraph.InvalidatePairsOutsideActiveImageIds(activeImageIds);

			GravityRefinement.RunGravityRefinement(options.GravityRefiner, poseGraph, _reconstruction, posePriors);
			Progress?.Report(new ControllerProgress(GravityRefinementStage, 1, 1, ""));
		}

		if (StopAndClearSeeds(seededFrameIds))
		{
			return;
		}

		if (!RotationAveraging.RunRotationAveraging(options.RotationEstimation, poseGraph, _reconstruction, posePriors))
		{
			Log.Error("Failed to solve rotation averaging");
			return;
		}

		Progress?.Report(new ControllerProgress(RotationAveragingStage, 1, 1, ""));
	}

	// The stop check after gravity seeding: when stopped, un-poses the seeded frames (their
	// rotation is a gravity prior and their translation NaN), so a stopped run leaves no
	// half-posed frames (entry 101).
	private bool StopAndClearSeeds(List<uint> seededFrameIds)
	{
		if (!CheckIfStopped())
		{
			return false;
		}

		foreach (uint frameId in seededFrameIds)
		{
			_reconstruction.Frame(frameId).ResetPose();
		}

		return true;
	}
}
