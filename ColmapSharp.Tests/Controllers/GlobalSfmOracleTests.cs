// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GlobalSfmOracleTests (C#-only): global SfM against pycolmap 4.2.0 on the same databases
// (oracle/fixture_global_sfm.py writes TestData/oracle/global_sfm.json; its header describes
// the cases). One test per pycolmap entry point:
// - ViewGraphCalibration.CalibrateViewGraph (pycolmap.calibrate_view_graph);
// - RotationAveraging.RunRotationAveraging (pycolmap.run_rotation_averaging);
// - GravityRefinement.RunGravityRefinement (pycolmap.run_gravity_refinement);
// - GlobalPipeline (pycolmap.global_mapping: rotation averaging, track establishment, global
//   positioning, bundle adjustment, retriangulation), after the calibration where the case
//   calibrates first.
// Each runs on the cases whose "steps" include it. GlobalSfmOracleTests.Fixture.cs rebuilds
// the input databases and holds the error measures.
//
// Tier C (outcome). Both sides start from the identical database and the same seed
// (single-threaded); what can differ is Eigen vs. LinearAlgebra/ and Ceres vs. Solver/.
// Compared:
// - exactly: success flags, registered images, 3D point counts, two-view configurations,
//   inlier counts, focal priors;
// - rotations: the largest relative rotation error between any two registered images of ours
//   and of pycolmap's (gauge invariant, so no alignment is needed);
// - positions: our projection centers mapped onto pycolmap's by a Sim3 (EstimateSim3d), the
//   largest residual relative to the scene's radius (RMS distance of pycolmap's centers from
//   their centroid);
// - focal lengths relative to pycolmap's, and each re-estimated relative pose's rotation and
//   translation direction.
// pycolmap's run_gravity_refinement cannot return the refined gravities (the fixture script
// explains why), so that test compares the number of refined gravities with the count COLMAP
// logs and checks the gravities against the ground truth at gravity_refinement_test.cc's bound.
// The bounds, and the errors measured when the fixture was written, are stated with the
// constants below. Ours is also checked against the ground truth, so a failure both sides
// share cannot pass silently.

using System.Text.Json;

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class GlobalSfmOracleTests
{
	// Measured when the fixture was written (largest over the cases): rotation averaging
	// 1.1e-6 deg from pycolmap, the pipeline 5.9e-8 deg and 2.2e-9 of the scene radius. Both
	// sides converge to the same solution; the bounds leave room for the solvers' stopping
	// points.
	private const double RotationAveragingToleranceDeg = 1e-5;
	private const double PipelineRotationToleranceDeg = 1e-5;
	private const double PipelinePositionTolerance = 1e-7;

	// The uncalibrated case's pipeline runs on our calibration, whose relative pose of pair
	// 2-3 differs from pycolmap's by 0.41 deg (UncalibratedPairsSeededDifferently). Measured:
	// 4.0e-4 deg and 2.5e-5 of the scene radius.
	private const string UncalibratedCase = "uncalibrated";
	private const double UncalibratedPipelineRotationToleranceDeg = 1e-3;
	private const double UncalibratedPipelinePositionTolerance = 1e-4;

	// Against the ground truth, so a failure both sides share cannot pass. Measured: without
	// noise rotation averaging recovers it to 1e-13 deg (the synthetic relative poses are
	// exact) and the pipeline to 3.5e-6 deg and 1.1e-7 of the radius; with corrupted relative
	// rotations rotation averaging stays within 2.9e-4 deg, and the pipeline, bundle adjusted
	// on 0.5 px noise, lands up to 0.11 deg and 7.2e-3 of the radius away.
	private const double GroundTruthRotationAveragingToleranceDeg = 5e-3;
	private const double GroundTruthPipelineRotationToleranceDeg = 0.2;
	private const double GroundTruthPositionTolerance = 0.01;

	// View graph calibration. Measured: focal lengths 4.2e-10 relative to pycolmap's (1.5e-8
	// in calibration_config_tagging, where the two broken F matrices drive every focal to about
	// 7574, far from the true 1280, on both sides); re-estimated relative poses 6.5e-7 deg in
	// rotation and 8.5e-7 deg in translation direction.
	private const double CalibrationFocalRelativeTolerance = 1e-7;
	private const double CalibrationPoseToleranceDeg = 1e-5;

	// Pairs whose re-estimated relative pose starts LO-RANSAC's local optimization from a
	// different five-point solution than pycolmap does (docs/CPP_DIVERGENCES.md entry 124):
	// same configuration and inlier count, but a pose 0.41 deg (rotation) and 0.25 deg
	// (translation direction) away, where the ten-step local optimization stops.
	private static readonly HashSet<string> UncalibratedPairsSeededDifferently = ["uncalibrated 2-3"];
	private const double SeededDifferentlyPoseToleranceDeg = 0.5;

	// gravity_refinement_test.cc's bound on every refined gravity.
	private const double GravityToleranceDeg = 1e-2;

	public static IEnumerable<string> CalibrationCases() => CasesWithStep("calibrate");

	public static IEnumerable<string> RotationAveragingCases() => CasesWithStep("rotation_averaging");

	public static IEnumerable<string> GravityCases() => CasesWithStep("gravity");

	public static IEnumerable<string> PipelineCases() => CasesWithStep("pipeline");

	private static ViewGraphCalibrationOptions CalibrationOptions(JsonElement expected)
	{
		var options = new ViewGraphCalibrationOptions { RandomSeed = Seed };
		foreach (JsonProperty option in expected.GetProperty("calibration_options").EnumerateObject())
		{
			switch (option.Name)
			{
				case "reestimate_relative_pose": options.ReestimateRelativePose = option.Value.GetBoolean(); break;
				case "max_calibration_error": options.MaxCalibrationError = option.Value.GetDouble(); break;
				default: throw new InvalidOperationException($"Unknown calibration option {option.Name}");
			}
		}

		return options;
	}

	[Test]
	[MethodDataSource(nameof(CalibrationCases))]
	public async Task CSharpOnly_ViewGraphCalibrationMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);
		JsonElement expectedResult = expected.GetProperty("calibration");

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		bool success = ViewGraphCalibration.CalibrateViewGraph(CalibrationOptions(expected), database);

		Dictionary<uint, Camera> cameras = database.ReadAllCameras().ToDictionary(c => c.CameraId);
		double maxFocalError = 0;
		var ourPriors = new List<bool>();
		var theirPriors = new List<bool>();
		foreach (JsonElement c in expectedResult.GetProperty("cameras").EnumerateArray())
		{
			Camera theirs = ReadCamera(c);
			Camera ours = cameras[theirs.CameraId];
			maxFocalError = Math.Max(maxFocalError, Math.Abs(ours.MeanFocalLength() / theirs.MeanFocalLength() - 1));
			ourPriors.Add(ours.HasPriorFocalLength);
			theirPriors.Add(theirs.HasPriorFocalLength);
		}

		var ourPairs = new List<string>();
		var theirPairs = new List<string>();
		double maxPoseError = 0;
		double maxSeededDifferentlyPoseError = 0;
		foreach (JsonElement p in expectedResult.GetProperty("pairs").EnumerateArray())
		{
			uint imageId1 = p.GetProperty("image_id1").GetUInt32();
			uint imageId2 = p.GetProperty("image_id2").GetUInt32();
			TwoViewGeometry ours = database.ReadTwoViewGeometry(imageId1, imageId2);
			Rigid3d? theirPose = MaybePose(p.GetProperty("cam2_from_cam1"));
			ourPairs.Add($"{imageId1}-{imageId2} {ours.Config} {ours.InlierMatches.Count} {ours.Cam2FromCam1 is not null}");
			theirPairs.Add($"{imageId1}-{imageId2} {(TwoViewGeometry.ConfigurationType)p.GetProperty("config").GetInt32()} "
				+ $"{p.GetProperty("num_inliers").GetInt32()} {theirPose is not null}");
			if (ours.Cam2FromCam1 is Rigid3d ourPose && theirPose is Rigid3d pose)
			{
				double poseError = Math.Max(
					RadToDeg(ourPose.Rotation.AngularDistance(pose.Rotation)), AngleDeg(ourPose.Translation, pose.Translation));
				if (UncalibratedPairsSeededDifferently.Contains($"{caseName} {imageId1}-{imageId2}"))
				{
					maxSeededDifferentlyPoseError = Math.Max(maxSeededDifferentlyPoseError, poseError);
				}
				else
				{
					maxPoseError = Math.Max(maxPoseError, poseError);
				}
			}
		}

		await Assert.That(success).IsEqualTo(expectedResult.GetProperty("success").GetBoolean());
		await Assert.That(ourPriors).IsEquivalentTo(theirPriors, TUnit.Assertions.Enums.CollectionOrdering.Matching);
		await Assert.That(ourPairs).IsEquivalentTo(theirPairs, TUnit.Assertions.Enums.CollectionOrdering.Matching);
		await Assert.That(maxFocalError).IsLessThanOrEqualTo(CalibrationFocalRelativeTolerance);
		await Assert.That(maxPoseError).IsLessThanOrEqualTo(CalibrationPoseToleranceDeg);
		await Assert.That(maxSeededDifferentlyPoseError).IsLessThanOrEqualTo(SeededDifferentlyPoseToleranceDeg);
	}

	[Test]
	[MethodDataSource(nameof(RotationAveragingCases))]
	public async Task CSharpOnly_RotationAveragingMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);
		JsonElement expectedResult = expected.GetProperty("rotation_averaging");

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		(Reconstruction reconstruction, PoseGraph poseGraph) = LoadReconstructionAndPoseGraph(database);
		bool success = RotationAveraging.RunRotationAveraging(
			new RotationEstimatorOptions { RandomSeed = Seed }, poseGraph, reconstruction, database.ReadAllPosePriors());

		Dictionary<uint, Quaterniond> ours = reconstruction.RegImageIds().ToDictionary(
			id => id, id => reconstruction.Image(id).CamFromWorld().Rotation);
		Dictionary<uint, Quaterniond> theirs = expectedResult.GetProperty("rotations").EnumerateArray().ToDictionary(
			r => r.GetProperty("image_id").GetUInt32(), r => Quaternion(r.GetProperty("q_xyzw")));
		Dictionary<uint, Quaterniond> groundTruth = Poses(expected.GetProperty("gt_cam_from_world"))
			.ToDictionary(p => p.Key, p => p.Value.Rotation);

		await Assert.That(success).IsEqualTo(expectedResult.GetProperty("success").GetBoolean());
		await Assert.That(ours.Keys.Order()).IsEquivalentTo(theirs.Keys.Order(), TUnit.Assertions.Enums.CollectionOrdering.Matching);
		await Assert.That(MaxRelativeRotationErrorDeg(ours, theirs)).IsLessThanOrEqualTo(RotationAveragingToleranceDeg);
		await Assert.That(MaxRelativeRotationErrorDeg(ours, groundTruth)).IsLessThanOrEqualTo(GroundTruthRotationAveragingToleranceDeg);
	}

	[Test]
	[MethodDataSource(nameof(GravityCases))]
	public async Task CSharpOnly_GravityRefinementMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);
		JsonElement expectedResult = expected.GetProperty("gravity");

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		(Reconstruction reconstruction, PoseGraph poseGraph) = LoadReconstructionAndPoseGraph(database);
		List<PosePrior> posePriors = database.ReadAllPosePriors();
		List<Vector3d> before = [.. posePriors.Select(p => p.Gravity)];
		GravityRefinement.RunGravityRefinement(new GravityRefinerOptions(), poseGraph, reconstruction, posePriors);

		int numRefined = posePriors.Where((p, i) => p.Gravity != before[i]).Count();
		Dictionary<uint, Rigid3d> groundTruth = Poses(expected.GetProperty("gt_cam_from_world"));
		var gravityInWorld = new Vector3d(0, 1, 0); // SyntheticDatasetOptions.prior_gravity_in_world
		double maxGravityError = posePriors.Max(p => AngleDeg(p.Gravity, groundTruth[(uint)p.CorrDataId.Id].Rotation * gravityInWorld));

		await Assert.That(numRefined).IsEqualTo(expectedResult.GetProperty("num_refined_gravities").GetInt32());
		await Assert.That(maxGravityError).IsLessThanOrEqualTo(GravityToleranceDeg);
	}

	[Test]
	[MethodDataSource(nameof(PipelineCases))]
	public async Task CSharpOnly_GlobalPipelineMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		if (expected.TryGetProperty("calibration", out _))
		{
			ViewGraphCalibration.CalibrateViewGraph(CalibrationOptions(expected), database);
		}

		var reconstructionManager = new ReconstructionManager();
		var options = new GlobalPipelineOptions { NumThreads = 1, RandomSeed = Seed };
		new GlobalPipeline(options, database, reconstructionManager).Run();

		List<JsonElement> expectedModels = [.. expected.GetProperty("pipeline").EnumerateArray()];
		Dictionary<uint, Rigid3d> groundTruth = Poses(expected.GetProperty("gt_cam_from_world"));
		await Assert.That(reconstructionManager.Size).IsEqualTo(expectedModels.Count);
		for (int i = 0; i < expectedModels.Count; ++i)
		{
			Reconstruction reconstruction = reconstructionManager.Get(i);
			Dictionary<uint, Rigid3d> ours = reconstruction.RegImageIds().ToDictionary(
				id => id, id => reconstruction.Image(id).CamFromWorld());
			Dictionary<uint, Rigid3d> theirs = Poses(expectedModels[i].GetProperty("cam_from_world"));
			Dictionary<uint, Quaterniond> ourRotations = ours.ToDictionary(p => p.Key, p => p.Value.Rotation);

			await Assert.That(ours.Keys.Order()).IsEquivalentTo(theirs.Keys.Order(), TUnit.Assertions.Enums.CollectionOrdering.Matching);
			await Assert.That(reconstruction.NumPoints3D).IsEqualTo(expectedModels[i].GetProperty("num_points3D").GetInt32());
			await Assert.That(MaxRelativeRotationErrorDeg(ourRotations, theirs.ToDictionary(p => p.Key, p => p.Value.Rotation)))
				.IsLessThanOrEqualTo(caseName == UncalibratedCase ? UncalibratedPipelineRotationToleranceDeg : PipelineRotationToleranceDeg);
			await Assert.That(MaxRelativeRotationErrorDeg(ourRotations, groundTruth.ToDictionary(p => p.Key, p => p.Value.Rotation)))
				.IsLessThanOrEqualTo(GroundTruthPipelineRotationToleranceDeg);
			await Assert.That(MaxAlignedPositionError(ours, theirs))
				.IsLessThanOrEqualTo(caseName == UncalibratedCase ? UncalibratedPipelinePositionTolerance : PipelinePositionTolerance);
			await Assert.That(MaxAlignedPositionError(ours, groundTruth)).IsLessThanOrEqualTo(GroundTruthPositionTolerance);
		}
	}

	// LoadReconstructionAndPoseGraph of rotation_averaging_test.cc and gravity_refinement_test.cc.
	private static (Reconstruction Reconstruction, PoseGraph PoseGraph) LoadReconstructionAndPoseGraph(Database database)
	{
		var cache = new DatabaseCache();
		cache.Load(database, new DatabaseCache.Options());
		var reconstruction = new Reconstruction();
		reconstruction.Load(cache);
		var poseGraph = new PoseGraph();
		poseGraph.Load(cache.CorrespondenceGraph);
		return (reconstruction, poseGraph);
	}
}
