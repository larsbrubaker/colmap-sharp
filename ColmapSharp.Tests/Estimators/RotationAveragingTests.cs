// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingTests: colmap/estimators/rotation_averaging_test.cc ported 1:1, one method
// per gtest case named Suite_Name, testing ColmapSharp/Estimators/RotationAveraging.cs,
// RotationEstimator.cs and, through them, RotationAveragingProblem*.cs and
// RotationAveragingSolver.cs. The database is InMemoryDatabase (not an SQLite file). Each
// ExpectEqualRotations collects the largest pairwise error and asserts it once. The C#-only
// cases are in RotationAveragingTests.CSharpOnly.cs. Tier C (outcome): relative rotations
// against ground truth within COLMAP's tolerances.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test; tests seed and draw before
// their first await (the PRNG is thread-local).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class RotationAveragingTests
{
	private sealed class TestData
	{
		public Reconstruction GtReconstruction { get; } = new();

		public Reconstruction Reconstruction { get; } = new();

		public PoseGraph PoseGraph { get; } = new();

		public List<PosePrior> PosePriors { get; set; } = [];
	}

	// CreateTestData + LoadReconstructionAndPoseGraph of rotation_averaging_test.cc.
	private static TestData CreateTestData(SyntheticDatasetOptions datasetOptions, SyntheticNoiseOptions? noiseOptions = null)
	{
		var data = new TestData();
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(datasetOptions, data.GtReconstruction, database);
		if (noiseOptions is not null)
		{
			Synthetic.SynthesizeNoise(noiseOptions, data.GtReconstruction, database);
		}

		var cache = new DatabaseCache();
		cache.Load(database, new DatabaseCache.Options());
		data.Reconstruction.Load(cache);
		data.PoseGraph.Load(cache.CorrespondenceGraph);
		data.PosePriors = database.ReadAllPosePriors();
		return data;
	}

	// C++'s PoseGraph copy constructor (edges copied in the same order).
	private static PoseGraph CopyPoseGraph(PoseGraph poseGraph)
	{
		var copy = new PoseGraph();
		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.Edges)
		{
			copy.Edges.Add(pairId, edge.Clone());
		}

		return copy;
	}

	private static RotationEstimatorOptions CreateRATestOptions(bool useGravity = false) => new()
	{
		SkipInitialization = false,
		UseGravity = useGravity,
		UseStratified = true,
	};

	// ExpectEqualRotations: the largest pairwise relative rotation error (radians).
	private static double MaxRelativeRotationError(Reconstruction gt, Reconstruction computed)
	{
		List<uint> regImageIds = gt.RegImageIds();
		double maxError = 0;
		for (int i = 0; i < regImageIds.Count; i++)
		{
			for (int j = 0; j < i; j++)
			{
				Quaterniond cam2FromCam1 = computed.Image(regImageIds[j]).CamFromWorld().Rotation
					* computed.Image(regImageIds[i]).CamFromWorld().Rotation.Inverse();
				Quaterniond cam2FromCam1Gt = gt.Image(regImageIds[j]).CamFromWorld().Rotation
					* gt.Image(regImageIds[i]).CamFromWorld().Rotation.Inverse();
				maxError = Math.Max(maxError, cam2FromCam1.AngularDistance(cam2FromCam1Gt));
			}
		}

		return maxError;
	}

	// Gauge-invariant mean rotation error (in degrees) over all registered image pairs,
	// comparing the computed relative rotations against the ground truth.
	private static double MeanRelativeRotationErrorDeg(Reconstruction gt, Reconstruction computed)
	{
		List<uint> regImageIds = gt.RegImageIds();
		double totalErrorRad = 0;
		int count = 0;
		for (int i = 0; i < regImageIds.Count; i++)
		{
			for (int j = 0; j < i; j++)
			{
				Quaterniond rel = computed.Image(regImageIds[j]).CamFromWorld().Rotation
					* computed.Image(regImageIds[i]).CamFromWorld().Rotation.Inverse();
				Quaterniond relGt = gt.Image(regImageIds[j]).CamFromWorld().Rotation
					* gt.Image(regImageIds[i]).CamFromWorld().Rotation.Inverse();
				totalErrorRad += rel.AngularDistance(relGt);
				count++;
			}
		}

		return MathUtils.RadToDeg(totalErrorRad / count);
	}

	private static void ResetSensorsFromRig(Reconstruction reconstruction)
	{
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensor) in rig.NonRefSensors.ToList())
			{
				if (sensor is not null)
				{
					reconstruction.Rig(rigId).ResetSensorFromRig(sensorId);
				}
			}
		}
	}

	// RunAndVerifyRotationAveraging: the largest error over the use_gravity values.
	private static double RunAndVerifyRotationAveraging(TestData data, bool[] useGravityValues)
	{
		double maxError = 0;
		foreach (bool useGravity in useGravityValues)
		{
			Reconstruction reconstructionCopy = data.Reconstruction.Clone();
			PoseGraph poseGraphCopy = CopyPoseGraph(data.PoseGraph);
			RotationAveraging.RunRotationAveraging(
				CreateRATestOptions(useGravity), poseGraphCopy, reconstructionCopy, data.PosePriors);
			maxError = Math.Max(maxError, MaxRelativeRotationError(data.GtReconstruction, reconstructionCopy));
		}

		return maxError;
	}

	private static SyntheticDatasetOptions SmallRigOptions(int numCamerasPerRig, int numFramesPerRig, bool priorGravity) => new()
	{
		NumRigs = 1,
		NumCamerasPerRig = numCamerasPerRig,
		NumFramesPerRig = numFramesPerRig,
		NumPoints3D = 50,
		SensorFromRigRotationStddev = 20.0,
		PriorGravity = priorGravity,
		TwoViewGeometryHasRelativePose = true,
	};

	private static SyntheticDatasetOptions NoisyOptions(int numCamerasPerRig) => new()
	{
		NumRigs = 2,
		NumCamerasPerRig = numCamerasPerRig,
		NumFramesPerRig = 7,
		NumPoints3D = 100,
		InlierMatchRatio = 0.6,
		PriorGravity = true,
		TwoViewGeometryHasRelativePose = true,
	};

	private static SyntheticNoiseOptions NoiseOptions() => new()
	{
		Point2DStddev = 1,
		PriorGravityStddev = 3e-1,
	};

	private static Dictionary<(uint, SensorId), Rigid3d> SnapshotSensorsFromRig(Reconstruction reconstruction)
	{
		var snapshot = new Dictionary<(uint, SensorId), Rigid3d>();
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				snapshot[(rigId, sensorId)] = sensorFromRig!.Value;
			}
		}

		return snapshot;
	}

	// Every non-ref sensor still has a sensor_from_rig equal to the snapshot's.
	private static bool SensorsFromRigMatch(Reconstruction reconstruction, Dictionary<(uint, SensorId), Rigid3d> snapshot)
	{
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? after) in rig.NonRefSensors)
			{
				if (after is not Rigid3d value || value != snapshot[(rigId, sensorId)])
				{
					return false;
				}
			}
		}

		return true;
	}

	[Test]
	public async Task RotationAveraging_WithoutNoise()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5, priorGravity: true));
		double maxError = RunAndVerifyRotationAveraging(data, [true, false]);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	[Test]
	public async Task RotationAveraging_WeightedNoiseFreeMatchesInvariant()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5, priorGravity: true));

		// Assign varying positive match counts so the edge weighting is non-trivial.
		int counter = 1;
		foreach (PoseGraph.Edge edge in data.PoseGraph.Edges.Values)
		{
			edge.NumMatches = 10 * (counter++ % 7) + 1;
		}

		double maxGtError = 0;
		double maxUnweightedError = 0;
		foreach (bool useGravity in new[] { true, false })
		{
			// Unweighted baseline.
			Reconstruction reconUnweighted = data.Reconstruction.Clone();
			RotationEstimatorOptions optionsUnweighted = CreateRATestOptions(useGravity);
			optionsUnweighted.Reweighting = RotationAveragingReweighting.Uniform;
			RotationAveraging.RunRotationAveraging(
				optionsUnweighted, CopyPoseGraph(data.PoseGraph), reconUnweighted, data.PosePriors);

			// Weighted.
			Reconstruction reconWeighted = data.Reconstruction.Clone();
			RotationEstimatorOptions optionsWeighted = CreateRATestOptions(useGravity);
			optionsWeighted.Reweighting = RotationAveragingReweighting.InlierMatchCount;
			RotationAveraging.RunRotationAveraging(
				optionsWeighted, CopyPoseGraph(data.PoseGraph), reconWeighted, data.PosePriors);

			// The weighted solution recovers the ground truth and, for a noise-free system, is
			// identical to the unweighted solution.
			maxGtError = Math.Max(maxGtError, MaxRelativeRotationError(data.GtReconstruction, reconWeighted));
			maxUnweightedError = Math.Max(maxUnweightedError, MaxRelativeRotationError(reconUnweighted, reconWeighted));
		}

		await Assert.That(maxGtError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
		await Assert.That(maxUnweightedError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	[Test]
	public async Task RotationAveraging_WeightedReducesErrorWithNoisyLowMatchEdges()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 15,
			NumPoints3D = 150,
			PriorGravity = false,
			TwoViewGeometryHasRelativePose = true,
		});

		// Inject controlled rotation noise into each relative pose and halve the match count
		// of the noisy edges. A uniform baseline match count ensures the only weight
		// difference between runs is the halving of the noisy edges.
		const double kNoiseThresholdDeg = 5.0;
		const int kBaselineMatches = 100;
		foreach (PoseGraph.Edge edge in data.PoseGraph.Edges.Values)
		{
			// Range kept just above the 5 deg threshold: noisy edges (5-8 deg) still carry
			// meaningful IRLS weight (sigma = 5 deg), so the 2x down-weighting has real
			// leverage; sub-5 deg edges anchor the solution.
			double noiseDeg = RandomUtils.RandomUniformReal(0.0, 8.0);

			// Seeded isotropic axis (RandomEigenVectord<3>() is NOT seeded by SetPRNGSeed).
			double axisX = RandomUtils.RandomGaussian(0.0, 1.0);
			double axisY = RandomUtils.RandomGaussian(0.0, 1.0);
			double axisZ = RandomUtils.RandomGaussian(0.0, 1.0);
			Vector3d axis = new Vector3d(axisX, axisY, axisZ).Normalized();
			Quaterniond perturb = Quaterniond.FromAngleAxis(new AngleAxisd(MathUtils.DegToRad(noiseDeg), axis));
			edge.Cam2FromCam1 = edge.Cam2FromCam1 with { Rotation = perturb * edge.Cam2FromCam1.Rotation };
			edge.NumMatches = kBaselineMatches;
			if (noiseDeg > kNoiseThresholdDeg)
			{
				edge.NumMatches /= 2; // Down-weight noisy edges.
			}
		}

		double Run(RotationAveragingReweighting reweighting)
		{
			Reconstruction reconstruction = data.Reconstruction.Clone();
			RotationEstimatorOptions options = CreateRATestOptions(useGravity: false);
			options.Reweighting = reweighting;
			options.RandomSeed = 0; // Deterministic solve.
			options.MaxRotationErrorDeg = 0; // Disable post-solve edge filtering so only the solver reweighting differs.
			RotationAveraging.RunRotationAveraging(options, CopyPoseGraph(data.PoseGraph), reconstruction, data.PosePriors);
			return MeanRelativeRotationErrorDeg(data.GtReconstruction, reconstruction);
		}

		double errorUnweighted = Run(RotationAveragingReweighting.Uniform);
		double errorWeighted = Run(RotationAveragingReweighting.InlierMatchCount);

		await Assert.That(errorWeighted).IsLessThan(errorUnweighted);
	}

	[Test]
	public async Task RotationAveraging_WithoutNoiseWithNonTrivialKnownRig()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: true));
		double maxError = RunAndVerifyRotationAveraging(data, [true, false]);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	[Test]
	public async Task RotationAveraging_WithoutNoiseWithNonTrivialUnknownRig()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: true));
		ResetSensorsFromRig(data.Reconstruction);

		// For unknown rigs, it is not supported to use gravity.
		double maxError = RunAndVerifyRotationAveraging(data, [false]);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	[Test]
	public async Task RotationAveraging_WithNoiseAndOutliers()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(NoisyOptions(numCamerasPerRig: 1), NoiseOptions());
		double maxError = RunAndVerifyRotationAveraging(data, [true, false]);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(3.0));
	}

	[Test]
	public async Task RotationAveraging_WithNoiseAndOutliersWithNonTrivialKnownRigs()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(NoisyOptions(numCamerasPerRig: 2), NoiseOptions());
		double maxError = RunAndVerifyRotationAveraging(data, [true, false]);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(2.0));
	}

	[Test]
	public async Task RotationAveraging_DeterministicRandomSeed()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5, priorGravity: false));
		RotationEstimatorOptions options = CreateRATestOptions();
		options.RandomSeed = 42;

		// Run twice with the same seed and verify identical results.
		Reconstruction reconstruction1 = data.Reconstruction.Clone();
		bool success1 = RotationAveraging.RunRotationAveraging(
			options, CopyPoseGraph(data.PoseGraph), reconstruction1, data.PosePriors);
		Reconstruction reconstruction2 = data.Reconstruction.Clone();
		bool success2 = RotationAveraging.RunRotationAveraging(
			options, CopyPoseGraph(data.PoseGraph), reconstruction2, data.PosePriors);

		// In the presence of optimizations like FMA, q.angularDistance(q) can be near-zero
		// instead of zero, so check equality explicitly instead of with ExpectEqualRotations.
		List<uint> regImageIds = reconstruction1.RegImageIds();
		bool identical = regImageIds.All(imageId =>
			reconstruction1.Image(imageId).CamFromWorld().Rotation.Coeffs
				.Equals(reconstruction2.Image(imageId).CamFromWorld().Rotation.Coeffs));

		await Assert.That(success1).IsTrue();
		await Assert.That(success2).IsTrue();
		await Assert.That(identical).IsTrue();
	}

	[Test]
	public async Task RotationAveraging_RidgeRegularizationDoesNotBiasSolution()
	{
		// Use a noisy multi-rig setup to make the solution non-trivial and the
		// regularization's effect non-degenerate.
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(NoisyOptions(numCamerasPerRig: 1), NoiseOptions());
		RotationEstimatorOptions options = CreateRATestOptions(useGravity: true);
		options.RandomSeed = 42;

		// Run once with no regularization.
		Reconstruction reconstructionNoRidge = data.Reconstruction.Clone();
		options.RidgeRegularization = 0;
		bool noRidgeSolved = RotationAveraging.RunRotationAveraging(
			options, CopyPoseGraph(data.PoseGraph), reconstructionNoRidge, data.PosePriors);

		// Run again with the same default ridge that the global mapper uses. The option must
		// flow through L1 and IRLS without biasing the solution.
		Reconstruction reconstructionRidge = data.Reconstruction.Clone();
		options.RidgeRegularization = 1e-9;
		bool ridgeSolved = RotationAveraging.RunRotationAveraging(
			options, CopyPoseGraph(data.PoseGraph), reconstructionRidge, data.PosePriors);

		await Assert.That(noRidgeSolved).IsTrue();
		await Assert.That(ridgeSolved).IsTrue();

		// The two solutions should be effectively identical since 1e-9 is far below any
		// meaningful residual scale in the optimization.
		double maxError = MaxRelativeRotationError(reconstructionNoRidge, reconstructionRidge);
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-12));
	}

	[Test]
	public async Task RotationAveraging_EmptyPoseGraph()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 3,
			NumPoints3D = 20,
			TwoViewGeometryHasRelativePose = true,
		});

		// Invalidate all edges so connected components are empty.
		foreach (PoseGraph.Edge edge in data.PoseGraph.Edges.Values)
		{
			edge.Valid = false;
		}

		bool success = RotationAveraging.RunRotationAveraging(
			CreateRATestOptions(), data.PoseGraph, data.Reconstruction, data.PosePriors);
		await Assert.That(success).IsFalse();
	}

	[Test]
	public async Task RotationAveraging_MultiImageRigFrameDeregisterDoesNotCrashOnSecondVisit()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 2,
			NumFramesPerRig = 4,
			NumPoints3D = 50,
			TwoViewGeometryHasRelativePose = true,
		});

		List<uint> frameIds = [.. data.Reconstruction.Frames.Keys.Order()];
		await Assert.That(frameIds.Count).IsEqualTo(4);
		uint isolatedFrameId = frameIds[^1];

		// 1. Collect every image_id that belongs to the isolated frame.
		var isolatedImageIds = new HashSet<uint>();
		foreach (DataId dataId in data.Reconstruction.Frame(isolatedFrameId).ImageIds())
		{
			isolatedImageIds.Add((uint)dataId.Id);
		}

		// 2. Strip every pose-graph edge touching the isolated frame. After this the frame is
		//    unreachable from any other frame in the pose-graph CC.
		var edgesToRemove = new List<(uint, uint)>();
		foreach (ulong pairId in data.PoseGraph.Edges.Keys)
		{
			(uint id1, uint id2) = Types.PairIdToImagePair(pairId);
			if (isolatedImageIds.Contains(id1) || isolatedImageIds.Contains(id2))
			{
				edgesToRemove.Add((id1, id2));
			}
		}

		await Assert.That(edgesToRemove).IsNotEmpty();
		foreach ((uint id1, uint id2) in edgesToRemove)
		{
			data.PoseGraph.DeleteEdge(id1, id2);
		}

		// 3. Pre-register the isolated frame with its GT pose. This puts it into
		//    reg_frame_ids_ even though no edges touch it.
		await Assert.That(data.GtReconstruction.Frame(isolatedFrameId).HasPose).IsTrue();
		data.Reconstruction.Frame(isolatedFrameId).SetRigFromWorld(data.GtReconstruction.Frame(isolatedFrameId).RigFromWorld());
		data.Reconstruction.RegisterFrame(isolatedFrameId);

		RotationEstimatorOptions options = CreateRATestOptions();
		options.MaxRotationErrorDeg = 1.0;

		// The awaits above may have moved this test to another thread (thread-local PRNG).
		RandomUtils.SetPRNGSeed(0);
		bool success = RotationAveraging.RunRotationAveraging(options, data.PoseGraph, data.Reconstruction, data.PosePriors);
		await Assert.That(success).IsTrue();

		// Post-condition: the isolated frame was deregistered cleanly. The other frames remain
		// registered with poses recovered by RA.
		await Assert.That(data.Reconstruction.Frame(isolatedFrameId).HasPose).IsFalse();
		for (int i = 0; i + 1 < frameIds.Count; ++i)
		{
			await Assert.That(data.Reconstruction.Frame(frameIds[i]).HasPose).IsTrue();
		}
	}

	[Test]
	public async Task RotationAveraging_GravityWithUnknownRigSensorsReturnsFalse()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: true));
		ResetSensorsFromRig(data.Reconstruction);

		// With gravity enabled and unknown rig sensors, EstimateRotations should fail inside
		// RunRotationAveraging because AllSensorsFromRigKnown returns false. However,
		// RunRotationAveraging takes the HasUnknownCamsFromRig path which creates an expanded
		// reconstruction (singleton rigs) that avoids the AllSensorsFromRigKnown check. To
		// directly hit the AllSensorsFromRigKnown check, we use RotationEstimator directly.
		RotationEstimatorOptions options = CreateRATestOptions(useGravity: true);
		HashSet<uint> activeImageIds = [.. data.Reconstruction.Images.Keys];

		var estimator = new RotationEstimator(options);
		bool success = estimator.EstimateRotations(data.PoseGraph, data.PosePriors, activeImageIds, data.Reconstruction);
		await Assert.That(success).IsFalse();
	}

	// Covers: InitializeRigRotationsFromImages standalone with multi-camera rig to exercise
	// cam_from_rig estimation and rig_from_world averaging.
	[Test]
	public async Task RotationAveraging_InitializeSensorFromRigUsingCamsFromWorld()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: false));

		// Build cams_from_world from the ground truth.
		Dictionary<uint, Rigid3d> camsFromWorld = GtCamsFromWorld(data);

		ResetSensorsFromRig(data.Reconstruction);

		bool success = RotationAveraging.InitializeRigRotationsFromImages(camsFromWorld, data.Reconstruction);

		double maxError = 0;
		foreach ((uint rigId, Rig rig) in data.Reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				maxError = Math.Max(maxError, sensorFromRig!.Value.Rotation.AngularDistance(
					data.GtReconstruction.Rig(rigId).SensorFromRig(sensorId).Rotation));
			}
		}

		await Assert.That(success).IsTrue();
		await Assert.That(maxError).IsLessThan(1e-6);
	}

	// When a sensor_from_rig is already fully calibrated (valid rotation AND translation),
	// InitializeRigRotationsFromImages must preserve it rather than resetting the translation
	// to NaN.
	[Test]
	public async Task RotationAveraging_InitializeSensorFromRigPreservesCalibratedRig()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: false));
		Dictionary<uint, Rigid3d> camsFromWorld = GtCamsFromWorld(data);

		// Snapshot the (already-calibrated) rig BEFORE initialization.
		Dictionary<(uint, SensorId), Rigid3d> snapshot = SnapshotSensorsFromRig(data.Reconstruction);
		await Assert.That(snapshot.Count).IsGreaterThan(0);

		bool success = RotationAveraging.InitializeRigRotationsFromImages(camsFromWorld, data.Reconstruction);

		await Assert.That(success).IsTrue();
		await Assert.That(SensorsFromRigMatch(data.Reconstruction, snapshot)).IsTrue();
	}

	[Test]
	public async Task RotationAveraging_RefineSensorFromRigFalsePreservesRig()
	{
		// A non-trivial multi-camera rig so both rotation AND translation are non-zero.
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SmallRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4, priorGravity: true));

		// Snapshot the rig BEFORE RA so we can compare element-wise.
		Dictionary<(uint, SensorId), Rigid3d> snapshot = SnapshotSensorsFromRig(data.Reconstruction);

		// Sanity check: at least one sensor should have a non-zero translation so the test
		// would actually catch the old "reset to zero" behaviour.
		await Assert.That(snapshot.Count).IsGreaterThan(0);

		// Run RA with refine_sensor_from_rig=false.
		RotationEstimatorOptions options = CreateRATestOptions(useGravity: true);
		options.RefineSensorFromRig = false;
		// The awaits above may have moved this test to another thread (thread-local PRNG).
		RandomUtils.SetPRNGSeed(0);
		bool success = RotationAveraging.RunRotationAveraging(options, data.PoseGraph, data.Reconstruction, data.PosePriors);

		// Every sensor_from_rig must match the snapshot exactly.
		await Assert.That(success).IsTrue();
		await Assert.That(SensorsFromRigMatch(data.Reconstruction, snapshot)).IsTrue();
	}

	private static Dictionary<uint, Rigid3d> GtCamsFromWorld(TestData data)
	{
		var camsFromWorld = new Dictionary<uint, Rigid3d>();
		foreach ((uint imageId, Image image) in data.GtReconstruction.Images)
		{
			if (image.HasPose)
			{
				camsFromWorld[imageId] = image.CamFromWorld();
			}
		}

		return camsFromWorld;
	}
}
