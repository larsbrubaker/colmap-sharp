// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPositioningTests: colmap/estimators/global_positioning_test.cc 1:1 for
// Estimators/GlobalPositioning.cs. Test names are <Suite>_<Test>. Tier C.
//
// Ported: GlobalPositioning.{Nominal, MultiCameraRig, RefineSensorFromRigFalsePreservesRig}.
//
// Translation notes: the SQLite test database is an InMemoryDatabase. PrngTestIsolation
// seeds the PRNG with 0 before every test, as COLMAP's gtest_main does; Setup draws
// before any await.

using ColmapSharp.Estimators;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public class GlobalPositioningTests
{
	// The shared setup of every case: a synthetic dataset, its pose graph, and a copy of the
	// ground truth that keeps only the frame rotations (translations reset).
	private static (Reconstruction Gt, Reconstruction Reconstruction, PoseGraph PoseGraph) Setup(
		int numRigs, int numCamerasPerRig, int numFramesPerRig)
	{
		var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = numRigs,
			NumCamerasPerRig = numCamerasPerRig,
			NumFramesPerRig = numFramesPerRig,
			NumPoints3D = 200,
			TwoViewGeometryHasRelativePose = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, gtReconstruction, database);

		DatabaseCache databaseCache = DatabaseCache.Create(database, new DatabaseCache.Options());
		var poseGraph = new PoseGraph();
		poseGraph.Load(databaseCache.CorrespondenceGraph);

		// Copy GT reconstruction and keep only rotations (reset translations).
		Reconstruction reconstruction = gtReconstruction.Clone();
		foreach (uint frameId in reconstruction.Frames.Keys.ToList())
		{
			Frame frame = reconstruction.Frame(frameId);
			frame.SetRigFromWorld(new Rigid3d(frame.RigFromWorld().Rotation, Vector3d.Zero));
		}

		return (gtReconstruction, reconstruction, poseGraph);
	}

	private static GlobalPositionerOptions TestOptions() => new() { RandomSeed = 42 };

	[Test]
	public async Task GlobalPositioning_RefineSensorFromRigFalsePreservesRig()
	{
		// Multi-camera rig so the sensor offsets are non-trivial - both rotation and
		// translation must round-trip. The rig calibration is left as-is.
		(_, Reconstruction reconstruction, PoseGraph poseGraph) = Setup(2, 3, 5);

		// Snapshot the rig BEFORE GP.
		var snapshot = new Dictionary<(uint RigId, SensorId SensorId), Rigid3d>();
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				await Assert.That(sensorFromRig.HasValue).IsTrue();
				snapshot[(rigId, sensorId)] = sensorFromRig!.Value;
			}
		}

		await Assert.That(snapshot.Count).IsGreaterThan(0);

		GlobalPositionerOptions options = TestOptions();
		options.RefineSensorFromRig = false;

		await Assert.That(GlobalPositioning.RunGlobalPositioning(options, poseGraph, reconstruction)).IsTrue();

		// Every sensor_from_rig must match the snapshot exactly.
		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRigAfter) in rig.NonRefSensors)
			{
				await Assert.That(sensorFromRigAfter.HasValue).IsTrue();
				Rigid3d sensorFromRigBefore = snapshot[(rigId, sensorId)];
				await Assert.That(Rigid3dEq(sensorFromRigAfter!.Value, sensorFromRigBefore)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GlobalPositioning_Nominal() => await RunAndExpectNearGroundTruth(1, 1, 10);

	[Test]
	public async Task GlobalPositioning_MultiCameraRig() => await RunAndExpectNearGroundTruth(2, 3, 5);

	// The shared body of Nominal and MultiCameraRig: they differ only in the dataset size.
	private static async Task RunAndExpectNearGroundTruth(int numRigs, int numCamerasPerRig, int numFramesPerRig)
	{
		(Reconstruction gtReconstruction, Reconstruction reconstruction, PoseGraph poseGraph) =
			Setup(numRigs, numCamerasPerRig, numFramesPerRig);

		bool success = GlobalPositioning.RunGlobalPositioning(TestOptions(), poseGraph, reconstruction);
		await Assert.That(success).IsTrue();

		await Assert.That(ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg: 0.1,
			maxProjCenterError: 0.5,
			maxScaleError: null,
			numObsTolerance: 0.0)).IsNull();
	}
}
