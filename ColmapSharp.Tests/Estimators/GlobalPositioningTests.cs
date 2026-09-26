// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPositioningTests: colmap/estimators/global_positioning_test.cc 1:1 for
// Estimators/GlobalPositioning.cs. Test names are <Suite>_<Test>. Tier C.
//
// Ported: GlobalPositioning.RefineSensorFromRigFalsePreservesRig.
// Waiting: GlobalPositioning.{Nominal, MultiCameraRig}. They assert through the
// ReconstructionNear matcher, which needs AlignReconstructionsViaProjCenters and
// ComputeImageAlignmentError (estimators/alignment, not ported yet; ReconstructionMatchers.cs).
// Until then, the C#-only tests at the bottom run the same two setups and check the same
// bounds (rotation 0.1 deg, projection center 0.5 after a similarity alignment) through a
// least-squares Sim3 of the projection centers. They do not stand in for the ported cases.
//
// Translation notes: the SQLite test database is an InMemoryDatabase. COLMAP's gtest_main
// reseeds the PRNG with 0 at every test start; Setup does the same, before any await.

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
		RandomUtils.SetPRNGSeed(0);

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

	// C#-only: the GlobalPositioning.Nominal setup, checked through a least-squares Sim3 of
	// the projection centers until ReconstructionNear is ported (file header).
	[Test]
	public async Task CSharpOnly_GlobalPositioning_NominalAlignsToGroundTruth() =>
		await RunAndCheckAlignment(1, 1, 10);

	// C#-only: the GlobalPositioning.MultiCameraRig setup, checked the same way.
	[Test]
	public async Task CSharpOnly_GlobalPositioning_MultiCameraRigAlignsToGroundTruth() =>
		await RunAndCheckAlignment(2, 3, 5);

	private static async Task RunAndCheckAlignment(int numRigs, int numCamerasPerRig, int numFramesPerRig)
	{
		(Reconstruction gt, Reconstruction reconstruction, PoseGraph poseGraph) =
			Setup(numRigs, numCamerasPerRig, numFramesPerRig);

		await Assert.That(GlobalPositioning.RunGlobalPositioning(TestOptions(), poseGraph, reconstruction)).IsTrue();

		List<uint> imageIds = gt.RegImageIds();
		var src = new Vector3d[imageIds.Count];
		var tgt = new Vector3d[imageIds.Count];
		for (int i = 0; i < imageIds.Count; ++i)
		{
			src[i] = reconstruction.Image(imageIds[i]).ProjectionCenter();
			tgt[i] = gt.Image(imageIds[i]).ProjectionCenter();
		}

		var gtFromEstimate = new Sim3d();
		await Assert.That(SimilarityTransform.EstimateSim3d(src, tgt, ref gtFromEstimate)).IsTrue();

		double maxRotationErrorRad = MathUtils.DegToRad(0.1);
		for (int i = 0; i < imageIds.Count; ++i)
		{
			Rigid3d gtCamFromWorld = gt.Image(imageIds[i]).CamFromWorld();
			Rigid3d camFromWorld = reconstruction.Image(imageIds[i]).CamFromWorld();
			Quaterniond alignedRotation = camFromWorld.Rotation * gtFromEstimate.Rotation.Inverse();
			await Assert.That(alignedRotation.AngularDistance(gtCamFromWorld.Rotation)).IsLessThanOrEqualTo(maxRotationErrorRad);
			await Assert.That((gtFromEstimate * src[i] - tgt[i]).Norm).IsLessThanOrEqualTo(0.5);
		}
	}
}
