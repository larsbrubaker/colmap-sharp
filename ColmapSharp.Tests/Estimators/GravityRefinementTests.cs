// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GravityRefinementTests: colmap/estimators/gravity_refinement_test.cc 1:1 for
// Estimators/GravityRefinement.cs. Test names are <Suite>_<Test>.
//
// Ported: GravityRefinement.{RefineGravity, RefineGravityWithNonTrivialRigs}.
// Tier C: COLMAP's own tolerance (1e-2 degrees).
//
// Translation notes: the SQLite test database is an InMemoryDatabase. PrngTestIsolation seeds
// the PRNG with 0 before every test, as COLMAP's gtest_main does; RunRefineGravity makes
// every draw before its first await (the PRNG is per thread).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class GravityRefinementTests
{
	private static (Reconstruction Reconstruction, PoseGraph PoseGraph) LoadReconstructionAndPoseGraph(Database database)
	{
		DatabaseCache databaseCache = DatabaseCache.Create(database, new DatabaseCache.Options());
		var reconstruction = new Reconstruction();
		reconstruction.Load(databaseCache);
		var poseGraph = new PoseGraph();
		poseGraph.Load(databaseCache.CorrespondenceGraph);
		return (reconstruction, poseGraph);
	}

	private static void SynthesizeGravityOutliers(List<PosePrior> posePriors, double outlierRatio = 0.0)
	{
		for (int i = 0; i < posePriors.Count; ++i)
		{
			PosePrior posePrior = posePriors[i];
			if (posePrior.HasGravity() && RandomUtils.RandomUniformReal(0.0, 1.0) < outlierRatio)
			{
				posePrior.Gravity = RandomEigen.RandomEigenVector3d().Normalized();
				posePriors[i] = posePrior;
			}
		}
	}

	private static async Task ExpectEqualGravity(
		Vector3d gravityInWorld, Reconstruction gt, List<PosePrior> posePriors, double maxGravityErrorDeg)
	{
		double maxGravityErrorRad = MathUtils.DegToRad(maxGravityErrorDeg);
		var imageToPosePrior = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in posePriors)
		{
			if (posePrior.CorrDataId.SensorId.Type == SensorType.Camera)
			{
				imageToPosePrior.TryAdd((uint)posePrior.CorrDataId.Id, posePrior);
			}
		}

		foreach (uint imageId in gt.RegImageIds())
		{
			Image image = gt.Image(imageId);
			if (!image.IsRefInFrame)
			{
				continue;
			}

			Vector3d gravityGt = gt.Image(imageId).CamFromWorld().Rotation * gravityInWorld;
			Vector3d gravityComputed = imageToPosePrior[imageId].Gravity;
			double gravityErrorRad = Triangulation.CalculateAngleBetweenVectors(gravityGt, gravityComputed);
			await Assert.That(gravityErrorRad).IsLessThan(maxGravityErrorRad);
		}
	}

	private static async Task RunRefineGravity(int numCamerasPerRig)
	{
		var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = numCamerasPerRig,
			NumFramesPerRig = 25,
			NumPoints3D = 100,
			PriorGravity = true,
			TwoViewGeometryHasRelativePose = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, gtReconstruction, database);

		(Reconstruction reconstruction, PoseGraph poseGraph) = LoadReconstructionAndPoseGraph(database);

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		SynthesizeGravityOutliers(posePriors, outlierRatio: 0.3);

		var optGravRefine = new GravityRefinerOptions();
		GravityRefinement.RunGravityRefinement(optGravRefine, poseGraph, reconstruction, posePriors);

		await ExpectEqualGravity(syntheticDatasetOptions.PriorGravityInWorld, gtReconstruction, posePriors, maxGravityErrorDeg: 1e-2);
	}

	[Test]
	public async Task GravityRefinement_RefineGravity() => await RunRefineGravity(numCamerasPerRig: 1);

	[Test]
	public async Task GravityRefinement_RefineGravityWithNonTrivialRigs() => await RunRefineGravity(numCamerasPerRig: 2);
}
