// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ObservationManagerTests.Merge: C#-only tests for ObservationManager.MergeAndFilterReconstructions
// (ColmapSharp/Sfm/ObservationManager.Filter.cs). COLMAP's observation_manager_test.cc has no
// case for it, so these do not stand in for a ported test. They reuse the merge scene of
// AlignmentTests.Alignment_MergeReconstructions: the source drops rig 1, the target rig 2,
// and the two share rig 3's frames.
//
// Tier A: the merge on an exact synthetic scene and the filter's counts are deterministic.

using ColmapSharp.Estimators;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;
using ColmapSharp.Tests.Estimators;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sfm;

public partial class ObservationManagerTests
{
	private const double MergeMaxReprojError = 1e-4;

	// The source and target of the merge scene, with the target's first 3D point moved far
	// off its observations so that every one of its observations fails the reprojection test.
	private static (Reconstruction Src, Reconstruction Tgt) MergeSceneWithOneCorruptTargetPoint()
	{
		// Reseeded on every call, not only at test start: a test that compares two merges
		// needs both built from the same scene.
		RandomUtils.SetPRNGSeed(0);
		Reconstruction srcReconstruction = AlignmentTests.GenerateReconstructionForMerge();
		Reconstruction tgtReconstruction = srcReconstruction.Clone();
		AlignmentTests.RemoveRigFrames(srcReconstruction, 1);
		AlignmentTests.RemoveRigFrames(tgtReconstruction, 2);
		srcReconstruction.TearDown();
		tgtReconstruction.TearDown();

		Point3D corruptPoint = tgtReconstruction.Point3D(tgtReconstruction.Point3DIds().Min());
		corruptPoint.Xyz += new Vector3d(100, 100, 100);
		return (srcReconstruction, tgtReconstruction);
	}

	// C#-only: after MergeAndFilterReconstructions no observation of the target exceeds the
	// reprojection threshold, while a plain MergeReconstructions of the same scene leaves the
	// corrupt point's observations in place.
	[Test]
	public async Task CSharpOnly_MergeAndFilterReconstructions_FiltersMergedTarget()
	{
		(Reconstruction srcReconstruction, Reconstruction tgtReconstruction) = MergeSceneWithOneCorruptTargetPoint();
		(Reconstruction plainSrc, Reconstruction plainTgt) = MergeSceneWithOneCorruptTargetPoint();

		bool merged = ObservationManager.MergeAndFilterReconstructions(MergeMaxReprojError, srcReconstruction, tgtReconstruction);
		bool plainMerged = Alignment.MergeReconstructions(MergeMaxReprojError, plainSrc, plainTgt);

		await Assert.That(merged).IsTrue();
		await Assert.That(plainMerged).IsTrue();
		await Assert.That(tgtReconstruction.NumRegFrames).IsEqualTo(plainTgt.NumRegFrames);
		await Assert.That(tgtReconstruction.ComputeNumObservations()).IsLessThan(plainTgt.ComputeNumObservations());

		int filteredAfterPlainMerge = new ObservationManager(plainTgt).FilterAllPoints3D(MergeMaxReprojError, 0);
		int filteredAfterMergeAndFilter = new ObservationManager(tgtReconstruction).FilterAllPoints3D(MergeMaxReprojError, 0);
		await Assert.That(filteredAfterPlainMerge).IsGreaterThan(0);
		await Assert.That(filteredAfterMergeAndFilter).IsEqualTo(0);
		await Assert.That(tgtReconstruction.ComputeNumObservations()).IsEqualTo(plainTgt.ComputeNumObservations());
	}

	// C#-only: when the merge fails (no shared registered images), the target is untouched.
	[Test]
	public async Task CSharpOnly_MergeAndFilterReconstructions_FailedMergeLeavesTarget()
	{
		Reconstruction srcReconstruction = AlignmentTests.GenerateReconstructionForMerge();
		Reconstruction tgtReconstruction = srcReconstruction.Clone();
		AlignmentTests.RemoveRigFrames(srcReconstruction, 1);
		AlignmentTests.RemoveRigFrames(srcReconstruction, 3);
		AlignmentTests.RemoveRigFrames(tgtReconstruction, 2);
		AlignmentTests.RemoveRigFrames(tgtReconstruction, 3);
		srcReconstruction.TearDown();
		tgtReconstruction.TearDown();
		Reconstruction origTgtReconstruction = tgtReconstruction.Clone();

		bool merged = ObservationManager.MergeAndFilterReconstructions(MergeMaxReprojError, srcReconstruction, tgtReconstruction);

		await Assert.That(merged).IsFalse();
		await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(tgtReconstruction, origTgtReconstruction)).IsNull();
	}
}
