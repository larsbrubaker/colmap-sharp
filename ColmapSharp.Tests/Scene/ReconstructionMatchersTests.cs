// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionMatchersTests: colmap/scene/reconstruction_matchers_test.cc (suite
// "Reconstruction"), testing the test-side ReconstructionMatchers.cs. The gmock StrictMock
// half of Eq checks that the matcher accepts each argument the mocked method is called with
// (r1 matches ReconstructionEq(r1), r2 matches ReconstructionEq(r2)); with predicates that is
// the matcher applied to (r1, r1) and (r2, r2), as in Rigid3dMatchersTests; Near likewise.
// ReconstructionNear runs LO-RANSAC on the thread PRNG, so Near evaluates every matcher
// before its first await.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.ReconstructionMatchers;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionMatchersTests
{
	[Test]
	public async Task Reconstruction_Eq()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction1 = new Reconstruction();
		var reconstruction2 = new Reconstruction();
		bool emptyEqual = ReconstructionEq(reconstruction1, reconstruction2);

		var syntheticDatasetOptions = new SyntheticDatasetOptions { NumPoints2DWithoutPoint3D = 0 };
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction1);

		reconstruction2 = reconstruction1.Clone();
		bool copyEqual = ReconstructionEq(reconstruction1, reconstruction2);

		reconstruction2 = reconstruction1.Clone();
		Rigid3d rigFromWorld = reconstruction2.Frame(1).RigFromWorld();
		Vector3d translation = rigFromWorld.Translation;
		reconstruction2.Frame(1).SetRigFromWorld(rigFromWorld with { Translation = new Vector3d(translation.X + 0.1, translation.Y, translation.Z) });
		bool poseChangedEqual = ReconstructionEq(reconstruction1, reconstruction2);

		reconstruction2 = reconstruction1.Clone();
		reconstruction2.DeleteObservation(1, 0);
		bool observationDeletedEqual = ReconstructionEq(reconstruction1, reconstruction2);

		await Assert.That(emptyEqual).IsTrue();
		await Assert.That(copyEqual).IsTrue();
		await Assert.That(poseChangedEqual).IsFalse();
		await Assert.That(observationDeletedEqual).IsFalse();

		// The mock: TestMethod(reconstruction1) once, TestMethod(reconstruction2) twice.
		await Assert.That(ReconstructionEq(reconstruction1, reconstruction1)).IsTrue();
		await Assert.That(ReconstructionEq(reconstruction2, reconstruction2)).IsTrue();
		await Assert.That(ReconstructionEq(reconstruction2, reconstruction2)).IsTrue();
		await Assert.That(ReconstructionEq(reconstruction2, reconstruction1)).IsFalse();
	}

	[Test]
	public async Task Reconstruction_Near()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction1 = new Reconstruction();
		var reconstruction2 = new Reconstruction();
		bool emptyNearUnaligned = ReconstructionNear(
			reconstruction1,
			reconstruction2,
			maxRotationErrorDeg: 0,
			maxProjCenterError: 0,
			maxScaleError: null,
			numObsTolerance: 0,
			align: false);
		bool emptyNearAligned = ReconstructionNear(
			reconstruction1,
			reconstruction2,
			maxRotationErrorDeg: 0,
			maxProjCenterError: 0,
			maxScaleError: null,
			numObsTolerance: 0,
			align: true);

		var syntheticDatasetOptions = new SyntheticDatasetOptions { NumPoints2DWithoutPoint3D = 0 };
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction1);

		reconstruction2 = reconstruction1.Clone();
		bool copyNear = ReconstructionNear(reconstruction1, reconstruction2);

		reconstruction2 = reconstruction1.Clone();
		Rigid3d rigFromWorld = reconstruction2.Frame(1).RigFromWorld();
		Vector3d translation = rigFromWorld.Translation;
		reconstruction2.Frame(1).SetRigFromWorld(rigFromWorld with { Translation = new Vector3d(translation.X + 0.1, translation.Y, translation.Z) });
		bool translationChangedNear = ReconstructionNear(reconstruction1, reconstruction2);

		reconstruction2 = reconstruction1.Clone();
		rigFromWorld = reconstruction2.Frame(1).RigFromWorld();
		reconstruction2.Frame(1).SetRigFromWorld(rigFromWorld with
		{
			Rotation = rigFromWorld.Rotation * Quaterniond.FromAngleAxis(new AngleAxisd(0.1, Vector3d.UnitX)),
		});
		bool rotationChangedNear = ReconstructionNear(reconstruction1, reconstruction2);

		reconstruction2 = reconstruction1.Clone();
		reconstruction2.DeleteObservation(1, 0);
		bool observationDeletedNear = ReconstructionNear(reconstruction1, reconstruction2);

		// The mock: TestMethod(reconstruction1) once, TestMethod(reconstruction2) twice.
		bool[] mockCalls =
		[
			ReconstructionNear(reconstruction1, reconstruction1),
			ReconstructionNear(reconstruction2, reconstruction2),
			ReconstructionNear(reconstruction2, reconstruction2),
		];

		await Assert.That(emptyNearUnaligned).IsTrue();
		await Assert.That(emptyNearAligned).IsFalse();
		await Assert.That(copyNear).IsTrue();
		await Assert.That(translationChangedNear).IsFalse();
		await Assert.That(rotationChangedNear).IsFalse();
		await Assert.That(observationDeletedNear).IsFalse();
		foreach (bool mockCallMatched in mockCalls)
		{
			await Assert.That(mockCallMatched).IsTrue();
		}
	}
}
