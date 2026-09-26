// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentControllerTests: colmap/controllers/bundle_adjustment_test.cc ported 1:1,
// testing ColmapSharp/Controllers/BundleAdjustmentController.cs.
//
// Tier C (outcome): ReconstructionNear against the synthetic ground truth with COLMAP's
// bounds; ReconstructionEq when the run stops before optimizing.
//
// Translation notes: a default OptionManager is `new BundleAdjustmentOptions()` (the
// controller takes OptionManager::bundle_adjustment, docs/CPP_DIVERGENCES.md entry 67);
// std::make_shared<Reconstruction>(gt) is gt.Clone(). The PRNG is seeded with 0 before
// every test (PrngTestIsolation, like gtest_main), and all work happens before the first
// await (the PRNG is per thread).

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class BundleAdjustmentControllerTests
{
	[Test]
	public async Task BundleAdjustmentController_EmptyReconstruction()
	{
		var reconstruction = new Reconstruction();

		var controller = new BundleAdjustmentController(new BundleAdjustmentOptions(), reconstruction);
		await Assert.That(controller.Run).ThrowsNothing();

		await Assert.That(reconstruction.NumRegImages).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
	}

	[Test]
	public async Task BundleAdjustmentController_StopsBeforeOptimization()
	{
		var gtReconstruction = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 3, NumPoints3D = 50 };
		Synthetic.SynthesizeDataset(syntheticOptions, gtReconstruction);

		Reconstruction reconstruction = gtReconstruction.Clone();
		var controller = new BundleAdjustmentController(new BundleAdjustmentOptions(), reconstruction);
		bool stopChecked = false;
		controller.SetCheckIfStoppedFunc(() =>
		{
			stopChecked = true;
			return true;
		});
		controller.Run();

		string? eq = ReconstructionMatchers.ExplainReconstructionEq(reconstruction, gtReconstruction);
		await Assert.That(stopChecked).IsTrue();
		await Assert.That(eq).IsNull();
	}

	[Test]
	public async Task BundleAdjustmentController_Reconstruction()
	{
		var gtReconstruction = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 2, NumFramesPerRig = 3, NumPoints3D = 100 };
		Synthetic.SynthesizeDataset(syntheticOptions, gtReconstruction);

		Reconstruction reconstruction = gtReconstruction.Clone();

		var noiseOptions = new SyntheticNoiseOptions
		{
			Point2DStddev = 0.1,
			Point3DStddev = 0.1,
			RigFromWorldRotationStddev = 0.1,
			RigFromWorldTranslationStddev = 0.1,
		};
		Synthetic.SynthesizeNoise(noiseOptions, reconstruction);

		var controller = new BundleAdjustmentController(new BundleAdjustmentOptions(), reconstruction);
		controller.Run();

		string? near = ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction, reconstruction, maxRotationErrorDeg: 0.1, maxProjCenterError: 0.1, maxScaleError: null, numObsTolerance: 0.0);
		await Assert.That(near).IsNull();
	}
}
