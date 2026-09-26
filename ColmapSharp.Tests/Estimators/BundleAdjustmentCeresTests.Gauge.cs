// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeresTests.Gauge: the FixGauge* cases of
// colmap/estimators/bundle_adjustment_ceres_test.cc 1:1 (Estimators/BundleAdjustmentCeres.Gauge.cs).
// BundleAdjustmentCeresTests.cs holds the rest of the file and the helpers. The expected
// num_effective_parameters_reduced counts are exact (Tier A problem layout).

using ColmapSharp.Estimators;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class BundleAdjustmentCeresTests
{
	private static async Task ExpectValidSolve(
		BundleAdjustmentOptions options, BundleAdjustmentConfig config, Reconstruction reconstruction, int numEffectiveParametersReduced)
	{
		BundleAdjustmentSummary summary =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction).Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(numEffectiveParametersReduced);
	}

	[Test]
	public async Task DefaultBundleAdjuster_FixGaugeWithThreePoints()
	{
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(2, 1, 1, 100);

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);

		await ExpectValidSolve(new BundleAdjustmentOptions(), config, reconstruction, 316);

		config.FixGauge(BundleAdjustmentGauge.ThreePoints);
		await ExpectValidSolve(new BundleAdjustmentOptions(), config, reconstruction, 307);

		config.AddConstantPoint(1);
		await ExpectValidSolve(new BundleAdjustmentOptions(), config, reconstruction, 307);

		config.AddConstantPoint(2);
		config.AddConstantPoint(3);
		await ExpectValidSolve(new BundleAdjustmentOptions(), config, reconstruction, 307);

		config.AddConstantPoint(4);
		await ExpectValidSolve(new BundleAdjustmentOptions(), config, reconstruction, 304);
	}

	[Test]
	public async Task DefaultBundleAdjuster_FixGaugeWithTwoCamsFromWorld()
	{
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(2, 2, 1, 100);

		var options = new BundleAdjustmentOptions();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		config.AddImage(4);

		options.RefineRigFromWorld = false;
		await ExpectValidSolve(options, config, reconstruction, 320);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 332);

		options.RefineRigFromWorld = false;
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		await ExpectValidSolve(options, config, reconstruction, 320);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 325);

		config.SetConstantRigFromWorldPose(1);
		await ExpectValidSolve(options, config, reconstruction, 325);

		config.SetConstantRigFromWorldPose(2);
		await ExpectValidSolve(options, config, reconstruction, 320);
	}

	[Test]
	public async Task DefaultBundleAdjuster_FixGaugeWithTwoCamsFromWorldFixSensorFromRig()
	{
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(2, 2, 1, 100);

		var options = new BundleAdjustmentOptions();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		config.AddImage(4);

		options.RefineRigFromWorld = false;
		options.RefineSensorFromRig = false;
		await ExpectValidSolve(options, config, reconstruction, 308);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 320);

		options.RefineRigFromWorld = false;
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		await ExpectValidSolve(options, config, reconstruction, 308);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 313);

		config.SetConstantRigFromWorldPose(1);
		await ExpectValidSolve(options, config, reconstruction, 313);

		config.SetConstantRigFromWorldPose(2);
		await ExpectValidSolve(options, config, reconstruction, 308);
	}

	[Test]
	public async Task DefaultBundleAdjuster_FixGaugeWithTwoCamsFromWorldNoReferenceSensor()
	{
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(2, 2, 1, 100);

		// Delete observations from the two reference images.
		Check.That(reconstruction.Image(1).IsRefInFrame);
		Check.That(reconstruction.Image(3).IsRefInFrame);
		foreach (uint imageId in new uint[] { 1, 3 })
		{
			for (uint i = 0; i < reconstruction.Image(imageId).NumPoints2D; ++i)
			{
				if (reconstruction.Image(imageId).Points2D[(int)i].HasPoint3D)
				{
					reconstruction.DeleteObservation(imageId, i);
				}
			}
		}

		// Only add two non-reference images.
		var options = new BundleAdjustmentOptions();
		var config = new BundleAdjustmentConfig();
		config.AddImage(2);
		config.AddImage(4);

		// refine_sensor_from_rig should have no effect when there are no reference sensors
		options.RefineRigFromWorld = true;
		options.RefineSensorFromRig = true;
		await ExpectValidSolve(options, config, reconstruction, 316);

		options.RefineRigFromWorld = false;
		await ExpectValidSolve(options, config, reconstruction, 304);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 316);

		options.RefineRigFromWorld = false;
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		await ExpectValidSolve(options, config, reconstruction, 304);

		options.RefineSensorFromRig = false;
		await ExpectValidSolve(options, config, reconstruction, 304);

		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 309);

		config.SetConstantRigFromWorldPose(1);
		await ExpectValidSolve(options, config, reconstruction, 309);
		options.RefineRigFromWorld = false;
		await ExpectValidSolve(options, config, reconstruction, 304);

		config.SetConstantRigFromWorldPose(2);
		options.RefineRigFromWorld = true;
		await ExpectValidSolve(options, config, reconstruction, 304);
		options.RefineRigFromWorld = false;
		await ExpectValidSolve(options, config, reconstruction, 304);
	}

	[Test]
	public async Task DefaultBundleAdjuster_FixGaugeWithTwoCamsFromWorldFallback()
	{
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(1, 2, 1, 100);

		var options = new BundleAdjustmentOptions();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);

		// The current implementation needs two reference cameras in different frames to fix
		// the gauge. If there are none, it falls back to fixing the gauge with three points.
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		BundleAdjustmentSummary summary =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction).Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);
		await Assert.That(GetCeresSummary(summary).NumEffectiveParameters).IsEqualTo(316);
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(307);
	}
}
