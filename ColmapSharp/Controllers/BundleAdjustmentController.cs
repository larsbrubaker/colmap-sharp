// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentController: port of colmap/controllers/bundle_adjustment.h and .cc - one
// global bundle adjustment of every registered image of a reconstruction, with the gauge
// fixed by two cameras (Estimators/BundleAdjustment*.cs). Tests:
// ColmapSharp.Tests/Controllers/BundleAdjustmentControllerTests.cs (bundle_adjustment_test.cc
// 1:1).
//
// Tier C (outcome): the adjusted reconstruction is compared with the ground truth.
//
// Translation note: COLMAP's controller reads OptionManager::bundle_adjustment; the option
// manager (the CLI's option registry) is not ported, so the controller takes that
// BundleAdjustmentOptions directly (docs/CPP_DIVERGENCES.md, entry 67). A default
// OptionManager holds default BundleAdjustmentOptions, so `new BundleAdjustmentOptions()` is
// the same configuration.

using ColmapSharp.Estimators;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::BundleAdjustmentController: controls the global bundle adjustment procedure.</summary>
public sealed class BundleAdjustmentController : BaseController
{
	private readonly BundleAdjustmentOptions _options;
	private readonly Reconstruction _reconstruction;

	/// <summary>Adjusts <paramref name="reconstruction"/> in place with <paramref name="options"/> when run.</summary>
	public BundleAdjustmentController(BundleAdjustmentOptions options, Reconstruction reconstruction)
	{
		_options = Check.NotNull(options);
		_reconstruction = reconstruction;
	}

	/// <summary>Runs one global bundle adjustment (nothing when no frame is registered).</summary>
	public override void Run()
	{
		Check.NotNull(_reconstruction);

		if (_reconstruction.NumRegFrames == 0)
		{
			Log.Error("Need at least one registered frame.");
			return;
		}

		if (CheckIfStopped())
		{
			return;
		}

		// Avoid degeneracies in bundle adjustment.
		new ObservationManager(_reconstruction).FilterObservationsWithNegativeDepth();

		BundleAdjustmentOptions baOptions = _options.Clone();
		baOptions.CheckIfStopped = CheckIfStopped;

		// Configure bundle adjustment.
		var baConfig = new BundleAdjustmentConfig();
		foreach (uint imageId in _reconstruction.RegImageIds())
		{
			baConfig.AddImage(imageId);
		}

		// Fixing the gauge with two cameras leads to a more stable optimization with fewer
		// steps as compared to fixing three points.
		baConfig.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		// Run bundle adjustment.
		BundleAdjuster bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(baOptions, baConfig, _reconstruction);
		bundleAdjuster.Solve(CancellationToken);
		_reconstruction.UpdatePoint3DErrors();
	}
}
