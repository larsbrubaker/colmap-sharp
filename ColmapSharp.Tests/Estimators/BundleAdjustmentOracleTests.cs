// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BundleAdjustmentOracleTests: C#-only oracle test (not a port of a *_test.cc) for
// DefaultBundleAdjuster (Estimators/BundleAdjustmentCeres*.cs) against pycolmap 4.2.0.
// Fixture: TestData/oracle/bundle_adjustment/ from oracle/bundle_adjustment.py: a noisy
// 16-image synthetic model (input/), pycolmap's solved model (solved/) and its solver summary.
//
// Tier C (outcome). The solver is a managed Ceres replacement, so the trajectory differs in
// the last bits; the bar is the same converged optimum:
// - final cost within 1e-6 relative (both converge on gradient_tolerance 1e-4),
// - the same residual and effective parameter counts (exact: the problem layout is Tier A),
// - iteration counts within a factor of two,
// - every pose within 1e-6 deg / 1e-6 and every point within 1e-6 of pycolmap's. The gauge
//   (TWO_CAMS_FROM_WORLD) fixes the same frames on both sides, so no Sim3 alignment is needed.
// The bounds leave room for convergence-tolerance differences, not for bugs. The initial cost
// is checked to 1e-12 relative: it is the same input evaluated by the same cost functions.

using System.Text.Json;

using ColmapSharp.Estimators;
using ColmapSharp.Scene;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class BundleAdjustmentOracleTests
{
	private static Reconstruction ReadModel(string name)
	{
		var reconstruction = new Reconstruction();
		reconstruction.ReadBinary(OracleFixture.PathOf(Path.Combine("bundle_adjustment", name)));
		return reconstruction;
	}

	[Test]
	public async Task DefaultBundleAdjuster_MatchesPycolmap()
	{
		JsonElement manifest = OracleFixture.Load("bundle_adjustment/manifest.json");
		Reconstruction reconstruction = ReadModel("input");
		Reconstruction expected = ReadModel("solved");

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		var options = new BundleAdjustmentOptions { PrintSummary = false };
		BundleAdjustmentSummary summary =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction).Solve();
		SolverSummary ceres = ((CeresBundleAdjustmentSummary)summary).CeresSummary;

		await Assert.That(summary.TerminationType).IsEqualTo(BundleAdjustmentTerminationType.Convergence);
		await Assert.That(ceres.NumResidualsReduced).IsEqualTo(manifest.GetProperty("num_residuals_reduced").GetInt32());
		await Assert.That(ceres.NumEffectiveParametersReduced)
			.IsEqualTo(manifest.GetProperty("num_effective_parameters_reduced").GetInt32());

		double expectedInitialCost = manifest.GetProperty("initial_cost").GetDouble();
		double expectedFinalCost = manifest.GetProperty("final_cost").GetDouble();
		await Assert.That(Math.Abs(ceres.InitialCost - expectedInitialCost) / expectedInitialCost).IsLessThan(1e-12);
		await Assert.That(Math.Abs(ceres.FinalCost - expectedFinalCost) / expectedFinalCost).IsLessThan(1e-6);

		int expectedIterations = manifest.GetProperty("num_iterations").GetInt32();
		int iterations = ceres.NumSuccessfulSteps + ceres.NumUnsuccessfulSteps;
		await Assert.That(iterations).IsLessThanOrEqualTo(2 * expectedIterations);
		await Assert.That(2 * iterations).IsGreaterThanOrEqualTo(expectedIterations);

		double maxRotationErrorDeg = 0;
		double maxTranslationError = 0;
		foreach (uint imageId in expected.RegImageIds())
		{
			var camFromWorld = reconstruction.Image(imageId).CamFromWorld();
			var expectedCamFromWorld = expected.Image(imageId).CamFromWorld();
			maxRotationErrorDeg = Math.Max(
				maxRotationErrorDeg, camFromWorld.Rotation.AngularDistance(expectedCamFromWorld.Rotation) * 180 / Math.PI);
			maxTranslationError = Math.Max(
				maxTranslationError, (camFromWorld.Translation - expectedCamFromWorld.Translation).Norm);
		}

		double maxPointError = 0;
		foreach ((ulong point3DId, Point3D point3D) in expected.Points3D)
		{
			maxPointError = Math.Max(maxPointError, (reconstruction.Point3D(point3DId).Xyz - point3D.Xyz).Norm);
		}

		await Assert.That(maxRotationErrorDeg).IsLessThan(1e-6);
		await Assert.That(maxTranslationError).IsLessThan(1e-6);
		await Assert.That(maxPointError).IsLessThan(1e-6);
	}
}
