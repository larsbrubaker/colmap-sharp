// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimationTests.Multiple: the filtering, seed-determinism and
// multiple-model cases of two_view_geometry_test.cc, 1:1: EstimateTwoViewGeometry.
// {DetectWatermark, IgnoreStationaryMatches, CalibratedDeterministic,
// UncalibratedDeterministic, UncalibratedDegensac, PlanarOrPanoramicDeterministic} and
// EstimateMultipleTwoViewGeometries.{SingleGeometry, NoGeometry, MultipleGeometries}.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class TwoViewGeometryEstimationTests
{
	private static TwoViewGeometry Estimate(TwoViewGeometryTestData data, TwoViewGeometryOptions options) =>
		TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			data.Camera1, data.Points1, data.Camera2, data.Points2, data.Matches, options);

	[Test]
	public async Task EstimateTwoViewGeometry_DetectWatermark()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(TwoRigOptions(100, true));

		var twoViewGeometryOptions = new TwoViewGeometryOptions { DetectWatermark = true };
		var log = new ExpectationLog();
		log.True(Estimate(testData, twoViewGeometryOptions).Config != TwoViewGeometry.ConfigurationType.Watermark, "no watermark");

		// Place the points on the left and right side of the images.
		for (int i = 0; i < testData.Matches.Count; ++i)
		{
			double y = (double)i / testData.Matches.Count * testData.Camera1.Height;
			testData.Points1[(int)testData.Matches[i].Point2DIdx1] = new Vector2d(0, y);
			testData.Points2[(int)testData.Matches[i].Point2DIdx2] = new Vector2d(testData.Camera2.Width - 1, y);
		}

		log.Equal(Estimate(testData, twoViewGeometryOptions).Config, TwoViewGeometry.ConfigurationType.Watermark, "left/right");

		// Place the points on the top and bottom side of the images.
		for (int i = 0; i < testData.Matches.Count; ++i)
		{
			double x = (double)i / testData.Matches.Count * testData.Camera1.Width;
			testData.Points1[(int)testData.Matches[i].Point2DIdx1] = new Vector2d(x, 0);
			testData.Points2[(int)testData.Matches[i].Point2DIdx2] = new Vector2d(x, testData.Camera2.Height - 1);
		}

		log.Equal(Estimate(testData, twoViewGeometryOptions).Config, TwoViewGeometry.ConfigurationType.Watermark, "top/bottom");

		// With disabled detection, expect a normal config.
		twoViewGeometryOptions.DetectWatermark = false;
		log.True(Estimate(testData, twoViewGeometryOptions).Config != TwoViewGeometry.ConfigurationType.Watermark, "disabled");
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task EstimateTwoViewGeometry_IgnoreStationaryMatches()
	{
		SyntheticDatasetOptions syntheticDatasetOptions = TwoRigOptions(500, true);
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(syntheticDatasetOptions);

		foreach (FeatureMatch match in testData.Matches)
		{
			testData.Points1[(int)match.Point2DIdx1] = testData.Points2[(int)match.Point2DIdx2];
		}

		var twoViewGeometryOptions = new TwoViewGeometryOptions();
		TwoViewGeometry geometry1 = Estimate(testData, twoViewGeometryOptions);
		var log = new ExpectationLog();
		log.Equal(geometry1.Config, TwoViewGeometry.ConfigurationType.PlanarOrPanoramic, "config 1");
		log.Equal(geometry1.InlierMatches.Count, syntheticDatasetOptions.NumPoints3D, "inliers 1");

		twoViewGeometryOptions.FilterStationaryMatches = true;
		TwoViewGeometry geometry2 = Estimate(testData, twoViewGeometryOptions);
		log.Equal(geometry2.Config, TwoViewGeometry.ConfigurationType.Degenerate, "config 2");
		log.Equal(geometry2.InlierMatches.Count, 0, "inliers 2");
		await Assert.That(log.Failures).IsEmpty();
	}

	// Runs the pair with seeds 42, 42 and 123 and returns the three geometries.
	private static TwoViewGeometry[] RunSeeds(TwoViewGeometryTestData testData, TwoViewGeometryOptions options)
	{
		options.RansacOptions.RandomSeed = 42;
		TwoViewGeometry geometry1 = Estimate(testData, options);
		options.RansacOptions.RandomSeed = 42;
		TwoViewGeometry geometry2 = Estimate(testData, options);
		options.RansacOptions.RandomSeed = 123;
		TwoViewGeometry geometry3 = Estimate(testData, options);
		return [geometry1, geometry2, geometry3];
	}

	private static SyntheticDatasetOptions NoisyOptions(bool hasPriorFocalLength)
	{
		SyntheticDatasetOptions options = TwoRigOptions(500, hasPriorFocalLength);
		options.InlierMatchRatio = 0.6;
		return options;
	}

	[Test]
	public async Task EstimateTwoViewGeometry_CalibratedDeterministic()
	{
		// Keep the noise moderate: low enough that the essential and fundamental matrix
		// support similar numbers of inliers so the scene is unambiguously classified as
		// CALIBRATED across platforms, yet high enough that different RANSAC seeds select
		// different inlier sets and thus yield different E.
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(
			NoisyOptions(true), new SyntheticNoiseOptions { Point2DStddev = 3 });

		// This test verifies RANSAC seed reproducibility, not the default calibration
		// verification boundary. Leave margin for platform-dependent inlier counts.
		var twoViewGeometryOptions = new TwoViewGeometryOptions { MinEFInlierRatio = 0.8 };
		TwoViewGeometry[] geometries = RunSeeds(testData, twoViewGeometryOptions);
		var log = new ExpectationLog();
		for (int i = 0; i < 3; ++i)
		{
			log.Equal(geometries[i].Config, TwoViewGeometry.ConfigurationType.Calibrated, $"config {i + 1}");
		}

		// Using the same random seed should produce identical results.
		log.Equal(geometries[0].E, geometries[1].E, "same seed E");
		// Using a different random seed may produce different results.
		log.True(geometries[0].E != geometries[2].E, "different seed E");
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task EstimateTwoViewGeometry_UncalibratedDeterministic()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(
			NoisyOptions(false), new SyntheticNoiseOptions { Point2DStddev = 5 });

		TwoViewGeometry[] geometries = RunSeeds(testData, new TwoViewGeometryOptions());
		var log = new ExpectationLog();
		for (int i = 0; i < 3; ++i)
		{
			log.Equal(geometries[i].Config, TwoViewGeometry.ConfigurationType.Uncalibrated, $"config {i + 1}");
		}

		// Using the same random seed should produce identical results.
		log.Equal(geometries[0].F, geometries[1].F, "same seed F");
		// Using a different random seed may produce different results.
		log.True(geometries[0].F != geometries[2].F, "different seed F");
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task EstimateTwoViewGeometry_UncalibratedDegensac()
	{
		SyntheticDatasetOptions syntheticDatasetOptions = NoisyOptions(false);
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(
			syntheticDatasetOptions, new SyntheticNoiseOptions { Point2DStddev = 5 });

		// The DEGENSAC-based fundamental matrix estimation is a drop-in replacement that
		// recovers a valid uncalibrated geometry on a general (non-planar) scene.
		var twoViewGeometryOptions = new TwoViewGeometryOptions { UseDegensac = true };
		twoViewGeometryOptions.RansacOptions.RandomSeed = 42;
		TwoViewGeometry geometry = Estimate(testData, twoViewGeometryOptions);
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Uncalibrated);
		await Assert.That(geometry.InlierMatches.Count).IsGreaterThanOrEqualTo(
			(int)(syntheticDatasetOptions.InlierMatchRatio * syntheticDatasetOptions.NumPoints3D * 0.8));
	}

	[Test]
	public async Task EstimateTwoViewGeometry_PlanarOrPanoramicDeterministic()
	{
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 2,
			NumFramesPerRig = 1,
			NumPoints3D = 500,
			InlierMatchRatio = 0.6,
			CameraHasPriorFocalLength = true,
			SensorFromRigTranslationStddev = 0,
		};
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(
			syntheticDatasetOptions, new SyntheticNoiseOptions { Point2DStddev = 5 });

		TwoViewGeometry[] geometries = RunSeeds(testData, new TwoViewGeometryOptions { ForceHUse = true });
		var log = new ExpectationLog();
		log.Equal(geometries[0].Config, TwoViewGeometry.ConfigurationType.PlanarOrPanoramic, "config 1");
		log.Equal(geometries[1].Config, TwoViewGeometry.ConfigurationType.PlanarOrPanoramic, "config 2");
		// Using the same random seed should produce identical results.
		log.Equal(geometries[0].H, geometries[1].H, "same seed H");
		// COLMAP re-checks geometry2's config here (not geometry3's); kept 1:1.
		log.Equal(geometries[1].Config, TwoViewGeometry.ConfigurationType.PlanarOrPanoramic, "config 2 again");
		// Using a different random seed may produce different results.
		log.True(geometries[0].H != geometries[2].H, "different seed H");
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task EstimateMultipleTwoViewGeometries_SingleGeometry()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(TwoRigOptions(100, true));

		var options = new TwoViewGeometryOptions { MultipleModels = true };
		options.RansacOptions.RandomSeed = 42;

		TwoViewGeometry geometry = Estimate(testData, options);
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
		await Assert.That(geometry.InlierMatches.Count).IsGreaterThan(0);
	}

	[Test]
	public async Task EstimateMultipleTwoViewGeometries_NoGeometry()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(TwoRigOptions(5, true));  // Too few points

		var options = new TwoViewGeometryOptions
		{
			MultipleModels = true,
			MinNumInliers = 100,  // Require too many inliers
		};
		options.RansacOptions.RandomSeed = 42;

		TwoViewGeometry geometry = Estimate(testData, options);
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Degenerate);
		await Assert.That(geometry.InlierMatches.Count).IsEqualTo(0);
	}

	[Test]
	public async Task EstimateMultipleTwoViewGeometries_MultipleGeometries()
	{
		// Create two separate synthetic datasets with different poses.
		var reconstruction1 = new Reconstruction();
		Synthetic.SynthesizeDataset(TwoRigOptions(100, true), reconstruction1);

		var reconstruction2 = new Reconstruction();
		Synthetic.SynthesizeDataset(TwoRigOptions(100, true), reconstruction2);

		Image image1 = reconstruction1.Image(1);
		Image image2 = reconstruction1.Image(2);
		Image image3 = reconstruction2.Image(1);
		Image image4 = reconstruction2.Image(2);

		Camera camera1 = reconstruction1.Camera(image1.CameraId).Clone();
		Camera camera2 = reconstruction1.Camera(image2.CameraId).Clone();
		var log = new ExpectationLog();
		log.True(camera1 == reconstruction2.Camera(image3.CameraId), "camera1 == camera3");
		log.True(camera2 == reconstruction2.Camera(image4.CameraId), "camera2 == camera4");

		var points1 = new List<Vector2d>();
		var points2 = new List<Vector2d>();
		var matches1 = new List<FeatureMatch>();
		ExtractPointsAndMatches(reconstruction1, image1, image2, points1, points2, [], matches1);

		var points3 = new List<Vector2d>();
		var points4 = new List<Vector2d>();
		var matches2 = new List<FeatureMatch>();
		ExtractPointsAndMatches(reconstruction2, image3, image4, points3, points4, [], matches2);

		uint matchesOffset = (uint)points1.Count;
		var allPoints1 = new List<Vector2d>(points1);
		allPoints1.AddRange(points3);
		var allPoints2 = new List<Vector2d>(points2);
		allPoints2.AddRange(points4);

		var allMatches = new List<FeatureMatch>(matches1);
		foreach (FeatureMatch match in matches2)
		{
			allMatches.Add(new FeatureMatch(match.Point2DIdx1 + matchesOffset, match.Point2DIdx2 + matchesOffset));
		}

		var twoViewOptions = new TwoViewGeometryOptions { MultipleModels = true };
		twoViewOptions.RansacOptions.RandomSeed = 42;

		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			camera1, allPoints1, camera2, allPoints2, allMatches, twoViewOptions);

		log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Multiple, "config");
		log.Equal(geometry.InlierMatches.Count, matches1.Count + matches2.Count, "inliers");
		await Assert.That(log.Failures).IsEmpty();
	}
}
