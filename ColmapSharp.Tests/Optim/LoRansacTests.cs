// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// LoRansacTests: colmap/optim/loransac_test.cc ported 1:1, one method per gtest TEST(Suite,
// Name) named Suite_Name, same checks and tolerances. Tests ColmapSharp/Optim/LoRansac.cs
// with SimilarityTransformEstimator3d as both the minimal and the local estimator. Tier C
// (outcome). The test data and ValidateReport are SimilarityTransformTestData in
// RansacTests.cs (the C++ files carry identical copies).
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test runs LO-RANSAC before its first await. num_threads = 4 is validated and then
// run serially (Ransac.cs header).

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class LoRansacTests
{
	private static LoRansac<SimilarityTransformEstimator3d, SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d> Create(RansacOptions options)
	{
		return new LoRansac<SimilarityTransformEstimator3d, SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(
			options, new SimilarityTransformEstimator3d(), new SimilarityTransformEstimator3d());
	}

	[Test]
	public async Task LORANSAC_Report()
	{
		var report = new RansacReport<Matrix3x4d, InlierSupportMeasurer.Support>();
		using (Assert.Multiple())
		{
			await Assert.That(report.Success).IsFalse();
			await Assert.That(report.NumTrials).IsEqualTo(0UL);
			await Assert.That(report.Support.NumInliers).IsEqualTo(0);
			await Assert.That(report.Support.ResidualSum).IsEqualTo(double.MaxValue);
			await Assert.That(report.InlierMask.Length).IsEqualTo(0);
		}
	}

	[Test]
	public async Task LORANSAC_SimilarityTransform()
	{
		var data = SimilarityTransformTestData.Generate();

		var options = new RansacOptions();
		options.MaxError = 10;
		options.RandomSeed = RandomUtils.DefaultPRNGSeed;
		var report = Create(options).Estimate(data.Src, data.Tgt);

		await data.ValidateReport(report);
	}

	[Test]
	public async Task LORANSAC_ParallelSimilarityTransform()
	{
		var data = SimilarityTransformTestData.Generate();

		var options = new RansacOptions();
		options.MaxError = 10;
		options.RandomSeed = RandomUtils.DefaultPRNGSeed;
		options.NumThreads = 4;
		var report = Create(options).Estimate(data.Src, data.Tgt);

		await data.ValidateReport(report);
	}
}
