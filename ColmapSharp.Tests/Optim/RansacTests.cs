// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RansacTests: colmap/optim/ransac_test.cc ported 1:1, one method per gtest TEST(Suite,
// Name) named Suite_Name, same checks and tolerances. Tests ColmapSharp/Optim/Ransac.cs
// with the SimilarityTransformEstimator3d of Estimators/Solvers/SimilarityTransform.cs.
// ComputeNumTrials is Tier A (exact trial counts); the estimations are Tier C (outcome:
// success, inlier set, and the model within COLMAP's 1e-6). The test data helpers are
// shared with LoRansacTests.cs, as the two C++ files carry identical copies.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test runs RANSAC before its first await (the PRNG is per thread and an await may
// resume elsewhere). ParallelSimilarityTransform sets num_threads = 4, which this port
// validates and then runs serially (Ransac.cs header).

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

/// <summary>The SimilarityTransformTestData struct and helpers of ransac_test.cc / loransac_test.cc.</summary>
internal sealed class SimilarityTransformTestData
{
	public Sim3d ExpectedTgtFromSrc { get; init; }

	public Vector3d[] Src { get; init; } = [];

	public Vector3d[] Tgt { get; init; } = [];

	public int NumSamples { get; init; }

	public int NumOutliers { get; init; }

	public static SimilarityTransformTestData Generate(int numSamples = 1000, int numOutliers = 400)
	{
		var expectedTgtFromSrc = new Sim3d(2, RandomEigen.RandomEigenQuaterniond(), new Vector3d(100, 10, 10));

		var src = new Vector3d[numSamples];
		var tgt = new Vector3d[numSamples];
		for (int i = 0; i < numSamples; ++i)
		{
			src[i] = new Vector3d(i, Math.Sqrt(i) + 2, Math.Sqrt(2 * i + 2));
			tgt[i] = expectedTgtFromSrc * src[i];
		}

		for (int i = 0; i < numOutliers; ++i)
		{
			double x = RandomUtils.RandomUniformReal(-3000.0, -2000.0);
			double y = RandomUtils.RandomUniformReal(-4000.0, -3000.0);
			double z = RandomUtils.RandomUniformReal(-5000.0, -4000.0);
			tgt[i] = new Vector3d(x, y, z);
		}

		return new SimilarityTransformTestData
		{
			ExpectedTgtFromSrc = expectedTgtFromSrc,
			Src = src,
			Tgt = tgt,
			NumSamples = numSamples,
			NumOutliers = numOutliers,
		};
	}

	/// <summary>ValidateReport of the C++ tests.</summary>
	public async Task ValidateReport(RansacReport<Matrix3x4d, InlierSupportMeasurer.Support> report)
	{
		using (Assert.Multiple())
		{
			await Assert.That(report.Success).IsTrue();
			await Assert.That(report.NumTrials).IsGreaterThan(0UL);

			await Assert.That(report.Support.NumInliers).IsEqualTo(NumSamples - NumOutliers);
			for (int i = 0; i < NumSamples; ++i)
			{
				await Assert.That(report.InlierMask[i]).IsEqualTo(i >= NumOutliers);
			}

			double matrixDiff = (ExpectedTgtFromSrc.ToMatrix() - report.Model).Norm();
			await Assert.That(matrixDiff).IsLessThan(1e-6);
		}
	}
}

public class RansacTests
{
	private static ulong ComputeNumTrials(ulong numInliers, ulong numSamples, double confidence, double numTrialsMultiplier)
	{
		return Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>.ComputeNumTrials(
			numInliers, numSamples, confidence, numTrialsMultiplier);
	}

	[Test]
	public async Task RANSAC_Options()
	{
		var options = new RansacOptions();
		using (Assert.Multiple())
		{
			await Assert.That(options.MaxError).IsEqualTo(0.0);
			await Assert.That(options.MinInlierRatio).IsEqualTo(0.1);
			await Assert.That(options.Confidence).IsEqualTo(0.99);
			await Assert.That(options.MinNumTrials).IsEqualTo(0);
			await Assert.That(options.MaxNumTrials).IsEqualTo(int.MaxValue);
		}
	}

	[Test]
	public async Task RANSAC_Report()
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
	public async Task RANSAC_NumTrials()
	{
		using (Assert.Multiple())
		{
			await Assert.That(ComputeNumTrials(1, 100, 0.99, 1.0)).IsEqualTo(18446744073709551615UL);
			await Assert.That(ComputeNumTrials(10, 100, 0.99, 1.0)).IsEqualTo(6204UL);
			await Assert.That(ComputeNumTrials(10, 100, 0.999, 1.0)).IsEqualTo(9305UL);
			await Assert.That(ComputeNumTrials(10, 100, 0.999, 2.0)).IsEqualTo(18610UL);
			await Assert.That(ComputeNumTrials(50, 100, 0.99, 1.0)).IsEqualTo(36UL);
			await Assert.That(ComputeNumTrials(50, 100, 0.999, 1.0)).IsEqualTo(54UL);
			await Assert.That(ComputeNumTrials(100, 100, 0.99, 1.0)).IsEqualTo(1UL);
			await Assert.That(ComputeNumTrials(100, 100, 0.999, 1.0)).IsEqualTo(1UL);
			await Assert.That(ComputeNumTrials(100, 100, 0, 1.0)).IsEqualTo(1UL);
		}
	}

	[Test]
	public async Task RANSAC_SimilarityTransform()
	{
		var data = SimilarityTransformTestData.Generate();

		var options = new RansacOptions();
		options.MaxError = 10;
		options.RandomSeed = RandomUtils.DefaultPRNGSeed;
		var ransac = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(options, new SimilarityTransformEstimator3d());
		var report = ransac.Estimate(data.Src, data.Tgt);

		await data.ValidateReport(report);
	}

	[Test]
	public async Task RANSAC_ParallelSimilarityTransform()
	{
		var data = SimilarityTransformTestData.Generate();

		var options = new RansacOptions();
		options.MaxError = 10;
		options.RandomSeed = RandomUtils.DefaultPRNGSeed;
		options.NumThreads = 4;
		var ransac = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(options, new SimilarityTransformEstimator3d());
		var report = ransac.Estimate(data.Src, data.Tgt);

		await data.ValidateReport(report);
	}

	[Test]
	public async Task RANSAC_ReproducibilityWithRandomSeed()
	{
		var data = SimilarityTransformTestData.Generate();

		var options1 = new RansacOptions();
		options1.MaxError = 10;
		options1.RandomSeed = 42;
		var ransac1 = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(options1, new SimilarityTransformEstimator3d());
		var report1 = ransac1.Estimate(data.Src, data.Tgt);

		RansacOptions options2 = options1;
		var ransac2 = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(options2, new SimilarityTransformEstimator3d());
		var report2 = ransac2.Estimate(data.Src, data.Tgt);

		// Now change the seed.
		options2.RandomSeed = 123;
		var ransac3 = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d>(options2, new SimilarityTransformEstimator3d());
		var report3 = ransac3.Estimate(data.Src, data.Tgt);

		// ASSERT_TRUE: the comparisons below are meaningless without success.
		await Assert.That(report1.Success).IsTrue();
		await Assert.That(report2.Success).IsTrue();

		using (Assert.Multiple())
		{
			// Results should be exactly the same.
			await Assert.That(report2.Support.NumInliers).IsEqualTo(report1.Support.NumInliers);
			await Assert.That(report2.InlierMask.SequenceEqual(report1.InlierMask)).IsTrue();
			await Assert.That(report2.Model == report1.Model).IsTrue();
		}

		await Assert.That(report3.Success).IsTrue();

		// Results should now differ.
		await Assert.That(report3.Model != report1.Model).IsTrue();
	}

	/// <summary>
	/// C#-only (not in ransac_test.cc): pins divergence 16. With as many
	/// data pairs as the estimator needs, PROSAC's first sample includes index
	/// total_num_samples; COLMAP reads past the end of the data there, this port fails a Check.
	/// </summary>
	[Test]
	public async Task CSharpOnly_ProgressiveSamplerIndexPastEndFailsCheck()
	{
		var data = SimilarityTransformTestData.Generate(numSamples: 3, numOutliers: 0);

		var options = new RansacOptions();
		options.MaxError = 10;
		var ransac = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d,
			InlierSupportMeasurer, InlierSupportMeasurer.Support, ProgressiveSampler>(
			options, new SimilarityTransformEstimator3d(), new InlierSupportMeasurer());

		await Assert.That(() => ransac.Estimate(data.Src, data.Tgt)).Throws<ArgumentException>();
	}

	/// <summary>
	/// C#-only (not in ransac_test.cc): COLMAP rejects num_threads != 1 for any sampler but
	/// RandomSampler; the serial port keeps that validation (divergence 17).
	/// </summary>
	[Test]
	public async Task CSharpOnly_ParallelRequiresRandomSampler()
	{
		var data = SimilarityTransformTestData.Generate(numSamples: 10, numOutliers: 0);

		var options = new RansacOptions();
		options.MaxError = 10;
		options.NumThreads = 2;
		var ransac = new Ransac<SimilarityTransformEstimator3d, Vector3d, Vector3d, Matrix3x4d,
			InlierSupportMeasurer, InlierSupportMeasurer.Support, CombinationSampler>(
			options, new SimilarityTransformEstimator3d(), new InlierSupportMeasurer());

		await Assert.That(() => ransac.Estimate(data.Src, data.Tgt)).Throws<ArgumentException>();
	}
}
