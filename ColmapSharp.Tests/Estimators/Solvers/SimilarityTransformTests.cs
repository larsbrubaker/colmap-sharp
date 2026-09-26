// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SimilarityTransformTests: colmap/estimators/solvers/similarity_transform_test.cc ported
// 1:1, one method per gtest TEST(Suite, Name) named Suite_Name, same checks and
// tolerances. Tests ColmapSharp/Estimators/Solvers/SimilarityTransform.cs. The estimates
// are Tier B (a 3x3 SVD) and the robust ones Tier C (LO-RANSAC); COLMAP's tolerances are the
// bars.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and draws everything before its first await (the PRNG is per
// thread and an await may resume elsewhere). The robust tests leave random_seed at -1, so
// RANSAC keeps drawing from that same thread PRNG, as in COLMAP.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;
using static ColmapSharp.Tests.Sim3dMatchers;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class SimilarityTransformTests
{
	private static (Vector3d[] Src, Vector3d[] Tgt) GenerateData(int numInliers, int numOutliers, Matrix3x4d tgtFromSrc)
	{
		var src = new List<Vector3d>();
		var tgt = new List<Vector3d>();

		// Generate inlier data.
		for (int i = 0; i < numInliers; ++i)
		{
			src.Add(new Vector3d(i, Math.Sqrt(i) + 2, Math.Sqrt(2 * i + 2)));
			tgt.Add(tgtFromSrc * src[^1].Homogeneous());
		}

		// Add some faulty data.
		for (int i = 0; i < numOutliers; ++i)
		{
			src.Add(new Vector3d(i, Math.Sqrt(i) + 2, Math.Sqrt(2 * i + 2)));
			double x = RandomUtils.RandomUniformReal(-3000.0, -2000.0);
			double y = RandomUtils.RandomUniformReal(-4000.0, -3000.0);
			double z = RandomUtils.RandomUniformReal(-5000.0, -4000.0);
			tgt.Add(new Vector3d(x, y, z));
		}

		return (src.ToArray(), tgt.ToArray());
	}

	private static async Task TestEstimateRigid3dWithNumCoords(int numCoords)
	{
		RandomUtils.SetPRNGSeed(0);
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		var gtTgtFromSrc = new Rigid3d(rotation, translation);
		var (src, tgt) = GenerateData(numCoords, 0, gtTgtFromSrc.ToMatrix());

		var tgtFromSrc = new Rigid3d();
		bool success = SimilarityTransform.EstimateRigid3d(src, tgt, ref tgtFromSrc);
		using (Assert.Multiple())
		{
			await Assert.That(success).IsTrue();
			await Assert.That(gtTgtFromSrc.Rotation.AngularDistance(tgtFromSrc.Rotation)).IsLessThan(1e-6);
			await Assert.That((gtTgtFromSrc.Translation - tgtFromSrc.Translation).Norm).IsLessThan(1e-6);
		}
	}

	[Test]
	public async Task Rigid3d_EstimateMinimal() => await TestEstimateRigid3dWithNumCoords(3);

	[Test]
	public async Task Rigid3d_EstimateOverDetermined() => await TestEstimateRigid3dWithNumCoords(100);

	[Test]
	public async Task Rigid3d_EstimateMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector3d[3];
		var tgtFromSrc = new Rigid3d();
		await Assert.That(SimilarityTransform.EstimateRigid3d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Rigid3d_EstimateNonMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector3d[5];
		var tgtFromSrc = new Rigid3d();
		await Assert.That(SimilarityTransform.EstimateRigid3d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Rigid3d_EstimateRobust()
	{
		RandomUtils.SetPRNGSeed(0);
		const int NumInliers = 1000;
		const int NumOutliers = 400;

		var gtTgtFromSrc = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), new Vector3d(100, 10, 10));
		var (src, tgt) = GenerateData(NumInliers, NumOutliers, gtTgtFromSrc.ToMatrix());

		// Robustly estimate transformation using RANSAC.
		var options = new RansacOptions();
		options.MaxError = 10;
		var report = SimilarityTransform.EstimateRigid3dRobust(src, tgt, options, out Rigid3d tgtFromSrc);

		using (Assert.Multiple())
		{
			await Assert.That(report.Success).IsTrue();
			await Assert.That(report.NumTrials).IsGreaterThan(0UL);

			// Make sure outliers were detected correctly.
			await Assert.That(report.Support.NumInliers).IsEqualTo(NumInliers);
			for (int i = 0; i < NumInliers + NumOutliers; ++i)
			{
				await Assert.That(report.InlierMask[i]).IsEqualTo(i < NumInliers);
			}

			await Assert.That(Rigid3dNear(tgtFromSrc, gtTgtFromSrc, rtol: 1e-6, ttol: 1e-6)).IsTrue();
		}
	}

	private static async Task TestEstimateSim3dWithNumCoords(int numCoords)
	{
		RandomUtils.SetPRNGSeed(0);
		double scale = RandomUtils.RandomUniformReal(0.1, 10.0);
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		var gtTgtFromSrc = new Sim3d(scale, rotation, translation);
		var (src, tgt) = GenerateData(numCoords, 0, gtTgtFromSrc.ToMatrix());

		var tgtFromSrc = new Sim3d();
		bool success = SimilarityTransform.EstimateSim3d(src, tgt, ref tgtFromSrc);
		using (Assert.Multiple())
		{
			await Assert.That(success).IsTrue();
			await Assert.That(Math.Abs(gtTgtFromSrc.Scale - tgtFromSrc.Scale)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(gtTgtFromSrc.Rotation.AngularDistance(tgtFromSrc.Rotation)).IsLessThan(1e-6);
			await Assert.That((gtTgtFromSrc.Translation - tgtFromSrc.Translation).Norm).IsLessThan(1e-6);
		}
	}

	[Test]
	public async Task Sim3d_EstimateMinimal() => await TestEstimateSim3dWithNumCoords(3);

	[Test]
	public async Task Sim3d_EstimateOverDetermined() => await TestEstimateSim3dWithNumCoords(100);

	[Test]
	public async Task Sim3d_EstimateMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector3d[3];
		var tgtFromSrc = new Sim3d();
		await Assert.That(SimilarityTransform.EstimateSim3d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Sim3d_EstimateNonMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector3d[5];
		var tgtFromSrc = new Sim3d();
		await Assert.That(SimilarityTransform.EstimateSim3d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Sim3d_EstimateRobust()
	{
		RandomUtils.SetPRNGSeed(0);
		const int NumInliers = 1000;
		const int NumOutliers = 400;

		var gtTgtFromSrc = new Sim3d(2, RandomEigen.RandomEigenQuaterniond(), new Vector3d(100, 10, 10));
		var (src, tgt) = GenerateData(NumInliers, NumOutliers, gtTgtFromSrc.ToMatrix());

		// Robustly estimate transformation using RANSAC.
		var options = new RansacOptions();
		options.MaxError = 10;
		var tgtFromSrc = new Sim3d();
		var report = SimilarityTransform.EstimateSim3dRobust(src, tgt, options, ref tgtFromSrc);

		using (Assert.Multiple())
		{
			await Assert.That(report.Success).IsTrue();
			await Assert.That(report.NumTrials).IsGreaterThan(0UL);

			// Make sure outliers were detected correctly.
			await Assert.That(report.Support.NumInliers).IsEqualTo(NumInliers);
			for (int i = 0; i < NumInliers + NumOutliers; ++i)
			{
				await Assert.That(report.InlierMask[i]).IsEqualTo(i < NumInliers);
			}

			await Assert.That(Sim3dNear(tgtFromSrc, gtTgtFromSrc, stol: 1e-8, rtol: 1e-8, ttol: 1e-8)).IsTrue();
		}
	}
}
