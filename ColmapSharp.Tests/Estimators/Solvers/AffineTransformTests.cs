// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AffineTransformTests: colmap/estimators/solvers/affine_transform_test.cc ported 1:1, one
// method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/Solvers/AffineTransform.cs. The estimates are Tier B (LU or SVD)
// and the robust one Tier C (LO-RANSAC); COLMAP's tolerances are the bars.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test draws everything before its first await (the PRNG is per thread). The
// robust test leaves random_seed at -1, so RANSAC keeps drawing from that same thread PRNG,
// as in COLMAP.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class AffineTransformTests
{
	private static (Vector2d[] Src, Vector2d[] Tgt) GenerateData(int numInliers, int numOutliers, Matrix2x3d tgtFromSrc)
	{
		var src = new List<Vector2d>();
		var tgt = new List<Vector2d>();

		// Generate inlier data.
		for (int i = 0; i < numInliers; ++i)
		{
			src.Add(new Vector2d(i, Math.Sqrt(i) + 2));
			tgt.Add(tgtFromSrc * src[^1].Homogeneous());
		}

		// Add some faulty data.
		for (int i = 0; i < numOutliers; ++i)
		{
			src.Add(new Vector2d(i, Math.Sqrt(i) + 2));
			double x = RandomUtils.RandomUniformReal(-3000.0, -2000.0);
			double y = RandomUtils.RandomUniformReal(-4000.0, -3000.0);
			tgt.Add(new Vector2d(x, y));
		}

		return (src.ToArray(), tgt.ToArray());
	}

	private static async Task TestEstimateAffine2dWithNumCoords(int numCoords)
	{
		Matrix2x3d gtTgtFromSrc = RandomEigen.RandomEigenMatrix2x3d();
		var (src, tgt) = GenerateData(numCoords, 0, gtTgtFromSrc);

		Matrix2x3d tgtFromSrc = Matrix2x3d.Zero;
		bool success = AffineTransform.EstimateAffine2d(src, tgt, ref tgtFromSrc);
		using (Assert.Multiple())
		{
			await Assert.That(success).IsTrue();
			await Assert.That(EigenMatrixNear(tgtFromSrc, gtTgtFromSrc, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task Affine2d_EstimateMinimal() => await TestEstimateAffine2dWithNumCoords(3);

	[Test]
	public async Task Affine2d_EstimateOverDetermined() => await TestEstimateAffine2dWithNumCoords(100);

	[Test]
	public async Task Affine2d_EstimateMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector2d[3];
		Matrix2x3d tgtFromSrc = Matrix2x3d.Zero;
		await Assert.That(AffineTransform.EstimateAffine2d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Affine2d_EstimateNonMinimalDegenerate()
	{
		var degenerateSrcTgt = new Vector2d[5];
		Matrix2x3d tgtFromSrc = Matrix2x3d.Zero;
		await Assert.That(AffineTransform.EstimateAffine2d(degenerateSrcTgt, degenerateSrcTgt, ref tgtFromSrc)).IsFalse();
	}

	[Test]
	public async Task Affine2d_EstimateRobust()
	{
		const int NumInliers = 1000;
		const int NumOutliers = 400;

		Matrix2x3d gtTgtFromSrc = RandomEigen.RandomEigenMatrix2x3d();
		var (src, tgt) = GenerateData(NumInliers, NumOutliers, gtTgtFromSrc);

		// Robustly estimate transformation using RANSAC.
		var options = new RansacOptions();
		options.MaxError = 10;
		Matrix2x3d tgtFromSrc = Matrix2x3d.Zero;
		var report = AffineTransform.EstimateAffine2dRobust(src, tgt, options, ref tgtFromSrc);

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

			await Assert.That(EigenMatrixNear(tgtFromSrc, gtTgtFromSrc, 1e-6)).IsTrue();
		}
	}
}
