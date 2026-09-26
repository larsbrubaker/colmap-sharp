// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedAbsolutePoseTests: colmap/estimators/solvers/generalized_absolute_pose_test.cc
// ported 1:1. The TEST_P suite ParameterizedGP3PEstimatorTests.Nominal is instantiated over
// the same six (num_cams, panoramic) pairs as INSTANTIATE_TEST_SUITE_P(GP3PEstimatorTests,
// ...), with the same checks and tolerances. Tests
// ColmapSharp/Estimators/Solvers/GeneralizedAbsolutePose.cs, PoseLib/Gp3p.cs (non-panoramic
// rigs) and the P3P fallback (panoramic rigs). Tier C: the models come out of RANSAC over
// Tier B solvers, and COLMAP's Rigid3dNear tolerances are the bar.
//
// The PRNG is seeded with 1 as COLMAP's test main does, and all random draws happen
// synchronously before the first await, because the PRNG is per thread. C++ leaves the
// evaluation order of constructor arguments unspecified; the draws follow clang's
// left-to-right order (the macOS reference build), which C# guarantees. Failed EXPECTs are
// collected in an ExpectationLog so one run lists them all, as gtest does.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class GeneralizedAbsolutePoseTests
{
	[Test]
	[Arguments(1, false)]
	[Arguments(2, false)]
	[Arguments(3, false)]
	[Arguments(4, false)]
	[Arguments(1, true)]
	[Arguments(2, true)]
	public async Task ParameterizedGP3PEstimatorTests_Nominal(int kNumCams, bool kPanoramic)
	{
		RandomUtils.SetPRNGSeed(1);

		// Note that we can estimate the minimal problem from only 3 points but we need a 4th
		// point to choose the correct solution. In theory, we don't need RANSAC as we generate
		// exact correspondences, but we use it in this test to do the choosing of the best
		// solution for us.
		const int kNumPoints = 4;
		const int kNumTrials = 10;

		var log = new ExpectationLog();
		for (int i = 0; i < kNumTrials; ++i)
		{
			var rigFromWorld = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
			Rigid3d worldFromRig = rigFromWorld.Inverse();

			var camsFromWorld = new Rigid3d[kNumCams];
			var camsFromRig = new Rigid3d[kNumCams];
			for (int c = 0; c < kNumCams; ++c)
			{
				if (kPanoramic)
				{
					Quaterniond camFromRigRotation = RandomEigen.RandomEigenQuaterniond();
					camsFromRig[c] = new Rigid3d(camFromRigRotation, camFromRigRotation * new Vector3d(1, 2, 3));
					camsFromWorld[c] = camsFromRig[c] * rigFromWorld;
				}
				else
				{
					camsFromWorld[c] = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
					camsFromRig[c] = camsFromWorld[c] * worldFromRig;
				}
			}

			var points2D = new List<GP3PObservation>();
			var points3D = new List<Vector3d>();
			var points3DOutlier = new List<Vector3d>();
			Matrix3d outlierRotation = new AngleAxisd(Math.PI / 2, Vector3d.UnitX).ToRotationMatrix();
			for (int k = 0; k < kNumPoints; ++k)
			{
				double rx = RandomUtils.RandomUniformReal(-0.5, 0.5);
				double ry = RandomUtils.RandomUniformReal(-0.5, 0.5);
				Vector3d rayInCam = new Vector3d(rx, ry, 1).Normalized();
				points2D.Add(new GP3PObservation(camsFromRig[k % kNumCams].ToMatrix(), rayInCam));
				Vector3d point3DInCam = rayInCam * RandomUtils.RandomUniformReal(0.1, 10.0);
				Rigid3d worldFromCam = camsFromWorld[k % kNumCams].Inverse();
				points3D.Add(worldFromCam * point3DInCam);
				points3DOutlier.Add(worldFromCam * (outlierRotation * point3DInCam));
			}

			GP3PObservation[] x = points2D.ToArray();
			foreach (GP3PEstimator.ResidualType residualType in new[]
				{ GP3PEstimator.ResidualType.CosineDistance, GP3PEstimator.ResidualType.ReprojectionError })
			{
				var options = new RansacOptions();
				options.MaxError = 1e-5;
				var ransac = new Ransac<GP3PEstimator, GP3PObservation, Vector3d, Rigid3d>(options, new GP3PEstimator(residualType));

				var report = ransac.Estimate(x, points3D.ToArray());

				string what = $"trial {i}, {residualType}";
				log.True(report.Success, $"{what}: report.success");
				log.True(
					Rigid3dNear(report.Model, rigFromWorld, rtol: 1e-6, ttol: 1e-6),
					$"{what}: model {report.Model} near {rigFromWorld}");

				// Test residuals of inlier points.
				var residuals = new double[x.Length];
				ransac.Estimator.Residuals(x, points3D.ToArray(), report.Model, residuals);
				log.Equal(residuals.Length, points2D.Count, $"{what}: residuals.size()");
				for (int k = 0; k < residuals.Length; ++k)
				{
					log.True(residuals[k] < 1e-10, $"{what}: residual {k} = {residuals[k]}");
				}

				// Test residuals of outlier points.
				var residualsOutlier = new double[x.Length];
				ransac.Estimator.Residuals(x, points3DOutlier.ToArray(), report.Model, residualsOutlier);
				log.Equal(residualsOutlier.Length, points2D.Count, $"{what}: residuals_outlier.size()");
				for (int k = 0; k < residualsOutlier.Length; ++k)
				{
					log.True(residualsOutlier[k] > 1e-2, $"{what}: outlier residual {k} = {residualsOutlier[k]}");
				}
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}
}
