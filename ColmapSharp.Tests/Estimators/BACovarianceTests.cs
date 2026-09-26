// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BACovarianceTests: colmap/estimators/covariance_test.cc 1:1. The TEST_P suite
// ParameterizedBACovarianceTests.CompareWithCeres, instantiated over seven
// (BACovarianceOptions, BACovarianceTestOptions) pairs, becomes one method with the pair as
// [Arguments] (params, fixed_points, fixed_cam_poses, fixed_cam_intrinsics), in COLMAP's
// instantiation order. Tests ColmapSharp/Estimators/Covariance*.cs against ceres::Covariance
// (Solver/Covariance.cs) at COLMAP's 1e-8 element tolerance (Tier B).

using ColmapSharp.Estimators;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class BACovarianceTests
{
	private static async Task ExpectNearEigenMatrixXd(MatrixXd mat1, MatrixXd mat2, double tol)
	{
		await Assert.That(mat1.Rows).IsEqualTo(mat2.Rows);
		await Assert.That(mat1.Cols).IsEqualTo(mat2.Cols);
		for (int i = 0; i < mat1.Rows; ++i)
		{
			for (int j = 0; j < mat1.Cols; ++j)
			{
				await Assert.That(mat1[i, j]).IsEqualTo(mat2[i, j]).Within(tol);
			}
		}
	}

	[Test]
	[Arguments(BACovarianceParams.All, false, false, false)]
	[Arguments(BACovarianceParams.All, true, false, false)]
	[Arguments(BACovarianceParams.All, false, false, true)]
	[Arguments(BACovarianceParams.All, false, true, false)]
	[Arguments(BACovarianceParams.Points, false, false, false)]
	[Arguments(BACovarianceParams.Poses, false, false, false)]
	[Arguments(BACovarianceParams.PosesAndPoints, false, false, false)]
	public async Task ParameterizedBACovarianceTests_CompareWithCeres(
		BACovarianceParams parameters, bool fixedPoints, bool fixedCamPoses, bool fixedCamIntrinsics)
	{
		var options = new BACovarianceOptions { Params = parameters };

		bool estimatePointCovs = options.Params is BACovarianceParams.Points
			or BACovarianceParams.PosesAndPoints or BACovarianceParams.All;
		bool estimatePoseCovs = options.Params is BACovarianceParams.Poses
			or BACovarianceParams.PosesAndPoints or BACovarianceParams.All;
		bool estimateOtherCovs = options.Params == BACovarianceParams.All;

		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 7,
			NumPoints3D = 200,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = 0.01 }, reconstruction);

		var config = new BundleAdjustmentConfig();
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			config.AddImage(imageId);
			if (fixedCamPoses)
			{
				config.SetConstantRigFromWorldPose(image.FrameId);
			}

			if (fixedCamIntrinsics)
			{
				config.SetConstantCamIntrinsics(image.CameraId);
			}
		}

		// Fix the Gauge by always setting at least 3 points as constant.
		Check.Gt(reconstruction.NumPoints3D, 3);
		int numConstantPoints = 0;
		foreach (ulong point3DId in reconstruction.Points3D.Keys)
		{
			if (++numConstantPoints <= 3 || fixedPoints)
			{
				config.AddConstantPoint(point3DId);
			}
		}

		BundleAdjuster bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(
			new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		// Cast to CeresBundleAdjuster to access Problem.
		var ceresBa = bundleAdjuster as CeresBundleAdjuster;
		await Assert.That(ceresBa).IsNotNull();
		Problem problem = ceresBa!.Problem;

		BACovariance? baCov = BACovarianceEstimation.EstimateBACovariance(options, reconstruction, ceresBa);
		await Assert.That(baCov).IsNotNull();

		List<PointParam> points = BACovarianceEstimation.GetPointParams(reconstruction, problem);
		if (fixedPoints)
		{
			await Assert.That(points).IsEmpty();
		}
		else
		{
			await Assert.That(points.Count).IsEqualTo(syntheticDatasetOptions.NumPoints3D - 3);
		}

		List<PoseParam> poses = BACovarianceEstimation.GetPoseParams(reconstruction, problem);
		if (fixedCamPoses)
		{
			await Assert.That(poses).IsEmpty();
		}
		else
		{
			await Assert.That(poses.Count).IsEqualTo(syntheticDatasetOptions.NumFramesPerRig);
		}

		List<ArraySegment<double>> others = BACovarianceEstimation.GetOtherParams(problem, poses, points);
		if (fixedCamIntrinsics)
		{
			await Assert.That(others).IsEmpty();
		}
		else
		{
			await Assert.That(others.Count).IsEqualTo(syntheticDatasetOptions.NumCamerasPerRig);
		}

		if (!fixedCamPoses && estimatePoseCovs)
		{
			await ComparePoseCovariances(baCov!, problem, poses);
		}

		if (!fixedCamIntrinsics && estimateOtherCovs)
		{
			await CompareOtherCovariances(baCov!, problem, others);
		}

		if (!fixedPoints && estimatePointCovs)
		{
			await ComparePointCovariances(baCov!, problem, poses, others, points);
		}
	}

	// "Comparing pose covariances". The C# Covariance takes the block list instead of Ceres'
	// block pairs; the pairs COLMAP asks for are all pairs of these blocks.
	private static async Task ComparePoseCovariances(BACovariance baCov, Problem problem, List<PoseParam> poses)
	{
		var ceresCovComputer = new Covariance();
		await Assert.That(ceresCovComputer.Compute(poses.Select(pose => pose.CamFromWorld).ToList(), problem)).IsTrue();

		foreach (PoseParam pose1 in poses)
		{
			foreach (PoseParam pose2 in poses)
			{
				var paramBlocks = new List<ArraySegment<double>>();

				int tangentSize1 = problem.ParameterBlockTangentSize(pose1.CamFromWorld);
				paramBlocks.Add(pose1.CamFromWorld);

				int tangentSize2 = 0;
				if (pose1.ImageId != pose2.ImageId)
				{
					tangentSize2 += problem.ParameterBlockTangentSize(pose2.CamFromWorld);
					paramBlocks.Add(pose2.CamFromWorld);
				}

				MatrixXd ceresCov = ceresCovComputer.GetCovarianceMatrixInTangentSpace(paramBlocks);

				if (pose1.ImageId == pose2.ImageId)
				{
					MatrixXd? cov = baCov.GetCamCovFromWorld(pose1.ImageId);
					await Assert.That(cov).IsNotNull();
					await ExpectNearEigenMatrixXd(ceresCov, cov!, tol: 1e-8);
				}
				else
				{
					MatrixXd? cov = baCov.GetCamCrossCovFromWorld(pose1.ImageId, pose2.ImageId);
					await Assert.That(cov).IsNotNull();
					await ExpectNearEigenMatrixXd(
						ceresCov.Block(0, tangentSize1, tangentSize1, tangentSize2), cov!, tol: 1e-8);
				}
			}
		}

		await Assert.That(baCov.GetCamCovFromWorld(Types.InvalidImageId)).IsNull();
		await Assert.That(baCov.GetCamCrossCovFromWorld(Types.InvalidImageId, poses[0].ImageId)).IsNull();
		await Assert.That(baCov.GetCamCrossCovFromWorld(poses[0].ImageId, Types.InvalidImageId)).IsNull();
	}

	// "Comparing other covariances".
	private static async Task CompareOtherCovariances(BACovariance baCov, Problem problem, List<ArraySegment<double>> others)
	{
		var ceresCovComputer = new Covariance();
		await Assert.That(ceresCovComputer.Compute(others.Where(other => other.Array is not null).ToList(), problem)).IsTrue();

		foreach (ArraySegment<double> other in others)
		{
			MatrixXd ceresCov = ceresCovComputer.GetCovarianceMatrixInTangentSpace([other]);
			await Assert.That(ceresCov.Rows).IsEqualTo(problem.ParameterBlockTangentSize(other));

			MatrixXd? cov = baCov.GetOtherParamsCov(other);
			await Assert.That(cov).IsNotNull();
			await ExpectNearEigenMatrixXd(ceresCov, cov!, tol: 1e-8);
		}

		await Assert.That(baCov.GetOtherParamsCov(default)).IsNull();
	}

	// "Comparing point covariances".
	private static async Task ComparePointCovariances(
		BACovariance baCov, Problem problem, List<PoseParam> poses, List<ArraySegment<double>> others, List<PointParam> points)
	{
		// Set all pose/other parameters as constant.
		foreach (PoseParam pose in poses)
		{
			problem.SetParameterBlockConstant(pose.CamFromWorld);
		}

		foreach (ArraySegment<double> other in others)
		{
			if (other.Array is not null)
			{
				problem.SetParameterBlockConstant(other);
			}
		}

		var ceresCovComputer = new Covariance();
		await Assert.That(ceresCovComputer.Compute(
			points.Where(point => point.Xyz.Array is not null).Select(point => point.Xyz).ToList(), problem)).IsTrue();

		foreach (PointParam point in points)
		{
			MatrixXd ceresCov = ceresCovComputer.GetCovarianceMatrixInTangentSpace([point.Xyz]);
			await Assert.That(ceresCov.Rows).IsEqualTo(problem.ParameterBlockTangentSize(point.Xyz));

			MatrixXd? cov = baCov.GetPointCov(point.Point3DId);
			await Assert.That(cov).IsNotNull();
			await ExpectNearEigenMatrixXd(ceresCov, cov!, tol: 1e-8);
		}

		await Assert.That(baCov.GetPointCov(Types.InvalidPoint3DId)).IsNull();
	}
}
