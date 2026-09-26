// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CalibrationTests: colmap/estimators/cost_functions/calibration_test.cc ported 1:1, one
// method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/CalibrationCostFunctions.cs; the functor is
// evaluated on Real (doubles), as the C++ calls operator() with doubles. Tier B.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class CalibrationTests
{
	private static double ResidualNorm<TFunctor>(TFunctor functor, params double[] focalLengths)
		where TFunctor : IAutoDiffFunctor
	{
		var residual = new double[2];
		if (!functor.Evaluate(Real.Cast(focalLengths), Real.CastWritable(residual)))
		{
			throw new InvalidOperationException("Evaluate returned false.");
		}

		return Math.Sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
	}

	private static Rigid3d RandomRigid3d()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Rigid3d(rotation, translation);
	}

	[Test]
	public async Task FetzerFocalLengthCostFunctor_ConvexCostLandscape()
	{
		const int NumTrials = 10;
		for (int i = 0; i < NumTrials; ++i)
		{
			const double FocalLength1 = 128;
			const double FocalLength2 = 256;
			var pp1 = new Vector2d(320, 240);
			var pp2 = new Vector2d(480, 320);
			Rigid3d cam2FromCam1 = RandomRigid3d();

			var k1 = new Matrix3d(FocalLength1, 0, pp1.X, 0, FocalLength1, pp1.Y, 0, 0, 1);
			var k2 = new Matrix3d(FocalLength2, 0, pp2.X, 0, FocalLength2, pp2.Y, 0, 0, 1);

			Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(k2, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), k1);

			var costFunctor = new FetzerFocalLengthCostFunctor(f, pp1, pp2);

			await Assert.That(ResidualNorm(costFunctor, FocalLength1, FocalLength2)).IsLessThan(1e-8);

			double previousCost = -1e-9;
			double modifiedFocalLength1 = FocalLength1;
			double modifiedFocalLength2 = FocalLength2;
			for (int j = 0; j < 10; ++j)
			{
				double cost = ResidualNorm(costFunctor, modifiedFocalLength1, modifiedFocalLength2);
				await Assert.That(cost).IsGreaterThan(previousCost);
				previousCost = cost;
				modifiedFocalLength1 *= 1.05;
				modifiedFocalLength2 *= 1.05;
			}

			previousCost = -1e-9;
			modifiedFocalLength1 = FocalLength1;
			modifiedFocalLength2 = FocalLength2;
			for (int j = 0; j < 10; ++j)
			{
				double cost = ResidualNorm(costFunctor, modifiedFocalLength1, modifiedFocalLength2);
				await Assert.That(cost).IsGreaterThan(previousCost);
				previousCost = cost;
				modifiedFocalLength1 *= 0.95;
				modifiedFocalLength2 *= 0.95;
			}
		}
	}

	[Test]
	public async Task FetzerFocalLengthSameCameraCostFunctor_ConvexCostLandscape()
	{
		const int NumTrials = 10;
		for (int i = 0; i < NumTrials; ++i)
		{
			const double FocalLength = 128;
			var pp = new Vector2d(320, 240);
			Rigid3d cam2FromCam1 = RandomRigid3d();

			var k = new Matrix3d(FocalLength, 0, pp.X, 0, FocalLength, pp.Y, 0, 0, 1);

			Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(k, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), k);

			var costFunctor = new FetzerFocalLengthSameCameraCostFunctor(f, pp);

			await Assert.That(ResidualNorm(costFunctor, FocalLength)).IsLessThan(1e-8);

			double previousCost = -1e-9;
			double modifiedFocalLength = FocalLength;
			for (int j = 0; j < 10; ++j)
			{
				double cost = ResidualNorm(costFunctor, modifiedFocalLength);
				await Assert.That(cost).IsGreaterThan(previousCost);
				previousCost = cost;
				modifiedFocalLength *= 1.05;
			}

			previousCost = -1e-9;
			modifiedFocalLength = FocalLength;
			for (int j = 0; j < 10; ++j)
			{
				double cost = ResidualNorm(costFunctor, modifiedFocalLength);
				await Assert.That(cost).IsGreaterThan(previousCost);
				previousCost = cost;
				modifiedFocalLength *= 0.95;
			}
		}
	}
}
