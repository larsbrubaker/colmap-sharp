// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AlignmentTests: colmap/estimators/cost_functions/alignment_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/AlignmentCostFunctions.cs. Tier B (1e-6).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class AlignmentTests
{
	// Sim3d::params: [qx, qy, qz, qw, tx, ty, tz, scale].
	private static double[] Params(Sim3d s, double scale) =>
		[s.Rotation.X, s.Rotation.Y, s.Rotation.Z, s.Rotation.W, s.Translation.X, s.Translation.Y, s.Translation.Z, scale];

	private static Sim3d RandomSim3d()
	{
		double scale = RandomUtils.RandomUniformReal(0.1, 10.0);
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Sim3d(scale, rotation, translation);
	}

	[Test]
	public async Task Point3DAlignmentCostFunctor_UseLogScale()
	{
		Sim3d bFromA = RandomSim3d();
		var pointInBPrior = new Vector3d(1.0, 2.0, 3.0);
		var pointInA = new Vector3d(3.0, 2.0, 1.0);
		Vector3d pointInB = bFromA * pointInA;
		var costFunction = Point3DAlignmentCostFunctor.Create(pointInBPrior, useLogScale: true);
		ArraySegment<double>[] parametersLogScale = [new([pointInA.X, pointInA.Y, pointInA.Z]), new(Params(bFromA, Math.Log(bFromA.Scale)))];
		var residuals = new double[3];
		await Assert.That(costFunction.Evaluate(parametersLogScale, residuals, default)).IsTrue();

		Vector3d error = pointInB - pointInBPrior;
		await Assert.That(EigenMatrixNear(new Vector3d(residuals[0], residuals[1], residuals[2]), error, 1e-6)).IsTrue();
	}

	[Test]
	public async Task Point3DAlignmentCostFunctor_DoNotUseLogScale()
	{
		Sim3d bFromA = RandomSim3d();
		var pointInBPrior = new Vector3d(1.0, 2.0, 3.0);
		var pointInA = new Vector3d(3.0, 2.0, 1.0);
		Vector3d pointInB = bFromA * pointInA;
		var costFunction = Point3DAlignmentCostFunctor.Create(pointInBPrior, useLogScale: false);
		ArraySegment<double>[] parametersLogScale = [new([pointInA.X, pointInA.Y, pointInA.Z]), new(Params(bFromA, bFromA.Scale))];
		var residuals = new double[3];
		await Assert.That(costFunction.Evaluate(parametersLogScale, residuals, default)).IsTrue();

		Vector3d error = pointInB - pointInBPrior;
		await Assert.That(EigenMatrixNear(new Vector3d(residuals[0], residuals[1], residuals[2]), error, 1e-6)).IsTrue();
	}
}
