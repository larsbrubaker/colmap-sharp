// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// UtilsTests: colmap/estimators/cost_functions/utils_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/CostFunctionUtils.cs through CostFunction.Evaluate,
// as the C++ goes through ceres::CostFunction::Evaluate. Tier B (1e-10).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class UtilsTests
{
	private static Vector3d ToVector3d(double[] v) => new(v[0], v[1], v[2]);

	private static MatrixXd Covariance()
	{
		MatrixXd covariance = MatrixXd.Identity(3);
		covariance[0, 0] = 4.0;
		return covariance;
	}

	[Test]
	public async Task NormalPriorCostFunctor_Nominal()
	{
		double[] prior = [1, 2, 3];

		var costFunction = NormalPriorCostFunctor.Create(prior);
		await Assert.That(costFunction).IsNotNull();
		await Assert.That(costFunction.NumResiduals).IsEqualTo(3);

		var residuals = new double[3];
		ArraySegment<double>[] parametersZero = [new(prior)];
		await Assert.That(costFunction.Evaluate(parametersZero, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), new Vector3d(0, 0, 0), 1e-10)).IsTrue();

		double[] param = [4, 5, 6];
		ArraySegment<double>[] parameters = [new(param)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), ToVector3d(param) - ToVector3d(prior), 1e-10)).IsTrue();
	}

	[Test]
	public async Task NormalErrorCostFunctor_Nominal()
	{
		double[] param0 = [1, 2, 3];
		double[] param1 = [4, 5, 6];

		var costFunction = NormalErrorCostFunctor.Create(3);
		await Assert.That(costFunction).IsNotNull();
		await Assert.That(costFunction.NumResiduals).IsEqualTo(3);

		var residuals = new double[3];
		ArraySegment<double>[] parametersZero = [new(param0), new(param0)];
		await Assert.That(costFunction.Evaluate(parametersZero, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), new Vector3d(0, 0, 0), 1e-10)).IsTrue();

		ArraySegment<double>[] parameters = [new(param0), new(param1)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), ToVector3d(param0) - ToVector3d(param1), 1e-10)).IsTrue();
	}

	[Test]
	public async Task CovarianceWeightedCostFunctor_NormalPriorCostFunctor()
	{
		double[] prior = [1, 2, 3];
		double[] param = [4, 5, 6];

		var costFunction = CovarianceWeightedCostFunctor.Create(Covariance(), new NormalPriorCostFunctor(prior));

		var residuals = new double[3];
		ArraySegment<double>[] parameters = [new(param)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		var expected = new Vector3d(
			0.5 * (param[0] - prior[0]),
			1.0 * (param[1] - prior[1]),
			1.0 * (param[2] - prior[2]));
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), expected, 1e-10)).IsTrue();
	}

	[Test]
	public async Task CovarianceWeightedCostFunctor_NormalErrorCostFunctor()
	{
		double[] param0 = [1, 2, 3];
		double[] param1 = [4, 5, 6];

		var costFunction = CovarianceWeightedCostFunctor.Create(Covariance(), new NormalErrorCostFunctor(3));

		var residuals = new double[3];
		ArraySegment<double>[] parameters = [new(param0), new(param1)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		var expected = new Vector3d(
			0.5 * (param0[0] - param1[0]),
			1.0 * (param0[1] - param1[1]),
			1.0 * (param0[2] - param1[2]));
		await Assert.That(EigenMatrixNear(ToVector3d(residuals), expected, 1e-10)).IsTrue();
	}
}
