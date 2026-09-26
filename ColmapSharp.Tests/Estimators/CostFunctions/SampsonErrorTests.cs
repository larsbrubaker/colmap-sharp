// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SampsonErrorTests: colmap/estimators/cost_functions/sampson_error_test.cc ported 1:1
// (SampsonErrorCostFunctor.Nominal), same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/SampsonError.cs through
// Solver/AutoDiffCostFunction.cs, as the C++ goes through ceres::CostFunction::Evaluate.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class SampsonErrorTests
{
	[Test]
	public async Task SampsonErrorCostFunctor_Nominal()
	{
		double[] camFromWorld = [0, 0, 0, 1, 0, 1, 0];
		ArraySegment<double>[] parameters = [new(camFromWorld)];
		var residuals = new double[1];

		var costFunction = SampsonErrorCostFunctor.Create(new Vector2d(0, 0), new Vector2d(0, 0));
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0);

		costFunction = SampsonErrorCostFunctor.Create(new Vector2d(0, 0), new Vector2d(1, 0));
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0] * residuals[0]).IsEqualTo(0.5).Within(1e-6);

		costFunction = SampsonErrorCostFunctor.Create(new Vector2d(0, 0), new Vector2d(1, 1));
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0] * residuals[0]).IsEqualTo(0.5).Within(1e-6);
	}
}
