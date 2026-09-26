// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinySampsonErrorTests: the TinyFundamentalSampsonErrorCostFunctor cases of
// colmap/estimators/cost_functions/tiny_sampson_error_test.cc ported 1:1
// (ParameterizationIsRankTwo, MatchesSquaredSampsonError, JacobianMatchesFiniteDifference),
// named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/TinySampsonError.cs.
//
// Not ported yet: the TinyTangentSampsonErrorCostFunctor, TinyFocalSampsonErrorCostFunctor
// and TinyOneSidedFocalTangentSampsonErrorCostFunctor cases (7 tests); those functors land
// with sampson_error.h (see TinySampsonError.cs).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class TinySampsonErrorTests
{
	// The factorized parameterization always yields a rank-2 matrix with singular
	// values (1, sigma, 0), which is what makes the truncation step of the 8-point
	// algorithm unnecessary.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_ParameterizationIsRankTwo()
	{
		Quaterniond qU = new AngleAxisd(0.8, new Vector3d(0.3, -1, 0.5).Normalized()).ToQuaternion();
		Quaterniond qV = new AngleAxisd(-0.4, new Vector3d(1, 0.2, -0.7).Normalized()).ToQuaternion();
		const double sigma = 0.6;
		double[] parameters = Params(qU, qV, sigma);

		Matrix3d f = TinyFundamentalSampsonErrorCostFunctor.FundamentalFromParams(parameters);
		Vector3d singularValues = Svd3d.Compute(f).SingularValues;
		using (Assert.Multiple())
		{
			await Assert.That(singularValues.X).IsEqualTo(1.0).Within(1e-12);
			await Assert.That(singularValues.Y).IsEqualTo(sigma).Within(1e-12);
			await Assert.That(singularValues.Z).IsEqualTo(0.0).Within(1e-12);
		}
	}

	// The batched functor's squared residuals match ComputeSquaredSampsonError at
	// several points in the 9-parameter space.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_MatchesSquaredSampsonError()
	{
		Vector2d[] points1 = FundamentalTestPoints1();
		Vector2d[] points2 = FundamentalTestPoints2();
		var functor = new TinyFundamentalSampsonErrorCostFunctor(points1, points2);

		Quaterniond[] quaternionsU =
		[
			new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion(),
			new AngleAxisd(0.2, new Vector3d(0.4, 1, -0.3).Normalized()).ToQuaternion(),
		];
		Quaterniond[] quaternionsV =
		[
			new AngleAxisd(0.3, new Vector3d(0.2, -1, 0.7).Normalized()).ToQuaternion(),
			new AngleAxisd(-0.6, new Vector3d(1, 0.1, 0.2).Normalized()).ToQuaternion(),
		];
		double[] sigmas = [0.85, 0.35];

		var log = new ExpectationLog();
		for (int k = 0; k < sigmas.Length; ++k)
		{
			double[] parameters = Params(quaternionsU[k].Normalized(), quaternionsV[k].Normalized(), sigmas[k]);

			var residuals = new double[points1.Length];
			if (!log.True(functor.Evaluate(parameters, residuals, default), $"k={k}: functor succeeds"))
			{
				break;
			}

			var expected = new List<double>();
			EssentialMatrix.ComputeSquaredSampsonError(
				points1, points2, TinyFundamentalSampsonErrorCostFunctor.FundamentalFromParams(parameters), expected);
			for (int i = 0; i < points1.Length; ++i)
			{
				log.Near(residuals[i] * residuals[i], expected[i], 1e-12, $"k={k} residual[{i}]^2");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The closed-form 9-parameter Jacobian matches central finite differences.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_JacobianMatchesFiniteDifference()
	{
		Vector2d[] points1 = FundamentalTestPoints1();
		Vector2d[] points2 = FundamentalTestPoints2();
		var functor = new TinyFundamentalSampsonErrorCostFunctor(points1, points2);
		int n = points1.Length;

		Quaterniond qU = new AngleAxisd(0.7, new Vector3d(0.2, -1, 0.5).Normalized()).ToQuaternion();
		Quaterniond qV = new AngleAxisd(0.5, new Vector3d(-0.6, 0.3, 1).Normalized()).ToQuaternion();
		double[] p = Params(qU, qV, 0.7);

		var residuals = new double[n];
		var jacobian = new double[n * 9];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		const double kEps = 1e-6;
		var log = new ExpectationLog();
		for (int l = 0; l < 9; ++l)
		{
			double[] pPlus = (double[])p.Clone();
			double[] pMinus = (double[])p.Clone();
			pPlus[l] += kEps;
			pMinus[l] -= kEps;
			var resPlus = new double[n];
			var resMinus = new double[n];
			functor.Evaluate(pPlus, resPlus, default);
			functor.Evaluate(pMinus, resMinus, default);
			for (int i = 0; i < n; ++i)
			{
				double finiteDiff = (resPlus[i] - resMinus[i]) / (2 * kEps);
				log.Near(jacobian[i + l * n], finiteDiff, 1e-5, $"J({i}, {l})");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Image points shared by the fundamental matrix functor tests, in the
	// normalized frame the refiner solves in.
	private static Vector2d[] FundamentalTestPoints1() =>
		[new(0.10, 0.20), new(-0.30, 0.10), new(0.20, -0.25), new(-0.15, -0.35), new(0.40, 0.05)];

	private static Vector2d[] FundamentalTestPoints2() =>
		[new(0.15, -0.10), new(0.05, 0.30), new(-0.20, -0.15), new(0.25, 0.35), new(-0.05, 0.45)];

	// {qU.x, qU.y, qU.z, qU.w, qV.x, qV.y, qV.z, qV.w, sigma}.
	private static double[] Params(Quaterniond qU, Quaterniond qV, double sigma) =>
		[qU.X, qU.Y, qU.Z, qU.W, qV.X, qV.Y, qV.Z, qV.W, sigma];
}
