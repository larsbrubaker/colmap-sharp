// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/trust_region_minimizer_test.cc (BSD-3-Clause,
// see THIRD_PARTY_NOTICES.md).
//
// TrustRegionMinimizerTests (C#-only in the sense of CLAUDE.md: Ceres' test, not COLMAP's):
// ColmapSharp/Solver/TrustRegionMinimizer.cs and its neighbors, same names, values and
// tolerances.
// - PowellsSingularFunctionUsingLevenbergMarquardt: Ceres drives the minimizer with a
//   hand-written evaluator whose template flags hold some of Powell's four variables fixed,
//   and notes this "is equivalent to constructing a problem and using the SubsetManifold";
//   the port does exactly that, so the problem, the evaluator and Plus are exercised too.
//   Same 14 column subsets, DENSE_QR, radius 1e4 (max 1e20), tolerances 1e-26, and the
//   optimum within 0.001.
// - JacobiScalingTest, GradientToleranceConvergenceUpdatesStep: as in Ceres, through Problem
//   and Solve (the latter runs the projected line search of a bounded problem; Ceres'
//   ExpCostFunctor is autodiff, here its derivative -exp(x) is written out, the same value).
// Not ported: PowellsSingularFunctionUsingDogleg (the dogleg strategy is not ported; COLMAP
// always uses Levenberg-Marquardt).

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

/// <summary>Powell's function f = (x1 + 10 x2, sqrt5 (x3 - x4), (x2 - 2 x3)^2, sqrt10 (x1 - x4)^2).</summary>
internal sealed class PowellCostFunction() : CostFunction(4, 4)
{
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		ReadOnlySpan<double> x = parameters[0];
		double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
		residuals[0] = x1 + (10.0 * x2);
		residuals[1] = Math.Sqrt(5.0) * (x3 - x4);
		residuals[2] = Math.Pow(x2 - (2.0 * x3), 2.0);
		residuals[3] = Math.Sqrt(10.0) * Math.Pow(x1 - x4, 2.0);
		if (jacobians.IsEmpty || jacobians[0].Array is null)
		{
			return true;
		}

		// Row-major 4 x 4, the columns of Ceres' PowellEvaluator2.
		Span<double> j = jacobians[0];
		double[] rows =
		[
			1.0, 10.0, 0.0, 0.0,
			0.0, 0.0, Math.Sqrt(5.0), -Math.Sqrt(5.0),
			0.0, 2.0 * (x2 - (2.0 * x3)), 4.0 * ((2.0 * x3) - x2), 0.0,
			Math.Sqrt(10.0) * 2.0 * (x1 - x4), 0.0, 0.0, Math.Sqrt(10.0) * 2.0 * (x4 - x1),
		];
		rows.CopyTo(j);
		return true;
	}
}

public class TrustRegionMinimizerTests
{
	// Holds a subset of the columns fixed and checks if the solver converges to the optimal
	// values or not.
	private static async Task IsTrustRegionSolveSuccessful(bool col1, bool col2, bool col3, bool col4)
	{
		double[] parameters = [3, -1, 0, 1.0];

		// If the column is inactive, then set its value to the optimal value.
		bool[] active = [col1, col2, col3, col4];
		var constant = new List<int>();
		for (int i = 0; i < 4; i++)
		{
			if (!active[i])
			{
				parameters[i] = 0.0;
				constant.Add(i);
			}
		}

		var problem = new Problem();
		problem.AddResidualBlock(new PowellCostFunction(), null, parameters);
		if (constant.Count > 0)
		{
			problem.SetManifold(parameters, new SubsetManifold(4, constant));
		}

		var options = new SolverOptions
		{
			LinearSolverType = LinearSolverType.DenseQr,
			GradientTolerance = 1e-26,
			FunctionTolerance = 1e-26,
			ParameterTolerance = 1e-26,
			InitialTrustRegionRadius = 1e4,
			MaxTrustRegionRadius = 1e20,
			MinLmDiagonal = 1e-6,
			MaxLmDiagonal = 1e32,
		};
		LeastSquaresSolver.Solve(options, problem);

		// The minimum is at x1 = x2 = x3 = x4 = 0.
		await Assert.That(parameters[0]).IsEqualTo(0.0).Within(0.001);
		await Assert.That(parameters[1]).IsEqualTo(0.0).Within(0.001);
		await Assert.That(parameters[2]).IsEqualTo(0.0).Within(0.001);
		await Assert.That(parameters[3]).IsEqualTo(0.0).Within(0.001);
	}

	[Test]
	public async Task PowellsSingularFunctionUsingLevenbergMarquardt()
	{
		// This case is excluded because this has a local minimum and does not find the
		// optimum. This should not affect the correctness of this test since we are testing
		// all the other 14 combinations of column activations.
		//
		//   IsSolveSuccessful<true, true, false, true>();
		await IsTrustRegionSolveSuccessful(true, true, true, true);
		await IsTrustRegionSolveSuccessful(true, true, true, false);
		await IsTrustRegionSolveSuccessful(true, false, true, true);
		await IsTrustRegionSolveSuccessful(false, true, true, true);
		await IsTrustRegionSolveSuccessful(true, true, false, false);
		await IsTrustRegionSolveSuccessful(true, false, true, false);
		await IsTrustRegionSolveSuccessful(false, true, true, false);
		await IsTrustRegionSolveSuccessful(true, false, false, true);
		await IsTrustRegionSolveSuccessful(false, true, false, true);
		await IsTrustRegionSolveSuccessful(false, false, true, true);
		await IsTrustRegionSolveSuccessful(true, false, false, false);
		await IsTrustRegionSolveSuccessful(false, true, false, false);
		await IsTrustRegionSolveSuccessful(false, false, true, false);
		await IsTrustRegionSolveSuccessful(false, false, false, true);
	}

	// Ceres' CurveCostFunction: one residual, the target length minus the perimeter of the
	// closed polygon through the vertices (one 2-vector block each).
	private sealed class CurveCostFunction : CostFunction
	{
		// std::numeric_limits<double>::min(), the smallest normal double.
		private const double MinNormal = 2.2250738585072014E-308;

		private readonly int numVertices;
		private readonly double targetLength;

		public CurveCostFunction(int numVertices, double targetLength)
			: base(1, [.. Enumerable.Repeat(2, numVertices)])
		{
			this.numVertices = numVertices;
			this.targetLength = targetLength;
		}

		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			residuals[0] = targetLength;
			for (int i = 0; i < numVertices; ++i)
			{
				int prev = (numVertices + i - 1) % numVertices;
				double length = 0.0;
				for (int dim = 0; dim < 2; dim++)
				{
					double diff = parameters[prev][dim] - parameters[i][dim];
					length += diff * diff;
				}

				residuals[0] -= Math.Sqrt(length);
			}

			if (jacobians.IsEmpty)
			{
				return true;
			}

			for (int i = 0; i < numVertices; ++i)
			{
				if (jacobians[i].Array is null)
				{
					continue;
				}

				int prev = (numVertices + i - 1) % numVertices;
				int next = (i + 1) % numVertices;
				double[] u = new double[2];
				double[] v = new double[2];
				double normU = 0.0, normV = 0.0;
				for (int dim = 0; dim < 2; dim++)
				{
					u[dim] = parameters[i][dim] - parameters[prev][dim];
					normU += u[dim] * u[dim];
					v[dim] = parameters[next][dim] - parameters[i][dim];
					normV += v[dim] * v[dim];
				}

				normU = Math.Sqrt(normU);
				normV = Math.Sqrt(normV);
				Span<double> jacobian = jacobians[i];
				for (int dim = 0; dim < 2; dim++)
				{
					jacobian[dim] = 0.0;
					if (normU > MinNormal)
					{
						jacobian[dim] -= u[dim] / normU;
					}

					if (normV > MinNormal)
					{
						jacobian[dim] += v[dim] / normV;
					}
				}
			}

			return true;
		}
	}

	[Test]
	public async Task JacobiScalingTest()
	{
		const int N = 6;
		var y = new ArraySegment<double>[N];
		const double pi = 3.1415926535897932384626433;
		for (int i = 0; i < N; i++)
		{
			double theta = i * 2.0 * pi / N;
			y[i] = new double[] { Math.Cos(theta), Math.Sin(theta) };
		}

		var problem = new Problem();
		problem.AddResidualBlock(new CurveCostFunction(N, 10.0), null, y);
		var options = new SolverOptions { LinearSolverType = LinearSolverType.DenseQr };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.FinalCost).IsLessThanOrEqualTo(1e-10);
	}

	// residual = 10 - exp(x).
	private sealed class ExpCostFunction() : CostFunction(1, 1)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			double x = parameters[0][0];
			residuals[0] = 10.0 - Math.Exp(x);
			if (!jacobians.IsEmpty && jacobians[0].Array is not null)
			{
				jacobians[0].AsSpan()[0] = -Math.Exp(x);
			}

			return true;
		}
	}

	[Test]
	public async Task GradientToleranceConvergenceUpdatesStep()
	{
		double[] x = [5];
		var problem = new Problem();
		problem.AddResidualBlock(new ExpCostFunction(), null, x);
		problem.SetParameterLowerBound(x, 0, 3.0);
		var options = new SolverOptions();
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(Math.Abs(3.0 - x[0])).IsLessThanOrEqualTo(1e-12);
		double expectedFinalCost = 0.5 * Math.Pow(10.0 - Math.Exp(3.0), 2);
		await Assert.That(Math.Abs(expectedFinalCost - summary.FinalCost)).IsLessThanOrEqualTo(1e-12);
	}
}
