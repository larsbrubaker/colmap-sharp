// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Examples ported from Ceres Solver 2.2.0 examples/helloworld.cc, examples/powell.cc and
// examples/curve_fitting.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md), with the expected
// outputs Ceres' tutorial (docs/source/nnls_tutorial.rst) prints for them.
//
// CeresExampleTests (C#-only: neither COLMAP nor Ceres has these as tests). Whole solves
// through Problem, AutoDiffCostFunction and LeastSquaresSolver, Tier C against the tutorial:
// - HelloWorld: min 1/2 (10 - x)^2 from x = 0.5 converges to 10 in two steps, and the
//   BriefReport has Ceres' exact shape and initial cost.
// - Powell: Powell's singular function as four scalar blocks; the tutorial's final point
//   (six significant digits) and iteration count, DENSE_QR.
// - CurveFitting: y = exp(m x + c) on the tutorial's 67 points; the tutorial's m, c and
//   final cost (six significant digits), with 5 rejected steps on the way.
// Powell and CurveFitting also run with the other linear solvers, which must match too.
// The tutorial's BriefReport shows "Iterations: 2" / "Iterations: 13"; Ceres 2.2 counts
// iteration 0 too (num_successful_steps + num_unsuccessful_steps over every iteration, which
// is what its FullReport for Powell shows: 15 iterations for rows 0..14), so the counts
// asserted here are the number of rows in the tutorial's trace.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class CeresExampleTests
{
	// examples/curve_fitting.cc: (x, y) pairs.
	private static readonly double[] CurveData =
	[
		0.000000e+00, 1.133898e+00,
		7.500000e-02, 1.334902e+00,
		1.500000e-01, 1.213546e+00,
		2.250000e-01, 1.252016e+00,
		3.000000e-01, 1.392265e+00,
		3.750000e-01, 1.314458e+00,
		4.500000e-01, 1.472541e+00,
		5.250000e-01, 1.536218e+00,
		6.000000e-01, 1.355679e+00,
		6.750000e-01, 1.463566e+00,
		7.500000e-01, 1.490201e+00,
		8.250000e-01, 1.658699e+00,
		9.000000e-01, 1.067574e+00,
		9.750000e-01, 1.464629e+00,
		1.050000e+00, 1.402653e+00,
		1.125000e+00, 1.713141e+00,
		1.200000e+00, 1.527021e+00,
		1.275000e+00, 1.702632e+00,
		1.350000e+00, 1.423899e+00,
		1.425000e+00, 1.543078e+00,
		1.500000e+00, 1.664015e+00,
		1.575000e+00, 1.732484e+00,
		1.650000e+00, 1.543296e+00,
		1.725000e+00, 1.959523e+00,
		1.800000e+00, 1.685132e+00,
		1.875000e+00, 1.951791e+00,
		1.950000e+00, 2.095346e+00,
		2.025000e+00, 2.361460e+00,
		2.100000e+00, 2.169119e+00,
		2.175000e+00, 2.061745e+00,
		2.250000e+00, 2.178641e+00,
		2.325000e+00, 2.104346e+00,
		2.400000e+00, 2.584470e+00,
		2.475000e+00, 1.914158e+00,
		2.550000e+00, 2.368375e+00,
		2.625000e+00, 2.686125e+00,
		2.700000e+00, 2.712395e+00,
		2.775000e+00, 2.499511e+00,
		2.850000e+00, 2.558897e+00,
		2.925000e+00, 2.309154e+00,
		3.000000e+00, 2.869503e+00,
		3.075000e+00, 3.116645e+00,
		3.150000e+00, 3.094907e+00,
		3.225000e+00, 2.471759e+00,
		3.300000e+00, 3.017131e+00,
		3.375000e+00, 3.232381e+00,
		3.450000e+00, 2.944596e+00,
		3.525000e+00, 3.385343e+00,
		3.600000e+00, 3.199826e+00,
		3.675000e+00, 3.423039e+00,
		3.750000e+00, 3.621552e+00,
		3.825000e+00, 3.559255e+00,
		3.900000e+00, 3.530713e+00,
		3.975000e+00, 3.561766e+00,
		4.050000e+00, 3.544574e+00,
		4.125000e+00, 3.867945e+00,
		4.200000e+00, 4.049776e+00,
		4.275000e+00, 3.885601e+00,
		4.350000e+00, 4.110505e+00,
		4.425000e+00, 4.345320e+00,
		4.500000e+00, 4.161241e+00,
		4.575000e+00, 4.363407e+00,
		4.650000e+00, 4.161576e+00,
		4.725000e+00, 4.619728e+00,
		4.800000e+00, 4.737410e+00,
		4.875000e+00, 4.727863e+00,
		4.950000e+00, 4.669206e+00,
	];

	private readonly struct HelloWorldFunctor : IAutoDiffFunctor
	{
		public bool Evaluate<T>(ReadOnlySpan<T> x, Span<T> residual)
			where T : struct, IScalar<T>
		{
			residual[0] = 10.0 - x[0];
			return true;
		}
	}

	// Powell's F1..F4, each over two scalar blocks.
	private readonly struct PowellFunctor(int term) : IAutoDiffFunctor
	{
		public bool Evaluate<T>(ReadOnlySpan<T> p, Span<T> residual)
			where T : struct, IScalar<T>
		{
			residual[0] = term switch
			{
				1 => p[0] + (10.0 * p[1]),
				2 => Math.Sqrt(5.0) * (p[0] - p[1]),
				3 => (p[0] - (2.0 * p[1])) * (p[0] - (2.0 * p[1])),
				_ => Math.Sqrt(10.0) * (p[0] - p[1]) * (p[0] - p[1]),
			};
			return true;
		}
	}

	private readonly struct ExponentialResidual(double x, double y) : IAutoDiffFunctor
	{
		public bool Evaluate<T>(ReadOnlySpan<T> p, Span<T> residual)
			where T : struct, IScalar<T>
		{
			residual[0] = y - T.Exp((p[0] * x) + p[1]);
			return true;
		}
	}

	public static IEnumerable<LinearSolverType> AllLinearSolvers() =>
		[
			LinearSolverType.DenseQr,
			LinearSolverType.DenseNormalCholesky,
			LinearSolverType.SparseNormalCholesky,
			LinearSolverType.DenseSchur,
			LinearSolverType.SparseSchur,
			LinearSolverType.IterativeSchur,
		];

	[Test]
	public async Task HelloWorld()
	{
		double[] x = [0.5];
		var problem = new Problem();
		problem.AddResidualBlock(new AutoDiffCostFunction<HelloWorldFunctor, Grad1>(default, 1, 1), null, x);
		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), problem);

		// Tutorial: "x : 0.5 -> 10" (six significant digits); rows 0..2 with costs
		// 4.512500e+01, 4.511598e-07, 5.012552e-16. The last step stops about 3.2e-8 short of
		// 10, so the final cost is matched rather than x to more digits than printed.
		await Assert.That(x[0]).IsEqualTo(10.0).Within(5e-6);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Convergence);
		await Assert.That(summary.Iterations.Count).IsEqualTo(3);
		await Assert.That(summary.InitialCost).IsEqualTo(45.125);
		await Assert.That(summary.Iterations[1].Cost).IsEqualTo(4.511598e-07).Within(5e-14);
		await Assert.That(summary.FinalCost).IsEqualTo(5.012552e-16).Within(5e-23);
		string report = summary.BriefReport();
		await Assert.That(report).StartsWith("Ceres Solver Report: Iterations: 3, Initial cost: 4.512500e+01, Final cost: ");
		await Assert.That(report).EndsWith(", Termination: CONVERGENCE");
	}

	[Test]
	[MethodDataSource(nameof(AllLinearSolvers))]
	public async Task Powell(LinearSolverType linearSolver)
	{
		double[] x1 = [3.0], x2 = [-1.0], x3 = [0.0], x4 = [1.0];
		var problem = new Problem();
		problem.AddResidualBlock(PowellTerm(1), null, x1, x2);
		problem.AddResidualBlock(PowellTerm(2), null, x3, x4);
		problem.AddResidualBlock(PowellTerm(3), null, x2, x3);
		problem.AddResidualBlock(PowellTerm(4), null, x1, x4);
		var options = new SolverOptions { MaxNumIterations = 100, LinearSolverType = linearSolver };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);

		// Tutorial: "Final x1 = 0.000146222, x2 = -1.46222e-05, x3 = 2.40957e-05,
		// x4 = 2.40957e-05", initial cost 1.075000e+02, final 1.120029e-15, 15 iterations,
		// all successful, gradient tolerance.
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Convergence);
		await Assert.That(summary.Message).StartsWith("Gradient tolerance reached.");
		await Assert.That(summary.InitialCost).IsEqualTo(1.075000e+02).Within(5e-5);
		await Assert.That(summary.NumSuccessfulSteps).IsEqualTo(15);
		await Assert.That(summary.NumUnsuccessfulSteps).IsEqualTo(0);
		await Assert.That(summary.FinalCost).IsEqualTo(1.120029e-15).Within(5e-22);
		await Assert.That(x1[0]).IsEqualTo(0.000146222).Within(5e-10);
		await Assert.That(x2[0]).IsEqualTo(-1.46222e-05).Within(5e-11);
		await Assert.That(x3[0]).IsEqualTo(2.40957e-05).Within(5e-11);
		await Assert.That(x4[0]).IsEqualTo(2.40957e-05).Within(5e-11);
	}

	private static AutoDiffCostFunction<PowellFunctor, Grad2> PowellTerm(int term) => new(new PowellFunctor(term), 1, 1, 1);

	/// <summary>
	/// The solvers that reproduce the tutorial's trajectory. ITERATIVE_SCHUR cannot: curve
	/// fitting's reduced system is 1 x 1, so CG's first step solves it exactly and the
	/// residual is often exactly zero. Ceres' CG then reports "Numerical failure. rho = r'z =
	/// 0" on its second iteration, which makes the LM step invalid; the solve ends in
	/// FAILURE with m and c left at 0 (CurveFitting_IterativeSchurFailsOnItsOneByOneReducedSystem).
	/// </summary>
	public static IEnumerable<LinearSolverType> DirectLinearSolvers() =>
		AllLinearSolvers().Where(type => type != LinearSolverType.IterativeSchur);

	[Test]
	[MethodDataSource(nameof(DirectLinearSolvers))]
	public async Task CurveFitting(LinearSolverType linearSolver)
	{
		double[] m = [0.0];
		double[] c = [0.0];
		var problem = new Problem();
		for (int i = 0; i < CurveData.Length / 2; ++i)
		{
			problem.AddResidualBlock(
				new AutoDiffCostFunction<ExponentialResidual, Grad2>(new ExponentialResidual(CurveData[2 * i], CurveData[(2 * i) + 1]), 1, 1, 1),
				null,
				m,
				c);
		}

		var options = new SolverOptions { MaxNumIterations = 25, LinearSolverType = linearSolver };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);

		// Tutorial: initial cost 1.211734e+02, final 1.056751e+00, m 0.291861, c 0.131439,
		// rows 0..13 with 5 rejected steps (1..5).
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Convergence);
		await Assert.That(summary.InitialCost).IsEqualTo(1.211734e+02).Within(5e-5);
		await Assert.That(summary.FinalCost).IsEqualTo(1.056751).Within(5e-7);
		await Assert.That(m[0]).IsEqualTo(0.291861).Within(5e-7);
		await Assert.That(c[0]).IsEqualTo(0.131439).Within(5e-7);
		await Assert.That(summary.Iterations.Count).IsEqualTo(14);
		await Assert.That(summary.NumUnsuccessfulSteps).IsEqualTo(5);
		await Assert.That(summary.NumResidualsReduced).IsEqualTo(67);
	}

	// C#-only: pins the ITERATIVE_SCHUR outcome on curve fitting described at
	// DirectLinearSolvers. Every step whose CG hits rho = r'z = 0 is invalid; after
	// max_num_consecutive_invalid_steps (5) of them the solve fails and the parameters are
	// restored to their starting values.
	[Test]
	public async Task CurveFitting_IterativeSchurFailsOnItsOneByOneReducedSystem()
	{
		double[] m = [0.0];
		double[] c = [0.0];
		var problem = new Problem();
		for (int i = 0; i < CurveData.Length / 2; ++i)
		{
			problem.AddResidualBlock(
				new AutoDiffCostFunction<ExponentialResidual, Grad2>(new ExponentialResidual(CurveData[2 * i], CurveData[(2 * i) + 1]), 1, 1, 1),
				null,
				m,
				c);
		}

		var options = new SolverOptions { MaxNumIterations = 25, LinearSolverType = LinearSolverType.IterativeSchur };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);

		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message)
			.IsEqualTo("Number of consecutive invalid steps more than Solver::Options::max_num_consecutive_invalid_steps: 5");
		await Assert.That(summary.IsSolutionUsable).IsFalse();
		await Assert.That(summary.InitialCost).IsEqualTo(1.211734e+02).Within(5e-5);
		await Assert.That(summary.Iterations[1].StepIsValid).IsFalse();
		await Assert.That(summary.Iterations[^1].StepIsValid).IsFalse();
		await Assert.That(m[0]).IsEqualTo(0.0);
		await Assert.That(c[0]).IsEqualTo(0.0);
	}
}
