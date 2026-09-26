// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinySolverTests: colmap/optim/tiny_solver_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Optim/TinySolver.cs, the TinySphereManifold of
// Estimators/CostFunctions/TinyManifold.cs, and Solver/TinySolverAutoDiffFunction.cs (the
// ceres::TinySolverAutoDiffFunction the manifold case wraps its functor in). Tier C: the
// convergence outcome, with COLMAP's bounds.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class TinySolverTests
{
	// A = [[1, 0], [0, 1], [1, 1]] and b = (1, 2, 4) of the linear case.
	private static readonly MatrixXd A = MatrixXd.FromRowMajor(3, 2, [1, 0, 0, 1, 1, 1]);
	private static readonly VectorXd B = new([1, 2, 4]);

	// Linear least squares: residual = A * p - b, with analytic Jacobian.
	private readonly struct LinearResidual : ITinySolverFunction
	{
		public static int NumParameters => 2;

		public int NumResiduals => 3;

		public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian)
		{
			VectorXd res = A * new VectorXd(parameters[..2]) - B;
			res.AsSpan().CopyTo(residuals);
			if (!jacobian.IsEmpty)
			{
				// Column-major, NUM_RESIDUALS x NUM_PARAMETERS.
				A.AsSpan().CopyTo(jacobian);
			}

			return true;
		}
	}

	[Test]
	public async Task TinySolver_EuclideanConvergesToNormalEquationsSolution()
	{
		VectorXd expected = new LDLT(A.TransposeTimesSelf()).Solve(A.TransposeTimes(B));

		var solver = TinySolver.Create<LinearResidual>();
		var options = new TinySolverOptions
		{
			GradientTolerance = 0,
			ParameterTolerance = 0,
			FunctionTolerance = 0,
			MaxNumIterations = 100,
		};
		double[] x = [0, 0];
		TinySolverSummary summary = solver.Solve(new LinearResidual(), x, options);

		await Assert.That(summary.Status).IsNotEqualTo(TinySolverStatus.CostFunctionFailed);
		await Assert.That((new VectorXd(x) - expected).Norm()).IsLessThan(1e-9);

		// Started from x = 0, so initial_cost = 0.5 * ||b||^2 = 0.5 * 21.
		await Assert.That(Math.Abs(summary.InitialCost - 10.5)).IsLessThanOrEqualTo(1e-9);

		// Cost must not increase, and at the least-squares optimum J'f(x) vanishes.
		await Assert.That(summary.FinalCost).IsGreaterThanOrEqualTo(0.0);
		await Assert.That(summary.FinalCost).IsLessThanOrEqualTo(summary.InitialCost);
		await Assert.That(Math.Abs(summary.GradientMaxNorm - 0.0)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(summary.Iterations).IsGreaterThanOrEqualTo(1);
		await Assert.That(summary.Iterations).IsLessThanOrEqualTo(options.MaxNumIterations);
	}

	// Autodiff functor minimizing ||x - target||^2; on the unit sphere the optimum is
	// target.normalized().
	private readonly struct SphereFitResidual(Vector3d target) : IAutoDiffFunctor
	{
		public Vector3d Target { get; } = target;

		public bool Evaluate<T>(ReadOnlySpan<T> x, Span<T> residuals)
			where T : struct, IScalar<T>
		{
			residuals[0] = x[0] - T.FromDouble(Target.X);
			residuals[1] = x[1] - T.FromDouble(Target.Y);
			residuals[2] = x[2] - T.FromDouble(Target.Z);
			return true;
		}
	}

	[Test]
	public async Task TinySolver_ManifoldConvergesAndStaysOnManifold()
	{
		var functor = new SphereFitResidual(new Vector3d(0.3, -0.7, 0.5));

		var f = new TinySolverAutoDiffFunction<SphereFitResidual, Grad3>(functor, 3);

		var solver = new TinySolver<TinySolverAutoDiffFunction<SphereFitResidual, Grad3>, TinySphereManifold>(default);
		var options = new TinySolverOptions
		{
			GradientTolerance = 0,
			ParameterTolerance = 0,
			FunctionTolerance = 0,
			MaxNumIterations = 100,
		};
		double[] x = [1, 0, 0]; // Unit seed.
		solver.Solve(f, x, options);

		var xv = new Vector3d(x[0], x[1], x[2]);
		await Assert.That(Math.Abs(xv.Norm - 1.0)).IsLessThanOrEqualTo(1e-9);
		await Assert.That((xv - functor.Target.Normalized()).Norm).IsLessThan(1e-6);
	}

	// A functor that always fails to evaluate.
	private readonly struct FailingResidual : ITinySolverFunction
	{
		public static int NumParameters => 2;

		public int NumResiduals => 2;

		public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian) => false;
	}

	[Test]
	public async Task TinySolver_ReportsCostFunctionFailure()
	{
		var solver = TinySolver.Create<FailingResidual>();
		double[] x = [1, 2];
		TinySolverSummary summary = solver.Solve(new FailingResidual(), x);
		await Assert.That(summary.Status).IsEqualTo(TinySolverStatus.CostFunctionFailed);
	}
}
