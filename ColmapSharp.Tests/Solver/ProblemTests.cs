// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ProblemTests (C#-only; Ceres' problem_test.cc is mostly about removal and evaluation APIs
// COLMAP does not use, and COLMAP has no test of its own): the bookkeeping and validation of
// ColmapSharp/Solver/Problem.cs and the preprocessing in LeastSquaresSolver.cs, with Ceres'
// behavior for each case (same errors, same fixed-cost and reduction rules).

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class ProblemTests
{
	// r = x - target, one residual per parameter.
	private sealed class Offset(int size, double target) : CostFunction(size, size)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			ReadOnlySpan<double> x = parameters[0];
			for (int i = 0; i < NumResiduals; i++)
			{
				residuals[i] = x[i] - target;
			}

			if (!jacobians.IsEmpty && jacobians[0].Array is not null)
			{
				Span<double> j = jacobians[0];
				j.Clear();
				for (int i = 0; i < NumResiduals; i++)
				{
					j[(i * NumResiduals) + i] = 1.0;
				}
			}

			return true;
		}
	}

	// r = x0 - x1 over two blocks of one value each.
	private sealed class Difference() : CostFunction(1, 1, 1)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			residuals[0] = parameters[0][0] - parameters[1][0];
			if (!jacobians.IsEmpty)
			{
				if (jacobians[0].Array is not null)
				{
					jacobians[0].AsSpan()[0] = 1.0;
				}

				if (jacobians[1].Array is not null)
				{
					jacobians[1].AsSpan()[0] = -1.0;
				}
			}

			return true;
		}
	}

	// Levenberg-Marquardt's first step on a linear problem is damped by the initial radius
	// (1e4), and Ceres' default function tolerance (1e-6) then stops it about 1e-4 short of
	// the solution; the tighter tolerance lets it get within the 1e-7 checked here.
	private static SolverOptions Tight() => new() { LinearSolverType = LinearSolverType.DenseQr, FunctionTolerance = 1e-15 };

	[Test]
	public async Task SlicesOfOneArray_AreDistinctBlocks()
	{
		double[] buffer = new double[6];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(3, 1.0), null, new ArraySegment<double>(buffer, 0, 3));
		problem.AddResidualBlock(new Offset(3, 2.0), null, new ArraySegment<double>(buffer, 3, 3));
		await Assert.That(problem.NumParameterBlocks).IsEqualTo(2);
		await Assert.That(problem.NumResiduals).IsEqualTo(6);
		await Assert.That(problem.HasParameterBlock(new ArraySegment<double>(buffer, 3, 3))).IsTrue();
		await Assert.That(problem.HasParameterBlock(new ArraySegment<double>(buffer, 1, 3))).IsFalse();

		SolverSummary summary = LeastSquaresSolver.Solve(Tight(), problem);
		await Assert.That(summary.IsSolutionUsable).IsTrue();
		double[] expected = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
		for (int i = 0; i < 6; i++)
		{
			await Assert.That(buffer[i]).IsEqualTo(expected[i]).Within(1e-7);
		}
	}

	[Test]
	public async Task OverlappingBlocks_Throw()
	{
		double[] buffer = new double[6];
		var problem = new Problem();
		problem.AddParameterBlock(new ArraySegment<double>(buffer, 0, 3));
		await Assert.That(() => problem.AddParameterBlock(new ArraySegment<double>(buffer, 2, 3))).Throws<ArgumentException>();
		await Assert.That(() => problem.AddParameterBlock(new ArraySegment<double>(buffer, 0, 4))).Throws<ArgumentException>();

		// Adding the same block again with the same size is allowed and changes nothing.
		problem.AddParameterBlock(new ArraySegment<double>(buffer, 0, 3));
		await Assert.That(problem.NumParameterBlocks).IsEqualTo(1);
	}

	[Test]
	public async Task InvalidResidualBlocks_Throw()
	{
		double[] x = [0.0];
		double[] y = [0.0];
		var problem = new Problem();
		await Assert.That(() => problem.AddResidualBlock(new Difference(), null, x, x)).Throws<ArgumentException>();
		await Assert.That(() => problem.AddResidualBlock(new Difference(), null, x)).Throws<ArgumentException>();
		await Assert.That(() => problem.SetParameterBlockConstant(y)).Throws<ArgumentException>();
		await Assert.That(() => problem.AddResidualBlock(new Offset(2, 0.0), null, x)).Throws<ArgumentException>();
	}

	[Test]
	public async Task ConstantBlocks_MoveTheirCostToFixedCost()
	{
		double[] a = [3.0];
		double[] b = [1.0];
		double[] c = [5.0];
		var problem = new Problem();
		problem.AddResidualBlock(new Difference(), null, a, b);
		problem.AddResidualBlock(new Difference(), null, b, c);
		problem.AddResidualBlock(new Offset(1, 0.0), null, c);
		problem.SetParameterBlockConstant(a);
		problem.SetParameterBlockConstant(c);
		await Assert.That(problem.IsParameterBlockConstant(a)).IsTrue();
		await Assert.That(problem.IsParameterBlockConstant(b)).IsFalse();

		SolverSummary summary = LeastSquaresSolver.Solve(Tight(), problem);

		// c's own residual (5^2 / 2) only touches a constant block.
		await Assert.That(summary.FixedCost).IsEqualTo(12.5);
		await Assert.That(summary.NumResidualBlocks).IsEqualTo(3);
		await Assert.That(summary.NumResidualBlocksReduced).IsEqualTo(2);
		await Assert.That(summary.NumParameterBlocksReduced).IsEqualTo(1);

		// b settles between a and c: cost (3 - 4)^2 / 2 + (4 - 5)^2 / 2 + 12.5.
		await Assert.That(b[0]).IsEqualTo(4.0).Within(1e-7);
		await Assert.That(summary.FinalCost).IsEqualTo(13.5).Within(1e-12);
		await Assert.That(a[0]).IsEqualTo(3.0);
		await Assert.That(c[0]).IsEqualTo(5.0);

		// Variable again, the next solve moves every block.
		problem.SetParameterBlockVariable(a);
		problem.SetParameterBlockVariable(c);
		summary = LeastSquaresSolver.Solve(Tight(), problem);
		await Assert.That(summary.NumParameterBlocksReduced).IsEqualTo(3);
		await Assert.That(summary.FinalCost).IsLessThan(1e-12);
	}

	[Test]
	public async Task ZeroDimensionalManifold_MakesTheBlockConstant()
	{
		double[] x = [1.0, 2.0];
		var problem = new Problem();
		problem.AddParameterBlock(x, new SubsetManifold(2, [0, 1]));
		problem.AddResidualBlock(new Offset(2, 0.0), null, x);
		await Assert.That(problem.IsParameterBlockConstant(x)).IsTrue();
		await Assert.That(problem.ParameterBlockTangentSize(x)).IsEqualTo(0);

		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Convergence);
		await Assert.That(summary.Message).IsEqualTo("Function tolerance reached. No non-constant parameter blocks found.");
		await Assert.That(summary.InitialCost).IsEqualTo(2.5);
		await Assert.That(summary.FinalCost).IsEqualTo(2.5);
	}

	[Test]
	public async Task InvalidOptions_FailWithCeresMessage()
	{
		double[] x = [1.0];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(1, 0.0), null, x);
		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions { NumThreads = 0 }, problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message).IsEqualTo(
			"Invalid configuration. Solver::Options::num_threads = 0. Violated constraint: Solver::Options::num_threads > 0");
	}

	[Test]
	public async Task NonFiniteParameters_Fail()
	{
		double[] x = [double.NaN];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(1, 0.0), null, x);
		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.IsSolutionUsable).IsFalse();
		await Assert.That(double.IsNaN(x[0])).IsTrue();
	}

	// A cost function whose evaluation throws.
	private sealed class Throwing() : CostFunction(1, 1)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians) =>
			throw new InvalidOperationException("cost function failed");
	}

	// Parallel.For wraps a worker's exception in an AggregateException; the evaluator must
	// surface the cost function's own exception for every thread count.
	[Test]
	[Arguments(1)]
	[Arguments(4)]
	public async Task CostFunctionException_SurfacesUnwrapped(int numThreads)
	{
		var problem = new Problem();
		var blocks = new double[10][];
		for (int i = 0; i < blocks.Length; i++)
		{
			blocks[i] = [1.0];
			problem.AddResidualBlock(i == 7 ? new Throwing() : new Offset(1, 0.0), null, blocks[i]);
		}

		var options = new SolverOptions { NumThreads = numThreads };
		await Assert.That(() => LeastSquaresSolver.Solve(options, problem))
			.Throws<InvalidOperationException>()
			.WithMessage("cost function failed");
	}
}
