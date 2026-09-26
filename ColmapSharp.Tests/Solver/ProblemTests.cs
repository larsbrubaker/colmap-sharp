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

	// Points p0..p3 each tied to a target and to two of the cameras c0, c1; c0 is anchored.
	private static (Problem Problem, double[][] Points, double[][] Cameras) PointsAndCameras()
	{
		var problem = new Problem();
		double[][] points = [[1.0], [2.0], [3.0], [4.0]];
		double[][] cameras = [[0.5], [-0.5]];
		for (int i = 0; i < points.Length; i++)
		{
			problem.AddResidualBlock(new Offset(1, i * 1.5), null, points[i]);
			problem.AddResidualBlock(new Difference(), null, points[i], cameras[0]);
			problem.AddResidualBlock(new Difference(), null, points[i], cameras[1]);
		}

		problem.AddResidualBlock(new Offset(1, 0.25), null, cameras[0]);
		return (problem, points, cameras);
	}

	private static ParameterBlockOrdering PointsFirst(double[][] points, double[][] cameras)
	{
		var ordering = new ParameterBlockOrdering();
		foreach (double[] point in points)
		{
			ordering.AddElementToGroup(point, 0);
		}

		foreach (double[] camera in cameras)
		{
			ordering.AddElementToGroup(camera, 1);
		}

		return ordering;
	}

	// C#-only: a user ordering with two groups (the global positioner's shape) eliminates the
	// first group and reaches the dense solver's optimum.
	[Test]
	[Arguments(LinearSolverType.SparseSchur)]
	[Arguments(LinearSolverType.DenseSchur)]
	[Arguments(LinearSolverType.IterativeSchur)]
	public async Task UserOrdering_SchurSolverMatchesDenseQr(LinearSolverType type)
	{
		(Problem reference, double[][] expected, _) = PointsAndCameras();
		LeastSquaresSolver.Solve(Tight(), reference);

		(Problem problem, double[][] points, double[][] cameras) = PointsAndCameras();
		SolverOptions options = Tight();
		options.LinearSolverType = type;
		options.LinearSolverOrdering = PointsFirst(points, cameras);
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.IsSolutionUsable).IsTrue();
		await Assert.That(summary.LinearSolverTypeUsed).IsEqualTo(type);
		for (int i = 0; i < points.Length; i++)
		{
			await Assert.That(Math.Abs(points[i][0] - expected[i][0])).IsLessThanOrEqualTo(1e-7);
		}
	}

	// C#-only: Ceres rejects a first elimination group that is not an independent set.
	[Test]
	public async Task UserOrdering_DependentFirstGroup_Fails()
	{
		(Problem problem, double[][] points, double[][] cameras) = PointsAndCameras();
		ParameterBlockOrdering ordering = PointsFirst(points, cameras);
		ordering.AddElementToGroup(cameras[0], 0);
		var options = new SolverOptions { LinearSolverType = LinearSolverType.SparseSchur, LinearSolverOrdering = ordering };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message)
			.IsEqualTo("The first elimination group in the parameter block ordering of size 5 is not an independent set");
	}

	// C#-only: when every block of the first group is constant, Ceres switches SPARSE_SCHUR to
	// SPARSE_NORMAL_CHOLESKY; unlike Ceres the caller's ordering is left as it was.
	[Test]
	public async Task UserOrdering_ConstantFirstGroup_SwitchesSolver()
	{
		(Problem problem, double[][] points, double[][] cameras) = PointsAndCameras();
		foreach (double[] point in points)
		{
			problem.SetParameterBlockConstant(point);
		}

		ParameterBlockOrdering ordering = PointsFirst(points, cameras);
		var options = new SolverOptions { LinearSolverType = LinearSolverType.SparseSchur, LinearSolverOrdering = ordering };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.IsSolutionUsable).IsTrue();
		await Assert.That(summary.LinearSolverTypeUsed).IsEqualTo(LinearSolverType.SparseNormalCholesky);
		await Assert.That(ordering.NumElements).IsEqualTo(6);
	}

	// C#-only: Ceres' IsFeasible rejects a variable block whose bounds leave no room.
	[Test]
	public async Task InfeasibleBounds_Fail()
	{
		double[] x = [1.0];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(1, 0.0), null, x);
		problem.SetParameterLowerBound(x, 0, 2.0);
		problem.SetParameterUpperBound(x, 0, 2.0);
		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message).StartsWith("ParameterBlock with size 1 has at least one infeasible bound.");
		await Assert.That(x[0]).IsEqualTo(1.0);
	}

	// C#-only: the start is projected onto the bounds and every step stays inside them, so a
	// minimum outside the box ends on its face.
	[Test]
	public async Task Bounds_ProjectStartAndSteps()
	{
		double[] x = [-3.0, 0.5];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(2, -5.0), null, x);
		problem.SetParameterLowerBound(x, 0, 0.0);
		problem.SetParameterLowerBound(x, 1, -1.0);
		SolverSummary summary = LeastSquaresSolver.Solve(Tight(), problem);
		await Assert.That(summary.IsSolutionUsable).IsTrue();
		await Assert.That(summary.IsConstrained).IsTrue();
		await Assert.That(x[0]).IsEqualTo(0.0);
		await Assert.That(x[1]).IsEqualTo(-1.0);
		foreach (IterationSummary iteration in summary.Iterations)
		{
			await Assert.That(iteration.Cost).IsGreaterThanOrEqualTo(0.5 * (25.0 + 16.0));
		}
	}

	// C#-only: Ceres CHECK-fails on an empty ordering; the port fails the solve instead
	// (docs/CPP_DIVERGENCES.md entry 37).
	[Test]
	public async Task UserOrdering_Empty_Fails()
	{
		(Problem problem, double[][] points, _) = PointsAndCameras();
		var options = new SolverOptions { LinearSolverType = LinearSolverType.SparseSchur, LinearSolverOrdering = new ParameterBlockOrdering() };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message).Contains("linear_solver_ordering is empty");
		await Assert.That(points[0][0]).IsEqualTo(1.0);
	}

	// C#-only: an ordering left empty by the reduction (all its blocks constant) CHECK-fails
	// in Ceres; the port fails the solve.
	[Test]
	public async Task UserOrdering_OnlyConstantBlocks_Fails()
	{
		(Problem problem, double[][] points, _) = PointsAndCameras();
		var ordering = new ParameterBlockOrdering();
		foreach (double[] point in points)
		{
			problem.SetParameterBlockConstant(point);
			ordering.AddElementToGroup(point, 0);
		}

		var options = new SolverOptions { LinearSolverType = LinearSolverType.SparseSchur, LinearSolverOrdering = ordering };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Failure);
		await Assert.That(summary.Message).Contains("only parameter blocks that are constant or unused");
	}

	// C#-only: as in Ceres' ProgramEvaluator, an evaluation that asks for a gradient counts as
	// a Jacobian evaluation, so a bounded solve's CUBIC line search trials land there; with
	// QUADRATIC interpolation they ask for no gradient and count as residual evaluations.
	[Test]
	[Arguments(LineSearchInterpolationType.Cubic)]
	[Arguments(LineSearchInterpolationType.Quadratic)]
	public async Task BoundedSolve_CountsLineSearchEvaluations(LineSearchInterpolationType interpolation)
	{
		double[] x = [-3.0, 0.5];
		var problem = new Problem();
		problem.AddResidualBlock(new Offset(2, -5.0), null, x);
		problem.SetParameterLowerBound(x, 0, 0.0);
		problem.SetParameterLowerBound(x, 1, -1.0);
		SolverOptions options = Tight();
		options.LineSearchInterpolationType = interpolation;
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		await Assert.That(summary.IsSolutionUsable).IsTrue();

		// Iteration 0 and every accepted step evaluate the Jacobian; every valid step
		// evaluates its candidate's cost; each line search evaluates its first trial plus one
		// per backtracking iteration (none of these searches gives up).
		int validSteps = summary.Iterations.Count(i => i.Iteration > 0 && i.StepIsValid);
		int lineSearchTrials = validSteps + summary.NumLineSearchSteps;
		bool cubic = interpolation == LineSearchInterpolationType.Cubic;
		await Assert.That(summary.NumJacobianEvaluations).IsEqualTo(summary.NumSuccessfulSteps + (cubic ? lineSearchTrials : 0));
		await Assert.That(summary.NumResidualEvaluations).IsEqualTo(validSteps + (cubic ? 0 : lineSearchTrials));
	}
}
