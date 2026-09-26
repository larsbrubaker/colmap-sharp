// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/problem_test.cc (Problem.SetAndGetParameter*
// Bound, the ProblemEvaluateTest cases), internal/ceres/parameter_block_test.cc (the bounds
// cases) and internal/ceres/evaluator_test_utils.cc (CompareEvaluations)
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ceres' tests, not COLMAP's: ColmapSharp/Solver/Problem.cs (bounds), Problem.Evaluate.cs
// and ParameterBlock.cs, same values and exact comparisons.
// - ProblemEvaluateTest runs each case with every combination of residuals, gradient and
//   Jacobian requested, as CheckAllEvaluationCombinations does; the CRS Jacobian is
//   expanded to a dense row-major matrix before comparing, like CRSToDenseMatrix.
// - The ParameterBlock tests build blocks through ParameterBlock's constructor, as Ceres does.
// Not ported: the rest of problem_test.cc and parameter_block_test.cc covers removal, the
// dynamic problem, EvaluateResidualBlock and evaluation callbacks (none used by COLMAP, none
// ported); the manifold cases of parameter_block_test.cc are covered by ManifoldTests and
// ProblemTests.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class CeresProblemBoundsTests
{
	[Test]
	public async Task SetAndGetParameterLowerBound()
	{
		var problem = new Problem();
		double[] x = [1.0, 2.0];
		problem.AddParameterBlock(x);

		await Assert.That(problem.GetParameterLowerBound(x, 0)).IsEqualTo(-double.MaxValue);
		await Assert.That(problem.GetParameterLowerBound(x, 1)).IsEqualTo(-double.MaxValue);

		problem.SetParameterLowerBound(x, 0, -1.0);
		await Assert.That(problem.GetParameterLowerBound(x, 0)).IsEqualTo(-1.0);
		await Assert.That(problem.GetParameterLowerBound(x, 1)).IsEqualTo(-double.MaxValue);

		problem.SetParameterLowerBound(x, 0, -2.0);
		await Assert.That(problem.GetParameterLowerBound(x, 0)).IsEqualTo(-2.0);
		await Assert.That(problem.GetParameterLowerBound(x, 1)).IsEqualTo(-double.MaxValue);

		problem.SetParameterLowerBound(x, 0, -double.MaxValue);
		await Assert.That(problem.GetParameterLowerBound(x, 0)).IsEqualTo(-double.MaxValue);
		await Assert.That(problem.GetParameterLowerBound(x, 1)).IsEqualTo(-double.MaxValue);
	}

	[Test]
	public async Task SetAndGetParameterUpperBound()
	{
		var problem = new Problem();
		double[] x = [1.0, 2.0];
		problem.AddParameterBlock(x);

		await Assert.That(problem.GetParameterUpperBound(x, 0)).IsEqualTo(double.MaxValue);
		await Assert.That(problem.GetParameterUpperBound(x, 1)).IsEqualTo(double.MaxValue);

		problem.SetParameterUpperBound(x, 0, -1.0);
		await Assert.That(problem.GetParameterUpperBound(x, 0)).IsEqualTo(-1.0);
		await Assert.That(problem.GetParameterUpperBound(x, 1)).IsEqualTo(double.MaxValue);

		problem.SetParameterUpperBound(x, 0, -2.0);
		await Assert.That(problem.GetParameterUpperBound(x, 0)).IsEqualTo(-2.0);
		await Assert.That(problem.GetParameterUpperBound(x, 1)).IsEqualTo(double.MaxValue);

		problem.SetParameterUpperBound(x, 0, double.MaxValue);
		await Assert.That(problem.GetParameterUpperBound(x, 0)).IsEqualTo(double.MaxValue);
		await Assert.That(problem.GetParameterUpperBound(x, 1)).IsEqualTo(double.MaxValue);
	}
}

public class CeresParameterBlockTests
{
	[Test]
	public async Task DefaultBounds()
	{
		double[] x = new double[2];
		var parameterBlock = new ParameterBlock(x, -1);
		await Assert.That(parameterBlock.UpperBoundForParameter(0)).IsEqualTo(double.MaxValue);
		await Assert.That(parameterBlock.UpperBoundForParameter(1)).IsEqualTo(double.MaxValue);
		await Assert.That(parameterBlock.LowerBoundForParameter(0)).IsEqualTo(-double.MaxValue);
		await Assert.That(parameterBlock.LowerBoundForParameter(1)).IsEqualTo(-double.MaxValue);
	}

	[Test]
	public async Task SetBounds()
	{
		double[] x = new double[2];
		var parameterBlock = new ParameterBlock(x, -1);
		parameterBlock.SetLowerBound(0, 1);
		parameterBlock.SetUpperBound(1, 1);

		await Assert.That(parameterBlock.LowerBoundForParameter(0)).IsEqualTo(1.0);
		await Assert.That(parameterBlock.LowerBoundForParameter(1)).IsEqualTo(-double.MaxValue);

		await Assert.That(parameterBlock.UpperBoundForParameter(0)).IsEqualTo(double.MaxValue);
		await Assert.That(parameterBlock.UpperBoundForParameter(1)).IsEqualTo(1.0);
	}

	[Test]
	public async Task PlusWithBoundsConstraints()
	{
		double[] x = [1.0, 0.0];
		double[] delta = [2.0, -10.0];
		var parameterBlock = new ParameterBlock(x, -1);
		parameterBlock.SetUpperBound(0, 2.0);
		parameterBlock.SetLowerBound(1, -1.0);
		double[] xPlusDelta = new double[2];
		parameterBlock.Plus(x, delta, xPlusDelta);
		await Assert.That(xPlusDelta[0]).IsEqualTo(2.0);
		await Assert.That(xPlusDelta[1]).IsEqualTo(-1.0);
	}
}

public class ProblemEvaluateTest
{
	private readonly Problem problem = new();
	private readonly double[] parameters = [1, 2, 3, 4, 5, 6];
	private readonly List<ArraySegment<double>> parameterBlocks = [];
	private readonly List<ResidualBlockId> residualBlocks = [];

	public ProblemEvaluateTest()
	{
		parameterBlocks.Add(new ArraySegment<double>(parameters, 0, 2));
		parameterBlocks.Add(new ArraySegment<double>(parameters, 2, 2));
		parameterBlocks.Add(new ArraySegment<double>(parameters, 4, 2));

		var costFunction = new QuadraticCostFunction(2, 2);

		// f(x, y)
		residualBlocks.Add(problem.AddResidualBlock(costFunction, null, parameterBlocks[0], parameterBlocks[1]));

		// g(y, z)
		residualBlocks.Add(problem.AddResidualBlock(costFunction, null, parameterBlocks[1], parameterBlocks[2]));

		// h(z, x)
		residualBlocks.Add(problem.AddResidualBlock(costFunction, null, parameterBlocks[2], parameterBlocks[0]));
	}

	// Rows/columns, cost, residuals, gradient, row-major Jacobian.
	private sealed record ExpectedEvaluation(int NumRows, int NumCols, double Cost, double[] Residuals, double[] Gradient, double[] Jacobian);

	private static readonly double[] AllResiduals = [-19.0, -35.0, -59.0, -87.0, -27.0, -43.0];

	private static readonly double[] AllGradient = [146.0, 484.0, 582.0, 1256.0, 1450.0, 2604.0];

	private static readonly double[] AllJacobian =
	[
		-2.0, 0.0, -12.0, 0.0, 0.0, 0.0,
		0.0, -4.0, 0.0, -16.0, 0.0, 0.0,
		0.0, 0.0, -6.0, 0.0, -20.0, 0.0,
		0.0, 0.0, 0.0, -8.0, 0.0, -24.0,
		-4.0, 0.0, 0.0, 0.0, -10.0, 0.0,
		0.0, -8.0, 0.0, 0.0, 0.0, -12.0,
	];

	[Test]
	public async Task MultipleParameterAndResidualBlocks()
	{
		await CheckAllEvaluationCombinations(new ProblemEvaluateOptions(), new ExpectedEvaluation(6, 6, 7607.0, AllResiduals, AllGradient, AllJacobian));
	}

	[Test]
	public async Task ParameterAndResidualBlocksPassedInOptions()
	{
		var options = new ProblemEvaluateOptions();
		options.ParameterBlocks.AddRange(parameterBlocks);
		options.ResidualBlocks.AddRange(residualBlocks);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(6, 6, 7607.0, AllResiduals, AllGradient, AllJacobian));
	}

	[Test]
	public async Task ReorderedResidualBlocks()
	{
		var options = new ProblemEvaluateOptions();
		options.ParameterBlocks.AddRange(parameterBlocks);

		// f, h, g
		options.ResidualBlocks.Add(residualBlocks[0]);
		options.ResidualBlocks.Add(residualBlocks[2]);
		options.ResidualBlocks.Add(residualBlocks[1]);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(
			6,
			6,
			7607.0,
			[-19.0, -35.0, -27.0, -43.0, -59.0, -87.0],
			AllGradient,
			[
				-2.0, 0.0, -12.0, 0.0, 0.0, 0.0,
				0.0, -4.0, 0.0, -16.0, 0.0, 0.0,
				-4.0, 0.0, 0.0, 0.0, -10.0, 0.0,
				0.0, -8.0, 0.0, 0.0, 0.0, -12.0,
				0.0, 0.0, -6.0, 0.0, -20.0, 0.0,
				0.0, 0.0, 0.0, -8.0, 0.0, -24.0,
			]));
	}

	[Test]
	public async Task ReorderedResidualBlocksAndReorderedParameterBlocks()
	{
		var options = new ProblemEvaluateOptions();

		// z, y, x
		options.ParameterBlocks.Add(parameterBlocks[2]);
		options.ParameterBlocks.Add(parameterBlocks[1]);
		options.ParameterBlocks.Add(parameterBlocks[0]);

		// f, h, g
		options.ResidualBlocks.Add(residualBlocks[0]);
		options.ResidualBlocks.Add(residualBlocks[2]);
		options.ResidualBlocks.Add(residualBlocks[1]);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(
			6,
			6,
			7607.0,
			[-19.0, -35.0, -27.0, -43.0, -59.0, -87.0],
			[1450.0, 2604.0, 582.0, 1256.0, 146.0, 484.0],
			[
				0.0, 0.0, -12.0, 0.0, -2.0, 0.0,
				0.0, 0.0, 0.0, -16.0, 0.0, -4.0,
				-10.0, 0.0, 0.0, 0.0, -4.0, 0.0,
				0.0, -12.0, 0.0, 0.0, 0.0, -8.0,
				-20.0, 0.0, -6.0, 0.0, 0.0, 0.0,
				0.0, -24.0, 0.0, -8.0, 0.0, 0.0,
			]));
	}

	[Test]
	public async Task ConstantParameterBlock()
	{
		problem.SetParameterBlockConstant(parameterBlocks[1]);
		await CheckAllEvaluationCombinations(new ProblemEvaluateOptions(), new ExpectedEvaluation(
			6,
			6,
			7607.0,
			AllResiduals,
			[146.0, 484.0, 0.0, 0.0, 1450.0, 2604.0],
			[
				-2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, -4.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, -20.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, -24.0,
				-4.0, 0.0, 0.0, 0.0, -10.0, 0.0,
				0.0, -8.0, 0.0, 0.0, 0.0, -12.0,
			]));
	}

	[Test]
	public async Task ExcludedAResidualBlock()
	{
		var options = new ProblemEvaluateOptions();
		options.ResidualBlocks.Add(residualBlocks[0]);
		options.ResidualBlocks.Add(residualBlocks[2]);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(
			4,
			6,
			2082.0,
			[-19.0, -35.0, -27.0, -43.0],
			[146.0, 484.0, 228.0, 560.0, 270.0, 516.0],
			[
				-2.0, 0.0, -12.0, 0.0, 0.0, 0.0,
				0.0, -4.0, 0.0, -16.0, 0.0, 0.0,
				-4.0, 0.0, 0.0, 0.0, -10.0, 0.0,
				0.0, -8.0, 0.0, 0.0, 0.0, -12.0,
			]));
	}

	[Test]
	public async Task ExcludedParameterBlock()
	{
		var options = new ProblemEvaluateOptions();

		// x, z
		options.ParameterBlocks.Add(parameterBlocks[0]);
		options.ParameterBlocks.Add(parameterBlocks[2]);
		options.ResidualBlocks.AddRange(residualBlocks);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(
			6,
			4,
			7607.0,
			AllResiduals,
			[146.0, 484.0, 1450.0, 2604.0],
			[
				-2.0, 0.0, 0.0, 0.0,
				0.0, -4.0, 0.0, 0.0,
				0.0, 0.0, -20.0, 0.0,
				0.0, 0.0, 0.0, -24.0,
				-4.0, 0.0, -10.0, 0.0,
				0.0, -8.0, 0.0, -12.0,
			]));
	}

	[Test]
	public async Task ExcludedParameterBlockAndExcludedResidualBlock()
	{
		var options = new ProblemEvaluateOptions();

		// x, z
		options.ParameterBlocks.Add(parameterBlocks[0]);
		options.ParameterBlocks.Add(parameterBlocks[2]);
		options.ResidualBlocks.Add(residualBlocks[0]);
		options.ResidualBlocks.Add(residualBlocks[1]);
		await CheckAllEvaluationCombinations(options, new ExpectedEvaluation(
			4,
			4,
			6318.0,
			[-19.0, -35.0, -59.0, -87.0],
			[38.0, 140.0, 1180.0, 2088.0],
			[
				-2.0, 0.0, 0.0, 0.0,
				0.0, -4.0, 0.0, 0.0,
				0.0, 0.0, -20.0, 0.0,
				0.0, 0.0, 0.0, -24.0,
			]));
	}

	[Test]
	public async Task Manifold()
	{
		problem.SetManifold(parameterBlocks[1], new SubsetManifold(2, [0]));
		await CheckAllEvaluationCombinations(new ProblemEvaluateOptions(), new ExpectedEvaluation(
			6,
			5,
			7607.0,
			AllResiduals,
			[146.0, 484.0, 1256.0, 1450.0, 2604.0],
			[
				-2.0, 0.0, 0.0, 0.0, 0.0,
				0.0, -4.0, -16.0, 0.0, 0.0,
				0.0, 0.0, 0.0, -20.0, 0.0,
				0.0, 0.0, -8.0, 0.0, -24.0,
				-4.0, 0.0, 0.0, -10.0, 0.0,
				0.0, -8.0, 0.0, 0.0, -12.0,
			]));
	}

	private async Task CheckAllEvaluationCombinations(ProblemEvaluateOptions options, ExpectedEvaluation expected)
	{
		for (int i = 0; i < 8; i++)
		{
			await EvaluateAndCompare(
				options,
				expected,
				(i & 1) != 0,
				(i & 2) != 0,
				(i & 4) != 0);
		}
	}

	private async Task EvaluateAndCompare(ProblemEvaluateOptions options, ExpectedEvaluation expected, bool wantResiduals, bool wantGradient, bool wantJacobian)
	{
		List<double>? residuals = wantResiduals ? [] : null;
		List<double>? gradient = wantGradient ? [] : null;
		CRSMatrix? jacobian = wantJacobian ? new CRSMatrix() : null;
		await Assert.That(problem.Evaluate(options, out double cost, residuals, gradient, jacobian)).IsTrue();

		// CompareEvaluations.
		await Assert.That(cost).IsEqualTo(expected.Cost);
		if (residuals is not null)
		{
			await Assert.That(residuals.Count).IsEqualTo(expected.NumRows);
			await Assert.That(residuals.SequenceEqual(expected.Residuals)).IsTrue();
		}

		if (gradient is not null)
		{
			await Assert.That(gradient.Count).IsEqualTo(expected.NumCols);
			await Assert.That(gradient.SequenceEqual(expected.Gradient)).IsTrue();
		}

		if (jacobian is not null)
		{
			await Assert.That(jacobian.NumRows).IsEqualTo(expected.NumRows);
			await Assert.That(jacobian.NumCols).IsEqualTo(expected.NumCols);

			// CRSToDenseMatrix, row-major.
			double[] dense = new double[jacobian.NumRows * jacobian.NumCols];
			for (int row = 0; row < jacobian.NumRows; row++)
			{
				for (int j = jacobian.Rows[row]; j < jacobian.Rows[row + 1]; j++)
				{
					dense[row * jacobian.NumCols + jacobian.Cols[j]] = jacobian.Values[j];
				}
			}

			await Assert.That(dense.SequenceEqual(expected.Jacobian)).IsTrue();
		}
	}

	// r_i = i - (j + 1) * x_ij^2
	private sealed class QuadraticCostFunction(int numResiduals, int numParameterBlocks)
		: CostFunction(numResiduals, Enumerable.Repeat(numResiduals, numParameterBlocks).ToArray())
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			int n = residuals.Length;
			for (int i = 0; i < n; i++)
			{
				residuals[i] = i;
				for (int j = 0; j < parameters.Length; j++)
				{
					residuals[i] -= (j + 1.0) * parameters[j][i] * parameters[j][i];
				}
			}

			if (jacobians.IsEmpty)
			{
				return true;
			}

			for (int j = 0; j < parameters.Length; j++)
			{
				if (jacobians[j].Array is null)
				{
					continue;
				}

				Span<double> jacobian = jacobians[j].AsSpan();
				jacobian.Clear();
				for (int i = 0; i < n; i++)
				{
					jacobian[i * n + i] = -2.0 * (j + 1.0) * parameters[j][i];
				}
			}

			return true;
		}
	}
}
