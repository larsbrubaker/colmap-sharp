// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/conjugate_gradients_solver_test.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ConjugateGradientsSolverTests (Ceres' test, not COLMAP's): ColmapSharp/Solver/
// IterativeSchurSolver.cs's ConjugateGradients on a 3x3 identity and a 3x3 SPD system with
// the identity preconditioner, same options, starting points and expectations. The
// TripletSparseMatrix operators become DenseSparseMatrix (same products); ASSERT_DOUBLE_EQ
// is gtest's 4-ULP comparison.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class ConjugateGradientsSolverTests
{
	// gtest's AlmostEquals: at most 4 units in the last place apart.
	private static bool DoubleEq(double expected, double actual)
	{
		// Sign-and-magnitude bits mapped onto one ordered integer line (-0 and +0 meet at 0).
		static long Ordered(double v)
		{
			long bits = BitConverter.DoubleToInt64Bits(v);
			return bits < 0 ? -(bits & long.MaxValue) : bits;
		}

		return !double.IsNaN(expected) && !double.IsNaN(actual) && Math.Abs(Ordered(expected) - Ordered(actual)) <= 4;
	}

	private static readonly ConjugateGradientsSolverOptions Options = new(
		MinNumIterations: 1, MaxNumIterations: 10, ResidualResetPeriod: 20, QTolerance: 0.0, RTolerance: 1e-9);

	[Test]
	public async Task Solves3x3IdentitySystem()
	{
		var a = new DenseSparseMatrix(3, 3);
		for (int i = 0; i < 3; i++)
		{
			a.Matrix[i, i] = 1.0;
		}

		double[] b = [1.0, 2.0, 3.0];
		double[] x = [1.0, 1.0, 1.0];

		LinearSolverSummary summary = ConjugateGradients.Solve(Options, a, b, new IdentityOperator(), x);

		await Assert.That(summary.TerminationType).IsEqualTo(LinearSolverTerminationType.Success);
		await Assert.That(summary.NumIterations).IsEqualTo(1);
		await Assert.That(DoubleEq(1, x[0])).IsTrue();
		await Assert.That(DoubleEq(2, x[1])).IsTrue();
		await Assert.That(DoubleEq(3, x[2])).IsTrue();
	}

	[Test]
	public async Task Solves3x3SymmetricSystem()
	{
		//      | 2  -1  0|
		//  A = |-1   2 -1| is symmetric positive definite.
		//      | 0  -1  2|
		var a = new DenseSparseMatrix(3, 3);
		double[] values = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				a.Matrix[i, j] = values[(3 * i) + j];
			}
		}

		double[] b = [-1.0, 0.0, 3.0];
		double[] x = [1.0, 1.0, 1.0];

		LinearSolverSummary summary = ConjugateGradients.Solve(Options, a, b, new IdentityOperator(), x);

		await Assert.That(summary.TerminationType).IsEqualTo(LinearSolverTerminationType.Success);
		await Assert.That(DoubleEq(0, x[0])).IsTrue();
		await Assert.That(DoubleEq(1, x[1])).IsTrue();
		await Assert.That(DoubleEq(2, x[2])).IsTrue();
	}
}
