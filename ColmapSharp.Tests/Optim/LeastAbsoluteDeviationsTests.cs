// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// LeastAbsoluteDeviationsTests: colmap/optim/least_absolute_deviations_test.cc ported 1:1.
// The TEST_P suite ParameterizedLeastAbsoluteDeviationsTests, instantiated over both solver
// types, becomes one method per case named Suite_Name with the solver type as [Arguments].
// Same expected values and tolerances. Tests ColmapSharp/Optim/LeastAbsoluteDeviations.cs
// (Tier C over the Tier B sparse Cholesky).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

using SolverType = ColmapSharp.Optim.LeastAbsoluteDeviationSolver.Options.SolverType;

namespace ColmapSharp.Tests.Optim;

public class LeastAbsoluteDeviationsTests
{
	private static LeastAbsoluteDeviationSolver.Options GetOptions(SolverType type) => new() { Solver = type };

	private static double L1Norm(VectorXd v) => v.AsSpan().ToArray().Sum(Math.Abs);

	// A(i, j) = i * cols + j + 1 with A(0, 0) = 10, the matrix of the two "determined" cases.
	private static SparseMatrixCsc CountingMatrix(int rows, int cols)
	{
		var triplets = new List<SparseTriplet>();
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				triplets.Add(new SparseTriplet(i, j, i * cols + j + 1));
			}
		}

		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(rows, cols, triplets);
		a.CoeffRef(0, 0) = 10;
		return a;
	}

	private static VectorXd OneToN(int n)
	{
		var b = new VectorXd(n);
		for (int i = 0; i < n; ++i)
		{
			b[i] = i + 1;
		}

		return b;
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_OverDetermined(SolverType type)
	{
		SparseMatrixCsc a = CountingMatrix(4, 3);
		VectorXd b = OneToN(a.Rows);
		VectorXd x = VectorXd.Zero(a.Cols);

		var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
		await Assert.That(solver.Solve(b, x)).IsTrue();

		// Reference solution obtained with Boyd's Matlab implementation.
		await Assert.That(EigenMatrixNear(x, new VectorXd([0, 0, 1 / 3.0]))).IsTrue();

		VectorXd residual = a * x - b;
		await Assert.That(residual.Norm()).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_WellDetermined(SolverType type)
	{
		SparseMatrixCsc a = CountingMatrix(3, 3);
		VectorXd b = OneToN(a.Rows);
		VectorXd x = VectorXd.Zero(a.Cols);

		var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
		await Assert.That(solver.Solve(b, x)).IsTrue();

		// Reference solution obtained with Boyd's Matlab implementation.
		await Assert.That(EigenMatrixNear(x, new VectorXd([0, 0, 1 / 3.0]))).IsTrue();

		VectorXd residual = a * x - b;
		await Assert.That(residual.Norm()).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_UnderDetermined(SolverType type)
	{
		// In this case, the system is rank-deficient and not positive semi-definite.
		SparseMatrixCsc a = SparseMatrixCsc.Zero(2, 3);
		await Assert.That(() => new LeastAbsoluteDeviationSolver(GetOptions(type), a))
			.Throws<InvalidOperationException>();
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_SimpleOverdeterminedSystem(SolverType type)
	{
		// The explicit zeros are inserted as in the C++ test, so they are part of the pattern.
		var a = SparseMatrixCsc.FromTriplets(4, 3,
		[
			new(0, 0, 1.0), new(0, 1, 0.0), new(0, 2, 0.0),
			new(1, 0, 0.0), new(1, 1, 1.0), new(1, 2, 0.0),
			new(2, 0, 0.0), new(2, 1, 0.0), new(2, 2, 1.0),
			new(3, 0, 1.0), new(3, 1, 1.0), new(3, 2, 1.0),
		]);

		var b = new VectorXd([1.0, 2.0, 3.0, 6.0]);

		VectorXd x = VectorXd.Zero(3);
		await Assert.That(L1Norm(a * x - b)).IsGreaterThan(1e-1);

		var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
		solver.Solve(b, x);
		await Assert.That(L1Norm(a * x - b)).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_DiagonalSystem(SolverType type)
	{
		const int n = 5;
		var triplets = new List<SparseTriplet>();
		for (int i = 0; i < n; ++i)
		{
			triplets.Add(new SparseTriplet(i, i, i + 1.0));
		}

		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(n, n, triplets);

		var b = new VectorXd(n);
		for (int i = 0; i < n; ++i)
		{
			b[i] = (i + 1.0) * (i + 2.0);
		}

		LeastAbsoluteDeviationSolver.Options options = GetOptions(type);
		options.MaxNumIterations = 100;

		VectorXd x = VectorXd.Zero(n);
		var solver = new LeastAbsoluteDeviationSolver(options, a);
		solver.Solve(b, x);

		// For diagonal systems, the L1 solution should be close to x_i = b_i / A_ii
		var expected = new VectorXd(n);
		for (int i = 0; i < n; ++i)
		{
			expected[i] = b[i] / a[i, i];
		}

		await Assert.That(L1Norm(x - expected)).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_OverdeterminedWithOutliers(SolverType type)
	{
		var triplets = new List<SparseTriplet>();
		for (int i = 0; i < 6; ++i)
		{
			triplets.Add(new SparseTriplet(i, 0, 1.0));
			triplets.Add(new SparseTriplet(i, 1, i));
		}

		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(6, 2, triplets);

		// Linear relationship b = 2 + 3*x, but with outliers
		var b = new VectorXd([2.0, 5.0, 8.0, 11.0, 1000.0, 17.0]); // b[4] is an outlier

		LeastAbsoluteDeviationSolver.Options options = GetOptions(type);
		options.MaxNumIterations = 1000;

		VectorXd x = VectorXd.Zero(2);
		var solver = new LeastAbsoluteDeviationSolver(options, a);
		solver.Solve(b, x);

		// The L1 solution should be more robust to the outlier than L2
		// Expected solution is approximately [2, 3]
		await Assert.That(x[0]).IsEqualTo(2.0).Within(1e-3);
		await Assert.That(x[1]).IsEqualTo(3.0).Within(1e-3);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_IdentityMatrix(SolverType type)
	{
		// Test with identity matrix - trivial case
		const int n = 4;
		SparseMatrixCsc a = SparseMatrixCsc.Identity(n);

		var b = new VectorXd([1.0, 2.0, 3.0, 4.0]);

		VectorXd x = VectorXd.Zero(n);
		var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
		solver.Solve(b, x);

		// Solution should be exactly b for identity matrix
		await Assert.That(EigenMatrixNear(x, b, 1e-3)).IsTrue();
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_ScaledIdentityMatrix(SolverType type)
	{
		// Test with scaled identity matrix
		const int n = 3;
		const double scale = 5.0;
		var triplets = new List<SparseTriplet>();
		for (int i = 0; i < n; ++i)
		{
			triplets.Add(new SparseTriplet(i, i, scale));
		}

		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(n, n, triplets);

		var b = new VectorXd([5.0, 10.0, 15.0]);

		VectorXd x = VectorXd.Zero(n);
		var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
		solver.Solve(b, x);

		// Solution should be b / scale
		VectorXd expected = b / scale;
		await Assert.That(EigenMatrixNear(x, expected, 1e-3)).IsTrue();
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_ToleranceSettings(SolverType type)
	{
		// Test that tighter tolerances produce more accurate results
		// Create a well-conditioned matrix with full column rank
		var a = SparseMatrixCsc.FromTriplets(5, 3,
		[
			new(0, 0, 3.0), new(0, 1, 0.5), new(0, 2, 0.2),
			new(1, 0, 0.5), new(1, 1, 2.5), new(1, 2, 0.3),
			new(2, 0, 0.2), new(2, 1, 0.3), new(2, 2, 2.0),
			new(3, 0, 1.0), new(3, 1, 1.5), new(3, 2, 0.5),
			new(4, 0, 0.7), new(4, 1, 0.6), new(4, 2, 1.8),
		]);

		var b = new VectorXd([1.0, 2.0, 3.0, 4.0, 5.0]);

		// Loose tolerance
		VectorXd x1 = VectorXd.Zero(3);
		{
			LeastAbsoluteDeviationSolver.Options options = GetOptions(type);
			options.AbsoluteTolerance = 1e-1;
			options.RelativeTolerance = 1e-1;

			var solver = new LeastAbsoluteDeviationSolver(options, a);
			solver.Solve(b, x1);
		}

		// Tight tolerance
		VectorXd x2 = VectorXd.Zero(3);
		{
			LeastAbsoluteDeviationSolver.Options options = GetOptions(type);
			options.AbsoluteTolerance = 1e-6;
			options.RelativeTolerance = 1e-4;
			options.MaxNumIterations = 2000;

			var solver = new LeastAbsoluteDeviationSolver(options, a);
			solver.Solve(b, x2);
		}

		// The tighter tolerance solution should have lower or equal residual
		double residual1 = L1Norm(a * x1 - b);
		double residual2 = L1Norm(a * x2 - b);
		await Assert.That(residual2).IsLessThan(0.99 * residual1);
	}

	[Test]
	[Arguments(SolverType.SimplicialLLT)]
	[Arguments(SolverType.SupernodalCholmodLLT)]
	public async Task ParameterizedLeastAbsoluteDeviationsTests_RidgeRegularization(SolverType type)
	{
		// Singular matrix (rank 1, two identical columns) makes A^T A not positive definite.
		// With ridge regularization, the solver still factorizes A^T A and Solve returns a
		// finite solution (no NaN). Without regularization, the solver should detect failure
		// and return false.
		var a = SparseMatrixCsc.FromTriplets(3, 2,
		[
			new(0, 0, 1.0), new(0, 1, 1.0),
			new(1, 0, 2.0), new(1, 1, 2.0),
			new(2, 0, 3.0), new(2, 1, 3.0),
		]);

		var b = new VectorXd([2.0, 4.0, 6.0]);

		// Without regularization, the singular A^T A is not factorizable.
		{
			var solver = new LeastAbsoluteDeviationSolver(GetOptions(type), a);
			await Assert.That(solver.Valid).IsFalse();
			VectorXd x = VectorXd.Zero(2);
			await Assert.That(solver.Solve(b, x)).IsFalse();
		}

		// With regularization, factorization succeeds and Solve returns a finite (non-NaN)
		// solution.
		{
			LeastAbsoluteDeviationSolver.Options options = GetOptions(type);
			options.RidgeRegularization = 1e-9;
			var solver = new LeastAbsoluteDeviationSolver(options, a);
			await Assert.That(solver.Valid).IsTrue();
			VectorXd x = VectorXd.Zero(2);
			await Assert.That(solver.Solve(b, x)).IsTrue();
			await Assert.That(x.AsSpan().ToArray().Any(double.IsNaN)).IsFalse();

			// Either column achieves residual ~ 0 since b lies in span(A.col(0)).
			await Assert.That(L1Norm(a * x - b)).IsLessThanOrEqualTo(1e-3);
		}
	}
}
