// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SparseCholeskyWithFallbackSolverTests: colmap/optim/sparse_cholesky_test.cc ported 1:1,
// one method per gtest TEST(Suite, Name) named Suite_Name, same expected values and
// tolerances. Tests ColmapSharp/Optim/SparseCholesky.cs (Tier B).
//
// IllConditionedChain draws its right-hand side from RandomUniformReal(-1, 1) per entry,
// which is what COLMAP's RandomEigenVectorXd does; the test only checks the residual, so
// the values themselves do not matter.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Optim;

public class SparseCholeskyWithFallbackSolverTests
{
	// 1D Laplacian of a chain of n nodes with the first diagonal entry increased by 1 to fix
	// the gauge (otherwise the matrix is rank-deficient). This is the pose-graph structure
	// that drives the rotation-averaging fix; it is PD but becomes ill-conditioned as n grows.
	private static SparseMatrixCsc ChainLaplacianGaugeFixed(int n)
	{
		var triplets = new List<SparseTriplet>();
		for (int i = 0; i < n; ++i)
		{
			double diag = 0;
			if (i > 0)
			{
				triplets.Add(new SparseTriplet(i, i - 1, -1));
				diag += 1;
			}

			if (i < n - 1)
			{
				triplets.Add(new SparseTriplet(i, i + 1, -1));
				diag += 1;
			}

			triplets.Add(new SparseTriplet(i, i, diag));
		}

		// Gauge fix: pin node 0.
		triplets.Add(new SparseTriplet(0, 0, 1));
		return SparseMatrixCsc.FromTriplets(n, n, triplets);
	}

	private static VectorXd LinSpaced(int size, double low, double high)
	{
		var v = new VectorXd(size);
		for (int i = 0; i < size; i++)
		{
			v[i] = size == 1 ? high : low + (high - low) * i / (size - 1);
		}

		return v;
	}

	private static bool AnyNaN(VectorXd v) => v.AsSpan().ToArray().Any(double.IsNaN);

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_ComputeAndSolveDiagonal()
	{
		var a = SparseMatrixCsc.FromTriplets(3, 3, [new(0, 0, 2), new(1, 1, 3), new(2, 2, 4)]);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsTrue();

		var b = new VectorXd([2, 6, 12]);
		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(EigenMatrixNear(x, new VectorXd([1, 2, 3]))).IsTrue();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_ComputeAndSolveChain()
	{
		SparseMatrixCsc a = ChainLaplacianGaugeFixed(10);
		VectorXd b = LinSpaced(a.Rows, 1, 10);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsTrue();

		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(EigenMatrixNear(a * x, b, 1e-9)).IsTrue();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_AnalyzeAndFactorizeReusedAcrossMatrices()
	{
		// Same sparsity, different numeric values — mirrors IRLS reuse pattern.
		SparseMatrixCsc a1 = ChainLaplacianGaugeFixed(8);
		SparseMatrixCsc a2 = a1.Clone();
		for (int i = 0; i < a2.Cols; ++i)
		{
			a2.CoeffRef(i, i) += 0.5;
		}

		var solver = new SparseCholeskyWithFallbackSolver();
		solver.AnalyzePattern(a1);

		VectorXd b = LinSpaced(a1.Rows, 1, 8);

		await Assert.That(solver.Factorize(a1)).IsTrue();
		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(EigenMatrixNear(a1 * x, b, 1e-9)).IsTrue();

		await Assert.That(solver.Factorize(a2)).IsTrue();
		await Assert.That(solver.Solve(b, out x)).IsTrue();
		await Assert.That(EigenMatrixNear(a2 * x, b, 1e-9)).IsTrue();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_ComputeReturnsFalseOnSingularMatrix()
	{
		// 2x2 rank-1 matrix: [[1,1],[1,1]]. Singular, so both supernodal and LDLT must detect
		// the failure and Compute must return false rather than NaN.
		var a = SparseMatrixCsc.FromTriplets(2, 2, [new(0, 0, 1), new(0, 1, 1), new(1, 0, 1), new(1, 1, 1)]);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsFalse();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_RidgeMakesSingularMatrixSolvable()
	{
		// Same singular matrix as above with a small ridge added to the diagonal is PD and
		// must factorize successfully.
		var a = SparseMatrixCsc.FromTriplets(
			2, 2, [new(0, 0, 1 + 1e-6), new(0, 1, 1), new(1, 0, 1), new(1, 1, 1 + 1e-6)]);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsTrue();

		var b = new VectorXd([2, 2]);
		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(AnyNaN(x)).IsFalse();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_FallsBackToLdltOnIndefiniteMatrix()
	{
		// diag(1, 1, -1e-20) is mathematically indefinite. Supernodal LLT fundamentally
		// requires strictly positive pivots (it takes square roots), so CHOLMOD reliably
		// rejects this across versions. SimplicialLDLT accepts indefinite matrices by allowing
		// negative entries in D. The wrapper must fall back transparently and Solve must
		// return the correct (exact for a diagonal system) solution rather than NaN.
		var a = SparseMatrixCsc.FromTriplets(3, 3, [new(0, 0, 1), new(1, 1, 1), new(2, 2, -1e-20)]);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsTrue();

		var b = new VectorXd([2, 3, -5e-20]);
		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(EigenMatrixNear(x, new VectorXd([2, 3, 5]), 1e-10)).IsTrue();
	}

	// C#-only (no counterpart in sparse_cholesky_test.cc): once the solver has fallen back
	// to LDLT, a Compute on a matrix with a different pattern must re-analyze the LDLT
	// stage too, not factorize the new values against the old pattern.
	[Test]
	public async Task SparseCholeskyWithFallbackSolver_ComputeNewPatternAfterFallback()
	{
		var indefinite = SparseMatrixCsc.FromTriplets(3, 3, [new(0, 0, 1), new(1, 1, 1), new(2, 2, -1e-20)]);
		SparseMatrixCsc chain = ChainLaplacianGaugeFixed(6);
		VectorXd b = LinSpaced(chain.Rows, 1, 6);

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(indefinite)).IsTrue();
		await Assert.That(solver.UsesLdlt).IsTrue();

		await Assert.That(solver.Compute(chain)).IsTrue();
		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(EigenMatrixNear(chain * x, b, 1e-9)).IsTrue();
	}

	[Test]
	public async Task SparseCholeskyWithFallbackSolver_IllConditionedChain()
	{
		// Long chain Laplacian + gauge fix. Mathematically PD but condition number grows as
		// O(n^2). Exercises the regime that motivated the fallback.
		SparseMatrixCsc a = ChainLaplacianGaugeFixed(500);
		var b = new VectorXd(a.Rows);
		for (int i = 0; i < b.Length; i++)
		{
			b[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		var solver = new SparseCholeskyWithFallbackSolver();
		await Assert.That(solver.Compute(a)).IsTrue();

		await Assert.That(solver.Solve(b, out VectorXd x)).IsTrue();
		await Assert.That(AnyNaN(x)).IsFalse();
		await Assert.That(EigenMatrixNear(a * x, b, 1e-6)).IsTrue();
	}
}
