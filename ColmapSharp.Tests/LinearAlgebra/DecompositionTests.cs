// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DecompositionTests (C#-only; COLMAP has no test for Eigen itself): PartialPivLU, LLT,
// LDLT, HouseholderQR and ColPivHouseholderQR in ColmapSharp/LinearAlgebra. Each
// decomposition must reconstruct its input from its factors and its Solve must leave a
// small residual; small hand-checked cases pin the conventions (pivot choice, reflector
// signs, rank threshold, non-positive-definite detection). Tier B: tolerances are 1e-12
// relative (isApprox's default) for reconstructions of well-conditioned O(1) inputs.
// numpy agreement is in DecompositionOracleTests.

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class DecompositionTests
{
	private static readonly MatrixXd General = MatrixXd.FromRowMajor(4, 4, [
		0.5, -1.2, 2.0, 0.3,
		1.7, 0.4, -0.6, 1.1,
		-0.9, 2.2, 0.8, -1.4,
		0.2, -0.3, 1.5, 0.9]);

	private static readonly MatrixXd Tall = MatrixXd.FromRowMajor(6, 3, [
		1.0, 0.2, -0.5,
		0.3, -1.1, 0.7,
		-0.8, 0.4, 1.3,
		0.6, 0.9, -0.2,
		-0.1, -0.7, 0.4,
		1.2, 0.5, 0.8]);

	private static MatrixXd Spd()
	{
		MatrixXd gram = Tall.TransposeTimesSelf();
		return gram + MatrixXd.Identity(3) * 0.1;
	}

	private static double Residual(MatrixXd a, VectorXd x, VectorXd b) => (a * x - b).Norm();

	[Test]
	public async Task PartialPivLU_ReconstructsSolvesAndInverts()
	{
		var lu = new PartialPivLU(General);
		var b = new VectorXd([1, -2, 0.5, 3]);
		VectorXd x = lu.Solve(b);
		using (Assert.Multiple())
		{
			await Assert.That((lu.PermutationP() * General).IsApprox(lu.MatrixL() * lu.MatrixU())).IsTrue();
			await Assert.That(Residual(General, x, b)).IsLessThan(1e-13);
			await Assert.That((General * lu.Inverse()).IsApprox(MatrixXd.Identity(4))).IsTrue();
			// The first row's pivot is the largest |entry| of column 0 (1.7, row 1).
			await Assert.That(lu.PermutationIndices()[0]).IsEqualTo(1);
		}
	}

	[Test]
	public async Task PartialPivLU_DeterminantMatchesClosedForm()
	{
		Matrix3d m3 = new(2, -1, 0.5, 3, 4, -2, 1, 0.25, 5);
		using (Assert.Multiple())
		{
			await Assert.That(MatrixXd.From(m3).Determinant()).IsEqualTo(m3.Determinant()).Within(1e-12);
			await Assert.That(MatrixXd.From(General.ToMatrix4d()).Determinant()).IsEqualTo(General.ToMatrix4d().Determinant()).Within(1e-12);
			await Assert.That(MatrixXd.FromRowMajor(2, 2, [0, 1, 1, 0]).Determinant()).IsEqualTo(-1.0);
		}
	}

	[Test]
	public async Task LLT_ReconstructsAndSolves()
	{
		MatrixXd a = Spd();
		var llt = new LLT(a);
		var b = new VectorXd([0.3, -1, 2]);
		MatrixXd l = llt.MatrixL();
		using (Assert.Multiple())
		{
			await Assert.That(llt.Info).IsEqualTo(ComputationInfo.Success);
			await Assert.That((l * llt.MatrixU()).IsApprox(a)).IsTrue();
			await Assert.That(l[0, 1]).IsEqualTo(0.0);
			await Assert.That(Residual(a, llt.Solve(b), b)).IsLessThan(1e-13);
		}
	}

	[Test]
	public async Task LLT_ReportsNotPositiveDefinite()
	{
		var indefinite = MatrixXd.FromRowMajor(2, 2, [1, 2, 2, 1]);
		await Assert.That(new LLT(indefinite).Info).IsEqualTo(ComputationInfo.NumericalIssue);
	}

	[Test]
	public async Task LDLT_ReconstructsAndSolves()
	{
		// Symmetric indefinite but with a nonsingular diagonally pivoted LDL^T, plus the
		// SPD case; A = P^T L D L^T P.
		var indefinite = MatrixXd.FromRowMajor(3, 3, [1, 0.5, 0.2, 0.5, -3, 0.4, 0.2, 0.4, 2]);
		foreach (MatrixXd a in new[] { Spd(), indefinite })
		{
			var ldlt = new LDLT(a);
			MatrixXd p = ldlt.PermutationP();
			MatrixXd reconstructed = p.Transpose() * ldlt.MatrixL() * MatrixXd.FromDiagonal(ldlt.VectorD())
				* ldlt.MatrixL().Transpose() * p;
			var b = new VectorXd([1, 2, -0.5]);
			using (Assert.Multiple())
			{
				await Assert.That(ldlt.Info).IsEqualTo(ComputationInfo.Success);
				await Assert.That(reconstructed.IsApprox(a)).IsTrue();
				await Assert.That(Residual(a, ldlt.Solve(b), b)).IsLessThan(1e-13);
			}
		}

		var indefiniteLdlt = new LDLT(indefinite);
		using (Assert.Multiple())
		{
			// |-3| is the largest diagonal entry, so it is pivoted first.
			await Assert.That(indefiniteLdlt.Transpositions()[0]).IsEqualTo(1);
			await Assert.That(indefiniteLdlt.IsPositive()).IsFalse();
			await Assert.That(indefiniteLdlt.IsNegative()).IsFalse();
			await Assert.That(new LDLT(Spd()).IsPositive()).IsTrue();
		}
	}

	[Test]
	public async Task LDLT_SemidefiniteSolveIsConsistent()
	{
		// Rank-1 PSD matrix v v^T: D has one nonzero entry, and a b in the range is solved.
		var v = MatrixXd.FromColumnMajor(3, 1, [1, 2, -1]);
		MatrixXd a = v * v.Transpose();
		var ldlt = new LDLT(a);
		VectorXd b = v.Col(0) * 3.0;
		using (Assert.Multiple())
		{
			await Assert.That(ldlt.VectorD()[1]).IsEqualTo(0.0);
			await Assert.That(ldlt.VectorD()[2]).IsEqualTo(0.0);
			await Assert.That(Residual(a, ldlt.Solve(b), b)).IsLessThan(1e-13);
		}
	}

	[Test]
	public async Task LDLT_ZeroPivotWithNonzeroColumnIsNumericalIssue()
	{
		// [[0, 1], [1, 0]] has no diagonally pivoted LDL^T: both diagonal entries are 0 but
		// the off-diagonal is not. tiny_solver rejects a step when info() != Success.
		var a = MatrixXd.FromRowMajor(2, 2, [0, 1, 1, 0]);
		await Assert.That(new LDLT(a).Info).IsEqualTo(ComputationInfo.NumericalIssue);
	}

	[Test]
	public async Task HouseholderQR_ReconstructsAndIsOrthogonal()
	{
		foreach (MatrixXd a in new[] { General, Tall, Tall.Transpose() })
		{
			var qr = new HouseholderQR(a);
			MatrixXd q = qr.HouseholderQ();
			using (Assert.Multiple())
			{
				await Assert.That((q * qr.MatrixR()).IsApprox(a)).IsTrue();
				await Assert.That(q.IsUnitary()).IsTrue();
				await Assert.That(qr.MatrixR().IsUpperTriangular()).IsTrue();
			}
		}
	}

	[Test]
	public async Task HouseholderQR_SignConventionAndLeastSquares()
	{
		// LAPACK/Eigen convention: R(0,0) = -sign(a00) * ||a0||.
		var column = MatrixXd.FromColumnMajor(3, 1, [3, 0, 4]);
		var qr = new HouseholderQR(column);
		var b = new VectorXd([1, 0, -1, 2, 0.5, 1]);
		VectorXd x = new HouseholderQR(Tall).Solve(b);

		// Normal equations residual A^T (A x - b) vanishes at the least-squares solution.
		VectorXd normalResidual = Tall.TransposeTimes(Tall * x - b);
		using (Assert.Multiple())
		{
			await Assert.That(qr.MatrixR()[0, 0]).IsEqualTo(-5.0).Within(1e-15);
			await Assert.That(qr.HouseholderQ()[0, 0]).IsEqualTo(-0.6).Within(1e-15);
			await Assert.That(normalResidual.Norm()).IsLessThan(1e-13);
			await Assert.That(new HouseholderQR(MatrixXd.FromColumnMajor(2, 1, [0, 0])).HCoeffs()[0]).IsEqualTo(0.0);
		}
	}

	[Test]
	public async Task ColPivHouseholderQR_ReconstructsRankAndSolve()
	{
		var qr = new ColPivHouseholderQR(Tall);
		MatrixXd reconstructed = qr.HouseholderQ() * qr.MatrixR();
		var b = new VectorXd([1, 0, -1, 2, 0.5, 1]);
		VectorXd x = qr.Solve(b);
		VectorXd expected = new HouseholderQR(Tall).Solve(b);

		// Rank-deficient: the third column is the sum of the first two.
		MatrixXd deficient = Tall.Clone();
		deficient.SetCol(2, Tall.Col(0) + Tall.Col(1));
		using (Assert.Multiple())
		{
			await Assert.That(reconstructed.IsApprox(Tall * qr.ColsPermutation())).IsTrue();
			await Assert.That(qr.Rank()).IsEqualTo(3);
			await Assert.That((x - expected).Norm()).IsLessThan(1e-13);
			await Assert.That(new ColPivHouseholderQR(deficient).Rank()).IsEqualTo(2);
			await Assert.That(new ColPivHouseholderQR(MatrixXd.Zero(3, 3)).Rank()).IsEqualTo(0);
			await Assert.That(new ColPivHouseholderQR(MatrixXd.Identity(3, 5)).Rank()).IsEqualTo(3);
		}

		// The largest-norm column is pivoted first.
		var scaled = MatrixXd.FromRowMajor(2, 3, [1, 0, 5, 0, 1, 0]);
		await Assert.That(new ColPivHouseholderQR(scaled).ColsPermutationIndices()[0]).IsEqualTo(2);
	}
}
