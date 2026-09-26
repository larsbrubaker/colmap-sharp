// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/covariance_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// CovarianceTests: the CovarianceTest fixture (x: 2, y: 3, z: 1 in one buffer, five residual
// blocks with fixed Jacobians) and the SPARSE_QR legs of NormalBehavior,
// ManifoldInTangentSpace and ManifoldInTangentSpaceWithConstantBlocks, with Ceres' expected
// matrices and tolerance (Frobenius difference / block size <= 1e-5). Tests
// ColmapSharp/Solver/Covariance.cs, which only implements what COLMAP calls
// (Compute over a block list + GetCovarianceMatrixInTangentSpace), so the ambient-space,
// DENSE_SVD, sparsity, threading, truncated-rank and large-scale cases of
// covariance_test.cc have no code under test here. Ceres checks every subset of the six
// block pairs; the full covariance is always computed here, so each block pair of the
// full matrix is compared once.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class CovarianceTests
{
	private sealed class UnaryCostFunction(int numResiduals, int blockSize, double[] jacobian)
		: CostFunction(numResiduals, blockSize)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			residuals[..NumResiduals].Fill(1);
			if (!jacobians.IsEmpty && jacobians[0].Array is not null)
			{
				jacobian.CopyTo(jacobians[0].AsSpan());
			}

			return true;
		}
	}

	private sealed class BinaryCostFunction(int numResiduals, int size1, int size2, double[] jacobian1, double[] jacobian2)
		: CostFunction(numResiduals, size1, size2)
	{
		public override bool Evaluate(
			ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
		{
			residuals[..NumResiduals].Fill(2);
			if (!jacobians.IsEmpty)
			{
				if (jacobians[0].Array is not null)
				{
					jacobian1.CopyTo(jacobians[0].AsSpan());
				}

				if (jacobians[1].Array is not null)
				{
					jacobian2.CopyTo(jacobians[1].AsSpan());
				}
			}

			return true;
		}
	}

	private sealed class PolynomialManifold : Manifold
	{
		public override int AmbientSize => 2;

		public override int TangentSize => 1;

		public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
		{
			xPlusDelta[0] = delta[0] * x[0];
			xPlusDelta[1] = delta[0] * x[1];
			return true;
		}

		public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
		{
			jacobian[0] = x[0];
			jacobian[1] = x[1];
			return true;
		}

		public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX) =>
			throw new InvalidOperationException("Should not be called");

		public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian) =>
			throw new InvalidOperationException("Should not be called");
	}

	private sealed class Fixture
	{
		public Fixture()
		{
			double[] p = Parameters;
			p[0] = 1;
			p[1] = 1;
			p[2] = 2;
			p[3] = 2;
			p[4] = 2;
			p[5] = 3;
			Problem.AddResidualBlock(new UnaryCostFunction(2, 2, [1.0, 0.0, 0.0, 1.0]), null, X);
			Problem.AddResidualBlock(
				new UnaryCostFunction(3, 3, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]), null, Y);
			Problem.AddResidualBlock(new UnaryCostFunction(1, 1, [5.0]), null, Z);
			Problem.AddResidualBlock(new BinaryCostFunction(1, 3, 2, [1.0, 2.0, 3.0], [-5.0, -6.0]), null, Y, X);
			Problem.AddResidualBlock(new BinaryCostFunction(1, 1, 2, [2.0], [3.0, -2.0]), null, Z, X);
		}

		public double[] Parameters { get; } = new double[6];

		public ArraySegment<double> X => new(Parameters, 0, 2);

		public ArraySegment<double> Y => new(Parameters, 2, 3);

		public ArraySegment<double> Z => new(Parameters, 5, 1);

		public Problem Problem { get; } = new();
	}

	// ComputeAndCompareCovarianceBlocksInTangentSpace: every block pair against the expected
	// tangent-space covariance, with Ceres' size-normalized Frobenius tolerance.
	private static async Task ComputeAndCompareInTangentSpace(Fixture fixture, int[] tangentSizes, double[] expected)
	{
		var covariance = new Covariance();
		ArraySegment<double>[] blocks = [fixture.X, fixture.Y, fixture.Z];
		await Assert.That(covariance.Compute(blocks, fixture.Problem)).IsTrue();
		MatrixXd actual = covariance.GetCovarianceMatrixInTangentSpace(blocks);
		int dof = tangentSizes.Sum();
		await Assert.That(actual.Rows).IsEqualTo(dof);
		var starts = new int[4];
		for (int i = 0; i < 3; i++)
		{
			starts[i + 1] = starts[i] + tangentSizes[i];
		}

		const double kTolerance = 1e-5;
		for (int bi = 0; bi < 3; bi++)
		{
			for (int bj = 0; bj < 3; bj++)
			{
				double squared = 0;
				for (int r = starts[bi]; r < starts[bi + 1]; r++)
				{
					for (int c = starts[bj]; c < starts[bj + 1]; c++)
					{
						double d = expected[r * dof + c] - actual[r, c];
						squared += d * d;
					}
				}

				double diffNorm = Math.Sqrt(squared) / (tangentSizes[bi] * tangentSizes[bj]);
				await Assert.That(diffNorm).IsEqualTo(0.0).Within(kTolerance);
			}
		}
	}

	[Test]
	public async Task CovarianceTest_NormalBehavior()
	{
		var fixture = new Fixture();

		// inv(J'J) computed using octave.
		double[] expectedCovariance =
		[
			7.0747e-02, -8.4923e-03, 1.6821e-02, 3.3643e-02, 5.0464e-02, -1.5809e-02,
			-8.4923e-03, 8.1352e-02, 2.4758e-02, 4.9517e-02, 7.4275e-02, 1.2978e-02,
			1.6821e-02, 2.4758e-02, 2.4904e-01, -1.9271e-03, -2.8906e-03, -6.5325e-05,
			3.3643e-02, 4.9517e-02, -1.9271e-03, 2.4615e-01, -5.7813e-03, -1.3065e-04,
			5.0464e-02, 7.4275e-02, -2.8906e-03, -5.7813e-03, 2.4133e-01, -1.9598e-04,
			-1.5809e-02, 1.2978e-02, -6.5325e-05, -1.3065e-04, -1.9598e-04, 3.9544e-02,
		];

		// Without manifolds the tangent space is the ambient space.
		await ComputeAndCompareInTangentSpace(fixture, [2, 3, 1], expectedCovariance);
	}

	[Test]
	public async Task CovarianceTest_ManifoldInTangentSpace()
	{
		var fixture = new Fixture();
		fixture.Problem.SetManifold(fixture.X, new PolynomialManifold());
		fixture.Problem.SetManifold(fixture.Y, new SubsetManifold(3, [2]));

		// inv((J*A)'*(J*A)), computed using octave.
		double[] expectedCovariance =
		[
			0.01766, 0.02158, 0.04316, -0.00122,
			0.02158, 0.24860, -0.00281, -0.00149,
			0.04316, -0.00281, 0.24439, -0.00298,
			-0.00122, -0.00149, -0.00298, 0.03457,
		];

		await ComputeAndCompareInTangentSpace(fixture, [1, 2, 1], expectedCovariance);
	}

	[Test]
	public async Task CovarianceTest_ManifoldInTangentSpaceWithConstantBlocks()
	{
		var fixture = new Fixture();
		fixture.Problem.SetManifold(fixture.X, new PolynomialManifold());
		fixture.Problem.SetParameterBlockConstant(fixture.X);
		fixture.Problem.SetManifold(fixture.Y, new SubsetManifold(3, [2]));
		fixture.Problem.SetParameterBlockConstant(fixture.Y);

		// pinv((J*A)'*(J*A)), computed using octave.
		double[] expectedCovariance =
		[
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.034482,
		];

		await ComputeAndCompareInTangentSpace(fixture, [1, 2, 1], expectedCovariance);
	}

	/// <summary>
	/// C#-only, after Ceres' ComputeCovarianceSparsityWithFreeParameterBlock: a variable
	/// block that no residual block uses is treated as constant (zero covariance) instead of
	/// making the Jacobian rank deficient, and the other blocks' covariance is unchanged.
	/// </summary>
	[Test]
	public async Task CSharpOnly_FreeParameterBlockHasZeroCovariance()
	{
		var fixture = new Fixture();
		double[] free = [4.0, 5.0];
		fixture.Problem.AddParameterBlock(free);

		var covariance = new Covariance();
		ArraySegment<double>[] blocks = [fixture.X, fixture.Y, fixture.Z, free];
		await Assert.That(covariance.Compute(blocks, fixture.Problem)).IsTrue();
		MatrixXd actual = covariance.GetCovarianceMatrixInTangentSpace(blocks);
		await Assert.That(actual.Rows).IsEqualTo(8);

		// The x-x block of NormalBehavior's inv(J'J), and every entry touching the free block.
		await Assert.That(actual[0, 0]).IsEqualTo(7.0747e-02).Within(1e-5);
		await Assert.That(actual[1, 1]).IsEqualTo(8.1352e-02).Within(1e-5);
		await Assert.That(actual[5, 5]).IsEqualTo(3.9544e-02).Within(1e-5);
		for (int i = 0; i < 8; i++)
		{
			for (int j = 6; j < 8; j++)
			{
				await Assert.That(actual[i, j]).IsEqualTo(0.0);
				await Assert.That(actual[j, i]).IsEqualTo(0.0);
			}
		}
	}

	/// <summary>
	/// C#-only: a rank-deficient Jacobian makes Compute fail, as Ceres' SPARSE_QR path does
	/// ("Jacobian matrix is rank deficient").
	/// </summary>
	[Test]
	public async Task CSharpOnly_RankDeficientJacobianFails()
	{
		double[] x = [1.0, 2.0];
		var problem = new Problem();
		problem.AddResidualBlock(new UnaryCostFunction(2, 2, [1.0, 1.0, 2.0, 2.0]), null, x);
		await Assert.That(new Covariance().Compute([x], problem)).IsFalse();
	}
}
