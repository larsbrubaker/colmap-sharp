// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SpectralTests (C#-only; COLMAP has no test for Eigen itself): hand-checked conventions of
// JacobiSVD, Svd3d, SelfAdjointEigenSolver, EigenSolver and FullPivLU that the numpy
// comparison in SpectralOracleTests does not pin: sorting and sign of singular values,
// thin/full factor shapes, non-finite input, eigenvalue order, complex-pair order. Tier B
// where a tolerance appears; exact where the value is exactly representable.

using System.Numerics;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class SpectralTests
{
	[Test]
	public async Task JacobiSVD_DiagonalIsSortedAndNonNegative()
	{
		// diag(1, -3, 2): singular values (3, 2, 1); the negative entry flips its U column.
		var svd = new JacobiSVD(MatrixXd.FromDiagonal(new VectorXd([1.0, -3.0, 2.0])), SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
		VectorXd s = svd.SingularValues();
		await Assert.That(s[0]).IsEqualTo(3.0);
		await Assert.That(s[1]).IsEqualTo(2.0);
		await Assert.That(s[2]).IsEqualTo(1.0);
		MatrixXd u = svd.MatrixU();
		MatrixXd v = svd.MatrixV();
		await Assert.That(u[1, 0] * v[1, 0]).IsEqualTo(-1.0);
		await Assert.That(u[2, 1] * v[2, 1]).IsEqualTo(1.0);
		await Assert.That(u[0, 2] * v[0, 2]).IsEqualTo(1.0);
	}

	[Test]
	public async Task JacobiSVD_ThinAndFullShapes()
	{
		MatrixXd tall = MatrixXd.FromRowMajor(5, 2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
		var thin = new JacobiSVD(tall, SvdOptions.ComputeThinU | SvdOptions.ComputeThinV);
		await Assert.That(thin.MatrixU().Cols).IsEqualTo(2);
		await Assert.That(thin.MatrixV().Cols).IsEqualTo(2);
		var wide = new JacobiSVD(tall.Transpose(), SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
		await Assert.That(wide.MatrixU().Rows).IsEqualTo(2);
		await Assert.That(wide.MatrixV().Rows).IsEqualTo(5);
		await Assert.That(wide.MatrixV().Cols).IsEqualTo(5);
		await Assert.That(wide.Rank()).IsEqualTo(2);
		await Assert.That(() => new JacobiSVD(tall).MatrixU()).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task JacobiSVD_NonFiniteInputIsInvalid()
	{
		MatrixXd a = MatrixXd.Identity(3);
		a[1, 2] = double.NaN;
		await Assert.That(new JacobiSVD(a).Info).IsEqualTo(ComputationInfo.InvalidInput);
		await Assert.That(Svd3d.Compute(a.ToMatrix3d()).Info).IsEqualTo(ComputationInfo.InvalidInput);
	}

	[Test]
	public async Task SelfAdjointEigenSolver_ReadsLowerTriangleAndSortsAscending()
	{
		// The upper triangle is garbage; only the lower one ([[2, 1], [1, 2]]) counts.
		MatrixXd a = MatrixXd.FromRowMajor(2, 2, [2, 100, 1, 2]);
		var solver = new SelfAdjointEigenSolver(a);
		VectorXd values = solver.Eigenvalues();
		await Assert.That(Math.Abs(values[0] - 1)).IsLessThanOrEqualTo(1e-15);
		await Assert.That(Math.Abs(values[1] - 3)).IsLessThanOrEqualTo(1e-15);
		MatrixXd vectors = solver.Eigenvectors();
		await Assert.That(Math.Abs(vectors[0, 0] + vectors[1, 0])).IsLessThanOrEqualTo(1e-15);
	}

	[Test]
	public async Task EigenSolver_ComplexPairListsPositiveImaginaryFirst()
	{
		// Rotation by 90 degrees plus 2: eigenvalues 2 +- i.
		var solver = new EigenSolver(MatrixXd.FromRowMajor(2, 2, [2, -1, 1, 2]));
		Complex[] values = solver.Eigenvalues();
		await Assert.That(values[0]).IsEqualTo(new Complex(2, 1));
		await Assert.That(values[1]).IsEqualTo(new Complex(2, -1));
		Complex[,] vectors = solver.Eigenvectors();
		await Assert.That(vectors[0, 1]).IsEqualTo(Complex.Conjugate(vectors[0, 0]));
	}

	// Regression: a thin U of a wide matrix went through the full N x N Householder Q
	// (3 x 10000 allocated ~800 MB). Thin factors must stay O(N).
	[Test]
	public async Task JacobiSVD_ThinFactorsOfWideInputStaySmall()
	{
		const int n = 20000;
		var a = new MatrixXd(3, n);
		for (int j = 0; j < n; j++)
		{
			a[0, j] = Math.Sin(j);
			a[1, j] = Math.Cos(0.5 * j);
			a[2, j] = Math.Sin(0.25 * j + 1);
		}

		long before = GC.GetAllocatedBytesForCurrentThread();
		var svd = new JacobiSVD(a, SvdOptions.ComputeThinU | SvdOptions.ComputeThinV);
		long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
		MatrixXd v = svd.MatrixV();
		MatrixXd u = svd.MatrixU();
		await Assert.That(allocated).IsLessThan(20_000_000L);
		await Assert.That(v.Rows).IsEqualTo(n);
		await Assert.That(v.Cols).IsEqualTo(3);
		var sigma = MatrixXd.FromDiagonal(svd.SingularValues());
		double error = (u * sigma * v.Transpose() - a).AsSpan().ToArray().Max(Math.Abs);
		await Assert.That(error).IsLessThanOrEqualTo(1e-11);
		double orthogonality = (v.Transpose() * v - MatrixXd.Identity(3)).AsSpan().ToArray().Max(Math.Abs);
		await Assert.That(orthogonality).IsLessThanOrEqualTo(1e-12);
	}

	// Regression: the 2x2 step's hypot underflowed to 0 and produced NaN with Info Success.
	[Test]
	public async Task JacobiSVD_TinyEntriesDoNotUnderflow()
	{
		MatrixXd a = MatrixXd.FromRowMajor(3, 3, [0, 1e-170, 1, 2e-170, 0, 0, 0, 0, 0]);
		var svd = new JacobiSVD(a, SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
		VectorXd s = svd.SingularValues();
		await Assert.That(svd.Info).IsEqualTo(ComputationInfo.Success);
		await Assert.That(Math.Abs(s[0] - 1)).IsLessThanOrEqualTo(1e-15);
		await Assert.That(s[1]).IsLessThanOrEqualTo(1e-15);
		MatrixXd u = svd.MatrixU();
		MatrixXd v = svd.MatrixV();
		await Assert.That(u.AsSpan().ToArray().All(double.IsFinite)).IsTrue();
		await Assert.That(v.AsSpan().ToArray().All(double.IsFinite)).IsTrue();
		double error = (u * MatrixXd.FromDiagonal(s) * v.Transpose() - a).AsSpan().ToArray().Max(Math.Abs);
		await Assert.That(error).IsLessThanOrEqualTo(1e-15);
	}

	// Regression: without scaling, squared norms under/overflowed, no sweep ran and the
	// unrotated diagonal came back as the eigenvalues.
	[Test]
	[Arguments(1e-170)]
	[Arguments(1e160)]
	public async Task SelfAdjointEigenSolver_ExtremeScales(double scale)
	{
		MatrixXd a = MatrixXd.Constant(2, 2, scale);
		var solver = new SelfAdjointEigenSolver(a);
		VectorXd values = solver.Eigenvalues();
		await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);
		await Assert.That(Math.Abs(values[0]) / scale).IsLessThanOrEqualTo(1e-15);
		await Assert.That(Math.Abs(values[1] / (2 * scale) - 1)).IsLessThanOrEqualTo(1e-15);
	}

	// Regression: products of ~1e200 entries overflowed in the Francis step.
	[Test]
	public async Task EigenSolver_HugeEntries()
	{
		MatrixXd unit = MatrixXd.FromRowMajor(3, 3, [1, 2, 0, 0.5, 1, 3, 1, 0, 2]);
		MatrixXd huge = unit * 1e200;
		var reference = new EigenSolver(unit, computeEigenvectors: false);
		var solver = new EigenSolver(huge);
		await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);
		Complex[] expected = reference.Eigenvalues();
		Complex[] actual = solver.Eigenvalues();
		for (int i = 0; i < 3; i++)
		{
			await Assert.That(Complex.Abs(actual[i] / 1e200 - expected[i])).IsLessThanOrEqualTo(1e-12);
		}
	}

	// Exactly rank-deficient inputs (COLMAP's everyday case) must converge in a few sweeps:
	// the absolute floor eps * ||A||_F stops rotations on roundoff next to a zero diagonal.
	private const int FewSweeps = 8;

	private static async Task AssertConvergesWithNullSpace(string label, MatrixXd a, int expectedRank)
	{
		var svd = new JacobiSVD(a, SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
		await Assert.That(svd.Info).IsEqualTo(ComputationInfo.Success).Because(label);
		await Assert.That(svd.Sweeps).IsLessThanOrEqualTo(FewSweeps).Because($"{label}: {svd.Sweeps} sweeps");
		await Assert.That(svd.Rank()).IsEqualTo(expectedRank).Because(label);
		MatrixXd v = svd.MatrixV();
		double scale = svd.SingularValues()[0];
		for (int i = expectedRank; i < a.Cols; i++)
		{
			await Assert.That((a * v.Col(i)).Norm()).IsLessThanOrEqualTo(1e-13 * scale).Because($"{label} null column {i}");
		}
	}

	private static Matrix3d TestRotation() => new Quaterniond(0.9, 0.2, -0.3, 0.25).Normalized().ToRotationMatrix();

	private static readonly Vector3d TestTranslation = new(0.4, -0.7, 0.3);

	[Test]
	public async Task JacobiSVD_ExactEssentialMatrixConverges()
	{
		Matrix3d essential = Rigid3d.CrossProductMatrix(TestTranslation) * TestRotation();
		await AssertConvergesWithNullSpace("essential", MatrixXd.From(essential), 2);
		Svd3d fixed3 = Svd3d.Compute(essential);
		await Assert.That(fixed3.Info).IsEqualTo(ComputationInfo.Success);
		await Assert.That(fixed3.Rank()).IsEqualTo(2);
	}

	[Test]
	[Arguments(8)]
	[Arguments(50)]
	public async Task JacobiSVD_NoiseFreeEightPointSystemConverges(int n)
	{
		// Rows kron(x2, x1) of the epipolar constraint x2^T E x1 = 0 for exact correspondences.
		Matrix3d rotation = TestRotation();
		var a = new MatrixXd(n, 9);
		for (int k = 0; k < n; k++)
		{
			var point = new Vector3d(Math.Sin(1.3 * k), Math.Cos(0.7 * k + 0.2), 4 + Math.Sin(0.37 * k));
			Vector3d x1 = point / point.Z;
			Vector3d moved = rotation * point + TestTranslation;
			Vector3d x2 = moved / moved.Z;
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					a[k, 3 * r + c] = x2[r] * x1[c];
				}
			}
		}

		await AssertConvergesWithNullSpace($"8-point n={n}", a, 8);
	}

	[Test]
	public async Task JacobiSVD_ZeroColumnConverges()
	{
		MatrixXd a = MatrixXd.FromRowMajor(4, 4, [
			1, 0, 2, -1,
			0.5, 0, -1, 3,
			2, 0, 0.3, 0.7,
			-1, 0, 1.5, 2]);
		await AssertConvergesWithNullSpace("zero column", a, 3);
	}

	// A zero column of A is an exact null vector, and it must come out exact: COLMAP's
	// TriangulatePoint rejects parallel rays by testing that vector's last coordinate for
	// == 0 (triangulation_test.cc TriangulatePoint.ParallelRays uses this very matrix).
	// The general two-angle rotation left cos(pi/2) = 6.1e-17 there.
	[Test]
	public async Task Svd4d_ZeroColumnNullVectorIsExact()
	{
		var a = new Matrix4d(
			-1, 0, 0, 0,
			0, -1, 0, 0,
			-1, 0, 0, -1,
			0, -1, 0, 0);
		Svd4d svd = Svd4d.Compute(a);
		Vector4d nullVector = svd.MatrixV.Col(3);
		using (Assert.Multiple())
		{
			await Assert.That(svd.SingularValues.W).IsEqualTo(0.0);
			await Assert.That(nullVector.X).IsEqualTo(0.0);
			await Assert.That(nullVector.Y).IsEqualTo(0.0);
			await Assert.That(Math.Abs(nullVector.Z)).IsEqualTo(1.0);
			await Assert.That(nullVector.W).IsEqualTo(0.0);
		}
	}

	[Test]
	public async Task JacobiSVD_RankOneOuterProductConverges()
	{
		var a = new MatrixXd(9, 9);
		for (int r = 0; r < 9; r++)
		{
			for (int c = 0; c < 9; c++)
			{
				a[r, c] = Math.Sin(r + 1.0) * Math.Cos(0.5 * c + 0.3);
			}
		}

		await AssertConvergesWithNullSpace("rank-1 9x9", a, 1);
	}

	[Test]
	public async Task FullPivLU_RankOfCollinearColumns()
	{
		MatrixXd a = MatrixXd.FromRowMajor(3, 4, [1, 2, 3, 4, 2, 4, 6, 8, 0, 0, 0, 0]);
		await Assert.That(new FullPivLU(a).Rank()).IsEqualTo(1);
		await Assert.That(new FullPivLU(MatrixXd.Identity(3, 5)).Rank()).IsEqualTo(3);
	}
}
