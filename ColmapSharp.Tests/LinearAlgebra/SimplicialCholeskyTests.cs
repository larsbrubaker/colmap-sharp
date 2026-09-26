// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SimplicialCholeskyTests (C#-only; COLMAP tests its sparse solvers only through
// sparse_cholesky_test.cc and least_absolute_deviations_test.cc, ported in Optim/):
// SimplicialCholesky and AmdOrdering in ColmapSharp/LinearAlgebra. Solutions are compared
// with the dense LLT on SPD matrices shaped like COLMAP's systems (2D grid Laplacians and a
// bundle-adjustment-like block-arrow normal matrix), for both factorization kinds, both
// orderings and both stored triangles. AMD is checked to be a permutation and to cut the
// fill of a grid Laplacian far below the natural ordering's, and a 100k-unknown Laplacian
// must factor in seconds.

using System.Diagnostics;

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class SimplicialCholeskyTests
{
	// 5-point Laplacian of a width x height grid plus `shift` on the diagonal (SPD for shift > 0).
	private static SparseMatrixCsc GridLaplacian(int width, int height, double shift)
	{
		var triplets = new List<SparseTriplet>();
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int i = y * width + x;
				double diag = shift;
				void Link(int j)
				{
					triplets.Add(new SparseTriplet(i, j, -1));
					diag += 1;
				}

				if (x > 0)
				{
					Link(i - 1);
				}

				if (x < width - 1)
				{
					Link(i + 1);
				}

				if (y > 0)
				{
					Link(i - width);
				}

				if (y < height - 1)
				{
					Link(i + width);
				}

				triplets.Add(new SparseTriplet(i, i, diag));
			}
		}

		int n = width * height;
		return SparseMatrixCsc.FromTriplets(n, n, triplets);
	}

	// Normal matrix J^T J + I of a synthetic BA problem: 6 parameters per camera, 3 per point,
	// every observation couples one camera with one point (a block-arrow pattern once points
	// are eliminated first, which AMD should discover).
	private static SparseMatrixCsc BlockArrowNormalMatrix(Random random, int cameras, int points, int observationsPerPoint)
	{
		int n = 6 * cameras + 3 * points;
		var triplets = new List<SparseTriplet>();
		int row = 0;
		for (int p = 0; p < points; p++)
		{
			for (int o = 0; o < observationsPerPoint; o++)
			{
				int c = random.Next(cameras);
				for (int r = 0; r < 2; r++, row++)
				{
					for (int k = 0; k < 6; k++)
					{
						triplets.Add(new SparseTriplet(row, 6 * c + k, random.NextDouble() * 2 - 1));
					}

					for (int k = 0; k < 3; k++)
					{
						triplets.Add(new SparseTriplet(row, 6 * cameras + 3 * p + k, random.NextDouble() * 2 - 1));
					}
				}
			}
		}

		SparseMatrixCsc jacobian = SparseMatrixCsc.FromTriplets(row, n, triplets);
		return jacobian.TransposeTimesSelf().AddToDiagonal(1.0);
	}

	private static VectorXd RandomVector(Random random, int n)
	{
		var v = new VectorXd(n);
		for (int i = 0; i < n; i++)
		{
			v[i] = random.NextDouble() * 2 - 1;
		}

		return v;
	}

	private static double RelativeError(VectorXd x, VectorXd reference) => (x - reference).Norm() / reference.Norm();

	private static async Task AssertMatchesDense(SparseMatrixCsc a, Random random)
	{
		VectorXd b = RandomVector(random, a.Rows);
		VectorXd reference = new LLT(a.ToDense()).Solve(b);
		foreach (SimplicialCholeskyKind kind in new[] { SimplicialCholeskyKind.LLT, SimplicialCholeskyKind.LDLT })
		{
			foreach (SparseOrdering ordering in new[] { SparseOrdering.Amd, SparseOrdering.Natural })
			{
				foreach (SymmetricPart part in new[] { SymmetricPart.Lower, SymmetricPart.Upper })
				{
					// Only the named triangle is given, so reading the other one would fail.
					SparseMatrixCsc stored = a.TriangularPart(part);
					var solver = new SimplicialCholesky(kind, ordering, part).Compute(stored);
					await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);
					await Assert.That(RelativeError(solver.Solve(b), reference)).IsLessThan(1e-10);
				}
			}
		}
	}

	[Test]
	public async Task GridLaplacian_MatchesDenseLlt()
	{
		await AssertMatchesDense(GridLaplacian(9, 7, 0.1), new Random(1));
	}

	[Test]
	public async Task BlockArrowNormalMatrix_MatchesDenseLlt()
	{
		var random = new Random(2);
		await AssertMatchesDense(BlockArrowNormalMatrix(random, 5, 20, 3), random);
	}

	[Test]
	public async Task RandomSpd_MatchesDenseLlt()
	{
		var random = new Random(3);
		for (int trial = 0; trial < 5; trial++)
		{
			SparseMatrixCsc j = SparseMatrixTests.RandomSparse(random, 40, 30, 0.08);
			await AssertMatchesDense(j.TransposeTimesSelf().AddToDiagonal(0.5), random);
		}
	}

	[Test]
	public async Task Amd_IsPermutationAndCutsFillOnGridLaplacian()
	{
		SparseMatrixCsc a = GridLaplacian(40, 40, 1.0);
		var amd = new SimplicialCholesky(SimplicialCholeskyKind.LLT, SparseOrdering.Amd);
		var natural = new SimplicialCholesky(SimplicialCholeskyKind.LLT, SparseOrdering.Natural);
		amd.AnalyzePattern(a);
		natural.AnalyzePattern(a);

		int[] sorted = amd.Permutation.ToArray();
		Array.Sort(sorted);

		// Natural order on a 40x40 grid fills the whole band of width 40 (about 64k entries);
		// minimum degree orderings land near 20k.
		using (Assert.Multiple())
		{
			await Assert.That(sorted).IsEquivalentTo(Enumerable.Range(0, a.Rows).ToArray());
			await Assert.That(natural.NonZerosL).IsGreaterThan(60_000);
			await Assert.That(amd.NonZerosL).IsLessThan(natural.NonZerosL / 2);
		}
	}

	[Test]
	public async Task Amd_OrdersIsolatedAndDisconnectedNodes()
	{
		// Two components plus an isolated node; every node must appear exactly once.
		var adjacency = new List<IReadOnlyList<int>> { new[] { 1 }, new[] { 0, 2 }, new[] { 1 }, Array.Empty<int>(), new[] { 5, 5, 4 }, new[] { 4 } };
		int[] perm = AmdOrdering.Compute(adjacency);
		Array.Sort(perm);
		await Assert.That(perm).IsEquivalentTo(new[] { 0, 1, 2, 3, 4, 5 });
	}

	[Test]
	public async Task Factorize_ReusesSymbolicAnalysisForNewValues()
	{
		var random = new Random(4);
		SparseMatrixCsc a1 = BlockArrowNormalMatrix(random, 4, 15, 3);
		SparseMatrixCsc a2 = a1.Clone();
		Span<double> values = a2.Values;
		for (int p = 0; p < values.Length; p++)
		{
			values[p] *= 1.5;
		}

		SparseMatrixCsc a3 = a2.AddToDiagonal(2.0);
		var solver = new SimplicialCholesky(SimplicialCholeskyKind.LDLT);
		solver.AnalyzePattern(a1);
		VectorXd b = RandomVector(random, a1.Rows);

		await Assert.That(solver.Factorize(a1)).IsTrue();
		await Assert.That(RelativeError(solver.Solve(b), new LLT(a1.ToDense()).Solve(b))).IsLessThan(1e-10);
		await Assert.That(solver.Factorize(a2)).IsTrue();
		await Assert.That(RelativeError(solver.Solve(b), new LLT(a2.ToDense()).Solve(b))).IsLessThan(1e-10);

		// a1 already stores its whole diagonal, so AddToDiagonal keeps the pattern.
		await Assert.That(solver.Factorize(a3)).IsTrue();
		await Assert.That(RelativeError(solver.Solve(b), new LLT(a3.ToDense()).Solve(b))).IsLessThan(1e-10);
		await Assert.That(() => solver.Factorize(GridLaplacian(2, 2, 1.0))).Throws<ArgumentException>();
	}

	[Test]
	public async Task Ldlt_SolvesIndefiniteAndLltRejectsIt()
	{
		// [[1, 2], [2, 1]] has eigenvalues 3 and -1.
		var a = SparseMatrixCsc.FromTriplets(2, 2, [new(0, 0, 1), new(1, 0, 2), new(0, 1, 2), new(1, 1, 1)]);
		var ldlt = new SimplicialCholesky(SimplicialCholeskyKind.LDLT).Compute(a);
		var llt = new SimplicialCholesky(SimplicialCholeskyKind.LLT).Compute(a);
		VectorXd x = ldlt.Solve(new VectorXd([5, 4]));
		using (Assert.Multiple())
		{
			await Assert.That(ldlt.Info).IsEqualTo(ComputationInfo.Success);
			await Assert.That(x[0]).IsEqualTo(1.0).Within(1e-14);
			await Assert.That(x[1]).IsEqualTo(2.0).Within(1e-14);
			await Assert.That(llt.Info).IsEqualTo(ComputationInfo.NumericalIssue);
			await Assert.That(() => llt.Solve(new VectorXd([5, 4]))).Throws<InvalidOperationException>();
		}
	}

	[Test]
	public async Task Performance_100kUnknownGridLaplacianFactorsInSeconds()
	{
		SparseMatrixCsc a = GridLaplacian(316, 316, 1e-3);
		VectorXd b = RandomVector(new Random(5), a.Rows);
		var watch = Stopwatch.StartNew();
		var solver = new SimplicialCholesky(SimplicialCholeskyKind.LLT).Compute(a);
		VectorXd x = solver.Solve(b);
		watch.Stop();
		Console.WriteLine($"100k Laplacian: nnz(L) = {solver.NonZerosL}, analyze+factorize+solve {watch.ElapsedMilliseconds} ms");

		using (Assert.Multiple())
		{
			await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);
			await Assert.That((a * x - b).Norm() / b.Norm()).IsLessThan(1e-10);
			await Assert.That(watch.Elapsed.TotalSeconds).IsLessThan(10.0);
		}
	}
}
