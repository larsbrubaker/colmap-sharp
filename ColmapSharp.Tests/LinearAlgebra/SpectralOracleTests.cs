// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SpectralOracleTests (C#-only; COLMAP has no test for Eigen itself): JacobiSVD,
// Svd3d/Svd4d, SelfAdjointEigenSolver, EigenSolver and FullPivLU.Rank against numpy
// (LAPACK) on random, rank-deficient and repeated-value matrices, in the shapes COLMAP
// decomposes. Fixture: oracle/linear_algebra_spectral.py → TestData/oracle/
// linear_algebra_spectral.json.
//
// Tier B. Tolerances (entries O(1), small matrices):
// - Singular values and symmetric eigenvalues: 1e-12 absolute (times the largest value).
// - Reconstructions U S V^T and V D V^T, orthogonality of U and V: 1e-12 absolute.
// - Singular vectors / eigenvectors: compared up to sign (up to a complex phase for the
//   general solver) and only for simple values (relative gap to the neighbors > 1e-3),
//   within 1e-9; a vector's error is ~eps / gap. Null-space columns (index >= rank) are
//   checked by ||A v|| <= 1e-12 instead, since any orthonormal null basis is correct.
// - Rank: exact. Minimum-norm solve: 1e-10 against numpy's pinv with the same cutoff.
// - General eigenvalues: matched as a multiset within 1e-9 (1e-7 for the defective
//   Jordan-block case, whose eigenvalue error is ~sqrt(eps)); every eigenvector satisfies
//   ||A v - lambda v|| <= 1e-9 (1e-7 defective).

using System.Numerics;
using System.Text.Json;

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class SpectralOracleTests
{
	private const string Fixture = "linear_algebra_spectral.json";

	private static JsonElement[] Cases(string name) =>
		OracleFixture.Load(Fixture).GetProperty(name).EnumerateArray().ToArray();

	private static double[] D(JsonElement c, string name) => OracleFixture.Doubles(c.GetProperty(name));

	private static bool IsSimple(double[] values, int i)
	{
		double scale = Math.Max(Math.Abs(values[0]), Math.Abs(values[^1]));
		for (int j = 0; j < values.Length; j++)
		{
			if (j != i && Math.Abs(values[j] - values[i]) <= 1e-3 * scale)
			{
				return false;
			}
		}

		return true;
	}

	private static double MaxAbs(MatrixXd m) => m.AsSpan().ToArray().Select(Math.Abs).DefaultIfEmpty(0).Max();

	private static async Task AssertColumnUpToSign(string label, MatrixXd actual, MatrixXd expected, int column, double tolerance)
	{
		double sign = actual.Col(column).Dot(expected.Col(column)) >= 0 ? 1 : -1;
		double error = (actual.Col(column) * sign - expected.Col(column)).MaxAbs();
		await Assert.That(error).IsLessThanOrEqualTo(tolerance).Because($"{label} column {column}");
	}

	[Test]
	public async Task JacobiSVD_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("svd"))
		{
			string name = c.GetProperty("name").GetString()!;
			int rows = c.GetProperty("rows").GetInt32();
			int cols = c.GetProperty("cols").GetInt32();
			MatrixXd a = MatrixXd.FromColumnMajor(rows, cols, D(c, "a"));
			var svd = new JacobiSVD(a, SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
			double[] s = D(c, "s");
			double scale = Math.Max(1, s[0]);
			VectorXd actualS = svd.SingularValues();
			await Assert.That(svd.Info).IsEqualTo(ComputationInfo.Success);
			await Assert.That(actualS.Length).IsEqualTo(s.Length);
			for (int i = 0; i < s.Length; i++)
			{
				await Assert.That(Math.Abs(actualS[i] - s[i])).IsLessThanOrEqualTo(1e-12 * scale).Because($"{name} s[{i}]");
				if (i > 0)
				{
					await Assert.That(actualS[i]).IsLessThanOrEqualTo(actualS[i - 1]);
				}
			}

			int rank = c.GetProperty("rank").GetInt32();
			await Assert.That(svd.Rank()).IsEqualTo(rank).Because(name);

			MatrixXd u = svd.MatrixU();
			MatrixXd v = svd.MatrixV();
			await Assert.That(u.Rows).IsEqualTo(rows);
			await Assert.That(u.Cols).IsEqualTo(rows);
			await Assert.That(v.Rows).IsEqualTo(cols);
			await Assert.That(v.Cols).IsEqualTo(cols);
			await Assert.That(MaxAbs(u.Transpose() * u - MatrixXd.Identity(rows))).IsLessThanOrEqualTo(1e-12).Because(name + " U orthogonal");
			await Assert.That(MaxAbs(v.Transpose() * v - MatrixXd.Identity(cols))).IsLessThanOrEqualTo(1e-12).Because(name + " V orthogonal");
			var sigma = new MatrixXd(rows, cols);
			for (int i = 0; i < s.Length; i++)
			{
				sigma[i, i] = actualS[i];
			}

			await Assert.That(MaxAbs(u * sigma * v.Transpose() - a)).IsLessThanOrEqualTo(1e-12 * scale).Because(name + " reconstruction");

			MatrixXd expectedU = MatrixXd.FromColumnMajor(rows, rows, D(c, "u"));
			MatrixXd expectedV = MatrixXd.FromColumnMajor(cols, cols, D(c, "v"));
			for (int i = 0; i < rank; i++)
			{
				if (IsSimple(s, i))
				{
					await AssertColumnUpToSign(name + " U", u, expectedU, i, 1e-9);
					await AssertColumnUpToSign(name + " V", v, expectedV, i, 1e-9);
				}
			}

			// Null space: the columns COLMAP reads with matrixV().rightCols<k>().
			for (int i = rank; i < cols; i++)
			{
				await Assert.That((a * v.Col(i)).Norm()).IsLessThanOrEqualTo(1e-12 * scale).Because($"{name} A v[{i}]");
			}

			VectorXd x = svd.Solve(new VectorXd(D(c, "b")));
			await Assert.That((x - new VectorXd(D(c, "x"))).MaxAbs()).IsLessThanOrEqualTo(1e-10).Because(name + " solve");

			if (rows == 3 && cols == 3)
			{
				await AssertFixedMatchesDynamic(Svd3d.Compute(a.ToMatrix3d()), a);
			}
			else if (rows == 4 && cols == 4)
			{
				Svd4d fixed4 = Svd4d.Compute(a.ToMatrix4d());
				await Assert.That(fixed4.MatrixU).IsEqualTo(u.ToMatrix4d());
				await Assert.That(fixed4.MatrixV).IsEqualTo(v.ToMatrix4d());
				await Assert.That(fixed4.Rank()).IsEqualTo(rank);
			}
		}
	}

	// Same kernel, same buffers: the fixed-size SVD is bit-identical to JacobiSVD.
	private static async Task AssertFixedMatchesDynamic(Svd3d fixed3, MatrixXd a)
	{
		var svd = new JacobiSVD(a, SvdOptions.ComputeFullU | SvdOptions.ComputeFullV);
		await Assert.That(fixed3.MatrixU).IsEqualTo(svd.MatrixU().ToMatrix3d());
		await Assert.That(fixed3.MatrixV).IsEqualTo(svd.MatrixV().ToMatrix3d());
		await Assert.That(fixed3.SingularValues).IsEqualTo(svd.SingularValues().ToVector3d());
		await Assert.That(fixed3.Rank()).IsEqualTo(svd.Rank());
	}

	[Test]
	public async Task SelfAdjointEigenSolver_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("symmetric"))
		{
			int n = c.GetProperty("n").GetInt32();
			MatrixXd a = MatrixXd.FromColumnMajor(n, n, D(c, "a"));
			var solver = new SelfAdjointEigenSolver(a);
			double[] values = D(c, "values");
			MatrixXd expected = MatrixXd.FromColumnMajor(n, n, D(c, "vectors"));
			VectorXd actual = solver.Eigenvalues();
			MatrixXd vectors = solver.Eigenvectors();
			string label = $"n={n}";
			await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);
			for (int i = 0; i < n; i++)
			{
				await Assert.That(Math.Abs(actual[i] - values[i])).IsLessThanOrEqualTo(1e-12 * 4).Because($"{label} lambda[{i}]");
				if (IsSimple(values, i))
				{
					await AssertColumnUpToSign(label, vectors, expected, i, 1e-9);
				}
			}

			await Assert.That(MaxAbs(vectors.Transpose() * vectors - MatrixXd.Identity(n))).IsLessThanOrEqualTo(1e-12);
			await Assert.That(MaxAbs(vectors * MatrixXd.FromDiagonal(actual) * vectors.Transpose() - a)).IsLessThanOrEqualTo(1e-12 * 4);
		}
	}

	[Test]
	public async Task EigenSolver_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("general"))
		{
			int n = c.GetProperty("n").GetInt32();
			MatrixXd a = MatrixXd.FromColumnMajor(n, n, D(c, "a"));
			double[] re = D(c, "values_re");
			double[] im = D(c, "values_im");
			bool defective = Enumerable.Range(0, n).Any(i => Enumerable.Range(0, n).Any(j => j != i && re[i] == re[j] && im[i] == im[j]));
			double tolerance = defective ? 1e-7 : 1e-9;
			var solver = new EigenSolver(a);
			Complex[] values = solver.Eigenvalues();
			Complex[,] vectors = solver.Eigenvectors();
			string label = $"n={n} a[0]={a[0, 0]:R}";
			await Assert.That(solver.Info).IsEqualTo(ComputationInfo.Success);

			var unused = Enumerable.Range(0, n).ToList();
			for (int i = 0; i < n; i++)
			{
				var expected = new Complex(re[i], im[i]);
				int best = unused.OrderBy(j => Complex.Abs(values[j] - expected)).First();
				await Assert.That(Complex.Abs(values[best] - expected)).IsLessThanOrEqualTo(tolerance).Because($"{label} lambda {expected}");
				unused.Remove(best);
			}

			for (int k = 0; k < n; k++)
			{
				double residual = 0;
				double norm = 0;
				for (int r = 0; r < n; r++)
				{
					Complex av = Complex.Zero;
					for (int j = 0; j < n; j++)
					{
						av += a[r, j] * vectors[j, k];
					}

					residual = Math.Max(residual, Complex.Abs(av - values[k] * vectors[r, k]));
					norm += Complex.Abs(vectors[r, k]) * Complex.Abs(vectors[r, k]);
				}

				await Assert.That(residual).IsLessThanOrEqualTo(tolerance).Because($"{label} residual {k}");
				await Assert.That(Math.Abs(norm - 1)).IsLessThanOrEqualTo(1e-12).Because($"{label} unit norm {k}");
				if (values[k].Imaginary == 0)
				{
					for (int r = 0; r < n; r++)
					{
						await Assert.That(vectors[r, k].Imaginary).IsEqualTo(0.0);
					}
				}
			}
		}
	}

	[Test]
	public async Task FullPivLU_RankMatchesNumpy()
	{
		foreach (JsonElement c in Cases("full_piv_lu_rank"))
		{
			int rows = c.GetProperty("rows").GetInt32();
			int cols = c.GetProperty("cols").GetInt32();
			var lu = new FullPivLU(MatrixXd.FromColumnMajor(rows, cols, D(c, "a")));
			await Assert.That(lu.Rank()).IsEqualTo(c.GetProperty("rank").GetInt32()).Because($"{rows}x{cols}");
		}
	}
}
