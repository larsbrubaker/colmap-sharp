// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DecompositionOracleTests (C#-only; COLMAP has no test for Eigen itself): the dense
// decompositions in ColmapSharp/LinearAlgebra against numpy (LAPACK) on random matrices.
// The fixture is written by oracle/linear_algebra_dense.py into
// TestData/oracle/linear_algebra_dense.json; that script says which LAPACK routine each
// field comes from.
//
// Tier B. Tolerances (entries are O(1), matrices are small and well conditioned):
// - HouseholderQR Q and R: 1e-12 absolute per entry, signs included (the reflector sign
//   convention is LAPACK's, which Eigen documents too).
// - Least-squares, LU and Cholesky solutions, inverse, Cholesky L: 1e-10 absolute per
//   entry, which leaves room for the condition number (the observed gaps are ~1e-15).
// - Determinant: 1e-12 relative.

using System.Text.Json;

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class DecompositionOracleTests
{
	private const string Fixture = "linear_algebra_dense.json";

	private static JsonElement[] Cases(string name) =>
		OracleFixture.Load(Fixture).GetProperty(name).EnumerateArray().ToArray();

	private static MatrixXd Matrix(JsonElement c, string name, int rows, int cols) =>
		MatrixXd.FromColumnMajor(rows, cols, OracleFixture.Doubles(c.GetProperty(name)));

	private static VectorXd Vector(JsonElement c, string name) => new(OracleFixture.Doubles(c.GetProperty(name)));

	private static async Task AssertNear(string label, ReadOnlyMemory<double> actual, double[] expected, double tolerance)
	{
		await Assert.That(actual.Length).IsEqualTo(expected.Length);
		for (int i = 0; i < expected.Length; i++)
		{
			double difference = Math.Abs(actual.Span[i] - expected[i]);
			await Assert.That(difference).IsLessThanOrEqualTo(tolerance).Because($"{label}[{i}]: {actual.Span[i]:R} vs {expected[i]:R}");
		}
	}

	[Test]
	public async Task HouseholderQR_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("qr"))
		{
			int rows = c.GetProperty("rows").GetInt32();
			int cols = c.GetProperty("cols").GetInt32();
			var qr = new HouseholderQR(Matrix(c, "a", rows, cols));
			string label = $"{rows}x{cols}";
			await AssertNear(label + " Q", qr.HouseholderQ().AsSpan().ToArray(), OracleFixture.Doubles(c.GetProperty("q")), 1e-12);
			await AssertNear(label + " R", qr.MatrixR().AsSpan().ToArray(), OracleFixture.Doubles(c.GetProperty("r")), 1e-12);
			if (c.TryGetProperty("x", out JsonElement x))
			{
				VectorXd solution = qr.Solve(Vector(c, "b"));
				await AssertNear(label + " x", solution.AsSpan().ToArray(), OracleFixture.Doubles(x), 1e-10);
				VectorXd pivoted = new ColPivHouseholderQR(Matrix(c, "a", rows, cols)).Solve(Vector(c, "b"));
				await AssertNear(label + " colpiv x", pivoted.AsSpan().ToArray(), OracleFixture.Doubles(x), 1e-10);
			}
		}
	}

	[Test]
	public async Task PartialPivLU_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("lu"))
		{
			int n = c.GetProperty("n").GetInt32();
			var lu = new PartialPivLU(Matrix(c, "a", n, n));
			string label = $"n={n}";
			await AssertNear(label + " x", lu.Solve(Vector(c, "b")).AsSpan().ToArray(), OracleFixture.Doubles(c.GetProperty("x")), 1e-10);
			await AssertNear(label + " inverse", lu.Inverse().AsSpan().ToArray(), OracleFixture.Doubles(c.GetProperty("inverse")), 1e-10);
			double det = c.GetProperty("det").GetDouble();
			await Assert.That(Math.Abs(lu.Determinant() - det)).IsLessThanOrEqualTo(1e-12 * Math.Abs(det));
		}
	}

	[Test]
	public async Task Cholesky_MatchesNumpy()
	{
		foreach (JsonElement c in Cases("spd"))
		{
			int n = c.GetProperty("n").GetInt32();
			MatrixXd a = Matrix(c, "a", n, n);
			var llt = new LLT(a);
			string label = $"n={n}";
			double[] x = OracleFixture.Doubles(c.GetProperty("x"));
			await Assert.That(llt.Info).IsEqualTo(ComputationInfo.Success);
			await AssertNear(label + " L", llt.MatrixL().AsSpan().ToArray(), OracleFixture.Doubles(c.GetProperty("l")), 1e-10);
			await AssertNear(label + " llt x", llt.Solve(Vector(c, "b")).AsSpan().ToArray(), x, 1e-10);
			await AssertNear(label + " ldlt x", new LDLT(a).Solve(Vector(c, "b")).AsSpan().ToArray(), x, 1e-10);
		}
	}
}
