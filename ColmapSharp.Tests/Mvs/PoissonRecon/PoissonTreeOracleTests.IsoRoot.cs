// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.IsoRoot (C#-only, not a COLMAP test): the iso-vertex root on crafted
// inputs that the extraction runs reach rarely or never - PoissonPolynomial.GetSolutions
// (Polynomial< 1 > and Polynomial< 2 >::getSolutions with Factor.h's Factor: real pair, double
// root, complex pair dropped or kept by EPS, fall-through to the linear Factor, no root) and
// PoissonLevelSetExtractor.AverageRoot (GetIsoVertex's root block: one root, two roots
// averaged, a linear Hermite, the linear fallback for NaN-scaled derivatives, the clamp at
// either end counted as bad, and the throw on equal ends) - against
// oracle/poisson_isoroot_harness.cc (TestData/oracle/poisson_isoroot.json), whose case tables
// the arrays below mirror row for row. Tier A, bit-identical doubles.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string IsoRootFixture = "poisson_isoroot.json";

	// c0, c1, c2, c, EPS (the harness's "quadratic" table).
	private static readonly double[][] QuadraticSolutionCases =
	[
		[1, -3, 2, 0, 0],
		[1, 2, 1, 0, 0],
		[1, 0, 1, 0, 0],
		[1, 0, 1, 0, 2],
		[1, 0, 4, 0, 1],
		[-1, 4, -3, 0, 0],
		[3, 2, 0, 1, 0],
		[3, 0, 0, 1, 0],
		[3, 2, 1e-12, 1, 1e-10],
		[5, 1e-12, 0, 1, 1e-10],
		[0.3, -1.7, 2.9, 0.1, 0],
		[0.25, 0.1, 0.3, 0.25, 0],
		[0.1, 0.7, -0.35, 0.4, 0],
		[0.1, 0.7, 0.5, -0.3, 0],
	];

	// c0, c1, c, EPS (the harness's "linear" table).
	private static readonly double[][] LinearSolutionCases = [[1, 2, 0, 0], [1, 0, 0, 0], [1, 1e-12, 0, 1e-10], [-0.3, 0.7, 0.2, 0]];

	// isoValue, x0, x1 (as floats), dx0, dx1 (the harness's "edges" table).
	private static readonly double[][] AverageRootCases =
	[
		[0.1, -0.3, 0.7, 0.9, 1.4],
		[0, -0.2, 0.05, 0.01, 0.6],
		[0, -1, 1, 2, 2],
		[1, 0, 1, 4, -2],
		[1, -1, 1, 2, 2],
		[-1, -1, 1, 2, 2],
		[0, -1, 1, 1, -1],
		[0, -1, 1, 0, 0],
		[2, 0, 1, 1, 1],
		[-1, 0, 1, 1, 1],
		[0.2, 0.1, 0.9, -3, 5],
		[0, 1, 1, 0, 0],
	];

	[Test]
	public async Task PolynomialGetSolutions_MatchesHarness()
	{
		JsonElement cases = OracleFixture.Load(IsoRootFixture).GetProperty("cases");
		var produced = new Cases("solutions");
		Span<double> roots = stackalloc double[2];
		for (int i = 0; i < QuadraticSolutionCases.Length; i++)
		{
			double[] q = QuadraticSolutionCases[i];
			int count = new PoissonPolynomial(q.AsSpan(0, 3)).GetSolutions(q[3], roots, q[4]);
			produced.F("quadratic" + i, [count, .. roots[..count].ToArray()]);
		}

		for (int i = 0; i < LinearSolutionCases.Length; i++)
		{
			double[] l = LinearSolutionCases[i];
			int count = new PoissonPolynomial(l.AsSpan(0, 2)).GetSolutions(l[2], roots, l[3]);
			produced.F("linear" + i, [count, .. roots[..count].ToArray()]);
		}

		await Assert.That(CompareRun(cases, "solutions", produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(QuadraticSolutionCases.Length + LinearSolutionCases.Length);
	}

	[Test]
	public async Task IsoVertexAverageRoot_MatchesHarness()
	{
		JsonElement cases = OracleFixture.Load(IsoRootFixture).GetProperty("cases");
		var produced = new Cases("averageroot");
		for (int i = 0; i < AverageRootCases.Length; i++)
		{
			double[] e = AverageRootCases[i];
			double root = 0;
			bool bad = false;
			bool threw = false;
			try
			{
				root = PoissonLevelSetExtractor.AverageRoot((float)e[0], (float)e[1], (float)e[2], e[3], e[4], out bad);
			}
			catch (InvalidOperationException)
			{
				threw = true;
			}

			produced.F(i.ToString(System.Globalization.CultureInfo.InvariantCulture), [root, bad ? 1 : 0, threw ? 1 : 0]);
		}

		await Assert.That(CompareRun(cases, "averageroot", produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(AverageRootCases.Length);
	}
}
