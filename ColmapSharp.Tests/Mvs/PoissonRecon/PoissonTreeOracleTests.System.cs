// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.System (C#-only, not a COLMAP test): the system assembly of
// Poisson::Solver::Solve after finalizeForMultigrid - the normal-divergence constraints
// (PoissonFemConstraints, with PoissonMultigrid's restriction and prolongation) - against the
// "system*" runs of oracle/poisson_system_harness.cc (TestData/oracle/poisson_system.json).
// Tier A, bit-identical floats. Each run replays the harness's input through the stages the
// other PoissonTreeOracleTests files check step by step, then compares the stages here.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	// oracle/poisson_system_harness.cc's system assembly.
	private const string SystemFixture = "poisson_system.json";

	[Test]
	[Arguments("system3", 3)]
	[Arguments("system5", 5)]
	[Arguments("system6", 6)]
	[Arguments("system8", 8)]
	public async Task FemConstraints_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(SystemFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p) = Finalize(cases, name, depth, produced);

		// Solve: addFEMConstraints( F , *normalInfo , constraints , solveDepth ), and the harness's
		// extra run with a shallower maxDepth.
		FemConstraintIntegrator f = PoissonFemConstraints.CreateDivergenceIntegrator();
		var shallow = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, f, p.Normals, shallow, depth - 2);
		produced.F("femconstraintsshallow", shallow.Select(v => (double)v).ToList());
		var constraints = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, f, p.Normals, constraints, depth);
		produced.F("femconstraints", constraints.Select(v => (double)v).ToList());
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	// The harness's Run up to and including finalizeForMultigrid, emitting its sorted slices.
	private static (FemTree Tree, SortedTreeNodes Sorted, Prepared P) Finalize(JsonElement cases, string name, int depth, Cases produced)
	{
		Prepared p = Prepare(cases, name, depth, new Cases(name));
		FemTree tree = p.Set.Tree;
		SortedTreeNodes sorted = PoissonFinalize.FinalizeForMultigrid(
			tree, MaxDegree, (int)p.Parameters.BaseDepth, (int)p.Parameters.FullDepth, p.Normals, out _, p.Interpolation, p.Normals, p.Density, p.Colors);
		var slices = new List<double>();
		for (int d = 0; d < sorted.Levels; d++)
		{
			slices.AddRange([sorted.Begin(d), sorted.End(d)]);
		}

		produced.I("sortedslices", slices);
		return (tree, sorted, p);
	}
}
