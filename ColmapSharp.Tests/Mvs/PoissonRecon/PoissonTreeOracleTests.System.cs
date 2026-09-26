// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.System (C#-only, not a COLMAP test): the system assembly of
// Poisson::Solver::Solve after finalizeForMultigrid - the normal-divergence constraints
// (PoissonFemConstraints, with PoissonMultigrid's restriction and prolongation), the point
// interpolation constraints, and the solver's per-depth matrix rows and prolongation
// constraints (PoissonSystem, PoissonSystemMatrix) - against the
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
	public async Task SystemConstraints_MatchHarness(string name, int depth)
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

		// Solve: addInterpolationConstraints( constraints , solveDepth , iInfo ).
		PoissonFemConstraints.AddInterpolationConstraints(tree, sorted, p.Interpolation, constraints, depth);
		produced.F("interpolationconstraints", constraints.Select(v => (double)v).ToList());
		AddSliceCases(produced, tree, sorted, p, depth);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	// The harness's solver-assembly block: per depth of at most SliceLimit nodes, the matrix rows,
	// prolongation constraints and inverse diagonals of
	// _getSliceMatrixAndProlongationConstraints over a synthetic prolonged solution, and
	// _getProlongedMatrixRowSize of every valid node.
	private static void AddSliceCases(Cases produced, FemTree tree, SortedTreeNodes sorted, Prepared p, int solveDepth)
	{
		const int SliceLimit = 12000;
		var f = new FemSystemIntegrator(PoissonFemConstraints.TestSignature, 1, 0.0, 1.0);
		var bsData = new PoissonPointEvaluator(PoissonFemConstraints.TestSignature, 1, solveDepth);
		var system = new PoissonSystem(tree, sorted, f, bsData, p.Interpolation, p.PointWeight);
		int maxDepth = PoissonMultigrid.MaxDepth(tree);
		var prolonged = new float[PoissonMultigrid.End(tree, sorted, maxDepth - 1)];
		for (int i = 0; i < prolonged.Length; i++)
		{
			prolonged[i] = (float)((long)(i * 37L % 101) - 50) / 64f;
		}

		var depths = new List<double>();
		var rowSizes = new List<double>();
		var columns = new List<double>();
		var values = new List<double>();
		var constraints = new List<double>();
		var diagonal = new List<double>();
		var prolongedRowSizes = new List<double>();
		var matrix = new PoissonSystemMatrix(27);
		var parentWindow = new int[27];
		for (int d = 1; d <= maxDepth; d++)
		{
			int begin = PoissonMultigrid.Begin(tree, sorted, d), end = PoissonMultigrid.End(tree, sorted, d);
			if (end - begin > SliceLimit)
			{
				continue;
			}

			depths.Add(d);
			f.Init(d);
			double[] cc = f.SetStencil();
			double[][] pc = f.SetParentChildStencils();
			var diag = new float[end - begin];
			var cons = new float[end - begin];
			system.GetSliceMatrixAndProlongationConstraints(matrix, diag, d, begin, end, prolonged, cons, cc, pc);
			for (int i = 0; i < end - begin; i++)
			{
				rowSizes.Add(matrix.RowSize(i));
				for (int j = 0; j < matrix.RowSize(i); j++)
				{
					columns.Add(matrix.Column(i, j));
					values.Add(matrix.Value(i, j));
				}

				constraints.Add(cons[i]);
				diagonal.Add(PoissonMultigrid.IsValidFem1Node(tree, sorted.TreeNodes[i + begin]) ? diag[i] : 0.0);
			}

			var key = new NeighborKey(tree, 1, 1, resetOnMissing: false);
			key.Set(d + tree.DepthOffset);
			for (int i = begin; i < end; i++)
			{
				int node = sorted.TreeNodes[i];
				if (!PoissonMultigrid.IsValidFem1Node(tree, node))
				{
					continue;
				}

				key.GetNeighbors(1, 1, tree.Parent(node), parentWindow);
				prolongedRowSizes.Add(system.GetProlongedMatrixRowSize(node, parentWindow));
			}
		}

		produced.I("slicedepths", depths);
		produced.I("slicerowsizes", rowSizes);
		produced.I("slicecolumns", columns);
		produced.F("slicevalues", values);
		produced.F("sliceconstraints", constraints);
		produced.F("slicediagonal", diagonal);
		produced.I("prolongedrowsizes", prolongedRowSizes);
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
