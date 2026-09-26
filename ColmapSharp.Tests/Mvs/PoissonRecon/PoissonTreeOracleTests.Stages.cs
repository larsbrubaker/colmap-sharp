// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.Stages (C#-only, not a COLMAP test): the solver-preparation stages of
// Poisson::Solver::Solve after the sample set - density (PoissonDensity), the normal and color
// fields (PoissonSplat), the point interpolation constraints (PoissonInterpolation) and
// _finalizeForMultigrid (PoissonFinalize) - step by step against the "density*" runs of
// oracle/poisson_tree_harness.cc, the composed FinalizeForMultigrid against the "density*final"
// runs (the vendored finalizeForMultigrid), and the FEM constraint and system integrators and
// the restriction/prolongation stencils against TestData/oracle/poisson_fem.json. Split from PoissonTreeOracleTests.cs (the tree stage and the shared helpers) by stage.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const int MaxDegree = 2;

	// oracle/poisson_fem_harness.cc's integrators and stencils.
	private const string FemFixture = "poisson_fem.json";

	[Test]
	[Arguments("density5", 5)]
	[Arguments("density3", 3)]
	[Arguments("density6", 6)]
	[Arguments("density8", 8)]
	public async Task DensityStage_MatchesHarness(string name, int depth)
	{
		var produced = new Cases(name);
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		Prepared p = Prepare(cases, name, depth, produced);
		FemTree tree = p.Set.Tree;

		// The tree-shaping steps of _finalizeForMultigrid, one by one as the harness runs them.
		int baseDepth = (int)p.Parameters.BaseDepth;
		int fullDepth = (int)p.Parameters.FullDepth;
		bool AddNode(int d, int x, int y, int z) => d <= fullDepth;
		PoissonFinalize.Reroot(tree, MaxDegree);
		produced.I("reroot", [tree.DepthOffset, tree.NodeCount, tree.MaxDepth(tree.SpaceRoot)]);
		produced.I("reroottree", DumpTree(tree));
		PoissonFinalize.SetFullDepth(tree, MaxDegree, baseDepth);
		produced.I("fulldepthtree", DumpTree(tree));
		PoissonFinalize.Refine(tree, MaxDegree, AddNode);
		produced.I("refinetree", DumpTree(tree));
		PoissonFinalize.MarkGhosts(tree, baseDepth, AddNode);
		produced.I("ghostflags", DumpFlags(tree));
		PoissonFinalize.ClipTree(tree, node => tree.LocalDepth(node) <= fullDepth || PoissonFinalize.HasNormalData(tree, p.Normals, node), baseDepth);
		produced.I("clipflags", DumpFlags(tree));
		int active = 0, ghost = 0;
		tree.ProcessNodes(tree.Root, node =>
		{
			active += tree.IsActive(node) ? 1 : 0;
			ghost += tree.IsActive(node) ? 0 : 1;
		});
		int maxDepth = tree.MaxDepth(tree.Root) - tree.DepthOffset;
		produced.I("clipinfo", [maxDepth, tree.NodeCount, active, ghost]);

		// The second half: supporting prolongation, Dirichlet element marks, sorting and re-keying.
		PoissonFinalize.SupportApproximateProlongation(tree, MaxDegree, maxDepth, baseDepth);
		produced.I("prolongtree", DumpTree(tree));
		PoissonFinalize.MarkNonBaseDirichletElements(tree, baseDepth);
		produced.I("prolongflags", DumpFlags(tree));
		var sorted = new SortedTreeNodes();
		int[] map = PoissonFinalize.SetSortedTreeNodes(tree, sorted, p.Interpolation, p.Normals, p.Density, p.Colors);
		AddSortedCases(produced, tree, sorted, map, p);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	[Test]
	[Arguments("density5", 5)]
	[Arguments("density3", 3)]
	[Arguments("density6", 6)]
	[Arguments("density8", 8)]
	public async Task FinalizeForMultigrid_MatchesTheVendoredFinalize(string name, int depth)
	{
		// The "<name>final" run calls the real FEMTree::finalizeForMultigrid<2,1> on a tree built
		// from the same input, so this also checks the harness's step-by-step copy of it.
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		Prepared p = Prepare(cases, name, depth, new Cases(name));
		FemTree tree = p.Set.Tree;
		SortedTreeNodes sorted = PoissonFinalize.FinalizeForMultigrid(
			tree, MaxDegree, (int)p.Parameters.BaseDepth, (int)p.Parameters.FullDepth, p.Normals, out int[] map, p.Interpolation, p.Normals, p.Density, p.Colors);
		var produced = new Cases(name + "final");
		AddSortedCases(produced, tree, sorted, map, p);
		await Assert.That(CompareRun(cases, name + "final", produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task MarkNonBaseDirichletElements_RejectsDirichletNodeFlags()
	{
		// C#-only guard: the port has no envelope, so a Dirichlet node flag means a caller the
		// port does not support; it must fail loudly rather than mark nothing.
		var tree = new FemTree();
		tree.SetFlags(tree.Root, FemTree.DirichletNodeFlag);
		await Assert.That(() => PoissonFinalize.MarkNonBaseDirichletElements(tree, 0)).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task FemConstraintIntegrator_MatchesHarness()
	{
		// Mirrors DumpFemConstraint in the harness: Solve's normal-divergence constraint.
		const int TestSig = 5, NormalSig = 7;
		var f = new FemConstraintIntegrator(TestSig, 1, NormalSig, 0, 3);
		for (int d = 0; d < 3; d++)
		{
			f.Weights[d, FemConstraintIntegrator.DerivativeIndex(1, d == 0 ? 1 : 0, d == 1 ? 1 : 0, d == 2 ? 1 : 0), 0] = 1;
		}

		var cc = new List<double>();
		var pc = new List<double>();
		var raw = new List<double>();
		var v = new double[3];
		for (int depth = 0; depth <= 6; depth++)
		{
			f.Init(depth);
			cc.AddRange(f.SetStencil());
			foreach (double[] stencil in f.SetParentChildStencils())
			{
				pc.AddRange(stencil);
			}

			if (depth != 3)
			{
				continue;
			}

			for (int a = -2; a <= 10; a++)
			{
				for (int b = -2; b <= 10; b++)
				{
					f.CcIntegrate([a, 1, 7], [b, 2, 7], v);
					raw.AddRange(v);
					f.PcIntegrate([a / 2, 1, 3], [b, 2, 7], v);
					raw.AddRange(v);
					f.CpIntegrate([b, 2, 7], [a / 2, 1, 3], v);
					raw.AddRange(v);
				}
			}
		}

		var produced = new Cases("femconstraint");
		produced.F("cc", cc);
		produced.F("pc", pc);
		produced.F("raw", raw);
		await Assert.That(CompareRun(OracleFixture.Load(FemFixture).GetProperty("cases"), "femconstraint", produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task FemSystemIntegrator_MatchesHarness()
	{
		// Mirrors DumpFemSystem in the harness: Solve's system F( { 0 , 1 } ).
		const int Sig = 5;
		var f = new FemSystemIntegrator(Sig, 1, 0.0, 1.0);
		var cc = new List<double>();
		var pc = new List<double>();
		var raw = new List<double>();
		for (int depth = 0; depth <= 6; depth++)
		{
			f.Init(depth);
			cc.AddRange(f.SetStencil());
			foreach (double[] stencil in f.SetParentChildStencils())
			{
				pc.AddRange(stencil);
			}

			if (depth != 3)
			{
				continue;
			}

			for (int a = -2; a <= 10; a++)
			{
				for (int b = -2; b <= 10; b++)
				{
					raw.Add(f.CcIntegrate([a, 0, 7], [b, 1, 7]));
					raw.Add(f.PcIntegrate([a / 2, 0, 3], [b, 1, 7]));
				}
			}
		}

		var produced = new Cases("femsystem");
		produced.F("cc", cc);
		produced.F("pc", pc);
		produced.F("raw", raw);
		produced.I("vanishesonconstants", [f.VanishesOnConstants ? 1 : 0]);
		await Assert.That(CompareRun(OracleFixture.Load(FemFixture).GetProperty("cases"), "femsystem", produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task RestrictionProlongation_MatchesHarness()
	{
		// Mirrors DumpRestrictionProlongation in the harness: the degree-1 Neumann basis.
		var rp = new RestrictionProlongation(5);
		var up = new List<double>();
		var down = new List<double>();
		var raw = new List<double>();
		for (int depth = 1; depth <= 6; depth++)
		{
			rp.Init(depth);
			up.AddRange(rp.SetStencil());
			foreach (double[] stencil in rp.SetStencils())
			{
				down.AddRange(stencil);
			}

			if (depth != 3)
			{
				continue;
			}

			for (int a = -1; a <= 5; a++)
			{
				for (int b = -2; b <= 9; b++)
				{
					raw.Add(rp.UpSampleCoefficient([a, 0, 2], [b, 1, 4]));
				}
			}
		}

		var produced = new Cases("restrictionprolongation");
		produced.F("up", up);
		produced.F("down", down);
		produced.F("raw", raw);
		await Assert.That(CompareRun(OracleFixture.Load(FemFixture).GetProperty("cases"), "restrictionprolongation", produced)).IsEqualTo(string.Empty);
	}

	// The pipeline up to finalizing, as the harness's density runs execute it, emitting the
	// density, splat and interpolation cases.
	private static Prepared Prepare(JsonElement cases, string name, int depth, Cases produced)
	{
		PoissonSampleSet set = Build(cases.GetProperty(name + "/input"), depth, confidence: false, out PoissonSolutionParameters parameters);
		FemTree tree = set.Tree;
		tree.ResetNodeIndices(0);
		int nodesBefore = tree.NodeCount;

		// Solve: setDensityEstimator< 1 , Reconstructor::WeightDegree >( samples , kernelDepth , samplesPerNode ).
		DensityEstimator density = PoissonDensity.SetDensityEstimator(set, 1, 2, (int)parameters.KernelDepth, parameters.SamplesPerNode);
		produced.F("density", DumpField(tree, density, 1));
		produced.I("densitytree", DumpTree(tree));
		produced.I("densityinfo", [density.Count, density.KernelDepth, density.CoDimension, nodesBefore, tree.NodeCount]);

		// Solve: the normal field (then negated) and the color field (then scaled per level).
		SparseNodeData normals = PoissonSplat.SetNormalField(set, density, (int)parameters.BaseDepth, (int)parameters.Depth, parameters.LowDepthCutOff, out var pointDepthAndWeight);
		produced.F("normals", DumpField(tree, normals, 3));
		produced.F("pointdepthandweight", [pointDepthAndWeight.DepthSum, pointDepthAndWeight.WeightSum, pointDepthAndWeight.TotalWeight]);
		produced.I("normaltree", DumpTree(tree));
		SparseNodeData colors = PoissonSplat.SetAuxField(set, parameters.PerLevelDataScaleFactor);
		produced.F("colors", DumpField(tree, colors, 4));
		produced.I("splatinfo", [normals.Count, colors.Count, tree.NodeCount]);

		// Solve's approximate point interpolation constraints (before finalizing).
		float pointWeight = parameters.PointWeight * PoissonInterpolation.AverageSampleWeight(pointDepthAndWeight.WeightSum, pointDepthAndWeight.TotalWeight);
		SparseNodeData interpolation = PoissonInterpolation.Build(set, 0.5f, pointWeight, (int)parameters.Depth, 1);
		produced.F("interpolation", DumpField(tree, interpolation, PoissonInterpolation.Width));
		produced.I("interpolationinfo", [interpolation.Count, tree.NodeCount]);
		return new Prepared(set, parameters, density, normals, colors, interpolation, pointWeight);
	}

	// The harness's setSortedTreeNodes cases.
	private static void AddSortedCases(Cases produced, FemTree tree, SortedTreeNodes sorted, int[] map, Prepared p)
	{
		produced.I("sortedmap", map.Select(m => (double)m).ToList());
		produced.I("sortedtree2", DumpTree(tree));
		produced.I("sortedflags", DumpFlags(tree));
		var slices = new List<double>();
		for (int d = 0; d < sorted.Levels; d++)
		{
			slices.AddRange([sorted.Begin(d), sorted.End(d)]);
		}

		produced.I("sortedslices", slices);
		var remapped = new List<double>();
		tree.ProcessNodes(tree.Root, node =>
		{
			int index = tree.NodeIndex(node);
			int normal = p.Normals.Index(index), density = p.Density.Index(index), color = p.Colors.Index(index);
			if (normal == -1 && density == -1 && color == -1)
			{
				return;
			}

			remapped.AddRange([index, normal, density, color]);
		});
		produced.I("remappedslots", remapped);
		produced.F("remappedinterpolation", DumpField(tree, p.Interpolation, PoissonInterpolation.Width));
	}

	private sealed record Prepared(
		PoissonSampleSet Set,
		PoissonSolutionParameters Parameters,
		DensityEstimator Density,
		SparseNodeData Normals,
		SparseNodeData Colors,
		SparseNodeData Interpolation,
		float PointWeight);
}
