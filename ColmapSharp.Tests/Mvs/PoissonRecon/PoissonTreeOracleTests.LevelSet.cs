// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet (C#-only, not a COLMAP test): what Poisson::Solver::Solve
// does after the linear solve - the coarse coefficients, the implicit function at each sample
// and the iso-value (PoissonImplicitEvaluator), and the level-set extractor's corner values and
// gradients at every leaf corner (PoissonCornerEvaluator) - against the "levelset*" runs of
// oracle/poisson_levelset_harness.cc (TestData/oracle/poisson_levelset.json). Tier A,
// bit-identical floats. Each run replays the harness's input through the stages the other
// PoissonTreeOracleTests files check, solves as Solve does, then compares the stages here.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	// oracle/poisson_levelset_harness.cc's post-solve stages.
	private const string LevelSetFixture = "poisson_levelset.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task PostSolveStages_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(cases, name, depth);

		var evaluator = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution);
		produced.F("coarsecoefficients", evaluator.Coarse.Select(v => (double)v).ToList());
		var values = new List<double>();
		for (int j = 0; j < p.Set.Count; j++)
		{
			float w = p.Set.Weight(j);
			if (w > 0)
			{
				values.Add(evaluator.Value(p.Set.Position(j, 0) / w, p.Set.Position(j, 1) / w, p.Set.Position(j, 2) / w, p.Set.Node(j)));
			}
		}

		produced.F("samplevalues", values);
		var stages = new List<PoissonProgress>();
		PoissonIsoValue iso = evaluator.IsoValue(p.Set, new SynchronousProgress<PoissonProgress>(stages.Add));
		produced.F("isovalue", [iso.ValueSum, iso.WeightSum, iso.Value]);

		// The extractor's corner evaluation at every corner of every leaf, in sorted-node order.
		var cornerEvaluator = new PoissonCornerEvaluator(tree, PoissonFemConstraints.TestSignature);
		var corners = new List<double>();
		Span<float> cornerValues = stackalloc float[4];
		int maxDepth = PoissonMultigrid.MaxDepth(tree);
		for (int d = 0; d <= maxDepth; d++)
		{
			int end = PoissonMultigrid.End(tree, sorted, d);
			for (int i = PoissonMultigrid.Begin(tree, sorted, d); i < end; i++)
			{
				int leaf = sorted.TreeNodes[i];
				if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
				{
					continue;
				}

				bool isInterior = cornerEvaluator.IsInterior(leaf);
				for (int c = 0; c < 8; c++)
				{
					cornerEvaluator.Values(leaf, c, solution, evaluator.Coarse, isInterior, cornerValues);
					for (int k = 0; k < 4; k++)
					{
						corners.Add(cornerValues[k]);
					}
				}
			}
		}

		produced.F("cornervalues", corners);
		await Assert.That(stages.All(s => s.Stage == PoissonStage.IsoValue) && stages[^1].Fraction == 1).IsTrue();
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task IsoValue_StopsWhenCancelled()
	{
		// C#-only: the average checks the token before its first sample.
		JsonElement cases = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(cases, "levelset3", 3);
		var evaluator = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution);
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		await Assert.That(() => evaluator.IsoValue(p.Set, cancellationToken: cancelled.Token)).Throws<OperationCanceledException>();
	}

	// oracle/poisson_solve.h's PoissonRun: Prepare, then SolveSystem (FEM and point constraints,
	// then the cascadic solve to the full depth).
	private static (FemTree Tree, SortedTreeNodes Sorted, Prepared P, float[] Solution) SolveAsSolveDoes(JsonElement cases, string name, int depth)
	{
		(FemTree tree, SortedTreeNodes sorted, Prepared p) = Finalize(cases, name, depth, new Cases(name));
		var constraints = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, PoissonFemConstraints.CreateDivergenceIntegrator(), p.Normals, constraints, depth);
		PoissonFemConstraints.AddInterpolationConstraints(tree, sorted, p.Interpolation, constraints, depth);
		var system = new FemSystemIntegrator(PoissonFemConstraints.TestSignature, 1, 0.0, 1.0);
		PoissonSolutionParameters parameters = p.Parameters;
		int baseDepth = (int)parameters.BaseDepth, solveDepth = (int)parameters.SolveDepth;
		var solver = new PoissonSystem(tree, sorted, system, new PoissonPointEvaluator(PoissonFemConstraints.TestSignature, 1, solveDepth), p.Interpolation, p.PointWeight);
		float[] solution = solver.Solve(constraints, baseDepth, baseDepth, solveDepth, (int)parameters.Iters, (int)parameters.BaseVCycles, parameters.CgSolverAccuracy, constrainsDCTerm: true);
		return (tree, sorted, p, solution);
	}

	// An IProgress that reports on the calling thread.
	private sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
	{
		public void Report(T value) => report(value);
	}
}
