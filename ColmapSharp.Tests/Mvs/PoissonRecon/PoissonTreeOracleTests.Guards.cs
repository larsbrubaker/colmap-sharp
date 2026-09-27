// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.Guards (C#-only, not a COLMAP test): PoissonMeshOutput.FromLevelSet
// only accepts an extractor whose Extract ran to completion (not before Extract, not after a
// cancelled or stopped one), and the extractor's step methods are not public, so Extract's
// run-once guard cannot be bypassed from outside the library.

using System.Reflection;
using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	[Test]
	public async Task LevelSetOutput_RequiresACompleteExtract()
	{
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, "levelset3", 3);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;

		var notRun = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		await Assert.That(notRun.IsComplete).IsFalse();
		await Assert.That(() => PoissonMeshOutput.FromLevelSet(notRun, p.Set.UnitCubeToModel)).Throws<InvalidOperationException>();

		var cancelledExtractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		await Assert.That(() => cancelledExtractor.Extract(cancelled.Token)).Throws<OperationCanceledException>();
		await Assert.That(cancelledExtractor.IsComplete).IsFalse();
		await Assert.That(() => PoissonMeshOutput.FromLevelSet(cancelledExtractor, p.Set.UnitCubeToModel)).Throws<InvalidOperationException>();

		var stopped = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		stopped.Extract(default, null, new LevelSetExtractHooks { Stages = LevelSetExtractStages.IsoEdges });
		await Assert.That(stopped.IsComplete).IsFalse();
		await Assert.That(() => PoissonMeshOutput.FromLevelSet(stopped, p.Set.UnitCubeToModel)).Throws<InvalidOperationException>();

		var complete = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		complete.Extract();
		await Assert.That(complete.IsComplete).IsTrue();
		await Assert.That(PoissonMeshOutput.FromLevelSet(complete, p.Set.UnitCubeToModel).VertexCount).IsGreaterThan(0);
	}

	[Test]
	public async Task LevelSetExtractor_StepMethodsAreNotPublic()
	{
		string[] steps =
		[
			"InitSlice", "InitSlab", "SetSliceValues", "SetSliceCornerValuesAndMcIndices", "SetSliceIsoVertices",
			"FinalizeSliceEdges", "SetSlabIsoVertices", "FinalizeSlabEdges", "SetXSliceIsoVertices", "SetSliceIsoEdges",
			"SetSlabIsoEdges", "FinalizeSlice", "FinalizeSlab", "CopyFinerSliceIsoEdgeKeys", "CopyFinerXSliceIsoEdgeKeys",
			"SetXSliceIsoEdges", "IsoSurface", "SetLevelSet",
		];
		string[] publicMethods = typeof(PoissonLevelSetExtractor).GetMethods(BindingFlags.Public | BindingFlags.Instance).Select(m => m.Name).ToArray();
		await Assert.That(steps.Where(publicMethods.Contains).ToArray()).IsEmpty();
		await Assert.That(publicMethods.Contains("Extract")).IsTrue();
	}
}
