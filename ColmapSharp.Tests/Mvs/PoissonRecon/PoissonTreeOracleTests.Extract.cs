// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.Extract (C#-only, not a COLMAP test): the level-set extraction end to
// end (PoissonLevelSetExtractor.Extract) and the output mesh (PoissonMeshOutput) - Solve's
// unitCubeToModel, the vertex and triangle counts, every vertex's model position, density value
// and PLY color bytes, and every triangle - against the "levelset*" runs of
// oracle/poisson_extract_harness.cc, which runs upstream's own Extract through
// TransformedOutputLevelSetVertexStream (TestData/oracle/poisson_extract.json; "levelset3" is
// stored in full, the rest as checksums), plus PlyUChar on the harness's crafted floats. Tier A,
// identical integers and bit-identical floats. Each run solves as PostSolveStages_MatchHarness
// does and extracts at Solve's iso-value with the density and the per-level-scaled color field,
// as COLMAP's --density and --colors do. Also pins Extract's cancellation and progress.
//
// The "vertexpairs*" runs are crafted inputs (stored in the extract fixture itself) where a
// coarse leaf borders finer ones and the surface crosses both halves of a shared edge, which
// the other runs do not reach. Counted with temporary instrumentation when they were added:
// vertexpairs1 (the shell at depth 6) walks loops across the back and front slices' pair maps;
// vertexpairs2 (the shell plus three small spheres, depth 7) makes pairs on slices and slabs
// and pushes both to coarser slices; vertexpairs3 (a thin slab of two sheets, depth 7) walks
// across the slab's pair map. All three reach the pushed-up face-edge map fallbacks, on
// slices and on cross faces, with non-empty iso-edges.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string ExtractFixture = "poisson_extract.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	[Arguments("vertexpairs1", 6)]
	[Arguments("vertexpairs2", 7)]
	[Arguments("vertexpairs3", 7)]
	public async Task LevelSetExtractMesh_MatchesHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(ExtractFixture).GetProperty("cases");

		// The "vertexpairs" run's input is in the extract fixture itself; the others share
		// poisson_levelset.json's.
		JsonElement input = cases.TryGetProperty(name + "/input", out _) ? cases : OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		produced.F("isovalue", [isoValue]);
		var xform = new List<double>();
		AddMatrix(xform, p.Set.UnitCubeToModel);
		produced.F("mesh/unitcubetomodel", xform);

		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);
		var fractions = new List<double>();
		extractor.Extract(progress: new SynchronousFractionProgress(fractions.Add));
		PoissonMeshOutput mesh = PoissonMeshOutput.FromLevelSet(extractor, p.Set.UnitCubeToModel);

		produced.I("mesh/vertexcount", [mesh.VertexCount]);
		produced.I("mesh/trianglecount", [mesh.Triangles.Length / 3]);
		produced.F("mesh/positions", mesh.Positions.Select(v => (double)v).ToList());
		produced.F("mesh/values", mesh.Values!.Select(v => (double)v).ToList());
		produced.I("mesh/colors", mesh.Colors!.Select(v => (double)v).ToList());
		produced.I("mesh/triangles", mesh.Triangles.Select(v => (double)v).ToList());
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(8);
		await Assert.That(mesh.ColorChannels).IsEqualTo(3);

		// One report per slab at the finest depth, ending at 1.
		await Assert.That(fractions.Count).IsEqualTo(1 << extractor.MaxDepth);
		await Assert.That(fractions[^1]).IsEqualTo(1.0);
		await Assert.That(() => extractor.Extract()).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task LevelSetExtract_WithoutDensityOrColors_OmitsThem()
	{
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, "levelset3", 3);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		extractor.Extract();
		PoissonMeshOutput mesh = PoissonMeshOutput.FromLevelSet(extractor, p.Set.UnitCubeToModel);

		// The geometry does not depend on the density or the colors.
		JsonElement cases = OracleFixture.Load(ExtractFixture).GetProperty("cases");
		string? positions = FirstMismatch(cases.GetProperty("levelset3/mesh/positions"), new Case(true, mesh.Positions.Select(v => (double)v).ToList()));
		string? triangles = FirstMismatch(cases.GetProperty("levelset3/mesh/triangles"), new Case(false, mesh.Triangles.Select(v => (double)v).ToList()));
		await Assert.That(positions).IsNull();
		await Assert.That(triangles).IsNull();
		await Assert.That(mesh.Values).IsNull();
		await Assert.That(mesh.Colors).IsNull();
		await Assert.That(mesh.ColorChannels).IsEqualTo(0);
	}

	[Test]
	public async Task LevelSetExtract_Cancelled_Throws()
	{
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, "levelset3", 3);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		await Assert.That(() => extractor.Extract(cancelled.Token)).Throws<OperationCanceledException>();
	}

	[Test]
	public async Task PlyUChar_MatchesHarness()
	{
		JsonElement cases = OracleFixture.Load(ExtractFixture).GetProperty("cases");
		double[] values = cases.GetProperty("plycolor/in").EnumerateArray().Select(e => e.GetDouble()).ToArray();
		var produced = new Cases("plycolor");
		produced.F("in", [.. values]);
		produced.I("out", values.Select(v => (double)PoissonMeshOutput.PlyUChar((float)v)).ToList());
		await Assert.That(CompareRun(cases, "plycolor", produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(2);
	}

	// IProgress that reports on the caller's thread (Progress<T> posts to the thread pool).
	private sealed class SynchronousFractionProgress(Action<double> report) : IProgress<double>
	{
		public void Report(double value) => report(value);
	}
}
