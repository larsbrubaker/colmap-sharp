// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.Trim (C#-only, not a COLMAP test): COLMAP's mesh trimming
// (PoissonSurfaceTrimmer, with TrimmerEdgeHashTable's libc++ iteration order in the island
// merge) against oracle/poisson_trim_harness.cc, which runs upstream's own RunSurfaceTrimmer
// with poisson_meshing.cc's arguments on the extracted meshes of
// PoissonTreeOracleTests.Extract, through the PLY files COLMAP writes between the two
// (TestData/oracle/poisson_trim.json; "levelset3" in full, the rest as checksums). Per run: the
// input's density range, then for each trim value (COLMAP's default 10, above every density
// here, and values inside the range that split polygons and merge islands) the kept vertex
// and triangle counts, positions, density values, color bytes and triangles. Tier A,
// identical integers and bit-identical floats. Also pins the trimmer's cancellation and
// progress and its refusal of a mesh without density values. Two crafted meshes add island
// merges the extracted meshes never reach: "crafted" (small components between two
// neighbors) and "hubs" (small components with 4 to 9 neighbors).

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string TrimFixture = "poisson_trim.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task SurfaceTrimmer_MatchesHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(TrimFixture).GetProperty("cases");
		PoissonMeshOutput mesh = ExtractAsSolveDoes(name, depth);
		var produced = new Cases(name);
		produced.F("mesh/valuerange", [mesh.Values!.Min(), mesh.Values!.Max()]);
		int trimCount = AddTrims(produced, cases, name, mesh);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(2 + (6 * trimCount));
	}

	[Test]
	[Arguments("crafted", 3)]
	[Arguments("hubs", 1)]
	public async Task SurfaceTrimmer_Crafted_MatchesHarness(string name, int trimCount)
	{
		// The harness's crafted meshes, read from the fixture (and echoed back). "crafted", a
		// wavy grid with a noisy density: trimming it leaves many small components, some
		// between two neighbors, so the libc++ neighbor order decides where they merge. "hubs":
		// each below-trim hub touches 4 to 9 kept blocks and merges with all of them, so the
		// output triangle order follows the whole neighbor order, which libc++'s iteration of
		// componentHalfEdges and componentBoundaryHalfEdges (through their insertion into the
		// next container) feeds as much as componentEdges'. Walking either of the first two in
		// insertion order, or reversed, changes the "hubs" triangles; "crafted" misses that.
		JsonElement cases = OracleFixture.Load(TrimFixture).GetProperty("cases");
		var produced = new Cases(name);
		float[] Floats(string key) => cases.GetProperty(name + "/input/" + key).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
		float[] positions = Floats("positions");
		float[] values = Floats("values");
		byte[] colors = cases.GetProperty(name + "/input/colors").EnumerateArray().Select(e => (byte)e.GetInt32()).ToArray();
		int[] triangles = cases.GetProperty(name + "/input/triangles").EnumerateArray().Select(e => e.GetInt32()).ToArray();
		var mesh = new PoissonMeshOutput(values.Length, 3, positions, values, colors, triangles);
		AddTrimmedMesh(produced, "input/", mesh);
		produced.F("mesh/valuerange", [values.Min(), values.Max()]);
		int trims = AddTrims(produced, cases, name, mesh);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(trims).IsEqualTo(trimCount);
		await Assert.That(produced.Count).IsEqualTo(6 + 2 + (6 * trimCount));
	}

	[Test]
	public async Task SurfaceTrimmer_ReportsProgressAndCancels()
	{
		PoissonMeshOutput mesh = ExtractAsSolveDoes("levelset3", 3);
		var fractions = new List<double>();
		PoissonSurfaceTrimmer.Trim(mesh, 2.6f, progress: new SynchronousFractionProgress(fractions.Add));
		await Assert.That(fractions).IsEquivalentTo(new List<double> { 0.25, 0.5, 0.75, 1.0 });

		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		await Assert.That(() => PoissonSurfaceTrimmer.Trim(mesh, 2.6f, cancelled.Token)).Throws<OperationCanceledException>();
	}

	[Test]
	public async Task SurfaceTrimmer_WithoutDensity_Throws()
	{
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, "levelset3", 3);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue);
		extractor.Extract();
		PoissonMeshOutput mesh = PoissonMeshOutput.FromLevelSet(extractor, p.Set.UnitCubeToModel);
		await Assert.That(() => PoissonSurfaceTrimmer.Trim(mesh, 2.6f)).Throws<ArgumentException>();
	}

	// Trims mesh at each of the run's trim values and adds the fixture's "trims" and per-trim
	// meshes; returns the number of trim values.
	private static int AddTrims(Cases produced, JsonElement cases, string name, PoissonMeshOutput mesh)
	{
		double[] trims = cases.GetProperty(name + "/mesh/trims").EnumerateArray().Select(e => e.GetDouble()).ToArray();
		produced.F("mesh/trims", [.. trims]);
		for (int t = 0; t < trims.Length; t++)
		{
			// poisson_meshing.cc formats the trim with std::to_string and the trimmer parses it
			// into a float; for these values that is the float nearest the double.
			AddTrimmedMesh(produced, $"mesh/trim{t}/", PoissonSurfaceTrimmer.Trim(mesh, (float)trims[t]));
		}

		return trims.Length;
	}

	private static void AddTrimmedMesh(Cases produced, string prefix, PoissonMeshOutput trimmed)
	{
		produced.I(prefix + "vertexcount", [trimmed.VertexCount]);
		produced.I(prefix + "trianglecount", [trimmed.Triangles.Length / 3]);
		produced.F(prefix + "positions", trimmed.Positions.Select(v => (double)v).ToList());
		produced.F(prefix + "values", trimmed.Values!.Select(v => (double)v).ToList());
		produced.I(prefix + "colors", trimmed.Colors!.Select(v => (double)v).ToList());
		produced.I(prefix + "triangles", trimmed.Triangles.Select(v => (double)v).ToList());
	}

	// The extracted mesh of PoissonTreeOracleTests.Extract's run: Solve's iso-value, with the
	// density and the per-level-scaled color field, as COLMAP's --density and --colors ask.
	private static PoissonMeshOutput ExtractAsSolveDoes(string name, int depth)
	{
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);
		extractor.Extract();
		return PoissonMeshOutput.FromLevelSet(extractor, p.Set.UnitCubeToModel);
	}
}
