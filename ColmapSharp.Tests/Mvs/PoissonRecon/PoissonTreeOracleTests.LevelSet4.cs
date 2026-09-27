// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet4 (C#-only, not a COLMAP test): every iso-vertex of the
// level-set extractor, on slice edges (PoissonLevelSetExtractor.IsoVertices) and on cross-slice
// edges (PoissonLevelSetExtractor.XSliceIsoVertices), interleaved as Extract runs them (a
// slab's cross-edge vertices before the next slice's edge vertices) - every vertex in write
// order (position, gradient, density depth, color), the colors alone in full for the smaller
// runs, each finalized slice's and slab's edge-vertex map, and the bad-root count - against the
// "levelset*" runs of oracle/poisson_levelset4_harness.cc (TestData/oracle/poisson_levelset4.json).
// Tier A, identical integers and bit-identical floats. Each run solves as
// PostSolveStages_MatchHarness does and extracts at Solve's iso-value with the density and the
// per-level-scaled color field, as COLMAP's --density and --colors do.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string LevelSet4Fixture = "poisson_levelset4.json";

	[Test]
	[Arguments("levelset3", 3, true)]
	[Arguments("levelset5", 5, true)]
	[Arguments("levelset6", 6, true)]
	[Arguments("levelset8", 8, false)]
	public async Task LevelSetIsoVertices_MatchHarness(string name, int depth, bool fullColors)
	{
		JsonElement cases = OracleFixture.Load(LevelSet4Fixture).GetProperty("cases");
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		produced.F("isovalue", [isoValue]);
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);

		var edgeMaps = new List<double>();
		var slabMaps = new List<double>();
		int step = 0;

		// A key map sorted by key (the harness sorts upstream's unordered_map the same way).
		void DumpMap(Dictionary<LevelSetKey, int> map, int d, int o, List<double> output)
		{
			var entries = map.Select(kv => (kv.Key.X, kv.Key.Y, kv.Key.Z, kv.Value)).Order().ToList();
			output.AddRange([step, d, o, entries.Count]);
			foreach ((uint x, uint y, uint z, int vertex) in entries)
			{
				output.AddRange([x, y, z, vertex]);
			}
		}

		void FinalizeSlice(int sliceAtMaxDepth)
		{
			extractor.FinalizeSliceEdges(sliceAtMaxDepth);
			for (int d = extractor.MaxDepth, o = sliceAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				DumpMap(extractor.SlabValues[d].SliceValues(o).EdgeVertexMap, d, o, edgeMaps);
				if ((o & 1) != 0)
				{
					break;
				}
			}
		}

		void FinalizeSlab(int slabAtMaxDepth)
		{
			extractor.FinalizeSlabEdges(slabAtMaxDepth);
			for (int d = extractor.MaxDepth, o = slabAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				DumpMap(extractor.SlabValues[d].XSliceValues(o).EdgeVertexMap, d, o, slabMaps);
				if ((o & 1) == 0)
				{
					break;
				}
			}
		}

		// Extract's slab loop, without the iso-edges and the polygons.
		extractor.InitSlice(0);
		extractor.InitSlab(0, true);
		extractor.SetSliceValues(0);
		extractor.SetSliceIsoVertices(0);
		FinalizeSlice(0);
		step++;
		for (int slab = 0; slab < 1 << extractor.MaxDepth; slab++)
		{
			extractor.InitSlice(slab + 1);
			if (slab != 0)
			{
				extractor.InitSlab(slab, false);
			}

			extractor.SetSliceValues(slab + 1);
			extractor.SetSlabIsoVertices(slab);
			extractor.SetSliceIsoVertices(slab + 1);
			FinalizeSlice(slab + 1);
			FinalizeSlab(slab);
			step++;
		}

		var vertices = new List<double>();
		var colors = new List<double>();
		foreach (LevelSetVertex v in extractor.Vertices)
		{
			vertices.AddRange([v.X, v.Y, v.Z, v.GradientX, v.GradientY, v.GradientZ, v.Depth]);
			vertices.AddRange(v.Data.Select(c => (double)c));
			colors.AddRange(v.Data.Select(c => (double)c));
		}

		produced.I("vertexcount", [extractor.Vertices.Count]);
		produced.F("vertices", vertices);
		if (fullColors)
		{
			produced.F("vertexcolors", colors);
		}

		produced.I("edgemaps", edgeMaps);
		produced.I("slabmaps", slabMaps);
		produced.I("badroots", [extractor.BadRootCount]);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(fullColors ? 7 : 6);
	}
}
