// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet3 (C#-only, not a COLMAP test): the level-set extractor's
// iso-vertices on slice edges (PoissonLevelSetExtractor.IsoVertices) - every vertex in output
// order (position, gradient, density depth, color), each finalized slice's edge-vertex map, the
// edge keys pushed into coarser slabs, and the bad-root count - in Extract's slab order without
// the cross-slice vertices, against the "levelset*" runs of oracle/poisson_levelset3_harness.cc
// (TestData/oracle/poisson_levelset3.json). Tier A, identical integers and bit-identical
// floats. Each run solves as PostSolveStages_MatchHarness does and extracts at Solve's
// iso-value with the density and the per-level-scaled color field, as COLMAP's --density and
// --colors do.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string LevelSet3Fixture = "poisson_levelset3.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task LevelSetSliceIsoVertices_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(LevelSet3Fixture).GetProperty("cases");
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		produced.F("isovalue", [isoValue]);
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);

		var edgeMaps = new List<double>();
		var slabKeys = new List<double>();
		int step = 0;
		void FinalizeSlice(int sliceAtMaxDepth)
		{
			extractor.FinalizeSliceEdges(sliceAtMaxDepth);
			for (int d = extractor.MaxDepth, o = sliceAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				// The map, sorted by key (the harness sorts upstream's unordered_map the same way).
				var entries = extractor.SlabValues[d].SliceValues(o).EdgeVertexMap
					.Select(kv => (kv.Key.X, kv.Key.Y, kv.Key.Z, kv.Value))
					.Order()
					.ToList();
				edgeMaps.AddRange([step, d, o, entries.Count]);
				foreach ((uint x, uint y, uint z, int vertex) in entries)
				{
					edgeMaps.AddRange([x, y, z, vertex]);
				}

				if ((o & 1) != 0)
				{
					break;
				}
			}

			for (int d = extractor.MaxDepth; d >= extractor.FullDepth; d--)
			{
				for (int parity = 0; parity < 2; parity++)
				{
					List<(LevelSetKey Key, int Vertex)> keys = extractor.SlabValues[d].SlabEdgeKeyValues(parity);
					if (keys.Count == 0)
					{
						continue;
					}

					slabKeys.AddRange([step, d, parity, keys.Count]);
					foreach ((LevelSetKey key, int vertex) in keys)
					{
						slabKeys.AddRange([key.X, key.Y, key.Z, vertex]);
					}
				}
			}

			step++;
		}

		// Extract's slab loop, without the cross-slice vertices, iso-edges and polygons.
		extractor.InitSlice(0);
		extractor.InitSlab(0, true);
		extractor.SetSliceValues(0);
		extractor.SetSliceIsoVertices(0);
		FinalizeSlice(0);
		for (int slab = 0; slab < 1 << extractor.MaxDepth; slab++)
		{
			extractor.InitSlice(slab + 1);
			if (slab != 0)
			{
				extractor.InitSlab(slab, false);
			}

			extractor.SetSliceValues(slab + 1);
			extractor.SetSliceIsoVertices(slab + 1);
			FinalizeSlice(slab + 1);
		}

		var vertices = new List<double>();
		foreach (LevelSetVertex v in extractor.Vertices)
		{
			vertices.AddRange([v.X, v.Y, v.Z, v.GradientX, v.GradientY, v.GradientZ, v.Depth]);
			vertices.AddRange(v.Data.Select(c => (double)c));
		}

		produced.I("vertexcount", [extractor.Vertices.Count]);
		produced.F("vertices", vertices);
		produced.I("edgemaps", edgeMaps);
		produced.I("slabkeys", slabKeys);
		produced.I("badroots", [extractor.BadRootCount]);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(6);
	}
}
