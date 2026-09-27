// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet5 (C#-only, not a COLMAP test): the level-set extractor's
// iso-edges (PoissonLevelSetExtractor.IsoEdges) with the iso-vertices they follow, run by
// PoissonLevelSetExtractor.Extract stopped before the polygons - for each finalized slice and slab: the
// edge keys that are set (own vertices and keys copied from finer edges), the iso-edges of
// each face that is set, the face-edge map and the vertex-pair map (sorted by key), plus the
// vertex count - against the "levelset*" runs of oracle/poisson_levelset5_harness.cc
// (TestData/oracle/poisson_levelset5.json; "levelset3" is stored in full, the rest as
// checksums). Tier A, identical integers. Each run solves as PostSolveStages_MatchHarness does
// and extracts at Solve's iso-value with the density and the per-level-scaled color field, as
// COLMAP's --density and --colors do.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string LevelSet5Fixture = "poisson_levelset5.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task LevelSetIsoEdges_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(LevelSet5Fixture).GetProperty("cases");
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		produced.F("isovalue", [isoValue]);
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);

		var slices = new IsoEdgeDumps();
		var slabs = new IsoEdgeDumps();

		// The harness's step: 0 for the first slice, then slab + 1 for each slab's slice and slab.
		void FinalizeSlice(int sliceAtMaxDepth)
		{
			int step = sliceAtMaxDepth;
			for (int d = extractor.MaxDepth, o = sliceAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				LevelSetSliceValues v = extractor.SlabValues[d].SliceValues(o);
				slices.Add(step, d, o, v.EdgeSet, v.EdgeKeys, v.CellIndices.Count(1), v.FaceSet, v.FaceEdges, v.CellIndices.Count(2), v.FaceEdgeMap, v.VertexPairMap);
				if ((o & 1) != 0)
				{
					break;
				}
			}
		}

		void FinalizeSlab(int slabAtMaxDepth)
		{
			int step = slabAtMaxDepth + 1;
			for (int d = extractor.MaxDepth, o = slabAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				LevelSetXSliceValues v = extractor.SlabValues[d].XSliceValues(o);
				slabs.Add(step, d, o, v.EdgeSet, v.EdgeKeys, v.CellIndices.Count(0), v.FaceSet, v.FaceEdges, v.CellIndices.Count(1), v.FaceEdgeMap, v.VertexPairMap);
				if ((o & 1) == 0)
				{
					break;
				}
			}
		}

		// Extract's slab loop, without IsoSurface (the polygons).
		extractor.Extract(CancellationToken.None, null, new LevelSetExtractHooks
		{
			Stages = LevelSetExtractStages.IsoEdges,
			SliceFinalized = FinalizeSlice,
			SlabFinalized = FinalizeSlab,
		});

		produced.I("vertexcount", [extractor.Vertices.Count]);
		produced.I("isoedges/slicekeys", slices.Keys);
		produced.I("isoedges/slicefaces", slices.Faces);
		produced.I("isoedges/slicefacemaps", slices.FaceMaps);
		produced.I("isoedges/slicepairs", slices.Pairs);
		produced.I("isoedges/slabkeys", slabs.Keys);
		produced.I("isoedges/slabfaces", slabs.Faces);
		produced.I("isoedges/slabfacemaps", slabs.FaceMaps);
		produced.I("isoedges/slabpairs", slabs.Pairs);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(10);
	}

	// The harness's four dumps of each finalized slice or slab (its Dump), in its layout.
	private sealed class IsoEdgeDumps
	{
		public List<double> Keys { get; } = [];

		public List<double> Faces { get; } = [];

		public List<double> FaceMaps { get; } = [];

		public List<double> Pairs { get; } = [];

		public void Add(int step, int d, int o, byte[] eSet, LevelSetKey[] edgeKeys, int eCount, byte[] fSet, LevelSetFaceEdges[] faceEdges, int fCount, Dictionary<LevelSetKey, List<LevelSetIsoEdge>> faceEdgeMap, Dictionary<LevelSetKey, LevelSetKey> vertexPairMap)
		{
			var keys = new List<double>();
			for (int i = 0; i < eCount; i++)
			{
				if (eSet[i] != 0)
				{
					keys.Add(i);
					AddKey(keys, edgeKeys[i]);
				}
			}

			Keys.AddRange([step, d, o, keys.Count / 4]);
			Keys.AddRange(keys);

			var faces = new List<double>();
			int n = 0;
			for (int i = 0; i < fCount; i++)
			{
				if (fSet[i] != 0)
				{
					LevelSetFaceEdges fe = faceEdges[i];
					faces.AddRange([i, fe.Count]);
					for (int j = 0; j < fe.Count; j++)
					{
						AddKey(faces, fe[j].First);
						AddKey(faces, fe[j].Second);
					}

					n++;
				}
			}

			Faces.AddRange([step, d, o, n]);
			Faces.AddRange(faces);

			// Sorted by key, as the harness sorts upstream's unordered_maps.
			FaceMaps.AddRange([step, d, o, faceEdgeMap.Count]);
			foreach ((LevelSetKey key, List<LevelSetIsoEdge> edges) in faceEdgeMap.OrderBy(kv => (kv.Key.X, kv.Key.Y, kv.Key.Z)))
			{
				AddKey(FaceMaps, key);
				FaceMaps.Add(edges.Count);
				foreach (LevelSetIsoEdge e in edges)
				{
					AddKey(FaceMaps, e.First);
					AddKey(FaceMaps, e.Second);
				}
			}

			Pairs.AddRange([step, d, o, vertexPairMap.Count]);
			foreach ((LevelSetKey key, LevelSetKey pair) in vertexPairMap.OrderBy(kv => (kv.Key.X, kv.Key.Y, kv.Key.Z)))
			{
				AddKey(Pairs, key);
				AddKey(Pairs, pair);
			}
		}

		private static void AddKey(List<double> output, LevelSetKey key) => output.AddRange([key.X, key.Y, key.Z]);
	}
}
