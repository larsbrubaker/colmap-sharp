// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet2 (C#-only, not a COLMAP test): the level-set extractor's
// slice and slab bookkeeping - the full depth, the slice and slab cell indices
// (LevelSetCellIndices), and each slice's corner values, gradients and marching-squares indices
// (PoissonLevelSetExtractor) - in Extract's slab order, against the "levelset*" runs of
// oracle/poisson_levelset2_harness.cc (TestData/oracle/poisson_levelset2.json). Tier A,
// identical integers and bit-identical floats. Each run solves as PostSolveStages_MatchHarness
// does, then extracts the level set at 0.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string LevelSet2Fixture = "poisson_levelset2.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task LevelSetSlices_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(LevelSet2Fixture).GetProperty("cases");
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, _, float[] solution) = SolveAsSolveDoes(input, name, depth);
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, 0f);
		produced.I("fulldepth", [extractor.FullDepth]);

		var dump = new LevelSetDump();
		int step = 0;
		void InitSlice(int sliceAtMaxDepth)
		{
			extractor.InitSlice(sliceAtMaxDepth);
			for (int d = extractor.MaxDepth; d >= extractor.FullDepth && (sliceAtMaxDepth >> (extractor.MaxDepth - d)) << (extractor.MaxDepth - d) == sliceAtMaxDepth; d--)
			{
				int slice = sliceAtMaxDepth >> (extractor.MaxDepth - d);
				dump.Cells(dump.SliceHeads, dump.SliceIndices, extractor.SlabValues[d].SliceValues(slice).CellIndices, step, d, slice);
			}

			step++;
		}

		void InitSlab(int slabAtMaxDepth, bool first)
		{
			extractor.InitSlab(slabAtMaxDepth, first);
			int slab = slabAtMaxDepth;
			for (int d = extractor.MaxDepth; d >= extractor.FullDepth; d--, slab >>= 1)
			{
				dump.Cells(dump.SlabHeads, dump.SlabIndices, extractor.SlabValues[d].SlabCellIndices(slab), step, d, slab);
				if ((slab & 1) != 0 && !first)
				{
					break;
				}
			}

			step++;
		}

		void SetSliceValues(int sliceAtMaxDepth)
		{
			extractor.SetSliceValues(sliceAtMaxDepth);
			for (int d = extractor.MaxDepth, o = sliceAtMaxDepth; d >= extractor.FullDepth; d--, o >>= 1)
			{
				dump.Corners(tree, sorted, extractor.SlabValues[d].SliceValues(o), step, d, o);
				if ((o & 1) != 0)
				{
					break;
				}
			}

			step++;
		}

		// Extract's slab loop, up to its corner values.
		InitSlice(0);
		InitSlab(0, true);
		SetSliceValues(0);
		for (int slab = 0; slab < 1 << extractor.MaxDepth; slab++)
		{
			InitSlice(slab + 1);
			if (slab != 0)
			{
				InitSlab(slab, false);
			}

			SetSliceValues(slab + 1);
		}

		produced.I("slicecells", dump.SliceHeads);
		produced.I("sliceindices", dump.SliceIndices);
		produced.I("slabcells", dump.SlabHeads);
		produced.I("slabindices", dump.SlabIndices);
		produced.I("cornerflags", dump.CornerFlags);
		produced.F("corners", dump.CornerValues);
		produced.I("mcindices", dump.McIndices);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	// The harness's dumps of the extractor's state, in its order.
	private sealed class LevelSetDump
	{
		public List<double> SliceHeads { get; } = [];

		public List<double> SliceIndices { get; } = [];

		public List<double> SlabHeads { get; } = [];

		public List<double> SlabIndices { get; } = [];

		public List<double> CornerFlags { get; } = [];

		public List<double> CornerValues { get; } = [];

		public List<double> McIndices { get; } = [];

		public void Cells(List<double> heads, List<double> indices, LevelSetCellIndices cells, int step, int d, int o)
		{
			heads.AddRange([step, d, o, cells.NodeOffset, cells.Size, cells.Count(0), cells.Count(1), cells.Count(2)]);
			for (int i = cells.NodeOffset; i < cells.NodeOffset + cells.Size; i++)
			{
				for (int k = 0; k < 3; k++)
				{
					for (int j = 0; j < HyperCube.ElementNum(2, k); j++)
					{
						indices.Add(cells.Index(k, i, j));
					}
				}
			}
		}

		public void Corners(FemTree tree, SortedTreeNodes sorted, LevelSetSliceValues values, int step, int d, int o)
		{
			CornerFlags.AddRange([step, d, o]);
			for (int v = 0; v < values.CellIndices.Count(0); v++)
			{
				CornerFlags.Add(values.CornerSet[v]);
				if (values.CornerSet[v] != 0)
				{
					CornerValues.Add(values.CornerValues[v]);
					for (int k = 0; k < 3; k++)
					{
						CornerValues.Add(values.CornerGradients![(3 * v) + k]);
					}
				}
			}

			for (int s = o - 1; s <= o; s++)
			{
				if (s < 0 || s >= 1 << d)
				{
					continue;
				}

				int global = d + tree.DepthOffset;
				int end = sorted.End(global, s + tree.LocalInset(d));
				for (int i = sorted.Begin(global, s + tree.LocalInset(d)); i < end; i++)
				{
					int leaf = sorted.TreeNodes[i];
					if (PoissonMultigrid.IsValidSpaceNode(tree, leaf) && !tree.IsActive(tree.FirstChild(leaf)))
					{
						McIndices.Add(values.McIndices[i - values.CellIndices.NodeOffset]);
					}
				}
			}
		}
	}
}
