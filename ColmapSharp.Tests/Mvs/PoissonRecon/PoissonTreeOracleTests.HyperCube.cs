// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.HyperCube (C#-only, not a COLMAP test): the level-set extractor's
// hypercube algebra (HyperCube) and its cached tables (HyperCubeTables) for the 1-, 2- and
// 3-cubes, against oracle/poisson_hypercube_harness.cc (TestData/oracle/poisson_hypercube.json).
// Tier A, identical integers.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	private const string HyperCubeFixture = "poisson_hypercube.json";

	[Test]
	public async Task HyperCubeTables_MatchHarness()
	{
		JsonElement cases = OracleFixture.Load(HyperCubeFixture).GetProperty("cases");
		var produced = new Cases("hypercube");
		for (int d = 1; d <= 3; d++)
		{
			string cube = $"d{d}/";
			var roots = new List<double>();
			for (int m = 0; m < 1 << (1 << d); m++)
			{
				roots.Add(HyperCube.HasMcRoots(d, m) ? 1 : 0);
			}

			produced.I(cube + "hasmcroots", roots);
			for (int k1 = 0; k1 <= d; k1++)
			{
				for (int k2 = 0; k2 <= d; k2++)
				{
					HyperCubeOverlapTable table = HyperCubeTables.Of(d, k1, k2);
					var flags = new List<double>();
					var elements = new List<double>();
					for (int e = 0; e < table.Overlap.Length; e++)
					{
						flags.AddRange(table.Overlap[e].Select(o => o ? 1.0 : 0.0));
						elements.AddRange(table.OverlapElements[e].Select(x => (double)x));
					}

					produced.I($"{cube}overlap{k1}{k2}/flags", flags);
					produced.I($"{cube}overlap{k1}{k2}/elements", elements);
				}
			}

			for (int k = 0; k <= d; k++)
			{
				AddElementTable(produced, d, k);
			}
		}

		// MCIndex over corner values with ties at the iso-value, as the harness builds them.
		var mc = new List<double>();
		Span<float> values = stackalloc float[8];
		for (int m = 0; m < 256; m++)
		{
			for (int c = 0; c < 8; c++)
			{
				values[c] = ((m >> c) & 1) != 0 ? -0.25f : (c % 3 == 0 ? 0.5f : 0f);
			}

			mc.Add(HyperCube.McIndex(values, 0f));
		}

		produced.I("mcindex", mc);
		await Assert.That(CompareRun(cases, "hypercube", produced)).IsEqualTo(string.Empty);
	}

	private static void AddElementTable(Cases produced, int d, int k)
	{
		string prefix = $"d{d}/k{k}/";
		HyperCubeElementTable table = HyperCubeTables.Of(d, k);
		var cellOffset = new List<double>();
		var coIndex = new List<double>();
		var index = new List<double>();
		var offsets = new List<double>();
		var antipodal = new List<double>();
		var factor = new List<double>();
		var mc = new List<double>();
		var directions = new List<double>();
		Span<int> x = stackalloc int[d];
		for (int e = 0; e < table.ElementNum; e++)
		{
			for (int i = 0; i < table.IncidentCubeNum; i++)
			{
				cellOffset.Add(table.CellOffset[e][i]);
				coIndex.Add(table.IncidentElementCoIndex[e][i]);
				index.Add(table.IncidentElementIndex[e][i]);
				HyperCube.CellOffset(d, k, e, i, x);
				foreach (int xi in x)
				{
					offsets.Add(xi);
				}
			}

			directions.AddRange(table.Directions[e].Select(dir => (double)(int)dir));
			antipodal.Add(HyperCube.Antipodal(d, k, e));
			HyperCube.Factor(d, k, e, out HyperCubeDirection dir, out int co);
			factor.Add((int)dir);
			factor.Add(co);
			for (int m = 0; m < 1 << (1 << d); m++)
			{
				mc.Add(HyperCube.ElementMcIndex(d, k, e, m));
			}
		}

		produced.I(prefix + "celloffset", cellOffset);
		produced.I(prefix + "incidentcoindex", coIndex);
		produced.I(prefix + "incidentindex", index);
		produced.I(prefix + "celloffsetantipodal", table.CellOffsetAntipodal.Select(v => (double)v).ToList());
		produced.I(prefix + "incidentcube", table.IncidentCube.Select(v => (double)v).ToList());
		produced.I(prefix + "directions", directions);
		produced.I(prefix + "antipodal", antipodal);
		produced.I(prefix + "factor", factor);
		produced.I(prefix + "celloffsetxyz", offsets);
		produced.I(prefix + "elementmcindex", mc);
		if (k + 1 == d)
		{
			produced.I(prefix + "oriented", Enumerable.Range(0, table.ElementNum).Select(e => HyperCube.IsOriented(d, e) ? 1.0 : 0.0).ToList());
		}
	}
}
