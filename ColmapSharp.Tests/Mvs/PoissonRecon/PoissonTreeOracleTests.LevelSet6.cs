// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.LevelSet6 (C#-only, not a COLMAP test): the level-set extractor's
// polygons (PoissonLevelSetExtractor.Polygons, MinimalAreaTriangulation), run through
// Extract's whole slab loop (PoissonLevelSetExtractor.Extract) - the total vertex count, every triangle in write order and each
// barycenter vertex AddIsoPolygons writes (index, position, gradient, depth, color) - against
// the "levelset*" runs of oracle/poisson_levelset6_harness.cc
// (TestData/oracle/poisson_levelset6.json; "levelset3" is stored in full, the rest as
// checksums). Tier A, identical integers and bit-identical floats. Each run solves as
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
	private const string LevelSet6Fixture = "poisson_levelset6.json";

	[Test]
	[Arguments("levelset3", 3)]
	[Arguments("levelset5", 5)]
	[Arguments("levelset6", 6)]
	[Arguments("levelset8", 8)]
	public async Task LevelSetPolygons_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(LevelSet6Fixture).GetProperty("cases");
		JsonElement input = OracleFixture.Load(LevelSetFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p, float[] solution) = SolveAsSolveDoes(input, name, depth);
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution).IsoValue(p.Set).Value;
		produced.F("isovalue", [isoValue]);
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, p.Density, p.Colors);

		// The vertices written while IsoSurface runs (after the slab is finalized) are the
		// barycenters.
		var barycenters = new List<int>();
		int before = 0;
		extractor.Extract(CancellationToken.None, null, new LevelSetExtractHooks
		{
			SlabFinalized = _ => before = extractor.Vertices.Count,
			SlabExtracted = _ => barycenters.AddRange(Enumerable.Range(before, extractor.Vertices.Count - before)),
		});

		var triangles = new List<double>();
		foreach (int[] polygon in extractor.Polygons)
		{
			triangles.Add(polygon.Length);
			triangles.AddRange(polygon.Select(v => (double)v));
		}

		var centers = new List<double>();
		foreach (int index in barycenters)
		{
			LevelSetVertex v = extractor.Vertices[index];
			centers.AddRange([v.X, v.Y, v.Z, v.GradientX, v.GradientY, v.GradientZ, v.Depth]);
			centers.AddRange(v.Data.Select(c => (double)c));
		}

		produced.I("vertexcount", [extractor.Vertices.Count]);
		produced.I("polygoncount", [extractor.Polygons.Count]);
		produced.I("polygons", triangles);
		produced.I("polygonbarycenterindices", barycenters.Select(v => (double)v).ToList());
		produced.F("polygonbarycenters", centers);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(6);
		await Assert.That(extractor.InvalidFaceCount).IsEqualTo(0);
	}

	[Test]
	public async Task MinimalAreaTriangulation_MatchesHarness()
	{
		// The harness's crafted polygons (NextUnit points, 4 to 9 vertices), each triangulated
		// by upstream's MinimalAreaTriangulation: the triangle count, then the triangles.
		JsonElement cases = OracleFixture.Load(LevelSet6Fixture).GetProperty("cases");
		var produced = new Cases("mat");
		double[] points = cases.GetProperty("mat/points").EnumerateArray().Select(e => e.GetDouble()).ToArray();
		double[] sizes = cases.GetProperty("mat/sizes").EnumerateArray().Select(e => e.GetDouble()).ToArray();
		produced.F("points", [.. points]);
		produced.I("sizes", [.. sizes]);
		var triangles = new List<double>();
		int offset = 0;
		foreach (double size in sizes)
		{
			int n = (int)size;
			float[] xyz = new float[3 * n];
			for (int k = 0; k < 3 * n; k++)
			{
				xyz[k] = (float)points[offset + k];
			}

			offset += 3 * n;
			List<(int A, int B, int C)> result = MinimalAreaTriangulation.Triangulate(xyz, n);
			triangles.Add(result.Count);
			foreach ((int a, int b, int c) in result)
			{
				triangles.AddRange([a, b, c]);
			}
		}

		produced.I("triangles", triangles);
		await Assert.That(CompareRun(cases, "mat", produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(3);
	}
}
