// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonMeshingOracleTests (C#-only, not a COLMAP test): PoissonMeshing end to end against the
// pinned pycolmap wheel's poisson_meshing (oracle/fixture_poisson_meshing.py writes
// TestData/oracle/poisson_meshing.json) on a noisy, colored sphere, screened and unscreened,
// with and without trimming. Tier C: every PoissonRecon stage is pinned bit-exact against the
// vendored C++ built at -O1 (PoissonTreeOracleTests), but the wheel is optimized and may round
// a few floats differently (docs/CPP_DIVERGENCES.md, entry 125): positions come out a few ulps
// apart, so MinimalAreaTriangulation splits some near-tie polygons along the other diagonal, and
// a few vertices near the trim value land on the other side of it. The bar: the same PLY
// property layout; untrimmed, the same vertex and triangle counts, positions within
// PositionTolerance, colors within one step, at most RetriangulatedFraction of the triangles
// different and the area within 1e-4; trimmed, counts and area within TrimmedCountFraction and
// the density values' range and mean close (the fixture stores only those for trimmed cases).
// The file wrapper against upstream's own code built at -O1, Tier A, is
// PoissonMeshingOracleTests.Exact.cs.

using System.Text;
using System.Text.Json;

using ColmapSharp.Mvs;
using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PoissonMeshingOracleTests
{
	private const string Fixture = "poisson_meshing.json";

	// Model units on a unit sphere; the wheel's positions differ by a few float ulps
	// (3.6e-7 at most when written).
	private const double PositionTolerance = 1e-6;
	private const double ValueTolerance = 1e-4;

	// Near-tie polygons retriangulated the other way (about 1% of the triangles when written).
	private const double RetriangulatedFraction = 0.03;

	// Vertices and triangles that cross the trim value (about 0.2% when written).
	private const double TrimmedCountFraction = 0.01;

	[Test]
	[Arguments("depth5")]
	[Arguments("depth5trim")]
	[Arguments("depth4unscreened")]
	[Arguments("depth6trim")]
	public async Task PoissonMeshing_MatchesPycolmap(string name)
	{
		JsonElement root = OracleFixture.Load(Fixture);
		JsonElement expected = root.GetProperty("cases").GetProperty(name);
		var options = new PoissonMeshingOptions
		{
			Depth = expected.GetProperty("depth").GetInt32(),
			PointWeight = expected.GetProperty("point_weight").GetDouble(),
			Trim = expected.GetProperty("trim").GetDouble(),
			NumThreads = 1,
		};

		PoissonMeshOutput mesh = PoissonMeshing.Run(options, Floats(root, "positions"), Floats(root, "normals"), Bytes(root, "colors"));

		using var ply = new MemoryStream();
		PoissonMeshing.WriteMeshPly(ply, mesh);
		string header = expected.GetProperty("header").GetString()!;
		string actualHeader = Encoding.ASCII.GetString(ply.ToArray(), 0, Math.Min(header.Length, (int)ply.Length));
		await Assert.That(WithoutCounts(actualHeader)).IsEqualTo(WithoutCounts(header));

		int expectedVertices = expected.GetProperty("vertexcount").GetInt32();
		int expectedTriangles = expected.GetProperty("trianglecount").GetInt32();
		double area = Area(mesh.Positions, mesh.Triangles);
		if (options.Trim == 0)
		{
			int[] expectedTriangleIndices = expected.GetProperty("triangles").EnumerateArray().Select(e => e.GetInt32()).ToArray();
			float[] expectedPositions = Floats(expected, "positions");
			double expectedArea = Area(expectedPositions, expectedTriangleIndices);

			// Untrimmed: the same vertices in the same order, a few ulps apart; the only other
			// difference is which diagonal MinimalAreaTriangulation picks for a near-tie polygon.
			await Assert.That(mesh.VertexCount).IsEqualTo(expectedVertices);
			await Assert.That(mesh.Triangles.Length / 3).IsEqualTo(expectedTriangles);
			await Assert.That(MaxAbsDiff(mesh.Positions, expectedPositions)).IsLessThanOrEqualTo(PositionTolerance);
			await Assert.That(mesh.Values).IsNull();
			byte[] colors = Bytes(expected, "colors");
			await Assert.That(MaxAbsDiff(mesh.Colors!.Select(c => (float)c).ToArray(), colors.Select(c => (float)c).ToArray())).IsLessThanOrEqualTo(1.0);
			await Assert.That(DifferentTriangleFraction(mesh.Triangles, expectedTriangleIndices)).IsLessThanOrEqualTo(RetriangulatedFraction);
			await Assert.That(Math.Abs(area - expectedArea) / expectedArea).IsLessThanOrEqualTo(1e-4);
		}
		else
		{
			// Trimmed: a vertex whose density sits within those ulps of the trim value can land on
			// the other side, so a few vertices and triangles come or go and the indices shift.
			// The fixture's area is Area over the wheel's mesh, summed in double the same way.
			double expectedArea = expected.GetProperty("area").GetDouble();
			await Assert.That(Math.Abs(mesh.VertexCount - expectedVertices)).IsLessThanOrEqualTo((int)(TrimmedCountFraction * expectedVertices));
			await Assert.That(Math.Abs((mesh.Triangles.Length / 3) - expectedTriangles)).IsLessThanOrEqualTo((int)(TrimmedCountFraction * expectedTriangles));
			await Assert.That(Math.Abs(area - expectedArea) / expectedArea).IsLessThanOrEqualTo(TrimmedCountFraction);
			// Merged islands keep some below-trim vertices, so compare the range and mean, not a bound.
			float[] values = mesh.Values!;
			await Assert.That((double)Math.Abs(values.Min() - Float(expected, "value_min"))).IsLessThanOrEqualTo(ValueTolerance);
			await Assert.That((double)Math.Abs(values.Max() - Float(expected, "value_max"))).IsLessThanOrEqualTo(ValueTolerance);
			await Assert.That(Math.Abs(values.Average() - expected.GetProperty("value_mean").GetDouble())).IsLessThanOrEqualTo(1e-3);
			await Assert.That(mesh.Colors!.Length).IsEqualTo(3 * mesh.VertexCount);
		}
	}

	// "element vertex 123" -> "element vertex": the property layout must match exactly.
	private static string WithoutCounts(string header) =>
		string.Join('\n', header.Split('\n').Select(line => line.StartsWith("element ", StringComparison.Ordinal) ? line[..line.LastIndexOf(' ')] : line));

	private static double Area(float[] positions, int[] triangles)
	{
		double area = 0;
		for (int t = 0; t < triangles.Length; t += 3)
		{
			int a = 3 * triangles[t], b = 3 * triangles[t + 1], c = 3 * triangles[t + 2];
			double ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
			double vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
			double cx = (uy * vz) - (uz * vy), cy = (uz * vx) - (ux * vz), cz = (ux * vy) - (uy * vx);
			area += 0.5 * Math.Sqrt((cx * cx) + (cy * cy) + (cz * cz));
		}

		return area;
	}

	// The fraction of triangles (as vertex sets) in one list but not the other.
	private static double DifferentTriangleFraction(int[] actual, int[] expected)
	{
		static HashSet<(int, int, int)> Set(int[] t)
		{
			var set = new HashSet<(int, int, int)>();
			for (int i = 0; i < t.Length; i += 3)
			{
				int[] v = [t[i], t[i + 1], t[i + 2]];
				Array.Sort(v);
				set.Add((v[0], v[1], v[2]));
			}

			return set;
		}

		HashSet<(int, int, int)> a = Set(actual);
		a.SymmetricExceptWith(Set(expected));
		return a.Count / (double)Math.Max(1, (actual.Length + expected.Length) / 3);
	}

	private static double MaxAbsDiff(float[] actual, float[] expected)
	{
		if (actual.Length != expected.Length)
		{
			return double.PositiveInfinity;
		}

		double max = 0;
		for (int i = 0; i < actual.Length; i++)
		{
			max = Math.Max(max, Math.Abs((double)actual[i] - expected[i]));
		}

		return max;
	}

	// The fixture writes floats in their shortest float32 form, so they parse exactly as float.
	private static float[] Floats(JsonElement element, string name) =>
		element.GetProperty(name).EnumerateArray().Select(ParseFloat).ToArray();

	private static float Float(JsonElement element, string name) => ParseFloat(element.GetProperty(name));

	private static float ParseFloat(JsonElement e) => float.Parse(e.GetRawText(), System.Globalization.CultureInfo.InvariantCulture);

	private static byte[] Bytes(JsonElement element, string name) =>
		element.GetProperty(name).EnumerateArray().Select(e => e.GetByte()).ToArray();
}
