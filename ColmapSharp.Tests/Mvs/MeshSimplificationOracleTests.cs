// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MeshSimplificationOracleTests: C#-only oracle checks of ColmapSharp/Mvs/MeshSimplification.cs
// against COLMAP 4.2.0 (TestData/oracle/mesh_simplification.json, written by
// oracle/fixture_mesh_simplification.py, which says where each case's numbers come from).
// - wavy: a curved grid through pycolmap. Tier C: the same faces, positions within 1e-5
//   (boundary systems can differ in the last bits, divergence 72; they
//   were byte-identical when generated).
// - flat: flat grids with boundary_weight 0 through pycolmap, where every cost ties at 0 and
//   the result is decided by the ported libc++ heap order (CollapseHeap.cs). Exact.
// - color: the curved grid with colors, from oracle/mesh_simplification_harness.cc (pycolmap
//   writes no colors; the harness's geometry was checked equal to pycolmap's). Exact.

using System.Text.Json;

using ColmapSharp.Mvs;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class MeshSimplificationOracleTests
{
	private const string Fixture = "mesh_simplification.json";

	public static IEnumerable<int> CaseIndices()
	{
		return Enumerable.Range(0, OracleFixture.Load(Fixture).GetProperty("cases").GetArrayLength());
	}

	[Test]
	[MethodDataSource(nameof(CaseIndices))]
	public async Task CSharpOnly_MatchesColmap(int caseIndex)
	{
		JsonElement root = OracleFixture.Load(Fixture);
		JsonElement expected = root.GetProperty("cases")[caseIndex];
		string input = expected.GetProperty("input").GetString()!;
		PlyMesh mesh = BuildInput(root, input);
		var options = new MeshSimplificationOptions
		{
			TargetFaceRatio = expected.GetProperty("target_face_ratio").GetDouble(),
			BoundaryWeight = expected.GetProperty("boundary_weight").GetDouble(),
			MaxError = expected.GetProperty("max_error").GetDouble(),
			InterpolateColors = expected.GetProperty("interpolate_colors").GetBoolean(),
		};
		double[] expectedVertices = OracleFixture.Doubles(expected.GetProperty("vertices"));
		long[] expectedFaces = OracleFixture.Int64s(expected.GetProperty("faces"));

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Vertices.Count * 3).IsEqualTo(expectedVertices.Length);
		await Assert.That(result.Faces.Count * 3).IsEqualTo(expectedFaces.Length);
		for (int i = 0; i < result.Faces.Count; i++)
		{
			PlyMeshFace face = result.Faces[i];
			await Assert.That(face.VertexIdx1).IsEqualTo((int)expectedFaces[i * 3]);
			await Assert.That(face.VertexIdx2).IsEqualTo((int)expectedFaces[i * 3 + 1]);
			await Assert.That(face.VertexIdx3).IsEqualTo((int)expectedFaces[i * 3 + 2]);
		}

		// The fixture's positions are float values written as doubles, so exact is ==.
		double tolerance = input == "wavy" ? 1e-5 : 0.0;
		double maxDiff = 0;
		for (int i = 0; i < result.Vertices.Count; i++)
		{
			PlyMeshVertex v = result.Vertices[i];
			maxDiff = Math.Max(maxDiff, Math.Abs(v.X - expectedVertices[i * 3]));
			maxDiff = Math.Max(maxDiff, Math.Abs(v.Y - expectedVertices[i * 3 + 1]));
			maxDiff = Math.Max(maxDiff, Math.Abs(v.Z - expectedVertices[i * 3 + 2]));
		}

		await Assert.That(maxDiff).IsLessThanOrEqualTo(tolerance);

		if (expected.TryGetProperty("colors", out JsonElement colorsElement))
		{
			long[] expectedColors = OracleFixture.Int64s(colorsElement);
			for (int i = 0; i < result.Vertices.Count; i++)
			{
				PlyMeshVertex v = result.Vertices[i];
				await Assert.That((long)v.R).IsEqualTo(expectedColors[i * 3]);
				await Assert.That((long)v.G).IsEqualTo(expectedColors[i * 3 + 1]);
				await Assert.That((long)v.B).IsEqualTo(expectedColors[i * 3 + 2]);
			}
		}
	}

	private static PlyMesh BuildInput(JsonElement root, string input)
	{
		if (input.StartsWith("flat", StringComparison.Ordinal))
		{
			int n = int.Parse(input["flat".Length..], System.Globalization.CultureInfo.InvariantCulture);
			return MeshSimplificationTests.CreateGridMesh(n);
		}

		double[] vertices = OracleFixture.Doubles(root.GetProperty("wavy_vertices"));
		long[] colors = OracleFixture.Int64s(root.GetProperty("wavy_colors"));
		long[] faces = OracleFixture.Int64s(root.GetProperty("wavy_faces"));
		bool colored = input == "wavy_colored";
		var mesh = new PlyMesh();
		for (int i = 0; i < vertices.Length / 3; i++)
		{
			float x = (float)vertices[i * 3], y = (float)vertices[i * 3 + 1], z = (float)vertices[i * 3 + 2];
			mesh.Vertices.Add(colored
				? new PlyMeshVertex(x, y, z, (byte)colors[i * 3], (byte)colors[i * 3 + 1], (byte)colors[i * 3 + 2])
				: new PlyMeshVertex(x, y, z));
		}

		for (int i = 0; i < faces.Length; i += 3)
		{
			mesh.Faces.Add(new PlyMeshFace((int)faces[i], (int)faces[i + 1], (int)faces[i + 2]));
		}

		return mesh;
	}
}
