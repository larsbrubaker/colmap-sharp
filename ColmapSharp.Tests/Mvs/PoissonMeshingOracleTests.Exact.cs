// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonMeshingOracleTests.Exact (C#-only, not a COLMAP test): PoissonMeshing's file wrapper
// against upstream's own RunPoissonRecon and RunSurfaceTrimmer, called with COLMAP's exact
// arguments by oracle/poisson_meshing_harness.cc, built with clang++ -O1 -ffp-contract=off
// -DRELEASE and run with one thread (oracle/fixture_poisson_meshing.py writes
// TestData/oracle/poisson_meshing_exact.json). Tier A: the output PLY must be byte-identical,
// header included. The one known exception is entry 116 of docs/CPP_DIVERGENCES.md
// (PoissonSplat.LogF is the double logarithm rounded to float, Apple's logf rounds a rare
// argument the other way): in depth6trim one output vertex's density "value" is one float ulp
// from upstream's. The test pins that exception precisely - that vertex, that property, one
// ulp exactly - and requires every other byte to match. The "empty" case is a PLY with red,
// green and blue but no points, whose output still declares those properties.

using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PoissonMeshingOracleTests
{
	private const string ExactFixture = "poisson_meshing_exact.json";

	[Test]
	[Arguments("depth5", -1)]
	[Arguments("depth5trim", -1)]
	[Arguments("depth4unscreened", -1)]
	[Arguments("depth6trim", KnownLogFVertex)]
	[Arguments("empty", -1)]
	public async Task PoissonMeshing_FileMatchesUpstreamExactly(string name, int logFVertex)
	{
		JsonElement root = OracleFixture.Load(ExactFixture);
		JsonElement expected = root.GetProperty("cases").GetProperty(name);
		byte[] input = Convert.FromBase64String(root.GetProperty("inputs").GetProperty(expected.GetProperty("input").GetString()!).GetString()!);
		byte[] expectedPly = Convert.FromBase64String(expected.GetProperty("ply").GetString()!);
		var options = new PoissonMeshingOptions
		{
			Depth = expected.GetProperty("depth").GetInt32(),
			PointWeight = expected.GetProperty("point_weight").GetDouble(),
			Trim = expected.GetProperty("trim").GetDouble(),
			NumThreads = 1,
		};

		string testDir = MvsTestUtils.CreateTestDir();
		string inputPath = Path.Combine(testDir, "points.ply"), outputPath = Path.Combine(testDir, "mesh.ply");
		File.WriteAllBytes(inputPath, input);
		await Assert.That(PoissonMeshing.Run(options, inputPath, outputPath)).IsTrue();
		byte[] actualPly = File.ReadAllBytes(outputPath);

		await Assert.That(actualPly.Length).IsEqualTo(expectedPly.Length);
		int headerLength = HeaderLength(expectedPly);
		await Assert.That(Encoding.ASCII.GetString(actualPly, 0, headerLength)).IsEqualTo(Encoding.ASCII.GetString(expectedPly, 0, headerLength));

		// Every differing byte must fall in the known vertex's density value.
		(int vertexCount, int recordSize, int valueOffset) = VertexLayout(Encoding.ASCII.GetString(expectedPly, 0, headerLength));
		var differences = new List<string>();
		for (int i = 0; i < expectedPly.Length; i++)
		{
			if (actualPly[i] == expectedPly[i])
			{
				continue;
			}

			int offset = i - headerLength;
			int vertex = offset / recordSize;
			bool inKnownValue = offset >= 0 && vertex < vertexCount && vertex == logFVertex && valueOffset >= 0
				&& offset % recordSize >= valueOffset && offset % recordSize < valueOffset + 4;
			if (!inKnownValue)
			{
				differences.Add($"byte {i} (vertex record {vertex}, offset {offset % recordSize}): {actualPly[i]} vs {expectedPly[i]}");
			}
		}

		await Assert.That(string.Join("; ", differences.Take(10))).IsEqualTo(string.Empty);
		if (logFVertex >= 0)
		{
			// Entry 116: the one value within one float ulp, and actually different, so the
			// exception is removed if LogF ever matches Apple's logf here.
			int at = headerLength + (logFVertex * recordSize) + valueOffset;
			int actualBits = BinaryPrimitives.ReadInt32LittleEndian(actualPly.AsSpan(at));
			int expectedBits = BinaryPrimitives.ReadInt32LittleEndian(expectedPly.AsSpan(at));
			await Assert.That(Math.Abs(actualBits - expectedBits)).IsEqualTo(1);
		}
	}

	// depth6trim's output vertex whose density value is one ulp off (docs/CPP_DIVERGENCES.md,
	// entry 116).
	private const int KnownLogFVertex = 898;

	private static int HeaderLength(byte[] ply)
	{
		byte[] end = Encoding.ASCII.GetBytes("end_header\n");
		return ply.AsSpan().IndexOf(end) + end.Length;
	}

	// The vertex count, the bytes per vertex record and the offset of "value" in it (-1 when absent).
	private static (int Count, int RecordSize, int ValueOffset) VertexLayout(string header)
	{
		int count = 0, size = 0, valueOffset = -1;
		bool inVertex = false;
		foreach (string line in header.Split('\n'))
		{
			string[] tokens = line.Split(' ');
			if (tokens[0] == "element")
			{
				inVertex = tokens[1] == "vertex";
				if (inVertex)
				{
					count = int.Parse(tokens[2], System.Globalization.CultureInfo.InvariantCulture);
				}
			}
			else if (inVertex && tokens[0] == "property")
			{
				if (tokens[2] == "value")
				{
					valueOffset = size;
				}

				size += tokens[1] == "uchar" ? 1 : 4;
			}
		}

		return (count, size, valueOffset);
	}
}
