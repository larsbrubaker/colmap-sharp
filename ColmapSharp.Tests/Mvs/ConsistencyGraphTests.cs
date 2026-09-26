// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ConsistencyGraphTests: colmap/mvs/consistency_graph_test.cc ported 1:1 (Suite_Name),
// testing ColmapSharp/Mvs/ConsistencyGraph.cs. Tier A (exact). GetImageIdxs returns a span
// here: `num_images == 0 && image_idxs == nullptr` becomes an empty span. The C#-only case
// at the end pins the file bytes.

using System.Text;

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class ConsistencyGraphTests
{
	[Test]
	public async Task ConsistencyGraph_Empty()
	{
		int[] data = [];
		var consistencyGraph = new ConsistencyGraph(2, 2, data);
		for (int i = 0; i < 2; ++i)
		{
			for (int j = 0; j < 2; ++j)
			{
				int[] imageIdxs = consistencyGraph.GetImageIdxs(0, 0).ToArray();
				await Assert.That(imageIdxs.Length).IsEqualTo(0);
			}
		}

		await Assert.That(consistencyGraph.GetNumBytes()).IsEqualTo(16);
	}

	[Test]
	public async Task ConsistencyGraph_Partial()
	{
		int[] data = [0, 0, 3, 5, 7, 33];
		var consistencyGraph = new ConsistencyGraph(2, 1, data);
		int[] imageIdxs = consistencyGraph.GetImageIdxs(0, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(3);
		await Assert.That(imageIdxs[0]).IsEqualTo(5);
		await Assert.That(imageIdxs[1]).IsEqualTo(7);
		await Assert.That(imageIdxs[2]).IsEqualTo(33);
		imageIdxs = consistencyGraph.GetImageIdxs(0, 1).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(0);
		await Assert.That(consistencyGraph.GetNumBytes()).IsEqualTo(32);
	}

	[Test]
	public async Task ConsistencyGraph_Zero()
	{
		int[] data = [0, 0, 0];
		var consistencyGraph = new ConsistencyGraph(2, 1, data);
		int[] imageIdxs = consistencyGraph.GetImageIdxs(0, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(0);
		imageIdxs = consistencyGraph.GetImageIdxs(0, 1).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(0);
		await Assert.That(consistencyGraph.GetNumBytes()).IsEqualTo(20);
	}

	[Test]
	public async Task ConsistencyGraph_Full()
	{
		int[] data = [0, 0, 3, 5, 7, 33, 0, 1, 1, 100];
		var consistencyGraph = new ConsistencyGraph(1, 2, data);
		int[] imageIdxs = consistencyGraph.GetImageIdxs(0, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(3);
		await Assert.That(imageIdxs[0]).IsEqualTo(5);
		await Assert.That(imageIdxs[1]).IsEqualTo(7);
		await Assert.That(imageIdxs[2]).IsEqualTo(33);
		imageIdxs = consistencyGraph.GetImageIdxs(1, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(1);
		await Assert.That(imageIdxs[0]).IsEqualTo(100);
		await Assert.That(consistencyGraph.GetNumBytes()).IsEqualTo(48);
	}

	[Test]
	public async Task ConsistencyGraph_DefaultConstructor()
	{
		var consistencyGraph = new ConsistencyGraph();
		await Assert.That(consistencyGraph.GetNumBytes()).IsEqualTo(0);
	}

	[Test]
	public async Task ConsistencyGraph_WriteReadRoundtrip()
	{
		int[] data = [0, 0, 3, 5, 7, 33, 0, 1, 1, 100];
		var original = new ConsistencyGraph(1, 2, data);

		string testDir = MvsTestUtils.CreateTestDir();
		string path = Path.Combine(testDir, "consistency_graph.bin");
		original.Write(path);

		var loaded = new ConsistencyGraph();
		loaded.Read(path);

		await Assert.That(loaded.GetNumBytes()).IsEqualTo(original.GetNumBytes());

		// Verify data is preserved
		int[] imageIdxs = loaded.GetImageIdxs(0, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(3);
		await Assert.That(imageIdxs[0]).IsEqualTo(5);
		await Assert.That(imageIdxs[1]).IsEqualTo(7);
		await Assert.That(imageIdxs[2]).IsEqualTo(33);
		imageIdxs = loaded.GetImageIdxs(1, 0).ToArray();
		await Assert.That(imageIdxs.Length).IsEqualTo(1);
		await Assert.That(imageIdxs[0]).IsEqualTo(100);
	}

	// C#-only: the exact bytes ConsistencyGraph::Write produces ("cols&rows&1&" + ints).
	[Test]
	public async Task ConsistencyGraph_WriteByteExact()
	{
		int[] data = [0, 1, 1, -7];
		var graph = new ConsistencyGraph(1, 2, data);
		string path = Path.Combine(MvsTestUtils.CreateTestDir(), "graph.bin");
		graph.Write(path);

		var expected = new List<byte>(Encoding.ASCII.GetBytes("1&2&1&"));
		foreach (int value in data)
		{
			expected.AddRange(BitConverter.GetBytes(value));
		}

		await Assert.That(File.ReadAllBytes(path)).IsEquivalentTo(expected.ToArray(), CollectionOrdering.Matching);
	}
}
