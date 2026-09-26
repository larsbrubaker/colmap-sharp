// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ConnectedComponentsTests: colmap/math/connected_components_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/ConnectedComponents.cs.
// Tier A for the component sets; gmock's UnorderedElementsAre becomes a comparison of the
// sorted members.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class ConnectedComponentsTests
{
	[Test]
	public async Task FindConnectedComponents_Empty()
	{
		var nodes = new HashSet<int>();
		var edges = new List<(int, int)>();
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(0);
	}

	[Test]
	public async Task FindConnectedComponents_SingleNode()
	{
		var nodes = new HashSet<int> { 1 };
		var edges = new List<(int, int)>();
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(1);
		using (Assert.Multiple())
		{
			await Assert.That(components[0].Count).IsEqualTo(1);
			await Assert.That(components[0][0]).IsEqualTo(1);
		}
	}

	[Test]
	public async Task FindConnectedComponents_TwoConnectedNodes()
	{
		var nodes = new HashSet<int> { 1, 2 };
		var edges = new List<(int, int)> { (1, 2) };
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(1);
		await Assert.That(components[0].Count).IsEqualTo(2);
	}

	[Test]
	public async Task FindConnectedComponents_TwoDisconnectedNodes()
	{
		var nodes = new HashSet<int> { 1, 2 };
		var edges = new List<(int, int)>();
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(2);
		using (Assert.Multiple())
		{
			await Assert.That(components[0].Count).IsEqualTo(1);
			await Assert.That(components[1].Count).IsEqualTo(1);
		}
	}

	[Test]
	public async Task FindConnectedComponents_ThreeComponents()
	{
		var nodes = new HashSet<int> { 1, 2, 3, 4, 5, 6 };
		var edges = new List<(int, int)> { (1, 2), (3, 4), (5, 6) };
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(3);
		using (Assert.Multiple())
		{
			foreach (var comp in components)
			{
				await Assert.That(comp.Count).IsEqualTo(2);
			}
		}
	}

	[Test]
	public async Task FindConnectedComponents_Chain()
	{
		var nodes = new HashSet<int> { 1, 2, 3, 4, 5 };
		var edges = new List<(int, int)> { (1, 2), (2, 3), (3, 4), (4, 5) };
		var components = ConnectedComponents.FindConnectedComponents(nodes, edges);
		await Assert.That(components.Count).IsEqualTo(1);
		await Assert.That(components[0].Count).IsEqualTo(5);
	}

	[Test]
	public async Task FindLargestConnectedComponent_Empty()
	{
		var nodes = new HashSet<int>();
		var edges = new List<(int, int)>();
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(largest.Count).IsEqualTo(0);
	}

	[Test]
	public async Task FindLargestConnectedComponent_SingleNode()
	{
		var nodes = new HashSet<int> { 42 };
		var edges = new List<(int, int)>();
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(Sorted(largest)).IsEquivalentTo(new[] { 42 });
	}

	[Test]
	public async Task FindLargestConnectedComponent_AllConnected()
	{
		var nodes = new HashSet<int> { 1, 2, 3, 4 };
		var edges = new List<(int, int)> { (1, 2), (2, 3), (3, 4) };
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(Sorted(largest)).IsEquivalentTo(new[] { 1, 2, 3, 4 });
	}

	[Test]
	public async Task FindLargestConnectedComponent_TwoComponentsDifferentSizes()
	{
		// Component 1: {1, 2, 3} (size 3)
		// Component 2: {10, 20} (size 2)
		var nodes = new HashSet<int> { 1, 2, 3, 10, 20 };
		var edges = new List<(int, int)> { (1, 2), (2, 3), (10, 20) };
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(Sorted(largest)).IsEquivalentTo(new[] { 1, 2, 3 });
	}

	[Test]
	public async Task FindLargestConnectedComponent_ManySmallOnelarger()
	{
		// 5 isolated nodes + 1 component of 3
		var nodes = new HashSet<int> { 1, 2, 3, 4, 5, 100, 200, 300 };
		var edges = new List<(int, int)> { (100, 200), (200, 300) };
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(Sorted(largest)).IsEquivalentTo(new[] { 100, 200, 300 });
	}

	[Test]
	public async Task FindLargestConnectedComponent_StringType()
	{
		var nodes = new HashSet<string> { "a", "b", "c", "x", "y" };
		var edges = new List<(string, string)> { ("a", "b"), ("b", "c") };
		var largest = ConnectedComponents.FindLargestConnectedComponent(nodes, edges);
		await Assert.That(Sorted(largest)).IsEquivalentTo(new[] { "a", "b", "c" });
	}

	// UnorderedElementsAre: compare as sorted sequences so the count and multiplicity count too.
	private static List<T> Sorted<T>(List<T> values)
	{
		return [.. values.OrderBy(v => v, Comparer<T>.Default)];
	}
}
