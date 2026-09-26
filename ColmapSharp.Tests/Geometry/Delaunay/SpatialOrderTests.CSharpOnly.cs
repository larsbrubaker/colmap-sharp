// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SpatialOrderTests (C#-only): ColmapSharp/Geometry/Delaunay/SpatialOrder.cs. The Hilbert key
// must be a Hilbert curve (a bijection on the grid whose consecutive cells share a face), and
// the BRIO order must be a deterministic permutation. Exact (integer) properties.

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry.Delaunay;

public class SpatialOrderTests
{
	[Test]
	public async Task CSharpOnly_HilbertKeyVisitsNeighboringCellsInOrder()
	{
		// The 8x8x8 corner block is the first 512 cells of the curve, since it starts at the
		// origin; walking them by key must step to a face neighbor every time.
		const int size = 8;
		var cellsByKey = new (int X, int Y, int Z)[size * size * size];
		var seen = new bool[cellsByKey.Length];
		int outOfRange = 0, duplicates = 0;
		for (int x = 0; x < size; ++x)
		{
			for (int y = 0; y < size; ++y)
			{
				for (int z = 0; z < size; ++z)
				{
					ulong key = SpatialOrder.HilbertKey((uint)x, (uint)y, (uint)z);
					if (key >= (ulong)cellsByKey.Length)
					{
						++outOfRange;
						continue;
					}

					if (seen[key])
					{
						++duplicates;
					}

					seen[key] = true;
					cellsByKey[key] = (x, y, z);
				}
			}
		}

		int jumps = 0;
		for (int k = 1; k < cellsByKey.Length; ++k)
		{
			var a = cellsByKey[k - 1];
			var b = cellsByKey[k];
			if (Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y) + Math.Abs(a.Z - b.Z) != 1)
			{
				++jumps;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(outOfRange).IsEqualTo(0);
			await Assert.That(duplicates).IsEqualTo(0);
			await Assert.That(jumps).IsEqualTo(0);
			await Assert.That(cellsByKey[0]).IsEqualTo((0, 0, 0));
		}
	}

	[Test]
	public async Task CSharpOnly_BrioOrderIsADeterministicPermutation()
	{
		var random = new Random(3);
		var points = new Vector3d[1000];
		for (int i = 0; i < points.Length; ++i)
		{
			points[i] = new Vector3d(random.NextDouble(), random.NextDouble(), random.NextDouble());
		}

		var first = SpatialOrder.BrioHilbertOrder(points);
		var second = SpatialOrder.BrioHilbertOrder(points);
		using (Assert.Multiple())
		{
			await Assert.That(first.Order()).IsEquivalentTo(Enumerable.Range(0, points.Length));
			await Assert.That(second).IsEquivalentTo(first, TUnit.Assertions.Enums.CollectionOrdering.Matching);
		}
	}
}
