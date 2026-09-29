// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullMarchingCubesTests: C#-only tests (not ports; COLMAP has no marching cubes) of
// ColmapSharp/Mvs/Silhouette/HullMarchingCubes.cs on small random grids, where every face
// configuration, including the ones the asymptotic decider resolves, turns up many times:
// - every directed edge appears exactly once and its reverse exists (closed, consistently wound,
//   no triangle doubled back on a neighbor's), and the enclosed volume is positive;
// - values exactly at the iso level (quantized grids, as the [1 2 1]/4 smoothing produces) give
//   no zero-area triangles.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.Silhouette;

public class HullMarchingCubesTests
{
	private enum ValueKind
	{
		Continuous,
		Binary,
		Quantized,
	}

	private static OccupancyGrid RandomGrid(Mt19937 random, ValueKind kind)
	{
		int nx = 1 + (int)(random.Next() % 3), ny = 1 + (int)(random.Next() % 3), nz = 1 + (int)(random.Next() % 3);
		var grid = new OccupancyGrid(Vector3d.Zero, 1, nx, ny, nz);
		for (int i = 0; i < grid.Values.Length; i++)
		{
			uint r = random.Next();
			grid.Values[i] = kind switch
			{
				ValueKind.Continuous => (float)(r / 4294967296.0),
				ValueKind.Binary => r % 2,
				_ => r % 5 * 0.25f,
			};
		}

		return grid;
	}

	private static bool HasZeroAreaTriangle(PlyMesh mesh)
	{
		foreach (PlyMeshFace f in mesh.Faces)
		{
			Vector3d a = VisualHullTestRig.Position(mesh, f.VertexIdx1);
			Vector3d b = VisualHullTestRig.Position(mesh, f.VertexIdx2);
			Vector3d c = VisualHullTestRig.Position(mesh, f.VertexIdx3);
			if ((b - a).Cross(c - a).Norm == 0)
			{
				return true;
			}
		}

		return false;
	}

	[Test]
	public async Task CSharpOnly_DecidedFaceDoesNotDoubleATriangle()
	{
		// The reviewer's minimal repro: two inside corners on the diagonal of the z = 0 face.
		var grid = new OccupancyGrid(Vector3d.Zero, 1, 2, 2, 2);
		float[] values = [1, 0, 0, 1, 0, 0, 0, 0];
		values.CopyTo(grid.Values, 0);
		PlyMesh mesh = HullMarchingCubes.Extract(grid, 0.5);
		(bool closed, bool consistent, double volume) = VisualHullTestRig.Topology(mesh);
		await Assert.That(closed).IsTrue();
		await Assert.That(consistent).IsTrue();
		await Assert.That(volume).IsGreaterThan(0);
	}

	[Test]
	[Arguments(0)]
	[Arguments(1)]
	[Arguments(2)]
	public async Task CSharpOnly_RandomGridsGiveClosedConsistentMeshes(int kindIndex)
	{
		var kind = (ValueKind)kindIndex;
		var random = new Mt19937(17u + (uint)kindIndex);
		int failures = 0, degenerate = 0, meshes = 0;
		for (int trial = 0; trial < 400; trial++)
		{
			PlyMesh mesh = HullMarchingCubes.Extract(RandomGrid(random, kind), 0.5);
			if (mesh.Faces.Count == 0)
			{
				continue;
			}

			meshes++;
			(bool closed, bool consistent, double volume) = VisualHullTestRig.Topology(mesh);
			if (!closed || !consistent || !(volume > 0))
			{
				failures++;
			}

			if (HasZeroAreaTriangle(mesh))
			{
				degenerate++;
			}
		}

		await Assert.That(meshes).IsGreaterThan(300);
		await Assert.That(failures).IsEqualTo(0);
		await Assert.That(degenerate).IsEqualTo(0);
	}
}
