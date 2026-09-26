// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionManagerTests: colmap/scene/reconstruction_manager_test.cc ported 1:1, one
// method per gtest case named Suite_Name. Tests ColmapSharp/Scene/ReconstructionManager.cs.
// CSharpOnly_WriteOrdersByPointCountThenIndex pins Write/Read (reconstruction_manager_test.cc
// has no case for them) and the index tie-break of docs/CPP_DIVERGENCES.md, entry 34.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionManagerTests
{
	[Test]
	public async Task ReconstructionManager_Empty()
	{
		var reconstructionManager = new ReconstructionManager();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	[Test]
	public async Task ReconstructionManager_AddGet()
	{
		var reconstructionManager = new ReconstructionManager();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
		for (int i = 0; i < 10; ++i)
		{
			int idx = reconstructionManager.Add();
			await Assert.That(reconstructionManager.Size).IsEqualTo(i + 1);
			await Assert.That(idx).IsEqualTo(i);
			await Assert.That(reconstructionManager.Get(idx).NumCameras).IsEqualTo(0);
			await Assert.That(reconstructionManager.Get(idx).NumImages).IsEqualTo(0);
			await Assert.That(reconstructionManager.Get(idx).NumPoints3D).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ReconstructionManager_Delete()
	{
		var reconstructionManager = new ReconstructionManager();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
		for (int i = 0; i < 10; ++i)
		{
			reconstructionManager.Add();
		}

		await Assert.That(reconstructionManager.Size).IsEqualTo(10);
		for (int i = 0; i < 10; ++i)
		{
			reconstructionManager.Delete(0);
			await Assert.That(reconstructionManager.Size).IsEqualTo(9 - i);
		}
	}

	[Test]
	public async Task ReconstructionManager_Clear()
	{
		var reconstructionManager = new ReconstructionManager();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
		for (int i = 0; i < 10; ++i)
		{
			reconstructionManager.Add();
		}

		await Assert.That(reconstructionManager.Size).IsEqualTo(10);
		reconstructionManager.Clear();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_WriteOrdersByPointCountThenIndex()
	{
		// Point counts 1, 2, 1, 2 by index: written as 1, 3, 0, 2.
		var manager = new ReconstructionManager();
		int[] counts = [1, 2, 1, 2];
		for (int i = 0; i < counts.Length; i++)
		{
			Reconstruction reconstruction = manager.Get(manager.Add());
			for (int j = 0; j < counts[i]; j++)
			{
				// x encodes the source index so the written directory can be traced back.
				reconstruction.AddPoint3D(new Vector3d(i, j, 0), new Track());
			}
		}

		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-manager-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		manager.Write(dir);

		int[] expectedSource = [1, 3, 0, 2];
		var readBack = new ReconstructionManager();
		for (int i = 0; i < expectedSource.Length; i++)
		{
			int idx = readBack.Read(Path.Combine(dir, i.ToString(System.Globalization.CultureInfo.InvariantCulture)));
			await Assert.That(idx).IsEqualTo(i);
			Reconstruction written = readBack.Get(idx);
			await Assert.That(written.NumPoints3D).IsEqualTo(counts[expectedSource[i]]);
			await Assert.That(written.Points3D.Values.First().Xyz.X).IsEqualTo((double)expectedSource[i]);
		}
	}
}
