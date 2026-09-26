// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionManagerTests: colmap/scene/reconstruction_manager_test.cc ported 1:1, one
// method per gtest case named Suite_Name. Tests ColmapSharp/Scene/ReconstructionManager.cs.

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
}
