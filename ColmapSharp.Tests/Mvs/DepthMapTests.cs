// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DepthMapTests: colmap/mvs/depth_map_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/DepthMap.cs. Tier A (exact), as in C++.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class DepthMapTests
{
	[Test]
	public async Task DepthMap_Empty()
	{
		var depthMap = new DepthMap();
		await Assert.That(depthMap.GetWidth()).IsEqualTo(0);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(0);
		await Assert.That(depthMap.GetDepth()).IsEqualTo(1);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(-1);
		await Assert.That(depthMap.GetDepthMax()).IsEqualTo(-1);
	}

	[Test]
	public async Task DepthMap_NonEmpty()
	{
		var depthMap = new DepthMap(1, 2, 0, 1);
		await Assert.That(depthMap.GetWidth()).IsEqualTo(1);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(2);
		await Assert.That(depthMap.GetDepth()).IsEqualTo(1);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(0);
		await Assert.That(depthMap.GetDepthMax()).IsEqualTo(1);
	}

	[Test]
	public async Task DepthMap_Rescale()
	{
		var depthMap = new DepthMap(6, 7, 0, 1);
		depthMap.Rescale(0.5f);
		await Assert.That(depthMap.GetWidth()).IsEqualTo(3);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(4);
		await Assert.That(depthMap.GetDepth()).IsEqualTo(1);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(0);
		await Assert.That(depthMap.GetDepthMax()).IsEqualTo(1);
	}

	[Test]
	public async Task DepthMap_Downsize()
	{
		var depthMap = new DepthMap(6, 7, 0, 1);
		depthMap.Downsize(2, 4);
		await Assert.That(depthMap.GetWidth()).IsEqualTo(2);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(2);
		await Assert.That(depthMap.GetDepth()).IsEqualTo(1);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(0);
		await Assert.That(depthMap.GetDepthMax()).IsEqualTo(1);
	}

	[Test]
	public async Task DepthMap_ToBitmap()
	{
		var depthMap = new DepthMap(2, 2, 0.1f, 0.9f);
		depthMap.Fill(0.9f);
		depthMap.Set(0, 0, 0, 0.1f);
		depthMap.Set(0, 1, 0, 0.5f);
		Bitmap bitmap = depthMap.ToBitmap(0, 100);
		await Assert.That(bitmap.Width).IsEqualTo(depthMap.GetWidth());
		await Assert.That(bitmap.Height).IsEqualTo(depthMap.GetHeight());
		await Assert.That(bitmap.IsRGB).IsTrue();
		await Assert.That(bitmap.GetPixel(0, 0)!.Value).IsEqualTo(new BitmapColor<byte>(0, 0, 128));
		await Assert.That(bitmap.GetPixel(0, 1)!.Value).IsEqualTo(new BitmapColor<byte>(128, 0, 0));
		await Assert.That(bitmap.GetPixel(1, 0)!.Value).IsEqualTo(new BitmapColor<byte>(128, 255, 127));
		await Assert.That(bitmap.GetPixel(1, 1)!.Value).IsEqualTo(new BitmapColor<byte>(128, 0, 0));
	}
}
