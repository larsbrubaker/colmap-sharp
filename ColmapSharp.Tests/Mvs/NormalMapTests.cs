// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// NormalMapTests: colmap/mvs/normal_map_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/NormalMap.cs. Tier A (exact), as in C++. The C#-only case at the end pins
// Rescale's re-normalization.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class NormalMapTests
{
	[Test]
	public async Task NormalMap_Empty()
	{
		var normalMap = new NormalMap();
		await Assert.That(normalMap.GetWidth()).IsEqualTo(0);
		await Assert.That(normalMap.GetHeight()).IsEqualTo(0);
		await Assert.That(normalMap.GetDepth()).IsEqualTo(3);
	}

	[Test]
	public async Task NormalMap_NonEmpty()
	{
		var normalMap = new NormalMap(1, 2);
		await Assert.That(normalMap.GetWidth()).IsEqualTo(1);
		await Assert.That(normalMap.GetHeight()).IsEqualTo(2);
		await Assert.That(normalMap.GetDepth()).IsEqualTo(3);
	}

	[Test]
	public async Task NormalMap_Rescale()
	{
		var normalMap = new NormalMap(6, 7);
		normalMap.Rescale(0.5f);
		await Assert.That(normalMap.GetWidth()).IsEqualTo(3);
		await Assert.That(normalMap.GetHeight()).IsEqualTo(4);
		await Assert.That(normalMap.GetDepth()).IsEqualTo(3);
	}

	[Test]
	public async Task NormalMap_Downsize()
	{
		var normalMap = new NormalMap(6, 7);
		normalMap.Downsize(2, 4);
		await Assert.That(normalMap.GetWidth()).IsEqualTo(2);
		await Assert.That(normalMap.GetHeight()).IsEqualTo(2);
		await Assert.That(normalMap.GetDepth()).IsEqualTo(3);
	}

	[Test]
	public async Task NormalMap_ToBitmap()
	{
		var normalMap = new NormalMap(2, 2);
		normalMap.Set(0, 0, 0, 0);
		normalMap.Set(0, 0, 1, 0);
		normalMap.Set(0, 0, 2, 1);
		normalMap.Set(0, 1, 0, 0);
		normalMap.Set(0, 1, 1, 1);
		normalMap.Set(0, 1, 2, 0);
		normalMap.Set(1, 0, 0, 1);
		normalMap.Set(1, 0, 1, 0);
		normalMap.Set(1, 0, 2, 0);
		normalMap.Set(1, 1, 0, 1 / MathF.Sqrt(2.0f));
		normalMap.Set(1, 1, 1, 1 / MathF.Sqrt(2.0f));
		normalMap.Set(1, 1, 2, 0);
		Bitmap bitmap = normalMap.ToBitmap();
		await Assert.That(bitmap.Width).IsEqualTo(normalMap.GetWidth());
		await Assert.That(bitmap.Height).IsEqualTo(normalMap.GetHeight());
		await Assert.That(bitmap.IsRGB).IsTrue();
		await Assert.That(bitmap.GetPixel(0, 0)!.Value).IsEqualTo(new BitmapColor<byte>(128, 128, 0));
		await Assert.That(bitmap.GetPixel(0, 1)!.Value).IsEqualTo(new BitmapColor<byte>(0, 128, 0));
		await Assert.That(bitmap.GetPixel(1, 0)!.Value).IsEqualTo(new BitmapColor<byte>(128, 0, 0));
		await Assert.That(bitmap.GetPixel(1, 1)!.Value).IsEqualTo(new BitmapColor<byte>(37, 37, 0));
	}

	// C#-only: a constant field of non-unit normals comes back as unit normals.
	[Test]
	public async Task NormalMap_RescaleRenormalizes()
	{
		var normalMap = new NormalMap(4, 4);
		for (int r = 0; r < 4; r++)
		{
			for (int c = 0; c < 4; c++)
			{
				normalMap.Set(r, c, 0, 0);
				normalMap.Set(r, c, 1, 0);
				normalMap.Set(r, c, 2, -2);
			}
		}

		normalMap.Rescale(0.5f);
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++)
			{
				await Assert.That(normalMap.Get(r, c, 2)).IsEqualTo(-1.0f);
			}
		}
	}
}
