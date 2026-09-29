// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteSegmenterSceneTests: C#-only tests (no COLMAP counterpart) of SilhouetteSegmenter on
// the rendered SyntheticObjectScene captures that Object mode is benchmarked on, against the
// scenes' true masks. The object turns nearly in place, so the median holds it and the Otsu
// fallback is what is tested. Outcome bar: IoU >= 0.975 on every frame.

using ColmapSharp.Mvs.Testing;
using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Segmentation;

public class SilhouetteSegmenterSceneTests
{
	// Regression: Otsu on L* split the sphere's own light and dark texture, the lighter half
	// joined the grey wall's class and was cut away (IoU 0.55-0.64 on frames 7-15).
	[Test]
	public async Task TexturedSphere_MatchesTrueMasks()
	{
		await AssertMasksMatch(SyntheticObjectKind.TexturedSphere);
	}

	[Test]
	public async Task DarkObject_MatchesTrueMasks()
	{
		await AssertMasksMatch(SyntheticObjectKind.DarkObject);
	}

	private static async Task AssertMasksMatch(SyntheticObjectKind kind)
	{
		var scene = SyntheticObjectScene.Generate(kind, 20, 320, 240, seed: 1, motionDuration: 0.27);
		SegmentationResult result = SilhouetteSegmenter.Segment(scene.Frames);

		// The object barely translates, so it survives into the median (correctly rejected).
		await Assert.That(result.Background).IsEqualTo(BackgroundModelKind.OtsuPersistentObject);
		for (int i = 0; i < scene.Frames.Count; ++i)
		{
			await Assert.That(IoU(result.Masks[i], scene.Masks[i])).IsGreaterThanOrEqualTo(0.975);
		}
	}

	private static double IoU(Bitmap mask, Bitmap truth)
	{
		byte[] m = mask.RowMajorData, t = truth.RowMajorData;
		int intersection = 0, union = 0;
		for (int p = 0; p < m.Length; ++p)
		{
			bool a = m[p] > 127, b = t[p] > 127;
			intersection += a && b ? 1 : 0;
			union += a || b ? 1 : 0;
		}

		return union == 0 ? 1.0 : (double)intersection / union;
	}
}
