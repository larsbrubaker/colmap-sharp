// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SegmentationBuildingBlockTests: C#-only tests (no COLMAP counterpart) of the internal pieces
// of ColmapSharp/Segmentation that the end-to-end SilhouetteSegmenterTests cannot isolate: the
// padded closing, the convex hull fill, the hull-aware trimap, the border-connected background
// colour model and the factor-aligned mask upsampling. Each test fails when its feature is
// turned off (checked by mutation when they were written).

using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Segmentation;

public class SegmentationBuildingBlockTests
{
	[Test]
	public async Task Close_DoesNotBridgeLobesAlongTheBorder()
	{
		// Two lobes cut by the top edge, 5 pixels apart. The frame edge is a real boundary, so
		// closing by r = 3 must not fill the gap along it (unpadded, it would).
		const int W = 30, H = 30;
		bool[] a = Mask(W, H, (x, y) => y <= 9 && ((x >= 5 && x <= 9) || (x >= 15 && x <= 19)));
		bool[] closed = BinaryMorphology.Close(a, W, H, 3.0);

		for (int p = 0; p < a.Length; ++p)
		{
			await Assert.That(!a[p] || closed[p]).IsTrue();
		}

		for (int x = 0; x < W; ++x)
		{
			await Assert.That(closed[x]).IsEqualTo(a[x]);
		}
	}

	[Test]
	public async Task ConvexHull_SinglePoint()
	{
		bool[] hull = BinaryMorphology.ConvexHull(Mask(8, 8, (x, y) => x == 3 && y == 5), 8, 8);
		await Assert.That(Count(hull)).IsEqualTo(1);
		await Assert.That(hull[5 * 8 + 3]).IsTrue();
	}

	[Test]
	public async Task ConvexHull_CollinearPoints()
	{
		bool[] hull = BinaryMorphology.ConvexHull(Mask(12, 8, (x, y) => y == 5 && (x == 2 || x == 5 || x == 8)), 12, 8);
		await Assert.That(ToString(hull, 12, 8)).IsEqualTo(ToString(Mask(12, 8, (x, y) => y == 5 && x >= 2 && x <= 8), 12, 8));
	}

	[Test]
	public async Task ConvexHull_VerticalLine()
	{
		bool[] line = Mask(8, 10, (x, y) => x == 4 && y >= 1 && y <= 7);
		await Assert.That(ToString(BinaryMorphology.ConvexHull(line, 8, 10), 8, 10)).IsEqualTo(ToString(line, 8, 10));
	}

	[Test]
	public async Task ConvexHull_Polygon()
	{
		// Three corners of a right triangle; the hull fill is every pixel with x + y <= 10.
		bool[] corners = Mask(14, 14, (x, y) => (x, y) is (0, 0) or (10, 0) or (0, 10));
		bool[] hull = BinaryMorphology.ConvexHull(corners, 14, 14);
		await Assert.That(ToString(hull, 14, 14)).IsEqualTo(ToString(Mask(14, 14, (x, y) => x + y <= 10), 14, 14));
	}

	[Test]
	public async Task Trimap_HullAndBandAreUnknownAndKeptOutOfTheBackgroundModel()
	{
		// A square object whose right side reads as wall in a deep notch (the lit underside):
		// the notch lies inside the object's hull, more than TrimapDilateRadius from the rest.
		const int W = 60, H = 60;
		bool[] initial = Mask(W, H, (x, y) =>
			x >= 10 && x <= 49 && y >= 10 && y <= 49 && !(x >= 26 && y >= 16 && y <= 43));
		var options = new SegmentationOptions();
		bool[] sure = BinaryMorphology.Erode(initial, W, H, options.TrimapErodeRadius);
		(byte[] trimap, bool[] backgroundModelAllowed) = SilhouetteSegmenter.BuildTrimap(initial, sure, W, H, options);

		// Deep in the notch: 14+ pixels from the initial mask, so only the hull makes it Unknown.
		int notch = 30 * W + 44;
		await Assert.That(trimap[notch]).IsEqualTo(GrabCutRefiner.Unknown);
		await Assert.That(backgroundModelAllowed[notch]).IsFalse();

		// 9 pixels above the object: inside the default 12-pixel band, outside a 6-pixel one.
		int band = 1 * W + 30;
		await Assert.That(trimap[band]).IsEqualTo(GrabCutRefiner.Unknown);
		await Assert.That(backgroundModelAllowed[band]).IsFalse();

		// Far outside: sure background, and it trains the background model.
		int wall = 30 * W + 58;
		await Assert.That(trimap[wall]).IsEqualTo(GrabCutRefiner.SureBackground);
		await Assert.That(backgroundModelAllowed[wall]).IsTrue();
	}

	[Test]
	public async Task BackgroundModel_LearnsOnlyFromBackgroundReachingTheBorder()
	{
		// A foreground ring enclosing a background hole (white label text inside a dark object).
		const int W = 20, H = 20;
		bool[] labels = Mask(W, H, (x, y) => x >= 5 && x <= 14 && y >= 5 && y <= 14 && !(x >= 8 && x <= 11 && y >= 8 && y <= 11));
		bool[] allowed = Mask(W, H, (x, y) => true);
		int[] indices = GrabCutRefiner.BackgroundModelIndices(labels, allowed, W, H);

		await Assert.That(indices.Contains(0)).IsTrue();
		await Assert.That(indices.Contains(9 * W + 9)).IsFalse();
		await Assert.That(indices.Length).IsEqualTo(W * H - 100);
	}

	[Test]
	public async Task UpsampleMask_AlignsWithTheDownsampleNearRightAndBottomEdges()
	{
		// 803 x 487 at factor 5 is a 160 x 97 grid covering 800 x 485 pixels, so the source is
		// not a whole number of blocks. Working column 150 starts at source column 750 and row 90
		// at source row 450; a size-ratio mapping would put the edges near 752 and 451.
		bool[] mask = Mask(160, 97, (x, y) => x < 150 && y < 90);
		Bitmap up = LabImage.UpsampleMask(mask, 160, 97, 5, 803, 487);

		await Assert.That(up.GetPixel(749, 10)!.Value.R).IsEqualTo((byte)255);
		await Assert.That(up.GetPixel(750, 10)!.Value.R).IsEqualTo((byte)0);
		await Assert.That(up.GetPixel(10, 449)!.Value.R).IsEqualTo((byte)255);
		await Assert.That(up.GetPixel(10, 450)!.Value.R).IsEqualTo((byte)0);
	}

	private static bool[] Mask(int width, int height, Func<int, int, bool> set)
	{
		var mask = new bool[width * height];
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				mask[y * width + x] = set(x, y);
			}
		}

		return mask;
	}

	private static int Count(bool[] mask) => mask.Count(v => v);

	// Row strings make a mismatch readable in the failure message.
	private static string ToString(bool[] mask, int width, int height)
	{
		var text = new System.Text.StringBuilder();
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				text.Append(mask[y * width + x] ? '#' : '.');
			}

			text.Append('\n');
		}

		return text.ToString();
	}
}
