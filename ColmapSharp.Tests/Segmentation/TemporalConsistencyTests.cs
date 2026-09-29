// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TemporalConsistencyTests: C#-only tests (no COLMAP counterpart) of the temporal consistency of
// SilhouetteSegmenter (SegmentationOptions.TemporalWindow, docs/QUALITY_PLAN.md stage 1a+). The
// frames are a dark elliptical blob drifting and turning slowly on a grey gradient wall, drawn
// here. A "slot" is a wall-coloured cut into the blob's side, in the blob's own frame so it turns
// with it: in a bitten frame it is a segmentation error (the true object is the whole ellipse,
// like the lit underside of the mouse that matches the wall), in a notched frame it is real shape
// (the truth excludes it). Outcome bar: IoU >= 0.97, as for stage 1a.

using ColmapSharp.Mvs.Testing;
using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Segmentation;

public class TemporalConsistencyTests
{
	private const int Width = 200;
	private const int Height = 120;
	private const int FrameCount = 10;
	private const double SemiMajor = 22.0;
	private const double SemiMinor = 15.0;

	[Test]
	public async Task OneFrameBite_IsRepaired()
	{
		const int bitten = 5;
		List<Frame> frames = Sequence(k => k == bitten ? Slot.Bite : Slot.None);
		Bitmap[] images = [.. frames.Select(f => f.Image)];

		SegmentationResult off = SilhouetteSegmenter.Segment(images);
		SegmentationResult on = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });

		// Without temporal consistency the bite is kept; with it, every frame matches the truth.
		await Assert.That(IoU(off.Masks[bitten], frames[bitten].Truth)).IsLessThan(0.95);
		for (int k = 0; k < FrameCount; ++k)
		{
			await Assert.That(IoU(on.Masks[k], frames[k].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task PersistentNotch_IsKept()
	{
		// The blob really gains a notch at frame 4 and keeps it: consensus must not fill it.
		List<Frame> frames = Sequence(k => k >= 4 ? Slot.Notch : Slot.None);
		Bitmap[] images = [.. frames.Select(f => f.Image)];

		SegmentationResult on = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });

		for (int k = 0; k < FrameCount; ++k)
		{
			await Assert.That(IoU(on.Masks[k], frames[k].Truth)).IsGreaterThanOrEqualTo(0.97);
		}

		for (int k = 4; k < FrameCount; ++k)
		{
			(int x, int y) = frames[k].SlotCenter;
			await Assert.That(on.Masks[k].GetPixel(x, y)!.Value.R).IsEqualTo((byte)0);
		}
	}

	[Test]
	public async Task WindowZero_LeavesMasksUnchanged_AndOnlyOutliersChange()
	{
		const int bitten = 5;
		Bitmap[] images = [.. Sequence(k => k == bitten ? Slot.Bite : Slot.None).Select(f => f.Image)];

		SegmentationResult baseline = SilhouetteSegmenter.Segment(images);
		SegmentationResult zero = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 0 });
		SegmentationResult on = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });

		for (int k = 0; k < images.Length; ++k)
		{
			await Assert.That(zero.Masks[k].RowMajorData.SequenceEqual(baseline.Masks[k].RowMajorData)).IsTrue();
			// Frames that are not outliers are left byte for byte as the single-frame segmenter made them.
			bool same = on.Masks[k].RowMajorData.SequenceEqual(baseline.Masks[k].RowMajorData);
			await Assert.That(same).IsEqualTo(k != bitten);
		}
	}

	[Test]
	public async Task Sequential_EqualsParallel()
	{
		Bitmap[] images = [.. Sequence(k => k == 5 ? Slot.Bite : Slot.None).Select(f => f.Image)];

		SegmentationResult parallel = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });
		SegmentationResult sequential = SilhouetteSegmenter.Segment(
			images,
			new SegmentationOptions { TemporalWindow = 2, MaxDegreeOfParallelism = 1 });

		for (int k = 0; k < images.Length; ++k)
		{
			await Assert.That(sequential.Masks[k].RowMajorData.SequenceEqual(parallel.Masks[k].RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task DarkObjectScene_MatchesTrueMasks()
	{
		// The rendered stand-in for the mouse capture (stage 0a), with and without temporal
		// consistency. Its lit, grey underside is the case that bit the real footage.
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.DarkObject, 16, 240, 180, seed: 3);
		SegmentationResult off = SilhouetteSegmenter.Segment(scene.Frames);
		SegmentationResult on = SilhouetteSegmenter.Segment(scene.Frames, new SegmentationOptions { TemporalWindow = 2 });

		var report = new List<string>();
		for (int k = 0; k < scene.Frames.Count; ++k)
		{
			bool[] truth = [.. scene.Masks[k].RowMajorData.Select(v => v != 0)];
			double iouOff = IoU(off.Masks[k], truth), iouOn = IoU(on.Masks[k], truth);
			report.Add($"frame {k}: off {iouOff:F4} on {iouOn:F4}");
			Console.WriteLine(report[^1]);
			await Assert.That(iouOff).IsGreaterThanOrEqualTo(0.97);
			await Assert.That(iouOn).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task TwoFrameBite_IsRepairedFromTheCleanFrames()
	{
		// Frames 5 and 6 are both bitten. A window of 3 is needed: at 2, half of each bitten
		// frame's neighbours share its bite, so neither stands out. Each is an outlier, so neither
		// may vote in the other's consensus; the bite is filled from the clean frames alone.
		List<Frame> frames = Sequence(k => k is 5 or 6 ? Slot.Bite : Slot.None);

		SegmentationResult on = SilhouetteSegmenter.Segment(
			[.. frames.Select(f => f.Image)],
			new SegmentationOptions { TemporalWindow = 3 });

		for (int k = 0; k < FrameCount; ++k)
		{
			await Assert.That(IoU(on.Masks[k], frames[k].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task TwoFrameBite_OutlierNeighbourGetsNoVote()
	{
		// The same two bitten frames, with the thresholds raised so that a bite pixel is filled
		// only when about 86% (0.6 / 0.7) of the weighted votes agree. The clean frames agree
		// fully; a vote from the other bitten frame, even down-weighted by its poor agreement,
		// would pull the bite's consensus below that.
		List<Frame> frames = Sequence(k => k is 5 or 6 ? Slot.Bite : Slot.None);

		SegmentationResult on = SilhouetteSegmenter.Segment(
			[.. frames.Select(f => f.Image)],
			new SegmentationOptions { TemporalWindow = 3, TemporalHighThreshold = 0.62, TemporalLowThreshold = 0.6 });

		await Assert.That(IoU(on.Masks[5], frames[5].Truth)).IsGreaterThanOrEqualTo(0.97);
		await Assert.That(IoU(on.Masks[6], frames[6].Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	[Test]
	public async Task FastInPlaneTurn_BiteIsRepaired()
	{
		// 0.3 rad (17 degrees) per frame: without the principal-axis rotation the neighbours'
		// outlines cross the frame's at up to 34 degrees and no consensus forms.
		const int bitten = 5;
		List<Frame> frames = Sequence(k => k == bitten ? Slot.Bite : Slot.None, angleStep: 0.3);

		SegmentationResult on = SilhouetteSegmenter.Segment(
			[.. frames.Select(f => f.Image)],
			new SegmentationOptions { TemporalWindow = 2 });

		for (int k = 0; k < FrameCount; ++k)
		{
			await Assert.That(IoU(on.Masks[k], frames[k].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task FastOutOfPlaneTurn_NoFrameChanges()
	{
		// The object turns about the image's vertical axis in one quick step around frame 4, so its
		// outline narrows in a way no similarity transform aligns: frame 4 differs from every
		// neighbour, but so do frames 3 and 5 from each other. Nothing is an error; nothing changes.
		List<Frame> frames = Sequence(_ => Slot.None, semiMinor: k => k < 4 ? 15.0 : k == 4 ? 10.0 : 6.0);
		Bitmap[] images = [.. frames.Select(f => f.Image)];

		SegmentationResult off = SilhouetteSegmenter.Segment(images);
		SegmentationResult on = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });

		for (int k = 0; k < FrameCount; ++k)
		{
			await AssertUnchanged(on, off, k);
		}
	}

	[Test]
	public async Task FrameTouchingBorder_IsLeftAloneAndNotANeighbour()
	{
		// Frame 7 jumps to the left edge, cut off by it: its moments are of the visible part, so
		// aligned by them it would read as an outlier itself, and as a misaligned neighbour of
		// the bitten frame 6 next to it. It must neither be changed nor vote.
		const int bitten = 6, atBorder = 7;
		List<Frame> frames = Sequence(
			k => k == bitten ? Slot.Bite : Slot.None,
			centerX: k => k == atBorder ? 4.0 : 70.0 + 6.0 * k);
		Bitmap[] images = [.. frames.Select(f => f.Image)];

		SegmentationResult off = SilhouetteSegmenter.Segment(images);
		SegmentationResult on = SilhouetteSegmenter.Segment(images, new SegmentationOptions { TemporalWindow = 2 });

		await AssertUnchanged(on, off, atBorder);
		await AssertUnchanged(on, off, atBorder + 1);
		await Assert.That(IoU(on.Masks[bitten], frames[bitten].Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	[Test]
	public async Task HysteresisLowBand_FillsBiteConnectedToObject()
	{
		// With the high threshold at 0.75 the bite's pixels (P = 0.7 * consensus <= 0.7) are
		// never seeds; they are filled only through the low band, by connecting to the object.
		const int bitten = 5;
		List<Frame> frames = Sequence(k => k == bitten ? Slot.Bite : Slot.None);

		SegmentationResult on = SilhouetteSegmenter.Segment(
			[.. frames.Select(f => f.Image)],
			new SegmentationOptions { TemporalWindow = 2, TemporalHighThreshold = 0.75 });

		await Assert.That(IoU(on.Masks[bitten], frames[bitten].Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	private enum Slot
	{
		None,
		Bite,
		Notch,
	}

	private sealed record Frame(Bitmap Image, bool[] Truth, (int X, int Y) SlotCenter);

	// Frame k: centre x 70 + 6k (unless centerX says otherwise), angle angleStep * k, semi-minor
	// axis SemiMinor (unless semiMinor says otherwise).
	private static List<Frame> Sequence(
		Func<int, Slot> slot,
		double angleStep = 0.06,
		Func<int, double>? centerX = null,
		Func<int, double>? semiMinor = null)
	{
		var frames = new List<Frame>();
		for (int k = 0; k < FrameCount; ++k)
		{
			double x = centerX?.Invoke(k) ?? 70.0 + 6.0 * k;
			frames.Add(Draw(x, angleStep * k, slot(k), (uint)(k + 1), semiMinor?.Invoke(k) ?? SemiMinor));
		}

		return frames;
	}

	private static async Task AssertUnchanged(SegmentationResult on, SegmentationResult off, int k)
	{
		await Assert.That(on.Masks[k].RowMajorData.SequenceEqual(off.Masks[k].RowMajorData)).IsTrue();
	}

	// The slot: |u| <= 0.2 and v <= -0.3 in the blob's unit frame, about 9 pixels wide and 10
	// deep, wider than the segmenter's closing (a radius of about 3 pixels) seals.
	private static Frame Draw(double centerX, double angle, Slot slot, uint seed, double semiMinor)
	{
		const double centerY = 62.0;
		var image = new Bitmap(Width, Height, asRgb: true);
		var truth = new bool[Width * Height];
		double cos = Math.Cos(angle), sin = Math.Sin(angle);
		uint state = seed * 2654435761u + 1u;
		for (int y = 0; y < Height; ++y)
		{
			for (int x = 0; x < Width; ++x)
			{
				double dx = x + 0.5 - centerX, dy = y + 0.5 - centerY;
				double u = (cos * dx + sin * dy) / SemiMajor;
				double v = (-sin * dx + cos * dy) / semiMinor;
				bool inBlob = u * u + v * v <= 1.0;
				bool inSlot = slot != Slot.None && inBlob && Math.Abs(u) <= 0.2 && v <= -0.3;
				double wall = 140.0 + 50.0 * x / Width;
				double grey = inBlob && !inSlot ? 32.0 : wall;
				state = state * 1664525u + 1013904223u;
				double noise = ((state >> 24) / 255.0 - 0.5) * 12.0;
				byte value = (byte)Math.Clamp(Math.Round(grey + noise), 0.0, 255.0);
				image.SetPixel(x, y, new BitmapColor<byte>(value, value, value));
				truth[y * Width + x] = inBlob && !(slot == Slot.Notch && inSlot);
			}
		}

		// The slot's centre (u = 0, v = -0.65) in image pixels.
		double cu = 0.0, cv = -0.65 * semiMinor;
		(int, int) slotCenter = ((int)(centerX + cos * cu - sin * cv), (int)(centerY + sin * cu + cos * cv));
		return new Frame(image, truth, slotCenter);
	}

	private static double IoU(Bitmap mask, bool[] truth)
	{
		int intersection = 0, union = 0;
		for (int y = 0; y < mask.Height; ++y)
		{
			for (int x = 0; x < mask.Width; ++x)
			{
				bool m = mask.GetPixel(x, y)!.Value.R != 0;
				bool t = truth[y * mask.Width + x];
				intersection += m && t ? 1 : 0;
				union += m || t ? 1 : 0;
			}
		}

		return union == 0 ? 1.0 : (double)intersection / union;
	}
}
