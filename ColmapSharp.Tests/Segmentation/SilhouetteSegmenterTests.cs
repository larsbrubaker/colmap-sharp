// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteSegmenterTests: C#-only tests (no COLMAP counterpart) of ColmapSharp/Segmentation.
// Each frame is drawn here: a dark elliptical blob on a grey horizontal-gradient wall, a thin
// dark cable from the blob to the top edge, a bright specular spot inside the blob, and
// deterministic noise. Frames move and rotate the blob (or spin it in place). The true mask is the ellipse alone
// (cable excluded, highlight included). Outcome bar: IoU >= 0.97 (docs/QUALITY_PLAN.md 1a).

using ColmapSharp.Controllers;
using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Segmentation;

public class SilhouetteSegmenterTests
{
	private const int Width = 200;
	private const int Height = 120;
	private const double SemiMajor = 22.0;
	private const double SemiMinor = 15.0;
	private const double CenterY = 70.0;

	[Test]
	public async Task TemporalMedian_MatchesBlobMask()
	{
		List<Frame> frames = MovingFrames(6);
		SegmentationResult result = SilhouetteSegmenter.Segment([.. frames.Select(f => f.Image)]);

		await Assert.That(result.UsedTemporalMedian).IsTrue();
		for (int i = 0; i < frames.Count; ++i)
		{
			await Assert.That(result.ForegroundFound[i]).IsTrue();
			await Assert.That(IoU(result.Masks[i], frames[i].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task ObjectSpinningInPlace_FallsBackToOtsu()
	{
		// The driving case: the object turns but stays put, so it survives into the median.
		var frames = new List<Frame>();
		for (int i = 0; i < 6; ++i)
		{
			frames.Add(Draw(100.0 + 2.0 * i, 0.3 * i, seed: (uint)(i + 11)));
		}

		SegmentationResult result = SilhouetteSegmenter.Segment([.. frames.Select(f => f.Image)]);

		await Assert.That(result.Background).IsEqualTo(BackgroundModelKind.OtsuPersistentObject);
		for (int i = 0; i < frames.Count; ++i)
		{
			await Assert.That(IoU(result.Masks[i], frames[i].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task Cable_IsRemoved()
	{
		List<Frame> frames = MovingFrames(6);
		SegmentationResult result = SilhouetteSegmenter.Segment([.. frames.Select(f => f.Image)]);

		for (int i = 0; i < frames.Count; ++i)
		{
			Frame frame = frames[i];
			int kept = 0;
			// The cable runs from the top edge down to the blob; test well clear of the blob.
			int bottom = (int)(CenterY - SemiMajor) - 4;
			for (int y = 0; y < bottom; ++y)
			{
				for (int x = (int)frame.CenterX - 4; x <= (int)frame.CenterX + 4; ++x)
				{
					kept += result.Masks[i].GetPixel(x, y)!.Value.R != 0 ? 1 : 0;
				}
			}

			await Assert.That(kept).IsEqualTo(0);
		}
	}

	[Test]
	public async Task SpecularHighlight_StaysForeground()
	{
		List<Frame> frames = MovingFrames(6);
		SegmentationResult result = SilhouetteSegmenter.Segment([.. frames.Select(f => f.Image)]);

		for (int i = 0; i < frames.Count; ++i)
		{
			Frame frame = frames[i];
			for (int y = (int)CenterY - 4; y <= (int)CenterY + 4; ++y)
			{
				for (int x = (int)frame.CenterX - 4; x <= (int)frame.CenterX + 4; ++x)
				{
					await Assert.That(result.Masks[i].GetPixel(x, y)!.Value.R).IsEqualTo((byte)255);
				}
			}
		}
	}

	[Test]
	public async Task SingleFrame_UsesOtsuFallback()
	{
		Frame frame = Draw(100.0, 0.3, seed: 7);
		SegmentationResult result = SilhouetteSegmenter.Segment([frame.Image]);

		await Assert.That(result.UsedTemporalMedian).IsFalse();
		await Assert.That(result.ForegroundFound[0]).IsTrue();
		await Assert.That(IoU(result.Masks[0], frame.Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	[Test]
	public async Task Segment_IsDeterministic()
	{
		List<Frame> frames = MovingFrames(6);
		Bitmap[] images = [.. frames.Select(f => f.Image)];
		SegmentationResult first = SilhouetteSegmenter.Segment(images);
		SegmentationResult second = SilhouetteSegmenter.Segment(images);

		for (int i = 0; i < images.Length; ++i)
		{
			await Assert.That(second.Masks[i].RowMajorData.SequenceEqual(first.Masks[i].RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task MaskSource_UsesImageReaderNaming()
	{
		List<Frame> frames = MovingFrames(6);
		var images = new InMemoryImageSource();
		for (int i = 0; i < frames.Count; ++i)
		{
			images.Add($"video/frame{i:D3}.jpg", frames[i].Image);
		}

		SilhouetteMaskSource masks = SilhouetteSegmenter.SegmentToMaskSource(images);

		await Assert.That(masks.ListNames().Count).IsEqualTo(frames.Count);
		await Assert.That(masks.ForegroundFound("video/frame002.jpg")).IsTrue();
		await Assert.That(() => masks.ForegroundFound("video/missing.jpg")).Throws<ArgumentException>();
		Bitmap? mask = masks.Read("video/frame002.jpg.png");
		await Assert.That(mask).IsNotNull();
		await Assert.That(mask!.IsGrey).IsTrue();
		await Assert.That(IoU(mask, frames[2].Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	[Test]
	public async Task FramesLargerThanWorkingSize_AreResampled()
	{
		// 800 x 480 at WorkingSize 200: segmented at a quarter of the size and upsampled back.
		var frames = new List<Frame>();
		for (int i = 0; i < 6; ++i)
		{
			frames.Add(Draw(new Scene(35.0 + 26.0 * i, 0.25 * i, (uint)(i + 21), Scale: 4)));
		}

		var options = new SegmentationOptions { WorkingSize = 200 };
		SegmentationResult result = SilhouetteSegmenter.Segment([.. frames.Select(f => f.Image)], options);

		await Assert.That(result.UsedTemporalMedian).IsTrue();
		for (int i = 0; i < frames.Count; ++i)
		{
			await Assert.That(result.Masks[i].Width).IsEqualTo(Width * 4);
			await Assert.That(result.Masks[i].Height).IsEqualTo(Height * 4);
			await Assert.That(IoU(result.Masks[i], frames[i].Truth)).IsGreaterThanOrEqualTo(0.97);
		}
	}

	[Test]
	public async Task NoObject_KeepsEverythingAndReportsNotFound()
	{
		Frame frame = Draw(new Scene(100.0, 0.0, 5, Blob: false));
		SegmentationResult result = SilhouetteSegmenter.Segment([frame.Image]);

		await Assert.That(result.ForegroundFound[0]).IsFalse();
		await Assert.That(result.Masks[0].RowMajorData.All(v => v == 255)).IsTrue();
	}

	[Test]
	public async Task LightObjectOnDarkWall_MatchesBlobMask()
	{
		Frame frame = Draw(new Scene(100.0, 0.4, 9, Light: true));
		SegmentationResult result = SilhouetteSegmenter.Segment([frame.Image]);

		await Assert.That(result.ForegroundFound[0]).IsTrue();
		await Assert.That(IoU(result.Masks[0], frame.Truth)).IsGreaterThanOrEqualTo(0.97);
	}

	[Test]
	public async Task Sequential_EqualsParallel()
	{
		Bitmap[] images = [.. MovingFrames(6).Select(f => f.Image)];
		SegmentationResult parallel = SilhouetteSegmenter.Segment(images);
		SegmentationResult sequential = SilhouetteSegmenter.Segment(images, new SegmentationOptions { MaxDegreeOfParallelism = 1 });

		for (int i = 0; i < images.Length; ++i)
		{
			await Assert.That(sequential.Masks[i].RowMajorData.SequenceEqual(parallel.Masks[i].RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task ThinRimChannel_IsClosed()
	{
		Frame frame = Draw(new Scene(100.0, 0.0, 13, Channel: true));
		SegmentationResult result = SilhouetteSegmenter.Segment([frame.Image]);

		await Assert.That(IoU(result.Masks[0], frame.Truth)).IsGreaterThanOrEqualTo(0.97);
		// The slot's pixels, from 2 inside the rim to its inner end, are foreground again.
		for (int x = (int)(100.0 + SemiMajor - 11.0); x <= (int)(100.0 + SemiMajor - 2.0); ++x)
		{
			await Assert.That(result.Masks[0].GetPixel(x, (int)CenterY)!.Value.R).IsEqualTo((byte)255);
		}
	}

	[Test]
	public async Task ObjectTouchingFrameEdge_GetsNoBorderStrips()
	{
		// A large blob (88 x 60) with 34 of its 88 columns cut off by the right edge.
		Frame frame = Draw(new Scene(Width - 10.0, 0.0, 17, Size: 2.0));
		SegmentationResult result = SilhouetteSegmenter.Segment([frame.Image]);

		await Assert.That(IoU(result.Masks[0], frame.Truth)).IsGreaterThanOrEqualTo(0.97);
		// The edge columns match the truth except within a pixel of the blob's outline.
		int mismatches = 0;
		foreach (int x in new[] { 0, Width - 1 })
		{
			for (int y = 0; y < Height; ++y)
			{
				mismatches += (result.Masks[0].GetPixel(x, y)!.Value.R != 0) != frame.Truth[y * Width + x] ? 1 : 0;
			}
		}

		await Assert.That(mismatches).IsLessThanOrEqualTo(2);
	}

	private sealed record Frame(Bitmap Image, bool[] Truth, double CenterX);

	private static List<Frame> MovingFrames(int count)
	{
		var frames = new List<Frame>();
		for (int i = 0; i < count; ++i)
		{
			// Far enough apart that every wall pixel is uncovered in most frames.
			frames.Add(Draw(35.0 + 26.0 * i, 0.25 * i, seed: (uint)(i + 1)));
		}

		return frames;
	}

	private static Frame Draw(double centerX, double angle, uint seed) => Draw(new Scene(centerX, angle, seed));

	// Scale multiplies the frame size (and every length); Light swaps to a light object on a
	// dark wall (no cable or highlight); Channel cuts a thin wall-coloured slot in from the rim.
	private sealed record Scene(double CenterX, double Angle, uint Seed, int Scale = 1, bool Blob = true, bool Light = false, bool Channel = false, double Size = 1.0);

	private static Frame Draw(Scene scene)
	{
		int s = scene.Scale, width = Width * s, height = Height * s;
		var image = new Bitmap(width, height, asRgb: true);
		var truth = new bool[width * height];
		double cos = Math.Cos(scene.Angle), sin = Math.Sin(scene.Angle);
		uint state = scene.Seed * 2654435761u + 1u;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				// Geometry in unscaled units, so every scene has the same shape.
				double dx = (x + 0.5) / s - scene.CenterX, dy = (y + 0.5) / s - CenterY;
				double u = (cos * dx + sin * dy) / (SemiMajor * scene.Size);
				double v = (-sin * dx + cos * dy) / (SemiMinor * scene.Size);
				bool inBlob = scene.Blob && u * u + v * v <= 1.0;
				bool decorated = scene.Blob && !scene.Light;
				bool onCable = decorated && !inBlob && dy < 0.0 && Math.Abs(dx) <= 1.0;
				bool onHighlight = decorated && dx * dx + dy * dy <= 9.0;
				// A 2-pixel slot from the right rim to 12 pixels deep, along the blob's row: well under
				// the closing diameter (CloseFraction 0.1 of a 30-pixel-wide blob is a radius of 3).
				bool onChannel = scene.Channel && inBlob && Math.Abs(dy) <= 1.0 && dx >= SemiMajor - 12.0;
				double wall = scene.Light ? 40.0 + 30.0 * x / width : 140.0 + 50.0 * x / width;
				double r, g, b;
				if (onHighlight)
				{
					r = g = b = 245.0;
				}
				else if (onChannel)
				{
					r = g = b = wall;
				}
				else if (inBlob || onCable)
				{
					r = scene.Light ? 215.0 : 32.0;
					g = scene.Light ? 210.0 : 30.0;
					b = scene.Light ? 200.0 : 36.0;
				}
				else
				{
					// The wall, brighter to the right.
					r = g = b = wall;
				}

				state = state * 1664525u + 1013904223u;
				double noise = ((state >> 24) / 255.0 - 0.5) * 12.0;
				image.SetPixel(x, y, new BitmapColor<byte>(Byte(r + noise), Byte(g + noise), Byte(b + noise)));
				truth[y * width + x] = inBlob;
			}
		}

		return new Frame(image, truth, scene.CenterX);
	}

	private static byte Byte(double value) => (byte)Math.Clamp(Math.Round(value), 0.0, 255.0);

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
