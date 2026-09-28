// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of PhotoDecoder (demo/ColmapDemo/PhotoDecoder.cs) and ColmapDemoApp.IsPhotoPath: that a
// decoded photo lands in the library's Bitmap with RGB channel order and the file's top row
// first, that the size cap applies, and that the atlas conversion back to agg is its inverse.

using ColmapSharp.Sensor;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace ColmapDemo.Tests;

public class PhotoDecoderTests
{
	// A 3x2 image: top row red, green, blue; bottom row white, black, grey.
	private static readonly Rgba32[] Pixels =
	[
		new Rgba32(255, 0, 0), new Rgba32(0, 255, 0), new Rgba32(0, 0, 255),
		new Rgba32(255, 255, 255), new Rgba32(0, 0, 0), new Rgba32(128, 128, 128),
	];

	private static byte[] Encode(bool jpeg, int width = 3, int height = 2, Rgba32[] pixels = null)
	{
		using var image = Image.LoadPixelData<Rgba32>(pixels ?? Pixels, width, height);
		using var stream = new MemoryStream();
		if (jpeg)
		{
			image.Save(stream, new JpegEncoder { Quality = 100 });
		}
		else
		{
			image.Save(stream, new PngEncoder());
		}

		return stream.ToArray();
	}

	[Test]
	public async Task PngDecodesAsRgbWithTopRowFirst()
	{
		Bitmap bitmap = PhotoDecoder.Decode(Encode(jpeg: false), maxImageSize: 0);

		await Assert.That(bitmap.Width).IsEqualTo(3);
		await Assert.That(bitmap.Height).IsEqualTo(2);
		await Assert.That(bitmap.IsRGB).IsTrue();
		byte[] expected =
		[
			255, 0, 0, 0, 255, 0, 0, 0, 255,
			255, 255, 255, 0, 0, 0, 128, 128, 128,
		];
		await Assert.That(bitmap.RowMajorData).IsEquivalentTo(expected);
	}

	[Test]
	public async Task JpegDecodesAsRgbWithTopRowFirst()
	{
		// JPEG is lossy (and chroma-subsampled), so a flat 16x16 red-over-blue image is used and
		// checked within a tolerance at the centers of the two halves.
		var pixels = new Rgba32[16 * 16];
		for (int i = 0; i < pixels.Length; i++)
		{
			pixels[i] = i < pixels.Length / 2 ? new Rgba32(255, 0, 0) : new Rgba32(0, 0, 255);
		}

		Bitmap bitmap = PhotoDecoder.Decode(Encode(jpeg: true, 16, 16, pixels), maxImageSize: 0);

		BitmapColor<byte> top = bitmap.GetPixel(8, 3)!.Value;
		BitmapColor<byte> bottom = bitmap.GetPixel(8, 12)!.Value;
		await Assert.That((int)top.R).IsGreaterThan(200);
		await Assert.That((int)top.B).IsLessThan(60);
		await Assert.That((int)bottom.B).IsGreaterThan(200);
		await Assert.That((int)bottom.R).IsLessThan(60);
	}

	[Test]
	public async Task MaxImageSizeCapsTheLongerSide()
	{
		var pixels = Enumerable.Repeat(new Rgba32(10, 20, 30), 40 * 20).ToArray();
		Bitmap bitmap = PhotoDecoder.Decode(Encode(jpeg: false, 40, 20, pixels), maxImageSize: 10);

		await Assert.That(bitmap.Width).IsEqualTo(10);
		await Assert.That(bitmap.Height).IsEqualTo(5);
	}

	[Test]
	public async Task ToImageBufferIsTheInverseOfFromImageBuffer()
	{
		Bitmap original = PhotoDecoder.Decode(Encode(jpeg: false), maxImageSize: 0);

		Bitmap roundTrip = PhotoDecoder.FromImageBuffer(PhotoDecoder.ToImageBuffer(original));

		await Assert.That(roundTrip.RowMajorData).IsEquivalentTo(original.RowMajorData);
	}

	[Test]
	public async Task NotAnImageThrows()
	{
		await Assert.That(() => PhotoDecoder.Decode(new byte[] { 1, 2, 3, 4 }, 0)).Throws<Exception>();
	}

	[Test]
	[Arguments("a/IMG_0001.JPG", true)]
	[Arguments("b.jpeg", true)]
	[Arguments("c.png", true)]
	[Arguments("d.tiff", true)]
	[Arguments("notes.txt", false)]
	[Arguments("clip.mov", false)]
	[Arguments("noextension", false)]
	public async Task IsPhotoPathTakesPhotoExtensionsOnly(string path, bool expected)
	{
		await Assert.That(ColmapDemoApp.IsPhotoPath(path)).IsEqualTo(expected);
	}
}
