// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PhotoDecoder: turns a photo file into the ColmapSharp.Sensor.Bitmap the library reads, and a
// library Bitmap (the texture atlas) back into an agg ImageBuffer for the viewport and for
// saving texture.png. The library decodes nothing itself (Sensor/Bitmap.cs), so the demo uses
// agg-sharp's ImageIO (ImageSharp) and ExifReader for the focal length.
// Tests: demo/ColmapDemo.Tests/PhotoDecoderTests.cs.
//
// The two layouts differ in both channel order and row order, which is what the conversions
// are for: agg's ImageBuffer is 32-bit BGRA with row 0 at the bottom (ImageIO flips on load),
// while colmap's Bitmap is RGB with row 0 at the top, as the file stores it.

using System;
using System.IO;
using ColmapSharp.Sensor;
using MatterHackers.Agg;
using MatterHackers.Agg.Image;

namespace ColmapDemo
{
	/// <summary>Photo file to library Bitmap, and library Bitmap to agg ImageBuffer.</summary>
	public static class PhotoDecoder
	{
		/// <summary>
		/// Decodes <paramref name="path"/> into an RGB Bitmap, shrunk so its longer side is at most
		/// <paramref name="maxImageSize"/> (0 or less keeps the full size), with the JPEG's EXIF
		/// (focal length, camera model) attached so the library can seed the camera intrinsics.
		/// </summary>
		public static Bitmap Decode(string path, int maxImageSize)
		{
			byte[] bytes = File.ReadAllBytes(path);
			return Decode(bytes, maxImageSize);
		}

		/// <summary>The in-memory form of <see cref="Decode(string, int)"/>.</summary>
		public static Bitmap Decode(byte[] fileBytes, int maxImageSize)
		{
			ImageBuffer image;
			using (var stream = new MemoryStream(fileBytes))
			{
				image = ImageIO.LoadImage(stream);
			}

			if (image == null || image.Width == 0)
			{
				throw new InvalidDataException("Not an image this app can read.");
			}

			Bitmap bitmap = FromImageBuffer(image);

			// A photo without EXIF (a PNG, a stripped JPEG) still works: the library then guesses
			// the focal length from the image size, as COLMAP does.
			ExifReader.ReadJpeg(fileBytes, bitmap);

			if (maxImageSize > 0 && Math.Max(bitmap.Width, bitmap.Height) > maxImageSize)
			{
				// Thumbnail keeps the metadata; the EXIF focal length is in millimetres (or 35 mm
				// equivalent), which the library turns into pixels from the bitmap's own size.
				bitmap.Thumbnail(maxImageSize);
			}

			return bitmap;
		}

		/// <summary>
		/// Converts agg's bottom-up BGRA ImageBuffer into a top-down RGB Bitmap (alpha dropped).
		/// </summary>
		public static Bitmap FromImageBuffer(ImageBuffer image)
		{
			int width = image.Width;
			int height = image.Height;
			var bitmap = new Bitmap(width, height, asRgb: true);
			byte[] source = image.GetBuffer(out _);
			int bytesPerPixel = image.BitDepth / 8;
			byte[] target = bitmap.RowMajorData;
			for (int row = 0; row < height; row++)
			{
				int sourceIndex = image.GetBufferOffsetXY(0, height - 1 - row);
				int targetIndex = row * width * 3;
				for (int x = 0; x < width; x++)
				{
					target[targetIndex++] = source[sourceIndex + ImageBuffer.OrderR];
					target[targetIndex++] = source[sourceIndex + ImageBuffer.OrderG];
					target[targetIndex++] = source[sourceIndex + ImageBuffer.OrderB];
					sourceIndex += bytesPerPixel;
				}
			}

			return bitmap;
		}

		/// <summary>
		/// Converts a top-down RGB (or grey) Bitmap into agg's bottom-up 32-bit BGRA ImageBuffer,
		/// opaque. The inverse of <see cref="FromImageBuffer"/>.
		/// </summary>
		public static ImageBuffer ToImageBuffer(Bitmap bitmap)
		{
			int width = bitmap.Width;
			int height = bitmap.Height;
			int channels = bitmap.Channels;
			var image = new ImageBuffer(width, height);
			byte[] target = image.GetBuffer(out _);
			byte[] source = bitmap.RowMajorData;
			for (int row = 0; row < height; row++)
			{
				int targetIndex = image.GetBufferOffsetXY(0, height - 1 - row);
				int sourceIndex = row * width * channels;
				for (int x = 0; x < width; x++)
				{
					byte r = source[sourceIndex];
					byte g = channels == 3 ? source[sourceIndex + 1] : r;
					byte b = channels == 3 ? source[sourceIndex + 2] : r;
					target[targetIndex + ImageBuffer.OrderR] = r;
					target[targetIndex + ImageBuffer.OrderG] = g;
					target[targetIndex + ImageBuffer.OrderB] = b;
					target[targetIndex + ImageBuffer.OrderA] = 255;
					targetIndex += 4;
					sourceIndex += channels;
				}
			}

			image.MarkImageChanged();
			return image;
		}
	}
}
