// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Bitmap: colmap/sensor/bitmap.h/.cc, the 8-bit grey or RGB pixel buffer (row-major,
// interleaved) that feature extraction, undistortion and MVS read images into. The class is
// split by responsibility:
//   Bitmap.cs          - storage, pixel access, interpolation, rotation, grey/RGB clones
//   Bitmap.Exif.cs     - the typed metadata store and the EXIF getters (focal length, GPS)
//   BitmapResize.cs    - Rescale/Thumbnail's resampling filters
//   BitmapColor.cs     - the pixel value type; JetColormap.cs - the colormap in bitmap.h
// Tests: ColmapSharp.Tests/Sensor/BitmapTests.cs (bitmap_test.cc; the file I/O cases are
// skipped, see PORTING_PLAN.md).
//
// No file decoding or encoding: COLMAP's Read/Write go through OpenImageIO, which is native
// and not ported (docs/LICENSE_AUDIT.md). The host decodes an image into RowMajorData and
// feeds its EXIF block to ExifReader (ExifReader.cs), which fills the metadata the EXIF
// getters read under OpenImageIO's attribute names.
//
// Tier A (exact) for everything here except Rescale (BitmapResize.cs, docs/CPP_DIVERGENCES.md
// entry 9). Translation notes:
// - C++ copy construction/assignment is Clone(); a default-constructed or cloned-empty
//   Bitmap has no metadata store, like COLMAP's null meta_data_, and metadata access on it
//   fails its THROW_CHECK_NOTNULL. C++ move semantics have no C# counterpart.
// - The linear-colorspace flag only matters to OIIO's Read/Write color conversion; it is
//   kept (and recorded as "oiio:ColorSpace" metadata, as COLMAP's constructor does) so a
//   host can round-trip it.

using System.Globalization;

using ColmapSharp.Util;

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::Bitmap: an 8-bit grey (1 channel) or RGB (3 channels) image, stored
/// row-major with interleaved channels.
/// </summary>
public sealed partial class Bitmap
{
	private int width;
	private int height;
	private int channels;
	private bool linearColorspace;
	private byte[] data;

	/// <summary>Empty bitmap (width, height and channels 0), without metadata.</summary>
	public Bitmap()
	{
		data = [];
	}

	/// <summary>Allocates a zero-filled bitmap with the given dimensions.</summary>
	public Bitmap(int width, int height, bool asRgb, bool linearColorspace = false)
	{
		this.width = width;
		this.height = height;
		channels = asRgb ? 3 : 1;
		this.linearColorspace = linearColorspace;
		data = new byte[width * height * channels];
		metaData = new BitmapMetaData();
		metaData.Set("oiio:ColorSpace", linearColorspace ? "linear" : "sRGB");
	}

	/// <summary>Width in pixels.</summary>
	public int Width => width;

	/// <summary>Height in pixels.</summary>
	public int Height => height;

	/// <summary>Number of channels: 1 for grey, 3 for RGB, 0 when empty.</summary>
	public int Channels => channels;

	/// <summary>Number of bits per pixel. This is 8 for grey and 24 for RGB images.</summary>
	public int BitsPerPixel => channels * 8;

	/// <summary>Number of bytes required to store image.</summary>
	public int NumBytes => data.Length;

	/// <summary>Scan line size in bytes, also known as stride.</summary>
	public int Pitch => width * channels;

	/// <summary>Whether the image is empty (i.e., width/height=0).</summary>
	public bool IsEmpty => NumBytes == 0;

	/// <summary>Whether the image has three channels.</summary>
	public bool IsRGB => channels == 3;

	/// <summary>Whether the image has one channel.</summary>
	public bool IsGrey => channels == 1;

	/// <summary>Whether the pixel values are in a linear (not sRGB) colorspace.</summary>
	public bool IsLinearColorspace => linearColorspace;

	/// <summary>
	/// The raw pixel array, row-major with interleaved channels (Pitch bytes per row). The
	/// host writes decoded pixels here; its length is fixed by the dimensions.
	/// </summary>
	public byte[] RowMajorData => data;

	/// <summary>
	/// Port of Bitmap::GetPixel. Null outside the image. For grey images all three
	/// components hold the grey value.
	/// </summary>
	public BitmapColor<byte>? GetPixel(int x, int y)
	{
		if (x < 0 || x >= width || y < 0 || y >= height)
		{
			return null;
		}

		if (IsGrey)
		{
			byte v = data[y * width + x];
			return new BitmapColor<byte>(v, v, v);
		}
		else if (IsRGB)
		{
			int pixel = (y * width + x) * channels;
			return new BitmapColor<byte>(data[pixel], data[pixel + 1], data[pixel + 2]);
		}

		return null;
	}

	/// <summary>
	/// Port of Bitmap::SetPixel. Returns false outside the image. For grayscale images, only
	/// the red element of the color is used.
	/// </summary>
	public bool SetPixel(int x, int y, BitmapColor<byte> color)
	{
		if (x < 0 || x >= width || y < 0 || y >= height)
		{
			return false;
		}

		if (IsGrey)
		{
			data[y * width + x] = color.R;
			return true;
		}
		else if (IsRGB)
		{
			int pixel = (y * width + x) * channels;
			data[pixel] = color.R;
			data[pixel + 1] = color.G;
			data[pixel + 2] = color.B;
			return true;
		}

		return false;
	}

	/// <summary>
	/// Port of Bitmap::Fill: fill entire bitmap with uniform color. For grayscale images,
	/// the red element is used.
	/// </summary>
	public void Fill(BitmapColor<byte> color)
	{
		if (IsGrey)
		{
			Array.Fill(data, color.R);
		}
		else
		{
			Check.Eq(data.Length % 3, 0);
			int i = 0;
			while (i < data.Length)
			{
				data[i++] = color.R;
				data[i++] = color.G;
				data[i++] = color.B;
			}
		}
	}

	/// <summary>
	/// Port of Bitmap::InterpolateNearestNeighbor: the pixel at the rounded position
	/// (std::round, half away from zero), or null outside the image.
	/// </summary>
	public BitmapColor<byte>? InterpolateNearestNeighbor(double x, double y)
	{
		int xx = (int)Math.Round(x, MidpointRounding.AwayFromZero);
		int yy = (int)Math.Round(y, MidpointRounding.AwayFromZero);
		return GetPixel(xx, yy);
	}

	/// <summary>
	/// Port of Bitmap::InterpolateBilinear. Null unless all four neighbors are inside the
	/// image. Accumulates in double and rounds to float at the end, as COLMAP does.
	/// </summary>
	public BitmapColor<float>? InterpolateBilinear(double x, double y)
	{
		int x0 = (int)Math.Floor(x);
		int x1 = x0 + 1;
		int y0 = (int)Math.Floor(y);
		int y1 = y0 + 1;

		if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height)
		{
			return null;
		}

		double dx = x - x0;
		double dy = y - y0;
		double dx_1 = 1 - dx;
		double dy_1 = 1 - dy;

		int pitch = width * channels;
		int line0 = y0 * pitch;
		int line1 = y1 * pitch;

		if (IsGrey)
		{
			// Top row, column-wise linear interpolation.
			double v0 = dx_1 * data[line0 + x0] + dx * data[line0 + x1];

			// Bottom row, column-wise linear interpolation.
			double v1 = dx_1 * data[line1 + x0] + dx * data[line1 + x1];

			// Row-wise linear interpolation.
			float r = (float)(dy_1 * v0 + dy * v1);
			return new BitmapColor<float>(r, r, r);
		}
		else if (IsRGB)
		{
			int p00 = line0 + 3 * x0;
			int p01 = line0 + 3 * x1;
			int p10 = line1 + 3 * x0;
			int p11 = line1 + 3 * x1;

			// Top row, column-wise linear interpolation.
			double v0_r = dx_1 * data[p00] + dx * data[p01];
			double v0_g = dx_1 * data[p00 + 1] + dx * data[p01 + 1];
			double v0_b = dx_1 * data[p00 + 2] + dx * data[p01 + 2];

			// Bottom row, column-wise linear interpolation.
			double v1_r = dx_1 * data[p10] + dx * data[p11];
			double v1_g = dx_1 * data[p10 + 1] + dx * data[p11 + 1];
			double v1_b = dx_1 * data[p10 + 2] + dx * data[p11 + 2];

			// Row-wise linear interpolation.
			return new BitmapColor<float>(
				(float)(dy_1 * v0_r + dy * v1_r),
				(float)(dy_1 * v0_g + dy * v1_g),
				(float)(dy_1 * v0_b + dy * v1_b));
		}

		return null;
	}

	/// <summary>
	/// Port of Bitmap::Rot90: rotate image by k * 90 degrees counter-clockwise (k may be
	/// negative). COLMAP delegates to OIIO's rotate90/180/270, which are exact pixel moves.
	/// </summary>
	public void Rot90(int k)
	{
		if (IsEmpty)
		{
			return;
		}
		k %= 4;
		if (k < 0)
		{
			k += 4;
		}
		if (k == 0)
		{
			return;
		}

		bool swapDims = k == 1 || k == 3;
		int newWidth = swapDims ? height : width;
		int newHeight = swapDims ? width : height;
		var newData = new byte[newWidth * newHeight * channels];

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// Where source pixel (x, y) lands after the counter-clockwise rotation.
				int nx, ny;
				if (k == 1)
				{
					nx = y;
					ny = width - 1 - x;
				}
				else if (k == 2)
				{
					nx = width - 1 - x;
					ny = height - 1 - y;
				}
				else
				{
					nx = height - 1 - y;
					ny = x;
				}
				Array.Copy(data, (y * width + x) * channels, newData, (ny * newWidth + nx) * channels, channels);
			}
		}

		width = newWidth;
		height = newHeight;
		data = newData;
		Check.NotNull(metaData);
	}

	/// <summary>Port of Bitmap::Clone (the C++ copy constructor): a deep copy.</summary>
	public Bitmap Clone()
	{
		var cloned = new Bitmap
		{
			width = width,
			height = height,
			channels = channels,
			linearColorspace = linearColorspace,
			data = (byte[])data.Clone(),
		};
		if (!IsEmpty)
		{
			cloned.metaData = Check.NotNull(metaData).Clone();
		}
		return cloned;
	}

	/// <summary>
	/// Port of Bitmap::CloneAsGrey. Uses COLMAP's Rec. 709 luma weights in float and rounds
	/// by adding 0.5 before truncating (the weighted sum is non-negative).
	/// </summary>
	public Bitmap CloneAsGrey()
	{
		if (IsGrey)
		{
			return Clone();
		}

		var cloned = new Bitmap
		{
			width = width,
			height = height,
			channels = 1,
			linearColorspace = linearColorspace,
			data = new byte[width * height],
		};
		for (int i = 0; i < cloned.data.Length; ++i)
		{
			// Each product and sum is rounded to float, as the C++ float expression is.
			cloned.data[i] = (byte)(.2126f * data[3 * i + 0] +
				.7152f * data[3 * i + 1] +
				.0722f * data[3 * i + 2] + .5f);
		}
		cloned.metaData = Check.NotNull(metaData).Clone();
		return cloned;
	}

	/// <summary>Port of Bitmap::CloneAsRGB: replicates the grey value into all channels.</summary>
	public Bitmap CloneAsRGB()
	{
		if (IsRGB)
		{
			return Clone();
		}

		Check.Eq(channels, 1);
		var cloned = new Bitmap
		{
			width = width,
			height = height,
			channels = 3,
			linearColorspace = linearColorspace,
			data = new byte[width * height * 3],
		};
		for (int i = 0; i < data.Length; ++i)
		{
			cloned.data[3 * i + 0] = data[i];
			cloned.data[3 * i + 1] = data[i];
			cloned.data[3 * i + 2] = data[i];
		}
		cloned.metaData = Check.NotNull(metaData).Clone();
		return cloned;
	}

	/// <summary>Port of operator&lt;&lt;(ostream, Bitmap).</summary>
	public override string ToString() =>
		string.Create(CultureInfo.InvariantCulture, $"Bitmap(width={width}, height={height}, channels={channels})");
}
