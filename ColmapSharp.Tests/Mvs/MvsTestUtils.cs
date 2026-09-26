// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MvsTestUtils: test infrastructure shared by the Mvs test files (C#-only, not a port):
// gtest's EXPECT_FLOAT_EQ comparison, COLMAP's CreateTestDir, and an in-memory
// IBitmapSource standing in for the image files COLMAP's tests write with Bitmap::Write
// (the library does no image decoding; the host supplies bitmaps).

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

namespace ColmapSharp.Tests.Mvs;

internal static class MvsTestUtils
{
	/// <summary>
	/// gtest's FloatingPoint&lt;float&gt;::AlmostEquals (EXPECT_FLOAT_EQ): within 4 ulps,
	/// comparing the sign-and-magnitude bit patterns as biased integers; NaN never matches.
	/// </summary>
	public static bool FloatEq(float lhs, float rhs)
	{
		if (float.IsNaN(lhs) || float.IsNaN(rhs))
		{
			return false;
		}

		return DistanceBetweenSignAndMagnitudeNumbers(BitConverter.SingleToUInt32Bits(lhs), BitConverter.SingleToUInt32Bits(rhs)) <= 4;
	}

	private static uint SignAndMagnitudeToBiased(uint sam)
	{
		const uint SignBitMask = 0x80000000u;
		return (sam & SignBitMask) != 0 ? ~sam + 1 : SignBitMask | sam;
	}

	private static uint DistanceBetweenSignAndMagnitudeNumbers(uint sam1, uint sam2)
	{
		uint biased1 = SignAndMagnitudeToBiased(sam1);
		uint biased2 = SignAndMagnitudeToBiased(sam2);
		return biased1 >= biased2 ? biased1 - biased2 : biased2 - biased1;
	}

	/// <summary>A fresh, empty temporary directory (colmap::CreateTestDir).</summary>
	public static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-mvs-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	/// <summary>Bitmaps served from memory by path, as a host would after decoding the files.</summary>
	public sealed class InMemoryBitmapSource : IBitmapSource
	{
		private readonly Dictionary<string, Bitmap> bitmaps = new();

		public void Add(string path, Bitmap bitmap) => bitmaps[path] = bitmap;

		public bool Exists(string path) => bitmaps.ContainsKey(path);

		public Bitmap Read(string path, bool asRgb)
		{
			Bitmap bitmap = bitmaps[path];
			return asRgb ? bitmap.CloneAsRGB() : bitmap.CloneAsGrey();
		}
	}
}
