// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BitmapRescaleOracleTests: C#-only oracle test (bitmap_test.cc pins only Rescale's output
// dimensions). Compares Bitmap.Rescale (ColmapSharp/Sensor/BitmapResize.cs) with pycolmap
// 4.2.0's OpenImageIO resize on the seeded images of oracle/fixture_bitmap_rescale.py.
//
// Tier B: every pixel within one gray level of pycolmap's (docs/CPP_DIVERGENCES.md, entry 9).

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class BitmapRescaleOracleTests
{
	[Test]
	public async Task Rescale_MatchesPycolmapWithinOneGrayLevel()
	{
		var root = OracleFixture.Load("bitmap_rescale.json");
		using (Assert.Multiple())
		{
			foreach (var testCase in root.GetProperty("cases").EnumerateArray())
			{
				int width = testCase.GetProperty("width").GetInt32();
				int height = testCase.GetProperty("height").GetInt32();
				int channels = testCase.GetProperty("channels").GetInt32();
				int newWidth = testCase.GetProperty("new_width").GetInt32();
				int newHeight = testCase.GetProperty("new_height").GetInt32();
				var filter = testCase.GetProperty("filter").GetString() == "box"
					? Bitmap.RescaleFilter.Box
					: Bitmap.RescaleFilter.Bilinear;
				long[] input = OracleFixture.Int64s(testCase.GetProperty("input"));
				long[] expected = OracleFixture.Int64s(testCase.GetProperty("output"));

				var bitmap = new Bitmap(width, height, asRgb: channels == 3);
				for (int i = 0; i < input.Length; i++)
				{
					bitmap.RowMajorData[i] = (byte)input[i];
				}
				bitmap.Rescale(newWidth, newHeight, filter);

				long maxDifference = 0;
				for (int i = 0; i < expected.Length; i++)
				{
					maxDifference = Math.Max(maxDifference, Math.Abs(bitmap.RowMajorData[i] - expected[i]));
				}
				string label = $"{width}x{height}x{channels} -> {newWidth}x{newHeight} {filter}";
				await Assert.That(bitmap.NumBytes).IsEqualTo(expected.Length).Because(label);
				await Assert.That(maxDifference).IsLessThanOrEqualTo(1).Because(label);
			}
		}
	}
}
