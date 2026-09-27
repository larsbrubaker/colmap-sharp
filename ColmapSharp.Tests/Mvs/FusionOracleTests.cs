// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FusionOracleTests (C#-only, not a port of a COLMAP test): runs ColmapSharp/Mvs/Fusion.cs on
// the workspaces in TestData/oracle/stereo_fusion and stereo_fusion_options
// (oracle/fixture_stereo_fusion.py) and compares it with the fused.ply and fused.ply.vis
// pycolmap 4.2.0's stereo_fusion wrote there with num_threads = 1. Tier C: the same point
// count and order, colors exact, positions and normals within float rounding of the
// projection products (docs/CPP_DIVERGENCES.md, entry 63), visibility the same image sets
// (the order within a set is entry 88). The options workspace adds noisy normals, a bounding
// box, max_image_size, short traversals and masks (see the fixture script).
//
// The images and masks are not checked in; they are the fixture script's closed-form
// patterns (ImageColor, MaskRows/MaskCols), served from memory as a host would after decoding
// the PNGs.

using System.Text.Json;

using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class FusionOracleTests
{
	// Measured: positions (magnitudes up to ~5) differ by under 1e-6 (a couple of ulps) and
	// normal components by under 1.2e-7; the bounds leave 10x room for FMA contraction on the
	// C++ side (entry 63).
	private const float PositionTolerance = 1e-5f;
	private const float NormalTolerance = 1e-6f;

	private static BitmapColor<byte> ImageColor(int imageIdx, int x, int y) => new(
		(byte)((5 * x + 3 * y + 17 * imageIdx) % 256),
		(byte)((11 * x + 7 * y) % 256),
		(byte)((x + 13 * y + 29 * imageIdx) % 256));

	// fixture_stereo_fusion.py mask_rows/mask_cols: the masked (zero) block of each mask.
	private static bool IsMasked(int imageIdx, int x, int y) =>
		y >= 2 * imageIdx + 16 && y < 2 * imageIdx + 24 && x >= 4 * imageIdx + 20 && x < 4 * imageIdx + 30;

	private static MvsTestUtils.InMemoryBitmapSource FixtureBitmaps(string workspacePath, string? maskPath = null)
	{
		var reconstruction = new Reconstruction();
		reconstruction.Read(Path.Combine(workspacePath, "sparse"));
		var bitmaps = new MvsTestUtils.InMemoryBitmapSource();
		int imageIdx = 0;
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			ColmapSharp.Scene.Image image = reconstruction.Image(imageId);
			Camera camera = image.CameraPtr;
			var bitmap = new Bitmap(camera.Width, camera.Height, asRgb: true);
			var mask = new Bitmap(camera.Width, camera.Height, asRgb: false);
			for (int y = 0; y < camera.Height; y++)
			{
				for (int x = 0; x < camera.Width; x++)
				{
					bitmap.SetPixel(x, y, ImageColor(imageIdx, x, y));
					mask.SetPixel(x, y, new BitmapColor<byte>(IsMasked(imageIdx, x, y) ? (byte)0 : (byte)255));
				}
			}

			bitmaps.Add(Path.Combine(workspacePath, "images", image.Name), bitmap);
			if (maskPath != null)
			{
				bitmaps.Add(Path.Combine(maskPath, image.Name + ".png"), mask);
			}

			imageIdx++;
		}

		return bitmaps;
	}

	private static async Task AssertMatchesOracle(StereoFusion fusion, string workspacePath)
	{
		List<PlyPoint> expected = Ply.ReadPly(Path.Combine(workspacePath, "fused.ply"));
		List<List<int>> expectedVisibility = StereoFusion.ReadPointsVisibility(Path.Combine(workspacePath, "fused.ply.vis"), expected.Count);

		IReadOnlyList<PlyPoint> actual = fusion.GetFusedPoints();
		IReadOnlyList<List<int>> actualVisibility = fusion.GetFusedPointsVisibility();
		await Assert.That(actual.Count).IsEqualTo(expected.Count);
		await Assert.That(actual.Count).IsGreaterThan(100);
		for (int i = 0; i < expected.Count; i++)
		{
			PlyPoint e = expected[i];
			PlyPoint a = actual[i];
			await Assert.That(Math.Abs(a.X - e.X)).IsLessThanOrEqualTo(PositionTolerance);
			await Assert.That(Math.Abs(a.Y - e.Y)).IsLessThanOrEqualTo(PositionTolerance);
			await Assert.That(Math.Abs(a.Z - e.Z)).IsLessThanOrEqualTo(PositionTolerance);
			await Assert.That(Math.Abs(a.Nx - e.Nx)).IsLessThanOrEqualTo(NormalTolerance);
			await Assert.That(Math.Abs(a.Ny - e.Ny)).IsLessThanOrEqualTo(NormalTolerance);
			await Assert.That(Math.Abs(a.Nz - e.Nz)).IsLessThanOrEqualTo(NormalTolerance);
			await Assert.That(a.R).IsEqualTo(e.R);
			await Assert.That(a.G).IsEqualTo(e.G);
			await Assert.That(a.B).IsEqualTo(e.B);
			await Assert.That(actualVisibility[i].SequenceEqual(expectedVisibility[i].Order())).IsTrue();
		}
	}

	private static StereoFusionOptions OptionsFixtureOptions(string workspacePath)
	{
		using JsonDocument json = JsonDocument.Parse(File.ReadAllText(Path.Combine(workspacePath, "options.json")));
		JsonElement root = json.RootElement;
		static (float, float, float) Box(JsonElement e) =>
			(e[0].GetSingle(), e[1].GetSingle(), e[2].GetSingle());
		return new StereoFusionOptions
		{
			NumThreads = 1,
			CheckNumImages = 10,
			MinNumPixels = root.GetProperty("min_num_pixels").GetInt32(),
			MaxNumPixels = root.GetProperty("max_num_pixels").GetInt32(),
			MaxTraversalDepth = root.GetProperty("max_traversal_depth").GetInt32(),
			MaxImageSize = root.GetProperty("max_image_size").GetInt32(),
			BoundingBoxMin = Box(root.GetProperty("bounding_box_min")),
			BoundingBoxMax = Box(root.GetProperty("bounding_box_max")),
			MaskPath = Path.Combine(workspacePath, "masks"),
		};
	}

	[Test]
	public async Task StereoFusion_MatchesPycolmap()
	{
		string workspacePath = OracleFixture.PathOf("stereo_fusion");
		var options = new StereoFusionOptions { NumThreads = 1, MinNumPixels = 3, CheckNumImages = 10 };
		var fusion = new StereoFusion(options, workspacePath, "COLMAP", "", "geometric", FixtureBitmaps(workspacePath));
		var reports = new List<StereoFusionProgress>();
		fusion.Run(progress: new SynchronousProgress(reports));

		await AssertMatchesOracle(fusion, workspacePath);

		// One report per fused image, the last with the final point count.
		await Assert.That(reports.Count).IsEqualTo(3);
		await Assert.That(reports[^1]).IsEqualTo(new StereoFusionProgress(3, 3, fusion.GetFusedPoints().Count));
	}

	[Test]
	public async Task StereoFusion_MatchesPycolmapWithOptions()
	{
		string workspacePath = OracleFixture.PathOf("stereo_fusion_options");
		StereoFusionOptions options = OptionsFixtureOptions(workspacePath);
		var fusion = new StereoFusion(options, workspacePath, "COLMAP", "", "geometric", FixtureBitmaps(workspacePath, options.MaskPath));
		fusion.Run();

		await AssertMatchesOracle(fusion, workspacePath);

		// The bounding box really cuts the plane: every point is inside it. (The masks, short
		// traversals and normal check are pinned by the count and order matching pycolmap's.)
		foreach (PlyPoint point in fusion.GetFusedPoints())
		{
			await Assert.That(point.X).IsLessThanOrEqualTo(options.BoundingBoxMax.X);
		}
	}

	[Test]
	public async Task StereoFusion_CachedWorkspaceMatchesLoaded()
	{
		// C#-only: the cached and pre-loaded workspaces give identical output, and the loaded
		// one runs with all threads for loading (entry 87: the fusion itself is sequential).
		string workspacePath = OracleFixture.PathOf("stereo_fusion");
		MvsTestUtils.InMemoryBitmapSource bitmaps = FixtureBitmaps(workspacePath);
		var loaded = new StereoFusion(new StereoFusionOptions { MinNumPixels = 3, CheckNumImages = 10 }, workspacePath, "COLMAP", "", "geometric", bitmaps);
		loaded.Run();
		var cached = new StereoFusion(new StereoFusionOptions { MinNumPixels = 3, CheckNumImages = 10, UseCache = true }, workspacePath, "COLMAP", "", "geometric", bitmaps);
		cached.Run();

		await Assert.That(cached.GetFusedPoints().SequenceEqual(loaded.GetFusedPoints())).IsTrue();
		await Assert.That(cached.GetFusedPointsVisibility().Count).IsEqualTo(loaded.GetFusedPointsVisibility().Count);
		for (int i = 0; i < loaded.GetFusedPointsVisibility().Count; i++)
		{
			await Assert.That(cached.GetFusedPointsVisibility()[i].SequenceEqual(loaded.GetFusedPointsVisibility()[i])).IsTrue();
		}
	}

	[Test]
	public async Task CSharpOnly_FusedOutputIsBitIdenticalForAnyThreadCount()
	{
		// C#-only: pins the exact bits of the fused output on both fixtures (positions,
		// normals, colors, visibility), so that parallelizing fusion (entry 87) can never
		// change a result. The hashes are the output of the one-thread traversal. Each fixture
		// runs with one thread, with four, with all cores, and through the cached workspace.
		foreach ((string name, string expectedHash) in new[]
		{
			("stereo_fusion", "EB769835871BF9D42B521EFD3837CA1D9CF9BD8F5A0A3505BCC78E5C223A0EDA"),
			("stereo_fusion_options", "840475B5A09FD93614B56F0D4616B998470B8812A83ADC974460F321F828A39B"),
		})
		{
			string workspacePath = OracleFixture.PathOf(name);
			foreach ((int numThreads, bool useCache) in new[] { (1, false), (4, false), (-1, false), (1, true) })
			{
				StereoFusionOptions options = name == "stereo_fusion"
					? new StereoFusionOptions { MinNumPixels = 3, CheckNumImages = 10 }
					: OptionsFixtureOptions(workspacePath);
				options.NumThreads = numThreads;
				options.UseCache = useCache;
				var fusion = new StereoFusion(
					options, workspacePath, "COLMAP", "", "geometric",
					FixtureBitmaps(workspacePath, name == "stereo_fusion" ? null : options.MaskPath));
				fusion.Run();

				await Assert.That(FusedOutputHash(fusion)).IsEqualTo(expectedHash);
			}
		}
	}

	private static string FusedOutputHash(StereoFusion fusion)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		foreach (PlyPoint p in fusion.GetFusedPoints())
		{
			writer.Write(p.X);
			writer.Write(p.Y);
			writer.Write(p.Z);
			writer.Write(p.Nx);
			writer.Write(p.Ny);
			writer.Write(p.Nz);
			writer.Write(p.R);
			writer.Write(p.G);
			writer.Write(p.B);
		}

		foreach (List<int> visibility in fusion.GetFusedPointsVisibility())
		{
			writer.Write(visibility.Count);
			foreach (int imageIdx in visibility)
			{
				writer.Write(imageIdx);
			}
		}

		writer.Flush();
		return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream.ToArray()));
	}

	// Progress<T> posts to the thread pool; this records the reports in order on the caller.
	private sealed class SynchronousProgress(List<StereoFusionProgress> reports) : IProgress<StereoFusionProgress>
	{
		public void Report(StereoFusionProgress value) => reports.Add(value);
	}
}
