// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FusionTests: colmap/mvs/fusion_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/Fusion.cs. Tier C (outcome), with COLMAP's own bounds.
//
// The fixture writes the images with Bitmap::Write in COLMAP; the library does no image
// decoding, so the same bitmaps are served by an in-memory IBitmapSource under the same
// paths (images/<name>). The sparse model, depth maps, normal maps and fusion.cfg are written
// to disk as in COLMAP. COLMAP's SetCheckIfStoppedFunc returning true becomes an already
// cancelled CancellationToken. CSharpOnly_UnreadableMaskThrows is C#-only (divergence 89).
// The oracle comparison against pycolmap.stereo_fusion is in FusionOracleTests.cs.

using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Mvs.MvsTestUtils;

namespace ColmapSharp.Tests.Mvs;

public class FusionTests
{
	[Test]
	public async Task StereoFusion_Integration()
	{
		string tempDir = CreateTestDir();
		Directory.CreateDirectory(Path.Combine(tempDir, "sparse"));
		Directory.CreateDirectory(Path.Combine(tempDir, "images"));
		Directory.CreateDirectory(Path.Combine(tempDir, "stereo"));
		Directory.CreateDirectory(Path.Combine(tempDir, "stereo", "depth_maps"));
		Directory.CreateDirectory(Path.Combine(tempDir, "stereo", "normal_maps"));

		// Create synthetic reconstruction with 2 overlapping images.
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			CameraWidth = 30,
			CameraHeight = 20,
		};
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		reconstruction.Write(Path.Combine(tempDir, "sparse"));

		// Create depth maps, normal maps, and consistency graphs for both images.
		var bitmaps = new InMemoryBitmapSource();
		var imageNames = new List<string>();
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			imageNames.Add(image.Name);

			// Create depth map with constant depth.
			var depthMap = new Mat<float>(syntheticDatasetOptions.CameraWidth, syntheticDatasetOptions.CameraHeight, 1);
			depthMap.Fill(5.0f);
			depthMap.Write(Path.Combine(tempDir, "stereo", "depth_maps", image.Name + ".geometric.bin"));

			// Create normal map pointing in z direction.
			var normalMap = new Mat<float>(syntheticDatasetOptions.CameraWidth, syntheticDatasetOptions.CameraHeight, 3);
			int numPixels = normalMap.GetHeight() * normalMap.GetWidth();
			for (int i = 0; i < numPixels; ++i)
			{
				// Same raw writes as the C++ GetPtr()[3 * i + c]. Mat stores channel planes, so
				// this interleaved pattern does not give every pixel (0, 0, 1): with 600 pixels
				// per plane, pixel j gets (1, 1, 1) if j % 3 == 2 and (0, 0, 0) otherwise. Kept
				// as COLMAP writes it; fused normals are normalized and zero ones dropped.
				normalMap.Data[3 * i + 0] = 0.0f; // nx
				normalMap.Data[3 * i + 1] = 0.0f; // ny
				normalMap.Data[3 * i + 2] = 1.0f; // nz
			}

			normalMap.Write(Path.Combine(tempDir, "stereo", "normal_maps", image.Name + ".geometric.bin"));

			// Create bitmap.
			var bitmap = new Bitmap(syntheticDatasetOptions.CameraWidth, syntheticDatasetOptions.CameraHeight, asRgb: true);
			bitmap.Fill(new BitmapColor<byte>(0, 64, 128));
			bitmaps.Add(Path.Combine(tempDir, "images", image.Name), bitmap);
		}

		// Write fusion config
		File.WriteAllText(Path.Combine(tempDir, "stereo", "fusion.cfg"), string.Concat(imageNames.Select(name => name + "\n")));

		// Run fusion
		var options = new StereoFusionOptions
		{
			MinNumPixels = 1,
			MaxNumPixels = 100,
			MaxTraversalDepth = 10,
			CheckNumImages = 10,
			UseCache = false,
		};

		var fusion = new StereoFusion(options, tempDir, "COLMAP", "", "geometric", bitmaps);
		fusion.Run();

		// Verify that some points were fused
		IReadOnlyList<PlyPoint> fusedPoints = fusion.GetFusedPoints();
		IReadOnlyList<List<int>> visibility = fusion.GetFusedPointsVisibility();

		await Assert.That(fusedPoints.Count).IsGreaterThan(0);
		await Assert.That(fusedPoints.Count).IsEqualTo(visibility.Count);

		foreach (PlyPoint point in fusedPoints)
		{
			await Assert.That(point.X).IsGreaterThan(-10.0f);
			await Assert.That(point.X).IsLessThan(10.0f);
			await Assert.That(point.Y).IsGreaterThan(-10.0f);
			await Assert.That(point.Y).IsLessThan(10.0f);
			await Assert.That(point.Z).IsGreaterThan(-10.0f);
			await Assert.That(point.Z).IsLessThan(10.0f);
			await Assert.That(point.R).IsEqualTo((byte)0);
			await Assert.That(point.G).IsEqualTo((byte)64);
			await Assert.That(point.B).IsEqualTo((byte)128);
			await Assert.That(FloatEq(point.Nx * point.Nx + point.Ny * point.Ny + point.Nz * point.Nz, 1.0f)).IsTrue();
		}

		foreach (List<int> vis in visibility)
		{
			await Assert.That(vis.Count).IsGreaterThan(0);
		}

		using var stop = new CancellationTokenSource();
		stop.Cancel();
		var cancelledFusion = new StereoFusion(options, tempDir, "COLMAP", "", "geometric", bitmaps);
		cancelledFusion.Run(stop.Token);
		await Assert.That(cancelledFusion.GetFusedPoints().Count).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_UnreadableMaskThrows()
	{
		// C#-only (divergence 89): a mask the source reports but cannot
		// decode fails the run instead of silently fusing the image unmasked.
		string workspacePath = OracleFixture.PathOf("stereo_fusion");
		string maskPath = Path.Combine(workspacePath, "masks");
		var source = new UnreadableMaskSource(maskPath);
		var options = new StereoFusionOptions { MaskPath = maskPath };
		var fusion = new StereoFusion(options, workspacePath, "COLMAP", "", "geometric", source);

		await Assert.That(() => fusion.Run()).Throws<InvalidDataException>();
		await Assert.That(source.MaskReads).IsEqualTo(1);
	}

	/// <summary>Serves a plain 40 x 30 image for every image path; every mask exists but fails to decode.</summary>
	private sealed class UnreadableMaskSource(string maskDir) : IBitmapSource
	{
		private int maskReads;

		public int MaskReads => maskReads;

		public bool Exists(string path) => true;

		public Bitmap Read(string path, bool asRgb)
		{
			if (path.StartsWith(maskDir, StringComparison.Ordinal))
			{
				Interlocked.Increment(ref maskReads);
				throw new InvalidDataException($"cannot decode {path}");
			}

			return new Bitmap(40, 30, asRgb);
		}
	}

	[Test]
	public async Task ReadPointsVisibility_RoundTrip()
	{
		string testDir = CreateTestDir();
		string visPath = Path.Combine(testDir, "test.vis");

		List<List<int>> expected =
		[
			[0, 1, 2],
			[1, 3],
			[],
			[0, 2, 3, 4],
		];
		StereoFusion.WritePointsVisibility(visPath, expected);

		List<List<int>> actual = StereoFusion.ReadPointsVisibility(visPath, expected.Count);

		await Assert.That(actual.Count).IsEqualTo(expected.Count);
		for (int i = 0; i < expected.Count; ++i)
		{
			await Assert.That(actual[i].Count).IsEqualTo(expected[i].Count);
			for (int j = 0; j < expected[i].Count; ++j)
			{
				await Assert.That(actual[i][j]).IsEqualTo(expected[i][j]);
			}
		}
	}

	[Test]
	public async Task ReadPointsVisibility_SizeMismatch()
	{
		string testDir = CreateTestDir();
		string visPath = Path.Combine(testDir, "test.vis");

		List<List<int>> data = [[0, 1], [2]];
		StereoFusion.WritePointsVisibility(visPath, data);

		await Assert.That(() => StereoFusion.ReadPointsVisibility(visPath, 5)).Throws<ArgumentException>();
	}
}
