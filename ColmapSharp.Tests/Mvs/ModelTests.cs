// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ModelTests: colmap/mvs/model_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/Model.cs. Tier A; EXPECT_FLOAT_EQ is gtest's 4-ulp comparison and
// EXPECT_NEAR on floats compares them promoted to double, as gtest does. The C#-only cases
// at the end pin the in-memory ReadFromCOLMAP overload and the tie order of
// GetMaxOverlappingImages (divergence 64).

using ColmapSharp.Mvs;
using ColmapSharp.Scene;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Mvs.MvsTestUtils;

namespace ColmapSharp.Tests.Mvs;

public class ModelTests
{
	private static readonly float[] K = [100, 0, 50, 0, 100, 50, 0, 0, 1];
	private static readonly float[] R = [1, 0, 0, 0, 1, 0, 0, 0, 1];

	private static Model.Point NewPoint(float x, float y, float z, params int[] track) =>
		new() { X = x, Y = y, Z = z, Track = [.. track] };

	[Test]
	public async Task Model_ReadCOLMAP()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 3,
			NumPoints3D = 10,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		string testDir = CreateTestDir();
		string sparsePath = Path.Combine(testDir, "sparse");
		Directory.CreateDirectory(sparsePath);
		reconstruction.WriteBinary(sparsePath);

		var model = new Model();
		model.ReadFromCOLMAP(testDir);
		await Assert.That(model.Images.Count).IsEqualTo(reconstruction.NumRegImages);
		await Assert.That(model.Points.Count).IsEqualTo(reconstruction.NumPoints3D);

		var modelLower = new Model();
		modelLower.Read(testDir, "colmap");
		await Assert.That(modelLower.Images.Count).IsEqualTo(reconstruction.NumRegImages);
		await Assert.That(modelLower.Points.Count).IsEqualTo(reconstruction.NumPoints3D);

		// Verify case insensitivity.
		var modelUpper = new Model();
		modelUpper.Read(testDir, "COLMAP");
		await Assert.That(modelUpper.Images.Count).IsEqualTo(reconstruction.NumRegImages);
		await Assert.That(modelUpper.Points.Count).IsEqualTo(reconstruction.NumPoints3D);

		string name = model.GetImageName(0);
		await Assert.That(name).IsNotEmpty();
		await Assert.That(model.GetImageIdx(name)).IsEqualTo(0);

		await Assert.That(() => model.GetImageIdx("nonexistent")).Throws<Exception>();
		await Assert.That(() => model.GetImageName(-1)).Throws<Exception>();
		await Assert.That(() => model.GetImageName(model.Images.Count)).Throws<Exception>();
	}

	[Test]
	public async Task Model_ComputeSharedPoints()
	{
		var model = new Model();
		// Projection center = -R^T * T = -T (for R=I).
		float[] t1 = [0, 0, 0];
		float[] t2 = [-1, 0, 0];
		float[] t3 = [-2, 0, 0];
		model.Images.Add(new Image("img0.jpg", 100, 100, K, R, t1));
		model.Images.Add(new Image("img1.jpg", 100, 100, K, R, t2));
		model.Images.Add(new Image("img2.jpg", 100, 100, K, R, t3));
		model.Points.Add(NewPoint(5.0f, 0.0f, 10.0f, 0, 1));

		// Point seen by images 0, 1, and 2.
		model.Points.Add(NewPoint(6.0f, 0.0f, 10.0f, 0, 1, 2));

		List<SortedDictionary<int, int>> shared = model.ComputeSharedPoints();
		await Assert.That(shared.Count).IsEqualTo(3);
		await Assert.That(shared[0][1]).IsEqualTo(2); // images 0,1 share 2 points
		await Assert.That(shared[1][0]).IsEqualTo(2);
		await Assert.That(shared[0][2]).IsEqualTo(1); // images 0,2 share 1 point
		await Assert.That(shared[2][0]).IsEqualTo(1);
		await Assert.That(shared[1][2]).IsEqualTo(1); // images 1,2 share 1 point
		await Assert.That(shared[2][1]).IsEqualTo(1);
	}

	[Test]
	public async Task Model_ComputeDepthRanges()
	{
		var model = new Model();
		float[] t = [0, 0, 0];
		model.Images.Add(new Image("img0.jpg", 100, 100, K, R, t));

		for (int i = 1; i <= 100; ++i)
		{
			model.Points.Add(NewPoint(0.0f, 0.0f, i, 0));
		}

		List<(float Min, float Max)> depthRanges = model.ComputeDepthRanges();
		await Assert.That(depthRanges.Count).IsEqualTo(1);

		// The points span depths in the range [1..100] and
		// the range is computed at percentiles 1% and 99%
		// with some additional padding.
		await Assert.That(FloatEq(depthRanges[0].Min, 1.5f)).IsTrue();
		await Assert.That(FloatEq(depthRanges[0].Max, 125.0f)).IsTrue();
	}

	[Test]
	public async Task Model_ComputeDepthRangesNoPoints()
	{
		var model = new Model();
		float[] t = [0, 0, 0];
		model.Images.Add(new Image("img0.jpg", 100, 100, K, R, t));

		List<(float Min, float Max)> depthRanges = model.ComputeDepthRanges();
		await Assert.That(depthRanges.Count).IsEqualTo(1);
		await Assert.That(FloatEq(depthRanges[0].Min, -1.0f)).IsTrue();
		await Assert.That(FloatEq(depthRanges[0].Max, -1.0f)).IsTrue();
	}

	[Test]
	public async Task Model_ComputeTriangulationAngles()
	{
		var model = new Model();
		// Projection center = -R^T * T = -T (for R=I).
		// Place cameras at (0,0,0) and (1,0,0).
		float[] t1 = [0, 0, 0];
		float[] t2 = [-1, 0, 0];
		model.Images.Add(new Image("img0.jpg", 100, 100, K, R, t1));
		model.Images.Add(new Image("img1.jpg", 100, 100, K, R, t2));

		// Point at midpoint between cameras, at depth 10.
		model.Points.Add(NewPoint(0.5f, 0.0f, 10.0f, 0, 1));

		List<SortedDictionary<int, float>> angles = model.ComputeTriangulationAngles(50);
		await Assert.That(angles.Count).IsEqualTo(2);
		float expectedAngle = 2.0f * MathF.Atan(0.5f / 10.0f);
		await Assert.That((double)angles[0][1]).IsEqualTo(expectedAngle).Within(1e-5f);
		// Angles should be symmetric.
		await Assert.That(FloatEq(angles[0][1], angles[1][0])).IsTrue();
	}

	[Test]
	public async Task Model_GetMaxOverlappingImages()
	{
		var model = new Model();
		// Projection center = -R^T * T = -T (for R=I).
		float[] t1 = [0, 0, 0];
		float[] t2 = [-1, 0, 0];
		float[] t3 = [-2, 0, 0];
		model.Images.Add(new Image("img0.jpg", 100, 100, K, R, t1));
		model.Images.Add(new Image("img1.jpg", 100, 100, K, R, t2));
		model.Images.Add(new Image("img2.jpg", 100, 100, K, R, t3));
		for (int i = 0; i < 10; ++i)
		{
			model.Points.Add(NewPoint(0.5f, 0.0f, 5.0f + i, 0, 1));
		}

		model.Points.Add(NewPoint(1.0f, 0.0f, 10.0f, 0, 2));

		List<List<int>> overlapping = model.GetMaxOverlappingImages(2, 0.0);
		await Assert.That(overlapping.Count).IsEqualTo(3);
		// Image 0 should have image 1 as top overlap.
		await Assert.That(overlapping[0]).IsNotEmpty();
		await Assert.That(overlapping[0][0]).IsEqualTo(1);
	}

	// C#-only: the in-memory overload gives the same model as reading the files, with
	// images in registration order and paths under the given images directory.
	[Test]
	public async Task Model_ReadCOLMAPInMemory()
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 3, NumPoints3D = 10 },
			reconstruction);

		var model = new Model();
		model.ReadFromCOLMAP(reconstruction, "imgs");
		await Assert.That(model.Images.Count).IsEqualTo(reconstruction.NumRegImages);
		await Assert.That(model.Points.Count).IsEqualTo(reconstruction.NumPoints3D);

		List<uint> regImageIds = reconstruction.RegImageIds();
		for (int i = 0; i < regImageIds.Count; i++)
		{
			string name = reconstruction.Image(regImageIds[i]).Name;
			await Assert.That(model.GetImageName(i)).IsEqualTo(name);
			await Assert.That(model.Images[i].GetPath()).IsEqualTo(Path.Combine("imgs", name));
		}

		int point = 0;
		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			await Assert.That(model.Points[point].X).IsEqualTo((float)point3D.Xyz.X);
			await Assert.That(model.Points[point].Track.Count).IsEqualTo(point3D.Track.Length);
			point++;
		}
	}

	// C#-only: images with equal shared-point counts come out in ascending index order, and
	// with fewer slots than tied candidates the cut-off keeps the lowest indices
	// (divergence 64).
	[Test]
	public async Task Model_GetMaxOverlappingImagesTies()
	{
		var model = new Model();
		for (int i = 0; i < 4; i++)
		{
			float[] t = [-i, 0, 0];
			model.Images.Add(new Image($"img{i}.jpg", 100, 100, K, R, t));
		}

		model.Points.Add(NewPoint(1.0f, 0.0f, 10.0f, 3, 0, 2, 1));

		// Every pair shares exactly one point, so each image has three tied candidates.
		List<List<int>> overlapping = model.GetMaxOverlappingImages(2, 0.0);
		await Assert.That(overlapping[0]).IsEquivalentTo(new List<int> { 1, 2 }, CollectionOrdering.Matching);
		await Assert.That(overlapping[3]).IsEquivalentTo(new List<int> { 0, 1 }, CollectionOrdering.Matching);
	}

	private static string WriteBundlerPmvs(MvsTestUtils.InMemoryBitmapSource bitmaps, int trackImageIdx)
	{
		string dir = CreateTestDir();
		File.WriteAllText(
			Path.Combine(dir, "bundle.rd.out"),
			"# Bundle file v0.3\n" +
			"1 1\n" +
			"500 0 0\n" +
			"1 0 0\n0 1 0\n0 0 1\n" +
			"0 0 0\n" +
			"0 0 -5\n" +
			"255 255 255\n" +
			$"1 {trackImageIdx} 0 1.5 2.5\n");
		bitmaps.Add(Path.Combine(dir, "visualize", "00000000.jpg"), new ColmapSharp.Sensor.Bitmap(40, 30, asRgb: true));
		return dir;
	}

	// C#-only: a Bundler PMVS track index must be a valid image index; COLMAP's
	// THROW_CHECK_LT(int, size_t) rejects a negative one through the unsigned conversion.
	[Test]
	public async Task Model_ReadBundlerPMVSNegativeTrackIndex()
	{
		var bitmaps = new MvsTestUtils.InMemoryBitmapSource();

		var valid = new Model();
		valid.ReadFromPMVS(WriteBundlerPmvs(bitmaps, 0), bitmaps);
		await Assert.That(valid.Images.Count).IsEqualTo(1);
		await Assert.That(valid.Images[0].GetWidth()).IsEqualTo(40);
		await Assert.That(valid.Points[0].Track).IsEquivalentTo(new List<int> { 0 }, CollectionOrdering.Matching);

		string negativeDir = WriteBundlerPmvs(bitmaps, -1);
		await Assert.That(() => new Model().ReadFromPMVS(negativeDir, bitmaps)).Throws<ArgumentException>();
	}
}
