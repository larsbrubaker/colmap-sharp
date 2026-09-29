// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TriangleBvhTests: C#-only tests (not ports; COLMAP has no test for its CGAL occlusion
// tree). They pin ColmapSharp/Mvs/TriangleBvh.cs, the CGAL AABB-tree replacement
// (divergence 90), and the texture mapping properties this port adds:
// occlusion is applied, and results are identical for any thread count.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class TriangleBvhTests
{
	// One triangle in the z = 0 plane: (0,0,0), (1,0,0), (0,1,0).
	private static TriangleBvh UnitTriangle(int id = 7) =>
		new([0, 0, 0, 1, 0, 0, 0, 1, 0], [id]);

	[Test]
	public async Task AnyHit_SegmentThroughTriangle()
	{
		TriangleBvh bvh = UnitTriangle();
		await Assert.That(bvh.AnyHit(0.25f, 0.25f, 1, 0.25f, 0.25f, -1, -1, 10)).IsTrue();

		// Stops short of the plane.
		await Assert.That(bvh.AnyHit(0.25f, 0.25f, 1, 0.25f, 0.25f, 0.5f, -1, 10)).IsFalse();

		// Passes beside the triangle.
		await Assert.That(bvh.AnyHit(0.75f, 0.75f, 1, 0.75f, 0.75f, -1, -1, 10)).IsFalse();
	}

	[Test]
	public async Task AnyHit_ExcludedIdAndDistance()
	{
		TriangleBvh bvh = UnitTriangle(7);
		await Assert.That(bvh.AnyHit(0.25f, 0.25f, 1, 0.25f, 0.25f, -1, 7, 10)).IsFalse();

		// The hit is at distance 1 from the origin: it counts only below maxHitDistance.
		await Assert.That(bvh.AnyHit(0.25f, 0.25f, 1, 0.25f, 0.25f, -1, -1, 1.0f)).IsFalse();
		await Assert.That(bvh.AnyHit(0.25f, 0.25f, 1, 0.25f, 0.25f, -1, -1, 1.001f)).IsTrue();
	}

	[Test]
	public async Task AnyHit_EdgesCountAndParallelSegmentsDoNot()
	{
		TriangleBvh bvh = UnitTriangle();

		// Through a corner and through the middle of the hypotenuse.
		await Assert.That(bvh.AnyHit(0, 0, 1, 0, 0, -1, -1, 10)).IsTrue();
		await Assert.That(bvh.AnyHit(0.5f, 0.5f, 1, 0.5f, 0.5f, -1, -1, 10)).IsTrue();

		// In the triangle's plane: CGAL reports a segment, which COLMAP does not count.
		await Assert.That(bvh.AnyHit(-1, 0.25f, 0, 2, 0.25f, 0, -1, 10)).IsFalse();
	}

	[Test]
	public async Task AnyHit_EmptyTree()
	{
		var bvh = new TriangleBvh([], []);
		await Assert.That(bvh.Count).IsEqualTo(0);
		await Assert.That(bvh.AnyHit(0, 0, 1, 0, 0, -1, -1, 10)).IsFalse();

		// A general direction (nonzero x and y) once walked past the empty root.
		await Assert.That(bvh.AnyHit(0, 0, 1, 0.5f, 0.3f, -1, -1, 10)).IsFalse();
	}

	[Test]
	public async Task AnyHit_MatchesBruteForceOnRandomTriangles()
	{
		// Many small random triangles (exercising SAH splits, deep trees and leaves) against
		// random segments; the reference answer ORs one single-triangle tree per triangle.
		var random = new Random(42);
		float Uniform(float lo, float hi) => lo + (hi - lo) * (float)random.NextDouble();
		const int numTriangles = 3000;
		var corners = new float[9 * numTriangles];
		var ids = new int[numTriangles];
		var singles = new TriangleBvh[numTriangles];
		for (int i = 0; i < numTriangles; i++)
		{
			float cx = Uniform(-5.0f, 5.0f);
			float cy = Uniform(-5.0f, 5.0f);
			float cz = Uniform(-5.0f, 5.0f);
			for (int k = 0; k < 3; k++)
			{
				corners[9 * i + 3 * k] = cx + Uniform(-0.3f, 0.3f);
				corners[9 * i + 3 * k + 1] = cy + Uniform(-0.3f, 0.3f);
				corners[9 * i + 3 * k + 2] = cz + Uniform(-0.3f, 0.3f);
			}

			ids[i] = i;
			singles[i] = new TriangleBvh(corners.AsSpan(9 * i, 9), [i]);
		}

		var bvh = new TriangleBvh(corners, ids);
		int hits = 0;
		for (int q = 0; q < 300; q++)
		{
			float ox = Uniform(-6.0f, 6.0f), oy = Uniform(-6.0f, 6.0f), oz = Uniform(-6.0f, 6.0f);
			float ex = Uniform(-6.0f, 6.0f), ey = Uniform(-6.0f, 6.0f), ez = Uniform(-6.0f, 6.0f);
			int excluded = q % 2 == 0 ? -1 : q;
			float maxDistance = Uniform(1.0f, 20.0f);
			bool expected = false;
			for (int i = 0; i < numTriangles && !expected; i++)
			{
				expected = singles[i].AnyHit(ox, oy, oz, ex, ey, ez, excluded, maxDistance);
			}

			bool actual = bvh.AnyHit(ox, oy, oz, ex, ey, ez, excluded, maxDistance);
			await Assert.That(actual).IsEqualTo(expected);

			// The counting query answers the same question for any excluded id.
			int count = bvh.CountHitsUpToTwo(ox, oy, oz, ex, ey, ez, maxDistance, out int onlyId);
			bool fromCount = count == 2 || (count == 1 && onlyId != excluded);
			await Assert.That(fromCount).IsEqualTo(expected);
			await Assert.That(onlyId >= 0).IsEqualTo(count == 1);
			hits += actual ? 1 : 0;
		}

		// The query mix must contain both answers to mean anything.
		await Assert.That(hits).IsGreaterThan(0);
		await Assert.That(hits).IsLessThan(300);
	}

	// Random small triangles; the build tests use 40,000, above the parallel threshold (32,768).
	private static (float[] Corners, int[] Ids) RandomTriangles(int count, int seed)
	{
		var random = new Random(seed);
		float Uniform(float lo, float hi) => lo + (hi - lo) * (float)random.NextDouble();
		var corners = new float[9 * count];
		var ids = new int[count];
		for (int i = 0; i < count; i++)
		{
			float cx = Uniform(-5, 5), cy = Uniform(-5, 5), cz = Uniform(-5, 5);
			for (int k = 0; k < 9; k++)
			{
				corners[9 * i + k] = (k % 3 == 0 ? cx : k % 3 == 1 ? cy : cz) + Uniform(-0.05f, 0.05f);
			}

			ids[i] = i;
		}

		return (corners, ids);
	}

	[Test]
	public async Task Build_ParallelLayoutMatchesSequential()
	{
		(float[] corners, int[] ids) = RandomTriangles(40_000, 7);
		var sequential = new TriangleBvh(corners, ids, numThreads: 1);
		await Assert.That(sequential.ParallelSplits).IsEqualTo(0);

		var expected = sequential.Layout;
		foreach (int threads in new[] { 2, 4, -1, -1, -1 })
		{
			var parallel = new TriangleBvh(corners, ids, threads);
			await Assert.That(parallel.ParallelSplits).IsGreaterThan(0);
			var actual = parallel.Layout;
			await Assert.That(actual.NodeBounds.AsSpan().SequenceEqual(expected.NodeBounds)).IsTrue();
			await Assert.That(actual.NodeFirst.AsSpan().SequenceEqual(expected.NodeFirst)).IsTrue();
			await Assert.That(actual.NodeCount.AsSpan().SequenceEqual(expected.NodeCount)).IsTrue();
			await Assert.That(actual.Ids.AsSpan().SequenceEqual(expected.Ids)).IsTrue();
		}
	}

	[Test]
	public async Task Build_HonorsCancellation()
	{
		(float[] corners, int[] ids) = RandomTriangles(40_000, 8);
		using var cts = new CancellationTokenSource();
		cts.Cancel();
		await Assert.That(() => new TriangleBvh(corners, ids, 4, cts.Token)).Throws<OperationCanceledException>();
	}

	[Test]
	public async Task AnyHit_CoincidentCentroidsStillBuild()
	{
		// Every triangle has the same centroid, so no centroid split exists.
		const int n = 50;
		var corners = new float[9 * n];
		var ids = new int[n];
		for (int i = 0; i < n; i++)
		{
			float s = 1 + i * 0.01f;
			float[] tri = [-s, -s, 0, 2 * s, -s, 0, -s, 2 * s, 0];
			tri.CopyTo(corners, 9 * i);
			ids[i] = i;
		}

		var bvh = new TriangleBvh(corners, ids);
		await Assert.That(bvh.AnyHit(0, 0, 1, 0, 0, -1, -1, 10)).IsTrue();
		await Assert.That(bvh.AnyHit(5, 5, 1, 5, 5, -1, -1, 10)).IsFalse();
	}

	// A grid of quads in z = 0 (n x n cells over [0, 1]^2) plus, when raised is set, a small
	// raised quad at z = 0.5 over the grid's center.
	private static PlyMesh MakeGrid(int n, bool raised)
	{
		var mesh = new PlyMesh();
		for (int y = 0; y <= n; y++)
		{
			for (int x = 0; x <= n; x++)
			{
				mesh.Vertices.Add(new PlyMeshVertex((float)x / n, (float)y / n, 0));
			}
		}

		for (int y = 0; y < n; y++)
		{
			for (int x = 0; x < n; x++)
			{
				int a = y * (n + 1) + x;
				mesh.Faces.Add(new PlyMeshFace(a, a + 1, a + n + 2));
				mesh.Faces.Add(new PlyMeshFace(a, a + n + 2, a + n + 1));
			}
		}

		if (raised)
		{
			int b = mesh.Vertices.Count;
			mesh.Vertices.Add(new PlyMeshVertex(0.3f, 0.3f, 0.5f));
			mesh.Vertices.Add(new PlyMeshVertex(0.7f, 0.3f, 0.5f));
			mesh.Vertices.Add(new PlyMeshVertex(0.7f, 0.7f, 0.5f));
			mesh.Vertices.Add(new PlyMeshVertex(0.3f, 0.7f, 0.5f));
			mesh.Faces.Add(new PlyMeshFace(b, b + 1, b + 2));
			mesh.Faces.Add(new PlyMeshFace(b, b + 2, b + 3));
		}

		return mesh;
	}

	[Test]
	public async Task MeshTextureMapping_OccludedFacesAreNotTextured()
	{
		// The camera above the grid's center sees the raised quad, which hides the grid cells
		// under it; the grid's corner cells stay visible.
		const int n = 10;
		PlyMesh mesh = MakeGrid(n, raised: true);
		List<Image> images = [TextureMappingTests.MakeTestImage(512, 512, new BitmapColor<byte>(90))];
		var options = new MeshTextureMappingOptions { ViewSelectionSmoothingIterations = 0, InpaintRadius = 0 };

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		int centerCell = 2 * ((n / 2) * n + n / 2);
		await Assert.That(result.FaceViewIds[centerCell]).IsEqualTo(-1);
		await Assert.That(result.FaceViewIds[centerCell + 1]).IsEqualTo(-1);
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(0);
		await Assert.That(result.FaceViewIds[^1]).IsEqualTo(0);
		await Assert.That(result.FaceViewIds[^2]).IsEqualTo(0);
	}

	// Views above the grid at C = (0.3 + 0.13 i, 0.4 + 0.07 i, 4 + 0.5 i), looking down, each
	// with its own gradient bitmap, so faces pick different views and seams appear.
	private static List<Image> MakeViews(int count)
	{
		var images = new List<Image>();
		for (int i = 0; i < count; i++)
		{
			float[] k = [300, 0, 128, 0, 300, 128, 0, 0, 1];
			float[] r = [1, 0, 0, 0, -1, 0, 0, 0, -1];

			// Camera centers C = (0.3 + 0.13 i, 0.4 + 0.07 i, 4 + 0.5 i); T = -R C.
			float[] t = [-(0.3f + 0.13f * i), 0.4f + 0.07f * i, 4 + 0.5f * i];
			var image = new Image($"view{i}.png", 256, 256, k, r, t);
			var bitmap = new Bitmap(256, 256, asRgb: true);
			for (int y = 0; y < 256; y++)
			{
				for (int x = 0; x < 256; x++)
				{
					bitmap.SetPixel(x, y, new BitmapColor<byte>((byte)(x + 40 * i), (byte)y, (byte)(x ^ y)));
				}
			}

			image.SetBitmap(bitmap);
			images.Add(image);
		}

		return images;
	}

	[Test]
	public async Task MeshTextureMapping_SameResultForAnyThreadCount()
	{
		// Several overlapping views (so smoothing, several regions, seams and color
		// correction all run); one thread and all threads must agree bit for bit.
		PlyMesh mesh = MakeGrid(40, raised: true);
		List<Image> images = MakeViews(4);

		var sequential = new MeshTextureMappingOptions { NumThreads = 1 };
		var parallel = new MeshTextureMappingOptions { NumThreads = -1 };
		MeshTextureMappingResult a = TextureMapping.MeshTextureMapping(mesh, images, sequential);
		MeshTextureMappingResult b = TextureMapping.MeshTextureMapping(mesh, images, parallel);

		await Assert.That(a.FaceViewIds.Distinct().Count()).IsGreaterThan(1);
		await Assert.That(b.FaceViewIds.SequenceEqual(a.FaceViewIds)).IsTrue();
		await Assert.That(b.FaceUvs.SequenceEqual(a.FaceUvs)).IsTrue();
		await Assert.That(b.AtlasWidth).IsEqualTo(a.AtlasWidth);
		await Assert.That(b.AtlasHeight).IsEqualTo(a.AtlasHeight);
		await Assert.That(b.TextureAtlas.RowMajorData.AsSpan().SequenceEqual(a.TextureAtlas.RowMajorData)).IsTrue();

		// The seams were real: color correction changed the atlas.
		var uncorrected = new MeshTextureMappingOptions { NumThreads = -1, ApplyColorCorrection = false };
		MeshTextureMappingResult c = TextureMapping.MeshTextureMapping(mesh, images, uncorrected);
		await Assert.That(c.TextureAtlas.RowMajorData.AsSpan().SequenceEqual(a.TextureAtlas.RowMajorData)).IsFalse();
	}

	[Test]
	public async Task MeshTextureMapping_HonorsCancellation()
	{
		PlyMesh mesh = MakeGrid(4, raised: false);
		List<Image> images = [TextureMappingTests.MakeTestImage(64, 64, new BitmapColor<byte>(90))];
		using var cts = new CancellationTokenSource();
		cts.Cancel();
		await Assert.That(() => TextureMapping.MeshTextureMapping(mesh, images, new MeshTextureMappingOptions(), null, cts.Token))
			.Throws<OperationCanceledException>();
	}

	// Reports synchronously on the calling thread, unlike Progress<T>, so the test sees the
	// order in which the library reports.
	private sealed class RecordingProgress(Action<double>? onReport = null) : IProgress<double>
	{
		public List<double> Values { get; } = [];

		public void Report(double value)
		{
			lock (Values)
			{
				Values.Add(value);
			}

			onReport?.Invoke(value);
		}
	}

	[Test]
	public async Task MeshTextureMapping_ProgressNeverGoesBackwards()
	{
		// A few thousand faces give several parallel view-selection chunks.
		PlyMesh mesh = MakeGrid(60, raised: true);
		List<Image> images = [TextureMappingTests.MakeTestImage(256, 256, new BitmapColor<byte>(90))];
		var progress = new RecordingProgress();
		TextureMapping.MeshTextureMapping(mesh, images, new MeshTextureMappingOptions { NumThreads = -1 }, progress);

		await Assert.That(progress.Values.Count).IsGreaterThan(3);

		// With a single image, view selection still reports per chunk of faces (8 here).
		int viewSelectionReports = progress.Values.Count(v => v > 0.05 && v < 0.7);
		await Assert.That(viewSelectionReports).IsGreaterThanOrEqualTo(5);
		for (int i = 1; i < progress.Values.Count; i++)
		{
			await Assert.That(progress.Values[i]).IsGreaterThanOrEqualTo(progress.Values[i - 1]);
		}

		await Assert.That(progress.Values[^1]).IsEqualTo(1.0);
	}

	[Test]
	public async Task MeshTextureMapping_CancelsDuringColorCorrection()
	{
		// Cancel once baking is reported done; color correction must notice.
		PlyMesh mesh = MakeGrid(20, raised: true);
		List<Image> images = MakeViews(4);
		using var cts = new CancellationTokenSource();
		var progress = new RecordingProgress(value =>
		{
			if (value >= 0.9)
			{
				cts.Cancel();
			}
		});
		await Assert.That(() => TextureMapping.MeshTextureMapping(mesh, images, new MeshTextureMappingOptions(), progress, cts.Token))
			.Throws<OperationCanceledException>();
		await Assert.That(progress.Values[^1]).IsEqualTo(0.9);
	}
}
