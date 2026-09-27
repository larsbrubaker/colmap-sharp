// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchRunTests: C#-only tests (COLMAP tests PatchMatch only through its CUDA
// pipeline, and the pycolmap wheel has no CUDA, so there is no oracle) for PatchMatch.Run in
// ColmapSharp/Mvs/PatchMatch.cs and PatchMatchCpu.cs. Tier C by construction: a synthetic
// scene - a textured slanted plane seen by a reference and two source cameras, rendered
// analytically - checks that the recovered depths and normals converge to the truth, that
// filtering and the consistency graph behave, and that results are bit-identical across
// thread counts and reruns (the determinism the CPU port promises, divergence 86).

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchRunTests
{
	private const float Fx = 40;
	private const float Slope = 0.3f;
	private const float PlaneZ = 4.0f;
	private static readonly float[] Identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];

	// Camera centres; every camera looks down +z with R = I, so T = -C.
	private static readonly float[][] Centers = [[0, 0, 0], [0.5f, 0, 0], [-0.4f, 0.3f, 0]];

	// The plane z = PlaneZ + Slope * X.
	private static double RayDepth(double cx, double cy, double cz, double dx, double dy) =>
		(PlaneZ + Slope * cx - cz) / (1 - Slope * dx);

	private static double Texture(double x, double y) =>
		0.5 + 0.2 * Math.Sin(9 * x + 1) * Math.Cos(8 * y) + 0.15 * Math.Sin(23 * x - 17 * y) + 0.1 * Math.Cos(31 * y + 5 * x);

	internal static (List<Image> Images, List<DepthMap> DepthMaps, List<NormalMap> NormalMaps) Scene(int width, int height)
	{
		float cx = width / 2.0f - 0.5f;
		float cy = height / 2.0f - 0.5f;
		float[] k = [Fx, 0, cx, 0, Fx, cy, 0, 0, 1];
		var images = new List<Image>();
		var depthMaps = new List<DepthMap>();
		var normalMaps = new List<NormalMap>();
		float normalNorm = MathF.Sqrt(Slope * Slope + 1);
		foreach (float[] c in Centers)
		{
			var image = new Image("img", width, height, k, Identity, [-c[0], -c[1], -c[2]]);
			var bitmap = new Bitmap(width, height, asRgb: false);
			var depthMap = new DepthMap(width, height, 0, 10);
			var normalMap = new NormalMap(width, height);
			for (int v = 0; v < height; ++v)
			{
				for (int u = 0; u < width; ++u)
				{
					double dx = (u - cx) / Fx;
					double dy = (v - cy) / Fx;
					double t = RayDepth(c[0], c[1], c[2], dx, dy);
					double x = c[0] + t * dx;
					double y = c[1] + t * dy;
					bitmap.RowMajorData[v * width + u] = (byte)Math.Round(20 + 200 * Texture(x, y));
					depthMap.Set(v, u, (float)t);
					normalMap.Set(v, u, 0, Slope / normalNorm);
					normalMap.Set(v, u, 1, 0);
					normalMap.Set(v, u, 2, -1 / normalNorm);
				}
			}

			image.SetBitmap(bitmap);
			images.Add(image);
			depthMaps.Add(depthMap);
			normalMaps.Add(normalMap);
		}

		return (images, depthMaps, normalMaps);
	}

	internal static PatchMatchOptions Options(int numIterations, bool geomConsistency, bool filter, int numThreads = -1) => new()
	{
		DepthMin = 2,
		DepthMax = 8,
		WindowRadius = 3,
		SigmaSpatial = 3,
		NumIterations = numIterations,
		GeomConsistency = geomConsistency,
		Filter = filter,
		FilterMinNumConsistent = 1,
		NumThreads = numThreads,
	};

	internal static PatchMatch.Problem Problem(List<Image> images, List<DepthMap>? depthMaps = null, List<NormalMap>? normalMaps = null) => new()
	{
		RefImageIdx = 0,
		SrcImageIdxs = [1, 2],
		Images = images,
		DepthMaps = depthMaps ?? [],
		NormalMaps = normalMaps ?? [],
	};

	/// <summary>The fraction of interior pixels whose depth is within <paramref name="tolerance"/> (relative) of the truth.</summary>
	private static double FractionWithin(DepthMap estimate, DepthMap truth, float tolerance, int border)
	{
		int good = 0;
		int total = 0;
		for (int row = border; row < truth.GetHeight() - border; ++row)
		{
			for (int col = border; col < truth.GetWidth() - border; ++col)
			{
				total++;
				if (MathF.Abs(estimate.Get(row, col) - truth.Get(row, col)) <= tolerance * truth.Get(row, col))
				{
					good++;
				}
			}
		}

		return good / (double)total;
	}

	[Test]
	public async Task Photometric_RecoversTheSlantedPlane()
	{
		(List<Image> images, List<DepthMap> truth, _) = Scene(48, 36);
		var progress = new List<double>();
		var patchMatch = new PatchMatch(Options(3, geomConsistency: false, filter: false), Problem(images));
		patchMatch.Run(progress: new SynchronousProgress(progress));

		DepthMap depthMap = patchMatch.GetDepthMap();
		await Assert.That(depthMap.GetWidth()).IsEqualTo(48);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(36);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(2.0f);
		await Assert.That(FractionWithin(depthMap, truth[0], 0.02f, 4)).IsGreaterThan(0.9);

		// Normals: the median angle to the true normal on well-estimated pixels is small.
		NormalMap normalMap = patchMatch.GetNormalMap();
		float normalNorm = MathF.Sqrt(Slope * Slope + 1);
		var cosines = new List<float>();
		for (int row = 4; row < 32; ++row)
		{
			for (int col = 4; col < 44; ++col)
			{
				cosines.Add((normalMap.Get(row, col, 0) * Slope - normalMap.Get(row, col, 2)) / normalNorm);
			}
		}

		cosines.Sort();
		await Assert.That(cosines[cosines.Count / 2]).IsGreaterThan(MathF.Cos(15 * MathF.PI / 180));

		// One report per sweep, ending at 1.
		await Assert.That(progress.Count).IsEqualTo(12);
		await Assert.That(progress[^1]).IsEqualTo(1.0);

		// No filtering: an empty consistency graph; selection probabilities per source.
		ConsistencyGraph graph = patchMatch.GetConsistencyGraph();
		int numConsistent = 0;
		for (int row = 0; row < 36; ++row)
		{
			for (int col = 0; col < 48; ++col)
			{
				numConsistent += graph.GetImageIdxs(row, col).Length;
			}
		}

		await Assert.That(numConsistent).IsEqualTo(0);
		await Assert.That(patchMatch.GetSelProbMap().GetDepth()).IsEqualTo(2);
	}

	[Test]
	public async Task Geometric_KeepsTheTruthAndMarksConsistentSources()
	{
		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = Scene(40, 30);
		var patchMatch = new PatchMatch(Options(1, geomConsistency: true, filter: true), Problem(images, truth, normals));
		patchMatch.Run();

		DepthMap depthMap = patchMatch.GetDepthMap();
		await Assert.That(FractionWithin(depthMap, truth[0], 0.01f, 4)).IsGreaterThan(0.9);

		// Interior pixels see both sources consistently; filtered pixels are zeroed.
		ConsistencyGraph graph = patchMatch.GetConsistencyGraph();
		int consistentPixels = 0;
		for (int row = 0; row < 30; ++row)
		{
			for (int col = 0; col < 40; ++col)
			{
				int[] imageIdxs = graph.GetImageIdxs(row, col).ToArray();
				if (imageIdxs.Length > 0)
				{
					consistentPixels++;
					foreach (int imageIdx in imageIdxs)
					{
						await Assert.That(imageIdx == 1 || imageIdx == 2).IsTrue();
					}
				}
				else
				{
					await Assert.That(depthMap.Get(row, col)).IsEqualTo(0.0f);
				}
			}
		}

		await Assert.That(consistentPixels).IsGreaterThan(40 * 30 / 2);
		int[] center = graph.GetImageIdxs(15, 20).ToArray();
		await Assert.That(center).IsEquivalentTo([1, 2], CollectionOrdering.Matching);
	}

	[Test]
	public async Task Results_IndependentOfThreadCountAndRerun()
	{
		(List<Image> images, _, _) = Scene(26, 19);
		(DepthMap Depth, NormalMap Normal, Mat<float> SelProb, ConsistencyGraph Graph) RunWith(int numThreads)
		{
			var patchMatch = new PatchMatch(Options(1, geomConsistency: false, filter: true, numThreads), Problem(images));
			patchMatch.Run();
			return (patchMatch.GetDepthMap(), patchMatch.GetNormalMap(), patchMatch.GetSelProbMap(), patchMatch.GetConsistencyGraph());
		}

		var single = RunWith(1);
		foreach (int numThreads in new[] { 1, 3, -1 })
		{
			var other = RunWith(numThreads);
			await Assert.That(other.Depth.Data).IsEquivalentTo(single.Depth.Data, CollectionOrdering.Matching);
			await Assert.That(other.Normal.Data).IsEquivalentTo(single.Normal.Data, CollectionOrdering.Matching);
			await Assert.That(other.SelProb.Data).IsEquivalentTo(single.SelProb.Data, CollectionOrdering.Matching);
			for (int row = 0; row < 19; ++row)
			{
				for (int col = 0; col < 26; ++col)
				{
					await Assert.That(other.Graph.GetImageIdxs(row, col).ToArray())
						.IsEquivalentTo(single.Graph.GetImageIdxs(row, col).ToArray(), CollectionOrdering.Matching);
				}
			}
		}
	}

	[Test]
	public async Task Run_CancelsAndGuardsResults()
	{
		(List<Image> images, _, _) = Scene(16, 12);
		var patchMatch = new PatchMatch(Options(1, geomConsistency: false, filter: false), Problem(images));
		await Assert.That(() => patchMatch.GetDepthMap()).Throws<InvalidOperationException>();

		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		await Assert.That(() => patchMatch.Run(cancellation.Token)).Throws<OperationCanceledException>();

		// A cancelled run leaves no results to read.
		await Assert.That(() => patchMatch.GetDepthMap()).Throws<InvalidOperationException>();

		// Run checks the problem first.
		var invalid = new PatchMatch(Options(1, geomConsistency: true, filter: false), Problem(images));
		await Assert.That(() => invalid.Run()).Throws<ArgumentException>();
	}

	// Progress<T> posts to the thread pool; this records reports as they happen.
	private sealed class SynchronousProgress(List<double> reports) : IProgress<double>
	{
		public void Report(double value) => reports.Add(value);
	}
}
