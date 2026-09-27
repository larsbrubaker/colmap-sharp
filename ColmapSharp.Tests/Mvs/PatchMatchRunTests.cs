// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchRunTests: C#-only tests (COLMAP tests PatchMatch only through its CUDA
// pipeline, and the pycolmap wheel has no CUDA, so there is no oracle) for PatchMatch.Run in
// ColmapSharp/Mvs/PatchMatch.cs and PatchMatchCpu.cs. Tier C by construction: a synthetic
// scene - a textured slanted plane seen by a reference and two source cameras, rendered
// analytically - checks that the recovered depths and normals converge to the truth, that
// filtering and the consistency graph behave, and that results are bit-identical across
// thread counts and reruns (the determinism the CPU port promises, divergence 86). The scene,
// options and truth measures are the library's (Mvs/Testing/PatchMatchSyntheticScene.cs), shared
// with the GPU tests and PatchMatchGpuConformance.

using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchRunTests
{
	[Test]
	public async Task Photometric_RecoversTheSlantedPlane()
	{
		(List<Image> images, List<DepthMap> truth, _) = PatchMatchSyntheticScene.Scene(48, 36);
		var progress = new List<double>();
		var patchMatch = new PatchMatch(PatchMatchSyntheticScene.Options(3, geomConsistency: false, filter: false), PatchMatchSyntheticScene.Problem(images));
		patchMatch.Run(progress: new SynchronousProgress(progress));

		DepthMap depthMap = patchMatch.GetDepthMap();
		await Assert.That(depthMap.GetWidth()).IsEqualTo(48);
		await Assert.That(depthMap.GetHeight()).IsEqualTo(36);
		await Assert.That(depthMap.GetDepthMin()).IsEqualTo(2.0f);
		await Assert.That(PatchMatchSyntheticScene.FractionWithin(depthMap, truth[0], 0.02f, 4)).IsGreaterThan(0.9);

		// Normals: the median angle to the true normal on well-estimated pixels is small.
		await Assert.That(PatchMatchSyntheticScene.MedianTrueNormalCosine(patchMatch.GetNormalMap(), 4)).IsGreaterThan(MathF.Cos(15 * MathF.PI / 180));

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
		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = PatchMatchSyntheticScene.Scene(40, 30);
		var patchMatch = new PatchMatch(PatchMatchSyntheticScene.Options(1, geomConsistency: true, filter: true), PatchMatchSyntheticScene.Problem(images, truth, normals));
		patchMatch.Run();

		DepthMap depthMap = patchMatch.GetDepthMap();
		await Assert.That(PatchMatchSyntheticScene.FractionWithin(depthMap, truth[0], 0.01f, 4)).IsGreaterThan(0.9);

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
		(List<Image> images, _, _) = PatchMatchSyntheticScene.Scene(26, 19);
		(DepthMap Depth, NormalMap Normal, Mat<float> SelProb, ConsistencyGraph Graph) RunWith(int numThreads)
		{
			var patchMatch = new PatchMatch(PatchMatchSyntheticScene.Options(1, geomConsistency: false, filter: true, numThreads), PatchMatchSyntheticScene.Problem(images));
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
		(List<Image> images, _, _) = PatchMatchSyntheticScene.Scene(16, 12);
		var patchMatch = new PatchMatch(PatchMatchSyntheticScene.Options(1, geomConsistency: false, filter: false), PatchMatchSyntheticScene.Problem(images));
		await Assert.That(() => patchMatch.GetDepthMap()).Throws<InvalidOperationException>();

		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		await Assert.That(() => patchMatch.Run(cancellation.Token)).Throws<OperationCanceledException>();

		// A cancelled run leaves no results to read.
		await Assert.That(() => patchMatch.GetDepthMap()).Throws<InvalidOperationException>();

		// Run checks the problem first.
		var invalid = new PatchMatch(PatchMatchSyntheticScene.Options(1, geomConsistency: true, filter: false), PatchMatchSyntheticScene.Problem(images));
		await Assert.That(() => invalid.Run()).Throws<ArgumentException>();
	}

	// Progress<T> posts to the thread pool; this records reports as they happen.
	private sealed class SynchronousProgress(List<double> reports) : IProgress<double>
	{
		public void Report(double value) => reports.Add(value);
	}
}
