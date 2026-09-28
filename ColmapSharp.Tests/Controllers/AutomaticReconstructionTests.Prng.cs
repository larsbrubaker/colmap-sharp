// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// C#-only test of AutomaticReconstructionController's sparse stage
// (ColmapSharp/Controllers/AutomaticReconstruction.cs, docs/CPP_DIVERGENCES.md entry 138):
// with the default random_seed (-1) the mapper's RANSAC draws from the thread's PRNG. COLMAP
// runs the mapper on the controller's own new thread, so it always starts from the default
// seed; the port runs it on the caller's thread, often a reused pool thread, so without a fresh
// PRNG the sparse model depended on whatever that thread drew before. The global mapper first
// re-estimates relative poses in ViewGraphCalibration's Parallel.For (entry 139); its test also
// compares 1 and 4 threads. The demo's GPU fallback
// test (demo/ColmapDemo.Tests/SessionGpuFallbackTests.cs) failed about one run in four from it.
//
// Tier A for this property: the same inputs give byte-identical sparse model files.

using ColmapSharp.Controllers;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	public async Task CSharpOnly_UnseededSparseModelIgnoresTheCallersPrng()
	{
		var gtReconstruction = new Reconstruction();
		var images = new InMemoryImageSource();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 4,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				NumPoints3D = 150,
				NumPoints2DWithoutPoint3D = 10,
			},
			gtReconstruction);
		Synthetic.SynthesizeImages(new SyntheticImageOptions(), gtReconstruction, images.Add);

		// Two callers whose threads drew different streams before; everything runs synchronously
		// on this thread, before the first await.
		RandomUtils.SetPRNGSeed(1234);
		Mt19937 callerPrng = RandomUtils.Prng!;
		Dictionary<string, byte[]> first = RunSparse(images);
		bool callerPrngKept = ReferenceEquals(RandomUtils.Prng, callerPrng);

		RandomUtils.SetPRNGSeed(98765);
		for (int i = 0; i < 1000; i++)
		{
			RandomUtils.RandomUniformReal(0.0, 1.0);
		}

		Dictionary<string, byte[]> second = RunSparse(images);

		await Assert.That(first.Count).IsGreaterThan(0);
		await Assert.That(second.Keys.Order(StringComparer.Ordinal))
			.IsEquivalentTo(first.Keys.Order(StringComparer.Ordinal));
		foreach ((string name, byte[] bytes) in first)
		{
			await Assert.That(second[name].AsSpan().SequenceEqual(bytes)).IsTrue().Because(name + " differs");
		}

		// The mapper's draws do not leak into the caller's stream.
		await Assert.That(callerPrngKept).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_UnseededGlobalModelIgnoresTheCallersPrngAndThreadCount()
	{
		// The global mapper first re-estimates the relative poses in ViewGraphCalibration, a
		// Parallel.For of unseeded RANSAC (entry 139). Each pair must start from a fresh PRNG, and
		// the calling thread's share of the pairs must not consume the mapper's stream.
		var gtReconstruction = new Reconstruction();
		var images = new InMemoryImageSource();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 5,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				NumPoints3D = 150,
				NumPoints2DWithoutPoint3D = 10,
			},
			gtReconstruction);
		Synthetic.SynthesizeImages(new SyntheticImageOptions(), gtReconstruction, images.Add);

		RandomUtils.SetPRNGSeed(1234);
		Mt19937 callerPrng = RandomUtils.Prng!;
		Dictionary<string, byte[]> oneThread = RunSparse(images, AutomaticReconstructionOptions.MapperType.Global, 1);
		bool callerPrngKept = ReferenceEquals(RandomUtils.Prng, callerPrng);

		RandomUtils.SetPRNGSeed(98765);
		for (int i = 0; i < 1000; i++)
		{
			RandomUtils.RandomUniformReal(0.0, 1.0);
		}

		Dictionary<string, byte[]> manyThreads = RunSparse(images, AutomaticReconstructionOptions.MapperType.Global, 4);

		await Assert.That(oneThread.Count).IsGreaterThan(0);
		await Assert.That(manyThreads.Keys.Order(StringComparer.Ordinal))
			.IsEquivalentTo(oneThread.Keys.Order(StringComparer.Ordinal));
		foreach ((string name, byte[] bytes) in oneThread)
		{
			await Assert.That(manyThreads[name].AsSpan().SequenceEqual(bytes)).IsTrue().Because(name + " differs");
		}

		await Assert.That(callerPrngKept).IsTrue();
	}

	// Extraction, matching and the selected mapper with the default (unseeded) options, in a
	// fresh workspace; returns the sparse model's files by relative path.
	private static Dictionary<string, byte[]> RunSparse(
		InMemoryImageSource images,
		AutomaticReconstructionOptions.MapperType mapper = AutomaticReconstructionOptions.MapperType.Incremental,
		int numThreads = -1)
	{
		string testDir = CreateTestDir();
		try
		{
			var options = new AutomaticReconstructionOptions
			{
				WorkspacePath = testDir,
				Images = images,
				Data = AutomaticReconstructionOptions.DataType.Individual,
				Quality = AutomaticReconstructionOptions.QualityLevel.Low,
				Dense = false,
				Mapper = mapper,
				NumThreads = numThreads,
			};
			var controller = new AutomaticReconstructionController(options, new ReconstructionManager());
			controller.Setup();
			controller.Run();

			string sparse = Path.Combine(testDir, "sparse");
			return Directory.GetFiles(sparse, "*", SearchOption.AllDirectories)
				.ToDictionary(f => Path.GetRelativePath(sparse, f), File.ReadAllBytes);
		}
		finally
		{
			try
			{
				Directory.Delete(testDir, recursive: true);
			}
			catch (IOException)
			{
				// A leftover temp folder is harmless.
			}
		}
	}
}
