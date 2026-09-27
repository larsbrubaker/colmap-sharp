// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionTests: colmap/controllers/automatic_reconstruction_test.cc ported
// 1:1, testing ColmapSharp/Controllers/AutomaticReconstruction*.cs. The TEST_P
// ParameterizedAutomaticReconstructionTests.Nominal becomes one method with an Arguments row
// per INSTANTIATE_TEST_SUITE_P value (INCREMENTAL, HIERARCHICAL; COLMAP leaves GLOBAL out).
//
// Tier C (outcome): the reconstruction from synthesized images against the ground truth
// after alignment, with COLMAP's bounds.
//
// Translation notes: the images folder is an InMemoryImageSource filled by SynthesizeImages;
// the workspace is a temporary folder. PrngTestIsolation seeds the PRNG with 0 before every
// test, as COLMAP's gtest_main does; the PRNG is per thread, so the test runs the controller
// before its first await.

using ColmapSharp.Controllers;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-autorecon-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	[Test]
	[Arguments(AutomaticReconstructionOptions.MapperType.Incremental)]
	[Arguments(AutomaticReconstructionOptions.MapperType.Hierarchical)]
	public async Task ParameterizedAutomaticReconstructionTests_Nominal(AutomaticReconstructionOptions.MapperType mapper)
	{
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var images = new InMemoryImageSource();

		var gtReconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 5,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 200,
			NumPoints2DWithoutPoint3D = 10,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, gtReconstruction);
		Synthetic.SynthesizeImages(new SyntheticImageOptions(), gtReconstruction, images.Add);

		var options = new AutomaticReconstructionOptions
		{
			WorkspacePath = workspacePath,
			Images = images,
			Data = AutomaticReconstructionOptions.DataType.Individual,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			SingleCamera = false,
			Dense = false, // Disable dense reconstruction to avoid GPU
			RandomSeed = 1,
			Mapper = mapper,
		};
		// options.use_gpu = false: there are no GPU stages.

		var reconstructionManager = new ReconstructionManager();
		var controller = new AutomaticReconstructionController(options, reconstructionManager);
		controller.Setup();
		controller.Run();

		int size = reconstructionManager.Size;
		string? nearExplanation = size == 0
			? "no reconstruction"
			: ReconstructionMatchers.ExplainReconstructionNear(
				reconstructionManager.Get(0),
				gtReconstruction,
				maxRotationErrorDeg: 0.6,
				maxProjCenterError: 0.1,
				maxScaleError: null,
				numObsTolerance: 0.9,
				align: true);

		try
		{
			Directory.Delete(testDir, recursive: true);
		}
		catch (IOException)
		{
			// A leftover temp folder is harmless.
		}

		await Assert.That(size).IsEqualTo(1);
		await Assert.That(nearExplanation).IsNull();
	}
}
