// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only, not a port. Also pins that the host's
// BaRefineFocalLength / BaRefineExtraParams (divergence 141) reach the mapper under every
// preset, defaulting to COLMAP's refinement. Pins that the host's Poisson
// options (AutomaticReconstructionOptions.PoissonMeshing, which COLMAP's controller does not
// have) reach the meshing step under every quality preset, and that NumThreads still comes from
// the controller's own NumThreads as in COLMAP. The step passes EffectivePoissonMeshing straight
// to PoissonMeshing.Run (AutomaticReconstruction.Dense.cs RunMeshing).

using ColmapSharp.Controllers;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.Low)]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.Medium)]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.High)]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.Extreme)]
	public async Task CSharpOnly_HostPoissonOptionsReachTheMesher(AutomaticReconstructionOptions.QualityLevel quality)
	{
		string testDir = CreateTestDir();
		var options = new AutomaticReconstructionOptions
		{
			WorkspacePath = testDir,
			Images = new InMemoryImageSource(),
			Quality = quality,
			NumThreads = 3,
		};
		options.PoissonMeshing.Depth = 9;
		options.PoissonMeshing.Trim = 4.5;
		options.PoissonMeshing.PointWeight = 2.0;
		options.PoissonMeshing.Color = false;
		options.PoissonMeshing.NumThreads = 7;

		var controller = new AutomaticReconstructionController(options, new ReconstructionManager());

		PoissonMeshingOptions effective = controller.EffectivePoissonMeshing;
		await Assert.That(effective.Depth).IsEqualTo(9);
		await Assert.That(effective.Trim).IsEqualTo(4.5);
		await Assert.That(effective.PointWeight).IsEqualTo(2.0);
		await Assert.That(effective.Color).IsFalse();
		await Assert.That(effective.NumThreads).IsEqualTo(3);
		Directory.Delete(testDir, recursive: true);
	}

	[Test]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.Low, AutomaticReconstructionOptions.DataType.Individual)]
	[Arguments(AutomaticReconstructionOptions.QualityLevel.Extreme, AutomaticReconstructionOptions.DataType.Video)]
	public async Task CSharpOnly_BaRefineOptionsReachTheMapper(
		AutomaticReconstructionOptions.QualityLevel quality,
		AutomaticReconstructionOptions.DataType data)
	{
		string testDir = CreateTestDir();
		var byDefault = new AutomaticReconstructionController(
			new AutomaticReconstructionOptions { WorkspacePath = testDir, Images = new InMemoryImageSource(), Quality = quality, Data = data },
			new ReconstructionManager());
		var known = new AutomaticReconstructionController(
			new AutomaticReconstructionOptions
			{
				WorkspacePath = testDir,
				Images = new InMemoryImageSource(),
				Quality = quality,
				Data = data,
				BaRefineFocalLength = false,
				BaRefineExtraParams = false,
			},
			new ReconstructionManager());

		await Assert.That(byDefault.EffectiveMapper.BaRefineFocalLength).IsTrue();
		await Assert.That(byDefault.EffectiveMapper.BaRefineExtraParams).IsTrue();
		await Assert.That(known.EffectiveMapper.BaRefineFocalLength).IsFalse();
		await Assert.That(known.EffectiveMapper.BaRefineExtraParams).IsFalse();
		Directory.Delete(testDir, recursive: true);
	}

	[Test]
	public async Task CSharpOnly_DefaultPoissonOptionsAreColmaps()
	{
		string testDir = CreateTestDir();
		var options = new AutomaticReconstructionOptions { WorkspacePath = testDir, Images = new InMemoryImageSource() };

		var controller = new AutomaticReconstructionController(options, new ReconstructionManager());

		PoissonMeshingOptions effective = controller.EffectivePoissonMeshing;
		await Assert.That(effective.Depth).IsEqualTo(13);
		await Assert.That(effective.Trim).IsEqualTo(10.0);
		await Assert.That(effective.PointWeight).IsEqualTo(1.0);
		await Assert.That(effective.Color).IsTrue();
		Directory.Delete(testDir, recursive: true);
	}
}
