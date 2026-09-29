// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GlobalPipelineTests.CSharpOnly: C#-only cases for ColmapSharp/Controllers/GlobalPipeline.cs
// (COLMAP has no progress reports and no CancellationToken): the stage progress,
// cancellation through BaseController.CancellationToken, and the order of equally large
// reconstructions (divergence 107).

using ColmapSharp.Controllers;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class GlobalPipelineTests
{
	private static SyntheticDatasetOptions SmallDatasetOptions() => new()
	{
		NumRigs = 1,
		NumCamerasPerRig = 1,
		NumFramesPerRig = 5,
		NumPoints3D = 50,
		CameraHasPriorFocalLength = true,
		TwoViewGeometryHasRelativePose = true,
	};

	// C#-only: every mapper stage is reported in order, followed by the finished component.
	[Test]
	public async Task CSharpOnly_ProgressReportsEveryStage()
	{
		using InMemoryDatabase database = Synthesize(SmallDatasetOptions(), new Reconstruction());

		var stages = new List<GlobalPipelineStage>();
		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(new GlobalPipelineOptions(), database, reconstructionManager)
		{
			Progress = new SynchronousPipelineProgress(p => stages.Add(p.Stage)),
		};
		mapper.Run();

		List<GlobalPipelineStage> startedStages =
			[.. stages.Where(s => s is not GlobalPipelineStage.ModelUpdated and not GlobalPipelineStage.ComponentFinished)];

		await Assert.That(reconstructionManager.Size).IsEqualTo(1);
		await Assert.That(startedStages).IsEquivalentTo(
			[
				GlobalPipelineStage.RotationAveraging,
				GlobalPipelineStage.TrackEstablishment,
				GlobalPipelineStage.GlobalPositioning,
				GlobalPipelineStage.BundleAdjustment,
				GlobalPipelineStage.Retriangulation,
			],
			TUnit.Assertions.Enums.CollectionOrdering.Matching);
		await Assert.That(stages.Contains(GlobalPipelineStage.ModelUpdated)).IsTrue();
		await Assert.That(stages[^1]).IsEqualTo(GlobalPipelineStage.ComponentFinished);
	}

	// C#-only: a cancelled token stops the pipeline before the first component is mapped.
	[Test]
	public async Task CSharpOnly_CancelledTokenMapsNothing()
	{
		using InMemoryDatabase database = Synthesize(SmallDatasetOptions(), new Reconstruction());

		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(new GlobalPipelineOptions(), database, reconstructionManager)
		{
			CancellationToken = new CancellationToken(canceled: true),
		};
		mapper.Run();

		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	// C#-only (divergence 107): two disconnected components of the same
	// size give two reconstructions with the same registered frame count; the stable sort
	// keeps them in component order, so the component with the smallest frame id comes first.
	[Test]
	public async Task CSharpOnly_EqualSizeReconstructionsKeepComponentOrder()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);
		List<HashSet<uint>> components = GroupImageIdsByRig(gt);
		DisconnectDatabaseComponents(components, database);

		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		// The component holding the smallest frame id, and so the smallest image id here
		// (one camera per rig), is the first input component.
		uint smallestImageId = gt.Images.Keys.Min();
		HashSet<uint> firstComponent = components.Single(c => c.Contains(smallestImageId));
		HashSet<uint> secondComponent = components.Single(c => !c.Contains(smallestImageId));

		await Assert.That(reconstructionManager.Size).IsEqualTo(2);
		await Assert.That(reconstructionManager.Get(0).NumRegFrames).IsEqualTo(reconstructionManager.Get(1).NumRegFrames);
		await Assert.That(firstComponent.SetEquals(reconstructionManager.Get(0).RegImageIds())).IsTrue();
		await Assert.That(secondComponent.SetEquals(reconstructionManager.Get(1).RegImageIds())).IsTrue();
	}

	// Progress<T> posts to the thread pool; this one reports inline so the test sees every
	// report before it asserts.
	private sealed class SynchronousPipelineProgress(Action<GlobalPipelineProgress> report) : IProgress<GlobalPipelineProgress>
	{
		public void Report(GlobalPipelineProgress value) => report(value);
	}
}
