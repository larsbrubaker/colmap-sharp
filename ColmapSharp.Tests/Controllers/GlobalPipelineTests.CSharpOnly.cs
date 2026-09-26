// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GlobalPipelineTests.CSharpOnly: C#-only cases for ColmapSharp/Controllers/GlobalPipeline.cs
// (COLMAP has no progress reports and no CancellationToken): the stage progress and
// cancellation through BaseController.CancellationToken.

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
		RandomUtils.SetPRNGSeed(0);
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
		RandomUtils.SetPRNGSeed(0);
		using InMemoryDatabase database = Synthesize(SmallDatasetOptions(), new Reconstruction());

		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(new GlobalPipelineOptions(), database, reconstructionManager)
		{
			CancellationToken = new CancellationToken(canceled: true),
		};
		mapper.Run();

		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	// Progress<T> posts to the thread pool; this one reports inline so the test sees every
	// report before it asserts.
	private sealed class SynchronousPipelineProgress(Action<GlobalPipelineProgress> report) : IProgress<GlobalPipelineProgress>
	{
		public void Report(GlobalPipelineProgress value) => report(value);
	}
}
