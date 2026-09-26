// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipelineTests.Host: C#-only tests (labeled as such; none stands in for a ported
// case) of what MatterCAD needs from ColmapSharp/Controllers/IncrementalPipeline*.cs beyond
// COLMAP's tests: cancellation through the CancellationToken, IProgress reports and the
// COLMAP callbacks, and point colors from host-decoded images (ReadImage, which replaces
// COLMAP's image_path).

using ColmapSharp.Controllers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class IncrementalPipelineTests
{
	// Collects reports synchronously (Progress<T> would post to the thread pool).
	private sealed class ListProgress : IProgress<IncrementalPipelineProgress>
	{
		public List<IncrementalPipelineProgress> Reports { get; } = [];

		public void Report(IncrementalPipelineProgress value) => Reports.Add(value);
	}

	private static SyntheticDatasetOptions SmallDataset() =>
		new() { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50, CameraHasPriorFocalLength = false };

	// C#-only.
	[Test]
	public async Task IncrementalPipeline_ReportsProgressAndCallbacks()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(SmallDataset(), gt);

		var manager = new ReconstructionManager();
		var pipeline = new IncrementalPipeline(new IncrementalPipelineOptions(), database, manager);
		var progress = new ListProgress();
		pipeline.Progress = progress;
		int initialCallbacks = 0;
		int nextCallbacks = 0;
		int lastCallbacks = 0;
		pipeline.AddCallback((int)IncrementalPipeline.CallbackType.InitialImagePairRegCallback, () => ++initialCallbacks);
		pipeline.AddCallback((int)IncrementalPipeline.CallbackType.NextImageRegCallback, () => ++nextCallbacks);
		pipeline.AddCallback((int)IncrementalPipeline.CallbackType.LastImageRegCallback, () => ++lastCallbacks);
		pipeline.Run();

		int numImages = gt.NumImages;
		List<IncrementalPipelineProgress> reports = progress.Reports;
		int imageReports = reports.Count(r => r.Stage == IncrementalPipelineStage.ImageRegistered);
		IncrementalPipelineProgress last = reports[^1];
		bool monotonic = reports.Zip(reports.Skip(1)).All(p => p.First.NumRegImages <= p.Second.NumRegImages);

		await Assert.That(manager.Size).IsEqualTo(1);
		await Assert.That(initialCallbacks).IsEqualTo(1);
		// The initial pair registers two images; every other image fires one next-image callback.
		await Assert.That(nextCallbacks).IsEqualTo(numImages - 2);
		await Assert.That(lastCallbacks).IsEqualTo(1);
		await Assert.That(imageReports).IsEqualTo(nextCallbacks);
		await Assert.That(reports[0].Stage).IsEqualTo(IncrementalPipelineStage.InitialPairRegistered);
		await Assert.That(reports[0].NumRegImages).IsEqualTo(2);
		await Assert.That(last.Stage).IsEqualTo(IncrementalPipelineStage.ModelFinished);
		await Assert.That(last.NumRegImages).IsEqualTo(numImages);
		await Assert.That(last.NumImages).IsEqualTo(numImages);
		await Assert.That(last.NumModels).IsEqualTo(1);
		await Assert.That(monotonic).IsTrue();
	}

	// C#-only: cancelling between registrations keeps the partial model, as COLMAP keeps it on
	// an interrupt.
	[Test]
	public async Task IncrementalPipeline_CancellationStopsBetweenRegistrations()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(SmallDataset(), gt);

		using var cts = new CancellationTokenSource();
		var manager = new ReconstructionManager();
		var pipeline = new IncrementalPipeline(new IncrementalPipelineOptions(), database, manager)
		{
			CancellationToken = cts.Token,
		};
		int nextCallbacks = 0;
		pipeline.AddCallback(
			(int)IncrementalPipeline.CallbackType.NextImageRegCallback,
			() =>
			{
				// Cancel once the first image after the initial pair is registered.
				if (++nextCallbacks == 1)
				{
					cts.Cancel();
				}
			});
		pipeline.Run();

		int numRegImages = manager.Size == 1 ? manager.Get(0).NumRegImages : -1;
		await Assert.That(manager.Size).IsEqualTo(1);
		await Assert.That(nextCallbacks).IsEqualTo(1);
		await Assert.That(numRegImages).IsEqualTo(3);
	}

	// C#-only: cancelled before the run, nothing is reconstructed.
	[Test]
	public async Task IncrementalPipeline_CancelledBeforeRunKeepsNothing()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(SmallDataset(), gt);

		using var cts = new CancellationTokenSource();
		cts.Cancel();
		var manager = new ReconstructionManager();
		var pipeline = new IncrementalPipeline(new IncrementalPipelineOptions(), database, manager)
		{
			CancellationToken = cts.Token,
		};
		pipeline.Run();

		await Assert.That(manager.Size).IsEqualTo(0);
	}

	// C#-only: ReadImage supplies the decoded images the points take their colors from.
	[Test]
	public async Task IncrementalPipeline_ExtractsColorsFromReadImage()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(SmallDataset(), gt);

		var requestedNames = new HashSet<string>();
		var options = new IncrementalPipelineOptions
		{
			ReadImage = name =>
			{
				requestedNames.Add(name);
				var bitmap = new Bitmap(1024, 768, asRgb: true);
				bitmap.Fill(new BitmapColor<byte>(10, 20, 30));
				return bitmap;
			},
		};
		ReconstructionManager manager = RunPipeline(options, database);
		Require(manager.Size == 1);
		Reconstruction reconstruction = manager.Get(0);
		int numColored = reconstruction.Points3D.Values.Count(p => p.Color == new Vector3ub(10, 20, 30));
		int numPoints = reconstruction.NumPoints3D;

		await Assert.That(requestedNames.Count).IsEqualTo(gt.NumImages);
		await Assert.That(numPoints).IsGreaterThan(0);
		await Assert.That(numColored).IsEqualTo(numPoints);
	}
}
