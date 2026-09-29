// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ObjectModeEndToEndTests: C#-only (not a port; COLMAP has no object mode). The whole pipeline
// on a small TexturedSphere with the benchmark's realistic motion (motionDuration 0.27), once as
// a Scene and once as an Object (AutomaticReconstructionOptions.Subject; divergence 142): the
// object run fills the fused cloud's gaps from the visual hull, drops pieces outside the
// silhouettes and pulls the Poisson surface inside the hull (AutomaticReconstruction.Object.cs).
// It must give a closed mesh of the dense points (not the hull fallback) and a better F-score
// than the scene run, through SyntheticBenchmark, the runner's own measurement. A second test
// stops an object run between Poisson and the clean-up and resumes it (divergence 135's rule).
//
// Both runs get the true intrinsics (KnownIntrinsics). Self-calibration collapses the focal on
// this small object (divergence 141; ratio 0.87 here), and then no hull or surface can match the
// truth (F about 0.02 either way), so the comparison would say nothing. The object run finds
// its own masks (SilhouetteSegmenter), the real Object-mode path; the scene run gets the scene's
// true masks (features and fusion), the best masks it could have, so the object run's gain is
// not a masking gain. Release numbers: object F 0.411 (16/20 placed, mask IoU >= 0.984 against
// the truth); scene F 0.274 with true masks, 0.313 without.

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class ObjectModeEndToEndTests
{
	[Test]
	public async Task CSharpOnly_ObjectModeGivesAClosedMeshThatBeatsSceneMode()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(
			SyntheticObjectKind.TexturedSphere, 20, 320, 240, seed: 1, motionDuration: 0.27);

		SyntheticBenchmarkResult Run(AutomaticReconstructionOptions.SubjectType subject)
		{
			var options = new SyntheticBenchmarkOptions
			{
				MapperSeed = 1,
				Subject = subject,
				KnownIntrinsics = true,
				UseTrueMasks = subject == AutomaticReconstructionOptions.SubjectType.Scene,
			};
			options.Surface.SampleCount = 5000;
			return SyntheticBenchmark.Run(scene, options);
		}

		SyntheticBenchmarkResult sceneRun = Run(AutomaticReconstructionOptions.SubjectType.Scene);
		SyntheticBenchmarkResult objectRun = Run(AutomaticReconstructionOptions.SubjectType.Object);
		Console.WriteLine(
			$"scene: {sceneRun.Metrics.NumRegistered} registered, F {sceneRun.Metrics.Surface.FScore:F3}, IoU {sceneRun.Metrics.SilhouetteIouMean:F3}, closed {sceneRun.MeshClosed}; "
			+ $"object: {objectRun.Metrics.NumRegistered} registered, F {objectRun.Metrics.Surface.FScore:F3}, IoU {objectRun.Metrics.SilhouetteIouMean:F3}, closed {objectRun.MeshClosed}");

		await Assert.That(objectRun.Metrics.SurfaceSource).IsEqualTo("mesh");
		await Assert.That(objectRun.MeshClosed).IsTrue();
		await Assert.That(objectRun.MeshIsVisualHull).IsFalse();
		await Assert.That(objectRun.Metrics.Surface.FScore).IsGreaterThan(sceneRun.Metrics.Surface.FScore);
	}

	// A run stopped after Poisson but before the object clean-up must not leave a mesh file that
	// a resume would take for a finished one, and the resumed run must finish the model.
	[Test]
	public async Task CSharpOnly_ObjectModeStoppedAfterPoissonLeavesNoMeshAndResumes()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(
			SyntheticObjectKind.TexturedSphere, 16, 240, 180, seed: 1, motionDuration: 0.27);
		string workspace = Path.Combine(Path.GetTempPath(), "colmap-sharp-object-resume-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(workspace);
		try
		{
			using var cts = new CancellationTokenSource();
			var stopAfterPoisson = new StopWhen(
				p => p.Stage == AutomaticReconstructionController.MeshingStage && p.Total > 0 && p.Done >= p.Total, cts);
			SyntheticBenchmarkOptions Options(IProgress<ControllerProgress>? progress) => new()
			{
				MapperSeed = 1,
				Subject = AutomaticReconstructionOptions.SubjectType.Object,
				KnownIntrinsics = true,
				WorkspacePath = workspace,
				Progress = progress,
			};

			string denseModel = Path.Combine(workspace, "dense", "0");
			string meshPath = Path.Combine(denseModel, "meshed-poisson.ply");
			bool cancelled = false;
			try
			{
				SyntheticBenchmark.Run(scene, Options(stopAfterPoisson), cts.Token);
			}
			catch (OperationCanceledException)
			{
				cancelled = true;
			}

			await Assert.That(stopAfterPoisson.Fired).IsTrue();
			await Assert.That(cancelled).IsTrue();
			await Assert.That(File.Exists(meshPath)).IsFalse();
			await Assert.That(Directory.GetFiles(denseModel, "*partial*").Length).IsEqualTo(0);

			SyntheticBenchmarkResult resumed = SyntheticBenchmark.Run(scene, Options(null));
			await Assert.That(File.Exists(meshPath)).IsTrue();
			await Assert.That(resumed.MeshClosed).IsTrue();
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	// Cancels once a report matches, synchronously on the reporting thread.
	private sealed class StopWhen(Func<ControllerProgress, bool> condition, CancellationTokenSource cts) : IProgress<ControllerProgress>
	{
		public bool Fired { get; private set; }

		public void Report(ControllerProgress value)
		{
			if (!Fired && condition(value))
			{
				Fired = true;
				cts.Cancel();
			}
		}
	}
}
