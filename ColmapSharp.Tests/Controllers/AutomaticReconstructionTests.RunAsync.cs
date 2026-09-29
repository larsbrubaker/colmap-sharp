// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only tests of
// AutomaticReconstructionController.RunAsync, the entry that awaits PatchMatch on the host's
// compute device (divergence 136), not a port. Over the textured scene of
// AutomaticReconstructionTests.CSharpOnly.cs they pin that RunAsync writes the same bytes as
// Run with no device, uses a device that cannot be waited on synchronously (the browser's)
// with the same outputs and no fallback warning, and stops part-way through the dense stage
// exactly where Run does.

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	[NotInParallel(nameof(Log))]
	public async Task CSharpOnly_RunAsyncGivesTheSameResultsAsRunOnAnyDevice()
	{
		string testDir = CreateTestDir();
		InMemoryImageSource images = TexturedSceneImages(NumAsyncViews);

		// Every stage from scratch, textured, in two workspaces: Run, then RunAsync.
		string syncWorkspace = Path.Combine(testDir, "sync");
		string asyncWorkspace = Path.Combine(testDir, "async");
		Directory.CreateDirectory(syncWorkspace);
		Directory.CreateDirectory(asyncWorkspace);
		var syncStages = new StageCollector();
		AutomaticReconstructionController syncController = NewAsyncTestController(syncWorkspace, images, resume: false, device: null, syncStages);
		syncController.Run();
		var asyncStages = new StageCollector();
		AutomaticReconstructionController asyncController = NewAsyncTestController(asyncWorkspace, images, resume: false, device: null, asyncStages);
		await asyncController.RunAsync();
		Dictionary<string, byte[]> syncFiles = WorkspaceFiles(syncWorkspace);
		Dictionary<string, byte[]> asyncFiles = WorkspaceFiles(asyncWorkspace);

		// The dense stages again over the async workspace's sparse model, awaited on a twin that
		// cannot be waited on synchronously, which the synchronous Run would refuse.
		string densePath = Path.Combine(asyncWorkspace, "dense", "0");
		foreach (string kind in new[] { "depth_maps", "normal_maps" })
		{
			Directory.Delete(Path.Combine(densePath, "stereo", kind), recursive: true);
		}

		File.Delete(Path.Combine(densePath, "fused.ply"));
		File.Delete(Path.Combine(densePath, "fused.ply.vis"));
		File.Delete(Path.Combine(densePath, "meshed-delaunay.ply"));
		var browserTwin = new ReferenceComputeDevice { SupportsBlockingWait = false };
		var messages = new List<string>();
		// LogCapture keeps only this run's warnings; a plain sink also collected the export and
		// texturing warnings of tests running in parallel and made the check below flaky.
		List<string> warnings;
		using (var capture = new LogCapture())
		{
			AutomaticReconstructionController gpuController = NewAsyncTestController(
				asyncWorkspace, images, resume: true, device: browserTwin, new MessageCollector(messages));
			await gpuController.RunAsync();
			warnings = capture.Messages.Select(entry => entry.Message).ToList();
		}

		Dictionary<string, byte[]> gpuFiles = WorkspaceFiles(asyncWorkspace);
		List<string> patchMatchMessages;
		lock (messages)
		{
			patchMatchMessages = messages.Where(m => m.StartsWith("PatchMatch ", StringComparison.Ordinal)).ToList();
		}

		DeleteTestDir(testDir);

		// Guard against a vacuous pass: the run got through every stage, texturing included.
		await Assert.That(syncFiles.ContainsKey(Path.Combine("dense", "0", "fused.ply"))).IsTrue();
		await Assert.That(syncFiles.ContainsKey(Path.Combine("dense", "0", "meshed-delaunay.ply"))).IsTrue();
		await Assert.That(syncFiles.ContainsKey(Path.Combine("dense", "0", "meshed-delaunay-textured", "mesh.ply"))).IsTrue();
		await Assert.That(syncFiles.Keys.Count(k => k.StartsWith(Path.Combine("sparse", "0"), StringComparison.Ordinal))).IsGreaterThan(0);
		await Assert.That(syncController.TexturedMeshes.Count).IsEqualTo(1);

		await Assert.That(Differences(asyncFiles, syncFiles)).IsEqualTo("");
		await Assert.That(string.Join(" | ", asyncStages.Stages())).IsEqualTo(string.Join(" | ", syncStages.Stages()));
		await Assert.That(asyncController.TexturedMeshes.Count).IsEqualTo(1);

		await Assert.That(Differences(DenseProducts(gpuFiles), DenseProducts(syncFiles))).IsEqualTo("");
		await Assert.That(browserTwin.ExecutedDispatchCount).IsGreaterThan(0);
		await Assert.That(patchMatchMessages.Count).IsEqualTo(NumAsyncViews);
		await Assert.That(patchMatchMessages.All(m => m.EndsWith(" (GPU)", StringComparison.Ordinal))).IsTrue();
		await Assert.That(string.Join(" | ", warnings)).IsEqualTo("");
	}

	// A host that stops once the first PatchMatch problem is written: Run and RunAsync (the
	// latter awaiting a non-blocking twin) both return normally, keep that problem's maps, run
	// no other problem, and write no fused points, mesh or texture.
	[Test]
	public async Task CSharpOnly_RunAsyncStopsMidDenseLikeRun()
	{
		string testDir = CreateTestDir();
		InMemoryImageSource images = TexturedSceneImages(NumAsyncViews);

		async Task<(Dictionary<string, byte[]> Files, int NumTextured)> RunStopped(bool useAsync)
		{
			string workspacePath = Path.Combine(testDir, useAsync ? "async" : "sync");
			Directory.CreateDirectory(workspacePath);
			using var stop = new CancellationTokenSource();
			ReferenceComputeDevice? device = useAsync ? new ReferenceComputeDevice { SupportsBlockingWait = false } : null;
			AutomaticReconstructionController controller = NewAsyncTestController(
				workspacePath, images, resume: false, device, new StopOnPatchMatch(stop));
			controller.CancellationToken = stop.Token;
			if (useAsync)
			{
				await controller.RunAsync();
			}
			else
			{
				controller.Run();
			}

			return (WorkspaceFiles(workspacePath), controller.TexturedMeshes.Count);
		}

		(Dictionary<string, byte[]> syncFiles, int syncTextured) = await RunStopped(useAsync: false);
		(Dictionary<string, byte[]> asyncFiles, int asyncTextured) = await RunStopped(useAsync: true);
		DeleteTestDir(testDir);

		string depthMaps = Path.Combine("dense", "0", "stereo", "depth_maps");
		await Assert.That(syncFiles.Keys.Count(k => k.StartsWith(depthMaps, StringComparison.Ordinal))).IsEqualTo(1);
		await Assert.That(syncFiles.Keys.Any(k => k.Contains("fused", StringComparison.Ordinal))).IsFalse();
		await Assert.That(syncFiles.Keys.Any(k => k.Contains("meshed", StringComparison.Ordinal))).IsFalse();
		await Assert.That(syncTextured).IsEqualTo(0);

		await Assert.That(string.Join(", ", asyncFiles.Keys.Order(StringComparer.Ordinal)))
			.IsEqualTo(string.Join(", ", syncFiles.Keys.Order(StringComparer.Ordinal)));
		await Assert.That(Differences(asyncFiles, syncFiles)).IsEqualTo("");
		await Assert.That(asyncTextured).IsEqualTo(0);
	}

	private const int NumAsyncViews = 4;

	private static InMemoryImageSource TexturedSceneImages(int numViews)
	{
		var images = new InMemoryImageSource();
		for (int i = 0; i < numViews; ++i)
		{
			images.Add($"view{i}.png", SyntheticObjectScene.RenderTexturedSphereOnWall(i, numViews, width: 200, height: 150));
		}

		return images;
	}

	// The textured scene's dense, textured Delaunay reconstruction; resume skips extraction and
	// matching (the sparse model is read back from the workspace).
	private static AutomaticReconstructionController NewAsyncTestController(
		string workspacePath, InMemoryImageSource images, bool resume, ReferenceComputeDevice? device, IProgress<ControllerProgress> progress)
	{
		var options = new AutomaticReconstructionOptions
		{
			WorkspacePath = workspacePath,
			Images = images,
			Data = AutomaticReconstructionOptions.DataType.Individual,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			Dense = true,
			RandomSeed = 1,
			Mesher = AutomaticReconstructionOptions.MesherType.Delaunay,
			TextureSink = new InMemoryBitmapStore(),
			Extraction = !resume,
			Matching = !resume,
			ComputeDevice = device,
		};
		var controller = new AutomaticReconstructionController(options, new ReconstructionManager())
		{
			Progress = progress,
		};
		controller.Setup();
		return controller;
	}

	// The maps, fused points and meshes of a workspace's files. A resumed run reads its sparse
	// model back from disk, whose images come in a different order than in the run that mapped
	// them, so it re-undistorts to a differently ordered dense/<i>/sparse and stereo/*.cfg
	// (with or without RunAsync); what PatchMatch and the later stages write is keyed by image
	// name and does not depend on that order.
	private static Dictionary<string, byte[]> DenseProducts(Dictionary<string, byte[]> files) =>
		files.Where(f => f.Key.StartsWith(Path.Combine("dense", "0") + Path.DirectorySeparatorChar, StringComparison.Ordinal)
				&& !f.Key.StartsWith(Path.Combine("dense", "0", "sparse"), StringComparison.Ordinal)
				&& !f.Key.EndsWith(".cfg", StringComparison.Ordinal))
			.ToDictionary(f => f.Key, f => f.Value, StringComparer.Ordinal);

	// Every file under the workspace, by path relative to it.
	private static Dictionary<string, byte[]> WorkspaceFiles(string workspacePath) =>
		Directory.GetFiles(workspacePath, "*", SearchOption.AllDirectories)
			.ToDictionary(path => Path.GetRelativePath(workspacePath, path), File.ReadAllBytes, StringComparer.Ordinal);

	private static void DeleteTestDir(string testDir)
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

	// Cancels on the first PatchMatch report, which comes right after the first problem's maps
	// are written. Synchronous (not Progress<T>), so the stop lands before the next problem.
	private sealed class StopOnPatchMatch(CancellationTokenSource stop) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value)
		{
			if (value.Message.StartsWith("PatchMatch ", StringComparison.Ordinal))
			{
				stop.Cancel();
			}
		}
	}
}
