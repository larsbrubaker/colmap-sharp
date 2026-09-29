// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only test of
// AutomaticReconstructionOptions.ComputeDevice (divergence 136), not a port.
// The dense stages of the textured scene (AutomaticReconstructionTests.CSharpOnly.cs) run three
// times over one sparse model: on the CPU, on ReferenceComputeDevice (the CPU twin of the WGSL
// kernels, so bit-identical to the CPU), and on a twin that cannot be waited on synchronously,
// which the synchronous controller must not use. All three must write the same depth maps,
// normal maps, fused.ply and Delaunay mesh (deterministic on a resume since divergence 137),
// and the progress must say where PatchMatch ran.

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
	public async Task CSharpOnly_ComputeDeviceGivesTheSameDenseResults()
	{
		const int NumViews = 4;
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var images = new InMemoryImageSource();
		for (int i = 0; i < NumViews; ++i)
		{
			images.Add($"view{i}.png", SyntheticObjectScene.RenderTexturedSphereOnWall(i, NumViews, width: 200, height: 150));
		}

		string densePath = Path.Combine(workspacePath, "dense", "0");

		// Runs the controller (the first time from scratch; later over the same sparse model),
		// returns the dense outputs by relative path and deletes them so the next run redoes
		// PatchMatch and fusion, and the PatchMatch progress messages.
		(Dictionary<string, byte[]> Outputs, List<string> Messages) RunDense(bool resume, ReferenceComputeDevice? device)
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
				Texture = false,
				Extraction = !resume,
				Matching = !resume,
				ComputeDevice = device,
			};
			var messages = new List<string>();
			var controller = new AutomaticReconstructionController(options, new ReconstructionManager())
			{
				Progress = new MessageCollector(messages),
			};
			controller.Setup();
			controller.Run();

			var outputs = new Dictionary<string, byte[]>(StringComparer.Ordinal);
			var files = new List<string> { Path.Combine(densePath, "fused.ply"), Path.Combine(densePath, "meshed-delaunay.ply") };
			foreach (string kind in new[] { "depth_maps", "normal_maps" })
			{
				string folder = Path.Combine(densePath, "stereo", kind);
				files.AddRange(Directory.Exists(folder) ? Directory.GetFiles(folder) : []);
			}

			foreach (string path in files.Where(File.Exists))
			{
				outputs.Add(Path.GetRelativePath(densePath, path), File.ReadAllBytes(path));
				File.Delete(path);
			}

			lock (messages)
			{
				return (outputs, messages.Where(m => m.StartsWith("PatchMatch ", StringComparison.Ordinal)).ToList());
			}
		}

		(Dictionary<string, byte[]> cpu, List<string> cpuMessages) = RunDense(resume: false, device: null);
		var twin = new ReferenceComputeDevice();
		(Dictionary<string, byte[]> gpu, List<string> gpuMessages) = RunDense(resume: true, device: twin);

		Dictionary<string, byte[]> nonBlocking;
		List<string> nonBlockingMessages;
		List<string> warnings;
		var browserTwin = new ReferenceComputeDevice { SupportsBlockingWait = false };
		using (var capture = new LogCapture())
		{
			(nonBlocking, nonBlockingMessages) = RunDense(resume: true, device: browserTwin);
			warnings = capture.Messages.Select(entry => entry.Message).ToList();
		}

		try
		{
			Directory.Delete(testDir, recursive: true);
		}
		catch (IOException)
		{
			// A leftover temp folder is harmless.
		}

		// Guard against a vacuous pass: the CPU run produced a fused cloud and depth maps.
		await Assert.That(cpu.ContainsKey("fused.ply")).IsTrue();
		await Assert.That(cpu.ContainsKey("meshed-delaunay.ply")).IsTrue();
		// Low quality turns geometric consistency off: one photometric map per view.
		await Assert.That(cpu.Keys.Count(k => k.Contains("depth_maps", StringComparison.Ordinal))).IsEqualTo(NumViews);
		await Assert.That(cpuMessages.Count).IsEqualTo(NumViews);
		await Assert.That(cpuMessages.Any(m => m.EndsWith(")", StringComparison.Ordinal))).IsFalse();

		await Assert.That(Differences(gpu, cpu)).IsEqualTo("");
		await Assert.That(twin.ExecutedDispatchCount).IsGreaterThan(0);
		await Assert.That(gpuMessages.Count).IsEqualTo(NumViews);
		await Assert.That(gpuMessages.All(m => m.EndsWith(" (GPU)", StringComparison.Ordinal))).IsTrue();

		await Assert.That(Differences(nonBlocking, cpu)).IsEqualTo("");
		await Assert.That(browserTwin.Commands.Count).IsEqualTo(0);
		await Assert.That(nonBlockingMessages.Any(m => m.EndsWith(")", StringComparison.Ordinal))).IsFalse();
		await Assert.That(warnings.Any(w => w.Contains("on the CPU", StringComparison.Ordinal))).IsTrue();
	}

	private static string Differences(Dictionary<string, byte[]> actual, Dictionary<string, byte[]> expected)
	{
		IEnumerable<string> keys = actual.Keys.Union(expected.Keys).Order(StringComparer.Ordinal);
		return string.Join(", ", keys.Where(k =>
			!actual.TryGetValue(k, out byte[]? a) || !expected.TryGetValue(k, out byte[]? e) || !a.AsSpan().SequenceEqual(e)));
	}

	// Records every report's Message. Locked: the stages report from worker threads.
	private sealed class MessageCollector(List<string> messages) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value)
		{
			lock (messages)
			{
				messages.Add(value.Message);
			}
		}
	}
}
