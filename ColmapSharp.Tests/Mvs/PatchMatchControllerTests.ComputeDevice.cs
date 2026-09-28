// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchControllerTests (continued): C#-only tests of PatchMatchController.ComputeDevice
// (ColmapSharp/Mvs/PatchMatchController.cs; docs/CPP_DIVERGENCES.md entry 136). COLMAP spreads
// problems over CUDA devices by index; here the host hands the controller one compute device.
// The device is ReferenceComputeDevice (ColmapSharp/Mvs/Testing/), the CPU twin of the WGSL
// kernels, so every map the controller writes with it must be byte-identical to the maps it
// writes without one. The workspace is PatchMatchControllerTests' Fixture, with its flat
// bitmaps replaced by a pattern so the maps carry more than a degenerate constant.

using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PatchMatchControllerTests
{
	[Test]
	public async Task RunAsync_WithTwinDevice_WritesTheSameMapsAsTheCpu()
	{
		var fixture = PatternedFixture();
		Directory.CreateDirectory(Path.Combine(fixture.TempDir, "stereo", "consistency_graphs"));
		fixture.WriteConfig(fixture.Names[0], "__all__", fixture.Names[1], "__all__", fixture.Names[2], "__all__", fixture.Names[3], "__all__");

		new PatchMatchController(RunOptions(true), fixture.TempDir, "COLMAP", "", fixture.Bitmaps).Run();
		Dictionary<string, byte[]> cpuMaps = TakeMaps(fixture);

		var reports = new List<ControllerProgress>();
		var device = new ReferenceComputeDevice();
		var controller = new PatchMatchController(RunOptions(true), fixture.TempDir, "COLMAP", "", fixture.Bitmaps)
		{
			ComputeDevice = device,
		};
		await controller.RunAsync(progress: new SynchronousProgress<ControllerProgress>(reports));
		Dictionary<string, byte[]> gpuMaps = TakeMaps(fixture);

		// Photometric and geometric depth, normal and consistency-graph files for four images.
		await Assert.That(cpuMaps.Count).IsEqualTo(24);
		await Assert.That(string.Join(",", gpuMaps.Keys.Order())).IsEqualTo(string.Join(",", cpuMaps.Keys.Order()));
		var differing = cpuMaps.Keys.Where(k => !cpuMaps[k].AsSpan().SequenceEqual(gpuMaps[k])).Order().ToList();
		await Assert.That(string.Join(",", differing)).IsEqualTo("");

		await Assert.That(device.ExecutedDispatchCount).IsGreaterThan(0);
		await Assert.That(reports.Count).IsEqualTo(8);
		await Assert.That(reports[0].Message).IsEqualTo(fixture.Names[0] + " (GPU)");
		await Assert.That(reports.All(r => r.Message.EndsWith(" (GPU)", StringComparison.Ordinal))).IsTrue();
	}

	[Test]
	[NotInParallel(nameof(Log))]
	public async Task RunAsync_DeviceTooSmall_FallsBackToCpuAndSaysSo()
	{
		var fixture = PatternedFixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");

		new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps).Run();
		Dictionary<string, byte[]> cpuMaps = TakeMaps(fixture);

		var reports = new List<ControllerProgress>();
		var device = new ReferenceComputeDevice(ComputeDeviceLimits.Defaults with { MaxStorageBufferBindingSize = 256 });
		var controller = new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps)
		{
			ComputeDevice = device,
		};

		IReadOnlyList<(LogLevel Level, string Message)> messages;
		using (var capture = new LogCapture())
		{
			await controller.RunAsync(progress: new SynchronousProgress<ControllerProgress>(reports));
			messages = capture.Messages;
		}

		Dictionary<string, byte[]> fallbackMaps = TakeMaps(fixture);

		// The warning names the problem and carries the planner's reason, which ends the
		// sentence a user reads.
		string prefix = "PatchMatch photometric " + fixture.Names[0] + ": ";
		await Assert.That(messages).Contains(m => m.Level == LogLevel.Warning
			&& m.Message.StartsWith(prefix, StringComparison.Ordinal)
			&& m.Message.EndsWith(" Using the CPU.", StringComparison.Ordinal));
		await Assert.That(device.Commands.Count).IsEqualTo(0);
		await Assert.That(reports.Count).IsEqualTo(1);
		await Assert.That(reports[0].Message).IsEqualTo(fixture.Names[0] + " (CPU)");
		await Assert.That(fallbackMaps.Count).IsEqualTo(2);
		await Assert.That(fallbackMaps.Keys.All(k => cpuMaps[k].AsSpan().SequenceEqual(fallbackMaps[k]))).IsTrue();
	}

	[Test]
	public async Task RunAsync_WithDevice_CancelledMidProblemWritesNothingForIt()
	{
		// As Run_CancelledMidProblemWritesNothingForIt, on the device (divergence 122): the
		// token reaches the GPU run, which stops before submitting any work for the aborted
		// problem; no exception escapes and the finished problem's maps stay.
		var fixture = PatternedFixture();
		fixture.WriteConfig(fixture.Names[0], "__all__", fixture.Names[1], "__all__");
		using var cancellation = new CancellationTokenSource();
		var device = new ReferenceComputeDevice();
		var controller = new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps)
		{
			ComputeDevice = device,
		};
		int started = 0;
		int dispatchesBeforeSecond = -1;
		controller.ProblemRunning = problemIdx =>
		{
			started++;
			if (problemIdx == 1)
			{
				dispatchesBeforeSecond = device.ExecutedDispatchCount;
				cancellation.Cancel();
			}
		};

		await controller.RunAsync(cancellation.Token);

		await Assert.That(started).IsEqualTo(2);
		await Assert.That(dispatchesBeforeSecond).IsGreaterThan(0);
		await Assert.That(device.ExecutedDispatchCount).IsEqualTo(dispatchesBeforeSecond);
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 0, "photometric"))).IsTrue();
		await Assert.That(File.Exists(MapPath(fixture, "normal_maps", 0, "photometric"))).IsTrue();
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 1, "photometric"))).IsFalse();
		await Assert.That(File.Exists(MapPath(fixture, "normal_maps", 1, "photometric"))).IsFalse();
	}

	[Test]
	public async Task Run_WithNonBlockingDevice_ThrowsBeforeWritingAnything()
	{
		var fixture = PatternedFixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");
		var device = new ReferenceComputeDevice { SupportsBlockingWait = false };
		var controller = new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps)
		{
			ComputeDevice = device,
		};

		InvalidOperationException? error = null;
		try
		{
			controller.Run();
		}
		catch (InvalidOperationException e)
		{
			error = e;
		}

		await Assert.That(error).IsNotNull();
		await Assert.That(error!.Message.Contains("RunAsync", StringComparison.Ordinal)).IsTrue();
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 0, "photometric"))).IsFalse();

		// The same device works through RunAsync.
		await controller.RunAsync();
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 0, "photometric"))).IsTrue();
		await Assert.That(device.ExecutedDispatchCount).IsGreaterThan(0);
	}

	// The fixture with each image a distinct diagonal pattern instead of a flat grey.
	private static Fixture PatternedFixture()
	{
		var fixture = new Fixture();
		for (int imageIdx = 0; imageIdx < fixture.Names.Count; ++imageIdx)
		{
			var bitmap = new Bitmap(CameraWidth, CameraHeight, asRgb: false);
			for (int y = 0; y < CameraHeight; ++y)
			{
				for (int x = 0; x < CameraWidth; ++x)
				{
					bitmap.SetPixel(x, y, new BitmapColor<byte>((byte)((x * 37 + y * 11 + imageIdx * 5) % 251)));
				}
			}

			fixture.Bitmaps.Add(Path.Combine(fixture.TempDir, "images", fixture.Names[imageIdx]), bitmap);
		}

		return fixture;
	}

	// Every file under stereo/{depth_maps,normal_maps,consistency_graphs}, by relative path,
	// deleted afterwards so the next run recomputes them rather than skipping.
	private static Dictionary<string, byte[]> TakeMaps(Fixture fixture)
	{
		var maps = new Dictionary<string, byte[]>(StringComparer.Ordinal);
		foreach (string kind in new[] { "depth_maps", "normal_maps", "consistency_graphs" })
		{
			string folder = Path.Combine(fixture.TempDir, "stereo", kind);
			if (!Directory.Exists(folder))
			{
				continue;
			}

			foreach (string path in Directory.GetFiles(folder))
			{
				maps.Add(kind + "/" + Path.GetFileName(path), File.ReadAllBytes(path));
				File.Delete(path);
			}
		}

		return maps;
	}
}
