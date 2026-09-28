// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of ReconstructionSession's GPU fallback (demo/ColmapDemo/ReconstructionSession.cs): when
// the compute device faults in the dense stage, the session says so (GpuFailed) and finishes on the
// CPU from the workspace the first attempt left, keeping the depth maps the GPU already wrote. The
// photos are the browser head's bundled samples (ColmapDemo.Browser/wwwroot/samples), shrunk to
// keep two whole runs quick.
//
// The device fails only after the first PatchMatch problem's maps were read back, so the retry
// mixes GPU and CPU depth maps. It wraps ReferenceComputeDevice, the bit-identical CPU twin of the
// WGSL kernels (see the library's AutomaticReconstructionTests.ComputeDevice.cs), so that mix still
// gives exactly the plain CPU run's mesh; on a real GPU it would only come close. The two sessions
// run on whatever pool threads the awaits land on, so this test catches an unseeded run that
// depends on what those threads drew before only when the scheduling happens to expose it. The
// library test AutomaticReconstructionTests.CSharpOnly_UnseededSparseModelIgnoresTheCallersPrng
// pins that property (docs/CPP_DIVERGENCES.md entry 138).

using ColmapSharp.Compute;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Util;

namespace ColmapDemo.Tests;

public class SessionGpuFallbackTests
{
	private static readonly string SampleDir = Path.Combine(AppContext.BaseDirectory, "samples");

	private static IReadOnlyList<string> Samples() =>
		Directory.GetFiles(SampleDir, "view*.png").OrderBy(p => p, StringComparer.Ordinal).ToList();

	private static async Task<(SessionResult Result, Exception GpuFault, List<string> Stages)> RunAsync(IComputeDevice device, string workspaceRoot)
	{
		var session = new ReconstructionSession(new SessionSettings
		{
			ComputeDevice = device,
			MaxImageSize = 240,
			WorkspaceRoot = workspaceRoot,
			KeepWorkspace = true,
		});
		Exception fault = null;
		var stages = new List<string>();
		session.GpuFailed += e => fault = e;
		session.ProgressChanged += p =>
		{
			if (stages.Count == 0 || stages[^1] != p.Stage)
			{
				stages.Add(p.Stage);
			}
		};
		SessionResult result = await session.RunAsync(Samples(), CancellationToken.None);
		return (result, fault, stages);
	}

	[Test]
	public async Task AGpuFaultInTheDenseStageFinishesOnTheCpuWithTheSameMesh()
	{
		string root = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		try
		{
			var device = new FaultingDevice();
			(SessionResult cpu, Exception cpuFault, _) = await RunAsync(null, Path.Combine(root, "cpu"));
			(SessionResult fallback, Exception fault, List<string> stages) = await RunAsync(device, Path.Combine(root, "fallback"));

			await Assert.That(cpuFault).IsNull();
			await Assert.That(fault?.Message).IsEqualTo(FaultingDevice.Message);
			await Assert.That(cpu.Mesh).IsNotNull();
			await Assert.That(fallback.Mesh).IsNotNull();

			// The retry reports only the stages from dense on: the stage list never goes backwards.
			await Assert.That(stages.Count(s => s == "Feature extraction")).IsEqualTo(1);
			await Assert.That(stages.Count(s => s == "Feature matching")).IsEqualTo(1);
			await Assert.That(stages.Count(s => s == "Sparse reconstruction")).IsEqualTo(1);

			// The maps the GPU finished before the fault are kept, and the CPU wrote the rest.
			string[] depthMaps = Directory.GetFiles(
				Path.Combine(fallback.WorkspacePath, "dense", "0", "stereo", "depth_maps"), "*.bin");
			int keptFromGpu = depthMaps.Count(f => File.GetLastWriteTimeUtc(f) < device.FaultTimeUtc);
			await Assert.That(keptFromGpu).IsGreaterThan(0);
			await Assert.That(keptFromGpu).IsLessThan(depthMaps.Length);

			await Assert.That(fallback.Mesh.Faces.Count).IsEqualTo(cpu.Mesh.Faces.Count);
			await Assert.That(fallback.Mesh.Vertices.Count).IsEqualTo(cpu.Mesh.Vertices.Count);
			bool sameVertices = true;
			for (int i = 0; i < cpu.Mesh.Vertices.Count; i++)
			{
				PlyMeshVertex a = cpu.Mesh.Vertices[i], b = fallback.Mesh.Vertices[i];
				sameVertices &= a.X == b.X && a.Y == b.Y && a.Z == b.Z;
			}

			await Assert.That(sameVertices).IsTrue();
			await Assert.That(fallback.IsTextured).IsEqualTo(cpu.IsTextured);
		}
		finally
		{
			Directory.Delete(root, recursive: true);
		}
	}

	// A GPU that works through the first PatchMatch problem, maps read back included, then fails
	// its next submit, as the browser's did with a validation error.
	private sealed class FaultingDevice : IComputeDevice
	{
		public const string Message = "The GPU reported an error running compute work: Validation: test";

		private readonly ReferenceComputeDevice inner = new ReferenceComputeDevice();

		private bool readBack;

		/// <summary>When the first submit failed; maps written before it came from the GPU.</summary>
		public DateTime FaultTimeUtc { get; private set; } = DateTime.MaxValue;

		public ComputeDeviceLimits Limits => this.inner.Limits;

		public bool SupportsBlockingWait => this.inner.SupportsBlockingWait;

		public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string label = null)
			=> this.inner.CreateBuffer(kind, size, initial, label);

		public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data) => this.inner.WriteBuffer(buffer, offset, data);

		public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor) => this.inner.CreateKernel(descriptor);

		public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
			=> this.inner.CreateBindGroup(kernel, group, entries);

		public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
			=> this.inner.Dispatch(kernel, groups, x, y, z);

		public ValueTask FlushAsync(CancellationToken cancellationToken = default)
			=> this.readBack ? this.Fail() : this.inner.FlushAsync(cancellationToken);

		public ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
		{
			if (this.FaultTimeUtc != DateTime.MaxValue)
			{
				return this.Fail();
			}

			this.readBack = true;
			return this.inner.ReadBufferAsync(buffer, offset, destination, cancellationToken);
		}

		private ValueTask Fail()
		{
			if (this.FaultTimeUtc == DateTime.MaxValue)
			{
				this.FaultTimeUtc = DateTime.UtcNow;
			}

			return ValueTask.FromException(new InvalidOperationException(Message));
		}
	}
}
