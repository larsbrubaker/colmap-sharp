// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReferenceComputeDeviceTests: C#-only tests of the CPU twin of the GPU PatchMatch kernels
// (ColmapSharp/Mvs/Testing/ReferenceComputeDevice*.cs). Each kernel twin, run through the
// compute seam on buffers packed as the WGSL lays them out (ReferenceComputeDeviceFixture),
// must give bit-identical results to the direct CPU call it stands for (Tier A: same code, same
// inputs). Also pinned: unknown kernels are rejected, buffers without initial data start as
// garbage, dispatches that reach only some elements leave the rest unwritten, and a kernel that
// would index past a binding faults the flush.

using System.Runtime.InteropServices;

using ColmapSharp.Compute;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Mvs.ReferenceComputeDeviceFixture;

namespace ColmapSharp.Tests.Mvs;

public class ReferenceComputeDeviceTests
{
	[Test]
	public async Task InitRandom_MatchesCpuInit()
	{
		var f = new ReferenceComputeDeviceFixture();
		await RunInitRandom(f);

		float[] state = await f.ReadFloats(f.State, 4 * PlaneSize);
		var cpu = new PatchMatchCpu(f.Options, f.Problem, Seed);
		await AssertBits(state[..PlaneSize], cpu.GetDepthMap().Data);
		await AssertBits(state[PlaneSize..], cpu.GetNormalMap().Data);
	}

	[Test]
	public async Task InitialCost_MatchesComputeInitialCost()
	{
		var f = new ReferenceComputeDeviceFixture();
		await RunInitRandom(f);
		uint groups = Groups((long)PlaneSize * NumSrc);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.InitialCost, groups);
		IComputeBindGroup group0 = f.Group(
			kernel, 0, (0, f.ByteToUnit), (1, f.Poses), (2, f.ProblemUniform), (3, f.Reference), (4, f.SourceImages),
			(5, f.SourceDepths), (6, f.State), (7, f.Costs));
		f.Device.Dispatch(kernel, [group0], groups);
		float[] costs = await f.ReadFloats(f.Costs, NumSrc * PlaneSize);

		var cpu = new PatchMatchCpu(f.Options, f.Problem, Seed);
		var expected = new Mat<float>(Width, Height, NumSrc);
		var pcc = new PatchMatchPhotoConsistency(
			f.RefImage, f.SrcImages, f.Transforms.Poses(0), new PatchMatchFrame(f.Transforms, 0), f.Options.WindowRadius,
			f.Options.WindowStep, (float)f.Options.SigmaSpatial, (float)f.Options.SigmaColor);
		PatchMatchKernel.ComputeInitialCost(expected, cpu.GetDepthMap(), cpu.GetNormalMap(), pcc, 1);
		await AssertBits(costs, expected.Data);
	}

	[Test]
	public async Task RotatePlanes_MatchesMatRotateAtWordOffsets()
	{
		var f = new ReferenceComputeDeviceFixture();
		const int SrcOffset = 5;
		const int DstOffset = 3;
		var planes = new Mat<float>(Width, Height, 2);
		for (int i = 0; i < planes.Data.Length; ++i)
		{
			planes.Data[i] = MathF.Sin(i) * 100;
		}

		float[] srcWords = [.. new float[SrcOffset], .. planes.Data];
		IComputeBuffer src = f.Device.CreateBuffer(ComputeBufferKind.Storage, 4 * srcWords.Length, MemoryMarshal.AsBytes(srcWords.AsSpan()));
		IComputeBuffer dst = f.Device.CreateBuffer(ComputeBufferKind.Storage, 4 * (DstOffset + planes.Data.Length + 2));
		IComputeBuffer rotate = f.Uniform(Width, Height, 2, SrcOffset, DstOffset, 0, 0, 0);
		uint groups = Groups(2 * PlaneSize);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.RotatePlanes, groups);
		f.Device.Dispatch(kernel, [f.Group(kernel, 0, (0, src), (1, dst)), f.Group(kernel, 1, (0, rotate))], groups);
		float[] result = await f.ReadFloats(dst, DstOffset + planes.Data.Length + 2);

		var expected = new Mat<float>(Height, Width, 2);
		planes.Rotate(expected, 1);
		await AssertBits(result[DstOffset..^2], expected.Data);

		// Words outside the planes are not written: they keep the garbage word.
		await Assert.That(BitConverter.SingleToUInt32Bits(result[0])).IsEqualTo(ReferenceComputeDevice.GarbageWord);
		await Assert.That(BitConverter.SingleToUInt32Bits(result[^1])).IsEqualTo(ReferenceComputeDevice.GarbageWord);
	}

	[Test]
	public async Task RotateNormals_MatchesRotateNormalMapThenRotate()
	{
		var f = new ReferenceComputeDeviceFixture();
		const int Offset = 7;
		var normals = new Mat<float>(Width, Height, 3);
		for (int i = 0; i < normals.Data.Length; ++i)
		{
			normals.Data[i] = MathF.Cos(i * 0.7f) - (i % 5 == 0 ? 0.0f : 0.25f);
		}

		normals.Data[4] = 0.0f;
		float[] srcWords = [.. new float[Offset], .. normals.Data];
		IComputeBuffer src = f.Device.CreateBuffer(ComputeBufferKind.Storage, 4 * srcWords.Length, MemoryMarshal.AsBytes(srcWords.AsSpan()));
		IComputeBuffer dst = f.Device.CreateBuffer(ComputeBufferKind.Storage, 4 * (Offset + normals.Data.Length));
		IComputeBuffer rotate = f.Uniform(Width, Height, 3, Offset, Offset, 0, 0, 0);
		uint groups = Groups(PlaneSize);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.RotateNormals, groups);
		f.Device.Dispatch(kernel, [f.Group(kernel, 0, (0, src), (1, dst)), f.Group(kernel, 1, (0, rotate))], groups);
		float[] result = await f.ReadFloats(dst, Offset + normals.Data.Length);

		PatchMatchKernel.RotateNormalMap(normals, 1);
		var expected = new Mat<float>(Height, Width, 3);
		normals.Rotate(expected, 1);
		await AssertBits(result[Offset..], expected.Data);
	}

	[Test]
	public async Task FilterPixels_WritesZeroOrOneToEveryMaskEntry()
	{
		var f = new ReferenceComputeDeviceFixture();
		await RunInitRandom(f);
		float[] selProbs = Enumerable.Range(0, NumSrc * PlaneSize).Select(i => (i * 37 % 101) / 100.0f).ToArray();
		IComputeBuffer sel = f.Device.CreateBuffer(ComputeBufferKind.Storage, 4 * selProbs.Length, MemoryMarshal.AsBytes(selProbs.AsSpan()));

		// The last sweep of one iteration runs in rotation 3; photometric filtering only.
		IComputeBuffer sweep = f.Uniform(3, 0, 0, 0, 0, 1, 0, 0);
		uint groups = Groups(PlaneSize);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.FilterPixels, groups);
		IComputeBindGroup group0 = f.Group(
			kernel, 0, (0, f.ByteToUnit), (1, f.Poses), (2, f.ProblemUniform), (4, f.SourceImages), (5, f.SourceDepths),
			(6, f.State), (7, f.Costs), (8, sel));
		f.Device.Dispatch(kernel, [group0, f.Group(kernel, 1, (0, sweep))], groups);

		byte[] maskBytes = new byte[4 * NumSrc * PlaneSize];
		await f.Device.ReadBufferAsync(f.Costs, 0, maskBytes);
		uint[] mask = MemoryMarshal.Cast<byte, uint>(maskBytes).ToArray();
		await Assert.That(mask.All(word => word is 0 or 1)).IsTrue();
		await Assert.That(mask.Contains(1u)).IsTrue();
		await Assert.That(mask.Contains(0u)).IsTrue();
	}

	[Test]
	public async Task CreateKernel_RejectsUnknownEntryPoint()
	{
		var device = new ReferenceComputeDevice();
		var descriptor = new ComputeKernelDescriptor(
			"other", "@compute @workgroup_size(64) fn main() {}", "main", [new ComputeKernelBinding(0, 0, ComputeBindingType.Storage)]);
		await Assert.That(() => device.CreateKernel(descriptor)).Throws<ArgumentException>();
		await Assert.That(device.Commands.Count).IsEqualTo(0);
	}

	[Test]
	public async Task CreateBuffer_WithoutInitialData_StartsAsGarbage()
	{
		var device = new ReferenceComputeDevice();
		IComputeBuffer empty = device.CreateBuffer(ComputeBufferKind.Storage, 8);
		IComputeBuffer partial = device.CreateBuffer(ComputeBufferKind.Storage, 8, [1, 2, 3, 4]);
		await Assert.That(MemoryMarshal.Cast<byte, uint>(device.GetContents(empty)).ToArray())
			.IsEquivalentTo(new[] { ReferenceComputeDevice.GarbageWord, ReferenceComputeDevice.GarbageWord });
		await Assert.That(device.GetContents(partial)).IsEquivalentTo(new byte[] { 1, 2, 3, 4, 0, 0, 0, 0 });
	}

	[Test]
	public async Task Dispatch_ReachingSomeElements_LeavesTheRestUnwritten()
	{
		var f = new ReferenceComputeDeviceFixture();

		// Composed for every pixel, dispatched with one workgroup: only pixels 0..63 run.
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.InitRandom, Groups(PlaneSize));
		f.Device.Dispatch(kernel, [f.Group(kernel, 0, (1, f.Poses), (2, f.ProblemUniform), (6, f.State))], 1);
		float[] depth = await f.ReadFloats(f.State, PlaneSize);
		await Assert.That(depth[..64].All(d => d >= 2 && d <= 8)).IsTrue();
		await Assert.That(depth[64..].All(d => BitConverter.SingleToUInt32Bits(d) == ReferenceComputeDevice.GarbageWord)).IsTrue();

		// More workgroups along x than composed for would run elements twice.
		await Assert.That(() => f.Device.Dispatch(kernel, [f.Group(kernel, 0, (1, f.Poses), (2, f.ProblemUniform), (6, f.State))], Groups(PlaneSize) + 1))
			.Throws<ArgumentOutOfRangeException>();
	}

	[Test]
	public async Task Kernel_IndexingPastABinding_FaultsTheFlush()
	{
		var f = new ReferenceComputeDeviceFixture();
		IComputeBuffer tooSmall = f.Device.CreateBuffer(ComputeBufferKind.Storage, 16);
		uint groups = Groups(PlaneSize);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.InitRandom, groups);
		f.Device.Dispatch(kernel, [f.Group(kernel, 0, (1, f.Poses), (2, f.ProblemUniform), (6, tooSmall))], groups);
		await Assert.That(async () => await f.Device.FlushAsync()).Throws<InvalidOperationException>();

		// The device is reusable after the fault.
		await Assert.That(f.Device.Commands.OfType<ColmapSharp.Compute.Testing.FlushCommand>().Count()).IsEqualTo(1);
	}

	private static async Task RunInitRandom(ReferenceComputeDeviceFixture f)
	{
		uint groups = Groups(PlaneSize);
		IComputeKernel kernel = f.Kernel(PatchMatchGpuKernel.InitRandom, groups);
		f.Device.Dispatch(kernel, [f.Group(kernel, 0, (1, f.Poses), (2, f.ProblemUniform), (6, f.State))], groups);
		await f.Device.FlushAsync();
	}

	private static async Task AssertBits(float[] actual, float[] expected)
	{
		await Assert.That(actual.Length).IsEqualTo(expected.Length);
		for (int i = 0; i < actual.Length; ++i)
		{
			if (BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i]))
			{
				await Assert.That(actual[i]).IsEqualTo(expected[i]).Because($"element {i}");
			}
		}
	}
}
