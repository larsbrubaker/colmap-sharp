// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RecordingComputeDeviceTests: C#-only tests (COLMAP has no compute seam; its GPU path is CUDA)
// of ColmapSharp/Compute/Testing/RecordingComputeDevice.cs. They pin every argument rule the
// IComputeDevice contract states - buffer sizes and limits, alignment, kernel layouts, bind
// group and dispatch layout matching, disposed handles - so the GPU PatchMatch orchestration
// tested against this double is known to keep the rules a real WebGPU device holds it to.
// Ordering, the write-after-dispatch hazard, flush, readback and cancellation are in
// RecordingComputeDeviceFlushTests.cs.

using ColmapSharp.Compute;
using ColmapSharp.Compute.Testing;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Compute;

public class RecordingComputeDeviceTests
{
	/// <summary>A kernel with one read-only storage binding at 0 and one uniform at 1, in group 0.</summary>
	internal static IComputeKernel StorageAndUniformKernel(RecordingComputeDevice device, string label = "k")
		=> device.CreateKernel(new ComputeKernelDescriptor(label, "// wgsl", "main", new[]
		{
			new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage),
			new ComputeKernelBinding(0, 1, ComputeBindingType.Uniform),
		}));

	/// <summary>A bind group for <see cref="StorageAndUniformKernel"/>'s group 0 over whole buffers.</summary>
	internal static IComputeBindGroup BindWhole(RecordingComputeDevice device, IComputeKernel kernel, IComputeBuffer storage, IComputeBuffer uniform)
		=> device.CreateBindGroup(kernel, 0, new[]
		{
			new ComputeBufferBinding(0, storage, 0, storage.Size),
			new ComputeBufferBinding(1, uniform, 0, uniform.Size),
		});

	[Test]
	public async Task Defaults_AreWebGpuDefaults()
	{
		var limits = ComputeDeviceLimits.Defaults;
		await Assert.That(limits.MaxBufferSize).IsEqualTo(268435456L);
		await Assert.That(limits.MaxStorageBufferBindingSize).IsEqualTo(134217728L);
		await Assert.That(limits.MaxStorageBuffersPerShaderStage).IsEqualTo(8);
		await Assert.That(limits.MaxUniformBufferBindingSize).IsEqualTo(65536L);
		await Assert.That(limits.MaxComputeWorkgroupsPerDimension).IsEqualTo(65535u);
		await Assert.That(limits.MaxComputeInvocationsPerWorkgroup).IsEqualTo(256);
		await Assert.That(limits.MinStorageBufferOffsetAlignment).IsEqualTo(256);
		await Assert.That(new RecordingComputeDevice().Limits).IsEqualTo(limits);
	}

	[Test]
	public async Task CreateBuffer_StoresInitialDataAndZeroFillsTheRest()
	{
		var device = new RecordingComputeDevice();
		var buffer = (RecordingComputeBuffer)device.CreateBuffer(ComputeBufferKind.Storage, 8, new byte[] { 1, 2, 3 }, "b");
		await Assert.That(buffer.GetContents()).IsEquivalentTo(new byte[] { 1, 2, 3, 0, 0, 0, 0, 0 }, CollectionOrdering.Matching);
		await Assert.That(buffer.Size).IsEqualTo(8L);
		await Assert.That(buffer.Kind).IsEqualTo(ComputeBufferKind.Storage);
		await Assert.That(buffer.Label).IsEqualTo("b");
		var command = device.CommandsOf<CreateBufferCommand>().Single();
		await Assert.That(command.Buffer).IsSameReferenceAs(buffer);
		await Assert.That(command.InitialLength).IsEqualTo(3);
	}

	[Test]
	public async Task CreateBuffer_RefusesBadSizes()
	{
		var device = new RecordingComputeDevice { Limits = ComputeDeviceLimits.Defaults with { MaxBufferSize = 64 } };
		Assert.Throws<ArgumentException>(() => device.CreateBuffer(ComputeBufferKind.Storage, 0));
		Assert.Throws<ArgumentException>(() => device.CreateBuffer(ComputeBufferKind.Storage, 6));
		Assert.Throws<ArgumentException>(() => device.CreateBuffer(ComputeBufferKind.Storage, 4, new byte[8]));
		Assert.Throws<InvalidOperationException>(() => device.CreateBuffer(ComputeBufferKind.Storage, 68));
		await Assert.That(device.CreateBuffer(ComputeBufferKind.Storage, 64).Size).IsEqualTo(64L);
		await Assert.That(device.Commands.Count).IsEqualTo(1);
	}

	[Test]
	public async Task WriteBuffer_StoresBytesAndChecksTheRange()
	{
		var device = new RecordingComputeDevice();
		var buffer = (RecordingComputeBuffer)device.CreateBuffer(ComputeBufferKind.Uniform, 12);
		device.WriteBuffer(buffer, 4, new byte[] { 9, 8, 7, 6 });
		await Assert.That(buffer.GetContents()).IsEquivalentTo(new byte[] { 0, 0, 0, 0, 9, 8, 7, 6, 0, 0, 0, 0 }, CollectionOrdering.Matching);
		await Assert.That(device.CommandsOf<WriteBufferCommand>().Single()).IsEqualTo(new WriteBufferCommand(buffer, 4, 4));

		Assert.Throws<ArgumentException>(() => device.WriteBuffer(buffer, 2, new byte[4]));
		Assert.Throws<ArgumentException>(() => device.WriteBuffer(buffer, 0, new byte[3]));
		Assert.Throws<ArgumentException>(() => device.WriteBuffer(buffer, 8, new byte[8]));
		Assert.Throws<ArgumentException>(() => device.WriteBuffer(new RecordingComputeDevice().CreateBuffer(ComputeBufferKind.Uniform, 4), 0, new byte[4]));
		buffer.Dispose();
		Assert.Throws<ObjectDisposedException>(() => device.WriteBuffer(buffer, 0, new byte[4]));
	}

	[Test]
	public async Task CreateKernel_EnforcesStorageBindingsPerStage()
	{
		var device = new RecordingComputeDevice { Limits = ComputeDeviceLimits.Defaults with { MaxStorageBuffersPerShaderStage = 2 } };

		// Uniforms do not count against the storage limit; read-only and read-write storage both do,
		// across every group.
		var fits = new[]
		{
			new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage),
			new ComputeKernelBinding(1, 0, ComputeBindingType.Storage),
			new ComputeKernelBinding(1, 1, ComputeBindingType.Uniform),
			new ComputeKernelBinding(2, 0, ComputeBindingType.Uniform),
		};
		var kernel = (RecordingComputeKernel)device.CreateKernel(new ComputeKernelDescriptor("fits", "src", "main", fits));
		await Assert.That(kernel.GroupLayouts.Count).IsEqualTo(3);
		await Assert.That(device.CommandsOf<CreateKernelCommand>().Single().Kernel).IsSameReferenceAs(kernel);

		var tooMany = fits.Append(new ComputeKernelBinding(2, 1, ComputeBindingType.ReadOnlyStorage)).ToArray();
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor("tooMany", "src", "main", tooMany)));
	}

	[Test]
	public async Task CreateKernel_RefusesMalformedLayouts()
	{
		var device = new RecordingComputeDevice();
		var storage = ComputeBindingType.Storage;
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor(
			"dup", "src", "main", new[] { new ComputeKernelBinding(0, 0, storage), new ComputeKernelBinding(0, 0, ComputeBindingType.Uniform) })));
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor(
			"gap", "src", "main", new[] { new ComputeKernelBinding(0, 0, storage), new ComputeKernelBinding(2, 0, storage) })));
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor(
			"negative", "src", "main", new[] { new ComputeKernelBinding(0, -1, storage) })));
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor("noSource", "", "main", Array.Empty<ComputeKernelBinding>())));
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor("noEntry", "src", "", Array.Empty<ComputeKernelBinding>())));
		Assert.Throws<ArgumentException>(() => device.CreateKernel(new ComputeKernelDescriptor("noList", "src", "main", null!)));
		await Assert.That(device.Commands.Count).IsEqualTo(0);
	}

	[Test]
	public async Task CreateBindGroup_RequiresExactlyTheDeclaredBindings()
	{
		var device = new RecordingComputeDevice();
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 16);
		var uniform = device.CreateBuffer(ComputeBufferKind.Uniform, 16);

		// Entries may come in any order; the group stores them sorted.
		var group = (RecordingComputeBindGroup)device.CreateBindGroup(kernel, 0, new[]
		{
			new ComputeBufferBinding(1, uniform, 0, 16),
			new ComputeBufferBinding(0, storage, 0, 16),
		});
		await Assert.That(group.Entries.Select(e => e.Binding)).IsEquivalentTo(new[] { 0, 1 }, CollectionOrdering.Matching);

		Assert.Throws<ArgumentException>(() => device.CreateBindGroup(kernel, 0, new[] { new ComputeBufferBinding(0, storage, 0, 16) }));
		Assert.Throws<ArgumentException>(() => device.CreateBindGroup(kernel, 0, new[]
		{
			new ComputeBufferBinding(0, storage, 0, 16),
			new ComputeBufferBinding(0, storage, 0, 16),
		}));
		Assert.Throws<ArgumentException>(() => device.CreateBindGroup(kernel, 0, new[]
		{
			new ComputeBufferBinding(0, storage, 0, 16),
			new ComputeBufferBinding(2, uniform, 0, 16),
		}));
		Assert.Throws<ArgumentException>(() => device.CreateBindGroup(kernel, 1, ReadOnlySpan<ComputeBufferBinding>.Empty));
	}

	[Test]
	public async Task CreateBindGroup_RequiresTheMatchingBufferKind()
	{
		var device = new RecordingComputeDevice();
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 16);
		var uniform = device.CreateBuffer(ComputeBufferKind.Uniform, 16);
		Assert.Throws<ArgumentException>(() => BindWhole(device, kernel, uniform, uniform));
		Assert.Throws<ArgumentException>(() => BindWhole(device, kernel, storage, storage));
		await Assert.That(device.CommandsOf<CreateBindGroupCommand>().Count).IsEqualTo(0);
	}

	[Test]
	public async Task CreateBindGroup_EnforcesAlignmentBoundsAndBindingSizeLimits()
	{
		var device = new RecordingComputeDevice
		{
			Limits = ComputeDeviceLimits.Defaults with { MaxStorageBufferBindingSize = 512, MaxUniformBufferBindingSize = 256 },
		};
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 1024);
		var uniform = device.CreateBuffer(ComputeBufferKind.Uniform, 512);

		IComputeBindGroup Bind(long storageOffset, long storageSize, long uniformSize) => device.CreateBindGroup(kernel, 0, new[]
		{
			new ComputeBufferBinding(0, storage, storageOffset, storageSize),
			new ComputeBufferBinding(1, uniform, 0, uniformSize),
		});

		// A 512-byte storage window at a 256-aligned offset, and a full 256-byte uniform, fit.
		await Assert.That(Bind(256, 512, 256).Group).IsEqualTo(0);
		Assert.Throws<ArgumentException>(() => Bind(128, 256, 16));   // offset not 256-aligned
		Assert.Throws<ArgumentException>(() => Bind(0, 6, 16));       // size not a multiple of 4
		Assert.Throws<ArgumentException>(() => Bind(0, 0, 16));       // empty binding
		Assert.Throws<ArgumentException>(() => Bind(768, 512, 16));   // runs past the buffer
		Assert.Throws<ArgumentException>(() => Bind(0, 516, 16));     // over the storage binding limit
		Assert.Throws<ArgumentException>(() => Bind(0, 16, 260));     // over the uniform binding limit
	}

	[Test]
	public async Task CreateBindGroup_RefusesDisposedOrForeignHandles()
	{
		var device = new RecordingComputeDevice();
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 16);
		var uniform = device.CreateBuffer(ComputeBufferKind.Uniform, 16);
		var foreign = new RecordingComputeDevice().CreateBuffer(ComputeBufferKind.Storage, 16);
		Assert.Throws<ArgumentException>(() => BindWhole(device, kernel, foreign, uniform));
		Assert.Throws<ArgumentException>(() => BindWhole(device, StorageAndUniformKernel(new RecordingComputeDevice()), storage, uniform));

		storage.Dispose();
		Assert.Throws<ObjectDisposedException>(() => BindWhole(device, kernel, storage, uniform));
		kernel.Dispose();
		Assert.Throws<ObjectDisposedException>(() => BindWhole(device, kernel, device.CreateBuffer(ComputeBufferKind.Storage, 16), uniform));
		await Assert.That(device.CommandsOf<CreateBindGroupCommand>().Count).IsEqualTo(0);
	}

	[Test]
	public async Task Dispatch_AcceptsBindGroupsOfAnyKernelWithAnIdenticalLayout()
	{
		var device = new RecordingComputeDevice();
		var first = StorageAndUniformKernel(device, "first");
		var second = StorageAndUniformKernel(device, "second");
		var group = BindWhole(device, first, device.CreateBuffer(ComputeBufferKind.Storage, 16), device.CreateBuffer(ComputeBufferKind.Uniform, 16));
		device.Dispatch(second, new[] { group }, 3, 2, 1);
		var dispatch = device.CommandsOf<DispatchCommand>().Single();
		await Assert.That(dispatch.Kernel).IsSameReferenceAs(second);
		await Assert.That(dispatch.Groups.Single()).IsSameReferenceAs(group);
		await Assert.That((dispatch.X, dispatch.Y, dispatch.Z)).IsEqualTo((3u, 2u, 1u));
	}

	[Test]
	public async Task Dispatch_RefusesLayoutMismatches()
	{
		var device = new RecordingComputeDevice();
		var kernel = StorageAndUniformKernel(device);
		var group = BindWhole(device, kernel, device.CreateBuffer(ComputeBufferKind.Storage, 16), device.CreateBuffer(ComputeBufferKind.Uniform, 16));

		// Same bindings, but binding 0 is read-write here: not the same layout.
		var readWrite = device.CreateKernel(new ComputeKernelDescriptor("rw", "src", "main", new[]
		{
			new ComputeKernelBinding(0, 0, ComputeBindingType.Storage),
			new ComputeKernelBinding(0, 1, ComputeBindingType.Uniform),
		}));
		Assert.Throws<ArgumentException>(() => device.Dispatch(readWrite, new[] { group }, 1));

		// Group 0 of this kernel matches, but the group is bound at index 1.
		var twoGroups = device.CreateKernel(new ComputeKernelDescriptor("two", "src", "main", new[]
		{
			new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage),
			new ComputeKernelBinding(0, 1, ComputeBindingType.Uniform),
			new ComputeKernelBinding(1, 0, ComputeBindingType.ReadOnlyStorage),
			new ComputeKernelBinding(1, 1, ComputeBindingType.Uniform),
		}));
		Assert.Throws<ArgumentException>(() => device.Dispatch(twoGroups, new[] { group, group }, 1));
		Assert.Throws<ArgumentException>(() => device.Dispatch(twoGroups, new[] { group }, 1));
		Assert.Throws<ArgumentException>(() => device.Dispatch(kernel, ReadOnlySpan<IComputeBindGroup>.Empty, 1));
		await Assert.That(device.CommandsOf<DispatchCommand>().Count).IsEqualTo(0);
	}

	[Test]
	public async Task Dispatch_EnforcesWorkgroupsPerDimension()
	{
		var device = new RecordingComputeDevice { Limits = ComputeDeviceLimits.Defaults with { MaxComputeWorkgroupsPerDimension = 100 } };
		var kernel = StorageAndUniformKernel(device);
		var group = BindWhole(device, kernel, device.CreateBuffer(ComputeBufferKind.Storage, 16), device.CreateBuffer(ComputeBufferKind.Uniform, 16));
		device.Dispatch(kernel, new[] { group }, 100, 100, 100);
		Assert.Throws<ArgumentOutOfRangeException>(() => device.Dispatch(kernel, new[] { group }, 101));
		Assert.Throws<ArgumentOutOfRangeException>(() => device.Dispatch(kernel, new[] { group }, 1, 101));
		Assert.Throws<ArgumentOutOfRangeException>(() => device.Dispatch(kernel, new[] { group }, 1, 1, 101));
		await Assert.That(device.PendingDispatchCount).IsEqualTo(1);
	}

	[Test]
	public async Task Dispatch_RefusesDisposedHandles()
	{
		var device = new RecordingComputeDevice();
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 16);
		var group = BindWhole(device, kernel, storage, device.CreateBuffer(ComputeBufferKind.Uniform, 16));

		// A bind group whose buffer has been disposed can no longer be dispatched.
		storage.Dispose();
		Assert.Throws<ObjectDisposedException>(() => device.Dispatch(kernel, new[] { group }, 1));

		var liveGroup = BindWhole(device, kernel, device.CreateBuffer(ComputeBufferKind.Storage, 16), device.CreateBuffer(ComputeBufferKind.Uniform, 16));
		liveGroup.Dispose();
		Assert.Throws<ObjectDisposedException>(() => device.Dispatch(kernel, new[] { liveGroup }, 1));
		kernel.Dispose();
		Assert.Throws<ObjectDisposedException>(() => device.Dispatch(kernel, new[] { group }, 1));
		await Assert.That(device.PendingDispatchCount).IsEqualTo(0);
	}
}
