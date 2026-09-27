// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpu.Run: creates a GPU PatchMatch run's buffers, kernels and bind groups once, then
// records PatchMatchCpu.Run's schedule on the device - init_random (photometric) and
// initial_cost, then per sweep backward_messages, the sweep_band dispatches, filter_pixels on the
// last sweep, and the rotation copies - flushing once per sweep. PatchMatchGpu.cs packs the
// buffer contents and decodes the results; PatchMatchGpuPlan.cs sizes everything and explains
// the buffer roles. Not a COLMAP port (PORTING_PLAN.md Phase 13).
//
// Buffer roles through the schedule. Each sweep rotates every map into another buffer, so which
// buffer plays which role depends only on the parity p of the sweep index k (the rotation k % 4
// has the same parity, since an iteration is four sweeps):
// - reference planes and state (depth, normals, column states) ping-pong: sweep k uses
//   reference[p] and state[p], and rotates them into reference[1 - p] and state[1 - p];
// - the source maps are three buffers A, B, C: C is always the previous selection map; the cost
//   map is A on even sweeps and B on odd ones, and the selection map the other. The rotation
//   copies the selection map into C first, then the cost map into the old selection buffer,
//   which is why cost and selection swap buffers every sweep (PatchMatchCpu.Rotate's order).
// - On the last sweep filter_pixels writes the consistency mask into the cost buffer; the cost
//   copy then moves the mask. Without filtering the last cost copy is skipped: nothing reads it.
// After 4 x iterations sweeps (an even count) the maps are back in rotation 0: depth and normals
// in state[0], the final selection probabilities in C, the mask in the last sweep's selection
// buffer (A).
//
// Every read_write buffer is bound once per dispatch (the seam forbids aliasing a writable
// binding), and every uniform is created with its data before anything is recorded (the seam's
// WriteBuffer hazard), so the run never writes a buffer.

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

internal sealed partial class PatchMatchGpu
{
	/// <summary>
	/// Runs the problem on <paramref name="device"/>: plans it (throws
	/// <see cref="InvalidOperationException"/> with the planner's reason if the device cannot run
	/// it; this method never falls back, so callers wanting a CPU fallback check
	/// <see cref="PatchMatchGpuPlan.TryCreate"/> first),
	/// records the schedule, flushes once per sweep, and reads back the results the getters
	/// return. Cancellation is checked before each sweep is recorded, so a canceled run leaves
	/// nothing pending on the device; progress is the fraction of sweeps done, as PatchMatchCpu
	/// reports it. A GPU error faults the returned task.
	/// </summary>
	public async Task RunAsync(IComputeDevice device, CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		ArgumentNullException.ThrowIfNull(device);
		if (!PatchMatchGpuPlan.TryCreate(ProblemShape, options, device.Limits, null, out PatchMatchGpuPlan? plan, out string? reason, SweepBandRows))
		{
			// The planner's reason ends "Using the CPU.", which is the caller's decision, not ours.
			throw new InvalidOperationException("The GPU cannot run this problem: " + PatchMatchGpuPlan.ReasonWithoutFallback(reason!));
		}

		cancellationToken.ThrowIfCancellationRequested();
		using var run = new GpuRun(this, device, plan!);
		// ConfigureAwait(false): the run may continue on another thread between device calls,
		// which IComputeDevice's threading contract allows (calls are never concurrent).
		await run.ExecuteAsync(cancellationToken, progress).ConfigureAwait(false);
	}

	/// <summary>The device resources of one run and the recorded schedule over them.</summary>
	private sealed class GpuRun : IDisposable
	{
		private readonly PatchMatchGpu pm;
		private readonly IComputeDevice device;
		private readonly PatchMatchGpuPlan plan;
		private readonly List<IDisposable> owned = [];
		private readonly Dictionary<PatchMatchGpuKernel, IComputeKernel> kernels = [];
		private readonly Dictionary<PatchMatchGpuKernel, uint> groupsX = [];

		private readonly IComputeBuffer byteTable;
		private readonly IComputeBuffer poses;
		private readonly IComputeBuffer problemUniform;
		private readonly IComputeBuffer[] reference = new IComputeBuffer[2];
		private readonly IComputeBuffer[] state = new IComputeBuffer[2];
		private readonly IComputeBuffer mapsA;
		private readonly IComputeBuffer mapsB;
		private readonly IComputeBuffer prevSelProbs;
		private readonly IComputeBuffer sourceImages;
		private readonly IComputeBuffer sourceDepths;

		// Bind groups: [parity] for group 0 of the parity-dependent kernels; one sweep group per
		// sweep; [orientation][band] band groups; rotation copies per parity.
		private readonly IComputeBindGroup[] backwardGroup = new IComputeBindGroup[2];
		private readonly IComputeBindGroup[] sweepBandGroup = new IComputeBindGroup[2];
		private readonly IComputeBindGroup?[] filterGroup = new IComputeBindGroup?[2];
		private readonly IComputeBindGroup[] sweepGroups;
		private readonly IComputeBindGroup[][] bandGroups = new IComputeBindGroup[2][];
		private readonly RotateCopy[][] rotateCopies = new RotateCopy[2][];
		private readonly IComputeBindGroup? initRandomGroup;
		private readonly IComputeBindGroup initialCostGroup;

		public GpuRun(PatchMatchGpu pm, IComputeDevice device, PatchMatchGpuPlan plan)
		{
			this.pm = pm;
			this.device = device;
			this.plan = plan;
			try
			{
				byteTable = Uniform(PatchMatchGpuBufferRole.ByteTable, PackByteTable(), "pm_byte_to_unit");
				poses = Uniform(PatchMatchGpuBufferRole.PoseTable, pm.PackPoseTable(), "pm_poses");
				problemUniform = Uniform(PatchMatchGpuBufferRole.Problem, pm.PackProblem(), "pm_problem");

				long referenceSize = Size(PatchMatchGpuBufferRole.ReferencePlanes);
				reference[0] = Storage(referenceSize, pm.PackReference(), "pm_reference_0");
				reference[1] = Storage(referenceSize, default, "pm_reference_1");
				long stateSize = Size(PatchMatchGpuBufferRole.State);
				state[0] = Storage(stateSize, pm.options.GeomConsistency ? pm.PackInitialState() : default, "pm_state_0");
				state[1] = Storage(stateSize, default, "pm_state_1");
				long mapsSize = Size(PatchMatchGpuBufferRole.SourceMaps);
				mapsA = Storage(mapsSize, default, "pm_source_maps_a");
				mapsB = Storage(mapsSize, default, "pm_source_maps_b");
				prevSelProbs = Storage(mapsSize, pm.PackInitialPrevSelProbs(), "pm_source_maps_c");
				long imagesSize = Size(PatchMatchGpuBufferRole.SourceImages);
				sourceImages = Storage(imagesSize, pm.PackSourceImages(imagesSize), "pm_source_images");
				sourceDepths = Storage(Size(PatchMatchGpuBufferRole.SourceDepths), pm.PackSourceDepths(), "pm_source_depths");

				foreach (PatchMatchGpuKernel kernel in PatchMatchGpuKernels.All)
				{
					if ((kernel == PatchMatchGpuKernel.InitRandom && plan.Geometric) || (kernel == PatchMatchGpuKernel.FilterPixels && !plan.Filter))
					{
						continue;
					}

					string entryPoint = PatchMatchGpuKernels.EntryPoint(kernel);
					uint x = plan.Dispatches.Single(d => d.Kernel == entryPoint).X;
					groupsX[kernel] = x;
					kernels[kernel] = Own(device.CreateKernel(PatchMatchGpuKernels.Descriptor(kernel, pm.ShaderShape, x)));
				}

				if (!plan.Geometric)
				{
					initRandomGroup = Group0(PatchMatchGpuKernel.InitRandom, parity: 0);
				}

				initialCostGroup = Group0(PatchMatchGpuKernel.InitialCost, parity: 0);
				for (int p = 0; p < 2; ++p)
				{
					backwardGroup[p] = Group0(PatchMatchGpuKernel.BackwardMessages, p);
					sweepBandGroup[p] = Group0(PatchMatchGpuKernel.SweepBand, p);
					if (plan.Filter)
					{
						filterGroup[p] = Group0(PatchMatchGpuKernel.FilterPixels, p);
					}

					bandGroups[p] = CreateBandGroups(p);
					rotateCopies[p] = CreateRotateCopies(p);
				}

				// One sweep uniform per sweep; its group layout (one uniform at binding 0) is the
				// same for backward_messages, sweep_band and filter_pixels.
				sweepGroups = new IComputeBindGroup[plan.SweepCount];
				long sweepSize = Size(PatchMatchGpuBufferRole.SweepUniforms);
				for (int k = 0; k < sweepGroups.Length; ++k)
				{
					IComputeBuffer sweepUniform = Own(device.CreateBuffer(
						ComputeBufferKind.Uniform, sweepSize, pm.PackSweep(k / 4, k % 4), $"pm_sweep_{k}"));
					sweepGroups[k] = UniformGroup(PatchMatchGpuKernel.BackwardMessages, PatchMatchGpuKernels.SweepGroup, sweepUniform);
				}
			}
			catch
			{
				Dispose();
				throw;
			}
		}

		public async Task ExecuteAsync(CancellationToken cancellationToken, IProgress<double>? progress)
		{
			int numIterations = pm.options.NumIterations;
			float totalNumSteps = numIterations * 4;
			int planeSize = pm.refWidth * pm.refHeight;
			int numSrc = pm.numSrc;
			for (int iter = 0; iter < numIterations; ++iter)
			{
				for (int sweep = 0; sweep < 4; ++sweep)
				{
					// Checked before recording, so a cancellation leaves nothing pending.
					cancellationToken.ThrowIfCancellationRequested();

					int k = iter * 4 + sweep;
					if (k == 0)
					{
						if (initRandomGroup != null)
						{
							Dispatch(PatchMatchGpuKernel.InitRandom, planeSize, initRandomGroup);
						}

						Dispatch(PatchMatchGpuKernel.InitialCost, (long)planeSize * numSrc, initialCostGroup);
					}

					RecordSweep(k, lastSweep: k == numIterations * 4 - 1);
					await device.FlushAsync(cancellationToken).ConfigureAwait(false);
					progress?.Report((iter * 4 + sweep + 1) / (double)totalNumSteps);
				}
			}

			// An even number of sweeps: the maps are back in rotation 0 (see the file header).
			var stateBytes = new byte[4L * 4 * planeSize];
			await device.ReadBufferAsync(state[0], 0, stateBytes, cancellationToken).ConfigureAwait(false);
			var selBytes = new byte[4L * numSrc * planeSize];
			await device.ReadBufferAsync(prevSelProbs, 0, selBytes, cancellationToken).ConfigureAwait(false);
			byte[] maskBytes = [];
			if (plan.Filter)
			{
				maskBytes = new byte[4L * numSrc * planeSize];
				await device.ReadBufferAsync(SelBuffer(1), 0, maskBytes, cancellationToken).ConfigureAwait(false);
			}

			pm.Decode(stateBytes, selBytes, maskBytes);
		}

		/// <summary>The recorded dispatches of sweep <paramref name="k"/>, its rotation included.</summary>
		private void RecordSweep(int k, bool lastSweep)
		{
			int p = k & 1;
			int width = p == 0 ? pm.refWidth : pm.refHeight;
			IComputeBindGroup sweepGroup = sweepGroups[k];

			Dispatch(PatchMatchGpuKernel.BackwardMessages, (long)width * pm.numSrc, backwardGroup[p], sweepGroup);
			foreach (IComputeBindGroup band in bandGroups[p])
			{
				Dispatch(PatchMatchGpuKernel.SweepBand, width, sweepBandGroup[p], sweepGroup, band);
			}

			if (lastSweep && plan.Filter)
			{
				Dispatch(PatchMatchGpuKernel.FilterPixels, (long)pm.refWidth * pm.refHeight, filterGroup[p]!, sweepGroup);
			}

			foreach (RotateCopy copy in rotateCopies[p])
			{
				// The last cost map is never read again; with filtering the buffer holds the mask.
				if (copy.IsCostCopy && lastSweep && !plan.Filter)
				{
					continue;
				}

				Dispatch(copy.Kernel, copy.Elements, copy.Buffers, copy.Uniform);
			}
		}

		// The cost map of a sweep of parity p is A on even sweeps, B on odd ones; the selection
		// map the other (see the file header).
		private IComputeBuffer CostBuffer(int p) => p == 0 ? mapsA : mapsB;

		private IComputeBuffer SelBuffer(int p) => p == 0 ? mapsB : mapsA;

		/// <summary>Group 0 of a sweep-side kernel for the buffers of parity <paramref name="parity"/>.</summary>
		private IComputeBindGroup Group0(PatchMatchGpuKernel kernel, int parity)
		{
			var entries = new List<ComputeBufferBinding>();
			foreach (ComputeKernelBinding binding in PatchMatchGpuKernels.Bindings(kernel))
			{
				if (binding.Group != 0)
				{
					continue;
				}

				IComputeBuffer buffer = binding.Binding switch
				{
					PatchMatchGpuKernels.ByteToUnitBinding => byteTable,
					PatchMatchGpuKernels.PosesBinding => poses,
					PatchMatchGpuKernels.ProblemBinding => problemUniform,
					PatchMatchGpuKernels.ReferenceBinding => reference[parity],
					PatchMatchGpuKernels.SourceImagesBinding => sourceImages,
					PatchMatchGpuKernels.SourceDepthsBinding => sourceDepths,
					PatchMatchGpuKernels.StateBinding => state[parity],
					PatchMatchGpuKernels.CostsBinding => CostBuffer(parity),
					PatchMatchGpuKernels.SelProbsBinding => SelBuffer(parity),
					PatchMatchGpuKernels.PrevSelProbsBinding => prevSelProbs,
					_ => throw new InvalidOperationException($"No buffer for binding {binding.Binding} of {kernel}."),
				};
				entries.Add(new ComputeBufferBinding(binding.Binding, buffer, 0, buffer.Size));
			}

			return Own(device.CreateBindGroup(kernels[kernel], 0, [.. entries]));
		}

		// The band uniforms of an orientation: rows [b R, min(rows, (b + 1) R)).
		private IComputeBindGroup[] CreateBandGroups(int orientation)
		{
			int rows = orientation == 0 ? pm.refHeight : pm.refWidth;
			int bandRows = plan.BandRows(orientation);
			var groups = new IComputeBindGroup[plan.BandCount(orientation)];
			long size = Size(PatchMatchGpuBufferRole.BandUniforms);
			for (int b = 0; b < groups.Length; ++b)
			{
				int rowStart = b * bandRows;
				byte[] data = PackBand(rowStart, Math.Min(rows, rowStart + bandRows));
				IComputeBuffer uniform = Own(device.CreateBuffer(ComputeBufferKind.Uniform, size, data, $"pm_band_{orientation}_{b}"));
				groups[b] = UniformGroup(PatchMatchGpuKernel.SweepBand, PatchMatchGpuKernels.BandGroup, uniform);
			}

			return groups;
		}

		// The rotation after a sweep of parity p, in PatchMatchCpu.Rotate's order: depth, normals,
		// reference planes, selection into the previous map, cost into the old selection buffer.
		// PatchMatchGpuPlan.RotateCopiesPerParity counts these for the plan's memory total.
		private RotateCopy[] CreateRotateCopies(int p)
		{
			int srcWidth = p == 0 ? pm.refWidth : pm.refHeight;
			int srcHeight = p == 0 ? pm.refHeight : pm.refWidth;
			int planeSize = srcWidth * srcHeight;
			int s = pm.numSrc;
			IComputeBindGroup stateCopy = CopyGroup(state[p], state[1 - p]);
			return
			[
				Copy(PatchMatchGpuKernel.RotatePlanes, stateCopy, PackRotate(srcWidth, srcHeight, 1, 0, 0), planeSize, $"depth_{p}", false),
				Copy(PatchMatchGpuKernel.RotateNormals, stateCopy, PackRotate(srcWidth, srcHeight, 3, planeSize, planeSize), planeSize, $"normals_{p}", false),
				Copy(PatchMatchGpuKernel.RotatePlanes, CopyGroup(reference[p], reference[1 - p]), PackRotate(srcWidth, srcHeight, 3, 0, 0), 3L * planeSize, $"reference_{p}", false),
				Copy(PatchMatchGpuKernel.RotatePlanes, CopyGroup(SelBuffer(p), prevSelProbs), PackRotate(srcWidth, srcHeight, s, 0, 0), (long)s * planeSize, $"sel_{p}", false),
				Copy(PatchMatchGpuKernel.RotatePlanes, CopyGroup(CostBuffer(p), SelBuffer(p)), PackRotate(srcWidth, srcHeight, s, 0, 0), (long)s * planeSize, $"cost_{p}", true),
			];
		}

		private RotateCopy Copy(PatchMatchGpuKernel kernel, IComputeBindGroup buffers, byte[] data, long elements, string name, bool isCostCopy)
		{
			IComputeBuffer uniform = Own(device.CreateBuffer(ComputeBufferKind.Uniform, Size(PatchMatchGpuBufferRole.RotateUniforms), data, "pm_rotate_" + name));
			return new RotateCopy(kernel, buffers, UniformGroup(kernel, PatchMatchGpuKernels.SweepGroup, uniform), elements, isCostCopy);
		}

		private IComputeBindGroup CopyGroup(IComputeBuffer source, IComputeBuffer destination) =>
			Own(device.CreateBindGroup(kernels[PatchMatchGpuKernel.RotatePlanes], 0,
			[
				new ComputeBufferBinding(PatchMatchGpuKernels.RotateSourceBinding, source, 0, source.Size),
				new ComputeBufferBinding(PatchMatchGpuKernels.RotateDestinationBinding, destination, 0, destination.Size),
			]));

		private IComputeBindGroup UniformGroup(PatchMatchGpuKernel kernel, int group, IComputeBuffer uniform) =>
			Own(device.CreateBindGroup(kernels[kernel], group, [new ComputeBufferBinding(0, uniform, 0, uniform.Size)]));

		/// <summary>
		/// Records <paramref name="kernel"/> over <paramref name="elements"/> invocations, folded as
		/// the kernel's PM_GROUPS_X expects: up to that many workgroups along x, the rest along y.
		/// </summary>
		private void Dispatch(PatchMatchGpuKernel kernel, long elements, params IComputeBindGroup[] groups)
		{
			uint maxX = groupsX[kernel];
			long workgroups = (elements + PatchMatchGpuKernels.WorkgroupSize - 1) / PatchMatchGpuKernels.WorkgroupSize;
			long x = Math.Min(workgroups, maxX);
			long y = (workgroups + x - 1) / x;
			device.Dispatch(kernels[kernel], groups, (uint)x, (uint)y);
		}

		private long Size(PatchMatchGpuBufferRole role) => plan.Buffers.First(b => b.Role == role).Size;

		private IComputeBuffer Uniform(PatchMatchGpuBufferRole role, byte[] data, string label) =>
			Own(device.CreateBuffer(ComputeBufferKind.Uniform, Size(role), data, label));

		private IComputeBuffer Storage(long size, ReadOnlySpan<byte> data, string label) =>
			Own(device.CreateBuffer(ComputeBufferKind.Storage, size, data, label));

		private T Own<T>(T handle)
			where T : IDisposable
		{
			owned.Add(handle);
			return handle;
		}

		public void Dispose()
		{
			// Bind groups and kernels before the buffers they reference; WebGPU would keep them
			// alive anyway, but this reads in dependency order.
			for (int i = owned.Count - 1; i >= 0; --i)
			{
				owned[i].Dispose();
			}

			owned.Clear();
		}
	}

	/// <summary>One rotation copy: kernel, source/destination group, uniform group, element count.</summary>
	private readonly record struct RotateCopy(
		PatchMatchGpuKernel Kernel, IComputeBindGroup Buffers, IComputeBindGroup Uniform, long Elements, bool IsCostCopy);
}
