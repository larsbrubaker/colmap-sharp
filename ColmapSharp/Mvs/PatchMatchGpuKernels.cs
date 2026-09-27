// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuKernels: the catalogue of GPU PatchMatch's compute kernels - for each, the entry
// point, the WGSL parts it is composed from (PatchMatchShaders.cs), the constants its text needs,
// and the bindings it declares - and the descriptor that puts them together. Not a COLMAP port
// (PORTING_PLAN.md Phase 13); the kernels replace patch_match_cuda.cu's, and each .wgsl kernel
// file in Mvs/Shaders/ names the CPU code it mirrors (PatchMatchCpu*.cs). PatchMatchGpuPlan.cs
// sizes the dispatches; the orchestrator packs the uniforms at the offsets that
// patch_match_layout.wgsl and patch_match_rotate.wgsl document (the sizes are below).
//
// Every binding has one fixed (group, binding) number across all kernels, so the host can keep
// one table: group 0 holds the problem's buffers (uniforms and storage), group 1 the per-sweep
// (or per-rotation-copy) uniform, group 2 the per-band uniform. A kernel declares the subset it
// uses. Each read_write buffer appears in exactly one binding of a kernel.
//
// Every kernel runs @workgroup_size(64) over a folded 2-D dispatch (patch_match_dispatch.wgsl);
// its PM_GROUPS_X constant is the dispatch's x size, so it is part of the kernel text. The
// cooperative sweep_band is the exception to one element per invocation: a workgroup per column.

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

/// <summary>The compute kernels of GPU PatchMatch.</summary>
internal enum PatchMatchGpuKernel
{
	/// <summary>Random initial depth and normal per pixel (photometric runs).</summary>
	InitRandom,

	/// <summary>The photo-consistency cost of every pixel against every source image.</summary>
	InitialCost,

	/// <summary>A sweep's backward messages and column-state initialization.</summary>
	BackwardMessages,

	/// <summary>One band of rows of a sweep.</summary>
	SweepBand,

	/// <summary>The last sweep's consistency filter.</summary>
	FilterPixels,

	/// <summary>90-degree rotation of 32-bit planes.</summary>
	RotatePlanes,

	/// <summary>90-degree rotation of the normal planes, rotating each normal.</summary>
	RotateNormals,
}

/// <summary>The per-problem values baked into the kernels' text.</summary>
/// <param name="RefWidth">Width of the unrotated reference image.</param>
/// <param name="RefHeight">Height of the unrotated reference image.</param>
/// <param name="NumSrcImages">Number of source images (S).</param>
/// <param name="SrcMaxWidth">Width of the source image layers (PatchMatchSourceImages.MaxWidth).</param>
/// <param name="SrcMaxHeight">Height of the source image layers.</param>
/// <param name="WindowRadius">PatchMatchOptions.WindowRadius.</param>
/// <param name="WindowStep">PatchMatchOptions.WindowStep.</param>
/// <param name="NumSamples">PatchMatchOptions.NumSamples.</param>
/// <param name="GeomConsistency">PatchMatchOptions.GeomConsistency: the sweep's geometric term.</param>
/// <param name="CooperativeSweep">
/// sweep_band's scheme: true runs one workgroup per column whose lanes split each row's work
/// (the default, many times faster); false runs one invocation per column as COLMAP does, kept
/// so a GPU test can pin the two to the same floats.
/// </param>
internal readonly record struct PatchMatchGpuShaderShape(
	int RefWidth,
	int RefHeight,
	int NumSrcImages,
	int SrcMaxWidth,
	int SrcMaxHeight,
	int WindowRadius,
	int WindowStep,
	int NumSamples,
	bool GeomConsistency,
	bool CooperativeSweep = true);

/// <summary>Parts, constants, bindings and descriptors of the GPU PatchMatch kernels.</summary>
internal static class PatchMatchGpuKernels
{
	/// <summary>Threads per workgroup of every kernel (PM_WORKGROUP_SIZE).</summary>
	public const int WorkgroupSize = 64;

	/// <summary>Bytes of the PmProblem uniform (patch_match_layout.wgsl).</summary>
	public const int ProblemUniformSize = 64;

	/// <summary>Bytes of the PmSweep uniform (patch_match_layout.wgsl).</summary>
	public const int SweepUniformSize = 32;

	/// <summary>Bytes of the PmBand uniform (patch_match_layout.wgsl).</summary>
	public const int BandUniformSize = 16;

	/// <summary>Bytes of the PmRotate uniform (patch_match_rotate.wgsl).</summary>
	public const int RotateUniformSize = 32;

	// Group 0: the problem's buffers, one number each across all kernels.
	public const int ByteToUnitBinding = 0;
	public const int PosesBinding = 1;
	public const int ProblemBinding = 2;
	public const int ReferenceBinding = 3;
	public const int SourceImagesBinding = 4;
	public const int SourceDepthsBinding = 5;
	public const int StateBinding = 6;

	/// <summary>The cost buffer; filter_pixels binds it here as the u32 mask.</summary>
	public const int CostsBinding = 7;
	public const int SelProbsBinding = 8;
	public const int PrevSelProbsBinding = 9;

	// rotate_planes / rotate_normals, group 0.
	public const int RotateSourceBinding = 0;
	public const int RotateDestinationBinding = 1;

	/// <summary>Group 1: the sweep uniform (the rotation uniform for the rotate kernels).</summary>
	public const int SweepGroup = 1;

	/// <summary>Group 2: the band uniform.</summary>
	public const int BandGroup = 2;

	/// <summary>
	/// Invocations sweep_band runs per column: a whole workgroup under the cooperative scheme,
	/// one invocation otherwise.
	/// </summary>
	public static int SweepInvocationsPerColumn(bool cooperative) => cooperative ? WorkgroupSize : 1;

	/// <summary>Hypotheses scored per pixel (PM_NUM_COSTS in patch_match_likelihood.wgsl).</summary>
	public const int NumCosts = 5;

	/// <summary>
	/// Bytes of <c>var&lt;workgroup&gt;</c> memory the cooperative sweep_band declares
	/// (patch_match_sweep_band.wgsl), every element 4 bytes: forward messages and sampling
	/// priors (S each), per sample the drawn source and hypothesis 0's cost (N each), per
	/// (sample, hypothesis) the NCC and geometric terms (N x NumCosts each), and the hypothesis
	/// sums (NumCosts). The serial scheme's module declares the same arrays, but a backend only
	/// allocates what the entry point reaches.
	/// </summary>
	public static long SweepWorkgroupBytes(int numSrc, int numSamples) =>
		4L * ((2L * numSrc) + (2L * numSamples) + (2L * numSamples * NumCosts) + NumCosts);

	/// <summary>Every kernel.</summary>
	public static IReadOnlyList<PatchMatchGpuKernel> All { get; } = Enum.GetValues<PatchMatchGpuKernel>();

	/// <summary>The WGSL entry point of <paramref name="kernel"/>.</summary>
	public static string EntryPoint(PatchMatchGpuKernel kernel) => kernel switch
	{
		PatchMatchGpuKernel.InitRandom => "init_random",
		PatchMatchGpuKernel.InitialCost => "initial_cost",
		PatchMatchGpuKernel.BackwardMessages => "backward_messages",
		PatchMatchGpuKernel.SweepBand => "sweep_band",

		// "filter" is a reserved word in WGSL.
		PatchMatchGpuKernel.FilterPixels => "filter_pixels",
		PatchMatchGpuKernel.RotatePlanes => "rotate_planes",
		PatchMatchGpuKernel.RotateNormals => "rotate_normals",
		_ => throw new ArgumentOutOfRangeException(nameof(kernel)),
	};

	/// <summary>The kernel's own .wgsl file (its bindings and entry point); the last part.</summary>
	public static string KernelFile(PatchMatchGpuKernel kernel) => kernel switch
	{
		PatchMatchGpuKernel.InitRandom => "patch_match_init_random.wgsl",
		PatchMatchGpuKernel.InitialCost => "patch_match_initial_cost.wgsl",
		PatchMatchGpuKernel.BackwardMessages => "patch_match_backward_messages.wgsl",
		PatchMatchGpuKernel.SweepBand => "patch_match_sweep_band.wgsl",
		PatchMatchGpuKernel.FilterPixels => "patch_match_filter.wgsl",
		PatchMatchGpuKernel.RotatePlanes or PatchMatchGpuKernel.RotateNormals => "patch_match_rotate.wgsl",
		_ => throw new ArgumentOutOfRangeException(nameof(kernel)),
	};

	/// <summary>The parts <paramref name="kernel"/>'s module is composed from, in order.</summary>
	public static IReadOnlyList<string> Parts(PatchMatchGpuKernel kernel)
	{
		string[] shared = kernel switch
		{
			PatchMatchGpuKernel.InitRandom or PatchMatchGpuKernel.BackwardMessages =>
				[PatchMatchShaders.Dispatch, PatchMatchShaders.Common, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood, PatchMatchShaders.Layout],
			PatchMatchGpuKernel.InitialCost =>
				[PatchMatchShaders.Dispatch, PatchMatchShaders.Common, PatchMatchShaders.Textures, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood, PatchMatchShaders.Layout, PatchMatchShaders.Ncc],
			PatchMatchGpuKernel.SweepBand =>
				[PatchMatchShaders.Dispatch, PatchMatchShaders.Common, PatchMatchShaders.Textures, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood, PatchMatchShaders.Layout, PatchMatchShaders.Ncc, PatchMatchShaders.GeomCost],
			PatchMatchGpuKernel.FilterPixels =>
				[PatchMatchShaders.Dispatch, PatchMatchShaders.Common, PatchMatchShaders.Textures, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood, PatchMatchShaders.Layout, PatchMatchShaders.GeomCost],
			PatchMatchGpuKernel.RotatePlanes or PatchMatchGpuKernel.RotateNormals => [PatchMatchShaders.Dispatch],
			_ => throw new ArgumentOutOfRangeException(nameof(kernel)),
		};
		return [.. shared, KernelFile(kernel)];
	}

	/// <summary>The names of the constants <paramref name="kernel"/>'s text needs, sorted.</summary>
	public static IReadOnlyList<string> RequiredConstants(PatchMatchGpuKernel kernel)
	{
		var names = new SortedSet<string>(StringComparer.Ordinal) { "PM_GROUPS_X" };
		IReadOnlyList<string> parts = Parts(kernel);
		if (parts.Contains(PatchMatchShaders.Textures))
		{
			names.Add("PM_SRC_MAX_WIDTH");
			names.Add("PM_SRC_MAX_HEIGHT");
		}

		if (parts.Contains(PatchMatchShaders.Geometry))
		{
			names.Add("PM_NUM_SRC_IMAGES");
		}

		if (parts.Contains(PatchMatchShaders.Layout))
		{
			names.Add("PM_REF_WIDTH");
			names.Add("PM_REF_HEIGHT");
		}

		// Ncc's window, which the sweep's resolution prior also reads.
		if (parts.Contains(PatchMatchShaders.Ncc))
		{
			names.Add("PM_WINDOW_RADIUS");
			names.Add("PM_WINDOW_STEP");
		}

		if (kernel == PatchMatchGpuKernel.SweepBand)
		{
			names.Add("PM_NUM_SAMPLES");
			names.Add("PM_GEOM_CONSISTENCY");
			names.Add("PM_SWEEP_COOPERATIVE");
		}

		return [.. names];
	}

	/// <summary>
	/// The constants header of <paramref name="kernel"/> for <paramref name="shape"/>, dispatched
	/// with <paramref name="groupsX"/> workgroups along x: exactly <see cref="RequiredConstants"/>.
	/// </summary>
	public static WgslConstants Constants(PatchMatchGpuKernel kernel, in PatchMatchGpuShaderShape shape, uint groupsX)
	{
		if (groupsX == 0)
		{
			throw new ArgumentOutOfRangeException(nameof(groupsX), "A dispatch needs at least one workgroup along x.");
		}

		var constants = new WgslConstants();
		foreach (string name in RequiredConstants(kernel))
		{
			_ = name switch
			{
				"PM_GROUPS_X" => constants.Add(name, groupsX),
				"PM_SRC_MAX_WIDTH" => constants.Add(name, shape.SrcMaxWidth),
				"PM_SRC_MAX_HEIGHT" => constants.Add(name, shape.SrcMaxHeight),
				"PM_NUM_SRC_IMAGES" => constants.Add(name, shape.NumSrcImages),
				"PM_REF_WIDTH" => constants.Add(name, shape.RefWidth),
				"PM_REF_HEIGHT" => constants.Add(name, shape.RefHeight),
				"PM_WINDOW_RADIUS" => constants.Add(name, shape.WindowRadius),
				"PM_WINDOW_STEP" => constants.Add(name, shape.WindowStep),
				"PM_NUM_SAMPLES" => constants.Add(name, shape.NumSamples),
				"PM_GEOM_CONSISTENCY" => constants.Add(name, shape.GeomConsistency),
				"PM_SWEEP_COOPERATIVE" => constants.Add(name, shape.CooperativeSweep),
				_ => throw new InvalidOperationException($"No value for the WGSL constant {name}."),
			};
		}

		return constants;
	}

	/// <summary>The bindings <paramref name="kernel"/>'s WGSL declares.</summary>
	public static IReadOnlyList<ComputeKernelBinding> Bindings(PatchMatchGpuKernel kernel)
	{
		const ComputeBindingType U = ComputeBindingType.Uniform;
		const ComputeBindingType R = ComputeBindingType.ReadOnlyStorage;
		const ComputeBindingType W = ComputeBindingType.Storage;
		return kernel switch
		{
			PatchMatchGpuKernel.InitRandom =>
			[
				new(0, PosesBinding, U), new(0, ProblemBinding, U), new(0, StateBinding, W),
			],
			PatchMatchGpuKernel.InitialCost =>
			[
				new(0, ByteToUnitBinding, U), new(0, PosesBinding, U), new(0, ProblemBinding, U),
				new(0, ReferenceBinding, R), new(0, SourceImagesBinding, R), new(0, SourceDepthsBinding, R),
				new(0, StateBinding, R), new(0, CostsBinding, W),
			],
			PatchMatchGpuKernel.BackwardMessages =>
			[
				new(0, PosesBinding, U), new(0, ProblemBinding, U), new(0, StateBinding, W),
				new(0, CostsBinding, R), new(0, SelProbsBinding, W), new(SweepGroup, 0, U),
			],
			PatchMatchGpuKernel.SweepBand =>
			[
				new(0, ByteToUnitBinding, U), new(0, PosesBinding, U), new(0, ProblemBinding, U),
				new(0, ReferenceBinding, R), new(0, SourceImagesBinding, R), new(0, SourceDepthsBinding, R),
				new(0, StateBinding, W), new(0, CostsBinding, W), new(0, SelProbsBinding, W),
				new(0, PrevSelProbsBinding, R), new(SweepGroup, 0, U), new(BandGroup, 0, U),
			],
			PatchMatchGpuKernel.FilterPixels =>
			[
				new(0, ByteToUnitBinding, U), new(0, PosesBinding, U), new(0, ProblemBinding, U),
				new(0, SourceImagesBinding, R), new(0, SourceDepthsBinding, R), new(0, StateBinding, W),
				new(0, CostsBinding, W), new(0, SelProbsBinding, R), new(SweepGroup, 0, U),
			],
			PatchMatchGpuKernel.RotatePlanes or PatchMatchGpuKernel.RotateNormals =>
			[
				new(0, RotateSourceBinding, R), new(0, RotateDestinationBinding, W), new(SweepGroup, 0, U),
			],
			_ => throw new ArgumentOutOfRangeException(nameof(kernel)),
		};
	}

	/// <summary>
	/// The descriptor of <paramref name="kernel"/> for <paramref name="shape"/>, dispatched with
	/// <paramref name="groupsX"/> workgroups along x. The same arguments give the same text.
	/// </summary>
	public static ComputeKernelDescriptor Descriptor(PatchMatchGpuKernel kernel, in PatchMatchGpuShaderShape shape, uint groupsX)
	{
		string source = PatchMatchShaders.Compose(Constants(kernel, shape, groupsX), [.. Parts(kernel)]);
		return new ComputeKernelDescriptor("patch_match_" + EntryPoint(kernel), source, EntryPoint(kernel), Bindings(kernel));
	}
}
