// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuPlan: the buffer and dispatch plan of a GPU PatchMatch run, and the decision
// whether a device can run it at all. Not a COLMAP port (COLMAP's PatchMatch allocates CUDA
// GpuMats as it goes and has no fallback); part of PORTING_PLAN.md Phase 13. The planner sizes
// every buffer the GPU orchestrator (PatchMatchGpu) creates, picks the rows per sweep dispatch,
// and checks all of it against the device's ComputeDeviceLimits (ColmapSharp/Compute/) and an
// optional host memory budget before a single buffer exists. When a check fails, the reason is
// a sentence a user can read, and PatchMatch runs on the CPU (PatchMatchCpu.cs) instead.
// Tests: ColmapSharp.Tests/Mvs/PatchMatchGpuPlanTests.cs (C#-only).
//
// Buffer layout (P = reference pixels, S = source images, L = source layer pixels, C = the
// larger of the reference width and height, all floats f32):
// - Reference planes, 2 buffers of 3P floats (image as byte / 255, bilateral sum, squared sum):
//   each rotation writes the rotated planes into the other buffer (ping-pong).
// - State, 2 buffers of 4P floats (depth, then the normal's three planes) followed by the
//   per-column sweep state, C x (S + 4) floats laid out as PatchMatchCpu.Sweep.cs's column
//   state [forward messages (S), prevDepth, prevNormal x3]. C columns cover both orientations.
//   Depth and normal ping-pong across rotations; the column area of each buffer is simply the
//   one used while that buffer is current (backward_messages rewrites it every sweep), which
//   costs one spare column area but keeps the depth, normal and column state in one binding.
// - Source maps, 3 buffers of S x P floats holding the cost, selection probability and
//   previous selection probability maps in rotation: after a sweep the selection map is
//   rotated into the previous map's buffer, the cost map into the old selection buffer, and
//   the old cost buffer becomes the new (fully rewritten) selection map.
// - Consistency mask: no buffer of its own. It is one u32 per (source, pixel), the size of a
//   cost map, and only exists on the last sweep, whose cost map is never read again (the
//   filter reads the selection map and depth/normal, not the costs). So the filter writes the
//   mask into the cost buffer, and after the selection map has been rotated into the previous
//   buffer the mask is rotated into the old selection buffer. Because that buffer still holds
//   float cost bits, the filter must write 0 or 1 to EVERY (source, pixel) entry, including
//   filtered pixels; the CPU can write only the 1s because its mask starts zeroed.
// - Source images, S x L bytes packed four per u32 (patch_match_textures.wgsl); source depth
//   maps, S x L floats when geometric consistency is on, else a 4-byte dummy (WGSL resolves
//   every declared name, so a photometric kernel still binds one).
// - Uniforms: the byte / 255 table (1024 bytes); the pose table, array<vec4<f32>, 8 + 43S>
//   as patch_match_geometry.wgsl reads it (floats [8r, 8r + 4) RefK(r), [8r + 4, 8r + 8)
//   RefInvK(r) for rotations r = 0..3, then source s of rotation r at float
//   32 + 43 (r S + s), unpadded; 4 x 43 S floats is always whole vec4s); one problem
//   uniform, one uniform per sweep and one per band per orientation, all created with their
//   data up front so nothing is written while dispatches are being recorded.
// Every storage buffer is bound whole, so a buffer's size is also its binding size.
//
// Dispatches are 1D workgroups of 64, folded into two dimensions when the workgroup count
// passes the device's per-dimension limit (PatchMatchGpuDispatch). A kernel computes its
// index (group.y * X + group.x) * 64 + local, and compares it with the element count, in u32:
// every count the plan accepts is below 2^31, but i32 intermediates could still wrap.

using System.Globalization;

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

/// <summary>What a <see cref="PatchMatchGpuPlan"/> buffer holds.</summary>
internal enum PatchMatchGpuBufferRole
{
	/// <summary>Reference image, bilateral sum and squared sum planes (ping-pong pair).</summary>
	ReferencePlanes,

	/// <summary>Depth and normal planes plus the per-column sweep state (ping-pong pair).</summary>
	State,

	/// <summary>Cost / selection / previous selection maps, and the mask on the last sweep (ring of three).</summary>
	SourceMaps,

	/// <summary>Source image layers, bytes packed four per u32.</summary>
	SourceImages,

	/// <summary>Source depth layers, or a 4-byte dummy for a photometric run.</summary>
	SourceDepths,

	/// <summary>The byte / 255 lookup table uniform.</summary>
	ByteTable,

	/// <summary>The per-source pose tables of all four rotations.</summary>
	PoseTable,

	/// <summary>The per-problem options and derived constants.</summary>
	Problem,

	/// <summary>One uniform per sweep (rotation, per-sweep constants).</summary>
	SweepUniforms,

	/// <summary>One uniform per row band of each orientation.</summary>
	BandUniforms,
}

/// <summary>
/// <paramref name="Count"/> buffers of <paramref name="Size"/> bytes each, all of one role
/// and kind. <paramref name="Description"/> names the contents in words a user reads in a
/// fallback reason.
/// </summary>
internal readonly record struct PatchMatchGpuBuffer(
	PatchMatchGpuBufferRole Role,
	ComputeBufferKind Kind,
	long Size,
	int Count,
	string Description)
{
	/// <summary>Bytes of all <see cref="Count"/> buffers.</summary>
	public long TotalBytes => Size * Count;
}

/// <summary>
/// The largest dispatch of one kernel: <paramref name="Elements"/> invocations run as
/// <paramref name="X"/> x <paramref name="Y"/> workgroups of
/// <see cref="PatchMatchGpuPlan.WorkgroupSize"/>. The linear index is folded into two
/// dimensions when one would exceed the device limit; the kernel recovers it as
/// <c>(group.y * X + group.x) * WorkgroupSize + local</c>, in u32, and skips indices at or
/// past <paramref name="Elements"/> (compared in u32 too).
/// </summary>
internal readonly record struct PatchMatchGpuDispatch(string Kernel, long Elements, uint X, uint Y);

/// <summary>The sizes of a PatchMatch problem the plan depends on.</summary>
/// <param name="RefWidth">Reference image width (rotation 0).</param>
/// <param name="RefHeight">Reference image height (rotation 0).</param>
/// <param name="NumSources">Number of source images S.</param>
/// <param name="SourceLayerWidth">Source layer width (the largest source width, PatchMatchSourceImages.MaxWidth).</param>
/// <param name="SourceLayerHeight">Source layer height (PatchMatchSourceImages.MaxHeight).</param>
internal readonly record struct PatchMatchGpuProblemShape(
	int RefWidth,
	int RefHeight,
	int NumSources,
	int SourceLayerWidth,
	int SourceLayerHeight);

/// <summary>
/// The buffers, band sizes and dispatch shapes of a GPU PatchMatch run on a particular
/// device, or (through <see cref="TryCreate"/>) the reason the device cannot run it.
/// </summary>
internal sealed class PatchMatchGpuPlan
{
	/// <summary>Invocations per workgroup of every PatchMatch kernel (1D).</summary>
	public const int WorkgroupSize = 64;

	/// <summary>
	/// Floats per source per rotation in the pose-table uniform: PatchMatchTransforms'
	/// NumTformParams, packed with no per-record padding.
	/// </summary>
	public const int PoseFloatsPerSource = PatchMatchTransforms.NumTformParams;

	/// <summary>
	/// vec4s at the start of the pose-table uniform: RefK(r) and RefInvK(r), one vec4 each,
	/// for the four rotations r.
	/// </summary>
	public const int PoseTableHeaderVec4 = 8;

	/// <summary>The byte / 255 table: 256 floats (patch_match_textures.wgsl's PmByteToUnit).</summary>
	public const int ByteTableBytes = 256 * 4;

	/// <summary>
	/// Bytes reserved for the problem uniform (options and constants derived in C#: depth
	/// range, NCC normalization, likelihood constants, filter thresholds). The kernels' WGSL
	/// struct must fit; 256 bytes is 64 scalars.
	/// </summary>
	public const int ProblemUniformBytes = 256;

	/// <summary>
	/// Bytes reserved for a sweep uniform: rotation index, perturbation, previous-selection
	/// weight and sweep flags (the rotation's reference frame is in the pose table).
	/// </summary>
	public const int SweepUniformBytes = 64;

	/// <summary>Bytes of a band uniform: first row and end row, padded to 16.</summary>
	public const int BandUniformBytes = 16;

	/// <summary>
	/// Storage bindings of sweep_band, the kernel with the most: reference planes, source
	/// images, source depths, state, cost, selection, previous selection.
	/// </summary>
	public const int SweepStorageBindings = 7;

	/// <summary>Uniform bindings of sweep_band: problem, byte table, pose table (group 0), sweep (group 1), band (group 2).</summary>
	public const int SweepUniformBindings = 5;

	/// <summary>Bind groups: 0 static buffers, 1 the sweep uniform, 2 the band uniform.</summary>
	public const int BindGroups = 3;

	/// <summary>
	/// The work one sweep_band dispatch aims for, in NCC window samples (see
	/// <see cref="WindowSamplesPerPixel"/>). 2^26 samples take about 50 ms at 1.3 billion
	/// window samples per second, a conservative figure for an integrated GPU, which keeps each
	/// dispatch far below the watchdogs that reset a GPU stuck in one submission (2 s on
	/// Windows) while still putting whole rows of work in each dispatch.
	/// </summary>
	public const long TargetWindowSamplesPerDispatch = 1L << 26;

	// PatchMatchCpu.Sweep.cs's hypotheses per pixel besides the current one, each scored
	// against every sampled source.
	private const int NewHypotheses = 4;

	private PatchMatchGpuPlan(
		PatchMatchGpuProblemShape shape,
		bool geometric,
		bool filter,
		int windowCount,
		long windowSamplesPerPixel,
		int[] bandRows,
		int[] bandCounts,
		PatchMatchGpuBuffer[] buffers,
		PatchMatchGpuDispatch[] dispatches)
	{
		Shape = shape;
		Geometric = geometric;
		Filter = filter;
		WindowCount = windowCount;
		WindowSamplesPerPixel = windowSamplesPerPixel;
		this.bandRows = bandRows;
		this.bandCounts = bandCounts;
		Buffers = buffers;
		Dispatches = dispatches;
		foreach (PatchMatchGpuBuffer buffer in buffers)
		{
			TotalBytes += buffer.TotalBytes;
			if (buffer.Kind == ComputeBufferKind.Storage)
			{
				LargestStorageBinding = Math.Max(LargestStorageBinding, buffer.Size);
			}
		}
	}

	private readonly int[] bandRows;
	private readonly int[] bandCounts;

	/// <summary>The problem sizes planned for.</summary>
	public PatchMatchGpuProblemShape Shape { get; }

	/// <summary>Whether the run uses geometric consistency (source depth maps bound).</summary>
	public bool Geometric { get; }

	/// <summary>Whether the last sweep filters (consistency mask written).</summary>
	public bool Filter { get; }

	/// <summary>Samples in the strided NCC window (PatchMatchPhotoConsistency.WindowCount).</summary>
	public int WindowCount { get; }

	/// <summary>
	/// The cost model of a swept pixel, in window samples: 4 new hypotheses scored against
	/// each of NumSamples sampled sources, and (when a new hypothesis wins) the winner scored
	/// against all S sources for the forward messages - WindowCount x (4 x NumSamples + S).
	/// </summary>
	public long WindowSamplesPerPixel { get; }

	/// <summary>Every buffer the run creates, in creation order.</summary>
	public IReadOnlyList<PatchMatchGpuBuffer> Buffers { get; }

	/// <summary>The largest dispatch of each kernel the run uses.</summary>
	public IReadOnlyList<PatchMatchGpuDispatch> Dispatches { get; }

	/// <summary>Bytes of every buffer together: what the run keeps resident on the device.</summary>
	public long TotalBytes { get; }

	/// <summary>The largest storage buffer, which is also the largest storage binding.</summary>
	public long LargestStorageBinding { get; }

	/// <summary>
	/// Rows per sweep_band dispatch in an orientation: 0 for rotations 0 and 2 (the reference
	/// width across, its height down), 1 for rotations 1 and 3 (height across, width down).
	/// </summary>
	public int BandRows(int orientation) => bandRows[orientation];

	/// <summary>Bands (sweep_band dispatches and band uniforms) per sweep in an orientation.</summary>
	public int BandCount(int orientation) => bandCounts[orientation];

	/// <summary>The number of sweep uniforms: four per iteration.</summary>
	public int SweepCount => Buffers.First(b => b.Role == PatchMatchGpuBufferRole.SweepUniforms).Count;

	/// <summary>
	/// Plans a run of <paramref name="options"/> on <paramref name="shape"/> for a device with
	/// <paramref name="limits"/>. Returns false with a user-readable
	/// <paramref name="fallbackReason"/> (ending in "Using the CPU.") when the device cannot
	/// run it: the first failing check, in a fixed order, so the reason is deterministic.
	/// <paramref name="memoryBudget"/>, when given, caps the bytes all buffers may take.
	/// <paramref name="bandRows"/>, when positive, replaces the planned rows per sweep_band
	/// dispatch (clamped to each orientation's rows); only tests set it, to force several bands
	/// per sweep on a small problem (PatchMatchGpuTwinTests).
	/// </summary>
	public static bool TryCreate(
		PatchMatchGpuProblemShape shape,
		PatchMatchOptions options,
		ComputeDeviceLimits limits,
		long? memoryBudget,
		out PatchMatchGpuPlan? plan,
		out string? fallbackReason,
		int bandRows = 0)
	{
		// A malformed problem is a caller bug, not a reason to fall back.
		Util.Check.That(shape.RefWidth > 0 && shape.RefHeight > 0);
		Util.Check.That(shape.NumSources > 0);
		Util.Check.That(shape.SourceLayerWidth > 0 && shape.SourceLayerHeight > 0);
		Util.Check.That(options.WindowRadius > 0 && options.WindowStep > 0);
		Util.Check.That(options.NumSamples > 0 && options.NumIterations > 0);

		plan = Build(shape, options, limits.MaxComputeWorkgroupsPerDimension, bandRows);
		fallbackReason = FirstFailure(plan, limits, memoryBudget);
		if (fallbackReason != null)
		{
			plan = null;
			return false;
		}

		return true;
	}

	private static PatchMatchGpuPlan Build(PatchMatchGpuProblemShape shape, PatchMatchOptions options, uint maxWorkgroupsPerDimension, int bandRowsOverride)
	{
		long w = shape.RefWidth;
		long h = shape.RefHeight;
		long s = shape.NumSources;
		long p = w * h;
		long layer = (long)shape.SourceLayerWidth * shape.SourceLayerHeight;
		long columns = Math.Max(w, h);
		bool geometric = options.GeomConsistency;
		bool filter = options.Filter;

		int windowCount = PatchMatchPhotoConsistency.WindowCountFor(options.WindowRadius, options.WindowStep);
		long samplesPerPixel = windowCount * (NewHypotheses * (long)options.NumSamples + s);

		// Orientation 0 sweeps w columns down h rows; orientation 1, h columns down w rows.
		int[] bandRows = bandRowsOverride > 0
			? [(int)Math.Min(bandRowsOverride, h), (int)Math.Min(bandRowsOverride, w)]
			: [RowsPerBand(w, h, samplesPerPixel), RowsPerBand(h, w, samplesPerPixel)];
		int[] bandCounts = [(int)CeilDiv(h, bandRows[0]), (int)CeilDiv(w, bandRows[1])];
		int sweeps = 4 * options.NumIterations;

		PatchMatchGpuBuffer[] buffers =
		[
			new(PatchMatchGpuBufferRole.ReferencePlanes, ComputeBufferKind.Storage, 3 * p * 4, 2, "reference image"),
			new(PatchMatchGpuBufferRole.State, ComputeBufferKind.Storage, (4 * p + columns * (s + 4)) * 4, 2, "depth and normal maps"),
			new(PatchMatchGpuBufferRole.SourceMaps, ComputeBufferKind.Storage, s * p * 4, 3, "matching costs of all source images"),
			new(PatchMatchGpuBufferRole.SourceImages, ComputeBufferKind.Storage, RoundUp(s * layer, 4), 1, "source images"),
			new(PatchMatchGpuBufferRole.SourceDepths, ComputeBufferKind.Storage, geometric ? s * layer * 4 : 4, 1, "source depth maps"),
			new(PatchMatchGpuBufferRole.ByteTable, ComputeBufferKind.Uniform, ByteTableBytes, 1, "color lookup table"),
			new(PatchMatchGpuBufferRole.PoseTable, ComputeBufferKind.Uniform, 16 * (PoseTableHeaderVec4 + PoseFloatsPerSource * s), 1, "camera poses of the source images"),
			new(PatchMatchGpuBufferRole.Problem, ComputeBufferKind.Uniform, ProblemUniformBytes, 1, "PatchMatch settings"),
			new(PatchMatchGpuBufferRole.SweepUniforms, ComputeBufferKind.Uniform, SweepUniformBytes, sweeps, "sweep settings"),
			new(PatchMatchGpuBufferRole.BandUniforms, ComputeBufferKind.Uniform, BandUniformBytes, bandCounts[0] + bandCounts[1], "row band settings"),
		];

		uint maxGroups = maxWorkgroupsPerDimension;
		var dispatches = new List<PatchMatchGpuDispatch>();
		if (!geometric)
		{
			dispatches.Add(Dispatch("init_random", p, maxGroups));
		}

		dispatches.Add(Dispatch("initial_cost", p * s, maxGroups));
		dispatches.Add(Dispatch("backward_messages", columns * s, maxGroups));
		dispatches.Add(Dispatch("sweep_band", columns, maxGroups));
		if (filter)
		{
			dispatches.Add(Dispatch("filter_pixels", p, maxGroups));
		}

		// The widest copy is a whole source map (or mask); the reference planes are 3 planes.
		dispatches.Add(Dispatch("rotate_planes", Math.Max(3, s) * p, maxGroups));
		dispatches.Add(Dispatch("rotate_normals", p, maxGroups));

		return new PatchMatchGpuPlan(shape, geometric, filter, windowCount, samplesPerPixel, bandRows, bandCounts, buffers, [.. dispatches]);
	}

	// The most rows whose window samples stay within TargetWindowSamplesPerDispatch, at least
	// one and at most the whole height.
	private static int RowsPerBand(long columns, long rows, long samplesPerPixel)
	{
		long perRow = columns * samplesPerPixel;
		long fit = TargetWindowSamplesPerDispatch / perRow;
		return (int)Math.Clamp(fit, 1, rows);
	}

	// Workgroups along x up to the device limit, the rest folded into y. A y past the limit
	// is kept (clamped to uint) for FirstFailure to reject.
	private static PatchMatchGpuDispatch Dispatch(string kernel, long elements, uint maxWorkgroupsPerDimension)
	{
		long groups = CeilDiv(elements, WorkgroupSize);
		long x = Math.Min(groups, maxWorkgroupsPerDimension);
		long y = CeilDiv(groups, x);
		return new(kernel, elements, (uint)x, (uint)Math.Min(y, uint.MaxValue));
	}

	private static string? FirstFailure(PatchMatchGpuPlan plan, ComputeDeviceLimits limits, long? memoryBudget)
	{
		if (limits.MaxComputeInvocationsPerWorkgroup < WorkgroupSize)
		{
			return $"The GPU runs at most {limits.MaxComputeInvocationsPerWorkgroup} threads per workgroup; PatchMatch needs {WorkgroupSize}. Using the CPU.";
		}

		if (limits.MaxBindGroups < BindGroups)
		{
			return $"The GPU allows {limits.MaxBindGroups} bind groups per shader; PatchMatch needs {BindGroups}. Using the CPU.";
		}

		if (limits.MaxStorageBuffersPerShaderStage < SweepStorageBindings)
		{
			return $"The GPU allows {limits.MaxStorageBuffersPerShaderStage} storage buffers per shader; PatchMatch needs {SweepStorageBindings}. Using the CPU.";
		}

		if (limits.MaxUniformBuffersPerShaderStage < SweepUniformBindings)
		{
			return $"The GPU allows {limits.MaxUniformBuffersPerShaderStage} uniform buffers per shader; PatchMatch needs {SweepUniformBindings}. Using the CPU.";
		}

		// The kernels index every storage buffer with i32 (the source images by byte), so no
		// buffer may hold more than int.MaxValue elements. Only reachable with raised limits.
		foreach (PatchMatchGpuBuffer buffer in plan.Buffers)
		{
			long elements = buffer.Role == PatchMatchGpuBufferRole.SourceImages ? buffer.Size : buffer.Size / 4;
			if (buffer.Kind == ComputeBufferKind.Storage && elements > int.MaxValue)
			{
				return $"This image is too large for the GPU: its {buffer.Description} need {Need(buffer.Size)}, more than the GPU can address in one buffer. {TrySmaller} Using the CPU.";
			}
		}

		foreach (PatchMatchGpuBuffer buffer in plan.Buffers)
		{
			if (buffer.Size > limits.MaxBufferSize)
			{
				return $"The GPU can hold at most {Limit(limits.MaxBufferSize)} in one buffer; this image's {buffer.Description} need {Need(buffer.Size)}. {TrySmaller} Using the CPU.";
			}
		}

		foreach (PatchMatchGpuBuffer buffer in plan.Buffers)
		{
			if (buffer.Kind == ComputeBufferKind.Storage && buffer.Size > limits.MaxStorageBufferBindingSize)
			{
				return $"The GPU can bind at most {Limit(limits.MaxStorageBufferBindingSize)} per buffer; this image's {buffer.Description} need {Need(buffer.Size)}. {TrySmaller} Using the CPU.";
			}

			if (buffer.Kind == ComputeBufferKind.Uniform && buffer.Size > limits.MaxUniformBufferBindingSize)
			{
				return $"The GPU can bind at most {Limit(limits.MaxUniformBufferBindingSize)} of uniform data per buffer; the {buffer.Description} need {Need(buffer.Size)}. {TryFewerSources} Using the CPU.";
			}
		}

		foreach (PatchMatchGpuDispatch d in plan.Dispatches)
		{
			if (d.Y > limits.MaxComputeWorkgroupsPerDimension)
			{
				// Speak in work items (invocations), not kernels or workgroups.
				long capacity = (long)limits.MaxComputeWorkgroupsPerDimension * limits.MaxComputeWorkgroupsPerDimension * WorkgroupSize;
				return $"This image is too large for the GPU to process in one pass: it needs {d.Elements.ToString("N0", CultureInfo.InvariantCulture)} work items, and the GPU runs at most {capacity.ToString("N0", CultureInfo.InvariantCulture)}. {TrySmaller} Using the CPU.";
			}
		}

		if (memoryBudget is long budget && plan.TotalBytes > budget)
		{
			return $"This image needs {Need(plan.TotalBytes)} of GPU memory, more than the {Limit(budget)} available. {TrySmaller} Using the CPU.";
		}

		return null;
	}

	// What a user can change to fit: every size and dispatch the plan checks grows with the
	// source image count or the image size (PatchMatchOptions.MaxImageSize).
	private const string TrySmaller = "Try fewer source images or a smaller maximum image size.";

	// The only uniform that grows with the problem is the pose table, which the image size
	// does not affect.
	private const string TryFewerSources = "Try fewer source images.";

	private static long CeilDiv(long a, long b) => (a + b - 1) / b;

	private static long RoundUp(long value, long multiple) => CeilDiv(value, multiple) * multiple;

	// A requirement is rounded up and a limit down, so a failing check never reads as
	// "needs 128 MiB, at most 128 MiB".
	private static string Need(long bytes) => FormatBytes(bytes, roundUp: true);

	private static string Limit(long bytes) => FormatBytes(bytes, roundUp: false);

	private static string FormatBytes(long bytes, bool roundUp)
	{
		const long KiB = 1024;
		const long MiB = 1024 * 1024;
		long unit = bytes >= MiB ? MiB : KiB;
		long value = roundUp ? CeilDiv(bytes, unit) : bytes / unit;
		return value.ToString("N0", CultureInfo.InvariantCulture) + (unit == MiB ? " MiB" : " KiB");
	}
}
