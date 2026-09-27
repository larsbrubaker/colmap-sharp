// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The plumbing of ReferenceComputeDevice (ReferenceComputeDevice.cs) between a recorded
// dispatch and a kernel twin (ReferenceComputeDevice.Kernels.cs): which PatchMatch kernel a
// descriptor is and the constants its text bakes in (parsed back out of the generated header,
// Compute/WgslConstants.cs), which element indices a dispatch's folded index reaches
// (patch_match_dispatch.wgsl), and typed views of the bound buffer ranges. Not a COLMAP port
// (PORTING_PLAN.md Phase 13).

using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

using ColmapSharp.Compute;
using ColmapSharp.Compute.Testing;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class ReferenceComputeDevice
{
	/// <summary>A kernel the twin knows: which one, and the constants its WGSL text holds.</summary>
	private sealed class TwinKernel
	{
		// One declaration of the generated header: `const NAME: i32 = 12i;` (u32 `12u`, bool).
		private static readonly Regex ConstantLine = new(
			@"^const (?<name>\w+): (?<type>i32|u32|bool) = (?<value>[^;]+);", RegexOptions.Multiline | RegexOptions.CultureInvariant);

		private readonly Dictionary<string, long> constants;

		private TwinKernel(PatchMatchGpuKernel kind, Dictionary<string, long> constants)
		{
			this.Kind = kind;
			this.constants = constants;
		}

		public PatchMatchGpuKernel Kind { get; }

		/// <summary>PM_GROUPS_X: the x size of the folded dispatch.</summary>
		public uint GroupsX => (uint)this.constants["PM_GROUPS_X"];

		/// <summary>
		/// The twin of <paramref name="descriptor"/>: its entry point must be a PatchMatch kernel,
		/// its bindings exactly that kernel's, and its header must define the kernel's constants.
		/// </summary>
		public static TwinKernel Parse(in ComputeKernelDescriptor descriptor)
		{
			PatchMatchGpuKernel? kind = null;
			foreach (PatchMatchGpuKernel candidate in PatchMatchGpuKernels.All)
			{
				if (PatchMatchGpuKernels.EntryPoint(candidate) == descriptor.EntryPoint)
				{
					kind = candidate;
				}
			}

			if (kind == null)
			{
				throw new ArgumentException(
					$"ReferenceComputeDevice runs only the GPU PatchMatch kernels; '{descriptor.EntryPoint}' is not one of their entry points.",
					nameof(descriptor));
			}

			var declared = new HashSet<ComputeKernelBinding>(descriptor.Bindings ?? []);
			if (!declared.SetEquals(PatchMatchGpuKernels.Bindings(kind.Value)))
			{
				throw new ArgumentException(
					$"Kernel '{descriptor.Label}' declares other bindings than {descriptor.EntryPoint}'s WGSL does"
					+ " (PatchMatchGpuKernels.Bindings), so the twin cannot tell what its buffers hold.",
					nameof(descriptor));
			}

			var constants = new Dictionary<string, long>(StringComparer.Ordinal);
			// Only the generated header (PatchMatchShaders.Compose puts it between these markers);
			// the parts declare constants of their own, in other literal forms.
			string source = descriptor.Source ?? string.Empty;
			const string HeaderMarker = "// ---- constants ----\n";
			int headerStart = source.IndexOf(HeaderMarker, StringComparison.Ordinal);
			int headerEnd = headerStart < 0 ? -1 : source.IndexOf("\n// ---- ", headerStart + HeaderMarker.Length, StringComparison.Ordinal);
			string header = headerStart < 0 ? string.Empty : source[(headerStart + HeaderMarker.Length)..(headerEnd < 0 ? source.Length : headerEnd)];
			foreach (Match match in ConstantLine.Matches(header))
			{
				string value = match.Groups["value"].Value.Trim();
				constants[match.Groups["name"].Value] = match.Groups["type"].Value switch
				{
					"bool" => value == "true" ? 1 : 0,
					"i32" when value == "(-2147483647i - 1i)" => int.MinValue,
					_ => long.Parse(value.TrimEnd('i', 'u'), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture),
				};
			}

			foreach (string name in PatchMatchGpuKernels.RequiredConstants(kind.Value))
			{
				if (!constants.ContainsKey(name))
				{
					throw new ArgumentException(
						$"Kernel '{descriptor.Label}' has no {name} in its constants header (PatchMatchGpuKernels.Constants writes it).",
						nameof(descriptor));
				}
			}

			return new TwinKernel(kind.Value, constants);
		}

		/// <summary>An integer or bool constant of the header (bools read 0 or 1).</summary>
		public int Int(string name) => checked((int)this.constants[name]);
	}

	/// <summary>
	/// Which element indices a dispatch reaches: invocation (x, y, local) of a folded dispatch
	/// runs element (y PM_GROUPS_X + x) 64 + local (patch_match_dispatch.wgsl).
	/// </summary>
	private readonly struct DispatchCoverage
	{
		private readonly uint groupsX;
		private readonly uint x;
		private readonly uint y;

		public DispatchCoverage(uint groupsX, uint x, uint y)
		{
			this.groupsX = groupsX;
			this.x = x;
			this.y = y;
		}

		/// <summary>
		/// Rejects a dispatch whose folded index would reach an element twice: x past
		/// PM_GROUPS_X wraps onto the next row of workgroups, and every z slice repeats the
		/// first. On a GPU both are races between invocations.
		/// </summary>
		public static void Validate(uint groupsX, uint x, uint z)
		{
			if (x > groupsX)
			{
				throw new ArgumentOutOfRangeException(
					nameof(x), x, $"The kernel was composed for {groupsX} workgroups along x (PM_GROUPS_X); more would run elements twice.");
			}

			if (z > 1)
			{
				throw new ArgumentOutOfRangeException(nameof(z), z, "The PatchMatch kernels ignore z, so a z size above 1 would run every element again.");
			}
		}

		/// <summary>True when some invocation of the dispatch runs element <paramref name="index"/>.</summary>
		public bool Covers(long index)
		{
			long workgroup = index / PatchMatchGpuKernels.WorkgroupSize;
			return workgroup % this.groupsX < this.x && workgroup / this.groupsX < this.y;
		}
	}

	/// <summary>The bound ranges of one recorded dispatch, as typed views of the live bytes.</summary>
	private readonly struct TwinBindings
	{
		private readonly DispatchCommand dispatch;

		public TwinBindings(DispatchCommand dispatch)
		{
			this.dispatch = dispatch;
		}

		public Span<byte> Bytes(int group, int binding)
		{
			foreach (ComputeBufferBinding entry in this.dispatch.Groups[group].Entries)
			{
				if (entry.Binding == binding)
				{
					var buffer = (RecordingComputeBuffer)entry.Buffer;
					return buffer.Memory.Slice(checked((int)entry.Offset), checked((int)entry.Size));
				}
			}

			throw new InvalidOperationException($"{this.dispatch.Kernel} has no binding {binding} in group {group}.");
		}

		public Span<float> Floats(int group, int binding) => MemoryMarshal.Cast<byte, float>(this.Bytes(group, binding));

		public Span<uint> Words(int group, int binding) => MemoryMarshal.Cast<byte, uint>(this.Bytes(group, binding));

		/// <summary>
		/// A group-0 binding holding at least <paramref name="count"/> 32-bit elements, or an error
		/// naming it: the WGSL would read or write past its end.
		/// </summary>
		public Span<float> Floats(int binding, long count, string what)
		{
			Span<float> values = this.Floats(0, binding);
			if (values.Length < count)
			{
				throw new InvalidOperationException(
					$"{this.dispatch.Kernel}: the {what} binding holds {values.Length} words but the kernel indexes {count}.");
			}

			return values;
		}

		/// <summary>A uniform of at least <paramref name="size"/> bytes.</summary>
		public ReadOnlySpan<byte> Uniform(int group, int binding, int size, string what)
		{
			Span<byte> bytes = this.Bytes(group, binding);
			if (bytes.Length < size)
			{
				throw new InvalidOperationException($"{this.dispatch.Kernel}: the {what} uniform binding has {bytes.Length} bytes, not {size}.");
			}

			return bytes;
		}
	}
}
