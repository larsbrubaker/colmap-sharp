// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// WgslBindingCheckerTests: C#-only tests (COLMAP has no compute seam) of
// ColmapSharp/Compute/WgslBindingChecker.cs, which compares a WGSL module's buffer
// declarations with its ComputeKernelDescriptor. The checker on the real composed PatchMatch
// shader parts is exercised in ColmapSharp.Tests/Mvs/PatchMatchShadersTests.cs.

using ColmapSharp.Compute;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Compute;

public class WgslBindingCheckerTests
{
	private const string Source = """
		struct Params { n: u32, }
		@group(0) @binding(0) var<storage, read> input: array<f32>;
		@binding(1) @group(0) var<storage, read_write> output: array<f32>;
		@group(1) @binding(0)
		var<uniform> params: Params;
		@group(0) @binding(2) var<storage> implicit_read: array<u32>;
		// @group(3) @binding(0) var<storage, read_write> commented_out: array<f32>;
		/* @group(4) @binding(0) var<uniform> also_commented: Params; */
		var<workgroup> tile: array<f32, 64>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) id: vec3<u32>) {
			var local: f32 = input[id.x];
			output[id.x] = local + f32(params.n) + f32(implicit_read[0]) + tile[0];
		}
		""";

	private static readonly ComputeKernelBinding[] Matching =
	{
		new(0, 0, ComputeBindingType.ReadOnlyStorage),
		new(0, 1, ComputeBindingType.Storage),
		new(0, 2, ComputeBindingType.ReadOnlyStorage),
		new(1, 0, ComputeBindingType.Uniform),
	};

	private static ComputeKernelDescriptor Descriptor(params ComputeKernelBinding[] bindings) => new("k", Source, "main", bindings);

	[Test]
	public async Task Parse_ReadsEveryBufferDeclarationAndSkipsComments()
	{
		(IReadOnlyList<WgslBindingDeclaration> declarations, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(Source);

		await Assert.That(unreadable.Count).IsEqualTo(0);
		await Assert.That(declarations).IsEquivalentTo(new List<WgslBindingDeclaration>
		{
			new(0, 0, ComputeBindingType.ReadOnlyStorage, "input"),
			new(0, 1, ComputeBindingType.Storage, "output"),
			new(1, 0, ComputeBindingType.Uniform, "params"),
			new(0, 2, ComputeBindingType.ReadOnlyStorage, "implicit_read"),
		});
	}

	[Test]
	public async Task Compare_MatchingDescriptor_HasNoProblems()
	{
		// Descriptor order does not matter.
		var reversed = Matching.Reverse().ToArray();

		await Assert.That(WgslBindingChecker.Compare(Descriptor(Matching)).Count).IsEqualTo(0);
		await Assert.That(WgslBindingChecker.Compare(Descriptor(reversed)).Count).IsEqualTo(0);
		WgslBindingChecker.Check(Descriptor(Matching));
	}

	[Test]
	public async Task Compare_DescriptorMissingABinding_ReportsIt()
	{
		IReadOnlyList<string> problems = WgslBindingChecker.Compare(Descriptor(Matching.Where(b => b.Group != 1).ToArray()));

		await Assert.That(problems).IsEquivalentTo(new List<string>
		{
			"WGSL declares @group(1) @binding(0) params as Uniform, but the descriptor has no such binding.",
		});
	}

	[Test]
	public async Task Compare_DescriptorWithAnExtraBinding_ReportsIt()
	{
		IReadOnlyList<string> problems = WgslBindingChecker.Compare(Descriptor(Matching.Append(new ComputeKernelBinding(2, 0, ComputeBindingType.Storage)).ToArray()));

		await Assert.That(problems).IsEquivalentTo(new List<string>
		{
			"The descriptor lists @group(2) @binding(0) as Storage, but the WGSL declares no such binding.",
		});
	}

	[Test]
	public async Task Compare_WrongAccess_ReportsIt()
	{
		ComputeKernelBinding[] wrong = Matching.Select(b => b.Binding == 1 ? b with { Type = ComputeBindingType.ReadOnlyStorage } : b).ToArray();

		IReadOnlyList<string> problems = WgslBindingChecker.Compare(Descriptor(wrong));

		await Assert.That(problems).IsEquivalentTo(new List<string>
		{
			"@group(0) @binding(1) output is Storage in the WGSL but ReadOnlyStorage in the descriptor.",
		});
		await Assert.That(() => WgslBindingChecker.Check(Descriptor(wrong))).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task Compare_UniformDeclaredAsStorage_ReportsIt()
	{
		ComputeKernelBinding[] wrong = Matching.Select(b => b.Group == 1 ? b with { Type = ComputeBindingType.ReadOnlyStorage } : b).ToArray();

		await Assert.That(WgslBindingChecker.Compare(Descriptor(wrong))).IsEquivalentTo(new List<string>
		{
			"@group(1) @binding(0) params is Uniform in the WGSL but ReadOnlyStorage in the descriptor.",
		});
	}

	[Test]
	public async Task Compare_DuplicatesAndUnreadableDeclarations_AreReported()
	{
		const string source = """
			const G: u32 = 0u;
			@group(0) @binding(0) var<storage, read> a: array<f32>;
			@group(0) @binding(0) var<storage, read> b: array<f32>;
			@group(G) @binding(1) var<storage, read> c: array<f32>;
			""";
		var descriptor = new ComputeKernelDescriptor("k", source, "main", new[]
		{
			new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage),
			new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage),
		});

		await Assert.That(WgslBindingChecker.Compare(descriptor)).IsEquivalentTo(new List<string>
		{
			"WGSL declares the buffer variable c without a literal @group and @binding.",
			"WGSL declares @group(0) @binding(0) twice (a and b).",
			"The descriptor lists @group(0) @binding(0) twice.",
		});
	}
}
