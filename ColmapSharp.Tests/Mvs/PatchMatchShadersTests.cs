// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchShadersTests: C#-only tests (COLMAP's PatchMatch kernels are CUDA) of
// ColmapSharp/Mvs/PatchMatchShaders.cs and ColmapSharp/Compute/WgslConstants.cs: the embedded
// .wgsl files load, composition is deterministic, the constants header writes every f32
// exactly, and sample kernels built on the shared parts agree with their descriptors
// (Compute/WgslBindingChecker.cs). None of this needs a GPU; whether a GPU compiles and runs
// the text is the conformance kit's job (PORTING_PLAN.md Phase 13).

using System.Globalization;
using System.Text.RegularExpressions;

using ColmapSharp.Compute;
using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchShadersTests
{
	// A kernel using every helper of the common and textures parts, with the bindings the
	// textures part requires. (Checked with naga 30, wgpu's WGSL front end, when written.)
	private const string SampleKernel = """
		@group(0) @binding(0) var<uniform> pm_byte_to_unit: PmByteToUnit;
		@group(0) @binding(1) var<storage, read> pm_source_images: array<u32>;
		@group(0) @binding(2) var<storage, read> pm_source_depths: array<f32>;
		@group(0) @binding(3) var<storage, read_write> out_values: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) id: vec3<u32>) {
			var rng = pm_random_init(vec2<u32>(1u, 2u), i32(id.x), 3, -2);
			out_values[id.x] = pm_random_next_uniform(&rng) + pm_sample_source_image(1.5, 2.5, 0) + pm_sample_source_depth(1.0, 1.0, 0) + cuda_min(1.0, PM_X) + cuda_max(0.0, 2.0);
		}
		""";

	private static readonly ComputeKernelBinding[] SampleKernelBindings =
	{
		new(0, 0, ComputeBindingType.Uniform),
		new(0, 1, ComputeBindingType.ReadOnlyStorage),
		new(0, 2, ComputeBindingType.ReadOnlyStorage),
		new(0, 3, ComputeBindingType.Storage),
	};

	private static WgslConstants SampleConstants() => new WgslConstants()
		.Add("PM_SRC_MAX_WIDTH", 640)
		.Add("PM_SRC_MAX_HEIGHT", 480)
		.Add("PM_X", 0.1f);

	[Test]
	public async Task Load_ReturnsEachEmbeddedFileWithItsHeader()
	{
		foreach (string part in new[] { PatchMatchShaders.Common, PatchMatchShaders.Textures })
		{
			string text = PatchMatchShaders.Load(part);
			await Assert.That(text.StartsWith("// Copyright (c) 2026, Lars Brubaker.", StringComparison.Ordinal)).IsTrue();
			await Assert.That(text.Contains('\r')).IsFalse();
		}

		await Assert.That(() => PatchMatchShaders.Load("no_such_part.wgsl")).Throws<ArgumentException>();
	}

	[Test]
	public async Task Compose_IsDeterministicAndOrdersItsParts()
	{
		// Constants added in a different order give the same header.
		WgslConstants reordered = new WgslConstants()
			.Add("PM_X", 0.1f)
			.Add("PM_SRC_MAX_HEIGHT", 480)
			.Add("PM_SRC_MAX_WIDTH", 640);

		string first = PatchMatchShaders.Compose(SampleConstants(), PatchMatchShaders.Common, PatchMatchShaders.Textures);
		string second = PatchMatchShaders.Compose(reordered, PatchMatchShaders.Common, PatchMatchShaders.Textures);

		await Assert.That(second).IsEqualTo(first);
		string expectedStart = "// Composed by ColmapSharp.Mvs.PatchMatchShaders.Compose from: patch_match_common.wgsl, patch_match_textures.wgsl\n"
			+ "// ---- constants ----\n"
			+ "const PM_SRC_MAX_HEIGHT: i32 = 480i;\n"
			+ "const PM_SRC_MAX_WIDTH: i32 = 640i;\n"
			+ "const PM_X: f32 = 0x1.99999Ap-4f; // 0.1\n"
			+ "\n// ---- patch_match_common.wgsl ----\n"
			+ PatchMatchShaders.Load(PatchMatchShaders.Common);
		await Assert.That(first.StartsWith(expectedStart, StringComparison.Ordinal)).IsTrue();
		await Assert.That(first.EndsWith("\n// ---- patch_match_textures.wgsl ----\n" + PatchMatchShaders.Load(PatchMatchShaders.Textures), StringComparison.Ordinal)).IsTrue();
		await Assert.That(() => PatchMatchShaders.Compose(SampleConstants(), PatchMatchShaders.Common, PatchMatchShaders.Common)).Throws<ArgumentException>();
		await Assert.That(() => PatchMatchShaders.Compose(SampleConstants())).Throws<ArgumentException>();
	}

	[Test]
	public async Task Constants_WriteEveryTypeInWgslSyntax()
	{
		string header = new WgslConstants()
			.Add("B_FALSE", false)
			.Add("B_TRUE", true)
			.Add("I_MIN", int.MinValue)
			.Add("I_NEG", -7)
			.Add("U_MAX", uint.MaxValue)
			.ToWgsl();

		await Assert.That(header).IsEqualTo(
			"const B_FALSE: bool = false;\n"
			+ "const B_TRUE: bool = true;\n"
			+ "const I_MIN: i32 = (-2147483647i - 1i);\n"
			+ "const I_NEG: i32 = -7i;\n"
			+ "const U_MAX: u32 = 4294967295u;\n");
		await Assert.That(() => new WgslConstants().Add("A", 1).Add("A", 2)).Throws<ArgumentException>();
		await Assert.That(() => new WgslConstants().Add("1A", 1)).Throws<ArgumentException>();
		await Assert.That(() => new WgslConstants().Add("A-B", 1)).Throws<ArgumentException>();
		await Assert.That(() => new WgslConstants().Add("F", float.NaN)).Throws<ArgumentException>();
		await Assert.That(() => new WgslConstants().Add("F", float.PositiveInfinity)).Throws<ArgumentException>();
	}

	[Test]
	public async Task Constants_WriteFloatsExactly()
	{
		var values = new List<float>
		{
			0.0f, -0.0f, 1.0f, -1.5f, 0.1f, 1.0f / 3.0f, float.MaxValue, float.MinValue, float.Epsilon, -float.Epsilon,
			BitConverter.UInt32BitsToSingle(0x00800000u), BitConverter.UInt32BitsToSingle(0x007FFFFFu),
			BitConverter.UInt32BitsToSingle(0x2F000000u), 2.3283064e-10f, 3.14159265f,
		};
		var picker = new Random(5);
		while (values.Count < 20000)
		{
			float value = BitConverter.UInt32BitsToSingle((uint)picker.NextInt64(0, 1L << 32));
			if (float.IsFinite(value))
			{
				values.Add(value);
			}
		}

		int mismatches = 0;
		foreach (float value in values)
		{
			if (BitConverter.SingleToUInt32Bits(ParseHexFloat(WgslConstants.FormatF32(value))) != BitConverter.SingleToUInt32Bits(value))
			{
				mismatches++;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
		await Assert.That(WgslConstants.FormatF32(1.0f)).IsEqualTo("0x1.000000p+0f");
		await Assert.That(WgslConstants.FormatF32(-0.0f)).IsEqualTo("-0.0f");
		await Assert.That(WgslConstants.FormatF32(float.Epsilon)).IsEqualTo("0x0.000002p-126f");
		await Assert.That(WgslConstants.FormatF32(float.MaxValue)).IsEqualTo("0x1.FFFFFEp+127f");
	}

	[Test]
	public async Task SampleKernel_OnCommonAndTextures_MatchesItsDescriptor()
	{
		string source = PatchMatchShaders.Compose(SampleConstants(), PatchMatchShaders.Common, PatchMatchShaders.Textures) + SampleKernel;
		var descriptor = new ComputeKernelDescriptor("sample", source, "main", SampleKernelBindings);

		// The shared parts declare no bindings of their own, so the kernel's are all there is.
		(IReadOnlyList<WgslBindingDeclaration> declared, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(source);
		await Assert.That(declared.Count).IsEqualTo(4);
		await Assert.That(unreadable.Count).IsEqualTo(0);
		await Assert.That(WgslBindingChecker.Compare(descriptor).Count).IsEqualTo(0);

		// The binding list in textures' header, mentioned in comments only, is not parsed.
		ComputeKernelBinding[] withoutDepths = SampleKernelBindings.Where(b => b.Binding != 2).ToArray();
		await Assert.That(WgslBindingChecker.Compare(descriptor with { Bindings = withoutDepths })).IsEquivalentTo(new List<string>
		{
			"WGSL declares @group(0) @binding(2) pm_source_depths as ReadOnlyStorage, but the descriptor has no such binding.",
		});
	}

	[Test]
	public async Task Common_DeclaresNoBindings()
	{
		string source = PatchMatchShaders.Compose(new WgslConstants(), PatchMatchShaders.Common);

		(IReadOnlyList<WgslBindingDeclaration> declared, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(source);

		await Assert.That(declared.Count + unreadable.Count).IsEqualTo(0);
	}

	[Test]
	public async Task Common_RandomConstantsAreSplitMix64s()
	{
		// The transliteration test copies these words; pin them to the 64-bit constants of
		// PatchMatchRandom.cs so a typo in the WGSL cannot hide behind the same typo in the copy.
		string common = PatchMatchShaders.Load(PatchMatchShaders.Common);

		await Assert.That(common.Contains(Words(0x9E3779B97F4A7C15UL), StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains(Words(0xBF58476D1CE4E5B9UL), StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains(Words(0x94D049BB133111EBUL), StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains("u64_shr(z, 30u)", StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains("u64_shr(z, 27u)", StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains("u64_shr(z, 31u)", StringComparison.Ordinal)).IsTrue();
	}

	private static string Words(ulong value)
		=> string.Create(CultureInfo.InvariantCulture, $"vec2<u32>(0x{(uint)value:X8}u, 0x{(uint)(value >> 32):X8}u)");

	// C#-only. WGSL makes a const-expression that evaluates to NaN or an infinity a shader-creation
	// error. Dawn's Tint (the browser) enforces it; naga (wgpu, the desktop) accepted
	// bitcast<f32>(PM_NAN_BITS), so every PatchMatch kernel failed to compile only in the browser
	// ("value nan cannot be represented as 'f32'"). A non-finite f32 has to be built from a runtime value.
	[Test]
	public async Task Kernels_NeverBitcastAConstantToANonFiniteFloat()
	{
		var parts = PatchMatchGpuKernels.All.SelectMany(PatchMatchGpuKernels.Parts).Distinct().ToList();
		// Comments stripped: the shaders explain the rule by quoting the forbidden form.
		var texts = parts.Select(part => Regex.Replace(PatchMatchShaders.Load(part), "//[^\\n]*", string.Empty)).ToList();

		var constants = new Dictionary<string, uint>(StringComparer.Ordinal);
		foreach (Match match in texts.SelectMany(text => Regex.Matches(text, @"const\s+(\w+)\s*:\s*u32\s*=\s*0x([0-9A-Fa-f]+)u\s*;")))
		{
			constants[match.Groups[1].Value] = uint.Parse(match.Groups[2].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
		}

		int constantBitcasts = 0;
		for (int i = 0; i < parts.Count; i++)
		{
			foreach (Match match in Regex.Matches(texts[i], @"bitcast<f32>\(\s*(?:0x([0-9A-Fa-f]+)u|(\w+))\s*\)"))
			{
				uint bits;
				if (match.Groups[1].Success)
				{
					bits = uint.Parse(match.Groups[1].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
				}
				else if (!constants.TryGetValue(match.Groups[2].Value, out bits))
				{
					continue;
				}

				constantBitcasts++;
				await Assert.That((bits & 0x7F800000u) != 0x7F800000u)
					.IsTrue()
					.Because($"{parts[i]}: {match.Value} is a const-expression with non-finite bits 0x{bits:X8}");
			}
		}

		// The finite literals in pm_random_next_uniform, so the scan is known to see bitcasts at all.
		await Assert.That(constantBitcasts).IsGreaterThanOrEqualTo(2);
	}

	/// <summary>
	/// An independent reading of FormatF32's literals: the significand and power of two as a
	/// double (exact for every f32), narrowed to float (exact too).
	/// </summary>
	private static float ParseHexFloat(string literal)
	{
		if (literal == "0.0f" || literal == "-0.0f")
		{
			return literal[0] == '-' ? -0.0f : 0.0f;
		}

		Match match = Regex.Match(literal, @"^(-?)0x([01])\.([0-9A-F]{6})p([+-]\d+)f$");
		if (!match.Success)
		{
			throw new FormatException($"Not a WGSL hex float literal: {literal}");
		}

		double significand = int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture)
			+ int.Parse(match.Groups[3].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture) / (double)(1 << 24);
		double value = Math.ScaleB(significand, int.Parse(match.Groups[4].Value, CultureInfo.InvariantCulture));
		return (float)(match.Groups[1].Value == "-" ? -value : value);
	}
}
