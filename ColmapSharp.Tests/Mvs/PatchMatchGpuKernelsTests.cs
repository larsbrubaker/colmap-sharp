// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuKernelsTests: C#-only tests (COLMAP's PatchMatch kernels are CUDA) of
// ColmapSharp/Mvs/PatchMatchGpuKernels.cs and the kernel .wgsl files. Without a GPU the suite
// checks that every kernel composes deterministically, agrees with its descriptor
// (WgslBindingChecker), stays within WebGPU's default binding limits, uses one binding number
// per buffer across kernels, and that the uniform structs have the sizes and field orders the host
// packs. TRANSLITERATION CHECKS, NOT PRODUCTION CODE, follow the pattern of
// PatchMatchShaderRngTransliterationTests: line-by-line C# copies of the WGSL index math
// (pm_to_original_pixel, the rotate kernels' moves, the folded dispatch index), run against
// PatchMatchTransforms.ToOriginalPixel, Mat.Rotate and PatchMatchKernel.RotateNormalMap. Keep the
// copies in step with the WGSL. Whether a GPU runs the text the same way is the conformance
// kit's job (PORTING_PLAN.md Phase 13); naga 30 validated every composed kernel (and translated
// it to MSL and SPIR-V) when it was written.

using System.Text.RegularExpressions;

using ColmapSharp.Compute;
using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchGpuKernelsTests
{
	private static readonly PatchMatchGpuShaderShape Photometric = new(64, 48, 3, 70, 50, 5, 1, 15, false);

	// A large window (uncached) and a non-default step.
	private static readonly PatchMatchGpuShaderShape Geometric = new(33, 71, 20, 80, 80, 20, 2, 7, true);

	[Test]
	public async Task SweepWorkgroupBytes_MatchesTheWgslDeclarations()
	{
		// The planner checks SweepWorkgroupBytes against the device; it must be what the WGSL
		// declares. Every var<workgroup> array's length, with the header's constants put in.
		foreach (PatchMatchGpuShaderShape shape in new[] { Photometric, Geometric })
		{
			string source = PatchMatchGpuKernels.Descriptor(PatchMatchGpuKernel.SweepBand, shape, 1).Source;
			long bytes = 0;
			var declaration = new Regex(@"var<workgroup> \w+: array<(f32|i32|u32), (?<length>[^>]+)>;");
			foreach (Match match in declaration.Matches(source))
			{
				long length = 1;
				foreach (string factor in match.Groups["length"].Value.Split('*'))
				{
					length *= factor.Trim() switch
					{
						"PM_NUM_SRC_IMAGES" => shape.NumSrcImages,
						"PM_NUM_SAMPLES" => shape.NumSamples,
						"PM_NUM_COSTS" => PatchMatchGpuKernels.NumCosts,
						string other => throw new InvalidOperationException($"Unknown workgroup array length term {other}."),
					};
				}

				bytes += 4 * length;
			}

			await Assert.That(bytes).IsEqualTo(PatchMatchGpuKernels.SweepWorkgroupBytes(shape.NumSrcImages, shape.NumSamples));
			await Assert.That(source).Contains($"const PM_NUM_COSTS: i32 = {PatchMatchGpuKernels.NumCosts}i;");
		}
	}

	[Test]
	public async Task EveryKernel_ComposesDeterministicallyAndMatchesItsDescriptor()
	{
		var problems = new List<string>();
		foreach (PatchMatchGpuShaderShape shape in new[] { Photometric, Geometric })
		{
			foreach (PatchMatchGpuKernel kernel in PatchMatchGpuKernels.All)
			{
				ComputeKernelDescriptor first = PatchMatchGpuKernels.Descriptor(kernel, shape, 1234);
				ComputeKernelDescriptor second = PatchMatchGpuKernels.Descriptor(kernel, shape, 1234);
				if (first.Source != second.Source)
				{
					problems.Add($"{kernel}: composition is not deterministic");
				}

				if (!first.Source.Contains("\nfn " + first.EntryPoint + "(", StringComparison.Ordinal))
				{
					problems.Add($"{kernel}: no entry point {first.EntryPoint}");
				}

				if (!first.Source.Contains("const PM_GROUPS_X: u32 = 1234u;", StringComparison.Ordinal))
				{
					problems.Add($"{kernel}: the folded dispatch width is missing");
				}

				(_, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(first.Source);
				problems.AddRange(unreadable.Select(u => $"{kernel}: {u}"));
				problems.AddRange(WgslBindingChecker.Compare(first).Select(p => $"{kernel}: {p}"));
			}
		}

		await Assert.That(problems).IsEmpty();
	}

	[Test]
	public async Task EntryPoints_AreThePlannersNames()
	{
		string[] names = PatchMatchGpuKernels.All.Select(PatchMatchGpuKernels.EntryPoint).ToArray();

		// filter_pixels, not filter: "filter" is a reserved word in WGSL.
		await Assert.That(names).IsEquivalentTo(new[]
		{
			"init_random", "initial_cost", "backward_messages", "sweep_band", "filter_pixels", "rotate_planes", "rotate_normals",
		});
	}

	[Test]
	public async Task EveryKernel_StaysWithinTheDefaultBindingLimits()
	{
		var problems = new List<string>();
		foreach (PatchMatchGpuKernel kernel in PatchMatchGpuKernels.All)
		{
			IReadOnlyList<ComputeKernelBinding> bindings = PatchMatchGpuKernels.Bindings(kernel);
			int storage = bindings.Count(b => b.Type != ComputeBindingType.Uniform);
			int uniform = bindings.Count(b => b.Type == ComputeBindingType.Uniform);
			int maxStorage = kernel == PatchMatchGpuKernel.SweepBand ? 7 : 8;
			if (storage > maxStorage)
			{
				problems.Add($"{kernel}: {storage} storage bindings");
			}

			if (uniform > 12)
			{
				problems.Add($"{kernel}: {uniform} uniform bindings");
			}

			int[] groups = bindings.Select(b => b.Group).Distinct().Order().ToArray();
			if (groups.Length > 3 || !groups.SequenceEqual(Enumerable.Range(0, groups.Length)))
			{
				problems.Add($"{kernel}: groups {string.Join(",", groups)}");
			}
		}

		await Assert.That(problems).IsEmpty();
		await Assert.That(PatchMatchGpuKernels.Bindings(PatchMatchGpuKernel.SweepBand).Count(b => b.Type != ComputeBindingType.Uniform)).IsEqualTo(7);
	}

	[Test]
	public async Task BindingNumbers_NameTheSameBufferInEveryKernel()
	{
		// The cost buffer is the u32 mask in filter_pixels.
		var expected = new Dictionary<(int, int), string[]>
		{
			[(0, PatchMatchGpuKernels.ByteToUnitBinding)] = ["pm_byte_to_unit"],
			[(0, PatchMatchGpuKernels.PosesBinding)] = ["pm_poses"],
			[(0, PatchMatchGpuKernels.ProblemBinding)] = ["pm_problem"],
			[(0, PatchMatchGpuKernels.ReferenceBinding)] = ["pm_reference"],
			[(0, PatchMatchGpuKernels.SourceImagesBinding)] = ["pm_source_images"],
			[(0, PatchMatchGpuKernels.SourceDepthsBinding)] = ["pm_source_depths"],
			[(0, PatchMatchGpuKernels.StateBinding)] = ["pm_state"],
			[(0, PatchMatchGpuKernels.CostsBinding)] = ["pm_costs", "pm_mask"],
			[(0, PatchMatchGpuKernels.SelProbsBinding)] = ["pm_sel_probs"],
			[(0, PatchMatchGpuKernels.PrevSelProbsBinding)] = ["pm_prev_sel_probs"],
			[(PatchMatchGpuKernels.SweepGroup, 0)] = ["pm_sweep"],
			[(PatchMatchGpuKernels.BandGroup, 0)] = ["pm_band"],
		};
		var rotateExpected = new Dictionary<(int, int), string>
		{
			[(0, PatchMatchGpuKernels.RotateSourceBinding)] = "pm_rotate_src",
			[(0, PatchMatchGpuKernels.RotateDestinationBinding)] = "pm_rotate_dst",
			[(PatchMatchGpuKernels.SweepGroup, 0)] = "pm_rotate",
		};

		var problems = new List<string>();
		foreach (PatchMatchGpuKernel kernel in PatchMatchGpuKernels.All)
		{
			string source = PatchMatchGpuKernels.Descriptor(kernel, Geometric, 1).Source;
			bool rotate = kernel is PatchMatchGpuKernel.RotatePlanes or PatchMatchGpuKernel.RotateNormals;
			foreach (WgslBindingDeclaration declaration in WgslBindingChecker.Parse(source).Declarations)
			{
				(int, int) key = (declaration.Group, declaration.Binding);
				bool known = rotate
					? rotateExpected.TryGetValue(key, out string? name) && name == declaration.Name
					: expected.TryGetValue(key, out string[]? names) && names.Contains(declaration.Name);
				if (!known)
				{
					problems.Add($"{kernel}: @group({declaration.Group}) @binding({declaration.Binding}) {declaration.Name}");
				}
			}
		}

		await Assert.That(problems).IsEmpty();
	}

	[Test]
	public async Task Constants_AreExactlyTheRequiredOnesAndEveryOneIsUsed()
	{
		var problems = new List<string>();
		foreach (PatchMatchGpuKernel kernel in PatchMatchGpuKernels.All)
		{
			WgslConstants constants = PatchMatchGpuKernels.Constants(kernel, Geometric, 9);
			string header = constants.ToWgsl();
			// The parts' code without comments (which mention names like PM_POSE_*).
			string body = Regex.Replace(string.Concat(PatchMatchGpuKernels.Parts(kernel).Select(PatchMatchShaders.Load)), "//[^\n]*", string.Empty);
			IReadOnlyList<string> required = PatchMatchGpuKernels.RequiredConstants(kernel);
			if (constants.Count != required.Count)
			{
				problems.Add($"{kernel}: {constants.Count} constants for {required.Count} required");
			}

			foreach (string name in required)
			{
				if (!header.Contains("const " + name + ":", StringComparison.Ordinal))
				{
					problems.Add($"{kernel}: {name} missing from the header");
				}

				if (!Regex.IsMatch(body, @"\b" + name + @"\b"))
				{
					problems.Add($"{kernel}: {name} is never used");
				}
			}

			// Every PM_ constant the parts use but do not declare comes from the header.
			foreach (Match use in Regex.Matches(body, @"\bPM_[A-Z0-9_]+\b"))
			{
				string name = use.Value;
				bool declared = Regex.IsMatch(body, @"\bconst " + name + @"\s*:") || required.Contains(name);
				if (!declared)
				{
					problems.Add($"{kernel}: {name} has no value");
				}
			}
		}

		await Assert.That(problems.Distinct().ToList()).IsEmpty();
		await Assert.That(() => PatchMatchGpuKernels.Constants(PatchMatchGpuKernel.SweepBand, Geometric, 0)).Throws<ArgumentOutOfRangeException>();
	}

	[Test]
	public async Task UniformStructs_HaveTheDocumentedFieldsAndSizes()
	{
		string layout = PatchMatchShaders.Load(PatchMatchShaders.Layout);
		string rotate = PatchMatchShaders.Load("patch_match_rotate.wgsl");

		await Assert.That(StructFields(layout, "PmProblem")).IsEquivalentTo(new[]
		{
			"lk", "seed", "depth_min", "depth_max", "geom_consistency_regularizer", "geom_consistency_max_cost",
			"filter_min_ncc_prob", "filter_cos_min_triangulation_angle", "filter_geom_consistency_max_cost",
			"filter_min_num_consistent", "spatial_normalization", "color_normalization",
		}, CollectionOrdering.Matching);
		await Assert.That(UniformSize(layout, "PmProblem")).IsEqualTo(PatchMatchGpuKernels.ProblemUniformSize);
		await Assert.That(StructFields(layout, "PmSweep")).IsEquivalentTo(new[]
		{
			"rotation", "phase", "perturbation", "normal_perturbation", "prev_sel_prob_weight",
			"filter_photo_consistency", "filter_geom_consistency", "pad0",
		}, CollectionOrdering.Matching);
		await Assert.That(UniformSize(layout, "PmSweep")).IsEqualTo(PatchMatchGpuKernels.SweepUniformSize);
		await Assert.That(UniformSize(layout, "PmBand")).IsEqualTo(PatchMatchGpuKernels.BandUniformSize);
		await Assert.That(StructFields(rotate, "PmRotate").Take(5)).IsEquivalentTo(new[]
		{
			"src_width", "src_height", "num_planes", "src_offset", "dst_offset",
		}, CollectionOrdering.Matching);
		await Assert.That(UniformSize(rotate, "PmRotate")).IsEqualTo(PatchMatchGpuKernels.RotateUniformSize);
	}

	[Test]
	public async Task ToOriginalPixelTransliteration_MatchesPatchMatchTransforms()
	{
		int mismatches = 0;
		foreach ((int refWidth, int refHeight) in new[] { (7, 4), (4, 7), (5, 5), (1, 3) })
		{
			for (int rotation = 0; rotation < 4; ++rotation)
			{
				int width = rotation % 2 == 0 ? refWidth : refHeight;
				int height = rotation % 2 == 0 ? refHeight : refWidth;
				for (int row = 0; row < height; ++row)
				{
					for (int col = 0; col < width; ++col)
					{
						if (WgslTransliteration.ToOriginalPixel(rotation, row, col, refWidth, refHeight)
							!= PatchMatchTransforms.ToOriginalPixel(rotation, row, col, refWidth, refHeight))
						{
							mismatches++;
						}
					}
				}
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	public async Task RotatePlanesTransliteration_MatchesMatRotate()
	{
		// Three 5 x 3 planes, placed at word offsets in larger buffers.
		const int Width = 5;
		const int Height = 3;
		const int Planes = 3;
		var mat = new Mat<float>(Width, Height, Planes);
		for (int i = 0; i < mat.Data.Length; ++i)
		{
			mat.Data[i] = i * 1.5f - 7;
		}

		var expected = new Mat<float>(Height, Width, Planes);
		mat.Rotate(expected, 1);

		uint[] src = new uint[11 + mat.Data.Length];
		for (int i = 0; i < mat.Data.Length; ++i)
		{
			src[11 + i] = BitConverter.SingleToUInt32Bits(mat.Data[i]);
		}

		uint[] dst = new uint[4 + mat.Data.Length];
		var rotate = new WgslTransliteration.Rotate(Width, Height, Planes, 11, 4);
		RunFolded((int)rotate.NumPlanes * Width * Height, index => WgslTransliteration.RotatePlanes(rotate, index, src, dst));

		float[] actual = dst.Skip(4).Select(BitConverter.UInt32BitsToSingle).ToArray();
		await Assert.That(actual).IsEquivalentTo(expected.Data, CollectionOrdering.Matching);
	}

	[Test]
	public async Task RotateNormalsTransliteration_MatchesRotateNormalMapThenRotate()
	{
		const int Width = 4;
		const int Height = 6;
		var normals = new Mat<float>(Width, Height, 3);
		for (int i = 0; i < normals.Data.Length; ++i)
		{
			normals.Data[i] = (i % 7) - 3.25f;
		}

		// Signed zeros and a NaN: the sign flip must match C#'s negation bit for bit.
		normals.Data[0] = 0.0f;
		normals.Data[1] = -0.0f;
		normals.Data[2] = float.NaN;

		uint[] src = new uint[2 + normals.Data.Length];
		for (int i = 0; i < normals.Data.Length; ++i)
		{
			src[2 + i] = BitConverter.SingleToUInt32Bits(normals.Data[i]);
		}

		PatchMatchKernel.RotateNormalMap(normals, 1);
		var expected = new Mat<float>(Height, Width, 3);
		normals.Rotate(expected, 1);

		uint[] dst = new uint[normals.Data.Length];
		var rotate = new WgslTransliteration.Rotate(Width, Height, 3, 2, 0);
		RunFolded(Width * Height, index => WgslTransliteration.RotateNormals(rotate, index, src, dst));

		uint[] expectedBits = expected.Data.Select(BitConverter.SingleToUInt32Bits).ToArray();
		await Assert.That(dst).IsEquivalentTo(expectedBits, CollectionOrdering.Matching);
	}

	[Test]
	public async Task FoldedIndexTransliteration_CoversEveryElementOnce()
	{
		// 1000 workgroups folded into rows of 7: every element index below 64000 exactly once.
		const uint GroupsX = 7;
		const uint Groups = 1000;
		uint groupsY = (Groups + GroupsX - 1) / GroupsX;
		var seen = new int[groupsY * GroupsX * 64];
		for (uint y = 0; y < groupsY; ++y)
		{
			for (uint x = 0; x < GroupsX; ++x)
			{
				for (uint local = 0; local < 64; ++local)
				{
					seen[WgslTransliteration.LinearIndex(x, y, local, GroupsX)]++;
				}
			}
		}

		await Assert.That(seen.All(count => count == 1)).IsTrue();
	}

	// Runs `element` for every index a folded dispatch of `count` elements reaches, with the
	// kernels' early return for indices past the count.
	private static void RunFolded(int count, Action<uint> element)
	{
		const uint GroupsX = 2;
		uint groups = (uint)((count + 63) / 64);
		uint groupsY = (groups + GroupsX - 1) / GroupsX;
		for (uint y = 0; y < groupsY; ++y)
		{
			for (uint x = 0; x < GroupsX; ++x)
			{
				for (uint local = 0; local < 64; ++local)
				{
					element(WgslTransliteration.LinearIndex(x, y, local, GroupsX));
				}
			}
		}
	}

	private static string[] StructFields(string wgsl, string name)
	{
		Match match = Regex.Match(wgsl, @"\nstruct " + name + @" \{\n(?<body>[^}]*)\}");
		return Regex.Matches(match.Groups["body"].Value, @"^\s*(\w+):", RegexOptions.Multiline).Select(m => m.Groups[1].Value).ToArray();
	}

	// WGSL's uniform layout for the member types these structs use.
	private static int UniformSize(string wgsl, string name)
	{
		Match match = Regex.Match(wgsl, @"\nstruct " + name + @" \{\n(?<body>[^}]*)\}");
		int offset = 0;
		int structAlign = 4;
		foreach (Match field in Regex.Matches(match.Groups["body"].Value, @":\s*([\w<>]+),"))
		{
			(int size, int align) = field.Groups[1].Value switch
			{
				"f32" or "i32" or "u32" => (4, 4),
				"vec2<u32>" => (8, 8),

				// A struct member of a uniform struct is aligned to 16.
				"PmLikelihood" => (16, 16),
				string other => throw new InvalidOperationException($"No layout for {other}."),
			};
			offset = (offset + align - 1) / align * align + size;
			structAlign = Math.Max(structAlign, align);
		}

		return (offset + structAlign - 1) / structAlign * structAlign;
	}

	/// <summary>
	/// Line-by-line C# copies of WGSL index math in patch_match_layout.wgsl,
	/// patch_match_dispatch.wgsl and patch_match_rotate.wgsl. NOT PRODUCTION CODE; see the file header.
	/// </summary>
	private static class WgslTransliteration
	{
		public static uint LinearIndex(uint workgroupX, uint workgroupY, uint localIndex, uint groupsX)
			=> (workgroupY * groupsX + workgroupX) * 64u + localIndex;

		public static (int Row, int Col) ToOriginalPixel(int rotation, int row0, int col0, int refWidth, int refHeight)
		{
			int row = row0;
			int col = col0;
			for (int k = rotation; k > 0; k = k - 1)
			{
				int prevWidth = ((k - 1) & 1) == 0 ? refWidth : refHeight;
				int newRow = col;
				col = prevWidth - 1 - row;
				row = newRow;
			}

			return (row, col);
		}

		public readonly record struct Rotate(uint SrcWidth, uint SrcHeight, uint NumPlanes, uint SrcOffset, uint DstOffset);

		public static void RotatePlanes(in Rotate rotate, uint index, uint[] src, uint[] dst)
		{
			uint planeSize = rotate.SrcWidth * rotate.SrcHeight;
			if (index >= rotate.NumPlanes * planeSize)
			{
				return;
			}

			uint plane = index / planeSize;
			uint offset = index % planeSize;
			uint row = offset / rotate.SrcWidth;
			uint col = offset % rotate.SrcWidth;
			uint planeStart = plane * planeSize;
			dst[rotate.DstOffset + planeStart + RotatedOffset(rotate, row, col)] = src[rotate.SrcOffset + index];
		}

		public static void RotateNormals(in Rotate rotate, uint index, uint[] src, uint[] dst)
		{
			uint planeSize = rotate.SrcWidth * rotate.SrcHeight;
			if (index >= planeSize)
			{
				return;
			}

			uint row = index / rotate.SrcWidth;
			uint col = index % rotate.SrcWidth;
			uint s = rotate.SrcOffset + index;
			uint d = rotate.DstOffset + RotatedOffset(rotate, row, col);
			uint x = src[s];
			dst[d] = src[s + planeSize];
			dst[d + planeSize] = x ^ 0x80000000u;
			dst[d + 2u * planeSize] = src[s + 2u * planeSize];
		}

		private static uint RotatedOffset(in Rotate rotate, uint row, uint col) => (rotate.SrcWidth - 1u - col) * rotate.SrcHeight + row;
	}
}
