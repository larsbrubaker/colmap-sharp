// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchShaderGeometryTests: C#-only tests (COLMAP's PatchMatch kernels are CUDA) of
// ColmapSharp/Mvs/Shaders/patch_match_geometry.wgsl and patch_match_likelihood.wgsl, and of the
// both-zeros rule of cuda_min/cuda_max in patch_match_common.wgsl. Without a GPU the suite can
// check that the parts load, compose with common and textures, declare no bindings of their own,
// define the helpers the kernels call, and keep the pose table offsets of
// ColmapSharp/Mvs/PatchMatchTransforms.cs. Two TRANSLITERATION CHECKS, NOT PRODUCTION CODE,
// follow the pattern of PatchMatchShaderRngTransliterationTests: line-by-line C# copies of
// pm_div and of pm_perturb_normal's trial loop (which replaces C#'s recursion), run against
// IEEE division and the production PatchMatchKernel.PerturbNormal. They show the rewrites are
// right; whether a GPU runs the text the same way is the conformance kit's job (PORTING_PLAN.md
// Phase 13). Keep the copies in step with the WGSL. (naga 30 validated the composed sample
// kernel below when it was written.)

using System.Text.RegularExpressions;

using ColmapSharp.Compute;
using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchShaderGeometryTests
{
	// A kernel calling every helper of geometry and likelihood, with the bindings textures and
	// geometry require.
	private const string SampleKernel = """
		struct Problem { lk: PmLikelihood, perturbation: f32, prev_weight: f32, }
		@group(0) @binding(0) var<uniform> pm_byte_to_unit: PmByteToUnit;
		@group(0) @binding(1) var<storage, read> pm_source_images: array<u32>;
		@group(0) @binding(2) var<storage, read> pm_source_depths: array<f32>;
		@group(0) @binding(3) var<uniform> pm_poses: PmPoseTable;
		@group(0) @binding(4) var<uniform> problem: Problem;
		@group(0) @binding(5) var<storage, read_write> out_values: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) id: vec3<u32>) {
			var rng = pm_random_init(vec2<u32>(1u, 2u), i32(id.x), 3, 0);
			let frame = pm_frame(1);
			let n = pm_generate_random_normal(frame, 2, i32(id.x), &rng);
			let pn = pm_perturb_normal(frame, 2, i32(id.x), problem.perturbation, n, &rng);
			let d = pm_perturb_depth(problem.perturbation, pm_generate_random_depth(1.0, 2.0, &rng), &rng);
			let p = pm_compute_point_at_depth(frame, 2.0, f32(id.x), d);
			let pd = pm_propagate_depth(frame, d, pn, 1.0, 2.0);
			let ang = pm_compute_viewing_angles(1, p, pn, 2);
			let h = pm_compose_homography(1, frame, 2, 2, i32(id.x), d, pn);
			let a = pm_compute_forward_message(problem.lk, 0.3, 0.5);
			let b = pm_compute_backward_message(problem.lk, 0.3, 0.5);
			let sp = pm_compute_sel_prob(a, b, 0.4, problem.prev_weight);
			var probs = array<f32, PM_NUM_SRC_IMAGES>(sp, pm_compute_tri_prob(problem.lk, ang.x), pm_compute_inc_prob(problem.lk, ang.y) * pm_compute_resolution_prob(h, 2.0, f32(id.x), 3));
			pm_transform_pdf_to_cdf(&probs);
			let costs = array<f32, PM_NUM_COSTS>(probs[0], probs[1], probs[2], pd, pm_div(1.0, d));
			let best = pm_find_min_cost(costs);
			out_values[id.x] = f32(best) + cuda_min(pm_sample_source_depth(1.0, 1.0, 0), pm_sample_source_image(1.5, 2.5, 0)) + pm_compute_ncc_prob(problem.lk, 0.2) + pm_dot3(p, pm_mat33_dot_vec3(h, n)) + cuda_max(1.0, 2.0);
		}
		""";

	private static readonly ComputeKernelBinding[] SampleKernelBindings =
	{
		new(0, 0, ComputeBindingType.Uniform),
		new(0, 1, ComputeBindingType.ReadOnlyStorage),
		new(0, 2, ComputeBindingType.ReadOnlyStorage),
		new(0, 3, ComputeBindingType.Uniform),
		new(0, 4, ComputeBindingType.Uniform),
		new(0, 5, ComputeBindingType.Storage),
	};

	// The helpers the GPU kernels (PORTING_PLAN.md Phase 13) call, by part.
	private static readonly string[] GeometryFunctions =
	{
		"pm_div", "pm_pose_float", "pm_pose_row", "pm_frame", "pm_mat33_dot_vec3", "pm_mat33_dot_vec3_homogeneous",
		"pm_dot3", "pm_dot_view_ray", "pm_generate_random_depth", "pm_generate_random_normal", "pm_perturb_depth",
		"pm_perturb_normal", "pm_compute_point_at_depth", "pm_propagate_depth", "pm_compute_viewing_angles",
		"pm_compose_homography",
	};

	private static readonly string[] LikelihoodFunctions =
	{
		"pm_compute_message", "pm_compute_forward_message", "pm_compute_backward_message", "pm_compute_sel_prob",
		"pm_compute_ncc_prob", "pm_compute_tri_prob", "pm_compute_inc_prob", "pm_compute_resolution_prob",
		"pm_find_min_cost", "pm_transform_pdf_to_cdf",
	};

	[Test]
	public async Task Load_GeometryAndLikelihoodHaveTheirHeaders()
	{
		foreach (string part in new[] { PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood })
		{
			string text = PatchMatchShaders.Load(part);
			await Assert.That(text.StartsWith("// Copyright (c) 2026, Lars Brubaker.", StringComparison.Ordinal)).IsTrue();
			await Assert.That(text.Contains("// Mirrors: ColmapSharp/Mvs/", StringComparison.Ordinal)).IsTrue();
		}
	}

	[Test]
	public async Task GeometryAndLikelihood_DeclareNoBindings()
	{
		string source = PatchMatchShaders.Compose(
			new WgslConstants().Add("PM_NUM_SRC_IMAGES", 2), PatchMatchShaders.Common, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood);

		(IReadOnlyList<WgslBindingDeclaration> declared, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(source);

		await Assert.That(declared.Count + unreadable.Count).IsEqualTo(0);
	}

	[Test]
	public async Task SampleKernel_OnAllParts_MatchesItsDescriptor()
	{
		WgslConstants constants = new WgslConstants()
			.Add("PM_NUM_SRC_IMAGES", 3)
			.Add("PM_SRC_MAX_WIDTH", 640)
			.Add("PM_SRC_MAX_HEIGHT", 480);
		string source = PatchMatchShaders.Compose(
			constants, PatchMatchShaders.Common, PatchMatchShaders.Textures, PatchMatchShaders.Geometry, PatchMatchShaders.Likelihood)
			+ SampleKernel;
		var descriptor = new ComputeKernelDescriptor("sample", source, "main", SampleKernelBindings);

		(IReadOnlyList<WgslBindingDeclaration> declared, IReadOnlyList<string> unreadable) = WgslBindingChecker.Parse(source);

		await Assert.That(declared.Count).IsEqualTo(SampleKernelBindings.Length);
		await Assert.That(unreadable.Count).IsEqualTo(0);
		await Assert.That(WgslBindingChecker.Compare(descriptor).Count).IsEqualTo(0);
	}

	[Test]
	public async Task Parts_DefineEveryHelperTheKernelsNeed()
	{
		string geometry = PatchMatchShaders.Load(PatchMatchShaders.Geometry);
		string likelihood = PatchMatchShaders.Load(PatchMatchShaders.Likelihood);
		var missing = new List<string>();
		foreach (string name in GeometryFunctions)
		{
			if (!geometry.Contains("\nfn " + name + "(", StringComparison.Ordinal))
			{
				missing.Add(PatchMatchShaders.Geometry + ": " + name);
			}
		}

		foreach (string name in LikelihoodFunctions)
		{
			if (!likelihood.Contains("\nfn " + name + "(", StringComparison.Ordinal))
			{
				missing.Add(PatchMatchShaders.Likelihood + ": " + name);
			}
		}

		await Assert.That(missing).IsEmpty();
		await Assert.That(geometry.Contains("\nstruct PmPoseTable {", StringComparison.Ordinal)).IsTrue();
		await Assert.That(geometry.Contains("\nstruct PmFrame {", StringComparison.Ordinal)).IsTrue();
		await Assert.That(likelihood.Contains("\nstruct PmLikelihood {", StringComparison.Ordinal)).IsTrue();
	}

	[Test]
	public async Task Geometry_PoseTableOffsetsMatchPatchMatchTransforms()
	{
		string geometry = PatchMatchShaders.Load(PatchMatchShaders.Geometry);

		await Assert.That(ConstI32(geometry, "PM_POSE_K_OFFSET")).IsEqualTo(PatchMatchTransforms.KOffset);
		await Assert.That(ConstI32(geometry, "PM_POSE_R_OFFSET")).IsEqualTo(PatchMatchTransforms.ROffset);
		await Assert.That(ConstI32(geometry, "PM_POSE_T_OFFSET")).IsEqualTo(PatchMatchTransforms.TOffset);
		await Assert.That(ConstI32(geometry, "PM_POSE_C_OFFSET")).IsEqualTo(PatchMatchTransforms.COffset);
		await Assert.That(ConstI32(geometry, "PM_POSE_P_OFFSET")).IsEqualTo(PatchMatchTransforms.POffset);
		await Assert.That(ConstI32(geometry, "PM_POSE_INV_P_OFFSET")).IsEqualTo(PatchMatchTransforms.InvPOffset);
		await Assert.That(ConstI32(geometry, "PM_NUM_TFORM_PARAMS")).IsEqualTo(PatchMatchTransforms.NumTformParams);

		// Four frames of eight floats come first; the four rotations' rows then fill whole
		// vec4s (4 * 43 * S floats), which PM_POSE_TABLE_VEC4S counts.
		await Assert.That(ConstI32(geometry, "PM_POSE_ROWS_OFFSET")).IsEqualTo(4 * 8);
		await Assert.That(geometry.Contains(
			"const PM_POSE_TABLE_VEC4S: i32 = 8i + PM_NUM_TFORM_PARAMS * PM_NUM_SRC_IMAGES;", StringComparison.Ordinal)).IsTrue();
	}

	[Test]
	public async Task Common_CudaMinMaxCombineBitsOnlyForTwoZeros()
	{
		// A GPU that flushes subnormals compares two different subnormals equal; combining their
		// bits would give neither operand, so the bit fallback is only for +0 and -0.
		string common = PatchMatchShaders.Load(PatchMatchShaders.Common);

		await Assert.That(common.Contains(
			"\tif (both_zero_f32(x, y)) {\n\t\treturn bitcast<f32>(bitcast<u32>(x) | bitcast<u32>(y));\n\t}\n\n\treturn x;\n}", StringComparison.Ordinal)).IsTrue();
		await Assert.That(common.Contains(
			"\tif (both_zero_f32(x, y)) {\n\t\treturn bitcast<f32>(bitcast<u32>(x) & bitcast<u32>(y));\n\t}\n\n\treturn x;\n}", StringComparison.Ordinal)).IsTrue();
		await Assert.That(Regex.Matches(common, @"bitcast<u32>\(x\) [|&] bitcast<u32>\(y\)").Count).IsEqualTo(3);
	}

	[Test]
	public async Task PmDivTransliteration_MatchesIeeeDivision()
	{
		float[] values =
		{
			0.0f, -0.0f, 1.0f, -1.0f, 2.5f, -3.75f, float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue,
			float.PositiveInfinity, float.NegativeInfinity, float.NaN,
		};
		int mismatches = 0;
		foreach (float a in values)
		{
			foreach (float b in values)
			{
				float expected = a / b;
				float actual = WgslTransliteration.PmDiv(a, b);
				bool same = float.IsNaN(expected)
					? float.IsNaN(actual)
					: BitConverter.SingleToUInt32Bits(actual) == BitConverter.SingleToUInt32Bits(expected);
				if (!same)
				{
					mismatches++;
				}
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	public async Task PerturbNormalTransliteration_MatchesRecursivePerturbNormal()
	{
		var images = new List<Image>
		{
			new("ref", 32, 24, [40, 0, 15.5f, 0, 42, 11.5f, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
			new("src", 32, 24, [40, 0, 15.5f, 0, 42, 11.5f, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [-0.5f, 0, 0]),
		};
		var frame = new PatchMatchFrame(new PatchMatchTransforms(images, 0, [1]), 0);

		// Normals from facing the camera to nearly grazing, so all four trials and the
		// give-up path occur; perturbations up to pi.
		int mismatches = 0;
		int gaveUp = 0;
		float[] expected = new float[3];
		for (int i = 0; i < 2000; ++i)
		{
			int row = i % 24;
			int col = i * 7 % 32;
			float tilt = i % 50 / 50.0f * 1.6f;
			float[] normal = [MathF.Sin(tilt), 0.1f * MathF.Sin(tilt), -MathF.Cos(tilt)];
			float perturbation = (float)((1 + i % 4) / 4.0 * Math.PI);
			var cpuRandom = new PatchMatchRandom(7, row, col, 3);
			var gpuRandom = new PatchMatchRandom(7, row, col, 3);

			PatchMatchKernel.PerturbNormal(frame, row, col, perturbation, normal, ref cpuRandom, expected);
			float[] actual = WgslTransliteration.PerturbNormal(frame, row, col, perturbation, normal, ref gpuRandom);

			bool same = cpuRandom.NextUniform() == gpuRandom.NextUniform();
			for (int k = 0; k < 3; ++k)
			{
				same &= BitConverter.SingleToUInt32Bits(actual[k]) == BitConverter.SingleToUInt32Bits(expected[k]);
			}

			if (!same)
			{
				mismatches++;
			}

			if (expected.AsSpan().SequenceEqual(normal))
			{
				gaveUp++;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
		await Assert.That(gaveUp).IsGreaterThan(0);
	}

	private static int ConstI32(string wgsl, string name)
	{
		Match match = Regex.Match(wgsl, @"\nconst " + name + @": i32 = (\d+)i;");
		if (!match.Success)
		{
			throw new InvalidOperationException($"{name} is not declared as a literal i32 constant.");
		}

		return int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
	}

	/// <summary>
	/// Line-by-line C# copies of WGSL functions in patch_match_geometry.wgsl. NOT PRODUCTION
	/// CODE; see the file header.
	/// </summary>
	private static class WgslTransliteration
	{
		private const uint PmNanBits = 0x7FC00000u;

		public static float PmDiv(float a, float b)
		{
			uint bBits = BitConverter.SingleToUInt32Bits(b);
			if ((bBits & 0x7FFFFFFFu) != 0u)
			{
				return a / b;
			}

			uint aBits = BitConverter.SingleToUInt32Bits(a);
			if ((aBits & 0x7FFFFFFFu) > 0x7F800000u)
			{
				return a;
			}

			if ((aBits & 0x7FFFFFFFu) == 0u)
			{
				return BitConverter.UInt32BitsToSingle(PmNanBits);
			}

			return BitConverter.UInt32BitsToSingle(((aBits ^ bBits) & 0x80000000u) | 0x7F800000u);
		}

		public static float[] PerturbNormal(in PatchMatchFrame frame, int row, int col, float perturbation0, float[] normal, ref PatchMatchRandom random)
		{
			const int MaxNumTrials = 3;
			float perturbation = perturbation0;
			int numTrials = 0;
			float[] perturbed;
			while (true)
			{
				float a1 = (random.NextUniform() - 0.5f) * perturbation;
				float a2 = (random.NextUniform() - 0.5f) * perturbation;
				float a3 = (random.NextUniform() - 0.5f) * perturbation;

				float sinA1 = MathF.Sin(a1);
				float sinA2 = MathF.Sin(a2);
				float sinA3 = MathF.Sin(a3);
				float cosA1 = MathF.Cos(a1);
				float cosA2 = MathF.Cos(a2);
				float cosA3 = MathF.Cos(a3);

				float[] r = new float[9];
				r[0] = cosA2 * cosA3;
				r[1] = -cosA2 * sinA3;
				r[2] = sinA2;
				r[3] = cosA1 * sinA3 + cosA3 * sinA1 * sinA2;
				r[4] = cosA1 * cosA3 - sinA1 * sinA2 * sinA3;
				r[5] = -cosA2 * sinA1;
				r[6] = sinA1 * sinA3 - cosA1 * cosA3 * sinA2;
				r[7] = cosA3 * sinA1 + cosA1 * sinA2 * sinA3;
				r[8] = cosA1 * cosA2;

				float[] perturbedNormal =
				[
					r[0] * normal[0] + r[1] * normal[1] + r[2] * normal[2],
					r[3] * normal[0] + r[4] * normal[1] + r[5] * normal[2],
					r[6] * normal[0] + r[7] * normal[1] + r[8] * normal[2],
				];

				float rayX = frame.InvK0 * col + frame.InvK1;
				float rayY = frame.InvK2 * row + frame.InvK3;
				float viewDot = perturbedNormal[0] * rayX + perturbedNormal[1] * rayY + perturbedNormal[2] * 1.0f;
				if (!float.IsNaN(viewDot) && viewDot >= 0.0f)
				{
					if (numTrials < MaxNumTrials)
					{
						perturbation = 0.5f * perturbation;
						numTrials = numTrials + 1;
						continue;
					}

					perturbed = [normal[0], normal[1], normal[2]];
					break;
				}

				float invNorm = 1.0f / MathF.Sqrt(perturbedNormal[0] * perturbedNormal[0] + perturbedNormal[1] * perturbedNormal[1] + perturbedNormal[2] * perturbedNormal[2]);
				perturbed = [perturbedNormal[0] * invNorm, perturbedNormal[1] * invNorm, perturbedNormal[2] * invNorm];
				break;
			}

			return perturbed;
		}
	}
}
