// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_common.wgsl: the self-contained helpers every GPU PatchMatch kernel shares -
// 64-bit integer arithmetic on vec2<u32>, the counter-based random numbers, an exactly
// rounded u32 -> f32 conversion, CUDA's NaN-ignoring float min/max, and bit-level float tests.
// It declares no bindings and uses no constants, so any kernel can include it.
// Mirrors: ColmapSharp/Mvs/PatchMatchRandom.cs (the RNG, bit for bit) and
//   PatchMatchKernel.CudaMin/CudaMax in ColmapSharp/Mvs/PatchMatchKernel.Geometry.cs.
// Ports: nothing from COLMAP directly; PatchMatchRandom replaces patch_match_cuda.cu's cuRAND
//   state (docs/CPP_DIVERGENCES.md, entry 86), and cuda_min/cuda_max keep CUDA's fminf/fmaxf
//   (entry 96).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs. The C#-only
// ColmapSharp.Tests/Mvs/PatchMatchShaderRngTransliterationTests.cs runs a line-by-line C# copy
// of the integer functions here against PatchMatchRandom; keep the two in step.
//
// Why integer emulation: WGSL has no u64, and its u32 -> f32 conversion may round either way
// when the value is not representable, so both are done with u32 operations whose results
// WGSL defines exactly. Why bit tests: WGSL lets an implementation assume floats are never
// NaN or infinite, so `x != x` and comparisons against infinity may be optimized away.

// ---- 64-bit unsigned integers as vec2<u32>(low word, high word) ----

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
	let lo = a.x + b.x;
	let carry = select(0u, 1u, lo < a.x);
	return vec2<u32>(lo, a.y + b.y + carry);
}

// The full 64-bit product of two u32s, from 16-bit limbs so no partial product overflows.
fn u32_mul_wide(a: u32, b: u32) -> vec2<u32> {
	let a0 = a & 0xFFFFu;
	let a1 = a >> 16u;
	let b0 = b & 0xFFFFu;
	let b1 = b >> 16u;
	let p00 = a0 * b0;
	let p01 = a0 * b1;
	let p10 = a1 * b0;
	let p11 = a1 * b1;
	// Bits 16..47 of the product: at most 3 * 0xFFFF, so no overflow.
	let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
	let lo = (p00 & 0xFFFFu) | (mid << 16u);
	let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
	return vec2<u32>(lo, hi);
}

// The low 64 bits of a 64 x 64 product (C#'s unchecked ulong multiply). The cross terms only
// reach the high word, where u32 multiplication wraps exactly as the dropped bits require.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
	let w = u32_mul_wide(a.x, b.x);
	return vec2<u32>(w.x, w.y + a.x * b.y + a.y * b.x);
}

// a >> n for 0 < n < 32 (WGSL rejects or masks shift counts of 32 and more).
fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
	return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}

// ---- The counter-based PatchMatch random numbers (PatchMatchRandom.cs) ----

// 0x9E3779B97F4A7C15, SplitMix64's Weyl increment.
const PM_RANDOM_GOLDEN: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);

// SplitMix64's output mixer (PatchMatchRandom.Mix).
fn pm_random_mix(z0: vec2<u32>) -> vec2<u32> {
	var z = z0;
	// 0xBF58476D1CE4E5B9
	z = u64_mul(z ^ u64_shr(z, 30u), vec2<u32>(0x1CE4E5B9u, 0xBF58476Du));
	// 0x94D049BB133111EB
	z = u64_mul(z ^ u64_shr(z, 27u), vec2<u32>(0x133111EBu, 0x94D049BBu));
	return z ^ u64_shr(z, 31u);
}

// One pixel's stream in one phase: the key and the number of draws taken so far.
struct PatchMatchRandom {
	key: vec2<u32>,
	counter: vec2<u32>,
}

// The stream of pixel (row, col) of the original, unrotated reference image in `phase`,
// under `seed` (vec2<u32>(low, high) of the C# ulong seed). The PatchMatchRandom constructor.
fn pm_random_init(seed: vec2<u32>, row: i32, col: i32, phase: i32) -> PatchMatchRandom {
	// ((ulong)(uint)row << 32) | (uint)col
	let pixel = vec2<u32>(bitcast<u32>(col), bitcast<u32>(row));
	// (ulong)(uint)phase
	let phase64 = vec2<u32>(bitcast<u32>(phase), 0u);
	let key = pm_random_mix(pm_random_mix(pm_random_mix(u64_add(seed, PM_RANDOM_GOLDEN)) ^ pixel) ^ phase64);
	var state: PatchMatchRandom;
	state.key = key;
	state.counter = vec2<u32>(0u, 0u);
	return state;
}

// The high 32 bits of the next draw (PatchMatchRandom.NextUniform before the float mapping).
fn pm_random_next_bits(state: ptr<function, PatchMatchRandom>) -> u32 {
	(*state).counter = u64_add((*state).counter, vec2<u32>(1u, 0u));
	return pm_random_mix(u64_add((*state).key, u64_mul((*state).counter, PM_RANDOM_GOLDEN))).y;
}

// The next uniform float in (0, 1], like curand_uniform: bits * 2^-32 + 2^-33. The conversion
// is emulated; the multiply is by a power of two and so exact, and WGSL rounds f32 `*` and
// `+` correctly, so the result matches C#'s float arithmetic whether or not the compiler
// fuses the two (the fused and unfused results agree when the product is exact).
fn pm_random_next_uniform(state: ptr<function, PatchMatchRandom>) -> f32 {
	let bits = pm_random_next_bits(state);
	return u32_to_f32_rne(bits) * bitcast<f32>(0x2F800000u) + bitcast<f32>(0x2F000000u);
}

// ---- Exact conversions and float bit tests ----

// v converted to the nearest f32, ties to even: what C#'s (float)uint does.
fn u32_to_f32_rne(v: u32) -> f32 {
	if (v == 0u) {
		return 0.0;
	}

	let msb = 31u - countLeadingZeros(v);
	let exponent = (msb + 127u) << 23u;
	if (msb <= 23u) {
		// At most 24 significant bits: exact.
		return bitcast<f32>(exponent | ((v << (23u - msb)) & 0x7FFFFFu));
	}

	let shift = msb - 23u;
	var significand = v >> shift;
	let remainder = v & ((1u << shift) - 1u);
	let half = 1u << (shift - 1u);
	if (remainder > half || (remainder == half && (significand & 1u) == 1u)) {
		significand = significand + 1u;
	}

	// Adding the significand without its implicit bit lets a round-up to 2^24 carry into the
	// exponent, which is the correctly rounded power of two.
	return bitcast<f32>(exponent + (significand - 0x800000u));
}

fn is_nan_f32(x: f32) -> bool {
	return (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u;
}

// Neither NaN nor infinite.
fn is_finite_f32(x: f32) -> bool {
	return (bitcast<u32>(x) & 0x7FFFFFFFu) < 0x7F800000u;
}

// CUDA's min(float, float), fminf: the smaller operand, or the other one when one is NaN.
// Equal operands return the OR of their bits, so min(+0, -0) is -0 as C#'s MathF.Min gives.
fn cuda_min(x: f32, y: f32) -> f32 {
	if (is_nan_f32(x)) {
		return y;
	}

	if (is_nan_f32(y)) {
		return x;
	}

	if (x < y) {
		return x;
	}

	if (y < x) {
		return y;
	}

	return bitcast<f32>(bitcast<u32>(x) | bitcast<u32>(y));
}

// CUDA's max(float, float), fmaxf: the larger operand, ignoring a NaN one. Equal operands
// return the AND of their bits, so max(+0, -0) is +0 as C#'s MathF.Max gives.
fn cuda_max(x: f32, y: f32) -> f32 {
	if (is_nan_f32(x)) {
		return y;
	}

	if (is_nan_f32(y)) {
		return x;
	}

	if (x > y) {
		return x;
	}

	if (y > x) {
		return y;
	}

	return bitcast<f32>(bitcast<u32>(x) & bitcast<u32>(y));
}
