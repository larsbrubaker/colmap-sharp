// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_ncc.wgsl: the photo-consistency cost of GPU PatchMatch - the bilaterally weighted
// NCC between a reference patch and its plane-induced warp into a source image, as 1 - NCC in
// [0, 2] - and the reference half of it (the window's colors and bilateral weights), which does
// not depend on the hypothesis or source image and so is prepared once per pixel.
// Mirrors: ColmapSharp/Mvs/PatchMatchKernel.Photometric.cs (PatchMatchPhotoConsistency's
//   PrepareWindow, the scalar Compute and FinishNcc) and BilateralWeightComputer in
//   ColmapSharp/Mvs/PatchMatchRefImage.cs, in the same operation order. The CPU's
//   ComputeFourLockstep is bit-identical to the scalar Compute, so the scalar path is the one
//   ported.
// Ports: patch_match_cuda.cu's PhotoConsistencyCostComputer, as the CPU port models it
//   (docs/CPP_DIVERGENCES.md, entries 95-97).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after common, textures, geometry, likelihood and
// layout. A kernel that includes it supplies PM_WINDOW_RADIUS: i32 and PM_WINDOW_STEP: i32 in its
// constants header, and PM_WINDOW_SHARED: bool (the cooperative sweep_band's shared window, below),
// and declares, at any group and binding it likes,
//   var<uniform> pm_problem: PmProblem;           (the bilateral normalizations)
//   var<storage, read> pm_reference: array<f32>;  (the reference buffer, patch_match_layout.wgsl)
//
// Window cache. Up to PM_WINDOW_CACHE_LIMIT samples, pm_prepare_window stores the window's colors
// and weights in private arrays, as the CPU's PrepareWindow does; above it (a large radius with
// step 1) private memory would be too large, so each evaluation recomputes color and weight from
// the reference plane. The values and the weight sum's addition order are the same either way
// (PatchMatchKernel.Photometric.cs explains why), so the two paths give the same floats only if
// the backend evaluates pm_bilateral_weight identically at both call sites - the same exp result
// and the same choice about contracting its multiply-adds, which a shader compiler may make
// differently per inlined site. Where it does not, the cached and uncached paths agree at Tier C.

// Samples per window axis and in the whole (strided) window (PatchMatchPhotoConsistency.WindowCount).
const PM_WINDOW_PER_AXIS: i32 = (2i * PM_WINDOW_RADIUS) / PM_WINDOW_STEP + 1i;
const PM_WINDOW_COUNT: i32 = PM_WINDOW_PER_AXIS * PM_WINDOW_PER_AXIS;

// The largest window whose colors and weights are kept in private arrays (11 x 11 is the default
// radius 5).
const PM_WINDOW_CACHE_LIMIT: i32 = 128i;
const PM_WINDOW_CACHED: bool = PM_WINDOW_COUNT <= PM_WINDOW_CACHE_LIMIT;
const PM_WINDOW_CACHE_SIZE: i32 = select(1i, PM_WINDOW_COUNT, PM_WINDOW_CACHED);

// Maximum photo consistency cost as 1 - min(NCC) (PatchMatchPhotoConsistency.MaxCost).
const PM_MAX_COST: f32 = 2.0f;

var<private> pm_window_colors: array<f32, PM_WINDOW_CACHE_SIZE>;
var<private> pm_window_weights: array<f32, PM_WINDOW_CACHE_SIZE>;

// Shared window (the cooperative sweep_band, PM_WINDOW_SHARED): the cached window lives in workgroup
// memory instead of every lane's private arrays. The lanes fill its slots together
// (pm_fill_workgroup_window), one lane sums the weights in the serial order
// (pm_sum_workgroup_window), and every NCC evaluation reads it. Each slot's color and weight come from
// the same pm_ref_texel and pm_bilateral_weight inputs as pm_prepare_window's, so the values match the
// private cache's; the kernel orders fill, sum and reads with its barriers. An uncached (large)
// window keeps recomputing per evaluation and leaves these arrays at one element.
const PM_WINDOW_FROM_WORKGROUP: bool = PM_WINDOW_SHARED && PM_WINDOW_CACHED;
const PM_WINDOW_WORKGROUP_SIZE: i32 = select(1i, PM_WINDOW_COUNT, PM_WINDOW_FROM_WORKGROUP);
var<workgroup> pm_wg_window_colors: array<f32, PM_WINDOW_WORKGROUP_SIZE>;
var<workgroup> pm_wg_window_weights: array<f32, PM_WINDOW_WORKGROUP_SIZE>;
var<workgroup> pm_wg_window_weight_sum: array<f32, 1>;

// The reference half of the NCC at one pixel: the window's weight sum and its centre color. With
// the cache on, pm_prepare_window has also filled pm_window_colors / pm_window_weights.
struct PmWindow {
	weight_sum: f32,
	center_color: f32,
}

// PatchMatchRefImage.Texel on the rotated reference plane: the pixel's byte / 255, or 0 outside
// the frame. The unsigned compares reject negatives too, like C#'s (uint) casts.
fn pm_ref_texel(rotation: i32, row: i32, col: i32) -> f32 {
	let width = pm_frame_width(rotation);
	let height = pm_frame_height(rotation);
	if (bitcast<u32>(row) >= bitcast<u32>(height) || bitcast<u32>(col) >= bitcast<u32>(width)) {
		return 0.0f;
	}

	return pm_reference[row * width + col];
}

// BilateralWeightComputer.Compute. exp is WGSL's builtin (Tier C; MathF.Exp in C#).
fn pm_bilateral_weight(row_diff: f32, col_diff: f32, color1: f32, color2: f32) -> f32 {
	let spatial_dist_squared = row_diff * row_diff + col_diff * col_diff;
	let color_dist = color1 - color2;
	return exp(-spatial_dist_squared * pm_problem.spatial_normalization - color_dist * color_dist * pm_problem.color_normalization);
}

// PrepareWindow: the window's reference colors and bilateral weights in COLMAP's loop order (into
// the private cache when it is on) and their weight sum accumulated in that order.
fn pm_prepare_window(rotation: i32, row: i32, col: i32) -> PmWindow {
	let ref_center_color = pm_ref_texel(rotation, row, col);
	var bilateral_weight_sum = 0.0f;
	var i = 0i;
	for (var window_row = -PM_WINDOW_RADIUS; window_row <= PM_WINDOW_RADIUS; window_row += PM_WINDOW_STEP) {
		for (var window_col = -PM_WINDOW_RADIUS; window_col <= PM_WINDOW_RADIUS; window_col += PM_WINDOW_STEP) {
			let ref_color = pm_ref_texel(rotation, row + window_row, col + window_col);
			let bilateral_weight = pm_bilateral_weight(f32(window_row), f32(window_col), ref_center_color, ref_color);
			if (PM_WINDOW_CACHED) {
				pm_window_colors[i] = ref_color;
				pm_window_weights[i] = bilateral_weight;
			}

			bilateral_weight_sum += bilateral_weight;
			i += 1i;
		}
	}

	return PmWindow(bilateral_weight_sum, ref_center_color);
}

// The shared window's slots lane, lane + lanes, ...: slot i is pm_prepare_window's i-th sample
// (row-major over the strided window), with the same integer offsets and so the same floats.
fn pm_fill_workgroup_window(rotation: i32, row: i32, col: i32, lane: i32, lanes: i32) {
	if (!PM_WINDOW_FROM_WORKGROUP) {
		return;
	}

	let ref_center_color = pm_ref_texel(rotation, row, col);
	for (var i = lane; i < PM_WINDOW_COUNT; i += lanes) {
		let window_row = (i / PM_WINDOW_PER_AXIS) * PM_WINDOW_STEP - PM_WINDOW_RADIUS;
		let window_col = (i % PM_WINDOW_PER_AXIS) * PM_WINDOW_STEP - PM_WINDOW_RADIUS;
		let ref_color = pm_ref_texel(rotation, row + window_row, col + window_col);
		pm_wg_window_colors[i] = ref_color;
		pm_wg_window_weights[i] = pm_bilateral_weight(f32(window_row), f32(window_col), ref_center_color, ref_color);
	}
}

// The shared window's weight sum, accumulated in pm_prepare_window's order; one lane calls it after
// the fill is visible and stores it for the others.
fn pm_sum_workgroup_window() {
	var bilateral_weight_sum = 0.0f;
	for (var i = 0i; i < PM_WINDOW_COUNT; i += 1i) {
		bilateral_weight_sum += pm_wg_window_weights[i];
	}

	pm_wg_window_weight_sum[0] = bilateral_weight_sum;
}

// 1 - NCC from the weighted source sums and the reference window's statistics (FinishNcc).
fn pm_finish_ncc(src_color_sum0: f32, src_color_squared_sum0: f32, src_ref_color_sum0: f32, bilateral_weight_sum: f32, ref_color_sum: f32, ref_color_squared_sum: f32) -> f32 {
	// The sum can be zero: when PM_WINDOW_STEP does not divide PM_WINDOW_RADIUS the centre (weight
	// exp(0) = 1) is never sampled, and every sampled weight can underflow to 0. pm_div keeps C#'s
	// result for that case explicit: an infinite reciprocal. Every source sum is then 0 (each term
	// has a zero weight factor), so the normalized sums are 0 * inf = NaN; see below for the cost.
	let inv_bilateral_weight_sum = pm_div(1.0f, bilateral_weight_sum);
	let src_color_sum = src_color_sum0 * inv_bilateral_weight_sum;
	let src_color_squared_sum = src_color_squared_sum0 * inv_bilateral_weight_sum;
	let src_ref_color_sum = src_ref_color_sum0 * inv_bilateral_weight_sum;

	let ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;
	let src_color_var = src_color_squared_sum - src_color_sum * src_color_sum;

	// Based on Jensen's Inequality for convex functions, the variance should always be larger
	// than 0. Do not make this threshold smaller. The terms are finite (samples are in [0, 1])
	// unless the weight sum was zero, when src_color_var is NaN. C# then compares false and falls
	// through to cuda_min(PM_MAX_COST, NaN) = PM_MAX_COST; a GPU whose NaN comparison answers true
	// returns PM_MAX_COST here. Either way the cost is PM_MAX_COST, so this needs no NaN guard.
	const MIN_VAR: f32 = 1e-5f;
	if (ref_color_var < MIN_VAR || src_color_var < MIN_VAR) {
		return PM_MAX_COST;
	}

	// Both variances are at least MIN_VAR, so the divisor is positive.
	let src_ref_color_covar = src_ref_color_sum - ref_color_sum * src_color_sum;
	let src_ref_color_var = sqrt(ref_color_var * src_color_var);
	return cuda_max(0.0f, cuda_min(PM_MAX_COST, 1.0f - src_ref_color_covar / src_ref_color_var));
}

// PatchMatchPhotoConsistency.Compute with the window prepared by pm_prepare_window for the same
// pixel: 1 - NCC in [0, 2] of the patch around pixel (row, col) warped into source image `src`
// by the plane at `depth` with `normal`; 2 when either patch has (nearly) no variance.
fn pm_compute_ncc_cost(rotation: i32, frame: PmFrame, row: i32, col: i32, depth: f32, normal: vec3<f32>, src: i32, window: PmWindow) -> f32 {
	let tform = pm_compose_homography(rotation, frame, src, row, col, depth, normal);

	let window_step = f32(PM_WINDOW_STEP);
	let step0 = window_step * tform[0];
	let step1 = window_step * tform[1];
	let step3 = window_step * tform[3];
	let step4 = window_step * tform[4];
	let step6 = window_step * tform[6];
	let step7 = window_step * tform[7];

	let row_start = row - PM_WINDOW_RADIUS;
	let col_start = col - PM_WINDOW_RADIUS;

	var col_src = tform[0] * f32(col_start) + tform[1] * f32(row_start) + tform[2];
	var row_src = tform[3] * f32(col_start) + tform[4] * f32(row_start) + tform[5];
	var z = tform[6] * f32(col_start) + tform[7] * f32(row_start) + tform[8];
	var base_col_src = col_src;
	var base_row_src = row_src;
	var base_z = z;

	let pixel = row * pm_frame_width(rotation) + col;
	let ref_color_sum = pm_reference[PM_PLANE_SIZE + pixel];
	let ref_color_squared_sum = pm_reference[2i * PM_PLANE_SIZE + pixel];
	var src_color_sum = 0.0f;
	var src_color_squared_sum = 0.0f;
	var src_ref_color_sum = 0.0f;

	var i = 0i;
	for (var window_row = 0i; window_row < PM_WINDOW_PER_AXIS; window_row += 1i) {
		for (var window_col = 0i; window_col < PM_WINDOW_PER_AXIS; window_col += 1i) {
			// Guard: a zero z gives C# an infinite inv_z (the sample then reads 0 or NaN
			// coordinates, which pm_sample_source_image returns 0 for).
			let inv_z = pm_div(1.0f, z);
			let norm_col_src = inv_z * col_src + 0.5f;
			let norm_row_src = inv_z * row_src + 0.5f;
			var ref_color: f32;
			var weight: f32;
			if (PM_WINDOW_FROM_WORKGROUP) {
				ref_color = pm_wg_window_colors[i];
				weight = pm_wg_window_weights[i];
			} else if (PM_WINDOW_CACHED) {
				ref_color = pm_window_colors[i];
				weight = pm_window_weights[i];
			} else {
				let offset_row = window_row * PM_WINDOW_STEP - PM_WINDOW_RADIUS;
				let offset_col = window_col * PM_WINDOW_STEP - PM_WINDOW_RADIUS;
				ref_color = pm_ref_texel(rotation, row + offset_row, col + offset_col);
				weight = pm_bilateral_weight(f32(offset_row), f32(offset_col), window.center_color, ref_color);
			}

			let src_color = pm_sample_source_image(norm_col_src, norm_row_src, src);

			let bilateral_weight_src = weight * src_color;

			src_color_sum += bilateral_weight_src;
			src_color_squared_sum += bilateral_weight_src * src_color;
			src_ref_color_sum += bilateral_weight_src * ref_color;

			// Accumulate warped source coordinates per row to reduce numerical errors. Note that
			// this is necessary since coordinates usually are in the order of 1000s as opposed to
			// the color values which are normalized to the range [0, 1].
			col_src += step0;
			row_src += step3;
			z += step6;
			i += 1i;
		}

		base_col_src += step1;
		base_row_src += step4;
		base_z += step7;

		col_src = base_col_src;
		row_src = base_row_src;
		z = base_z;
	}

	return pm_finish_ncc(src_color_sum, src_color_squared_sum, src_ref_color_sum, window.weight_sum, ref_color_sum, ref_color_squared_sum);
}
