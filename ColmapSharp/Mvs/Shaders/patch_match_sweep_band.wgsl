// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_sweep_band.wgsl: the sweep_band kernel of GPU PatchMatch - one band of rows
// [row_start, row_end) of a top-to-bottom sweep. Per row it proposes five depth/normal hypotheses
// (current, propagated from the row above, random, and the two mixes), scores them against source
// images drawn by Monte Carlo sampling from the selection probabilities, keeps the best, and
// updates costs, forward messages and selection probabilities. One invocation per column of the
// current rotation's frame, walking the band's rows in order; element count width
// (pm_frame_width(rotation); the host dispatches for C = max(width, height) columns). A column
// resumes from, and leaves for the next band, its column state (patch_match_layout.wgsl), so the
// bands of a sweep give the same result as one pass; backward_messages must have run first.
// Mirrors: PatchMatchCpu.SweepRows in ColmapSharp/Mvs/PatchMatchCpu.Sweep.cs, line by line (its
//   ComputeFour scores hypotheses 1-4 bit-identically to four scalar Computes, which run here).
// Ports: the row loop of patch_match_cuda.cu's SweepFromTopToBottom, as the CPU port models it
//   (docs/CPP_DIVERGENCES.md, entries 86, 95 and 96).
// Parts: dispatch, common, textures, geometry, likelihood, layout, ncc, geom_cost
//   (ColmapSharp/Mvs/PatchMatchGpuKernels.cs); constants PM_NUM_SAMPLES: i32 and
//   PM_GEOM_CONSISTENCY: bool (the geometric term) besides those of the parts.
// Bindings: group 0 - 0 pm_byte_to_unit, 1 pm_poses, 2 pm_problem, 3 pm_reference,
//   4 pm_source_images, 5 pm_source_depths, 6 pm_state (read_write), 7 pm_costs (read_write),
//   8 pm_sel_probs (read_write: backward messages in, selection probabilities out),
//   9 pm_prev_sel_probs (read); group 1 - 0 pm_sweep; group 2 - 0 pm_band. Seven storage bindings.

@group(0) @binding(0) var<uniform> pm_byte_to_unit: PmByteToUnit;
@group(0) @binding(1) var<uniform> pm_poses: PmPoseTable;
@group(0) @binding(2) var<uniform> pm_problem: PmProblem;
@group(0) @binding(3) var<storage, read> pm_reference: array<f32>;
@group(0) @binding(4) var<storage, read> pm_source_images: array<u32>;
@group(0) @binding(5) var<storage, read> pm_source_depths: array<f32>;
@group(0) @binding(6) var<storage, read_write> pm_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> pm_costs: array<f32>;
@group(0) @binding(8) var<storage, read_write> pm_sel_probs: array<f32>;
@group(0) @binding(9) var<storage, read> pm_prev_sel_probs: array<f32>;
@group(1) @binding(0) var<uniform> pm_sweep: PmSweep;
@group(2) @binding(0) var<uniform> pm_band: PmBand;

// The normal of hypothesis i: {curr, prev, rand, rand, curr} (PatchMatchCpu.HypothesisNormal).
fn pm_hypothesis_normal(i: i32, curr: vec3<f32>, prev: vec3<f32>, rand: vec3<f32>) -> vec3<f32> {
	if (i == 0i || i == 4i) {
		return curr;
	}

	if (i == 1i) {
		return prev;
	}

	return rand;
}

fn pm_read_normal(pixel: i32) -> vec3<f32> {
	return vec3<f32>(pm_state[PM_PLANE_SIZE + pixel], pm_state[2i * PM_PLANE_SIZE + pixel], pm_state[3i * PM_PLANE_SIZE + pixel]);
}

// geom_consistency_regularizer * ComputeGeomConsistencyCost (PatchMatchCpu.GeomCost's use).
fn pm_geom_term(rotation: i32, frame: PmFrame, row: i32, col: i32, depth: f32, image_idx: i32) -> f32 {
	return pm_problem.geom_consistency_regularizer
		* pm_compute_geom_consistency_cost(rotation, frame, f32(row), f32(col), depth, image_idx, pm_problem.geom_consistency_max_cost);
}

@compute @workgroup_size(64)
fn sweep_band(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let rotation = pm_sweep.rotation;
	let width = pm_frame_width(rotation);
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= u32(width)) {
		return;
	}

	let col = i32(index);
	let frame = pm_frame(rotation);
	let state = PM_COLUMN_STATES_OFFSET + col * PM_COLUMN_STATE_SIZE;

	// The column state: forward messages, previous depth and normal.
	var forward_message: array<f32, PM_NUM_SRC_IMAGES>;
	for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
		forward_message[image_idx] = pm_state[state + image_idx];
	}

	var prev_depth = pm_state[state + PM_NUM_SRC_IMAGES];
	var prev_normal = vec3<f32>(pm_state[state + PM_NUM_SRC_IMAGES + 1i], pm_state[state + PM_NUM_SRC_IMAGES + 2i], pm_state[state + PM_NUM_SRC_IMAGES + 3i]);

	var sampling_probs: array<f32, PM_NUM_SRC_IMAGES>;
	var hypothesis_costs: array<f32, PM_NUM_COSTS>;
	var hypothesis_depths: array<f32, PM_NUM_COSTS>;

	for (var row = pm_band.row_start; row < pm_band.row_end; row += 1i) {
		let pixel = row * width + col;
		let orig = pm_to_original_pixel(rotation, row, col);
		var random = pm_random_init(pm_problem.seed, orig.x, orig.y, pm_sweep.phase);

		// Propagate the depth at which the current ray intersects with the plane of the normal of
		// the previous ray. This helps to better estimate the depth of very oblique structures,
		// i.e. pixels whose normal direction is significantly different from their viewing
		// direction.
		prev_depth = pm_propagate_depth(frame, prev_depth, prev_normal, f32(row - 1i), f32(row));

		// Read parameters for current pixel from previous sweep.
		let curr_depth = pm_state[pixel];
		let curr_normal = pm_read_normal(pixel);

		// Generate random parameters. The normal's perturbation angle is C#'s double product
		// (float)(perturbation * Math.PI), passed in the sweep uniform.
		let rand_depth = pm_perturb_depth(pm_sweep.perturbation, curr_depth, &random);
		let rand_normal = pm_perturb_normal(frame, row, col, pm_sweep.normal_perturbation, curr_normal, &random);

		// Read in the backward message, compute selection probabilities and modulate selection
		// probabilities with priors.
		let point = pm_compute_point_at_depth(frame, f32(row), f32(col), curr_depth);
		for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
			let idx = image_idx * PM_PLANE_SIZE + pixel;
			let alpha = pm_compute_forward_message(pm_problem.lk, pm_costs[idx], forward_message[image_idx]);
			let sel_prob = pm_compute_sel_prob(alpha, pm_sel_probs[idx], pm_prev_sel_probs[idx], pm_sweep.prev_sel_prob_weight);

			let angles = pm_compute_viewing_angles(rotation, point, curr_normal, image_idx);
			let tri_prob = pm_compute_tri_prob(pm_problem.lk, angles.x);
			let inc_prob = pm_compute_inc_prob(pm_problem.lk, angles.y);

			let h = pm_compose_homography(rotation, frame, image_idx, row, col, curr_depth, curr_normal);
			let res_prob = pm_compute_resolution_prob(h, f32(row), f32(col), PM_WINDOW_RADIUS);

			sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
		}

		pm_transform_pdf_to_cdf(&sampling_probs);

		// The reference half of the NCC is the same for every hypothesis and source image at this
		// pixel.
		let window = pm_prepare_window(rotation, row, col);

		// Compute matching cost using Monte Carlo sampling of source images. Images with higher
		// selection probability are more likely to be sampled. Hence, if only very few source
		// images see the reference image pixel, the same source image is likely to be sampled
		// many times. Instead of taking the best K probabilities, this sampling scheme has the
		// advantage of being adaptive to any distribution of selection probabilities.
		for (var i = 0i; i < PM_NUM_COSTS; i += 1i) {
			hypothesis_costs[i] = 0.0f;
		}

		hypothesis_depths[0] = curr_depth;
		hypothesis_depths[1] = prev_depth;
		hypothesis_depths[2] = rand_depth;
		hypothesis_depths[3] = curr_depth;
		hypothesis_depths[4] = rand_depth;

		for (var sample = 0i; sample < PM_NUM_SAMPLES; sample += 1i) {
			let rand_prob = pm_random_next_uniform(&random) - PM_FLOAT_EPSILON;

			var src_image_idx = -1i;
			for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
				// Guard: a NaN probability (all priors zero) fails C#'s `>`.
				let prob = sampling_probs[image_idx];
				if (!is_nan_f32(prob) && prob > rand_prob) {
					src_image_idx = image_idx;
					break;
				}
			}

			if (src_image_idx == -1i) {
				continue;
			}

			hypothesis_costs[0] += pm_costs[src_image_idx * PM_PLANE_SIZE + pixel];
			if (PM_GEOM_CONSISTENCY) {
				hypothesis_costs[0] += pm_geom_term(rotation, frame, row, col, hypothesis_depths[0], src_image_idx);
			}

			// Hypotheses 1-4: the CPU scores them in SIMD lockstep, bit-identical to these scalar
			// evaluations.
			for (var i = 1i; i < PM_NUM_COSTS; i += 1i) {
				let normal = pm_hypothesis_normal(i, curr_normal, prev_normal, rand_normal);
				hypothesis_costs[i] += pm_compute_ncc_cost(rotation, frame, row, col, hypothesis_depths[i], normal, src_image_idx, window);
				if (PM_GEOM_CONSISTENCY) {
					hypothesis_costs[i] += pm_geom_term(rotation, frame, row, col, hypothesis_depths[i], src_image_idx);
				}
			}
		}

		// Find the parameters of the minimum cost.
		let min_cost_idx = pm_find_min_cost(hypothesis_costs);
		let best_depth = hypothesis_depths[min_cost_idx];
		let best_normal = pm_hypothesis_normal(min_cost_idx, curr_normal, prev_normal, rand_normal);

		// Save best new parameters.
		pm_state[pixel] = best_depth;
		pm_state[PM_PLANE_SIZE + pixel] = best_normal.x;
		pm_state[2i * PM_PLANE_SIZE + pixel] = best_normal.y;
		pm_state[3i * PM_PLANE_SIZE + pixel] = best_normal.z;

		// Use the new cost to recompute the updated forward message and the selection probability.
		for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
			let idx = image_idx * PM_PLANE_SIZE + pixel;

			// Determine the cost for best depth.
			var cost: f32;
			if (min_cost_idx == 0i) {
				cost = pm_costs[idx];
			} else {
				cost = pm_compute_ncc_cost(rotation, frame, row, col, best_depth, best_normal, image_idx, window);
				pm_costs[idx] = cost;
			}

			let alpha = pm_compute_forward_message(pm_problem.lk, cost, forward_message[image_idx]);
			let prob = pm_compute_sel_prob(alpha, pm_sel_probs[idx], pm_prev_sel_probs[idx], pm_sweep.prev_sel_prob_weight);
			forward_message[image_idx] = alpha;
			pm_sel_probs[idx] = prob;
		}

		// Update previous depth for next row.
		prev_depth = best_depth;
		prev_normal = best_normal;
	}

	// Leave the column state for the next band.
	for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
		pm_state[state + image_idx] = forward_message[image_idx];
	}

	pm_state[state + PM_NUM_SRC_IMAGES] = prev_depth;
	pm_state[state + PM_NUM_SRC_IMAGES + 1i] = prev_normal.x;
	pm_state[state + PM_NUM_SRC_IMAGES + 2i] = prev_normal.y;
	pm_state[state + PM_NUM_SRC_IMAGES + 3i] = prev_normal.z;
}
