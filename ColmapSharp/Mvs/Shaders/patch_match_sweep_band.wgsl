// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_sweep_band.wgsl: the sweep_band kernel of GPU PatchMatch - one band of rows
// [row_start, row_end) of a top-to-bottom sweep. Per row it proposes five depth/normal hypotheses
// (current, propagated from the row above, random, and the two mixes), scores them against source
// images drawn by Monte Carlo sampling from the selection probabilities, keeps the best, and
// updates costs, forward messages and selection probabilities. Two schemes, chosen by the
// PM_SWEEP_COOPERATIVE constant: cooperative (the default) runs one workgroup per column of the
// current rotation's frame, its lanes splitting each row's work (see "Cooperative sweep" below);
// serial runs one invocation per column, as COLMAP does. Both walk the band's rows in order and,
// in WGSL semantics, give the same floats; on Metal only the photometric sums come out
// bit-identical, because wgpu compiles with fast math (see "Cooperative sweep"). Element count width (cooperative: width workgroups) with width =
// pm_frame_width(rotation); the host composes for C = max(width, height) columns. A column
// resumes from, and leaves for the next band, its column state (patch_match_layout.wgsl), so the
// bands of a sweep give the same result as one pass; backward_messages must have run first.
// Mirrors: PatchMatchCpu.SweepRows in ColmapSharp/Mvs/PatchMatchCpu.Sweep.cs, line by line (its
//   ComputeFour scores hypotheses 1-4 bit-identically to four scalar Computes, which run here).
// Ports: the row loop of patch_match_cuda.cu's SweepFromTopToBottom, as the CPU port models it
//   (docs/CPP_DIVERGENCES.md, entries 86, 95 and 96).
// Parts: dispatch, common, textures, geometry, likelihood, layout, ncc, geom_cost
//   (ColmapSharp/Mvs/PatchMatchGpuKernels.cs); constants PM_NUM_SAMPLES: i32,
//   PM_GEOM_CONSISTENCY: bool (the geometric term) and PM_SWEEP_COOPERATIVE: bool besides those
//   of the parts.
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
	// A const condition, so both branches are in uniform control flow (the cooperative one has
	// barriers). Checked by hand, not by a validator: the cooperative path's only early return
	// depends on workgroup_id and the uniform pm_sweep; its row loop's bounds come from the
	// uniform pm_band; and no barrier sits inside a per-lane loop or a lane-dependent branch.
	if (PM_SWEEP_COOPERATIVE) {
		pm_sweep_band_cooperative(workgroup_id, local_index);
	} else {
		pm_sweep_band_serial(workgroup_id, local_index);
	}
}

// v1: one invocation per column, the whole row's work in that invocation (COLMAP's scheme).
fn pm_sweep_band_serial(workgroup_id: vec3<u32>, local_index: u32) {
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
				let first = pm_wg_first_sample[image_idx];
				if (first != -1i) {
					cost = pm_wg_ncc[first * PM_NUM_COSTS + min_cost_idx];
				} else {
					cost = pm_compute_ncc_cost(rotation, frame, row, col, best_depth, best_normal, image_idx, window);
				}

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

// ---- Cooperative sweep: one workgroup per column ----
//
// The workgroup's PM_WORKGROUP_SIZE lanes walk the column's rows together (the row loop's bounds
// are uniform, so every barrier is reached by all lanes) and split each row's independent work:
// the per-source loops (sampling priors, the best hypothesis's cost and messages) go to lane
// image_idx mod lanes, and the (sample, hypothesis) cost evaluations to lane item mod lanes. Each
// value is computed by the same function, from the same inputs, as the serial path computes it;
// every order-dependent step - the PDF-to-CDF sums, the random draws, and each hypothesis's sum
// over samples - runs in one lane in the serial order. So in WGSL semantics the cooperative and
// serial paths give the same floats. On Metal only the photometric sums are bit-identical:
// wgpu-hal compiles with the default MTLCompileOptions, i.e. fastMathEnabled, so the compiler may
// reassociate `cost += ncc; cost += geom` differently in each kernel, while a photometric
// hypothesis gets one addition per sample. ColmapGpuTests pins photometric runs bit for bit and
// geometric runs within tight bounds on a real GPU. The CPU twin (ReferenceComputeDevice) checks
// this scheme's plan, dispatch shape and band coverage, but runs PatchMatchCpu.SweepRows, not this
// WGSL; the real-GPU test is what pins the kernel. Every lane redundantly carries the row's
// scalars (propagated depth, current and random hypotheses): they are deterministic, so all lanes
// hold the same values, and recomputing is cheaper than sharing. The reference window is the
// exception: its colors and bilateral weights (an exp each) are filled into workgroup memory by
// all lanes together and its weight sum taken by lane 0 in the serial order (patch_match_ncc.wgsl's
// shared window), which on Metal took ~26% off a sweep with bit-identical photometric output.

// Per-source forward messages: lane image_idx mod lanes owns entry image_idx for the whole band.
var<workgroup> pm_wg_forward_message: array<f32, PM_NUM_SRC_IMAGES>;
// The row's sampling priors (before the CDF transform).
var<workgroup> pm_wg_sampling_probs: array<f32, PM_NUM_SRC_IMAGES>;
// Per sample: the drawn source image (-1 when none) and hypothesis 0's stored cost for it.
var<workgroup> pm_wg_sample_src: array<i32, PM_NUM_SAMPLES>;
var<workgroup> pm_wg_sample_cost0: array<f32, PM_NUM_SAMPLES>;
// Per (sample, hypothesis): the NCC cost (hypotheses 1-4) and the geometric term.
var<workgroup> pm_wg_ncc: array<f32, PM_NUM_SAMPLES * PM_NUM_COSTS>;
var<workgroup> pm_wg_geom: array<f32, PM_NUM_SAMPLES * PM_NUM_COSTS>;
// The five hypotheses' summed costs.
var<workgroup> pm_wg_hypothesis_costs: array<f32, PM_NUM_COSTS>;
// Per source: the first sample that drew it (-1 when none). Its hypothesis costs are the NCC costs
// the final per-source loop would recompute - the same function on the same inputs - so that loop
// reads them from pm_wg_ncc instead. Bit-identical on M5 Metal and ~10% off a sweep.
var<workgroup> pm_wg_first_sample: array<i32, PM_NUM_SRC_IMAGES>;

fn pm_hypothesis_depth(i: i32, curr: f32, prev: f32, rand: f32) -> f32 {
	if (i == 0i || i == 3i) {
		return curr;
	}

	if (i == 1i) {
		return prev;
	}

	return rand;
}

fn pm_sweep_band_cooperative(workgroup_id: vec3<u32>, local_index: u32) {
	let rotation = pm_sweep.rotation;
	let width = pm_frame_width(rotation);

	// The workgroup's column; workgroup_id is uniform, so this return is too.
	let col_index = workgroup_id.y * PM_GROUPS_X + workgroup_id.x;
	if (col_index >= u32(width)) {
		return;
	}

	let lane = i32(local_index);
	let lanes = i32(PM_WORKGROUP_SIZE);
	let col = i32(col_index);
	let frame = pm_frame(rotation);
	let state = PM_COLUMN_STATES_OFFSET + col * PM_COLUMN_STATE_SIZE;

	for (var image_idx = lane; image_idx < PM_NUM_SRC_IMAGES; image_idx += lanes) {
		pm_wg_forward_message[image_idx] = pm_state[state + image_idx];
	}

	var prev_depth = pm_state[state + PM_NUM_SRC_IMAGES];
	var prev_normal = vec3<f32>(pm_state[state + PM_NUM_SRC_IMAGES + 1i], pm_state[state + PM_NUM_SRC_IMAGES + 2i], pm_state[state + PM_NUM_SRC_IMAGES + 3i]);

	for (var row = pm_band.row_start; row < pm_band.row_end; row += 1i) {
		let pixel = row * width + col;
		let orig = pm_to_original_pixel(rotation, row, col);
		var random = pm_random_init(pm_problem.seed, orig.x, orig.y, pm_sweep.phase);

		prev_depth = pm_propagate_depth(frame, prev_depth, prev_normal, f32(row - 1i), f32(row));
		let curr_depth = pm_state[pixel];
		let curr_normal = pm_read_normal(pixel);
		let rand_depth = pm_perturb_depth(pm_sweep.perturbation, curr_depth, &random);
		let rand_normal = pm_perturb_normal(frame, row, col, pm_sweep.normal_perturbation, curr_normal, &random);

		// The sampling priors, one source per lane.
		let point = pm_compute_point_at_depth(frame, f32(row), f32(col), curr_depth);
		for (var image_idx = lane; image_idx < PM_NUM_SRC_IMAGES; image_idx += lanes) {
			let idx = image_idx * PM_PLANE_SIZE + pixel;
			let alpha = pm_compute_forward_message(pm_problem.lk, pm_costs[idx], pm_wg_forward_message[image_idx]);
			let sel_prob = pm_compute_sel_prob(alpha, pm_sel_probs[idx], pm_prev_sel_probs[idx], pm_sweep.prev_sel_prob_weight);

			let angles = pm_compute_viewing_angles(rotation, point, curr_normal, image_idx);
			let tri_prob = pm_compute_tri_prob(pm_problem.lk, angles.x);
			let inc_prob = pm_compute_inc_prob(pm_problem.lk, angles.y);

			let h = pm_compose_homography(rotation, frame, image_idx, row, col, curr_depth, curr_normal);
			let res_prob = pm_compute_resolution_prob(h, f32(row), f32(col), PM_WINDOW_RADIUS);

			pm_wg_sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
		}

		// The shared reference window's slots, split across the lanes (patch_match_ncc.wgsl).
		pm_fill_workgroup_window(rotation, row, col, lane, lanes);

		workgroupBarrier();

		// Lane 0: the CDF and the draws, in the serial order, and hypothesis 0's stored costs.
		if (lane == 0i) {
			// The shared window's weight sum, in the serial order, after the fill barrier above.
			if (PM_WINDOW_FROM_WORKGROUP) {
				pm_sum_workgroup_window();
			}

			var sampling_probs: array<f32, PM_NUM_SRC_IMAGES>;
			for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
				sampling_probs[image_idx] = pm_wg_sampling_probs[image_idx];
			}

			pm_transform_pdf_to_cdf(&sampling_probs);
			for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
				pm_wg_first_sample[image_idx] = -1i;
			}

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

				pm_wg_sample_src[sample] = src_image_idx;
				if (src_image_idx != -1i) {
					if (pm_wg_first_sample[src_image_idx] == -1i) {
						pm_wg_first_sample[src_image_idx] = sample;
					}
					pm_wg_sample_cost0[sample] = pm_costs[src_image_idx * PM_PLANE_SIZE + pixel];
				}
			}
		}

		// The reference half of the NCC: an uncached window's weight sum, the same in every lane.
		var window: PmWindow;
		if (!PM_WINDOW_FROM_WORKGROUP) {
			window = pm_prepare_window(rotation, row, col);
		}

		// storageBarrier orders lane 0's pm_costs reads above before the other lanes' pm_costs
		// writes below (a cross-lane write-after-read on the same pixel). The workgroup barrier
		// also publishes lane 0's shared window weight sum.
		storageBarrier();
		workgroupBarrier();

		if (PM_WINDOW_FROM_WORKGROUP) {
			window = PmWindow(pm_wg_window_weight_sum[0], pm_ref_texel(rotation, row, col));
		}

		// The (sample, hypothesis) evaluations, one per lane: hypotheses 1-4 get the NCC cost,
		// every hypothesis the geometric term.
		for (var item = lane; item < PM_NUM_SAMPLES * PM_NUM_COSTS; item += lanes) {
			let sample = item / PM_NUM_COSTS;
			let i = item % PM_NUM_COSTS;
			let src_image_idx = pm_wg_sample_src[sample];
			if (src_image_idx == -1i) {
				continue;
			}

			let depth = pm_hypothesis_depth(i, curr_depth, prev_depth, rand_depth);
			if (i != 0i) {
				let normal = pm_hypothesis_normal(i, curr_normal, prev_normal, rand_normal);
				pm_wg_ncc[item] = pm_compute_ncc_cost(rotation, frame, row, col, depth, normal, src_image_idx, window);
			}

			if (PM_GEOM_CONSISTENCY) {
				pm_wg_geom[item] = pm_geom_term(rotation, frame, row, col, depth, src_image_idx);
			}
		}

		workgroupBarrier();

		// Lane i sums hypothesis i over the samples in the serial order.
		if (lane < PM_NUM_COSTS) {
			var cost = 0.0f;
			for (var sample = 0i; sample < PM_NUM_SAMPLES; sample += 1i) {
				if (pm_wg_sample_src[sample] == -1i) {
					continue;
				}

				let item = sample * PM_NUM_COSTS + lane;
				if (lane == 0i) {
					cost += pm_wg_sample_cost0[sample];
				} else {
					cost += pm_wg_ncc[item];
				}

				if (PM_GEOM_CONSISTENCY) {
					cost += pm_wg_geom[item];
				}
			}

			pm_wg_hypothesis_costs[lane] = cost;
		}

		// storageBarrier orders every lane's pm_state[pixel] reads at the top of the row before
		// lane 0 writes the best parameters there below.
		storageBarrier();
		workgroupBarrier();

		var hypothesis_costs: array<f32, PM_NUM_COSTS>;
		for (var i = 0i; i < PM_NUM_COSTS; i += 1i) {
			hypothesis_costs[i] = pm_wg_hypothesis_costs[i];
		}

		let min_cost_idx = pm_find_min_cost(hypothesis_costs);
		let best_depth = pm_hypothesis_depth(min_cost_idx, curr_depth, prev_depth, rand_depth);
		let best_normal = pm_hypothesis_normal(min_cost_idx, curr_normal, prev_normal, rand_normal);

		if (lane == 0i) {
			pm_state[pixel] = best_depth;
			pm_state[PM_PLANE_SIZE + pixel] = best_normal.x;
			pm_state[2i * PM_PLANE_SIZE + pixel] = best_normal.y;
			pm_state[3i * PM_PLANE_SIZE + pixel] = best_normal.z;
		}

		// The best hypothesis's cost, forward message and selection probability, one source per
		// lane (the lane that owns its forward message).
		for (var image_idx = lane; image_idx < PM_NUM_SRC_IMAGES; image_idx += lanes) {
			let idx = image_idx * PM_PLANE_SIZE + pixel;
			var cost: f32;
			if (min_cost_idx == 0i) {
				cost = pm_costs[idx];
			} else {
				let first = pm_wg_first_sample[image_idx];
				if (first != -1i) {
					cost = pm_wg_ncc[first * PM_NUM_COSTS + min_cost_idx];
				} else {
					cost = pm_compute_ncc_cost(rotation, frame, row, col, best_depth, best_normal, image_idx, window);
				}

				pm_costs[idx] = cost;
			}

			let alpha = pm_compute_forward_message(pm_problem.lk, cost, pm_wg_forward_message[image_idx]);
			let prob = pm_compute_sel_prob(alpha, pm_sel_probs[idx], pm_prev_sel_probs[idx], pm_sweep.prev_sel_prob_weight);
			pm_wg_forward_message[image_idx] = alpha;
			pm_sel_probs[idx] = prob;
		}

		prev_depth = best_depth;
		prev_normal = best_normal;

		// The next row overwrites the shared arrays. (The storage write-after-read hazards within
		// the row are ordered by the storage barriers above; across rows the pixels differ.)
		workgroupBarrier();
	}

	for (var image_idx = lane; image_idx < PM_NUM_SRC_IMAGES; image_idx += lanes) {
		pm_state[state + image_idx] = pm_wg_forward_message[image_idx];
	}

	if (lane == 0i) {
		pm_state[state + PM_NUM_SRC_IMAGES] = prev_depth;
		pm_state[state + PM_NUM_SRC_IMAGES + 1i] = prev_normal.x;
		pm_state[state + PM_NUM_SRC_IMAGES + 2i] = prev_normal.y;
		pm_state[state + PM_NUM_SRC_IMAGES + 3i] = prev_normal.z;
	}
}
