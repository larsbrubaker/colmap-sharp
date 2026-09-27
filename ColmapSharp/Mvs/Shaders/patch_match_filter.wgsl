// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_filter.wgsl: the filter_pixels kernel of GPU PatchMatch (not "filter", a reserved
// word in WGSL) - after the last sweep's rows and before its rotation, marks the source images
// each pixel's best depth and normal are consistent with, and clears the pixel when fewer than
// filter_min_num_consistent are. One invocation per pixel of the current rotation's frame;
// element count P, element e is row e / width, column e % width. It writes 0 or 1 to every
// (source, pixel) entry of the mask, because the mask reuses the cost buffer (still holding cost
// bits), where the CPU writes only the 1s of a zeroed mask.
// Mirrors: PatchMatchCpu.FilterPixels / FilterPixel in ColmapSharp/Mvs/PatchMatchCpu.Sweep.cs
//   (PatchMatchCpu.Sweep.cs explains why filtering after the sweep equals filtering per row).
// Ports: the filtering step of patch_match_cuda.cu's SweepFromTopToBottom, as the CPU port models
//   it (docs/CPP_DIVERGENCES.md, entry 96).
// Parts: dispatch, common, textures, geometry, likelihood, layout, geom_cost
//   (ColmapSharp/Mvs/PatchMatchGpuKernels.cs).
// Bindings: group 0 - 0 pm_byte_to_unit, 1 pm_poses, 2 pm_problem, 4 pm_source_images (textures
//   declares both), 5 pm_source_depths, 6 pm_state (read_write), 7 pm_mask (read_write: the cost
//   buffer, as u32), 8 pm_sel_probs (read); group 1 - 0 pm_sweep.

@group(0) @binding(0) var<uniform> pm_byte_to_unit: PmByteToUnit;
@group(0) @binding(1) var<uniform> pm_poses: PmPoseTable;
@group(0) @binding(2) var<uniform> pm_problem: PmProblem;
@group(0) @binding(4) var<storage, read> pm_source_images: array<u32>;
@group(0) @binding(5) var<storage, read> pm_source_depths: array<f32>;
@group(0) @binding(6) var<storage, read_write> pm_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> pm_mask: array<u32>;
@group(0) @binding(8) var<storage, read> pm_sel_probs: array<f32>;
@group(1) @binding(0) var<uniform> pm_sweep: PmSweep;

fn pm_filter_geom_cost(rotation: i32, frame: PmFrame, row: i32, col: i32, depth: f32, image_idx: i32) -> f32 {
	return pm_compute_geom_consistency_cost(rotation, frame, f32(row), f32(col), depth, image_idx, pm_problem.geom_consistency_max_cost);
}

// C#'s `cost <= max`, false for a NaN cost.
fn pm_geom_consistent(cost: f32) -> bool {
	return !is_nan_f32(cost) && cost <= pm_problem.filter_geom_consistency_max_cost;
}

// C#'s `selProb >= minNccProb`, false for a NaN probability.
fn pm_photo_consistent(sel_prob: f32) -> bool {
	return !is_nan_f32(sel_prob) && sel_prob >= pm_problem.filter_min_ncc_prob;
}

@compute @workgroup_size(64)
fn filter_pixels(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= u32(PM_PLANE_SIZE)) {
		return;
	}

	let rotation = pm_sweep.rotation;
	let width = pm_frame_width(rotation);
	let pixel = i32(index);
	let row = pixel / width;
	let col = pixel % width;
	let frame = pm_frame(rotation);
	let best_depth = pm_state[pixel];
	let best_normal = vec3<f32>(pm_state[PM_PLANE_SIZE + pixel], pm_state[2i * PM_PLANE_SIZE + pixel], pm_state[3i * PM_PLANE_SIZE + pixel]);
	let filter_photo = pm_sweep.filter_photo_consistency != 0u;
	let filter_geom = pm_sweep.filter_geom_consistency != 0u;

	var num_consistent = 0i;

	let best_point = pm_compute_point_at_depth(frame, f32(row), f32(col), best_depth);

	for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
		let idx = image_idx * PM_PLANE_SIZE + pixel;
		let angles = pm_compute_viewing_angles(rotation, best_point, best_normal, image_idx);
		// Guard: C#'s comparisons are false for NaN cosines, so the image is then not skipped.
		let above_min_angle = !is_nan_f32(angles.x) && angles.x > pm_problem.filter_cos_min_triangulation_angle;
		let back_facing = !is_nan_f32(angles.y) && angles.y <= 0.0f;
		var consistent = false;
		if (!(above_min_angle || back_facing)) {
			if (!filter_geom) {
				consistent = pm_photo_consistent(pm_sel_probs[idx]);
			} else if (!filter_photo) {
				consistent = pm_geom_consistent(pm_filter_geom_cost(rotation, frame, row, col, best_depth, image_idx));
			} else {
				// C#'s && evaluates the geometric cost only for a photo-consistent image; the cost
				// has no side effects, so evaluating it only then is just the cheaper order.
				consistent = pm_photo_consistent(pm_sel_probs[idx])
					&& pm_geom_consistent(pm_filter_geom_cost(rotation, frame, row, col, best_depth, image_idx));
			}
		}

		pm_mask[idx] = select(0u, 1u, consistent);
		if (consistent) {
			num_consistent += 1i;
		}
	}

	if (num_consistent < pm_problem.filter_min_num_consistent) {
		pm_state[pixel] = 0.0f;
		pm_state[PM_PLANE_SIZE + pixel] = 0.0f;
		pm_state[2i * PM_PLANE_SIZE + pixel] = 0.0f;
		pm_state[3i * PM_PLANE_SIZE + pixel] = 0.0f;
		for (var image_idx = 0i; image_idx < PM_NUM_SRC_IMAGES; image_idx += 1i) {
			pm_mask[image_idx * PM_PLANE_SIZE + pixel] = 0u;
		}
	}
}
