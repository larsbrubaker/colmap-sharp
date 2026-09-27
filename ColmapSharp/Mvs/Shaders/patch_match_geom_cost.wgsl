// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_geom_cost.wgsl: the geometric-consistency cost of GPU PatchMatch - the
// forward-backward reprojection error of a hypothesis through a source image's depth map.
// Mirrors: PatchMatchKernel.ComputeGeomConsistencyCost in
//   ColmapSharp/Mvs/PatchMatchKernel.Geometry.cs, in the same operation order.
// Ports: patch_match_cuda.cu's ComputeGeomConsistencyCost, as the CPU port models it
//   (docs/CPP_DIVERGENCES.md, entry 96).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after common, textures (pm_sample_source_depth)
// and geometry (the pose table, pm_div, pm_compute_point_at_depth). It declares no bindings.
// Guards against non-finite values follow patch_match_geometry.wgsl's rules ("Guard:").

// The forward-backward reprojection error, capped at `max_cost`, of the point at `depth` on pixel
// (row, col) of the frame rotated `rotation` times through source image `image_idx`'s depth map;
// `max_cost` when the point projects outside the source depth map (depth 0).
fn pm_compute_geom_consistency_cost(rotation: i32, frame: PmFrame, row: f32, col: f32, depth: f32, image_idx: i32, max_cost: f32) -> f32 {
	let pose = pm_pose_row(rotation, image_idx);

	// Extract projection matrices for source image.
	var p: array<f32, 12>;
	var inv_p: array<f32, 12>;
	for (var i = 0i; i < 12i; i = i + 1i) {
		p[i] = pm_pose_float(pose + PM_POSE_P_OFFSET + i);
		inv_p[i] = pm_pose_float(pose + PM_POSE_INV_P_OFFSET + i);
	}

	// Project point in reference image to world.
	let forward_point = pm_compute_point_at_depth(frame, row, col, depth);

	// Project world point to source image.
	// Guard: a point on the source's principal plane gives C# an infinite inverse.
	let inv_forward_z = pm_div(1.0f, p[8] * forward_point.x + p[9] * forward_point.y + p[10] * forward_point.z + p[11]);
	var src_col = inv_forward_z * (p[0] * forward_point.x + p[1] * forward_point.y + p[2] * forward_point.z + p[3]);
	var src_row = inv_forward_z * (p[4] * forward_point.x + p[5] * forward_point.y + p[6] * forward_point.z + p[7]);

	// Extract depth in source image. (Non-finite coordinates sample 0, as in C#.)
	let src_depth = pm_sample_source_depth(src_col + 0.5f, src_row + 0.5f, image_idx);

	// Projection outside of source image.
	if (src_depth == 0.0f) {
		return max_cost;
	}

	// Project point in source image to world.
	src_col *= src_depth;
	src_row *= src_depth;
	let backward_point_x = inv_p[0] * src_col + inv_p[1] * src_row + inv_p[2] * src_depth + inv_p[3];
	let backward_point_y = inv_p[4] * src_col + inv_p[5] * src_row + inv_p[6] * src_depth + inv_p[7];
	let backward_point_z = inv_p[8] * src_col + inv_p[9] * src_row + inv_p[10] * src_depth + inv_p[11];
	// Guard: likewise for a backward point on the reference's principal plane.
	let inv_backward_point_z = pm_div(1.0f, backward_point_z);

	// Project world point back to reference image.
	let backward_col = inv_backward_point_z * (frame.k.x * backward_point_x + frame.k.y * backward_point_z);
	let backward_row = inv_backward_point_z * (frame.k.z * backward_point_y + frame.k.w * backward_point_z);

	// Return truncated reprojection error between original observation and the forward-backward
	// projected observation. cuda_min returns max_cost for a NaN error, as CUDA's fminf does.
	let diff_col = col - backward_col;
	let diff_row = row - backward_row;
	return cuda_min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}
