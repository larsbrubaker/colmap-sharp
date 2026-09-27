// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_backward_messages.wgsl: the backward_messages kernel of GPU PatchMatch - the first
// step of a sweep. Per column and source image it runs up the column computing the backward
// messages into the selection probability planes, and initializes the column's state for the
// first row. One invocation per (source image, column) of the current rotation's frame; element
// count width S (width = pm_frame_width(rotation); the host dispatches for C = max(width, height)
// columns, so the guard uses the actual width), element e is source e / width at column e % width.
// Mirrors: PatchMatchCpu.BackwardMessages in ColmapSharp/Mvs/PatchMatchCpu.Sweep.cs.
// Ports: the backward pass of patch_match_cuda.cu's SweepFromTopToBottom, as the CPU port models it.
// Parts: dispatch, common, geometry, likelihood, layout (ColmapSharp/Mvs/PatchMatchGpuKernels.cs).
// Bindings: group 0 - 1 pm_poses (geometry's helpers name it), 2 pm_problem, 6 pm_state
//   (read_write: the depth/normal planes are read, the column states written), 7 pm_costs (read),
//   8 pm_sel_probs (read_write); group 1 - 0 pm_sweep.

@group(0) @binding(1) var<uniform> pm_poses: PmPoseTable;
@group(0) @binding(2) var<uniform> pm_problem: PmProblem;
@group(0) @binding(6) var<storage, read_write> pm_state: array<f32>;
@group(0) @binding(7) var<storage, read> pm_costs: array<f32>;
@group(0) @binding(8) var<storage, read_write> pm_sel_probs: array<f32>;
@group(1) @binding(0) var<uniform> pm_sweep: PmSweep;

@compute @workgroup_size(64)
fn backward_messages(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let width = pm_frame_width(pm_sweep.rotation);
	let height = pm_frame_height(pm_sweep.rotation);
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= u32(width) * u32(PM_NUM_SRC_IMAGES)) {
		return;
	}

	let image_idx = i32(index / u32(width));
	let col = i32(index % u32(width));
	let state = PM_COLUMN_STATES_OFFSET + col * PM_COLUMN_STATE_SIZE;

	// Compute backward message for all rows. Note that the backward messages are temporarily
	// stored in the sel_prob_map and replaced row by row as the updated forward messages are
	// computed by the sweep.
	var beta = PM_UNIFORM_PROB;
	for (var row = height - 1i; row >= 0i; row = row - 1i) {
		let idx = image_idx * PM_PLANE_SIZE + row * width + col;
		beta = pm_compute_backward_message(pm_problem.lk, pm_costs[idx], beta);
		pm_sel_probs[idx] = beta;
	}

	// Initialize forward message.
	pm_state[state + image_idx] = PM_UNIFORM_PROB;

	// Parameters for first row in column (written once per column, by its first source).
	if (image_idx == 0i) {
		pm_state[state + PM_NUM_SRC_IMAGES] = pm_state[col];
		pm_state[state + PM_NUM_SRC_IMAGES + 1i] = pm_state[PM_PLANE_SIZE + col];
		pm_state[state + PM_NUM_SRC_IMAGES + 2i] = pm_state[2i * PM_PLANE_SIZE + col];
		pm_state[state + PM_NUM_SRC_IMAGES + 3i] = pm_state[3i * PM_PLANE_SIZE + col];
	}
}
