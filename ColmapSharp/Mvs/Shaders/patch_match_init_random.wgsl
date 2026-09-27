// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_init_random.wgsl: the init_random kernel of GPU PatchMatch - a uniform random depth
// in (depth_min, depth_max] and a uniform camera-facing random normal for every pixel, the start
// of a photometric run. One invocation per pixel of the unrotated reference image; element count
// P = PM_REF_WIDTH * PM_REF_HEIGHT.
// Mirrors: PatchMatchCpu.InitRandomDepthAndNormalMaps in ColmapSharp/Mvs/PatchMatchCpu.cs (the
//   same streams: PatchMatchRandom.InitDepthPhase and InitNormalPhase keyed on the pixel).
// Ports: patch_match_cuda.cu's FillWithRandomNumbers and InitNormalMap, as the CPU port models
//   them (docs/CPP_DIVERGENCES.md, entry 86).
// Parts: dispatch, common, geometry, likelihood, layout (ColmapSharp/Mvs/PatchMatchGpuKernels.cs).
// Bindings: group 0 - 1 pm_poses, 2 pm_problem, 6 pm_state (read_write: the depth and normal planes).

@group(0) @binding(1) var<uniform> pm_poses: PmPoseTable;
@group(0) @binding(2) var<uniform> pm_problem: PmProblem;
@group(0) @binding(6) var<storage, read_write> pm_state: array<f32>;

// PatchMatchRandom.InitDepthPhase and InitNormalPhase.
const PM_INIT_DEPTH_PHASE: i32 = -2i;
const PM_INIT_NORMAL_PHASE: i32 = -1i;

@compute @workgroup_size(64)
fn init_random(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= u32(PM_PLANE_SIZE)) {
		return;
	}

	let pixel = i32(index);
	let row = pixel / PM_REF_WIDTH;
	let col = pixel % PM_REF_WIDTH;
	let frame = pm_frame(0i);

	var depth_random = pm_random_init(pm_problem.seed, row, col, PM_INIT_DEPTH_PHASE);
	pm_state[pixel] = pm_generate_random_depth(pm_problem.depth_min, pm_problem.depth_max, &depth_random);

	var normal_random = pm_random_init(pm_problem.seed, row, col, PM_INIT_NORMAL_PHASE);
	let normal = pm_generate_random_normal(frame, row, col, &normal_random);
	pm_state[PM_PLANE_SIZE + pixel] = normal.x;
	pm_state[2i * PM_PLANE_SIZE + pixel] = normal.y;
	pm_state[3i * PM_PLANE_SIZE + pixel] = normal.z;
}
