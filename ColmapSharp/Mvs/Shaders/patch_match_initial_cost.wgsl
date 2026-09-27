// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_initial_cost.wgsl: the initial_cost kernel of GPU PatchMatch - the
// photo-consistency cost of every pixel's starting depth and normal against every source image,
// in the unrotated frame, before the first sweep. One invocation per (source image, pixel);
// element count P S, element e is source e / P at pixel e % P (the cost buffer's own index).
// Mirrors: PatchMatchKernel.ComputeInitialCost in ColmapSharp/Mvs/PatchMatchKernel.Photometric.cs.
// Ports: patch_match_cuda.cu's ComputeInitialCost, as the CPU port models it
//   (docs/CPP_DIVERGENCES.md, entries 95-97).
// Parts: dispatch, common, textures, geometry, likelihood, layout, ncc (ColmapSharp/Mvs/PatchMatchGpuKernels.cs).
// Bindings: group 0 - 0 pm_byte_to_unit, 1 pm_poses, 2 pm_problem, 3 pm_reference,
//   4 pm_source_images, 5 pm_source_depths (textures declares it; a 4-byte dummy when photometric),
//   6 pm_state (read), 7 pm_costs (read_write).

@group(0) @binding(0) var<uniform> pm_byte_to_unit: PmByteToUnit;
@group(0) @binding(1) var<uniform> pm_poses: PmPoseTable;
@group(0) @binding(2) var<uniform> pm_problem: PmProblem;
@group(0) @binding(3) var<storage, read> pm_reference: array<f32>;
@group(0) @binding(4) var<storage, read> pm_source_images: array<u32>;
@group(0) @binding(5) var<storage, read> pm_source_depths: array<f32>;
@group(0) @binding(6) var<storage, read> pm_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> pm_costs: array<f32>;

@compute @workgroup_size(64)
fn initial_cost(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= u32(PM_PLANE_SIZE) * u32(PM_NUM_SRC_IMAGES)) {
		return;
	}

	let image_idx = i32(index / u32(PM_PLANE_SIZE));
	let pixel = i32(index % u32(PM_PLANE_SIZE));
	let row = pixel / PM_REF_WIDTH;
	let col = pixel % PM_REF_WIDTH;

	// The costs are computed before the first rotation.
	let rotation = 0i;
	let frame = pm_frame(rotation);
	let depth = pm_state[pixel];
	let normal = vec3<f32>(pm_state[PM_PLANE_SIZE + pixel], pm_state[2i * PM_PLANE_SIZE + pixel], pm_state[3i * PM_PLANE_SIZE + pixel]);
	let window = pm_prepare_window(rotation, row, col);
	pm_costs[image_idx * PM_PLANE_SIZE + pixel] = pm_compute_ncc_cost(rotation, frame, row, col, depth, normal, image_idx, window);
}
