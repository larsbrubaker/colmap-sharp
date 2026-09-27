// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_rotate.wgsl: the rotate_planes and rotate_normals kernels of GPU PatchMatch - the
// 90-degree counter-clockwise rotation of the maps between sweeps, as copies from one buffer into
// another. rotate_planes moves any number of 32-bit planes (floats or the u32 mask, copied as
// bits); rotate_normals moves the three normal planes and rotates each normal about the z-axis,
// (x, y, z) to (y, -x, z). One invocation per source element; rotate_planes' element count is
// num_planes P (the host dispatches for max(3, S) P), rotate_normals' is P; element e is plane
// e / P, source row (e % P) / src_width, source column (e % P) % src_width.
// Mirrors: Mat.Rotate (MoveTiled with mirrorColumns) in ColmapSharp/Mvs/Mat.Transforms.cs, and
//   PatchMatchKernel.RotateNormalMap in ColmapSharp/Mvs/PatchMatchKernel.Geometry.cs followed by
//   Mat.Rotate, as PatchMatchCpu.Rotate applies them. Pure bit moves: Tier A.
// Ports: gpu_mat.h's Rotate (cuda_rotate.h's CudaRotateKernel) and patch_match_cuda.cu's
//   RotateNormalMap kernel, as the CPU port models them.
// Parts: dispatch (ColmapSharp/Mvs/PatchMatchGpuKernels.cs).
// Bindings: group 0 - 0 pm_rotate_src (read), 1 pm_rotate_dst (read_write); group 1 - 0 pm_rotate.
//
// The planes start at word offsets given in the uniform rather than at bind-group offsets, which
// must be aligned to 256 bytes that plane sizes need not be.

// One rotation copy (32 bytes):
//   offset  0  src_width, 4 src_height   the source frame (the destination is src_height wide)
//   offset  8  num_planes                planes to move (rotate_normals: 3, and ignores it)
//   offset 12  src_offset, 16 dst_offset word offset of the first plane in each buffer
//   offset 20  padding (3 words)
struct PmRotate {
	src_width: u32,
	src_height: u32,
	num_planes: u32,
	src_offset: u32,
	dst_offset: u32,
	pad0: u32,
	pad1: u32,
	pad2: u32,
}

@group(0) @binding(0) var<storage, read> pm_rotate_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> pm_rotate_dst: array<u32>;
@group(1) @binding(0) var<uniform> pm_rotate: PmRotate;

// Mat.Rotate's move for element (row, col) of a src_width x src_height plane: the output is
// src_height wide and takes it at row src_width - 1 - col, column row. Returns the offset in the
// destination plane.
fn pm_rotated_offset(row: u32, col: u32) -> u32 {
	return (pm_rotate.src_width - 1u - col) * pm_rotate.src_height + row;
}

@compute @workgroup_size(64)
fn rotate_planes(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let plane_size = pm_rotate.src_width * pm_rotate.src_height;
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= pm_rotate.num_planes * plane_size) {
		return;
	}

	let plane = index / plane_size;
	let offset = index % plane_size;
	let row = offset / pm_rotate.src_width;
	let col = offset % pm_rotate.src_width;
	let plane_start = plane * plane_size;
	pm_rotate_dst[pm_rotate.dst_offset + plane_start + pm_rotated_offset(row, col)] = pm_rotate_src[pm_rotate.src_offset + index];
}

@compute @workgroup_size(64)
fn rotate_normals(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let plane_size = pm_rotate.src_width * pm_rotate.src_height;
	let index = pm_linear_index(workgroup_id, local_index);
	if (index >= plane_size) {
		return;
	}

	let row = index / pm_rotate.src_width;
	let col = index % pm_rotate.src_width;
	let src = pm_rotate.src_offset + index;
	let dst = pm_rotate.dst_offset + pm_rotated_offset(row, col);

	// RotateNormalMap: (x, y, z) to (y, -x, z). The negation flips the sign bit, exactly as C#'s
	// unary minus does for every float (zeros and NaNs included).
	let x = pm_rotate_src[src];
	pm_rotate_dst[dst] = pm_rotate_src[src + plane_size];
	pm_rotate_dst[dst + plane_size] = x ^ 0x80000000u;
	pm_rotate_dst[dst + 2u * plane_size] = pm_rotate_src[src + 2u * plane_size];
}
