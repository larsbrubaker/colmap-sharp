// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_dispatch.wgsl: the folded dispatch index every GPU PatchMatch kernel uses. A kernel
// has more elements than one dispatch dimension holds (65535 workgroups by default), so the host
// dispatches (x, y) = (min(groups, max per dimension), ceil(groups / x)) workgroups of 64 threads,
// and element (y PM_GROUPS_X + x) 64 + local runs in workgroup (x, y). The index is u32
// throughout: with raised limits it can exceed i32. Each kernel returns when the index is at or
// past its element count (the last row of workgroups overshoots).
// Mirrors: the dispatch shapes of ColmapSharp/Mvs/PatchMatchGpuPlan.cs; PM_GROUPS_X comes from
//   ColmapSharp/Mvs/PatchMatchGpuKernels.cs.
// Ports: nothing from COLMAP; CUDA launches its grids in two dimensions natively.
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs. A kernel that includes it supplies
//   PM_GROUPS_X: u32   the x size of its folded dispatch
// in its constants header, and declares @workgroup_size(64) (PM_WORKGROUP_SIZE).

// Threads per workgroup of every kernel.
const PM_WORKGROUP_SIZE: u32 = 64u;

// The invocation's element index under the folded dispatch.
fn pm_linear_index(workgroup_id: vec3<u32>, local_index: u32) -> u32 {
	return (workgroup_id.y * PM_GROUPS_X + workgroup_id.x) * PM_WORKGROUP_SIZE + local_index;
}
