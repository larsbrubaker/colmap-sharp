// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_layout.wgsl: the buffer layouts, uniform structs and index math the GPU PatchMatch
// kernels share - the problem, sweep and band uniforms, the size of each rotated frame, the map
// from a rotated pixel back to the original reference image (which keys the random streams), and
// where each plane lives in the state buffer.
// Mirrors: ColmapSharp/Mvs/PatchMatchCpu.cs and PatchMatchCpu.Sweep.cs (the maps, SweepOptions,
//   ColumnStateSize) and PatchMatchTransforms.ToOriginalPixel in
//   ColmapSharp/Mvs/PatchMatchTransforms.cs; ColmapSharp/Mvs/PatchMatchGpuKernels.cs writes the
//   constants and states the uniform sizes (the host packs the uniforms at the offsets below).
// Ports: nothing from COLMAP directly; the layouts replace patch_match_cuda.cu's GpuMat members.
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after patch_match_likelihood.wgsl (PmLikelihood).
// A kernel that includes it supplies, in its constants header,
//   PM_REF_WIDTH: i32, PM_REF_HEIGHT: i32   the unrotated reference image
// and PM_NUM_SRC_IMAGES as geometry does.
//
// Buffers (f32 unless said otherwise; P = PM_REF_WIDTH * PM_REF_HEIGHT; every plane is stored in
// the layout of the current rotation, row-major with the rotated frame's width):
//   reference  [image | sum | squared sum], 3 P: the rotated PatchMatchRefImage, the image plane
//              holding byte / 255.0f (the exact Texel value), the others its SumImage and
//              SquaredSumImage.
//   state      [depth P | normal x P | normal y P | normal z P | column states], the column state
//              of column c at 4 P + c (S + 4): S forward messages, the previous depth and the
//              previous normal (PatchMatchCpu.ColumnStateSize), for C = max(width, height) columns:
//              16 P + 4 C (S + 4) bytes.
//   cost, sel, prev   S planes each (costMap, selProbMap, prevSelProbMap); after the last sweep
//              filter_pixels writes the u32 consistency mask (S planes) into the cost buffer.

// ---- Uniforms ----

// The per-problem values, computed in C# exactly as PatchMatchCpu computes them (64 bytes):
//   offset  0  lk                                PatchMatchLikelihood's four floats
//   offset 16  seed                              the C# ulong seed as (low word, high word)
//   offset 24  depth_min, 28 depth_max           (float)options.DepthMin / DepthMax
//   offset 32  geom_consistency_regularizer      (float)options.GeomConsistencyRegularizer
//   offset 36  geom_consistency_max_cost         (float)options.GeomConsistencyMaxCost
//   offset 40  filter_min_ncc_prob               likelihood.ComputeNCCProb(1.0f - filterMinNcc)
//   offset 44  filter_cos_min_triangulation_angle  MathF.Cos(filterMinTriangulationAngle)
//   offset 48  filter_geom_consistency_max_cost  (float)options.FilterGeomConsistencyMaxCost
//   offset 52  filter_min_num_consistent         options.FilterMinNumConsistent (i32)
//   offset 56  spatial_normalization             BilateralWeightComputer's 1 / (2 sigma_spatial²)
//   offset 60  color_normalization               BilateralWeightComputer's 1 / (2 sigma_color²)
struct PmProblem {
	lk: PmLikelihood,
	seed: vec2<u32>,
	depth_min: f32,
	depth_max: f32,
	geom_consistency_regularizer: f32,
	geom_consistency_max_cost: f32,
	filter_min_ncc_prob: f32,
	filter_cos_min_triangulation_angle: f32,
	filter_geom_consistency_max_cost: f32,
	filter_min_num_consistent: i32,
	spatial_normalization: f32,
	color_normalization: f32,
}

// One sweep's SweepOptions (32 bytes):
//   offset  0  rotation               0..3, the rotation the maps are in during the sweep
//   offset  4  phase                  PatchMatchRandom.SweepPhase(iteration, sweep)
//   offset  8  perturbation           1.0f / MathF.Pow(2.0f, iter + sweep / 4.0f)
//   offset 12  normal_perturbation    (float)(perturbation * Math.PI)
//   offset 16  prev_sel_prob_weight   (iter * 4 + sweep) / (float)(iterations * 4)
//   offset 20  filter_photo_consistency, 24 filter_geom_consistency   0 or 1
//   offset 28  padding
struct PmSweep {
	rotation: i32,
	phase: i32,
	perturbation: f32,
	normal_perturbation: f32,
	prev_sel_prob_weight: f32,
	filter_photo_consistency: u32,
	filter_geom_consistency: u32,
	pad0: u32,
}

// One band of rows [row_start, row_end) of a sweep (16 bytes).
struct PmBand {
	row_start: i32,
	row_end: i32,
	pad0: i32,
	pad1: i32,
}

// ---- Frames and indices ----

// FLT_EPSILON (PatchMatchCpu.FloatEpsilon).
const PM_FLOAT_EPSILON: f32 = 1.1920929e-07f;

// Probability for boundary pixels (PatchMatchCpu.UniformProb).
const PM_UNIFORM_PROB: f32 = 0.5f;

const PM_PLANE_SIZE: i32 = PM_REF_WIDTH * PM_REF_HEIGHT;

// Floats per column state (PatchMatchCpu.ColumnStateSize).
const PM_COLUMN_STATE_SIZE: i32 = PM_NUM_SRC_IMAGES + 4i;

// Where the column states start in the state buffer.
const PM_COLUMN_STATES_OFFSET: i32 = 4i * PM_PLANE_SIZE;

fn pm_frame_width(rotation: i32) -> i32 {
	return select(PM_REF_HEIGHT, PM_REF_WIDTH, (rotation & 1i) == 0i);
}

fn pm_frame_height(rotation: i32) -> i32 {
	return select(PM_REF_WIDTH, PM_REF_HEIGHT, (rotation & 1i) == 0i);
}

// The pixel (row, col) of the original reference image that pixel (row, col) of the frame rotated
// `rotation` times came from (PatchMatchTransforms.ToOriginalPixel): Mat.Rotate maps (y, x) of a
// frame of width w to (w - 1 - x, y); undo it once per rotation.
fn pm_to_original_pixel(rotation: i32, row0: i32, col0: i32) -> vec2<i32> {
	var row = row0;
	var col = col0;
	for (var k = rotation; k > 0i; k = k - 1i) {
		// The frame before rotation k is PM_REF_WIDTH wide when k - 1 is even.
		let prev_width = select(PM_REF_HEIGHT, PM_REF_WIDTH, ((k - 1i) & 1i) == 0i);
		let new_row = col;
		col = prev_width - 1i - row;
		row = new_row;
	}

	return vec2<i32>(row, col);
}
