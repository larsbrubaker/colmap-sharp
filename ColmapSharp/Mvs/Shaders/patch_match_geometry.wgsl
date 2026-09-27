// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_geometry.wgsl: the geometry helpers of GPU PatchMatch - the pose table and
// reference frames, the random and perturbed depth/normal hypotheses, depth propagation along a
// column, viewing angles and the plane-induced homography - plus the zero-divisor-safe division
// the GPU helpers share.
// Mirrors: ColmapSharp/Mvs/PatchMatchKernel.Geometry.cs (Mat33DotVec3,
//   Mat33DotVec3Homogeneous, DotProduct3, GenerateRandomDepth, GenerateRandomNormal,
//   PerturbDepth, PerturbNormal, ComputePointAtDepth, PropagateDepth, ComputeViewingAngles,
//   ComposeHomography, DotViewRay) in the same operation order, and the pose table layout of
//   ColmapSharp/Mvs/PatchMatchTransforms.cs. ComputeGeomConsistencyCost needs the source depth
//   sampling of patch_match_textures.wgsl and is in patch_match_geom_cost.wgsl; RotateNormalMap
//   is the rotate_normals kernel of patch_match_rotate.wgsl.
// Ports: patch_match_cuda.cu's __device__ geometry helpers, as the CPU port models them
//   (docs/CPP_DIVERGENCES.md, entries 86 and 96).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after patch_match_common.wgsl (it uses the
// RNG and the float bit tests). A kernel that includes this file supplies, in its constants
// header,
//   PM_NUM_SRC_IMAGES: i32   (PatchMatchTransforms.NumSrcImages)
// and declares, at any group and binding it likes,
//   var<uniform> pm_poses: PmPoseTable;
//
// Pose table contract (the C# packer writes exactly this; one uniform for all four rotations).
// The buffer is a flat float[] read four floats per vec4, float i at values[i / 4][i % 4]:
//   floats [8 r, 8 r + 4)        rotation r's ref_K     {fx, cx, fy, cy}          r = 0..3
//   floats [8 r + 4, 8 r + 8)    rotation r's ref_inv_K {1/fx, -cx/fx, 1/fy, -cy/fy}
//                                (PatchMatchTransforms.RefK(r) and RefInvK(r); vec4 2r and 2r+1)
//   floats [32 + 43 (r S + s), 32 + 43 (r S + s) + 43)
//                                source s's row of rotation r's pose table, S = PM_NUM_SRC_IMAGES:
//                                PatchMatchTransforms.Poses(r)[43 s .. 43 s + 43] verbatim, i.e.
//                                K at +0 (4), R at +4 (9, row-major), T at +13 (3), C at +16 (3),
//                                P at +19 (12, row-major 3x4), inverse P at +31 (12)
// That is 32 + 172 S floats = 8 + 43 S vec4s (16 (8 + 43 S) bytes), with no padding: 172 S is a
// multiple of four. A uniform binding is limited to 64 KiB by default, so 16 (8 + 43 S) bytes caps S
// at 95 source images (PatchMatchGpuPlan enforces it). The PM_POSE_* offsets below must equal
// PatchMatchTransforms' *Offset constants (PatchMatchShaderGeometryTests pins them).
//
// Non-finite values. C# relies on IEEE infinities and NaNs in a few places; WGSL lets a compiler
// assume they never occur and makes a division by zero indeterminate. So every division whose
// divisor can be zero goes through pm_div, which returns the IEEE result for a zero divisor from
// bits, and every comparison C# makes on a value that can be NaN tests for NaN by its bits first
// (the guards are marked "Guard:"). Arithmetic on an infinity or NaN that reached a runtime value
// is left to the hardware, which is IEEE on every GPU this targets.
//
// sin and cos in pm_perturb_normal are WGSL builtins, not MathF's: results are Tier C there.

// ---- Division and non-finite values ----

// A quiet NaN. Which NaN C# produces (for 0 / 0) depends on the CPU; no caller looks at the bits.
const PM_NAN_BITS: u32 = 0x7FC00000u;

// a / b with IEEE-754's result for a zero divisor, which WGSL leaves indeterminate: NaN for
// 0 / 0 or NaN / 0, otherwise an infinity whose sign is the XOR of the operands' signs.
fn pm_div(a: f32, b: f32) -> f32 {
	let b_bits = bitcast<u32>(b);
	if ((b_bits & 0x7FFFFFFFu) != 0u) {
		return a / b;
	}

	let a_bits = bitcast<u32>(a);
	if (is_nan_f32(a)) {
		return a;
	}

	if ((a_bits & 0x7FFFFFFFu) == 0u) {
		return bitcast<f32>(PM_NAN_BITS);
	}

	return bitcast<f32>(((a_bits ^ b_bits) & 0x80000000u) | 0x7F800000u);
}

// ---- Pose table and reference frames ----

// Offsets in a pose row: PatchMatchTransforms.KOffset .. InvPOffset and NumTformParams.
const PM_POSE_K_OFFSET: i32 = 0i;
const PM_POSE_R_OFFSET: i32 = 4i;
const PM_POSE_T_OFFSET: i32 = 13i;
const PM_POSE_C_OFFSET: i32 = 16i;
const PM_POSE_P_OFFSET: i32 = 19i;
const PM_POSE_INV_P_OFFSET: i32 = 31i;
const PM_NUM_TFORM_PARAMS: i32 = 43i;

// The float offset of the pose rows (after the four frames).
const PM_POSE_ROWS_OFFSET: i32 = 32i;

const PM_POSE_TABLE_VEC4S: i32 = 8i + PM_NUM_TFORM_PARAMS * PM_NUM_SRC_IMAGES;

// The frames and pose tables of all four rotations, laid out as the header says.
struct PmPoseTable {
	values: array<vec4<f32>, PM_POSE_TABLE_VEC4S>,
}

// Float `index` of the pose table.
fn pm_pose_float(index: i32) -> f32 {
	return pm_poses.values[index >> 2u][index & 3i];
}

// The float offset of source `image_idx`'s row in rotation `rotation`'s pose table; the pose
// row is poses.Slice(imageIdx * NumTformParams, NumTformParams) of C#'s Poses(rotation).
fn pm_pose_row(rotation: i32, image_idx: i32) -> i32 {
	return PM_POSE_ROWS_OFFSET + (rotation * PM_NUM_SRC_IMAGES + image_idx) * PM_NUM_TFORM_PARAMS;
}

// PatchMatchFrame: ref_K {fx, cx, fy, cy} as k and ref_inv_K {1/fx, -cx/fx, 1/fy, -cy/fy} as
// inv_k, so frame.K0 is k.x and frame.InvK3 is inv_k.w.
struct PmFrame {
	k: vec4<f32>,
	inv_k: vec4<f32>,
}

// The frame of rotation `rotation` (new PatchMatchFrame(transforms, rotation)).
fn pm_frame(rotation: i32) -> PmFrame {
	var frame: PmFrame;
	frame.k = pm_poses.values[2i * rotation];
	frame.inv_k = pm_poses.values[2i * rotation + 1i];
	return frame;
}

// ---- Small vector helpers ----

// result = mat * vec for a row-major 3x3 matrix (Mat33DotVec3).
fn pm_mat33_dot_vec3(mat: array<f32, 9>, vec: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z,
		mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z,
		mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z);
}

// The dehomogenized mat * (x, y, 1) (Mat33DotVec3Homogeneous).
fn pm_mat33_dot_vec3_homogeneous(mat: array<f32, 9>, x: f32, y: f32) -> vec2<f32> {
	// Guard: a zero z gives C# an infinite inv_z.
	let inv_z = pm_div(1.0f, mat[6] * x + mat[7] * y + mat[8]);
	return vec2<f32>(inv_z * (mat[0] * x + mat[1] * y + mat[2]), inv_z * (mat[3] * x + mat[4] * y + mat[5]));
}

// vec1 · vec2 in C#'s order (DotProduct3). WGSL's dot() leaves the order to the implementation.
fn pm_dot3(vec1: vec3<f32>, vec2: vec3<f32>) -> f32 {
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

// normal · (ref_inv_K ray of pixel (row, col)) (DotViewRay).
fn pm_dot_view_ray(frame: PmFrame, row: i32, col: i32, normal: vec3<f32>) -> f32 {
	let ray_x = frame.inv_k.x * f32(col) + frame.inv_k.y;
	let ray_y = frame.inv_k.z * f32(row) + frame.inv_k.w;
	return normal.x * ray_x + normal.y * ray_y + normal.z * 1.0f;
}

// ---- Random and perturbed hypotheses ----

// A uniform depth in (depth_min, depth_max] (GenerateRandomDepth).
fn pm_generate_random_depth(depth_min: f32, depth_max: f32, random: ptr<function, PatchMatchRandom>) -> f32 {
	return pm_random_next_uniform(random) * (depth_max - depth_min) + depth_min;
}

// A uniformly distributed unit normal facing the camera at pixel (row, col)
// (GenerateRandomNormal). Unbiased sampling of normal, according to George Marsaglia,
// "Choosing a Point from the Surface of a Sphere", 1972.
fn pm_generate_random_normal(frame: PmFrame, row: i32, col: i32, random: ptr<function, PatchMatchRandom>) -> vec3<f32> {
	var v1 = 0.0f;
	var v2 = 0.0f;
	var s = 2.0f;
	// Every draw is finite and in (0, 1], so s is finite and the loop needs no NaN guard.
	loop {
		if (!(s >= 1.0f)) {
			break;
		}

		v1 = 2.0f * pm_random_next_uniform(random) - 1.0f;
		v2 = 2.0f * pm_random_next_uniform(random) - 1.0f;
		s = v1 * v1 + v2 * v2;
	}

	// s < 1, so the square root's argument is positive.
	let s_norm = sqrt(1.0f - s);
	var normal = vec3<f32>(2.0f * v1 * s_norm, 2.0f * v2 * s_norm, 1.0f - 2.0f * s);

	// Make sure normal is looking away from camera. (The normal is finite, so is the dot.)
	if (pm_dot_view_ray(frame, row, col, normal) > 0.0f) {
		normal = vec3<f32>(-normal.x, -normal.y, -normal.z);
	}

	return normal;
}

// A uniform depth within ±perturbation (relative) of `depth` (PerturbDepth).
fn pm_perturb_depth(perturbation: f32, depth: f32, random: ptr<function, PatchMatchRandom>) -> f32 {
	let depth_min = (1.0f - perturbation) * depth;
	let depth_max = (1.0f + perturbation) * depth;
	return pm_generate_random_depth(depth_min, depth_max, random);
}

// `normal` rotated by random angles in ±perturbation/2 about each axis; if the result faces
// away from the camera, retries with half the perturbation, up to three times, and then keeps
// the input normal (PerturbNormal). C# recurses with numTrials + 1; WGSL has no recursion, so
// the trials are a loop that draws, halves and gives up in the same order.
fn pm_perturb_normal(frame: PmFrame, row: i32, col: i32, perturbation0: f32, normal: vec3<f32>, random: ptr<function, PatchMatchRandom>) -> vec3<f32> {
	const MAX_NUM_TRIALS: i32 = 3i;
	var perturbation = perturbation0;
	var num_trials = 0i;
	// WGSL wants a return statement after the loop, so each outcome sets this and breaks.
	var perturbed: vec3<f32>;
	loop {
		// Perturbation rotation angles.
		let a1 = (pm_random_next_uniform(random) - 0.5f) * perturbation;
		let a2 = (pm_random_next_uniform(random) - 0.5f) * perturbation;
		let a3 = (pm_random_next_uniform(random) - 0.5f) * perturbation;

		// WGSL builtins (Tier C; MathF.Sin/Cos in C#).
		let sin_a1 = sin(a1);
		let sin_a2 = sin(a2);
		let sin_a3 = sin(a3);
		let cos_a1 = cos(a1);
		let cos_a2 = cos(a2);
		let cos_a3 = cos(a3);

		// R = Rx * Ry * Rz
		var r: array<f32, 9>;
		r[0] = cos_a2 * cos_a3;
		r[1] = -cos_a2 * sin_a3;
		r[2] = sin_a2;
		r[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
		r[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
		r[5] = -cos_a2 * sin_a1;
		r[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
		r[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
		r[8] = cos_a1 * cos_a2;

		// Perturb the normal vector.
		let perturbed_normal = pm_mat33_dot_vec3(r, normal);

		// Make sure the perturbed normal is still looking in the same direction as the viewing
		// direction, otherwise try again but with smaller perturbation.
		// Guard: a NaN normal (from a NaN input) fails C#'s `>= 0` and is normalized below.
		let view_dot = pm_dot_view_ray(frame, row, col, perturbed_normal);
		if (!is_nan_f32(view_dot) && view_dot >= 0.0f) {
			if (num_trials < MAX_NUM_TRIALS) {
				perturbation = 0.5f * perturbation;
				num_trials = num_trials + 1i;
				continue;
			}

			perturbed = normal;
			break;
		}

		// Make sure normal has unit norm. The dot is negative here, so the normal is not zero
		// and the square root is positive (or NaN, which the hardware propagates).
		let inv_norm = 1.0f / sqrt(pm_dot3(perturbed_normal, perturbed_normal));
		perturbed = vec3<f32>(perturbed_normal.x * inv_norm, perturbed_normal.y * inv_norm, perturbed_normal.z * inv_norm);
		break;
	}

	return perturbed;
}

// ---- Points, propagation and viewing angles ----

// The point at `depth` along the viewing ray of pixel (row, col) (ComputePointAtDepth).
fn pm_compute_point_at_depth(frame: PmFrame, row: f32, col: f32, depth: f32) -> vec3<f32> {
	return vec3<f32>(depth * (frame.inv_k.x * col + frame.inv_k.y), depth * (frame.inv_k.z * row + frame.inv_k.w), depth);
}

// Transfer depth on plane from viewing ray at row1 to row2 (PropagateDepth): the intersection
// of the viewing ray through row2 with the plane at row1 defined by the given depth and normal
// (in the column's y-z plane); depth1 when the two are nearly parallel.
fn pm_propagate_depth(frame: PmFrame, depth1: f32, normal1: vec3<f32>, row1: f32, row2: f32) -> f32 {
	// Point along first viewing ray.
	let x1 = depth1 * (frame.inv_k.z * row1 + frame.inv_k.w);
	let y1 = depth1;

	// Point on plane defined by point along first viewing ray and plane normal1.
	let x2 = x1 + normal1.z;
	let y2 = y1 - normal1.y;

	// Point on second viewing ray (whose origin is (0, 0)); its y is 1.
	let x4 = frame.inv_k.z * row2 + frame.inv_k.w;

	// Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
	let denom = x2 - x1 + x4 * (y1 - y2);
	const EPS: f32 = 1e-5f;
	// Guard: a NaN denom fails C#'s `< Eps` and falls through to the (NaN) quotient.
	if (!is_nan_f32(denom) && abs(denom) < EPS) {
		return depth1;
	}

	// |denom| >= Eps here (or NaN), so the divisor is not zero.
	let nom = y1 * x2 - x1 * y2;
	return nom / denom;
}

// The cosines of the triangulation angle between the reference and source image `image_idx`
// at `point`, and of the incident angle between the source's viewing direction and `normal`
// (ComputeViewingAngles), as (cos_triangulation_angle, cos_incident_angle).
fn pm_compute_viewing_angles(rotation: i32, point: vec3<f32>, normal: vec3<f32>, image_idx: i32) -> vec2<f32> {
	// Projection center of source image.
	let c = pm_pose_row(rotation, image_idx) + PM_POSE_C_OFFSET;

	// Ray from point to camera.
	let sx = vec3<f32>(pm_pose_float(c) - point.x, pm_pose_float(c + 1i) - point.y, pm_pose_float(c + 2i) - point.z);

	// Length of ray from reference image to point.
	// Guard: a zero point (depth 0) gives C# an infinite inverse norm.
	let rx_inv_norm = pm_div(1.0f, sqrt(pm_dot3(point, point)));

	// Length of ray from point to source image.
	// Guard: a point at the source's projection center, likewise.
	let sx_inv_norm = pm_div(1.0f, sqrt(pm_dot3(sx, sx)));

	let cos_incident_angle = pm_dot3(sx, normal) * sx_inv_norm;
	let cos_triangulation_angle = -pm_dot3(sx, point) * rx_inv_norm * sx_inv_norm;
	return vec2<f32>(cos_triangulation_angle, cos_incident_angle);
}

// ---- Homography ----

// The homography H = K * (R - T * n' / d) * Kref^-1 from the reference image to source image
// `image_idx` induced by the plane through the point at `depth` on pixel (row, col) with
// `normal`, row-major (ComposeHomography).
fn pm_compose_homography(rotation: i32, frame: PmFrame, image_idx: i32, row: i32, col: i32, depth: f32, normal: vec3<f32>) -> array<f32, 9> {
	let pose = pm_pose_row(rotation, image_idx);

	// Calibration of source image.
	let k0 = pm_pose_float(pose + PM_POSE_K_OFFSET);
	let k1 = pm_pose_float(pose + PM_POSE_K_OFFSET + 1i);
	let k2 = pm_pose_float(pose + PM_POSE_K_OFFSET + 2i);
	let k3 = pm_pose_float(pose + PM_POSE_K_OFFSET + 3i);

	// Relative rotation between reference and source image.
	var r: array<f32, 9>;
	for (var i = 0i; i < 9i; i = i + 1i) {
		r[i] = pm_pose_float(pose + PM_POSE_R_OFFSET + i);
	}

	// Relative translation between reference and source image.
	let t0 = pm_pose_float(pose + PM_POSE_T_OFFSET);
	let t1 = pm_pose_float(pose + PM_POSE_T_OFFSET + 1i);
	let t2 = pm_pose_float(pose + PM_POSE_T_OFFSET + 2i);

	// Distance to the plane.
	let dist = depth * (normal.x * (frame.inv_k.x * f32(col) + frame.inv_k.y) + normal.y * (frame.inv_k.z * f32(row) + frame.inv_k.w) + normal.z);
	// Guard: a plane through the camera centre (or depth 0) gives C# an infinite inv_dist.
	let inv_dist = pm_div(1.0f, dist);

	let inv_dist_n0 = inv_dist * normal.x;
	let inv_dist_n1 = inv_dist * normal.y;
	let inv_dist_n2 = inv_dist * normal.z;

	var h: array<f32, 9>;
	h[0] = frame.inv_k.x * (k0 * (r[0] + inv_dist_n0 * t0) + k1 * (r[6] + inv_dist_n0 * t2));
	h[1] = frame.inv_k.z * (k0 * (r[1] + inv_dist_n1 * t0) + k1 * (r[7] + inv_dist_n1 * t2));
	h[2] = k0 * (r[2] + inv_dist_n2 * t0) + k1 * (r[8] + inv_dist_n2 * t2)
		+ frame.inv_k.y * (k0 * (r[0] + inv_dist_n0 * t0) + k1 * (r[6] + inv_dist_n0 * t2))
		+ frame.inv_k.w * (k0 * (r[1] + inv_dist_n1 * t0) + k1 * (r[7] + inv_dist_n1 * t2));
	h[3] = frame.inv_k.x * (k2 * (r[3] + inv_dist_n0 * t1) + k3 * (r[6] + inv_dist_n0 * t2));
	h[4] = frame.inv_k.z * (k2 * (r[4] + inv_dist_n1 * t1) + k3 * (r[7] + inv_dist_n1 * t2));
	h[5] = k2 * (r[5] + inv_dist_n2 * t1) + k3 * (r[8] + inv_dist_n2 * t2)
		+ frame.inv_k.y * (k2 * (r[3] + inv_dist_n0 * t1) + k3 * (r[6] + inv_dist_n0 * t2))
		+ frame.inv_k.w * (k2 * (r[4] + inv_dist_n1 * t1) + k3 * (r[7] + inv_dist_n1 * t2));
	h[6] = frame.inv_k.x * (r[6] + inv_dist_n0 * t2);
	h[7] = frame.inv_k.z * (r[7] + inv_dist_n1 * t2);
	h[8] = r[8] + frame.inv_k.y * (r[6] + inv_dist_n0 * t2) + frame.inv_k.w * (r[7] + inv_dist_n1 * t2) + inv_dist_n2 * t2;
	return h;
}
