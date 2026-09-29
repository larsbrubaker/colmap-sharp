// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_likelihood.wgsl: the selection-probability model of GPU PatchMatch - the
// hidden-Markov model over "source image sees this pixel" of Schönberger et al., "Pixelwise View
// Selection for Unstructured Multi-View Stereo" (ECCV 2016): forward/backward messages along a
// column, the selection probability, and the NCC, triangulation-angle, incident-angle and
// resolution priors, plus FindMinCost and TransformPDFToCDF.
// Mirrors: ColmapSharp/Mvs/PatchMatchLikelihood.cs (PatchMatchLikelihood and
//   PatchMatchKernel.FindMinCost / TransformPDFToCDF) in the same operation order.
// Ports: patch_match_cuda.cu's LikelihoodComputer, FindMinCost and TransformPDFToCDF, as the CPU
//   port models them (divergence 96).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after patch_match_common.wgsl (cuda_min,
// cuda_max, is_nan_f32) and patch_match_geometry.wgsl (pm_div, pm_mat33_dot_vec3_homogeneous).
// It declares no bindings. A kernel that includes it supplies PM_NUM_SRC_IMAGES: i32 (as
// geometry does) and a PmLikelihood value, typically a field of its problem uniform.
//
// The model's four floats are computed on the CPU, because C# derives them through MathF.Cos
// and a double-precision erf (see PmLikelihood). Guards against non-finite values follow
// patch_match_geometry.wgsl's rules and are marked "Guard:". exp is WGSL's builtin (Tier C).

// PatchMatchLikelihood's fields, exactly as its constructor computes them in C# (the packer
// copies them bit for bit; 16 bytes, so it can sit in a uniform):
//   cos_min_triangulation_angle      MathF.Cos(minTriangulationAngle)
//   inv_incident_angle_sigma_square  -0.5f / (incidentAngleSigma * incidentAngleSigma)
//   inv_ncc_sigma_square             -0.5f / (nccSigma * nccSigma)
//   ncc_norm_factor                  PatchMatchLikelihood.ComputeNCCCostNormFactor(nccSigma)
struct PmLikelihood {
	cos_min_triangulation_angle: f32,
	inv_incident_angle_sigma_square: f32,
	inv_ncc_sigma_square: f32,
	ncc_norm_factor: f32,
}

// The costs FindMinCost compares: the sweep's five hypotheses.
const PM_NUM_COSTS: i32 = 5i;

// ---- Messages and selection probability ----

// Compute the forward or backward message (ComputeMessage).
fn pm_compute_message(lk: PmLikelihood, forward: bool, cost: f32, prev: f32) -> f32 {
	// Written with the f suffix so they are f32 constants: an abstract-float `1.0 - 0.99999`
	// would be evaluated in double first. (The subtraction is exact either way.)
	const UNIFORM_PROB: f32 = 0.5f;
	const NO_CHANGE_PROB: f32 = 0.99999f;
	const CHANGE_PROB: f32 = 1.0f - NO_CHANGE_PROB;
	let emission = pm_compute_ncc_prob(lk, cost);

	var zn0: f32; // Message for selection probability = 0.
	var zn1: f32; // Message for selection probability = 1.
	if (forward) {
		zn0 = (prev * CHANGE_PROB + (1.0f - prev) * NO_CHANGE_PROB) * UNIFORM_PROB;
		zn1 = (prev * NO_CHANGE_PROB + (1.0f - prev) * CHANGE_PROB) * emission;
	} else {
		zn0 = prev * emission * CHANGE_PROB + (1.0f - prev) * UNIFORM_PROB * NO_CHANGE_PROB;
		zn1 = prev * emission * NO_CHANGE_PROB + (1.0f - prev) * UNIFORM_PROB * CHANGE_PROB;
	}

	// Guard: with prev = 1 and an emission that underflowed to 0, the backward message is
	// 0 / 0, NaN in C#.
	return pm_div(zn1, zn0 + zn1);
}

// Compute forward message from current cost and forward message of previous / neighboring
// pixel (ComputeForwardMessage).
fn pm_compute_forward_message(lk: PmLikelihood, cost: f32, prev: f32) -> f32 {
	return pm_compute_message(lk, true, cost, prev);
}

// Compute backward message from current cost and backward message of previous / neighboring
// pixel (ComputeBackwardMessage).
fn pm_compute_backward_message(lk: PmLikelihood, cost: f32, prev: f32) -> f32 {
	return pm_compute_message(lk, false, cost, prev);
}

// The selection probability from the forward and backward messages, blended with the previous
// sweep's probability by prev_weight (ComputeSelProb).
fn pm_compute_sel_prob(alpha: f32, beta: f32, prev: f32, prev_weight: f32) -> f32 {
	let zn0 = (1.0f - alpha) * (1.0f - beta);
	let zn1 = alpha * beta;
	// Guard: alpha = 1 and beta = 0 (or the reverse) make this 0 / 0, NaN in C#.
	let curr = pm_div(zn1, zn0 + zn1);
	return prev_weight * prev + (1.0f - prev_weight) * curr;
}

// ---- Priors ----

// Compute NCC probability. Note that cost = 1 - NCC (ComputeNCCProb).
fn pm_compute_ncc_prob(lk: PmLikelihood, cost: f32) -> f32 {
	return exp(cost * cost * lk.inv_ncc_sigma_square) * lk.ncc_norm_factor;
}

// The triangulation angle prior: 1 at or above the minimum angle, falling to 0 at a zero angle
// (ComputeTriProb).
fn pm_compute_tri_prob(lk: PmLikelihood, cos_triangulation_angle: f32) -> f32 {
	// Guard: a NaN cosine (from a point at depth 0) fails C#'s `>` and gives 1.
	if (!is_nan_f32(cos_triangulation_angle) && cos_triangulation_angle > lk.cos_min_triangulation_angle) {
		// Guard: a zero minimum angle makes the divisor 0 (and C#'s quotient infinite).
		let scaled = 1.0f - pm_div(1.0f - cos_triangulation_angle, 1.0f - lk.cos_min_triangulation_angle);
		let likelihood = 1.0f - scaled * scaled;
		return cuda_min(1.0f, cuda_max(0.0f, likelihood));
	}

	return 1.0f;
}

// The incident angle prior: 1 head-on, decreasing towards grazing views (ComputeIncProb).
fn pm_compute_inc_prob(lk: PmLikelihood, cos_incident_angle: f32) -> f32 {
	let x = 1.0f - cuda_max(0.0f, cos_incident_angle);
	return exp(x * x * lk.inv_incident_angle_sigma_square);
}

// The warping/resolution prior: the ratio (at most 1) of the areas of the
// (2 window_radius + 1)² reference patch around (row, col) and its image under h
// (ComputeResolutionProb).
fn pm_compute_resolution_prob(h: array<f32, 9>, row: f32, col: f32, window_radius: i32) -> f32 {
	let window_size = 2i * window_radius + 1i;

	// Warp corners of patch in reference image to source image.
	let radius = f32(window_radius);
	let src1 = pm_mat33_dot_vec3_homogeneous(h, col - radius, row - radius);
	let src2 = pm_mat33_dot_vec3_homogeneous(h, col - radius, row + radius);
	let src3 = pm_mat33_dot_vec3_homogeneous(h, col + radius, row + radius);
	let src4 = pm_mat33_dot_vec3_homogeneous(h, col + radius, row - radius);

	// Compute area of patches in reference and source image.
	let ref_area = f32(window_size * window_size);
	let src_area = abs(0.5f * (src1.x * src2.y - src2.x * src1.y - src1.x * src4.y + src2.x * src3.y
		- src3.x * src2.y + src4.x * src1.y + src3.x * src4.y - src4.x * src3.y));

	// Guard: a corner mapped to infinity (a zero z in the homography) makes src_area NaN or
	// +inf. C# then takes the false branch of `>`: ref_area / NaN is NaN, ref_area / inf is 0.
	if (is_nan_f32(src_area)) {
		return src_area;
	}

	if (bitcast<u32>(src_area) == 0x7F800000u) {
		return 0.0f;
	}

	// ref_area >= 1, so a zero src_area takes the first branch and neither quotient divides
	// by zero.
	if (ref_area > src_area) {
		return src_area / ref_area;
	}

	return ref_area / src_area;
}

// ---- Small array helpers ----

// Index of the minimum cost; the last one among equal minima (FindMinCost).
fn pm_find_min_cost(costs: array<f32, PM_NUM_COSTS>) -> i32 {
	var min_cost = costs[0];
	var min_cost_idx = 0i;
	for (var idx = 1i; idx < PM_NUM_COSTS; idx = idx + 1i) {
		// Guard: C#'s `<=` is false when either side is NaN.
		let cost = costs[idx];
		if (!is_nan_f32(cost) && !is_nan_f32(min_cost) && cost <= min_cost) {
			min_cost = cost;
			min_cost_idx = idx;
		}
	}

	return min_cost_idx;
}

// Normalizes probs and replaces it with its cumulative sums (TransformPDFToCDF).
fn pm_transform_pdf_to_cdf(probs: ptr<function, array<f32, PM_NUM_SRC_IMAGES>>) {
	var prob_sum = 0.0f;
	for (var i = 0i; i < PM_NUM_SRC_IMAGES; i = i + 1i) {
		prob_sum += (*probs)[i];
	}

	// Guard: the probabilities are products of non-negative priors, so a zero sum means every
	// one is zero; C# then multiplies each by an infinite inverse and gets NaN everywhere
	// (which no sampling threshold exceeds). Written out, since WGSL leaves 1 / 0 open.
	if ((bitcast<u32>(prob_sum) & 0x7FFFFFFFu) == 0u) {
		for (var i = 0i; i < PM_NUM_SRC_IMAGES; i = i + 1i) {
			(*probs)[i] = pm_quiet_nan();
		}

		return;
	}

	let inv_prob_sum = 1.0f / prob_sum;

	var cum_prob = 0.0f;
	for (var i = 0i; i < PM_NUM_SRC_IMAGES; i = i + 1i) {
		let prob = (*probs)[i] * inv_prob_sum;
		cum_prob += prob;
		(*probs)[i] = cum_prob;
	}
}
