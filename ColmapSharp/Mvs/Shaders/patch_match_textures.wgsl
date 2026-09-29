// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// patch_match_textures.wgsl: sampling of the source images (bilinear) and source depth maps
// (point) of a GPU PatchMatch problem, the layered CUDA textures of patch_match_cuda.cu.
// Mirrors: ColmapSharp/Mvs/PatchMatchTextures.cs (PatchMatchSourceImages.Sample and
//   PatchMatchSourceDepthMaps.Sample), in the same operation order.
// Ports: patch_match_cuda.cu's tex2DLayered reads as the CPU port models them: unnormalized
//   coordinates, texel centres at +0.5, border addressing (0 outside the layer), and exact
//   float bilinear weights (divergence 95).
// Composed by ColmapSharp/Mvs/PatchMatchShaders.cs after patch_match_common.wgsl (it uses
// is_finite_f32). A kernel that includes this file supplies, in its constants header,
//   PM_SRC_MAX_WIDTH: i32, PM_SRC_MAX_HEIGHT: i32   (the layer size, as in C#)
// and declares, at any group and binding it likes,
//   var<uniform> pm_byte_to_unit: PmByteToUnit;           byte / 255.0f for every byte
//   var<storage, read> pm_source_images: array<u32>;      the layers' bytes, four per word,
//                                                         little-endian (C#'s byte[] copied)
//   var<storage, read> pm_source_depths: array<f32>;      the depth layers (WGSL resolves
//                                                         every name, called or not, so a
//                                                         photometric kernel still declares
//                                                         it and binds a 4-byte dummy)
// Layer l, row r, column c is element (l * PM_SRC_MAX_HEIGHT + r) * PM_SRC_MAX_WIDTH + c of
// both, as in PatchMatchSourceImages and PatchMatchSourceDepthMaps.

// PatchMatchSourceImages.ByteToUnit: 256 floats. A uniform array's stride is 16 bytes, so
// they are packed four to a vec4; the buffer bytes are the plain float[256].
struct PmByteToUnit {
	values: array<vec4<f32>, 64>,
}

// byte / 255.0f, looked up.
fn pm_byte_to_unit_value(b: u32) -> f32 {
	return pm_byte_to_unit.values[b >> 2u][b & 3u];
}

// The byte at element `index` of the source image layers.
fn pm_source_byte(index: i32) -> u32 {
	let i = bitcast<u32>(index);
	return (pm_source_images[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

// PatchMatchSourceImages.Texel: texel (col, row) of the layer starting at row
// `layer_offset`, or 0 outside the layer. The unsigned compares reject negatives too, like
// C#'s (uint) casts.
fn pm_source_texel(col: i32, row: i32, layer_offset: i32) -> f32 {
	if (bitcast<u32>(col) >= bitcast<u32>(PM_SRC_MAX_WIDTH) || bitcast<u32>(row) >= bitcast<u32>(PM_SRC_MAX_HEIGHT)) {
		return 0.0;
	}

	return pm_byte_to_unit_value(pm_source_byte((layer_offset + row) * PM_SRC_MAX_WIDTH + col));
}

// PatchMatchSourceImages.Sample: the bilinearly interpolated value in [0, 1] at unnormalized
// coordinates (x, y) of `layer`. C# takes one path when all four texels are inside and
// another at the border; both compute this blend, so one path with per-texel checks is the
// same function.
fn pm_sample_source_image(x: f32, y: f32, layer: i32) -> f32 {
	// C# returns 0 for NaN and infinite coordinates through its range test below (every
	// comparison with NaN is false); WGSL may assume those values never occur, so they are
	// tested by their bits. A finite x stays finite after subtracting 0.5.
	if (!is_finite_f32(x) || !is_finite_f32(y)) {
		return 0.0;
	}

	let px = x - 0.5;
	let py = y - 0.5;
	let fx = floor(px);
	let fy = floor(py);

	// Far outside the layer every texel is border: 0. This also keeps the int conversions
	// below in range.
	if (!(fx >= -1.0 && fx < f32(PM_SRC_MAX_WIDTH) && fy >= -1.0 && fy < f32(PM_SRC_MAX_HEIGHT))) {
		return 0.0;
	}

	let wx = px - fx;
	let wy = py - fy;
	let ix = i32(fx);
	let iy = i32(fy);
	let layer_offset = layer * PM_SRC_MAX_HEIGHT;
	let c00 = pm_source_texel(ix, iy, layer_offset);
	let c10 = pm_source_texel(ix + 1, iy, layer_offset);
	let c01 = pm_source_texel(ix, iy + 1, layer_offset);
	let c11 = pm_source_texel(ix + 1, iy + 1, layer_offset);
	return (c00 * (1.0 - wx) + c10 * wx) * (1.0 - wy) + (c01 * (1.0 - wx) + c11 * wx) * wy;
}

// PatchMatchSourceDepthMaps.Sample: the depth of the texel containing (x, y) of `layer`, or 0
// outside it (point filtering, border addressing).
fn pm_sample_source_depth(x: f32, y: f32, layer: i32) -> f32 {
	// Non-finite coordinates fail C#'s range test; see pm_sample_source_image.
	if (!is_finite_f32(x) || !is_finite_f32(y)) {
		return 0.0;
	}

	let fx = floor(x);
	let fy = floor(y);
	if (!(fx >= 0.0 && fx < f32(PM_SRC_MAX_WIDTH) && fy >= 0.0 && fy < f32(PM_SRC_MAX_HEIGHT))) {
		return 0.0;
	}

	return pm_source_depths[(layer * PM_SRC_MAX_HEIGHT + i32(fy)) * PM_SRC_MAX_WIDTH + i32(fx)];
}
