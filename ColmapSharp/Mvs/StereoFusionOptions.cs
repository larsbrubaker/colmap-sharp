// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// StereoFusionOptions: the options struct of colmap/mvs/fusion.h (StereoFusionOptions and
// its Check). StereoFusion (Fusion.cs) runs with these. Print is not ported (it writes
// LOG(INFO) lines, which the library does not route). Tests: ColmapSharp.Tests/Mvs/FusionTests.cs.
//
// Translation notes:
// - The double-typed options whose C++ defaults are float literals (max_reproj_error = 2.0f,
//   max_depth_error = 0.01f, max_normal_error = 10.0f) keep the float value widened to
//   double, so 0.01f is 0.009999999776482582 here as in COLMAP.
// - bounding_box (a pair of Eigen::Vector3f) becomes two float triples.

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::StereoFusionOptions.</summary>
public sealed class StereoFusionOptions
{
	/// <summary>
	/// Path for PNG masks, served by the fusion's bitmap source (same format expected as
	/// ImageReaderOptions). Empty: no masks.
	/// </summary>
	public string MaskPath { get; set; } = "";

	/// <summary>
	/// The number of threads to use when loading the workspace. The fusion traversal itself
	/// always runs on one thread (divergence 87).
	/// </summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Maximum image size in either dimension.</summary>
	public int MaxImageSize { get; set; } = -1;

	/// <summary>Minimum number of fused pixels to produce a point.</summary>
	public int MinNumPixels { get; set; } = 5;

	/// <summary>Maximum number of pixels to fuse into a single point.</summary>
	public int MaxNumPixels { get; set; } = 10000;

	/// <summary>Maximum depth in consistency graph traversal.</summary>
	public int MaxTraversalDepth { get; set; } = 100;

	/// <summary>Maximum relative difference between measured and projected pixel.</summary>
	public double MaxReprojError { get; set; } = 2.0f;

	/// <summary>Maximum relative difference between measured and projected depth.</summary>
	public double MaxDepthError { get; set; } = 0.01f;

	/// <summary>Maximum angular difference in degrees of normals of pixels to be fused.</summary>
	public double MaxNormalError { get; set; } = 10.0f;

	/// <summary>Number of overlapping images to transitively check for fusing points.</summary>
	public int CheckNumImages { get; set; } = 50;

	/// <summary>Flag indicating whether to use LRU cache or pre-load all data.</summary>
	public bool UseCache { get; set; }

	/// <summary>
	/// Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth maps, normal
	/// maps, and consistency graphs of this number of images in memory. A higher value leads
	/// to less disk access and faster fusion, while a lower value leads to reduced memory
	/// usage. Note that a single image can consume a lot of memory, if the consistency graph
	/// is dense.
	/// </summary>
	public double CacheSize { get; set; } = 32.0;

	/// <summary>Lower corner of the box outside which fused pixels are dropped.</summary>
	public (float X, float Y, float Z) BoundingBoxMin { get; set; } = (-float.MaxValue, -float.MaxValue, -float.MaxValue);

	/// <summary>Upper corner of the box outside which fused pixels are dropped.</summary>
	public (float X, float Y, float Z) BoundingBoxMax { get; set; } = (float.MaxValue, float.MaxValue, float.MaxValue);

	/// <summary>A copy (C++ copies the options struct by value).</summary>
	public StereoFusionOptions Clone() => (StereoFusionOptions)MemberwiseClone();

	/// <summary>Port of StereoFusionOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check() =>
		MinNumPixels >= 0
		&& MinNumPixels <= MaxNumPixels
		&& MaxTraversalDepth > 0
		&& MaxReprojError >= 0
		&& MaxDepthError >= 0
		&& MaxNormalError >= 0
		&& CheckNumImages > 0
		&& CacheSize > 0;
}

/// <summary>
/// Progress of a <see cref="StereoFusion"/> run, reported after each fused image.
/// </summary>
/// <param name="NumFusedImages">Images fused so far.</param>
/// <param name="NumImages">Images in the model (COLMAP's "[i/n]" denominator).</param>
/// <param name="NumFusedPoints">Points fused so far.</param>
public readonly record struct StereoFusionProgress(int NumFusedImages, int NumImages, int NumFusedPoints);
