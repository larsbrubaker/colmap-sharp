// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshingOptions: port of colmap::mvs::DelaunayMeshingOptions (mvs/delaunay_meshing.h
// and its Check() in delaunay_meshing.cc). Used by DelaunayMeshingInput (the triangulation
// step) and DelaunayMeshingEdgeWeightComputer. Tier A: plain value checks.

namespace ColmapSharp.Mvs;

/// <summary>
/// Options of Delaunay meshing (Labatut, Pons and Keriven 2009), as in COLMAP.
/// </summary>
public sealed class DelaunayMeshingOptions
{
	/// <summary>
	/// Unify input points into one cell in the Delaunay triangulation that fall within a
	/// reprojected radius of the given pixels.
	/// </summary>
	public double MaxProjDist { get; set; } = 20.0;

	/// <summary>
	/// Maximum relative depth difference between input point and a vertex of an existing
	/// cell in the Delaunay triangulation, otherwise a new vertex is created.
	/// </summary>
	public double MaxDepthDist { get; set; } = 0.05;

	/// <summary>
	/// The standard deviation of wrt. the number of images seen by each point. Increasing
	/// this value decreases the influence of points seen in few images.
	/// </summary>
	public double VisibilitySigma { get; set; } = 3.0;

	/// <summary>
	/// The factor applied to the computed distance sigma, which is automatically computed as
	/// the 25th percentile of edge lengths. A higher value increases surface smoothness.
	/// </summary>
	public double DistanceSigmaFactor { get; set; } = 1.0;

	/// <summary>A higher quality regularization leads to a smoother surface.</summary>
	public double QualityRegularization { get; set; } = 1.0;

	/// <summary>
	/// Filtering threshold for outlier surface mesh faces: if the longest side of a face
	/// exceeds the side lengths of all faces at <see cref="MaxSideLengthPercentile"/> by this
	/// factor, the face is discarded.
	/// </summary>
	public double MaxSideLengthFactor { get; set; } = 25.0;

	/// <summary>The percentile used with <see cref="MaxSideLengthFactor"/>.</summary>
	public double MaxSideLengthPercentile { get; set; } = 95.0;

	/// <summary>The number of threads to use for reconstruction; -1 means all threads.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Port of DelaunayMeshingOptions::Check (CHECK_OPTION_* return false when violated).</summary>
	public bool Check() =>
		MaxProjDist >= 0
		&& MaxDepthDist >= 0
		&& MaxDepthDist <= 1
		&& VisibilitySigma > 0
		&& DistanceSigmaFactor > 0
		&& QualityRegularization >= 0
		&& MaxSideLengthFactor >= 0
		&& MaxSideLengthPercentile >= 0
		&& MaxSideLengthPercentile <= 100
		&& NumThreads >= -1
		&& NumThreads != 0;
}
