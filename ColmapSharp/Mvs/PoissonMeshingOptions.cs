// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoissonMeshingOptions: port of colmap::mvs::PoissonMeshingOptions (mvs/poisson_meshing.h and
// its Check() in poisson_meshing.cc). Used by PoissonMeshing, which turns them into the
// PoissonRecon and SurfaceTrimmer arguments COLMAP passes. Tier A: plain value checks.

namespace ColmapSharp.Mvs;

/// <summary>Options for screened Poisson surface reconstruction (port of colmap::mvs::PoissonMeshingOptions).</summary>
public sealed class PoissonMeshingOptions
{
	/// <summary>
	/// This floating point value specifies the importance that interpolation of the point
	/// samples is given in the formulation of the screened Poisson equation. The results of the
	/// original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0.
	/// </summary>
	public double PointWeight { get; set; } = 1.0;

	/// <summary>
	/// This integer is the maximum depth of the tree that will be used for surface
	/// reconstruction. Running at depth d corresponds to solving on a voxel grid whose
	/// resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts
	/// the octree to the sampling density, the specified reconstruction depth is only an upper
	/// bound.
	/// </summary>
	public int Depth { get; set; } = 13;

	/// <summary>
	/// Whether to color the vertices. As in COLMAP, this has no effect on the result: PoissonRecon
	/// ignores --colors for PLY input and colors the mesh whenever the input points have colors.
	/// </summary>
	public bool Color { get; set; } = true;

	/// <summary>
	/// This floating point values specifies the value for mesh trimming. The subset of the mesh
	/// with signal value less than the trim value is discarded.
	/// </summary>
	public double Trim { get; set; } = 10.0;

	/// <summary>
	/// The number of threads used for the Poisson reconstruction. The port runs sequentially
	/// (its results do not depend on the thread count), so this is only validated.
	/// </summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Port of PoissonMeshingOptions::Check (CHECK_OPTION_*: false, not a throw, when invalid).</summary>
	public bool Check() =>
		PointWeight >= 0
		&& Depth > 0
		&& Trim >= 0
		&& NumThreads >= -1
		&& NumThreads != 0;
}
