// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchOptions: colmap/mvs/patch_match_options.h and .cc - the settings of PatchMatch
// stereo (PatchMatch.cs, the per-image problem, and PatchMatchController.cs, which runs one
// problem per reference image). Tests: ColmapSharp.Tests/Mvs/PatchMatchTests.cs
// (COLMAP has no patch_match_test.cc; those cases are C#-only).
//
// Translation notes:
// - The defaults are written exactly as COLMAP writes them: several double options are
//   initialized from float literals (sigma_color = 0.2f), so they hold the float's value
//   (0.20000000298023224), not the decimal one.
// - gpu_index is not ported: PatchMatch runs on the CPU here, so there is no device to
//   pick (docs/CPP_DIVERGENCES.md, entry 86). NumThreads bounds the CPU parallelism.
// - Print() is not ported (it is a LOG(INFO) listing; only warnings and errors go to
//   Util/Log.cs).

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::PatchMatchOptions.</summary>
public sealed class PatchMatchOptions
{
	/// <summary>
	/// Maximum possible window radius for the photometric consistency cost
	/// (kMaxPatchMatchWindowRadius). In COLMAP it equals THREADS_PER_BLOCK in
	/// patch_match_cuda.cu, a shared-memory limit; the CPU port keeps the same bound so the
	/// same options are valid on both.
	/// </summary>
	public const int MaxPatchMatchWindowRadius = 32;

	/// <summary>Depth range in which to randomly sample depth hypotheses (-1: from the sparse model).</summary>
	public double DepthMin { get; set; } = -1.0f;

	/// <summary>Depth range in which to randomly sample depth hypotheses (-1: from the sparse model).</summary>
	public double DepthMax { get; set; } = -1.0f;

	/// <summary>Spatial sigma of the bilaterally weighted NCC (&lt;= 0: the window radius).</summary>
	public double SigmaSpatial { get; set; } = -1;

	/// <summary>Color sigma of the bilaterally weighted NCC.</summary>
	public double SigmaColor { get; set; } = 0.2f;

	/// <summary>Spread of the NCC likelihood function.</summary>
	public double NccSigma { get; set; } = 0.6f;

	/// <summary>Minimum triangulation angle in degrees.</summary>
	public double MinTriangulationAngle { get; set; } = 1.0f;

	/// <summary>Spread of the incident angle likelihood function.</summary>
	public double IncidentAngleSigma { get; set; } = 0.9f;

	/// <summary>
	/// The relative weight of the geometric consistency term w.r.t. to the
	/// photo-consistency term.
	/// </summary>
	public double GeomConsistencyRegularizer { get; set; } = 0.3f;

	/// <summary>
	/// Maximum geometric consistency cost in terms of the forward-backward reprojection
	/// error in pixels.
	/// </summary>
	public double GeomConsistencyMaxCost { get; set; } = 3.0f;

	/// <summary>Minimum NCC coefficient for pixel to be photo-consistent.</summary>
	public double FilterMinNcc { get; set; } = 0.1f;

	/// <summary>Minimum triangulation angle to be stable.</summary>
	public double FilterMinTriangulationAngle { get; set; } = 3.0f;

	/// <summary>
	/// Maximum forward-backward reprojection error for pixel to be geometrically consistent.
	/// </summary>
	public double FilterGeomConsistencyMaxCost { get; set; } = 1.0f;

	/// <summary>
	/// Cache size in gigabytes for patch match, which keeps the bitmaps, depth maps, and
	/// normal maps of this number of images in memory. A higher value leads to less disk
	/// access and faster computation, while a lower value leads to reduced memory usage.
	/// Note that a single image can consume a lot of memory, if the consistency graph is
	/// dense.
	/// </summary>
	public double CacheSize { get; set; } = 32.0;

	/// <summary>Maximum image size in either dimension (-1: no limit).</summary>
	public int MaxImageSize { get; set; } = -1;

	/// <summary>Half window size to compute NCC photo-consistency cost.</summary>
	public int WindowRadius { get; set; } = 5;

	/// <summary>
	/// Number of pixels to skip when computing NCC. For a value of 1, every pixel is used to
	/// compute the NCC. For larger values, only every n-th row and column is used and the
	/// computation speed thereby increases roughly by a factor of window_step^2. Note that
	/// not all combinations of window sizes and steps produce nice results, especially if
	/// the step is greater than 2.
	/// </summary>
	public int WindowStep { get; set; } = 1;

	/// <summary>Number of random samples to draw in Monte Carlo sampling.</summary>
	public int NumSamples { get; set; } = 15;

	/// <summary>
	/// Number of coordinate descent iterations. Each iteration consists of four sweeps from
	/// left to right, top to bottom, and vice versa.
	/// </summary>
	public int NumIterations { get; set; } = 5;

	/// <summary>Minimum number of source images have to be consistent for pixel not to be filtered.</summary>
	public int FilterMinNumConsistent { get; set; } = 2;

	/// <summary>Number of threads for processing. -1 uses all available threads.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// Whether to add a regularized geometric consistency term to the cost function. If
	/// true, the problem's depth maps and normal maps must not be null.
	/// </summary>
	public bool GeomConsistency { get; set; } = true;

	/// <summary>Whether to enable filtering.</summary>
	public bool Filter { get; set; } = true;

	/// <summary>Whether to tolerate missing images/maps in the problem setup.</summary>
	public bool AllowMissingFiles { get; set; } = false;

	/// <summary>Whether to write the consistency graph.</summary>
	public bool WriteConsistencyGraph { get; set; } = false;

	/// <summary>A copy of these options (C++ copies the struct by value).</summary>
	public PatchMatchOptions Clone() => (PatchMatchOptions)MemberwiseClone();

	/// <summary>
	/// Port of PatchMatchOptions::Check: false when an option is out of range
	/// (CHECK_OPTION_* logs and returns false in COLMAP).
	/// </summary>
	public bool Check()
	{
		// The float literals of the C++ comparisons are exact small values, so comparing
		// against the double constants below is the same test.
		if (DepthMin != -1.0f || DepthMax != -1.0f)
		{
			if (!(DepthMin <= DepthMax) || !(DepthMin >= 0.0f))
			{
				return false;
			}
		}

		return WindowRadius <= MaxPatchMatchWindowRadius
			&& SigmaColor > 0.0f
			&& WindowRadius > 0
			&& WindowStep > 0
			&& WindowStep <= 2
			&& NumSamples > 0
			&& NccSigma > 0.0f
			&& MinTriangulationAngle >= 0.0f
			&& MinTriangulationAngle < 180.0f
			&& IncidentAngleSigma > 0.0f
			&& NumIterations > 0
			&& GeomConsistencyRegularizer >= 0.0f
			&& GeomConsistencyMaxCost >= 0.0f
			&& FilterMinNcc >= -1.0f
			&& FilterMinNcc <= 1.0f
			&& FilterMinTriangulationAngle >= 0.0f
			&& FilterMinTriangulationAngle <= 180.0f
			&& FilterMinNumConsistent >= 0
			&& FilterGeomConsistencyMaxCost >= 0.0f
			&& CacheSize > 0
			&& NumThreads >= -1;
	}
}
