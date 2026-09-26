// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometry: port of colmap/scene/two_view_geometry.h and .cc, the verified geometry
// of an image pair - its configuration, the E/F/H matrices, the relative pose and the inlier
// feature matches. The two-view estimator (Phase 6) produces it, the database stores it, and
// CorrespondenceGraph.cs holds one per image pair. Tests:
// ColmapSharp.Tests/Scene/TwoViewGeometryTests.cs (two_view_geometry_test.cc 1:1).
//
// Tier B for Invert (H goes through a 3x3 inverse, Matrix3d.Inverse); everything else is
// Tier A bookkeeping.
//
// Translation notes:
// - A class (it owns the match list); C++ copy-assignment becomes Clone(). The optional
//   matrices and pose are nullable value types.
// - config is an int in COLMAP holding a ConfigurationType; here it is the enum itself, with
//   COLMAP's numeric values, so storage code can cast it to and from int.
// - `std::optional<Camera> camera1, camera2` (intrinsics a two-view solver estimated) are
//   nullable Camera references; Clone copies them (Camera.Clone), since C++ copies the
//   optional by value, and Invert swaps them.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::TwoViewGeometry: the geometric relation verified between two images.
/// </summary>
public sealed class TwoViewGeometry
{
	/// <summary>Port of TwoViewGeometry::ConfigurationType, with COLMAP's numeric values.</summary>
	public enum ConfigurationType
	{
		/// <summary>UNDEFINED.</summary>
		Undefined = 0,

		/// <summary>DEGENERATE: no overlap or not enough inliers.</summary>
		Degenerate = 1,

		/// <summary>CALIBRATED: essential matrix.</summary>
		Calibrated = 2,

		/// <summary>
		/// UNCALIBRATED: fundamental matrix. A pair whose focal lengths the two-view solver
		/// recovered is also UNCALIBRATED; it carries F built from the estimated focal.
		/// </summary>
		Uncalibrated = 3,

		/// <summary>PLANAR: homography, planar scene with baseline.</summary>
		Planar = 4,

		/// <summary>PANORAMIC: homography, pure rotation without baseline.</summary>
		Panoramic = 5,

		/// <summary>PLANAR_OR_PANORAMIC: homography, planar or panoramic.</summary>
		PlanarOrPanoramic = 6,

		/// <summary>WATERMARK: pure 2D translation in image borders.</summary>
		Watermark = 7,

		/// <summary>
		/// MULTIPLE: the inlier matches result from multiple individual, non-degenerate
		/// configurations.
		/// </summary>
		Multiple = 8,

		/// <summary>CALIBRATED_RIG: relative pose (metric) from a calibrated (non-panoramic) rig.</summary>
		CalibratedRig = 9,
	}

	/// <summary>The configuration of the two-view geometry.</summary>
	public ConfigurationType Config { get; set; } = ConfigurationType.Undefined;

	/// <summary>Essential matrix, if estimated.</summary>
	public Matrix3d? E { get; set; }

	/// <summary>Fundamental matrix, if estimated.</summary>
	public Matrix3d? F { get; set; }

	/// <summary>Homography matrix, if estimated.</summary>
	public Matrix3d? H { get; set; }

	/// <summary>Relative pose from the first to the second camera, if estimated.</summary>
	public Rigid3d? Cam2FromCam1 { get; set; }

	/// <summary>
	/// Side 1's intrinsics as recovered by the two-view solver, or null if that side was not
	/// estimated.
	/// </summary>
	public Camera? Camera1 { get; set; }

	/// <summary>
	/// Side 2's intrinsics as recovered by the two-view solver, or null if that side was not
	/// estimated.
	/// </summary>
	public Camera? Camera2 { get; set; }

	/// <summary>Inlier matches of the configuration (FeatureMatches).</summary>
	public List<FeatureMatch> InlierMatches { get; set; } = [];

	/// <summary>Median triangulation angle; -1 when not computed.</summary>
	public double TriAngle { get; set; } = -1;

	/// <summary>A deep copy (C++ copy construction).</summary>
	public TwoViewGeometry Clone()
	{
		return new TwoViewGeometry
		{
			Config = Config,
			E = E,
			F = F,
			H = H,
			Cam2FromCam1 = Cam2FromCam1,
			Camera1 = Camera1?.Clone(),
			Camera2 = Camera2?.Clone(),
			InlierMatches = new List<FeatureMatch>(InlierMatches),
			TriAngle = TriAngle,
		};
	}

	/// <summary>Inverts the geometry to match swapped cameras.</summary>
	public void Invert()
	{
		if (F is Matrix3d f)
		{
			F = f.Transpose();
		}

		if (E is Matrix3d e)
		{
			E = e.Transpose();
		}

		if (H is Matrix3d h)
		{
			H = h.Inverse();
		}

		if (Cam2FromCam1 is Rigid3d cam2FromCam1)
		{
			Cam2FromCam1 = cam2FromCam1.Inverse();
		}

		(Camera1, Camera2) = (Camera2, Camera1);

		System.Span<FeatureMatch> matches = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(InlierMatches);
		for (int i = 0; i < matches.Length; i++)
		{
			matches[i] = new FeatureMatch(matches[i].Point2DIdx2, matches[i].Point2DIdx1);
		}
	}
}
