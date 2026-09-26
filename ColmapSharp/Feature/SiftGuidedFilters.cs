// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftGuidedFilters: the geometric half of SiftCPUFeatureMatcher::MatchGuided in
// colmap/feature/sift.cc - CamRaysWithJac / ComputeCamRaysWithJac,
// UseEssentialMatrixForGuidedMatching and the three guided filters (essential matrix with the
// tangent Sampson error on bearings, fundamental matrix and homography on pixels). Neighbors:
// SiftMatcher.cs (the caller), Geometry/EssentialMatrix.cs (the tangent Sampson error),
// Scene/Camera.cs (CamRayFromImgWithJac). Tests: SiftMatcherTests.cs (MatchGuidedSiftFeaturesCPU).
//
// The F and H filters are float expressions evaluated in Eigen's order, left to right, for
// 3 x 3 times 3-vector products ((a0*b0 + a1*b1) + a2*b2), without FMA. The filters are pure
// functions of (i1, i2), so the parallel distance scan may call them from any thread.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>The guided-matching filters of COLMAP's SIFT CPU matcher.</summary>
internal static class SiftGuidedFilters
{
	/// <summary>
	/// Port of UseEssentialMatrixForGuidedMatching. Selects the epipolar model used to guide
	/// matching. The essential matrix is preferred whenever the intrinsics are known, which
	/// properly handles non-pinhole camera models where the fundamental matrix relationship
	/// does not hold. The intrinsics are known either from priors (calibrated configs) or
	/// because the two-view solver recovered them: such pairs are labeled UNCALIBRATED but
	/// carry the estimated intrinsics in camera1/camera2. Either side alone suffices, to
	/// support solvers that estimate only one side against an already calibrated view.
	/// </summary>
	public static bool UseEssentialMatrixForGuidedMatching(TwoViewGeometry geometry)
	{
		if (!geometry.E.HasValue)
		{
			return false;
		}

		return geometry.Config switch
		{
			TwoViewGeometry.ConfigurationType.Calibrated or TwoViewGeometry.ConfigurationType.CalibratedRig => true,
			TwoViewGeometry.ConfigurationType.Uncalibrated => geometry.Camera1 is not null || geometry.Camera2 is not null,
			_ => false,
		};
	}

	/// <summary>
	/// The guided filter of MatchGuided: (i1, i2) true rejects the pair on geometric grounds.
	/// Null when the two-view geometry carries no usable model, in which case COLMAP returns
	/// without matches.
	/// </summary>
	public static Func<int, int, bool>? Create(
		double maxError, FeatureMatcherImage image1, FeatureMatcherImage image2, TwoViewGeometry twoViewGeometry)
	{
		bool useEssentialMatrix = UseEssentialMatrixForGuidedMatching(twoViewGeometry);
		bool useFundamentalMatrix = !useEssentialMatrix &&
			twoViewGeometry.Config == TwoViewGeometry.ConfigurationType.Uncalibrated &&
			twoViewGeometry.F.HasValue;
		bool useHomography =
			twoViewGeometry.Config is TwoViewGeometry.ConfigurationType.Planar
				or TwoViewGeometry.ConfigurationType.Panoramic
				or TwoViewGeometry.ConfigurationType.PlanarOrPanoramic &&
			twoViewGeometry.H.HasValue;

		// Normalize with the intrinsics the solver estimated where available (its focal, not
		// the camera's stale default), else the given cameras.
		Camera effectiveCamera1 = twoViewGeometry.Camera1 ?? image1.Camera!;
		Camera effectiveCamera2 = twoViewGeometry.Camera2 ?? image2.Camera!;

		// A spherical camera has no meaningful image plane, so a homography over pixel
		// coordinates is not a valid model for it. No estimator produces one today; fail
		// loudly rather than silently dividing by a near-zero z.
		if (useHomography)
		{
			Check.That(!effectiveCamera1.IsSpherical);
			Check.That(!effectiveCamera2.IsSpherical);
		}

		IReadOnlyList<FeatureKeypoint> keypoints1 = image1.Keypoints!;
		IReadOnlyList<FeatureKeypoint> keypoints2 = image2.Keypoints!;
		double maxResidualDouble = maxError * maxError;
		float maxResidual = (float)maxResidualDouble;

		if (useEssentialMatrix)
		{
			// The essential matrix path scores in pixels with the tangent Sampson error,
			// matching the two-view verification that produced E. Bearings are used rather
			// than normalized image plane coordinates so that the filter is defined for every
			// central camera model, including omnidirectional ones whose back hemisphere has
			// no image plane representation at all.
			CamRayWithJac?[] camRays1 = ComputeCamRaysWithJac(effectiveCamera1, keypoints1);
			CamRayWithJac?[] camRays2 = ComputeCamRaysWithJac(effectiveCamera2, keypoints2);
			Matrix3d e = twoViewGeometry.E!.Value;
			return (i1, i2) =>
			{
				if (camRays1[i1] is not CamRayWithJac ray1 || camRays2[i2] is not CamRayWithJac ray2)
				{
					return true;
				}

				return EssentialMatrix.ComputeSquaredTangentSampsonError(
					ray1.Ray, ray1.Jacobian, ray2.Ray, ray2.Jacobian, e) > maxResidualDouble;
			};
		}

		if (useFundamentalMatrix)
		{
			float[] f = ToFloat(twoViewGeometry.F!.Value);
			return (i1, i2) =>
			{
				FeatureKeypoint k1 = keypoints1[i1];
				FeatureKeypoint k2 = keypoints2[i2];

				// epipolar_line1 = F * p1, epipolar_line2 = F^T * p2 with p = (x, y, 1).
				float l10 = (f[0] * k1.X) + (f[1] * k1.Y) + f[2];
				float l11 = (f[3] * k1.X) + (f[4] * k1.Y) + f[5];
				float l12 = (f[6] * k1.X) + (f[7] * k1.Y) + f[8];
				float l20 = (f[0] * k2.X) + (f[3] * k2.Y) + f[6];
				float l21 = (f[1] * k2.X) + (f[4] * k2.Y) + f[7];
				float nom = (k2.X * l10) + (k2.Y * l11) + l12;
				float denomSq = (l10 * l10) + (l11 * l11) + (l20 * l20) + (l21 * l21);
				return nom * nom > maxResidual * denomSq;
			};
		}

		if (useHomography)
		{
			float[] h = ToFloat(twoViewGeometry.H!.Value);
			return (i1, i2) =>
			{
				FeatureKeypoint k1 = keypoints1[i1];
				FeatureKeypoint k2 = keypoints2[i2];

				// ((H * p1).hnormalized() - p2).squaredNorm().
				float q0 = (h[0] * k1.X) + (h[1] * k1.Y) + h[2];
				float q1 = (h[3] * k1.X) + (h[4] * k1.Y) + h[5];
				float q2 = (h[6] * k1.X) + (h[7] * k1.Y) + h[8];
				float dx = (q0 / q2) - k2.X;
				float dy = (q1 / q2) - k2.Y;
				return (dx * dx) + (dy * dy) > maxResidual;
			};
		}

		return null;
	}

	// Row-major float copy of a 3 x 3 matrix (Matrix3d::cast<float>()).
	private static float[] ToFloat(in Matrix3d m)
	{
		var result = new float[9];
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				result[(3 * r) + c] = (float)m[r, c];
			}
		}

		return result;
	}

	/// <summary>
	/// Port of ComputeCamRaysWithJac: unit bearing vectors and their pixel Jacobians for a set
	/// of keypoints; null for a keypoint that cannot be unprojected.
	/// Keypoints that cannot be unprojected - back-hemisphere pixels of an omnidirectional
	/// camera have no normalized image plane representation, and iterative undistortion can
	/// fail - must be excluded from matching by the caller. Note that encoding invalidity as
	/// an extreme coordinate does *not* work: the Sampson error is a ratio whose numerator and
	/// denominator scale together, so a point pushed to infinity along a direction d converges
	/// to the finite distance between its partner and the epipolar line of d, which admits
	/// rather than rejects partners lying near that one line.
	/// </summary>
	private static CamRayWithJac?[] ComputeCamRaysWithJac(Camera camera, IReadOnlyList<FeatureKeypoint> keypoints)
	{
		var camRays = new CamRayWithJac?[keypoints.Count];
		for (int i = 0; i < camRays.Length; ++i)
		{
			camRays[i] = camera.CamRayFromImgWithJac(new Vector2d(keypoints[i].X, keypoints[i].Y));
		}

		return camRays;
	}
}
