// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sim3dMatchers: the Sim3dEq and Sim3dNear gmock matchers of colmap/geometry/sim3_matchers.h,
// as predicates, next to Rigid3dMatchers.cs. First user:
// Estimators/Solvers/SimilarityTransformTests.cs. Its own tests are
// Geometry/Sim3dMatchersTests.cs (sim3_matchers_test.cc 1:1).
//
// Semantics, as in COLMAP: Eq compares scale, rotation coefficients and translation exactly
// with !(a == b) so NaN never matches. Near requires |scale difference| <= stol, rotation
// angular distance <= rtol, then, if rhs's translation is zero (isZero(), every coefficient
// within 1e-12), lhs translation norm <= ttol, else lhs.translation.isApprox(rhs, ttol).
// Faithful to an upstream slip: COLMAP's Sim3dNearMatcher initializes its scale tolerance
// from rtol (`stol_(rtol)`), so stol is ignored and the scale is compared against rtol.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Tests;

/// <summary>
/// Port of COLMAP's Sim3dEq and Sim3dNear matchers.
/// </summary>
internal static class Sim3dMatchers
{
	/// <summary>Sim3dEq(rhs) applied to lhs: exact equality of scale, rotation and translation.</summary>
	public static bool Sim3dEq(Sim3d lhs, Sim3d rhs)
	{
		if (!(lhs.Scale == rhs.Scale))
		{
			return false;
		}

		if (!(lhs.Rotation.Coeffs == rhs.Rotation.Coeffs))
		{
			return false;
		}

		return lhs.Translation == rhs.Translation;
	}

	/// <summary>Sim3dNear(rhs, stol, rtol, ttol) applied to lhs.</summary>
	public static bool Sim3dNear(
		Sim3d lhs,
		Sim3d rhs,
		double stol = LinearAlgebraConstants.DummyPrecision,
		double rtol = LinearAlgebraConstants.DummyPrecision,
		double ttol = LinearAlgebraConstants.DummyPrecision)
	{
		// COLMAP's matcher stores stol_(rtol); see the file header.
		_ = stol;
		double scaleTolerance = rtol;

		// Note the !(a <= b), to handle NaNs.
		if (!(Math.Abs(lhs.Scale - rhs.Scale) <= scaleTolerance))
		{
			return false;
		}

		if (!(lhs.Rotation.AngularDistance(rhs.Rotation) <= rtol))
		{
			return false;
		}

		Vector3d rhsTranslation = rhs.Translation;
		bool rhsIsZero = Math.Abs(rhsTranslation.X) <= LinearAlgebraConstants.DummyPrecision
			&& Math.Abs(rhsTranslation.Y) <= LinearAlgebraConstants.DummyPrecision
			&& Math.Abs(rhsTranslation.Z) <= LinearAlgebraConstants.DummyPrecision;
		if (rhsIsZero)
		{
			// isApprox() is not well-defined for zero matrices.
			return lhs.Translation.Norm <= ttol;
		}

		return lhs.Translation.IsApprox(rhsTranslation, ttol);
	}
}
