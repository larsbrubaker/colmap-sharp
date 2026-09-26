// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Rigid3dMatchers: the Rigid3dEq and Rigid3dNear gmock matchers of
// colmap/geometry/rigid3_matchers.h, as predicates: EXPECT_THAT(a, Rigid3dNear(b, rtol,
// ttol)) becomes Assert.That(Rigid3dNear(a, b, rtol, ttol)).IsTrue(). Test infrastructure
// next to EigenMatchers.cs; first users are Scene/CorrespondenceGraphTests.cs. Its own
// tests are Geometry/Rigid3dMatchersTests.cs (rigid3_matchers_test.cc 1:1).
//
// Semantics, as in COLMAP: Eq compares rotation coefficients and translation exactly with
// !(a == b) so NaN never matches. Near requires the rotations' angular distance <= rtol,
// then, if rhs's translation is zero (isZero(), every coefficient within 1e-12), the lhs
// translation norm <= ttol, else lhs.translation.isApprox(rhs.translation, ttol).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Tests;

/// <summary>
/// Port of COLMAP's Rigid3dEq and Rigid3dNear matchers.
/// </summary>
internal static class Rigid3dMatchers
{
	/// <summary>Rigid3dEq(rhs) applied to lhs: exact equality of rotation coefficients and translation.</summary>
	public static bool Rigid3dEq(Rigid3d lhs, Rigid3d rhs)
	{
		if (!(lhs.Rotation.Coeffs == rhs.Rotation.Coeffs))
		{
			return false;
		}

		return lhs.Translation == rhs.Translation;
	}

	/// <summary>Rigid3dNear(rhs, rtol, ttol) applied to lhs.</summary>
	public static bool Rigid3dNear(
		Rigid3d lhs,
		Rigid3d rhs,
		double rtol = LinearAlgebraConstants.DummyPrecision,
		double ttol = LinearAlgebraConstants.DummyPrecision)
	{
		// Note the !(a <= b), to handle NaNs.
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
