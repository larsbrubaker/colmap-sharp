// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryTests: colmap/scene/two_view_geometry_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/TwoViewGeometry.cs.
// EigenMatrixNear on a Matrix3d uses Eigen's default precision (1e-12, relative).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Scene;

public class TwoViewGeometryTests
{
	[Test]
	public async Task TwoViewGeometry_Default()
	{
		var twoViewGeometry = new TwoViewGeometry();
		using (Assert.Multiple())
		{
			await Assert.That(twoViewGeometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Undefined);
			await Assert.That(twoViewGeometry.F.HasValue).IsFalse();
			await Assert.That(twoViewGeometry.E.HasValue).IsFalse();
			await Assert.That(twoViewGeometry.H.HasValue).IsFalse();
			await Assert.That(twoViewGeometry.Cam2FromCam1.HasValue).IsFalse();
			await Assert.That(twoViewGeometry.InlierMatches).IsEmpty();
		}
	}

	[Test]
	public async Task TwoViewGeometry_Invert()
	{
		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			F = Matrix3d.Identity,
			E = Matrix3d.Identity,
			H = Matrix3d.Identity,
			Cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(0, 1, 2)),
			InlierMatches = [new FeatureMatch(0, 1), new FeatureMatch(2, 3)],
		};

		twoViewGeometry.Invert();
		await CheckState(twoViewGeometry, new Vector3d(-0.0, -1, -2), [new FeatureMatch(1, 0), new FeatureMatch(3, 2)]);

		twoViewGeometry.Invert();
		await CheckState(twoViewGeometry, new Vector3d(0, 1, 2), [new FeatureMatch(0, 1), new FeatureMatch(2, 3)]);
	}

	// C#-only (two_view_geometry_test.cc does not cover camera1/camera2): Invert swaps the
	// per-side intrinsics, and Clone copies them by value like the C++ copy of an optional.
	[Test]
	public async Task InvertAndCloneCameras()
	{
		Camera camera1 = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100, 10, 10);
		var twoViewGeometry = new TwoViewGeometry { Camera1 = camera1 };
		TwoViewGeometry copy = twoViewGeometry.Clone();
		twoViewGeometry.Invert();
		using (Assert.Multiple())
		{
			await Assert.That(twoViewGeometry.Camera1).IsNull();
			await Assert.That(ReferenceEquals(twoViewGeometry.Camera2, camera1)).IsTrue();
			await Assert.That(copy.Camera1 == camera1).IsTrue();
			await Assert.That(ReferenceEquals(copy.Camera1, camera1)).IsFalse();
			await Assert.That(copy.Camera2).IsNull();
		}
	}

	private static async Task CheckState(TwoViewGeometry twoViewGeometry, Vector3d translation, FeatureMatch[] matches)
	{
		using (Assert.Multiple())
		{
			await Assert.That(twoViewGeometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
			await Assert.That(twoViewGeometry.F!.Value.IsApprox(Matrix3d.Identity)).IsTrue();
			await Assert.That(twoViewGeometry.E!.Value.IsApprox(Matrix3d.Identity)).IsTrue();
			await Assert.That(twoViewGeometry.H!.Value.IsApprox(Matrix3d.Identity)).IsTrue();
			await Assert.That(twoViewGeometry.Cam2FromCam1.HasValue).IsTrue();
			await Assert.That(twoViewGeometry.Cam2FromCam1!.Value.Rotation.Coeffs.IsApprox(Quaterniond.Identity.Coeffs)).IsTrue();
			await Assert.That(EigenMatrixNear(twoViewGeometry.Cam2FromCam1.Value.Translation, translation)).IsTrue();
			await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx1).IsEqualTo(matches[0].Point2DIdx1);
			await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx2).IsEqualTo(matches[0].Point2DIdx2);
			await Assert.That(twoViewGeometry.InlierMatches[1].Point2DIdx1).IsEqualTo(matches[1].Point2DIdx1);
			await Assert.That(twoViewGeometry.InlierMatches[1].Point2DIdx2).IsEqualTo(matches[1].Point2DIdx2);
		}
	}
}
