// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ProjectionTests: colmap/scene/projection_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/Projection.cs. EIGEN_PI is
// Math.PI. The random point of CalculateSquaredReprojectionError.Nominal is the first draw
// after the per-test seed of 0 (PrngTestIsolation, as COLMAP's gtest_main).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using static ColmapSharp.Scene.Projection;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ProjectionTests
{
	[Test]
	public async Task CalculateSquaredReprojectionError_Nominal()
	{
		var camFromWorld = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Matrix3x4d camFromWorldMat = camFromWorld.ToMatrix();

		Vector3d point3D = RandomEigen.RandomEigenVector3d().CwiseAbs();
		Vector3d point2DH = camFromWorldMat * point3D.Homogeneous();
		Vector2d point2D = point2DH.HNormalized();

		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 0, 0);

		using (Assert.Multiple())
		{
			await Assert.That(CalculateSquaredReprojectionError(point2D, point3D, camFromWorld, camera)).IsEqualTo(0.0).Within(1e-6);
			await Assert.That(CalculateSquaredReprojectionError(point2D, point3D, camFromWorldMat, camera)).IsEqualTo(0.0).Within(1e-6);

			await Assert.That(CalculateSquaredReprojectionError(point2D + Vector2d.Ones, point3D, camFromWorld, camera)).IsEqualTo(2.0).Within(1e-6);
			await Assert.That(CalculateSquaredReprojectionError(point2D + Vector2d.Ones, point3D, camFromWorldMat, camera)).IsEqualTo(2.0).Within(1e-6);
		}
	}

	[Test]
	public async Task CalculateSquaredReprojectionError_Spherical()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		var camFromWorld = new Rigid3d();
		double pxPerRad = camera.Width / (2.0 * Math.PI);

		using (Assert.Multiple())
		{
			// Exact observations have ~0 error, including a back-hemisphere match that
			// straddles the azimuth seam (observed at x = 0, point projects to x = w), which a
			// raw pixel error would penalize by ~width^2.
			foreach (Vector3d camPoint in new[] { new Vector3d(0, 0, 1), new Vector3d(0, 0, -1), new Vector3d(1, 0, 0) })
			{
				Vector2d imgPoint = camera.ImgFromCam(camPoint)!.Value;
				await Assert.That(CalculateSquaredReprojectionError(imgPoint, camPoint, camFromWorld, camera)).IsEqualTo(0.0).Within(1e-9);
			}

			// Check that the error is continuous across the seam.
			await Assert.That(CalculateSquaredReprojectionError(
				new Vector2d(0.0, camera.Height / 2.0), new Vector3d(0, 0, -1), camFromWorld, camera)).IsEqualTo(0.0).Within(1e-9);

			// Pole invariance: the same angular offset yields the same squared error at the
			// equator and near the pole. Raw equirectangular pixel error would diverge towards
			// the pole, dropping otherwise-valid observations.
			const double kOffset = 0.01; // radians
			var equatorPoint = new Vector3d(0, 0, 1);
			Vector2d equatorObs = camera.ImgFromCam(new Vector3d(Math.Sin(kOffset), 0, Math.Cos(kOffset)))!.Value;
			var polePoint = new Vector3d(0, -1, 0);
			Vector2d poleObs = camera.ImgFromCam(new Vector3d(0, -Math.Cos(kOffset), Math.Sin(kOffset)))!.Value;
			double expected = (kOffset * pxPerRad) * (kOffset * pxPerRad);
			await Assert.That(CalculateSquaredReprojectionError(equatorObs, equatorPoint, camFromWorld, camera)).IsEqualTo(expected).Within(1e-6);
			await Assert.That(CalculateSquaredReprojectionError(poleObs, polePoint, camFromWorld, camera)).IsEqualTo(expected).Within(1e-6);
		}
	}

	[Test]
	public async Task CalculateAngularReprojectionError_Nominal()
	{
		var camFromWorld = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Matrix3x4d camFromWorldMat = camFromWorld.ToMatrix();

		var camera = new Camera { ModelId = CameraModelId.SimplePinhole, Params = [1, 0, 0] };

		double Error(double x, double y, double pX, double pY, double pZ) =>
			CalculateAngularReprojectionError(new Vector2d(x, y), new Vector3d(pX, pY, pZ), camFromWorldMat, camera);

		using (Assert.Multiple())
		{
			await Assert.That(Error(0, 0, 0, 0, 1)).IsEqualTo(0.0).Within(1e-6);
			await Assert.That(Error(0, 0, 0, 1, 1)).IsEqualTo(Math.PI / 4).Within(1e-6);
			await Assert.That(Error(0, 0, 0, 5, 5)).IsEqualTo(Math.PI / 4).Within(1e-6);
			await Assert.That(Error(1, 0, 0, 0, 1)).IsEqualTo(Math.PI / 4).Within(1e-6);
			await Assert.That(Error(2, 0, 0, 0, 1)).IsEqualTo(1.10714872).Within(1e-6);
			await Assert.That(Error(2, 0, 1, 0, 1)).IsEqualTo(1.10714872 - Math.PI / 4).Within(1e-6);
			await Assert.That(Error(2, 0, 5, 0, 5)).IsEqualTo(1.10714872 - Math.PI / 4).Within(1e-6);
			await Assert.That(Error(1, 0, -1, 0, 1)).IsEqualTo(Math.PI / 2).Within(1e-6);
			await Assert.That(Error(1, 0, -1, 0, 0)).IsEqualTo(Math.PI * 3 / 4).Within(1e-6);
			await Assert.That(Error(1, 0, -1, 0, -1)).IsEqualTo(Math.PI).Within(1e-6);
			await Assert.That(Error(1, 0, 0, 0, -1)).IsEqualTo(Math.PI * 3 / 4).Within(1e-6);
		}
	}

	[Test]
	public async Task HasPointPositiveDepth_Nominal()
	{
		var camFromWorld = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Matrix3x4d camFromWorldMat = camFromWorld.ToMatrix();

		using (Assert.Multiple())
		{
			// In the image plane
			await Assert.That(HasPointPositiveDepth(camFromWorldMat, new Vector3d(0, 0, 0))).IsFalse();
			await Assert.That(HasPointPositiveDepth(camFromWorldMat, new Vector3d(0, 2, 0))).IsFalse();

			// Infront of camera
			await Assert.That(HasPointPositiveDepth(camFromWorldMat, new Vector3d(0, 0, 1))).IsTrue();

			// Behind camera
			await Assert.That(HasPointPositiveDepth(camFromWorldMat, new Vector3d(0, 0, -1))).IsFalse();
		}
	}
}
