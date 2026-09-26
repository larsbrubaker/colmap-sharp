// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AbsolutePoseTests: colmap/estimators/solvers/absolute_pose_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/Solvers/AbsolutePose.cs and Epnp.cs, and through them PoseLib's
// P3P and P4Pf (Estimators/Solvers/PoseLib/). The solvers are Tier B and the RANSAC runs
// Tier C; COLMAP's tolerances are the bars.
//
// gtest's EXPECT_* keep going after a failure; so does this port: every failed expectation
// goes to an ExpectationLog and each test asserts the log is empty at the end.
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each RANSAC test starts
// with RandomUtils.SetPRNGSeed(0) and runs synchronously up to its final assertion.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class AbsolutePoseTests
{
	private static Vector3d[] Points3D() =>
	[
		new(1, 1, 1),
		new(0, 1, 1),
		new(3, 1.0, 4),
		new(3, 1.1, 4),
		new(3, 1.2, 4),
		new(3, 1.3, 4),
		new(3, 1.4, 4),
		new(2, 1, 7),
	];

	private static Vector3d[] Faulty(Vector3d[] points3D) =>
		points3D.Select(p => new Vector3d(20, p.Y, p.Z)).ToArray();

	private static void CheckResiduals(ExpectationLog log, double[] residuals, bool exact, string what)
	{
		for (int i = 0; i < residuals.Length; ++i)
		{
			if (exact)
			{
				log.True(residuals[i] < 1e-3, $"{what}: residual[{i}] = {residuals[i]} < 1e-3");
			}
			else
			{
				log.True(residuals[i] > 0.1, $"{what}: faulty residual[{i}] = {residuals[i]} > 0.1");
			}
		}
	}

	private static void CheckPnpRansac<TEstimator>(ExpectationLog log, TEstimator estimator, ImgFromCamFunc imgFromCamFunc, double maxError, double poseTol)
		where TEstimator : IEstimator<Point2DWithRay, Vector3d, Matrix3x4d>
	{
		Vector3d[] points3D = Points3D();
		Vector3d[] points3DFaulty = Faulty(points3D);

		// NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
		for (double qx = 0; qx < 1; qx += 0.2)
		{
			// NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
			for (double tx = 0; tx < 1; tx += 0.1)
			{
				var expectedCamFromWorld = new Rigid3d(new Quaterniond(1, qx, 0, 0).Normalized(), new Vector3d(tx, 0, 0));

				// Project points to camera coordinate system.
				var points2D = new Point2DWithRay[points3D.Length];
				for (int i = 0; i < points3D.Length; ++i)
				{
					Vector3d ray = (expectedCamFromWorld * points3D[i]).Normalized();
					points2D[i] = new Point2DWithRay(imgFromCamFunc(ray)!.Value, ray);
				}

				var options = new RansacOptions { MaxError = maxError };
				var ransac = new Ransac<TEstimator, Point2DWithRay, Vector3d, Matrix3x4d>(options, estimator);
				var report = ransac.Estimate(points2D, points3D);

				string what = $"qx={qx} tx={tx}";
				log.True(report.Success, $"{what}: success");
				double err = (expectedCamFromWorld.ToMatrix() - report.Model).Norm();
				log.True(err < poseTol, $"{what}: pose error {err} < {poseTol}");

				// Test residuals of exact points.
				var residuals = new double[points2D.Length];
				ransac.Estimator.Residuals(points2D, points3D, report.Model, residuals);
				CheckResiduals(log, residuals, exact: true, what);

				// Test residuals of faulty points.
				ransac.Estimator.Residuals(points2D, points3DFaulty, report.Model, residuals);
				CheckResiduals(log, residuals, exact: false, what);
			}
		}
	}

	private static ImgFromCamFunc PinholeImgFromCam()
	{
		Camera camera = Camera.CreateFromModelId(Types.InvalidCameraId, CameraModelId.Pinhole, 12, 34, 56);
		return camPoint => camera.ImgFromCam(camPoint);
	}

	[Test]
	public async Task AbsolutePose_P3P()
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		ImgFromCamFunc imgFromCamFunc = PinholeImgFromCam();
		CheckPnpRansac(log, new P3PEstimator(imgFromCamFunc), imgFromCamFunc, maxError: 1e-3, poseTol: 1e-5);
		await Assert.That(log.Failures).IsEmpty();
	}

	private static void CheckP4pf(ExpectationLog log, bool shareFocalLength)
	{
		Vector3d[] points3D = Points3D();
		Vector3d[] points3DFaulty = Faulty(points3D);

		// NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
		for (double qx = 0; qx < 1; qx += 0.2)
		{
			// NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
			for (double tx = 0; tx < 1; tx += 0.1)
			{
				// NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
				for (double f = 0.5; f < 20; f += 2)
				{
					Vector2d focalLengths = shareFocalLength ? new Vector2d(f, f) : new Vector2d(f, 1.5 * f);
					var expectedCamFromWorld = new Rigid3d(new Quaterniond(1, qx, 0, 0).Normalized(), new Vector3d(tx, 0, 0));

					// Project points to camera coordinate system.
					var points2D = new Vector2d[points3D.Length];
					for (int i = 0; i < points3D.Length; ++i)
					{
						Vector2d normalized = (expectedCamFromWorld * points3D[i]).HNormalized();
						// The shared test scales by the scalar f, the separate one component-wise.
						points2D[i] = shareFocalLength ? f * normalized : focalLengths.CwiseProduct(normalized);
					}

					var options = new RansacOptions { MaxError = 1e-5 };
					var ransac = new Ransac<P4PFEstimator, Vector2d, Vector3d, P4PFModel>(options, new P4PFEstimator(shareFocalLength));
					var report = ransac.Estimate(points2D, points3D);

					string what = $"qx={qx} tx={tx} f={f}";
					log.True(report.Success, $"{what}: success");
					if (shareFocalLength)
					{
						// In shared mode both focal lengths must be identical.
						log.Equal(report.Model.FocalLengths.X, report.Model.FocalLengths.Y, $"{what}: fx == fy");
						log.Near(report.Model.FocalLengths.X, f, 1e-3, $"{what}: focal");
					}
					else
					{
						log.Near(report.Model.FocalLengths.X, focalLengths.X, 1e-3, $"{what}: fx");
						log.Near(report.Model.FocalLengths.Y, focalLengths.Y, 1e-3, $"{what}: fy");
					}

					double err = (expectedCamFromWorld.ToMatrix() - report.Model.CamFromWorld).Norm();
					log.True(err < 1e-3, $"{what}: pose error {err} < 1e-3");

					// Test residuals of exact points.
					var residuals = new double[points2D.Length];
					new P4PFEstimator().Residuals(points2D, points3D, report.Model, residuals);
					CheckResiduals(log, residuals, exact: true, what);

					// Test residuals of faulty points.
					new P4PFEstimator().Residuals(points2D, points3DFaulty, report.Model, residuals);
					CheckResiduals(log, residuals, exact: false, what);
				}
			}
		}
	}

	[Test]
	public async Task AbsolutePose_P4PFSharedFocalLength()
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		CheckP4pf(log, shareFocalLength: true);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task AbsolutePose_P4PFSeparateFocalLengths()
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		CheckP4pf(log, shareFocalLength: false);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task AbsolutePose_EPNP()
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		ImgFromCamFunc imgFromCamFunc = PinholeImgFromCam();
		CheckPnpRansac(log, new EPNPEstimator(imgFromCamFunc), imgFromCamFunc, maxError: 1e-5, poseTol: 1e-3);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task AbsolutePose_EPNP_BrokenSolveSignCase()
	{
		Vector2d[] imagePoints =
		[
			new(-2.6783007931074532e-01, 5.3457197430746251e-01),
			new(-4.2629907287470264e-01, 7.5623350319519789e-01),
			new(-1.6767413005963930e-01, -1.3387172544910089e-01),
			new(-5.6616329720373559e-02, 2.3621156497739373e-01),
			new(-1.7721225948969935e-01, 2.3395366792735982e-02),
			new(-5.1836259886632222e-02, -4.4380694271927049e-02),
			new(-3.5897765845560037e-01, 1.6252721078589397e-01),
			new(2.7057324473684058e-01, -1.4067450104631887e-01),
			new(-2.5811166424334520e-01, 8.0167171300227366e-02),
			new(2.0239567448222310e-02, -3.2845953375344145e-01),
			new(4.2571014715170657e-01, -2.8321173570154773e-01),
			new(-5.4597596412987237e-01, 9.1431935871671977e-02),
		];

		var points2D = new Point2DWithRay[imagePoints.Length];
		for (int i = 0; i < imagePoints.Length; ++i)
		{
			points2D[i] = new Point2DWithRay(imagePoints[i], imagePoints[i].Homogeneous().Normalized());
		}

		Vector3d[] points3D =
		[
			new(4.4276865308679305e+00, -1.3384364366019632e+00, -3.5997423085253892e+00),
			new(2.7278555252512309e+00, -3.8152996187231392e-01, -2.6558518399902824e+00),
			new(4.8548566083054894e+00, -1.4756197433631739e+00, -6.8274946022490501e-01),
			new(3.1523013527998449e+00, -1.3377020437938025e+00, -1.6443269301929087e+00),
			new(3.8551679771512073e+00, -1.0557700545885551e+00, -1.1695994508851486e+00),
			new(5.9571373150353812e+00, -2.6120646101684555e+00, -1.0841441206050342e+00),
			new(6.3287088499358894e+00, -1.1761274755817175e+00, -2.5951879774151583e+00),
			new(2.3005305990121250e+00, -1.4019796626800123e+00, -4.4485464455072321e-01),
			new(5.9816859934587354e+00, -1.4211814511691452e+00, -2.0285923889293449e+00),
			new(5.2543344690665457e+00, -2.3389255564264144e+00, 4.3708173185524052e-01),
			new(3.2181599245991688e+00, -2.8906671988445098e+00, 2.6825718150064348e-01),
			new(4.4592895306946758e+00, -9.1235241641579902e-03, -1.6555237117970871e+00),
		];

		var models = new List<Matrix3x4d>();
		var estimator = new EPNPEstimator(point3DInCam => point3DInCam.HNormalized());
		estimator.Estimate(points2D, points3D, models);

		await Assert.That(models.Count).IsEqualTo(1);

		double reproj = 0.0;
		for (int i = 0; i < points3D.Length; ++i)
		{
			reproj += ((models[0] * points3D[i].Homogeneous()).HNormalized() - points2D[i].ImagePoint).Norm;
		}

		await Assert.That(reproj < 0.2).IsTrue();
	}

	[Test]
	public async Task ComputeSquaredReprojectionError_Nominal()
	{
		Camera camera = Camera.CreateFromModelId(Types.InvalidCameraId, CameraModelId.SimplePinhole, 12, 34, 56);
		ImgFromCamFunc imgFromCamFunc = camPoint => camera.ImgFromCam(camPoint);

		Vector3d[] points3D = [new(-1, 0, 1), new(-1, 1, 1), new(0, 0, -1), new(0, 0, 0)];

		Point2DWithRay[] points2D =
		[
			new(new Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY()), Vector3d.Zero),
			new(new Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY()), Vector3d.Zero),
			new(Vector2d.Zero, Vector3d.Zero),
			new(Vector2d.Zero, Vector3d.Zero),
		];

		var camFromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));

		var residuals = new double[points2D.Length];
		AbsolutePose.ComputeSquaredReprojectionError(points2D, points3D, camFromWorld.ToMatrix(), imgFromCamFunc, residuals);

		double[] expected = [0, camera.FocalLength() * camera.FocalLength(), double.MaxValue, double.MaxValue];
		// testing::ElementsAre: same values in the same order.
		await Assert.That(residuals.SequenceEqual(expected)).IsTrue();
	}
}
