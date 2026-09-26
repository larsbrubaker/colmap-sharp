// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ModelsJacobianTests: colmap/sensor/models_jacobian_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, with the same parameters, grids and tolerances. Tests
// the analytic ImgFromCamWithJac kernels (ColmapSharp/Sensor/*CameraModels.Jacobian.cs),
// CameraModels.CameraModelImgFromCamWithJac and CameraModels.CamRayFromImgJacobian.
//
// COLMAP checks the analytic Jacobians against ImgFromCam evaluated on
// ceres::Jet<double, num_params + 3>; here that is Jet<TGrad> with the same width
// (Solver/JetGradients.cs), in one pass.
//
// Failed expectations are collected in an ExpectationLog (see ModelsTests.cs), so one run
// reports all of them; ASSERT_* return early from the helper after recording.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class ModelsJacobianTests
{
	/// <summary>
	/// ImgFromCam differentiated by forward-mode autodiff: derivatives with respect to the
	/// params first, then u, v, w (the seeding of COLMAP's TestImgFromCamWithJac), in one
	/// pass of a Jet exactly num_params + 3 wide, as COLMAP's
	/// ceres::Jet&lt;double, num_params + 3&gt;.
	/// </summary>
	private static bool AutodiffImgFromCam<TModel>(double[] parameters, double u, double v, double w, out double x, out double y, double[] dx, double[] dy)
		where TModel : struct, ICameraModel<TModel>
	{
		return (parameters.Length + 3) switch
		{
			5 => AutodiffImgFromCam<TModel, Grad5>(parameters, u, v, w, out x, out y, dx, dy),
			6 => AutodiffImgFromCam<TModel, Grad6>(parameters, u, v, w, out x, out y, dx, dy),
			7 => AutodiffImgFromCam<TModel, Grad7>(parameters, u, v, w, out x, out y, dx, dy),
			8 => AutodiffImgFromCam<TModel, Grad8>(parameters, u, v, w, out x, out y, dx, dy),
			9 => AutodiffImgFromCam<TModel, Grad9>(parameters, u, v, w, out x, out y, dx, dy),
			11 => AutodiffImgFromCam<TModel, Grad11>(parameters, u, v, w, out x, out y, dx, dy),
			15 => AutodiffImgFromCam<TModel, Grad15>(parameters, u, v, w, out x, out y, dx, dy),
			19 => AutodiffImgFromCam<TModel, Grad19>(parameters, u, v, w, out x, out y, dx, dy),
			_ => throw new ArgumentOutOfRangeException(nameof(parameters), $"no Jet width for {parameters.Length} params"),
		};
	}

	private static bool AutodiffImgFromCam<TModel, TGrad>(double[] parameters, double u, double v, double w, out double x, out double y, double[] dx, double[] dy)
		where TModel : struct, ICameraModel<TModel>
		where TGrad : unmanaged, IJetGradient
	{
		int numParams = parameters.Length;
		int numDerivs = numParams + 3;
		double[] values = [.. parameters, u, v, w];
		var jets = new Jet<TGrad>[numDerivs];
		for (int i = 0; i < numDerivs; i++)
		{
			jets[i] = Jet<TGrad>.Variable(values[i], i);
		}

		x = 0;
		y = 0;
		if (!TModel.ImgFromCam<Jet<TGrad>>(jets.AsSpan(0, numParams), jets[numParams], jets[numParams + 1], jets[numParams + 2], out Jet<TGrad> xJet, out Jet<TGrad> yJet))
		{
			return false;
		}

		x = xJet.A;
		y = yJet.A;
		for (int i = 0; i < numDerivs; i++)
		{
			dx[i] = xJet.Derivative(i);
			dy[i] = yJet.Derivative(i);
		}

		return true;
	}

	// Validate ImgFromCamWithJac against ImgFromCam using Jets.
	private static void TestImgFromCamWithJac<TModel>(ExpectationLog log, double[] parameters, double u, double v, double w)
		where TModel : struct, ICameraModel<TModel>
	{
		int kNumParams = TModel.NumParams;
		const int kNumUvw = 3;
		int kNumDerivs = kNumParams + kNumUvw;
		string at = $"({u}, {v}, {w})";

		// Compute using ImgFromCamWithJac
		double[] jParams = new double[2 * kNumParams];
		double[] jUvw = new double[2 * kNumUvw];
		if (!log.True(TModel.ImgFromCamWithJac(parameters, u, v, w, out double xJac, out double yJac, jParams, jUvw), $"ImgFromCamWithJac {at}"))
		{
			return;
		}

		// Compute using ImgFromCam with Jets for auto-differentiation
		// Jets track derivatives: first kNumParams for params, next 3 for u, v, w.
		double[] xV = new double[kNumDerivs];
		double[] yV = new double[kNumDerivs];
		if (!log.True(AutodiffImgFromCam<TModel>(parameters, u, v, w, out double xJet, out double yJet, xV, yV), $"ImgFromCam<Jet> {at}"))
		{
			return;
		}

		// Compare function values
		log.Near(xJac, xJet, 1e-10, $"x {at}");
		log.Near(yJac, yJet, 1e-10, $"y {at}");

		// Compare Jacobian w.r.t. params (2 x num_params, row-major)
		for (int i = 0; i < kNumParams; ++i)
		{
			log.Near(jParams[i], xV[i], 1e-10, $"J_params mismatch at dx/dparam[{i}] {at}");
			log.Near(jParams[kNumParams + i], yV[i], 1e-10, $"J_params mismatch at dy/dparam[{i}] {at}");
		}

		// Compare Jacobian w.r.t. uvw (2 x 3, row-major)
		for (int i = 0; i < kNumUvw; ++i)
		{
			log.Near(jUvw[i], xV[kNumParams + i], 1e-10, $"J_uvw mismatch at dx/d(uvw)[{i}] {at}");
			log.Near(jUvw[kNumUvw + i], yV[kNumParams + i], 1e-10, $"J_uvw mismatch at dy/d(uvw)[{i}] {at}");
		}
	}

	// Validate the runtime dispatch and the unprojection Jacobian derived from it.
	private static void TestCamRayJacobian<TModel>(ExpectationLog log, double[] parameters, double u, double v, double w)
		where TModel : struct, ICameraModel<TModel>
	{
		var uvw = new Vector3d(u, v, w);
		string at = $"({u}, {v}, {w})";

		// Reference: the templated per-model kernel, written 2x3 row-major.
		double[] jRefData = new double[6];
		if (!log.True(TModel.ImgFromCamWithJac(parameters, u, v, w, out double xRef, out double yRef, default, jRefData), $"ImgFromCamWithJac {at}"))
		{
			return;
		}

		Matrix2x3d jRef = Matrix2x3d.FromRowMajor(jRefData);

		// 1. The runtime dispatch must agree with the templated kernel. The compiler
		// may round the separately optimized call paths slightly differently.
		Vector2d? xy = CameraModels.CameraModelImgFromCamWithJac(TModel.ModelId, parameters, uvw, out Matrix2x3d jUvw);
		if (!log.True(xy.HasValue, $"dispatch has value {at}"))
		{
			return;
		}

		log.Near(xy!.Value.X, xRef, 1e-10, $"dispatch x {at}");
		log.Near(xy.Value.Y, yRef, 1e-10, $"dispatch y {at}");
		log.True(jUvw.IsApprox(jRef, 1e-12), $"dispatch J_uvw isApprox {at}");

		// Passing nullptr must skip the Jacobian but still project.
		Vector2d? xyNoJac = CameraModels.CameraModelImgFromCamWithJac(TModel.ModelId, parameters, uvw);
		if (!log.True(xyNoJac.HasValue, $"dispatch without Jacobian has value {at}"))
		{
			return;
		}

		log.True(xyNoJac!.Value.IsApprox(xy.Value, 1e-12), $"dispatch without Jacobian isApprox {at}");

		// 2. Central projection depends only on the ray direction, so the projection
		// is homogeneous of degree zero and Euler's identity gives J_uvw * uvw == 0.
		// This is the assumption that makes the pseudo-inverse below equal the
		// tangent-plane unprojection Jacobian; if it fails, that derivation is wrong.
		log.True((jUvw * uvw).Norm <= 1e-10 * jUvw.Norm() * uvw.Norm, $"J_uvw * uvw == 0 {at}");

		// The closed-form pseudo-inverse is only valid at a unit bearing.
		Vector3d camRay = uvw.Normalized();
		Matrix3x2d? jRay = CameraModels.CamRayFromImgJacobian(camRay, jUvw);
		if (!log.True(jRay.HasValue, $"CamRayFromImgJacobian has value {at}"))
		{
			return;
		}

		// 3. Pseudo-inverse round trip: J_uvw is surjective onto image space.
		log.True((jUvw * jRay!.Value - Matrix2d.Identity).Norm() <= 1e-10, $"J_uvw * J_ray == I {at}");

		// 4. The recovered Jacobian maps into the tangent plane at the ray.
		log.True((jRay.Value.Transpose() * uvw).Norm <= 1e-10 * jRay.Value.Norm() * uvw.Norm, $"J_ray^T * uvw == 0 {at}");
		log.True((jRay.Value.Transpose() * camRay).Norm <= 1e-10 * jRay.Value.Norm(), $"J_ray^T * cam_ray == 0 {at}");
	}

	// Validate the analytic ImgFromCamWithJac over a grid of camera-space points.
	private static void TestModelImgFromCamWithJac<TModel>(ExpectationLog log, double[] parameters)
		where TModel : struct, ICameraModel<TModel>
	{
		// The float loop counters are COLMAP's: the grid is 11 x 11 values that drift from
		// multiples of 0.1 exactly as the C++ ones do.
		for (double u = -0.5; u <= 0.5; u += 0.1)
		{
			for (double v = -0.5; v <= 0.5; v += 0.1)
			{
				foreach (double w in (double[])[0.5, 1.0, 2.0])
				{
					TestImgFromCamWithJac<TModel>(log, parameters, u, v, w);
					TestCamRayJacobian<TModel>(log, parameters, u, v, w);
				}
			}
		}
	}

	private static async Task Run(Action<ExpectationLog> body)
	{
		var log = new ExpectationLog();
		body(log);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public Task SimplePinhole_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<SimplePinholeCameraModel>(log, [655.123, 386.123, 511.123]));

	[Test]
	public Task Pinhole_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<PinholeCameraModel>(log, [651.123, 655.123, 386.123, 511.123]));

	[Test]
	public Task SimpleRadial_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<SimpleRadialCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModelImgFromCamWithJac<SimpleRadialCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
	});

	[Test]
	public Task Radial_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0, 0]);
		TestModelImgFromCamWithJac<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0.1, 0]);
		TestModelImgFromCamWithJac<RadialCameraModel>(log, [651.123, 386.123, 511.12, 0, 0.05]);
		TestModelImgFromCamWithJac<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0.05, 0.03]);
	});

	[Test]
	public Task OpenCV_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<OpenCVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001]));

	[Test]
	public Task FullOpenCV_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<FullOpenCVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001]));

	[Test]
	public Task FOV_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.9]);
		TestModelImgFromCamWithJac<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.5]);
	});

	[Test]
	public Task SimpleRadialFisheye_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<SimpleRadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModelImgFromCamWithJac<SimpleRadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
	});

	[Test]
	public Task RadialFisheye_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0, 0]);
		TestModelImgFromCamWithJac<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0.1, 0.02]);
	});

	[Test]
	public Task OpenCVFisheye_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<OpenCVFisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0]);
		TestModelImgFromCamWithJac<OpenCVFisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001]);
	});

	[Test]
	public Task ThinPrismFisheye_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<ThinPrismFisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0, 0, 0, 0, 0]);
		TestModelImgFromCamWithJac<ThinPrismFisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001, 0.001, 0.002, 0.001, -0.001]);
	});

	[Test]
	public Task RadTanThinPrismFisheye_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<RadTanThinPrismFisheyeModel>(log, [651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
		TestModelImgFromCamWithJac<RadTanThinPrismFisheyeModel>(log, [651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.005, 0.001, 0.0005, 0.0002, -0.001, 0.001, 0.001, -0.001, 0.0005, -0.0005]);
	});

	[Test]
	public Task SimpleFisheye_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<SimpleFisheyeCameraModel>(log, [651.123, 386.123, 511.123]));

	[Test]
	public Task Fisheye_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<FisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123]));

	[Test]
	public Task SimpleDivision_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
		TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, -0.1]);
	});

	[Test]
	public Task Division_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0]);
		TestModelImgFromCamWithJac<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.1]);
		TestModelImgFromCamWithJac<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.1]);
	});

	[Test]
	public Task EUCM_ImgFromCamWithJac() => Run(log =>
	{
		TestModelImgFromCamWithJac<EUCMCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.0, 1.0]);
		TestModelImgFromCamWithJac<EUCMCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.6, 1.2]);
	});

	[Test]
	public Task Equirectangular_ImgFromCamWithJac() => Run(log =>
		TestModelImgFromCamWithJac<EquirectangularCameraModel>(log, [1000, 500]));

	[Test]
	public Task CamRayFromImgJacobian_RankDeficientReturnsNullopt() => Run(log =>
	{
		// Rank 1: both image directions respond identically, so the projection is
		// not locally invertible and there is no unprojection Jacobian.
		var camRay = new Vector3d(0.0, 0.0, 1.0);
		var rank1 = new Matrix2x3d(1.0, 2.0, 3.0, 2.0, 4.0, 6.0);
		log.False(CameraModels.CamRayFromImgJacobian(camRay, rank1).HasValue, "rank 1");

		log.False(CameraModels.CamRayFromImgJacobian(camRay, Matrix2x3d.Zero).HasValue, "zero");

		// A well-conditioned Jacobian is accepted and inverts cleanly.
		var fullRank = new Matrix2x3d(100.0, 0.0, 0.0, 0.0, 100.0, 0.0);
		Matrix3x2d? jRay = CameraModels.CamRayFromImgJacobian(camRay, fullRank);
		if (!log.True(jRay.HasValue, "full rank has value"))
		{
			return;
		}

		log.True((fullRank * jRay!.Value - Matrix2d.Identity).Norm() <= 1e-12, "full rank round trip");
	});
}
