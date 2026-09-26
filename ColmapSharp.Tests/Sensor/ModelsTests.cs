// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ModelsTests: colmap/sensor/models_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, with the same parameters, grids, expected values and
// tolerances. Tests the camera models in ColmapSharp/Sensor. The templated helpers
// (TestModel<CameraModel>, TestCamToCamFromImg, ...) stay generic over the model struct,
// the C# form of COLMAP's CRTP models (CameraModelBase.cs).
//
// gtest's EXPECT_* keep going after a failure; so does this port: the helpers record every
// failed expectation in an ExpectationLog and each test asserts the log is empty at the
// end, so one run reports all failures rather than the first.
// ASSERT_* (which abort the helper) return early from the helper after recording.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class ModelsTests
{
	private static bool FisheyeCameraModelIsValidPixel<TModel>(ReadOnlySpan<double> parameters, Vector2d xy)
		where TModel : struct, IPerspectiveFisheyeCameraModel<TModel>
	{
		TModel.FisheyeFromImg(ColmapSharp.Solver.Real.Cast(parameters), xy.X, xy.Y, out ColmapSharp.Solver.Real ruu, out ColmapSharp.Solver.Real rvv);
		double uu = ruu;
		double vv = rvv;
		double theta = Math.Sqrt(uu * uu + vv * vv);
		return theta < Math.PI / 2.0;
	}

	private static void TestCamToCamFromImg<TModel>(ExpectationLog log, double[] parameters, double u0, double v0, double w0)
		where TModel : struct, ICameraModel<TModel>
	{
		CameraModelMath.ImgFromCam<TModel>(parameters, u0, v0, w0, out double x, out double y);
		Vector2d? xy = CameraModels.CameraModelImgFromCam(TModel.ModelId, parameters, new Vector3d(u0, v0, w0));
		if (!log.True(xy.HasValue, $"ImgFromCam({u0}, {v0}, {w0}) has value"))
		{
			return;
		}

		log.Equal(x, xy!.Value.X, "x");
		log.Equal(y, xy.Value.Y, "y");
		TModel.CamFromImg(parameters, x, y, out double u, out double v);
		log.Near(u, u0 / w0, 1e-6, $"u at ({u0}, {v0}, {w0})");
		log.Near(v, v0 / w0, 1e-6, $"v at ({u0}, {v0}, {w0})");
	}

	private static void TestCamFromImgToImg<TModel>(ExpectationLog log, double[] parameters, double x0, double y0)
		where TModel : struct, ICameraModel<TModel>
	{
		TModel.CamFromImg(parameters, x0, y0, out double u, out double v);
		Vector2d? uv = CameraModels.CameraModelCamFromImg(TModel.ModelId, parameters, new Vector2d(x0, y0));
		if (!log.True(uv.HasValue, $"CamFromImg({x0}, {y0}) has value"))
		{
			return;
		}

		log.Equal(u, uv!.Value.X, "u");
		log.Equal(v, uv.Value.Y, "v");
		foreach (double w in new[] { 0.5, 1.0, 2.0 })
		{
			if (!log.True(CameraModelMath.ImgFromCam<TModel>(parameters, w * u, w * v, w, out double x, out double y), $"ImgFromCam at ({x0}, {y0}), w={w}"))
			{
				return;
			}

			log.Near(x, x0, 1e-6, $"x at ({x0}, {y0}), w={w}");
			log.Near(y, y0, 1e-6, $"y at ({x0}, {y0}), w={w}");
		}
	}

	// Round-trip a pixel through the 3D bearing interface: CamRayFromImg yields a unit ray,
	// ImgFromCam must project it back to the same pixel.
	private static void TestCamRayFromImgToImg<TModel>(ExpectationLog log, double[] parameters, double x0, double y0)
		where TModel : struct, ICameraModel<TModel>
	{
		Vector3d? ray = CameraModels.CameraModelCamRayFromImg(TModel.ModelId, parameters, new Vector2d(x0, y0));
		if (!log.True(ray.HasValue, $"CamRayFromImg({x0}, {y0}) has value"))
		{
			return;
		}

		log.Near(ray!.Value.Norm, 1.0, 1e-12, $"ray norm at ({x0}, {y0})");
		Vector2d? xy = CameraModels.CameraModelImgFromCam(TModel.ModelId, parameters, ray.Value);
		if (!log.True(xy.HasValue, $"ImgFromCam(ray at ({x0}, {y0})) has value"))
		{
			return;
		}

		// The pixel round-trip is floored by the iterative Newton undistortion in CamFromImg
		// (~1e-7 worst case); matches the tolerance of the sibling CamFromImg/ImgFromCam
		// round-trip in TestCamFromImgToImg.
		log.Near(xy!.Value.X, x0, 1e-6, $"ray x at ({x0}, {y0})");
		log.Near(xy.Value.Y, y0, 1e-6, $"ray y at ({x0}, {y0})");
	}

	private static void TestModel<TModel>(ExpectationLog log, double[] parameters)
		where TModel : struct, ICameraModel<TModel>
	{
		CameraModelId id = TModel.ModelId;
		log.True(CameraModels.CameraModelVerifyParams(id, parameters), "VerifyParams(params)");

		double[] defaultParams = CameraModels.CameraModelInitializeParams(id, 100, 100, 100);
		log.True(CameraModels.CameraModelVerifyParams(id, defaultParams), "VerifyParams(default)");

		log.Equal(CameraModels.CameraModelParamsInfo(id), TModel.ParamsInfo, "ParamsInfo");
		log.Equal(CameraModels.CameraModelFocalLengthIdxs(id).ToArray(), TModel.FocalLengthIdxs.ToArray(), "FocalLengthIdxs");
		log.Equal(CameraModels.CameraModelPrincipalPointIdxs(id).ToArray(), TModel.PrincipalPointIdxs.ToArray(), "PrincipalPointIdxs");
		log.Equal(CameraModels.CameraModelExtraParamsIdxs(id).ToArray(), TModel.ExtraParamsIdxs.ToArray(), "ExtraParamsIdxs");
		log.True(CameraModels.CameraModelMetaDataParamsIdxs(id).IsEmpty, "MetaDataParamsIdxs empty");
		log.Equal(CameraModels.CameraModelNumParams(id), TModel.NumParams, "NumParams");

		log.False(CameraModels.CameraModelHasBogusParams(id, defaultParams, 100, 100, 0.1, 2.0, 1.0), "bogus (0.1, 2.0, 1.0)");
		log.True(CameraModels.CameraModelHasBogusParams(id, defaultParams, 100, 100, 0.1, 0.5, 1.0), "bogus (0.1, 0.5, 1.0)");
		log.True(CameraModels.CameraModelHasBogusParams(id, defaultParams, 100, 100, 1.5, 2.0, 1.0), "bogus (1.5, 2.0, 1.0)");
		if (TModel.ExtraParamsIdxs.Length > 0)
		{
			log.True(CameraModels.CameraModelHasBogusParams(id, defaultParams, 100, 100, 0.1, 2.0, -0.1), "bogus (0.1, 2.0, -0.1)");
		}

		log.Equal(CameraModels.CameraModelCamFromImgThreshold(id, parameters, 0), 0.0, "threshold 0");
		log.True(CameraModels.CameraModelCamFromImgThreshold(id, parameters, 1) > 0, "threshold 1 > 0");
		log.Equal(CameraModels.CameraModelCamFromImgThreshold(id, defaultParams, 1), 1.0 / 100.0, "default threshold");

		log.True(CameraModels.ExistsCameraModelWithName(TModel.ModelName), "exists name");
		log.False(CameraModels.ExistsCameraModelWithName(TModel.ModelName + "FOO"), "exists name FOO");

		log.True(CameraModels.ExistsCameraModelWithId(id), "exists id");
		log.False(CameraModels.ExistsCameraModelWithId((CameraModelId)123456789), "exists id 123456789");

		log.Equal(CameraModels.CameraModelNameToId(CameraModels.CameraModelIdToName(id)), id, "name/id round trip");
		log.Equal(CameraModels.CameraModelIdToName(CameraModels.CameraModelNameToId(TModel.ModelName)), TModel.ModelName, "id/name round trip");

		for (double u = -0.5; u <= 0.5; u += 0.1)
		{
			for (double v = -0.5; v <= 0.5; v += 0.1)
			{
				foreach (double w in new[] { 0.5, 1.0, 2.0 })
				{
					TestCamToCamFromImg<TModel>(log, parameters, u, v, w);
				}
			}
		}

		for (int x = 0; x <= 800; x += 50)
		{
			for (int y = 0; y <= 800; y += 50)
			{
				if (CameraModels.CameraModelIsPerspectiveFisheye(id) && !IsValidFisheyePixel<TModel>(parameters, new Vector2d(x, y)))
				{
					continue;
				}

				TestCamFromImgToImg<TModel>(log, parameters, x, y);
				TestCamRayFromImgToImg<TModel>(log, parameters, x, y);
			}
		}

		ReadOnlySpan<int> ppIdxs = TModel.PrincipalPointIdxs;
		TestCamFromImgToImg<TModel>(log, parameters, parameters[ppIdxs[0]], parameters[ppIdxs[1]]);
		TestCamRayFromImgToImg<TModel>(log, parameters, parameters[ppIdxs[0]], parameters[ppIdxs[1]]);

		// Analytic ImgFromCamWithJac is validated separately in models_jacobian_test.cc.
	}

	// C++ FisheyeCameraModelIsValidPixel dispatches by id over the fisheye models; the
	// generic TModel here is already that model, so this only picks the constrained call.
	private static bool IsValidFisheyePixel<TModel>(ReadOnlySpan<double> parameters, Vector2d xy)
		where TModel : struct, ICameraModel<TModel>
	{
		return TModel.ModelId switch
		{
			CameraModelId.SimpleRadialFisheye => FisheyeCameraModelIsValidPixel<SimpleRadialFisheyeCameraModel>(parameters, xy),
			CameraModelId.RadialFisheye => FisheyeCameraModelIsValidPixel<RadialFisheyeCameraModel>(parameters, xy),
			CameraModelId.OpenCVFisheye => FisheyeCameraModelIsValidPixel<OpenCVFisheyeCameraModel>(parameters, xy),
			CameraModelId.ThinPrismFisheye => FisheyeCameraModelIsValidPixel<ThinPrismFisheyeCameraModel>(parameters, xy),
			CameraModelId.RadTanThinPrismFisheye => FisheyeCameraModelIsValidPixel<RadTanThinPrismFisheyeModel>(parameters, xy),
			CameraModelId.SimpleFisheye => FisheyeCameraModelIsValidPixel<SimpleFisheyeCameraModel>(parameters, xy),
			CameraModelId.Fisheye => FisheyeCameraModelIsValidPixel<FisheyeCameraModel>(parameters, xy),
			_ => throw new ArgumentException("Camera model does not exist or is not a fisheye camera"),
		};
	}

	private static async Task Run(Action<ExpectationLog> body)
	{
		var log = new ExpectationLog();
		body(log);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public Task SimplePinhole_Nominal() =>
		Run(log => TestModel<SimplePinholeCameraModel>(log, [655.123, 386.123, 511.123]));

	[Test]
	public Task Pinhole_Nominal() =>
		Run(log => TestModel<PinholeCameraModel>(log, [651.123, 655.123, 386.123, 511.123]));

	[Test]
	public Task Spherical_Nominal() => Run(log =>
	{
		// params = (w, h) of the equirectangular image.
		double[] parameters = [800, 400];
		CameraModelId id = EquirectangularCameraModel.ModelId;
		log.True(CameraModels.CameraModelVerifyParams(id, parameters), "VerifyParams");

		log.Equal(CameraModels.CameraModelParamsInfo(id), "w,h", "ParamsInfo");
		log.True(CameraModels.CameraModelFocalLengthIdxs(id).IsEmpty, "FocalLengthIdxs empty");
		log.True(CameraModels.CameraModelPrincipalPointIdxs(id).IsEmpty, "PrincipalPointIdxs empty");
		log.True(CameraModels.CameraModelExtraParamsIdxs(id).IsEmpty, "ExtraParamsIdxs empty");
		log.Equal(CameraModels.CameraModelMetaDataParamsIdxs(id).ToArray(), new[] { 0, 1 }, "MetaDataParamsIdxs");
		log.Equal(CameraModels.CameraModelNumParams(id), 2, "NumParams");

		// Perspective models have no metadata parameters.
		log.True(CameraModels.CameraModelMetaDataParamsIdxs(PinholeCameraModel.ModelId).IsEmpty, "pinhole metadata empty");

		// EQUIRECTANGULAR is non-perspective, spherical, and never has bogus parameters.
		log.False(CameraModels.CameraModelIsPerspective(id), "equirect perspective");
		log.True(CameraModels.CameraModelIsPerspective(PinholeCameraModel.ModelId), "pinhole perspective");
		log.True(CameraModels.CameraModelIsSpherical(id), "equirect spherical");
		log.False(CameraModels.CameraModelIsSpherical(PinholeCameraModel.ModelId), "pinhole spherical");
		log.False(CameraModels.CameraModelIsSpherical(OpenCVFisheyeCameraModel.ModelId), "opencv fisheye spherical");
		log.False(CameraModels.CameraModelHasBogusParams(id, parameters, 800, 400, 0.1, 2.0, 1.0), "bogus");

		// InitializeParams ignores the focal length and returns (w, h).
		log.Equal(CameraModels.CameraModelInitializeParams(id, focalLength: 123, 800, 400), parameters, "InitializeParams");

		// Full-sphere bearing round-trip CamRayFromImg -> ImgFromCam over the image interior
		// (avoiding the azimuth seam at x in {0, w} and the poles at y in {0, h}, where the
		// azimuth is undefined).
		for (int xi = 40; xi <= 760; xi += 40)
		{
			for (int yi = 40; yi <= 360; yi += 40)
			{
				TestCamRayFromImgToImg<EquirectangularCameraModel>(log, parameters, xi, yi);
			}
		}

		// Back-hemisphere pixels (azimuth near +/-pi) have no forward 2D representation, so
		// the 2D CamFromImg fails there while CamRayFromImg still yields a valid unit bearing.
		log.False(CameraModels.CameraModelCamFromImg(id, parameters, new Vector2d(0, 200)).HasValue, "back CamFromImg");
		log.True(CameraModels.CameraModelCamRayFromImg(id, parameters, new Vector2d(0, 200)).HasValue, "back CamRayFromImg");

		// A forward-hemisphere pixel (azimuth ~0, image center column) also round-trips
		// through the 2D CamFromImg / ImgFromCam path.
		TestCamFromImgToImg<EquirectangularCameraModel>(log, parameters, 400, 200);
	});

	[Test]
	public Task SimpleRadial_Nominal() => Run(log =>
	{
		TestModel<SimpleRadialCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModel<SimpleRadialCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
	});

	[Test]
	public Task Radial_Nominal() => Run(log =>
	{
		TestModel<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0, 0]);
		TestModel<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0.1, 0]);
		TestModel<RadialCameraModel>(log, [651.123, 386.123, 511.12, 0, 0.05]);
		TestModel<RadialCameraModel>(log, [651.123, 386.123, 511.123, 0.05, 0.03]);
	});

	[Test]
	public Task OpenCV_Nominal() =>
		Run(log => TestModel<OpenCVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001]));

	[Test]
	public Task OpenCVFisheye_Nominal() =>
		Run(log => TestModel<OpenCVFisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001]));

	[Test]
	public Task FullOpenCV_Nominal() => Run(log => TestModel<FullOpenCVCameraModel>(
		log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001]));

	[Test]
	public Task FOV_Nominal() => Run(log =>
	{
		TestModel<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0]);
		TestModel<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.9]);
		TestModel<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 1e-6]);
		TestModel<FOVCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 1e-2]);
		log.Equal(CameraModels.CameraModelInitializeParams(FOVCameraModel.ModelId, 100, 100, 100)[^1], 1e-2, "FOV default omega");
	});

	[Test]
	public Task SimpleRadialFisheye_Nominal() => Run(log =>
	{
		TestModel<SimpleRadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModel<SimpleRadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
	});

	[Test]
	public Task RadialFisheye_Nominal() => Run(log =>
	{
		TestModel<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0, 0]);
		TestModel<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0, 0.1]);
		TestModel<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0, 0.05]);
		TestModel<RadialFisheyeCameraModel>(log, [651.123, 386.123, 511.123, 0, 0.03]);
	});

	[Test]
	public Task ThinPrismFisheye_Nominal() => Run(log => TestModel<ThinPrismFisheyeCameraModel>(
		log, [651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001]));

	[Test]
	public Task RadTanThinPrismFisheye_Nominal() => Run(log => TestModel<RadTanThinPrismFisheyeModel>(
		log,
		[
			651.123, 655.123, 386.123, 511.123, -0.0232, 0.0924, -0.0591, 0.003, 0.0048, -0.0009, 0.0002, 0.0005,
			-0.0009, -0.0001, 0.00007, -0.00017,
		]));

	[Test]
	public Task SimpleDivision_Nominal() => Run(log =>
	{
		TestModel<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, 0]);
		TestModel<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, 0.1]);
		TestModel<SimpleDivisionCameraModel>(log, [651.123, 386.123, 511.123, -0.1]);
	});

	[Test]
	public Task Division_Nominal() => Run(log =>
	{
		TestModel<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0]);
		TestModel<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.1]);
		TestModel<DivisionCameraModel>(log, [651.123, 655.123, 386.123, 511.123, -0.1]);
	});

	[Test]
	public Task SimpleFisheyeCamera_Nominal() =>
		Run(log => TestModel<SimpleFisheyeCameraModel>(log, [651.123, 386.123, 511.123]));

	[Test]
	public Task FisheyeCamera_Nominal() =>
		Run(log => TestModel<FisheyeCameraModel>(log, [651.123, 655.123, 386.123, 511.123]));

	[Test]
	public Task EUCMCamera_Nominal() => Run(log =>
	{
		TestModel<EUCMCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.56, 0.87]);
		TestModel<EUCMCameraModel>(log, [400, 400, 400, 400, 0.88, 0.64]);
		TestModel<EUCMCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.0, 1.0]);
		TestModel<EUCMCameraModel>(log, [651.123, 655.123, 386.123, 511.123, 0.5, 1.0]);
	});

	[Test]
	public Task EUCMCamera_RejectsInvalidExtraParams() => Run(log =>
	{
		CameraModelId id = EUCMCameraModel.ModelId;
		log.True(CameraModels.CameraModelHasBogusParams(id, [651.123, 655.123, 386.123, 511.123, -0.01, 0.87], 1000, 1000, 0.1, 2.0, 1.0), "alpha < 0");
		log.True(CameraModels.CameraModelHasBogusParams(id, [651.123, 655.123, 386.123, 511.123, 1.01, 0.87], 1000, 1000, 0.1, 2.0, 1.0), "alpha > 1");
		log.True(CameraModels.CameraModelHasBogusParams(id, [651.123, 655.123, 386.123, 511.123, 0.56, 0.00], 1000, 1000, 0.1, 2.0, 1.0), "beta == 0");
		log.True(CameraModels.CameraModelHasBogusParams(id, [651.123, 655.123, 386.123, 511.123, 0.56, -0.3], 1000, 1000, 0.1, 2.0, 1.0), "beta < 0");
	});

	[Test]
	public async Task CameraModelRescale_Perspective()
	{
		// Distinct per-axis scale factors to verify each is applied to the right parameter;
		// all results are exactly representable.
		const double scaleX = 2.0;
		const double scaleY = 3.0;

		// Two focal lengths (fx, fy): each scales along its own axis, as does the principal
		// point (cx, cy).
		double[] pinhole = [100, 200, 50, 80]; // fx, fy, cx, cy
		CameraModels.CameraModelRescale(PinholeCameraModel.ModelId, scaleX, scaleY, pinhole);
		await Assert.That(pinhole.SequenceEqual(new double[] { 200, 600, 100, 240 })).IsTrue();

		// Single shared focal length scales by the mean of the two factors.
		double[] simplePinhole = [100, 50, 80]; // f, cx, cy
		CameraModels.CameraModelRescale(SimplePinholeCameraModel.ModelId, scaleX, scaleY, simplePinhole);
		await Assert.That(simplePinhole.SequenceEqual(new double[] { 250, 100, 240 })).IsTrue(); // f *= 2.5

		// Extra (distortion) parameters are resolution independent and untouched.
		double[] simpleRadial = [100, 50, 80, 0.3]; // f, cx, cy, k
		CameraModels.CameraModelRescale(SimpleRadialCameraModel.ModelId, scaleX, scaleY, simpleRadial);
		await Assert.That(simpleRadial.SequenceEqual(new double[] { 250, 100, 240, 0.3 })).IsTrue();
	}

	[Test]
	public async Task CameraModelRescale_Spherical()
	{
		// The (w, h) image-size parameters track the rescaled image dimensions.
		double[] parameters = [800, 400]; // w, h
		CameraModels.CameraModelRescale(EquirectangularCameraModel.ModelId, 2.0, 0.5, parameters);
		await Assert.That(parameters.SequenceEqual(new double[] { 1600, 200 })).IsTrue();
	}
}
