// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraTests: colmap/scene/camera_test.cc ported 1:1, one method per gtest TEST(Suite,
// Name) named Suite_Name, testing ColmapSharp/Scene/Camera.cs. COLMAP's std::domain_error
// for an unknown camera model is ArgumentException here (Sensor/CameraModels.cs); C++'s
// `Camera other = camera;` is Clone(); operator<< is ToString(). The span-returning
// index properties are read inside an Action, since a lambda cannot return a span.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class CameraTests
{
	[Test]
	public async Task Camera_Empty()
	{
		var camera = new Camera();
		using (Assert.Multiple())
		{
			await Assert.That(camera.CameraId).IsEqualTo(InvalidCameraId);
			await Assert.That(camera.SensorId).IsEqualTo(new SensorId(SensorType.Camera, InvalidCameraId));
			await Assert.That(camera.ModelId).IsEqualTo(CameraModelId.Invalid);
			await Assert.That(camera.ModelName).IsEqualTo("");
			await Assert.That(camera.Width).IsEqualTo(0);
			await Assert.That(camera.Height).IsEqualTo(0);
			await Assert.That(camera.HasPriorFocalLength).IsFalse();
			await Assert.That(() => { _ = camera.FocalLengthIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.PrincipalPointIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.ExtraParamsIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.MetaDataParamsIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => camera.ParamsInfo).Throws<ArgumentException>();
			await Assert.That(camera.ParamsToString()).IsEqualTo("");
			await Assert.That(camera.Params.Length).IsEqualTo(0);
			await Assert.That(camera.Params.Length).IsEqualTo(0);
			await Assert.That(ReferenceEquals(camera.Params, camera.Params)).IsTrue();
		}
	}

	[Test]
	public async Task Camera_Equals()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		Camera other = camera.Clone();
		await Assert.That(camera == other).IsTrue();
		camera.SetFocalLength(2.0);
		await Assert.That(camera != other).IsTrue();
		other.SetFocalLength(2.0);
		await Assert.That(camera == other).IsTrue();
	}

	[Test]
	public async Task Camera_Print()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await Assert.That(camera.ToString()).IsEqualTo(
			"Camera(camera_id=1, model=SIMPLE_PINHOLE, width=1, height=1, params=[1, 0.5, 0.5] (f, cx, cy))");
	}

	[Test]
	public async Task Camera_CameraId()
	{
		var camera = new Camera();
		await Assert.That(camera.CameraId).IsEqualTo(InvalidCameraId);
		camera.CameraId = 1;
		await Assert.That(camera.CameraId).IsEqualTo(1u);
	}

	[Test]
	public async Task Camera_SensorId()
	{
		var camera = new Camera { CameraId = 1 };
		await Assert.That(camera.SensorId).IsEqualTo(new SensorId(SensorType.Camera, 1));
	}

	[Test]
	public async Task Camera_FocalLength()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		using (Assert.Multiple())
		{
			await Assert.That(camera.FocalLength()).IsEqualTo(1.0);
			await Assert.That(camera.FocalLengthX()).IsEqualTo(1.0);
			await Assert.That(camera.FocalLengthY()).IsEqualTo(1.0);
			camera.SetFocalLength(2.0);
			await Assert.That(camera.FocalLength()).IsEqualTo(2.0);
			await Assert.That(camera.FocalLengthX()).IsEqualTo(2.0);
			await Assert.That(camera.FocalLengthY()).IsEqualTo(2.0);
			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
			await Assert.That(camera.FocalLengthX()).IsEqualTo(1.0);
			await Assert.That(camera.FocalLengthY()).IsEqualTo(1.0);
			camera.SetFocalLengthX(2.0);
			await Assert.That(camera.FocalLengthX()).IsEqualTo(2.0);
			await Assert.That(camera.FocalLengthY()).IsEqualTo(1.0);
			await Assert.That(camera.MeanFocalLength()).IsEqualTo(1.5);
			camera.SetFocalLengthY(2.0);
			await Assert.That(camera.FocalLengthX()).IsEqualTo(2.0);
			await Assert.That(camera.FocalLengthY()).IsEqualTo(2.0);
			await Assert.That(camera.MeanFocalLength()).IsEqualTo(2.0);
		}
	}

	[Test]
	public async Task Camera_PrincipalPoint()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
		using (Assert.Multiple())
		{
			await Assert.That(camera.PrincipalPointX()).IsEqualTo(0.5);
			await Assert.That(camera.PrincipalPointY()).IsEqualTo(0.5);
			await Assert.That(camera.PrincipalPoint()).IsEqualTo(new Vector2d(0.5, 0.5));
			camera.SetPrincipalPointX(2.0);
			await Assert.That(camera.PrincipalPointX()).IsEqualTo(2.0);
			await Assert.That(camera.PrincipalPointY()).IsEqualTo(0.5);
			await Assert.That(camera.PrincipalPoint()).IsEqualTo(new Vector2d(2.0, 0.5));
			camera.SetPrincipalPointY(2.0);
			await Assert.That(camera.PrincipalPointX()).IsEqualTo(2.0);
			await Assert.That(camera.PrincipalPointY()).IsEqualTo(2.0);
			await Assert.That(camera.PrincipalPoint()).IsEqualTo(new Vector2d(2.0, 2.0));
		}
	}

	[Test]
	public async Task Camera_ParamIdxs()
	{
		var camera = new Camera();
		using (Assert.Multiple())
		{
			await Assert.That(() => { _ = camera.FocalLengthIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.PrincipalPointIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.ExtraParamsIdxs.Length; }).Throws<ArgumentException>();
			await Assert.That(() => { _ = camera.MetaDataParamsIdxs.Length; }).Throws<ArgumentException>();
			camera.ModelId = CameraModelId.FullOpenCV;
			await Assert.That(camera.FocalLengthIdxs.Length).IsEqualTo(2);
			await Assert.That(camera.PrincipalPointIdxs.Length).IsEqualTo(2);
			await Assert.That(camera.ExtraParamsIdxs.Length).IsEqualTo(8);
			await Assert.That(camera.MetaDataParamsIdxs.Length).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Camera_CalibrationMatrix()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
		Matrix3d k = camera.CalibrationMatrix();
		Matrix3d kRef = Matrix3d.FromRows(new Vector3d(1, 0, 0.5), new Vector3d(0, 1, 0.5), new Vector3d(0, 0, 1));
		await Assert.That(k == kRef).IsTrue();
	}

	[Test]
	public async Task Camera_ParamsInfo()
	{
		var camera = new Camera();
		await Assert.That(() => camera.ParamsInfo).Throws<ArgumentException>();
		camera.ModelId = CameraModelId.SimpleRadial;
		await Assert.That(camera.ParamsInfo).IsEqualTo("f, cx, cy, k");
	}

	[Test]
	public async Task Camera_ParamsToString()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await Assert.That(camera.ParamsToString()).IsEqualTo("1, 0.5, 0.5");
	}

	[Test]
	public async Task Camera_ParamsFromString()
	{
		var camera = new Camera { ModelId = CameraModelId.SimplePinhole };
		double[] parameters = [1.0, 0.5, 0.5];
		using (Assert.Multiple())
		{
			await Assert.That(camera.SetParamsFromString("1, 0.5, 0.5")).IsTrue();
			await Assert.That(camera.Params.SequenceEqual(parameters)).IsTrue();
			await Assert.That(camera.SetParamsFromString("1, 0.5")).IsFalse();
			await Assert.That(camera.Params.SequenceEqual(parameters)).IsTrue();
		}
	}

	[Test]
	public async Task Camera_VerifyParams()
	{
		var camera = new Camera();
		await Assert.That(() => camera.VerifyParams()).Throws<ArgumentException>();
		camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await Assert.That(camera.VerifyParams()).IsTrue();
		camera.Params = camera.Params[..2];
		await Assert.That(camera.VerifyParams()).IsFalse();
	}

	[Test]
	public async Task Camera_IsUndistorted()
	{
		using (Assert.Multiple())
		{
			Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
			await Assert.That(camera.IsUndistorted()).IsTrue();
			camera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 1.0, 1, 1);
			await Assert.That(camera.IsUndistorted()).IsTrue();
			camera.Params = [1.0, 0.5, 0.5, 0.005];
			await Assert.That(camera.IsUndistorted()).IsFalse();
			camera = Camera.CreateFromModelId(1, CameraModelId.Radial, 1.0, 1, 1);
			await Assert.That(camera.IsUndistorted()).IsTrue();
			camera.Params = [1.0, 0.5, 0.5, 0.0, 0.005];
			await Assert.That(camera.IsUndistorted()).IsFalse();
			camera = Camera.CreateFromModelId(1, CameraModelId.OpenCV, 1.0, 1, 1);
			await Assert.That(camera.IsUndistorted()).IsTrue();
			camera.Params = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001];
			await Assert.That(camera.IsUndistorted()).IsFalse();
			camera = Camera.CreateFromModelId(1, CameraModelId.FullOpenCV, 1.0, 1, 1);
			await Assert.That(camera.IsUndistorted()).IsTrue();
			camera.Params = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001];
			await Assert.That(camera.IsUndistorted()).IsFalse();
		}
	}

	[Test]
	public async Task Camera_HasBogusParams()
	{
		var camera = new Camera();
		await Assert.That(() => camera.HasBogusParams(0.0, 0.0, 0.0)).Throws<ArgumentException>();
		using (Assert.Multiple())
		{
			camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 1.0)).IsFalse();
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 0.0)).IsFalse();
			await Assert.That(camera.HasBogusParams(0.1, 0.99, 1.0)).IsTrue();
			await Assert.That(camera.HasBogusParams(1.01, 1.1, 1.0)).IsTrue();
			camera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 1.0, 1, 1);
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 1.0)).IsFalse();
			camera.Params[3] = 1.01;
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 1.0)).IsTrue();
			camera.Params[3] = -0.5;
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 1.0)).IsFalse();
			camera.Params[3] = -1.01;
			await Assert.That(camera.HasBogusParams(0.1, 1.1, 1.0)).IsTrue();
		}
	}

	[Test]
	public async Task Camera_CreateFromModelId()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await CheckSimplePinholeUnitCamera(camera);
	}

	[Test]
	public async Task Camera_CreateFromModelName()
	{
		Camera camera = Camera.CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
		await CheckSimplePinholeUnitCamera(camera);
	}

	[Test]
	public async Task Camera_CamFromImg()
	{
		var camera = new Camera();
		await Assert.That(() => camera.CamFromImg(Vector2d.Zero)).Throws<ArgumentException>();
		camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await Assert.That(camera.CamFromImg(new Vector2d(0.0, 0.0))!.Value).IsEqualTo(new Vector2d(-0.5, -0.5));
		await Assert.That(camera.CamFromImg(new Vector2d(0.5, 0.5))!.Value).IsEqualTo(new Vector2d(0, 0));
	}

	[Test]
	public async Task Camera_CamFromImgThreshold()
	{
		var camera = new Camera();
		await Assert.That(() => camera.CamFromImgThreshold(0)).Throws<ArgumentException>();
		using (Assert.Multiple())
		{
			camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
			await Assert.That(camera.CamFromImgThreshold(0)).IsEqualTo(0.0);
			await Assert.That(camera.CamFromImgThreshold(1)).IsEqualTo(1.0);
			camera.SetFocalLength(2.0);
			await Assert.That(camera.CamFromImgThreshold(1)).IsEqualTo(0.5);
			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
			camera.SetFocalLengthY(3.0);
			await Assert.That(camera.CamFromImgThreshold(1)).IsEqualTo(0.5);
		}
	}

	[Test]
	public async Task Camera_ImgFromCam()
	{
		var camera = new Camera();
		await Assert.That(() => camera.ImgFromCam(Vector3d.Zero)).Throws<ArgumentException>();
		camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		await Assert.That(camera.ImgFromCam(new Vector3d(0.0, 0.0, 1))!.Value).IsEqualTo(new Vector2d(0.5, 0.5));
		await Assert.That(camera.ImgFromCam(new Vector3d(-0.5, -0.5, 1))!.Value).IsEqualTo(new Vector2d(0.0, 0.0));
	}

	[Test]
	public async Task Camera_ImgFromCamWithoutCheiralityCheck()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
		var camPoint = new Vector3d(-0.5, -0.5, -1);
		await Assert.That(camera.ImgFromCam(camPoint).HasValue).IsFalse();
		await Assert.That(camera.ImgFromCam(camPoint, checkCheirality: false)!.Value).IsEqualTo(new Vector2d(1.0, 1.0));
	}

	[Test]
	public async Task Camera_CamRayFromImgWithJac()
	{
		// Covers a perspective model, a distorted one, and a spherical one. For the spherical
		// camera the sampled pixels include the back hemisphere, where CamFromImg fails
		// outright but CamRayFromImg (and hence this) must not.
		Camera[] cameras =
		[
			Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 650.0, 1024, 768),
			Camera.CreateFromModelId(2, CameraModelId.SimpleRadial, 650.0, 1024, 768),
			Camera.CreateFromModelId(3, CameraModelId.Equirectangular, 0.0, 1000, 500),
		];

		foreach (Camera camera in cameras)
		{
			foreach (double x in new[] { 1.0, 250.0, 512.0, 800.0, 999.0 })
			{
				foreach (double y in new[] { 1.0, 120.0, 250.0, 400.0, 499.0 })
				{
					var imagePoint = new Vector2d(x, y);
					var rayAndJac = camera.CamRayFromImgWithJac(imagePoint);
					await Assert.That(rayAndJac.HasValue).IsTrue().Because($"model {camera.ModelName} at {imagePoint}");
					var (camRay, jRay) = rayAndJac!.Value;

					// The bearing is a unit vector that reprojects to the source pixel.
					await Assert.That(camRay.Norm).IsEqualTo(1.0).Within(1e-12);
					Vector2d? reprojected = camera.ImgFromCam(camRay);
					await Assert.That(reprojected.HasValue).IsTrue();
					await Assert.That((reprojected!.Value - imagePoint).Norm).IsLessThanOrEqualTo(1e-8);

					// The Jacobian inverts the projection on the tangent plane.
					await Assert.That(camera.ImgFromCamWithJac(camRay, out Matrix2x3d jUvw).HasValue).IsTrue();
					await Assert.That((jUvw * jRay - Matrix2d.Identity).Norm()).IsLessThanOrEqualTo(1e-10);
					await Assert.That((jRay.Transpose() * camRay).Norm).IsLessThanOrEqualTo(1e-10 * jRay.Norm());
				}
			}
		}
	}

	[Test]
	public async Task Camera_CamRayFromImgWithJacUnprojectable()
	{
		// A pixel the camera cannot unproject (iterative undistortion diverges) must yield no
		// ray + Jacobian, rather than a garbage one.
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.OpenCV, 100.0, 100, 200);
		camera.Params[4] = -0.5; // k1
		camera.Params[5] = 0.5; // k2
		camera.Params[6] = -0.5; // p1
		var unprojectable = new Vector2d(50.0, 150.0);
		await Assert.That(camera.CamFromImg(unprojectable).HasValue).IsFalse();
		await Assert.That(camera.CamRayFromImgWithJac(unprojectable).HasValue).IsFalse();
	}

	[Test]
	public async Task Camera_Rescale()
	{
		using (Assert.Multiple())
		{
			Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 1, 1);
			camera.Rescale(2.0);
			await Assert.That(camera.Width).IsEqualTo(2);
			await Assert.That(camera.Height).IsEqualTo(2);
			await Assert.That(camera.FocalLength()).IsEqualTo(2.0);
			await Assert.That(camera.PrincipalPointX()).IsEqualTo(1.0);
			await Assert.That(camera.PrincipalPointY()).IsEqualTo(1.0);

			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
			camera.Rescale(2.0);
			await CheckPinhole(camera, 2, 2, 2, 1);

			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 2, 2);
			camera.Rescale(0.5);
			await CheckPinhole(camera, 1, 1, 0.5, 0.5);

			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 2, 2);
			camera.Rescale(1, 1);
			await CheckPinhole(camera, 1, 1, 0.5, 0.5);

			camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 2, 2);
			camera.Rescale(4, 4);
			await CheckPinhole(camera, 4, 4, 2, 2);
		}
	}

	[Test]
	public async Task Camera_Spherical()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		using (Assert.Multiple())
		{
			await Assert.That(camera.Params.SequenceEqual([1000.0, 500.0])).IsTrue();
			await Assert.That(camera.IsPerspective).IsFalse();
			await Assert.That(camera.IsSpherical).IsTrue();

			// No focal length / pinhole image plane. The (w, h) parameters form the metadata
			// group; there is no principal point or extra (distortion).
			await Assert.That(camera.FocalLengthIdxs.IsEmpty).IsTrue();
			await Assert.That(camera.PrincipalPointIdxs.IsEmpty).IsTrue();
			await Assert.That(camera.ExtraParamsIdxs.IsEmpty).IsTrue();
			await Assert.That(camera.MetaDataParamsIdxs.Length).IsEqualTo(2);
			await Assert.That(camera.MeanFocalLength()).IsEqualTo(0.0);
			// CalibrationMatrix is undefined without a focal length.
			await Assert.That(() => camera.CalibrationMatrix()).ThrowsException();
			// Spherical images have no lens distortion to undistort.
			await Assert.That(camera.IsUndistorted()).IsTrue();

			// Rescaling keeps the (w, h) parameters consistent with the dimensions.
			camera.Rescale(0.5);
			await Assert.That(camera.Width).IsEqualTo(500);
			await Assert.That(camera.Height).IsEqualTo(250);
			await Assert.That(camera.Params.SequenceEqual([500.0, 250.0])).IsTrue();
			await Assert.That(camera.IsSpherical).IsTrue();

			// A perspective camera is not spherical.
			Camera pinhole = Camera.CreateFromModelId(2, CameraModelId.Pinhole, 1.0, 1, 1);
			await Assert.That(pinhole.IsPerspective).IsTrue();
			await Assert.That(pinhole.IsSpherical).IsFalse();
		}
	}

	[Test]
	public async Task Camera_IsPerspectivePinhole()
	{
		Camera pinhole = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1.0, 1, 1);
		Camera radial = Camera.CreateFromModelId(2, CameraModelId.SimpleRadial, 1.0, 1, 1);
		Camera fisheye = Camera.CreateFromModelId(3, CameraModelId.OpenCVFisheye, 1.0, 1, 1);
		Camera spherical = Camera.CreateFromModelId(4, CameraModelId.Equirectangular, 0.0, 1000, 500);
		using (Assert.Multiple())
		{
			await Assert.That(pinhole.IsPerspectivePinhole).IsTrue();
			await Assert.That(radial.IsPerspectivePinhole).IsTrue();
			await Assert.That(fisheye.IsPerspective).IsTrue();
			await Assert.That(fisheye.IsPerspectiveFisheye).IsTrue();
			await Assert.That(fisheye.IsPerspectivePinhole).IsFalse();
			await Assert.That(spherical.IsPerspectiveFisheye).IsFalse();
			await Assert.That(spherical.IsPerspectivePinhole).IsFalse();
		}
	}

	// The identical checks of camera_test.cc's CreateFromModelId and CreateFromModelName.
	private static async Task CheckSimplePinholeUnitCamera(Camera camera)
	{
		using (Assert.Multiple())
		{
			await Assert.That(camera.CameraId).IsEqualTo(1u);
			await Assert.That(camera.ModelId).IsEqualTo(CameraModelId.SimplePinhole);
			await Assert.That(camera.ModelName).IsEqualTo("SIMPLE_PINHOLE");
			await Assert.That(camera.Width).IsEqualTo(1);
			await Assert.That(camera.Height).IsEqualTo(1);
			await Assert.That(camera.HasPriorFocalLength).IsFalse();
			await Assert.That(camera.FocalLengthIdxs.Length).IsEqualTo(1);
			await Assert.That(camera.PrincipalPointIdxs.Length).IsEqualTo(2);
			await Assert.That(camera.ExtraParamsIdxs.Length).IsEqualTo(0);
			await Assert.That(camera.MetaDataParamsIdxs.Length).IsEqualTo(0);
			await Assert.That(camera.ParamsInfo).IsEqualTo("f, cx, cy");
			await Assert.That(camera.ParamsToString()).IsEqualTo("1, 0.5, 0.5");
			await Assert.That(camera.FocalLength()).IsEqualTo(1.0);
			await Assert.That(camera.PrincipalPointX()).IsEqualTo(0.5);
			await Assert.That(camera.PrincipalPointY()).IsEqualTo(0.5);
			await Assert.That(camera.VerifyParams()).IsTrue();
			await Assert.That(camera.HasBogusParams(0.1, 2.0, 1.0)).IsFalse();
			await Assert.That(camera.HasBogusParams(0.1, 0.5, 1.0)).IsTrue();
			await Assert.That(camera.Params.Length).IsEqualTo(SimplePinholeCameraModel.NumParams);
			await Assert.That(camera.Params.Length).IsEqualTo(SimplePinholeCameraModel.NumParams);
		}
	}

	// The PINHOLE checks of camera_test.cc's Rescale: size, equal fx/fy, equal cx/cy.
	private static async Task CheckPinhole(Camera camera, int width, int height, double focal, double principal)
	{
		await Assert.That(camera.Width).IsEqualTo(width);
		await Assert.That(camera.Height).IsEqualTo(height);
		await Assert.That(camera.FocalLengthX()).IsEqualTo(focal);
		await Assert.That(camera.FocalLengthY()).IsEqualTo(focal);
		await Assert.That(camera.PrincipalPointX()).IsEqualTo(principal);
		await Assert.That(camera.PrincipalPointY()).IsEqualTo(principal);
	}
}
