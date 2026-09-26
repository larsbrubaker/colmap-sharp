// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// UndistortionTests: colmap/image/undistortion_test.cc ported 1:1 (Suite_Name). Tests
// ColmapSharp/ImageProcessing/Undistortion.cs.
//
// Tier A for the camera and reconstruction cases (exact expected values, as in COLMAP);
// RectifyStereoCameras_Nominal is Tier B with COLMAP's 1e-5 tolerance.
// UndistortImage_WarpOptions compares a direct warp with a warp-then-Rescale, so it runs
// through Bitmap.Rescale (Tier B, docs/CPP_DIVERGENCES.md entry 9); its bounds are COLMAP's.
// CSharpOnly_UndistortReconstruction_KeepsCameraIds is C#-only and pins divergence entry 60;
// CSharpOnly_RectifyAndUndistortStereoImages_FarSourceSamples is C#-only and pins entry 117.

using ColmapSharp.Geometry;
using ColmapSharp.ImageProcessing;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.ImageProcessing;

public class UndistortionTests
{
	private static ulong TotalAbsoluteDifference(Bitmap bitmap1, Bitmap bitmap2)
	{
		byte[] data1 = bitmap1.RowMajorData;
		byte[] data2 = bitmap2.RowMajorData;
		Check.Eq(data1.Length, data2.Length);
		ulong totalAbsoluteDifference = 0;
		for (int i = 0; i < data1.Length; i++)
		{
			totalAbsoluteDifference += (ulong)Math.Abs(data1[i] - data2[i]);
		}
		return totalAbsoluteDifference;
	}

	private static double MeanAbsoluteDifference(Bitmap bitmap1, Bitmap bitmap2)
	{
		Check.Gt(bitmap1.RowMajorData.Length, 0);
		return (double)TotalAbsoluteDifference(bitmap1, bitmap2) / bitmap1.RowMajorData.Length;
	}

	private static int MaxAbsoluteDifference(Bitmap bitmap1, Bitmap bitmap2)
	{
		byte[] data1 = bitmap1.RowMajorData;
		byte[] data2 = bitmap2.RowMajorData;
		Check.Eq(data1.Length, data2.Length);
		int maxAbsoluteDifference = 0;
		for (int i = 0; i < data1.Length; i++)
		{
			maxAbsoluteDifference = Math.Max(maxAbsoluteDifference, Math.Abs(data1[i] - data2[i]));
		}
		return maxAbsoluteDifference;
	}

	[Test]
	public async Task UndistortCamera_Nominal()
	{
		var options = new UndistortCameraOptions();

		Camera distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		Camera undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(1);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(1);
			await Assert.That(undistortedCamera.Width).IsEqualTo(1);
			await Assert.That(undistortedCamera.Height).IsEqualTo(1);
		}

		distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 1, 1, 1);
		undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(1);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(1);
			await Assert.That(undistortedCamera.Width).IsEqualTo(1);
			await Assert.That(undistortedCamera.Height).IsEqualTo(1);
		}

		distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera.Params[3] = 0.5;
		undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.PrincipalPointX()).IsEqualTo(84.0 / 2.0);
			await Assert.That(undistortedCamera.PrincipalPointY()).IsEqualTo(84.0 / 2.0);
			await Assert.That(undistortedCamera.Width).IsEqualTo(84);
			await Assert.That(undistortedCamera.Height).IsEqualTo(84);
		}

		options.BlankPixels = 1;
		undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.Width).IsEqualTo(90);
			await Assert.That(undistortedCamera.Height).IsEqualTo(90);
		}

		options.MaxScale = 0.75;
		undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.Width).IsEqualTo(75);
			await Assert.That(undistortedCamera.Height).IsEqualTo(75);
		}

		options.MaxScale = 1.0;
		options.RoiMinX = 0.1;
		options.RoiMinY = 0.2;
		options.RoiMaxX = 0.9;
		options.RoiMaxY = 0.8;
		undistortedCamera = Undistortion.UndistortCamera(options, distortedCamera);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.Width).IsEqualTo(80);
			await Assert.That(undistortedCamera.Height).IsEqualTo(60);
			await Assert.That(undistortedCamera.PrincipalPointX()).IsEqualTo(40);
			await Assert.That(undistortedCamera.PrincipalPointY()).IsEqualTo(30);
		}
	}

	[Test]
	public async Task UndistortCamera_MaxCamPointNorm()
	{
		// Fisheye camera with off-center principal point: pixels far from (cx, cy) have theta
		// close to pi/2, so CamFromImg returns very large values (|cam_point| = tan(theta))
		// that blow up the output dimensions.
		Camera distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleFisheye, 130, 200, 100);
		distortedCamera.SetPrincipalPointX(10);
		distortedCamera.SetPrincipalPointY(50);

		var options = new UndistortCameraOptions
		{
			// Exercise the path driven by extreme samples.
			BlankPixels = 1.0,
		};

		// Default (max_cam_point_norm = -1): the extreme cam_points push the scale factor
		// against max_scale, so the output is the maximum allowed size.
		Camera undistortedCameraUnbounded = Undistortion.UndistortCamera(options, distortedCamera);
		await Assert.That((double)undistortedCameraUnbounded.Width).IsEqualTo(distortedCamera.Width * options.MaxScale);
		await Assert.That((double)undistortedCameraUnbounded.Height).IsEqualTo(distortedCamera.Height * options.MaxScale);

		// With a finite threshold, extreme border samples are skipped, yielding a smaller
		// output that is no longer hitting the max_scale clamp.
		options.MaxCamPointNorm = 2.0;
		Camera undistortedCameraBounded = Undistortion.UndistortCamera(options, distortedCamera);
		await Assert.That(undistortedCameraBounded.Width).IsLessThan(undistortedCameraUnbounded.Width);
		await Assert.That(undistortedCameraBounded.Height).IsLessThan(undistortedCameraUnbounded.Height);

		// max_cam_point_norm = 0 is invalid (would skip every sample).
		options.MaxCamPointNorm = 0;
		await Assert.That(() => Undistortion.UndistortCamera(options, distortedCamera)).ThrowsException();
	}

	[Test]
	public async Task UndistortCamera_BlankPixels()
	{
		var options = new UndistortCameraOptions { BlankPixels = 1 };

		Camera distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera.Params[3] = 0.5;

		var distortedImage = new Bitmap(100, 100, false);
		distortedImage.Fill(new BitmapColor<byte>(255));

		Undistortion.UndistortImage(options, distortedImage, distortedCamera, out Bitmap undistortedImage, out Camera undistortedCamera);

		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.PrincipalPointX()).IsEqualTo(90.0 / 2.0);
			await Assert.That(undistortedCamera.PrincipalPointY()).IsEqualTo(90.0 / 2.0);
			await Assert.That(undistortedCamera.Width).IsEqualTo(90);
			await Assert.That(undistortedCamera.Height).IsEqualTo(90);
		}

		// Make sure that there is no blank pixel.
		int numBlankPixels = 0;
		int numMissing = 0;
		for (int y = 0; y < undistortedImage.Height; y++)
		{
			for (int x = 0; x < undistortedImage.Width; x++)
			{
				BitmapColor<byte>? color = undistortedImage.GetPixel(x, y);
				if (color is null)
				{
					numMissing++;
				}
				else if (color.Value == new BitmapColor<byte>(0))
				{
					numBlankPixels++;
				}
			}
		}

		await Assert.That(numMissing).IsEqualTo(0);
		await Assert.That(numBlankPixels).IsGreaterThan(0);
	}

	[Test]
	public async Task UndistortCamera_NoBlankPixels()
	{
		var options = new UndistortCameraOptions { BlankPixels = 0 };

		Camera distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera.Params[3] = 0.5;

		var distortedImage = new Bitmap(100, 100, false);
		distortedImage.Fill(new BitmapColor<byte>(255));

		Undistortion.UndistortImage(options, distortedImage, distortedCamera, out Bitmap undistortedImage, out Camera undistortedCamera);

		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(100);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(100);
			await Assert.That(undistortedCamera.PrincipalPointX()).IsEqualTo(84.0 / 2.0);
			await Assert.That(undistortedCamera.PrincipalPointY()).IsEqualTo(84.0 / 2.0);
			await Assert.That(undistortedCamera.Width).IsEqualTo(84);
			await Assert.That(undistortedCamera.Height).IsEqualTo(84);
		}

		// Make sure that there is no blank pixel.
		int numMissing = 0;
		int numBlank = 0;
		for (int y = 0; y < undistortedImage.Height; y++)
		{
			for (int x = 0; x < undistortedImage.Width; x++)
			{
				BitmapColor<byte>? color = undistortedImage.GetPixel(x, y);
				if (color is null)
				{
					numMissing++;
				}
				else if (color.Value.R == 0 || color.Value.G == 0 || color.Value.B == 0)
				{
					numBlank++;
				}
			}
		}

		await Assert.That(numMissing).IsEqualTo(0);
		await Assert.That(numBlank).IsEqualTo(0);
	}

	[Test]
	public async Task UndistortImage_WarpOptions()
	{
		Camera distortedCamera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera.Params[3] = 0.5;

		var distortedImage = new Bitmap(100, 100, true);
		for (int y = 0; y < distortedImage.Height; y++)
		{
			for (int x = 0; x < distortedImage.Width; x++)
			{
				distortedImage.SetPixel(x, y, new BitmapColor<byte>((byte)x, (byte)y, (byte)((x + y) / 2)));
			}
		}

		var options = new UndistortCameraOptions();
		Undistortion.UndistortImage(options, distortedImage, distortedCamera, out Bitmap directImage, out Camera directCamera);
		double directScale = Math.Min(
			(double)directCamera.Width / distortedCamera.Width,
			(double)directCamera.Height / distortedCamera.Height);
		await Assert.That(directScale).IsGreaterThanOrEqualTo(options.WarpOptions.DirectWarpMinScale);

		options.WarpOptions.DirectWarpMinScale = 1.0;
		Undistortion.UndistortImage(options, distortedImage, distortedCamera, out Bitmap resizedImage, out Camera resizedCamera);

		using (Assert.Multiple())
		{
			await Assert.That(directCamera == resizedCamera).IsTrue();
			await Assert.That(directImage.Width).IsEqualTo(resizedImage.Width);
			await Assert.That(directImage.Height).IsEqualTo(resizedImage.Height);
			await Assert.That(directImage.RowMajorData.AsSpan().SequenceEqual(resizedImage.RowMajorData)).IsFalse();

			// The extra resize changes rounding, but not by a perceptible amount for smooth
			// image content.
			await Assert.That(MeanAbsoluteDifference(directImage, resizedImage)).IsLessThan(0.5);
			await Assert.That(MaxAbsoluteDifference(directImage, resizedImage)).IsLessThanOrEqualTo(1);
		}
	}

	/// <summary>
	/// The frame/image scaffolding the UndistortReconstruction cases share: a single-camera
	/// rig, and one registered frame with one image (named "image{id}", with the given
	/// observations) per id 1..numImages.
	/// </summary>
	private static Reconstruction CreateReconstruction(Camera camera, int numImages, IReadOnlyList<Vector2d> points)
	{
		var reconstruction = new Reconstruction();
		reconstruction.AddCamera(camera);
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(new SensorId(SensorType.Camera, 1));
		reconstruction.AddRig(rig);

		for (uint imageId = 1; imageId <= numImages; imageId++)
		{
			var frame = new Frame { FrameId = imageId };
			frame.SetRigId(1);
			frame.SetRigFromWorld(new Rigid3d());
			var image = new Image { ImageId = imageId, Name = "image" + imageId };
			image.SetCameraId(1);
			image.SetFrameId(frame.FrameId);
			image.SetPoints2D(points);
			frame.AddDataId(image.DataId);
			reconstruction.AddFrame(frame);
			reconstruction.AddImage(image);
			reconstruction.RegisterFrame(frame.FrameId);
		}

		return reconstruction;
	}

	[Test]
	public async Task UndistortReconstruction_Nominal()
	{
		const int kNumImages = 10;
		const int kNumPoints2D = 10;

		Camera camera = Camera.CreateFromModelId(1, CameraModelId.OpenCV, 1, 1, 1);
		camera.Params[4] = 1.0;
		Reconstruction reconstruction = CreateReconstruction(
			camera, kNumImages, Enumerable.Repeat(Vector2d.Ones, kNumPoints2D).ToList());

		var options = new UndistortCameraOptions();
		Undistortion.UndistortReconstruction(options, reconstruction);
		foreach (Camera undistortedCamera in reconstruction.Cameras.Values)
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
		}

		foreach (Image image in reconstruction.Images.Values)
		{
			foreach (Point2D point2D in image.Points2D)
			{
				await Assert.That(point2D.Xy != Vector2d.Ones).IsTrue();
			}
		}
	}

	[Test]
	public async Task UndistortReconstruction_RescalesAlreadyUndistortedCameras()
	{
		const int kNumImages = 3;

		// Already-undistorted (PINHOLE) camera at a high resolution. focal=100,
		// width=height=100 => principal point at (50, 50).
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 100, 100, 100);
		await Assert.That(camera.IsUndistorted()).IsTrue();

		var pointXy = new Vector2d(60, 40);
		Reconstruction reconstruction = CreateReconstruction(camera, kNumImages, [pointXy]);

		var options = new UndistortCameraOptions { MaxImageSize = 50 };
		Undistortion.UndistortReconstruction(options, reconstruction);

		// The camera resolution and intrinsics are rescaled to match max_image_size.
		Camera undistortedCamera = reconstruction.Camera(1);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.Width).IsEqualTo(50);
			await Assert.That(undistortedCamera.Height).IsEqualTo(50);
			await Assert.That(undistortedCamera.FocalLengthX()).IsEqualTo(50).Within(1e-6);
			await Assert.That(undistortedCamera.FocalLengthY()).IsEqualTo(50).Within(1e-6);
			await Assert.That(undistortedCamera.PrincipalPointX()).IsEqualTo(25).Within(1e-6);
			await Assert.That(undistortedCamera.PrincipalPointY()).IsEqualTo(25).Within(1e-6);
		}

		// The observations are rescaled consistently with the camera: (60, 40) maps to
		// normalized (0.1, -0.1) and back through the halved intrinsics to
		// (0.1 * 50 + 25, -0.1 * 50 + 25) = (30, 20).
		foreach (Image image in reconstruction.Images.Values)
		{
			await Assert.That(image.NumPoints2D).IsEqualTo(1u);
			await Assert.That(EigenMatrixNear(image.Points2D[0].Xy, new Vector2d(30, 20), 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task UndistortReconstruction_RescalesSphericalCameras()
	{
		const int kNumImages = 3;

		// Spherical (EQUIRECTANGULAR) camera at a high resolution. It has no pinhole plane to
		// undistort to, but can still be resized to a smaller image of the same model.
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		await Assert.That(camera.IsSpherical).IsTrue();

		// A front-hemisphere pixel (azimuth 36 deg) and a back-hemisphere pixel (azimuth
		// -144 deg, behind the camera). The back-hemisphere observation has no forward
		// normalized (CamFromImg) representation, so it must be handled by plain linear
		// scaling rather than a bearing round-trip.
		Vector2d[] pointsXy = [new Vector2d(600, 200), new Vector2d(100, 400)];
		Reconstruction reconstruction = CreateReconstruction(camera, kNumImages, pointsXy);

		var options = new UndistortCameraOptions { MaxImageSize = 250 };
		Undistortion.UndistortReconstruction(options, reconstruction);

		// The camera keeps its model but is resized so its larger dimension (width) matches
		// max_image_size: scale = 250 / 1000 = 0.25.
		Camera undistortedCamera = reconstruction.Camera(1);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("EQUIRECTANGULAR");
			await Assert.That(undistortedCamera.Width).IsEqualTo(250);
			await Assert.That(undistortedCamera.Height).IsEqualTo(125);
		}

		// The observations are rescaled consistently with the camera: the fractional position
		// within the image is preserved (a plain 0.25 scaling here), so both the front- and
		// back-hemisphere points map through without loss.
		foreach (Image image in reconstruction.Images.Values)
		{
			await Assert.That(image.NumPoints2D).IsEqualTo(2u);
			await Assert.That(EigenMatrixNear(image.Points2D[0].Xy, new Vector2d(150, 50), 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(image.Points2D[1].Xy, new Vector2d(25, 100), 1e-6)).IsTrue();
		}
	}

	// C#-only (docs/CPP_DIVERGENCES.md, entry 60): COLMAP overwrites each stored camera with a
	// default-constructed one, whose camera_id is invalid. Here the ids are kept, so code keyed
	// by camera or sensor id (Crop re-adds every camera by its CameraId) still works after
	// undistortion, as MatterCAD needs when it undistorts and then densifies the same model.
	[Test]
	public async Task CSharpOnly_UndistortReconstruction_KeepsCameraIds()
	{
		var reconstruction = new Reconstruction();
		for (uint cameraId = 1; cameraId <= 2; cameraId++)
		{
			Camera camera = Camera.CreateFromModelId(cameraId, CameraModelId.SimpleRadial, 100, 100, 100);
			camera.Params[3] = 0.1;
			reconstruction.AddCameraWithTrivialRig(camera);
			var image = new Image { ImageId = cameraId, Name = "image" + cameraId };
			image.SetCameraId(cameraId);
			image.SetPoints2D([new Vector2d(60, 40)]);
			reconstruction.AddImageWithTrivialFrame(image, new Rigid3d());
		}

		Undistortion.UndistortReconstruction(new UndistortCameraOptions(), reconstruction);

		foreach (var (cameraId, camera) in reconstruction.Cameras)
		{
			await Assert.That(camera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(camera.CameraId).IsEqualTo(cameraId);
			await Assert.That(reconstruction.Rig(cameraId).IsRefSensor(camera.SensorId)).IsTrue();
		}

		Reconstruction cropped = reconstruction.Crop(new AlignedBox3d(new Vector3d(-1, -1, -1), new Vector3d(1, 1, 1)));
		await Assert.That(cropped.NumCameras).IsEqualTo(2);
	}

	[Test]
	public async Task RectifyStereoCameras_Nominal()
	{
		Camera camera1 = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 1, 1);
		Camera camera2 = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, 1, 1);

		var cam2FromCam1 = new Rigid3d(
			Quaterniond.FromRotationMatrix(Pose.EulerAnglesToRotationMatrix(0.1, 0.2, 0.3)),
			new Vector3d(0.1, 0.2, 0.3));

		Undistortion.RectifyStereoCameras(camera1, camera2, cam2FromCam1, out Matrix3d h1, out Matrix3d h2, out Matrix4d q);

		var h1Ref = new Matrix3d(
			-0.202759, -0.815848, -0.897034, 0.416329, 0.733069, -0.199657,
			0.910839, -0.175408, 0.942638);
		await Assert.That(EigenMatrixNear(h1, h1Ref.Transpose(), 1e-5)).IsTrue();

		var h2Ref = new Matrix3d(
			-0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
			0.963533, 0.292411, 1.12528);
		await Assert.That(EigenMatrixNear(h2, h2Ref.Transpose(), 1e-5)).IsTrue();

		var qRef = new Matrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.67261, -0.5, -0.5, 1, 0);
		await Assert.That(EigenMatrixNear(q, qRef, 1e-5)).IsTrue();
	}

	[Test]
	public async Task RectifyAndUndistortStereoImages_Nominal()
	{
		var options = new UndistortCameraOptions();

		// Create two distorted cameras with radial distortion.
		Camera distortedCamera1 = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera1.Params[3] = 0.1; // Add some radial distortion

		Camera distortedCamera2 = Camera.CreateFromModelId(2, CameraModelId.SimpleRadial, 100, 100, 100);
		distortedCamera2.Params[3] = 0.1; // Add some radial distortion

		// Create dummy distorted images.
		var distortedImage1 = new Bitmap(100, 100, true);
		distortedImage1.Fill(new BitmapColor<byte>(255, 0, 0)); // Red image

		var distortedImage2 = new Bitmap(100, 100, true);
		distortedImage2.Fill(new BitmapColor<byte>(0, 255, 0)); // Green image

		// Create relative pose between cameras (typical stereo baseline).
		var cam2FromCam1 = new Rigid3d(
			Quaterniond.FromRotationMatrix(Pose.EulerAnglesToRotationMatrix(0.0, 0.05, 0.0)),
			new Vector3d(0.1, 0.0, 0.0)); // 0.1m baseline

		// Rectify and undistort stereo images.
		Undistortion.RectifyAndUndistortStereoImages(
			options,
			distortedImage1,
			distortedImage2,
			distortedCamera1,
			distortedCamera2,
			cam2FromCam1,
			out Bitmap undistortedImage1,
			out Bitmap undistortedImage2,
			out Camera undistortedCamera,
			out Matrix4d _);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedCamera.ModelName).IsEqualTo("PINHOLE");
			await Assert.That(undistortedCamera.Width).IsEqualTo(undistortedImage1.Width);
			await Assert.That(undistortedCamera.Height).IsEqualTo(undistortedImage1.Height);
			await Assert.That(undistortedImage1.Width).IsEqualTo(undistortedImage2.Width);
			await Assert.That(undistortedImage1.Height).IsEqualTo(undistortedImage2.Height);
		}
	}

	// C#-only (docs/CPP_DIVERGENCES.md, entry 117): with PRNG seed 25 the synthetic stereo pair
	// that UndistortersTests.StereoImageRectifier_Integration builds rectifies some target
	// pixels to source points beyond int range. Bitmap.InterpolateBilinear used to index out
	// of its array there, which made that test fail whenever an earlier test on the same
	// thread left the thread-static PRNG in such a state.
	[Test]
	public async Task CSharpOnly_RectifyAndUndistortStereoImages_FarSourceSamples()
	{
		RandomUtils.SetPRNGSeed(25);
		var syntheticOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			CameraWidth = 100,
			CameraHeight = 100,
		};
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(syntheticOptions, reconstruction);
		List<uint> imageIds = reconstruction.RegImageIds();
		ColmapSharp.Scene.Image image1 = reconstruction.Image(imageIds[0]);
		ColmapSharp.Scene.Image image2 = reconstruction.Image(imageIds[1]);
		var bitmap1 = new Bitmap(100, 100, true);
		var bitmap2 = new Bitmap(100, 100, true);

		Undistortion.RectifyAndUndistortStereoImages(
			new UndistortCameraOptions(),
			bitmap1,
			bitmap2,
			reconstruction.Camera(image1.CameraId),
			reconstruction.Camera(image2.CameraId),
			image2.CamFromWorld() * image1.CamFromWorld().Inverse(),
			out Bitmap undistortedImage1,
			out Bitmap undistortedImage2,
			out Camera undistortedCamera,
			out Matrix4d _);
		using (Assert.Multiple())
		{
			await Assert.That(undistortedImage1.Width).IsEqualTo(undistortedCamera.Width);
			await Assert.That(undistortedImage2.Height).IsEqualTo(undistortedCamera.Height);
		}
	}
}
