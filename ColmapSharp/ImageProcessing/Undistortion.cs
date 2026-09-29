// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Undistortion: colmap/image/undistortion.h/.cc - UndistortCameraOptions, UndistortCamera,
// UndistortImage, UndistortReconstruction, RectifyStereoCameras and
// RectifyAndUndistortStereoImages. The pixel warps they use are in Warp.cs. The file-writing
// undistorter controllers (controllers/undistorters.cc) build on these and are not here.
//
// Tier A (exact) for UndistortCamera, UndistortReconstruction and the image warps when
// warping directly (scalar camera-model arithmetic and bilinear lookups). RectifyStereoCameras
// is Tier B: it inverts a matrix and composes rotations through our Eigen replacements.
// Images that go through Bitmap.Rescale (large downscales, spherical cameras with a
// max_image_size) inherit its Tier B (divergence 9).
//
// Translation notes:
// - C++ output pointers become return values or out parameters.
// - COLMAP's Camera is a value type; `reconstruction->Camera(id) = UndistortCamera(...)`
//   overwrites every field of the stored camera, including camera_id, which becomes
//   kInvalidCameraId because UndistortCamera builds a default-constructed Camera. Here
//   UndistortReconstruction writes the undistorted model, size, parameters and prior flag
//   into the stored Camera object (which the images' CameraPtr also reference) but keeps its
//   CameraId (divergence 60): code keyed by camera or sensor id would
//   otherwise collide on the invalid id. UndistortCamera itself still returns a camera with
//   the invalid id, as in COLMAP.
// - std::min/std::max are written out as (b < a) ? b : a and (a < b) ? b : a where a NaN
//   could reach them, so NaN resolves the same way as in C++.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.ImageProcessing;

/// <summary>Port of colmap::UndistortCameraOptions.</summary>
public sealed class UndistortCameraOptions
{
	/// <summary>The amount of blank pixels in the undistorted image in the range [0, 1].</summary>
	public double BlankPixels { get; set; }

	/// <summary>Minimum scale change of camera used to satisfy the blank pixel constraint.</summary>
	public double MinScale { get; set; } = 0.2;

	/// <summary>Maximum scale change of camera used to satisfy the blank pixel constraint.</summary>
	public double MaxScale { get; set; } = 2.0;

	/// <summary>Maximum image size in terms of width or height of the undistorted camera (-1: none).</summary>
	public int MaxImageSize { get; set; } = -1;

	/// <summary>
	/// The 4 factors in the range [0, 1] that define the ROI (region of interest) in the original
	/// image. The bounding box pixel coordinates are (RoiMinX * Width, RoiMinY * Height) and
	/// (RoiMaxX * Width, RoiMaxY * Height).
	/// </summary>
	public double RoiMinX { get; set; }

	/// <summary>See <see cref="RoiMinX"/>.</summary>
	public double RoiMinY { get; set; }

	/// <summary>See <see cref="RoiMinX"/>.</summary>
	public double RoiMaxX { get; set; } = 1.0;

	/// <summary>See <see cref="RoiMinX"/>.</summary>
	public double RoiMaxY { get; set; } = 1.0;

	/// <summary>
	/// Maximum norm of the undistorted camera-space point (= tan(theta), theta the angle from
	/// the optical axis) used when tracing border pixels to compute the output image
	/// dimensions. For fisheye cameras with off-center principal points, border pixels outside
	/// the valid fisheye circle can have theta approaching pi/2, so tan(theta) diverges and
	/// produces extreme output dimensions. Points with a larger norm are skipped. -1 disables
	/// the check (default).
	/// </summary>
	public double MaxCamPointNorm { get; set; } = -1;

	/// <summary>Options controlling image warping during undistortion.</summary>
	public WarpImageOptions WarpOptions { get; set; } = new();
}

/// <summary>Port of the free functions in colmap/image/undistortion.h.</summary>
public static class Undistortion
{
	/// <summary>
	/// Port of colmap::UndistortCamera: a PINHOLE camera with the same focal length whose
	/// dimensions are adjusted so that either every undistorted pixel has a distorted
	/// counterpart (no blank border, BlankPixels = 0) or every distorted pixel lands in the
	/// undistorted image (BlankPixels = 1), or in between. The relative principal point
	/// location is preserved; the scale is subject to MinScale, MaxScale and MaxImageSize.
	/// </summary>
	public static Camera UndistortCamera(UndistortCameraOptions options, Camera camera)
	{
		Check.Ge(options.BlankPixels, 0);
		Check.Le(options.BlankPixels, 1);
		Check.Gt(options.MinScale, 0.0);
		Check.Le(options.MinScale, options.MaxScale);
		Check.Ne(options.MaxImageSize, 0);
		Check.Ge(options.RoiMinX, 0.0);
		Check.Ge(options.RoiMinY, 0.0);
		Check.Le(options.RoiMaxX, 1.0);
		Check.Le(options.RoiMaxY, 1.0);
		Check.Lt(options.RoiMinX, options.RoiMaxX);
		Check.Lt(options.RoiMinY, options.RoiMaxY);

		// Undistortion produces a pinhole image, which is only well-defined for perspective
		// cameras. Omnidirectional models (e.g. EQUIRECTANGULAR) have no pinhole image plane
		// and cannot be undistorted; callers skip them, but guard here too.
		Check.That(camera.IsPerspective);

		var undistortedCamera = new Camera
		{
			ModelId = CameraModelId.Pinhole,
			Width = camera.Width,
			Height = camera.Height,
			Params = new double[4],
		};

		// Copy focal length parameters.
		Check.Le(camera.FocalLengthIdxs.Length, 2, "Not more than two focal length parameters supported.");
		undistortedCamera.SetFocalLengthX(camera.FocalLengthX());
		undistortedCamera.SetFocalLengthY(camera.FocalLengthY());

		// Copy principal point parameters.
		undistortedCamera.SetPrincipalPointX(camera.PrincipalPointX());
		undistortedCamera.SetPrincipalPointY(camera.PrincipalPointY());

		// Modify undistorted camera parameters based on ROI if enabled.
		int roiMinX = 0;
		int roiMinY = 0;
		int roiMaxX = camera.Width;
		int roiMaxY = camera.Height;

		bool roiEnabled = options.RoiMinX > 0.0 || options.RoiMinY > 0.0 || options.RoiMaxX < 1.0 || options.RoiMaxY < 1.0;

		if (roiEnabled)
		{
			roiMinX = (int)Math.Round(options.RoiMinX * camera.Width, MidpointRounding.AwayFromZero);
			roiMinY = (int)Math.Round(options.RoiMinY * camera.Height, MidpointRounding.AwayFromZero);
			roiMaxX = (int)Math.Round(options.RoiMaxX * camera.Width, MidpointRounding.AwayFromZero);
			roiMaxY = (int)Math.Round(options.RoiMaxY * camera.Height, MidpointRounding.AwayFromZero);

			// Make sure that the roi is valid.
			roiMinX = Math.Min(roiMinX, camera.Width - 1);
			roiMinY = Math.Min(roiMinY, camera.Height - 1);
			roiMaxX = Math.Max(roiMaxX, roiMinX + 1);
			roiMaxY = Math.Max(roiMaxY, roiMinY + 1);

			undistortedCamera.Width = roiMaxX - roiMinX;
			undistortedCamera.Height = roiMaxY - roiMinY;

			undistortedCamera.SetPrincipalPointX(camera.PrincipalPointX() - roiMinX);
			undistortedCamera.SetPrincipalPointY(camera.PrincipalPointY() - roiMinY);
		}

		// Scale in order to match the boundary of the undistorted image.
		if (roiEnabled || (camera.ModelId != CameraModelId.SimplePinhole && camera.ModelId != CameraModelId.Pinhole))
		{
			ScaleToBorder(options, camera, undistortedCamera, roiMinX, roiMinY, roiMaxX, roiMaxY);
		}

		if (options.MaxImageSize > 0)
		{
			RescaleToMaxImageSize(options, undistortedCamera);
		}

		return undistortedCamera;
	}

	/// <summary>
	/// Port of colmap::UndistortImage: undistort the image so that its viewing geometry follows
	/// a pinhole camera model (see <see cref="UndistortCamera"/>). Spherical cameras keep their
	/// model and are only downscaled to MaxImageSize.
	/// </summary>
	public static void UndistortImage(
		UndistortCameraOptions options,
		Bitmap distortedBitmap,
		Camera distortedCamera,
		out Bitmap undistortedBitmap,
		out Camera undistortedCamera,
		CancellationToken cancellationToken = default)
	{
		Check.Eq(distortedCamera.Width, distortedBitmap.Width);
		Check.Eq(distortedCamera.Height, distortedBitmap.Height);

		if (distortedCamera.IsSpherical)
		{
			// Spherical cameras (e.g. EQUIRECTANGULAR) have no pinhole image plane to undistort
			// to, so keep the model and only apply the max_image_size limit. The
			// equirectangular pixel<->angle map is linear in the image dimensions, so this is a
			// plain downscale. The per-pixel bearing warp used below cannot be reused: CamFromImg
			// has no forward normalized coordinates for the back hemisphere and would blank out
			// those pixels.
			undistortedCamera = distortedCamera.Clone();
			RescaleToMaxImageSize(options, undistortedCamera);
			undistortedBitmap = distortedBitmap.Clone();
			if (undistortedCamera.Width != distortedCamera.Width || undistortedCamera.Height != distortedCamera.Height)
			{
				undistortedBitmap.Rescale(undistortedCamera.Width, undistortedCamera.Height);
			}
			return;
		}

		undistortedCamera = UndistortCamera(options, distortedCamera);

		undistortedBitmap = Warp.WarpImageBetweenCameras(
			options.WarpOptions, distortedCamera, undistortedCamera, distortedBitmap, cancellationToken);

		distortedBitmap.CloneMetadata(undistortedBitmap);
	}

	/// <summary>
	/// Port of colmap::UndistortReconstruction: undistort all cameras in the reconstruction
	/// and accordingly all observations in their images.
	/// </summary>
	public static void UndistortReconstruction(UndistortCameraOptions options, Reconstruction reconstruction)
	{
		// A snapshot of the cameras before undistortion (C++ copies the whole map).
		var distortedCameras = new Dictionary<uint, Camera>();
		foreach (var (cameraId, camera) in reconstruction.Cameras)
		{
			distortedCameras.Add(cameraId, camera.Clone());
		}

		// Leave a camera unchanged exactly when the image undistortion also copies its images
		// through unchanged, so that the reconstruction stays consistent with the output
		// images (see COLMAPUndistorter::Undistort). Both spherical cameras (which have no
		// pinhole image plane to undistort to) and already-undistorted perspective cameras are
		// otherwise copied through as-is. When a max_image_size is requested, they are still
		// resized to match the rescaled output images: perspective cameras through
		// undistortion, spherical cameras by resizing to a smaller image of the same model.
		bool KeepUnchanged(Camera camera) => camera.IsUndistorted() && options.MaxImageSize < 0;

		// Each camera is handled independently, so the (hash map) visiting order is irrelevant.
		foreach (var (cameraId, distortedCamera) in distortedCameras)
		{
			if (KeepUnchanged(distortedCamera))
			{
				continue;
			}

			Camera undistortedCamera = reconstruction.Camera(cameraId);
			if (distortedCamera.IsSpherical)
			{
				// Only reached with a max_image_size: resize the spherical camera in place,
				// keeping its model.
				AssignCamera(undistortedCamera, distortedCamera);
				RescaleToMaxImageSize(options, undistortedCamera);
			}
			else
			{
				AssignCamera(undistortedCamera, UndistortCamera(options, distortedCamera));
			}
		}

		foreach (Image image in reconstruction.Images.Values)
		{
			Camera distortedCamera = distortedCameras[image.CameraId];
			// Cameras left unchanged above need no observation rewrite.
			if (KeepUnchanged(distortedCamera))
			{
				continue;
			}

			Camera undistortedCamera = image.CameraPtr;
			List<Point2D> points2D = image.Points2D;

			if (distortedCamera.IsSpherical)
			{
				// Spherical models are only resized (see above). The equirectangular
				// pixel<->angle map is linear in the image dimensions, so observations scale
				// linearly. Unlike the perspective round-trip below, this stays valid for
				// back-hemisphere observations, whose bearing has no forward normalized
				// (CamFromImg) representation.
				double scaleX = (double)undistortedCamera.Width / distortedCamera.Width;
				double scaleY = (double)undistortedCamera.Height / distortedCamera.Height;
				for (int point2DIdx = 0; point2DIdx < points2D.Count; point2DIdx++)
				{
					Point2D point2D = points2D[point2DIdx];
					point2D.Xy = new Vector2d(point2D.Xy.X * scaleX, point2D.Xy.Y * scaleY);
					points2D[point2DIdx] = point2D;
				}
				continue;
			}

			for (int point2DIdx = 0; point2DIdx < points2D.Count; point2DIdx++)
			{
				Point2D point2D = points2D[point2DIdx];
				Vector2d? camPoint = distortedCamera.CamFromImg(point2D.Xy);
				Vector2d? undistortedPoint = camPoint is { } cam ? undistortedCamera.ImgFromCam(cam.Homogeneous()) : null;
				point2D.Xy = undistortedPoint ?? new Vector2d(double.NaN, double.NaN);
				points2D[point2DIdx] = point2D;
			}
		}
	}

	/// <summary>
	/// Port of colmap::RectifyStereoCameras: the rectification homographies that transform two
	/// (already undistorted, pinhole) images so that corresponding pixels lie on the same
	/// scanline, and the matrix Q that maps disparities to world coordinates as
	/// [x, y, disparity, 1] * Q = [X, Y, Z, 1] * w.
	/// </summary>
	public static void RectifyStereoCameras(
		Camera camera1,
		Camera camera2,
		Rigid3d cam2FromCam1,
		out Matrix3d h1,
		out Matrix3d h2,
		out Matrix4d q)
	{
		Check.That(camera1.ModelId == CameraModelId.SimplePinhole || camera1.ModelId == CameraModelId.Pinhole);
		Check.That(camera2.ModelId == CameraModelId.SimplePinhole || camera2.ModelId == CameraModelId.Pinhole);

		// Compute the average rotation between the first and the second camera.
		AngleAxisd fullCam2FromCam1 = AngleAxisd.FromQuaternion(cam2FromCam1.Rotation);
		var halfCam2FromCam1 = new AngleAxisd(fullCam2FromCam1.Angle * -0.5, fullCam2FromCam1.Axis);

		Matrix3d r2 = halfCam2FromCam1.ToRotationMatrix();
		Matrix3d r1 = r2.Transpose();

		// Determine the translation, such that it coincides with the X-axis.
		Vector3d t = r2 * cam2FromCam1.Translation;

		Vector3d xUnitVector = Vector3d.UnitX;
		if (t.Dot(xUnitVector) < 0)
		{
			xUnitVector = xUnitVector * -1;
		}

		Vector3d rotationAxis = t.Cross(xUnitVector);

		Matrix3d rX;
		if (rotationAxis.Norm < LinearAlgebraConstants.MachineEpsilon)
		{
			rX = Matrix3d.Identity;
		}
		else
		{
			double angle = Math.Acos(Math.Abs(t.Dot(xUnitVector)) / (t.Norm * xUnitVector.Norm));
			rX = new AngleAxisd(angle, rotationAxis.Normalized()).ToRotationMatrix();
		}

		// Apply the X-axis correction.
		r1 = rX * r1;
		r2 = rX * r2;
		t = rX * t;

		// Determine the intrinsic calibration matrix.
		double meanFocalLength1 = camera1.MeanFocalLength();
		double meanFocalLength2 = camera2.MeanFocalLength();
		double f = meanFocalLength2 < meanFocalLength1 ? meanFocalLength2 : meanFocalLength1;
		double cx = camera1.PrincipalPointX();
		double cy = (camera1.PrincipalPointY() + camera2.PrincipalPointY()) / 2;
		var k = new Matrix3d(
			f, 0, cx,
			0, f, cy,
			0, 0, 1);

		// Compose the homographies.
		h1 = k * r1 * camera1.CalibrationMatrix().Inverse();
		h2 = k * r2 * camera2.CalibrationMatrix().Inverse();

		// Determine the inverse projection matrix that transforms disparity values to 3D world
		// coordinates: [x, y, disparity, 1] * Q = [X, Y, Z, 1] * w.
		q = new Matrix4d(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, -1 / t.X,
			-cy, -cx, f, 0);
	}

	/// <summary>
	/// Port of colmap::RectifyAndUndistortStereoImages: undistort both images to the pinhole
	/// camera of <paramref name="distortedCamera1"/> and rectify them with the given relative
	/// pose.
	/// </summary>
	public static void RectifyAndUndistortStereoImages(
		UndistortCameraOptions options,
		Bitmap distortedImage1,
		Bitmap distortedImage2,
		Camera distortedCamera1,
		Camera distortedCamera2,
		Rigid3d cam2FromCam1,
		out Bitmap undistortedImage1,
		out Bitmap undistortedImage2,
		out Camera undistortedCamera,
		out Matrix4d q,
		CancellationToken cancellationToken = default)
	{
		Check.Eq(distortedCamera1.Width, distortedImage1.Width);
		Check.Eq(distortedCamera1.Height, distortedImage1.Height);
		Check.Eq(distortedCamera2.Width, distortedImage2.Width);
		Check.Eq(distortedCamera2.Height, distortedImage2.Height);

		undistortedCamera = UndistortCamera(options, distortedCamera1);

		// COLMAP first allocates both outputs and clones the distorted images' metadata into
		// them, but the warps below reassign the outputs to freshly allocated bitmaps, so that
		// metadata never survives; the outputs here are the warps' bitmaps directly.
		RectifyStereoCameras(undistortedCamera, undistortedCamera, cam2FromCam1, out Matrix3d h1, out Matrix3d h2, out q);

		undistortedImage1 = Warp.WarpImageWithHomographyBetweenCameras(
			options.WarpOptions, h1.Inverse(), distortedCamera1, undistortedCamera, distortedImage1, cancellationToken);
		undistortedImage2 = Warp.WarpImageWithHomographyBetweenCameras(
			options.WarpOptions, h2.Inverse(), distortedCamera2, undistortedCamera, distortedImage2, cancellationToken);
	}

	/// <summary>
	/// The border-tracing part of UndistortCamera: trace the ROI border pixels of the distorted
	/// camera into the undistorted one and scale its dimensions and principal point so that
	/// the blank-pixel constraint holds.
	/// </summary>
	private static void ScaleToBorder(
		UndistortCameraOptions options,
		Camera camera,
		Camera undistortedCamera,
		int roiMinX,
		int roiMinY,
		int roiMaxX,
		int roiMaxY)
	{
		// For fisheye camera, CamFromImg returns perspective-normalized coords where
		// |cam_point| = tan(theta). Near theta = pi/2, tan(theta) diverges: border pixels
		// outside the valid fisheye circle (common with off-center principal points) produce
		// extreme coordinates that blow up the output dimensions. Skip any cam_point whose norm
		// exceeds the threshold.
		Check.Ne(options.MaxCamPointNorm, 0);
		double maxCamPointNormSq = options.MaxCamPointNorm < 0
			? double.PositiveInfinity
			: options.MaxCamPointNorm * options.MaxCamPointNorm;

		// Determine min/max coordinates along top / bottom image border.
		double leftMinX = double.MaxValue;
		double leftMaxX = double.MinValue;
		double rightMinX = double.MaxValue;
		double rightMaxX = double.MinValue;

		for (int y = roiMinY; y < roiMaxY; y++)
		{
			// Left border.
			if (TraceBorderPoint(camera, undistortedCamera, new Vector2d(0.5, y + 0.5), maxCamPointNormSq) is { } p1)
			{
				leftMinX = StdMin(leftMinX, p1.X);
				leftMaxX = StdMax(leftMaxX, p1.X);
			}
			// Right border.
			if (TraceBorderPoint(camera, undistortedCamera, new Vector2d(camera.Width - 0.5, y + 0.5), maxCamPointNormSq) is { } p2)
			{
				rightMinX = StdMin(rightMinX, p2.X);
				rightMaxX = StdMax(rightMaxX, p2.X);
			}
		}

		// Determine min, max coordinates along left / right image border.
		double topMinY = double.MaxValue;
		double topMaxY = double.MinValue;
		double bottomMinY = double.MaxValue;
		double bottomMaxY = double.MinValue;

		for (int x = roiMinX; x < roiMaxX; x++)
		{
			// Top border.
			if (TraceBorderPoint(camera, undistortedCamera, new Vector2d(x + 0.5, 0.5), maxCamPointNormSq) is { } p1)
			{
				topMinY = StdMin(topMinY, p1.Y);
				topMaxY = StdMax(topMaxY, p1.Y);
			}
			// Bottom border.
			if (TraceBorderPoint(camera, undistortedCamera, new Vector2d(x + 0.5, camera.Height - 0.5), maxCamPointNormSq) is { } p2)
			{
				bottomMinY = StdMin(bottomMinY, p2.Y);
				bottomMaxY = StdMax(bottomMaxY, p2.Y);
			}
		}

		double cx = undistortedCamera.PrincipalPointX();
		double cy = undistortedCamera.PrincipalPointY();

		// Scale such that undistorted image contains all pixels of distorted image.
		double minScaleX = StdMin(cx / (cx - leftMinX), (undistortedCamera.Width - 0.5 - cx) / (rightMaxX - cx));
		double minScaleY = StdMin(cy / (cy - topMinY), (undistortedCamera.Height - 0.5 - cy) / (bottomMaxY - cy));

		// Scale such that there are no blank pixels in undistorted image.
		double maxScaleX = StdMax(cx / (cx - leftMaxX), (undistortedCamera.Width - 0.5 - cx) / (rightMinX - cx));
		double maxScaleY = StdMax(cy / (cy - topMaxY), (undistortedCamera.Height - 0.5 - cy) / (bottomMinY - cy));

		// Interpolate scale according to blank_pixels.
		double scaleX = 1.0 / (minScaleX * options.BlankPixels + maxScaleX * (1.0 - options.BlankPixels));
		double scaleY = 1.0 / (minScaleY * options.BlankPixels + maxScaleY * (1.0 - options.BlankPixels));

		// Clip the scaling factors.
		scaleX = MathUtils.Clamp(scaleX, options.MinScale, options.MaxScale);
		scaleY = MathUtils.Clamp(scaleY, options.MinScale, options.MaxScale);

		// Scale undistorted camera dimensions.
		int origUndistortedCameraWidth = undistortedCamera.Width;
		int origUndistortedCameraHeight = undistortedCamera.Height;
		undistortedCamera.Width = (int)StdMax(1.0, scaleX * undistortedCamera.Width);
		undistortedCamera.Height = (int)StdMax(1.0, scaleY * undistortedCamera.Height);

		// Scale the principal point according to the new dimensions of the camera.
		undistortedCamera.SetPrincipalPointX(
			undistortedCamera.PrincipalPointX() * undistortedCamera.Width / origUndistortedCameraWidth);
		undistortedCamera.SetPrincipalPointY(
			undistortedCamera.PrincipalPointY() * undistortedCamera.Height / origUndistortedCameraHeight);
	}

	/// <summary>
	/// A distorted border pixel mapped into the undistorted camera, or null when it does not
	/// lift, exceeds the camera-point norm limit, or does not project.
	/// </summary>
	private static Vector2d? TraceBorderPoint(Camera camera, Camera undistortedCamera, Vector2d imagePoint, double maxCamPointNormSq)
	{
		if (camera.CamFromImg(imagePoint) is { } camPoint && camPoint.SquaredNorm < maxCamPointNormSq)
		{
			return undistortedCamera.ImgFromCam(camPoint.Homogeneous());
		}

		return null;
	}

	/// <summary>
	/// Port of the anonymous RescaleToMaxImageSize: rescale the camera in place so that neither
	/// dimension exceeds MaxImageSize, keeping its model. A no-op without a size limit or when
	/// the camera already fits.
	/// </summary>
	private static void RescaleToMaxImageSize(UndistortCameraOptions options, Camera camera)
	{
		if (options.MaxImageSize < 0)
		{
			return;
		}

		double maxImageScale = StdMin(
			options.MaxImageSize / (double)camera.Width,
			options.MaxImageSize / (double)camera.Height);
		if (maxImageScale < 1.0)
		{
			camera.Rescale(maxImageScale);
		}
	}

	/// <summary>
	/// C++ copy assignment of a Camera, except that the target keeps its CameraId (and so its
	/// SensorId); COLMAP would copy the source's, which UndistortCamera leaves invalid
	/// (divergence 60).
	/// </summary>
	private static void AssignCamera(Camera target, Camera source)
	{
		target.ModelId = source.ModelId;
		target.Width = source.Width;
		target.Height = source.Height;
		target.Params = (double[])source.Params.Clone();
		target.HasPriorFocalLength = source.HasPriorFocalLength;
	}

	/// <summary>std::min(a, b): (b &lt; a) ? b : a.</summary>
	private static double StdMin(double a, double b) => b < a ? b : a;

	/// <summary>std::max(a, b): (a &lt; b) ? b : a.</summary>
	private static double StdMax(double a, double b) => a < b ? b : a;
}
