// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Camera: port of colmap/scene/camera.h and camera.cc, the intrinsics (model, image size,
// parameter vector) that images share. Every method forwards to the runtime-dispatched
// model functions of Sensor/CameraModels.cs, so the results are those of the camera models
// (Tier A, see Sensor/CameraModelBase.cs). Images reference cameras (Scene/Image.cs);
// TwoViewGeometry.cs carries optional per-side estimates. Tests:
// ColmapSharp.Tests/Scene/CameraTests.cs (camera_test.cc 1:1).
//
// Design (later phases build on it):
// - A class, because images hold references to a shared camera (C++ Camera*), which the
//   Reconstruction (a later Phase 4 port) owns. C++ copies cameras by value
//   (`Camera other = camera;`); here that is Clone(), and assigning a reference shares the
//   camera. Equality is COLMAP's value equality (operator==), as for Sensor/Rig.cs.
// - Params is a plain double[] that callers may replace or write into. Phase 8's bundle
//   adjustment registers it as a parameter block and writes the refined intrinsics
//   straight into it, as COLMAP's Ceres problem does with `camera.params.data()`. Code that
//   holds the array across a SetParamsFromString or a Params assignment holds the old one.
// - Width and height are int (size_t in C++), matching Sensor/CameraModels.cs.
// - std::round is half away from zero, hence MidpointRounding.AwayFromZero in Rescale.

using System.Globalization;
using System.Runtime.InteropServices;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Camera: the intrinsic parameters of a camera. Cameras may be shared
/// between multiple images, e.g., if the same "physical" camera took multiple pictures with
/// the exact same lens and intrinsics (focal length, etc.). The distortion model is given
/// by the camera model.
/// </summary>
public sealed class Camera : IEquatable<Camera>
{
	/// <summary>The unique identifier of the camera; <see cref="InvalidCameraId"/> until set.</summary>
	public uint CameraId { get; set; } = InvalidCameraId;

	/// <summary>The identifier of the camera model.</summary>
	public CameraModelId ModelId { get; set; } = CameraModelId.Invalid;

	/// <summary>The image width, 0 if not initialized.</summary>
	public int Width { get; set; }

	/// <summary>The image height, 0 if not initialized.</summary>
	public int Height { get; set; }

	/// <summary>
	/// The focal length, principal point, and extra parameters; empty if the camera model is
	/// not specified. Mutable in place (see the file header).
	/// </summary>
	public double[] Params { get; set; } = [];

	/// <summary>
	/// Whether there is a good prior for the focal length, e.g. manually provided, extracted
	/// from EXIF, or from view graph calibration.
	/// </summary>
	public bool HasPriorFocalLength { get; set; }

	/// <summary>The camera model name, or "" for an unknown model.</summary>
	public string ModelName => CameraModels.CameraModelIdToName(ModelId);

	/// <summary>The sensor id (CAMERA, camera_id).</summary>
	public SensorId SensorId => new(SensorType.Camera, CameraId);

	/// <summary>Indices of the focal length parameters; throws for an unknown model.</summary>
	public ReadOnlySpan<int> FocalLengthIdxs => CameraModels.CameraModelFocalLengthIdxs(ModelId);

	/// <summary>Indices of the principal point parameters; throws for an unknown model.</summary>
	public ReadOnlySpan<int> PrincipalPointIdxs => CameraModels.CameraModelPrincipalPointIdxs(ModelId);

	/// <summary>Indices of the extra (distortion) parameters; throws for an unknown model.</summary>
	public ReadOnlySpan<int> ExtraParamsIdxs => CameraModels.CameraModelExtraParamsIdxs(ModelId);

	/// <summary>Indices of the metadata parameters; throws for an unknown model.</summary>
	public ReadOnlySpan<int> MetaDataParamsIdxs => CameraModels.CameraModelMetaDataParamsIdxs(ModelId);

	/// <summary>Human-readable parameter order; throws for an unknown model.</summary>
	public string ParamsInfo => CameraModels.CameraModelParamsInfo(ModelId);

	/// <summary>
	/// Whether the camera model is perspective, i.e. has a focal length and a finite pinhole
	/// image plane (so positive-depth cheirality applies). EQUIRECTANGULAR is not.
	/// </summary>
	public bool IsPerspective => CameraModels.CameraModelIsPerspective(ModelId);

	/// <summary>Whether the model is spherical (the EQUIRECTANGULAR panorama model).</summary>
	public bool IsSpherical => CameraModels.CameraModelIsSpherical(ModelId);

	/// <summary>Whether the model is perspective and fisheye.</summary>
	public bool IsPerspectiveFisheye => CameraModels.CameraModelIsPerspectiveFisheye(ModelId);

	/// <summary>Whether the model is perspective and not fisheye.</summary>
	public bool IsPerspectivePinhole => CameraModels.CameraModelIsPerspectivePinhole(ModelId);

	/// <summary>
	/// A camera of the given model with all focal lengths set to
	/// <paramref name="focalLength"/> and the principal point at the image center.
	/// </summary>
	public static Camera CreateFromModelId(uint cameraId, CameraModelId modelId, double focalLength, int width, int height)
	{
		Check.That(CameraModels.ExistsCameraModelWithId(modelId));
		return new Camera
		{
			CameraId = cameraId,
			ModelId = modelId,
			Width = width,
			Height = height,
			Params = CameraModels.CameraModelInitializeParams(modelId, focalLength, width, height),
		};
	}

	/// <summary><see cref="CreateFromModelId"/> with the model given by name.</summary>
	public static Camera CreateFromModelName(uint cameraId, string modelName, double focalLength, int width, int height) =>
		CreateFromModelId(cameraId, CameraModels.CameraModelNameToId(modelName), focalLength, width, height);

	/// <summary>A copy with its own parameter array (C++ copy construction).</summary>
	public Camera Clone() => new()
	{
		CameraId = CameraId,
		ModelId = ModelId,
		Width = Width,
		Height = Height,
		Params = (double[])Params.Clone(),
		HasPriorFocalLength = HasPriorFocalLength,
	};

	/// <summary>The mean of the focal length parameters; 0 for spherical models.</summary>
	public double MeanFocalLength()
	{
		// The omnidirectional EQUIRECTANGULAR model has no focal length.
		if (IsSpherical)
		{
			return 0.0;
		}

		ReadOnlySpan<int> focalLengthIdxs = FocalLengthIdxs;
		double focalLength = 0;
		foreach (int idx in focalLengthIdxs)
		{
			focalLength += Params[idx];
		}

		return focalLength / focalLengthIdxs.Length;
	}

	/// <summary>The single focal length; throws if the model has separate fx and fy.</summary>
	public double FocalLength()
	{
		ReadOnlySpan<int> idxs = FocalLengthIdxs;
		Check.Eq(idxs.Length, 1);
		return Params[idxs[0]];
	}

	/// <summary>The x focal length (the single one for one-focal models).</summary>
	public double FocalLengthX() => Params[FocalLengthIdxs[0]];

	/// <summary>The y focal length (the single one for one-focal models).</summary>
	public double FocalLengthY()
	{
		ReadOnlySpan<int> idxs = FocalLengthIdxs;
		return Params[idxs[(idxs.Length == 1) ? 0 : 1]];
	}

	/// <summary>Sets every focal length parameter to <paramref name="f"/>.</summary>
	public void SetFocalLength(double f)
	{
		foreach (int idx in FocalLengthIdxs)
		{
			Params[idx] = f;
		}
	}

	/// <summary>Sets fx; throws unless the model has separate fx and fy.</summary>
	public void SetFocalLengthX(double fx)
	{
		ReadOnlySpan<int> idxs = FocalLengthIdxs;
		Check.Eq(idxs.Length, 2);
		Params[idxs[0]] = fx;
	}

	/// <summary>Sets fy; throws unless the model has separate fx and fy.</summary>
	public void SetFocalLengthY(double fy)
	{
		ReadOnlySpan<int> idxs = FocalLengthIdxs;
		Check.Eq(idxs.Length, 2);
		Params[idxs[1]] = fy;
	}

	/// <summary>The principal point x; throws unless the model has one.</summary>
	public double PrincipalPointX()
	{
		ReadOnlySpan<int> idxs = PrincipalPointIdxs;
		Check.Eq(idxs.Length, 2);
		return Params[idxs[0]];
	}

	/// <summary>The principal point y; throws unless the model has one.</summary>
	public double PrincipalPointY()
	{
		ReadOnlySpan<int> idxs = PrincipalPointIdxs;
		Check.Eq(idxs.Length, 2);
		return Params[idxs[1]];
	}

	/// <summary>The principal point (cx, cy).</summary>
	public Vector2d PrincipalPoint() => new(PrincipalPointX(), PrincipalPointY());

	/// <summary>Sets the principal point x.</summary>
	public void SetPrincipalPointX(double cx)
	{
		ReadOnlySpan<int> idxs = PrincipalPointIdxs;
		Check.Eq(idxs.Length, 2);
		Params[idxs[0]] = cx;
	}

	/// <summary>Sets the principal point y.</summary>
	public void SetPrincipalPointY(double cy)
	{
		ReadOnlySpan<int> idxs = PrincipalPointIdxs;
		Check.Eq(idxs.Length, 2);
		Params[idxs[1]] = cy;
	}

	/// <summary>
	/// The intrinsic calibration matrix from focal length and principal point, excluding
	/// distortion. This is the affine part of the projection, a projective camera matrix only
	/// for pinhole models: fisheye normalized coordinates are angular, so callers relying on
	/// x ~ K * [R | t] * X must restrict themselves to IsPerspectivePinhole cameras.
	/// </summary>
	public Matrix3d CalibrationMatrix()
	{
		Check.That(IsPerspective, "CalibrationMatrix() only defined for perspective cameras.");
		return Matrix3d.FromRows(
			new Vector3d(FocalLengthX(), 0, PrincipalPointX()),
			new Vector3d(0, FocalLengthY(), PrincipalPointY()),
			new Vector3d(0, 0, 1));
	}

	/// <summary>The parameters as a comma-separated list, e.g. "1, 0.5, 0.5".</summary>
	public string ParamsToString() => Misc.VectorToCsv(Params);

	/// <summary>
	/// Sets the parameters from a comma-separated list. Returns false, leaving the
	/// parameters unchanged, if the list does not fit the model.
	/// </summary>
	public bool SetParamsFromString(string value)
	{
		List<double> newCameraParams = Misc.CsvToDoubleVector(value);
		if (!CameraModels.CameraModelVerifyParams(ModelId, CollectionsMarshal.AsSpan(newCameraParams)))
		{
			return false;
		}

		Params = [.. newCameraParams];
		return true;
	}

	/// <summary>Whether the parameter count matches the model; throws for an unknown model.</summary>
	public bool VerifyParams() => CameraModels.CameraModelVerifyParams(ModelId, Params);

	/// <summary>
	/// Whether every extra (distortion) parameter is within 1e-8 of zero. Spherical cameras
	/// have no pinhole image plane to undistort to and count as undistorted, so undistortion
	/// is a no-op for them.
	/// </summary>
	public bool IsUndistorted()
	{
		if (IsSpherical)
		{
			return true;
		}

		foreach (int idx in ExtraParamsIdxs)
		{
			if (Math.Abs(Params[idx]) > 1e-8)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Whether the camera has bogus parameters (see CameraModelHasBogusParams).</summary>
	public bool HasBogusParams(double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam) =>
		CameraModels.CameraModelHasBogusParams(ModelId, Params, Width, Height, minFocalLengthRatio, maxFocalLengthRatio, maxExtraParam);

	/// <summary>Lifts an image point to the normalized camera plane (not unit normalized), or null.</summary>
	public Vector2d? CamFromImg(Vector2d imagePoint) => CameraModels.CameraModelCamFromImg(ModelId, Params, imagePoint);

	/// <summary>
	/// Unprojects a pixel to a unit bearing in the camera frame. Unlike CamFromImg (limited
	/// to the forward hemisphere) this works for any pixel the model can unproject,
	/// including back-facing rays of omnidirectional cameras.
	/// </summary>
	public Vector3d? CamRayFromImg(Vector2d imagePoint) => CameraModels.CameraModelCamRayFromImg(ModelId, Params, imagePoint);

	/// <summary>Converts a pixel threshold in the image plane to the camera frame.</summary>
	public double CamFromImgThreshold(double threshold) => CameraModels.CameraModelCamFromImgThreshold(ModelId, Params, threshold);

	/// <summary>
	/// Projects a point from the camera frame to the image plane, or null. Without the
	/// cheirality check, points behind the camera are projected as well and only points on
	/// the camera plane fail.
	/// </summary>
	public Vector2d? ImgFromCam(Vector3d camPoint, bool checkCheirality = true) =>
		CameraModels.CameraModelImgFromCam(ModelId, Params, camPoint, checkCheirality);

	/// <summary>
	/// <see cref="ImgFromCam"/> that also computes the Jacobian d(x, y) / d(u, v, w)
	/// (zero when the projection fails).
	/// </summary>
	public Vector2d? ImgFromCamWithJac(Vector3d camPoint, out Matrix2x3d jUvw, bool checkCheirality = true) =>
		CameraModels.CameraModelImgFromCamWithJac(ModelId, Params, camPoint, out jUvw, checkCheirality);

	/// <summary><see cref="ImgFromCam"/> through the analytic kernel, skipping the Jacobian (C++ nullptr).</summary>
	public Vector2d? ImgFromCamWithJac(Vector3d camPoint, bool checkCheirality = true) =>
		CameraModels.CameraModelImgFromCamWithJac(ModelId, Params, camPoint, checkCheirality);

	/// <summary>
	/// Unprojects a pixel to a unit bearing together with the Jacobian d(u, v, w) / d(x, y)
	/// of that bearing with respect to the pixel. The Jacobian maps image-space
	/// perturbations into the tangent plane of the unit sphere at the bearing, which lets an
	/// epipolar residual be evaluated in pixel units for any central camera model. Null if
	/// the pixel cannot be unprojected or the projection is rank deficient there.
	/// </summary>
	public CamRayWithJac? CamRayFromImgWithJac(Vector2d imagePoint)
	{
		if (CamRayFromImg(imagePoint) is not Vector3d camRay)
		{
			return null;
		}

		if (!ImgFromCamWithJac(camRay, out Matrix2x3d jUvw).HasValue)
		{
			return null;
		}

		if (CameraModels.CamRayFromImgJacobian(camRay, jUvw) is not Matrix3x2d jRay)
		{
			return null;
		}

		return new CamRayWithJac(camRay, jRay);
	}

	/// <summary>
	/// Rescales the image dimensions by <paramref name="scale"/> (rounded to whole pixels)
	/// and the focal length and principal point accordingly.
	/// </summary>
	public void Rescale(double scale)
	{
		Check.Gt(scale, 0.0);
		double scaleX = Math.Round(scale * Width, MidpointRounding.AwayFromZero) / Width;
		double scaleY = Math.Round(scale * Height, MidpointRounding.AwayFromZero) / Height;
		Width = (int)Math.Round(scale * Width, MidpointRounding.AwayFromZero);
		Height = (int)Math.Round(scale * Height, MidpointRounding.AwayFromZero);
		CameraModels.CameraModelRescale(ModelId, scaleX, scaleY, Params);
	}

	/// <summary>Rescales to the given image dimensions, and the parameters accordingly.</summary>
	public void Rescale(int newWidth, int newHeight)
	{
		double scaleX = (double)newWidth / Width;
		double scaleY = (double)newHeight / Height;
		Width = newWidth;
		Height = newHeight;
		CameraModels.CameraModelRescale(ModelId, scaleX, scaleY, Params);
	}

	/// <summary>The C++ operator==: same id, model, size, prior flag and exactly equal parameters.</summary>
	public bool Equals(Camera? other)
	{
		if (other is null
			|| CameraId != other.CameraId
			|| ModelId != other.ModelId
			|| Width != other.Width
			|| Height != other.Height
			|| HasPriorFocalLength != other.HasPriorFocalLength
			|| Params.Length != other.Params.Length)
		{
			return false;
		}

		// std::vector<double> == compares with double ==, so NaN never equals (unlike
		// SequenceEqual's EqualityComparer<double>).
		for (int i = 0; i < Params.Length; i++)
		{
			if (Params[i] != other.Params[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => Equals(obj as Camera);

	/// <summary>Hash of the identity fields (the parameters may change in place).</summary>
	public override int GetHashCode() => HashCode.Combine(CameraId, ModelId);

	/// <summary>The C++ operator==.</summary>
	public static bool operator ==(Camera? a, Camera? b) => a is null ? b is null : a.Equals(b);

	/// <summary>The C++ operator!=.</summary>
	public static bool operator !=(Camera? a, Camera? b) => !(a == b);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g. "Camera(camera_id=1, model=SIMPLE_PINHOLE, width=1,
	/// height=1, params=[1, 0.5, 0.5] (f, cx, cy))".
	/// </summary>
	public override string ToString()
	{
		bool validModel = CameraModels.ExistsCameraModelWithId(ModelId);
		string cameraIdStr = CameraId != InvalidCameraId ? CameraId.ToString(CultureInfo.InvariantCulture) : "Invalid";
		string paramsInfo = validModel ? ParamsInfo : "?";
		string modelName = validModel ? ModelName : "Invalid";
		return string.Create(
			CultureInfo.InvariantCulture,
			$"Camera(camera_id={cameraIdStr}, model={modelName}, width={Width}, height={Height}, params=[{ParamsToString()}] ({paramsInfo}))");
	}
}
