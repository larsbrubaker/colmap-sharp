// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The perspective fisheye camera models of colmap/sensor/models.h (the
// PERSPECTIVE_FISHEYE_CAMERA_MODEL_CASES): OPENCV_FISHEYE, SIMPLE_RADIAL_FISHEYE,
// RADIAL_FISHEYE, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE, SIMPLE_FISHEYE and
// FISHEYE. Each projects the normalized point onto the equidistant fisheye plane
// (CameraModelMath.FisheyeFromNormal), distorts it there, and maps it to pixels;
// CamFromImg undoes the steps in reverse, the distortion by the iterative undistortion.
// THIN_PRISM_FISHEYE and RAD_TAN_THIN_PRISM_FISHEYE live in FisheyeCameraModels.ThinPrism.cs.
// Siblings: PinholeCameraModels.cs, OtherCameraModels.cs; base and design notes in
// CameraModelBase.cs. Every expression keeps COLMAP's operator order (Tier A).

using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

/// <summary>
/// OpenCV fish-eye camera model: pinhole with radial distortion up to the 4th degree of
/// coefficients, suitable for the large radial distortions of fish-eye cameras.
/// Parameters: fx, fy, cx, cy, k1, k2, k3, k4.
/// See http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
/// </summary>
public readonly partial struct OpenCVFisheyeCameraModel : IPerspectiveFisheyeCameraModel<OpenCVFisheyeCameraModel>, IDistortedCameraModel<OpenCVFisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.OpenCVFisheye;

	/// <inheritdoc/>
	public static string ModelName => "OPENCV_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k1, k2, k3, k4";

	/// <inheritdoc/>
	public static int NumParams => 8;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [4, 5, 6, 7];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0, 0, 0, 0, 0];

	/// <inheritdoc/>
	public static void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>
	{
		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		x = f1 * uu + c1;
		y = f2 * vv + c2;
	}

	/// <inheritdoc/>
	public static void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		uu = (x - c1) / f1;
		vv = (y - c2) / f2;
	}

	/// <inheritdoc/>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		CameraModelMath.FisheyeFromNormal(u / w, v / w, out T uu, out T vv);

		// Distortion
		Distortion(parameters[4..], uu, vv, out T duu, out T dvv);

		// Transform to image coordinates
		ImgFromFisheye(parameters, uu + duu, vv + dvv, out x, out y);
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		FisheyeFromImg(Real.Cast(parameters), x, y, out Real ruu, out Real rvv);
		double uu = ruu;
		double vv = rvv;
		if (!CameraModelMath.IterativeUndistortion<OpenCVFisheyeCameraModel>(parameters[4..], ref uu, ref vv))
		{
			u = 0;
			v = 0;
			return false;
		}

		CameraModelMath.NormalFromFisheye<Real>(uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k1 = extraParams[0];
		T k2 = extraParams[1];
		T k3 = extraParams[2];
		T k4 = extraParams[3];

		T theta2 = u * u + v * v;
		T theta4 = theta2 * theta2;
		T theta6 = theta4 * theta2;
		T theta8 = theta4 * theta4;

		T radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
		du = u * radial;
		dv = v * radial;
	}
}

/// <summary>
/// Simple camera model with one focal length and one radial distortion parameter,
/// suitable for fish-eye cameras: OPENCV_FISHEYE with a single radial coefficient.
/// Parameters: f, cx, cy, k.
/// </summary>
public readonly partial struct SimpleRadialFisheyeCameraModel : IPerspectiveFisheyeCameraModel<SimpleRadialFisheyeCameraModel>, IDistortedCameraModel<SimpleRadialFisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.SimpleRadialFisheye;

	/// <inheritdoc/>
	public static string ModelName => "SIMPLE_RADIAL_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "f, cx, cy, k";

	/// <inheritdoc/>
	public static int NumParams => 4;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [1, 2];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [3];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, width / 2.0, height / 2.0, 0];

	/// <inheritdoc/>
	public static void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		x = f * uu + c1;
		y = f * vv + c2;
	}

	/// <inheritdoc/>
	public static void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		uu = (x - c1) / f;
		vv = (y - c2) / f;
	}

	/// <inheritdoc/>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		CameraModelMath.FisheyeFromNormal(u / w, v / w, out T uu, out T vv);

		// Distortion
		Distortion(parameters[3..], uu, vv, out T duu, out T dvv);

		// Transform to image coordinates
		ImgFromFisheye(parameters, uu + duu, vv + dvv, out x, out y);
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		FisheyeFromImg(Real.Cast(parameters), x, y, out Real ruu, out Real rvv);
		double uu = ruu;
		double vv = rvv;
		if (!CameraModelMath.IterativeUndistortion<SimpleRadialFisheyeCameraModel>(parameters[3..], ref uu, ref vv))
		{
			u = 0;
			v = 0;
			return false;
		}

		CameraModelMath.NormalFromFisheye<Real>(uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k = extraParams[0];

		T theta2 = u * u + v * v;
		T radial = k * theta2;
		du = u * radial;
		dv = v * radial;
	}
}

/// <summary>
/// Simple camera model with one focal length and two radial distortion parameters,
/// suitable for fish-eye cameras: OPENCV_FISHEYE with two radial coefficients.
/// Parameters: f, cx, cy, k1, k2.
/// </summary>
public readonly partial struct RadialFisheyeCameraModel : IPerspectiveFisheyeCameraModel<RadialFisheyeCameraModel>, IDistortedCameraModel<RadialFisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.RadialFisheye;

	/// <inheritdoc/>
	public static string ModelName => "RADIAL_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "f, cx, cy, k1, k2";

	/// <inheritdoc/>
	public static int NumParams => 5;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [1, 2];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [3, 4];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, width / 2.0, height / 2.0, 0, 0];

	/// <inheritdoc/>
	public static void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		x = f * uu + c1;
		y = f * vv + c2;
	}

	/// <inheritdoc/>
	public static void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		uu = (x - c1) / f;
		vv = (y - c2) / f;
	}

	/// <inheritdoc/>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		CameraModelMath.FisheyeFromNormal(u / w, v / w, out T uu, out T vv);

		// Distortion
		Distortion(parameters[3..], uu, vv, out T duu, out T dvv);

		// Transform to image coordinates
		ImgFromFisheye(parameters, uu + duu, vv + dvv, out x, out y);
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		FisheyeFromImg(Real.Cast(parameters), x, y, out Real ruu, out Real rvv);
		double uu = ruu;
		double vv = rvv;
		if (!CameraModelMath.IterativeUndistortion<RadialFisheyeCameraModel>(parameters[3..], ref uu, ref vv))
		{
			u = 0;
			v = 0;
			return false;
		}

		CameraModelMath.NormalFromFisheye<Real>(uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k1 = extraParams[0];
		T k2 = extraParams[1];

		T theta2 = u * u + v * v;
		T theta4 = theta2 * theta2;

		T radial = k1 * theta2 + k2 * theta4;
		du = u * radial;
		dv = v * radial;
	}
}

/// <summary>
/// Simple equidistant fisheye camera model (theta = r) without distortion parameters, for
/// fish-eye cameras whose distortion can be ignored or has been pre-corrected. One focal
/// length. Parameters: f, cx, cy.
/// </summary>
public readonly partial struct SimpleFisheyeCameraModel : IPerspectiveFisheyeCameraModel<SimpleFisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.SimpleFisheye;

	/// <inheritdoc/>
	public static string ModelName => "SIMPLE_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "f, cx, cy";

	/// <inheritdoc/>
	public static int NumParams => 3;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [1, 2];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, width / 2.0, height / 2.0];

	/// <inheritdoc/>
	public static void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		x = f * uu + c1;
		y = f * vv + c2;
	}

	/// <inheritdoc/>
	public static void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		uu = (x - c1) / f;
		vv = (y - c2) / f;
	}

	/// <inheritdoc/>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		CameraModelMath.FisheyeFromNormal(u / w, v / w, out T uu, out T vv);

		// No distortion

		// Transform to image coordinates
		ImgFromFisheye(parameters, uu, vv, out x, out y);
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		FisheyeFromImg(Real.Cast(parameters), x, y, out Real uu, out Real vv);

		// No undistortion needed
		CameraModelMath.NormalFromFisheye(uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}
}

/// <summary>
/// Equidistant fisheye camera model (theta = r) without distortion parameters, for
/// fish-eye cameras whose distortion can be ignored or has been pre-corrected. Two focal
/// lengths. Parameters: fx, fy, cx, cy.
/// </summary>
public readonly partial struct FisheyeCameraModel : IPerspectiveFisheyeCameraModel<FisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.Fisheye;

	/// <inheritdoc/>
	public static string ModelName => "FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy";

	/// <inheritdoc/>
	public static int NumParams => 4;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0];

	/// <inheritdoc/>
	public static void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>
	{
		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		x = f1 * uu + c1;
		y = f2 * vv + c2;
	}

	/// <inheritdoc/>
	public static void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		uu = (x - c1) / f1;
		vv = (y - c2) / f2;
	}

	/// <inheritdoc/>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		CameraModelMath.FisheyeFromNormal(u / w, v / w, out T uu, out T vv);

		// No distortion

		// Transform to image coordinates
		ImgFromFisheye(parameters, uu, vv, out x, out y);
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		FisheyeFromImg(Real.Cast(parameters), x, y, out Real uu, out Real vv);

		// No undistortion needed
		CameraModelMath.NormalFromFisheye(uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}
}
