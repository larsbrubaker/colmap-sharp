// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The perspective pinhole camera models of colmap/sensor/models.h that project as
// x = X / Z and then distort the normalized plane: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL,
// RADIAL, OPENCV, FULL_OPENCV and FOV. The division models and EUCM, also pinhole-based,
// are in OtherCameraModels.cs; the fisheye models in FisheyeCameraModels.cs. The shared
// base behavior and the design notes are in CameraModelBase.cs.
//
// Every expression keeps COLMAP's operator order (Tier A), e.g. `f * u / w + c1` is
// ((f * u) / w) + c1, so results are bit-identical to the C++ for the same input.

using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

/// <summary>
/// Simple Pinhole camera model. No distortion is assumed; only the focal length and the
/// principal point are modeled. Parameters: f, cx, cy.
/// See https://en.wikipedia.org/wiki/Pinhole_camera_model
/// </summary>
public readonly partial struct SimplePinholeCameraModel : IPerspectivePinholeCameraModel<SimplePinholeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.SimplePinhole;

	/// <inheritdoc/>
	public static string ModelName => "SIMPLE_PINHOLE";

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
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		// No Distortion

		// Transform to image coordinates
		x = f * u / w + c1;
		y = f * v / w + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];

		u = (x - c1) / f;
		v = (y - c2) / f;
		return true;
	}
}

/// <summary>
/// Pinhole camera model. No distortion is assumed; only the focal lengths and the principal
/// point are modeled. Parameters: fx, fy, cx, cy.
/// See https://en.wikipedia.org/wiki/Pinhole_camera_model
/// </summary>
public readonly partial struct PinholeCameraModel : IPerspectivePinholeCameraModel<PinholeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.Pinhole;

	/// <inheritdoc/>
	public static string ModelName => "PINHOLE";

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
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		// No Distortion

		// Transform to image coordinates
		x = f1 * u / w + c1;
		y = f2 * v / w + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];

		u = (x - c1) / f1;
		v = (y - c2) / f2;
		return true;
	}
}

/// <summary>
/// Simple camera model with one focal length and one radial distortion parameter. Similar
/// to VisualSfM's model, except that the distortion is applied to the projections and not
/// to the measurements. Parameters: f, cx, cy, k.
/// </summary>
public readonly partial struct SimpleRadialCameraModel : IPerspectivePinholeCameraModel<SimpleRadialCameraModel>, IDistortedCameraModel<SimpleRadialCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.SimpleRadial;

	/// <inheritdoc/>
	public static string ModelName => "SIMPLE_RADIAL";

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
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		T uu = u / w;
		T vv = v / w;

		// Distortion
		Distortion(parameters[3..], uu, vv, out T du, out T dv);
		x = uu + du;
		y = vv + dv;

		// Transform to image coordinates
		x = f * x + c1;
		y = f * y + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];

		// Lift points to normalized plane
		u = (x - c1) / f;
		v = (y - c2) / f;
		return CameraModelMath.IterativeUndistortion<SimpleRadialCameraModel>(parameters[3..], ref u, ref v);
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k = extraParams[0];

		T u2 = u * u;
		T v2 = v * v;
		T r2 = u2 + v2;
		T radial = k * r2;
		du = u * radial;
		dv = v * radial;
	}
}

/// <summary>
/// Simple camera model with one focal length and two radial distortion parameters,
/// equivalent to Bundler's model (except for an inverse z-axis in the camera coordinate
/// system). Parameters: f, cx, cy, k1, k2.
/// </summary>
public readonly partial struct RadialCameraModel : IPerspectivePinholeCameraModel<RadialCameraModel>, IDistortedCameraModel<RadialCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.Radial;

	/// <inheritdoc/>
	public static string ModelName => "RADIAL";

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
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];

		T uu = u / w;
		T vv = v / w;

		// Distortion
		Distortion(parameters[3..], uu, vv, out T du, out T dv);
		x = uu + du;
		y = vv + dv;

		// Transform to image coordinates
		x = f * x + c1;
		y = f * y + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];

		// Lift points to normalized plane
		u = (x - c1) / f;
		v = (y - c2) / f;
		return CameraModelMath.IterativeUndistortion<RadialCameraModel>(parameters[3..], ref u, ref v);
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k1 = extraParams[0];
		T k2 = extraParams[1];

		T u2 = u * u;
		T v2 = v * v;
		T r2 = u2 + v2;
		T radial = k1 * r2 + k2 * r2 * r2;
		du = u * radial;
		dv = v * radial;
	}
}

/// <summary>
/// OpenCV camera model: pinhole with radial and tangential distortion (up to 2nd degree of
/// coefficients). Not suitable for the large radial distortions of fish-eye cameras.
/// Parameters: fx, fy, cx, cy, k1, k2, p1, p2.
/// See http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
/// </summary>
public readonly partial struct OpenCVCameraModel : IPerspectivePinholeCameraModel<OpenCVCameraModel>, IDistortedCameraModel<OpenCVCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.OpenCV;

	/// <inheritdoc/>
	public static string ModelName => "OPENCV";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k1, k2, p1, p2";

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
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		T uu = u / w;
		T vv = v / w;

		// Distortion
		Distortion(parameters[4..], uu, vv, out T du, out T dv);
		x = uu + du;
		y = vv + dv;

		// Transform to image coordinates
		x = f1 * x + c1;
		y = f2 * y + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];

		// Lift points to normalized plane
		u = (x - c1) / f1;
		v = (y - c2) / f2;
		return CameraModelMath.IterativeUndistortion<OpenCVCameraModel>(parameters[4..], ref u, ref v);
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k1 = extraParams[0];
		T k2 = extraParams[1];
		T p1 = extraParams[2];
		T p2 = extraParams[3];
		T two = T.FromDouble(2);

		T u2 = u * u;
		T uv = u * v;
		T v2 = v * v;
		T r2 = u2 + v2;
		T radial = k1 * r2 + k2 * r2 * r2;
		du = u * radial + two * p1 * uv + p2 * (r2 + two * u2);
		dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2);
	}
}

/// <summary>
/// Full OpenCV camera model: pinhole with rational radial and tangential distortion.
/// Parameters: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6.
/// See http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
/// </summary>
public readonly partial struct FullOpenCVCameraModel : IPerspectivePinholeCameraModel<FullOpenCVCameraModel>, IDistortedCameraModel<FullOpenCVCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.FullOpenCV;

	/// <inheritdoc/>
	public static string ModelName => "FULL_OPENCV";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";

	/// <inheritdoc/>
	public static int NumParams => 12;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [4, 5, 6, 7, 8, 9, 10, 11];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0, 0, 0, 0, 0, 0, 0, 0, 0];

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

		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		T uu = u / w;
		T vv = v / w;

		// Distortion
		Distortion(parameters[4..], uu, vv, out T du, out T dv);
		x = uu + du;
		y = vv + dv;

		// Transform to image coordinates
		x = f1 * x + c1;
		y = f2 * y + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];

		// Lift points to normalized plane
		u = (x - c1) / f1;
		v = (y - c2) / f2;
		return CameraModelMath.IterativeUndistortion<FullOpenCVCameraModel>(parameters[4..], ref u, ref v);
	}

	/// <inheritdoc/>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k1 = extraParams[0];
		T k2 = extraParams[1];
		T p1 = extraParams[2];
		T p2 = extraParams[3];
		T k3 = extraParams[4];
		T k4 = extraParams[5];
		T k5 = extraParams[6];
		T k6 = extraParams[7];
		T one = T.FromDouble(1);
		T two = T.FromDouble(2);

		T u2 = u * u;
		T uv = u * v;
		T v2 = v * v;
		T r2 = u2 + v2;
		T r4 = r2 * r2;
		T r6 = r4 * r2;
		T radial = (one + k1 * r2 + k2 * r4 + k3 * r6) / (one + k4 * r2 + k5 * r4 + k6 * r6);
		du = u * radial + two * p1 * uv + p2 * (r2 + two * u2) - u;
		dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2) - v;
	}
}

/// <summary>
/// FOV camera model: pinhole with a one-parameter radial distortion, used for example by
/// Project Tango for its equidistant calibration type. Parameters: fx, fy, cx, cy, omega.
/// See: Frederic Devernay, Olivier Faugeras. Straight lines have to be straight: Automatic
/// calibration and removal of distortion from scenes of structured environments. Machine
/// vision and applications, 2001.
/// </summary>
public readonly partial struct FOVCameraModel : IPerspectivePinholeCameraModel<FOVCameraModel>, IDistortedCameraModel<FOVCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.FOV;

	/// <inheritdoc/>
	public static string ModelName => "FOV";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, omega";

	/// <inheritdoc/>
	public static int NumParams => 5;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [4];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0, 1e-2];

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

		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];

		// Distortion
		Distortion(parameters[4..], u / w, v / w, out x, out y);

		// Transform to image coordinates
		x = f1 * x + c1;
		y = f2 * y + c2;
		return true;
	}

	/// <summary>
	/// Closed-form unprojection through <see cref="Undistortion"/>; FOV does not use the
	/// iterative undistortion.
	/// </summary>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];

		// Lift points to normalized plane
		double uu = (x - c1) / f1;
		double vv = (y - c2) / f2;

		// Undistortion
		Undistortion(Real.Cast(parameters[4..]), uu, vv, out Real ru, out Real rv);
		u = ru;
		v = rv;
		return true;
	}

	/// <summary>
	/// The FOV distortion. Unlike the other models it returns the distorted point itself,
	/// not an offset: ImgFromCam uses (du, dv) as the image-plane point.
	/// </summary>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T omega = extraParams[0];

		// Chosen arbitrarily.
		T kEpsilon = T.FromDouble(1e-4);

		T radius2 = u * u + v * v;
		T omega2 = omega * omega;

		T factor;
		if (omega2 < kEpsilon)
		{
			// Derivation of this case with Matlab:
			// syms radius omega;
			// factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
			//                  (radius * omega);
			// simplify(taylor(factor, omega, 'order', 3))
			factor = (omega2 * radius2) / T.FromDouble(3) - omega2 / T.FromDouble(12) + T.FromDouble(1);
		}
		else if (radius2 < kEpsilon)
		{
			// Derivation of this case with Matlab:
			// syms radius omega;
			// factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
			//                  (radius * omega);
			// simplify(taylor(factor, radius, 'order', 3))
			T tanHalfOmega = T.Tan(omega / T.FromDouble(2));
			factor = (T.FromDouble(-2) * tanHalfOmega * (T.FromDouble(4) * radius2 * tanHalfOmega * tanHalfOmega - T.FromDouble(3)))
				/ (T.FromDouble(3) * omega);
		}
		else
		{
			T radius = T.Sqrt(radius2);
			T numerator = T.Atan(radius * T.FromDouble(2) * T.Tan(omega / T.FromDouble(2)));
			factor = numerator / (radius * omega);
		}

		du = u * factor;
		dv = v * factor;
	}

	/// <summary>The closed-form inverse of <see cref="Distortion"/>.</summary>
	public static void Undistortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T omega = extraParams[0];

		// Chosen arbitrarily.
		T kEpsilon = T.FromDouble(1e-4);

		T radius2 = u * u + v * v;
		T omega2 = omega * omega;

		T factor;
		if (omega2 < kEpsilon)
		{
			// Derivation of this case with Matlab:
			// syms radius omega;
			// factor(radius) = tan(radius * omega) / ...
			//                  (radius * 2*tan(omega/2));
			// simplify(taylor(factor, omega, 'order', 3))
			factor = (omega2 * radius2) / T.FromDouble(3) - omega2 / T.FromDouble(12) + T.FromDouble(1);
		}
		else if (radius2 < kEpsilon)
		{
			// Derivation of this case with Matlab:
			// syms radius omega;
			// factor(radius) = tan(radius * omega) / ...
			//                  (radius * 2*tan(omega/2));
			// simplify(taylor(factor, radius, 'order', 3))
			factor = (omega * (omega * omega * radius2 + T.FromDouble(3))) / (T.FromDouble(6) * T.Tan(omega / T.FromDouble(2)));
		}
		else
		{
			T radius = T.Sqrt(radius2);
			T numerator = T.Tan(radius * omega);
			factor = numerator / (radius * T.FromDouble(2) * T.Tan(omega / T.FromDouble(2)));
		}

		du = u * factor;
		dv = v * factor;
	}
}
