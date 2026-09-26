// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The camera models of colmap/sensor/models.h with closed-form projection both ways:
// SIMPLE_DIVISION and DIVISION (Fitzgibbon's division model), EUCM (the enhanced unified
// model), and the spherical EQUIRECTANGULAR panorama model. Siblings:
// PinholeCameraModels.cs, FisheyeCameraModels.cs; base and design notes in
// CameraModelBase.cs. Every expression keeps COLMAP's operator order (Tier A).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

/// <summary>
/// Simple Division camera model: Fitzgibbon's one-parameter division model with a single
/// focal length, closed form both ways. Parameters: f, cx, cy, k.
/// See "Simultaneous linear estimation of multiple view geometry and lens distortion" by
/// A. Fitzgibbon 2001.
/// </summary>
public readonly partial struct SimpleDivisionCameraModel : IPerspectivePinholeCameraModel<SimpleDivisionCameraModel>, IDistortedCameraModel<SimpleDivisionCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.SimpleDivision;

	/// <inheritdoc/>
	public static string ModelName => "SIMPLE_DIVISION";

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

	/// <summary>
	/// Division model projection: (xp, 1 + k*|xp|^2) ~= (x(1:2), x3), solving the quadratic
	/// rho*k*r2 - x3*r + rho = 0. Ignores <paramref name="checkCheirality"/>, as COLMAP does.
	/// </summary>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		T f = parameters[0];
		T c1 = parameters[1];
		T c2 = parameters[2];
		T k = parameters[3];

		T rho = T.Sqrt(u * u + v * v);
		T discSq = w * w - T.FromDouble(4) * rho * rho * k;
		if (discSq < T.FromDouble(0))
		{
			return false;
		}

		T disc = T.Sqrt(discSq);
		T r = T.FromDouble(2) / (w + disc);
		x = f * r * u + c1;
		y = f * r * v + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];
		double k = parameters[3];

		// Lift to normalized coordinates
		double x0 = (x - c1) / f;
		double y0 = (y - c2) / f;
		double r2 = x0 * x0 + y0 * y0;

		// Closed-form unprojection for division model
		double denom = 1.0 + k * r2;
		u = x0 / denom;
		v = y0 / denom;
		return true;
	}

	/// <summary>
	/// The division model doesn't use standard additive distortion, but COLMAP defines this
	/// for compatibility with the iterative undistortion.
	/// </summary>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k = extraParams[0];
		T r2 = u * u + v * v;
		T factor = k * r2 / (T.FromDouble(1) + k * r2);
		du = -u * factor;
		dv = -v * factor;
	}
}

/// <summary>
/// Division camera model: Fitzgibbon's one-parameter division model with separate fx/fy
/// focal lengths, closed form both ways. Parameters: fx, fy, cx, cy, k.
/// See "Simultaneous linear estimation of multiple view geometry and lens distortion" by
/// A. Fitzgibbon 2001.
/// </summary>
public readonly partial struct DivisionCameraModel : IPerspectivePinholeCameraModel<DivisionCameraModel>, IDistortedCameraModel<DivisionCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.Division;

	/// <inheritdoc/>
	public static string ModelName => "DIVISION";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k";

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
		[focalLength, focalLength, width / 2.0, height / 2.0, 0];

	/// <summary>
	/// Division model projection, as in <see cref="SimpleDivisionCameraModel"/>. Ignores
	/// <paramref name="checkCheirality"/>, as COLMAP does.
	/// </summary>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		T f1 = parameters[0];
		T f2 = parameters[1];
		T c1 = parameters[2];
		T c2 = parameters[3];
		T k = parameters[4];

		T rho = T.Sqrt(u * u + v * v);
		T discSq = w * w - T.FromDouble(4) * rho * rho * k;
		if (discSq < T.FromDouble(0))
		{
			return false;
		}

		T disc = T.Sqrt(discSq);
		T r = T.FromDouble(2) / (w + disc);
		x = f1 * r * u + c1;
		y = f2 * r * v + c2;
		return true;
	}

	/// <inheritdoc/>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];
		double k = parameters[4];

		// Lift to normalized coordinates
		double x0 = (x - c1) / f1;
		double y0 = (y - c2) / f2;
		double r2 = x0 * x0 + y0 * y0;

		// Closed-form unprojection for division model
		double denom = 1.0 + k * r2;
		u = x0 / denom;
		v = y0 / denom;
		return true;
	}

	/// <summary>
	/// The division model doesn't use standard additive distortion, but COLMAP defines this
	/// for compatibility with the iterative undistortion.
	/// </summary>
	public static void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>
	{
		T k = extraParams[0];
		T r2 = u * u + v * v;
		T factor = k * r2 / (T.FromDouble(1) + k * r2);
		du = -u * factor;
		dv = -v * factor;
	}
}

/// <summary>
/// EUCM camera model, described in "An Enhanced Unified Camera Model", Bogdan Khomutenko,
/// Gaetan Garcia, Philippe Martinet, 2018. Parameters: fx, fy, cx, cy, alpha, beta.
/// </summary>
public readonly partial struct EUCMCameraModel : IPerspectivePinholeCameraModel<EUCMCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.EUCM;

	/// <inheritdoc/>
	public static string ModelName => "EUCM";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, alpha, beta";

	/// <inheritdoc/>
	public static int NumParams => 6;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [4, 5];

	/// <summary>
	/// The base magnitude check, plus EUCM's own validity range: alpha in [0, 1], beta &gt; 0.
	/// </summary>
	public static bool HasBogusExtraParams(ReadOnlySpan<double> parameters, double maxExtraParam)
	{
		if (CameraModelMath.HasBogusExtraParams<EUCMCameraModel>(parameters, maxExtraParam))
		{
			return true;
		}

		double alpha = parameters[4];
		double beta = parameters[5];
		return alpha < 0 || alpha > 1 || beta <= 0;
	}

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0, 0.0, 1.0];

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
		T alpha = parameters[4];
		T beta = parameters[5];

		T rho2 = beta * (u * u + v * v) + w * w;
		if (rho2 < T.FromDouble(0))
		{
			return false;
		}

		T rho = T.Sqrt(rho2);
		T den = alpha * rho + (1.0 - alpha) * w;
		if (!CameraModelMath.HasProjectableDepth(den, checkCheirality))
		{
			return false;
		}

		x = u / den;
		y = v / den;

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
		double alpha = parameters[4];
		double beta = parameters[5];

		// Lift points to normalized plane
		u = (x - c1) / f1;
		v = (y - c2) / f2;

		double r2 = u * u + v * v;
		double gamma = 1.0 - alpha;
		double radicand = 1.0 - (alpha - gamma) * beta * r2;
		if (radicand < 0)
		{
			return false;
		}

		double helperDen = alpha * Math.Sqrt(radicand) + gamma;
		if (helperDen < LinearAlgebraConstants.MachineEpsilon)
		{
			return false;
		}

		double helper = (1.0 - alpha * alpha * beta * r2) / helperDen;
		if (helper < LinearAlgebraConstants.MachineEpsilon)
		{
			return false;
		}

		u /= helper;
		v /= helper;
		return true;
	}
}

/// <summary>
/// Equirectangular (spherical panorama) camera model. Maps the full 360x180 degree sphere
/// onto an image: the azimuth spans the width and the elevation spans the height. The model
/// is fully specified by the image dimensions: no focal length, principal point or lens
/// distortion. Parameters: w, h.
/// </summary>
public readonly partial struct EquirectangularCameraModel : ISphericalCameraModel<EquirectangularCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.Equirectangular;

	/// <inheritdoc/>
	public static string ModelName => "EQUIRECTANGULAR";

	/// <inheritdoc/>
	public static string ParamsInfo => "w,h";

	/// <inheritdoc/>
	public static int NumParams => 2;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> MetaDataParamsIdxs => [0, 1];

	/// <summary>Ignores the focal length and returns (width, height).</summary>
	public static double[] InitializeParams(double focalLength, int width, int height) => [width, height];

	/// <summary>Never bogus: the parameters are the image size.</summary>
	public static bool HasBogusParams(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam) => false;

	/// <summary>
	/// EQUIRECTANGULAR has no focal length, so it converts pixel thresholds with the angular
	/// resolution at the equator (2 pi rad per W pixels in azimuth).
	/// </summary>
	public static double CamFromImgThreshold(ReadOnlySpan<double> parameters, double threshold) =>
		threshold * (2.0 * Math.PI) / parameters[0];

	/// <summary>
	/// Unlike pinhole/fisheye models that require w &gt; 0, EQUIRECTANGULAR accepts any
	/// non-zero direction: all 4 pi of the sphere are representable. Ignores
	/// <paramref name="checkCheirality"/>.
	/// </summary>
	public static bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>
	{
		x = default;
		y = default;
		T width = parameters[0];
		T height = parameters[1];
		T horizontal = T.Sqrt(u * u + w * w);

		// Degenerate: zero direction vector.
		if (horizontal + T.Abs(v) < T.FromDouble(LinearAlgebraConstants.MachineEpsilon))
		{
			return false;
		}

		// Azimuth theta in (-pi, pi], measured from +Z axis (forward). +X is theta = +pi/2.
		T theta = T.Atan2(u, w);

		// Elevation phi in [-pi/2, pi/2], measured from the equator. -Y (up) is +pi/2.
		T phi = T.Atan2(-v, horizontal);
		x = (theta / T.FromDouble(2.0 * Math.PI) + T.FromDouble(0.5)) * width;
		y = (T.FromDouble(0.5) - phi / T.FromDouble(Math.PI)) * height;
		return true;
	}

	/// <summary>
	/// Inverse equirectangular projection to normalized coordinates (u = X/Z, v = Y/Z),
	/// valid only when the ray is in the forward hemisphere (Z &gt; 0). Back-hemisphere
	/// pixels return false; use CamRayFromImg for the full-sphere bearing.
	/// </summary>
	public static bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v)
	{
		u = 0;
		v = 0;
		double width = parameters[0];
		double height = parameters[1];
		double theta = 2.0 * Math.PI * (x / width - 0.5);
		double phi = Math.PI * (0.5 - y / height);
		double cosPhi = Math.Cos(phi);
		double rx = cosPhi * Math.Sin(theta);
		double ry = -Math.Sin(phi);
		double rz = cosPhi * Math.Cos(theta);
		if (rz <= LinearAlgebraConstants.MachineEpsilon)
		{
			return false;
		}

		u = rx / rz;
		v = ry / rz;
		return true;
	}

	/// <summary>
	/// The perspective default goes through the 2D CamFromImg, which fails for
	/// back-hemisphere pixels. EQUIRECTANGULAR produces a valid unit bearing for any pixel,
	/// so this computes the ray directly from the azimuth/elevation parametrization.
	/// </summary>
	public static bool CamRayFromImg(ReadOnlySpan<double> parameters, double x, double y, out double rx, out double ry, out double rz)
	{
		double width = parameters[0];
		double height = parameters[1];
		double theta = 2.0 * Math.PI * (x / width - 0.5);
		double phi = Math.PI * (0.5 - y / height);
		double cosPhi = Math.Cos(phi);
		rx = cosPhi * Math.Sin(theta);
		ry = -Math.Sin(phi);
		rz = cosPhi * Math.Cos(theta);
		return true;
	}
}
