// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The thin-prism perspective fisheye camera models of colmap/sensor/models.h:
// THIN_PRISM_FISHEYE and RAD_TAN_THIN_PRISM_FISHEYE, whose distortion adds tangential and
// thin-prism terms to the radial one. Split out of FisheyeCameraModels.cs, which holds the
// other fisheye models and describes the shared projection; the analytic Jacobians of all
// seven are in FisheyeCameraModels.Jacobian.cs. Every expression keeps COLMAP's operator
// order (Tier A).

using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

/// <summary>
/// Fisheye camera model with radial and tangential distortion coefficients and additional
/// coefficients accounting for thin-prism distortion, described in "Camera Calibration
/// with Distortion Models and Accuracy Evaluation", J Weng et al., TPAMI, 1992.
/// Parameters: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.
/// </summary>
public readonly partial struct ThinPrismFisheyeCameraModel : IPerspectiveFisheyeCameraModel<ThinPrismFisheyeCameraModel>, IDistortedCameraModel<ThinPrismFisheyeCameraModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.ThinPrismFisheye;

	/// <inheritdoc/>
	public static string ModelName => "THIN_PRISM_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";

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
		if (!CameraModelMath.IterativeUndistortion<ThinPrismFisheyeCameraModel>(parameters[4..], ref uu, ref vv))
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
		T p1 = extraParams[2];
		T p2 = extraParams[3];
		T k3 = extraParams[4];
		T k4 = extraParams[5];
		T sx1 = extraParams[6];
		T sy1 = extraParams[7];
		T two = T.FromDouble(2);

		T u2 = u * u;
		T uv = u * v;
		T v2 = v * v;
		T r2 = u2 + v2;
		T r4 = r2 * r2;
		T r6 = r4 * r2;
		T r8 = r6 * r2;

		T radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
		du = u * radial + two * p1 * uv + p2 * (r2 + two * u2) + sx1 * r2;
		dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2) + sy1 * r2;
	}
}

/// <summary>
/// RadTanThinPrismFisheye camera model (Project Aria's Fisheye624): radial and
/// tangential distortion coefficients and additional coefficients accounting for
/// thin-prism distortion.
/// Parameters: fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3.
/// See https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models#the-fisheyeradtanthinprism-fisheye624-model
/// </summary>
public readonly partial struct RadTanThinPrismFisheyeModel : IPerspectiveFisheyeCameraModel<RadTanThinPrismFisheyeModel>, IDistortedCameraModel<RadTanThinPrismFisheyeModel>
{
	/// <inheritdoc/>
	public static CameraModelId ModelId => CameraModelId.RadTanThinPrismFisheye;

	/// <inheritdoc/>
	public static string ModelName => "RAD_TAN_THIN_PRISM_FISHEYE";

	/// <inheritdoc/>
	public static string ParamsInfo => "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3";

	/// <inheritdoc/>
	public static int NumParams => 16;

	/// <inheritdoc/>
	public static ReadOnlySpan<int> FocalLengthIdxs => [0, 1];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> PrincipalPointIdxs => [2, 3];

	/// <inheritdoc/>
	public static ReadOnlySpan<int> ExtraParamsIdxs => [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

	/// <inheritdoc/>
	public static double[] InitializeParams(double focalLength, int width, int height) =>
		[focalLength, focalLength, width / 2.0, height / 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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
		if (!CameraModelMath.IterativeUndistortion<RadTanThinPrismFisheyeModel>(parameters[4..], ref uu, ref vv))
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
		const int kNumRadialParams = 6;
		T p0 = extraParams[6];
		T p1 = extraParams[7];
		T s0 = extraParams[8];
		T s1 = extraParams[9];
		T s2 = extraParams[10];
		T s3 = extraParams[11];
		T two = T.FromDouble(2);

		T theta2 = u * u + v * v;
		T thRadial = T.FromDouble(1);
		T thetaPower = T.FromDouble(1);
		for (int i = 0; i < kNumRadialParams; i++)
		{
			thetaPower *= theta2;
			thRadial += extraParams[i] * thetaPower;
		}

		T x = thRadial * u;
		T y = thRadial * v;

		T x2 = x * x;
		T y2 = y * y;
		T xy = x * y;
		T r2 = x2 + y2;
		T r4 = r2 * r2;

		T dxTang = two * p1 * xy + p0 * (r2 + two * x2);
		T dyTang = two * p0 * xy + p1 * (r2 + two * y2);

		T dxTp = s0 * r2 + s1 * r4;
		T dyTp = s2 * r2 + s3 * r4;

		T xDistorted = x + dxTang + dxTp;
		T yDistorted = y + dyTang + dyTp;

		du = xDistorted - u;
		dv = yDistorted - v;
	}
}
