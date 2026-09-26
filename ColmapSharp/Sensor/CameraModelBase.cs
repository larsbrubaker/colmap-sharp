// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The camera model hierarchy of colmap/sensor/models.h: BaseCameraModel,
// BasePerspectiveCameraModel, BasePerspectivePinholeCameraModel,
// BasePerspectiveFisheyeCameraModel and BaseSphericalCameraModel, plus HasProjectableDepth.
// The models are in PinholeCameraModels.cs, FisheyeCameraModels.cs and
// OtherCameraModels.cs; CameraModels.cs is the runtime dispatch by CameraModelId.
// Tests: ColmapSharp.Tests/Sensor/ModelsTests.cs (models_test.cc 1:1) and
// CameraModelOracleTests.cs.
//
// Design (a decision later phases build on):
// - COLMAP's CRTP (`struct M : BasePerspectiveCameraModel<M>`) becomes a static-abstract
//   interface hierarchy: every model is an empty `readonly struct M : I...CameraModel<M>`,
//   and the shared base behavior lives in `static virtual` interface members that a model
//   can re-implement (EUCM's HasBogusExtraParams does). Generic code takes
//   `TModel : struct, ICameraModel<TModel>` and calls `TModel.ImgFromCam(...)`, which is
//   the C# spelling of `CameraModel::ImgFromCam` in a template, so COLMAP's templated tests
//   and, later, templated cost functions port one to one.
// - The projection math (`template <typename T> ImgFromCam`, Distortion, the fisheye
//   helpers) is generic over `T : struct, IScalar<T>` (Solver/Scalar.cs): Real for double
//   evaluation, Jet2 now for IterativeUndistortion, and Phase 7's Jet<N> for bundle
//   adjustment. Functions COLMAP only defines for double (CamFromImg, thresholds, bogus
//   checks) take double.
// - Parameter arrays are spans: `const T* params` is ReadOnlySpan<T>, `&params[4]` is
//   params[4..]. Width and height (size_t in C++) are int.
// - Distortion is on its own interface, IDistortedCameraModel, because COLMAP declares it
//   for every perspective model but only defines it for the ones that have one; C# needs a
//   body for every abstract member, and "no stubs" rules out empty ones.
//
// Tier A: every operation keeps COLMAP's evaluation order, so projection and unprojection,
// including the Newton iteration, are bit-identical to COLMAP for the same input, up to the
// platform libm and FMA contraction in the C++ build (docs/CPP_DIVERGENCES.md, entry 12;
// CameraModelOracleTests pins which results are bit-identical to the pycolmap wheel).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::BaseCameraModel together with the members the CAMERA_MODEL_* macros
/// declare on every model. Implemented by empty structs, one per camera model.
/// </summary>
public interface ICameraModel<TSelf>
	where TSelf : struct, ICameraModel<TSelf>
{
	/// <summary>C++ <c>model_id</c>.</summary>
	static abstract CameraModelId ModelId { get; }

	/// <summary>C++ <c>model_name</c>, e.g. "SIMPLE_RADIAL".</summary>
	static abstract string ModelName { get; }

	/// <summary>C++ <c>params_info</c>: the parameter order, e.g. "f, cx, cy, k".</summary>
	static abstract string ParamsInfo { get; }

	/// <summary>C++ <c>num_params</c>.</summary>
	static abstract int NumParams { get; }

	/// <summary>C++ <c>focal_length_idxs</c>; empty for spherical models.</summary>
	static abstract ReadOnlySpan<int> FocalLengthIdxs { get; }

	/// <summary>C++ <c>principal_point_idxs</c>; empty for spherical models.</summary>
	static abstract ReadOnlySpan<int> PrincipalPointIdxs { get; }

	/// <summary>C++ <c>extra_params_idxs</c>; empty for spherical models.</summary>
	static abstract ReadOnlySpan<int> ExtraParamsIdxs { get; }

	/// <summary>C++ <c>metadata_idxs</c>; empty for perspective models.</summary>
	static abstract ReadOnlySpan<int> MetaDataParamsIdxs { get; }

	/// <summary>Derives from BasePerspectiveCameraModel.</summary>
	static abstract bool IsPerspective { get; }

	/// <summary>Derives from BasePerspectivePinholeCameraModel.</summary>
	static abstract bool IsPerspectivePinhole { get; }

	/// <summary>Derives from BasePerspectiveFisheyeCameraModel.</summary>
	static abstract bool IsPerspectiveFisheye { get; }

	/// <summary>Derives from BaseSphericalCameraModel.</summary>
	static abstract bool IsSpherical { get; }

	/// <summary>C++ <c>InitializeParams</c>: default parameters for an image size.</summary>
	static abstract double[] InitializeParams(double focalLength, int width, int height);

	/// <summary>
	/// Projects camera coordinates (u, v, w) to pixel coordinates. Returns false if the
	/// projection fails; <paramref name="checkCheirality"/> selects whether points behind
	/// the camera are rejected (see <see cref="CameraModelMath.HasProjectableDepth"/>).
	/// </summary>
	static abstract bool ImgFromCam<T>(ReadOnlySpan<T> parameters, T u, T v, T w, out T x, out T y, bool checkCheirality = true)
		where T : struct, IScalar<T>;

	/// <summary>
	/// <see cref="ImgFromCam"/> in double with analytic Jacobians (C++
	/// <c>ImgFromCamWithJac</c>, colmap/sensor/models_jacobian.h). <paramref name="jParams"/>
	/// receives the 2 x NumParams Jacobian d(x, y) / d(params) and <paramref name="jUvw"/>
	/// the 2x3 Jacobian d(x, y) / d(u, v, w), both row-major; an empty span (C++ nullptr)
	/// skips that Jacobian. Every COLMAP model provides it (has_img_from_cam_with_jac).
	/// The kernels are in the *CameraModels.Jacobian.cs files.
	/// </summary>
	static abstract bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true);

	/// <summary>
	/// Lifts pixel coordinates to normalized camera coordinates (u, v, 1). Returns false if
	/// lifting fails. On failure the outputs are unspecified: C++ leaves them untouched for
	/// the fisheye models and EQUIRECTANGULAR and partly written for EUCM, while this port
	/// writes 0 (fisheye, EQUIRECTANGULAR) or the partial values (EUCM). Callers must not
	/// read them after a false return.
	/// </summary>
	static abstract bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v);

	/// <summary>Unprojects a pixel to a unit bearing vector in the camera frame.</summary>
	static abstract bool CamRayFromImg(ReadOnlySpan<double> parameters, double x, double y, out double rx, out double ry, out double rz);

	/// <summary>Converts a pixel threshold to normalized camera units.</summary>
	static abstract double CamFromImgThreshold(ReadOnlySpan<double> parameters, double threshold);

	/// <summary>Whether the parameters are implausible for the image size.</summary>
	static abstract bool HasBogusParams(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam);

	/// <summary>Rescales the parameters in place for a new image resolution.</summary>
	static abstract void Rescale(double scaleX, double scaleY, Span<double> parameters);
}

/// <summary>
/// A model with an additive distortion of the normalized (or fisheye) plane,
/// (u, v) -> (u + du, v + dv), which CamFromImg inverts by Newton iteration. Port of the
/// <c>Distortion</c> member of the perspective models that define it.
/// </summary>
public interface IDistortedCameraModel<TSelf> : IPerspectiveCameraModel<TSelf>
	where TSelf : struct, IDistortedCameraModel<TSelf>
{
	/// <summary>The distortion (du, dv) at (u, v), given the extra parameters only.</summary>
	static abstract void Distortion<T>(ReadOnlySpan<T> extraParams, T u, T v, out T du, out T dv)
		where T : struct, IScalar<T>;
}

/// <summary>
/// Port of colmap::BasePerspectiveCameraModel: models with a focal length and a finite image
/// plane. Provides the shared bogus-parameter checks, the focal-length-based threshold,
/// the forward-hemisphere ray unprojection and rescaling.
/// </summary>
public interface IPerspectiveCameraModel<TSelf> : ICameraModel<TSelf>
	where TSelf : struct, IPerspectiveCameraModel<TSelf>
{
	static ReadOnlySpan<int> ICameraModel<TSelf>.MetaDataParamsIdxs => [];

	static bool ICameraModel<TSelf>.IsPerspective => true;

	static bool ICameraModel<TSelf>.IsSpherical => false;

	static bool ICameraModel<TSelf>.HasBogusParams(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam) =>
		TSelf.HasBogusPrincipalPoint(parameters, width, height)
		|| TSelf.HasBogusFocalLength(parameters, width, height, minFocalLengthRatio, maxFocalLengthRatio)
		|| TSelf.HasBogusExtraParams(parameters, maxExtraParam);

	/// <summary>Whether a focal length over the larger image side is outside the ratio range.</summary>
	static virtual bool HasBogusFocalLength(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio)
	{
		double invMaxSize = 1.0 / Math.Max(width, height);
		foreach (int idx in TSelf.FocalLengthIdxs)
		{
			double focalLengthRatio = parameters[idx] * invMaxSize;
			if (focalLengthRatio < minFocalLengthRatio || focalLengthRatio > maxFocalLengthRatio)
			{
				return true;
			}
		}

		return false;
	}

	/// <summary>Whether the principal point lies outside [0, width] x [0, height].</summary>
	static virtual bool HasBogusPrincipalPoint(ReadOnlySpan<double> parameters, int width, int height)
	{
		double cx = parameters[TSelf.PrincipalPointIdxs[0]];
		double cy = parameters[TSelf.PrincipalPointIdxs[1]];
		return cx < 0 || cx > width || cy < 0 || cy > height;
	}

	/// <summary>Whether any extra parameter's magnitude exceeds the maximum.</summary>
	static virtual bool HasBogusExtraParams(ReadOnlySpan<double> parameters, double maxExtraParam) =>
		CameraModelMath.HasBogusExtraParams<TSelf>(parameters, maxExtraParam);

	static double ICameraModel<TSelf>.CamFromImgThreshold(ReadOnlySpan<double> parameters, double threshold)
	{
		double meanFocalLength = 0;
		foreach (int idx in TSelf.FocalLengthIdxs)
		{
			meanFocalLength += parameters[idx];
		}

		meanFocalLength /= TSelf.FocalLengthIdxs.Length;
		return threshold / meanFocalLength;
	}

	// Default: CamFromImg normalized. Correct for perspective and fisheye-with-FOV <= 180
	// degree cameras; the returned ray always has rz > 0.
	static bool ICameraModel<TSelf>.CamRayFromImg(ReadOnlySpan<double> parameters, double x, double y, out double rx, out double ry, out double rz)
	{
		rx = 0;
		ry = 0;
		rz = 0;
		if (!TSelf.CamFromImg(parameters, x, y, out double u, out double v))
		{
			return false;
		}

		double norm = Math.Sqrt(u * u + v * v + 1.0);
		rx = u / norm;
		ry = v / norm;
		rz = 1.0 / norm;
		return true;
	}

	// A single shared focal length scales by the mean factor; separate fx/fy scale
	// independently. The principal point follows the image dimensions. Extra (distortion)
	// parameters are resolution independent and left untouched.
	static void ICameraModel<TSelf>.Rescale(double scaleX, double scaleY, Span<double> parameters)
	{
		ReadOnlySpan<int> focal = TSelf.FocalLengthIdxs;
		if (focal.Length == 1)
		{
			parameters[focal[0]] *= 0.5 * (scaleX + scaleY);
		}
		else
		{
			parameters[focal[0]] *= scaleX;
			parameters[focal[1]] *= scaleY;
		}

		parameters[TSelf.PrincipalPointIdxs[0]] *= scaleX;
		parameters[TSelf.PrincipalPointIdxs[1]] *= scaleY;
	}
}

/// <summary>
/// Port of colmap::BasePerspectivePinholeCameraModel: models that project as x = X / Z and
/// then deform the normalized plane, so a calibration matrix K is meaningful for them.
/// </summary>
public interface IPerspectivePinholeCameraModel<TSelf> : IPerspectiveCameraModel<TSelf>
	where TSelf : struct, IPerspectivePinholeCameraModel<TSelf>
{
	static bool ICameraModel<TSelf>.IsPerspectivePinhole => true;

	static bool ICameraModel<TSelf>.IsPerspectiveFisheye => false;
}

/// <summary>
/// Port of colmap::BasePerspectiveFisheyeCameraModel: models that project through the
/// equidistant fisheye plane (uu, vv) = theta * (u, v) / r before distortion.
/// </summary>
public interface IPerspectiveFisheyeCameraModel<TSelf> : IPerspectiveCameraModel<TSelf>
	where TSelf : struct, IPerspectiveFisheyeCameraModel<TSelf>
{
	static bool ICameraModel<TSelf>.IsPerspectivePinhole => false;

	static bool ICameraModel<TSelf>.IsPerspectiveFisheye => true;

	/// <summary>Fisheye-plane coordinates to pixels (focal length and principal point).</summary>
	static abstract void ImgFromFisheye<T>(ReadOnlySpan<T> parameters, T uu, T vv, out T x, out T y)
		where T : struct, IScalar<T>;

	/// <summary>Pixels to fisheye-plane coordinates.</summary>
	static abstract void FisheyeFromImg<T>(ReadOnlySpan<T> parameters, T x, T y, out T uu, out T vv)
		where T : struct, IScalar<T>;
}

/// <summary>
/// Port of colmap::BaseSphericalCameraModel: omnidirectional models whose only parameters
/// are metadata (the image size).
/// </summary>
public interface ISphericalCameraModel<TSelf> : ICameraModel<TSelf>
	where TSelf : struct, ISphericalCameraModel<TSelf>
{
	static ReadOnlySpan<int> ICameraModel<TSelf>.FocalLengthIdxs => [];

	static ReadOnlySpan<int> ICameraModel<TSelf>.PrincipalPointIdxs => [];

	static ReadOnlySpan<int> ICameraModel<TSelf>.ExtraParamsIdxs => [];

	static bool ICameraModel<TSelf>.IsPerspective => false;

	static bool ICameraModel<TSelf>.IsPerspectivePinhole => false;

	static bool ICameraModel<TSelf>.IsPerspectiveFisheye => false;

	static bool ICameraModel<TSelf>.IsSpherical => true;

	// Only the image dimensions, carried by the metadata group, track the rescaled image.
	static void ICameraModel<TSelf>.Rescale(double scaleX, double scaleY, Span<double> parameters)
	{
		parameters[TSelf.MetaDataParamsIdxs[0]] *= scaleX;
		parameters[TSelf.MetaDataParamsIdxs[1]] *= scaleY;
	}
}

/// <summary>
/// The shared numeric helpers of the camera model bases in colmap/sensor/models.h.
/// </summary>
public static partial class CameraModelMath
{
	/// <summary>
	/// Port of colmap::HasProjectableDepth. Rejects points at or behind the camera plane if
	/// <paramref name="checkCheirality"/>, otherwise only points on the plane, where the
	/// projection diverges.
	/// </summary>
	public static bool HasProjectableDepth<T>(T w, bool checkCheirality)
		where T : struct, IScalar<T>
	{
		T epsilon = T.FromDouble(LinearAlgebraConstants.MachineEpsilon);
		return checkCheirality ? w >= epsilon : T.Abs(w) >= epsilon;
	}

	/// <summary>
	/// The base implementation of HasBogusExtraParams, callable from a model that extends it
	/// (C++ <c>BasePerspectiveCameraModel&lt;M&gt;::HasBogusExtraParams</c>).
	/// </summary>
	public static bool HasBogusExtraParams<TModel>(ReadOnlySpan<double> parameters, double maxExtraParam)
		where TModel : struct, ICameraModel<TModel>
	{
		foreach (int idx in TModel.ExtraParamsIdxs)
		{
			if (Math.Abs(parameters[idx]) > maxExtraParam)
			{
				return true;
			}
		}

		return false;
	}

	/// <summary>
	/// BasePerspectiveFisheyeCameraModel::FisheyeFromNormal: normalized plane to the
	/// equidistant fisheye plane, scaling (u, v) by atan(r) / r.
	/// </summary>
	public static void FisheyeFromNormal<T>(T u, T v, out T uu, out T vv)
		where T : struct, IScalar<T>
	{
		uu = u;
		vv = v;
		T r = T.Sqrt(u * u + v * v);
		if (r > T.FromDouble(LinearAlgebraConstants.MachineEpsilon))
		{
			T theta = T.Atan(r);
			uu *= theta / r;
			vv *= theta / r;
		}
	}

	/// <summary>
	/// BasePerspectiveFisheyeCameraModel::NormalFromFisheye: the inverse of
	/// <see cref="FisheyeFromNormal"/>, scaling by tan(theta) / theta as sin / (theta cos).
	/// </summary>
	public static void NormalFromFisheye<T>(T uu, T vv, out T u, out T v)
		where T : struct, IScalar<T>
	{
		u = uu;
		v = vv;
		T theta = T.Sqrt(uu * uu + vv * vv);
		T thetaCosTheta = theta * T.Cos(theta);
		if (thetaCosTheta > T.FromDouble(LinearAlgebraConstants.MachineEpsilon))
		{
			T scale = T.Sin(theta) / thetaCosTheta;
			u *= scale;
			v *= scale;
		}
	}

	/// <summary>
	/// The double instantiation of <c>CameraModel::ImgFromCam</c>, for callers holding plain
	/// doubles (COLMAP's tests call it with double*).
	/// </summary>
	public static bool ImgFromCam<TModel>(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, bool checkCheirality = true)
		where TModel : struct, ICameraModel<TModel>
	{
		bool ok = TModel.ImgFromCam(Real.Cast(parameters), u, v, w, out Real rx, out Real ry, checkCheirality);
		x = rx;
		y = ry;
		return ok;
	}
}
