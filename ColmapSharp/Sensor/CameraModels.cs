// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraModels: the runtime-dispatched free functions of colmap/sensor/models.h and
// models.cc (CameraModelImgFromCam, CameraModelCamFromImg, CameraModelNameToId, ...), which
// COLMAP writes as a switch over CAMERA_MODEL_CASES. Here the switch is a table of
// CameraModelOps<TModel>, one per model, built once; each entry forwards to the static
// members of its model struct (PinholeCameraModels.cs, FisheyeCameraModels.cs,
// OtherCameraModels.cs), so the dispatched result is the same computation as calling the
// model directly. Unknown ids throw where COLMAP's switch throws std::domain_error
// ("Camera model does not exist").
//
// Not here yet: CameraModelImgFromCamWithJac and CamRayFromImgJacobian, which need the
// analytic per-model Jacobians of colmap/sensor/models_jacobian.h (a separate port).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of the camera model free functions of colmap/sensor/models.h / models.cc.
/// </summary>
public static class CameraModels
{
	private static readonly CameraModelOps[] Ops =
	[
		new CameraModelOps<SimplePinholeCameraModel>(),
		new CameraModelOps<PinholeCameraModel>(),
		new CameraModelOps<SimpleRadialCameraModel>(),
		new CameraModelOps<RadialCameraModel>(),
		new CameraModelOps<OpenCVCameraModel>(),
		new CameraModelOps<OpenCVFisheyeCameraModel>(),
		new CameraModelOps<FullOpenCVCameraModel>(),
		new CameraModelOps<FOVCameraModel>(),
		new CameraModelOps<SimpleRadialFisheyeCameraModel>(),
		new CameraModelOps<RadialFisheyeCameraModel>(),
		new CameraModelOps<ThinPrismFisheyeCameraModel>(),
		new CameraModelOps<RadTanThinPrismFisheyeModel>(),
		new CameraModelOps<SimpleDivisionCameraModel>(),
		new CameraModelOps<DivisionCameraModel>(),
		new CameraModelOps<SimpleFisheyeCameraModel>(),
		new CameraModelOps<FisheyeCameraModel>(),
		new CameraModelOps<EUCMCameraModel>(),
		new CameraModelOps<EquirectangularCameraModel>(),
	];

	private static readonly Dictionary<string, CameraModelId> NameToId =
		Ops.ToDictionary(op => op.ModelName, op => op.ModelId, StringComparer.Ordinal);

	/// <summary>ExistsCameraModelWithName.</summary>
	public static bool ExistsCameraModelWithName(string modelName) => NameToId.ContainsKey(modelName);

	/// <summary>ExistsCameraModelWithId.</summary>
	public static bool ExistsCameraModelWithId(CameraModelId modelId) => Find(modelId) is not null;

	/// <summary>CameraModelNameToId: the id, or <see cref="CameraModelId.Invalid"/> if unknown.</summary>
	public static CameraModelId CameraModelNameToId(string modelName) =>
		NameToId.TryGetValue(modelName, out CameraModelId id) ? id : CameraModelId.Invalid;

	/// <summary>CameraModelIdToName: the name, or "" if unknown.</summary>
	public static string CameraModelIdToName(CameraModelId modelId) => Find(modelId)?.ModelName ?? "";

	/// <summary>
	/// CameraModelInitializeParams: all focal lengths set to <paramref name="focalLength"/>
	/// and the principal point at the image center. Assumes image measurements are within
	/// [0, dim], i.e. that the upper left corner is the (0, 0) coordinate (rather than the
	/// center of the upper left pixel).
	/// </summary>
	public static double[] CameraModelInitializeParams(CameraModelId modelId, double focalLength, int width, int height) =>
		Get(modelId).InitializeParams(focalLength, width, height);

	/// <summary>CameraModelParamsInfo: the human-readable parameter order.</summary>
	public static string CameraModelParamsInfo(CameraModelId modelId) => Get(modelId).ParamsInfo;

	/// <summary>CameraModelFocalLengthIdxs (empty for spherical models).</summary>
	public static ReadOnlySpan<int> CameraModelFocalLengthIdxs(CameraModelId modelId) => Get(modelId).FocalLengthIdxs;

	/// <summary>CameraModelPrincipalPointIdxs (empty for spherical models).</summary>
	public static ReadOnlySpan<int> CameraModelPrincipalPointIdxs(CameraModelId modelId) => Get(modelId).PrincipalPointIdxs;

	/// <summary>CameraModelExtraParamsIdxs (empty for spherical models).</summary>
	public static ReadOnlySpan<int> CameraModelExtraParamsIdxs(CameraModelId modelId) => Get(modelId).ExtraParamsIdxs;

	/// <summary>CameraModelMetaDataParamsIdxs (empty for perspective models).</summary>
	public static ReadOnlySpan<int> CameraModelMetaDataParamsIdxs(CameraModelId modelId) => Get(modelId).MetaDataParamsIdxs;

	/// <summary>CameraModelNumParams.</summary>
	public static int CameraModelNumParams(CameraModelId modelId) => Get(modelId).NumParams;

	/// <summary>CameraModelVerifyParams: whether the parameter count matches the model.</summary>
	public static bool CameraModelVerifyParams(CameraModelId modelId, ReadOnlySpan<double> parameters) =>
		parameters.Length == Get(modelId).NumParams;

	/// <summary>
	/// CameraModelHasBogusParams: principal point outside the image, a focal length over the
	/// larger image side outside [min, max] ratio, or an extra parameter's magnitude above
	/// <paramref name="maxExtraParam"/>.
	/// </summary>
	public static bool CameraModelHasBogusParams(CameraModelId modelId, ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam) =>
		Get(modelId).HasBogusParams(parameters, width, height, minFocalLengthRatio, maxFocalLengthRatio, maxExtraParam);

	/// <summary>
	/// CameraModelImgFromCam: camera coordinates (u, v, w) to pixels, or null if the
	/// projection fails. The inverse of <see cref="CameraModelCamFromImg"/>.
	/// </summary>
	public static Vector2d? CameraModelImgFromCam(CameraModelId modelId, ReadOnlySpan<double> parameters, Vector3d uvw, bool checkCheirality = true) =>
		Get(modelId).ImgFromCam(parameters, uvw.X, uvw.Y, uvw.Z, out double x, out double y, checkCheirality) ? new Vector2d(x, y) : null;

	/// <summary>
	/// CameraModelCamFromImg: pixels to normalized camera coordinates (u, v), or null if
	/// lifting fails. Limited to the forward hemisphere; see
	/// <see cref="CameraModelCamRayFromImg"/>.
	/// </summary>
	public static Vector2d? CameraModelCamFromImg(CameraModelId modelId, ReadOnlySpan<double> parameters, Vector2d xy) =>
		Get(modelId).CamFromImg(parameters, xy.X, xy.Y, out double u, out double v) ? new Vector2d(u, v) : null;

	/// <summary>
	/// CameraModelCamRayFromImg: a pixel to a unit bearing vector in the camera frame, for
	/// any pixel the model can unproject, including back-facing rays of omnidirectional
	/// cameras. Prefer this to CamFromImg + homogeneous + normalize when a 3D ray is needed.
	/// </summary>
	public static Vector3d? CameraModelCamRayFromImg(CameraModelId modelId, ReadOnlySpan<double> parameters, Vector2d xy) =>
		Get(modelId).CamRayFromImg(parameters, xy.X, xy.Y, out double rx, out double ry, out double rz) ? new Vector3d(rx, ry, rz) : null;

	/// <summary>
	/// CameraModelCamFromImgThreshold: a pixel threshold in normalized camera units (divided
	/// by the mean focal length for perspective models).
	/// </summary>
	public static double CameraModelCamFromImgThreshold(CameraModelId modelId, ReadOnlySpan<double> parameters, double threshold) =>
		Get(modelId).CamFromImgThreshold(parameters, threshold);

	/// <summary>CameraModelIsPerspectiveFisheye: false for unknown ids, as in COLMAP.</summary>
	public static bool CameraModelIsPerspectiveFisheye(CameraModelId modelId) => Find(modelId)?.IsPerspectiveFisheye ?? false;

	/// <summary>CameraModelIsPerspective: has a focal length and a finite image plane.</summary>
	public static bool CameraModelIsPerspective(CameraModelId modelId) => Get(modelId).IsPerspective;

	/// <summary>CameraModelIsPerspectivePinhole: projects as X / Z, then deforms the plane.</summary>
	public static bool CameraModelIsPerspectivePinhole(CameraModelId modelId) => Get(modelId).IsPerspectivePinhole;

	/// <summary>CameraModelIsSpherical.</summary>
	public static bool CameraModelIsSpherical(CameraModelId modelId) => Get(modelId).IsSpherical;

	/// <summary>
	/// CameraModelRescale: rescales the parameters in place for a new image resolution,
	/// given the per-axis scale factors (new_dim / old_dim).
	/// </summary>
	public static void CameraModelRescale(CameraModelId modelId, double scaleX, double scaleY, Span<double> parameters) =>
		Get(modelId).Rescale(scaleX, scaleY, parameters);

	private static CameraModelOps? Find(CameraModelId modelId)
	{
		int index = (int)modelId;
		return index >= 0 && index < Ops.Length ? Ops[index] : null;
	}

	private static CameraModelOps Get(CameraModelId modelId) =>
		Find(modelId) ?? throw new ArgumentException("Camera model does not exist", nameof(modelId));

	/// <summary>One model's static members behind a virtual interface, for id dispatch.</summary>
	private abstract class CameraModelOps
	{
		public abstract CameraModelId ModelId { get; }

		public abstract string ModelName { get; }

		public abstract string ParamsInfo { get; }

		public abstract int NumParams { get; }

		public abstract ReadOnlySpan<int> FocalLengthIdxs { get; }

		public abstract ReadOnlySpan<int> PrincipalPointIdxs { get; }

		public abstract ReadOnlySpan<int> ExtraParamsIdxs { get; }

		public abstract ReadOnlySpan<int> MetaDataParamsIdxs { get; }

		public abstract bool IsPerspective { get; }

		public abstract bool IsPerspectivePinhole { get; }

		public abstract bool IsPerspectiveFisheye { get; }

		public abstract bool IsSpherical { get; }

		public abstract double[] InitializeParams(double focalLength, int width, int height);

		public abstract bool ImgFromCam(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, bool checkCheirality);

		public abstract bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v);

		public abstract bool CamRayFromImg(ReadOnlySpan<double> parameters, double x, double y, out double rx, out double ry, out double rz);

		public abstract double CamFromImgThreshold(ReadOnlySpan<double> parameters, double threshold);

		public abstract bool HasBogusParams(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam);

		public abstract void Rescale(double scaleX, double scaleY, Span<double> parameters);
	}

	private sealed class CameraModelOps<TModel> : CameraModelOps
		where TModel : struct, ICameraModel<TModel>
	{
		public override CameraModelId ModelId => TModel.ModelId;

		public override string ModelName => TModel.ModelName;

		public override string ParamsInfo => TModel.ParamsInfo;

		public override int NumParams => TModel.NumParams;

		public override ReadOnlySpan<int> FocalLengthIdxs => TModel.FocalLengthIdxs;

		public override ReadOnlySpan<int> PrincipalPointIdxs => TModel.PrincipalPointIdxs;

		public override ReadOnlySpan<int> ExtraParamsIdxs => TModel.ExtraParamsIdxs;

		public override ReadOnlySpan<int> MetaDataParamsIdxs => TModel.MetaDataParamsIdxs;

		public override bool IsPerspective => TModel.IsPerspective;

		public override bool IsPerspectivePinhole => TModel.IsPerspectivePinhole;

		public override bool IsPerspectiveFisheye => TModel.IsPerspectiveFisheye;

		public override bool IsSpherical => TModel.IsSpherical;

		public override double[] InitializeParams(double focalLength, int width, int height) =>
			TModel.InitializeParams(focalLength, width, height);

		public override bool ImgFromCam(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, bool checkCheirality) =>
			CameraModelMath.ImgFromCam<TModel>(parameters, u, v, w, out x, out y, checkCheirality);

		public override bool CamFromImg(ReadOnlySpan<double> parameters, double x, double y, out double u, out double v) =>
			TModel.CamFromImg(parameters, x, y, out u, out v);

		public override bool CamRayFromImg(ReadOnlySpan<double> parameters, double x, double y, out double rx, out double ry, out double rz) =>
			TModel.CamRayFromImg(parameters, x, y, out rx, out ry, out rz);

		public override double CamFromImgThreshold(ReadOnlySpan<double> parameters, double threshold) =>
			TModel.CamFromImgThreshold(parameters, threshold);

		public override bool HasBogusParams(ReadOnlySpan<double> parameters, int width, int height, double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam) =>
			TModel.HasBogusParams(parameters, width, height, minFocalLengthRatio, maxFocalLengthRatio, maxExtraParam);

		public override void Rescale(double scaleX, double scaleY, Span<double> parameters) =>
			TModel.Rescale(scaleX, scaleY, parameters);
	}
}
