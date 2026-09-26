// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraCostFunctions: colmap::CreateCameraCostFunction from
// colmap/estimators/cost_functions/reprojection_error.h - builds the cost function of a
// camera-model-templated functor for a model id known only at run time, as the C++ switch
// over CAMERA_MODEL_SWITCH_CASES does.
//
// C++ passes the functor template itself (`CreateCameraCostFunction<ReprojErrorCostFunctor>
// (model_id, args...)`). C# cannot pass an open generic type, so the functor family is an
// ICameraCostFunctionFactory struct carrying the constructor arguments, whose generic
// Create<TModel> the switch calls with the model struct; the named helpers below are the
// five families COLMAP dispatches (bundle_adjustment_ceres.cc, pose.cc,
// generalized_pose.cc). As in COLMAP, ReprojErrorCostFunctor and
// ReprojErrorConstantPoseCostFunctor map to their analytic-Jacobian cost functions
// (AnalyticalReprojectionError.cs), because every camera model implements
// ImgFromCamWithJac (has_img_from_cam_with_jac); the other families are autodiff
// (ReprojectionError.cs). An unknown id throws, where the C++ switch falls off the end.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// A family of camera-model-templated cost functions with its constructor arguments: the C#
/// stand-in for the template template parameter of colmap::CreateCameraCostFunction.
/// </summary>
public interface ICameraCostFunctionFactory
{
	/// <summary>The cost function for camera model <typeparamref name="TModel"/>.</summary>
	CostFunction Create<TModel>()
		where TModel : struct, ICameraModel<TModel>;
}

/// <summary>Port of colmap::CreateCameraCostFunction and its uses.</summary>
public static class CameraCostFunctions
{
	/// <summary>
	/// colmap::CreateCameraCostFunction: the factory's cost function for the camera model
	/// with the given id.
	/// </summary>
	public static CostFunction CreateCameraCostFunction<TFactory>(CameraModelId cameraModelId, in TFactory factory)
		where TFactory : struct, ICameraCostFunctionFactory => cameraModelId switch
		{
			CameraModelId.SimplePinhole => factory.Create<SimplePinholeCameraModel>(),
			CameraModelId.Pinhole => factory.Create<PinholeCameraModel>(),
			CameraModelId.SimpleRadial => factory.Create<SimpleRadialCameraModel>(),
			CameraModelId.Radial => factory.Create<RadialCameraModel>(),
			CameraModelId.OpenCV => factory.Create<OpenCVCameraModel>(),
			CameraModelId.OpenCVFisheye => factory.Create<OpenCVFisheyeCameraModel>(),
			CameraModelId.FullOpenCV => factory.Create<FullOpenCVCameraModel>(),
			CameraModelId.FOV => factory.Create<FOVCameraModel>(),
			CameraModelId.SimpleRadialFisheye => factory.Create<SimpleRadialFisheyeCameraModel>(),
			CameraModelId.RadialFisheye => factory.Create<RadialFisheyeCameraModel>(),
			CameraModelId.ThinPrismFisheye => factory.Create<ThinPrismFisheyeCameraModel>(),
			CameraModelId.RadTanThinPrismFisheye => factory.Create<RadTanThinPrismFisheyeModel>(),
			CameraModelId.SimpleDivision => factory.Create<SimpleDivisionCameraModel>(),
			CameraModelId.Division => factory.Create<DivisionCameraModel>(),
			CameraModelId.SimpleFisheye => factory.Create<SimpleFisheyeCameraModel>(),
			CameraModelId.Fisheye => factory.Create<FisheyeCameraModel>(),
			CameraModelId.EUCM => factory.Create<EUCMCameraModel>(),
			CameraModelId.Equirectangular => factory.Create<EquirectangularCameraModel>(),
			_ => throw new ArgumentException("Camera model does not exist", nameof(cameraModelId)),
		};

	/// <summary>
	/// <c>CreateCameraCostFunction&lt;ReprojErrorCostFunctor&gt;(model_id, point2D)</c>: the
	/// analytic <see cref="AnalyticalReprojErrorCostFunction{TModel}"/>.
	/// </summary>
	public static CostFunction CreateReprojErrorCostFunction(CameraModelId cameraModelId, Vector2d point2D) =>
		CreateCameraCostFunction(cameraModelId, new ReprojErrorFactory(point2D));

	/// <summary>
	/// <c>CreateCameraCostFunction&lt;ReprojErrorConstantPoseCostFunctor&gt;(model_id, point2D,
	/// cam_from_world)</c>: the analytic
	/// <see cref="AnalyticalReprojErrorConstantPoseCostFunction{TModel}"/>.
	/// </summary>
	public static CostFunction CreateReprojErrorConstantPoseCostFunction(CameraModelId cameraModelId, Vector2d point2D, Rigid3d camFromWorld) =>
		CreateCameraCostFunction(cameraModelId, new ReprojErrorConstantPoseFactory(point2D, camFromWorld));

	/// <summary><c>CreateCameraCostFunction&lt;ReprojErrorConstantPoint3DCostFunctor&gt;</c>.</summary>
	public static CostFunction CreateReprojErrorConstantPoint3DCostFunction(CameraModelId cameraModelId, Vector2d point2D, Vector3d point3DInWorld) =>
		CreateCameraCostFunction(cameraModelId, new ReprojErrorConstantPoint3DFactory(point2D, point3DInWorld));

	/// <summary><c>CreateCameraCostFunction&lt;RigReprojErrorCostFunctor&gt;</c>.</summary>
	public static CostFunction CreateRigReprojErrorCostFunction(CameraModelId cameraModelId, Vector2d point2D) =>
		CreateCameraCostFunction(cameraModelId, new RigReprojErrorFactory(point2D));

	/// <summary><c>CreateCameraCostFunction&lt;RigReprojErrorConstantRigCostFunctor&gt;</c>.</summary>
	public static CostFunction CreateRigReprojErrorConstantRigCostFunction(CameraModelId cameraModelId, Vector2d point2D, Rigid3d camFromRig) =>
		CreateCameraCostFunction(cameraModelId, new RigReprojErrorConstantRigFactory(point2D, camFromRig));

	private readonly struct ReprojErrorFactory(Vector2d point2D) : ICameraCostFunctionFactory
	{
		public CostFunction Create<TModel>()
			where TModel : struct, ICameraModel<TModel> =>
			new AnalyticalReprojErrorCostFunction<TModel>(point2D);
	}

	private readonly struct ReprojErrorConstantPoseFactory(Vector2d point2D, Rigid3d camFromWorld) : ICameraCostFunctionFactory
	{
		public CostFunction Create<TModel>()
			where TModel : struct, ICameraModel<TModel> =>
			new AnalyticalReprojErrorConstantPoseCostFunction<TModel>(point2D, camFromWorld);
	}

	private readonly struct ReprojErrorConstantPoint3DFactory(Vector2d point2D, Vector3d point3DInWorld) : ICameraCostFunctionFactory
	{
		public CostFunction Create<TModel>()
			where TModel : struct, ICameraModel<TModel> =>
			ReprojErrorConstantPoint3DCostFunctor<TModel>.Create(point2D, point3DInWorld);
	}

	private readonly struct RigReprojErrorFactory(Vector2d point2D) : ICameraCostFunctionFactory
	{
		public CostFunction Create<TModel>()
			where TModel : struct, ICameraModel<TModel> =>
			RigReprojErrorCostFunctor<TModel>.Create(point2D);
	}

	private readonly struct RigReprojErrorConstantRigFactory(Vector2d point2D, Rigid3d camFromRig) : ICameraCostFunctionFactory
	{
		public CostFunction Create<TModel>()
			where TModel : struct, ICameraModel<TModel> =>
			RigReprojErrorConstantRigCostFunctor<TModel>.Create(point2D, camFromRig);
	}
}
