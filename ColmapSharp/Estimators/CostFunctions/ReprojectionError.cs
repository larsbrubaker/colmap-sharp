// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReprojectionError: the autodiff half of colmap/estimators/cost_functions/reprojection_error.h -
// WrapEquirectangularHorizontalSeam and the five bundle-adjustment reprojection functors
// (ReprojErrorCostFunctor, ReprojErrorConstantPoseCostFunctor,
// ReprojErrorConstantPoint3DCostFunctor, RigReprojErrorCostFunctor,
// RigReprojErrorConstantRigCostFunctor), each generic over the camera model like the C++
// templates (Sensor/CameraModelBase.cs). The analytic-Jacobian cost functions are in
// AnalyticalReprojectionError.cs and the dispatch by CameraModelId (CreateCameraCostFunction)
// in CameraCostFunctions.cs. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/ReprojectionErrorTests.cs.
//
// Pose blocks use COLMAP's Rigid3d params layout [qx, qy, qz, qw, tx, ty, tz] (one block of
// 7), and every functor lists its blocks in the C++ order. Evaluate reads them from the
// concatenated span IAutoDiffFunctor passes. The quaternion algebra is QuaternionT.cs.
// Tier B: the arithmetic follows the C++ expression order, but Eigen's SIMD grouping on the
// double path is not reproduced (see QuaternionT.cs; divergence 115).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>The shared pieces of the reprojection functors.</summary>
public static class ReprojectionErrors
{
	/// <summary>
	/// Periodic (azimuthal) camera models such as EQUIRECTANGULAR wrap the x image coordinate
	/// at the ±π seam, so a raw pixel residual can jump by ~width across the seam (e.g. an
	/// observation at x ≈ 0 whose 3D point reprojects to x ≈ width). Wraps the x-residual
	/// into [-width/2, width/2) so the bundle-adjustment cost stays continuous across the
	/// seam. The offset is locally constant, so it does not perturb the residual's
	/// derivatives. No-op for non-periodic camera models. (Elevation has no wrap, so y is
	/// untouched.) Port of colmap::WrapEquirectangularHorizontalSeam.
	/// </summary>
	public static void WrapEquirectangularHorizontalSeam<TModel, T>(ReadOnlySpan<T> cameraParams, Span<T> residuals)
		where TModel : struct, ICameraModel<TModel>
		where T : struct, IScalar<T>
	{
		if (TModel.ModelId == CameraModelId.Equirectangular)
		{
			T width = cameraParams[0];
			residuals[0] -= width * T.Floor(residuals[0] / width + T.FromDouble(0.5));
		}
	}

	/// <summary>
	/// Projects a camera-frame point and writes projection - point2D (seam-wrapped) to the
	/// residuals, or zeros when the projection fails: the common tail of every functor.
	/// </summary>
	internal static bool ProjectResidual<TModel, T>(in Vector3T<T> pointInCam, ReadOnlySpan<T> cameraParams, Vector2d point2D, Span<T> residuals)
		where TModel : struct, ICameraModel<TModel>
		where T : struct, IScalar<T>
	{
		if (TModel.ImgFromCam(cameraParams, pointInCam.X, pointInCam.Y, pointInCam.Z, out T x, out T y))
		{
			residuals[0] = x - point2D.X;
			residuals[1] = y - point2D.Y;
			WrapEquirectangularHorizontalSeam<TModel, T>(cameraParams, residuals);
		}
		else
		{
			residuals[0] = T.FromDouble(0);
			residuals[1] = T.FromDouble(0);
		}

		return true;
	}
}

/// <summary>
/// Standard bundle adjustment cost function for variable camera pose, calibration, and
/// point parameters. Blocks: point3D_in_world (3), cam_from_world (7), camera params.
/// Port of colmap::ReprojErrorCostFunctor.
/// </summary>
public readonly struct ReprojErrorCostFunctor<TModel>(Vector2d point2D) : ISizedAutoDiffFunctor
	where TModel : struct, ICameraModel<TModel>
{
	private static readonly int[] Sizes = [3, 7, TModel.NumParams];

	private readonly Vector2d _point2D = point2D;

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector2d point2D) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new ReprojErrorCostFunctor<TModel>(point2D));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Vector3T<T> pointInCam =
			QuaternionT<T>.Map(parameters[3..]) * Vector3T<T>.Map(parameters) + Vector3T<T>.Map(parameters[7..]);
		return ReprojectionErrors.ProjectResidual<TModel, T>(pointInCam, parameters[10..], _point2D, residuals);
	}
}

/// <summary>
/// Bundle adjustment cost function for variable camera calibration and point parameters,
/// and fixed camera pose. Since the pose is constant, it is stored as a precomputed rotation
/// matrix and translation rather than a quaternion: applying a fixed rotation as a
/// matrix-vector product is faster than a quaternion rotation on every evaluation.
/// Blocks: point3D_in_world (3), camera params.
/// Port of colmap::ReprojErrorConstantPoseCostFunctor.
/// </summary>
public readonly struct ReprojErrorConstantPoseCostFunctor<TModel> : ISizedAutoDiffFunctor
	where TModel : struct, ICameraModel<TModel>
{
	private static readonly int[] Sizes = [3, TModel.NumParams];

	private readonly Vector2d _point2D;
	private readonly Matrix3d _camFromWorldRotation;
	private readonly Vector3d _camFromWorldTranslation;

	/// <summary>Creates the functor for an observation and the fixed pose.</summary>
	public ReprojErrorConstantPoseCostFunctor(Vector2d point2D, Rigid3d camFromWorld)
	{
		_point2D = point2D;
		_camFromWorldRotation = camFromWorld.Rotation.ToRotationMatrix();
		_camFromWorldTranslation = camFromWorld.Translation;
	}

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector2d point2D, Rigid3d camFromWorld) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new ReprojErrorConstantPoseCostFunctor<TModel>(point2D, camFromWorld));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Matrix3d r = _camFromWorldRotation;
		Vector3d t = _camFromWorldTranslation;
		T x = parameters[0], y = parameters[1], z = parameters[2];
		var pointInCam = new Vector3T<T>(
			r[0, 0] * x + r[0, 1] * y + r[0, 2] * z + t.X,
			r[1, 0] * x + r[1, 1] * y + r[1, 2] * z + t.Y,
			r[2, 0] * x + r[2, 1] * y + r[2, 2] * z + t.Z);
		return ReprojectionErrors.ProjectResidual<TModel, T>(pointInCam, parameters[3..], _point2D, residuals);
	}
}

/// <summary>
/// Bundle adjustment cost function for variable camera pose and calibration parameters, and
/// fixed point. Blocks: cam_from_world (7), camera params.
/// Port of colmap::ReprojErrorConstantPoint3DCostFunctor.
/// </summary>
public readonly struct ReprojErrorConstantPoint3DCostFunctor<TModel>(Vector2d point2D, Vector3d point3DInWorld)
	: ISizedAutoDiffFunctor
	where TModel : struct, ICameraModel<TModel>
{
	private static readonly int[] Sizes = [7, TModel.NumParams];

	private readonly Vector2d _point2D = point2D;
	private readonly Vector3d _point3DInWorld = point3DInWorld;

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector2d point2D, Vector3d point3DInWorld) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new ReprojErrorConstantPoint3DCostFunctor<TModel>(point2D, point3DInWorld));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Vector3T<T> pointInCam =
			QuaternionT<T>.Map(parameters) * Vector3T<T>.From(_point3DInWorld) + Vector3T<T>.Map(parameters[4..]);
		return ReprojectionErrors.ProjectResidual<TModel, T>(pointInCam, parameters[7..], _point2D, residuals);
	}
}

/// <summary>
/// Rig bundle adjustment cost function for variable camera pose and calibration and point
/// parameters. Different from the standard bundle adjustment function, this cost function
/// is suitable for camera rigs with consistent relative poses of the cameras within the
/// rig. The cost function first projects points into the local system of the camera rig
/// and then into the local system of the camera within the rig. Blocks: point3D_in_world
/// (3), cam_from_rig (7), rig_from_world (7), camera params.
/// Port of colmap::RigReprojErrorCostFunctor.
/// </summary>
public readonly struct RigReprojErrorCostFunctor<TModel>(Vector2d point2D) : ISizedAutoDiffFunctor
	where TModel : struct, ICameraModel<TModel>
{
	private static readonly int[] Sizes = [3, 7, 7, TModel.NumParams];

	private readonly Vector2d _point2D = point2D;

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector2d point2D) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new RigReprojErrorCostFunctor<TModel>(point2D));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Vector3T<T> pointInRig =
			QuaternionT<T>.Map(parameters[10..]) * Vector3T<T>.Map(parameters) + Vector3T<T>.Map(parameters[14..]);
		Vector3T<T> pointInCam =
			QuaternionT<T>.Map(parameters[3..]) * pointInRig + Vector3T<T>.Map(parameters[7..]);
		return ReprojectionErrors.ProjectResidual<TModel, T>(pointInCam, parameters[17..], _point2D, residuals);
	}
}

/// <summary>
/// Rig bundle adjustment cost function for variable camera pose and camera calibration and
/// point parameters but fixed rig extrinsic poses. Blocks: point3D_in_world (3),
/// rig_from_world (7), camera params.
/// Port of colmap::RigReprojErrorConstantRigCostFunctor.
/// </summary>
public readonly struct RigReprojErrorConstantRigCostFunctor<TModel>(Vector2d point2D, Rigid3d camFromRig)
	: ISizedAutoDiffFunctor
	where TModel : struct, ICameraModel<TModel>
{
	private static readonly int[] Sizes = [3, 7, TModel.NumParams];

	private readonly Vector2d _point2D = point2D;
	private readonly Rigid3d _camFromRig = camFromRig;

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector2d point2D, Rigid3d camFromRig) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new RigReprojErrorConstantRigCostFunctor<TModel>(point2D, camFromRig));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		// C++ casts cam_from_rig to T and runs RigReprojErrorCostFunctor on it.
		Vector3T<T> pointInRig =
			QuaternionT<T>.Map(parameters[3..]) * Vector3T<T>.Map(parameters) + Vector3T<T>.Map(parameters[7..]);
		Vector3T<T> pointInCam =
			QuaternionT<T>.From(_camFromRig.Rotation) * pointInRig + Vector3T<T>.From(_camFromRig.Translation);
		return ReprojectionErrors.ProjectResidual<TModel, T>(pointInCam, parameters[10..], _point2D, residuals);
	}
}
