// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeres.Default: the DefaultBundleAdjuster of
// colmap/estimators/bundle_adjustment_ceres.cc with the file-local helpers it calls:
// the residual setup (AddImageToProblem for trivial and rig frames, AddPointToProblem for
// tracks reaching outside the configured images), ParameterizeCameras (SubsetManifold over
// the fixed focal / principal point / extra / metadata parameters), ParameterizeRigsAndFrames
// (7-value pose blocks with the EigenQuaternion x Euclidean<3> product manifold, or
// SubsetManifold for a constant rotation), ParameterizePoints, and the two gauge fixes
// (FixGaugeWithTwoCamsFromWorld, FixGaugeWithThreePoints, in BundleAdjustmentCeres.Gauge.cs).
// Options, summary and the solve call are in BundleAdjustmentCeres.cs.
//
// Parameter blocks are the scene's own storage, so the solver writes the result in place as
// Ceres does through COLMAP's raw pointers: Point3D.XyzParams, Camera.Params,
// Frame.RigFromWorldStorage.Params and Rig.SensorFromRigStorage(...).Params.
//
// Translation notes:
// - Iteration order (divergence 39): COLMAP iterates the config's
//   FlatHashSets (abseil's hash order, which is seeded per process) when adding residuals
//   and its FlatHashMap of per-point observation counts when choosing the three gauge
//   points. Here the config's images and points are visited in ascending id order and the
//   observation counts in first-seen order, so a run is reproducible.
// - The LOG(WARNING)s on a failed gauge fix and the LOG(ERROR) on an unusable solution go to
//   Util/Log.cs; the "Bundle adjustment report" (print_summary, LOG(INFO)) is not printed.
//   A failed gauge fix leaves the problem as COLMAP leaves it.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of the file-local colmap::DefaultBundleAdjuster.</summary>
internal sealed partial class DefaultBundleAdjuster : CeresBundleAdjuster
{
	private readonly Problem problem = new();
	private readonly LossFunction lossFunction;
	private readonly SortedSet<uint> parameterizedCameraIds = [];
	private readonly SortedSet<uint> parameterizedImageIds = [];

	// point3D_id -> number of its observations in the problem, in first-seen order.
	// Keep this append-only (never Remove): first-seen iteration relies on Dictionary
	// enumerating in insertion order, which only holds while nothing is removed.
	private readonly Dictionary<ulong, int> point3DNumObservations = [];

	public DefaultBundleAdjuster(BundleAdjustmentOptions options, BundleAdjustmentConfig config, Reconstruction reconstruction)
		: base(options, config)
	{
		lossFunction = OptionsInternal.Ceres!.CreateLossFunction();

		// Verify that reconstruction is internally consistent.
		Check.That(reconstruction.IsValid());

		// Set up problem.
		// Warning: AddPointsToProblem assumes that AddImageToProblem is called first. Do not
		// change order of instructions!
		foreach (uint imageId in ConfigInternal.Images.Order())
		{
			AddImageToProblem(imageId, reconstruction);
		}

		foreach (ulong point3DId in ConfigInternal.VariablePoints.Order())
		{
			AddPointToProblem(point3DId, reconstruction);
		}

		foreach (ulong point3DId in ConfigInternal.ConstantPoints.Order())
		{
			AddPointToProblem(point3DId, reconstruction);
		}

		ParameterizeCameras(reconstruction);
		ParameterizeRigsAndFrames(reconstruction);
		ParameterizePoints(reconstruction);

		switch (ConfigInternal.FixedGauge)
		{
			case BundleAdjustmentGauge.Unspecified:
				break;
			case BundleAdjustmentGauge.TwoCamsFromWorld:
				FixGaugeWithTwoCamsFromWorld(reconstruction);
				break;
			case BundleAdjustmentGauge.ThreePoints:
				FixGaugeWithThreePoints(reconstruction);
				break;
			default:
				throw new InvalidOperationException("Unknown BundleAdjustmentGauge");
		}
	}

	/// <inheritdoc/>
	public override Problem Problem => problem;

	/// <summary>Images that received at least one residual, ascending.</summary>
	public IReadOnlySet<uint> ParameterizedImageIds => parameterizedImageIds;

	/// <inheritdoc/>
	public override BundleAdjustmentSummary Solve(CancellationToken cancellationToken = default)
	{
		if (problem.NumResiduals == 0)
		{
			return new BundleAdjustmentSummary();
		}

		SolverSummary ceresSummary = CeresBundleAdjusters.Solve(OptionsInternal, ConfigInternal, problem, cancellationToken);
		return CeresBundleAdjusters.CreateSummaryAndLogFailure(ceresSummary, "Bundle adjustment");
	}

	private bool SkipForTrackLength(Point3D point3D) =>
		OptionsInternal.MinTrackLength > 0 && point3D.Track.Length < OptionsInternal.MinTrackLength;

	private void CountObservation(ulong point3DId)
	{
		point3DNumObservations.TryGetValue(point3DId, out int count);
		point3DNumObservations[point3DId] = count + 1;
	}

	private void AddImageToProblem(uint imageId, Reconstruction reconstruction)
	{
		Image image = reconstruction.Image(imageId);
		if (image.IsRefInFrame)
		{
			AddImageWithTrivialFrame(image, reconstruction);
		}
		else
		{
			AddImageWithNonTrivialFrame(image, reconstruction);
		}
	}

	private void AddImageWithTrivialFrame(Image image, Reconstruction reconstruction)
	{
		Camera camera = image.CameraPtr;
		bool constantCamFromWorld =
			!OptionsInternal.RefineRigFromWorld || ConfigInternal.HasConstantRigFromWorldPose(image.FrameId);

		Check.That(image.IsRefInFrame);
		Rigid3dStorage rigFromWorld = image.FramePtr.RigFromWorldStorage;

		// Add residuals to bundle adjustment problem.
		int numObservations = 0;
		foreach (Point2D point2D in image.Points2D)
		{
			if (!point2D.HasPoint3D || ConfigInternal.IsIgnoredPoint(point2D.Point3DId))
			{
				continue;
			}

			Point3D point3D = reconstruction.Point3D(point2D.Point3DId);
			Check.Gt(point3D.Track.Length, 1);

			// Skip points with track length below minimum.
			if (SkipForTrackLength(point3D))
			{
				continue;
			}

			numObservations += 1;
			CountObservation(point2D.Point3DId);

			if (constantCamFromWorld)
			{
				problem.AddResidualBlock(
					CameraCostFunctions.CreateReprojErrorConstantPoseCostFunction(camera.ModelId, point2D.Xy, rigFromWorld.Value),
					lossFunction,
					point3D.XyzParams,
					camera.Params);
			}
			else
			{
				problem.AddResidualBlock(
					CameraCostFunctions.CreateReprojErrorCostFunction(camera.ModelId, point2D.Xy),
					lossFunction,
					point3D.XyzParams,
					rigFromWorld.Params,
					camera.Params);
			}
		}

		if (numObservations > 0)
		{
			parameterizedCameraIds.Add(image.CameraId);
			parameterizedImageIds.Add(image.ImageId);
		}
	}

	private void AddImageWithNonTrivialFrame(Image image, Reconstruction reconstruction)
	{
		Camera camera = image.CameraPtr;
		SensorId sensorId = camera.SensorId;

		bool constantSensorFromRig =
			!OptionsInternal.RefineSensorFromRig || ConfigInternal.HasConstantSensorFromRigPose(sensorId);
		bool constantRigFromWorld =
			!OptionsInternal.RefineRigFromWorld || ConfigInternal.HasConstantRigFromWorldPose(image.FrameId);

		Check.That(!image.IsRefInFrame);
		Rigid3dStorage sensorFromRig = image.FramePtr.RigPtr.SensorFromRigStorage(sensorId);
		Rigid3dStorage rigFromWorld = image.FramePtr.RigFromWorldStorage;
		Rigid3d? camFromWorld = constantSensorFromRig && constantRigFromWorld
			? sensorFromRig.Value * rigFromWorld.Value
			: null;

		// Add residuals to bundle adjustment problem.
		int numObservations = 0;
		foreach (Point2D point2D in image.Points2D)
		{
			if (!point2D.HasPoint3D || ConfigInternal.IsIgnoredPoint(point2D.Point3DId))
			{
				continue;
			}

			Point3D point3D = reconstruction.Point3D(point2D.Point3DId);
			Check.Gt(point3D.Track.Length, 1);

			// Skip points with track length below minimum.
			if (SkipForTrackLength(point3D))
			{
				continue;
			}

			numObservations += 1;
			CountObservation(point2D.Point3DId);

			// The !constant_sensor_from_rig && constant_rig_from_world is rare enough that we
			// do not have a specialized cost function for it.
			if (constantSensorFromRig && constantRigFromWorld)
			{
				problem.AddResidualBlock(
					CameraCostFunctions.CreateReprojErrorConstantPoseCostFunction(camera.ModelId, point2D.Xy, camFromWorld!.Value),
					lossFunction,
					point3D.XyzParams,
					camera.Params);
			}
			else if (!constantRigFromWorld && constantSensorFromRig)
			{
				problem.AddResidualBlock(
					CameraCostFunctions.CreateRigReprojErrorConstantRigCostFunction(camera.ModelId, point2D.Xy, sensorFromRig.Value),
					lossFunction,
					point3D.XyzParams,
					rigFromWorld.Params,
					camera.Params);
			}
			else
			{
				problem.AddResidualBlock(
					CameraCostFunctions.CreateRigReprojErrorCostFunction(camera.ModelId, point2D.Xy),
					lossFunction,
					point3D.XyzParams,
					sensorFromRig.Params,
					rigFromWorld.Params,
					camera.Params);
			}
		}

		if (numObservations > 0)
		{
			parameterizedCameraIds.Add(image.CameraId);
			parameterizedImageIds.Add(image.ImageId);
		}
	}

	private void AddPointToProblem(ulong point3DId, Reconstruction reconstruction)
	{
		Check.That(!ConfigInternal.IsIgnoredPoint(point3DId));
		Point3D point3D = reconstruction.Point3D(point3DId);

		// Skip points with track length below minimum.
		if (SkipForTrackLength(point3D))
		{
			return;
		}

		point3DNumObservations.TryGetValue(point3DId, out int numObservations);
		point3DNumObservations[point3DId] = numObservations;

		// Is 3D point already fully contained in the problem? I.e. its entire track is
		// contained in `variable_image_ids`, `constant_image_ids`, `constant_x_image_ids`.
		if (numObservations == point3D.Track.Length)
		{
			return;
		}

		foreach (TrackElement trackEl in point3D.Track.Elements)
		{
			// Skip observations that were already added in `FillImages`.
			if (ConfigInternal.HasImage(trackEl.ImageId))
			{
				continue;
			}

			numObservations += 1;
			point3DNumObservations[point3DId] = numObservations;

			Image image = reconstruction.Image(trackEl.ImageId);
			Camera camera = image.CameraPtr;
			Point2D point2D = image.Points2D[(int)trackEl.Point2DIdx];

			Rigid3d camFromWorld = image.IsRefInFrame
				? image.FramePtr.RigFromWorld()
				: image.FramePtr.RigPtr.SensorFromRig(camera.SensorId) * image.FramePtr.RigFromWorld();
			problem.AddResidualBlock(
				CameraCostFunctions.CreateReprojErrorConstantPoseCostFunction(camera.ModelId, point2D.Xy, camFromWorld),
				lossFunction,
				point3D.XyzParams,
				camera.Params);

			// Do not optimize intrinsics if the corresponding images were not included
			// explicitly in the config.
			if (parameterizedCameraIds.Add(image.CameraId))
			{
				ConfigInternal.SetConstantCamIntrinsics(image.CameraId);
			}
		}
	}

	private void ParameterizeCameras(Reconstruction reconstruction)
	{
		bool constantCamera = !OptionsInternal.RefineFocalLength
			&& !OptionsInternal.RefinePrincipalPoint
			&& !OptionsInternal.RefineExtraParams;
		foreach (uint cameraId in parameterizedCameraIds)
		{
			Camera camera = reconstruction.Camera(cameraId);

			if (constantCamera || ConfigInternal.HasConstantCamIntrinsics(cameraId))
			{
				problem.SetParameterBlockConstant(camera.Params);
				continue;
			}

			var constCameraParams = new List<int>(camera.Params.Length);

			// Metadata parameters (e.g. the (w, h) image dimensions of spherical models) are
			// sensor properties and are never optimized.
			constCameraParams.AddRange(camera.MetaDataParamsIdxs);
			if (!OptionsInternal.RefineFocalLength)
			{
				constCameraParams.AddRange(camera.FocalLengthIdxs);
			}

			if (!OptionsInternal.RefinePrincipalPoint)
			{
				constCameraParams.AddRange(camera.PrincipalPointIdxs);
			}

			if (!OptionsInternal.RefineExtraParams)
			{
				constCameraParams.AddRange(camera.ExtraParamsIdxs);
			}

			if (constCameraParams.Count == camera.Params.Length)
			{
				problem.SetParameterBlockConstant(camera.Params);
			}
			else if (constCameraParams.Count > 0)
			{
				ManifoldHelpers.SetManifold(
					problem, camera.Params, ManifoldHelpers.CreateSubsetManifold(camera.Params.Length, constCameraParams));
			}
		}
	}

	// CostFunction assumes unit quaternions: Eigen's in-place rotation().normalize().
	private static void NormalizeRotation(Rigid3dStorage pose)
	{
		Rigid3d value = pose.Value;
		pose.Value = new Rigid3d(value.Rotation.Normalized(), value.Translation);
	}

	private static Manifold PoseManifold() =>
		ManifoldHelpers.CreateProductManifold(
			ManifoldHelpers.CreateEigenQuaternionManifold(), ManifoldHelpers.CreateEuclideanManifold(3));

	private void ParameterizeRigsAndFrames(Reconstruction reconstruction)
	{
		var parameterizedRigIds = new HashSet<uint>();
		var parameterizedSensorIds = new HashSet<SensorId>();
		var parameterizedFrameIds = new HashSet<uint>();
		foreach (uint imageId in parameterizedImageIds)
		{
			Image image = reconstruction.Image(imageId);
			parameterizedRigIds.Add(image.FramePtr.RigId);

			// Parameterize sensor_from_rig.
			SensorId sensorId = image.CameraPtr.SensorId;
			if (parameterizedSensorIds.Add(sensorId) && !image.IsRefInFrame)
			{
				Rigid3dStorage sensorFromRig = image.FramePtr.RigPtr.SensorFromRigStorage(sensorId);
				NormalizeRotation(sensorFromRig);
				if (problem.HasParameterBlock(sensorFromRig.Params))
				{
					ManifoldHelpers.SetManifold(problem, sensorFromRig.Params, PoseManifold());
					if (!OptionsInternal.RefineSensorFromRig || ConfigInternal.HasConstantSensorFromRigPose(sensorId))
					{
						problem.SetParameterBlockConstant(sensorFromRig.Params);
					}
				}
			}

			// Parameterize rig_from_world.
			if (parameterizedFrameIds.Add(image.FrameId))
			{
				Rigid3dStorage rigFromWorld = image.FramePtr.RigFromWorldStorage;
				NormalizeRotation(rigFromWorld);
				if (problem.HasParameterBlock(rigFromWorld.Params))
				{
					if (!OptionsInternal.RefineRigFromWorld || ConfigInternal.HasConstantRigFromWorldPose(image.FrameId))
					{
						problem.SetParameterBlockConstant(rigFromWorld.Params);
					}
					else if (OptionsInternal.ConstantRigFromWorldRotation)
					{
						ManifoldHelpers.SetManifold(problem, rigFromWorld.Params, ManifoldHelpers.CreateSubsetManifold(7, [0, 1, 2, 3]));
					}
					else
					{
						ManifoldHelpers.SetManifold(problem, rigFromWorld.Params, PoseManifold());
					}
				}
			}
		}

		// Set the rig poses as constant, if the reference sensor is not part of the problem.
		// Otherwise, the relative pose between the sensors is not well constrained. Notice
		// that this does not handle degenerate configurations and assumes the observations in
		// the problem constrain the relative poses sufficiently.
		foreach (uint rigId in parameterizedRigIds)
		{
			Rig rig = reconstruction.Rig(rigId);
			if (parameterizedSensorIds.Contains(rig.RefSensorId))
			{
				continue;
			}

			foreach (SensorId sensorId in rig.NonRefSensors.Keys)
			{
				Check.That(rig.HasSensorFromRig(sensorId));
				double[] sensorFromRig = rig.SensorFromRigStorage(sensorId).Params;
				if (problem.HasParameterBlock(sensorFromRig))
				{
					problem.SetParameterBlockConstant(sensorFromRig);
				}
			}
		}
	}

	private void ParameterizePoints(Reconstruction reconstruction)
	{
		// First-seen order: relies on point3DNumObservations staying append-only.
		foreach ((ulong point3DId, int numObservations) in point3DNumObservations)
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			if (!OptionsInternal.RefinePoints3D || point3D.Track.Length > numObservations)
			{
				problem.SetParameterBlockConstant(point3D.XyzParams);
			}
		}

		foreach (ulong point3DId in ConfigInternal.ConstantPoints)
		{
			problem.SetParameterBlockConstant(reconstruction.Point3D(point3DId).XyzParams);
		}
	}
}
