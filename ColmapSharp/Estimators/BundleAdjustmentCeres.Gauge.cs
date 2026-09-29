// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeres.Gauge: the gauge fixes of colmap/estimators/bundle_adjustment_ceres.cc
// (FixedGaugeWithThreePoints, FixGaugeWithThreePoints, FixGaugeWithTwoCamsFromWorld), part of
// DefaultBundleAdjuster (BundleAdjustmentCeres.Default.cs). Bundle adjustment has a 7-DOF
// similarity gauge freedom; these hold either three linearly independent points constant, or
// one camera pose plus the largest baseline coordinate of a second one.
//
// Translation notes:
// - Eigen's colPivHouseholderQr().rank() is LinearAlgebra/ColPivHouseholderQR.cs (same
//   threshold rule); Eigen's maxCoeff(&index) returns the first maximum, as here.
// - The three candidate points are taken in the problem's first-seen order of points
//   (divergence 39), where COLMAP takes abseil hash order.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

internal sealed partial class DefaultBundleAdjuster
{
	private void FixGaugeWithThreePoints(Reconstruction reconstruction)
	{
		var fixedGauge = new FixedGaugeWithThreePoints();

		// First check if we already fixed enough points in the problem.
		// First-seen order: relies on point3DNumObservations staying append-only.
		foreach (ulong point3DId in point3DNumObservations.Keys)
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			if (problem.IsParameterBlockConstant(point3D.XyzParams)
				&& fixedGauge.MaybeAddFixedPoint(point3D.Xyz)
				&& fixedGauge.NumFixedPoints >= 3)
			{
				return;
			}
		}

		// Otherwise, fix sufficient points in the problem.
		// First-seen order: relies on point3DNumObservations staying append-only.
		foreach (ulong point3DId in point3DNumObservations.Keys)
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			if (!problem.IsParameterBlockConstant(point3D.XyzParams) && fixedGauge.MaybeAddFixedPoint(point3D.Xyz))
			{
				problem.SetParameterBlockConstant(point3D.XyzParams);
				if (fixedGauge.NumFixedPoints >= 3)
				{
					return;
				}
			}
		}

		Log.Warning($"Failed to fix Gauge due to insufficient number of fixed points: {fixedGauge.NumFixedPoints}");
	}

	// Whether a sensor is either a reference sensor, or a non-reference sensor with
	// sensor_from_rig fixed.
	private bool IsParameterizedConstSensor(Image image)
	{
		SensorId sensorId = image.CameraPtr.SensorId;
		if (image.FramePtr.RigPtr.IsRefSensor(sensorId))
		{
			return true;
		}

		double[] sensorFromRig = image.FramePtr.RigPtr.SensorFromRigStorage(sensorId).Params;
		if (problem.HasParameterBlock(sensorFromRig) && problem.IsParameterBlockConstant(sensorFromRig))
		{
			return true;
		}

		// Cover corner case when ReprojErrorConstantPoseCostFunctor is used.
		return ConfigInternal.HasConstantSensorFromRigPose(sensorId) || !OptionsInternal.RefineSensorFromRig;
	}

	// Note that the following implementation does not handle all degenerate edge cases well,
	// e.g., where the selected two cameras are not well constrained with respect to each
	// other with shared observations. Furthermore, the implementation could be more
	// sophisticated for multi-camera rigs by selecting camera pairs within a rig, etc.
	private void FixGaugeWithTwoCamsFromWorld(Reconstruction reconstruction)
	{
		// No need to fix the Gauge if all frames are constant.
		if (!OptionsInternal.RefineRigFromWorld)
		{
			return;
		}

		Image? image1 = null;
		Image? image2 = null;

		// First, search through the already fixed cameras in the problem.
		foreach (uint imageId in parameterizedImageIds)
		{
			Image image = reconstruction.Image(imageId);
			if (ConfigInternal.HasConstantRigFromWorldPose(image.FrameId) && IsParameterizedConstSensor(image))
			{
				if (image1 is null)
				{
					image1 = image;
				}
				else if (image1.FrameId != image.FrameId)
				{
					// No need to fix the Gauge if two frames are already fixed.
					return;
				}
			}
		}

		// Otherwise, search through the variable cameras in the problem.
		int frame2FromWorldFixedDim = 0;
		foreach (uint imageId in parameterizedImageIds)
		{
			Image image = reconstruction.Image(imageId);
			double[] rigFromWorldParams = image.FramePtr.RigFromWorldStorage.Params;
			if (image1 is null && IsParameterizedConstSensor(image))
			{
				image1 = image;
			}
			else if (image1 is not null
				&& image1.FrameId != image.FrameId
				&& IsParameterizedConstSensor(image)
				&& problem.HasParameterBlock(rigFromWorldParams))
			{
				// Check if one of the baseline dimensions is large enough and choose it as the
				// fixed coordinate. If there is no such pair of frames, then the scale is not
				// constrained well.
				Vector3d baseline = (image1.FramePtr.RigFromWorld() * image.FramePtr.RigFromWorld().Inverse()).Translation;
				Vector3d absBaseline = baseline.CwiseAbs();
				int maxCoeffIdx = 0;
				for (int i = 1; i < 3; i++)
				{
					if (absBaseline[i] > absBaseline[maxCoeffIdx])
					{
						maxCoeffIdx = i;
					}
				}

				if (absBaseline[maxCoeffIdx] > 1e-9)
				{
					image2 = image;
					frame2FromWorldFixedDim = maxCoeffIdx;
					break;
				}
			}
		}

		// TODO(jsch): Notice that we could alternatively fall back to fixing the Gauge between
		// two cameras in the same frame or in different frames. Since there are many different
		// combinations to iterate through, we instead fall back to fixing the Gauge with three
		// points for simplicity. Furthermore, once we support IMUs or other sensors, we should
		// fix the Gauge differently.
		if (image1 is null || image2 is null)
		{
			Log.Warning("Failed to fix Gauge with two cameras. Falling back to fixing Gauge with three points.");
			FixGaugeWithThreePoints(reconstruction);
			return;
		}

		if (!ConfigInternal.HasConstantRigFromWorldPose(image1.FrameId))
		{
			problem.SetParameterBlockConstant(image1.FramePtr.RigFromWorldStorage.Params);
		}

		if (!ConfigInternal.HasConstantRigFromWorldPose(image2.FrameId))
		{
			double[] frame2FromWorld = image2.FramePtr.RigFromWorldStorage.Params;
			if (OptionsInternal.ConstantRigFromWorldRotation)
			{
				ManifoldHelpers.SetManifold(
					problem, frame2FromWorld, ManifoldHelpers.CreateSubsetManifold(7, [0, 1, 2, 3, 4 + frame2FromWorldFixedDim]));
			}
			else
			{
				ManifoldHelpers.SetManifold(
					problem,
					frame2FromWorld,
					ManifoldHelpers.CreateProductManifold(
						ManifoldHelpers.CreateEigenQuaternionManifold(),
						ManifoldHelpers.CreateSubsetManifold(3, [frame2FromWorldFixedDim])));
			}
		}
	}

	private sealed class FixedGaugeWithThreePoints
	{
		// The coordinates of the fixed points as columns.
		private readonly MatrixXd fixedPoints = new(3, 3);

		// The number of fixed points for the Gauge.
		public int NumFixedPoints { get; private set; }

		public bool MaybeAddFixedPoint(Vector3d point)
		{
			if (NumFixedPoints >= 3)
			{
				return false;
			}

			SetColumn(NumFixedPoints, point);
			if (new ColPivHouseholderQR(fixedPoints).Rank() > NumFixedPoints)
			{
				++NumFixedPoints;
				return true;
			}

			SetColumn(NumFixedPoints, default);
			return false;
		}

		private void SetColumn(int col, Vector3d value)
		{
			fixedPoints[0, col] = value.X;
			fixedPoints[1, col] = value.Y;
			fixedPoints[2, col] = value.Z;
		}
	}
}
