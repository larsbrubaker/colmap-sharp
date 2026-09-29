// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingProblem, state half: ComputeResiduals, UpdateState, AverageStepSize and
// ApplyResultsToReconstruction of colmap/estimators/rotation_averaging_impl.cc. Setup and the
// constraint matrix are in RotationAveragingProblem.cs (see its header for the translation
// notes); RotationAveragingSolver.cs drives these per iteration.
//
// Translation notes:
// - std::remainder is Math.IEEERemainder (both round the quotient to nearest, ties to even).
// - Pair constraints (and so the jitter draws of the 1-DOF residual) are visited in
//   ascending pair id order, frames and cameras in ascending id order
//   (divergence 44).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

public sealed partial class RotationAveragingProblem
{
	/// <summary>
	/// Computes the 1-DOF residual for gravity-aligned rotation constraints:
	/// (angle_2 - angle_1) - angle_12, wrapped to [-pi, pi] with jitter near the boundaries
	/// to avoid local minima.
	/// </summary>
	private static double ComputeGravityAligned1DofResidual(double angle12, double angle1, double angle2)
	{
		double residual = Math.IEEERemainder((angle2 - angle1) - angle12, 2 * Math.PI);

		// Inject random noise if the angle is too close to the boundary to break the possible
		// balance at the local minima.
		const double kEps = 0.01;
		if (Math.Abs(residual) > Math.PI - kEps)
		{
			double jitter = RandomUtils.RandomUniformReal(0.0, kEps);
			if (residual < 0)
			{
				residual += jitter;
			}
			else
			{
				residual -= jitter;
			}
		}

		return residual;
	}

	/// <summary>Computes the residual vector b from the current rotation estimates.</summary>
	public void ComputeResiduals()
	{
		// Set PRNG seed for deterministic jitter injection.
		if (_options.RandomSeed >= 0)
		{
			RandomUtils.SetPRNGSeed((uint)_options.RandomSeed);
		}

		foreach (PairConstraint constraint in _pairConstraints.Values)
		{
			uint frameId1 = _imageIdToFrameId[constraint.ImageId1];
			uint frameId2 = _imageIdToFrameId[constraint.ImageId2];
			int frameParamIdx1 = _frameIdToParamIdx[frameId1];
			int frameParamIdx2 = _frameIdToParamIdx[frameId2];

			switch (constraint.Constraint)
			{
				case GravityAligned1Dof constraint1Dof:
					// 1-DOF case: compute Y-axis angle residual.
					_residuals[constraint.RowIndex] = ComputeGravityAligned1DofResidual(
						constraint1Dof.AngleCam2FromCam1,
						_estimatedRotations[frameParamIdx1],
						_estimatedRotations[frameParamIdx2]);
					break;

				case Full3Dof full:
				{
					// 3-DOF case: compute full rotation error.
					Matrix3d estimatedCam1FromWorld = EstimatedFrameRotation(frameId1, frameParamIdx1);
					Matrix3d estimatedCam2FromWorld = EstimatedFrameRotation(frameId2, frameParamIdx2);

					if (constraint.Cam1FromRigParamIdx != -1)
					{
						estimatedCam1FromWorld = Pose.AngleAxisToRotationMatrix(
							Segment3(_estimatedRotations, constraint.Cam1FromRigParamIdx)) * estimatedCam1FromWorld;
					}

					if (constraint.Cam2FromRigParamIdx != -1)
					{
						estimatedCam2FromWorld = Pose.AngleAxisToRotationMatrix(
							Segment3(_estimatedRotations, constraint.Cam2FromRigParamIdx)) * estimatedCam2FromWorld;
					}

					SetSegment3(_residuals, constraint.RowIndex, -Pose.RotationMatrixToAngleAxis(
						estimatedCam2FromWorld.Transpose() * full.RCam2FromCam1 * estimatedCam1FromWorld));
					break;
				}

				default:
					throw new InvalidOperationException("Unknown constraint type");
			}
		}

		// Fixed frame residual.
		int fixedFrameParamIdx = _frameIdToParamIdx[_fixedFrameId];
		if (_numGaugeFixingResiduals == 1)
		{
			_residuals[_residuals.Length - 1] = _estimatedRotations[fixedFrameParamIdx] - _fixedFrameRotation.Y;
		}
		else
		{
			SetSegment3(_residuals, _residuals.Length - 3, Pose.RotationMatrixToAngleAxis(
				Pose.AngleAxisToRotationMatrix(_fixedFrameRotation).Transpose()
				* Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, fixedFrameParamIdx))));
		}
	}

	// The estimated rig_from_world rotation of a frame as used by the 3-DOF residual (gravity
	// frames through their Y angle only when gravity mode is on).
	private Matrix3d EstimatedFrameRotation(uint frameId, int frameParamIdx)
	{
		if (_options.UseGravity && GetFrameGravityOrNull(frameId) is not null)
		{
			return Pose.RotationFromYAxisAngle(_estimatedRotations[frameParamIdx]);
		}

		return Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, frameParamIdx));
	}

	/// <summary>Updates the rotation estimates by applying the solution step.</summary>
	public void UpdateState(VectorXd step)
	{
		// Update frame rotations.
		foreach ((uint frameId, int frameParamIdx) in _frameIdToParamIdx)
		{
			if (!HasFrameGravity(frameId))
			{
				Matrix3d estimatedRigFromWorld = Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, frameParamIdx));
				SetSegment3(_estimatedRotations, frameParamIdx, Pose.RotationMatrixToAngleAxis(
					estimatedRigFromWorld * Pose.AngleAxisToRotationMatrix(-Segment3(step, frameParamIdx))));
			}
			else
			{
				_estimatedRotations[frameParamIdx] -= step[frameParamIdx];
			}
		}

		if (_cameraIdToParamIdx.Count == 0)
		{
			return;
		}

		// Compute current frame rotations for cam_from_rig averaging.
		var frameRotations = new Dictionary<uint, Matrix3d>(_frameIdToParamIdx.Count);
		foreach ((uint frameId, int frameParamIdx) in _frameIdToParamIdx)
		{
			frameRotations[frameId] = HasFrameGravity(frameId)
				? Pose.RotationFromYAxisAngle(_estimatedRotations[frameParamIdx])
				: Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, frameParamIdx));
		}

		// Update the global rotations for cam_from_rig cameras.
		// Note: the update is non-trivial, and we need to average the rotations from all the
		// frames.
		foreach ((uint cameraId, int cameraParamIdx) in _cameraIdToParamIdx)
		{
			Matrix3d estimatedCamFromRig = Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, cameraParamIdx));
			Matrix3d rUpdate = Pose.AngleAxisToRotationMatrix(-Segment3(step, cameraParamIdx));

			// COLMAP's camera_to_frame_ids_[camera_id] default-constructs an empty list for a
			// camera without active frames; its average is then taken over no samples.
			List<uint> frameIds = _cameraToFrameIds.TryGetValue(cameraId, out List<uint>? ids) ? ids : [];
			var rigRotations = new List<Quaterniond>(frameIds.Count);
			foreach (uint frameId in frameIds)
			{
				Matrix3d r = frameRotations[frameId];
				rigRotations.Add(Quaterniond.FromRotationMatrix(estimatedCamFromRig * r * rUpdate * r.Transpose()));
			}

			// Average the rotations for the rig.
			Quaterniond rAve = Pose.AverageQuaternions(rigRotations, Enumerable.Repeat(1.0, rigRotations.Count).ToArray());
			SetSegment3(_estimatedRotations, cameraParamIdx, Pose.RotationMatrixToAngleAxis(rAve.ToRotationMatrix()));
		}
	}

	/// <summary>Returns the average rotation step size for convergence checking.</summary>
	public double AverageStepSize(VectorXd step)
	{
		double totalUpdate = 0;
		foreach ((uint frameId, int frameParamIdx) in _frameIdToParamIdx)
		{
			totalUpdate += HasFrameGravity(frameId)
				? Math.Abs(step[frameParamIdx])
				: Segment3(step, frameParamIdx).Norm;
		}

		return totalUpdate / _frameIdToParamIdx.Count;
	}

	/// <summary>
	/// Writes the optimized rotations back to the reconstruction: every problem frame gets a
	/// rig_from_world with unknown (NaN) translation, and, with RefineSensorFromRig, every
	/// estimated camera a cam_from_rig with unknown translation.
	/// </summary>
	public void ApplyResultsToReconstruction(Reconstruction reconstruction)
	{
		var kUnknownTranslation = new Vector3d(double.NaN, double.NaN, double.NaN);

		foreach ((uint frameId, int frameParamIdx) in _frameIdToParamIdx)
		{
			Matrix3d rotation = HasFrameGravity(frameId)
				? Pose.GravityAlignedRotation(GetFrameGravityOrNull(frameId)!.Value)
					* Pose.RotationFromYAxisAngle(_estimatedRotations[frameParamIdx])
				: Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, frameParamIdx));
			reconstruction.Frame(frameId).SetRigFromWorld(
				new Rigid3d(Quaterniond.FromRotationMatrix(rotation), kUnknownTranslation));
		}

		if (!_options.RefineSensorFromRig)
		{
			return;
		}

		foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
		{
			// Materialized: SetSensorFromRig writes into the collection being enumerated.
			foreach (SensorId sensorId in rig.NonRefSensors.Keys.ToList())
			{
				if (!_cameraIdToParamIdx.TryGetValue(sensorId.Id, out int paramIdx))
				{
					continue; // Skip cameras that are not estimated.
				}

				var camFromRig = new Rigid3d(
					Quaterniond.FromRotationMatrix(Pose.AngleAxisToRotationMatrix(Segment3(_estimatedRotations, paramIdx))),
					kUnknownTranslation); // No translation yet.
				reconstruction.Rig(rigId).SetSensorFromRig(sensorId, camFromRig);
			}
		}
	}
}
