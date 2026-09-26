// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingProblem: colmap/estimators/rotation_averaging_impl.h and .cc, the linear
// system A x = b of global rotation averaging (x: rig_from_world rotations of the active
// frames plus unknown cam_from_rig rotations, in tangent space; b: residuals of the relative
// rotation constraints of the pose graph's valid edges; A: the sparse constraint matrix with
// one gauge-fixing block). Split by responsibility:
// - this file: constraint types, setup (parameter allocation, pair constraints, the
//   constraint matrix and the optional INLIER_MATCH_COUNT reweighting) and accessors;
// - RotationAveragingProblem.State.cs: residuals, state update, step size and writing the
//   result back to the Reconstruction.
// RotationAveragingSolver.cs solves it (L1 then IRLS); the options are in
// RotationEstimatorOptions.cs. Tests: ColmapSharp.Tests/Estimators/RotationAveragingTests.cs.
//
// Tier C (iterative, with a random jitter near the +-pi boundary of the 1-DOF residual).
//
// Translation notes:
// - COLMAP's NodeHashMap/FlatHashSet iteration decides the parameter layout, the row order,
//   the order of the jitter draws and which frame fixes the gauge. Here active frames and
//   estimated cameras are laid out in ascending id order, pair constraints in ascending pair
//   id order, and the gauge is fixed at the smallest-id (gravity-aligned, if any) frame
//   (docs/CPP_DIVERGENCES.md, entry 44).
// - COLMAP looks up a pair's frame parameter with operator[], which silently inserts index 0
//   for a frame outside the active set (and ComputeResiduals then throws from .at()). Callers
//   invalidate such pairs first (PoseGraph.InvalidatePairsOutsideActiveImageIds); here the
//   constructor throws KeyNotFoundException for them instead of corrupting the layout.
// - std::variant<GravityAligned1DOF, Full3DOF> is the RotationConstraint record hierarchy.
// - VLOG messages are dropped, like LOG(INFO) everywhere in the library.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Rotation averaging problem formulated as linear system A*x = b where
/// x = [rig_from_world rotations, unknown cam_from_rig rotations],
/// b = residuals from relative rotation constraints and
/// A = sparse matrix encoding constraint equations.
/// Port of colmap::RotationAveragingProblem.
/// </summary>
public sealed partial class RotationAveragingProblem
{
	/// <summary>A relative rotation constraint (std::variant of the two kinds below).</summary>
	public abstract record RotationConstraint;

	/// <summary>1-DOF constraint when both frames have gravity priors.</summary>
	/// <param name="AngleCam2FromCam1">Relative Y-axis rotation.</param>
	/// <param name="XzError">Squared error in x,z axes for IRLS.</param>
	public sealed record GravityAligned1Dof(double AngleCam2FromCam1, double XzError) : RotationConstraint;

	/// <summary>3-DOF constraint for the general case (no gravity or partial gravity).</summary>
	public sealed record Full3Dof(Matrix3d RCam2FromCam1) : RotationConstraint;

	/// <summary>Preprocessed constraint for an image pair, built once during setup.</summary>
	public sealed class PairConstraint
	{
		/// <summary>First image of the pair.</summary>
		public uint ImageId1 { get; internal set; } = Types.InvalidImageId;

		/// <summary>Second image of the pair.</summary>
		public uint ImageId2 { get; internal set; } = Types.InvalidImageId;

		/// <summary>Starting row in matrix A (1 row for 1-DOF, 3 rows for 3-DOF).</summary>
		public int RowIndex { get; internal set; } = -1;

		/// <summary>Column index of the unknown cam1_from_rig rotation (-1 if known).</summary>
		public int Cam1FromRigParamIdx { get; internal set; } = -1;

		/// <summary>Column index of the unknown cam2_from_rig rotation (-1 if known).</summary>
		public int Cam2FromRigParamIdx { get; internal set; } = -1;

		/// <summary>The 1-DOF or 3-DOF constraint.</summary>
		public RotationConstraint Constraint { get; internal set; } = null!;
	}

	private readonly RotationEstimatorOptions _options;

	// Pose priors indexed by frame ID.
	private readonly Dictionary<uint, PosePrior> _frameToPosePrior;

	// Linear system components: matrix A and vector b.
	private SparseMatrixCsc _constraintMatrix = SparseMatrixCsc.Zero(0, 0);
	private VectorXd _residuals = new(0);

	// Optional reweighting operator W applied to the residual space (rows of A and b); the
	// solver works on the reweighted system min ||W (A x - b)||. Populated when the
	// reweighting scheme is not Uniform. Stored as the diagonal of W (one weight per row).
	private VectorXd? _residualReweighting;

	// Current rotation estimates in tangent space (angle-axis).
	private VectorXd _estimatedRotations = new(0);

	// Parameter index mappings (inserted in ascending id order, never removed, so they
	// enumerate in ascending id order).
	private readonly Dictionary<uint, int> _frameIdToParamIdx = [];
	private readonly Dictionary<uint, int> _cameraIdToParamIdx = [];

	// Preprocessed constraints for each image pair, inserted in ascending pair id order.
	private readonly Dictionary<ulong, PairConstraint> _pairConstraints = [];

	// Gauge fixing (removes rotational ambiguity).
	private uint _fixedFrameId = Types.InvalidFrameId;
	private Vector3d _fixedFrameRotation;
	private int _numGaugeFixingResiduals = 3; // 1 for gravity-aligned, 3 otherwise.

	// Cached lookups for ComputeResiduals and UpdateState.
	private readonly Dictionary<uint, uint> _imageIdToFrameId = [];
	private readonly SortedDictionary<uint, uint> _cameraIdToRigId = [];
	private readonly Dictionary<uint, List<uint>> _cameraToFrameIds = [];

	// Active frames for the current solve, ascending.
	private readonly SortedSet<uint> _activeFrameIds = [];

	/// <summary>
	/// Builds the problem over the valid edges of <paramref name="poseGraph"/> between
	/// <paramref name="activeImageIds"/>, initializing the unknowns from the reconstruction's
	/// current rig_from_world and cam_from_rig rotations.
	/// </summary>
	public RotationAveragingProblem(
		PoseGraph poseGraph,
		IReadOnlyList<PosePrior> posePriors,
		RotationEstimatorOptions options,
		IReadOnlySet<uint> activeImageIds,
		Reconstruction reconstruction)
	{
		_options = options.Clone();

		// Derive active frame ids from the active image ids, and cache mappings.
		foreach (uint imageId in activeImageIds)
		{
			Image image = reconstruction.Image(imageId);
			uint frameId = image.FrameId;
			_activeFrameIds.Add(frameId);
			_imageIdToFrameId[imageId] = frameId;
			_cameraIdToRigId[image.CameraId] = image.FramePtr.RigId;
		}

		_frameToPosePrior = ExtractFrameToPosePrior(reconstruction, posePriors);

		int numParams = AllocateParameters(reconstruction);
		BuildPairConstraints(poseGraph, reconstruction);
		BuildConstraintMatrix(numParams, poseGraph, reconstruction);
	}

	/// <summary>The constraint matrix A.</summary>
	public SparseMatrixCsc ConstraintMatrix => _constraintMatrix;

	/// <summary>The residual vector b.</summary>
	public VectorXd Residuals => _residuals;

	/// <summary>
	/// Diagonal of the residual-space reweighting operator W (one weight per residual row),
	/// or null when no reweighting is configured.
	/// </summary>
	public VectorXd? ResidualReweighting => _residualReweighting;

	/// <summary>Number of unknowns (columns of A).</summary>
	public int NumParameters => _constraintMatrix.Cols;

	/// <summary>Number of residual rows of A.</summary>
	public int NumResiduals => _constraintMatrix.Rows;

	/// <summary>Rows of the gauge-fixing block at the end of A (1 or 3).</summary>
	public int NumGaugeFixingResiduals => _numGaugeFixingResiduals;

	/// <summary>The pair constraints by pair id, in ascending pair id order.</summary>
	public IReadOnlyDictionary<ulong, PairConstraint> PairConstraints => _pairConstraints;

	/// <summary>
	/// Constraint matrix A with the reweighting operator applied to its rows (W * A), or the
	/// plain constraint matrix when no reweighting is configured.
	/// </summary>
	public SparseMatrixCsc WeightedConstraintMatrix()
	{
		if (_residualReweighting is null)
		{
			return _constraintMatrix;
		}

		SparseMatrixCsc weighted = _constraintMatrix.Clone();
		ReadOnlySpan<int> rows = weighted.RowIndices;
		Span<double> values = weighted.Values;
		for (int p = 0; p < values.Length; p++)
		{
			values[p] = _residualReweighting[rows[p]] * values[p];
		}

		return weighted;
	}

	/// <summary>
	/// Residual vector b with the reweighting operator applied (W * b), or the plain
	/// residuals when no reweighting is configured.
	/// </summary>
	public VectorXd WeightedResiduals()
	{
		if (_residualReweighting is null)
		{
			return _residuals;
		}

		var weighted = new VectorXd(_residuals.Length);
		for (int i = 0; i < weighted.Length; i++)
		{
			weighted[i] = _residualReweighting[i] * _residuals[i];
		}

		return weighted;
	}

	private static Dictionary<uint, PosePrior> ExtractFrameToPosePrior(
		Reconstruction reconstruction, IReadOnlyList<PosePrior> posePriors)
	{
		var frameToPosePrior = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in posePriors)
		{
			if (posePrior.CorrDataId.SensorId.Type != SensorType.Camera)
			{
				continue;
			}

			uint imageId = (uint)posePrior.CorrDataId.Id;
			if (!reconstruction.Images.TryGetValue(imageId, out Image? image))
			{
				continue;
			}

			if (image.IsRefInFrame)
			{
				uint frameId = image.FrameId;
				Check.That(frameToPosePrior.TryAdd(frameId, posePrior), $"Duplicate pose prior for frame{frameId}");
			}
		}

		return frameToPosePrior;
	}

	private Vector3d? GetFrameGravityOrNull(uint frameId)
	{
		if (!_frameToPosePrior.TryGetValue(frameId, out PosePrior prior) || !prior.HasGravity())
		{
			return null;
		}

		return prior.Gravity;
	}

	/// <summary>True if the frame has a gravity prior and gravity mode is enabled.</summary>
	private bool HasFrameGravity(uint frameId) => _options.UseGravity && GetFrameGravityOrNull(frameId) is not null;

	private static bool HasNaN(Vector3d v) => double.IsNaN(v.X) || double.IsNaN(v.Y) || double.IsNaN(v.Z);

	private static Vector3d AngleAxisVector(Quaterniond rotation)
	{
		AngleAxisd angleAxis = AngleAxisd.FromQuaternion(rotation);
		return angleAxis.Angle * angleAxis.Axis;
	}

	private static Vector3d Segment3(VectorXd v, int start) => new(v[start], v[start + 1], v[start + 2]);

	private static void SetSegment3(VectorXd v, int start, Vector3d value)
	{
		v[start] = value.X;
		v[start + 1] = value.Y;
		v[start + 2] = value.Z;
	}

	/// <summary>Allocates parameter indices for frames and cameras, initializes rotations.</summary>
	private int AllocateParameters(Reconstruction reconstruction)
	{
		// Identify cameras that need cam_from_rig estimation (non-reference cameras without
		// calibrated extrinsics).
		var camFromRigRotations = new Dictionary<uint, Vector3d>();
		var estimatedCameraIds = new SortedSet<uint>();
		if (_options.RefineSensorFromRig)
		{
			foreach ((uint cameraId, uint rigId) in _cameraIdToRigId)
			{
				var sensorId = new SensorId(SensorType.Camera, cameraId);
				if (reconstruction.Rig(rigId).IsRefSensor(sensorId))
				{
					continue;
				}

				Rigid3d? camFromRig = reconstruction.Rig(rigId).MaybeSensorFromRig(sensorId);
				if (camFromRig is null || HasNaN(camFromRig.Value.Translation))
				{
					if (estimatedCameraIds.Add(cameraId) && camFromRig is not null)
					{
						camFromRigRotations[cameraId] = AngleAxisVector(camFromRig.Value.Rotation);
					}
				}
			}
		}

		// Cache camera_id -> frame_id mapping for UpdateState cam_from_rig averaging.
		foreach (uint frameId in _activeFrameIds)
		{
			foreach (DataId dataId in reconstruction.Frame(frameId).ImageIds())
			{
				uint cameraId = reconstruction.Image((uint)dataId.Id).CameraId;
				if (estimatedCameraIds.Contains(cameraId))
				{
					if (!_cameraToFrameIds.TryGetValue(cameraId, out List<uint>? frameIds))
					{
						frameIds = [];
						_cameraToFrameIds[cameraId] = frameIds;
					}

					frameIds.Add(frameId);
				}
			}
		}

		// Allocate frame parameters and cache frame info.
		var rotations = new List<double>(3 * (_activeFrameIds.Count + estimatedCameraIds.Count));
		foreach (uint frameId in _activeFrameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			_frameIdToParamIdx[frameId] = rotations.Count;

			if (HasFrameGravity(frameId))
			{
				// Gravity-aligned frame: 1-DOF (Y-axis rotation only).
				Matrix3d rigFromWorldRotation = frame.MaybeRigFromWorld is Rigid3d rigFromWorld
					? rigFromWorld.Rotation.ToRotationMatrix()
					: Matrix3d.Identity;
				double angle = Pose.YAxisAngleFromRotation(
					Pose.GravityAlignedRotation(GetFrameGravityOrNull(frameId)!.Value).Transpose() * rigFromWorldRotation);
				rotations.Add(angle);

				// Use first gravity-aligned frame as fixed frame.
				if (_fixedFrameId == Types.InvalidFrameId)
				{
					_fixedFrameRotation = new Vector3d(0, angle, 0);
					_fixedFrameId = frameId;
					_numGaugeFixingResiduals = 1;
				}
			}
			else
			{
				// General frame: 3-DOF.
				Vector3d rigFromWorld = frame.MaybeRigFromWorld is Rigid3d pose
					? AngleAxisVector(pose.Rotation)
					: Vector3d.Zero;
				rotations.Add(rigFromWorld.X);
				rotations.Add(rigFromWorld.Y);
				rotations.Add(rigFromWorld.Z);
			}
		}

		// Allocate camera parameters (for unknown cam_from_rig rotations).
		foreach (uint cameraId in estimatedCameraIds)
		{
			_cameraIdToParamIdx[cameraId] = rotations.Count;
			Vector3d camFromRig = camFromRigRotations.TryGetValue(cameraId, out Vector3d r) ? r : Vector3d.Zero;
			rotations.Add(camFromRig.X);
			rotations.Add(camFromRig.Y);
			rotations.Add(camFromRig.Z);
		}

		// If no gravity-aligned frame found, use first active frame as fixed.
		if (_fixedFrameId == Types.InvalidFrameId && _activeFrameIds.Count > 0)
		{
			uint frameId = _activeFrameIds.Min;
			Frame frame = reconstruction.Frame(frameId);
			_fixedFrameId = frameId;

			// Use identity rotation if frame doesn't have a pose yet.
			_fixedFrameRotation = frame.HasPose ? AngleAxisVector(frame.RigFromWorld().Rotation) : Vector3d.Zero;
			_numGaugeFixingResiduals = 3;
		}

		_estimatedRotations = new VectorXd(rotations.ToArray());
		return rotations.Count;
	}

	/// <summary>Builds the PairConstraint of each valid image pair.</summary>
	private void BuildPairConstraints(PoseGraph poseGraph, Reconstruction reconstruction)
	{
		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.ValidEdges().OrderBy(kv => kv.Key))
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			Image image1 = reconstruction.Image(imageId1);
			Image image2 = reconstruction.Image(imageId2);
			Frame frame1 = image1.FramePtr;
			Frame frame2 = image2.FramePtr;

			int frameParamIdx1 = _frameIdToParamIdx[frame1.FrameId];
			int frameParamIdx2 = _frameIdToParamIdx[frame2.FrameId];

			// Get known cam_from_rig transforms (null for reference cameras or cameras with
			// unknown cam_from_rig that need to be estimated).
			Rigid3d? cam1FromRig1 = null;
			Rigid3d? cam2FromRig2 = null;
			if (!image1.IsRefInFrame && !_cameraIdToParamIdx.ContainsKey(image1.CameraId))
			{
				cam1FromRig1 = reconstruction.Rig(frame1.RigId).SensorFromRig(image1.CameraPtr.SensorId);
			}

			if (!image2.IsRefInFrame && !_cameraIdToParamIdx.ContainsKey(image2.CameraId))
			{
				cam2FromRig2 = reconstruction.Rig(frame2.RigId).SensorFromRig(image2.CameraPtr.SensorId);
			}

			// Skip self-loops within the same frame when both cam_from_rig are known.
			if (cam1FromRig1 is not null && cam2FromRig2 is not null && frameParamIdx1 == frameParamIdx2)
			{
				continue;
			}

			// Compute relative rotation between rigs.
			Matrix3d rCam2FromCam1 = ((cam2FromRig2 ?? new Rigid3d()).Rotation.Inverse()
				* edge.Cam2FromCam1.Rotation
				* (cam1FromRig1 ?? new Rigid3d()).Rotation).ToRotationMatrix();

			Vector3d? frameGravity1 = GetFrameGravityOrNull(frame1.FrameId);
			Vector3d? frameGravity2 = GetFrameGravityOrNull(frame2.FrameId);

			// Apply gravity alignment transformations if available.
			if (_options.UseGravity)
			{
				if (frameGravity1 is Vector3d gravity1)
				{
					rCam2FromCam1 = rCam2FromCam1 * Pose.GravityAlignedRotation(gravity1);
				}

				if (frameGravity2 is Vector3d gravity2)
				{
					rCam2FromCam1 = Pose.GravityAlignedRotation(gravity2).Transpose() * rCam2FromCam1;
				}
			}

			// Create constraint based on gravity availability.
			var constraint = new PairConstraint { ImageId1 = imageId1, ImageId2 = imageId2 };
			if (_options.UseGravity && frameGravity1 is not null && frameGravity2 is not null)
			{
				// Both frames have gravity: use 1-DOF constraint.
				Vector3d aa = Pose.RotationMatrixToAngleAxis(rCam2FromCam1);
				constraint.Constraint = new GravityAligned1Dof(aa.Y, aa.X * aa.X + aa.Z * aa.Z);
			}
			else
			{
				// General case: use 3-DOF constraint.
				constraint.Constraint = new Full3Dof(rCam2FromCam1);
			}

			_pairConstraints[pairId] = constraint;
		}
	}

	/// <summary>Builds the sparse matrix A and the residual-space reweighting operator W.</summary>
	private void BuildConstraintMatrix(int numParams, PoseGraph poseGraph, Reconstruction reconstruction)
	{
		if (numParams == 0)
		{
			return;
		}

		var coeffs = new List<SparseTriplet>();
		int currRow = 0;

		foreach ((ulong pairId, PairConstraint constraint) in _pairConstraints)
		{
			Image image1 = reconstruction.Image(constraint.ImageId1);
			Image image2 = reconstruction.Image(constraint.ImageId2);
			if (!_activeFrameIds.Contains(image1.FrameId) || !_activeFrameIds.Contains(image2.FrameId))
			{
				continue;
			}

			int frameParamIdx1 = _frameIdToParamIdx[image1.FrameId];
			int frameParamIdx2 = _frameIdToParamIdx[image2.FrameId];

			// Look up camera parameter indices (-1 if cam_from_rig is known).
			int cam1ParamIdx = _cameraIdToParamIdx.TryGetValue(image1.CameraId, out int idx1) ? idx1 : -1;
			int cam2ParamIdx = _cameraIdToParamIdx.TryGetValue(image2.CameraId, out int idx2) ? idx2 : -1;

			constraint.RowIndex = currRow;
			constraint.Cam1FromRigParamIdx = cam1ParamIdx;
			constraint.Cam2FromRigParamIdx = cam2ParamIdx;

			if (constraint.Constraint is GravityAligned1Dof)
			{
				// 1-DOF constraint: single row.
				coeffs.Add(new SparseTriplet(currRow, frameParamIdx1, -1));
				coeffs.Add(new SparseTriplet(currRow, frameParamIdx2, 1));
				currRow++;
				continue;
			}

			// 3-DOF constraint: three rows.
			if (!_options.UseGravity || GetFrameGravityOrNull(image1.FrameId) is null)
			{
				for (int i = 0; i < 3; i++)
				{
					coeffs.Add(new SparseTriplet(currRow + i, frameParamIdx1 + i, -1));
				}
			}
			else
			{
				// Gravity-aligned frame1: only Y-axis contributes.
				coeffs.Add(new SparseTriplet(currRow + 1, frameParamIdx1, -1));
			}

			if (!_options.UseGravity || GetFrameGravityOrNull(image2.FrameId) is null)
			{
				for (int i = 0; i < 3; i++)
				{
					coeffs.Add(new SparseTriplet(currRow + i, frameParamIdx2 + i, 1));
				}
			}
			else
			{
				// Gravity-aligned frame2: only Y-axis contributes.
				coeffs.Add(new SparseTriplet(currRow + 1, frameParamIdx2, 1));
			}

			// Add cam_from_rig terms if being estimated.
			if (cam1ParamIdx != -1)
			{
				for (int i = 0; i < 3; i++)
				{
					coeffs.Add(new SparseTriplet(currRow + i, cam1ParamIdx + i, -1));
				}
			}

			if (cam2ParamIdx != -1)
			{
				for (int i = 0; i < 3; i++)
				{
					coeffs.Add(new SparseTriplet(currRow + i, cam2ParamIdx + i, 1));
				}
			}

			currRow += 3;
		}

		// Add gauge-fixing constraint for the fixed frame.
		int fixedFrameParamIdx = _frameIdToParamIdx[_fixedFrameId];
		for (int i = 0; i < _numGaugeFixingResiduals; i++)
		{
			coeffs.Add(new SparseTriplet(currRow + i, fixedFrameParamIdx + i, 1));
		}

		currRow += _numGaugeFixingResiduals;

		_constraintMatrix = SparseMatrixCsc.FromTriplets(currRow, numParams, coeffs);
		_residuals = new VectorXd(currRow);

		// Optionally build the residual-space reweighting operator W. Weights are normalized to
		// (0, 1] for numerical stability (a global scale leaves the minimizer unchanged). Only
		// the rows owned by a pair constraint are reweighted; the gauge-fixing rows (which are
		// not part of the pair constraints) retain their default weight of 1.
		if (_options.Reweighting == RotationAveragingReweighting.InlierMatchCount)
		{
			double maxNumMatches = 0;
			foreach (ulong pairId in _pairConstraints.Keys)
			{
				maxNumMatches = Math.Max(maxNumMatches, poseGraph.Edges[pairId].NumMatches);
			}

			// The inlier match count (normalized to (0, 1]) is used directly as the diagonal
			// reweighting entry.
			VectorXd reweightingDiagonal = VectorXd.Ones(currRow);
			if (maxNumMatches > 0)
			{
				foreach ((ulong pairId, PairConstraint constraint) in _pairConstraints)
				{
					double reweighting = poseGraph.Edges[pairId].NumMatches / maxNumMatches;
					int rows = constraint.Constraint is GravityAligned1Dof ? 1 : 3;
					for (int i = 0; i < rows; i++)
					{
						reweightingDiagonal[constraint.RowIndex + i] = reweighting;
					}
				}
			}

			_residualReweighting = reweightingDiagonal;
		}
	}
}
