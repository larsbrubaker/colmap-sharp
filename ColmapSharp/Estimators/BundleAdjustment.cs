// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustment: colmap/estimators/bundle_adjustment.h and .cc, the solver-agnostic half
// of bundle adjustment: the gauge / termination / backend enums, BundleAdjustmentSummary,
// BundleAdjustmentConfig (which images, points, cameras and poses take part and which are
// held constant), BundleAdjustmentOptions, the abstract BundleAdjuster and the
// CreateDefaultBundleAdjuster factory. The Ceres backend (the only one ported; Caspar is a
// GPU backend and out of scope) is BundleAdjustmentCeres.cs next to this file, built on
// Solver/ (the Ceres replacement) and Estimators/CostFunctions/.
// Tests: ColmapSharp.Tests/Estimators/BundleAdjustmentTests.cs (bundle_adjustment_test.cc).
//
// Tier A for the config bookkeeping; the solve itself is Tier C (see BundleAdjustmentCeres.cs).
//
// Translation notes:
// - COLMAP's FlatHashSet members become HashSet exposed as IReadOnlySet. Where the Ceres
//   backend iterates them, it sorts first so the problem layout does not depend on hash
//   order (docs/CPP_DIVERGENCES.md entry 39).
// - The backend options are held by shared_ptr in COLMAP with a deep-copying copy
//   constructor; here they are classes and Clone() deep-copies them.
// - check_if_stopped (std::function<bool()>) becomes Func<bool>?; BundleAdjuster.Solve also
//   takes a CancellationToken, which stops the solve the same way (USER_SUCCESS).
// - CreatePosePriorBundleAdjuster is not ported yet: it needs AlignReconstructionToPosePriors
//   (estimators/alignment), which has not been ported. PosePriorBundleAdjustmentOptions is.

using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::BundleAdjustmentGauge.</summary>
public enum BundleAdjustmentGauge
{
	/// <summary>Leave the gauge free.</summary>
	Unspecified = -1,

	/// <summary>Fix one camera pose and one translation coordinate of another.</summary>
	TwoCamsFromWorld = 0,

	/// <summary>Fix three linearly independent 3D points.</summary>
	ThreePoints = 1,
}

/// <summary>Port of colmap::BundleAdjustmentTerminationType.</summary>
public enum BundleAdjustmentTerminationType
{
	/// <summary>Converged.</summary>
	Convergence = 0,

	/// <summary>Hit an iteration or time limit; the solution is usable.</summary>
	NoConvergence = 1,

	/// <summary>Failed; the parameters are unchanged.</summary>
	Failure = 2,

	/// <summary>Stopped by the user (cancellation); the solution is usable.</summary>
	UserSuccess = 3,

	/// <summary>Aborted by the user; the parameters are unchanged.</summary>
	UserFailure = 4,
}

/// <summary>Port of colmap::BundleAdjustmentBackend. Only CERES is available here.</summary>
public enum BundleAdjustmentBackend
{
	/// <summary>The managed Ceres replacement (Solver/).</summary>
	Ceres = 0,

	/// <summary>COLMAP's GPU backend; not available in ColmapSharp.</summary>
	Caspar = 1,
}

/// <summary>Port of colmap::BundleAdjustmentSummary.</summary>
public class BundleAdjustmentSummary
{
	/// <summary>How the solve ended.</summary>
	public BundleAdjustmentTerminationType TerminationType { get; set; } = BundleAdjustmentTerminationType.Failure;

	/// <summary>
	/// Number of residuals connected to at least one variable parameter block. Excludes
	/// residuals where all connected parameters are constant.
	/// </summary>
	public int NumResiduals { get; set; }

	/// <summary>True for CONVERGENCE, NO_CONVERGENCE and USER_SUCCESS.</summary>
	public bool IsSolutionUsable() =>
		TerminationType is BundleAdjustmentTerminationType.Convergence
			or BundleAdjustmentTerminationType.NoConvergence
			or BundleAdjustmentTerminationType.UserSuccess;

	/// <summary>One-line report.</summary>
	public virtual string BriefReport() =>
		$"Bundle adjustment report: termination={TerminationTypeToString(TerminationType)}, num_residuals={NumResiduals}";

	/// <summary>COLMAP's enum name (BundleAdjustmentTerminationTypeToString).</summary>
	public static string TerminationTypeToString(BundleAdjustmentTerminationType type) => type switch
	{
		BundleAdjustmentTerminationType.Convergence => "CONVERGENCE",
		BundleAdjustmentTerminationType.NoConvergence => "NO_CONVERGENCE",
		BundleAdjustmentTerminationType.Failure => "FAILURE",
		BundleAdjustmentTerminationType.UserSuccess => "USER_SUCCESS",
		BundleAdjustmentTerminationType.UserFailure => "USER_FAILURE",
		_ => throw new ArgumentOutOfRangeException(nameof(type)),
	};
}

/// <summary>Port of colmap::BundleAdjustmentConfig: what a bundle adjustment problem contains.</summary>
public sealed class BundleAdjustmentConfig
{
	private readonly HashSet<uint> constantCamIntrinsics = [];
	private readonly HashSet<uint> imageIds = [];
	private readonly HashSet<ulong> variablePoint3DIds = [];
	private readonly HashSet<ulong> constantPoint3DIds = [];
	private readonly HashSet<ulong> ignoredPoint3DIds = [];
	private readonly HashSet<SensorId> constantSensorFromRigPoses = [];
	private readonly HashSet<uint> constantRigFromWorldPoses = [];

	/// <summary>How the gauge is fixed.</summary>
	public BundleAdjustmentGauge FixedGauge { get; private set; } = BundleAdjustmentGauge.Unspecified;

	/// <summary>Sets how the gauge is fixed.</summary>
	public void FixGauge(BundleAdjustmentGauge gauge) => FixedGauge = gauge;

	/// <summary>Number of added images.</summary>
	public int NumImages => imageIds.Count;

	/// <summary>Number of variable plus constant points.</summary>
	public int NumPoints => variablePoint3DIds.Count + constantPoint3DIds.Count;

	/// <summary>Number of variable points.</summary>
	public int NumVariablePoints => variablePoint3DIds.Count;

	/// <summary>Number of constant points.</summary>
	public int NumConstantPoints => constantPoint3DIds.Count;

	/// <summary>Number of cameras with constant intrinsics.</summary>
	public int NumConstantCamIntrinsics => constantCamIntrinsics.Count;

	/// <summary>Number of constant sensor_from_rig poses.</summary>
	public int NumConstantSensorFromRigPoses => constantSensorFromRigPoses.Count;

	/// <summary>Number of constant rig_from_world poses.</summary>
	public int NumConstantRigFromWorldPoses => constantRigFromWorldPoses.Count;

	/// <summary>Added images.</summary>
	public IReadOnlySet<uint> Images => imageIds;

	/// <summary>Variable points.</summary>
	public IReadOnlySet<ulong> VariablePoints => variablePoint3DIds;

	/// <summary>Constant points.</summary>
	public IReadOnlySet<ulong> ConstantPoints => constantPoint3DIds;

	/// <summary>Cameras with constant intrinsics.</summary>
	public IReadOnlySet<uint> ConstantCamIntrinsics => constantCamIntrinsics;

	/// <summary>Sensors with constant sensor_from_rig.</summary>
	public IReadOnlySet<SensorId> ConstantSensorFromRigPoses => constantSensorFromRigPoses;

	/// <summary>Frames with constant rig_from_world.</summary>
	public IReadOnlySet<uint> ConstantRigFromWorldPoses => constantRigFromWorldPoses;

	/// <summary>
	/// The number of residuals for the given reconstruction: the number of observations
	/// (added images, plus the tracks of added points outside them) times two.
	/// </summary>
	public int NumResiduals(Reconstruction reconstruction)
	{
		// Count the number of observations for all added images.
		int numObservations = 0;
		foreach (uint imageId in imageIds)
		{
			foreach (Point2D point2D in reconstruction.Image(imageId).Points2D)
			{
				if (point2D.HasPoint3D && !IsIgnoredPoint(point2D.Point3DId))
				{
					++numObservations;
				}
			}
		}

		// Count the number of observations for all added 3D points that are not already
		// added as part of the images above.
		int NumObservationsForPoint(ulong point3DId)
		{
			int numObservationsForPoint = 0;
			foreach (TrackElement trackEl in reconstruction.Point3D(point3DId).Track.Elements)
			{
				if (!imageIds.Contains(trackEl.ImageId))
				{
					++numObservationsForPoint;
				}
			}

			return numObservationsForPoint;
		}

		foreach (ulong point3DId in variablePoint3DIds)
		{
			numObservations += NumObservationsForPoint(point3DId);
		}

		foreach (ulong point3DId in constantPoint3DIds)
		{
			numObservations += NumObservationsForPoint(point3DId);
		}

		return 2 * numObservations;
	}

	/// <summary>Adds an image.</summary>
	public void AddImage(uint imageId) => imageIds.Add(imageId);

	/// <summary>Whether the image was added.</summary>
	public bool HasImage(uint imageId) => imageIds.Contains(imageId);

	/// <summary>Removes an image (no-op if absent).</summary>
	public void RemoveImage(uint imageId) => imageIds.Remove(imageId);

	/// <summary>Holds a camera's intrinsics constant.</summary>
	public void SetConstantCamIntrinsics(uint cameraId) => constantCamIntrinsics.Add(cameraId);

	/// <summary>Lets a camera's intrinsics vary (the default).</summary>
	public void SetVariableCamIntrinsics(uint cameraId) => constantCamIntrinsics.Remove(cameraId);

	/// <summary>Whether the camera's intrinsics are constant.</summary>
	public bool HasConstantCamIntrinsics(uint cameraId) => constantCamIntrinsics.Contains(cameraId);

	/// <summary>Holds a sensor's sensor_from_rig constant.</summary>
	public void SetConstantSensorFromRigPose(SensorId sensorId) => constantSensorFromRigPoses.Add(sensorId);

	/// <summary>Lets a sensor's sensor_from_rig vary.</summary>
	public void SetVariableSensorFromRigPose(SensorId sensorId) => constantSensorFromRigPoses.Remove(sensorId);

	/// <summary>Whether the sensor's sensor_from_rig is constant.</summary>
	public bool HasConstantSensorFromRigPose(SensorId sensorId) => constantSensorFromRigPoses.Contains(sensorId);

	/// <summary>Holds a frame's rig_from_world constant.</summary>
	public void SetConstantRigFromWorldPose(uint frameId) => constantRigFromWorldPoses.Add(frameId);

	/// <summary>Lets a frame's rig_from_world vary.</summary>
	public void SetVariableRigFromWorldPose(uint frameId) => constantRigFromWorldPoses.Remove(frameId);

	/// <summary>Whether the frame's rig_from_world is constant.</summary>
	public bool HasConstantRigFromWorldPose(uint frameId) => constantRigFromWorldPoses.Contains(frameId);

	/// <summary>Adds a variable point; it may not also be constant.</summary>
	public void AddVariablePoint(ulong point3DId)
	{
		Check.That(!HasConstantPoint(point3DId));
		variablePoint3DIds.Add(point3DId);
	}

	/// <summary>Adds a constant point; it may not also be variable.</summary>
	public void AddConstantPoint(ulong point3DId)
	{
		Check.That(!HasVariablePoint(point3DId));
		constantPoint3DIds.Add(point3DId);
	}

	/// <summary>Excludes a point (and its observations) from the problem.</summary>
	public void IgnorePoint(ulong point3DId)
	{
		Check.That(!HasVariablePoint(point3DId));
		Check.That(!HasConstantPoint(point3DId));
		ignoredPoint3DIds.Add(point3DId);
	}

	/// <summary>Whether the point was added as variable or constant.</summary>
	public bool HasPoint(ulong point3DId) => HasVariablePoint(point3DId) || HasConstantPoint(point3DId);

	/// <summary>Whether the point was added as variable.</summary>
	public bool HasVariablePoint(ulong point3DId) => variablePoint3DIds.Contains(point3DId);

	/// <summary>Whether the point was added as constant.</summary>
	public bool HasConstantPoint(ulong point3DId) => constantPoint3DIds.Contains(point3DId);

	/// <summary>Whether the point is ignored.</summary>
	public bool IsIgnoredPoint(ulong point3DId) => ignoredPoint3DIds.Contains(point3DId);

	/// <summary>Removes a variable point.</summary>
	public void RemoveVariablePoint(ulong point3DId) => variablePoint3DIds.Remove(point3DId);

	/// <summary>Removes a constant point.</summary>
	public void RemoveConstantPoint(ulong point3DId) => constantPoint3DIds.Remove(point3DId);

	/// <summary>A deep copy (C++ copy construction).</summary>
	public BundleAdjustmentConfig Clone()
	{
		var copy = new BundleAdjustmentConfig { FixedGauge = FixedGauge };
		copy.constantCamIntrinsics.UnionWith(constantCamIntrinsics);
		copy.imageIds.UnionWith(imageIds);
		copy.variablePoint3DIds.UnionWith(variablePoint3DIds);
		copy.constantPoint3DIds.UnionWith(constantPoint3DIds);
		copy.ignoredPoint3DIds.UnionWith(ignoredPoint3DIds);
		copy.constantSensorFromRigPoses.UnionWith(constantSensorFromRigPoses);
		copy.constantRigFromWorldPoses.UnionWith(constantRigFromWorldPoses);
		return copy;
	}
}

/// <summary>Port of colmap::BundleAdjustmentOptions (with BundleAdjustmentBackendOptions).</summary>
public sealed class BundleAdjustmentOptions
{
	/// <summary>Ceres-specific options (used when Backend is Ceres).</summary>
	public CeresBundleAdjustmentOptions? Ceres { get; set; } = new();

	/// <summary>Whether to refine the focal length parameter group.</summary>
	public bool RefineFocalLength { get; set; } = true;

	/// <summary>Whether to refine the principal point parameter group.</summary>
	public bool RefinePrincipalPoint { get; set; }

	/// <summary>Whether to refine the extra parameter group.</summary>
	public bool RefineExtraParams { get; set; } = true;

	/// <summary>Whether to refine sensor_from_rig.</summary>
	public bool RefineSensorFromRig { get; set; } = true;

	/// <summary>Whether to refine rig_from_world.</summary>
	public bool RefineRigFromWorld { get; set; } = true;

	/// <summary>
	/// Whether to refine the 3D point positions. When false, all 3D points are treated as
	/// constant, enabling refinement of only camera intrinsics and poses. This is useful when
	/// 3D points come from a reference model and should not be modified.
	/// </summary>
	public bool RefinePoints3D { get; set; } = true;

	/// <summary>
	/// Minimum track length for a 3D point to be included in bundle adjustment. Points with
	/// fewer observations are ignored.
	/// </summary>
	public int MinTrackLength { get; set; }

	/// <summary>
	/// Whether to keep the rotation component of rig_from_world constant. Only takes effect
	/// when RefineRigFromWorld is true. When true, only translation is refined.
	/// </summary>
	public bool ConstantRigFromWorldRotation { get; set; }

	/// <summary>Whether to print a final summary (COLMAP logs it; the library has no log yet).</summary>
	public bool PrintSummary { get; set; } = true;

	/// <summary>Solver backend to use for bundle adjustment.</summary>
	public BundleAdjustmentBackend Backend { get; set; } = BundleAdjustmentBackend.Ceres;

	/// <summary>
	/// Optional cooperative cancellation callback, evaluated after each solver iteration.
	/// Returning true ends the solve with USER_SUCCESS.
	/// </summary>
	public Func<bool>? CheckIfStopped { get; set; }

	/// <summary>Port of BundleAdjustmentOptions::Check: throws if Ceres is null, false on a bad option.</summary>
	public bool Check() => Util.Check.NotNull(Ceres).Check();

	/// <summary>A deep copy, including the backend options (COLMAP's copy constructor).</summary>
	public BundleAdjustmentOptions Clone()
	{
		var copy = (BundleAdjustmentOptions)MemberwiseClone();
		copy.Ceres = Ceres?.Clone();
		return copy;
	}
}

/// <summary>Port of colmap::BundleAdjuster: the backend-independent interface.</summary>
public abstract class BundleAdjuster
{
	/// <summary>Copies the options and config; throws if the options fail Check.</summary>
	protected BundleAdjuster(BundleAdjustmentOptions options, BundleAdjustmentConfig config)
	{
		ArgumentNullException.ThrowIfNull(options);
		ArgumentNullException.ThrowIfNull(config);
		OptionsInternal = options.Clone();
		ConfigInternal = config.Clone();
		Check.That(OptionsInternal.Check());
	}

	/// <summary>The adjuster's copy of the options.</summary>
	public BundleAdjustmentOptions Options => OptionsInternal;

	/// <summary>The adjuster's copy of the config.</summary>
	public BundleAdjustmentConfig Config => ConfigInternal;

	/// <summary>The mutable options (COLMAP's protected options_).</summary>
	protected BundleAdjustmentOptions OptionsInternal { get; }

	/// <summary>The mutable config (COLMAP's protected config_).</summary>
	protected BundleAdjustmentConfig ConfigInternal { get; }

	/// <summary>
	/// Runs the solve, writing the result into the reconstruction. Cancelling
	/// <paramref name="cancellationToken"/> stops after the current iteration with USER_SUCCESS.
	/// </summary>
	public abstract BundleAdjustmentSummary Solve(CancellationToken cancellationToken = default);
}

/// <summary>The bundle adjuster factories of bundle_adjustment.h.</summary>
public static class BundleAdjusters
{
	/// <summary>Port of colmap::CreateDefaultBundleAdjuster.</summary>
	public static BundleAdjuster CreateDefaultBundleAdjuster(
		BundleAdjustmentOptions options, BundleAdjustmentConfig config, Reconstruction reconstruction)
	{
		return options.Backend switch
		{
			BundleAdjustmentBackend.Ceres => CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction),
			BundleAdjustmentBackend.Caspar => throw new InvalidOperationException(
				"Caspar BA backend selected but ColmapSharp has no Caspar (GPU) backend; use the Ceres backend."),
			_ => throw new InvalidOperationException($"Unknown bundle adjustment backend: {(int)options.Backend}"),
		};
	}
}

/// <summary>Port of colmap::PosePriorBundleAdjustmentOptions (with its backend options).</summary>
public sealed class PosePriorBundleAdjustmentOptions
{
	/// <summary>Ceres-specific options.</summary>
	public CeresPosePriorBundleAdjustmentOptions? Ceres { get; set; } = new();

	/// <summary>Fallback if no prior position covariance is provided.</summary>
	public double PriorPositionFallbackStddev { get; set; } = 1.0;

	/// <summary>Sim3 alignment options.</summary>
	public RansacOptions AlignmentRansacOptions { get; set; } = new();

	/// <summary>Port of PosePriorBundleAdjustmentOptions::Check.</summary>
	public bool Check() => PriorPositionFallbackStddev > 0 && Util.Check.NotNull(Ceres).Check();

	/// <summary>A deep copy, including the backend options.</summary>
	public PosePriorBundleAdjustmentOptions Clone()
	{
		var copy = (PosePriorBundleAdjustmentOptions)MemberwiseClone();
		copy.Ceres = Ceres?.Clone();
		return copy;
	}
}
