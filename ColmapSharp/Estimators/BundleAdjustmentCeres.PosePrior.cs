// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeres.PosePrior: the PosePriorBundleAdjuster of
// colmap/estimators/bundle_adjustment_ceres.cc and its factory
// CreatePosePriorCeresBundleAdjuster. It robustly aligns the reconstruction to the camera
// position priors (Estimators/Alignment.cs, AlignReconstructionToPosePriors), normalizes it,
// builds a DefaultBundleAdjuster (BundleAdjustmentCeres.Default.cs) on the normalized scene
// and adds one covariance-weighted position prior residual per parameterized image
// (Estimators/CostFunctions/PosePriorCostFunctions.cs). Solve maps the result back to the
// prior (metric) frame. With fewer than three usable priors, or when the alignment fails, the
// priors cannot fix the 7-DoF gauge and it is fixed with two cameras instead.
// Tests: ColmapSharp.Tests/Estimators/BundleAdjustmentCeresTests.PosePrior.cs and
// BundleAdjustmentTests.Nominal.cs (PosePriorBundleAdjusterBackendTest).
//
// Tier C (outcome), like the default adjuster; the alignment is Tier C through LO-RANSAC.
//
// Translation notes:
// - The LOG(WARNING) on a failed alignment and the LOG(ERROR) on an unusable solution go to
//   Util/Log.cs; the VLOG(2) alignment error report is dropped, like LOG(INFO).
// - As in COLMAP, a problem without residuals returns an empty summary without undoing the
//   normalization, so the reconstruction is left in the normalized frame in that case.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of the file-local colmap::PosePriorBundleAdjuster.</summary>
internal sealed class PosePriorBundleAdjuster : CeresBundleAdjuster
{
	private readonly PosePriorBundleAdjustmentOptions priorOptions;
	private readonly List<PosePrior> posePriors;
	private readonly Reconstruction reconstruction;
	private readonly DefaultBundleAdjuster defaultBundleAdjuster;
	private readonly LossFunction? priorLossFunction;
	private readonly Sim3d normalizedFromMetric = Sim3d.Identity;

	public PosePriorBundleAdjuster(
		BundleAdjustmentOptions options,
		PosePriorBundleAdjustmentOptions priorOptions,
		BundleAdjustmentConfig config,
		IEnumerable<PosePrior> posePriors,
		Reconstruction reconstruction)
		: base(options, config)
	{
		ArgumentNullException.ThrowIfNull(priorOptions);
		ArgumentNullException.ThrowIfNull(posePriors);
		ArgumentNullException.ThrowIfNull(reconstruction);
		this.priorOptions = priorOptions.Clone();
		this.reconstruction = reconstruction;

		Check.That(this.priorOptions.Check());

		// Filter irrelevant pose priors.
		this.posePriors = posePriors
			.Where(posePrior => posePrior.HasPosition()
				&& posePrior.CorrDataId.SensorId.Type == SensorType.Camera
				&& ConfigInternal.HasImage((uint)posePrior.CorrDataId.Id))
			.ToList();

		bool usePriorPosition = this.posePriors.Count >= 3 && AlignReconstruction();

		// Fix 7-DOFs of the BA problem if the pose priors cannot constrain them.
		if (usePriorPosition)
		{
			// Normalize the reconstruction to avoid any numerical instability but do not
			// transform priors as they will be transformed when added to the problem.
			normalizedFromMetric = reconstruction.Normalize(fixedScale: true);
		}
		else
		{
			ConfigInternal.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		}

		// WARNING: Do not move this above the reconstruction normalization.
		defaultBundleAdjuster = new DefaultBundleAdjuster(OptionsInternal, ConfigInternal, reconstruction);

		if (usePriorPosition)
		{
			CeresPosePriorBundleAdjustmentOptions ceresPriorOptions = this.priorOptions.Ceres!;
			priorLossFunction = CeresBundleAdjustmentOptions.CreateLossFunction(
				ceresPriorOptions.PriorPositionLossFunctionType, ceresPriorOptions.PriorPositionLossScale);

			// Only consider parameterized images for pose priors. Notice that some images may
			// be configured to be included in the BA problem but have no reprojection
			// constraints, etc.
			IReadOnlySet<uint> parameterizedImageIds = defaultBundleAdjuster.ParameterizedImageIds;
			foreach (PosePrior posePrior in this.posePriors)
			{
				uint imageId = (uint)posePrior.CorrDataId.Id;
				if (parameterizedImageIds.Contains(imageId))
				{
					AddImagePosePriorToProblem(imageId, posePrior, reconstruction);
				}
			}
		}
	}

	/// <inheritdoc/>
	public override Problem Problem => defaultBundleAdjuster.Problem;

	/// <inheritdoc/>
	public override BundleAdjustmentSummary Solve(CancellationToken cancellationToken = default)
	{
		Problem problem = defaultBundleAdjuster.Problem;
		if (problem.NumResiduals == 0)
		{
			return new BundleAdjustmentSummary();
		}

		SolverSummary ceresSummary = CeresBundleAdjusters.Solve(OptionsInternal, ConfigInternal, problem, cancellationToken);

		reconstruction.Transform(normalizedFromMetric.Inverse());

		return CeresBundleAdjusters.CreateSummaryAndLogFailure(ceresSummary, "Pose prior bundle adjustment");
	}

	private void AddImagePosePriorToProblem(uint imageId, PosePrior posePrior, Reconstruction reconstruction)
	{
		Image image = reconstruction.Image(imageId);

		bool constantSensorFromRig =
			image.IsRefInFrame
			|| !OptionsInternal.RefineSensorFromRig
			|| ConfigInternal.HasConstantSensorFromRigPose(image.CameraPtr.SensorId);
		bool constantRigFromWorld =
			!OptionsInternal.RefineRigFromWorld || ConfigInternal.HasConstantRigFromWorldPose(image.FrameId);
		if (constantSensorFromRig && constantRigFromWorld)
		{
			return;
		}

		Problem problem = defaultBundleAdjuster.Problem;
		Frame frame = image.FramePtr;
		double[] rigFromWorld = frame.RigFromWorldStorage.Params;

		Vector3d normalizedPosition = normalizedFromMetric * posePrior.Position;
		Matrix3d normalizedFromMetricScaledRotation =
			normalizedFromMetric.Scale * normalizedFromMetric.Rotation.ToRotationMatrix();
		double fallbackStddev = priorOptions.PriorPositionFallbackStddev;
		Matrix3d positionCov = posePrior.HasPositionCov()
			? posePrior.PositionCovariance
			: fallbackStddev * fallbackStddev * Matrix3d.Identity;
		Matrix3d normalizedPositionCov =
			normalizedFromMetricScaledRotation * positionCov * normalizedFromMetricScaledRotation.Transpose();

		if (image.IsRefInFrame)
		{
			problem.AddResidualBlock(
				CovarianceWeightedCostFunctor.Create(
					MatrixXd.From(normalizedPositionCov),
					new AbsolutePosePositionPriorCostFunctor(normalizedPosition)),
				priorLossFunction,
				rigFromWorld);
		}
		else
		{
			double[] camFromRig = frame.RigPtr.SensorFromRigStorage(image.CameraPtr.SensorId).Params;
			problem.AddResidualBlock(
				CovarianceWeightedCostFunctor.Create(
					MatrixXd.From(normalizedPositionCov),
					new AbsoluteRigPosePositionPriorCostFunctor(normalizedPosition)),
				priorLossFunction,
				camFromRig,
				rigFromWorld);

			// Reprojection residuals may omit constant poses, so the prior can add their
			// parameter blocks after the default parameterization pass.
			if (constantSensorFromRig)
			{
				problem.SetParameterBlockConstant(camFromRig);
			}
		}

		if (constantRigFromWorld)
		{
			problem.SetParameterBlockConstant(rigFromWorld);
		}
	}

	private bool AlignReconstruction()
	{
		var metricFromOrig = new Sim3d();
		if (!Alignment.AlignReconstructionToPosePriors(
				reconstruction,
				posePriors,
				priorOptions.AlignmentRansacOptions,
				priorOptions.PriorPositionFallbackStddev,
				ref metricFromOrig))
		{
			Log.Warning("Alignment w.r.t. prior positions failed");
			return false;
		}

		reconstruction.Transform(metricFromOrig);
		return true;
	}
}
