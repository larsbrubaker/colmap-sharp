// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ViewGraphCalibration: colmap/estimators/view_graph_calibration.h and .cc - estimates the
// focal lengths of cameras without a focal prior from the fundamental matrices of the view
// graph (Fetzer et al., WACV 2020; the residuals are Estimators/CostFunctions/
// CalibrationCostFunctions.cs), then upgrades the pairs with a small calibration error to
// CALIBRATED (E from F and the new K), tags the rest DEGENERATE, and optionally re-estimates
// the relative poses (TwoViewGeometryEstimation.Calibrated.cs). It works directly on a
// Scene/Database. The least squares solve is Solver/ (Ceres replacement): parameter lower
// bounds keep the focal lengths positive, and Problem.Evaluate reads back the per-pair
// residuals.
// Tests: ColmapSharp.Tests/Estimators/ViewGraphCalibrationTests.cs
// (view_graph_calibration_test.cc 1:1). Tier C (a nonlinear solve and RANSAC).
//
// Translation notes:
// - COLMAP's NodeHashMaps become Dictionaries filled in the database's read order (camera
//   id, image id, pair id). Their iteration order never reaches the result here: residual
//   blocks follow the pair list, and each camera and pair is written independently.
// - The focal of a camera is a double[1] (COLMAP points Ceres at a NodeHashMap value).
// - ReestimateRelativePoses' ThreadPool becomes Parallel.For; each task writes its own
//   pair. The matches are read up front instead of under a mutex inside the task, which
//   only changes when they are read.
// - LOG/VLOG output is dropped (PORTING_PLAN.md).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::ViewGraphCalibrationOptions.</summary>
public sealed class ViewGraphCalibrationOptions
{
	/// <summary>Random seed for RANSAC-based estimation (-1 for random).</summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>
	/// Whether to cross-validate prior focal lengths by checking the ratio of calibrated vs
	/// uncalibrated pairs per camera. When enabled, UNCALIBRATED pairs are converted to
	/// CALIBRATED if both cameras have reliable priors.
	/// </summary>
	public bool CrossValidatePriorFocalLengths { get; set; } = true;

	/// <summary>
	/// Minimum ratio of calibrated pairs for a camera to be considered valid during
	/// cross-validation.
	/// </summary>
	public double MinCalibratedPairRatio { get; set; } = 0.5;

	/// <summary>Whether to re-estimate relative poses after focal length calibration.</summary>
	public bool ReestimateRelativePose { get; set; } = true;

	/// <summary>The minimum ratio of the estimated focal length to the prior focal length.</summary>
	public double MinFocalLengthRatio { get; set; } = 0.1;

	/// <summary>The maximum ratio of the estimated focal length to the prior focal length.</summary>
	public double MaxFocalLengthRatio { get; set; } = 10;

	/// <summary>The maximum calibration error for an image pair.</summary>
	public double MaxCalibrationError { get; set; } = 2.0;

	/// <summary>Scaling factor for the loss function.</summary>
	public double LossFunctionScale { get; set; } = 0.01;

	/// <summary>The options for the solver.</summary>
	public SolverOptions SolverOptions { get; set; } = new()
	{
		NumThreads = -1,
		MaxNumIterations = 100,
		FunctionTolerance = 1e-5,
	};

	/// <summary>Options for relative pose re-estimation: the RANSAC max error.</summary>
	public double RelposeMaxError { get; set; } = 1.0;

	/// <summary>Options for relative pose re-estimation: the minimum number of inliers.</summary>
	public int RelposeMinNumInliers { get; set; } = 30;

	/// <summary>Options for relative pose re-estimation: the minimum inlier ratio.</summary>
	public double RelposeMinInlierRatio { get; set; } = 0.25;

	/// <summary>Create loss function for given options (a Cauchy loss).</summary>
	public LossFunction CreateLossFunction() => new CauchyLoss(LossFunctionScale);
}

/// <summary>Port of colmap::CalibrateViewGraph and its helpers.</summary>
public static class ViewGraphCalibration
{
	// Lower bound for focal length optimization to prevent numerical issues.
	private const double FocalLengthLowerBound = 1e-3;

	// Input for focal length calibration: an image pair with its F matrix.
	private readonly record struct FocalLengthCalibInput(ulong PairId, uint CameraId1, uint CameraId2, Matrix3d F);

	// Result of focal length calibration.
	private sealed class FocalLengthCalibResult
	{
		// Optimized focal lengths per camera.
		public Dictionary<uint, double> FocalLengths { get; } = [];

		// Squared calibration error per image pair (unitless, relative error).
		public Dictionary<ulong, double> CalibrationErrorsSq { get; } = [];

		// Whether the calibration succeeded.
		public bool Success { get; set; }
	}

	// A focal length being optimized: the solver's memory and the starting value.
	private sealed class FocalLengthState(double focal)
	{
		public double[] Optimized { get; } = [focal];

		public double Initial { get; } = focal;
	}

	/// <summary>
	/// Calibrate the view graph by estimating focal lengths from fundamental matrices. This
	/// operates directly on the database, reading both UNCALIBRATED and CALIBRATED two-view
	/// geometries along with their associated cameras. It optimizes focal lengths and updates
	/// the camera intrinsics in the database. Image pairs with low calibration error have
	/// their essential matrices computed and relative poses re-estimated, then are upgraded
	/// to CALIBRATED. Pairs with high calibration error are tagged as DEGENERATE.
	/// Port of colmap::CalibrateViewGraph.
	/// </summary>
	public static bool CalibrateViewGraph(ViewGraphCalibrationOptions options, Database database)
	{
		Check.NotNull(database);

		// Read cameras and build image_id -> camera mapping.
		var cameras = new Dictionary<uint, Camera>();
		foreach (Camera camera in database.ReadAllCameras())
		{
			cameras[camera.CameraId] = camera;
		}

		var imageIdToCamera = new Dictionary<uint, Camera>();
		foreach (Image image in database.ReadAllImages())
		{
			imageIdToCamera[image.ImageId] = cameras[image.CameraId];
		}

		// Read UNCALIBRATED and CALIBRATED two-view geometries. A pair whose intrinsics a
		// two-view solver estimated is UNCALIBRATED too: its focal is re-estimated from F
		// rather than trusted, so any estimated intrinsics in camera1/camera2 are ignored here
		// and cleared on upgrade to CALIBRATED.
		var pairs = new List<(ulong PairId, TwoViewGeometry Tvg)>();
		foreach ((ulong pairId, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			if (tvg.Config == TwoViewGeometry.ConfigurationType.Uncalibrated
				|| tvg.Config == TwoViewGeometry.ConfigurationType.Calibrated)
			{
				pairs.Add((pairId, tvg));
			}
		}

		if (pairs.Count == 0)
		{
			// COLMAP: LOG(WARNING) << "No image pairs to calibrate".
			return true;
		}

		if (options.CrossValidatePriorFocalLengths)
		{
			CrossValidatePriorFocalLengths(options.MinCalibratedPairRatio, imageIdToCamera, pairs);
		}

		// Recompute F from E for CALIBRATED pairs using current calibration. This goes through
		// the calibration matrices, so it is restricted to pinhole models for the same reason
		// as CalibrateFocalLengths below.
		foreach ((ulong pairId, TwoViewGeometry tvg) in pairs)
		{
			if (tvg.Config != TwoViewGeometry.ConfigurationType.Calibrated || !tvg.Cam2FromCam1.HasValue)
			{
				continue;
			}

			(Camera camera1, Camera camera2) = CamerasOf(pairId, imageIdToCamera);
			if (!camera1.IsPerspectivePinhole || !camera2.IsPerspectivePinhole)
			{
				continue;
			}

			tvg.F = EssentialMatrix.FundamentalFromEssentialMatrix(
				camera2.CalibrationMatrix(),
				EssentialMatrix.EssentialMatrixFromPose(tvg.Cam2FromCam1.Value),
				camera1.CalibrationMatrix());
		}

		// Prepare inputs and run the optimization.
		var inputs = new List<FocalLengthCalibInput>(pairs.Count);
		foreach ((ulong pairId, TwoViewGeometry tvg) in pairs)
		{
			(Camera camera1, Camera camera2) = CamerasOf(pairId, imageIdToCamera);
			if (!camera1.IsPerspectivePinhole || !camera2.IsPerspectivePinhole)
			{
				continue;
			}

			Check.That(tvg.F.HasValue, "Two-view geometry must have F matrix for focal length calibration");
			inputs.Add(new FocalLengthCalibInput(pairId, camera1.CameraId, camera2.CameraId, tvg.F!.Value));
		}

		FocalLengthCalibResult calibResult = CalibrateFocalLengths(options, inputs, cameras);
		if (!calibResult.Success)
		{
			return false;
		}

		// Update cameras with estimated focal lengths.
		foreach ((uint cameraId, double focalLength) in calibResult.FocalLengths)
		{
			Camera camera = cameras[cameraId];
			camera.SetFocalLength(focalLength);
			camera.HasPriorFocalLength = true;
			database.UpdateCamera(camera);
		}

		// Process pairs: tag degenerate or compute E matrix.
		double maxCalibrationErrorSq = options.MaxCalibrationError * options.MaxCalibrationError;
		var validPairIndices = new List<int>();
		for (int i = 0; i < pairs.Count; ++i)
		{
			(ulong pairId, TwoViewGeometry tvg) = pairs[i];
			if (tvg.Config != TwoViewGeometry.ConfigurationType.Calibrated
				&& tvg.Config != TwoViewGeometry.ConfigurationType.Uncalibrated)
			{
				continue;
			}

			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			if (!calibResult.CalibrationErrorsSq.TryGetValue(pairId, out double errorSq))
			{
				continue;
			}

			if (errorSq > maxCalibrationErrorSq)
			{
				tvg.Config = TwoViewGeometry.ConfigurationType.Degenerate;
				database.UpdateTwoViewGeometry(imageId1, imageId2, tvg);
			}
			else
			{
				Check.That(tvg.F.HasValue, "Two-view geometry must have F matrix for E computation");
				(Camera camera1, Camera camera2) = CamerasOf(pairId, imageIdToCamera);
				UpgradeToCalibrated(tvg, camera1, camera2);
				validPairIndices.Add(i);
			}
		}

		// Re-estimate relative poses for valid pairs.
		if (options.ReestimateRelativePose && validPairIndices.Count > 0)
		{
			var validPairs = new List<(ulong PairId, TwoViewGeometry Tvg)>(validPairIndices.Count);
			foreach (int idx in validPairIndices)
			{
				validPairs.Add(pairs[idx]);
			}

			ReestimateRelativePoses(options, validPairs, imageIdToCamera, database);
			foreach ((ulong pairId, TwoViewGeometry tvg) in validPairs)
			{
				(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
				database.UpdateTwoViewGeometry(imageId1, imageId2, tvg);
			}
		}
		else
		{
			foreach (int idx in validPairIndices)
			{
				(ulong pairId, TwoViewGeometry tvg) = pairs[idx];
				(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
				database.UpdateTwoViewGeometry(imageId1, imageId2, tvg);
			}
		}

		return true;
	}

	private static (Camera Camera1, Camera Camera2) CamerasOf(ulong pairId, Dictionary<uint, Camera> imageIdToCamera)
	{
		(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
		return (imageIdToCamera[imageId1], imageIdToCamera[imageId2]);
	}

	// E from F and the cameras' K, then CALIBRATED. E is now built from these K, and
	// consumers calibrate the rays with camera1/camera2 whenever set, so clear them so the
	// rays use the same K as E, not the solver's own focal.
	private static void UpgradeToCalibrated(TwoViewGeometry tvg, Camera camera1, Camera camera2)
	{
		tvg.E = EssentialMatrix.EssentialFromFundamentalMatrix(
			camera2.CalibrationMatrix(), tvg.F!.Value, camera1.CalibrationMatrix());
		tvg.Config = TwoViewGeometry.ConfigurationType.Calibrated;
		tvg.Camera1 = null;
		tvg.Camera2 = null;
	}

	// Cross-validate prior focal lengths by checking the ratio of calibrated vs uncalibrated
	// pairs per camera. UNCALIBRATED pairs are converted to CALIBRATED if both cameras have
	// reliable priors.
	private static void CrossValidatePriorFocalLengths(
		double minCalibratedPairRatio,
		Dictionary<uint, Camera> imageIdToCamera,
		List<(ulong PairId, TwoViewGeometry Tvg)> pairs)
	{
		// For each camera, count the number of calibrated vs uncalibrated pairs.
		var cameraCounter = new Dictionary<uint, (int Total, int Calibrated)>();
		foreach ((ulong pairId, TwoViewGeometry tvg) in pairs)
		{
			(Camera camera1, Camera camera2) = CamerasOf(pairId, imageIdToCamera);
			if (!camera1.HasPriorFocalLength || !camera2.HasPriorFocalLength)
			{
				continue;
			}

			int calibrated = tvg.Config == TwoViewGeometry.ConfigurationType.Calibrated ? 1 : 0;
			(int total1, int calibrated1) = cameraCounter.GetValueOrDefault(camera1.CameraId);
			cameraCounter[camera1.CameraId] = (total1 + 1, calibrated1 + calibrated);
			(int total2, int calibrated2) = cameraCounter.GetValueOrDefault(camera2.CameraId);
			cameraCounter[camera2.CameraId] = (total2 + 1, calibrated2 + calibrated);
		}

		// Camera is valid if the ratio of calibrated pairs exceeds threshold. A camera that
		// was never counted is invalid (COLMAP's operator[] default-inserts false).
		var cameraValidity = new Dictionary<uint, bool>(cameraCounter.Count);
		foreach ((uint cameraId, (int total, int calibrated)) in cameraCounter)
		{
			double ratio = (double)calibrated / total;
			cameraValidity[cameraId] = ratio > minCalibratedPairRatio;
		}

		// Convert UNCALIBRATED pairs to CALIBRATED if both cameras are valid. Compute E from F
		// using the prior camera calibrations.
		foreach ((ulong pairId, TwoViewGeometry tvg) in pairs)
		{
			if (tvg.Config != TwoViewGeometry.ConfigurationType.Uncalibrated)
			{
				continue;
			}

			(Camera camera1, Camera camera2) = CamerasOf(pairId, imageIdToCamera);

			// Computing E from F assumes a pinhole projection for both cameras. See
			// CalibrateFocalLengths for why fisheye models are excluded.
			if (camera1.IsPerspectivePinhole && camera2.IsPerspectivePinhole
				&& cameraValidity.GetValueOrDefault(camera1.CameraId)
				&& cameraValidity.GetValueOrDefault(camera2.CameraId))
			{
				Check.That(tvg.F.HasValue, "UNCALIBRATED two-view geometry must have F matrix");
				UpgradeToCalibrated(tvg, camera1, camera2);
			}
		}
	}

	// Re-estimate relative poses for all pairs using calibrated cameras.
	private static void ReestimateRelativePoses(
		ViewGraphCalibrationOptions options,
		List<(ulong PairId, TwoViewGeometry Tvg)> pairs,
		Dictionary<uint, Camera> imageIdToCamera,
		Database database)
	{
		var twoViewOptions = new TwoViewGeometryOptions
		{
			ComputeRelativePose = true,
			MinNumInliers = options.RelposeMinNumInliers,
			MinInlierRatio = options.RelposeMinInlierRatio,
		};
		twoViewOptions.RansacOptions.MaxError = options.RelposeMaxError;
		twoViewOptions.RansacOptions.RandomSeed = options.RandomSeed;

		// Pre-read all keypoints and matches from the database.
		var imagePoints = new Dictionary<uint, List<Vector2d>>();
		var matches = new List<FeatureMatch>[pairs.Count];
		for (int i = 0; i < pairs.Count; ++i)
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairs[i].PairId);
			foreach (uint imageId in (ReadOnlySpan<uint>)[imageId1, imageId2])
			{
				if (!imagePoints.ContainsKey(imageId))
				{
					List<FeatureKeypoint> keypoints = database.ReadKeypoints(imageId);
					var points = new List<Vector2d>(keypoints.Count);
					foreach (FeatureKeypoint keypoint in keypoints)
					{
						points.Add(new Vector2d(keypoint.X, keypoint.Y));
					}

					imagePoints[imageId] = points;
				}
			}

			matches[i] = database.ReadMatches(imageId1, imageId2);
		}

		int numThreads = Threading.GetEffectiveNumThreads(options.SolverOptions.NumThreads);
		Parallel.For(0, pairs.Count, new ParallelOptions { MaxDegreeOfParallelism = numThreads }, i =>
		{
			(ulong pairId, _) = pairs[i];
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			TwoViewGeometry tvg = TwoViewGeometryEstimation.EstimateCalibratedTwoViewGeometry(
				imageIdToCamera[imageId1],
				imagePoints[imageId1],
				imageIdToCamera[imageId2],
				imagePoints[imageId2],
				matches[i],
				twoViewOptions);
			pairs[i] = (pairId, tvg);
		});
	}

	// Core optimization for focal length calibration. This is a pure function with no I/O.
	// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices with
	// Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.
	private static FocalLengthCalibResult CalibrateFocalLengths(
		ViewGraphCalibrationOptions options,
		List<FocalLengthCalibInput> inputs,
		Dictionary<uint, Camera> cameras)
	{
		var result = new FocalLengthCalibResult();

		if (inputs.Count == 0)
		{
			result.Success = true;
			return result;
		}

		// Initialize focal lengths from all perspective pinhole cameras. Only these are
		// calibrated below, and every camera seeded here is reported back to the caller,
		// which marks it as having a prior focal length.
		var focalLengths = new Dictionary<uint, FocalLengthState>(cameras.Count);
		foreach ((uint cameraId, Camera camera) in cameras)
		{
			if (camera.IsPerspectivePinhole)
			{
				focalLengths[cameraId] = new FocalLengthState(camera.MeanFocalLength());
			}
		}

		var problem = new Problem();
		LossFunction lossFunction = options.CreateLossFunction();

		foreach (FocalLengthCalibInput input in inputs)
		{
			// The focal length is recovered from F in closed form, which requires the
			// calibration to act projectively on the rays. This holds for pinhole models:
			// their distortion is zero-initialized at this stage and mild distortion is
			// absorbed by the epipolar fit. It does not hold for fisheye models, whose angular
			// projection is part of the model itself, so they would need a different
			// formulation and are skipped here.
			if (!cameras[input.CameraId1].IsPerspectivePinhole || !cameras[input.CameraId2].IsPerspectivePinhole)
			{
				continue;
			}

			if (input.CameraId1 == input.CameraId2)
			{
				problem.AddResidualBlock(
					FetzerFocalLengthSameCameraCostFunctor.Create(input.F, cameras[input.CameraId1].PrincipalPoint()),
					lossFunction,
					focalLengths[input.CameraId1].Optimized);
			}
			else
			{
				problem.AddResidualBlock(
					FetzerFocalLengthCostFunctor.Create(
						input.F,
						cameras[input.CameraId1].PrincipalPoint(),
						cameras[input.CameraId2].PrincipalPoint()),
					lossFunction,
					focalLengths[input.CameraId1].Optimized,
					focalLengths[input.CameraId2].Optimized);
			}
		}

		// Parameterize cameras (fix those with prior, set lower bound).
		int numCameras = 0;
		foreach ((uint cameraId, Camera camera) in cameras)
		{
			if (!camera.IsPerspectivePinhole)
			{
				continue;
			}

			double[] focal = focalLengths[cameraId].Optimized;
			if (!problem.HasParameterBlock(focal))
			{
				continue;
			}

			problem.SetParameterLowerBound(focal, 0, FocalLengthLowerBound);
			if (camera.HasPriorFocalLength)
			{
				problem.SetParameterBlockConstant(focal);
			}
			else
			{
				numCameras++;
			}
		}

		if (numCameras == 0)
		{
			// COLMAP: LOG(INFO) << "No cameras to optimize".
			foreach ((uint cameraId, FocalLengthState focal) in focalLengths)
			{
				result.FocalLengths[cameraId] = focal.Initial;
			}

			result.Success = true;
			return result;
		}

		// Set solver options.
		SolverOptions solverOptions = options.SolverOptions.Clone();
		solverOptions.LinearSolverType = cameras.Count < 50
			? LinearSolverType.DenseNormalCholesky
			: LinearSolverType.SparseNormalCholesky;
		solverOptions.NumThreads = Threading.GetEffectiveNumThreads(solverOptions.NumThreads);

		SolverSummary summary = LeastSquaresSolver.Solve(solverOptions, problem);
		if (!summary.IsSolutionUsable)
		{
			// COLMAP: LOG(ERROR) << "Ceres solver failed".
			result.Success = false;
			return result;
		}

		// Validate focal lengths and revert degenerate ones.
		foreach ((uint cameraId, Camera camera) in cameras)
		{
			if (!camera.IsPerspectivePinhole)
			{
				continue;
			}

			FocalLengthState focal = focalLengths[cameraId];
			if (!problem.HasParameterBlock(focal.Optimized))
			{
				continue;
			}

			double focalLengthRatio = focal.Optimized[0] / focal.Initial;
			if (focalLengthRatio > options.MaxFocalLengthRatio || focalLengthRatio < options.MinFocalLengthRatio)
			{
				// Reset to original focal length.
				focal.Optimized[0] = focal.Initial;
			}
		}

		foreach ((uint cameraId, FocalLengthState focal) in focalLengths)
		{
			result.FocalLengths[cameraId] = focal.Optimized[0];
		}

		// Evaluate calibration errors. Every input added a residual block of two residuals
		// (the inputs are all pinhole), so the residuals line up with the inputs.
		var evalOptions = new ProblemEvaluateOptions
		{
			NumThreads = solverOptions.NumThreads,
			ApplyLossFunction = false,
		};
		var residuals = new List<double>();
		problem.Evaluate(evalOptions, out _, residuals, null, null);

		int residualIdx = 0;
		foreach (FocalLengthCalibInput input in inputs)
		{
			double r0 = residuals[residualIdx];
			double r1 = residuals[residualIdx + 1];
			result.CalibrationErrorsSq[input.PairId] = (r0 * r0) + (r1 * r1);
			residualIdx += 2;
		}

		result.Success = true;
		return result;
	}
}
