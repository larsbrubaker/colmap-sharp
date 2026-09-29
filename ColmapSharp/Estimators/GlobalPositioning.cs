// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPositioning: colmap/estimators/global_positioning.h and .cc - the positioning step
// of global SfM. With the rotations known, it solves jointly for the frame centers, the 3D
// points and one scale per observation from point-to-camera direction constraints (BATA,
// Estimators/CostFunctions/MotionAveragingCostFunctions.cs), with the scales bounded below,
// a Huber loss (halved for cameras without a focal prior), a user ParameterBlockOrdering
// (scales, then points, then centers) and SPARSE_SCHUR (Solver/). Unknown sensor_from_rig
// translations of multi-camera rigs are estimated too.
// Tests: ColmapSharp.Tests/Estimators/GlobalPositioningTests.cs
// (global_positioning_test.cc 1:1, see its header for the cases that wait). Tier C.
//
// Translation notes:
// - Ceres points at COLMAP's own memory (a NodeHashMap of centers, Point3D::xyz, a vector
//   of scales). Here each center, point and camera-in-rig position is a double[3], and the
//   scales are one-element slices of one double[] (Solver/Problem.cs identifies a block by
//   array and offset). ConvertBackResults writes the frames, rigs and points back; COLMAP
//   writes the points in place, which is the same by the end of Solve.
// - The random frame starts are drawn while iterating reconstruction.Frames, which enumerates
//   in ascending frame id (Util/IdMap.cs); COLMAP iterates a NodeHashMap, so a given seed
//   hands its draws to different frames. The frame centers, points and cameras-in-rig are
//   kept sorted by id too, which sets the element order inside the ordering groups. The
//   final positions agree within the tests' tolerances (divergence 48).
// - The CUDA options (use_gpu, gpu_index, min_num_images_gpu_solver) are not ported: the
//   solver runs on the CPU, as COLMAP does without CUDA.
// - LOG(WARNING)/LOG(ERROR) go to Util/Log.cs; LOG(INFO)/VLOG output is dropped.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::GlobalPositionerOptions (without the CUDA options).</summary>
public sealed class GlobalPositionerOptions
{
	/// <summary>Whether to initialize the camera positions randomly.</summary>
	public bool GenerateRandomPositions { get; set; } = true;

	/// <summary>Whether to initialize the track positions randomly.</summary>
	public bool GenerateRandomPoints { get; set; } = true;

	/// <summary>
	/// Whether to initialize the camera scales to a constant 1 or derive them from the
	/// initialized camera and point positions.
	/// </summary>
	public bool GenerateScales { get; set; } = true;

	/// <summary>Whether to optimize the frame positions.</summary>
	public bool OptimizePositions { get; set; } = true;

	/// <summary>Whether to optimize the 3D points.</summary>
	public bool OptimizePoints { get; set; } = true;

	/// <summary>Whether to optimize the scales.</summary>
	public bool OptimizeScales { get; set; } = true;

	/// <summary>When false, treat sensor_from_rig as a fixed (pre-calibrated) parameter.</summary>
	public bool RefineSensorFromRig { get; set; } = true;

	/// <summary>Constrain the minimum number of views per track.</summary>
	public int MinNumViewPerTrack { get; set; } = 3;

	/// <summary>
	/// PRNG seed for random initialization. If -1 (default), the thread's PRNG is used as it
	/// is; if >= 0, it is reseeded with the given value.
	/// </summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>Scaling factor for the loss function.</summary>
	public double LossFunctionScale { get; set; } = 0.1;

	/// <summary>
	/// Whether to use custom parameter block ordering for Schur-based solvers. Disable for
	/// deterministic behavior when using a fixed random seed.
	/// </summary>
	public bool UseParameterBlockOrdering { get; set; } = true;

	/// <summary>The options for the solver.</summary>
	public SolverOptions SolverOptions { get; set; } = new()
	{
		NumThreads = -1,
		MaxNumIterations = 100,
		FunctionTolerance = 1e-5,
	};

	/// <summary>A Huber loss at <see cref="LossFunctionScale"/>.</summary>
	public LossFunction CreateLossFunction() => new HuberLoss(LossFunctionScale);

	/// <summary>A copy (C++ copies the options struct by value).</summary>
	public GlobalPositionerOptions Clone()
	{
		var copy = (GlobalPositionerOptions)MemberwiseClone();
		copy.SolverOptions = SolverOptions.Clone();
		return copy;
	}
}

/// <summary>Port of colmap::GlobalPositioner.</summary>
public sealed class GlobalPositioner
{
	private readonly GlobalPositionerOptions options;

	private Problem problem = new();

	// Loss functions for reweighted terms.
	private LossFunction? lossFunction;
	private LossFunction? lossFunctionPtcamUncalibrated;
	private LossFunction? lossFunctionPtcamCalibrated;

	// Auxiliary scale variables, one block per slot in use.
	private double[] scales = [];
	private int numScales;

	// Frame centers (world coordinates) during optimization, so that RigFromWorld() keeps the
	// cam_from_world convention.
	private readonly SortedDictionary<uint, double[]> frameCenters = [];

	// Camera-in-rig positions when cam_from_rig is unknown and needs to be estimated.
	private readonly SortedDictionary<SensorId, double[]> camsInRig = [];

	// The 3D points during optimization.
	private readonly SortedDictionary<ulong, double[]> points = [];

	/// <summary>Creates a positioner; a non-negative random seed reseeds the thread's PRNG.</summary>
	public GlobalPositioner(GlobalPositionerOptions options)
	{
		this.options = options.Clone();
		if (this.options.RandomSeed >= 0)
		{
			RandomUtils.SetPRNGSeed(unchecked((uint)this.options.RandomSeed));
		}
	}

	/// <summary>The options (mutable, as COLMAP's GetOptions()).</summary>
	public GlobalPositionerOptions Options => options;

	/// <summary>
	/// Returns true if the optimization was a success, false if there was a failure. Assumes
	/// the tracks are already filtered.
	/// </summary>
	public bool Solve(PoseGraph poseGraph, Reconstruction reconstruction)
	{
		if (reconstruction.NumImages == 0)
		{
			Log.Error($"Number of images = {reconstruction.NumImages}");
			return false;
		}

		if (reconstruction.NumPoints3D == 0)
		{
			Log.Error($"Number of tracks = {reconstruction.NumPoints3D}");
			return false;
		}

		// Setup the problem.
		SetupProblem(reconstruction);

		// Initialize camera translations to be random. Also, convert the camera pose
		// translation to be the camera center.
		InitializeRandomPositions(poseGraph, reconstruction);

		// Add the point to camera constraints to the problem.
		AddPointToCameraConstraints(reconstruction);

		if (options.UseParameterBlockOrdering)
		{
			AddCamerasAndPointsToParameterGroups();
		}

		// Parameterize the variables, set image poses / tracks / scales to be constant if
		// desired.
		ParameterizeVariables();

		options.SolverOptions.NumThreads = Threading.GetEffectiveNumThreads(options.SolverOptions.NumThreads);
		SolverSummary summary = LeastSquaresSolver.Solve(options.SolverOptions, problem);

		ConvertBackResults(reconstruction);
		return summary.IsSolutionUsable;
	}

	private void SetupProblem(Reconstruction reconstruction)
	{
		problem = new Problem();
		lossFunction = options.CreateLossFunction();

		// Clear temporary storage from previous runs.
		frameCenters.Clear();
		camsInRig.Clear();
		points.Clear();

		// Allocate enough memory for the scales. One for each residual. Due to possibly
		// invalid tracks, the actual number of residuals may be smaller.
		int totalObservations = 0;
		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			totalObservations += point3D.Track.Length;
		}

		scales = new double[totalObservations];
		numScales = 0;
	}

	// Initialize all constrained frame centers, randomly or from the current poses.
	private void InitializeRandomPositions(PoseGraph poseGraph, Reconstruction reconstruction)
	{
		var constrainedPositions = new HashSet<uint>();
		foreach ((ulong pairId, _) in poseGraph.ValidEdges())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			constrainedPositions.Add(reconstruction.Image(imageId1).FrameId);
			constrainedPositions.Add(reconstruction.Image(imageId2).FrameId);
		}

		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			if (point3D.Track.Length < options.MinNumViewPerTrack)
			{
				continue;
			}

			foreach (TrackElement observation in point3D.Track.Elements)
			{
				Check.That(reconstruction.ExistsImage(observation.ImageId));
				Image image = reconstruction.Image(observation.ImageId);
				if (!image.HasPose)
				{
					continue;
				}

				constrainedPositions.Add(image.FrameId);
			}
		}

		// Initialize frame centers in temporary storage. The reconstruction poses remain in
		// cam_from_world convention.
		foreach ((uint frameId, Frame frame) in reconstruction.Frames)
		{
			if (!constrainedPositions.Contains(frameId))
			{
				continue;
			}

			Vector3d center = options.GenerateRandomPositions && options.OptimizePositions
				? 100.0 * RandVector3d(-1, 1)
				: frame.RigFromWorld().TgtOriginInSrc();
			frameCenters[frameId] = [center.X, center.Y, center.Z];
		}
	}

	private void AddPointToCameraConstraints(Reconstruction reconstruction)
	{
		// Down-weight uncalibrated cameras.
		lossFunctionPtcamUncalibrated = new ScaledLoss(lossFunction, 0.5);
		lossFunctionPtcamCalibrated = lossFunction;

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			if (point3D.Track.Length < options.MinNumViewPerTrack)
			{
				continue;
			}

			AddPoint3DToProblem(point3DId, point3D, reconstruction);
		}
	}

	// COLMAP's frame_centers_[frame_id]: operator[] inserts a zero center if missing.
	private double[] FrameCenter(uint frameId)
	{
		if (!frameCenters.TryGetValue(frameId, out double[]? center))
		{
			center = new double[3];
			frameCenters[frameId] = center;
		}

		return center;
	}

	private void AddPoint3DToProblem(ulong point3DId, Point3D point3D, Reconstruction reconstruction)
	{
		bool randomInitialization = options.OptimizePoints && options.GenerateRandomPoints;

		// Only set the points to be random if they are needed to be optimized.
		Vector3d initialXyz = randomInitialization ? 100.0 * RandVector3d(-1, 1) : point3D.Xyz;
		double[] xyz = [initialXyz.X, initialXyz.Y, initialXyz.Z];
		points[point3DId] = xyz;

		// For each view in the track add the point to camera correspondences.
		foreach (TrackElement observation in point3D.Track.Elements)
		{
			if (!reconstruction.ExistsImage(observation.ImageId))
			{
				continue;
			}

			Image image = reconstruction.Image(observation.ImageId);
			if (!image.HasPose)
			{
				continue;
			}

			Vector3d? camRay = image.CameraPtr.CamRayFromImg(image.Point2DAt(observation.Point2DIdx).Xy);
			if (camRay is null)
			{
				Log.Warning(
					$"Ignoring feature because it failed to project: point3D_id={point3DId}, image_id={observation.ImageId}, "
					+ $"feature_id={observation.Point2DIdx}");
				continue;
			}

			Vector3d camFromPoint3DDir = image.CamFromWorld().Rotation.Inverse() * camRay.Value;

			Check.That(numScales < scales.Length, "Not enough capacity was reserved for the scales.");
			var scale = new ArraySegment<double>(scales, numScales++, 1);
			scale[0] = 1;

			if (!options.GenerateScales && randomInitialization)
			{
				double[] center = FrameCenter(image.FrameId);
				Vector3d camFromPoint3DTranslation =
					new Vector3d(xyz[0], xyz[1], xyz[2]) - new Vector3d(center[0], center[1], center[2]);
				// std::max(1e-5, x) returns 1e-5 when x is NaN (a zero translation gives 0/0), as
				// the comparison 1e-5 < NaN is false; Math.Max would return NaN.
				double initialScale = camFromPoint3DDir.Dot(camFromPoint3DTranslation) / camFromPoint3DTranslation.SquaredNorm;
				scale[0] = 1e-5 < initialScale ? initialScale : 1e-5;
			}

			// For calibrated and uncalibrated cameras, use different loss functions. Down
			// weight the uncalibrated cameras.
			Camera camera = reconstruction.Camera(image.CameraId);
			LossFunction? loss = camera.HasPriorFocalLength ? lossFunctionPtcamCalibrated : lossFunctionPtcamUncalibrated;

			if (image.IsRefInFrame)
			{
				// If the image is not part of a camera rig, use the standard BATA error.
				problem.AddResidualBlock(
					BATAPairwiseDirectionCostFunctor.Create(camFromPoint3DDir), loss, FrameCenter(image.FrameId), xyz, scale);
			}
			else
			{
				// If the image is part of a camera rig, use the RigBATA error.
				Rig rig = reconstruction.Rig(image.FramePtr.RigId);
				SensorId sensorId = image.CameraPtr.SensorId;
				Rigid3d camFromRig = rig.SensorFromRig(sensorId);
				Vector3d t = camFromRig.Translation;

				if (!(double.IsNaN(t.X) || double.IsNaN(t.Y) || double.IsNaN(t.Z)))
				{
					Vector3d camFromRigDir = image.CamFromWorld().Rotation.Inverse() * camFromRig.Translation;
					problem.AddResidualBlock(
						RigBATAPairwiseDirectionConstantRigCostFunctor.Create(camFromPoint3DDir, camFromRigDir),
						loss,
						xyz,
						FrameCenter(image.FrameId),
						scale);
				}
				else
				{
					// NaN translation means the sensor's cam_from_rig must be re-estimated, which
					// requires refine_sensor_from_rig=true.
					Check.That(
						options.RefineSensorFromRig,
						$"sensor_from_rig has NaN translation but refine_sensor_from_rig=false (image_id={observation.ImageId})");
					if (!camsInRig.TryGetValue(sensorId, out double[]? camInRig))
					{
						// Will be initialized to random values in ParameterizeVariables().
						camInRig = new double[3];
						camsInRig[sensorId] = camInRig;
					}

					problem.AddResidualBlock(
						RigBATAPairwiseDirectionCostFunctor.Create(camFromPoint3DDir, image.FramePtr.RigFromWorld().Rotation),
						loss,
						xyz,
						FrameCenter(image.FrameId),
						camInRig,
						scale);
				}
			}

			problem.SetParameterLowerBound(scale, 0, 1e-5);
		}
	}

	// Create a custom ordering for Schur-based problems.
	private void AddCamerasAndPointsToParameterGroups()
	{
		var parameterOrdering = new ParameterBlockOrdering();
		options.SolverOptions.LinearSolverOrdering = parameterOrdering;

		// Add scale parameters to group 0 (large and independent).
		for (int i = 0; i < numScales; ++i)
		{
			parameterOrdering.AddElementToGroup(new ArraySegment<double>(scales, i, 1), 0);
		}

		// Add point parameters to group 1.
		int groupId = 1;
		if (points.Count > 0)
		{
			foreach (double[] xyz in points.Values)
			{
				if (problem.HasParameterBlock(xyz))
				{
					parameterOrdering.AddElementToGroup(xyz, groupId);
				}
			}

			groupId++;
		}

		foreach (double[] center in frameCenters.Values)
		{
			if (problem.HasParameterBlock(center))
			{
				parameterOrdering.AddElementToGroup(center, groupId);
			}
		}

		// Add the cam_in_rig to be estimated into the parameter group.
		foreach (double[] center in camsInRig.Values)
		{
			if (problem.HasParameterBlock(center))
			{
				parameterOrdering.AddElementToGroup(center, groupId);
			}
		}
	}

	// Parameterize the variables, set some variables to be constant if desired. For the global
	// positioning, do not set any camera to be constant for easier convergence.
	private void ParameterizeVariables()
	{
		// Initialize cams_in_rig with random values if optimizing positions.
		if (options.OptimizePositions)
		{
			foreach (double[] center in camsInRig.Values)
			{
				if (problem.HasParameterBlock(center))
				{
					Vector3d random = RandVector3d(-1, 1);
					center[0] = random.X;
					center[1] = random.Y;
					center[2] = random.Z;
				}
			}
		}

		// If not optimizing positions, set frame centers to be constant.
		if (!options.OptimizePositions)
		{
			foreach (double[] center in frameCenters.Values)
			{
				if (problem.HasParameterBlock(center))
				{
					problem.SetParameterBlockConstant(center);
				}
			}
		}

		// If do not optimize the points, set them to be constant.
		if (!options.OptimizePoints)
		{
			foreach (double[] xyz in points.Values)
			{
				if (problem.HasParameterBlock(xyz))
				{
					problem.SetParameterBlockConstant(xyz);
				}
			}
		}

		// If do not optimize the scales, set the scales to be constant.
		if (!options.OptimizeScales)
		{
			for (int i = 0; i < numScales; ++i)
			{
				var scale = new ArraySegment<double>(scales, i, 1);
				if (problem.HasParameterBlock(scale))
				{
					problem.SetParameterBlockConstant(scale);
				}
			}
		}

		// Set the first scale to be constant to remove the gauge ambiguity.
		for (int i = 0; i < numScales; ++i)
		{
			var scale = new ArraySegment<double>(scales, i, 1);
			if (problem.HasParameterBlock(scale))
			{
				problem.SetParameterBlockConstant(scale);
				break;
			}
		}

		// Do not use iterative solvers, for its suboptimal performance.
		options.SolverOptions.LinearSolverType = LinearSolverType.SparseSchur;
	}

	// During the optimization, the camera translation is set to be the camera center. Convert
	// the results back to camera poses (and write the optimized points back).
	private void ConvertBackResults(Reconstruction reconstruction)
	{
		foreach ((uint frameId, double[] c) in frameCenters)
		{
			Frame frame = reconstruction.Frame(frameId);
			Quaterniond rotation = frame.RigFromWorld().Rotation;
			frame.SetRigFromWorld(new Rigid3d(rotation, rotation * -new Vector3d(c[0], c[1], c[2])));
		}

		foreach ((SensorId sensorId, double[] c) in camsInRig)
		{
			// Find the rig containing this sensor.
			foreach (Rig rig in reconstruction.Rigs.Values)
			{
				if (!rig.HasSensor(sensorId))
				{
					continue;
				}

				Quaterniond rotation = rig.SensorFromRig(sensorId).Rotation;
				rig.SetSensorFromRig(sensorId, new Rigid3d(rotation, rotation * -new Vector3d(c[0], c[1], c[2])));
				break;
			}
		}

		foreach ((ulong point3DId, double[] xyz) in points)
		{
			reconstruction.Point3D(point3DId).Xyz = new Vector3d(xyz[0], xyz[1], xyz[2]);
		}
	}

	private static Vector3d RandVector3d(double low, double high)
	{
		double x = RandomUtils.RandomUniformReal(low, high);
		double y = RandomUtils.RandomUniformReal(low, high);
		double z = RandomUtils.RandomUniformReal(low, high);
		return new Vector3d(x, y, z);
	}
}

/// <summary>Port of colmap::RunGlobalPositioning.</summary>
public static class GlobalPositioning
{
	/// <summary>Solve global positioning using point-to-camera constraints.</summary>
	public static bool RunGlobalPositioning(GlobalPositionerOptions options, PoseGraph poseGraph, Reconstruction reconstruction)
	{
		var positioner = new GlobalPositioner(options);
		return positioner.Solve(poseGraph, reconstruction);
	}
}
