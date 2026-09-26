// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GravityRefinement: colmap/estimators/gravity_refinement.h and .cc - refines the gravity
// directions stored in pose priors (Geometry/PosePrior.cs) using the relative rotations of
// a Scene/PoseGraph. Frames whose gravity disagrees with too many neighbors (measured as
// the non-upright part of the gravity-aligned relative rotation) get a new gravity: the
// average of the neighbors' gravities carried over by the relative rotations, refined on
// the sphere (Solver/SphereProductManifolds.cs) under an arctan loss.
// Tests: ColmapSharp.Tests/Estimators/GravityRefinementTests.cs
// (gravity_refinement_test.cc 1:1). Tier C (a nonlinear solve over noisy priors).
//
// Translation notes:
// - COLMAP holds PosePrior* into the caller's vector; PosePrior is a struct here, so the
//   maps hold indices into the caller's List and write back through them. As in COLMAP, a
//   frame refined earlier changes the neighbor gravities a later frame sees.
// - The error-prone frames, the per-frame neighbor pairs and the adjacency sets are sorted
//   (COLMAP iterates absl hash sets). Because of the previous point, the frame order can
//   change the result (docs/CPP_DIVERGENCES.md entry 47).
// - COLMAP's outlier check after the solve compares each neighbor gravity with itself (the
//   loop variable shadows the refined gravity), so the error is always 0 and every solved
//   frame is accepted. Kept as is, since this is the behavior COLMAP's tests pin.
// - LOG output is dropped (PORTING_PLAN.md).

using System.Runtime.InteropServices;

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::GravityRefinerOptions.</summary>
public sealed class GravityRefinerOptions
{
	/// <summary>The minimal ratio that the gravity vector should be consistent with.</summary>
	public double MaxOutlierRatio { get; set; } = 0.5;

	/// <summary>The maximum allowed angle error in degree.</summary>
	public double MaxGravityError { get; set; } = 1.0;

	/// <summary>Only refine the gravity of the images with more than min_neighbors.</summary>
	public int MinNumNeighbors { get; set; } = 7;

	/// <summary>The options for the solver.</summary>
	public SolverOptions SolverOptions { get; set; } = new()
	{
		NumThreads = -1,
		MaxNumIterations = 100,
		FunctionTolerance = 1e-5,
	};

	/// <summary>An arctan loss at 1 - cos(max_gravity_error).</summary>
	public LossFunction CreateLossFunction() =>
		new ArctanLoss(1 - Math.Cos(MathUtils.DegToRad(MaxGravityError)));
}

/// <summary>Port of colmap::GravityRefiner.</summary>
public sealed class GravityRefiner(GravityRefinerOptions options)
{
	private readonly GravityRefinerOptions options = options;

	/// <summary>
	/// Refine gravity stored in <paramref name="posePriors"/> using relative rotations from
	/// the pose graph. Only priors of cameras that are the reference sensor of their frame
	/// are used and refined.
	/// </summary>
	public void RefineGravity(PoseGraph poseGraph, Reconstruction reconstruction, List<PosePrior> posePriors)
	{
		SortedDictionary<uint, SortedSet<uint>> adjacencyList = CreateImageAdjacencyList(poseGraph);
		if (adjacencyList.Count == 0)
		{
			// COLMAP: LOG(INFO) << "Adjacency list not established".
			return;
		}

		var imageToPosePrior = new Dictionary<uint, int>();
		var frameToPosePrior = new Dictionary<uint, int>();
		for (int i = 0; i < posePriors.Count; ++i)
		{
			PosePrior posePrior = posePriors[i];
			if (posePrior.CorrDataId.SensorId.Type == SensorType.Camera)
			{
				uint imageId = (uint)posePrior.CorrDataId.Id;
				Image image = reconstruction.Image(imageId);

				// TODO(jsch): Can only handle trivial frames.
				if (image.IsRefInFrame)
				{
					Check.That(imageToPosePrior.TryAdd(imageId, i), $"Duplicate pose prior for image {imageId}");
					Check.That(frameToPosePrior.TryAdd(image.FrameId, i), $"Duplicate pose prior for frame {image.FrameId}");
				}
			}
		}

		// Identify the images that are error prone.
		SortedSet<uint> errorProneFrames = IdentifyErrorProneGravity(poseGraph, reconstruction, posePriors, imageToPosePrior);
		if (errorProneFrames.Count == 0)
		{
			// COLMAP: LOG(INFO) << "No error prone frames found".
			return;
		}

		// Get the relevant pair ids for frames.
		var adjacencyListFramesToPairId = new Dictionary<uint, SortedSet<ulong>>();
		foreach ((uint imageId, SortedSet<uint> neighbors) in adjacencyList)
		{
			uint frameId = reconstruction.Image(imageId).FrameId;
			if (!adjacencyListFramesToPairId.TryGetValue(frameId, out SortedSet<ulong>? pairIds))
			{
				pairIds = [];
				adjacencyListFramesToPairId[frameId] = pairIds;
			}

			foreach (uint neighbor in neighbors)
			{
				pairIds.Add(Types.ImagePairToPairId(imageId, neighbor));
			}
		}

		LossFunction lossFunction = options.CreateLossFunction();

		// Iterate through the error prone images.
		foreach (uint frameId in errorProneFrames)
		{
			SortedSet<ulong> neighbors = adjacencyListFramesToPairId[frameId];
			var gravities = new List<Vector3d>(neighbors.Count);

			var problem = new Problem();
			Vector3d prior = posePriors[frameToPosePrior[frameId]].Gravity;
			double[] gravity = [prior.X, prior.Y, prior.Z];
			foreach (ulong pairId in neighbors)
			{
				(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
				PoseGraph.Edge edge = poseGraph.EdgeRef(imageId1, imageId2).Edge;

				Vector3d? imageGravity1 = GetImageGravityOrNull(posePriors, imageToPosePrior, imageId1);
				Vector3d? imageGravity2 = GetImageGravityOrNull(posePriors, imageToPosePrior, imageId2);
				if (imageGravity1 is null || imageGravity2 is null)
				{
					continue;
				}

				Image image1 = reconstruction.Image(imageId1);
				Image image2 = reconstruction.Image(imageId2);

				// Get the cam_from_rig.
				Rigid3d cam1FromRig1 = new();
				Rigid3d cam2FromRig2 = new();
				if (!image1.IsRefInFrame)
				{
					cam1FromRig1 = image1.FramePtr.RigPtr.SensorFromRig(new SensorId(SensorType.Camera, image1.CameraId));
				}

				if (!image2.IsRefInFrame)
				{
					cam2FromRig2 = image2.FramePtr.RigPtr.SensorFromRig(new SensorId(SensorType.Camera, image2.CameraId));
				}

				// Note: for the case where both cameras are from the same frames, we only
				// consider a single cost term.
				if (image1.FrameId == frameId)
				{
					gravities.Add((edge.Cam2FromCam1 * cam1FromRig1).Inverse().Rotation.ToRotationMatrix() * imageGravity2.Value);
				}
				else if (image2.FrameId == frameId)
				{
					gravities.Add((cam2FromRig2.Inverse() * edge.Cam2FromCam1).Rotation.ToRotationMatrix() * imageGravity1.Value);
				}

				// COLMAP indexes gravities[counter] here; every pair of this frame's
				// adjacency contains an image of the frame, so it is the entry just added.
				Vector3d last = gravities[^1];
				problem.AddResidualBlock(NormalPriorCostFunctor.Create([last.X, last.Y, last.Z]), lossFunction, gravity);
			}

			if (gravities.Count < options.MinNumNeighbors)
			{
				continue;
			}

			// Initialize and set the manifold.
			Vector3d average = Pose.AverageDirections(gravities);
			gravity[0] = average.X;
			gravity[1] = average.Y;
			gravity[2] = average.Z;
			ManifoldHelpers.SetManifold(problem, gravity, ManifoldHelpers.CreateSphereManifold(3));

			// Then, run refinement.
			SolverOptions solverOptions = options.SolverOptions.Clone();
			solverOptions.NumThreads = Threading.GetEffectiveNumThreads(solverOptions.NumThreads);
			LeastSquaresSolver.Solve(solverOptions, problem);

			// Check the error with respect to the neighbors. As in COLMAP, the loop variable
			// shadows the refined gravity, so this compares each neighbor with itself (file
			// header).
			int counterOutlier = 0;
			foreach (Vector3d neighborGravity in gravities)
			{
				double error = MathUtils.RadToDeg(
					Math.Acos(Math.Max(Math.Min(neighborGravity.Dot(neighborGravity), 1.0), -1.0)));
				if (error > options.MaxGravityError * 2)
				{
					counterOutlier++;
				}
			}

			// If the refined gravity now consistent with more images, then accept it.
			if ((double)counterOutlier / gravities.Count < options.MaxOutlierRatio)
			{
				Span<PosePrior> priors = CollectionsMarshal.AsSpan(posePriors);
				priors[frameToPosePrior[frameId]].Gravity = new Vector3d(gravity[0], gravity[1], gravity[2]);
			}
		}
	}

	private static Vector3d? GetImageGravityOrNull(List<PosePrior> posePriors, Dictionary<uint, int> imageToPosePrior, uint imageId)
	{
		if (!imageToPosePrior.TryGetValue(imageId, out int index) || !posePriors[index].HasGravity())
		{
			return null;
		}

		return posePriors[index].Gravity;
	}

	private static SortedDictionary<uint, SortedSet<uint>> CreateImageAdjacencyList(PoseGraph poseGraph)
	{
		var adjacencyList = new SortedDictionary<uint, SortedSet<uint>>();
		foreach ((ulong pairId, _) in poseGraph.ValidEdges())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			AddNeighbor(adjacencyList, imageId1, imageId2);
			AddNeighbor(adjacencyList, imageId2, imageId1);
		}

		return adjacencyList;
	}

	private static void AddNeighbor(SortedDictionary<uint, SortedSet<uint>> adjacencyList, uint imageId, uint neighbor)
	{
		if (!adjacencyList.TryGetValue(imageId, out SortedSet<uint>? neighbors))
		{
			neighbors = [];
			adjacencyList[imageId] = neighbors;
		}

		neighbors.Add(neighbor);
	}

	private SortedSet<uint> IdentifyErrorProneGravity(
		PoseGraph poseGraph,
		Reconstruction reconstruction,
		List<PosePrior> posePriors,
		Dictionary<uint, int> imageToPosePrior)
	{
		var errorProneFrames = new SortedSet<uint>();
		double maxGravityErrorRad = MathUtils.DegToRad(options.MaxGravityError);

		// frame_id: (mistake, total). Set the counter of all frames to 0.
		var frameCounter = new SortedDictionary<uint, (int Mistakes, int Total)>();
		foreach (uint frameId in reconstruction.Frames.Keys)
		{
			frameCounter[frameId] = (0, 0);
		}

		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.ValidEdges())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			Vector3d? imageGravity1 = GetImageGravityOrNull(posePriors, imageToPosePrior, imageId1);
			Vector3d? imageGravity2 = GetImageGravityOrNull(posePriors, imageToPosePrior, imageId2);
			if (imageGravity1 is null || imageGravity2 is null)
			{
				continue;
			}

			uint frameId1 = reconstruction.Image(imageId1).FrameId;
			uint frameId2 = reconstruction.Image(imageId2).FrameId;

			// Calculate the gravity aligned relative rotation.
			Matrix3d rRel = Pose.GravityAlignedRotation(imageGravity2.Value).Transpose()
				* edge.Cam2FromCam1.Rotation.ToRotationMatrix()
				* Pose.GravityAlignedRotation(imageGravity1.Value);

			// Convert it to the closest upright rotation.
			Matrix3d rRelUp = Pose.RotationFromYAxisAngle(Pose.YAxisAngleFromRotation(rRel));

			// Increment the total count, and the mistake count.
			bool mistake = Quaterniond.FromRotationMatrix(rRel).AngularDistance(Quaterniond.FromRotationMatrix(rRelUp))
				> maxGravityErrorRad;
			int increment = mistake ? 1 : 0;
			(int mistakes1, int total1) = frameCounter[frameId1];
			frameCounter[frameId1] = (mistakes1 + increment, total1 + 1);
			(int mistakes2, int total2) = frameCounter[frameId2];
			frameCounter[frameId2] = (mistakes2 + increment, total2 + 1);
		}

		// Filter the frames with too many mistakes.
		foreach ((uint frameId, (int mistakes, int total)) in frameCounter)
		{
			if (total < options.MinNumNeighbors)
			{
				continue;
			}

			if ((double)mistakes / total >= options.MaxOutlierRatio)
			{
				errorProneFrames.Add(frameId);
			}
		}

		return errorProneFrames;
	}
}

/// <summary>Port of colmap::RunGravityRefinement.</summary>
public static class GravityRefinement
{
	/// <summary>
	/// Refine gravity stored in pose priors using relative rotations from the pose graph.
	/// </summary>
	public static void RunGravityRefinement(
		GravityRefinerOptions options, PoseGraph poseGraph, Reconstruction reconstruction, List<PosePrior> posePriors)
	{
		var refiner = new GravityRefiner(options);
		refiner.RefineGravity(poseGraph, reconstruction, posePriors);
	}
}
