// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationEstimator: the RotationEstimator class of colmap/estimators/rotation_averaging.h
// and .cc (EstimateRotations, the stratified gravity-aligned pre-solve, the solve itself and
// the maximum-spanning-tree initialization). It builds a RotationAveragingProblem
// (RotationAveragingProblem*.cs) and solves it with RotationAveragingSolver.cs; the free
// functions (RunRotationAveraging and friends) and the anonymous-namespace helpers are in
// RotationAveraging.cs. Tests: ColmapSharp.Tests/Estimators/RotationAveragingTests.cs
// (rotation_averaging_test.cc 1:1).
//
// Tier C (iterative).
//
// Translation notes:
// - The spanning tree numbers the active images in ascending id order (COLMAP: FlatHashSet
//   order, which picks the tree's root, node 0) and adds the valid edges in ascending pair
//   id order (docs/CPP_DIVERGENCES.md, entry 44).
// - COLMAP's cams_from_world[parent] default-constructs the root's entry (identity rotation,
//   zero translation) the first time a child of the root is visited; the root's entry is
//   seeded the same way here.
// - LOG(INFO)/LOG(ERROR) messages are dropped (no logging sink yet, PORTING_PLAN.md).

using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// High-level interface for rotation averaging: combines problem setup and solving into a
/// single call. Port of colmap::RotationEstimator.
/// </summary>
public sealed class RotationEstimator
{
	private readonly RotationEstimatorOptions _options;

	/// <summary>An estimator with a copy of <paramref name="options"/>.</summary>
	public RotationEstimator(RotationEstimatorOptions options)
	{
		_options = options.Clone();
	}

	/// <summary>
	/// Estimates the global orientations of all views: solves rotation averaging over
	/// <paramref name="activeImageIds"/> and registers their frames with the computed poses
	/// (unknown translation). Returns true on successful estimation.
	/// </summary>
	public bool EstimateRotations(
		PoseGraph poseGraph,
		IReadOnlyList<PosePrior> posePriors,
		IReadOnlySet<uint> activeImageIds,
		Reconstruction reconstruction)
	{
		bool useGravity = RotationAveraging.UseGravity(_options, posePriors);
		if (useGravity && !RotationAveraging.AllSensorsFromRigKnown(reconstruction))
		{
			return false;
		}

		// Handle stratified solving for mixed gravity systems.
		if (useGravity && _options.UseStratified
			&& !MaybeSolveGravityAlignedSubset(poseGraph, posePriors, activeImageIds, reconstruction))
		{
			return false;
		}

		// Solve the full system.
		if (!SolveRotationAveraging(poseGraph, posePriors, activeImageIds, reconstruction))
		{
			return false;
		}

		// Register frames with computed poses.
		foreach (uint imageId in activeImageIds)
		{
			uint frameId = reconstruction.Image(imageId).FrameId;
			Check.That(reconstruction.Frame(frameId).HasPose);
			reconstruction.RegisterFrame(frameId);
		}

		return true;
	}

	/// <summary>
	/// Maybe solves 1-DOF rotation averaging on the gravity-aligned subset. This is the first
	/// phase of stratified solving for mixed gravity systems.
	/// </summary>
	private bool MaybeSolveGravityAlignedSubset(
		PoseGraph poseGraph,
		IReadOnlyList<PosePrior> posePriors,
		IReadOnlySet<uint> activeImageIds,
		Reconstruction reconstruction)
	{
		// Build map from image to pose prior.
		var imageToPosePrior = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in posePriors)
		{
			if (posePrior.CorrDataId.SensorId.Type == SensorType.Camera)
			{
				imageToPosePrior[(uint)posePrior.CorrDataId.Id] = posePrior;
			}
		}

		// Separate pairs into gravity-aligned subset (ascending pair id order, so that the
		// subset's edge map enumerates deterministically).
		var gravityPoseGraph = new PoseGraph();
		int numTotalPairs = 0;
		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.ValidEdges().OrderBy(kv => kv.Key))
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			if (!reconstruction.ExistsImage(imageId1) || !reconstruction.ExistsImage(imageId2))
			{
				continue;
			}

			if (!activeImageIds.Contains(imageId1) || !activeImageIds.Contains(imageId2))
			{
				continue;
			}

			numTotalPairs++;

			bool image1HasGravity = imageToPosePrior.TryGetValue(imageId1, out PosePrior prior1) && prior1.HasGravity();
			bool image2HasGravity = imageToPosePrior.TryGetValue(imageId2, out PosePrior prior2) && prior2.HasGravity();
			if (image1HasGravity && image2HasGravity)
			{
				// A fresh edge: only the relative pose carries over (num_matches is 0).
				gravityPoseGraph.Edges.Add(pairId, new PoseGraph.Edge(edge.Cam2FromCam1));
			}
		}

		int numGravityPairs = gravityPoseGraph.NumEdges;

		// Only solve if we have a meaningful subset. Skip if no gravity pairs, or if most pairs
		// (>95%) have gravity since solving the subset separately provides little benefit over
		// the full system.
		bool shouldSolve = numGravityPairs > 0 && numGravityPairs <= numTotalPairs * 0.95;
		if (!shouldSolve)
		{
			return true;
		}

		Reconstruction gravityReconstruction = reconstruction.Clone();

		// Compute largest connected component for gravity subset.
		HashSet<uint> gravityImageIds = RotationAveraging.ComputeLargestConnectedComponentImageIds(
			gravityPoseGraph, gravityReconstruction, filterUnregistered: false);
		gravityPoseGraph.InvalidatePairsOutsideActiveImageIds(gravityImageIds);

		if (!SolveRotationAveraging(gravityPoseGraph, posePriors, gravityImageIds, gravityReconstruction))
		{
			return false;
		}

		foreach ((uint gravityFrameId, Frame gravityFrame) in gravityReconstruction.Frames)
		{
			if (gravityFrame.HasPose)
			{
				reconstruction.Frame(gravityFrameId).SetRigFromWorld(gravityFrame.RigFromWorld());
			}
		}

		if (_options.RefineSensorFromRig)
		{
			foreach ((uint gravityRigId, Rig gravityRig) in gravityReconstruction.Rigs)
			{
				foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in gravityRig.NonRefSensors)
				{
					if (gravityRig.HasSensorFromRig(sensorId))
					{
						reconstruction.Rig(gravityRigId).SetSensorFromRig(sensorId, sensorFromRig);
					}
				}
			}
		}

		return true;
	}

	/// <summary>Core rotation averaging solver.</summary>
	private bool SolveRotationAveraging(
		PoseGraph poseGraph,
		IReadOnlyList<PosePrior> posePriors,
		IReadOnlySet<uint> activeImageIds,
		Reconstruction reconstruction)
	{
		// Initialize rotations from maximum spanning tree. Note that without initialization,
		// the gravity-aligned rotation averaging is prone to random flips by 180deg.
		if (!_options.SkipInitialization)
		{
			InitializeFromMaximumSpanningTree(poseGraph, activeImageIds, reconstruction);
		}

		// Build the optimization problem.
		var problem = new RotationAveragingProblem(poseGraph, posePriors, _options, activeImageIds, reconstruction);

		// Solve and apply results.
		if (!new RotationAveragingSolver(_options).Solve(problem))
		{
			return false;
		}

		problem.ApplyResultsToReconstruction(reconstruction);
		return true;
	}

	/// <summary>
	/// Compute maximum spanning tree of the pose graph weighted by inlier count. Returns the
	/// root image id and fills <paramref name="parents"/>.
	/// </summary>
	private static uint ComputeMaximumPoseGraphSpanningTree(
		PoseGraph poseGraph, IReadOnlySet<uint> imageIds, Dictionary<uint, uint> parents)
	{
		// Build mapping between image_id and contiguous indices.
		List<uint> idxToImageId = [.. imageIds.Order()];
		var imageIdToIdx = new Dictionary<uint, int>(idxToImageId.Count);
		for (int i = 0; i < idxToImageId.Count; i++)
		{
			imageIdToIdx[idxToImageId[i]] = i;
		}

		// Build edges and weights from the pose graph.
		var edges = new List<(int, int)>();
		var weights = new List<float>();
		foreach ((ulong pairId, PoseGraph.Edge edge) in poseGraph.ValidEdges().OrderBy(kv => kv.Key))
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			if (!imageIdToIdx.TryGetValue(imageId1, out int idx1) || !imageIdToIdx.TryGetValue(imageId2, out int idx2))
			{
				continue;
			}

			edges.Add((idx1, idx2));
			weights.Add(edge.NumMatches);
		}

		// Compute spanning tree using generic algorithm.
		SpanningTree tree = SpanningTrees.ComputeMaximumSpanningTree(idxToImageId.Count, edges, weights);

		// Convert back to image_id based parent map.
		parents.Clear();
		for (int i = 0; i < idxToImageId.Count; i++)
		{
			if (tree.Parents[i] >= 0)
			{
				parents[idxToImageId[i]] = idxToImageId[tree.Parents[i]];
			}
		}

		return idxToImageId[tree.Root];
	}

	/// <summary>Initializes rotations from the maximum spanning tree.</summary>
	private void InitializeFromMaximumSpanningTree(
		PoseGraph poseGraph, IReadOnlySet<uint> activeImageIds, Reconstruction reconstruction)
	{
		// Compute maximum spanning tree over active images.
		var parents = new Dictionary<uint, uint>();
		uint root = ComputeMaximumPoseGraphSpanningTree(poseGraph, activeImageIds, parents);
		Check.That(activeImageIds.Contains(root));

		// Iterate through the tree to initialize the rotation. Establish child info.
		var children = new Dictionary<uint, List<uint>>();
		foreach (uint imageId in reconstruction.Images.Keys)
		{
			if (activeImageIds.Contains(imageId))
			{
				children[imageId] = [];
			}
		}

		foreach ((uint child, uint parent) in parents)
		{
			if (root == child)
			{
				continue;
			}

			if (!children.TryGetValue(parent, out List<uint>? list))
			{
				list = [];
				children[parent] = list;
			}

			list.Add(child);
		}

		var indexes = new Queue<uint>();
		indexes.Enqueue(root);

		// The root stays at the identity rotation; like COLMAP's operator[], its entry exists
		// only once a child of the root is visited (see the header).
		var camsFromWorld = new Dictionary<uint, Rigid3d>();
		if (children.TryGetValue(root, out List<uint>? rootChildren) && rootChildren.Count > 0)
		{
			camsFromWorld[root] = new Rigid3d();
		}
		while (indexes.Count > 0)
		{
			uint curr = indexes.Dequeue();

			// Add all children into the tree.
			if (children.TryGetValue(curr, out List<uint>? currChildren))
			{
				foreach (uint child in currChildren)
				{
					indexes.Enqueue(child);
				}
			}

			// If it is root, then fix it to be the original estimation.
			if (curr == root)
			{
				continue;
			}

			// Directly use the relative pose for estimation rotation.
			// GetEdge(parent, curr) returns curr_from_parent.
			uint parent = parents[curr];
			PoseGraph.Edge edge = poseGraph.GetEdge(parent, curr);
			Rigid3d existing = camsFromWorld.TryGetValue(curr, out Rigid3d value) ? value : new Rigid3d();
			camsFromWorld[curr] = existing with { Rotation = (edge.Cam2FromCam1 * camsFromWorld[parent]).Rotation };
		}

		RotationAveraging.InitializeRigRotationsFromImages(camsFromWorld, reconstruction, _options.RefineSensorFromRig);
	}
}
