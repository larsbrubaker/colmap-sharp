// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPipelineTests.Components: the multi-component cases of
// colmap/controllers/global_pipeline_test.cc (ReconstructOnlyLargestComponent onwards),
// ported 1:1. Helpers and conventions are in GlobalPipelineTests.cs.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class GlobalPipelineTests
{
	// End-to-end: a database whose view graph splits into two disconnected components should
	// yield one reconstruction per component, each registering exactly that component's
	// images.
	[Test]
	public async Task GlobalPipeline_MultiComponents()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = false,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		// Split the images into two groups (one per rig) and cut all cross-group matches so
		// the view graph decomposes into two connected components. Grouping by rig keeps each
		// component's ground truth well-defined.
		List<HashSet<uint>> expectedComponents = GroupImageIdsByRig(gt);
		Require(expectedComponents.Count == 2);
		Require(expectedComponents[0].Count == 5);
		Require(expectedComponents[1].Count == 5);
		DisconnectDatabaseComponents(expectedComponents, database);

		var options = new GlobalPipelineOptions();
		Require(options.MultipleModels);
		ViewGraphCalibration.CalibrateViewGraph(new ViewGraphCalibrationOptions(), database);
		ReconstructionManager reconstructionManager = RunPipeline(options, database);

		// Expect one reconstruction per component, each covering its own images.
		Require(reconstructionManager.Size == 2);
		bool sameSets = SameSets(RegImageIdSetsPerReconstruction(reconstructionManager), expectedComponents);

		// Each recovered component must also match the ground truth of its cluster.
		var mismatches = new List<string>();
		for (int i = 0; i < reconstructionManager.Size; ++i)
		{
			Reconstruction reconstruction = reconstructionManager.Get(i);
			HashSet<uint> reconstructionImageIds = [.. reconstruction.RegImageIds()];
			HashSet<uint>? group = expectedComponents.Find(c => c.SetEquals(reconstructionImageIds));
			Require(group is not null);
			Reconstruction gtSubset = ExtractGroundTruthSubset(gt, group!);
			string? near = ReconstructionMatchers.ExplainReconstructionNear(gtSubset, reconstruction, 1e-2, 1e-4);
			if (near is not null)
			{
				mismatches.Add(near);
			}
		}

		await Assert.That(sameSets).IsTrue();
		await Assert.That(mismatches).IsEmpty();
	}

	[Test]
	public async Task GlobalPipeline_ReconstructOnlyLargestComponent()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		DisconnectDatabaseComponents(GroupImageIdsByRig(gt), database);

		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions { MultipleModels = false }, database);

		Require(reconstructionManager.Size == 1);
		await Assert.That(reconstructionManager.Get(0).NumRegFrames).IsEqualTo(5);
	}

	// The current component must be visible through the reconstruction manager while
	// callbacks run, and a stop request must prevent subsequent components from starting.
	[Test]
	public async Task GlobalPipeline_MultiComponentsStopAfterFirstComponent()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		DisconnectDatabaseComponents(GroupImageIdsByRig(gt), database);

		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(new GlobalPipelineOptions(), database, reconstructionManager);

		bool stopRequested = false;
		bool callbackSawInProgressReconstruction = false;
		mapper.AddCallback((int)GlobalPipeline.CallbackType.ModelUpdateCallback, () =>
		{
			callbackSawInProgressReconstruction =
				reconstructionManager.Size == 1 && reconstructionManager.Get(0).NumRegFrames > 0;
			stopRequested = true;
		});
		mapper.SetCheckIfStoppedFunc(() => stopRequested);
		mapper.Run();

		await Assert.That(callbackSawInProgressReconstruction).IsTrue();
		await Assert.That(stopRequested).IsTrue();
		await Assert.That(reconstructionManager.Size).IsEqualTo(1);
	}

	// Components that cannot meet min_model_size are discarded before invoking the expensive
	// global mapper.
	[Test]
	public async Task GlobalPipeline_MultiComponentsBelowMinModelSize()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 2,
				NumPoints3D = 20,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		DisconnectDatabaseComponents(GroupImageIdsByRig(gt), database);

		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(new GlobalPipelineOptions { MinModelSize = 3 }, database, reconstructionManager);
		bool callbackCalled = false;
		mapper.AddCallback((int)GlobalPipeline.CallbackType.ModelUpdateCallback, () => callbackCalled = true);
		mapper.Run();

		await Assert.That(callbackCalled).IsFalse();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	// Components that become too small only after rotation filtering must also be discarded
	// before invoking the full global mapper. In particular, the rejected bridge edges must
	// not reconnect the residual components and cause repeated mapping attempts.
	[Test]
	public async Task GlobalPipeline_MultiComponentsBelowMinModelSizeAfterRotationFiltering()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 3,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
				PriorGravity = true,
			},
			gt);

		List<HashSet<uint>> expectedComponents = GroupImageIdsByRig(gt);
		Require(expectedComponents.Count == 2);
		Require(expectedComponents[0].Count == 3);
		Require(expectedComponents[1].Count == 3);

		// Keep and corrupt every cross-component edge. Gravity anchors the two groups, so
		// rotation filtering rejects the bridges and recovers two three-frame components from
		// one initial six-frame component.
		BridgeGroupsWithOutlierEdges(expectedComponents, gt.NumImages * gt.NumImages, database);

		// First verify that the setup is successfully decomposed into the expected components
		// when they meet the minimum size.
		var baselineOptions = new GlobalPipelineOptions { RandomSeed = 1, MinModelSize = 3 };
		baselineOptions.Mapper.RotationAveragingOptions.UseGravity = true;
		ReconstructionManager baselineManager = RunPipeline(baselineOptions, database);
		Require(baselineManager.Size == 2);
		bool baselineSets = SameSets(RegImageIdSetsPerReconstruction(baselineManager), expectedComponents);

		var options = new GlobalPipelineOptions { RandomSeed = 1, MinModelSize = 4 };
		options.Mapper.RotationAveragingOptions.UseGravity = true;
		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(options, database, reconstructionManager);
		bool callbackCalled = false;
		mapper.AddCallback((int)GlobalPipeline.CallbackType.ModelUpdateCallback, () => callbackCalled = true);
		mapper.Run();

		await Assert.That(baselineSets).IsTrue();
		await Assert.That(callbackCalled).IsFalse();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	// Multi-camera rigs with unknown sensor_from_rig must be calibrated independently in each
	// disconnected component.
	[Test]
	public async Task GlobalPipeline_MultiComponentsWithUnknownSensorFromRig()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 2,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		List<HashSet<uint>> expectedComponents = GroupImageIdsByRig(gt);
		Require(expectedComponents.Count == 2);
		DisconnectDatabaseComponents(expectedComponents, database);

		foreach (Rig rig in database.ReadAllRigs())
		{
			foreach (SensorId sensorId in rig.SensorIds())
			{
				if (!rig.IsRefSensor(sensorId))
				{
					rig.ResetSensorFromRig(sensorId);
				}
			}

			database.UpdateRig(rig);
		}

		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		Require(reconstructionManager.Size == 2);
		await Assert.That(SameSets(RegImageIdSetsPerReconstruction(reconstructionManager), expectedComponents)).IsTrue();
	}

	// Splits the sorted registered image ids of the ground truth into two halves.
	private static List<HashSet<uint>> SplitInTwoHalves(Reconstruction gt)
	{
		List<uint> imageIds = gt.RegImageIds();
		imageIds.Sort();
		Require(imageIds.Count == 10);
		return [[.. imageIds.Take(5)], [.. imageIds.Skip(5)]];
	}

	// End-to-end: two clusters bridged only by a couple of outlier edges (with bogus relative
	// rotations) are still recovered as two separate reconstructions. The two
	// mutually-inconsistent bridges cannot both be satisfied by any global rotation solution,
	// so rotation averaging leaves each with a large residual and
	// FilterEdgesByRelativeRotation removes them, splitting the view graph.
	[Test]
	public async Task GlobalPipeline_MultiComponentsWithOutlierEdges()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				// Known focal lengths let us skip view graph calibration, which would otherwise
				// re-estimate (and thereby "fix") the injected outlier edges.
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		List<HashSet<uint>> expectedComponents = SplitInTwoHalves(gt);

		// Connect the two groups only through two outlier bridge edges.
		BridgeGroupsWithOutlierEdges(expectedComponents, 2, database);

		var options = new GlobalPipelineOptions();
		Require(options.MultipleModels);
		ReconstructionManager reconstructionManager = RunPipeline(options, database);

		// The outlier bridges must be rejected, recovering the two clusters.
		Require(reconstructionManager.Size == 2);
		await Assert.That(SameSets(RegImageIdSetsPerReconstruction(reconstructionManager), expectedComponents)).IsTrue();
	}

	// End-to-end (gravity variant): even when *every* cross-cluster edge is a bogus-rotation
	// outlier - a regime the default gravity-free solver cannot resolve because the
	// inter-cluster orientation gauge is free - gravity priors anchor each cluster to the
	// vertical. The random bridge rotations then exceed the rotation error threshold and are
	// filtered, recovering the two clusters.
	[Test]
	public async Task GlobalPipeline_MultiComponentsWithOutlierEdgesUsingGravity()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 100,
				// Known focal lengths let us skip view graph calibration, which would otherwise
				// re-estimate (and thereby "fix") the injected outlier edges.
				CameraHasPriorFocalLength = true,
				TwoViewGeometryHasRelativePose = true,
				PriorGravity = true,
			},
			gt);

		List<HashSet<uint>> expectedComponents = SplitInTwoHalves(gt);

		// Corrupt every cross-cluster edge into an outlier bridge (passing a count larger than
		// the number of cross pairs keeps and randomizes all of them).
		BridgeGroupsWithOutlierEdges(expectedComponents, 10 * 10, database);

		var options = new GlobalPipelineOptions();
		Require(options.MultipleModels);
		// Gravity priors pin each cluster to the vertical, making the random-rotation bridges
		// detectable regardless of the otherwise free inter-cluster gauge.
		options.Mapper.RotationAveragingOptions.UseGravity = true;
		ReconstructionManager reconstructionManager = RunPipeline(options, database);

		// Despite every cross edge being an outlier, gravity lets rotation averaging filter
		// them all and recover the two clusters.
		Require(reconstructionManager.Size == 2);
		await Assert.That(SameSets(RegImageIdSetsPerReconstruction(reconstructionManager), expectedComponents)).IsTrue();
	}

	// End-to-end: with no matches at all, the view graph is empty and the pipeline produces
	// no reconstructions.
	[Test]
	public async Task GlobalPipeline_MultiComponentsEmptyViewGraph()
	{
		RandomUtils.SetPRNGSeed(1);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 3,
				NumPoints3D = 20,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		// Remove all matches so the view graph is empty.
		database.ClearTwoViewGeometries();
		database.ClearMatches();

		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}
}
