// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipelineTests.Paths: C#-only smoke tests (labeled as such; none stands in for a
// ported case) that drive ColmapSharp/Controllers/IncrementalPipeline*.cs and
// Sfm/IncrementalMapper*.cs down paths incremental_pipeline_test.cc does not reach on its
// synthetic scenes: redundant-point global bundle adjustment with a non-empty redundant set
// (IgnorePoint and the second, points-only solve), the second initialization relaxation
// (init_min_tri_angle /= 2), the max_model_overlap break, and TriangulateReconstruction
// (COLMAP has no test of it). Each scene was checked with temporary instrumentation to take
// the path it names.
//
// Not covered: the structure-less fallback after structure-based registration found no
// image. Initialization keeps a model only with at least abs_pose_min_num_inliers 3D
// points, and a synthetic image sees every point, so each next image passes the
// structure-based candidate filter; scenes with outlier matches, 2D noise and a strict
// inlier ratio all registered structure-based. Structure-less registration itself runs in
// the ported StructureLessRegistrationOnly.
//
// Tier C (outcome): registered images and ReconstructionNear against the ground truth.

using ColmapSharp.Controllers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class IncrementalPipelineTests
{
	// C#-only: with a high minimum coverage gain, global bundle adjustment ignores redundant
	// points (IgnorePoint) and then solves for them alone in a second solve.
	[Test]
	public async Task IncrementalPipeline_IgnoreRedundantPoints3DNonEmpty()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 200 },
			gt);
		var options = new IncrementalPipelineOptions();
		options.MapperOptions.BaGlobalIgnoreRedundantPoints3D = true;
		options.MapperOptions.BaGlobalIgnoreRedundantPoints3DMinCoverageGain = 0.9;
		ReconstructionManager manager = RunPipeline(options, database);
		Require(manager.Size == 1);
		int numRegImages = manager.Get(0).NumRegImages;
		string? near = Near(gt, manager, 1e-2, 1e-4);

		await Assert.That(numRegImages).IsEqualTo(gt.NumImages);
		await Assert.That(near).IsNull();
	}

	// C#-only: no pair reaches the initial triangulation angle, nor after halving the
	// minimum inliers; halving the angle in the second relaxation step initializes the model.
	[Test]
	public async Task IncrementalPipeline_SecondInitRelaxation()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 100 },
			gt);
		var options = new IncrementalPipelineOptions();
		// 100 degrees fails (also with half the inliers); 50 initializes.
		options.MapperOptions.InitMinTriAngle = 100;
		ReconstructionManager manager = RunPipeline(options, database);
		Require(manager.Size == 1, $"Expected 1 reconstruction, got {manager.Size}");
		int numRegImages = manager.Get(0).NumRegImages;
		string? near = Near(gt, manager, 1e-2, 1e-4);

		await Assert.That(numRegImages).IsEqualTo(gt.NumImages);
		await Assert.That(near).IsNull();
	}

	// C#-only: two disconnected scenes and a provided initial pair in the first. The first
	// model reconstructs that scene; every further model starts from the same pair again, so
	// it shares images with the first model and stops at max_model_overlap, and is discarded
	// as too small. Only the first model remains.
	[Test]
	public async Task IncrementalPipeline_MaxModelOverlapStopsSharedModel()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		var gt1 = new Reconstruction();
		var gt2 = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 50 };
		Synthetic.SynthesizeDataset(syntheticOptions, gt1, database);
		syntheticOptions.NumFramesPerRig = 4;
		Synthetic.SynthesizeDataset(syntheticOptions, gt2, database);

		var options = new IncrementalPipelineOptions
		{
			InitImageId1 = 1,
			InitImageId2 = 2,
			MaxModelOverlap = 2,
			InitNumTrials = 2,
		};
		ReconstructionManager manager = RunPipeline(options, database);

		Require(manager.Size == 1, $"Expected 1 reconstruction, got {manager.Size}");
		int numRegImages = manager.Get(0).NumRegImages;
		string? near = ReconstructionMatchers.ExplainReconstructionNear(gt1, manager.Get(0), 1e-2, 1e-4);
		await Assert.That(numRegImages).IsEqualTo(5);
		await Assert.That(near).IsNull();
	}

	// C#-only (COLMAP has no test of TriangulateReconstruction): the point triangulator's
	// use. The ground-truth poses and intrinsics are kept, the points are triangulated again
	// from the database's matches and colored from the images.
	[Test]
	public async Task IncrementalPipeline_TriangulateReconstruction()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 4, NumPoints3D = 100 },
			gt);

		Reconstruction reconstruction = gt.Clone();
		reconstruction.DeleteAllPoints2DAndPoints3D();
		var reconstructionManager = new ReconstructionManager();
		reconstructionManager.Set(reconstructionManager.Add(), reconstruction);

		// The point triangulator's configuration (colmap/exe/sfm.cc, RunPointTriangulatorImpl).
		var color = new BitmapColor<byte>(20, 40, 220);
		var options = new IncrementalPipelineOptions
		{
			LoadAllImages = true,
			FixExistingFrames = true,
			BaRefineFocalLength = false,
			BaRefinePrincipalPoint = false,
			BaRefineExtraParams = false,
			ReadImage = _ =>
			{
				var bitmap = new Bitmap(1024, 768, asRgb: true);
				bitmap.Fill(color);
				return bitmap;
			},
		};
		var pipeline = new IncrementalPipeline(options, database, reconstructionManager);
		pipeline.TriangulateReconstruction(reconstruction);

		int numPoints = reconstruction.NumPoints3D;
		bool allColored = reconstruction.Points3D.Values.All(p => p.Color == new Vector3ub(color.R, color.G, color.B));
		string? near = ReconstructionMatchers.ExplainReconstructionNear(gt, reconstruction, 1e-6, 1e-6, align: false);

		await Assert.That(numPoints).IsEqualTo(gt.NumPoints3D);
		await Assert.That(allColored).IsTrue();
		await Assert.That(near).IsNull();
	}
}
