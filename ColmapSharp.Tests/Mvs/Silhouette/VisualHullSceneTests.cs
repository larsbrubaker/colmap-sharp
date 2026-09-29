// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullSceneTests: a C#-only test (not a port; COLMAP has no visual hull) of the visual hull
// on the benchmark's rendered DarkObject (Mvs/Testing/SyntheticObjectScene): 40 frames at
// 480x360 over the realistic 0.27 of the pendulum swing, with the TRUE cameras and TRUE masks.
// The hull is scored with the benchmark's own surface metrics (accuracy, completeness, F-score
// against the true mesh) and by the silhouette IoU of its reprojection. The object only turns
// about a near-vertical axis, so no view looks at the top or bottom: the hull is tight around
// the sides and loose (a ridge where the cones meet) on top and underneath, which is what drags
// accuracy and precision down. The analytic-shape tests are in VisualHullTests.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.Silhouette;

public class VisualHullSceneTests
{
	[Test]
	public async Task CSharpOnly_DarkObjectHullFromTrueCamerasAndMasks()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(
			SyntheticObjectKind.DarkObject, 40, 480, 360, seed: 1, motionDuration: 0.27);
		BenchmarkMesh truth = BenchmarkMesh.FromScene(scene);
		var views = new List<VisualHullView>();
		for (int f = 0; f < scene.Frames.Count; f++)
		{
			views.Add(new VisualHullView(scene.Camera, scene.CamFromWorld[f], scene.Masks[f]));
		}

		// The mesh vertices stand in for a sparse cloud to bound the carving.
		AlignedBox3d box = VisualHullBounds.FromPoints(scene.MeshVertices);
		VisualHull hull = VisualHull.Build(views, box, new VisualHullOptions { Resolution = 128 });
		SurfaceScores scores = SurfaceMetrics.Compute(truth, BenchmarkMesh.FromPly(hull.Mesh), null, new SurfaceMetricOptions());

		BenchmarkMesh hullMesh = BenchmarkMesh.FromPly(hull.Mesh);
		double iouSum = 0, iouMin = 1;
		for (int f = 0; f < views.Count; f++)
		{
			Bitmap rendered = SilhouetteMetrics.RenderMask(scene.Camera, scene.CamFromWorld[f], hullMesh);
			double iou = SilhouetteMetrics.Iou(rendered, scene.Masks[f]);
			iouSum += iou;
			iouMin = Math.Min(iouMin, iou);
		}

		Console.WriteLine(
			$"DarkObject hull: voxel {hull.Grid.VoxelSize:G4}, accuracy {scores.AccuracyPct:F3}%, completeness "
			+ $"{scores.CompletenessPct:F3}%, precision {scores.Precision:F3}, recall {scores.Recall:F3}, F {scores.FScore:F3} "
			+ $"(tau {scores.TauPct}%), silhouette IoU mean {iouSum / views.Count:F4} min {iouMin:F4}");

		(bool closed, bool consistent, double volume) = VisualHullTestRig.Topology(hull.Mesh);
		await Assert.That(closed && consistent && volume > 0).IsTrue();

		// Measured: accuracy 0.353%, completeness 0.271%, precision 0.924, recall 0.943, F 0.934 at
		// tau 1%; IoU mean 0.9916, min 0.9879. The bars sit just under those.
		await Assert.That(iouSum / views.Count).IsGreaterThan(0.985);
		await Assert.That(scores.Precision).IsGreaterThan(0.9);
		await Assert.That(scores.Recall).IsGreaterThan(0.92);
		await Assert.That(scores.FScore).IsGreaterThan(0.92);
	}
}
