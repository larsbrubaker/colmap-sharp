// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkEvaluator: scores one reconstruction of a SyntheticObjectScene against its truth
// (docs/QUALITY_PLAN.md, stage 0b). Not a COLMAP port. It picks the model to score (the one
// with the most registered frames; a split shows as NumModels > 1 and a lower registered
// fraction), then combines the metric pieces of this folder:
// - registered fraction and model count;
// - pose error after a robust Sim3 (PoseMetrics);
// - accuracy, completeness and F-score of the mesh, or of the fused points when there is no
//   mesh, mapped into the truth frame with that Sim3 (SurfaceMetrics);
// - silhouette IoU of the mesh through each registered frame's estimated camera
//   (SilhouetteMetrics);
// - the focal ratio, estimated / true mean focal length of the scored model's first registered
//   frame's camera (the scenes share one camera). Bundle adjustment on a small object in a
//   narrow view can collapse the focal (divergence 141), so it is scored as |ln ratio|.
// BenchmarkMetrics.Values flattens the result into named numbers, each with the direction that
// is better, which is what the runner aggregates over seeds and writes as JSON.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>One named benchmark number and which direction is better.</summary>
public readonly record struct BenchmarkValue(string Name, double Value, bool HigherIsBetter);

/// <summary>All scores of one reconstruction of a synthetic scene.</summary>
public sealed record BenchmarkMetrics
{
	/// <summary>Frames in the scene.</summary>
	public int NumFrames { get; init; }

	/// <summary>Models the mapper produced (more than one is a split).</summary>
	public int NumModels { get; init; }

	/// <summary>Frames registered in the scored (largest) model.</summary>
	public int NumRegistered { get; init; }

	/// <summary>NumRegistered / NumFrames.</summary>
	public double RegisteredFraction => NumFrames == 0 ? 0 : NumRegistered / (double)NumFrames;

	/// <summary>Pose error of the scored model's frames.</summary>
	public PoseScores Pose { get; init; } = new();

	/// <summary>Surface scores ("mesh", "points" or "none" in <see cref="SurfaceSource"/>).</summary>
	public SurfaceScores Surface { get; init; } = new();

	/// <summary>What <see cref="Surface"/> was computed from.</summary>
	public string SurfaceSource { get; init; } = "none";

	/// <summary>Mean silhouette IoU over the registered frames (NaN without a mesh).</summary>
	public double SilhouetteIouMean { get; init; } = double.NaN;

	/// <summary>Lowest silhouette IoU over the registered frames (NaN without a mesh).</summary>
	public double SilhouetteIouMin { get; init; } = double.NaN;

	/// <summary>Estimated / true mean focal length (NaN when nothing registered).</summary>
	public double FocalRatio { get; init; } = double.NaN;

	/// <summary>The scores as named numbers, in a fixed order.</summary>
	public IReadOnlyList<BenchmarkValue> Values() =>
	[
		new("registered_fraction", RegisteredFraction, true),
		new("num_models", NumModels, false),
		new("focal_log_error", Math.Abs(Math.Log(FocalRatio)), false),
		new("rotation_error_median_deg", Pose.MedianRotationDeg, false),
		new("rotation_error_max_deg", Pose.MaxRotationDeg, false),
		new("position_error_median_pct", Pose.MedianPositionPct, false),
		new("position_error_max_pct", Pose.MaxPositionPct, false),
		new("accuracy_pct", Surface.AccuracyPct, false),
		new("completeness_pct", Surface.CompletenessPct, false),
		new("precision", Surface.Precision, true),
		new("recall", Surface.Recall, true),
		new("f_score", Surface.FScore, true),
		new("excluded_fraction", Surface.ExcludedFraction, false),
		new("silhouette_iou_mean", SilhouetteIouMean, true),
		new("silhouette_iou_min", SilhouetteIouMin, true),
	];
}

/// <summary>Scores a reconstruction of a <see cref="SyntheticObjectScene"/>.</summary>
public static class BenchmarkEvaluator
{
	/// <summary>
	/// The index of the model to score: the most registered images, the lowest index on a tie;
	/// -1 when there are no models.
	/// </summary>
	public static int SelectModel(IReadOnlyList<Reconstruction> models)
	{
		int best = -1;
		for (int i = 0; i < models.Count; i++)
		{
			if (best < 0 || models[i].NumRegImages > models[best].NumRegImages)
			{
				best = i;
			}
		}

		return best;
	}

	/// <summary>
	/// Scores <paramref name="models"/> against <paramref name="scene"/>, whose frame k was fed
	/// to the reconstruction under the name <paramref name="frameNames"/>[k]. The surface is
	/// <paramref name="mesh"/>, or <paramref name="points"/> when there is no mesh, and must
	/// belong to the model <see cref="SelectModel"/> picks, in that model's frame.
	/// </summary>
	public static BenchmarkMetrics Evaluate(
		SyntheticObjectScene scene,
		IReadOnlyList<string> frameNames,
		IReadOnlyList<Reconstruction> models,
		BenchmarkMesh? mesh,
		IReadOnlyList<Vector3d>? points,
		SurfaceMetricOptions? surfaceOptions = null)
	{
		Check.Eq(frameNames.Count, scene.Frames.Count);
		surfaceOptions ??= new SurfaceMetricOptions();
		BenchmarkMesh truth = BenchmarkMesh.FromScene(scene);
		double diagonal = truth.Diagonal();

		int modelIdx = SelectModel(models);
		if (modelIdx < 0)
		{
			return new BenchmarkMetrics
			{
				NumFrames = scene.Frames.Count,
				Surface = SurfaceMetrics.Compute(truth, null, null, surfaceOptions),
			};
		}

		Reconstruction model = models[modelIdx];
		var frameIndex = new Dictionary<string, int>();
		for (int k = 0; k < frameNames.Count; k++)
		{
			frameIndex[frameNames[k]] = k;
		}

		// Registered frames in frame order, so the pose inputs do not depend on image ids.
		var registered = new List<(int Frame, Scene.Image Image)>();
		foreach (uint imageId in model.RegImageIds())
		{
			Scene.Image image = model.Image(imageId);
			if (frameIndex.TryGetValue(image.Name, out int k))
			{
				registered.Add((k, image));
			}
		}

		registered.Sort((a, b) => a.Frame.CompareTo(b.Frame));

		PoseScores pose = PoseMetrics.Compute(
			[.. registered.Select(r => r.Image.CamFromWorld())],
			[.. registered.Select(r => scene.CamFromWorld[r.Frame])],
			diagonal);

		SurfaceScores surface;
		string source;
		if (!pose.Aligned || (mesh is null && points is null))
		{
			surface = SurfaceMetrics.Compute(truth, null, null, surfaceOptions);
			source = "none";
		}
		else if (mesh is not null)
		{
			surface = SurfaceMetrics.Compute(truth, mesh.Transformed(pose.TruthFromEstimate), null, surfaceOptions);
			source = "mesh";
		}
		else
		{
			Sim3d toTruth = pose.TruthFromEstimate;
			surface = SurfaceMetrics.Compute(truth, null, [.. points!.Select(p => toTruth * p)], surfaceOptions);
			source = "points";
		}

		double iouMean = double.NaN, iouMin = double.NaN;
		if (mesh is not null && registered.Count > 0)
		{
			var ious = new double[registered.Count];
			Parallel.For(0, registered.Count, i =>
			{
				(int frame, Scene.Image image) = registered[i];
				Bitmap rendered = SilhouetteMetrics.RenderMask(model.Camera(image.CameraId), image.CamFromWorld(), mesh);
				ious[i] = SilhouetteMetrics.Iou(rendered, scene.Masks[frame]);
			});

			double sum = 0;
			foreach (double iou in ious)
			{
				sum += iou;
			}

			iouMean = sum / ious.Length;
			iouMin = ious.Min();
		}

		double focalRatio = registered.Count == 0 ? double.NaN
			: model.Camera(registered[0].Image.CameraId).MeanFocalLength() / scene.Camera.MeanFocalLength();
		return new BenchmarkMetrics
		{
			NumFrames = scene.Frames.Count,
			NumModels = models.Count,
			FocalRatio = focalRatio,
			NumRegistered = registered.Count,
			Pose = pose,
			Surface = surface,
			SurfaceSource = source,
			SilhouetteIouMean = iouMean,
			SilhouetteIouMin = iouMin,
		};
	}
}
