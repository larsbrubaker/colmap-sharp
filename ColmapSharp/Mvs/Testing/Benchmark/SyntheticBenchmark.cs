// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticBenchmark: runs AutomaticReconstructionController on one SyntheticObjectScene and
// scores the result (docs/QUALITY_PLAN.md, stage 0b). Not a COLMAP port. The frames go in as an
// in-memory image source named frame000.png, frame001.png, ..., with one SIMPLE_PINHOLE
// camera; the workspace is a fresh temporary folder, deleted afterwards (or the caller's, kept). The controller's
// stages are timed with StageTimer, and the scored model's dense mesh (or its fused points when
// meshing gave nothing) is read back from dense/<model>/ for BenchmarkEvaluator. The ColmapSharp.Benchmarks runner calls this once
// per (scene, mapper seed), and the test suite's end-to-end check calls it too, so both measure
// the same thing.

using System.Diagnostics;
using System.Globalization;

using ColmapSharp.Controllers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>Settings of one <see cref="SyntheticBenchmark.Run"/>.</summary>
public sealed class SyntheticBenchmarkOptions
{
	/// <summary>The reconstruction quality preset.</summary>
	public AutomaticReconstructionOptions.QualityLevel Quality { get; set; } = AutomaticReconstructionOptions.QualityLevel.Low;

	/// <summary>Individual (exhaustive matching) or Video (sequential).</summary>
	public AutomaticReconstructionOptions.DataType Data { get; set; } = AutomaticReconstructionOptions.DataType.Individual;

	/// <summary>The mapper's (and RANSAC's) random seed, AutomaticReconstructionOptions.RandomSeed.</summary>
	public int MapperSeed { get; set; } = 1;

	/// <summary>Whether all frames share one camera (they do in the scenes).</summary>
	public bool SingleCamera { get; set; } = true;

	/// <summary>Whether to run the dense stages (without them there is no surface to score).</summary>
	public bool Dense { get; set; } = true;

	/// <summary>The mesher.</summary>
	public AutomaticReconstructionOptions.MesherType Mesher { get; set; } = AutomaticReconstructionOptions.MesherType.Poisson;

	/// <summary>Poisson octree depth (the demo's default).</summary>
	public int PoissonDepth { get; set; } = 11;

	/// <summary>Poisson density trim (the demo's default).</summary>
	public double PoissonTrim { get; set; } = 5;

	/// <summary>
	/// Known intrinsics: give the reconstruction the scene's true focal and principal point and
	/// keep bundle adjustment from refining them, as with a calibrated camera or a trusted EXIF
	/// focal. Off by default: the default run calibrates itself, like a photo set without EXIF.
	/// </summary>
	public bool KnownIntrinsics { get; set; }

	/// <summary>
	/// Scene or Object (AutomaticReconstructionOptions.Subject). Object without true masks segments
	/// the frames itself (SilhouetteSegmenter), as a host would.
	/// </summary>
	public AutomaticReconstructionOptions.SubjectType Subject { get; set; } = AutomaticReconstructionOptions.SubjectType.Scene;

	/// <summary>Whether to hand the scene's true masks to the reconstruction (features and fusion).</summary>
	public bool UseTrueMasks { get; set; }

	/// <summary>Whether to add KLT video tracks to the matches (AutomaticReconstructionOptions.VideoTracking; video data only).</summary>
	public bool VideoTracking { get; set; }

	/// <summary>How <see cref="VideoTracking"/> tracks (null: the defaults).</summary>
	public Feature.Tracking.VideoTrackingOptions? VideoTrackingOptions { get; set; }

	/// <summary>Threads for every stage (-1: all).</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Options of the surface scores.</summary>
	public SurfaceMetricOptions Surface { get; set; } = new();

	/// <summary>
	/// The workspace folder to use and keep, or null (the default) for a fresh temporary folder
	/// that is deleted afterwards. A kept workspace lets a test inspect or resume a run.
	/// </summary>
	public string? WorkspacePath { get; set; }

	/// <summary>Also receives the controller's progress (synchronously, on the reporting thread).</summary>
	public IProgress<ControllerProgress>? Progress { get; set; }
}

/// <summary>The scores and stage timings of one benchmark run.</summary>
public sealed record SyntheticBenchmarkResult(
	BenchmarkMetrics Metrics,
	IReadOnlyList<(string Stage, double Seconds)> StageSeconds,
	double TotalSeconds)
{
	/// <summary>Whether the scored mesh is closed (every edge on two faces); false without a mesh.</summary>
	public bool MeshClosed { get; init; }

	/// <summary>
	/// Whether the scored mesh is object mode's visual hull itself (its fallback when the dense
	/// stages gave no mesh), rather than a mesh of the dense points.
	/// </summary>
	public bool MeshIsVisualHull { get; init; }

	/// <summary>
	/// The scored model's registered cameras (CamFromWorld) by frame index, so two runs can be
	/// compared on the frames both placed (PoseMetrics.Compute on the common subset).
	/// </summary>
	public IReadOnlyDictionary<int, Rigid3d> RegisteredPoses { get; init; } = new Dictionary<int, Rigid3d>();
}

/// <summary>Reconstructs a synthetic scene with the full controller and scores it.</summary>
public static class SyntheticBenchmark
{
	/// <summary>The name frame <paramref name="k"/> is fed to the reconstruction under.</summary>
	public static string FrameName(int k) => "frame" + k.ToString("D3", CultureInfo.InvariantCulture) + ".png";

	/// <summary>Runs the reconstruction of <paramref name="scene"/> and scores it.</summary>
	public static SyntheticBenchmarkResult Run(
		SyntheticObjectScene scene,
		SyntheticBenchmarkOptions options,
		CancellationToken cancellationToken = default)
	{
		Check.NotNull(scene);
		Check.NotNull(options);
		var frameNames = new string[scene.Frames.Count];
		var images = new InMemoryImageSource();
		var masks = new InMemoryImageSource();
		for (int k = 0; k < frameNames.Length; k++)
		{
			frameNames[k] = FrameName(k);
			images.Add(frameNames[k], scene.Frames[k]);
			masks.Add(frameNames[k] + ".png", scene.Masks[k]);
		}

		string workspace = options.WorkspacePath ?? Path.Combine(Path.GetTempPath(), "colmap-sharp-benchmark-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(workspace);
		try
		{
			var reconOptions = new AutomaticReconstructionOptions
			{
				WorkspacePath = workspace,
				Images = images,
				Masks = options.UseTrueMasks ? masks : null,
				Subject = options.Subject,
				Data = options.Data,
				VideoTracking = options.VideoTracking,
				Quality = options.Quality,
				SingleCamera = options.SingleCamera,
				// SIMPLE_PINHOLE, one focal length: the scenes' camera has fx = fy, and with PINHOLE's
				// two free focals bundle adjustment on a small object in a narrow view traded them
				// off (fx 196 against fy 940 on a 16-frame sphere run), which says nothing about the
				// reconstruction the benchmark means to measure.
				CameraModel = "SIMPLE_PINHOLE",
				CameraParams = options.KnownIntrinsics ? TrueSimplePinholeParams(scene) : "",
				BaRefineFocalLength = !options.KnownIntrinsics,
				BaRefineExtraParams = !options.KnownIntrinsics,
				Dense = options.Dense,
				Mesher = options.Mesher,
				Texture = false,
				NumThreads = options.NumThreads,
				RandomSeed = options.MapperSeed,
			};
			if (options.VideoTrackingOptions is not null)
			{
				reconOptions.VideoTrackingOptions = options.VideoTrackingOptions;
			}

			reconOptions.PoissonMeshing.Depth = options.PoissonDepth;
			reconOptions.PoissonMeshing.Trim = options.PoissonTrim;

			var models = new ReconstructionManager();
			var timer = new StageTimer();
			var controller = new AutomaticReconstructionController(reconOptions, models)
			{
				Progress = options.Progress is null ? timer : new TeeProgress(timer, options.Progress),
				CancellationToken = cancellationToken,
			};
			var clock = Stopwatch.StartNew();
			controller.Setup();
			controller.Run();
			double total = clock.Elapsed.TotalSeconds;
			IReadOnlyList<(string Stage, double Seconds)> stages = timer.Stages();

			var reconstructions = Enumerable.Range(0, models.Size).Select(models.Get).ToList();
			int modelIdx = BenchmarkEvaluator.SelectModel(reconstructions);
			(BenchmarkMesh? mesh, List<Vector3d>? points, bool closed) = modelIdx < 0 ? (null, null, false) : ReadSurface(workspace, modelIdx, options.Mesher);
			BenchmarkMetrics metrics = BenchmarkEvaluator.Evaluate(scene, frameNames, reconstructions, mesh, points, options.Surface);
			var poses = new Dictionary<int, Rigid3d>();
			if (modelIdx >= 0)
			{
				Reconstruction model = reconstructions[modelIdx];
				foreach (uint imageId in model.RegImageIds())
				{
					int k = Array.IndexOf(frameNames, model.Image(imageId).Name);
					if (k >= 0)
					{
						poses[k] = model.Image(imageId).CamFromWorld();
					}
				}
			}

			return new SyntheticBenchmarkResult(metrics, stages, total) {
				MeshClosed = closed,
				MeshIsVisualHull = modelIdx >= 0 && IsVisualHullMesh(workspace, modelIdx, options.Mesher),
				RegisteredPoses = poses,
			};
		}
		finally
		{
			try
			{
				if (options.WorkspacePath is null)
				{
					Directory.Delete(workspace, recursive: true);
				}
			}
			catch (IOException)
			{
				// A leftover temp folder is harmless.
			}
		}
	}

	// The scene's PINHOLE camera (fx = fy) as SIMPLE_PINHOLE parameters "f, cx, cy".
	private static string TrueSimplePinholeParams(SyntheticObjectScene scene)
	{
		Check.That(scene.Camera.Params[0] == scene.Camera.Params[1], "The scene camera must have fx = fy");
		return string.Join(",", new[] { scene.Camera.Params[0], scene.Camera.Params[2], scene.Camera.Params[3] }
			.Select(v => v.ToString("R", CultureInfo.InvariantCulture)));
	}

	// Whether dense/<model>'s mesh has exactly the vertices and faces of its visual-hull.ply.
	private static bool IsVisualHullMesh(string workspace, int modelIdx, AutomaticReconstructionOptions.MesherType mesher)
	{
		string densePath = Path.Combine(workspace, "dense", modelIdx.ToString(CultureInfo.InvariantCulture));
		string meshPath = Path.Combine(densePath, mesher == AutomaticReconstructionOptions.MesherType.Delaunay ? "meshed-delaunay.ply" : "meshed-poisson.ply");
		string hullPath = Path.Combine(densePath, AutomaticReconstructionController.VisualHullMeshName);
		if (!File.Exists(meshPath) || !File.Exists(hullPath))
		{
			return false;
		}

		PlyMesh mesh = Ply.ReadPlyMesh(meshPath).Mesh, hull = Ply.ReadPlyMesh(hullPath).Mesh;
		return mesh.Faces.SequenceEqual(hull.Faces)
			&& mesh.Vertices.Select(v => (v.X, v.Y, v.Z)).SequenceEqual(hull.Vertices.Select(v => (v.X, v.Y, v.Z)));
	}

	private sealed class TeeProgress(IProgress<ControllerProgress> first, IProgress<ControllerProgress> second) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value)
		{
			first.Report(value);
			second.Report(value);
		}
	}

	// The model's mesh when meshing produced faces, else its fused points when there are any.
	private static (BenchmarkMesh? Mesh, List<Vector3d>? Points, bool Closed) ReadSurface(string workspace, int modelIdx, AutomaticReconstructionOptions.MesherType mesher)
	{
		string densePath = Path.Combine(workspace, "dense", modelIdx.ToString(CultureInfo.InvariantCulture));
		string meshPath = Path.Combine(densePath, mesher == AutomaticReconstructionOptions.MesherType.Delaunay ? "meshed-delaunay.ply" : "meshed-poisson.ply");
		if (File.Exists(meshPath))
		{
			PlyTexturedMesh ply = Ply.ReadPlyMesh(meshPath);
			if (ply.Mesh.Faces.Count > 0)
			{
				return (BenchmarkMesh.FromPly(ply.Mesh), null, Silhouette.HullSurfaceFusion.IsClosed(ply.Mesh));
			}
		}

		string fusedPath = Path.Combine(densePath, "fused.ply");
		if (File.Exists(fusedPath))
		{
			List<PlyPoint> fused = Ply.ReadPly(fusedPath);
			if (fused.Count > 0)
			{
				return (null, [.. fused.Select(p => new Vector3d(p.X, p.Y, p.Z))], false);
			}
		}

		return (null, null, false);
	}
}
