// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReconstructionSession: one photos-to-mesh run, without any UI - decode the photos
// (PhotoDecoder.cs), run the library's AutomaticReconstructionController in a fresh temp
// workspace, and hand back the sparse points and the (textured) mesh, converted for the
// viewport by MeshBridge.cs. ColmapDemoApp drives it from a background task on the Mac, and on
// the browser's one thread (where SessionSettings.YieldAsync hands the page its turns), and
// marshals its events to the UI thread; tests and a headless try-out drive it directly.
// SaveMesh writes the result as OBJ (+ MTL + <name>.png) or PLY; SaveMeshZip puts the OBJ's
// files in one zip, since a browser download is one file. Each run's temp workspace is
// deleted when the run ends (COLMAP_DEMO_KEEP_WORKSPACE=1 keeps it).
//
// Its events fire on whatever thread the run is on (the worker, on the Mac); listeners marshal.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;
using MatterHackers.Agg.Image;
using MatterHackers.PolygonMesh;

namespace ColmapDemo
{
	/// <summary>The demo's run settings. Defaults are chosen for a quick run on a small photo set.</summary>
	public sealed class SessionSettings
	{
		/// <summary>
		/// Photos are shrunk so their longer side is at most this many pixels before anything
		/// else sees them. The Low preset already caps SIFT (3200 * 0.3125 = 1000) and PatchMatch
		/// and fusion (1000), so shrinking up front mostly saves decode memory and upload time.
		/// </summary>
		public int MaxImageSize { get; set; } = 1000;

		/// <summary>The library's quality preset; Low keeps a demo run in the tens of seconds.</summary>
		public AutomaticReconstructionOptions.QualityLevel Quality { get; set; } = AutomaticReconstructionOptions.QualityLevel.Low;

		/// <summary>
		/// Poisson trim. COLMAP's 10 cuts the sparse fused cloud of a handful of photos down to
		/// nothing (PORTING_PLAN.md Phase 11); 5 keeps the surface near the samples while still
		/// dropping most of the far-flung "balloon" Poisson closes the surface with.
		/// </summary>
		public double PoissonTrim { get; set; } = 5;

		/// <summary>
		/// Poisson octree depth (COLMAP's default is 13). Measured on 6 synthetic 320x240 views
		/// (8104 fused points, trim 5): depth 13 meshed in 48.8 s, depth 11 in 10.9 s with the very
		/// same mesh (60805 faces; the octree adapts to the sampling, so the extra levels were
		/// never used), depth 10 in 7.7 s with 60697 faces. On all three, 99.5% of the fused points
		/// lie within 1% of the model's size from a mesh vertex. 11 keeps the full result on this set
		/// and leaves headroom for the denser clouds of 1000 px photos, at a quarter of the time.
		/// </summary>
		public int PoissonDepth { get; set; } = 11;

		/// <summary>Where each run's temp workspace folder is made.</summary>
		public string WorkspaceRoot { get; set; } = Path.Combine(Path.GetTempPath(), "ColmapDemo");

		/// <summary>
		/// Keep the run's workspace (depth maps, fused.ply, meshes) instead of deleting it when the
		/// run ends: a developer option, on when COLMAP_DEMO_KEEP_WORKSPACE=1.
		/// </summary>
		public bool KeepWorkspace { get; set; } = Environment.GetEnvironmentVariable("COLMAP_DEMO_KEEP_WORKSPACE") == "1";

		/// <summary>The GPU for PatchMatch, or null to run it on the CPU.</summary>
		public IComputeDevice ComputeDevice { get; set; }

		/// <summary>
		/// What the run awaits between units of work (each decoded photo, each pipeline stage, each
		/// PatchMatch problem), or null for <see cref="Task.Yield"/>. A host whose run shares the UI
		/// thread (the browser) supplies one that really gives the page a turn to paint.
		/// </summary>
		public Func<ValueTask> YieldAsync { get; set; }
	}

	/// <summary>What a finished run produced.</summary>
	public sealed class SessionResult
	{
		/// <summary>The sparse points of every model.</summary>
		public IReadOnlyList<ColoredPoint> SparsePoints { get; init; } = Array.Empty<ColoredPoint>();

		/// <summary>The first model's mesh as the library wrote it, or null when meshing gave none.</summary>
		public PlyMesh Mesh { get; init; }

		/// <summary>The mesh's per-corner UVs (6 per face) when textured, else null.</summary>
		public float[] FaceUvs { get; init; }

		/// <summary>The texture atlas, or null when no face was seen by any photo.</summary>
		public ImageBuffer Atlas { get; init; }

		/// <summary>The mesh converted for the viewport, or null.</summary>
		public Mesh PreviewMesh { get; init; }

		/// <summary>The workspace the run wrote into; deleted when the run ended unless KeepWorkspace.</summary>
		public string WorkspacePath { get; init; }

		/// <summary>Whether a textured mesh came out.</summary>
		public bool IsTextured => this.Atlas != null;
	}

	/// <summary>One photos-to-mesh run.</summary>
	public sealed class ReconstructionSession
	{
		private readonly SessionSettings settings;

		public ReconstructionSession(SessionSettings settings)
		{
			this.settings = settings ?? new SessionSettings();
		}

		/// <summary>Every progress report of the library (Stage is one of ColmapDemoApp.Stages).</summary>
		public event Action<ControllerProgress> ProgressChanged;

		/// <summary>The sparse model's points, as soon as the sparse stage has finished.</summary>
		public event Action<IReadOnlyList<ColoredPoint>> SparseReady;

		/// <summary>
		/// Decodes <paramref name="photoPaths"/> and runs the whole pipeline, synchronously on the
		/// calling thread between awaits (call it from a background task in a desktop UI).
		/// Throws <see cref="OperationCanceledException"/> when <paramref name="cancel"/> fires.
		/// </summary>
		public async Task<SessionResult> RunAsync(IReadOnlyList<string> photoPaths, CancellationToken cancel)
		{
			Func<ValueTask> yieldToHost = this.settings.YieldAsync ?? YieldToEventLoop;
			var images = new InMemoryImageSource();
			var names = new HashSet<string>(StringComparer.Ordinal);
			for (int i = 0; i < photoPaths.Count; i++)
			{
				cancel.ThrowIfCancellationRequested();
				this.ProgressChanged?.Invoke(new ControllerProgress("Loading photos", i, photoPaths.Count, Path.GetFileName(photoPaths[i])));
				Bitmap bitmap = PhotoDecoder.Decode(photoPaths[i], this.settings.MaxImageSize);

				// The library keys images by name; two folders can both hold "IMG_0001.JPG".
				string name = Path.GetFileName(photoPaths[i]);
				for (int n = 2; !names.Add(name); n++)
				{
					name = Path.GetFileNameWithoutExtension(photoPaths[i]) + "-" + n + Path.GetExtension(photoPaths[i]);
				}

				images.Add(name, bitmap);

				// Decoding is the first long stretch of a run; on a shared UI thread the page would
				// otherwise sit on "Starting..." until every photo is in.
				await yieldToHost().ConfigureAwait(false);
			}

			return await this.RunAsync(images, cancel).ConfigureAwait(false);
		}

		/// <summary>Runs the pipeline on already-decoded photos.</summary>
		public async Task<SessionResult> RunAsync(InMemoryImageSource images, CancellationToken cancel)
		{
			string workspace = Path.Combine(this.settings.WorkspaceRoot, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + "-" + Guid.NewGuid().ToString("N")[..8]);
			Directory.CreateDirectory(workspace);
			try
			{
				return await this.RunInWorkspaceAsync(images, workspace, cancel).ConfigureAwait(false);
			}
			finally
			{
				// The result holds everything the app shows and saves, so the depth maps and meshes
				// on disk (tens of MB a run) go, however the run ended.
				if (!this.settings.KeepWorkspace)
				{
					try
					{
						Directory.Delete(workspace, recursive: true);
					}
					catch (IOException e)
					{
						// Never let cleanup hide the run's own outcome; the OS temp cleaner gets it later.
						Console.Error.WriteLine($"Could not delete the workspace {workspace}: {e.Message}");
					}
				}
			}
		}

		private async Task<SessionResult> RunInWorkspaceAsync(InMemoryImageSource images, string workspace, CancellationToken cancel)
		{

			var options = new AutomaticReconstructionOptions
			{
				WorkspacePath = workspace,
				Images = images,
				Data = AutomaticReconstructionOptions.DataType.Individual,
				Quality = this.settings.Quality,
				Dense = true,
				Mesher = AutomaticReconstructionOptions.MesherType.Poisson,
				Texture = true,
				ComputeDevice = this.settings.ComputeDevice,
			};

			options.PoissonMeshing.Trim = this.settings.PoissonTrim;
			options.PoissonMeshing.Depth = this.settings.PoissonDepth;

			var models = new ReconstructionManager();
			var controller = new AutomaticReconstructionController(options, models)
			{
				CancellationToken = cancel,
				YieldAsync = this.settings.YieldAsync,
			};

			IReadOnlyList<ColoredPoint> sparse = null;
			controller.Progress = new SyncProgress(p =>
			{
				// The dense stage's heading is the first report after the sparse mapper finished and
				// wrote the models, so the points are complete here (reported on the run's thread,
				// which is the only one touching the models).
				if (sparse == null && p.Stage == AutomaticReconstructionController.DenseStage)
				{
					sparse = MeshBridge.SparsePoints(models);
					this.SparseReady?.Invoke(sparse);
				}

				this.ProgressChanged?.Invoke(p);
			});

			controller.Setup();
			await controller.RunAsync().ConfigureAwait(false);
			cancel.ThrowIfCancellationRequested();

			if (sparse == null)
			{
				sparse = MeshBridge.SparsePoints(models);
				this.SparseReady?.Invoke(sparse);
			}

			TexturedModelMesh textured = controller.TexturedMeshes.FirstOrDefault();
			PlyMesh mesh = textured?.Mesh ?? ReadFirstMesh(workspace);
			ImageBuffer atlas = null;
			float[] uvs = null;
			if (textured != null && !textured.Texture.TextureAtlas.IsEmpty)
			{
				atlas = PhotoDecoder.ToImageBuffer(textured.Texture.TextureAtlas);
				uvs = textured.Texture.FaceUvs;
			}

			bool hasMesh = mesh != null && mesh.Faces.Count > 0;
			return new SessionResult
			{
				SparsePoints = sparse,
				Mesh = hasMesh ? mesh : null,
				FaceUvs = hasMesh ? uvs : null,
				Atlas = hasMesh ? atlas : null,
				PreviewMesh = hasMesh ? MeshBridge.ToAggMesh(mesh, atlas, uvs) : null,
				WorkspacePath = workspace,
			};
		}

		/// <summary>
		/// Writes <paramref name="result"/>'s mesh to <paramref name="path"/>: a ".ply" path gets a
		/// PLY, anything else an OBJ (plus an MTL of the same name). A textured mesh also gets its
		/// texture beside it as a PNG of the same name (model.obj -> model.png), overwriting any
		/// earlier one; an untextured OBJ carries vertex colors instead.
		/// </summary>
		public static void SaveMesh(SessionResult result, string path)
		{
			if (result?.Mesh == null)
			{
				throw new InvalidOperationException("There is no mesh to save yet.");
			}

			var texturedMesh = new PlyTexturedMesh(result.Mesh);
			if (result.IsTextured)
			{
				// The texture is named after the mesh (model.obj -> model.png) so two meshes saved in
				// one folder do not share, or overwrite, one texture.png.
				texturedMesh.FaceUvs = result.FaceUvs.ToList();
				texturedMesh.TextureFile = Path.GetFileNameWithoutExtension(path) + ".png";
				string texturePath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path)), texturedMesh.TextureFile);

				// ImageIO.SaveImageData(string, ...) leaves an existing file alone and reports failure
				// only through its result; a stream always overwrites, and a failure throws.
				using var texture = new FileStream(texturePath, FileMode.Create, FileAccess.Write);
				if (!ImageIO.SaveImageData(texture, ".png", result.Atlas))
				{
					throw new IOException($"Could not encode the texture {texturePath}.");
				}
			}

			if (string.Equals(Path.GetExtension(path), ".ply", StringComparison.OrdinalIgnoreCase))
			{
				// COLMAP's textured PLY: per-face texcoord lists and a TextureFile comment.
				Ply.WriteBinaryPlyMesh(path, texturedMesh);
			}
			else if (result.IsTextured)
			{
				ObjWriter.WriteTexturedObj(path, texturedMesh);
			}
			else
			{
				ObjWriter.WriteObj(path, result.Mesh);
			}
		}

		/// <summary>
		/// Writes <paramref name="result"/>'s mesh as OBJ + MTL (+ PNG when textured) into one zip at
		/// <paramref name="zipPath"/>, the entries named after the zip (mesh.zip holds mesh.obj,
		/// mesh.mtl, mesh.png, as <see cref="SaveMesh"/> names them). The browser's save is a single
		/// download, and three loose files would be three prompts.
		/// </summary>
		public static void SaveMeshZip(SessionResult result, string zipPath)
		{
			string staging = Path.Combine(Path.GetTempPath(), "ColmapDemo", "zip-" + Guid.NewGuid().ToString("N"));
			Directory.CreateDirectory(staging);
			try
			{
				SaveMesh(result, Path.Combine(staging, Path.GetFileNameWithoutExtension(zipPath) + ".obj"));
				if (File.Exists(zipPath))
				{
					File.Delete(zipPath);
				}

				ZipFile.CreateFromDirectory(staging, zipPath, CompressionLevel.Optimal, includeBaseDirectory: false);
			}
			finally
			{
				Directory.Delete(staging, recursive: true);
			}
		}

		private static async ValueTask YieldToEventLoop() => await Task.Yield();

		// A run whose texturing was skipped still leaves the Poisson mesh on disk.
		private static PlyMesh ReadFirstMesh(string workspace)
		{
			string meshPath = Path.Combine(workspace, "dense", "0", "meshed-poisson.ply");
			return File.Exists(meshPath) ? Ply.ReadPlyMesh(meshPath).Mesh : null;
		}

		// Progress<T> would post to a SynchronizationContext; the listeners marshal themselves.
		private sealed class SyncProgress : IProgress<ControllerProgress>
		{
			private readonly Action<ControllerProgress> report;

			public SyncProgress(Action<ControllerProgress> report) => this.report = report;

			public void Report(ControllerProgress value) => this.report(value);
		}
	}
}
