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
// The mapper can build several separate models; the one with the most placed photos is shown and
// meshed, and PhotoPlacement.cs says which photos were placed.
// When the GPU fails during the dense stage, the run goes on without it: GpuFailed is raised and
// the dense stage is run again on the CPU in the same workspace, from the sparse model on disk.
//
// A known focal length needs photos of one size (SessionSettings.KnownFocalSizeError); the run
// refuses others before any stage.
// Its events fire on whatever thread the run is on (the worker, on the Mac); listeners marshal.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ColmapDemo.Compute;
using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;
using MatterHackers.Agg.Image;
using MatterHackers.PolygonMesh;

namespace ColmapDemo
{
	/// <summary>What a finished run produced.</summary>
	public sealed class SessionResult
	{
		/// <summary>The sparse points of the shown model (the one with the most placed photos).</summary>
		public IReadOnlyList<ColoredPoint> SparsePoints { get; init; } = Array.Empty<ColoredPoint>();

		/// <summary>The shown model's mesh as the library wrote it, or null when meshing gave none.</summary>
		public PlyMesh Mesh { get; init; }

		/// <summary>The mesh's per-corner UVs (6 per face) when textured, else null.</summary>
		public float[] FaceUvs { get; init; }

		/// <summary>The texture atlas, or null when no face was seen by any photo.</summary>
		public ImageBuffer Atlas { get; init; }

		/// <summary>The mesh converted for the viewport, or null.</summary>
		public Mesh PreviewMesh { get; init; }

		/// <summary>Which photos the mapper placed, and in how many separate groups.</summary>
		public PhotoPlacement Placement { get; init; }

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
		/// The GPU failed during the run, with the fault; the run goes on on the CPU. The session
		/// never uses the device again; the host owns it, and should dispose it and stop offering it.
		/// </summary>
		public event Action<Exception> GpuFailed;

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
			var orderedNames = new List<string>(photoPaths.Count);

			// A turn for the host before the first decode too, so "Starting..." gets painted.
			await yieldToHost().ConfigureAwait(false);
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
				orderedNames.Add(name);

				// Decoding is the first long stretch of a run; on a shared UI thread the page would
				// otherwise sit on "Starting..." until every photo is in.
				await yieldToHost().ConfigureAwait(false);
			}

			return await this.RunAsync(images, orderedNames, cancel).ConfigureAwait(false);
		}

		/// <summary>Runs the pipeline on already-decoded photos.</summary>
		public Task<SessionResult> RunAsync(InMemoryImageSource images, CancellationToken cancel) =>
			this.RunAsync(images, images.ListNames(), cancel);

		// photoNames: the photos' names in input order, the order PhotoPlacement reports them in.
		private async Task<SessionResult> RunAsync(InMemoryImageSource images, IReadOnlyList<string> photoNames, CancellationToken cancel)
		{
			// Before any stage (and any workspace): a known focal needs photos of one size.
			if (this.settings.HasKnownFocal)
			{
				string sizeError = SessionSettings.KnownFocalSizeError(images.ListNames().Select(n => images.Read(n)).Where(b => b != null).Select(b => (b.Width, b.Height)));
				if (sizeError != null)
				{
					throw new InvalidOperationException(sizeError);
				}
			}

			string workspace = Path.Combine(this.settings.WorkspaceRoot, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + "-" + Guid.NewGuid().ToString("N")[..8]);
			Directory.CreateDirectory(workspace);
			try
			{
				return await this.RunInWorkspaceAsync(images, photoNames, workspace, cancel).ConfigureAwait(false);
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

		private async Task<SessionResult> RunInWorkspaceAsync(InMemoryImageSource images, IReadOnlyList<string> photoNames, string workspace, CancellationToken cancel)
		{
			FaultWatchingComputeDevice gpu = this.settings.ComputeDevice == null ? null : new FaultWatchingComputeDevice(this.settings.ComputeDevice);
			IReadOnlyList<ColoredPoint> sparse = null;
			ReconstructionManager models;
			AutomaticReconstructionController controller = this.CreateController(images, workspace, gpu, cancel, retry: false, s => sparse = s, () => sparse, out models);
			try
			{
				await controller.RunAsync().ConfigureAwait(false);
			}
			catch (Exception e) when (gpu?.Fault != null && !cancel.IsCancellationRequested && sparse != null)
			{
				// The GPU failed in the dense stage (the sparse points are out, so the mapper is done).
				// The sparse model is on disk and every finished depth map too, so a controller that
				// skips extraction and matching reads the model back and PatchMatch skips the problems
				// that have their maps; what is left runs on the CPU.
				Console.WriteLine("COLMAP_DEMO GPU failed, retrying the dense stage on the CPU: " + e);
				this.GpuFailed?.Invoke(gpu.Fault);
				controller = this.CreateController(images, workspace, null, cancel, retry: true, s => sparse = s, () => sparse, out models);
				await controller.RunAsync().ConfigureAwait(false);
			}

			cancel.ThrowIfCancellationRequested();

			// The dense stage ran on every model (dense/<i> is model i of this controller's manager,
			// in the order the mapper built them); the largest is the one shown.
			int shown = PhotoPlacement.LargestModelIndex(models);
			if (sparse == null)
			{
				sparse = MeshBridge.SparsePoints(models, shown);
				this.SparseReady?.Invoke(sparse);
			}

			TexturedModelMesh textured = controller.TexturedMeshes.FirstOrDefault(m => m.ModelIdx == shown);
			PlyMesh mesh = textured?.Mesh ?? ReadMesh(workspace, shown, this.settings.Mesher);
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
				Placement = PhotoPlacement.FromModels(photoNames, models, shown),
				WorkspacePath = workspace,
			};
		}

		// A controller for the run, or (retry) for re-running only the sparse read-back and the dense
		// stage in the workspace a first controller left: extraction and matching are off, and the
		// progress it repeats from before the dense stage is not reported again.
		private AutomaticReconstructionController CreateController(
			InMemoryImageSource images,
			string workspace,
			IComputeDevice device,
			CancellationToken cancel,
			bool retry,
			Action<IReadOnlyList<ColoredPoint>> setSparse,
			Func<IReadOnlyList<ColoredPoint>> getSparse,
			out ReconstructionManager modelsOut)
		{
			var options = new AutomaticReconstructionOptions
			{
				WorkspacePath = workspace,
				Images = images,
				ComputeDevice = device,
				Extraction = !retry,
				Matching = !retry,
			};

			// A known focal is in 35 mm terms; its pixels depend on the photos' size after shrinking.
			IReadOnlyList<string> names = images.ListNames();
			Bitmap first = names.Count > 0 ? images.Read(names[0]) : null;
			this.settings.ApplyTo(options, first?.Width ?? 0, first?.Height ?? 0);

			var models = new ReconstructionManager();
			modelsOut = models;
			var controller = new AutomaticReconstructionController(options, models)
			{
				CancellationToken = cancel,
				YieldAsync = this.settings.YieldAsync,
			};

			controller.Progress = new SyncProgress(p =>
			{
				if (retry && p.Stage == AutomaticReconstructionController.SparseStage)
				{
					return;
				}

				// The dense stage's heading is the first report after the sparse mapper finished and
				// wrote the models, so the points are complete here (reported on the run's thread,
				// which is the only one touching the models).
				if (getSparse() == null && p.Stage == AutomaticReconstructionController.DenseStage)
				{
					IReadOnlyList<ColoredPoint> points = MeshBridge.SparsePoints(models, PhotoPlacement.LargestModelIndex(models));

					// For comparing captures: how much of the input the mapper actually used.
					var registered = Enumerable.Range(0, models.Size).Select(m => models.Get(m).NumRegImages);
					Console.WriteLine($"COLMAP_DEMO sparse: {models.Size} model(s), registered images [{string.Join(", ", registered)}] of {images.ListNames().Count}, {points.Count} points");
					setSparse(points);
					this.SparseReady?.Invoke(points);
				}

				this.ProgressChanged?.Invoke(p);
			});

			controller.Setup();
			return controller;
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

		/// <summary>
		/// Model <paramref name="modelIndex"/>'s mesh as <paramref name="mesher"/> wrote it in
		/// <paramref name="workspace"/>, or null when there is none: a run whose texturing was skipped
		/// still leaves it on disk (none when Dense is off).
		/// </summary>
		public static PlyMesh ReadMesh(string workspace, int modelIndex, AutomaticReconstructionOptions.MesherType mesher)
		{
			if (modelIndex < 0)
			{
				return null;
			}

			string meshPath = Path.Combine(workspace, "dense", modelIndex.ToString(System.Globalization.CultureInfo.InvariantCulture), mesher == AutomaticReconstructionOptions.MesherType.Delaunay ? "meshed-delaunay.ply" : "meshed-poisson.ply");
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
