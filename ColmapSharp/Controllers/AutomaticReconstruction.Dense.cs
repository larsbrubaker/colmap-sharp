// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionController, dense part: port of RunDenseMapper from
// colmap/controllers/automatic_reconstruction.cc. Per sparse model it undistorts the images
// (ColmapUndistorter.cs) into <workspace>/dense/<i>, runs PatchMatch stereo
// (Mvs/PatchMatchController.cs), fuses the depth maps (Mvs/Fusion.cs) into fused.ply and
// fused.ply.vis, and meshes them (Mvs/PoissonMeshing.cs or Mvs/DelaunayMeshing.cs). With
// Options.Texture on it then textures the mesh (AutomaticReconstruction.Texture.cs, C#-only,
// divergence 135); a model whose fused.ply and mesh already exist is then
// still undistorted and textured once per controller, since the texture lives in memory. Also
// under entry 135, a mesher that fails or is cancelled leaves no mesh file behind, and a mesh
// file without a PLY mesh header counts as missing, so a resume re-meshes the model rather
// than trusting (or failing on) a partial file. The rest of the controller is in
// AutomaticReconstruction.cs.
//
// Translation notes (divergence 134):
// - COLMAP skips PatchMatch (and so everything after undistortion) without CUDA; the
//   PatchMatch algorithm is ported to the CPU here, so it runs - on the host's compute device
//   when Options.ComputeDevice is set (entry 136): always under RunAsync, and under Run only
//   when the device can be waited on synchronously.
// - Delaunay meshing is a CGAL-free port here, so it runs like a COLMAP build with CGAL.
//   Advancing-front meshing is CGAL code (out of scope) and is skipped with a warning like a
//   build without CGAL.
// - The undistorted images stay in memory (InMemoryBitmapStore) rather than as files under
//   dense/<i>/images, so a model is undistorted again when its images are not in the store
//   (e.g. in a new controller over an existing workspace), not only when dense/<i> is missing.
// - Delaunay meshing reseeds the thread's PRNG first, so a resumed workspace re-meshes to the
//   same bytes as the run that built it (divergence 137).

using ColmapSharp.Compute;
using ColmapSharp.ImageProcessing;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

public sealed partial class AutomaticReconstructionController
{
	/// <summary>The Stage of the progress reports of the dense steps.</summary>
	public const string DenseStage = "Dense reconstruction";

	/// <summary>The Stage of the progress reports of stereo fusion.</summary>
	public const string FusionStage = "Stereo fusion";

	/// <summary>The Stage of the progress reports of surface meshing (Done of 1000).</summary>
	public const string MeshingStage = "Surface meshing";

	// The key prefix under which the fusion reads the host's masks (StereoFusionOptions.MaskPath).
	// It starts with a NUL, which no file path on any OS can contain, so no workspace key (a
	// relative WorkspacePath "masks/ws" included) can ever be routed to the masks.
	internal const string MaskRoot = "\0masks";

	private readonly InMemoryBitmapStore undistortedImages = new();

	// Port of RunDenseMapper. An iterator, so that the synchronous and asynchronous entries share
	// it (see RunStages): it yields each model's PatchMatchController for the caller to run and
	// goes on with that model's fusion when resumed, and yields null after each other step (a
	// yield point for RunAsync, which Run skips). A yield break is COLMAP's return.
	private IEnumerable<PatchMatchController?> RunDenseMapper(bool deviceIsAwaited)
	{
		Heading(DenseStage);

		Directory.CreateDirectory(Path.Combine(options.WorkspacePath, "dense"));

		for (int i = 0; i < reconstructionManager.Size; ++i)
		{
			if (CheckIfStopped())
			{
				yield break;
			}

			string densePath = Path.Combine(options.WorkspacePath, "dense", i.ToString(System.Globalization.CultureInfo.InvariantCulture));
			string fusedPath = Path.Combine(densePath, "fused.ply");

			string meshingPath = options.Mesher switch
			{
				AutomaticReconstructionOptions.MesherType.Poisson => Path.Combine(densePath, "meshed-poisson.ply"),
				AutomaticReconstructionOptions.MesherType.Delaunay => Path.Combine(densePath, "meshed-delaunay.ply"),
				_ => Path.Combine(densePath, "meshed-advancing-front.ply"),
			};

			// A mesh file without a PLY mesh header is what an interrupted Poisson run leaves
			// (PoissonMeshing.Run creates its output before reconstructing). COLMAP would skip
			// such a model for good; here it counts as missing, so the model is re-meshed from
			// its fused.ply (divergence 135).
			if (File.Exists(meshingPath) && !HasPlyMeshHeader(meshingPath))
			{
				Log.Warning($"Re-meshing model {i}: {meshingPath} is not a PLY mesh");
				File.Delete(meshingPath);
			}

			// Texturing needs the undistorted images even when the mesh exists, and its result
			// lives in memory, so a model whose mesh exists is still textured once per controller.
			bool haveDense = File.Exists(fusedPath) && File.Exists(meshingPath);
			bool needTexture = NeedsTexturing(i);
			if (haveDense && !needTexture)
			{
				// Skipping dense reconstruction for model i as it already exists.
				continue;
			}

			// Image undistortion.

			string imagesPrefix = Path.Combine(densePath, "images") + Path.DirectorySeparatorChar;
			bool haveImages = undistortedImages.Paths.Any(p => p.StartsWith(imagesPrefix, StringComparison.Ordinal));
			if (!Directory.Exists(densePath) || !haveImages)
			{
				Directory.CreateDirectory(densePath);

				var undistortionOptions = new UndistortCameraOptions
				{
					MaxImageSize = optionManager.PatchMatchStereo.MaxImageSize,
				};
				var undistorterOptions = new ColmapUndistorter.Options { NumThreads = options.NumThreads };
				var undistorter = new ColmapUndistorter(undistorterOptions, undistortionOptions,
					reconstructionManager.Get(i), options.Images!, densePath, undistortedImages)
				{
					Progress = Under(DenseStage),
					CancellationToken = CancellationToken,
				};
				undistorter.SetCheckIfStoppedFunc(CheckIfStopped);
				undistorter.Run();
				yield return null;
			}

			if (CheckIfStopped())
			{
				yield break;
			}

			IBitmapSource bitmaps = options.Masks is null
				? undistortedImages
				: new WorkspaceBitmapSource(undistortedImages, options.Masks);

			if (!haveDense)
			{
				// Patch match stereo (on the host's compute device or the CPU; COLMAP needs
				// CUDA here), run by the caller.
				yield return new PatchMatchController(
					optionManager.PatchMatchStereo, densePath, "COLMAP", "", bitmaps)
				{
					ComputeDevice = deviceIsAwaited ? options.ComputeDevice : BlockingComputeDevice(),
				};

				if (!RunFusion(i, densePath, fusedPath, bitmaps))
				{
					yield break;
				}

				yield return null;
				if (!RunMeshing(densePath, fusedPath, meshingPath))
				{
					yield break;
				}

				yield return null;
			}

			if (CheckIfStopped())
			{
				yield break;
			}

			// No mesh here means Poisson reconstruction failed (it logged why); COLMAP goes on
			// to the next model, and so does texturing.
			if (needTexture && File.Exists(meshingPath))
			{
				RunTexturing(i, densePath, meshingPath, undistortedImages);
				yield return null;
			}
		}
	}

	// Fusion of model i: the part of RunDenseMapper's loop after PatchMatch. Returns false
	// where that loop returns (once stopped).
	private bool RunFusion(int i, string densePath, string fusedPath, IBitmapSource bitmaps)
	{
		if (CheckIfStopped())
		{
			return false;
		}

		// Stereo fusion.

		if (!File.Exists(fusedPath))
		{
			StereoFusionOptions fusionOptions = optionManager.StereoFusion.Clone();
			int numRegImages = reconstructionManager.Get(i).NumRegImages;
			fusionOptions.MinNumPixels = Math.Min(numRegImages + 1, fusionOptions.MinNumPixels);
			var fuser = new StereoFusion(fusionOptions, densePath, "COLMAP", "",
				optionManager.PatchMatchStereo.GeomConsistency ? "geometric" : "photometric", bitmaps);
			fuser.Run(CancellationToken, Forward<StereoFusionProgress>(
				p => new ControllerProgress(FusionStage, p.NumFusedImages, p.NumImages, "")));

			Ply.WriteBinaryPlyPoints(fusedPath, fuser.GetFusedPoints());
			StereoFusion.WritePointsVisibility(fusedPath + ".vis", fuser.GetFusedPointsVisibility());
		}

		return true;
	}

	// Meshing of model i: the part of RunDenseMapper's loop after fusion. Returns false where
	// that loop returns (once stopped, and after the advancing-front warning).
	private bool RunMeshing(string densePath, string fusedPath, string meshingPath)
	{
		if (CheckIfStopped())
		{
			return false;
		}

		// Surface meshing.

		if (!File.Exists(meshingPath))
		{
			if (options.Mesher == AutomaticReconstructionOptions.MesherType.AdvancingFront)
			{
				Log.Warning("Skipping advancing front meshing because CGAL is not available");
				return false;
			}

			IProgress<double>? meshingProgress = Forward<double>(
				v => new ControllerProgress(MeshingStage, (int)(v * 1000), 1000, ""));

			// A mesher that fails or is cancelled must not leave a partial mesh behind, or the
			// next run would take it for a finished one (divergence 135). COLMAP ignores
			// PoissonMeshing's result and keeps the file it created.
			try
			{
				bool meshed = options.Mesher == AutomaticReconstructionOptions.MesherType.Poisson
					? PoissonMeshing.Run(optionManager.PoissonMeshing, fusedPath, meshingPath, CancellationToken, meshingProgress)
					: RunDenseDelaunayMeshing(densePath, fusedPath, meshingPath, meshingProgress);
				if (!meshed)
				{
					File.Delete(meshingPath);
				}
			}
			catch
			{
				File.Delete(meshingPath);
				throw;
			}
		}

		return true;
	}

	// The host's compute device if the synchronous Run can use it: one that cannot be waited on
	// synchronously (the browser) would make PatchMatchController.Run throw, so PatchMatch runs
	// on the CPU instead, with a warning pointing the host at RunAsync.
	private IComputeDevice? BlockingComputeDevice()
	{
		IComputeDevice? device = options.ComputeDevice;
		if (device == null || device.SupportsBlockingWait)
		{
			return device;
		}

		Log.Warning("The GPU cannot be used by the synchronous dense reconstruction (call RunAsync to use it); running PatchMatch stereo on the CPU.");
		return null;
	}

	// Port of mvs::DenseDelaunayMeshing(options, dense_path, output_path): reads the
	// undistorted sparse model, fused.ply and fused.ply.vis of the dense workspace.
	// Returns true (it throws on failure), to match PoissonMeshing.Run's result.
	private bool RunDenseDelaunayMeshing(string densePath, string fusedPath, string meshingPath, IProgress<double>? progress)
	{
		var reconstruction = new Reconstruction();
		reconstruction.Read(Path.Combine(densePath, "sparse"));
		List<PlyPoint> plyPoints = Ply.ReadPly(fusedPath);
		List<List<int>> visibility = StereoFusion.ReadPointsVisibility(fusedPath + ".vis", plyPoints.Count);

		// Delaunay meshing shuffles its points with the thread's PRNG. COLMAP leaves that
		// where the sparse mapper's RANSAC (or an earlier model's meshing) left it, so a
		// resumed workspace meshes differently from the run that built it. Reseed so each
		// model's mesh depends only on its inputs (divergence 137).
		RandomUtils.SetPRNGSeed();
		PlyMesh mesh = DelaunayMeshing.DenseDelaunayMeshing(
			optionManager.DelaunayMeshing, reconstruction, plyPoints, visibility, progress, CancellationToken);
		Ply.WriteBinaryPlyMesh(meshingPath, new PlyTexturedMesh(mesh));
		return true;
	}

	// Whether path starts with a complete PLY header ("ply" through "end_header") that has an
	// element vertex with x, y and z properties and an element face: the cheap check that a
	// mesh file was written, without reading its body. Bounded, since a damaged file may
	// hold anything.
	private static bool HasPlyMeshHeader(string path)
	{
		const int MaxHeaderLines = 1000;
		using FileStream stream = FileOpen.OpenRead(path);
		var reader = new PlyByteReader(stream);
		if (reader.ReadLine()?.Trim() != "ply")
		{
			return false;
		}

		bool inVertex = false;
		bool haveFace = false;
		var coordinates = new HashSet<string>(StringComparer.Ordinal);
		for (int lineIdx = 0; lineIdx < MaxHeaderLines; ++lineIdx)
		{
			string? line = reader.ReadLine();
			if (line is null)
			{
				return false;
			}

			string[] elems = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
			if (elems.Length == 1 && elems[0] == "end_header")
			{
				return haveFace && coordinates.Count == 3;
			}

			if (elems.Length >= 2 && elems[0] == "element")
			{
				inVertex = elems[1] == "vertex";
				haveFace |= elems[1] == "face";
			}
			else if (inVertex && elems.Length == 3 && elems[0] == "property" && elems[2] is "x" or "y" or "z")
			{
				coordinates.Add(elems[2]);
			}
		}

		return false;
	}

	// The undistorted images, plus the host's masks under MaskRoot/<name> for the fusion.
	// Internal for AutomaticReconstructionTests.CSharpOnly_MaskSourceRoutesByReservedKey.
	internal sealed class WorkspaceBitmapSource(InMemoryBitmapStore images, IImageSource masks) : IBitmapSource
	{
		private static readonly string Prefix = MaskRoot + Path.DirectorySeparatorChar;

		public bool Exists(string path) =>
			path.StartsWith(Prefix, StringComparison.Ordinal) ? masks.Exists(MaskName(path)) : images.Exists(path);

		public Bitmap Read(string path, bool asRgb)
		{
			if (!path.StartsWith(Prefix, StringComparison.Ordinal))
			{
				return images.Read(path, asRgb);
			}

			Bitmap mask = masks.Read(MaskName(path)) ?? throw new IOException($"Failed to read mask {path}");
			return asRgb ? mask.CloneAsRGB() : mask.CloneAsGrey();
		}

		// IImageSource names use '/' separators.
		private static string MaskName(string path) => path[Prefix.Length..].Replace(Path.DirectorySeparatorChar, '/');
	}
}
