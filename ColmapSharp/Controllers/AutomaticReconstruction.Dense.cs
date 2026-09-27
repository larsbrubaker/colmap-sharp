// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionController, dense part: port of RunDenseMapper from
// colmap/controllers/automatic_reconstruction.cc. Per sparse model it undistorts the images
// (ColmapUndistorter.cs) into <workspace>/dense/<i>, runs PatchMatch stereo
// (Mvs/PatchMatchController.cs), fuses the depth maps (Mvs/Fusion.cs) into fused.ply and
// fused.ply.vis, and meshes them (Mvs/PoissonMeshing.cs or Mvs/DelaunayMeshing.cs). The rest
// of the controller is in AutomaticReconstruction.cs.
//
// Translation notes (docs/CPP_DIVERGENCES.md entry 134):
// - COLMAP skips PatchMatch (and so everything after undistortion) without CUDA; the
//   PatchMatch algorithm is ported to the CPU here, so it runs.
// - Delaunay meshing is a CGAL-free port here, so it runs like a COLMAP build with CGAL.
//   Advancing-front meshing is CGAL code (out of scope) and is skipped with a warning like a
//   build without CGAL.
// - The undistorted images stay in memory (InMemoryBitmapStore) rather than as files under
//   dense/<i>/images, so a model is undistorted again when its images are not in the store
//   (e.g. in a new controller over an existing workspace), not only when dense/<i> is missing.

using ColmapSharp.ImageProcessing;
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
	private const string MaskRoot = "masks";

	private readonly InMemoryBitmapStore undistortedImages = new();

	private void RunDenseMapper()
	{
		Heading(DenseStage);

		Directory.CreateDirectory(Path.Combine(options.WorkspacePath, "dense"));

		for (int i = 0; i < reconstructionManager.Size; ++i)
		{
			if (CheckIfStopped())
			{
				return;
			}

			string densePath = Path.Combine(options.WorkspacePath, "dense", i.ToString(System.Globalization.CultureInfo.InvariantCulture));
			string fusedPath = Path.Combine(densePath, "fused.ply");

			string meshingPath = options.Mesher switch
			{
				AutomaticReconstructionOptions.MesherType.Poisson => Path.Combine(densePath, "meshed-poisson.ply"),
				AutomaticReconstructionOptions.MesherType.Delaunay => Path.Combine(densePath, "meshed-delaunay.ply"),
				_ => Path.Combine(densePath, "meshed-advancing-front.ply"),
			};

			if (File.Exists(fusedPath) && File.Exists(meshingPath))
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
					Progress = Progress,
					CancellationToken = CancellationToken,
				};
				undistorter.SetCheckIfStoppedFunc(CheckIfStopped);
				undistorter.Run();
			}

			if (CheckIfStopped())
			{
				return;
			}

			IBitmapSource bitmaps = options.Masks is null
				? undistortedImages
				: new WorkspaceBitmapSource(undistortedImages, options.Masks);

			// Patch match stereo (on the CPU; COLMAP needs CUDA here).

			var patchMatchController = new PatchMatchController(
				optionManager.PatchMatchStereo, densePath, "COLMAP", "", bitmaps);
			patchMatchController.Run(CancellationToken, Progress);

			if (CheckIfStopped())
			{
				return;
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

			if (CheckIfStopped())
			{
				return;
			}

			// Surface meshing.

			if (!File.Exists(meshingPath))
			{
				IProgress<double>? meshingProgress = Forward<double>(
					v => new ControllerProgress(MeshingStage, (int)(v * 1000), 1000, ""));
				if (options.Mesher == AutomaticReconstructionOptions.MesherType.Poisson)
				{
					PoissonMeshing.Run(optionManager.PoissonMeshing, fusedPath, meshingPath, CancellationToken, meshingProgress);
				}
				else if (options.Mesher == AutomaticReconstructionOptions.MesherType.Delaunay)
				{
					RunDenseDelaunayMeshing(densePath, fusedPath, meshingPath, meshingProgress);
				}
				else
				{
					Log.Warning("Skipping advancing front meshing because CGAL is not available");
					return;
				}
			}
		}
	}

	// Port of mvs::DenseDelaunayMeshing(options, dense_path, output_path): reads the
	// undistorted sparse model, fused.ply and fused.ply.vis of the dense workspace.
	private void RunDenseDelaunayMeshing(string densePath, string fusedPath, string meshingPath, IProgress<double>? progress)
	{
		var reconstruction = new Reconstruction();
		reconstruction.Read(Path.Combine(densePath, "sparse"));
		List<PlyPoint> plyPoints = Ply.ReadPly(fusedPath);
		List<List<int>> visibility = StereoFusion.ReadPointsVisibility(fusedPath + ".vis", plyPoints.Count);
		PlyMesh mesh = DelaunayMeshing.DenseDelaunayMeshing(
			optionManager.DelaunayMeshing, reconstruction, plyPoints, visibility, progress, CancellationToken);
		Ply.WriteBinaryPlyMesh(meshingPath, new PlyTexturedMesh(mesh));
	}

	// The undistorted images, plus the host's masks under MaskRoot/<name> for the fusion.
	private sealed class WorkspaceBitmapSource(InMemoryBitmapStore images, IImageSource masks) : IBitmapSource
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
