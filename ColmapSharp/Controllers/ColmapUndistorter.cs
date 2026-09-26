// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ColmapUndistorter: port of colmap::COLMAPUndistorter (controllers/undistorters.h/.cc).
// It undistorts the images of a reconstruction to pinhole cameras and lays out the MVS
// workspace mvs::PatchMatchController reads: images/, sparse/ (the undistorted
// reconstruction), stereo/ with its depth/normal/consistency folders, patch-match.cfg,
// fusion.cfg and the run-colmap-*.sh scripts. The shared helpers and translation notes are
// in Undistorters.cs. Tests: ColmapSharp.Tests/Controllers/UndistortersTests.cs.
//
// Addition for MatterCAD: after a run that was not stopped, UndistortedReconstruction holds
// the undistorted reconstruction COLMAP writes to sparse/, so the host can build an
// Mvs.Model from it (Model.ReadFromCOLMAP(reconstruction, <output>/images)) and read the
// images from an InMemoryBitmapStore sink, with no file round trip.

using ColmapSharp.ImageProcessing;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using System.Text;

namespace ColmapSharp.Controllers;

/// <summary>
/// Port of colmap::COLMAPUndistorter: undistorts images and exports undistorted cameras, as
/// required by the MVS PatchMatch controller.
/// </summary>
public sealed class ColmapUndistorter : BaseController
{
	/// <summary>Port of COLMAPUndistorter::Options.</summary>
	public sealed class Options
	{
		/// <summary>
		/// How many images to use as patch match source images when generating the patch
		/// match config file.
		/// </summary>
		public int NumPatchMatchSrcImages { get; set; } = 20;

		/// <summary>List of images to undistort. If empty, all images are undistorted.</summary>
		public List<uint> ImageIds { get; set; } = [];

		/// <summary>
		/// JPEG quality setting in the range [0, 100]. A value of -1 uses the default
		/// (quality 100). Lower values produce smaller file sizes.
		/// </summary>
		public int JpegQuality { get; set; } = -1;

		/// <summary>
		/// Number of threads to use for undistortion. A value of -1 uses all available CPU
		/// cores.
		/// </summary>
		public int NumThreads { get; set; } = -1;
	}

	private readonly Options options;
	private readonly UndistortCameraOptions cameraOptions;
	private readonly Reconstruction reconstruction;
	private readonly IImageSource imageSource;
	private readonly string outputPath;
	private readonly IBitmapSink imageSink;

	/// <summary>
	/// Undistorts the images of <paramref name="reconstruction"/>, read by name from
	/// <paramref name="imageSource"/> (COLMAP's image_path), into the workspace at
	/// <paramref name="outputPath"/>; the images go to <paramref name="imageSink"/> under
	/// <c>outputPath/images/&lt;name&gt;</c>.
	/// </summary>
	public ColmapUndistorter(
		Options options,
		UndistortCameraOptions cameraOptions,
		Reconstruction reconstruction,
		IImageSource imageSource,
		string outputPath,
		IBitmapSink imageSink)
	{
		this.options = options;
		this.cameraOptions = cameraOptions;
		this.reconstruction = reconstruction;
		this.imageSource = imageSource;
		this.outputPath = outputPath;
		this.imageSink = imageSink;
		Check.Ge(options.NumPatchMatchSrcImages, 1);
		Check.Ge(options.JpegQuality, -1);
		Check.Le(options.JpegQuality, 100);
	}

	/// <summary>
	/// The undistorted reconstruction written to sparse/, or null before a run and after a
	/// stopped one (C#-only, see the header).
	/// </summary>
	public Reconstruction? UndistortedReconstruction { get; private set; }

	/// <summary>
	/// Receives the progress of the run (COLMAP's "[i/n]" LOG(INFO) lines), one report per
	/// image in order, on the thread running <see cref="Run"/>.
	/// </summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <inheritdoc/>
	public override void Run()
	{
		UndistortedReconstruction = null;

		Directory.CreateDirectory(Path.Combine(outputPath, "images"));
		Directory.CreateDirectory(Path.Combine(outputPath, "sparse"));
		Directory.CreateDirectory(Path.Combine(outputPath, "stereo"));
		Directory.CreateDirectory(Path.Combine(outputPath, "stereo", "depth_maps"));
		Directory.CreateDirectory(Path.Combine(outputPath, "stereo", "normal_maps"));
		Directory.CreateDirectory(Path.Combine(outputPath, "stereo", "consistency_graphs"));
		reconstruction.CreateImageDirs(Path.Combine(outputPath, "images"));
		reconstruction.CreateImageDirs(Path.Combine(outputPath, "stereo", "depth_maps"));
		reconstruction.CreateImageDirs(Path.Combine(outputPath, "stereo", "normal_maps"));
		reconstruction.CreateImageDirs(Path.Combine(outputPath, "stereo", "consistency_graphs"));

		List<uint> imageIds = options.ImageIds.Count == 0 ? reconstruction.RegImageIds() : options.ImageIds;
		int numImages = imageIds.Count;

		// Only use the image names for the successfully undistorted images when writing the
		// MVS config files.
		var imageNames = new List<string>(numImages);
		bool finished = Undistorters.RunTasks(
			this,
			options.NumThreads,
			numImages,
			i => Undistort(imageIds[i]),
			(i, undistorted) =>
			{
				string name = reconstruction.Image(imageIds[i]).Name;
				Progress?.Report(new ControllerProgress(Undistorters.UndistortionStage, i + 1, numImages, name));
				if (undistorted)
				{
					imageNames.Add(name);
				}
			});
		if (!finished)
		{
			Log.Warning("Stopped image undistortion before writing the sparse model and stereo configuration files.");
			return;
		}

		Reconstruction undistortedReconstruction = reconstruction.Clone();
		Undistortion.UndistortReconstruction(cameraOptions, undistortedReconstruction);
		undistortedReconstruction.Write(Path.Combine(outputPath, "sparse"));

		WritePatchMatchConfig(imageNames);
		WriteFusionConfig(imageNames);

		WriteScript(geometric: false);
		WriteScript(geometric: true);

		UndistortedReconstruction = undistortedReconstruction;
	}

	private bool Undistort(uint imageId)
	{
		Image image = reconstruction.Image(imageId);
		Camera camera = image.CameraPtr;

		string outputImagePath = Path.Combine(outputPath, "images", image.Name);

		// Non-perspective cameras (e.g. EQUIRECTANGULAR) have no pinhole image plane to
		// undistort to. Without a size limit they are copied through unchanged (they cannot be
		// rescaled to a pinhole image for MVS); with a max_image_size they still go through
		// UndistortImage below, which resizes them to a smaller image of the same model.
		if (!camera.IsPerspective && cameraOptions.MaxImageSize < 0 && imageSource.Exists(image.Name))
		{
			Log.Warning(
				$"Cannot undistort image {image.Name} with non-perspective camera model {camera.ModelName}; " +
				"copying the original image.");
			return Undistorters.CopyImage(imageSource, image.Name, imageSink, outputImagePath);
		}

		// Already-undistorted perspective images are copied through only when no rescaling is
		// requested; with a max_image_size they still go through UndistortImage below so the
		// size limit is applied.
		if (camera.IsUndistorted() && cameraOptions.MaxImageSize < 0 && imageSource.Exists(image.Name))
		{
			return Undistorters.CopyImage(imageSource, image.Name, imageSink, outputImagePath);
		}

		Bitmap? distortedBitmap = Undistorters.ReadDistorted(imageSource, image.Name, "Cannot read image at path: ");
		if (distortedBitmap is null)
		{
			return false;
		}

		Undistortion.UndistortImage(cameraOptions, distortedBitmap, camera, out Bitmap undistortedBitmap, out _);

		Undistorters.MaybeSetJpegQuality(outputImagePath, undistortedBitmap, options.JpegQuality);

		return imageSink.Write(outputImagePath, undistortedBitmap);
	}

	private void WritePatchMatchConfig(List<string> imageNames)
	{
		var file = new StringBuilder();
		foreach (string imageName in imageNames)
		{
			file.Append(imageName).Append('\n');
			file.Append("__auto__, ").Append(options.NumPatchMatchSrcImages).Append('\n');
		}

		Undistorters.WriteTextFile(Path.Combine(outputPath, "stereo", "patch-match.cfg"), file);
	}

	private void WriteFusionConfig(List<string> imageNames)
	{
		var file = new StringBuilder();
		foreach (string imageName in imageNames)
		{
			file.Append(imageName).Append('\n');
		}

		Undistorters.WriteTextFile(Path.Combine(outputPath, "stereo", "fusion.cfg"), file);
	}

	private void WriteScript(bool geometric)
	{
		string path = Path.Combine(outputPath, geometric ? "run-colmap-geometric.sh" : "run-colmap-photometric.sh");
		var file = new StringBuilder();
		file.Append("# You must set $COLMAP_EXE_PATH to \n")
			.Append("# the directory containing the COLMAP executables.\n");
		Undistorters.WriteColmapCommands(geometric, ".", "COLMAP", "option-all", "", "", file);
		Undistorters.WriteTextFile(path, file);
	}
}
