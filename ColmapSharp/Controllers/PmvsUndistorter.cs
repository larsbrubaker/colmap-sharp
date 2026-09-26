// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PmvsUndistorter and CmpMvsUndistorter: ports of colmap::PMVSUndistorter and
// colmap::CMPMVSUndistorter (controllers/undistorters.h/.cc). They undistort the registered
// images to pinhole cameras and write them with projection matrices in the layouts of
// CMVS/PMVS (pmvs/visualize/%08d.jpg, pmvs/txt/%08d.txt, bundle file, vis.dat, option-all and
// run scripts) and CMP-MVS (%05d.jpg and %05d_P.txt, numbered from 1). The shared helpers and
// translation notes are in Undistorters.cs. Tests:
// ColmapSharp.Tests/Controllers/UndistortersTests.cs.
//
// Translation note: option-all's "CPU" line is std::thread::hardware_concurrency(), here
// Environment.ProcessorCount (the logical processors available to the process).

using System.Globalization;
using System.Text;

using ColmapSharp.ImageProcessing;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::PMVSUndistorter: undistorts images and prepares data for CMVS/PMVS.</summary>
public sealed class PmvsUndistorter : BaseController
{
	/// <summary>Port of PMVSUndistorter::Options.</summary>
	public sealed class Options
	{
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
	/// Undistorts the registered images of <paramref name="reconstruction"/>, read by name
	/// from <paramref name="imageSource"/>, into <c>outputPath/pmvs</c>; the images go to
	/// <paramref name="imageSink"/>.
	/// </summary>
	public PmvsUndistorter(
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
		Check.Ge(options.JpegQuality, -1);
		Check.Le(options.JpegQuality, 100);
	}

	/// <summary>
	/// Receives the progress of the run (COLMAP's "[i/n]" LOG(INFO) lines), one report per
	/// image in order, on the thread running <see cref="Run"/>.
	/// </summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <inheritdoc/>
	public override void Run()
	{
		Directory.CreateDirectory(Path.Combine(outputPath, "pmvs"));
		Directory.CreateDirectory(Path.Combine(outputPath, "pmvs", "txt"));
		Directory.CreateDirectory(Path.Combine(outputPath, "pmvs", "visualize"));
		Directory.CreateDirectory(Path.Combine(outputPath, "pmvs", "models"));

		List<uint> regImageIds = reconstruction.RegImageIds();
		int numImages = regImageIds.Count;
		bool finished = Undistorters.RunTasks(
			this,
			options.NumThreads,
			numImages,
			i => Undistort(regImageIds, i),
			(i, _) => Progress?.Report(new ControllerProgress(
				Undistorters.UndistortionStage, i + 1, numImages, reconstruction.Image(regImageIds[i]).Name)));
		if (!finished)
		{
			Log.Warning("Stopped image undistortion before writing the bundle and configuration files.");
			return;
		}

		Reconstruction undistortedReconstruction = reconstruction.Clone();
		Undistortion.UndistortReconstruction(cameraOptions, undistortedReconstruction);
		string bundlePath = Path.Combine(outputPath, "pmvs", "bundle.rd.out");
		ReconstructionIO.ExportBundler(undistortedReconstruction, bundlePath, FileUtils.AddFileExtension(bundlePath, ".list.txt"));

		WriteVisibilityData(regImageIds);
		WriteOptionFile(numImages);

		WritePmvsScript();
		WriteCmvsPmvsScript();
		WriteColmapScript(false);
		WriteColmapScript(true);
		WriteCmvsColmapScript(false);
		WriteCmvsColmapScript(true);
	}

	private bool Undistort(List<uint> regImageIds, int regImageIdx)
	{
		string index = regImageIdx.ToString("D8", CultureInfo.InvariantCulture);
		string outputImagePath = Path.Combine(outputPath, "pmvs", "visualize", index + ".jpg");
		string projMatrixPath = Path.Combine(outputPath, "pmvs", "txt", index + ".txt");

		Image image = reconstruction.Image(regImageIds[regImageIdx]);
		Camera camera = image.CameraPtr;

		Bitmap? distortedBitmap = Undistorters.ReadDistorted(imageSource, image.Name, "Cannot read image at path ");
		if (distortedBitmap is null)
		{
			return false;
		}

		Undistortion.UndistortImage(
			cameraOptions, distortedBitmap, camera, out Bitmap undistortedBitmap, out Camera undistortedCamera);

		Undistorters.WriteProjectionMatrix(projMatrixPath, undistortedCamera, image, "CONTOUR");
		Undistorters.MaybeSetJpegQuality(outputImagePath, undistortedBitmap, options.JpegQuality);
		return imageSink.Write(outputImagePath, undistortedBitmap);
	}

	private void WriteVisibilityData(List<uint> regImageIds)
	{
		var file = new StringBuilder();
		file.Append("VISDATA\n");
		file.Append(regImageIds.Count).Append('\n');

		int imageIdx = 0;
		foreach (uint imageId in regImageIds)
		{
			Image image = reconstruction.Image(imageId);
			var visibleImageIds = new HashSet<uint>();
			foreach (Point2D point2D in image.Points2D)
			{
				if (point2D.HasPoint3D)
				{
					Point3D point3D = reconstruction.Point3D(point2D.Point3DId);
					foreach (TrackElement trackEl in point3D.Track.Elements)
					{
						if (trackEl.ImageId != imageId)
						{
							visibleImageIds.Add(trackEl.ImageId);
						}
					}
				}
			}

			var sortedVisibleImageIds = new List<uint>(visibleImageIds);
			sortedVisibleImageIds.Sort();

			file.Append(imageIdx++).Append(' ').Append(visibleImageIds.Count);
			foreach (uint visibleImageId in sortedVisibleImageIds)
			{
				file.Append(' ').Append(visibleImageId);
			}

			file.Append('\n');
		}

		Undistorters.WriteTextFile(Path.Combine(outputPath, "pmvs", "vis.dat"), file);
	}

	private void WritePmvsScript()
	{
		var file = new StringBuilder();
		file.Append("# You must set $PMVS_EXE_PATH to \n")
			.Append("# the directory containing the CMVS-PMVS executables.\n");
		file.Append("$PMVS_EXE_PATH/pmvs2 pmvs/ option-all\n");
		Undistorters.WriteTextFile(Path.Combine(outputPath, "run-pmvs.sh"), file);
	}

	private void WriteCmvsPmvsScript()
	{
		var file = new StringBuilder();
		file.Append("# You must set $PMVS_EXE_PATH to \n")
			.Append("# the directory containing the CMVS-PMVS executables.\n");
		file.Append("$PMVS_EXE_PATH/cmvs pmvs/\n");
		file.Append("$PMVS_EXE_PATH/genOption pmvs/\n");
		file.Append("find pmvs/ -iname \"option-*\" | sort | while read file_name\n");
		file.Append("do\n");
		file.Append("    option_name=$(basename \"$file_name\")\n");
		file.Append("    if [ \"$option_name\" = \"option-all\" ]; then\n");
		file.Append("        continue\n");
		file.Append("    fi\n");
		file.Append("    $PMVS_EXE_PATH/pmvs2 pmvs/ $option_name\n");
		file.Append("done\n");
		Undistorters.WriteTextFile(Path.Combine(outputPath, "run-cmvs-pmvs.sh"), file);
	}

	private void WriteColmapScript(bool geometric)
	{
		string path = Path.Combine(outputPath, geometric ? "run-colmap-geometric.sh" : "run-colmap-photometric.sh");
		var file = new StringBuilder();
		file.Append("# You must set $COLMAP_EXE_PATH to \n")
			.Append("# the directory containing the COLMAP executables.\n");
		Undistorters.WriteColmapCommands(geometric, "pmvs", "PMVS", "option-all", "option-all-", "", file);
		Undistorters.WriteTextFile(path, file);
	}

	private void WriteCmvsColmapScript(bool geometric)
	{
		string path = Path.Combine(
			outputPath, geometric ? "run-cmvs-colmap-geometric.sh" : "run-cmvs-colmap-photometric.sh");
		var file = new StringBuilder();
		file.Append("# You must set $PMVS_EXE_PATH to \n")
			.Append("# the directory containing the CMVS-PMVS executables\n");
		file.Append("# and you must set $COLMAP_EXE_PATH to \n")
			.Append("# the directory containing the COLMAP executables.\n");
		file.Append("$PMVS_EXE_PATH/cmvs pmvs/\n");
		file.Append("$PMVS_EXE_PATH/genOption pmvs/\n");
		file.Append("find pmvs/ -iname \"option-*\" | sort | while read file_name\n");
		file.Append("do\n");
		file.Append("    workspace_path=$(dirname \"$file_name\")\n");
		file.Append("    option_name=$(basename \"$file_name\")\n");
		file.Append("    if [ \"$option_name\" = \"option-all\" ]; then\n");
		file.Append("        continue\n");
		file.Append("    fi\n");
		file.Append("    rm -rf \"$workspace_path/stereo\"\n");
		Undistorters.WriteColmapCommands(geometric, "pmvs", "PMVS", "$option_name", "$option_name-", "    ", file);
		file.Append("done\n");
		Undistorters.WriteTextFile(path, file);
	}

	private void WriteOptionFile(int numRegImages)
	{
		var file = new StringBuilder();
		file.Append("# Generated by COLMAP - all images, no clustering.\n");

		file.Append("level 1\n");
		file.Append("csize 2\n");
		file.Append("threshold 0.7\n");
		file.Append("wsize 7\n");
		file.Append("minImageNum 3\n");
		file.Append("CPU ").Append(Environment.ProcessorCount).Append('\n');
		file.Append("setEdge 0\n");
		file.Append("useBound 0\n");
		file.Append("useVisData 1\n");
		file.Append("sequence -1\n");
		file.Append("maxAngle 10\n");
		file.Append("quad 2.0\n");

		file.Append("timages ").Append(numRegImages);
		for (int i = 0; i < numRegImages; ++i)
		{
			file.Append(' ').Append(i);
		}

		file.Append('\n');

		file.Append("oimages 0\n");
		Undistorters.WriteTextFile(Path.Combine(outputPath, "pmvs", "option-all"), file);
	}
}

/// <summary>Port of colmap::CMPMVSUndistorter: undistorts images and prepares data for CMP-MVS.</summary>
public sealed class CmpMvsUndistorter : BaseController
{
	/// <summary>Port of CMPMVSUndistorter::Options.</summary>
	public sealed class Options
	{
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
	/// Undistorts the registered images of <paramref name="reconstruction"/>, read by name
	/// from <paramref name="imageSource"/>, into <paramref name="outputPath"/>; the images go
	/// to <paramref name="imageSink"/>.
	/// </summary>
	public CmpMvsUndistorter(
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
		Check.Ge(options.JpegQuality, -1);
		Check.Le(options.JpegQuality, 100);
	}

	/// <summary>
	/// Receives the progress of the run (COLMAP's "[i/n]" LOG(INFO) lines), one report per
	/// image in order, on the thread running <see cref="Run"/>.
	/// </summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <inheritdoc/>
	public override void Run()
	{
		List<uint> regImageIds = reconstruction.RegImageIds();
		int numImages = regImageIds.Count;
		Undistorters.RunTasks(
			this,
			options.NumThreads,
			numImages,
			i => Undistort(regImageIds, i),
			(i, _) => Progress?.Report(new ControllerProgress(
				Undistorters.UndistortionStage, i + 1, numImages, reconstruction.Image(regImageIds[i]).Name)));
	}

	private bool Undistort(List<uint> regImageIds, int regImageIdx)
	{
		string index = (regImageIdx + 1).ToString("D5", CultureInfo.InvariantCulture);
		string outputImagePath = Path.Combine(outputPath, index + ".jpg");
		string projMatrixPath = Path.Combine(outputPath, index + "_P.txt");

		Image image = reconstruction.Image(regImageIds[regImageIdx]);
		Camera camera = image.CameraPtr;

		Bitmap? distortedBitmap = Undistorters.ReadDistorted(imageSource, image.Name, "Cannot read image at path ");
		if (distortedBitmap is null)
		{
			return false;
		}

		Undistortion.UndistortImage(
			cameraOptions, distortedBitmap, camera, out Bitmap undistortedBitmap, out Camera undistortedCamera);

		Undistorters.WriteProjectionMatrix(projMatrixPath, undistortedCamera, image, "CONTOUR");
		Undistorters.MaybeSetJpegQuality(outputImagePath, undistortedBitmap, options.JpegQuality);
		return imageSink.Write(outputImagePath, undistortedBitmap);
	}
}
