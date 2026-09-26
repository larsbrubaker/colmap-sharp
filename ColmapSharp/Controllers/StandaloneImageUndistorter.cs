// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// StandaloneImageUndistorter and StereoImageRectifier: ports of
// colmap::StandaloneImageUndistorter (undistorts named images with given cameras, without a
// reconstruction) and colmap::StereoImageRectifier (undistorts and rectifies image pairs,
// writing both images and the disparity-to-depth matrix Q.txt into <image1>-<image2>/), from
// controllers/undistorters.h/.cc. The shared helpers and translation notes are in
// Undistorters.cs. Tests: ColmapSharp.Tests/Controllers/UndistortersTests.cs.

using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.ImageProcessing;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>
/// Port of colmap::StandaloneImageUndistorter: undistorts images and exports undistorted
/// cameras without the need for a reconstruction; the image names and cameras are given.
/// </summary>
public sealed class StandaloneImageUndistorter : BaseController
{
	/// <summary>Port of StandaloneImageUndistorter::Options.</summary>
	public sealed class Options
	{
		/// <summary>The images and cameras to undistort.</summary>
		public List<(string ImageName, Camera Camera)> ImageNamesAndCameras { get; set; } = [];

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
	private readonly IImageSource imageSource;
	private readonly string outputPath;
	private readonly IBitmapSink imageSink;

	/// <summary>
	/// Undistorts the listed images, read by name from <paramref name="imageSource"/>; each
	/// goes to <paramref name="imageSink"/> as <c>outputPath/&lt;name&gt;</c>.
	/// </summary>
	public StandaloneImageUndistorter(
		Options options,
		UndistortCameraOptions cameraOptions,
		IImageSource imageSource,
		string outputPath,
		IBitmapSink imageSink)
	{
		this.options = options;
		this.cameraOptions = cameraOptions;
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
		Directory.CreateDirectory(outputPath);

		int numImages = options.ImageNamesAndCameras.Count;
		Undistorters.RunTasks(
			this,
			options.NumThreads,
			numImages,
			Undistort,
			(i, _) => Progress?.Report(new ControllerProgress(
				Undistorters.UndistortionStage, i + 1, numImages, options.ImageNamesAndCameras[i].ImageName)));
	}

	private bool Undistort(int imageIdx)
	{
		(string imageName, Camera camera) = options.ImageNamesAndCameras[imageIdx];

		string outputImagePath = Path.Combine(outputPath, imageName);

		// Check if the image is already undistorted and copy from source if no scaling is
		// needed.
		if (camera.IsUndistorted() && cameraOptions.MaxImageSize < 0 && imageSource.Exists(imageName))
		{
			return Undistorters.CopyImage(imageSource, imageName, imageSink, outputImagePath);
		}

		Bitmap? distortedBitmap = Undistorters.ReadDistorted(imageSource, imageName, "Cannot read image at path ");
		if (distortedBitmap is null)
		{
			return false;
		}

		Undistortion.UndistortImage(cameraOptions, distortedBitmap, camera, out Bitmap undistortedBitmap, out _);

		Undistorters.MaybeSetJpegQuality(outputImagePath, undistortedBitmap, options.JpegQuality);

		return imageSink.Write(outputImagePath, undistortedBitmap);
	}
}

/// <summary>Port of colmap::StereoImageRectifier: rectifies stereo image pairs.</summary>
public sealed class StereoImageRectifier : BaseController
{
	/// <summary>Port of StereoImageRectifier::Options.</summary>
	public sealed class Options
	{
		/// <summary>The stereo image pairs to rectify.</summary>
		public List<(uint ImageId1, uint ImageId2)> StereoPairs { get; set; } = [];

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
	/// Rectifies the pairs of <paramref name="options"/>, images read by name from
	/// <paramref name="imageSource"/>; each pair goes to <paramref name="imageSink"/> under
	/// <c>outputPath/&lt;name1&gt;-&lt;name2&gt;/</c>, with Q.txt written beside them.
	/// </summary>
	public StereoImageRectifier(
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
		int numPairs = options.StereoPairs.Count;
		Undistorters.RunTasks(
			this,
			options.NumThreads,
			numPairs,
			i =>
			{
				Rectify(options.StereoPairs[i].ImageId1, options.StereoPairs[i].ImageId2);
				return true;
			},
			(i, _) => Progress?.Report(new ControllerProgress(
				Undistorters.RectificationStage, i + 1, numPairs, StereoPairName(options.StereoPairs[i]))));
	}

	private string StereoPairName((uint ImageId1, uint ImageId2) pair) =>
		reconstruction.Image(pair.ImageId1).Name.Replace("/", "-", StringComparison.Ordinal) + "-" +
		reconstruction.Image(pair.ImageId2).Name.Replace("/", "-", StringComparison.Ordinal);

	private void Rectify(uint imageId1, uint imageId2)
	{
		Image image1 = reconstruction.Image(imageId1);
		Image image2 = reconstruction.Image(imageId2);
		Camera camera1 = reconstruction.Camera(image1.CameraId);
		Camera camera2 = reconstruction.Camera(image2.CameraId);

		string imageName1 = image1.Name.Replace("/", "-", StringComparison.Ordinal);
		string imageName2 = image2.Name.Replace("/", "-", StringComparison.Ordinal);

		string stereoPairName = imageName1 + "-" + imageName2;

		Directory.CreateDirectory(Path.Combine(outputPath, stereoPairName));

		string outputImagePath1 = Path.Combine(outputPath, stereoPairName, imageName1);
		string outputImagePath2 = Path.Combine(outputPath, stereoPairName, imageName2);

		Bitmap? distortedBitmap1 = Undistorters.ReadDistorted(imageSource, image1.Name, "Cannot read image at path ");
		if (distortedBitmap1 is null)
		{
			return;
		}

		Bitmap? distortedBitmap2 = Undistorters.ReadDistorted(imageSource, image2.Name, "Cannot read image at path ");
		if (distortedBitmap2 is null)
		{
			return;
		}

		Rigid3d cam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse();

		Undistortion.RectifyAndUndistortStereoImages(
			cameraOptions,
			distortedBitmap1,
			distortedBitmap2,
			camera1,
			camera2,
			cam2FromCam1,
			out Bitmap undistortedBitmap1,
			out Bitmap undistortedBitmap2,
			out _,
			out Matrix4d q);

		Undistorters.MaybeSetJpegQuality(outputImagePath1, undistortedBitmap1, options.JpegQuality);
		Undistorters.MaybeSetJpegQuality(outputImagePath2, undistortedBitmap2, options.JpegQuality);

		// COLMAP ignores Bitmap::Write's result here.
		imageSink.Write(outputImagePath1, undistortedBitmap1);
		imageSink.Write(outputImagePath2, undistortedBitmap2);

		var qFile = new StringBuilder();
		Undistorters.WriteMatrix(qFile, 4, 4, (r, c) => q[r, c]);
		Undistorters.WriteTextFile(Path.Combine(outputPath, stereoPairName, "Q.txt"), qFile);
	}
}
