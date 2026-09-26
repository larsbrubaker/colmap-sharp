// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Undistorters: the shared part of colmap/controllers/undistorters.h/.cc - the anonymous
// namespace helpers (MaybeSetJpegQuality, WriteMatrix, WriteProjectionMatrix,
// WriteCOLMAPCommands) and the task loop all five controllers share. The controllers are in
// ColmapUndistorter.cs, PmvsUndistorter.cs (PMVS and CMP-MVS) and
// StandaloneImageUndistorter.cs (standalone and stereo rectification); the image math is
// ImageProcessing/Undistortion.cs. Tests: ColmapSharp.Tests/Controllers/UndistortersTests.cs
// (undistorters_test.cc 1:1).
//
// Tier A (exact) for the text outputs; the images inherit Undistortion.cs's tiers.
//
// Translation notes (they apply to every controller):
// - Images are read through an IImageSource (the image_path folder; Bitmap::Read decoding is
//   the host's, docs/CPP_DIVERGENCES.md entry 82) and written through an IBitmapSink under
//   the path COLMAP would write (Bitmap::Write encoding is the host's, entry 98). Text files
//   (configs, scripts, projection matrices, bundle files) are written to the file system as
//   in COLMAP, and so are the directories COLMAP creates.
// - COLMAP's ThreadPool runs every image as a task and the main thread waits on the futures
//   in order, checking CheckIfStopped before each; Stop() drops the tasks not yet started.
//   Here RunTasks starts the images in batches of num_threads (Parallel.For, each task
//   writes only its own result slot) and checks CheckIfStopped before each image's result
//   is consumed, starting the next batch only when not stopped. Stopping thus finishes the
//   batch in flight, where COLMAP finishes the tasks its threads had already picked up.
// - Stopping (CheckIfStopped: the stop function or the CancellationToken property) keeps
//   COLMAP's semantics: Run returns normally, without OperationCanceledException.
// - The LOG(INFO) "Undistorting image [i/n]" lines become each controller's Progress; the
//   headings, timers and other LOG(INFO) lines are not ported. LOG(WARNING)/LOG(ERROR) go
//   to Util/Log.cs.
// - std::filesystem::path streamed into a script is quoted (std::quoted), and
//   `workspace_path / name` joins with '/', as on the POSIX systems the scripts are for.

using System.Globalization;
using System.Text;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>The helpers colmap/controllers/undistorters.cc keeps in its anonymous namespace.</summary>
internal static class Undistorters
{
	/// <summary>The <see cref="ControllerProgress.Stage"/> of the undistorters.</summary>
	public const string UndistortionStage = "Image undistortion";

	/// <summary>The <see cref="ControllerProgress.Stage"/> of <see cref="StereoImageRectifier"/>.</summary>
	public const string RectificationStage = "Stereo rectification";

	/// <summary>
	/// Port of MaybeSetJpegQuality: asks the encoder for <paramref name="jpegQuality"/> when
	/// the output is a JPEG and a quality was given.
	/// </summary>
	public static void MaybeSetJpegQuality(string path, Bitmap bitmap, int jpegQuality)
	{
		if ((FileUtils.HasFileExtension(path, ".jpg") || FileUtils.HasFileExtension(path, ".jpeg")) && jpegQuality > 0)
		{
			bitmap.SetMetaData("Compression", "jpeg:" + jpegQuality.ToString(CultureInfo.InvariantCulture));
		}
	}

	/// <summary>
	/// Reads the distorted image <paramref name="name"/> like Bitmap::Read(path) (as RGB),
	/// logging COLMAP's error and returning null when it cannot be read.
	/// </summary>
	public static Bitmap? ReadDistorted(IImageSource imageSource, string name, string errorPrefix)
	{
		Bitmap? bitmap = ImageReader.ReadBitmap(imageSource, name, asRgb: true);
		if (bitmap is null)
		{
			Log.Error(errorPrefix + name);
		}

		return bitmap;
	}

	/// <summary>
	/// COLMAP's FileCopy of an image that needs no undistortion: the host's decoded image is
	/// handed to the sink unchanged (entry 98). False when it cannot be read.
	/// </summary>
	public static bool CopyImage(IImageSource imageSource, string name, IBitmapSink sink, string outputPath)
	{
		Bitmap? bitmap = imageSource.Read(name);
		return bitmap is not null && sink.Write(outputPath, bitmap);
	}

	/// <summary>Port of WriteMatrix: one row per line, entries at the stream's default precision.</summary>
	public static void WriteMatrix(StringBuilder file, int rows, int cols, Func<int, int, double> matrix)
	{
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols - 1; ++c)
			{
				file.Append(CppStreamFormat.FormatDouble(matrix(r, c))).Append(' ');
			}

			file.Append(CppStreamFormat.FormatDouble(matrix(r, cols - 1))).Append('\n');
		}
	}

	/// <summary>
	/// Port of WriteProjectionMatrix: writes P = K * [R t] to <paramref name="path"/>,
	/// preceded by <paramref name="header"/> when it is not empty.
	/// </summary>
	public static void WriteProjectionMatrix(string path, Camera camera, Image image, string header)
	{
		Check.That(camera.ModelId == CameraModelId.Pinhole);

		var calibMatrix = new Matrix3d(
			camera.FocalLengthX(), 0, camera.PrincipalPointX(),
			0, camera.FocalLengthY(), camera.PrincipalPointY(),
			0, 0, 1);

		Matrix3x4d imgFromWorld = calibMatrix * image.CamFromWorld().ToMatrix();

		var file = new StringBuilder();
		if (header.Length > 0)
		{
			file.Append(header).Append('\n');
		}

		WriteMatrix(file, 3, 4, (r, c) => imgFromWorld[r, c]);
		WriteTextFile(path, file);
	}

	/// <summary>
	/// Writes <paramref name="text"/> to <paramref name="path"/>, replacing it (std::ofstream
	/// with std::ios::trunc and THROW_CHECK_FILE_OPEN).
	/// </summary>
	public static void WriteTextFile(string path, StringBuilder text)
	{
		using var writer = new StreamWriter(FileOpen.OpenWrite(path));
		writer.Write(text.ToString());
	}

	/// <summary>std::filesystem::path's operator&lt;&lt;: the path in std::quoted form.</summary>
	public static string QuotePath(string path)
	{
		var quoted = new StringBuilder("\"");
		foreach (char c in path)
		{
			if (c == '"' || c == '\\')
			{
				quoted.Append('\\');
			}

			quoted.Append(c);
		}

		return quoted.Append('"').ToString();
	}

	/// <summary>
	/// Port of WriteCOLMAPCommands: the patch_match_stereo, stereo_fusion, poisson_mesher and
	/// delaunay_mesher calls of the generated shell scripts.
	/// </summary>
	public static void WriteColmapCommands(
		bool geometric,
		string workspacePath,
		string workspaceFormat,
		string pmvsOptionName,
		string outputPrefix,
		string indent,
		StringBuilder file)
	{
		string workspace = QuotePath(workspacePath);
		string Joined(string name) => QuotePath(workspacePath + "/" + name);

		file.Append(indent).Append("$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n");
		file.Append(indent).Append("  --workspace_path ").Append(workspace).Append(" \\\n");
		file.Append(indent).Append("  --workspace_format ").Append(workspaceFormat).Append(" \\\n");
		if (workspaceFormat == "PMVS")
		{
			file.Append(indent).Append("  --pmvs_option_name ").Append(pmvsOptionName).Append(" \\\n");
		}

		file.Append(indent).Append("  --PatchMatchStereo.max_image_size 2000 \\\n");
		file.Append(indent).Append(geometric
			? "  --PatchMatchStereo.geom_consistency true\n"
			: "  --PatchMatchStereo.geom_consistency false\n");

		file.Append(indent).Append("$COLMAP_EXE_PATH/colmap stereo_fusion \\\n");
		file.Append(indent).Append("  --workspace_path ").Append(workspace).Append(" \\\n");
		file.Append(indent).Append("  --workspace_format ").Append(workspaceFormat).Append(" \\\n");
		if (workspaceFormat == "PMVS")
		{
			file.Append(indent).Append("  --pmvs_option_name ").Append(pmvsOptionName).Append(" \\\n");
		}

		file.Append(indent).Append(geometric ? "  --input_type geometric \\\n" : "  --input_type photometric \\\n");
		file.Append(indent).Append("  --output_path ").Append(Joined(outputPrefix + "fused.ply")).Append(" \\\n");

		file.Append(indent).Append("$COLMAP_EXE_PATH/colmap poisson_mesher \\\n");
		file.Append(indent).Append("  --input_path ").Append(Joined(outputPrefix + "fused.ply")).Append(" \\\n");
		file.Append(indent).Append("  --output_path ").Append(Joined(outputPrefix + "meshed-poisson.ply")).Append(" \\\n");

		file.Append(indent).Append("$COLMAP_EXE_PATH/colmap delaunay_mesher \\\n");
		file.Append(indent).Append("  --input_path ").Append(Joined(outputPrefix)).Append(" \\\n");
		file.Append(indent).Append("  --input_type dense \\\n");
		file.Append(indent).Append("  --output_path ").Append(Joined(outputPrefix + "meshed-delaunay.ply")).Append(" \\\n");
	}

	/// <summary>
	/// The ThreadPool + futures loop of every controller's Run: runs <paramref name="task"/>
	/// for indices 0..count-1 on up to <paramref name="numThreads"/> threads and hands each
	/// result to <paramref name="consume"/> in index order, checking the controller's stop
	/// state before each. Returns false when stopped (see the header for how the in-flight
	/// work differs from COLMAP).
	/// </summary>
	public static bool RunTasks<T>(
		BaseController controller,
		int numThreads,
		int count,
		Func<int, T> task,
		Action<int, T> consume)
	{
		int batchSize = Threading.GetEffectiveNumThreads(numThreads);
		Check.Gt(batchSize, 0);
		var results = new T[count];
		var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = batchSize };
		int numStarted = 0;
		for (int i = 0; i < count; ++i)
		{
			if (controller.CheckIfStopped())
			{
				return false;
			}

			if (i == numStarted)
			{
				int end = Math.Min(count, i + batchSize);
				Parallel.For(i, end, parallelOptions, j => results[j] = task(j));
				numStarted = end;
			}

			consume(i, results[i]);
		}

		return true;
	}
}
