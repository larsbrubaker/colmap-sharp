// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIO: the exporters of colmap/scene/reconstruction_io.cc (ExportNVM, ExportCam,
// ExportRecon3D here; ExportBundler, ExportPLY, ExportVRML in ReconstructionIO.Export2.cs),
// which write a Reconstruction for other tools (VisualSfM, Meshlab, Bundler, VRML viewers).
// COLMAP's own formats are ReconstructionIOText*.cs / ReconstructionIOBinary.cs.
//
// Text is written exactly as COLMAP's streams write it: the files use precision(17) in the
// classic locale, but the per-point ostringstreams that build the track lines keep the
// default precision 6 (so observation coordinates are written with 6 significant digits),
// and VRML files are never given a precision. Doubles go through Util/CppStreamFormat.cs,
// Eigen matrices through Util/EigenStreamFormat.cs. An unsupported camera model makes an
// exporter return false after it has already written what came before, as in COLMAP, with
// its LOG(WARNING) (trailing newline included where COLMAP's text has one) to Util/Log.cs. COLMAP walks Points3D() (and, for VRML,
// Images()), which are hash maps; here they are walked in ascending id order
// (docs/CPP_DIVERGENCES.md, entry 21, which already covers every Reconstruction walk).

using System.Globalization;
using System.Text;

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Exports a reconstruction in other tools' formats. Port of reconstruction_io.h.</summary>
public static partial class ReconstructionIO
{
	// "Ensure that we don't lose any precision by storing in text."
	private const int FilePrecision = 17;

	/// <summary>
	/// Port of colmap::ExportNVM: NVM_V3 (http://ccwu.me/vsfm/doc.html#nvm). Only
	/// SIMPLE_RADIAL and pinhole models unless <paramref name="skipDistortion"/>, which
	/// writes k = 0 and the mean focal length for every model. Returns false for an
	/// unsupported model.
	/// </summary>
	public static bool ExportNVM(Reconstruction reconstruction, string path, bool skipDistortion = false)
	{
		using StreamWriter file = OpenText(path);

		// White space added for compatibility with Meshlab.
		file.Write("NVM_V3 \n \n");
		file.Write(Int(reconstruction.NumRegImages) + "  \n");

		var imageIdToIdx = new Dictionary<uint, long>();
		long imageIdx = 0;

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			Camera camera = reconstruction.Camera(image.CameraId);

			double k;
			if (skipDistortion || IsPinhole(camera))
			{
				k = 0.0;
			}
			else if (camera.ModelId == CameraModelId.SimpleRadial)
			{
				k = -1 * camera.Params[camera.ExtraParamsIdxs[0]];
			}
			else
			{
				Log.Warning("NVM only supports `SIMPLE_RADIAL` and pinhole camera models.\n");
				return false;
			}

			var projCenter = image.ProjectionCenter();
			var camFromWorld = image.CamFromWorld();

			var line = new StringBuilder();
			line.Append(image.Name).Append(' ');
			line.Append(Dbl(camera.MeanFocalLength())).Append(' ');
			line.Append(Dbl(camFromWorld.Rotation.W)).Append(' ');
			line.Append(Dbl(camFromWorld.Rotation.X)).Append(' ');
			line.Append(Dbl(camFromWorld.Rotation.Y)).Append(' ');
			line.Append(Dbl(camFromWorld.Rotation.Z)).Append(' ');
			line.Append(Dbl(projCenter.X)).Append(' ');
			line.Append(Dbl(projCenter.Y)).Append(' ');
			line.Append(Dbl(projCenter.Z)).Append(' ');
			line.Append(Dbl(k)).Append(' ');
			line.Append("0\n");
			file.Write(line.ToString());

			imageIdToIdx[imageId] = imageIdx;
			imageIdx += 1;
		}

		file.Write("\n" + Int(reconstruction.NumPoints3D) + "\n");

		foreach (ulong point3DId in SortedPoint3DIds(reconstruction))
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			var text = new StringBuilder();
			AppendXyz(text, point3D, " ", " ");
			text.Append(' ');
			AppendColor(text, point3D, " ");
			text.Append(' ');

			var line = new StringBuilder();
			var imageIds = new HashSet<uint>();
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				// Make sure that each point only has a single observation per image,
				// since VisualSfM does not support with multiple observations.
				if (!imageIds.Contains(trackEl.ImageId))
				{
					Image image = reconstruction.Image(trackEl.ImageId);
					Point2D point2D = Point2DAt(image, trackEl.Point2DIdx);
					// COLMAP's image_id_to_idx_[id] default-inserts 0 for an image that is
					// not registered.
					line.Append(Int(imageIdToIdx.GetValueOrDefault(trackEl.ImageId))).Append(' ');
					line.Append(Int(trackEl.Point2DIdx)).Append(' ');
					line.Append(Dbl6(point2D.Xy.X)).Append(' ');
					line.Append(Dbl6(point2D.Xy.Y)).Append(' ');
					imageIds.Add(trackEl.ImageId);
				}
			}

			text.Append(Int(imageIds.Count)).Append(' ');
			text.Append(DropLastChar(line)).Append('\n');
			file.Write(text.ToString());
		}

		return true;
	}

	/// <summary>
	/// Port of colmap::ExportCam: one "&lt;name without extension&gt;.cam" file per registered
	/// image under <paramref name="path"/>, with the pose (translation, row-major rotation) on
	/// the first line and focal length (relative to the larger image side), k1, k2, the
	/// pixel aspect ratio fy / fx and the principal point (relative to width and height) on
	/// the second. SIMPLE_RADIAL, RADIAL and pinhole models only, unless
	/// <paramref name="skipDistortion"/>. Returns false for an unsupported model.
	/// </summary>
	public static bool ExportCam(Reconstruction reconstruction, string path, bool skipDistortion = false)
	{
		reconstruction.CreateImageDirs(path);
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			Camera camera = reconstruction.Camera(image.CameraId);

			(string name, _) = SplitFileExtension(image.Name);
			string namePath = Path.Combine(path, name + ".cam");
			using StreamWriter file = OpenText(namePath);

			double k1, k2;
			if (skipDistortion || IsPinhole(camera))
			{
				k1 = 0.0;
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.SimpleRadial)
			{
				k1 = camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.Radial)
			{
				k1 = camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = camera.Params[camera.ExtraParamsIdxs[1]];
			}
			else
			{
				Log.Warning("CAM only supports `SIMPLE_RADIAL`, `RADIAL`, and pinhole camera models.\n");
				return false;
			}

			// If both k1 and k2 values are non-zero, then the CAM format assumes
			// a Bundler-like radial distortion model, which converts well from
			// COLMAP. However, if k2 is zero, then a different model is used
			// that does not translate as well, so we avoid setting k2 to zero.
			if (k1 != 0.0 && k2 == 0.0)
			{
				k2 = 1e-10;
			}

			double fx = camera.FocalLengthX();
			double fy = camera.FocalLengthY();
			double focalLength = camera.Width * fy < camera.Height * fx
				? fy / camera.Height
				: fx / camera.Width;

			var camFromWorld = image.CamFromWorld();
			var r = camFromWorld.Rotation.ToRotationMatrix();
			var text = new StringBuilder();
			text.Append(Dbl(camFromWorld.Translation.X)).Append(' ')
				.Append(Dbl(camFromWorld.Translation.Y)).Append(' ')
				.Append(Dbl(camFromWorld.Translation.Z));
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					text.Append(' ').Append(Dbl(r[row, col]));
				}
			}

			text.Append('\n');
			text.Append(Dbl(focalLength)).Append(' ').Append(Dbl(k1)).Append(' ').Append(Dbl(k2)).Append(' ')
				.Append(Dbl(fy / fx)).Append(' ')
				.Append(Dbl(camera.PrincipalPointX() / camera.Width)).Append(' ')
				.Append(Dbl(camera.PrincipalPointY() / camera.Height)).Append('\n');
			file.Write(text.ToString());
		}

		return true;
	}

	/// <summary>
	/// Port of colmap::ExportRecon3D: <c>Recon/synth_0.out</c>, <c>Recon/urd-images.txt</c>
	/// and <c>Recon/imagemap_0.txt</c> under <paramref name="path"/> (see reconstruction_io.h
	/// for the layout). SIMPLE_RADIAL, RADIAL and pinhole models only, unless
	/// <paramref name="skipDistortion"/>. Returns false for an unsupported model.
	/// </summary>
	public static bool ExportRecon3D(Reconstruction reconstruction, string path, bool skipDistortion = false)
	{
		Directory.CreateDirectory(path);
		string basePath = Path.Combine(path, "Recon");
		Directory.CreateDirectory(basePath);

		using StreamWriter synthFile = OpenText(Path.Combine(basePath, "synth_0.out"));
		using StreamWriter imageListFile = OpenText(Path.Combine(basePath, "urd-images.txt"));
		using StreamWriter imageMapFile = OpenText(Path.Combine(basePath, "imagemap_0.txt"));

		// Write header info
		synthFile.Write("colmap 1.0\n");
		synthFile.Write(Int(reconstruction.NumRegImages) + " " + Int(reconstruction.NumPoints3D) + "\n");

		var imageIdToIdx = new Dictionary<uint, long>();
		long imageIdx = 0;

		// Write image/camera info
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			Camera camera = reconstruction.Camera(image.CameraId);

			double k1, k2;
			if (skipDistortion || IsPinhole(camera))
			{
				k1 = 0.0;
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.SimpleRadial)
			{
				k1 = -1 * camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.Radial)
			{
				k1 = -1 * camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = -1 * camera.Params[camera.ExtraParamsIdxs[1]];
			}
			else
			{
				Log.Warning("Recon3D only supports `SIMPLE_RADIAL`, `RADIAL`, and pinhole camera models.");
				return false;
			}

			double scale = 1.0 / Math.Max(camera.Width, camera.Height);
			var camFromWorld = image.CamFromWorld();
			var rotation = camFromWorld.Rotation.ToRotationMatrix();
			var translation = camFromWorld.Translation;
			synthFile.Write(Dbl(scale * camera.MeanFocalLength()) + " " + Dbl(k1) + " " + Dbl(k2) + "\n");
			synthFile.Write(EigenStreamFormat.FormatMatrix(3, 3, (r, c) => rotation[r, c], FilePrecision) + "\n");
			synthFile.Write(EigenStreamFormat.FormatMatrix(1, 3, (_, c) => translation[c], FilePrecision) + "\n");

			imageIdToIdx[imageId] = imageIdx;
			imageListFile.Write(image.Name + "\n" + Int(camera.Width) + " " + Int(camera.Height) + "\n");
			imageMapFile.Write(Int(imageIdx) + "\n");

			imageIdx += 1;
		}

		imageListFile.Close();
		imageMapFile.Close();

		// Write point info
		foreach (ulong point3DId in SortedPoint3DIds(reconstruction))
		{
			Point3D p = reconstruction.Point3D(point3DId);
			var text = new StringBuilder();
			AppendXyz(text, p, " ", " ");
			text.Append('\n');
			AppendColor(text, p, " ");
			text.Append('\n');

			var line = new StringBuilder();
			var imageIds = new HashSet<uint>();
			foreach (TrackElement trackEl in p.Track.Elements)
			{
				// Make sure that each point only has a single observation per image,
				// since VisualSfM does not support with multiple observations.
				if (!imageIds.Contains(trackEl.ImageId))
				{
					Image image = reconstruction.Image(trackEl.ImageId);
					Camera camera = reconstruction.Camera(image.CameraId);
					Point2D point2D = Point2DAt(image, trackEl.Point2DIdx);

					double scale = 1.0 / Math.Max(camera.Width, camera.Height);

					line.Append(Int(imageIdToIdx.GetValueOrDefault(trackEl.ImageId))).Append(' ');
					line.Append(Int(trackEl.Point2DIdx)).Append(' ');
					// Use a scale of -1.0 to mark as invalid as it is not needed currently
					line.Append("-1.0 ");
					line.Append(Dbl6((point2D.Xy.X - camera.PrincipalPointX()) * scale)).Append(' ');
					line.Append(Dbl6((point2D.Xy.Y - camera.PrincipalPointY()) * scale)).Append(' ');
					imageIds.Add(trackEl.ImageId);
				}
			}

			text.Append(Int(imageIds.Count)).Append(' ');
			text.Append(DropLastChar(line)).Append('\n');
			synthFile.Write(text.ToString());
		}

		return true;
	}

	/// <summary>
	/// Port of colmap::SplitFileExtension (util/file.cc): "dir/a.b.jpg" is ("dir/a.b",
	/// ".jpg"); a name without a '.' has no extension. Like COLMAP's StringSplit (boost::split
	/// with token_compress_on), runs of '.' count as one separator, so "a..jpg" has root "a".
	/// </summary>
	internal static (string Root, string Ext) SplitFileExtension(string path)
	{
		var parts = new List<string>();
		int start = 0;
		int i = 0;
		while (i < path.Length)
		{
			if (path[i] == '.')
			{
				parts.Add(path[start..i]);
				while (i < path.Length && path[i] == '.')
				{
					i++;
				}

				start = i;
			}
			else
			{
				i++;
			}
		}

		parts.Add(path[start..]);
		if (parts.Count == 1)
		{
			return (parts[0], string.Empty);
		}

		string root = string.Join(".", parts.Take(parts.Count - 1));
		string ext = parts[^1].Length == 0 ? string.Empty : "." + parts[^1];
		return (root, ext);
	}

	private static bool IsPinhole(Camera camera) =>
		camera.ModelId == CameraModelId.SimplePinhole || camera.ModelId == CameraModelId.Pinhole;

	// std::ofstream(path, std::ios::trunc) + THROW_CHECK_FILE_OPEN; UTF-8 without a BOM, so
	// image names come out as the bytes COLMAP's std::string holds.
	private static StreamWriter OpenText(string path) =>
		new(FileOpen.OpenWrite(path), new UTF8Encoding(false));

	private static List<ulong> SortedPoint3DIds(Reconstruction reconstruction) =>
		ReconstructionIOUtils.ExtractSortedIds(reconstruction.Points3D);

	// Image::Point2D(idx), which is std::vector::at.
	private static Point2D Point2DAt(Image image, uint point2DIdx)
	{
		if (point2DIdx >= image.NumPoints2D)
		{
			throw new ArgumentOutOfRangeException(nameof(point2DIdx), "vector");
		}

		return image.Points2D[(int)point2DIdx];
	}

	private static void AppendXyz(StringBuilder text, Point3D point3D, string separator1, string separator2) =>
		text.Append(Dbl(point3D.Xyz.X)).Append(separator1).Append(Dbl(point3D.Xyz.Y)).Append(separator2).Append(Dbl(point3D.Xyz.Z));

	// static_cast<int>(color(i)), separated.
	private static void AppendColor(StringBuilder text, Point3D point3D, string separator) =>
		text.Append(Int(point3D.Color.X)).Append(separator).Append(Int(point3D.Color.Y)).Append(separator).Append(Int(point3D.Color.Z));

	// line_string.substr(0, line_string.size() - 1): drops the trailing space; for an empty
	// line the size_t wraps to npos and the result stays empty.
	private static string DropLastChar(StringBuilder line) =>
		line.Length == 0 ? string.Empty : line.ToString(0, line.Length - 1);

	private static string Dbl(double value) => CppStreamFormat.FormatDouble(value, FilePrecision);

	// An ostringstream (or file) without precision(): the default 6.
	private static string Dbl6(double value) => CppStreamFormat.FormatDouble(value);

	private static string Int(long value) => value.ToString(CultureInfo.InvariantCulture);
}
