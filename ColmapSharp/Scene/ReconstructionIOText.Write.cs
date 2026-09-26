// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOText.Write: the writers of colmap/scene/reconstruction_io_text.cc (the
// readers are in ReconstructionIOText.cs). COLMAP streams every double with
// precision(17) in the classic locale, which is printf's %.17g (Util/CppStreamFormat.cs), and
// the files are byte-identical to pycolmap's write_text. Two quirks are kept on purpose
// because they are part of the format COLMAP writes: a points3D.txt line always ends in a
// space before the track (so a point with an empty track ends "ERROR \n"), and an images.txt
// POINTS2D line keeps its trailing space (COLMAP's seekp(-1) does not truncate the
// ostringstream).

using System.Globalization;
using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public static partial class ReconstructionIOText
{
	// "Ensure that we don't loose any precision by storing in text."
	private const int TextPrecision = 17;

	/// <summary>WriteRigsText: every rig, in ascending id order.</summary>
	public static void WriteRigsText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		var text = new StringBuilder();
		text.Append("# Rig calib list with one line of data per calib:\n");
		text.Append("#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n");
		text.Append("# Number of rigs: ").Append(Int(reconstruction.NumRigs)).Append('\n');

		foreach (uint rigId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Rigs))
		{
			Rig rig = reconstruction.Rig(rigId);
			var line = new StringBuilder();
			line.Append(Int(rigId)).Append(' ');
			line.Append(Int(rig.NumSensors)).Append(' ');

			if (rig.NumSensors > 0)
			{
				line.Append(rig.RefSensorId.Type.ToColmapString()).Append(' ');
				line.Append(Int(rig.RefSensorId.Id)).Append(' ');
			}

			foreach (var (sensorId, sensorFromRig) in rig.NonRefSensors)
			{
				line.Append(sensorId.Type.ToColmapString()).Append(' ');
				line.Append(Int(sensorId.Id)).Append(' ');
				if (sensorFromRig.HasValue)
				{
					line.Append("1 ");
					AppendRigid3d(line, sensorFromRig.Value);
				}
				else
				{
					line.Append("0 ");
				}
			}

			AppendWithoutLastChar(text, line);
			text.Append('\n');
		}

		WriteText(stream, text);
	}

	/// <summary>WriteRigsText to a file (created or truncated).</summary>
	public static void WriteRigsText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenWrite(path);
		WriteRigsText(reconstruction, file);
	}

	/// <summary>WriteCamerasText: every camera, in ascending id order.</summary>
	public static void WriteCamerasText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		var text = new StringBuilder();
		text.Append("# Camera list with one line of data per camera:\n");
		text.Append("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
		text.Append("# Number of cameras: ").Append(Int(reconstruction.NumCameras)).Append('\n');

		foreach (uint cameraId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Cameras))
		{
			Camera camera = reconstruction.Camera(cameraId);
			var line = new StringBuilder();
			line.Append(Int(cameraId)).Append(' ');
			line.Append(camera.ModelName).Append(' ');
			line.Append(Int(camera.Width)).Append(' ');
			line.Append(Int(camera.Height)).Append(' ');
			foreach (double param in camera.Params)
			{
				line.Append(Dbl(param)).Append(' ');
			}

			AppendWithoutLastChar(text, line);
			text.Append('\n');
		}

		WriteText(stream, text);
	}

	/// <summary>WriteCamerasText to a file (created or truncated).</summary>
	public static void WriteCamerasText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenWrite(path);
		WriteCamerasText(reconstruction, file);
	}

	/// <summary>WriteFramesText: the frames with a pose, in ascending id order.</summary>
	public static void WriteFramesText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		List<uint> frameIds = ReconstructionIOUtils.ExtractSortedIds(reconstruction.Frames, frame => frame.HasPose);

		var text = new StringBuilder();
		text.Append("# Frame list with one line of data per frame:\n");
		text.Append("#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n");
		text.Append("# Number of frames: ").Append(Int(frameIds.Count)).Append('\n');

		foreach (uint frameId in frameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			text.Append(Int(frameId)).Append(' ');
			text.Append(Int(frame.RigId)).Append(' ');
			AppendRigid3d(text, frame.RigFromWorld());

			text.Append(Int(frame.NumDataIds));
			foreach (DataId dataId in frame.DataIds)
			{
				text.Append(' ').Append(dataId.SensorId.Type.ToColmapString())
					.Append(' ').Append(Int(dataId.SensorId.Id))
					.Append(' ').Append(dataId.Id.ToString(CultureInfo.InvariantCulture));
			}

			text.Append('\n');
		}

		WriteText(stream, text);
	}

	/// <summary>WriteFramesText to a file (created or truncated).</summary>
	public static void WriteFramesText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenWrite(path);
		WriteFramesText(reconstruction, file);
	}

	/// <summary>WriteImagesText: the registered images, in registration order, two lines each.</summary>
	public static void WriteImagesText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		var text = new StringBuilder();
		text.Append("# Image list with two lines of data per image:\n");
		text.Append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
		text.Append("#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
		text.Append("# Number of images: ").Append(Int(reconstruction.NumRegImages))
			.Append(", mean observations per image: ").Append(Dbl(reconstruction.ComputeMeanObservationsPerRegImage()))
			.Append('\n');

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			text.Append(Int(imageId)).Append(' ');
			AppendRigid3d(text, image.CamFromWorld());
			text.Append(Int(image.CameraId)).Append(' ');
			text.Append(image.Name).Append('\n');

			foreach (Point2D point2D in image.Points2D)
			{
				text.Append(Dbl(point2D.Xy.X)).Append(' ');
				text.Append(Dbl(point2D.Xy.Y)).Append(' ');
				if (point2D.HasPoint3D)
				{
					text.Append(point2D.Point3DId.ToString(CultureInfo.InvariantCulture)).Append(' ');
				}
				else
				{
					text.Append("-1 ");
				}
			}

			// COLMAP moves the put pointer back over the last space here (seekp(-1)), but
			// ostringstream::str() still returns it, so the trailing space stays.
			text.Append('\n');
		}

		WriteText(stream, text);
	}

	/// <summary>WriteImagesText to a file (created or truncated).</summary>
	public static void WriteImagesText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenWrite(path);
		WriteImagesText(reconstruction, file);
	}

	/// <summary>WritePoints3DText: every 3D point, in ascending id order.</summary>
	public static void WritePoints3DText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		var text = new StringBuilder();
		text.Append("# 3D point list with one line of data per point:\n");
		text.Append("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
		text.Append("# Number of points: ").Append(Int(reconstruction.NumPoints3D))
			.Append(", mean track length: ").Append(Dbl(reconstruction.ComputeMeanTrackLength()))
			.Append('\n');

		foreach (ulong point3DId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Points3D))
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			text.Append(point3DId.ToString(CultureInfo.InvariantCulture)).Append(' ');
			text.Append(Dbl(point3D.Xyz.X)).Append(' ');
			text.Append(Dbl(point3D.Xyz.Y)).Append(' ');
			text.Append(Dbl(point3D.Xyz.Z)).Append(' ');
			text.Append(Int(point3D.Color.X)).Append(' ');
			text.Append(Int(point3D.Color.Y)).Append(' ');
			text.Append(Int(point3D.Color.Z)).Append(' ');
			text.Append(Dbl(point3D.Error)).Append(' ');

			var line = new StringBuilder();
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				line.Append(Int(trackEl.ImageId)).Append(' ');
				line.Append(Int(trackEl.Point2DIdx)).Append(' ');
			}

			// substr(0, size() - 1) of an empty string is the empty string.
			AppendWithoutLastChar(text, line);
			text.Append('\n');
		}

		WriteText(stream, text);
	}

	/// <summary>WritePoints3DText to a file (created or truncated).</summary>
	public static void WritePoints3DText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenWrite(path);
		WritePoints3DText(reconstruction, file);
	}

	private static string Dbl(double value) => CppStreamFormat.FormatDouble(value, TextPrecision);

	private static string Int(long value) => value.ToString(CultureInfo.InvariantCulture);

	// QW QX QY QZ TX TY TZ, each followed by a space.
	private static void AppendRigid3d(StringBuilder text, Rigid3d transform)
	{
		text.Append(Dbl(transform.Rotation.W)).Append(' ');
		text.Append(Dbl(transform.Rotation.X)).Append(' ');
		text.Append(Dbl(transform.Rotation.Y)).Append(' ');
		text.Append(Dbl(transform.Rotation.Z)).Append(' ');
		text.Append(Dbl(transform.Translation.X)).Append(' ');
		text.Append(Dbl(transform.Translation.Y)).Append(' ');
		text.Append(Dbl(transform.Translation.Z)).Append(' ');
	}

	// line_string.substr(0, line_string.size() - 1): drops the trailing separator.
	private static void AppendWithoutLastChar(StringBuilder text, StringBuilder line)
	{
		if (line.Length > 0)
		{
			text.Append(line, 0, line.Length - 1);
		}
	}

	private static void WriteText(Stream stream, StringBuilder text)
	{
		byte[] bytes = new UTF8Encoding(false).GetBytes(text.ToString());
		stream.Write(bytes, 0, bytes.Length);
	}
}
