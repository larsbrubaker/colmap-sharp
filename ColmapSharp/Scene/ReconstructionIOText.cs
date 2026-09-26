// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOText: port of colmap/scene/reconstruction_io_text.h/.cc, COLMAP's rigs.txt,
// cameras.txt, frames.txt, images.txt and points3D.txt formats. This file holds the readers;
// ReconstructionIOText.Write.cs holds the writers. Lines are read with std::getline +
// StringTrim semantics, blank and '#' lines are skipped, and values are parsed token by token
// in the classic locale (CppLineTokens.cs). Each reader/writer has a Stream overload (the
// stream is left open) and a path overload. A THROW_CHECK(line_stream >> a >> b) becomes
// Check.That(TryRead(a) & TryRead(b)): the non-short-circuit '&' keeps every out variable
// definitely assigned, and the check still fails if any extraction does. Neighbors: ReconstructionIOBinary.cs (the .bin
// formats), ReconstructionIOUtils.cs, Reconstruction.IO.cs. Tests:
// ColmapSharp.Tests/Scene/ReconstructionIOOracleTests.cs.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap/scene/reconstruction_io_text.h.</summary>
public static partial class ReconstructionIOText
{
	/// <summary>ReadRigsText.</summary>
	public static void ReadRigsText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		foreach (CppLineTokens lineStream in DataLines(stream))
		{
			var rig = new Rig();

			// ID, NUM_SENSORS
			Check.That(
				lineStream.TryReadUInt32(out uint rigId) & lineStream.TryReadUInt64(out ulong numSensors),
				null,
				"line_stream >> rig_id >> num_sensors");
			rig.RigId = rigId;

			if (numSensors > 0)
			{
				// REF_SENSOR
				Check.That(
					lineStream.TryReadString(out string item) & lineStream.TryReadUInt32(out uint refId),
					null,
					"line_stream >> item >> ref_sensor_id.id");
				rig.AddRefSensor(new SensorId(SensorTypeExtensions.SensorTypeFromString(item), refId));
			}

			// SENSORS
			if (numSensors > 1)
			{
				for (ulong i = 0; i < numSensors - 1; ++i)
				{
					Check.That(
						lineStream.TryReadString(out string item) & lineStream.TryReadUInt32(out uint id) &
						lineStream.TryReadInt32(out int hasPoseInt),
						null,
						"line_stream >> item >> sensor_id.id >> has_pose_int");
					var sensorId = new SensorId(SensorTypeExtensions.SensorTypeFromString(item), id);
					bool hasPose = hasPoseInt == 1;

					Rigid3d? sensorFromRig = null;
					if (hasPose)
					{
						Check.That(TryReadRigid3d(lineStream, out Rigid3d pose), null, "line_stream >> sensor_from_rig");
						sensorFromRig = pose;
					}

					rig.AddSensor(sensorId, sensorFromRig);
				}
			}

			reconstruction.AddRig(rig);
		}
	}

	/// <summary>ReadRigsText from a file.</summary>
	public static void ReadRigsText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenRead(path);
		ReadRigsText(reconstruction, file);
	}

	/// <summary>ReadCamerasText. PARAMS are read until the line ends or a token is not a number.</summary>
	public static void ReadCamerasText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		foreach (CppLineTokens lineStream in DataLines(stream))
		{
			// ID, MODEL, WIDTH, HEIGHT
			Check.That(
				lineStream.TryReadUInt32(out uint cameraId) & lineStream.TryReadString(out string item) &
				lineStream.TryReadUInt64(out ulong width) & lineStream.TryReadUInt64(out ulong height),
				null,
				"line_stream >> camera.camera_id >> item >> camera.width >> camera.height");
			var camera = new Camera
			{
				CameraId = cameraId,
				ModelId = CameraModels.CameraModelNameToId(item),
				Width = checked((int)width),
				Height = checked((int)height),
			};

			// PARAMS
			var cameraParams = new List<double>(CameraModels.CameraModelNumParams(camera.ModelId));
			while (lineStream.TryReadDouble(out double param))
			{
				cameraParams.Add(param);
			}

			camera.Params = [.. cameraParams];
			Check.That(camera.VerifyParams());
			reconstruction.AddCamera(camera);
		}
	}

	/// <summary>ReadCamerasText from a file.</summary>
	public static void ReadCamerasText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenRead(path);
		ReadCamerasText(reconstruction, file);
	}

	/// <summary>ReadFramesText.</summary>
	public static void ReadFramesText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		foreach (CppLineTokens lineStream in DataLines(stream))
		{
			// ID, RIG_ID
			Check.That(
				lineStream.TryReadUInt32(out uint frameId) & lineStream.TryReadUInt32(out uint rigId),
				null,
				"line_stream >> frame_id >> rig_id");
			var frame = new Frame { FrameId = frameId };
			frame.SetRigId(rigId);

			// RIG_FROM_WORLD, DATA_IDS
			Check.That(
				TryReadRigid3d(lineStream, out Rigid3d rigFromWorld) & lineStream.TryReadUInt32(out uint numDataIds),
				null,
				"line_stream >> rig_from_world >> num_data_ids");
			frame.SetRigFromWorld(rigFromWorld);
			for (uint i = 0; i < numDataIds; ++i)
			{
				Check.That(
					lineStream.TryReadString(out string item) & lineStream.TryReadUInt32(out uint sensorId) &
					lineStream.TryReadUInt64(out ulong dataId),
					null,
					"line_stream >> item >> data_id.sensor_id.id >> data_id.id");
				var sensor = new SensorId(SensorTypeExtensions.SensorTypeFromString(item), sensorId);
				frame.AddDataId(new DataId { SensorId = sensor, Id = dataId });
			}

			reconstruction.AddFrame(frame);
		}
	}

	/// <summary>ReadFramesText from a file.</summary>
	public static void ReadFramesText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenRead(path);
		ReadFramesText(reconstruction, file);
	}

	/// <summary>
	/// ReadImagesText: two lines per image. Without any rigs and frames in the
	/// reconstruction, the file is a legacy one: every camera gets a trivial rig and every
	/// image a frame of its own.
	/// </summary>
	public static void ReadImagesText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");

		// Handle backwards-compatibility for when we didn't have rigs and frames.
		bool isLegacyReconstruction = reconstruction.NumRigs == 0 && reconstruction.NumFrames == 0;
		if (isLegacyReconstruction)
		{
			ReconstructionIOUtils.CreateOneRigPerCamera(reconstruction);
		}

		Dictionary<uint, Frame> imageToFrame = ReconstructionIOUtils.ExtractImageToFramePtr(reconstruction);

		List<(string Text, bool ValidUtf8)> lines = CppLineTokens.ReadLines(stream);
		var points2D = new List<Vector2d>();
		var point3DIds = new List<ulong>();

		for (int lineIdx = 0; lineIdx < lines.Count; ++lineIdx)
		{
			string line = CppLineTokens.Trim(lines[lineIdx].Text);
			if (line.Length == 0 || line[0] == '#')
			{
				continue;
			}

			var lineStream1 = new CppLineTokens(line);

			// ID, CAM_FROM_WORLD, CAMERA_ID
			Check.That(
				lineStream1.TryReadUInt32(out uint imageId) & TryReadRigid3d(lineStream1, out Rigid3d camFromWorld) &
				lineStream1.TryReadUInt32(out uint cameraId),
				null,
				"line_stream1 >> image_id >> cam_from_world >> camera_id");

			var image = new Image { ImageId = imageId };
			image.SetCameraId(cameraId);

			if (isLegacyReconstruction)
			{
				ReconstructionIOUtils.CreateFrameForImage(image, camFromWorld, reconstruction);
				image.SetFrameId(image.ImageId);
				image.SetFramePtr(reconstruction.Frame(image.ImageId));
			}
			else
			{
				Frame frame = ReconstructionIOUtils.FrameOfImage(imageToFrame, image.ImageId);
				image.SetFrameId(frame.FrameId);
				image.SetFramePtr(frame);
			}

			// std::string keeps any bytes; a C# string needs the name to be UTF-8.
			if (!lines[lineIdx].ValidUtf8)
			{
				throw ReconstructionIOUtils.NonUtf8ImageName(imageId);
			}

			// NAME
			Check.That(lineStream1.TryReadString(out string name), null, "line_stream1 >> image.Name()");
			image.Name = name;

			// POINTS2D
			if (++lineIdx >= lines.Count)
			{
				break;
			}

			line = CppLineTokens.Trim(lines[lineIdx].Text);
			var lineStream2 = new CppLineTokens(line);

			points2D.Clear();
			point3DIds.Clear();

			if (line.Length != 0)
			{
				while (lineStream2.TryReadDouble(out double x) && lineStream2.TryReadDouble(out double y))
				{
					points2D.Add(new Vector2d(x, y));

					Check.That(lineStream2.TryReadInt64(out long point3DIdSigned), null, "line_stream2 >> point3D_id_signed");
					point3DIds.Add(point3DIdSigned == -1 ? Types.InvalidPoint3DId : unchecked((ulong)point3DIdSigned));
				}
			}

			image.SetPoints2D(points2D);

			for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				if (point3DIds[(int)point2DIdx] != Types.InvalidPoint3DId)
				{
					image.SetPoint3DForPoint2D(point2DIdx, point3DIds[(int)point2DIdx]);
				}
			}

			reconstruction.AddImage(image);
		}
	}

	/// <summary>ReadImagesText from a file.</summary>
	public static void ReadImagesText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenRead(path);
		ReconstructionIOUtils.NameFileInErrors(path, () => ReadImagesText(reconstruction, file));
	}

	/// <summary>ReadPoints3DText. TRACK pairs are read until the line ends or a token is not a number.</summary>
	public static void ReadPoints3DText(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		foreach (CppLineTokens lineStream in DataLines(stream))
		{
			// ID, XYZ, RGB
			Check.That(
				lineStream.TryReadUInt64(out ulong point3DId) & lineStream.TryReadDouble(out double x) &
				lineStream.TryReadDouble(out double y) & lineStream.TryReadDouble(out double z) &
				lineStream.TryReadInt32(out int r) & lineStream.TryReadInt32(out int g) &
				lineStream.TryReadInt32(out int b),
				null,
				"line_stream >> point3D_id >> point3D.xyz >> r >> g >> b");

			// ERROR
			Check.That(lineStream.TryReadDouble(out double error), null, "line_stream >> point3D.error");

			// static_cast<uint8_t>(int) keeps the low byte.
			var point3D = new Point3D
			{
				Xyz = new Vector3d(x, y, z),
				Color = new Vector3ub(unchecked((byte)r), unchecked((byte)g), unchecked((byte)b)),
				Error = error,
			};

			// TRACK
			while (lineStream.TryReadUInt32(out uint imageId) && lineStream.TryReadUInt32(out uint point2DIdx))
			{
				point3D.Track.AddElement(imageId, point2DIdx);
			}

			point3D.Track.Compress();
			reconstruction.AddPoint3D(point3DId, point3D);
		}
	}

	/// <summary>ReadPoints3DText from a file.</summary>
	public static void ReadPoints3DText(Reconstruction reconstruction, string path)
	{
		using FileStream file = ReconstructionIOUtils.OpenRead(path);
		ReadPoints3DText(reconstruction, file);
	}

	// The trimmed, non-blank, non-comment lines of a text file, tokenized.
	private static IEnumerable<CppLineTokens> DataLines(Stream stream)
	{
		foreach (var (rawLine, _) in CppLineTokens.ReadLines(stream))
		{
			string line = CppLineTokens.Trim(rawLine);
			if (line.Length == 0 || line[0] == '#')
			{
				continue;
			}

			yield return new CppLineTokens(line);
		}
	}

	// QW QX QY QZ TX TY TZ, stored verbatim (not normalized).
	private static bool TryReadRigid3d(CppLineTokens lineStream, out Rigid3d transform)
	{
		transform = default;
		if (!(lineStream.TryReadDouble(out double qw) && lineStream.TryReadDouble(out double qx) &&
			lineStream.TryReadDouble(out double qy) && lineStream.TryReadDouble(out double qz) &&
			lineStream.TryReadDouble(out double tx) && lineStream.TryReadDouble(out double ty) &&
			lineStream.TryReadDouble(out double tz)))
		{
			return false;
		}

		transform = new Rigid3d(new Quaterniond(qw, qx, qy, qz), new Vector3d(tx, ty, tz));
		return true;
	}
}
