// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOBinary: port of colmap/scene/reconstruction_io_binary.h/.cc, COLMAP's
// rigs.bin, cameras.bin, frames.bin, images.bin and points3D.bin formats. Every value is
// little-endian (colmap/util/endian.h's WriteBinaryLittleEndian), quaternions are written
// w, x, y, z, and an image name is its UTF-8 bytes followed by a NUL. Each reader/writer has
// a Stream overload (the stream is left open) and a path overload.
// Neighbors: ReconstructionIOText.cs (the .txt formats), ReconstructionIOUtils.cs (shared
// helpers), Reconstruction.IO.cs (ReadBinary/WriteBinary on a directory). Tests:
// ColmapSharp.Tests/Scene/ReconstructionIOOracleTests.cs (Tier A: byte-identical to
// pycolmap's write_binary).

using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap/scene/reconstruction_io_binary.h.</summary>
public static class ReconstructionIOBinary
{
	private static readonly UTF8Encoding StrictUtf8 = new(false, true);

	/// <summary>ReadRigsBinary.</summary>
	public static void ReadRigsBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		ulong numRigs = reader.ReadUInt64();
		for (ulong i = 0; i < numRigs; ++i)
		{
			var rig = new Rig { RigId = reader.ReadUInt32() };
			uint numSensors = reader.ReadUInt32();

			if (numSensors > 0)
			{
				rig.AddRefSensor(ReadSensorId(reader));
			}

			if (numSensors > 1)
			{
				for (uint j = 0; j < numSensors - 1; ++j)
				{
					SensorId sensorId = ReadSensorId(reader);
					bool hasPose = reader.ReadByte() != 0;
					Rigid3d? sensorFromRig = hasPose ? ReadRigid3d(reader) : null;
					rig.AddSensor(sensorId, sensorFromRig);
				}
			}

			reconstruction.AddRig(rig);
		}
	}

	/// <summary>ReadRigsBinary from a file.</summary>
	public static void ReadRigsBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		ReadRigsBinary(reconstruction, file);
	}

	/// <summary>ReadCamerasBinary.</summary>
	public static void ReadCamerasBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		ulong numCameras = reader.ReadUInt64();
		for (ulong i = 0; i < numCameras; ++i)
		{
			var camera = new Camera
			{
				CameraId = reader.ReadUInt32(),
				ModelId = (CameraModelId)reader.ReadInt32(),
			};
			camera.Width = checked((int)reader.ReadUInt64());
			camera.Height = checked((int)reader.ReadUInt64());
			var cameraParams = new double[CameraModels.CameraModelNumParams(camera.ModelId)];
			for (int p = 0; p < cameraParams.Length; ++p)
			{
				cameraParams[p] = reader.ReadDouble();
			}

			camera.Params = cameraParams;
			Check.That(camera.VerifyParams());
			reconstruction.AddCamera(camera);
		}
	}

	/// <summary>ReadCamerasBinary from a file.</summary>
	public static void ReadCamerasBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		ReadCamerasBinary(reconstruction, file);
	}

	/// <summary>ReadFramesBinary.</summary>
	public static void ReadFramesBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		ulong numFrames = reader.ReadUInt64();
		for (ulong i = 0; i < numFrames; ++i)
		{
			var frame = new Frame { FrameId = reader.ReadUInt32() };
			frame.SetRigId(reader.ReadUInt32());
			frame.SetRigFromWorld(ReadRigid3d(reader));

			uint numDataIds = reader.ReadUInt32();
			for (uint j = 0; j < numDataIds; ++j)
			{
				SensorId sensorId = ReadSensorId(reader);
				frame.AddDataId(new DataId { SensorId = sensorId, Id = reader.ReadUInt64() });
			}

			reconstruction.AddFrame(frame);
		}
	}

	/// <summary>ReadFramesBinary from a file.</summary>
	public static void ReadFramesBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		ReadFramesBinary(reconstruction, file);
	}

	/// <summary>
	/// ReadImagesBinary. Without any rigs and frames in the reconstruction, the file is a
	/// legacy one: every camera gets a trivial rig and every image a frame of its own.
	/// </summary>
	public static void ReadImagesBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		// Handle backwards-compatibility for when we didn't have rigs and frames.
		bool isLegacyReconstruction = reconstruction.NumRigs == 0 && reconstruction.NumFrames == 0;
		if (isLegacyReconstruction)
		{
			ReconstructionIOUtils.CreateOneRigPerCamera(reconstruction);
		}

		Dictionary<uint, Frame> imageToFrame = ReconstructionIOUtils.ExtractImageToFramePtr(reconstruction);

		var points2D = new List<Vector2d>();
		var point3DIds = new List<ulong>();
		var nameBytes = new List<byte>();

		ulong numRegImages = reader.ReadUInt64();
		for (ulong i = 0; i < numRegImages; ++i)
		{
			var image = new Image { ImageId = reader.ReadUInt32() };
			Rigid3d camFromWorld = ReadRigid3d(reader);
			image.SetCameraId(reader.ReadUInt32());

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

			nameBytes.Clear();
			for (byte c = reader.ReadByte(); c != 0; c = reader.ReadByte())
			{
				nameBytes.Add(c);
			}

			// std::string keeps any bytes; a C# string needs the name to be UTF-8.
			try
			{
				image.Name = StrictUtf8.GetString(nameBytes.ToArray());
			}
			catch (DecoderFallbackException)
			{
				throw ReconstructionIOUtils.NonUtf8ImageName(image.ImageId);
			}

			ulong numPoints2D = reader.ReadUInt64();
			points2D.Clear();
			point3DIds.Clear();
			for (ulong j = 0; j < numPoints2D; ++j)
			{
				double x = reader.ReadDouble();
				double y = reader.ReadDouble();
				points2D.Add(new Vector2d(x, y));
				point3DIds.Add(reader.ReadUInt64());
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

	/// <summary>ReadImagesBinary from a file.</summary>
	public static void ReadImagesBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		ReconstructionIOUtils.NameFileInErrors(path, () => ReadImagesBinary(reconstruction, file));
	}

	/// <summary>ReadPoints3DBinary.</summary>
	public static void ReadPoints3DBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanRead, null, "stream.good()");
		using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

		ulong numPoints3D = reader.ReadUInt64();
		for (ulong i = 0; i < numPoints3D; ++i)
		{
			ulong point3DId = reader.ReadUInt64();
			var point3D = new Point3D
			{
				Xyz = new Vector3d(reader.ReadDouble(), reader.ReadDouble(), reader.ReadDouble()),
				Color = new Vector3ub(reader.ReadByte(), reader.ReadByte(), reader.ReadByte()),
				Error = reader.ReadDouble(),
			};

			ulong trackLength = reader.ReadUInt64();
			for (ulong j = 0; j < trackLength; ++j)
			{
				uint imageId = reader.ReadUInt32();
				uint point2DIdx = reader.ReadUInt32();
				point3D.Track.AddElement(imageId, point2DIdx);
			}

			point3D.Track.Compress();
			reconstruction.AddPoint3D(point3DId, point3D);
		}
	}

	/// <summary>ReadPoints3DBinary from a file.</summary>
	public static void ReadPoints3DBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		ReadPoints3DBinary(reconstruction, file);
	}

	/// <summary>WriteRigsBinary: every rig, in ascending id order.</summary>
	public static void WriteRigsBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

		writer.Write((ulong)reconstruction.NumRigs);
		foreach (uint rigId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Rigs))
		{
			Rig rig = reconstruction.Rig(rigId);
			writer.Write(rigId);
			writer.Write((uint)rig.NumSensors);
			if (rig.NumSensors > 0)
			{
				WriteSensorId(writer, rig.RefSensorId);
			}

			foreach (var (sensorId, sensorFromRig) in rig.NonRefSensors)
			{
				WriteSensorId(writer, sensorId);
				writer.Write((byte)(sensorFromRig.HasValue ? 1 : 0));
				if (sensorFromRig.HasValue)
				{
					WriteRigid3d(writer, sensorFromRig.Value);
				}
			}
		}
	}

	/// <summary>WriteRigsBinary to a file (created or truncated).</summary>
	public static void WriteRigsBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteRigsBinary(reconstruction, file);
	}

	/// <summary>WriteCamerasBinary: every camera, in ascending id order.</summary>
	public static void WriteCamerasBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

		writer.Write((ulong)reconstruction.NumCameras);
		foreach (uint cameraId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Cameras))
		{
			Camera camera = reconstruction.Camera(cameraId);
			writer.Write(cameraId);
			writer.Write((int)camera.ModelId);
			writer.Write((ulong)camera.Width);
			writer.Write((ulong)camera.Height);
			foreach (double param in camera.Params)
			{
				writer.Write(param);
			}
		}
	}

	/// <summary>WriteCamerasBinary to a file (created or truncated).</summary>
	public static void WriteCamerasBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteCamerasBinary(reconstruction, file);
	}

	/// <summary>WriteFramesBinary: the frames with a pose, in ascending id order.</summary>
	public static void WriteFramesBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

		List<uint> frameIds = ReconstructionIOUtils.ExtractSortedIds(reconstruction.Frames, frame => frame.HasPose);

		writer.Write((ulong)frameIds.Count);
		foreach (uint frameId in frameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			writer.Write(frameId);
			writer.Write(frame.RigId);
			WriteRigid3d(writer, frame.RigFromWorld());

			writer.Write((uint)frame.NumDataIds);
			foreach (DataId dataId in frame.DataIds)
			{
				WriteSensorId(writer, dataId.SensorId);
				writer.Write(dataId.Id);
			}
		}
	}

	/// <summary>WriteFramesBinary to a file (created or truncated).</summary>
	public static void WriteFramesBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteFramesBinary(reconstruction, file);
	}

	/// <summary>WriteImagesBinary: the registered images, in registration order.</summary>
	public static void WriteImagesBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

		writer.Write((ulong)reconstruction.NumRegImages);
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			writer.Write(imageId);
			WriteRigid3d(writer, image.CamFromWorld());
			writer.Write(image.CameraId);

			writer.Write(Encoding.UTF8.GetBytes(image.Name));
			writer.Write((byte)0);

			writer.Write((ulong)image.NumPoints2D);
			foreach (Point2D point2D in image.Points2D)
			{
				writer.Write(point2D.Xy.X);
				writer.Write(point2D.Xy.Y);
				writer.Write(point2D.Point3DId);
			}
		}
	}

	/// <summary>WriteImagesBinary to a file (created or truncated).</summary>
	public static void WriteImagesBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteImagesBinary(reconstruction, file);
	}

	/// <summary>WritePoints3DBinary: every 3D point, in ascending id order.</summary>
	public static void WritePoints3DBinary(Reconstruction reconstruction, Stream stream)
	{
		Check.That(stream.CanWrite, null, "stream.good()");
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

		writer.Write((ulong)reconstruction.NumPoints3D);
		foreach (ulong point3DId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Points3D))
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			writer.Write(point3DId);
			writer.Write(point3D.Xyz.X);
			writer.Write(point3D.Xyz.Y);
			writer.Write(point3D.Xyz.Z);
			writer.Write(point3D.Color.X);
			writer.Write(point3D.Color.Y);
			writer.Write(point3D.Color.Z);
			writer.Write(point3D.Error);

			writer.Write((ulong)point3D.Track.Length);
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				writer.Write(trackEl.ImageId);
				writer.Write(trackEl.Point2DIdx);
			}
		}
	}

	/// <summary>WritePoints3DBinary to a file (created or truncated).</summary>
	public static void WritePoints3DBinary(Reconstruction reconstruction, string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WritePoints3DBinary(reconstruction, file);
	}

	// A sensor_t as (int type, uint32 id).
	private static SensorId ReadSensorId(BinaryReader reader)
	{
		var type = (SensorType)reader.ReadInt32();
		return new SensorId(type, reader.ReadUInt32());
	}

	private static void WriteSensorId(BinaryWriter writer, SensorId sensorId)
	{
		writer.Write((int)sensorId.Type);
		writer.Write(sensorId.Id);
	}

	// A Rigid3d as qw, qx, qy, qz, tx, ty, tz, stored verbatim (not normalized).
	private static Rigid3d ReadRigid3d(BinaryReader reader)
	{
		double qw = reader.ReadDouble();
		double qx = reader.ReadDouble();
		double qy = reader.ReadDouble();
		double qz = reader.ReadDouble();
		double tx = reader.ReadDouble();
		double ty = reader.ReadDouble();
		double tz = reader.ReadDouble();
		return new Rigid3d(new Quaterniond(qw, qx, qy, qz), new Vector3d(tx, ty, tz));
	}

	private static void WriteRigid3d(BinaryWriter writer, Rigid3d transform)
	{
		writer.Write(transform.Rotation.W);
		writer.Write(transform.Rotation.X);
		writer.Write(transform.Rotation.Y);
		writer.Write(transform.Rotation.Z);
		writer.Write(transform.Translation.X);
		writer.Write(transform.Translation.Y);
		writer.Write(transform.Translation.Z);
	}
}
