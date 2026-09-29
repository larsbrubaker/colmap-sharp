// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RigConfig.ApplyRigConfig: port of the ApplyRigConfig half of colmap/scene/rig.cc (with its
// helpers UpdateRigAndCameraCalibsFromReconstruction, UpdateRigsAndFramesFromDatabase and
// CopyCameraIntrinsics). It replaces the rigs and frames of a Database (and, if given, of a
// Reconstruction) with those of the configs; RigConfig.cs holds the config type and its
// JSON reader. Tests: ColmapSharp.Tests/Scene/RigConfigTests.cs (rig_test.cc 1:1).
//
// Ordering (CLAUDE.md, hash container iteration):
// - COLMAP groups images per frame name in a std::map<std::string>, so frames are written in
//   the names' UTF-8 byte order; Utf8OrderComparer gives the same order.
// - COLMAP hands the reconstruction its rigs and frames in NodeHashMap order (an
//   std::unordered_map or boost::unordered_node_map, util/hash_containers.h), which sets the
//   registration order of the frames and is implementation-defined per build and standard
//   library. Here they go in ascending id order (divergence 81).
// - COLMAP's LOG(WARNING) goes to Util/Log.cs; its LOG(INFO) lines are dropped.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class RigConfig
{
	/// <summary>
	/// Applies the given rig configuration to the database and optionally derives camera rig
	/// extrinsics and intrinsics from the reconstruction, if not defined in the config. If the
	/// reconstruction is provided, it is also updated with the provided config and any
	/// previous rigs/frames are cleared and overwritten. Existing rigs and frames in the
	/// database and reconstruction will be cleared. Any unspecified images in the provided
	/// configurations will be configured as trivial rigs/frames. Port of colmap::ApplyRigConfig.
	/// </summary>
	public static void ApplyRigConfig(IReadOnlyList<RigConfig> configs, Database database, Reconstruction? reconstruction = null)
	{
		database.ClearFrames();
		database.ClearRigs();

		List<Image> images = database.ReadAllImages();
		var configuredImageIds = new HashSet<uint>();

		foreach (RigConfig config in configs)
		{
			var rig = new Rig();

			int numCameras = config.Cameras.Count;

			var cameraIds = new uint?[numCameras];
			var frameNameToImages = new SortedDictionary<string, List<Image>>(Utf8OrderComparer.Instance);
			foreach (Image image in images)
			{
				for (int cameraIdx = 0; cameraIdx < numCameras; ++cameraIdx)
				{
					RigCamera configCamera = config.Cameras[cameraIdx];
					if (StringStartsWith(image.Name, configCamera.ImagePrefix))
					{
						string frameName = StringGetAfter(image.Name, configCamera.ImagePrefix);
						if (!frameNameToImages.TryGetValue(frameName, out List<Image>? frameImages))
						{
							frameImages = [];
							frameNameToImages.Add(frameName, frameImages);
						}

						frameImages.Add(image);
						if (cameraIds[cameraIdx] is uint cameraId)
						{
							Check.Eq(cameraId, image.CameraId,
								"Inconsistent cameras for images with prefix: " + configCamera.ImagePrefix
								+ ". Consider setting --ImageReader.single_camera_per_folder during feature "
								+ "extraction or manually assign consistent camera_id's.");
						}
						else
						{
							cameraIds[cameraIdx] = image.CameraId;
							if (configCamera.Camera is not null)
							{
								Camera databaseCamera = database.ReadCamera(image.CameraId);
								CopyCameraIntrinsics(configCamera.Camera, databaseCamera);
								database.UpdateCamera(databaseCamera);
								if (reconstruction is not null)
								{
									CopyCameraIntrinsics(configCamera.Camera, reconstruction.Camera(image.CameraId));
								}
							}
						}
					}
				}
			}

			var uniqueCameraIds = new HashSet<uint>();
			for (int cameraIdx = 0; cameraIdx < numCameras; ++cameraIdx)
			{
				RigCamera configCamera = config.Cameras[cameraIdx];
				Check.That(cameraIds[cameraIdx] is not null, "At least one image must exist for each camera in the rig");
				uint cameraId = cameraIds[cameraIdx]!.Value;
				if (!uniqueCameraIds.Add(cameraId))
				{
					// Clone the camera, if multiple cameras in the rig share a camera.
					cameraId = database.WriteCamera(database.ReadCamera(cameraId));
					cameraIds[cameraIdx] = cameraId;
				}

				if (configCamera.RefSensor)
				{
					rig.AddRefSensor(new SensorId(SensorType.Camera, cameraId));
				}
				else
				{
					rig.AddSensor(new SensorId(SensorType.Camera, cameraId), configCamera.CamFromRig);
				}
			}

			rig.RigId = database.WriteRig(rig);

			foreach (List<Image> frameImages in frameNameToImages.Values)
			{
				var frame = new Frame();
				frame.SetRigId(rig.RigId);
				foreach (Image image in frameImages)
				{
					DataId dataId = image.DataId;
					Check.That(rig.HasSensor(dataId.SensorId),
						$"{rig} must not contain Image(image_id={image.ImageId}, camera_id={image.CameraId}, name={image.Name})");
					frame.AddDataId(dataId);
					configuredImageIds.Add(image.ImageId);
				}

				frame.FrameId = database.WriteFrame(frame);
			}

			if (reconstruction is not null)
			{
				UpdateRigAndCameraCalibsFromReconstruction(reconstruction, frameNameToImages, rig, database);
			}
		}

		// Create trivial rigs/frames for images without configuration.
		// This is necessary because we clear rigs/frames above.
		var cameraToRigId = new Dictionary<uint, uint>();
		foreach (Image image in images)
		{
			if (configuredImageIds.Contains(image.ImageId))
			{
				continue;
			}

			var sensorId = new SensorId(SensorType.Camera, image.CameraId);
			if (!cameraToRigId.TryGetValue(image.CameraId, out uint rigId))
			{
				var rig = new Rig();
				rig.AddRefSensor(sensorId);
				rigId = database.WriteRig(rig);
				cameraToRigId.Add(image.CameraId, rigId);
			}

			var frame = new Frame();
			frame.SetRigId(rigId);
			frame.AddDataId(new DataId(sensorId, image.ImageId));
			frame.FrameId = database.WriteFrame(frame);
		}

		if (reconstruction is not null)
		{
			UpdateRigsAndFramesFromDatabase(database, reconstruction);
		}
	}

	// Update the database with extracted rig and calibrations from the given reconstruction
	// derived as follows:
	//   * Compute the sensor_from_rig poses as the average of the relative poses between
	//     registered sensors in the reconstruction.
	//   * Set the camera calibration parameters from the first frame with an image of the
	//     camera.
	private static void UpdateRigAndCameraCalibsFromReconstruction(
		Reconstruction reconstruction,
		SortedDictionary<string, List<Image>> frameNameToImages,
		Rig rig,
		Database database)
	{
		// Only looked up by camera, so the dictionary's order cannot leak.
		var rigFromCams = new Dictionary<uint, (List<Quaterniond> Rotations, Vector3d TranslationSum)>();
		var updatedCameras = new HashSet<uint>();
		foreach (List<Image> images in frameNameToImages.Values)
		{
			Image? refImage = null;
			foreach (Image image in images)
			{
				if (rig.IsRefSensor(image.DataId.SensorId))
				{
					refImage = image;
				}
			}

			if (refImage is null)
			{
				continue;
			}

			Image? rigCalibRefImage = reconstruction.FindImageWithName(refImage.Name);
			if (rigCalibRefImage is null || !rigCalibRefImage.HasPose)
			{
				continue;
			}

			Rigid3d refCamFromWorld = rigCalibRefImage.CamFromWorld();
			if (updatedCameras.Add(rigCalibRefImage.CameraId))
			{
				Camera refCamera = rigCalibRefImage.CameraPtr.Clone();
				refCamera.CameraId = refImage.CameraId;
				database.UpdateCamera(refCamera);
			}

			foreach (Image image in images)
			{
				if (image.CameraId != refImage.CameraId)
				{
					Image? rigCalibImage = reconstruction.FindImageWithName(image.Name);
					if (rigCalibImage is null || !rigCalibImage.HasPose)
					{
						continue;
					}

					Rigid3d rigFromCam = refCamFromWorld * rigCalibImage.CamFromWorld().Inverse();
					if (updatedCameras.Add(rigCalibImage.CameraId))
					{
						Camera camera = rigCalibImage.CameraPtr.Clone();
						camera.CameraId = image.CameraId;
						database.UpdateCamera(camera);
					}

					if (rigFromCams.TryGetValue(image.CameraId, out var entry))
					{
						entry.Rotations.Add(rigFromCam.Rotation);
						rigFromCams[image.CameraId] = (entry.Rotations, entry.TranslationSum + rigFromCam.Translation);
					}
					else
					{
						rigFromCams.Add(image.CameraId, ([rigFromCam.Rotation], rigFromCam.Translation));
					}
				}
			}
		}

		// Compute the average sensor_from_rig poses over all frames.
		foreach (SensorId sensorId in rig.NonRefSensors.Keys.ToList())
		{
			if (rig.NonRefSensors[sensorId].HasValue)
			{
				// Do not compute it for explicitly provided poses in the config.
				continue;
			}

			if (!rigFromCams.TryGetValue((uint)sensorId.Id, out var entry))
			{
				Log.Warning(
					$"Failed to derive sensor_from_rig transformation for camera {sensorId.Id}, because the image was not "
					+ "registered in the given reconstruction.");
				continue;
			}

			var rigFromCamRotations = entry.Rotations;
			var rigFromCam = new Rigid3d(
				Pose.AverageQuaternions(rigFromCamRotations, Enumerable.Repeat(1.0, rigFromCamRotations.Count).ToArray()),
				entry.TranslationSum / rigFromCamRotations.Count);
			rig.SetSensorFromRig(sensorId, rigFromCam.Inverse());
		}

		database.UpdateRig(rig);
	}

	private static void UpdateRigsAndFramesFromDatabase(Database database, Reconstruction reconstruction)
	{
		List<Frame> databaseFrames = database.ReadAllFrames();

		var databaseRigs = new Dictionary<uint, Rig>();
		foreach (Rig rig in database.ReadAllRigs())
		{
			databaseRigs.Add(rig.RigId, rig);
		}

		// Sorted by id: the order in which they reach SetRigsAndFrames (see the file header).
		var reconstructionRigs = new SortedDictionary<uint, Rig>();

		// Create O(1) lookup table from image names to images. The first image of a name wins,
		// as with NodeHashMap::emplace.
		var imageNameToImage = new Dictionary<string, Image>(StringComparer.Ordinal);
		foreach (Image image in reconstruction.Images.Values)
		{
			imageNameToImage.TryAdd(image.Name, image);
		}

		void VisitFrameData(Action<Frame, Image, Image> visitor)
		{
			foreach (Frame databaseFrame in databaseFrames)
			{
				foreach (DataId dataId in databaseFrame.ImageIds())
				{
					Image databaseImage = database.ReadImage((uint)dataId.Id);
					if (!imageNameToImage.TryGetValue(databaseImage.Name, out Image? reconstructionImage))
					{
						continue;
					}

					visitor(databaseFrame, databaseImage, reconstructionImage);
				}
			}
		}

		Rig ReconstructionRig(uint rigId)
		{
			if (!reconstructionRigs.TryGetValue(rigId, out Rig? rig))
			{
				rig = new Rig();
				reconstructionRigs.Add(rigId, rig);
			}

			rig.RigId = rigId;
			return rig;
		}

		// Update reference sensors in reconstruction rigs.
		// (must be done before updating the non-reference sensors).
		VisitFrameData((databaseFrame, databaseImage, reconstructionImage) =>
		{
			Rig databaseRig = databaseRigs[databaseFrame.RigId];
			Rig reconstructionRig = ReconstructionRig(databaseFrame.RigId);

			SensorId databaseSensorId = databaseImage.DataId.SensorId;
			SensorId reconstructionSensorId = reconstructionImage.CameraPtr.SensorId;

			if (!reconstructionRig.IsRefSensor(reconstructionSensorId) && databaseRig.IsRefSensor(databaseSensorId))
			{
				reconstructionRig.AddRefSensor(reconstructionSensorId);
			}
		});

		// Update non-reference sensors in reconstruction rigs.
		VisitFrameData((databaseFrame, databaseImage, reconstructionImage) =>
		{
			Rig databaseRig = databaseRigs[databaseFrame.RigId];
			Rig reconstructionRig = ReconstructionRig(databaseFrame.RigId);

			SensorId databaseSensorId = databaseImage.DataId.SensorId;
			SensorId reconstructionSensorId = reconstructionImage.CameraPtr.SensorId;

			if (!reconstructionRig.NonRefSensors.ContainsKey(reconstructionSensorId)
				&& databaseRig.NonRefSensors.ContainsKey(databaseSensorId))
			{
				reconstructionRig.AddSensor(reconstructionSensorId, databaseRig.SensorFromRig(databaseSensorId));
			}
		});

		// Update reconstruction frames.
		var reconstructionFrames = new SortedDictionary<uint, Frame>();
		VisitFrameData((databaseFrame, databaseImage, reconstructionImage) =>
		{
			Rig databaseRig = databaseRigs[databaseFrame.RigId];
			SensorId databaseSensorId = databaseImage.DataId.SensorId;
			if (!reconstructionFrames.TryGetValue(databaseFrame.FrameId, out Frame? reconstructionFrame))
			{
				reconstructionFrame = new Frame();
				reconstructionFrames.Add(databaseFrame.FrameId, reconstructionFrame);
			}

			reconstructionFrame.FrameId = databaseFrame.FrameId;
			reconstructionFrame.SetRigId(databaseFrame.RigId);
			reconstructionFrame.AddDataId(reconstructionImage.DataId);
			if (reconstructionImage.HasPose)
			{
				if (databaseRig.IsRefSensor(databaseSensorId))
				{
					reconstructionFrame.SetRigFromWorld(reconstructionImage.CamFromWorld());
				}
				else
				{
					reconstructionFrame.SetRigFromWorld(
						databaseRig.SensorFromRig(databaseSensorId).Inverse() * reconstructionImage.CamFromWorld());
				}
			}
		});

		reconstruction.SetRigsAndFrames(reconstructionRigs.Values, reconstructionFrames.Values);
	}

	private static void CopyCameraIntrinsics(Camera src, Camera dst)
	{
		dst.ModelId = src.ModelId;
		dst.Params = (double[])src.Params.Clone();
		dst.HasPriorFocalLength = src.HasPriorFocalLength;
	}

	// Port of colmap::StringStartsWith: an empty prefix matches nothing.
	private static bool StringStartsWith(string str, string prefix) =>
		prefix.Length != 0 && str.StartsWith(prefix, StringComparison.Ordinal);

	// Port of colmap::StringGetAfter: the text after the LAST occurrence of the key (rfind),
	// the whole string for an empty key, and "" if the key does not occur. ApplyRigConfig only
	// calls it with a key StringStartsWith matched, so the empty-key branch is unreachable
	// here; it is kept to mirror the C++.
	private static string StringGetAfter(string str, string key)
	{
		if (key.Length == 0)
		{
			return str;
		}

		int found = str.LastIndexOf(key, StringComparison.Ordinal);
		return found >= 0 ? str[(found + key.Length)..] : "";
	}

	/// <summary>
	/// Orders strings by their UTF-8 bytes, as std::map&lt;std::string&gt; does, without encoding
	/// them. UTF-8 byte order is code point order, and UTF-16 code unit order differs from it
	/// only where a surrogate (U+D800-U+DFFF, standing for a code point above U+FFFF) meets a
	/// unit in U+E000-U+FFFF; moving the surrogates above that range fixes it.
	/// </summary>
	private sealed class Utf8OrderComparer : IComparer<string>
	{
		public static readonly Utf8OrderComparer Instance = new();

		public int Compare(string? x, string? y)
		{
			if (x is null || y is null)
			{
				return x is null ? (y is null ? 0 : -1) : 1;
			}

			int length = Math.Min(x.Length, y.Length);
			for (int i = 0; i < length; i++)
			{
				if (x[i] != y[i])
				{
					return FixUp(x[i]).CompareTo(FixUp(y[i]));
				}
			}

			return x.Length.CompareTo(y.Length);
		}

		private static int FixUp(char c) => c >= 0xE000 ? c - 0x800 : c >= 0xD800 ? c + 0x2000 : c;
	}
}
