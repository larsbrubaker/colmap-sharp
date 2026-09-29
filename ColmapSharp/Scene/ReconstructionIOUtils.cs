// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOUtils: port of colmap/scene/reconstruction_io_utils.h/.cc, the helpers the
// binary (ReconstructionIOBinary.cs) and text (ReconstructionIOText.cs) readers and writers
// share: sorted id extraction for deterministic output, and the rig/frame synthesis that
// lets a model written before COLMAP had rigs and frames (only cameras, images and
// points3D) load as one trivial rig per camera and one frame per image. Also the
// directory check (THROW_CHECK_DIR_EXISTS) the path overloads use; opening files is
// Util/FileOpen.cs. Tests: ColmapSharp.Tests/Scene/ReconstructionIOOracleTests.cs.

using ColmapSharp.Geometry;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap/scene/reconstruction_io_utils.h.</summary>
public static class ReconstructionIOUtils
{
	/// <summary>
	/// ExtractSortedIds: the keys of <paramref name="data"/> whose value passes
	/// <paramref name="filter"/> (all keys without one), in ascending order.
	/// </summary>
	public static List<TKey> ExtractSortedIds<TKey, TValue>(
		IReadOnlyDictionary<TKey, TValue> data,
		Func<TValue, bool>? filter = null)
		where TKey : IComparable<TKey>
	{
		var ids = new List<TKey>(data.Count);
		foreach (var (id, value) in data)
		{
			if (filter is null || filter(value))
			{
				ids.Add(id);
			}
		}

		// Ids are unique, so the unstable sort cannot reorder ties.
		ids.Sort();
		return ids;
	}

	/// <summary>
	/// CreateOneRigPerCamera: adds, for every camera, a rig with the camera's id whose only
	/// (reference) sensor is that camera.
	/// </summary>
	public static void CreateOneRigPerCamera(Reconstruction reconstruction)
	{
		foreach (var (cameraId, camera) in reconstruction.Cameras)
		{
			var rig = new Rig { RigId = cameraId };
			rig.AddRefSensor(camera.SensorId);
			reconstruction.AddRig(rig);
		}
	}

	/// <summary>
	/// CreateFrameForImage: adds a frame with the image's id, on the rig with the image's
	/// camera id (see <see cref="CreateOneRigPerCamera"/>), holding only that image, posed
	/// at <paramref name="camFromWorld"/>.
	/// </summary>
	public static void CreateFrameForImage(Image image, Rigid3d camFromWorld, Reconstruction reconstruction)
	{
		var frame = new Frame { FrameId = image.ImageId };
		frame.SetRigId(image.CameraId);
		frame.AddDataId(image.DataId);
		frame.SetRigFromWorld(camFromWorld);
		reconstruction.AddFrame(frame);
	}

	/// <summary>
	/// ExtractImageToFramePtr: for every image id referenced by a frame, the reconstruction's
	/// frame that holds it. Throws if two frames claim the same image.
	/// </summary>
	public static Dictionary<uint, Frame> ExtractImageToFramePtr(Reconstruction reconstruction)
	{
		var imageToFrame = new Dictionary<uint, Frame>();
		foreach (var (frameId, frame) in reconstruction.Frames)
		{
			foreach (DataId dataId in frame.ImageIds())
			{
				Check.That(
					imageToFrame.TryAdd((uint)dataId.Id, reconstruction.Frame(frameId)),
					null,
					"image_to_frame.emplace(data_id.id, &reconstruction.Frame(frame_id)).second");
			}
		}

		return imageToFrame;
	}

	/// <summary>
	/// <c>image_to_frame.at(image_id)</c> as a check with a message: the frame holding the
	/// image, or a failure naming the image when no frame lists it.
	/// </summary>
	internal static Frame FrameOfImage(Dictionary<uint, Frame> imageToFrame, uint imageId)
	{
		Check.That(
			imageToFrame.TryGetValue(imageId, out Frame? frame),
			$"Image {imageId} is not in any frame of the reconstruction. Its frames file must list it as a data id of a frame.",
			"image_to_frame.count(image.ImageId())");
		return frame;
	}

	/// <summary>
	/// The error for an image name that is not valid UTF-8. COLMAP keeps names as raw bytes;
	/// a C# string cannot hold them faithfully (divergence 25).
	/// </summary>
	internal static InvalidDataException NonUtf8ImageName(uint imageId) =>
		new($"Image {imageId} has a name that is not valid UTF-8. Rename the image file and re-save the model with the name in UTF-8, then load it again.");

	/// <summary>
	/// Runs a reader on a file; an <see cref="InvalidDataException"/> from it is rethrown with
	/// the file's path in front, so the user knows which file to fix.
	/// </summary>
	internal static void NameFileInErrors(string path, Action read)
	{
		try
		{
			read();
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidDataException($"{path}: {ex.Message}", ex);
		}
	}

	/// <summary>THROW_CHECK_DIR_EXISTS(path).</summary>
	internal static void CheckDirExists(string path)
	{
		Check.That(Directory.Exists(path), $"Directory \"{path}\" does not exist.", "colmap::ExistsDir(path_val)");
	}
}
