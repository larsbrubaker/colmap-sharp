// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction: port of colmap/scene/reconstruction.h and reconstruction.cc, the model
// the SfM pipeline builds: rigs (Sensor/Rig.cs), cameras (Camera.cs), frames (Frame.cs),
// images (Image.cs) and 3D points (Point3D.cs), with the bookkeeping that keeps the 2D-3D
// links on both sides consistent. This file holds the storage, the accessors, copying and
// the add/delete/register operations (the first half of reconstruction.cc);
// Reconstruction.Queries.cs holds validation, tear-down, normalization, transforms,
// cropping, lookups and statistics (the second half); Reconstruction.IO.cs holds
// Read/Write/ReadText/ReadBinary/WriteText/WriteBinary. Tests:
// ColmapSharp.Tests/Scene/ReconstructionTests*.cs (reconstruction_test.cc).
//
// Load and TranscribeImageIdsToDatabase are in Reconstruction.Database.cs, ConvertToPLY and
// ImportPLY in Reconstruction.Ply.cs.
// Not ported yet: ExtractColorsForImage/ExtractColorsForAllImages (they read image files
// through Bitmap::Read, which is not ported: the host decodes images).
//
// Design (later phases build on it):
// - Ownership. The reconstruction owns every rig, camera, frame, image and 3D point.
//   Frame.RigPtr, Image.CameraPtr and Image.FramePtr are non-owning references into it,
//   set when an object is added and rewired on copy. Accessors (Camera(id), Frame(id), ...)
//   return the owned object itself, so edits through them are C++'s edits through a
//   reference. Add* take their argument by value in C++, so they store a Clone() and the
//   caller's object stays independent; a later edit to it does not reach the model.
// - Bundle adjustment writes into Frame.RigFromWorldStorage and Camera.Params in place;
//   nothing here replaces those arrays (Transform and Normalize go through
//   Frame.SetRigFromWorld, which writes into the same storage).
// - Iteration order. COLMAP keeps its objects in hash maps; here they are in
//   Util/IdMap.cs, which iterates in ascending id order (docs/CPP_DIVERGENCES.md, entry
//   21). RegFrameIds is COLMAP's vector, in registration order, exactly as in C++.
// - C++ copy construction/assignment is Clone(). As in C++, the copied frames are not
//   finalized (Frame's copy constructor resets the flag).
// - LOG(WARNING) messages are dropped (PORTING_PLAN.md, Phase 4).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Reconstruction: holds all information about a single reconstructed
/// model. It is used by the mapping and bundle adjustment classes.
/// </summary>
public sealed partial class Reconstruction
{
	private readonly IdMap<uint, Rig> _rigs = new();
	private readonly IdMap<uint, Camera> _cameras = new();
	private readonly IdMap<uint, Frame> _frames = new();
	private readonly IdMap<uint, Image> _images = new();
	private readonly IdMap<ulong, Point3D> _points3D = new();

	// Unique set of frame_ids where Frame(frame_id).HasPose. Note that we intentionally use
	// a list instead of a set here leading to O(n) complexity on calls to
	// RegisterFrame/DeRegisterFrame, because we iterate very often over the set of
	// registered frames.
	private readonly List<uint> _regFrameIds = [];
	private int _numRegImages;

	// Total number of added 3D points, used to generate unique identifiers.
	private ulong _maxPoint3DId;

	/// <summary>Number of rigs.</summary>
	public int NumRigs => _rigs.Count;

	/// <summary>Number of cameras.</summary>
	public int NumCameras => _cameras.Count;

	/// <summary>Number of frames.</summary>
	public int NumFrames => _frames.Count;

	/// <summary>Number of registered frames.</summary>
	public int NumRegFrames => _regFrameIds.Count;

	/// <summary>Number of registered images (the images of the registered frames).</summary>
	public int NumRegImages => _numRegImages;

	/// <summary>Number of images.</summary>
	public int NumImages => _images.Count;

	/// <summary>Number of 3D points.</summary>
	public int NumPoints3D => _points3D.Count;

	/// <summary>All rigs, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Rig> Rigs => _rigs;

	/// <summary>All cameras, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Camera> Cameras => _cameras;

	/// <summary>All frames, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Frame> Frames => _frames;

	/// <summary>The registered frame ids, in registration order.</summary>
	public IReadOnlyList<uint> RegFrameIds => _regFrameIds;

	/// <summary>All images, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Image> Images => _images;

	/// <summary>All 3D points, in ascending id order.</summary>
	public IReadOnlyDictionary<ulong, Point3D> Points3D => _points3D;

	/// <summary>The rig with the given id; throws if it does not exist.</summary>
	public Rig Rig(uint rigId) =>
		_rigs.TryGetValue(rigId, out Rig? rig) ? rig : throw new KeyNotFoundException($"Rig with ID {rigId} does not exist");

	/// <summary>The camera with the given id; throws if it does not exist.</summary>
	public Camera Camera(uint cameraId) =>
		_cameras.TryGetValue(cameraId, out Camera? camera) ? camera : throw new KeyNotFoundException($"Camera with ID {cameraId} does not exist");

	/// <summary>The frame with the given id; throws if it does not exist.</summary>
	public Frame Frame(uint frameId) =>
		_frames.TryGetValue(frameId, out Frame? frame) ? frame : throw new KeyNotFoundException($"Frame with ID {frameId} does not exist");

	/// <summary>The image with the given id; throws if it does not exist.</summary>
	public Image Image(uint imageId) =>
		_images.TryGetValue(imageId, out Image? image) ? image : throw new KeyNotFoundException($"Image with ID {imageId} does not exist");

	/// <summary>The 3D point with the given id; throws if it does not exist.</summary>
	public Point3D Point3D(ulong point3DId) =>
		_points3D.TryGetValue(point3DId, out Point3D? point3D) ? point3D : throw new KeyNotFoundException($"Point3D with ID {point3DId} does not exist");

	/// <summary>Whether the rig exists.</summary>
	public bool ExistsRig(uint rigId) => _rigs.ContainsKey(rigId);

	/// <summary>Whether the camera exists.</summary>
	public bool ExistsCamera(uint cameraId) => _cameras.ContainsKey(cameraId);

	/// <summary>Whether the frame exists.</summary>
	public bool ExistsFrame(uint frameId) => _frames.ContainsKey(frameId);

	/// <summary>Whether the image exists.</summary>
	public bool ExistsImage(uint imageId) => _images.ContainsKey(imageId);

	/// <summary>Whether the 3D point exists.</summary>
	public bool ExistsPoint3D(ulong point3DId) => _points3D.ContainsKey(point3DId);

	/// <summary>
	/// A deep copy (C++ copy construction and assignment) whose frames and images point at
	/// the copy's own rigs, cameras and frames.
	/// </summary>
	public Reconstruction Clone()
	{
		var copy = new Reconstruction
		{
			_numRegImages = _numRegImages,
			_maxPoint3DId = _maxPoint3DId,
		};
		foreach (var (rigId, rig) in _rigs)
		{
			copy._rigs.TryAdd(rigId, rig.Clone());
		}

		foreach (var (cameraId, camera) in _cameras)
		{
			copy._cameras.TryAdd(cameraId, camera.Clone());
		}

		foreach (var (frameId, frame) in _frames)
		{
			Frame frameCopy = frame.Clone();
			frameCopy.ResetRigPtr();
			frameCopy.SetRigPtr(copy.Rig(frameCopy.RigId));
			copy._frames.TryAdd(frameId, frameCopy);
		}

		foreach (var (imageId, image) in _images)
		{
			Image imageCopy = image.Clone();
			imageCopy.ResetCameraPtr();
			imageCopy.SetCameraPtr(copy.Camera(imageCopy.CameraId));
			imageCopy.ResetFramePtr();
			imageCopy.SetFramePtr(copy.Frame(imageCopy.FrameId));
			copy._images.TryAdd(imageId, imageCopy);
		}

		foreach (var (point3DId, point3D) in _points3D)
		{
			copy._points3D.TryAdd(point3DId, point3D.Clone());
		}

		copy._regFrameIds.AddRange(_regFrameIds);
		return copy;
	}

	/// <summary>
	/// Identifiers of all registered images, frame by frame in registration order. Throws
	/// if a registered frame references an image that was not added.
	/// </summary>
	public List<uint> RegImageIds()
	{
		var regImageIds = new List<uint>();
		foreach (uint frameId in _regFrameIds)
		{
			Frame frame = Frame(frameId);
			foreach (DataId dataId in frame.ImageIds())
			{
				Check.That(
					ExistsImage((uint)dataId.Id),
					$"The reconstruction object is broken as image {dataId.Id} in frame {frame.FrameId} does not exist in the reconstruction. The most likely cause is missing AddImage(*) calls after adding frames.");
				regImageIds.Add((uint)dataId.Id);
			}
		}

		return regImageIds;
	}

	/// <summary>Identifiers of all 3D points.</summary>
	public HashSet<ulong> Point3DIds() => [.. _points3D.Keys];

	/// <summary>
	/// Adds a new rig calibration. Its camera sensors must have been added before.
	/// </summary>
	public void AddRig(Rig rig)
	{
		void CheckExistsSensor(SensorId sensorId)
		{
			if (sensorId.Type == SensorType.Camera)
			{
				Check.That(
					ExistsCamera(sensorId.Id),
					$"Camera {sensorId.Id} from rig {rig.RigId} not found in the reconstruction. Note that AddCamera should be called before AddRig.");
			}
		}

		CheckExistsSensor(rig.RefSensorId);
		foreach (SensorId sensorId in rig.NonRefSensors.Keys)
		{
			CheckExistsSensor(sensorId);
		}

		Check.That(_rigs.TryAdd(rig.RigId, rig.Clone()));
	}

	/// <summary>
	/// Adds a new camera. There is only one camera per image, while multiple images might be
	/// taken by the same camera.
	/// </summary>
	public void AddCamera(Camera camera)
	{
		Check.That(camera.VerifyParams());
		Check.That(_cameras.TryAdd(camera.CameraId, camera.Clone()));
	}

	/// <summary>
	/// Adds a new camera and also a rig with the same id (rig_id = camera_id), with the
	/// camera being its only sensor.
	/// </summary>
	public void AddCameraWithTrivialRig(Camera camera)
	{
		Check.That(
			!ExistsRig(camera.CameraId),
			$"AddCameraWithTrivialRig tried to add a rig with the same id as the camera, but failed because Rig {camera.CameraId}already exists in the reconstruction. ");
		var rig = new Rig { RigId = camera.CameraId };
		rig.AddRefSensor(camera.SensorId);
		AddCamera(camera);
		AddRig(rig);
	}

	/// <summary>
	/// Adds a new frame. Its rig must have been added before. If its rig reference is unset,
	/// it is set to the added rig. A frame with a pose is registered.
	/// </summary>
	public void AddFrame(Frame frame)
	{
		Check.That(frame.HasRigId);
		Rig rig = Rig(frame.RigId);
		foreach (DataId dataId in frame.DataIds)
		{
			switch (dataId.SensorId.Type)
			{
				case SensorType.Camera:
					// COLMAP's expression text, which callers match on.
					Check.That(rig.HasSensor(dataId.SensorId), null, "rig.HasSensor(data_id.sensor_id)");
					break;
				case SensorType.Imu:
					// Note that we do not (yet) support IMU measurement data.
					break;
				default:
					throw new ArgumentException($"Invalid sensor type: {dataId.SensorId.Type.ToColmapString()}");
			}
		}

		if (frame.HasRigPtr)
		{
			Check.That(ReferenceEquals(frame.RigPtr, rig), null, "frame.RigPtr() == &rig");
		}

		Frame stored = frame.Clone();
		if (!stored.HasRigPtr)
		{
			stored.SetRigPtr(rig);
		}

		bool isRegistered = stored.HasPose;
		uint frameId = stored.FrameId;
		Check.That(_frames.TryAdd(frameId, stored), null, "inserted");
		// We finalize the data ids, otherwise some internal bookkeeping (e.g., counting
		// reg_image_ids_) will be incorrect.
		stored.FinalizeDataIds();
		if (isRegistered)
		{
			Check.Ne(frameId, InvalidFrameId);
			RegisterFrame(frameId);
		}
	}

	/// <summary>
	/// Adds a new image. Its camera and frame must have been added before. If its camera or
	/// frame references are unset, they are set to the added ones.
	/// </summary>
	public void AddImage(Image image)
	{
		Check.That(image.HasCameraId);
		Camera camera = Camera(image.CameraId);
		if (image.HasCameraPtr)
		{
			Check.That(ReferenceEquals(image.CameraPtr, camera), null, "image.CameraPtr() == &camera");
		}

		Check.That(image.HasFrameId);
		Frame frame = Frame(image.FrameId);
		Check.That(frame.HasDataId(image.DataId), null, "frame.HasDataId(image.DataId())");
		if (image.HasFramePtr)
		{
			Check.That(ReferenceEquals(image.FramePtr, frame), null, "image.FramePtr() == &frame");
		}

		Image stored = image.Clone();
		if (!stored.HasCameraPtr)
		{
			stored.SetCameraPtr(camera);
		}

		if (!stored.HasFramePtr)
		{
			stored.SetFramePtr(frame);
		}

		Check.That(_images.TryAdd(stored.ImageId, stored));
	}

	/// <summary>
	/// Adds a new image and also a frame with the same id (frame_id = image_id), with the
	/// image being its only data. The frame's rig is the one with the image's camera id
	/// (rig_id = camera_id), which must hold exactly that camera.
	/// </summary>
	public void AddImageWithTrivialFrame(Image image)
	{
		Check.That(
			!ExistsFrame(image.ImageId),
			$"AddImageWithTrivialFrame tried to add a frame with the same id as the image, but failed because Frame {image.ImageId}already exists in the reconstruction.");
		Check.That(
			ExistsRig(image.CameraId),
			$"Rig {image.CameraId} that contains Camera {image.CameraId} does not exist in the reconstruction.");
		Rig rig = Rig(image.CameraId);
		Check.Eq(rig.NumSensors, 1, "AddImageWithTrivialFrame requires that the camera is from a rig that contains exactly one sensor (the camera itself).");
		Check.That(rig.IsRefSensor(Camera(image.CameraId).SensorId));
		var frame = new Frame { FrameId = image.ImageId };
		frame.SetRigId(image.CameraId);
		frame.AddDataId(image.DataId);
		Image stored = image.Clone();
		if (stored.HasFrameId)
		{
			Check.Eq(stored.FrameId, frame.FrameId);
		}
		else
		{
			stored.SetFrameId(frame.FrameId);
		}

		AddFrame(frame);
		AddImage(stored);
	}

	/// <summary>
	/// <see cref="AddImageWithTrivialFrame(Scene.Image)"/>, then registers the frame with the
	/// given pose.
	/// </summary>
	public void AddImageWithTrivialFrame(Image image, Rigid3d camFromWorld)
	{
		uint frameId = image.ImageId;
		AddImageWithTrivialFrame(image);
		Frame(frameId).SetRigFromWorld(camFromWorld);
		RegisterFrame(frameId);
	}

	/// <summary>
	/// Adds a new 3D point with a known id, linking its track's observations to it.
	/// </summary>
	public void AddPoint3D(ulong point3DId, Point3D point3D) => AddOwnedPoint3D(point3DId, point3D.Clone());

	/// <summary>Adds a new 3D point (with a copy of the track) and returns its unique id.</summary>
	public ulong AddPoint3D(Vector3d xyz, Track track, Vector3ub color = default) =>
		AddOwnedPoint3D(xyz, track.Clone(), color);

	// AddPoint3D for a point (or track) nobody else holds, so no copy is needed.
	private ulong AddOwnedPoint3D(Vector3d xyz, Track track, Vector3ub color)
	{
		ulong point3DId = ++_maxPoint3DId;
		AddOwnedPoint3D(point3DId, new Point3D { Xyz = xyz, Track = track, Color = color });
		return point3DId;
	}

	private void AddOwnedPoint3D(ulong point3DId, Point3D stored)
	{
		_maxPoint3DId = Math.Max(_maxPoint3DId, point3DId);

		foreach (TrackElement trackEl in stored.Track.Elements)
		{
			Image image = Image(trackEl.ImageId);
			Point2D point2D = image.Point2DAt(trackEl.Point2DIdx);
			if (point2D.HasPoint3D)
			{
				Check.Eq(point2D.Point3DId, point3DId);
			}
			else
			{
				image.SetPoint3DForPoint2D(trackEl.Point2DIdx, point3DId);
			}

			Check.Le(image.NumPoints3D, image.NumPoints2D);
		}

		Check.That(_points3D.TryAdd(point3DId, stored));
	}

	/// <summary>Adds an observation to an existing 3D point.</summary>
	public void AddObservation(ulong point3DId, TrackElement trackEl)
	{
		Image image = Image(trackEl.ImageId);
		Check.That(!image.Point2DAt(trackEl.Point2DIdx).HasPoint3D);

		image.SetPoint3DForPoint2D(trackEl.Point2DIdx, point3DId);
		Check.Le(image.NumPoints3D, image.NumPoints2D);

		Point3D(point3DId).Track.AddElement(trackEl);
	}

	/// <summary>
	/// Merges two 3D points and returns the id of the new 3D point. The location (and color)
	/// of the merged point is the average of the two, weighted by their track lengths.
	/// </summary>
	public ulong MergePoints3D(ulong point3DId1, ulong point3DId2)
	{
		Point3D point3D1 = Point3D(point3DId1);
		Point3D point3D2 = Point3D(point3DId2);
		double length1 = point3D1.Track.Length;
		double length2 = point3D2.Track.Length;

		Vector3d mergedXyz = ((length1 * point3D1.Xyz) + (length2 * point3D2.Xyz)) / (length1 + length2);
		Vector3d mergedRgb = ((length1 * ToDouble(point3D1.Color)) + (length2 * ToDouble(point3D2.Color))) / (length1 + length2);

		var mergedTrack = new Track();
		mergedTrack.Reserve(point3D1.Track.Length + point3D2.Track.Length);
		mergedTrack.AddElements(point3D1.Track.Elements);
		mergedTrack.AddElements(point3D2.Track.Elements);

		DeletePoint3D(point3DId1);
		DeletePoint3D(point3DId2);

		// Eigen's cast<uint8_t>() truncates toward zero, like a C cast.
		var mergedColor = new Vector3ub((byte)mergedRgb.X, (byte)mergedRgb.Y, (byte)mergedRgb.Z);
		return AddOwnedPoint3D(mergedXyz, mergedTrack, mergedColor);
	}

	/// <summary>Deletes a 3D point and all its references in the observing images.</summary>
	public void DeletePoint3D(ulong point3DId)
	{
		// Note: Do not change order of these instructions.
		Track track = Point3D(point3DId).Track;
		foreach (TrackElement trackEl in track.Elements)
		{
			Image(trackEl.ImageId).ResetPoint3DForPoint2D(trackEl.Point2DIdx);
		}

		_points3D.Remove(point3DId);
	}

	/// <summary>
	/// Deletes one observation from an image and the corresponding 3D point. Note that this
	/// deletes the entire 3D point if its track has two elements before the call.
	/// </summary>
	public void DeleteObservation(uint imageId, uint point2DIdx)
	{
		// Note: Do not change order of these instructions.
		Image image = Image(imageId);
		ulong point3DId = image.Point2DAt(point2DIdx).Point3DId;
		Point3D point3D = Point3D(point3DId);

		if (point3D.Track.Length <= 2)
		{
			DeletePoint3D(point3DId);
			return;
		}

		point3D.Track.DeleteElement(imageId, point2DIdx);
		image.ResetPoint3DForPoint2D(point2DIdx);
	}

	/// <summary>Deletes all 2D points of all images and all 3D points.</summary>
	public void DeleteAllPoints2DAndPoints3D()
	{
		_points3D.Clear();
		foreach (Image image in _images.Values)
		{
			image.SetPoints2D(Array.Empty<Vector2d>());
		}
	}

	/// <summary>
	/// Replaces all rigs and frames with the given ones, re-linking every image to its new
	/// frame. Frames with a pose are registered.
	/// </summary>
	public void SetRigsAndFrames(IEnumerable<Rig> rigs, IEnumerable<Frame> frames)
	{
		_rigs.Clear();
		foreach (Rig rig in rigs)
		{
			AddRig(rig);
		}

		_frames.Clear();
		_regFrameIds.Clear();
		_numRegImages = 0;
		var imageToFrameIds = new Dictionary<uint, uint>();
		foreach (Frame frame in frames)
		{
			foreach (DataId dataId in frame.ImageIds())
			{
				Check.That(imageToFrameIds.TryAdd((uint)dataId.Id, frame.FrameId));
			}

			AddFrame(frame);
		}

		foreach (Image image in _images.Values)
		{
			image.ResetFramePtr();
			image.SetFrameId(imageToFrameIds[image.ImageId]);
			image.SetFramePtr(Frame(image.FrameId));
		}
	}

	/// <summary>Registers an existing frame, which must have a pose. A no-op if registered.</summary>
	public void RegisterFrame(uint frameId)
	{
		Frame frame = Frame(frameId);
		Check.That(frame.HasPose);
		if (!_regFrameIds.Contains(frameId))
		{
			_regFrameIds.Add(frameId);
			_numRegImages += frame.ImageIds().Count();
		}
	}

	/// <summary>
	/// De-registers an existing frame: deletes the observations of its images, clears its
	/// pose and removes it from the registered frames. A no-op if not registered.
	/// </summary>
	public void DeRegisterFrame(uint frameId)
	{
		if (!_regFrameIds.Contains(frameId))
		{
			// COLMAP logs "Ignoring de-registration of frame ..., which is not registered."
			return;
		}

		Frame frame = Frame(frameId);
		foreach (DataId dataId in frame.ImageIds())
		{
			uint imageId = (uint)dataId.Id;
			Image image = Image(imageId);
			uint numPoints2D = image.NumPoints2D;
			for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
			{
				if (image.Point2DAt(point2DIdx).HasPoint3D)
				{
					DeleteObservation(imageId, point2DIdx);
				}
			}

			--_numRegImages;
		}

		frame.ResetPose();
		_regFrameIds.RemoveAll(id => id == frameId);
	}

	private static Vector3d ToDouble(Vector3ub color) => new(color.X, color.Y, color.Z);
}
