// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Frame: port of colmap/scene/frame.h and frame.cc, a (posed) instantiation of a rig
// (Sensor/Rig.cs) together with the ids of the measurements its sensors took. Images
// (Scene/Image.cs) get their pose from their frame. Tests:
// ColmapSharp.Tests/Scene/FrameTests.cs (frame_test.cc 1:1).
//
// Design (later phases build on it):
// - Ownership. The Reconstruction (a later Phase 4 port) owns rigs, frames, images and
//   cameras. C++'s raw `Rig* rig_ptr_` becomes a non-owning C# reference that the
//   Reconstruction sets with SetRigPtr and clears with ResetRigPtr, under COLMAP's rules:
//   whoever removes a rig resets the frames pointing at it. A stale reference keeps the old
//   rig alive rather than dangling, so forgetting a reset is a silent bug, not a crash.
// - Pose storage. rig_from_world is COLMAP's std::optional<Rigid3d>. Here the value lives
//   in a Geometry/Rigid3dStorage the frame owns for its whole lifetime, plus a HasPose flag.
//   Phase 8's bundle adjustment registers RigFromWorldStorage.Params as ONE 7-value
//   parameter block [qx, qy, qz, qw, tx, ty, tz] (COLMAP 4.2.0's
//   `rig_from_world.params.data()`, with the EigenQuaternion x Euclidean<3> product
//   manifold) and writes into it in place, as Ceres does through `Rigid3d&` in COLMAP;
//   ResetPose and SetRigFromWorld keep the same array, so a held block stays valid.
//   Contract for the optimizer: fetch RigFromWorldStorage (and Camera.Params) when setting
//   up a problem, and while that problem is live do not ResetPose, assign Camera.Params or
//   call SetParamsFromString. ResetPose only clears HasPose (the array keeps the old
//   values, which the solver would go on refining), and replacing Params swaps in a new
//   array the solver does not see.
// - Copying. C++ copy construction/assignment is Clone(): it copies ids, data ids and the
//   pose (into new storage), shares the rig reference, and is never finalized.
// - DataIds is COLMAP's std::set<data_t>, a SortedSet here with the same (type, sensor id,
//   id) order, so printing and iteration match.

using System.Globalization;
using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Frame. Frames represent (posed) instantiations of rigs with associated
/// measurements for the different sensors. The captured sensor measurements are defined by
/// the list of data ids.
/// </summary>
public sealed class Frame : IEquatable<Frame>
{
	private readonly SortedSet<DataId> _dataIds = [];

	// The rig_from_world transformation; valid only while _hasPose. If the rig is null, the
	// frame is a single sensor case, where rig modeling is no longer needed.
	private readonly Rigid3dStorage _rigFromWorld = new();
	private bool _hasPose;

	private uint _rigId = InvalidRigId;
	private Rig? _rigPtr;

	/// <summary>The unique identifier of the frame; <see cref="InvalidFrameId"/> until set.</summary>
	public uint FrameId { get; set; } = InvalidFrameId;

	/// <summary>The frame's associated data, in (type, sensor id, id) order.</summary>
	public IReadOnlySet<DataId> DataIds => _dataIds;

	/// <summary>Number of associated data ids.</summary>
	public int NumDataIds => _dataIds.Count;

	/// <summary>
	/// Whether the data ids are final. Set when the frame is added to a reconstruction, to
	/// keep cached counters like num_reg_images_ consistent.
	/// </summary>
	public bool HasFinalDataIds { get; private set; }

	/// <summary>
	/// The unique identifier of the rig; <see cref="InvalidRigId"/> until set. Multiple
	/// frames may share the same rig.
	/// </summary>
	public uint RigId => _rigId;

	/// <summary>Whether the rig id has been set.</summary>
	public bool HasRigId => _rigId != InvalidRigId;

	/// <summary>
	/// The shared rig object, typically only set when the frame was added to a
	/// reconstruction; throws if unset.
	/// </summary>
	public Rig RigPtr => Check.NotNull(_rigPtr);

	/// <summary>Whether the rig reference is set.</summary>
	public bool HasRigPtr => _rigPtr is not null;

	/// <summary>Whether the frame has a rig_from_world pose.</summary>
	public bool HasPose => _hasPose;

	/// <summary>The rig_from_world pose, or null.</summary>
	public Rigid3d? MaybeRigFromWorld => _hasPose ? _rigFromWorld.Value : null;

	/// <summary>
	/// The mutable storage behind the pose, for an optimizer to write into (see the file
	/// header). Throws if the frame has no pose.
	/// </summary>
	public Rigid3dStorage RigFromWorldStorage
	{
		get
		{
			Check.That(_hasPose, "Frame does not have a valid pose.");
			return _rigFromWorld;
		}
	}

	/// <summary>Adds a data id. Throws if the frame is finalized or the rig lacks the sensor.</summary>
	public void AddDataId(DataId dataId)
	{
		Check.That(!HasFinalDataIds, "Cannot add data id to a finalized frame. Data ids must be added before the frame is added to a reconstruction.");
		if (HasRigPtr)
		{
			Check.That(RigPtr.HasSensor(dataId.SensorId));
		}

		_dataIds.Add(dataId);
	}

	/// <summary>Whether the data is associated with the frame.</summary>
	public bool HasDataId(DataId dataId) => _dataIds.Contains(dataId);

	/// <summary>Finalizes the data ids, preventing further modifications.</summary>
	public void FinalizeDataIds() => HasFinalDataIds = true;

	/// <summary>Clears all the associated data. Throws if the frame is finalized.</summary>
	public void ClearDataIds()
	{
		Check.That(!HasFinalDataIds, "Cannot clear data ids of a finalized frame.");
		_dataIds.Clear();
	}

	/// <summary>
	/// Replaces the data ids and finalizes them, even if already finalized. This is C++'s
	/// "copy the frame, clear and re-add its data ids, finalize, move-assign back" in
	/// Reconstruction::TranscribeImageIdsToDatabase, done in place so the images' FramePtr
	/// and bundle adjustment's RigFromWorldStorage keep referring to this object.
	/// </summary>
	internal void ReplaceFinalizedDataIds(IReadOnlyList<DataId> dataIds)
	{
		HasFinalDataIds = false;
		ClearDataIds();
		foreach (DataId dataId in dataIds)
		{
			AddDataId(dataId);
		}

		FinalizeDataIds();
	}

	/// <summary>All data ids of a sensor type, in order.</summary>
	public IEnumerable<DataId> DataIdsOfType(SensorType type) => _dataIds.Where(dataId => dataId.SensorId.Type == type);

	/// <summary>All image data ids, in order.</summary>
	public IEnumerable<DataId> ImageIds() => DataIdsOfType(SensorType.Camera);

	/// <summary>Sets the rig id; throws for an invalid id or when the rig reference is set.</summary>
	public void SetRigId(uint rigId)
	{
		Check.Ne(rigId, InvalidRigId);
		Check.That(!HasRigPtr);
		_rigId = rigId;
	}

	/// <summary>
	/// Sets the shared rig. Every camera data id must be a sensor of the rig. Without a rig
	/// set yet, the rig's id must equal <see cref="RigId"/>; replacing a set rig adopts the
	/// new rig's id.
	/// </summary>
	public void SetRigPtr(Rig rig)
	{
		Check.NotNull(rig);
		Check.Ne(rig.RigId, InvalidRigId);
		foreach (DataId dataId in _dataIds)
		{
			switch (dataId.SensorId.Type)
			{
				case SensorType.Camera:
					Check.That(rig.HasSensor(dataId.SensorId));
					break;
				case SensorType.Imu:
					// Note that we do not (yet) support IMU measurement data.
					break;
				default:
					throw new ArgumentException($"Invalid sensor type: {dataId.SensorId.Type.ToColmapString()}");
			}
		}

		if (HasRigPtr)
		{
			_rigId = rig.RigId;
			_rigPtr = rig;
		}
		else
		{
			Check.Eq(rig.RigId, _rigId);
			_rigPtr = rig;
		}
	}

	/// <summary>Clears the rig reference.</summary>
	public void ResetRigPtr() => _rigPtr = null;

	/// <summary>The rig_from_world pose; throws if the frame has none.</summary>
	public Rigid3d RigFromWorld()
	{
		Check.That(_hasPose, "Frame does not have a valid pose.");
		return _rigFromWorld.Value;
	}

	/// <summary>Sets the rig_from_world pose.</summary>
	public void SetRigFromWorld(Rigid3d rigFromWorld)
	{
		_rigFromWorld.Value = rigFromWorld;
		_hasPose = true;
	}

	/// <summary>Sets the rig_from_world pose, or clears it with null.</summary>
	public void SetRigFromWorld(Rigid3d? rigFromWorld)
	{
		if (rigFromWorld is Rigid3d value)
		{
			SetRigFromWorld(value);
		}
		else
		{
			ResetPose();
		}
	}

	/// <summary>Clears the pose.</summary>
	public void ResetPose() => _hasPose = false;

	/// <summary>The sensor_from_world transformation of a sensor of the rig.</summary>
	public Rigid3d SensorFromWorld(SensorId sensorId)
	{
		Rig rig = Check.NotNull(_rigPtr);
		if (rig.IsRefSensor(sensorId))
		{
			return RigFromWorld();
		}

		return rig.SensorFromRig(sensorId) * RigFromWorld();
	}

	/// <summary>Sets rig_from_world from the cam_from_world pose of one of the rig's cameras.</summary>
	public void SetCamFromWorld(uint cameraId, Rigid3d camFromWorld)
	{
		Rig rig = Check.NotNull(_rigPtr);
		var sensorId = new SensorId(SensorType.Camera, cameraId);
		if (rig.IsRefSensor(sensorId))
		{
			SetRigFromWorld(camFromWorld);
		}
		else
		{
			Rigid3d camFromRig = rig.SensorFromRig(sensorId);
			SetRigFromWorld(camFromRig.Inverse() * camFromWorld);
		}
	}

	/// <summary>
	/// A copy (C++ copy construction): same ids, data ids and pose (in new storage), the
	/// same rig reference, and not finalized.
	/// </summary>
	public Frame Clone()
	{
		var copy = new Frame
		{
			FrameId = FrameId,
			_rigId = _rigId,
			_rigPtr = _rigPtr,
		};
		copy._dataIds.UnionWith(_dataIds);
		copy.SetRigFromWorld(MaybeRigFromWorld);
		return copy;
	}

	/// <summary>The C++ operator==: same ids, data ids and (exactly equal) pose.</summary>
	public bool Equals(Frame? other) =>
		other is not null
		&& FrameId == other.FrameId
		&& _rigId == other._rigId
		&& _dataIds.SetEquals(other._dataIds)
		&& MaybeRigFromWorld == other.MaybeRigFromWorld;

	/// <inheritdoc/>
	public override bool Equals(object? obj) => Equals(obj as Frame);

	/// <summary>Hash of the frame id (everything else may change in place).</summary>
	public override int GetHashCode() => FrameId.GetHashCode();

	/// <summary>The C++ operator==.</summary>
	public static bool operator ==(Frame? a, Frame? b) => a is null ? b is null : a.Equals(b);

	/// <summary>The C++ operator!=.</summary>
	public static bool operator !=(Frame? a, Frame? b) => !(a == b);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g.
	/// "Frame(frame_id=1, rig_id=2, has_pose=0, data_ids=[(CAMERA, 1, 3), (IMU, 0, 2)])".
	/// </summary>
	public override string ToString()
	{
		var stream = new StringBuilder();
		stream.Append("Frame(frame_id=").Append(FrameId.ToString(CultureInfo.InvariantCulture)).Append(", rig_id=");
		// COLMAP's "Invalid" branch (a set rig id equal to kInvalidRigId) cannot be reached,
		// since HasRigId is exactly rig_id != kInvalidRigId.
		stream.Append(HasRigId ? RigId.ToString(CultureInfo.InvariantCulture) : "Unknown");
		stream.Append(", has_pose=").Append(HasPose ? '1' : '0').Append(", data_ids=[");
		stream.AppendJoin(", ", _dataIds.Select(dataId => string.Create(
			CultureInfo.InvariantCulture,
			$"({dataId.SensorId.Type.ToColmapString()}, {dataId.SensorId.Id}, {dataId.Id})")));
		stream.Append("])");
		return stream.ToString();
	}
}
