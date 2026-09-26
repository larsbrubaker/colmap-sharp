// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Rig: colmap/sensor/rig.h and rig.cc, a set of rigidly mounted sensors with their
// sensor_from_rig transforms. Keyed by Util/Types.cs's SensorId; the transforms are
// Geometry/Rigid3d. The scene-level rig helpers (colmap/scene/rig.h) are a separate port.
// Tests: ColmapSharp.Tests/Sensor/RigTests.cs (rig_test.cc 1:1).
//
// C++'s std::map becomes a SortedDictionary with the same (type, id) key order, so
// NonRefSensors and the printed form iterate as COLMAP's do. C++ hands out Rigid3d& and
// std::optional<Rigid3d>& for in-place edits; Rigid3d is a readonly struct here, so edits
// go through SetSensorFromRig / ResetSensorFromRig and NonRefSensors is read-only. Bundle
// adjustment edits sensor_from_rig in place through Ceres parameter blocks
// (bundle_adjustment_ceres.cc); Phase 8 gives it mutable parameter storage rather than
// references into this dictionary.
//
// Rig is a class (C++ copies it by value) with COLMAP's value equality; copying a Rig
// reference shares it.

using System.Globalization;
using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::Rig. Rigs represent a collection of rigidly mounted sensors and the
/// associated sensor from rig transformations. The reference sensor is defined to have
/// identity pose in the rig frame. This design is mainly for two purposes: 1) In
/// visual-inertial optimization, one of the IMUs is generally used as the reference frame
/// since it is metric. 2) Not having a reference frame brings a 6 DoF Gauge for each rig,
/// which is not ideal particularly when it comes to covariance estimation.
/// </summary>
public sealed class Rig : IEquatable<Rig>
{
	// sensor_from_rig transformations of the non-reference sensors; null where unknown.
	private readonly SortedDictionary<SensorId, Rigid3d?> _sensorsFromRig = [];

	/// <summary>Unique identifier of the rig; <see cref="InvalidRigId"/> until set.</summary>
	public uint RigId { get; set; } = InvalidRigId;

	/// <summary>
	/// The reference sensor, which has the identity transformation to the rig (the first
	/// added sensor); <see cref="InvalidSensorId"/> until added.
	/// </summary>
	public SensorId RefSensorId { get; private set; } = InvalidSensorId;

	/// <summary>Number of sensors, including the reference sensor.</summary>
	public int NumSensors
	{
		get
		{
			int numSensors = _sensorsFromRig.Count;
			if (RefSensorId != InvalidSensorId)
			{
				numSensors += 1;
			}

			return numSensors;
		}
	}

	/// <summary>All sensors except the reference sensor, in (type, id) order.</summary>
	public IReadOnlyDictionary<SensorId, Rigid3d?> NonRefSensors => _sensorsFromRig;

	/// <summary>
	/// Adds the reference sensor. Must be called before all <see cref="AddSensor"/> calls.
	/// </summary>
	public void AddRefSensor(SensorId refSensorId)
	{
		Check.That(RefSensorId == InvalidSensorId, "Reference sensor already set");
		RefSensorId = refSensorId;
	}

	/// <summary>Adds a non-reference sensor, optionally with its sensor_from_rig transform.</summary>
	public void AddSensor(SensorId sensorId, Rigid3d? sensorFromRig = null)
	{
		Check.Ge(NumSensors, 1, "The reference sensor needs to be added first before other sensors.");
		Check.That(!HasSensor(sensorId), $"Sensor ({sensorId.Type.ToColmapString()}, {sensorId.Id}) is inserted twice into the rig");
		_sensorsFromRig.Add(sensorId, sensorFromRig);
	}

	/// <summary>Whether the sensor (reference or not) is in the rig.</summary>
	public bool HasSensor(SensorId sensorId) => sensorId == RefSensorId || _sensorsFromRig.ContainsKey(sensorId);

	/// <summary>Whether the sensor is the reference sensor of the rig.</summary>
	public bool IsRefSensor(SensorId sensorId) => sensorId == RefSensorId;

	/// <summary>Whether a non-reference sensor has a known sensor_from_rig transform.</summary>
	public bool HasSensorFromRig(SensorId sensorId) =>
		sensorId != RefSensorId && HasSensor(sensorId) && _sensorsFromRig[sensorId].HasValue;

	/// <summary>All sensor ids, including the reference sensor, in (type, id) order.</summary>
	public SortedSet<SensorId> SensorIds()
	{
		var sensorIds = new SortedSet<SensorId> { RefSensorId };
		foreach (SensorId sensorId in _sensorsFromRig.Keys)
		{
			sensorIds.Add(sensorId);
		}

		return sensorIds;
	}

	/// <summary>
	/// The sensor_from_rig transform. Throws for the reference sensor, an unknown sensor, or
	/// a sensor whose transform is unset (C++ std::optional::value throws).
	/// </summary>
	public Rigid3d SensorFromRig(SensorId sensorId) =>
		FindSensorFromRigOrThrow(sensorId) ?? throw new InvalidOperationException("bad optional access");

	/// <summary>The sensor_from_rig transform, or null if unset. Throws like SensorFromRig otherwise.</summary>
	public Rigid3d? MaybeSensorFromRig(SensorId sensorId) => FindSensorFromRigOrThrow(sensorId);

	/// <summary>Sets (or, with null, clears) a non-reference sensor's sensor_from_rig transform.</summary>
	public void SetSensorFromRig(SensorId sensorId, Rigid3d? sensorFromRig)
	{
		FindSensorFromRigOrThrow(sensorId);
		_sensorsFromRig[sensorId] = sensorFromRig;
	}

	/// <summary>Clears a non-reference sensor's sensor_from_rig transform.</summary>
	public void ResetSensorFromRig(SensorId sensorId) => SetSensorFromRig(sensorId, null);

	/// <summary>The C++ operator==: same id, reference sensor and transforms (exact).</summary>
	public bool Equals(Rig? other)
	{
		if (other is null
			|| RigId != other.RigId
			|| RefSensorId != other.RefSensorId
			|| _sensorsFromRig.Count != other._sensorsFromRig.Count)
		{
			return false;
		}

		// std::map == compares the ordered (key, value) sequences; the lifted Rigid3d? ==
		// is std::optional's (both empty, or both set and Rigid3d == holds).
		foreach (var (mine, theirs) in _sensorsFromRig.Zip(other._sensorsFromRig))
		{
			if (mine.Key != theirs.Key || mine.Value != theirs.Value)
			{
				return false;
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => Equals(obj as Rig);

	/// <summary>Hash of the identity fields (the transforms may change in place).</summary>
	public override int GetHashCode() => HashCode.Combine(RigId, RefSensorId);

	/// <summary>The C++ operator==.</summary>
	public static bool operator ==(Rig? a, Rig? b) => a is null ? b is null : a.Equals(b);

	/// <summary>The C++ operator!=.</summary>
	public static bool operator !=(Rig? a, Rig? b) => !(a == b);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g.
	/// "Rig(rig_id=0, ref_sensor_id=(IMU, 0), sensors=[(CAMERA, 1), (CAMERA, 2)])".
	/// </summary>
	public override string ToString()
	{
		string rigIdStr = RigId != InvalidRigId ? RigId.ToString(CultureInfo.InvariantCulture) : "Invalid";
		var stream = new StringBuilder();
		stream.Append("Rig(rig_id=").Append(rigIdStr).Append(", ref_sensor_id=").Append(FormatSensor(RefSensorId)).Append(", sensors=[");
		stream.AppendJoin(", ", _sensorsFromRig.Keys.Select(FormatSensor));
		stream.Append("])");
		return stream.ToString();
	}

	// "(TYPE, id)", how COLMAP streams a sensor_t's fields.
	private static string FormatSensor(SensorId sensorId) =>
		string.Create(CultureInfo.InvariantCulture, $"({sensorId.Type.ToColmapString()}, {sensorId.Id})");

	private Rigid3d? FindSensorFromRigOrThrow(SensorId sensorId)
	{
		Check.That(sensorId != RefSensorId, "The reference sensor does not have a SensorFromRig transformation, which is fixed to identity");
		Check.That(_sensorsFromRig.TryGetValue(sensorId, out Rigid3d? sensorFromRig), $"Sensor ({sensorId.Type.ToColmapString()}, {sensorId.Id}) not found in the rig");
		return sensorFromRig;
	}
}
