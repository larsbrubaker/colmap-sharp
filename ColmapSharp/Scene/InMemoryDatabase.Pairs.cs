// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// InMemoryDatabase.Pairs: the two-view geometry table of InMemoryDatabase (the
// two_view_geometries half of colmap/scene/database_sqlite.cc) and the row types, indices
// and constraint checks shared by both halves. See InMemoryDatabase.cs for the behavior
// this reproduces.
//
// A two-view geometry is stored under the normalized pair (inverted on write when
// image_id1 > image_id2, inverted back on read), holding exactly the columns COLMAP stores:
// config, inlier matches, F, E, H, cam2_from_cam1 and the two optional cameras. The
// triangulation angle is not a column, so a read returns the default (-1) like COLMAP.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

public sealed partial class InMemoryDatabase
{
	private readonly SortedDictionary<ulong, TwoViewGeometry> twoViewGeometries = [];

	// The UNIQUE indices on images(name), frame_data(data_id, sensor_type) and
	// pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type), kept for fast lookups.
	private readonly Dictionary<string, uint> imageIdsByName = new(StringComparer.Ordinal);
	private readonly Dictionary<(ulong DataId, SensorType Type), uint> frameIdsByData = [];
	private readonly Dictionary<DataId, uint> posePriorIdsByData = [];

	/// <inheritdoc/>
	public override bool ExistsTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			return twoViewGeometries.ContainsKey(ImagePairToPairId(imageId1, imageId2));
		}
	}

	/// <inheritdoc/>
	public override long NumInlierMatches() => Locked(() => twoViewGeometries.Values.Sum(geometry => (long)geometry.InlierMatches.Count));

	/// <inheritdoc/>
	public override long NumVerifiedImagePairs() => Locked(() => twoViewGeometries.Count);

	/// <inheritdoc/>
	public override TwoViewGeometry ReadTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			if (!twoViewGeometries.TryGetValue(ImagePairToPairId(imageId1, imageId2), out TwoViewGeometry? stored))
			{
				return new TwoViewGeometry();
			}

			TwoViewGeometry twoViewGeometry = CopyStoredColumns(stored);
			if (ShouldSwapImagePair(imageId1, imageId2))
			{
				twoViewGeometry.Invert();
			}

			return twoViewGeometry;
		}
	}

	/// <inheritdoc/>
	public override List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> ReadTwoViewGeometries()
	{
		// COLMAP selects the rows with any inlier, matrix, pose or camera set.
		return Locked(() => twoViewGeometries
			.Where(entry => entry.Value.InlierMatches.Count > 0
				|| entry.Value.F.HasValue || entry.Value.E.HasValue || entry.Value.H.HasValue
				|| entry.Value.Cam2FromCam1.HasValue
				|| entry.Value.Camera1 is not null || entry.Value.Camera2 is not null)
			.Select(entry => (entry.Key, CopyStoredColumns(entry.Value)))
			.ToList());
	}

	/// <inheritdoc/>
	public override List<(ulong PairId, int NumInliers)> ReadTwoViewGeometryNumInliers() =>
		Locked(() => twoViewGeometries
			.Where(entry => entry.Value.InlierMatches.Count > 0)
			.Select(entry => (entry.Key, entry.Value.InlierMatches.Count))
			.ToList());

	/// <inheritdoc/>
	public override void WriteTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		lock (sync)
		{
			EnsureOpen();
			ulong pairId = ImagePairToPairId(imageId1, imageId2);
			Check.That(
				!twoViewGeometries.ContainsKey(pairId),
				$"Two view geometry between image {imageId1} and {imageId2} already exists.");

			// Invert the two-view geometry if the image pair has to be swapped.
			TwoViewGeometry stored = CopyStoredColumns(twoViewGeometry);
			if (ShouldSwapImagePair(imageId1, imageId2))
			{
				stored.Invert();
			}

			twoViewGeometries.Add(pairId, stored);
		}
	}

	/// <inheritdoc/>
	public override void UpdateTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		lock (sync)
		{
			// Do nothing if the image pair does not exist, to align with the UPDATE behavior
			// in SQL.
			if (ExistsTwoViewGeometry(imageId1, imageId2))
			{
				DeleteTwoViewGeometry(imageId1, imageId2);
				WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
			}
		}
	}

	/// <inheritdoc/>
	public override void DeleteTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			twoViewGeometries.Remove(ImagePairToPairId(imageId1, imageId2));
		}
	}

	/// <inheritdoc/>
	public override void DeleteInlierMatches(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			if (!ExistsTwoViewGeometry(imageId1, imageId2))
			{
				return;
			}

			TwoViewGeometry geometry = ReadTwoViewGeometry(imageId1, imageId2);
			geometry.InlierMatches.Clear();
			UpdateTwoViewGeometry(imageId1, imageId2, geometry);
		}
	}

	/// <inheritdoc/>
	public override void ClearTwoViewGeometries() => Locked(() => { twoViewGeometries.Clear(); return 0; });

	// The columns of the two_view_geometries table, copied (TriAngle is not stored).
	private static TwoViewGeometry CopyStoredColumns(TwoViewGeometry source) => new()
	{
		Config = source.Config,
		F = source.F,
		E = source.E,
		H = source.H,
		Cam2FromCam1 = source.Cam2FromCam1,
		Camera1 = source.Camera1?.Clone(),
		Camera2 = source.Camera2?.Clone(),
		InlierMatches = [.. source.InlierMatches],
	};

	private static void MaybeThrowDeprecatedPosePriorError(bool isDeprecatedImagePrior)
	{
		if (isDeprecatedImagePrior)
		{
			throw new InvalidOperationException(
				"PosePrior API has changed: pose priors are now associated with frames, not images. Please "
				+ "update your code to use frames instead of image IDs. Data is automatically migrated upon "
				+ "opening a database. Update your API usage accordingly and add pose priors to frame data.");
		}
	}

	// sqlite's "FOREIGN KEY constraint failed".
	private static void CheckForeignKey(bool condition)
	{
		if (!condition)
		{
			throw new InvalidOperationException("SQLite error: FOREIGN KEY constraint failed");
		}
	}

	// sqlite's "UNIQUE constraint failed: <columns>".
	private static void CheckUnique(bool condition, string columns)
	{
		if (!condition)
		{
			throw new InvalidOperationException($"SQLite error: UNIQUE constraint failed: {columns}");
		}
	}

	// The next id of an AUTOINCREMENT table: one more than the largest id it ever held.
	private static uint NextAutoIncrementId(long sequence) => checked((uint)(sequence + 1));

	// ReadCameraRow: the parameter count must fit the model.
	private static Camera ReadCameraRow(Camera stored)
	{
		Check.Eq(stored.Params.Length, CameraModels.CameraModelNumParams(stored.ModelId));
		return stored.Clone();
	}

	private T Locked<T>(Func<T> read)
	{
		lock (sync)
		{
			EnsureOpen();
			return read();
		}
	}

	private void EnsureOpen()
	{
		if (closed)
		{
			throw new InvalidOperationException("The database is closed.");
		}
	}

	private uint? FindImageWithName(string name) => imageIdsByName.TryGetValue(name, out uint imageId) ? imageId : null;

	// The images LEFT JOIN frame_data: the frame id is that of the frame holding the image as
	// camera data, if any.
	private Image ToImage(uint imageId, ImageRow row)
	{
		var image = new Image { ImageId = imageId, Name = row.Name };
		image.SetCameraId(row.CameraId);
		if (frameIdsByData.TryGetValue((imageId, SensorType.Camera), out uint frameId))
		{
			image.SetFrameId(frameId);
		}

		return image;
	}

	// UNIQUE index index_name, CHECK image_id_check and the camera foreign key.
	private void CheckImageConstraints(uint imageId, string name, uint cameraId)
	{
		Check.That((ulong)imageId < MaxNumImages, "image_id_check");
		CheckUnique(!imageIdsByName.TryGetValue(name, out uint other) || other == imageId, "images.name");
		CheckForeignKey(cameras.ContainsKey(cameraId));
	}

	// UNIQUE indices rig_ref_sensor_assignment and rig_sensor_assignment.
	private void CheckRigConstraints(RigRow row, uint? ignoreRigId)
	{
		foreach (var (rigId, other) in rigs)
		{
			if (rigId == ignoreRigId)
			{
				continue;
			}

			CheckUnique(other.RefSensorId != row.RefSensorId, "rigs.ref_sensor_id, rigs.ref_sensor_type");
			foreach (var (sensorId, _) in row.Sensors)
			{
				CheckUnique(other.Sensors.All(sensor => sensor.SensorId != sensorId), "rig_sensors.sensor_id, rig_sensors.sensor_type");
			}
		}

		CheckUnique(
			row.Sensors.Select(sensor => sensor.SensorId).Distinct().Count() == row.Sensors.Count,
			"rig_sensors.sensor_id, rig_sensors.sensor_type");
	}

	// UNIQUE index frame_sensor_assignment on frame_data(data_id, sensor_type).
	private void CheckFrameDataConstraints(List<DataId> dataIds, uint? ignoreFrameId)
	{
		var seen = new HashSet<(ulong, SensorType)>();
		foreach (DataId dataId in dataIds)
		{
			var key = (dataId.Id, dataId.SensorId.Type);
			bool free = !frameIdsByData.TryGetValue(key, out uint owner) || owner == ignoreFrameId;
			CheckUnique(free && seen.Add(key), "frame_data.data_id, frame_data.sensor_type");
		}
	}

	// UNIQUE index pose_prior_data_assignment.
	private void CheckPosePriorConstraints(DataId corrDataId, uint? ignorePosePriorId)
	{
		CheckUnique(
			!posePriorIdsByData.TryGetValue(corrDataId, out uint owner) || owner == ignorePosePriorId,
			"pose_priors.corr_data_id, pose_priors.corr_sensor_id, pose_priors.corr_sensor_type");
	}

	// A rigs row with its rig_sensors rows.
	private sealed record RigRow(SensorId RefSensorId, List<(SensorId SensorId, Rigid3d? SensorFromRig)> Sensors)
	{
		public static RigRow FromRig(Rig rig) =>
			new(rig.RefSensorId, rig.NonRefSensors.Select(entry => (entry.Key, entry.Value)).ToList());

		public Rig ToRig(uint rigId)
		{
			var rig = new Rig { RigId = rigId };
			rig.AddRefSensor(RefSensorId);
			foreach (var (sensorId, sensorFromRig) in Sensors)
			{
				rig.AddSensor(sensorId, sensorFromRig);
			}

			return rig;
		}
	}

	// A frames row with its frame_data rows.
	private sealed record FrameRow(uint RigId, List<DataId> DataIds)
	{
		public Frame ToFrame(uint frameId)
		{
			var frame = new Frame { FrameId = frameId };
			frame.SetRigId(RigId);
			foreach (DataId dataId in DataIds)
			{
				frame.AddDataId(dataId);
			}

			return frame;
		}
	}

	// An images row.
	private sealed record ImageRow(string Name, uint CameraId);
}
