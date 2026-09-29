// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// InMemoryDatabase: the Database (Database.cs) implementation of this library - a managed,
// in-memory replacement for COLMAP's SqliteDatabase (colmap/scene/database_sqlite.cc), whose
// SQLite engine is native and out of scope (docs/LICENSE_AUDIT.md). It reproduces the
// behavior that SQLite schema and statements give COLMAP:
// - ids: rigs, cameras, frames and images are AUTOINCREMENT tables, so a new id is one more
//   than the largest id the table ever held (explicit ids included; clearing a table does
//   not reset it). Pose priors are a plain INTEGER PRIMARY KEY: one more than the largest
//   current id, or 1.
// - uniqueness (the UNIQUE indices and primary keys): image names; a sensor is the
//   reference sensor of at most one rig and a non-reference sensor of at most one rig; a
//   (data id, sensor type) belongs to at most one frame; one pose prior per corresponding
//   data id; one keypoints/descriptors row per image; one matches/two-view geometry row
//   per pair. Violations throw, like SQLite's constraint errors.
// - foreign keys (PRAGMA foreign_keys=ON): images need an existing camera, frames an
//   existing rig, keypoints and descriptors an existing image. Deleting rigs cascades to
//   their frames, deleting images to their keypoints and descriptors, and clearing the
//   cameras fails while any image exists. Matches and two-view geometries have no keys.
// - iteration: every Read* returns rows ordered by id (the rowid order of COLMAP's
//   SELECTs); pairs are ordered by pair id.
// - UPDATE of a missing row is a no-op; reads of a missing row return defaults.
// Every stored object is a private copy, so callers can keep mutating what they wrote or
// read. A lock serializes all calls (COLMAP's SQLite connection is not thread-safe at all).
// A write that violates a constraint leaves no partial rows, see divergence 23.
// The two-view geometry half is InMemoryDatabase.Pairs.cs.
//
// Tests: ColmapSharp.Tests/Scene/DatabaseTests.cs (database_test.cc 1:1). Tier A (exact).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// An in-memory <see cref="Database"/> with the observable behavior of COLMAP's SQLite
/// database (ids, constraints, ordering), see the file header.
/// </summary>
public sealed partial class InMemoryDatabase : Database
{
	private readonly object sync = new();

	private readonly SortedDictionary<uint, RigRow> rigs = [];
	private readonly SortedDictionary<uint, Camera> cameras = [];
	private readonly SortedDictionary<uint, FrameRow> frames = [];
	private readonly SortedDictionary<uint, ImageRow> images = [];
	private readonly SortedDictionary<uint, PosePrior> posePriors = [];
	private readonly SortedDictionary<uint, RowMajorMatrix<float>> keypoints = [];
	private readonly SortedDictionary<uint, FeatureDescriptors> descriptors = [];
	private readonly SortedDictionary<ulong, RowMajorMatrix<uint>> matches = [];

	// The sqlite_sequence of the AUTOINCREMENT tables: the largest id ever inserted.
	private long rigSequence;
	private long cameraSequence;
	private long frameSequence;
	private long imageSequence;

	// max(pose_prior_id) of the current rows, 0 when empty (rows are only ever removed all at
	// once, by ClearPosePriors).
	private long maxPosePriorId;

	private bool closed;

	/// <inheritdoc/>
	public override void Close()
	{
		lock (sync)
		{
			closed = true;
		}
	}

	/// <inheritdoc/>
	public override bool ExistsRig(uint rigId)
	{
		lock (sync)
		{
			EnsureOpen();
			return rigs.ContainsKey(rigId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsCamera(uint cameraId)
	{
		lock (sync)
		{
			EnsureOpen();
			return cameras.ContainsKey(cameraId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsFrame(uint frameId)
	{
		lock (sync)
		{
			EnsureOpen();
			return frames.ContainsKey(frameId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsImage(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return images.ContainsKey(imageId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsImageWithName(string name)
	{
		lock (sync)
		{
			EnsureOpen();
			return FindImageWithName(name) is not null;
		}
	}

	/// <inheritdoc/>
	public override bool ExistsPosePrior(uint posePriorId, bool isDeprecatedImagePrior = true)
	{
		MaybeThrowDeprecatedPosePriorError(isDeprecatedImagePrior);
		lock (sync)
		{
			EnsureOpen();
			return posePriors.ContainsKey(posePriorId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsKeypoints(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return keypoints.ContainsKey(imageId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsDescriptors(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return descriptors.ContainsKey(imageId);
		}
	}

	/// <inheritdoc/>
	public override bool ExistsMatches(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			return matches.ContainsKey(ImagePairToPairId(imageId1, imageId2));
		}
	}

	/// <inheritdoc/>
	public override long NumRigs() => Locked(() => rigs.Count);

	/// <inheritdoc/>
	public override long NumCameras() => Locked(() => cameras.Count);

	/// <inheritdoc/>
	public override long NumFrames() => Locked(() => frames.Count);

	/// <inheritdoc/>
	public override long NumImages() => Locked(() => images.Count);

	/// <inheritdoc/>
	public override long NumPosePriors() => Locked(() => posePriors.Count);

	/// <inheritdoc/>
	public override long NumKeypoints() => Locked(() => keypoints.Values.Sum(blob => (long)blob.Rows));

	/// <inheritdoc/>
	public override long MaxNumKeypoints() => Locked(() => keypoints.Values.Select(blob => (long)blob.Rows).DefaultIfEmpty(0).Max());

	/// <inheritdoc/>
	public override long NumKeypointsForImage(uint imageId) =>
		Locked(() => keypoints.TryGetValue(imageId, out var blob) ? blob.Rows : 0L);

	/// <inheritdoc/>
	public override long NumDescriptors() => Locked(() => descriptors.Values.Sum(desc => (long)desc.Data.Rows));

	/// <inheritdoc/>
	public override long MaxNumDescriptors() =>
		Locked(() => descriptors.Values.Select(desc => (long)desc.Data.Rows).DefaultIfEmpty(0).Max());

	/// <inheritdoc/>
	public override long NumDescriptorsForImage(uint imageId) =>
		Locked(() => descriptors.TryGetValue(imageId, out var desc) ? desc.Data.Rows : 0L);

	/// <inheritdoc/>
	public override long NumMatches() => Locked(() => matches.Values.Sum(blob => (long)blob.Rows));

	/// <inheritdoc/>
	public override long NumMatchedImagePairs() => Locked(() => matches.Count);

	/// <inheritdoc/>
	public override Rig ReadRig(uint rigId)
	{
		lock (sync)
		{
			EnsureOpen();
			return rigs.TryGetValue(rigId, out RigRow? row) ? row.ToRig(rigId) : new Rig();
		}
	}

	/// <inheritdoc/>
	public override Rig? ReadRigWithSensor(SensorId sensorId)
	{
		lock (sync)
		{
			EnsureOpen();

			// COLMAP looks among the non-reference sensors first, then the reference sensors.
			foreach (var (rigId, row) in rigs)
			{
				if (row.Sensors.Any(sensor => sensor.SensorId == sensorId))
				{
					return row.ToRig(rigId);
				}
			}

			foreach (var (rigId, row) in rigs)
			{
				if (row.RefSensorId == sensorId)
				{
					return row.ToRig(rigId);
				}
			}

			return null;
		}
	}

	/// <inheritdoc/>
	public override List<Rig> ReadAllRigs() => Locked(() => rigs.Select(entry => entry.Value.ToRig(entry.Key)).ToList());

	/// <inheritdoc/>
	public override Camera ReadCamera(uint cameraId)
	{
		lock (sync)
		{
			EnsureOpen();
			return cameras.TryGetValue(cameraId, out Camera? camera) ? ReadCameraRow(camera) : new Camera();
		}
	}

	/// <inheritdoc/>
	public override List<Camera> ReadAllCameras() => Locked(() => cameras.Values.Select(ReadCameraRow).ToList());

	/// <inheritdoc/>
	public override Frame ReadFrame(uint frameId)
	{
		lock (sync)
		{
			EnsureOpen();
			return frames.TryGetValue(frameId, out FrameRow? row) ? row.ToFrame(frameId) : new Frame();
		}
	}

	/// <inheritdoc/>
	public override List<Frame> ReadAllFrames() => Locked(() => frames.Select(entry => entry.Value.ToFrame(entry.Key)).ToList());

	/// <inheritdoc/>
	public override Image ReadImage(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return images.TryGetValue(imageId, out ImageRow? row) ? ToImage(imageId, row) : new Image();
		}
	}

	/// <inheritdoc/>
	public override Image? ReadImageWithName(string name)
	{
		lock (sync)
		{
			EnsureOpen();
			uint? imageId = FindImageWithName(name);
			return imageId is uint id ? ToImage(id, images[id]) : null;
		}
	}

	/// <inheritdoc/>
	public override List<Image> ReadAllImages() => Locked(() => images.Select(entry => ToImage(entry.Key, entry.Value)).ToList());

	/// <inheritdoc/>
	public override PosePrior ReadPosePrior(uint posePriorId, bool isDeprecatedImagePrior = true)
	{
		MaybeThrowDeprecatedPosePriorError(isDeprecatedImagePrior);
		lock (sync)
		{
			EnsureOpen();
			return posePriors.TryGetValue(posePriorId, out PosePrior posePrior) ? posePrior : new PosePrior();
		}
	}

	/// <inheritdoc/>
	public override List<PosePrior> ReadAllPosePriors() => Locked(() => posePriors.Values.ToList());

	/// <inheritdoc/>
	public override RowMajorMatrix<float> ReadKeypointsBlob(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return keypoints.TryGetValue(imageId, out var blob) ? blob.Clone() : new RowMajorMatrix<float>(0, 0);
		}
	}

	/// <inheritdoc/>
	public override List<FeatureKeypoint> ReadKeypoints(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();

			// Convert straight from the stored blob: the keypoint list is the only copy.
			return keypoints.TryGetValue(imageId, out var blob) ? FeatureKeypointsFromBlob(blob) : [];
		}
	}

	/// <inheritdoc/>
	public override FeatureDescriptors ReadDescriptors(uint imageId)
	{
		lock (sync)
		{
			EnsureOpen();
			return descriptors.TryGetValue(imageId, out FeatureDescriptors? desc) ? desc.Clone() : new FeatureDescriptors();
		}
	}

	/// <inheritdoc/>
	public override RowMajorMatrix<uint> ReadMatchesBlob(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			if (!matches.TryGetValue(ImagePairToPairId(imageId1, imageId2), out var blob))
			{
				return new RowMajorMatrix<uint>(0, 2);
			}

			return ShouldSwapImagePair(imageId1, imageId2) ? SwappedFeatureMatchesBlob(blob) : blob.Clone();
		}
	}

	/// <inheritdoc/>
	public override List<FeatureMatch> ReadMatches(uint imageId1, uint imageId2) =>
		FeatureMatchesFromBlob(ReadMatchesBlob(imageId1, imageId2));

	/// <inheritdoc/>
	public override List<(ulong PairId, RowMajorMatrix<uint> Matches)> ReadAllMatchesBlob() =>
		Locked(() => matches.Where(entry => entry.Value.Rows > 0).Select(entry => (entry.Key, entry.Value.Clone())).ToList());

	/// <inheritdoc/>
	public override List<(ulong PairId, List<FeatureMatch> Matches)> ReadAllMatches() =>
		Locked(() => matches.Where(entry => entry.Value.Rows > 0)
			.Select(entry => (entry.Key, FeatureMatchesFromBlob(entry.Value))).ToList());

	/// <inheritdoc/>
	public override List<(ulong PairId, int NumMatches)> ReadNumMatches() =>
		Locked(() => matches.Where(entry => entry.Value.Rows > 0).Select(entry => (entry.Key, entry.Value.Rows)).ToList());

	/// <inheritdoc/>
	public override uint WriteRig(Rig rig, bool useRigId = false)
	{
		Check.That(rig.NumSensors > 0, "Rig must have at least one sensor");
		lock (sync)
		{
			EnsureOpen();
			if (useRigId)
			{
				Check.That(!rigs.ContainsKey(rig.RigId), "rig_id must be unique");
			}

			var row = RigRow.FromRig(rig);
			CheckRigConstraints(row, ignoreRigId: null);
			uint rigId = useRigId ? rig.RigId : NextAutoIncrementId(rigSequence);
			rigSequence = Math.Max(rigSequence, rigId);
			rigs.Add(rigId, row);
			return rigId;
		}
	}

	/// <inheritdoc/>
	public override uint WriteCamera(Camera camera, bool useCameraId = false)
	{
		lock (sync)
		{
			EnsureOpen();
			if (useCameraId)
			{
				Check.That(!cameras.ContainsKey(camera.CameraId), "camera_id must be unique");
			}

			uint cameraId = useCameraId ? camera.CameraId : NextAutoIncrementId(cameraSequence);
			cameraSequence = Math.Max(cameraSequence, cameraId);
			Camera stored = camera.Clone();
			stored.CameraId = cameraId;
			cameras.Add(cameraId, stored);
			return cameraId;
		}
	}

	/// <inheritdoc/>
	public override uint WriteFrame(Frame frame, bool useFrameId = false)
	{
		lock (sync)
		{
			EnsureOpen();
			if (useFrameId)
			{
				Check.That(!frames.ContainsKey(frame.FrameId), "frame_id must be unique");
			}

			CheckForeignKey(rigs.ContainsKey(frame.RigId));
			var row = new FrameRow(frame.RigId, [.. frame.DataIds]);
			CheckFrameDataConstraints(row.DataIds, ignoreFrameId: null);
			uint frameId = useFrameId ? frame.FrameId : NextAutoIncrementId(frameSequence);
			frameSequence = Math.Max(frameSequence, frameId);
			frames.Add(frameId, row);
			IndexFrameData(frameId, row);
			return frameId;
		}
	}

	/// <inheritdoc/>
	public override uint WriteImage(Image image, bool useImageId = false)
	{
		lock (sync)
		{
			EnsureOpen();
			if (image.HasFrameId)
			{
				Check.That(ReadFrame(image.FrameId).HasDataId(image.DataId));
			}

			if (useImageId)
			{
				Check.That(!images.ContainsKey(image.ImageId), "image_id must be unique");
			}

			uint imageId = useImageId ? image.ImageId : NextAutoIncrementId(imageSequence);
			CheckImageConstraints(imageId, image.Name, image.CameraId);
			imageSequence = Math.Max(imageSequence, imageId);
			images.Add(imageId, new ImageRow(image.Name, image.CameraId));
			imageIdsByName.Add(image.Name, imageId);
			return imageId;
		}
	}

	/// <inheritdoc/>
	public override uint WritePosePrior(PosePrior posePrior, bool usePosePriorId = false)
	{
		lock (sync)
		{
			EnsureOpen();
			if (usePosePriorId)
			{
				Check.That(!posePriors.ContainsKey(posePrior.PosePriorId), "pose_prior_id must be unique");
			}

			CheckPosePriorConstraints(posePrior.CorrDataId, ignorePosePriorId: null);

			// A plain INTEGER PRIMARY KEY: max(rowid) + 1, or 1 for an empty table.
			uint posePriorId = usePosePriorId
				? posePrior.PosePriorId
				: checked((uint)(maxPosePriorId + 1));
			PosePrior stored = posePrior;
			stored.PosePriorId = posePriorId;
			posePriors.Add(posePriorId, stored);
			maxPosePriorId = Math.Max(maxPosePriorId, posePriorId);
			posePriorIdsByData.Add(stored.CorrDataId, posePriorId);
			return posePriorId;
		}
	}

	/// <inheritdoc/>
	public override void WriteKeypoints(uint imageId, IReadOnlyList<FeatureKeypoint> keypoints) =>
		WriteKeypoints(imageId, FeatureKeypointsToBlob(keypoints));

	/// <inheritdoc/>
	public override void WriteKeypoints(uint imageId, RowMajorMatrix<float> blob)
	{
		lock (sync)
		{
			EnsureOpen();
			CheckUnique(!keypoints.ContainsKey(imageId), "keypoints.image_id");
			CheckForeignKey(images.ContainsKey(imageId));
			keypoints.Add(imageId, blob.Clone());
		}
	}

	/// <inheritdoc/>
	public override void WriteDescriptors(uint imageId, FeatureDescriptors descriptors)
	{
		lock (sync)
		{
			EnsureOpen();
			CheckUnique(!this.descriptors.ContainsKey(imageId), "descriptors.image_id");
			CheckForeignKey(images.ContainsKey(imageId));
			this.descriptors.Add(imageId, descriptors.Clone());
		}
	}

	/// <inheritdoc/>
	public override void WriteMatches(uint imageId1, uint imageId2, IReadOnlyList<FeatureMatch> matches) =>
		WriteMatches(imageId1, imageId2, FeatureKeypoints.MatchesToMatrix(matches));

	/// <inheritdoc/>
	public override void WriteMatches(uint imageId1, uint imageId2, RowMajorMatrix<uint> blob)
	{
		// FeatureMatchesBlob has two columns at compile time in COLMAP.
		Check.Eq(blob.Cols, 2);
		lock (sync)
		{
			EnsureOpen();
			ulong pairId = ImagePairToPairId(imageId1, imageId2);
			CheckUnique(!matches.ContainsKey(pairId), "matches.pair_id");
			matches.Add(pairId, ShouldSwapImagePair(imageId1, imageId2) ? SwappedFeatureMatchesBlob(blob) : blob.Clone());
		}
	}

	/// <inheritdoc/>
	public override void UpdateRig(Rig rig)
	{
		lock (sync)
		{
			EnsureOpen();
			var row = RigRow.FromRig(rig);
			if (!rigs.ContainsKey(rig.RigId))
			{
				// The UPDATE and DELETE match nothing; re-inserting the sensors violates the
				// rig_sensors foreign key.
				CheckForeignKey(row.Sensors.Count == 0);
				return;
			}

			CheckRigConstraints(row, ignoreRigId: rig.RigId);
			rigs[rig.RigId] = row;
		}
	}

	/// <inheritdoc/>
	public override void UpdateCamera(Camera camera)
	{
		lock (sync)
		{
			EnsureOpen();
			if (cameras.ContainsKey(camera.CameraId))
			{
				cameras[camera.CameraId] = camera.Clone();
			}
		}
	}

	/// <inheritdoc/>
	public override void UpdateFrame(Frame frame)
	{
		lock (sync)
		{
			EnsureOpen();
			var row = new FrameRow(frame.RigId, [.. frame.DataIds]);
			if (!frames.ContainsKey(frame.FrameId))
			{
				// As in UpdateRig: only re-inserting the frame data can fail (foreign key).
				CheckForeignKey(row.DataIds.Count == 0);
				return;
			}

			CheckForeignKey(rigs.ContainsKey(frame.RigId));
			CheckFrameDataConstraints(row.DataIds, ignoreFrameId: frame.FrameId);
			UnindexFrameData(frames[frame.FrameId]);
			frames[frame.FrameId] = row;
			IndexFrameData(frame.FrameId, row);
		}
	}

	/// <inheritdoc/>
	public override void UpdateImage(Image image)
	{
		lock (sync)
		{
			EnsureOpen();
			if (image.HasFrameId)
			{
				Check.That(ReadFrame(image.FrameId).HasDataId(image.DataId));
			}

			if (!images.ContainsKey(image.ImageId))
			{
				return;
			}

			CheckImageConstraints(image.ImageId, image.Name, image.CameraId);
			imageIdsByName.Remove(images[image.ImageId].Name);
			images[image.ImageId] = new ImageRow(image.Name, image.CameraId);
			imageIdsByName.Add(image.Name, image.ImageId);
		}
	}

	/// <inheritdoc/>
	public override void UpdatePosePrior(PosePrior posePrior)
	{
		lock (sync)
		{
			EnsureOpen();
			if (!posePriors.ContainsKey(posePrior.PosePriorId))
			{
				return;
			}

			CheckPosePriorConstraints(posePrior.CorrDataId, ignorePosePriorId: posePrior.PosePriorId);
			posePriorIdsByData.Remove(posePriors[posePrior.PosePriorId].CorrDataId);
			posePriors[posePrior.PosePriorId] = posePrior;
			posePriorIdsByData.Add(posePrior.CorrDataId, posePrior.PosePriorId);
		}
	}

	/// <inheritdoc/>
	public override void UpdateKeypoints(uint imageId, IReadOnlyList<FeatureKeypoint> keypoints) =>
		UpdateKeypoints(imageId, FeatureKeypointsToBlob(keypoints));

	/// <inheritdoc/>
	public override void UpdateKeypoints(uint imageId, RowMajorMatrix<float> blob)
	{
		lock (sync)
		{
			EnsureOpen();
			if (keypoints.ContainsKey(imageId))
			{
				keypoints[imageId] = blob.Clone();
			}
		}
	}

	/// <inheritdoc/>
	public override void UpdateDescriptors(uint imageId, FeatureDescriptors descriptors)
	{
		lock (sync)
		{
			EnsureOpen();
			if (this.descriptors.ContainsKey(imageId))
			{
				this.descriptors[imageId] = descriptors.Clone();
			}
		}
	}

	/// <inheritdoc/>
	public override void DeleteMatches(uint imageId1, uint imageId2)
	{
		lock (sync)
		{
			EnsureOpen();
			matches.Remove(ImagePairToPairId(imageId1, imageId2));
		}
	}

	/// <inheritdoc/>
	public override void ClearAllTables()
	{
		lock (sync)
		{
			ClearMatches();
			ClearTwoViewGeometries();
			ClearDescriptors();
			ClearKeypoints();
			ClearPosePriors();
			ClearFrames();
			ClearImages();
			ClearRigs();
			ClearCameras();
		}
	}

	/// <inheritdoc/>
	public override void ClearRigs()
	{
		lock (sync)
		{
			EnsureOpen();
			rigs.Clear();

			// frames.rig_id REFERENCES rigs ON DELETE CASCADE.
			frames.Clear();
			frameIdsByData.Clear();
		}
	}

	/// <inheritdoc/>
	public override void ClearCameras()
	{
		lock (sync)
		{
			EnsureOpen();

			// images.camera_id REFERENCES cameras without a cascade: the DELETE fails.
			CheckForeignKey(images.Count == 0);
			cameras.Clear();
		}
	}

	/// <inheritdoc/>
	public override void ClearFrames()
	{
		lock (sync)
		{
			EnsureOpen();
			frames.Clear();
			frameIdsByData.Clear();
		}
	}

	/// <inheritdoc/>
	public override void ClearImages()
	{
		lock (sync)
		{
			EnsureOpen();
			images.Clear();
			imageIdsByName.Clear();

			// keypoints/descriptors.image_id REFERENCES images ON DELETE CASCADE.
			keypoints.Clear();
			descriptors.Clear();
		}
	}

	/// <inheritdoc/>
	public override void ClearPosePriors() => Locked(() => { posePriors.Clear(); posePriorIdsByData.Clear(); maxPosePriorId = 0; return 0; });

	/// <inheritdoc/>
	public override void ClearDescriptors() => Locked(() => { descriptors.Clear(); return 0; });

	/// <inheritdoc/>
	public override void ClearKeypoints() => Locked(() => { keypoints.Clear(); return 0; });

	/// <inheritdoc/>
	public override void ClearMatches() => Locked(() => { matches.Clear(); return 0; });

	private void IndexFrameData(uint frameId, FrameRow row)
	{
		foreach (DataId dataId in row.DataIds)
		{
			frameIdsByData.Add((dataId.Id, dataId.SensorId.Type), frameId);
		}
	}

	private void UnindexFrameData(FrameRow row)
	{
		foreach (DataId dataId in row.DataIds)
		{
			frameIdsByData.Remove((dataId.Id, dataId.SensorId.Type));
		}
	}

	/// <summary>Nothing to do: every call is applied immediately (and serialized by a lock).</summary>
	public override void BeginTransaction() => Locked(() => 0);

	/// <summary>Nothing to do: every call is applied immediately (and serialized by a lock).</summary>
	public override void EndTransaction() => Locked(() => 0);
}
