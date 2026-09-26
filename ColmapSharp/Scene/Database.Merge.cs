// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Database.Merge: the implementation-independent functions of colmap/scene/database.cc -
// Database::Merge and LoadRandomDatabaseDescriptors - plus the keypoint/match blob
// conversions COLMAP keeps in database_sqlite.cc (FeatureKeypointsToBlob/FromBlob,
// FeatureMatchesToBlob/FromBlob, SwapFeatureMatchesBlob), which every store needs. The
// abstract API is Database.cs; the store is InMemoryDatabase.cs. Tests:
// ColmapSharp.Tests/Scene/DatabaseTests.cs.
//
// Tier A (exact). LoadRandomDatabaseDescriptors draws through RandomSampler (mt19937),
// like COLMAP.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

public abstract partial class Database
{
	/// <summary>
	/// Port of Database::Merge: writes the contents of both databases into
	/// <paramref name="mergedDatabase"/> under new ids (cameras, rigs, images, frames, pose
	/// priors, matches, two-view geometries, in that order). Image names must be unique
	/// across the two databases.
	/// </summary>
	public static void Merge(Database database1, Database database2, Database mergedDatabase)
	{
		// Merge the cameras.
		Dictionary<uint, uint> newCameraIds1 = MergeCameras(database1, mergedDatabase);
		Dictionary<uint, uint> newCameraIds2 = MergeCameras(database2, mergedDatabase);

		// Merge the rigs.
		Dictionary<uint, uint> newRigIds1 = MergeRigs(database1, mergedDatabase, newCameraIds1);
		Dictionary<uint, uint> newRigIds2 = MergeRigs(database2, mergedDatabase, newCameraIds2);

		// Merge the images.
		Dictionary<uint, uint> newImageIds1 = MergeImages(database1, mergedDatabase, newCameraIds1);
		Dictionary<uint, uint> newImageIds2 = MergeImages(database2, mergedDatabase, newCameraIds2);

		// Merge the frames.
		MergeFrames(database1, mergedDatabase, newRigIds1, newCameraIds1, newImageIds1);
		MergeFrames(database2, mergedDatabase, newRigIds2, newCameraIds2, newImageIds2);

		// Merge the pose priors.
		MergePosePriors(database1, mergedDatabase, newCameraIds1, newImageIds1);
		MergePosePriors(database2, mergedDatabase, newCameraIds2, newImageIds2);

		// Merge the matches.
		MergeMatches(database1, mergedDatabase, newImageIds1);
		MergeMatches(database2, mergedDatabase, newImageIds2);

		// Merge the two-view geometries.
		MergeTwoViewGeometries(database1, mergedDatabase, newImageIds1);
		MergeTwoViewGeometries(database2, mergedDatabase, newImageIds2);
	}

	/// <summary>
	/// Port of LoadRandomDatabaseDescriptors: up to <paramref name="maxNumDescriptors"/>
	/// descriptors drawn uniformly at random over all images (all of them if negative or at
	/// least the total), as float, in database order. All images must share one type.
	/// </summary>
	public static FeatureDescriptorsFloat LoadRandomDatabaseDescriptors(Database database, int maxNumDescriptors)
	{
		List<Image> images = database.ReadAllImages();
		long totalNumDescriptors = database.NumDescriptors();

		if (totalNumDescriptors == 0)
		{
			return new FeatureDescriptorsFloat();
		}

		var result = new FeatureDescriptorsFloat { Type = FeatureExtractorType.Undefined };

		var descriptorIdxs = new List<int>();
		if (maxNumDescriptors < 0 || maxNumDescriptors >= totalNumDescriptors)
		{
			for (int i = 0; i < totalNumDescriptors; ++i)
			{
				descriptorIdxs.Add(i);
			}
		}
		else
		{
			// Random subset of images in the database.
			Check.Le((long)maxNumDescriptors, totalNumDescriptors);
			var randomSampler = new RandomSampler(maxNumDescriptors);
			randomSampler.Initialize(checked((int)totalNumDescriptors));
			randomSampler.Sample(descriptorIdxs);
			descriptorIdxs.Sort();
		}

		int imageIdx = -1;
		FeatureDescriptorsFloat imageDescriptors = new();
		long imageDescriptorStart = 0;
		long imageDescriptorEnd = 0;
		void ReadNextImage()
		{
			++imageIdx;
			Check.Lt(imageIdx, images.Count);
			Image image = images[imageIdx];
			imageDescriptors = database.ReadDescriptors(image.ImageId).ToFloat();
			imageDescriptorStart = imageDescriptorEnd;
			imageDescriptorEnd = imageDescriptorStart + imageDescriptors.Data.Rows;
		}

		RowMajorMatrix<float>? data = null;
		int descriptorRow = 0;
		foreach (int descriptorIdx in descriptorIdxs)
		{
			while (descriptorIdx >= imageDescriptorEnd)
			{
				ReadNextImage();
			}

			// Check that all images have the same feature type.
			if (result.Type == FeatureExtractorType.Undefined)
			{
				Check.That(imageDescriptors.Type != FeatureExtractorType.Undefined);
				result.Type = imageDescriptors.Type;
				data = new RowMajorMatrix<float>(descriptorIdxs.Count, imageDescriptors.Data.Cols);
				result.Data = data;
			}
			else
			{
				Check.That(result.Type == imageDescriptors.Type, "All images must have the same feature type");
				Check.Eq(imageDescriptors.Data.Cols, data!.Cols, "All images must have the same descriptor dimensionality");
			}

			imageDescriptors.Data.Row((int)(descriptorIdx - imageDescriptorStart)).CopyTo(data!.Row(descriptorRow));
			++descriptorRow;
		}

		Check.Eq(descriptorRow, descriptorIdxs.Count);

		return result;
	}

	/// <summary>FeatureKeypointsToBlob: an N x 6 blob [x, y, a11, a12, a21, a22].</summary>
	protected static RowMajorMatrix<float> FeatureKeypointsToBlob(IReadOnlyList<FeatureKeypoint> keypoints)
	{
		const int NumCols = 6;
		var blob = new RowMajorMatrix<float>(keypoints.Count, NumCols);
		for (int i = 0; i < keypoints.Count; ++i)
		{
			FeatureKeypoint keypoint = keypoints[i];
			blob[i, 0] = keypoint.X;
			blob[i, 1] = keypoint.Y;
			blob[i, 2] = keypoint.A11;
			blob[i, 3] = keypoint.A12;
			blob[i, 4] = keypoint.A21;
			blob[i, 5] = keypoint.A22;
		}

		return blob;
	}

	/// <summary>
	/// FeatureKeypointsFromBlob: keypoints from an N x 2 (location), N x 4 (location, scale,
	/// orientation) or N x 6 (location, affine shape) blob.
	/// </summary>
	protected static List<FeatureKeypoint> FeatureKeypointsFromBlob(RowMajorMatrix<float> blob)
	{
		var keypoints = FeatureKeypoints.Create(blob.Rows);
		if (blob.Rows == 0)
		{
			return keypoints;
		}
		else if (blob.Cols == 2)
		{
			for (int i = 0; i < blob.Rows; ++i)
			{
				keypoints[i] = new FeatureKeypoint(blob[i, 0], blob[i, 1]);
			}
		}
		else if (blob.Cols == 4)
		{
			for (int i = 0; i < blob.Rows; ++i)
			{
				keypoints[i] = new FeatureKeypoint(blob[i, 0], blob[i, 1], blob[i, 2], blob[i, 3]);
			}
		}
		else if (blob.Cols == 6)
		{
			for (int i = 0; i < blob.Rows; ++i)
			{
				keypoints[i] = new FeatureKeypoint(blob[i, 0], blob[i, 1], blob[i, 2], blob[i, 3], blob[i, 4], blob[i, 5]);
			}
		}
		else
		{
			throw new InvalidOperationException("Keypoint format not supported");
		}

		return keypoints;
	}

	/// <summary>FeatureMatchesFromBlob: matches from an N x 2 blob.</summary>
	protected static List<FeatureMatch> FeatureMatchesFromBlob(RowMajorMatrix<uint> blob)
	{
		Check.Eq(blob.Cols, 2);
		var matches = new List<FeatureMatch>(blob.Rows);
		for (int i = 0; i < blob.Rows; ++i)
		{
			matches.Add(new FeatureMatch(blob[i, 0], blob[i, 1]));
		}

		return matches;
	}

	/// <summary>SwapFeatureMatchesBlob: a copy with the two columns exchanged.</summary>
	protected static RowMajorMatrix<uint> SwappedFeatureMatchesBlob(RowMajorMatrix<uint> blob)
	{
		RowMajorMatrix<uint> swapped = blob.Clone();
		for (int i = 0; i < swapped.Rows; ++i)
		{
			(swapped[i, 0], swapped[i, 1]) = (swapped[i, 1], swapped[i, 0]);
		}

		return swapped;
	}

	private static Dictionary<uint, uint> MergeCameras(Database source, Database merged)
	{
		var newCameraIds = new Dictionary<uint, uint>();
		foreach (Camera camera in source.ReadAllCameras())
		{
			newCameraIds.Add(camera.CameraId, merged.WriteCamera(camera));
		}

		return newCameraIds;
	}

	private static Dictionary<uint, uint> MergeRigs(Database source, Database merged, Dictionary<uint, uint> newCameraIds)
	{
		var newRigIds = new Dictionary<uint, uint>();
		foreach (Rig rig in source.ReadAllRigs())
		{
			newRigIds.Add(rig.RigId, merged.WriteRig(UpdateRig(rig, newCameraIds)));
		}

		return newRigIds;
	}

	private static Rig UpdateRig(Rig rig, Dictionary<uint, uint> newCameraIds)
	{
		if (rig.NumSensors == 0)
		{
			return rig;
		}

		var updatedRig = new Rig { RigId = rig.RigId };
		updatedRig.AddRefSensor(MapCameraSensor(rig.RefSensorId, newCameraIds));
		foreach (var (sensorId, sensorFromRig) in rig.NonRefSensors)
		{
			updatedRig.AddSensor(MapCameraSensor(sensorId, newCameraIds), sensorFromRig);
		}

		return updatedRig;
	}

	private static SensorId MapCameraSensor(SensorId sensorId, Dictionary<uint, uint> newCameraIds) =>
		sensorId.Type == SensorType.Camera ? sensorId with { Id = newCameraIds[sensorId.Id] } : sensorId;

	private static Dictionary<uint, uint> MergeImages(Database source, Database merged, Dictionary<uint, uint> newCameraIds)
	{
		var newImageIds = new Dictionary<uint, uint>();
		foreach (Image image in source.ReadAllImages())
		{
			image.SetCameraId(newCameraIds[image.CameraId]);
			image.SetFrameId(InvalidFrameId);
			Check.That(
				!merged.ExistsImageWithName(image.Name),
				$"The two databases must not contain images with the same name, but there are images with name {image.Name} in both databases");
			uint newImageId = merged.WriteImage(image);
			newImageIds.Add(image.ImageId, newImageId);
			List<FeatureKeypoint> keypoints = source.ReadKeypoints(image.ImageId);
			FeatureDescriptors descriptors = source.ReadDescriptors(image.ImageId);
			merged.WriteKeypoints(newImageId, keypoints);
			merged.WriteDescriptors(newImageId, descriptors);
		}

		return newImageIds;
	}

	private static void MergeFrames(
		Database source,
		Database merged,
		Dictionary<uint, uint> newRigIds,
		Dictionary<uint, uint> newCameraIds,
		Dictionary<uint, uint> newImageIds)
	{
		foreach (Frame frame in source.ReadAllFrames())
		{
			var updatedFrame = frame.Clone();
			updatedFrame.SetRigId(newRigIds[frame.RigId]);
			updatedFrame.ClearDataIds();
			foreach (DataId dataId in frame.DataIds)
			{
				updatedFrame.AddDataId(MapCameraData(dataId, newCameraIds, newImageIds));
			}

			merged.WriteFrame(updatedFrame);
		}
	}

	private static void MergePosePriors(
		Database source,
		Database merged,
		Dictionary<uint, uint> newCameraIds,
		Dictionary<uint, uint> newImageIds)
	{
		foreach (PosePrior posePrior in source.ReadAllPosePriors())
		{
			PosePrior updatedPosePrior = posePrior; // A value copy (PosePrior is a struct).
			updatedPosePrior.CorrDataId = MapCameraData(posePrior.CorrDataId, newCameraIds, newImageIds);
			merged.WritePosePrior(updatedPosePrior);
		}
	}

	private static DataId MapCameraData(DataId dataId, Dictionary<uint, uint> newCameraIds, Dictionary<uint, uint> newImageIds)
	{
		if (dataId.SensorId.Type != SensorType.Camera)
		{
			throw new InvalidOperationException($"Data type not supported: {dataId.SensorId.Type.ToColmapString()}");
		}

		return new DataId(
			new SensorId(SensorType.Camera, newCameraIds[dataId.SensorId.Id]),
			newImageIds[checked((uint)dataId.Id)]);
	}

	private static void MergeMatches(Database source, Database merged, Dictionary<uint, uint> newImageIds)
	{
		foreach (var (pairId, matches) in source.ReadAllMatches())
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			merged.WriteMatches(newImageIds[imageId1], newImageIds[imageId2], matches);
		}
	}

	private static void MergeTwoViewGeometries(Database source, Database merged, Dictionary<uint, uint> newImageIds)
	{
		foreach (var (pairId, twoViewGeometry) in source.ReadTwoViewGeometries())
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			merged.WriteTwoViewGeometry(newImageIds[imageId1], newImageIds[imageId2], twoViewGeometry);
		}
	}
}
