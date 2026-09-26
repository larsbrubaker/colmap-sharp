// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction.Database: the two parts of colmap/scene/reconstruction.cc that talk to the
// feature database: Load (fill the model from a DatabaseCache, DatabaseCache.cs) and
// TranscribeImageIdsToDatabase (renumber the images to a database's ids, matched by name).
// The rest of the class is in Reconstruction.cs and Reconstruction.Queries.cs. Tests:
// ColmapSharp.Tests/Scene/ReconstructionTests.Database.cs (reconstruction_test.cc).
//
// Tier A (exact). Both walk the cache's / model's IdMaps in ascending id order
// (docs/CPP_DIVERGENCES.md, entries 21 and 33); neither result depends on that order,
// except which offending image a failing check reports.

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class Reconstruction
{
	/// <summary>
	/// Port of Reconstruction::Load: adds the cameras, rigs, frames and images of the cache
	/// that are not in the model yet. Existing ones must agree with the cache (same camera
	/// model and size, rig sensors, frame data, image name and point count); an existing
	/// image without points takes the cache's points.
	/// </summary>
	public void Load(DatabaseCache databaseCache)
	{
		foreach ((uint cameraId, Camera camera) in databaseCache.Cameras)
		{
			if (ExistsCamera(cameraId))
			{
				Camera existingCamera = Camera(cameraId);
				Check.That(existingCamera.ModelId == camera.ModelId);
				Check.Eq(existingCamera.Width, camera.Width);
				Check.Eq(existingCamera.Height, camera.Height);
			}
			else
			{
				AddCamera(camera);
			}
		}

		foreach ((uint rigId, Rig rig) in databaseCache.Rigs)
		{
			if (ExistsRig(rigId))
			{
				Rig existingRig = Rig(rigId);
				Check.That(existingRig.RefSensorId == rig.RefSensorId);
				Check.That(existingRig.SensorIds().SetEquals(rig.SensorIds()));
			}
			else
			{
				AddRig(rig);
			}
		}

		foreach ((uint frameId, Frame frame) in databaseCache.Frames)
		{
			if (ExistsFrame(frameId))
			{
				Frame existingFrame = Frame(frameId);
				Check.That(existingFrame.RigId == frame.RigId);
				Check.That(existingFrame.DataIds.SetEquals(frame.DataIds));
			}
			else
			{
				AddFrame(frame);
			}
		}

		foreach ((uint imageId, Image image) in databaseCache.Images)
		{
			if (ExistsImage(imageId))
			{
				Image existingImage = Image(imageId);
				Check.That(existingImage.Name == image.Name);
				if (existingImage.NumPoints2D == 0)
				{
					existingImage.SetPoints2D(image.Points2D);
				}
				else
				{
					Check.Eq(image.NumPoints2D, existingImage.NumPoints2D);
				}
			}
			else
			{
				AddImage(image);
			}
		}
	}

	/// <summary>
	/// Port of Reconstruction::TranscribeImageIdsToDatabase: gives every image the id of the
	/// database image with the same name and rewrites the frames' data ids and the tracks to
	/// match. Throws if an image name is not in the database.
	/// </summary>
	public void TranscribeImageIdsToDatabase(Database database)
	{
		var oldToNewImageIds = new Dictionary<uint, uint>(NumImages);
		var newImages = new List<Image>(NumImages);
		var newImageIds = new HashSet<uint>(NumImages);

		foreach (Image image in _images.Values)
		{
			Image databaseImage = database.ReadImageWithName(image.Name)
				?? throw new InvalidOperationException($"Image with name {image.Name} does not exist in database");
			oldToNewImageIds.Add(image.ImageId, databaseImage.ImageId);
			image.ImageId = databaseImage.ImageId;
			Check.That(newImageIds.Add(databaseImage.ImageId));
			newImages.Add(image);
		}

		_images.Clear();
		foreach (Image image in newImages)
		{
			_images.TryAdd(image.ImageId, image);
		}

		// Transcribe frame data.
		foreach (Frame frame in _frames.Values)
		{
			var newDataIds = new List<DataId>(frame.NumDataIds);
			foreach (DataId dataId in frame.DataIds)
			{
				newDataIds.Add(dataId.SensorId.Type == SensorType.Camera
					? new DataId(dataId.SensorId, oldToNewImageIds[(uint)dataId.Id])
					: dataId);
			}

			// We finalize the data ids, otherwise some internal bookkeeping (e.g., counting
			// reg_image_ids_) will be incorrect.
			frame.ReplaceFinalizedDataIds(newDataIds);
		}

		// Transcribe point tracks.
		foreach (Point3D point3D in _points3D.Values)
		{
			List<TrackElement> elements = point3D.Track.Elements;
			for (int i = 0; i < elements.Count; ++i)
			{
				elements[i] = new TrackElement(oldToNewImageIds[elements[i].ImageId], elements[i].Point2DIdx);
			}
		}
	}
}
