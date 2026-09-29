// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction, second half of colmap/scene/reconstruction.cc: consistency validation,
// tear-down, normalization and similarity transforms, cropping, lookups by name, summary
// statistics, reprojection error updates, image directory creation and operator<<. Storage
// and the add/delete/register operations are in Reconstruction.cs, whose header describes
// the design.
//
// Tiers: IsValid, TearDown, Crop's bookkeeping, the lookups and the counts are Tier A.
// Normalize/ComputeCentroid are Tier B through Geometry/Normalization.cs's centroid sum
// (divergence 15) and Transform is Tier B through Sim3d's rotation of
// vectors (entry 6). UpdatePoint3DErrors follows the camera models' tier.

using System.Globalization;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class Reconstruction
{
	/// <summary>
	/// Whether the reconstruction is internally consistent: rig, frame, image and camera
	/// references and ids agree, and the 2D-3D links match in both directions. Like COLMAP,
	/// logs a warning (Util/Log.cs) naming the first inconsistency.
	/// </summary>
	public bool IsValid()
	{
		// Check object associations: rig-frame-image-camera references and pointers.
		foreach (var (rigId, rig) in _rigs)
		{
			foreach (SensorId sensorId in rig.SensorIds())
			{
				// Only camera sensors are currently supported.
				if (sensorId.Type == SensorType.Camera && !ExistsCamera(sensorId.Id))
				{
					return Invalid($"Rig {rigId} has sensor (camera) {sensorId.Id} which does not exist");
				}
			}
		}

		foreach (var (frameId, frame) in _frames)
		{
			if (!frame.HasRigId)
			{
				return Invalid($"Frame {frameId} has no rig_id");
			}

			if (!ExistsRig(frame.RigId))
			{
				return Invalid($"Frame {frameId} references non-existent rig {frame.RigId}");
			}

			if (!frame.HasRigPtr)
			{
				return Invalid($"Frame {frameId} has no rig pointer");
			}

			if (!ReferenceEquals(frame.RigPtr, _rigs[frame.RigId]))
			{
				return Invalid($"Frame {frameId} rig pointer does not match rig_id");
			}

			foreach (DataId dataId in frame.DataIds)
			{
				if (!frame.RigPtr!.HasSensor(dataId.SensorId))
				{
					return Invalid($"Frame {frameId} has data with sensor_id {dataId.SensorId.Id} that does not exist in rig {frame.RigId}");
				}

				// Only camera data is currently supported.
				if (dataId.SensorId.Type == SensorType.Camera && !ExistsImage((uint)dataId.Id))
				{
					return Invalid($"Frame {frameId} references image {dataId.Id} which does not exist");
				}
			}
		}

		foreach (var (imageId, image) in _images)
		{
			if (!image.HasCameraId)
			{
				return Invalid($"Image {imageId} has no camera_id");
			}

			if (!ExistsCamera(image.CameraId))
			{
				return Invalid($"Image {imageId} references non-existent camera {image.CameraId}");
			}

			if (!image.HasCameraPtr)
			{
				return Invalid($"Image {imageId} has no camera pointer");
			}

			if (!ReferenceEquals(image.CameraPtr, _cameras[image.CameraId]))
			{
				return Invalid($"Image {imageId} camera pointer does not match camera_id");
			}

			if (!image.HasFrameId)
			{
				return Invalid($"Image {imageId} has no frame_id");
			}

			if (!ExistsFrame(image.FrameId))
			{
				return Invalid($"Image {imageId} references non-existent frame {image.FrameId}");
			}

			if (!image.HasFramePtr)
			{
				return Invalid($"Image {imageId} has no frame pointer");
			}

			if (!ReferenceEquals(image.FramePtr, _frames[image.FrameId]))
			{
				return Invalid($"Image {imageId} frame pointer does not match frame_id");
			}

			if (!image.FramePtr!.HasDataId(image.DataId))
			{
				return Invalid($"Image {imageId} data_id not found in frame {image.FrameId}");
			}

			// Check 2D-3D associations: point2D -> point3D direction.
			uint actualNumPoints3D = 0;
			uint point2DIdx = 0;
			foreach (Point2D point2D in image.Points2D)
			{
				if (point2D.HasPoint3D)
				{
					++actualNumPoints3D;
					if (!ExistsPoint3D(point2D.Point3DId))
					{
						return Invalid($"Image {imageId} point2D {point2DIdx} references non-existent point3D {point2D.Point3DId}");
					}
				}

				++point2DIdx;
			}

			if (image.NumPoints3D != actualNumPoints3D)
			{
				return Invalid($"Image {imageId} NumPoints3D()={image.NumPoints3D} does not match actual count={actualNumPoints3D}");
			}
		}

		// Check 2D-3D associations: point3D -> point2D direction.
		foreach (var (point3DId, point3D) in _points3D)
		{
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				if (!_images.TryGetValue(trackEl.ImageId, out Image? image))
				{
					return Invalid($"Point3D {point3DId} track references image {trackEl.ImageId} which does not exist");
				}

				if (trackEl.Point2DIdx >= image.NumPoints2D)
				{
					return Invalid(
						$"Point3D {point3DId} track references point2D {trackEl.Point2DIdx} in image {trackEl.ImageId} "
						+ $"which only has {image.NumPoints2D} points");
				}

				Point2D point2D = image.Point2DAt(trackEl.Point2DIdx);
				if (!point2D.HasPoint3D)
				{
					return Invalid(
						$"Point3D {point3DId} track references point2D {trackEl.Point2DIdx} in image {trackEl.ImageId} "
						+ "which has no point3D set");
				}

				if (point2D.Point3DId != point3DId)
				{
					return Invalid(
						$"Point3D {point3DId} track references point2D {trackEl.Point2DIdx} in image {trackEl.ImageId} "
						+ $"which points to different point3D {point2D.Point3DId}");
				}
			}
		}

		// Check registered frames exist and have poses.
		foreach (uint frameId in _regFrameIds)
		{
			if (!ExistsFrame(frameId))
			{
				return Invalid($"Registered frame {frameId} does not exist");
			}

			if (!Frame(frameId).HasPose)
			{
				return Invalid($"Registered frame {frameId} has no pose");
			}
		}

		return true;
	}

	// IsValid's LOG(WARNING) << message; return false.
	private static bool Invalid(string message)
	{
		Log.Warning(message);
		return false;
	}

	/// <summary>
	/// Finalizes the reconstruction after it has finished: removes all frames that are not
	/// registered with their images, the rigs no registered frame uses with their cameras,
	/// and compresses the tracks. A finalized scene cannot be used for reconstruction.
	/// </summary>
	public void TearDown()
	{
		// Remove all non-registered frames/images.
		var keepRigIds = new HashSet<uint>();
		foreach (var (frameId, frame) in _frames.ToList())
		{
			if (!frame.HasPose)
			{
				foreach (DataId dataId in frame.ImageIds())
				{
					_images.Remove((uint)dataId.Id);
				}

				_frames.Remove(frameId);
			}
			else
			{
				keepRigIds.Add(frame.RigId);
			}
		}

		// Remove all unused rigs and corresponding sensors.
		foreach (var (rigId, rig) in _rigs.ToList())
		{
			if (!keepRigIds.Contains(rigId))
			{
				foreach (SensorId sensorId in rig.SensorIds())
				{
					if (sensorId.Type == SensorType.Camera)
					{
						_cameras.Remove(sensorId.Id);
					}
				}

				_rigs.Remove(rigId);
			}
		}

		// Compress tracks.
		foreach (Point3D point3D in _points3D.Values)
		{
			point3D.Track.Compress();
		}
	}

	/// <summary>
	/// Normalizes the scene by scaling and translation to improve numerical stability of
	/// algorithms. Translates the scene such that the mean of the camera centers (or point
	/// locations) is at the origin, and, unless <paramref name="fixedScale"/>, scales it
	/// such that the bounding box of the camera centers (or points) between the given
	/// percentiles has a diagonal of <paramref name="extent"/>. Returns the applied
	/// new_from_old_world transform (identity with fewer than two elements).
	/// </summary>
	public Sim3d Normalize(
		bool fixedScale = false,
		double extent = 10.0,
		double minPercentile = 0.1,
		double maxPercentile = 0.9,
		bool useImages = true)
	{
		Check.Gt(extent, 0);

		if ((useImages && NumRegFrames < 2) || (!useImages && _points3D.Count < 2))
		{
			return new Sim3d();
		}

		var (bbox, centroid) = ComputeBBoxAndCentroid(minPercentile, maxPercentile, useImages);

		// Calculate scale and translation, such that translation is applied before scaling.
		double scale = 1.0;
		if (!fixedScale)
		{
			double oldExtent = bbox.Diagonal().Norm;
			if (oldExtent >= LinearAlgebraConstants.MachineEpsilon)
			{
				scale = extent / oldExtent;
			}
		}

		var tform = new Sim3d(scale, Quaterniond.Identity, -scale * centroid);
		Transform(tform);
		return tform;
	}

	/// <summary>The centroid of the camera centers or 3D points within the percentile range.</summary>
	public Vector3d ComputeCentroid(double minPercentile = 0.1, double maxPercentile = 0.9, bool useImages = false) =>
		ComputeBBoxAndCentroid(minPercentile, maxPercentile, useImages).Centroid;

	/// <summary>The bounding box of the camera centers or 3D points within the percentile range.</summary>
	public AlignedBox3d ComputeBoundingBox(double minPercentile = 0.0, double maxPercentile = 1.0, bool useImages = false) =>
		ComputeBBoxAndCentroid(minPercentile, maxPercentile, useImages).Bbox;

	/// <summary>
	/// Applies the 3D similarity transformation to all frames, rig extrinsics and points.
	/// </summary>
	public void Transform(Sim3d newFromOldWorld)
	{
		foreach (Rig rig in _rigs.Values)
		{
			foreach (var (sensorId, sensorFromRig) in rig.NonRefSensors.ToList())
			{
				if (sensorFromRig is Rigid3d value)
				{
					rig.SetSensorFromRig(sensorId, value with { Translation = value.Translation * newFromOldWorld.Scale });
				}
			}
		}

		foreach (Frame frame in _frames.Values)
		{
			if (frame.HasPose)
			{
				frame.SetRigFromWorld(Pose.TransformCameraWorld(newFromOldWorld, frame.RigFromWorld()));
			}
		}

		foreach (Point3D point3D in _points3D.Values)
		{
			point3D.Xyz = newFromOldWorld * point3D.Xyz;
		}
	}

	/// <summary>
	/// A cropped reconstruction holding the 3D points inside <paramref name="bbox"/> (with
	/// new ids, in ascending order of the old ones). All cameras, rigs, frames and images
	/// are kept, but only the frames observing a kept point stay registered.
	/// </summary>
	public Reconstruction Crop(AlignedBox3d bbox)
	{
		var croppedReconstruction = new Reconstruction();
		foreach (Camera camera in _cameras.Values)
		{
			croppedReconstruction.AddCamera(camera);
		}

		foreach (Rig rig in _rigs.Values)
		{
			croppedReconstruction.AddRig(rig);
		}

		foreach (Frame frame in _frames.Values)
		{
			Frame frameCopy = frame.Clone();
			frameCopy.ResetRigPtr();
			croppedReconstruction.AddFrame(frameCopy);
		}

		foreach (Image image in _images.Values)
		{
			Image imageCopy = image.Clone();
			imageCopy.ResetCameraPtr();
			imageCopy.ResetFramePtr();
			uint numPoints2D = imageCopy.NumPoints2D;
			for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
			{
				imageCopy.ResetPoint3DForPoint2D(point2DIdx);
			}

			croppedReconstruction.AddImage(imageCopy);
		}

		var croppedFrameIds = new HashSet<uint>();
		foreach (Point3D point3D in _points3D.Values)
		{
			if (bbox.Contains(point3D.Xyz))
			{
				foreach (TrackElement trackEl in point3D.Track.Elements)
				{
					croppedFrameIds.Add(Image(trackEl.ImageId).FrameId);
				}

				croppedReconstruction.AddPoint3D(point3D.Xyz, point3D.Track, point3D.Color);
			}
		}

		foreach (uint frameId in croppedReconstruction.Frames.Keys)
		{
			if (!croppedFrameIds.Contains(frameId))
			{
				croppedReconstruction.DeRegisterFrame(frameId);
			}
		}

		return croppedReconstruction;
	}

	/// <summary>
	/// The image with the given name (the one with the smallest id if several share it), or
	/// null. Note that this uses linear search.
	/// </summary>
	public Image? FindImageWithName(string name)
	{
		foreach (Image image in _images.Values)
		{
			if (image.Name == name)
			{
				return image;
			}
		}

		return null;
	}

	/// <summary>
	/// The (this, other) image id pairs of the images registered here whose name belongs to
	/// an image with a pose in <paramref name="other"/>, in this reconstruction's
	/// registration order.
	/// </summary>
	public List<(uint ImageId, uint OtherImageId)> FindCommonRegImageIds(Reconstruction other)
	{
		var commonRegImageIds = new List<(uint, uint)>();
		foreach (uint frameId in _regFrameIds)
		{
			foreach (DataId dataId in Frame(frameId).ImageIds())
			{
				Image image = Image((uint)dataId.Id);
				Image? otherImage = other.FindImageWithName(image.Name);
				if (otherImage is not null && otherImage.FramePtr.HasPose)
				{
					commonRegImageIds.Add((image.ImageId, otherImage.ImageId));
				}
			}
		}

		return commonRegImageIds;
	}

	/// <summary>The number of observations (triangulated 2D points) of the registered images.</summary>
	public long ComputeNumObservations()
	{
		long numObs = 0;
		foreach (uint imageId in RegImageIds())
		{
			numObs += Image(imageId).NumPoints3D;
		}

		return numObs;
	}

	/// <summary>The mean track length: observations per 3D point (0 without points).</summary>
	public double ComputeMeanTrackLength()
	{
		if (_points3D.Count == 0)
		{
			return 0.0;
		}

		return ComputeNumObservations() / (double)_points3D.Count;
	}

	/// <summary>The mean number of observations per registered image (0 without any).</summary>
	public double ComputeMeanObservationsPerRegImage()
	{
		if (NumRegImages == 0)
		{
			return 0.0;
		}

		return ComputeNumObservations() / (double)NumRegImages;
	}

	/// <summary>The mean of the 3D points' reprojection errors, over the points that have one.</summary>
	public double ComputeMeanReprojectionError()
	{
		double errorSum = 0.0;
		long numValidErrors = 0;
		foreach (Point3D point3D in _points3D.Values)
		{
			if (point3D.HasError)
			{
				errorSum += point3D.Error;
				numValidErrors += 1;
			}
		}

		if (numValidErrors == 0)
		{
			return 0.0;
		}

		return errorSum / numValidErrors;
	}

	/// <summary>Updates the mean reprojection errors of all 3D points (0 for an empty track).</summary>
	public void UpdatePoint3DErrors()
	{
		foreach (Point3D point3D in _points3D.Values)
		{
			if (point3D.Track.Length == 0)
			{
				point3D.Error = 0;
				continue;
			}

			double error = 0;
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				Image image = Image(trackEl.ImageId);
				Point2D point2D = image.Point2DAt(trackEl.Point2DIdx);
				error += Math.Sqrt(Projection.CalculateSquaredReprojectionError(
					point2D.Xy, point3D.Xyz, image.CamFromWorld(), image.CameraPtr));
			}

			point3D.Error = error / point3D.Track.Length;
		}
	}

	/// <summary>
	/// Creates the sub-directories the image names imply ("a/b/c.jpg" creates "a" and
	/// "a/b") under <paramref name="path"/>.
	/// </summary>
	public void CreateImageDirs(string path)
	{
		var imageDirs = new SortedSet<string>(StringComparer.Ordinal);
		foreach (Image image in _images.Values)
		{
			string[] nameSplit = image.Name.Split('/');
			if (nameSplit.Length > 1)
			{
				string dir = path;
				for (int i = 0; i < nameSplit.Length - 1; ++i)
				{
					dir = System.IO.Path.Combine(dir, nameSplit[i]);
					imageDirs.Add(dir);
				}
			}
		}

		foreach (string dir in imageDirs)
		{
			Directory.CreateDirectory(dir);
		}
	}

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g. "Reconstruction(num_rigs=1, num_cameras=1, num_frames=2,
	/// num_reg_frames=2, num_images=2, num_points3D=3)".
	/// </summary>
	public override string ToString() => string.Create(
		CultureInfo.InvariantCulture,
		$"Reconstruction(num_rigs={NumRigs}, num_cameras={NumCameras}, num_frames={NumFrames}, num_reg_frames={NumRegFrames}, num_images={NumImages}, num_points3D={NumPoints3D})");

	private (AlignedBox3d Bbox, Vector3d Centroid) ComputeBBoxAndCentroid(double minPercentile, double maxPercentile, bool useImages)
	{
		int numElements = useImages ? NumRegFrames : _points3D.Count;
		if (numElements == 0)
		{
			return (new AlignedBox3d(Vector3d.Zero, Vector3d.Zero), Vector3d.Zero);
		}

		// Coordinates of image centers or point locations.
		var coordsX = new List<double>();
		var coordsY = new List<double>();
		var coordsZ = new List<double>();
		if (useImages)
		{
			foreach (uint frameId in _regFrameIds)
			{
				foreach (DataId dataId in Frame(frameId).ImageIds())
				{
					Vector3d projCenter = Image((uint)dataId.Id).ProjectionCenter();
					coordsX.Add(projCenter.X);
					coordsY.Add(projCenter.Y);
					coordsZ.Add(projCenter.Z);
				}
			}
		}
		else
		{
			foreach (Point3D point3D in _points3D.Values)
			{
				coordsX.Add(point3D.Xyz.X);
				coordsY.Add(point3D.Xyz.Y);
				coordsZ.Add(point3D.Xyz.Z);
			}
		}

		return Normalization.ComputeBoundingBoxAndCentroid(
			minPercentile, maxPercentile, coordsX, coordsY, coordsZ);
	}
}
