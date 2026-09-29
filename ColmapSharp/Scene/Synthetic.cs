// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Synthetic: port of colmap/scene/synthetic.cc, the synthetic dataset generator most of
// COLMAP's downstream tests build their scenes with. SynthesizeDataset puts 3D points on the
// unit sphere, rigs of cameras on a sphere of radius 5 looking at the origin, projects the
// points into every image, and (with a database) writes keypoints, descriptors, matches and
// two-view geometries. SynthesizeNoise perturbs it. Options: SyntheticOptions.cs. The match
// synthesis is Synthetic.Matches.cs, the image rendering Synthetic.Images.cs.
//
// Randomness: every draw goes through RandomUtils (COLMAP's thread-local PRNG) or a local
// Mt19937, in COLMAP's order, so a seeded run reproduces pycolmap's draw sequence. Where C++
// builds a value from several draws in one expression (Eigen::Vector3d(RandomGaussian(...),
// RandomGaussian(...), ...)), the draws happen left to right, as clang (the macOS pycolmap
// wheel) evaluates constructor arguments.
//
// Iteration order: COLMAP iterates Reconstruction::Points3D()/Images() and a local
// NodeHashMap of pairs, whose order is abseil's hash order. Here those are ascending id
// order (IdMap, divergences 21 and 31). The draw sequence is the same;
// only which point or pair receives which draws, and the order of each image's 2D points
// before the shuffle, differ.
//
// Tests: ColmapSharp.Tests/Scene/SyntheticTests.cs (synthetic_test.cc 1:1) and
// SyntheticOracleTests.cs (pycolmap fixture). Tier A for ids, counts and poses; Tier B for
// values that pass through Gaussian draws (divergence 1).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap/scene/synthetic.h: synthetic reconstructions and databases for tests.
/// </summary>
public static partial class Synthetic
{
	private static readonly GPSTransform GpsTransform = new();
	private const double Lat0 = 47.37851943807808;
	private const double Lon0 = 8.549099927632087;
	private const double Alt0 = 451.5;

	/// <summary>
	/// Adds a synthetic dataset to <paramref name="reconstruction"/> and, if given, to
	/// <paramref name="database"/> (whose ids are then used). Port of colmap::SynthesizeDataset.
	/// </summary>
	public static void SynthesizeDataset(SyntheticDatasetOptions options, Reconstruction reconstruction, Database? database = null)
	{
		Check.Gt(options.NumRigs, 0);
		Check.Gt(options.NumCamerasPerRig, 0);
		Check.Gt(options.NumFramesPerRig, 0);
		Check.Ge(options.NumPoints3D, 0);
		Check.That(options.TrackLength == -1 || options.TrackLength >= 2);
		if (options.TrackLength > 0)
		{
			int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
			if (options.TrackLength > numImages)
			{
				Log.Warning(
					$"track_length ({options.TrackLength}) exceeds number of images ({numImages}), skipping observation pruning.");
			}
		}

		Check.Ne((int)options.FeatureType, (int)FeatureExtractorType.Undefined);
		Check.Ge(options.NumPoints2DWithoutPoint3D, 0);
		Check.Ge(options.SensorFromRigTranslationStddev, 0.0);
		Check.Ge(options.SensorFromRigRotationStddev, 0.0);
		Check.Ge(options.MatchSparsity, 0.0);
		Check.Le(options.MatchSparsity, 1.0);
		Check.That(options.ImageExtension.Length > 0);
		Check.That(options.ImageExtension[0] == '.');

		if (RandomUtils.Prng == null)
		{
			RandomUtils.SetPRNGSeed();
		}

		// Synthesize 3D points on unit sphere centered at origin.
		var newPoints3DIds = new HashSet<ulong>(options.NumPoints3D);
		for (int point3DIdx = 0; point3DIdx < options.NumPoints3D; ++point3DIdx)
		{
			newPoints3DIds.Add(reconstruction.AddPoint3D(RandomEigen.RandomEigenVector3d().Normalized(), new Track()));
		}

		int totalNumImages = database == null ? 0 : (int)database.NumImages();
		int totalNumDescriptors = database == null ? 0 : (int)database.NumDescriptors();

		int descDim = options.FeatureType switch
		{
			FeatureExtractorType.Sift => 128,
			_ => throw new ArgumentException($"Invalid FeatureExtractorType specified: {(int)options.FeatureType}"),
		};

		for (int rigIdx = 0; rigIdx < options.NumRigs; ++rigIdx)
		{
			var rig = new Rig();
			var cameraSensorIds = new List<SensorId>(options.NumCamerasPerRig);
			for (int cameraIdx = 0; cameraIdx < options.NumCamerasPerRig; ++cameraIdx)
			{
				var camera = new Camera
				{
					Width = options.CameraWidth,
					Height = options.CameraHeight,
					ModelId = options.CameraModelId,
					Params = (double[])options.CameraParams.Clone(),
				};
				Check.That(camera.VerifyParams());
				camera.HasPriorFocalLength = options.CameraHasPriorFocalLength;
				camera.CameraId = database == null
					? (uint)(rigIdx * options.NumCamerasPerRig + cameraIdx + 1)
					: database.WriteCamera(camera);
				reconstruction.AddCamera(camera);

				if (rig.NumSensors == 0)
				{
					rig.AddRefSensor(camera.SensorId);
				}
				else
				{
					rig.AddSensor(camera.SensorId, SynthesizeSensorFromRig(options));
				}

				cameraSensorIds.Add(camera.SensorId);
			}

			rig.RigId = database == null ? (uint)(rigIdx + 1) : database.WriteRig(rig);
			reconstruction.AddRig(rig);

			for (int frameIdx = 0; frameIdx < options.NumFramesPerRig; ++frameIdx)
			{
				SynthesizeFrame(options, reconstruction, database, rig, cameraSensorIds, rigIdx, frameIdx, descDim, newPoints3DIds, ref totalNumImages, ref totalNumDescriptors);
			}
		}

		if (database != null)
		{
			switch (options.MatchConfig)
			{
				case SyntheticMatchConfig.Exhaustive:
					SynthesizeExhaustiveMatches(options.InlierMatchRatio, options.TwoViewGeometryHasRelativePose, reconstruction, database);
					break;
				case SyntheticMatchConfig.Chained:
					SynthesizeChainedMatches(options.InlierMatchRatio, options.TwoViewGeometryHasRelativePose, reconstruction, database);
					break;
				case SyntheticMatchConfig.Sparse:
					SynthesizeSparseMatches(options.InlierMatchRatio, options.TwoViewGeometryHasRelativePose, options.MatchSparsity, reconstruction, database);
					break;
				default:
					throw new ArgumentException("Invalid MatchConfig specified");
			}
		}

		if (options.TrackLength > 0)
		{
			PruneTracks(options.TrackLength, reconstruction);
		}

		reconstruction.UpdatePoint3DErrors();
	}

	// The non-reference sensor's pose in its rig: a random rotation around the z-axis (which
	// keeps the 2D points in front of the camera) and a random offset.
	private static Rigid3d SynthesizeSensorFromRig(SyntheticDatasetOptions options)
	{
		Quaterniond rotation = Quaterniond.Identity;
		Vector3d translation = Vector3d.Zero;
		if (options.SensorFromRigRotationStddev > 0)
		{
			// Generate a random rotation around the Z-axis.
			// This is to avoid 2D points fall behind the camera.
			double angle = Math.Clamp(RandomUtils.RandomGaussian(0.0, options.SensorFromRigRotationStddev), -180.0, 180.0);
			rotation = Quaterniond.FromAngleAxis(new AngleAxisd(MathUtils.DegToRad(angle), new Vector3d(0, 0, 1)));
		}

		if (options.SensorFromRigTranslationStddev > 0)
		{
			double x = RandomUtils.RandomGaussian(0.0, options.SensorFromRigTranslationStddev);
			double y = RandomUtils.RandomGaussian(0.0, options.SensorFromRigTranslationStddev);
			double z = RandomUtils.RandomGaussian(0.0, options.SensorFromRigTranslationStddev);
			translation = new Vector3d(x, y, z);
		}

		return new Rigid3d(rotation, translation);
	}

	private static void SynthesizeFrame(
		SyntheticDatasetOptions options,
		Reconstruction reconstruction,
		Database? database,
		Rig rig,
		List<SensorId> cameraSensorIds,
		int rigIdx,
		int frameIdx,
		int descDim,
		HashSet<ulong> newPoints3DIds,
		ref int totalNumImages,
		ref int totalNumDescriptors)
	{
		var frame = new Frame();
		frame.SetRigId(rig.RigId);

		// Synthesize frames as sphere centered at world origin.
		Vector3d viewDir = -RandomEigen.RandomEigenVector3d().Normalized();
		Vector3d projCenter = -5 * viewDir;
		Quaterniond rigFromWorldRotation = Quaterniond.FromTwoVectors(viewDir, new Vector3d(0, 0, 1));
		var rigFromWorld = new Rigid3d(rigFromWorldRotation, rigFromWorldRotation * -projCenter);
		frame.SetRigFromWorld(rigFromWorld);

		var images = new List<Image>(options.NumCamerasPerRig);
		var camsFromWorld = new List<Rigid3d>(options.NumCamerasPerRig);
		foreach (SensorId sensorId in cameraSensorIds)
		{
			++totalNumImages;

			var image = new Image { Name = $"camera{sensorId.Id:D6}_frame{frameIdx:D6}{options.ImageExtension}" };
			images.Add(image);
			image.SetCameraId(sensorId.Id);
			image.ImageId = database == null ? (uint)totalNumImages : database.WriteImage(image);

			frame.AddDataId(image.DataId);

			// Need to compose cam_from_world manually, because the frame/rig pointer
			// references are not yet set up.
			Camera camera = reconstruction.Camera(image.CameraId);
			Rigid3d sensorFromRig = rig.IsRefSensor(camera.SensorId) ? new Rigid3d() : rig.SensorFromRig(camera.SensorId);
			Rigid3d camFromWorld = sensorFromRig * rigFromWorld;
			camsFromWorld.Add(camFromWorld);

			if (options.PriorPosition || options.PriorGravity)
			{
				WritePosePrior(options, Check.NotNull(database), image, camFromWorld);
			}
		}

		uint frameId = database == null
			? (uint)(rigIdx * options.NumFramesPerRig + frameIdx + 1)
			: database.WriteFrame(frame);
		frame.FrameId = frameId;
		reconstruction.AddFrame(frame);

		for (int cameraIdx = 0; cameraIdx < options.NumCamerasPerRig; ++cameraIdx)
		{
			Image image = images[cameraIdx];
			Camera camera = reconstruction.Camera(image.CameraId);
			Rigid3d camFromWorld = camsFromWorld[cameraIdx];

			image.SetFrameId(frameId);

			List<Point2D> points2D = SynthesizePoints2D(options, reconstruction, newPoints3DIds, camera, camFromWorld);

			if (database != null)
			{
				WriteFeatures(options, database, image.ImageId, points2D, descDim, ref totalNumDescriptors);
			}

			for (int point2DIdx = 0; point2DIdx < points2D.Count; ++point2DIdx)
			{
				Point2D point2D = points2D[point2DIdx];
				if (point2D.HasPoint3D)
				{
					reconstruction.Point3D(point2D.Point3DId).Track.AddElement(image.ImageId, (uint)point2DIdx);
				}
			}

			image.SetPoints2D(points2D);

			database?.UpdateImage(image);
			reconstruction.AddImage(image);
		}
	}

	private static void WritePosePrior(SyntheticDatasetOptions options, Database database, Image image, Rigid3d camFromWorld)
	{
		var posePrior = new PosePrior();

		if (options.PriorPosition)
		{
			posePrior.Position = camFromWorld.TgtOriginInSrc();
			posePrior.CoordinateSystem = PosePriorCoordinateSystem.Cartesian;
			switch (options.PriorPositionCoordinateSystem)
			{
				case PosePriorCoordinateSystem.Cartesian:
					break;
				case PosePriorCoordinateSystem.Wgs84:
					PosePriorPositionCartesianToWgs84(ref posePrior);
					break;
				default:
					throw new ArgumentException("Invalid PosePrior::CoordinateSystem specified");
			}
		}

		if (options.PriorGravity)
		{
			posePrior.Gravity = (camFromWorld.Rotation * options.PriorGravityInWorld).Normalized();
		}

		posePrior.CorrDataId = image.DataId;
		posePrior.PosePriorId = database.WritePosePrior(posePrior);
	}

	// The image's 2D points: the projections of the new 3D points that land inside the image,
	// then uniform random points without a 3D point, shuffled.
	private static List<Point2D> SynthesizePoints2D(
		SyntheticDatasetOptions options,
		Reconstruction reconstruction,
		HashSet<ulong> newPoints3DIds,
		Camera camera,
		Rigid3d camFromWorld)
	{
		var points2D = new List<Point2D>(options.NumPoints3D + options.NumPoints2DWithoutPoint3D);

		// Create 3D point observations by projecting 3D points to the image.
		foreach (var (point3DId, point3D) in reconstruction.Points3D)
		{
			if (!newPoints3DIds.Contains(point3DId))
			{
				// If a non-empty reconstruction is given, only add tracks for newly added
				// images and 3D points.
				continue;
			}

			Vector2d? projPoint2D = camera.ImgFromCam(camFromWorld * point3D.Xyz);
			Check.That(projPoint2D.HasValue);
			Vector2d xy = projPoint2D.Value;
			if (xy.X >= 0 && xy.Y >= 0 && xy.X <= camera.Width && xy.Y <= camera.Height)
			{
				points2D.Add(new Point2D { Xy = xy, Point3DId = point3DId });
			}
		}

		// Synthesize uniform random 2D points without 3D points.
		for (int i = 0; i < options.NumPoints2DWithoutPoint3D; ++i)
		{
			double x = RandomUtils.RandomUniformReal(0.0, (double)camera.Width);
			double y = RandomUtils.RandomUniformReal(0.0, (double)camera.Height);
			points2D.Add(new Point2D { Xy = new Vector2d(x, y) });
		}

		// Shuffle 2D points, so each image has 3D points ordered differently.
		LibcxxRandom.Shuffle(points2D, Check.NotNull(RandomUtils.Prng));
		return points2D;
	}

	// Keypoints at the 2D points and a descriptor per point that is unique to its 3D point
	// (seeded by the 3D point id), or unique on its own for a point without one.
	private static void WriteFeatures(
		SyntheticDatasetOptions options,
		Database database,
		uint imageId,
		List<Point2D> points2D,
		int descDim,
		ref int totalNumDescriptors)
	{
		var keypoints = new List<FeatureKeypoint>(points2D.Count);
		var descriptors = new FeatureDescriptors(options.FeatureType, new RowMajorMatrix<byte>(points2D.Count, descDim));
		for (int point2DIdx = 0; point2DIdx < points2D.Count; ++point2DIdx)
		{
			Point2D point2D = points2D[point2DIdx];
			keypoints.Add(new FeatureKeypoint((float)point2D.Xy.X, (float)point2D.Xy.Y));
			// Generate a unique descriptor for each 3D point. If the 2D point does not
			// observe a 3D point, generate a random unique descriptor. mt19937's seed is
			// uint_fast32_t (32 bits with libc++), so the 64-bit point id is truncated.
			uint seed = point2D.HasPoint3D
				? unchecked((uint)point2D.Point3DId)
				: unchecked((uint)(options.NumPoints3D + ++totalNumDescriptors));
			var featureGenerator = new Mt19937(seed);
			for (int d = 0; d < descriptors.Data.Cols; ++d)
			{
				descriptors.Data[point2DIdx, d] = (byte)LibcxxRandom.UniformInt(featureGenerator, 0, 255);
			}
		}

		database.WriteKeypoints(imageId, keypoints);
		database.WriteDescriptors(imageId, descriptors);
	}

	// Prunes every longer track to exactly trackLength observations, dropping a random
	// subset. Points are visited in ascending id order (COLMAP: hash order, entry 31).
	private static void PruneTracks(int trackLength, Reconstruction reconstruction)
	{
		var point3DIds = new List<ulong>(reconstruction.Points3D.Keys);
		foreach (ulong point3DId in point3DIds)
		{
			Track track = reconstruction.Point3D(point3DId).Track;
			if (track.Length <= trackLength)
			{
				continue;
			}

			var elements = new List<TrackElement>(track.Elements);
			LibcxxRandom.Shuffle(elements, Check.NotNull(RandomUtils.Prng));
			int numToDelete = elements.Count - trackLength;
			for (int i = 0; i < numToDelete; ++i)
			{
				reconstruction.DeleteObservation(elements[i].ImageId, elements[i].Point2DIdx);
			}
		}
	}

	private static void PosePriorPositionCartesianToWgs84(ref PosePrior posePrior)
	{
		posePrior.Position = GpsTransform.ENUToEllipsoid([posePrior.Position], Lat0, Lon0, Alt0)[0];
		posePrior.CoordinateSystem = PosePriorCoordinateSystem.Wgs84;
	}

	private static void PosePriorPositionWgs84ToCartesian(ref PosePrior posePrior)
	{
		posePrior.Position = GpsTransform.EllipsoidToENU([posePrior.Position], Lat0, Lon0, Alt0)[0];
		posePrior.CoordinateSystem = PosePriorCoordinateSystem.Cartesian;
	}
}
