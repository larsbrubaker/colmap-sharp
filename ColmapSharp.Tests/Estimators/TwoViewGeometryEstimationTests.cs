// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimationTests: colmap/estimators/two_view_geometry_test.cc 1:1 for
// Estimators/TwoViewGeometryEstimation*.cs. Test names are <Suite>_<Test>.
//
// Ported: EstimateTwoViewGeometryPose.{Calibrated, FailureDueToInsufficientMatches,
// Uncalibrated, Planar, PlanarOrPanoramic}, TwoViewGeometryFromKnownRelativePose.Nominal,
// MaybeDecomposeRelativePoses.{Nominal, UsesSolverEstimatedIntrinsics,
// MissingMatrixFromOldDatabase}.
// The EstimateTwoViewGeometry.* and EstimateMultipleTwoViewGeometries.* cases are in the
// .Estimate, .Focal and .Multiple partial files. Waiting: EstimateRigTwoViewGeometries.Nominal
// (needs estimators/generalized_pose).
//
// Translation notes: gtest's SetPRNGSeed is RandomUtils.SetPRNGSeed; SQLite in-memory
// databases are InMemoryDatabase. ExtractPointsAndMatches walks Points3D in point-id order
// (COLMAP walks an unordered_map); the checks below are order-insensitive (medians, counts).

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public partial class TwoViewGeometryEstimationTests
{
	private static readonly TwoViewGeometry.ConfigurationType[] PoseConfigs =
	[
		TwoViewGeometry.ConfigurationType.Calibrated,
		TwoViewGeometry.ConfigurationType.Uncalibrated,
		TwoViewGeometry.ConfigurationType.Planar,
		TwoViewGeometry.ConfigurationType.Panoramic,
	];

	private static void ExtractPointsAndMatches(
		Reconstruction reconstruction,
		Image image1,
		Image image2,
		List<Vector2d> points1,
		List<Vector2d> points2,
		List<Vector3d> points3D,
		List<FeatureMatch> matches)
	{
		points1.Clear();
		points2.Clear();
		matches.Clear();

		foreach (Point2D point2D in image1.Points2D)
		{
			points1.Add(point2D.Xy);
		}

		foreach (Point2D point2D in image2.Points2D)
		{
			points2.Add(point2D.Xy);
		}

		foreach (ulong point3DId in reconstruction.Points3D.Keys.OrderBy(id => id))
		{
			Point3D point3D = reconstruction.Points3D[point3DId];
			Track track = point3D.Track;
			if (track.Length != 2)
			{
				throw new InvalidOperationException("Check failed: track.Length() == 2");
			}

			points3D.Add(point3D.Xyz);

			TrackElement elem1 = track.Element(0);
			TrackElement elem2 = track.Element(1);

			uint idx1;
			uint idx2;
			if (elem1.ImageId == image1.ImageId && elem2.ImageId == image2.ImageId)
			{
				idx1 = elem1.Point2DIdx;
				idx2 = elem2.Point2DIdx;
			}
			else if (elem1.ImageId == image2.ImageId && elem2.ImageId == image1.ImageId)
			{
				idx1 = elem2.Point2DIdx;
				idx2 = elem1.Point2DIdx;
			}
			else
			{
				throw new InvalidOperationException("Invalid track element.");
			}

			matches.Add(new FeatureMatch(idx1, idx2));
		}
	}

	private sealed class TwoViewGeometryPoseTestData
	{
		public Camera Camera1 = null!;
		public Camera Camera2 = null!;
		public List<Vector2d> Points1 = [];
		public List<Vector2d> Points2 = [];
		public TwoViewGeometry Geometry = new();
	}

	private static TwoViewGeometryPoseTestData CreateTwoViewGeometryPoseTestData(TwoViewGeometry.ConfigurationType config)
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 50,
			CameraHasPriorFocalLength = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		Image image1 = reconstruction.Image(1);
		Image image2 = reconstruction.Image(2);

		var data = new TwoViewGeometryPoseTestData
		{
			Camera1 = reconstruction.Camera(image1.CameraId),
			Camera2 = reconstruction.Camera(image2.CameraId),
		};
		data.Geometry.Config = config;
		Rigid3d cam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse();

		if (config == TwoViewGeometry.ConfigurationType.Calibrated)
		{
			data.Geometry.E = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
		}
		else if (config == TwoViewGeometry.ConfigurationType.Uncalibrated)
		{
			data.Geometry.F = EssentialMatrix.FundamentalFromEssentialMatrix(
				data.Camera2.CalibrationMatrix(),
				EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1),
				data.Camera1.CalibrationMatrix());
		}
		else if (config == TwoViewGeometry.ConfigurationType.Planar)
		{
			Vector3d homographyPlaneNormal =
				image1.CamFromWorld().Rotation * -(image1.ViewingDirection() + image2.ViewingDirection()).Normalized();
			const double HomographyPlaneDistance = 1;
			data.Geometry.H = HomographyMatrix.HomographyMatrixFromPose(
				data.Camera1.CalibrationMatrix(),
				data.Camera2.CalibrationMatrix(),
				cam2FromCam1.Rotation.ToRotationMatrix(),
				cam2FromCam1.Translation,
				homographyPlaneNormal,
				HomographyPlaneDistance);
		}
		else if (config == TwoViewGeometry.ConfigurationType.Panoramic)
		{
			cam2FromCam1 = new Rigid3d(cam2FromCam1.Rotation, Vector3d.Zero);
			data.Geometry.H = HomographyMatrix.HomographyMatrixFromPose(
				data.Camera1.CalibrationMatrix(),
				data.Camera2.CalibrationMatrix(),
				cam2FromCam1.Rotation.ToRotationMatrix(),
				cam2FromCam1.Translation,
				new Vector3d(0, 0, 1),
				1);
		}
		else
		{
			throw new InvalidOperationException("Invalid configuration.");
		}

		data.Geometry.Cam2FromCam1 = cam2FromCam1;

		var points3D = new List<Vector3d>();
		ExtractPointsAndMatches(reconstruction, image1, image2, data.Points1, data.Points2, points3D, data.Geometry.InlierMatches);

		data.Geometry.TriAngle = config == TwoViewGeometry.ConfigurationType.Panoramic
			? 0
			: MathUtils.Median(Triangulation.CalculateTriangulationAngles(
				image1.ProjectionCenter(), image2.ProjectionCenter(), points3D).AsSpan());

		return data;
	}

	private static bool CheckEqualTwoViewGeometry(
		TwoViewGeometry geometry,
		TwoViewGeometry expectedGeometry,
		double triAngleTol,
		double rotationTol,
		double translationTol,
		bool normalizedTranslation)
	{
		Rigid3d actual = geometry.Cam2FromCam1 ?? throw new InvalidOperationException("Check failed: geometry.cam2_from_cam1.has_value()");
		Rigid3d expected = expectedGeometry.Cam2FromCam1 ?? throw new InvalidOperationException("Check failed: expected_geometry.cam2_from_cam1.has_value()");
		double triAngleError = Math.Abs(geometry.TriAngle - expectedGeometry.TriAngle);
		double rotationError = actual.Rotation.AngularDistance(expected.Rotation);
		double translationError = (actual.Translation
			- (normalizedTranslation ? expected.Translation.Normalized() : expected.Translation)).Norm;
		return !(triAngleError > triAngleTol || rotationError > rotationTol || translationError > translationTol);
	}

	private static async Task RunPoseTest(
		TwoViewGeometry.ConfigurationType config, double triAngleTol, double translationTol, bool normalizedTranslation)
	{
		const int NumTests = 100;
		var log = new ExpectationLog();
		int numFailures = 0;
		for (uint seed = 0; seed < NumTests; ++seed)
		{
			RandomUtils.SetPRNGSeed(seed);
			TwoViewGeometryPoseTestData testData = CreateTwoViewGeometryPoseTestData(config);

			var geometry = new TwoViewGeometry
			{
				Config = testData.Geometry.Config,
				E = testData.Geometry.E,
				F = testData.Geometry.F,
				H = testData.Geometry.H,
				InlierMatches = [.. testData.Geometry.InlierMatches],
			};
			log.True(TwoViewGeometryEstimation.EstimateTwoViewGeometryPose(
				testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, geometry), $"seed {seed}: pose estimated");
			if (!CheckEqualTwoViewGeometry(geometry, testData.Geometry, triAngleTol, 1e-6, translationTol, normalizedTranslation))
			{
				numFailures++;
			}
		}

		log.Equal(numFailures, 0, "num_failures");
		await Assert.That(log.Failures).IsEmpty();
	}

	// Each C++ case copies only the matrix its config uses (E, F or H); the others are unset
	// in the test data, so copying all three is the same.
	[Test]
	public async Task EstimateTwoViewGeometryPose_Calibrated() =>
		await RunPoseTest(TwoViewGeometry.ConfigurationType.Calibrated, 1e-6, 1e-6, normalizedTranslation: true);

	[Test]
	public async Task EstimateTwoViewGeometryPose_FailureDueToInsufficientMatches()
	{
		var log = new ExpectationLog();
		foreach (TwoViewGeometry.ConfigurationType config in PoseConfigs)
		{
			TwoViewGeometryPoseTestData testData = CreateTwoViewGeometryPoseTestData(config);
			testData.Geometry.InlierMatches.Clear();

			var geometry = new TwoViewGeometry
			{
				Config = testData.Geometry.Config,
				E = testData.Geometry.E,
				InlierMatches = [.. testData.Geometry.InlierMatches],
			};
			log.False(TwoViewGeometryEstimation.EstimateTwoViewGeometryPose(
				testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, geometry), $"{config}");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task EstimateTwoViewGeometryPose_Uncalibrated() =>
		await RunPoseTest(TwoViewGeometry.ConfigurationType.Uncalibrated, 1e-6, 1e-6, normalizedTranslation: true);

	[Test]
	public async Task EstimateTwoViewGeometryPose_Planar() =>
		await RunPoseTest(TwoViewGeometry.ConfigurationType.Planar, 1e-3, 1e-5, normalizedTranslation: false);

	[Test]
	public async Task EstimateTwoViewGeometryPose_PlanarOrPanoramic()
	{
		const int NumTests = 100;
		var log = new ExpectationLog();
		int numFailures = 0;
		for (uint seed = 0; seed < NumTests; ++seed)
		{
			RandomUtils.SetPRNGSeed(seed);
			foreach (TwoViewGeometry.ConfigurationType config in new[]
				{ TwoViewGeometry.ConfigurationType.Planar, TwoViewGeometry.ConfigurationType.Panoramic })
			{
				TwoViewGeometryPoseTestData testData = CreateTwoViewGeometryPoseTestData(config);

				var geometry = new TwoViewGeometry
				{
					Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic,
					H = testData.Geometry.H,
					InlierMatches = [.. testData.Geometry.InlierMatches],
				};
				log.True(TwoViewGeometryEstimation.EstimateTwoViewGeometryPose(
					testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, geometry), $"seed {seed} {config}: pose estimated");
				log.Equal(geometry.Config, config, $"seed {seed}: config");
				if (!CheckEqualTwoViewGeometry(geometry, testData.Geometry, 1e-3, 1e-6, 1e-6, normalizedTranslation: false))
				{
					numFailures++;
				}
			}
		}

		log.Equal(numFailures, 0, "num_failures");
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task TwoViewGeometryFromKnownRelativePose_Nominal()
	{
		const int NumTests = 100;
		var log = new ExpectationLog();
		for (uint seed = 0; seed < NumTests; ++seed)
		{
			RandomUtils.SetPRNGSeed(seed);
			TwoViewGeometryPoseTestData testData = CreateTwoViewGeometryPoseTestData(TwoViewGeometry.ConfigurationType.Calibrated);

			TwoViewGeometry geometry = TwoViewGeometryEstimation.TwoViewGeometryFromKnownRelativePose(
				testData.Camera1,
				testData.Points1,
				testData.Camera2,
				testData.Points2,
				testData.Geometry.Cam2FromCam1!.Value,
				testData.Geometry.InlierMatches,
				minNumInliers: 15,
				maxError: 4.0);

			log.Equal(geometry.Cam2FromCam1, testData.Geometry.Cam2FromCam1, $"seed {seed}: cam2_from_cam1");
			log.Equal(geometry.E, testData.Geometry.E, $"seed {seed}: E");
			log.True(geometry.InlierMatches.SequenceEqual(testData.Geometry.InlierMatches), $"seed {seed}: inlier_matches");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task MaybeDecomposeRelativePoses_Nominal()
	{
		var database = new InMemoryDatabase();

		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 50,
			CameraHasPriorFocalLength = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction, database);

		// Load the database into a cache.
		var cacheOptions = new DatabaseCache.Options();
		DatabaseCache cache = DatabaseCache.Create(database, cacheOptions);

		// Verify the two-view geometry exists but has no decomposed pose yet.
		CorrespondenceGraph corrGraph = cache.CorrespondenceGraph;
		TwoViewGeometry geometryBefore = corrGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryBefore.Cam2FromCam1.HasValue).IsFalse();

		// Decompose poses - should update cache without throwing.
		TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(cache);

		// Verify the geometry was updated with a decomposed pose.
		TwoViewGeometry geometryAfter = corrGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryAfter.Cam2FromCam1.HasValue).IsTrue();

		// Calling again should skip already decomposed geometries.
		TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(cache);

		TwoViewGeometry geometrySecond = corrGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryAfter.Cam2FromCam1!.Value.Rotation.Coeffs == geometrySecond.Cam2FromCam1!.Value.Rotation.Coeffs).IsTrue();
		await Assert.That(geometryAfter.Cam2FromCam1!.Value.Translation == geometrySecond.Cam2FromCam1!.Value.Translation).IsTrue();
	}

	// A pair whose focal a two-view solver recovered is UNCALIBRATED but carries the
	// estimated intrinsics in camera1/camera2. MaybeDecomposeRelativePoses must calibrate the
	// rays with those, not the camera's stale default focal. The pose survives a wrong focal
	// (it comes from E alone); tri_angle, measured between the rays, does not.
	[Test]
	public async Task MaybeDecomposeRelativePoses_UsesSolverEstimatedIntrinsics()
	{
		var database = new InMemoryDatabase();

		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			NumPoints3D = 100,
			CameraHasPriorFocalLength = false,
			// A focal well away from the no-prior default of 1.2 * max(width, height), so
			// that using the wrong one is actually observable.
			CameraParams = [1000, 512, 384, 0.05],
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction, database);

		Image image1 = reconstruction.Image(1);
		Image image2 = reconstruction.Image(2);
		await Assert.That(image1.CameraId).IsEqualTo(image2.CameraId);
		Rigid3d gtCam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse();

		// Emulate an intrinsics-estimating solver: the true focal is surfaced via
		// camera1/camera2, while the camera stored in the database still carries the default
		// focal that must not be used to calibrate the rays.
		Camera trueCamera = image1.CameraPtr.Clone();

		TwoViewGeometry geometry = database.ReadTwoViewGeometry(1, 2);
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Uncalibrated);
		await Assert.That(geometry.E.HasValue).IsTrue();
		geometry.Camera1 = trueCamera.Clone();
		geometry.Camera2 = trueCamera.Clone();
		database.UpdateTwoViewGeometry(1, 2, geometry);

		// Reference run: the database camera happens to carry the correct focal, so the
		// estimated intrinsics and the database camera agree.
		var cacheOptions = new DatabaseCache.Options();
		DatabaseCache referenceCache = DatabaseCache.Create(database, cacheOptions);
		TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(referenceCache);
		TwoViewGeometry referenceGeometry =
			referenceCache.CorrespondenceGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(referenceGeometry.Cam2FromCam1.HasValue).IsTrue();
		await Assert.That(referenceGeometry.TriAngle).IsGreaterThan(0);

		// Now give the database camera the focal it would actually carry without a prior,
		// i.e. ImageReaderOptions::default_focal_length_factor times the larger image
		// dimension. The result must be unchanged: the decomposition has to key off the
		// estimated intrinsics, not the database camera.
		Camera defaultCamera = trueCamera.Clone();
		defaultCamera.SetFocalLength(1.2 * Math.Max(trueCamera.Width, trueCamera.Height));
		defaultCamera.HasPriorFocalLength = false;
		await Assert.That(Math.Abs((defaultCamera.FocalLength() / trueCamera.FocalLength()) - 1.0)).IsGreaterThan(0.1);
		database.UpdateCamera(defaultCamera);

		DatabaseCache cache = DatabaseCache.Create(database, cacheOptions);
		TwoViewGeometry geometryBefore = cache.CorrespondenceGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryBefore.Cam2FromCam1.HasValue).IsFalse();
		await Assert.That(geometryBefore.Camera1 is not null).IsTrue();

		TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(cache);

		TwoViewGeometry geometryAfter = cache.CorrespondenceGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryAfter.Cam2FromCam1.HasValue).IsTrue();

		// The pose is recovered from E alone, so it is insensitive to the focal used to
		// calibrate the rays; the triangulation angle is measured between those rays and is
		// not.
		await Assert.That(Rigid3dNear(geometryAfter.Cam2FromCam1!.Value, referenceGeometry.Cam2FromCam1!.Value, 1e-6, 1e-6)).IsTrue();
		await Assert.That(Math.Abs(geometryAfter.TriAngle - referenceGeometry.TriAngle)).IsLessThanOrEqualTo(1e-6);

		// Sanity check that the reference itself is meaningful.
		Rigid3d reference = referenceGeometry.Cam2FromCam1!.Value;
		await Assert.That(Rigid3dNear(
			new Rigid3d(reference.Rotation, reference.Translation.Normalized()),
			new Rigid3d(gtCam2FromCam1.Rotation, gtCam2FromCam1.Translation.Normalized()),
			1e-3,
			1e-2)).IsTrue();
	}

	// Regression test for https://github.com/colmap/colmap/issues/4387: older COLMAP
	// databases stored the two-view geometry config without persisting the E/F/H matrices.
	// MaybeDecomposeRelativePoses must refit the missing matrix from the inlier matches
	// instead of crashing in EstimateTwoViewGeometryPose's THROW_CHECK on geometry->E/F/H.
	[Test]
	public async Task MaybeDecomposeRelativePoses_MissingMatrixFromOldDatabase()
	{
		var database = new InMemoryDatabase();

		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 50,
			CameraHasPriorFocalLength = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction, database);

		// Simulate an older database that has the configuration but no E matrix.
		TwoViewGeometry geometryLegacy = database.ReadTwoViewGeometry(1, 2);
		await Assert.That(geometryLegacy.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
		await Assert.That(geometryLegacy.E.HasValue).IsTrue();
		geometryLegacy.E = null;
		database.UpdateTwoViewGeometry(1, 2, geometryLegacy);

		var cacheOptions = new DatabaseCache.Options();
		DatabaseCache cache = DatabaseCache.Create(database, cacheOptions);

		CorrespondenceGraph corrGraph = cache.CorrespondenceGraph;
		TwoViewGeometry geometryBefore = corrGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryBefore.Cam2FromCam1.HasValue).IsFalse();

		TwoViewGeometryEstimation.MaybeDecomposeRelativePoses(cache);

		TwoViewGeometry geometryAfter = corrGraph.ExtractTwoViewGeometry(1, 2, extractInlierMatches: false);
		await Assert.That(geometryAfter.Cam2FromCam1.HasValue).IsTrue();
	}
}
