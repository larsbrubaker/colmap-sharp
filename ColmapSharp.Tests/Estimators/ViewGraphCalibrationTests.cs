// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ViewGraphCalibrationTests: colmap/estimators/view_graph_calibration_test.cc 1:1 for
// Estimators/ViewGraphCalibration.cs. Test names are <Suite>_<Test>.
//
// Ported: CalibrateViewGraph.{Nominal, PriorFocalLength, ConfigTagging,
// RelativePoseReestimation, SphericalCamerasAreIgnored, FisheyeCamerasAreIgnored}.
// Tier C (a nonlinear solve and RANSAC): COLMAP's own tolerances.
//
// Translation notes: SQLite in-memory databases are InMemoryDatabase, which, like SQLite's
// rowid order, returns the two-view geometries in pair-id order. COLMAP's gtest_main reseeds
// the PRNG with 0 at every test start; each test does the same with RandomUtils.SetPRNGSeed(0)
// and makes every draw before its first await (the PRNG is per thread).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public class ViewGraphCalibrationTests
{
	private static SyntheticDatasetOptions PinholeOptions(bool hasPriorFocalLength) => new()
	{
		NumRigs = 10,
		NumCamerasPerRig = 1,
		NumFramesPerRig = 1,
		NumPoints3D = 200,
		CameraModelId = CameraModelId.SimplePinhole,
		CameraParams = [1280, 512, 384],
		CameraHasPriorFocalLength = hasPriorFocalLength,
	};

	[Test]
	public async Task CalibrateViewGraph_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(PinholeOptions(false), reconstruction, database);

		// Store ground truth focal lengths.
		var gtFocals = new Dictionary<uint, double>();
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			gtFocals[cameraId] = camera.MeanFocalLength();
		}

		// Add noise to focal lengths of the first two cameras.
		// TODO: investigate view graph calibration cost functor and use more challenging
		// test setup.
		foreach (uint cameraId in reconstruction.Cameras.Keys)
		{
			if (cameraId >= 2)
			{
				continue;
			}

			Camera camera = database.ReadCamera(cameraId);
			double noise = RandomUtils.RandomUniformReal(-50.0, 50.0);
			foreach (int idx in camera.FocalLengthIdxs)
			{
				camera.Params[idx] += noise;
			}

			camera.HasPriorFocalLength = false;
			database.UpdateCamera(camera);
		}

		var calibOptions = new ViewGraphCalibrationOptions { ReestimateRelativePose = false };
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		// Verify focal lengths are calibrated close to ground truth.
		foreach ((uint cameraId, double gtFocal) in gtFocals)
		{
			Camera camera = database.ReadCamera(cameraId);
			await Assert.That(camera.HasPriorFocalLength).IsTrue();
			await Assert.That(Math.Abs(camera.MeanFocalLength() - gtFocal)).IsLessThanOrEqualTo(1.0);
		}

		// Verify pairs are now CALIBRATED with valid E matrices.
		foreach ((_, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			await Assert.That(tvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
			await Assert.That(tvg.E.HasValue).IsTrue();
		}
	}

	[Test]
	public async Task CalibrateViewGraph_PriorFocalLength()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(PinholeOptions(true), reconstruction, database);

		// Store original focal lengths (which have priors).
		var originalFocals = new Dictionary<uint, double>();
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			originalFocals[cameraId] = camera.MeanFocalLength();
		}

		var calibOptions = new ViewGraphCalibrationOptions { ReestimateRelativePose = false };
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		// Verify cameras with priors are unchanged.
		foreach ((uint cameraId, double originalFocal) in originalFocals)
		{
			Camera camera = database.ReadCamera(cameraId);
			await Assert.That(camera.MeanFocalLength()).IsEqualTo(originalFocal);
		}
	}

	[Test]
	public async Task CalibrateViewGraph_ConfigTagging()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(PinholeOptions(false), reconstruction, database);

		// Add large noise to F matrices for 3 pairs to ensure they become degenerate.
		var perturbedPairs = new HashSet<ulong>();
		var ones = new Matrix3d(1, 1, 1, 1, 1, 1, 1, 1, 1);
		foreach ((ulong pairId, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			if (perturbedPairs.Count >= 3)
			{
				break;
			}

			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			TwoViewGeometry perturbedTvg = tvg.Clone();
			perturbedTvg.F = perturbedTvg.F!.Value + ones;
			database.UpdateTwoViewGeometry(imageId1, imageId2, perturbedTvg);
			perturbedPairs.Add(pairId);
		}

		var calibOptions = new ViewGraphCalibrationOptions
		{
			ReestimateRelativePose = false,
			MaxCalibrationError = 0.01,
		};
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		// Verify perturbed pairs became DEGENERATE, others became CALIBRATED.
		foreach ((ulong pairId, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			if (perturbedPairs.Contains(pairId))
			{
				await Assert.That(tvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Degenerate);
			}
			else
			{
				await Assert.That(tvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
			}
		}
	}

	[Test]
	public async Task CalibrateViewGraph_RelativePoseReestimation()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(PinholeOptions(false), reconstruction, database);

		// Store ground truth relative poses and perturb them in the database. The
		// perturbation must exceed test thresholds (0.1 rad rotation, 0.1 normalized
		// translation error) to ensure re-estimation actually runs.
		var gtPoses = new Dictionary<ulong, Rigid3d>();
		foreach ((ulong pairId, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			Image image1 = reconstruction.Image(imageId1);
			Image image2 = reconstruction.Image(imageId2);
			gtPoses[pairId] = image2.CamFromWorld() * image1.CamFromWorld().Inverse();

			// Perturb the relative pose stored in database.
			TwoViewGeometry perturbedTvg = tvg.Clone();
			if (perturbedTvg.Cam2FromCam1.HasValue)
			{
				// The C++ evaluates these draws as one expression; its argument order is
				// unspecified, so any order is as faithful. They are drawn left to right.
				double angle = RandomUtils.RandomUniformReal(0.3, 0.7);
				double ax = RandomUtils.RandomUniformReal(-1.0, 1.0);
				double ay = RandomUtils.RandomUniformReal(-1.0, 1.0);
				double az = RandomUtils.RandomUniformReal(-1.0, 1.0);
				double tx = RandomUtils.RandomUniformReal(-1.0, 1.0);
				double ty = RandomUtils.RandomUniformReal(-1.0, 1.0);
				double tz = RandomUtils.RandomUniformReal(-1.0, 1.0);
				var perturbation = new Rigid3d(
					Quaterniond.FromAngleAxis(new AngleAxisd(angle, new Vector3d(ax, ay, az).Normalized())),
					new Vector3d(tx, ty, tz));
				Rigid3d perturbed = perturbation * perturbedTvg.Cam2FromCam1.Value;
				perturbedTvg.Cam2FromCam1 = perturbed with { Translation = perturbed.Translation.Normalized() };
			}

			database.UpdateTwoViewGeometry(imageId1, imageId2, perturbedTvg);
		}

		var calibOptions = new ViewGraphCalibrationOptions { ReestimateRelativePose = true };
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		// Verify relative poses are estimated correctly.
		foreach ((ulong pairId, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			if (tvg.Config != TwoViewGeometry.ConfigurationType.Calibrated)
			{
				continue;
			}

			await Assert.That(tvg.Cam2FromCam1.HasValue).IsTrue();
			Rigid3d cam2FromCam1 = tvg.Cam2FromCam1!.Value;
			await Assert.That(Math.Abs(cam2FromCam1.Translation.Norm - 1.0)).IsLessThanOrEqualTo(1e-6);

			// Normalize ground truth translation since estimated pose has unit scale.
			Rigid3d gtPose = gtPoses[pairId];
			var gtPoseNormalized = new Rigid3d(gtPose.Rotation, gtPose.Translation.Normalized());
			await Assert.That(Rigid3dNear(cam2FromCam1, gtPoseNormalized, 0.01, 0.01)).IsTrue();
		}
	}

	[Test]
	public async Task CalibrateViewGraph_SphericalCamerasAreIgnored()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();

		// Spherical (omnidirectional) cameras have no focal length and produce CALIBRATED
		// two-view geometries without a fundamental matrix. View graph calibration must skip
		// them gracefully instead of trying to optimize a (non-existent) focal length.
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 10,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 200,
			CameraModelId = CameraModelId.Equirectangular,
			CameraWidth = 1000,
			CameraHeight = 500,
			CameraParams = [1000, 500],
		};

		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		// Store original camera parameters to verify they remain untouched.
		var originalParams = new Dictionary<uint, double[]>();
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			originalParams[cameraId] = [.. camera.Params];
		}

		var calibOptions = new ViewGraphCalibrationOptions { ReestimateRelativePose = false };
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		// Spherical camera parameters must be left unchanged.
		foreach ((uint cameraId, double[] parameters) in originalParams)
		{
			Camera camera = database.ReadCamera(cameraId);
			await Assert.That(camera.IsSpherical).IsTrue();
			await Assert.That(camera.Params.SequenceEqual(parameters)).IsTrue();
		}

		// Spherical pairs stay CALIBRATED with valid E matrices and no F matrix.
		foreach ((_, TwoViewGeometry tvg) in database.ReadTwoViewGeometries())
		{
			await Assert.That(tvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
			await Assert.That(tvg.E.HasValue).IsTrue();
			await Assert.That(tvg.F.HasValue).IsFalse();
		}
	}

	[Test]
	public async Task CalibrateViewGraph_FisheyeCamerasAreIgnored()
	{
		RandomUtils.SetPRNGSeed(0);

		var database = new InMemoryDatabase();

		// A fisheye camera projects angularly, so its focal length cannot be recovered from a
		// fundamental matrix. View graph calibration must skip it entirely, leaving both its
		// parameters and its focal length prior flag untouched, rather than reporting a focal
		// length it never optimized.
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 10,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 200,
			CameraModelId = CameraModelId.OpenCVFisheye,
			CameraParams = [1280, 1280, 512, 384, 0, 0, 0, 0],
			CameraHasPriorFocalLength = false,
		};

		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		// Store original camera parameters to verify they remain untouched.
		var originalParams = new Dictionary<uint, double[]>();
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			originalParams[cameraId] = [.. camera.Params];
		}

		var calibOptions = new ViewGraphCalibrationOptions { ReestimateRelativePose = false };
		await Assert.That(ViewGraphCalibration.CalibrateViewGraph(calibOptions, database)).IsTrue();

		foreach ((uint cameraId, double[] parameters) in originalParams)
		{
			Camera camera = database.ReadCamera(cameraId);
			await Assert.That(camera.IsPerspectiveFisheye).IsTrue();
			await Assert.That(camera.Params.SequenceEqual(parameters)).IsTrue();

			// Never calibrated, so it must not be marked as having a prior focal length,
			// which would make downstream stages trust an unestimated value.
			await Assert.That(camera.HasPriorFocalLength).IsFalse();
		}
	}
}
