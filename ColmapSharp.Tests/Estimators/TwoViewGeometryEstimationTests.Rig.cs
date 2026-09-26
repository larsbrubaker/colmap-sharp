// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimationTests.Rig: EstimateRigTwoViewGeometries.Nominal of
// colmap/estimators/two_view_geometry_test.cc with its CreateRigTwoViewGeometryTestData
// helper, for Estimators/TwoViewGeometryEstimation.Rig.cs. Tier C, COLMAP's tolerances.
//
// Translation notes: the in-memory SQLite database is an InMemoryDatabase. COLMAP copies the
// two rigs out of the reconstruction; here they are the reconstruction's own (read-only use).

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public partial class TwoViewGeometryEstimationTests
{
	private sealed record RigTwoViewGeometryTestData(
		Rig Rig1,
		Rig Rig2,
		List<((uint ImageId1, uint ImageId2) ImagePair, List<FeatureMatch> Matches)> Matches,
		Reconstruction Reconstruction);

	private static RigTwoViewGeometryTestData CreateRigTwoViewGeometryTestData(SyntheticDatasetOptions syntheticDatasetOptions)
	{
		var reconstruction = new Reconstruction();
		var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction, database);

		Check.Eq(reconstruction.NumRigs, 2);

		Rig rig1 = reconstruction.Rig(1);
		Rig rig2 = reconstruction.Rig(2);
		var matches = new List<((uint ImageId1, uint ImageId2) ImagePair, List<FeatureMatch> Matches)>();
		foreach ((ulong pairId, List<FeatureMatch> pairMatches) in database.ReadAllMatches())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			Camera camera1 = reconstruction.Camera(reconstruction.Image(imageId1).CameraId);
			Camera camera2 = reconstruction.Camera(reconstruction.Image(imageId2).CameraId);
			if (rig1.HasSensor(camera1.SensorId) && rig2.HasSensor(camera2.SensorId))
			{
				matches.Add(((imageId1, imageId2), pairMatches));
			}
			else if (rig1.HasSensor(camera2.SensorId) && rig2.HasSensor(camera1.SensorId))
			{
				matches.Add(((imageId2, imageId1), pairMatches));
			}

			// else: Ignore matches between sensors in the same rig.
		}

		return new RigTwoViewGeometryTestData(rig1, rig2, matches, reconstruction);
	}

	[Test]
	public async Task EstimateRigTwoViewGeometries_Nominal()
	{
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 3,
			NumFramesPerRig = 1,
			NumPoints3D = 200,
			// Use only inlier matches so that pose recovery is exact and thus numerically
			// stable across platforms and build configurations. Outlier robustness of two-view
			// geometry estimation is covered by the EstimateTwoViewGeometry.*Deterministic
			// tests above.
			InlierMatchRatio = 1.0,
			CameraHasPriorFocalLength = true,
		};
		RigTwoViewGeometryTestData testData = CreateRigTwoViewGeometryTestData(syntheticDatasetOptions);

		var twoViewGeometryOptions = new TwoViewGeometryOptions();
		twoViewGeometryOptions.RansacOptions.RandomSeed = 42;
		var geometries = TwoViewGeometryEstimation.EstimateRigTwoViewGeometries(
			testData.Rig1,
			testData.Rig2,
			testData.Reconstruction.Images,
			testData.Reconstruction.Cameras,
			testData.Matches,
			twoViewGeometryOptions);
		await Assert.That(geometries.Count).IsEqualTo(testData.Matches.Count);
		foreach (((uint ImageId1, uint ImageId2) imagePair, TwoViewGeometry geometry) in geometries)
		{
			await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.CalibratedRig);
			await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
			Rigid3d expected = testData.Reconstruction.Image(imagePair.ImageId2).CamFromWorld()
				* testData.Reconstruction.Image(imagePair.ImageId1).CamFromWorld().Inverse();
			await Assert.That(Rigid3dNear(geometry.Cam2FromCam1!.Value, expected, rtol: 1e-2, ttol: 1e-3)).IsTrue();
			await Assert.That(geometry.InlierMatches.Count).IsGreaterThan(0);
		}
	}
}
