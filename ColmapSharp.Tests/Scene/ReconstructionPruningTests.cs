// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionPruningTests: colmap/scene/reconstruction_pruning_test.cc, one method per
// gtest case named Suite_Name. Tests ColmapSharp/Scene/ReconstructionPruning.cs.
//
// All four cases are ported; CSharpOnly_SameTilePointIsRedundant is an extra C#-only check.
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does;
// tests synthesize before their first await. testing::UnorderedElementsAre(...) is
// IsEquivalentTo.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionPruningTests
{
	[Test]
	public async Task FindRedundantPoints3D_Empty()
	{
		var reconstruction = new Reconstruction();
		await Assert.That(ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0, reconstruction)).IsEmpty();
	}

	[Test]
	public async Task FindRedundantPoints3D_VaryingCoverageGain()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 5,
			NumPoints3D = 100,
			TrackLength = 5,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		// Give every point the same coverage so the number selected at each threshold is
		// independent of random image-tile occupancy.
		foreach (Image image in reconstruction.Images.Values)
		{
			for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				image.Point2DAt(point2DIdx).Xy = new Vector2d(image.CameraPtr.Width / 2.0, image.CameraPtr.Height / 2.0);
			}
		}

		await Assert.That(ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0, reconstruction)).IsEmpty();
		foreach (var (minCoverageGain, expectedNumRedundantPoints3D) in new[] { (0.1, 92), (0.4, 98), (0.7, 99), (10.0, 100) })
		{
			List<ulong> redundantPoint3DIds = ReconstructionPruning.FindRedundantPoints3D(minCoverageGain, reconstruction);
			await Assert.That(redundantPoint3DIds.Count).IsEqualTo(expectedNumRedundantPoints3D);
		}
	}

	[Test]
	public async Task FindRedundantPoints3D_VaryingTrackLength()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 5,
			NumPoints2DWithoutPoint3D = 0,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		var expectedRedundantPoint3DIds = new List<ulong>();
		var results = new List<List<ulong>>();
		var expectations = new List<List<ulong>>();
		foreach (var (point3DId, point3D) in reconstruction.Points3D)
		{
			TrackElement trackEl = point3D.Track.Element(0);
			reconstruction.Image(trackEl.ImageId).ResetPoint3DForPoint2D(trackEl.Point2DIdx);
			reconstruction.Point3D(point3DId).Track.DeleteElement(0);
			expectedRedundantPoint3DIds.Add(point3DId);
			results.Add(ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0.1, reconstruction));
			expectations.Add([.. expectedRedundantPoint3DIds]);
		}

		for (int i = 0; i < results.Count; ++i)
		{
			await Assert.That(results[i]).IsEquivalentTo(expectations[i]);
		}
	}

	[Test]
	public async Task FindRedundantPoints3D_VaryingSpatialDistribution()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 4,
			NumFramesPerRig = 1,
			NumPoints3D = 4,
			NumPoints2DWithoutPoint3D = 0,
			// Ensure all cameras in the rig have identical poses.
			SensorFromRigTranslationStddev = 0.0,
			SensorFromRigRotationStddev = 0.0,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		// Generate a synthetic dataset where all points are identically distributed in a
		// circle around the camera center, i.e., each of them has the same coverage.
		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			Check(point3D.Track.Length == syntheticDatasetOptions.NumCamerasPerRig);
			for (int i = 0; i < syntheticDatasetOptions.NumCamerasPerRig; ++i)
			{
				TrackElement trackEl = point3D.Track.Element(i);
				double angle = Math.PI / 4 + i * Math.PI / 2;
				double radius = 0.25 * Math.Sqrt(
					syntheticDatasetOptions.CameraWidth * syntheticDatasetOptions.CameraWidth
					+ syntheticDatasetOptions.CameraHeight * syntheticDatasetOptions.CameraHeight);
				reconstruction.Image(trackEl.ImageId).Point2DAt(trackEl.Point2DIdx).Xy = new Vector2d(
					Math.Sin(angle) * radius + syntheticDatasetOptions.CameraWidth / 2.0,
					Math.Cos(angle) * radius + syntheticDatasetOptions.CameraHeight / 2.0);
			}
		}

		ref Point2D GetPoint2D(ulong point3DId, int trackElIdx)
		{
			TrackElement trackEl = reconstruction.Point3D(point3DId).Track.Element(trackElIdx);
			return ref reconstruction.Image(trackEl.ImageId).Point2DAt(trackEl.Point2DIdx);
		}

		// Make sure point 3 has the largest coverage and selected first.
		reconstruction.Point3D(3).Track.AddElement(1, 4);
		var point2D = new Point2D { Xy = Vector2d.Zero, Point3DId = 3 };
		reconstruction.Image(1).Points2D.Add(point2D);

		// Make sure point 1 has larger coverage gain by moving it to another tile, while
		// point 2 and 4 have redundant and thus smaller coverage.
		GetPoint2D(1, 0).Xy *= 2;
		List<ulong> redundant1 = ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0.6, reconstruction);

		// Now change point 2 to have redundant coverage with point 1 and point 4 to have
		// unique coverage.
		GetPoint2D(2, 0).Xy *= 2;
		GetPoint2D(4, 2).Xy *= 2;
		List<ulong> redundant2 = ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0.6, reconstruction);

		await Assert.That(redundant1).IsEquivalentTo(new List<ulong> { 2, 4 });
		await Assert.That(redundant2).IsEquivalentTo(new List<ulong> { 1, 2 });
	}

	// CHECK_EQ in a test body: aborts the test on failure.
	private static void Check(bool condition)
	{
		if (!condition)
		{
			throw new InvalidOperationException("Check failed: point3D.track.Length() == num_cameras_per_rig");
		}
	}

	/// <summary>
	/// C#-only: two single-observation points. In the
	/// same image tile the second one's gain drops from 1 - 1/sqrt(2) to 1/sqrt(2) - 1/sqrt(3)
	/// once the first is selected; in different tiles both keep the full gain. With equal
	/// gains the larger id is selected first, as in COLMAP's (gain, point3D_id) max-heap.
	/// </summary>
	[Test]
	public async Task CSharpOnly_SameTilePointIsRedundant()
	{
		static Reconstruction Build(Vector2d xy1, Vector2d xy2)
		{
			var reconstruction = new Reconstruction();
			reconstruction.AddCameraWithTrivialRig(Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 80, 80));
			var image = new Image { ImageId = 1, Name = "image" };
			image.SetCameraId(1);
			image.SetPoints2D([xy1, xy2]);
			reconstruction.AddImageWithTrivialFrame(image);
			var track1 = new Track();
			track1.AddElement(1, 0);
			reconstruction.AddPoint3D(Vector3d.Zero, track1);
			var track2 = new Track();
			track2.AddElement(1, 1);
			reconstruction.AddPoint3D(Vector3d.Zero, track2);
			return reconstruction;
		}

		Reconstruction sameTile = Build(new Vector2d(5, 5), new Vector2d(6, 6));
		Reconstruction differentTiles = Build(new Vector2d(5, 5), new Vector2d(75, 5));
		using (Assert.Multiple())
		{
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.2, sameTile)).IsEquivalentTo(new List<ulong> { 1 });
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.1, sameTile)).IsEmpty();
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.2, differentTiles)).IsEmpty();
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.3, differentTiles)).IsEquivalentTo(new List<ulong> { 1, 2 });
		}
	}
}
