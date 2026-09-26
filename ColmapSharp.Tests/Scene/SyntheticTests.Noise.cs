// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SyntheticTests.Noise: the SynthesizeNoise and SynthesizeImages cases of
// colmap/scene/synthetic_test.cc, 1:1. SyntheticTests.cs holds the SynthesizeDataset cases
// and the translation notes.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Scene;

public partial class SyntheticTests
{
	[Test]
	public async Task SynthesizeNoise_Point2DNoise()
	{
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions();
		Synthetic.SynthesizeDataset(options, reconstruction, database);
		double errorBefore = reconstruction.ComputeMeanReprojectionError();

		var syntheticNoiseOptions = new SyntheticNoiseOptions { Point2DStddev = 0.1 };
		Synthetic.SynthesizeNoise(syntheticNoiseOptions, reconstruction, database);
		await Assert.That(errorBefore).IsLessThan(1e-3);
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsGreaterThan(1e-3);

		foreach (var (imageId, image) in reconstruction.Images)
		{
			List<FeatureKeypoint> keypoints = database.ReadKeypoints(imageId);
			for (int point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				var keypoint = new Vector2d(keypoints[point2DIdx].X, keypoints[point2DIdx].Y);
				await Assert.That(EigenMatrixNear(keypoint, image.Points2D[point2DIdx].Xy, 1e-6)).IsTrue();
			}
		}
	}

	[Test]
	public async Task SynthesizeNoise_Point3DNoise()
	{
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions();

		Synthetic.SynthesizeDataset(options, reconstruction, database);
		double errorBefore = reconstruction.ComputeMeanReprojectionError();

		var syntheticNoiseOptions = new SyntheticNoiseOptions { Point3DStddev = 0.1 };
		Synthetic.SynthesizeNoise(syntheticNoiseOptions, reconstruction, database);
		await Assert.That(errorBefore).IsLessThan(1e-3);
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsGreaterThan(1e-3);
	}

	[Test]
	public async Task SynthesizeNoise_RigFromWorldNoise()
	{
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions();

		Synthetic.SynthesizeDataset(options, reconstruction, database);
		double errorBefore = reconstruction.ComputeMeanReprojectionError();

		var syntheticNoiseOptions = new SyntheticNoiseOptions
		{
			RigFromWorldTranslationStddev = 0.1,
			RigFromWorldRotationStddev = 0.1,
		};
		Synthetic.SynthesizeNoise(syntheticNoiseOptions, reconstruction, database);
		await Assert.That(errorBefore).IsLessThan(1e-3);
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsGreaterThan(1e-3);
	}

	private static Dictionary<uint, PosePrior> ReadPosePriors(Database database)
	{
		var posePriors = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in database.ReadAllPosePriors())
		{
			posePriors.Add(posePrior.PosePriorId, posePrior);
		}

		return posePriors;
	}

	[Test]
	public async Task SynthesizeNoise_PriorPositionNoise()
	{
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			PriorPosition = true,
			PriorPositionCoordinateSystem = PosePriorCoordinateSystem.Wgs84,
		};

		Synthetic.SynthesizeDataset(options, reconstruction, database);
		Dictionary<uint, PosePrior> origPosePriors = ReadPosePriors(database);

		var syntheticNoiseOptions = new SyntheticNoiseOptions { PriorPositionStddev = 0.1 };
		Synthetic.SynthesizeNoise(syntheticNoiseOptions, reconstruction, database);
		foreach (PosePrior posePrior in database.ReadAllPosePriors())
		{
			Vector3d origPosition = origPosePriors[posePrior.PosePriorId].Position;
			// Check that some noise was added.
			await Assert.That(posePrior.Position != origPosition).IsTrue();
			// Check that the noisy positions are somewhat close to the original ones.
			await Assert.That(EigenMatrixNear(posePrior.Position, origPosition, 10 * syntheticNoiseOptions.PriorPositionStddev)).IsTrue();
			await Assert.That(posePrior.HasPositionCov()).IsTrue();
			await Assert.That(posePrior.PositionCovariance.Trace() / 3.0)
				.IsEqualTo(syntheticNoiseOptions.PriorPositionStddev * syntheticNoiseOptions.PriorPositionStddev).Within(1e-6);
		}
	}

	[Test]
	public async Task SynthesizeNoise_PriorGravityNoise()
	{
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { PriorGravity = true };

		Synthetic.SynthesizeDataset(options, reconstruction, database);
		Dictionary<uint, PosePrior> origPosePriors = ReadPosePriors(database);

		var syntheticNoiseOptions = new SyntheticNoiseOptions { PriorGravityStddev = 0.1 };
		Synthetic.SynthesizeNoise(syntheticNoiseOptions, reconstruction, database);
		foreach (PosePrior posePrior in database.ReadAllPosePriors())
		{
			double angle = Math.Acos(posePrior.Gravity.Dot(origPosePriors[posePrior.PosePriorId].Gravity));
			// Check that some noise was added.
			await Assert.That(angle).IsGreaterThan(0);
			// Check that the noisy directions are somewhat close to the original ones.
			await Assert.That(angle).IsLessThan(10 * syntheticNoiseOptions.PriorGravityStddev);
		}
	}

	[Test]
	public async Task SynthesizeImages_Nominal()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			NumPoints3D = 80,
			NumPoints2DWithoutPoint3D = 20,
			CameraWidth = 320,
			CameraHeight = 240,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		var written = new Dictionary<string, Bitmap>();
		Synthetic.SynthesizeImages(new SyntheticImageOptions(), reconstruction, (name, bitmap) => written.Add(name, bitmap));

		foreach (Image image in reconstruction.Images.Values)
		{
			await Assert.That(written.TryGetValue(image.Name, out Bitmap? bitmap)).IsTrue();
			await Assert.That(bitmap!.Width).IsEqualTo(image.CameraPtr.Width);
			await Assert.That(bitmap.Height).IsEqualTo(image.CameraPtr.Height);
		}
	}
}
