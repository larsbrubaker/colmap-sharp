// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SyntheticTests: colmap/scene/synthetic_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/Synthetic*.cs. The oracle
// comparison against pycolmap is SyntheticOracleTests.cs.
//
// Translation notes: Database::Open(kInMemorySqliteDatabasePath) is an InMemoryDatabase.
// COLMAP's gtest_main seeds the PRNG with 0 before every test; the PRNG is thread-local and
// awaits may resume on another thread, so every test seeds and synthesizes before its first
// await. CreateTestDir() is a fresh temp directory. One deviation, forced by excluded image
// I/O: SynthesizeImages.Nominal's images go to a sink instead of files
// (docs/CPP_DIVERGENCES.md entry 32), so the test reads the width and height of the bitmaps
// the sink received rather than decoding PNG files.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;
using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public partial class SyntheticTests
{
	[Test]
	public async Task SynthesizeDataset_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 3, NumFramesPerRig = 3 };
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		string testDir = Path.Combine(Path.GetTempPath(), "colmapsharp-synthetic-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(testDir);
		try
		{
			string sparsePath = Path.Combine(testDir, "sparse");
			Directory.CreateDirectory(sparsePath);
			reconstruction.Write(sparsePath);
		}
		finally
		{
			Directory.Delete(testDir, true);
		}

		await Assert.That(database.NumRigs()).IsEqualTo(options.NumRigs);
		await Assert.That(reconstruction.NumRigs).IsEqualTo(options.NumRigs);
		foreach (Rig rig in reconstruction.Rigs.Values)
		{
			await Assert.That(rig.NumSensors).IsGreaterThanOrEqualTo(options.NumCamerasPerRig);
		}

		await Assert.That(database.NumCameras()).IsEqualTo(options.NumRigs * options.NumCamerasPerRig);
		await Assert.That(reconstruction.NumCameras).IsEqualTo(options.NumRigs * options.NumCamerasPerRig);
		foreach (var (cameraId, camera) in reconstruction.Cameras)
		{
			await Assert.That(camera == database.ReadCamera(cameraId)).IsTrue();
			await Assert.That(camera.ModelId).IsEqualTo(options.CameraModelId);
		}

		await Assert.That(database.NumFrames()).IsEqualTo(options.NumRigs * options.NumFramesPerRig);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(options.NumRigs * options.NumFramesPerRig);
		foreach (var (frameId, frame) in reconstruction.Frames)
		{
			Frame reconstructionFrame = frame.Clone();
			await Assert.That(reconstructionFrame.HasPose).IsTrue();
			reconstructionFrame.ResetPose();
			await Assert.That(reconstructionFrame == database.ReadFrame(frameId)).IsTrue();
			await Assert.That(reconstructionFrame.NumDataIds).IsEqualTo(reconstructionFrame.RigPtr.NumSensors);
		}

		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		await Assert.That(database.NumImages()).IsEqualTo(numImages);
		await Assert.That(reconstruction.NumImages).IsEqualTo(numImages);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(options.NumRigs * options.NumFramesPerRig);
		var imageNames = new HashSet<string>();
		foreach (var (imageId, image) in reconstruction.Images)
		{
			await Assert.That(image.Name).IsEqualTo(database.ReadImage(imageId).Name);
			await Assert.That(image.Name.EndsWith(options.ImageExtension, StringComparison.Ordinal)).IsTrue();
			imageNames.Add(image.Name);
			await Assert.That((int)image.NumPoints2D).IsEqualTo(database.ReadKeypoints(imageId).Count);
			await Assert.That((int)image.NumPoints2D).IsEqualTo(database.ReadDescriptors(imageId).Data.Rows);
			await Assert.That(database.ReadDescriptors(imageId).Data.Cols).IsEqualTo(128);
			await Assert.That((int)image.NumPoints2D).IsEqualTo(options.NumPoints3D + options.NumPoints2DWithoutPoint3D);
			await Assert.That((int)image.NumPoints3D).IsEqualTo(options.NumPoints3D);
		}

		await Assert.That(imageNames.Count).IsEqualTo(reconstruction.NumImages);

		int numImagePairs = reconstruction.NumImages * (reconstruction.NumImages - 1) / 2;
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumInlierMatches()).IsEqualTo(numImagePairs * options.NumPoints3D);

		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(0).Within(1e-6);
		await Assert.That(reconstruction.ComputeCentroid(0, 1).Norm).IsEqualTo(0).Within(0.2);
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(reconstruction.NumImages).Within(0.1);
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(reconstruction.NumImages * options.NumPoints3D);

		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(options.NumPoints3D);

		// All observations should be perfect and have sufficient triangulation angle. No
		// points or observations should be filtered.
		const double kMaxReprojError = 1e-3;
		const double kMinTriAngleDeg = 0.4;
		var projCenters = new Dictionary<uint, Vector3d>();
		foreach (ulong point3DId in reconstruction.Point3DIds())
		{
			Point3D point3D = reconstruction.Point3D(point3DId);

			// Make sure all descriptors of the same 3D point have identical features.
			FeatureDescriptors desc0 = database.ReadDescriptors(point3D.Track.Element(0).ImageId);
			byte[] descriptors = desc0.Data.Row((int)point3D.Track.Element(0).Point2DIdx).ToArray();

			double maxTriAngle = 0;
			for (int i1 = 0; i1 < point3D.Track.Length; ++i1)
			{
				TrackElement trackEl = point3D.Track.Element(i1);
				uint imageId1 = trackEl.ImageId;
				Image image1 = reconstruction.Image(imageId1);
				Camera camera1 = reconstruction.Camera(image1.CameraId);
				Point2D point2D = image1.Point2DAt(trackEl.Point2DIdx);
				double squaredReprojError = Projection.CalculateSquaredReprojectionError(point2D.Xy, point3D.Xyz, image1.CamFromWorld(), camera1);
				await Assert.That(squaredReprojError).IsLessThanOrEqualTo(kMaxReprojError * kMaxReprojError);
				FeatureDescriptors desc1 = database.ReadDescriptors(point3D.Track.Element(i1).ImageId);
				await Assert.That(desc1.Data.Row((int)point3D.Track.Element(i1).Point2DIdx).ToArray()).IsEquivalentTo(descriptors);

				if (!projCenters.TryGetValue(imageId1, out Vector3d projCenter1))
				{
					projCenter1 = image1.ProjectionCenter();
					projCenters.Add(imageId1, projCenter1);
				}

				for (int i2 = 0; i2 < i1; ++i2)
				{
					uint imageId2 = point3D.Track.Element(i2).ImageId;
					Vector3d projCenter2 = projCenters[imageId2];
					maxTriAngle = Math.Max(maxTriAngle, Triangulation.CalculateTriangulationAngle(projCenter1, projCenter2, point3D.Xyz));
				}
			}

			await Assert.That(maxTriAngle).IsGreaterThanOrEqualTo(MathUtils.DegToRad(kMinTriAngleDeg));
		}
	}

	[Test]
	public async Task SynthesizeDataset_MultipleTimes()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 3, NumFramesPerRig = 3 };
		Synthetic.SynthesizeDataset(options, reconstruction, database);
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		await Assert.That(database.NumRigs()).IsEqualTo(2 * options.NumRigs);
		await Assert.That((long)reconstruction.NumRigs).IsEqualTo(database.NumRigs());

		await Assert.That(database.NumCameras()).IsEqualTo(2 * options.NumRigs * options.NumCamerasPerRig);
		await Assert.That((long)reconstruction.NumCameras).IsEqualTo(database.NumCameras());

		await Assert.That(database.NumFrames()).IsEqualTo(2 * options.NumRigs * options.NumFramesPerRig);
		await Assert.That(database.NumFrames()).IsEqualTo(reconstruction.NumFrames);

		await Assert.That(database.NumImages()).IsEqualTo(2 * options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig);
		await Assert.That(database.NumImages()).IsEqualTo(reconstruction.NumImages);

		int numImagePairs = reconstruction.NumImages * (reconstruction.NumImages - 1) / 2;
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(numImagePairs);

		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(2 * options.NumPoints3D);
	}

	[Test]
	public async Task SynthesizeDataset_WithPriors()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			PriorPosition = true,
			PriorGravity = true,
			PriorGravityInWorld = RandomEigen.RandomEigenVector3d().Normalized(),
		};
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		var imageToPrior = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in posePriors)
		{
			await Assert.That(posePrior.CorrDataId.SensorId.Type).IsEqualTo(SensorType.Camera);
			imageToPrior[(uint)posePrior.CorrDataId.Id] = posePrior;
		}

		foreach (var (imageId, image) in reconstruction.Images)
		{
			await Assert.That(imageToPrior.ContainsKey(imageId)).IsTrue();
			PosePrior posePrior = imageToPrior[imageId];
			await Assert.That(EigenMatrixNear(image.ProjectionCenter(), posePrior.Position, 1e-9)).IsTrue();
			await Assert.That(EigenMatrixNear(image.CamFromWorld().Rotation * options.PriorGravityInWorld, posePrior.Gravity, 1e-9)).IsTrue();
		}
	}

	[Test]
	public async Task SynthesizeDataset_MultiReconstruction()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction1 = new Reconstruction();
		var reconstruction2 = new Reconstruction();
		var options = new SyntheticDatasetOptions();
		Synthetic.SynthesizeDataset(options, reconstruction1, database);
		Synthetic.SynthesizeDataset(options, reconstruction2, database);

		int numCameras = options.NumRigs * options.NumCamerasPerRig;
		await Assert.That(database.NumCameras()).IsEqualTo(2 * numCameras);
		await Assert.That(reconstruction1.NumCameras).IsEqualTo(numCameras);
		await Assert.That(reconstruction1.NumCameras).IsEqualTo(numCameras);
		int numImages = numCameras * options.NumFramesPerRig;
		await Assert.That(database.NumImages()).IsEqualTo(2 * numImages);
		await Assert.That(reconstruction1.NumImages).IsEqualTo(numImages);
		await Assert.That(reconstruction2.NumImages).IsEqualTo(numImages);
		await Assert.That(reconstruction1.NumRegFrames).IsEqualTo(numImages);
		await Assert.That(reconstruction2.NumRegFrames).IsEqualTo(numImages);
		int numImagePairs = numImages * (numImages - 1) / 2;
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(2 * numImagePairs);
		await Assert.That(database.NumInlierMatches()).IsEqualTo(2 * numImagePairs * options.NumPoints3D);
	}

	[Test]
	public async Task SynthesizeDataset_ExhaustiveMatches()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { MatchConfig = SyntheticMatchConfig.Exhaustive };
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		int numImagePairs = numImages * (numImages - 1) / 2;
		await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumInlierMatches()).IsEqualTo(numImagePairs * options.NumPoints3D);
	}

	[Test]
	public async Task SynthesizeDataset_ChainedMatches()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { MatchConfig = SyntheticMatchConfig.Chained };
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		int numImagePairs = numImages - 1;
		await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumInlierMatches()).IsEqualTo(numImagePairs * options.NumPoints3D);
		foreach (var (pairId, _) in database.ReadAllMatches())
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			await Assert.That(imageId1 + 1).IsEqualTo(imageId2);
		}

		foreach (var (pairId, _) in database.ReadTwoViewGeometries())
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			await Assert.That(imageId1 + 1).IsEqualTo(imageId2);
		}
	}

	[Test]
	public async Task SynthesizeDataset_SparseMatchesZeroSparsity()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { MatchConfig = SyntheticMatchConfig.Sparse, MatchSparsity = 0.0 };
		Synthetic.SynthesizeDataset(options, reconstruction, database);
		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		int numImagePairs = numImages * (numImages - 1) / 2;
		await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(numImagePairs);
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(numImagePairs);
	}

	[Test]
	public async Task SynthesizeDataset_SparseMatchesFullSparsity()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions { MatchConfig = SyntheticMatchConfig.Sparse, MatchSparsity = 1.0 };
		Synthetic.SynthesizeDataset(options, reconstruction, database);
		await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(0);
		await Assert.That(database.NumVerifiedImagePairs()).IsEqualTo(0);
	}

	[Test]
	public async Task SynthesizeDataset_SparseMatchesPartialSparsity()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 8,
			MatchConfig = SyntheticMatchConfig.Sparse,
			MatchSparsity = 0.5,
		};
		Synthetic.SynthesizeDataset(options, reconstruction, database);

		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		int numExhaustivePairs = numImages * (numImages - 1) / 2;
		await Assert.That(database.NumMatchedImagePairs()).IsEqualTo(numExhaustivePairs / 2);

		// Verify graph connectivity using UnionFind.
		var uf = new UnionFind<uint>();
		uf.Reserve(numImages);
		foreach (var (pairId, _) in database.ReadAllMatches())
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			uf.Union(imageId1, imageId2);
		}

		List<uint> regImageIds = reconstruction.RegImageIds();
		uint root = uf.Find(regImageIds[0]);
		foreach (uint imageId in regImageIds)
		{
			await Assert.That(uf.Find(imageId)).IsEqualTo(root);
		}
	}

	[Test]
	public async Task SynthesizeDataset_NoDatabase()
	{
		RandomUtils.SetPRNGSeed(0);
		var options = new SyntheticDatasetOptions();
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(options, reconstruction);
		await Task.CompletedTask;
	}

	[Test]
	public async Task SynthesizeDataset_TrackLength()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 5,
			NumPoints3D = 50,
			NumPoints2DWithoutPoint3D = 0,
			TrackLength = 3,
		};
		Synthetic.SynthesizeDataset(options, reconstruction);

		int numImages = options.NumRigs * options.NumCamerasPerRig * options.NumFramesPerRig;
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(numImages);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(options.NumPoints3D);
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(options.TrackLength).Within(1e-6);
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(options.NumPoints3D * options.TrackLength);
	}

	[Test]
	public async Task SynthesizeDataset_Determinism()
	{
		var options = new SyntheticDatasetOptions();

		var reconstruction1 = new Reconstruction();
		RandomUtils.SetPRNGSeed(42);
		Synthetic.SynthesizeDataset(options, reconstruction1);

		var reconstruction2 = new Reconstruction();
		RandomUtils.SetPRNGSeed(42);
		Synthetic.SynthesizeDataset(options, reconstruction2);

		string? explanation = ReconstructionMatchers.ExplainReconstructionEq(reconstruction1, reconstruction2);
		await Assert.That(explanation).IsNull();
	}
}
