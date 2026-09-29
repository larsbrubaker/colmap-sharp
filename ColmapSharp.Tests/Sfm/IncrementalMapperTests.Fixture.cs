// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapperTests.Fixture: the shared setup of IncrementalMapperTests.cs, ported from
// colmap/sfm/incremental_mapper_test.cc: the synthetic dataset options, the Require helper for
// ASSERT_* preconditions, and the Fixture class that stands in for the gtest fixtures
// IncrementalMapperTest and IncrementalMapperLargeDatasetTest. The tests themselves, and the
// translation notes, are in IncrementalMapperTests.cs. A partial class because Fixture is
// nested and private to the tests, as the gtest fixtures are to their TEST_Fs.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Sfm;

public partial class IncrementalMapperTests
{
	private static SyntheticDatasetOptions DefaultSyntheticOptions() => new()
	{
		NumRigs = 2,
		NumCamerasPerRig = 1,
		NumFramesPerRig = 5,
		NumPoints3D = 100,
	};

	// ASSERT_* inside the C++ tests and helpers: fail the test when a precondition does not
	// hold.
	private static void Require(bool condition, string message = "Test precondition failed")
	{
		if (!condition)
		{
			throw new InvalidOperationException(message);
		}
	}

	private sealed class Fixture
	{
		public readonly Reconstruction GtReconstruction = new();
		public readonly DatabaseCache Cache;
		public readonly IncrementalMapper Mapper;
		public Reconstruction Reconstruction;
		public readonly IncrementalMapper.Options Options = new();
		public readonly IncrementalTriangulator.Options TriOptions = new();
		public uint ImageId1 = InvalidImageId;
		public uint ImageId2 = InvalidImageId;
		public Rigid3d Cam2FromCam1 = Rigid3d.Identity;

		// SetUp; numFramesPerRig is 12 for IncrementalMapperLargeDatasetTest.
		// priorPosition is for the C#-only pose-prior test at the bottom.
		public Fixture(int numFramesPerRig = 5, bool priorPosition = false)
		{
			SyntheticDatasetOptions syntheticOptions = DefaultSyntheticOptions();
			syntheticOptions.NumFramesPerRig = numFramesPerRig;
			syntheticOptions.PriorPosition = priorPosition;
			using var database = new InMemoryDatabase();
			Synthetic.SynthesizeDataset(syntheticOptions, GtReconstruction, database);
			Cache = DatabaseCache.Create(database, new DatabaseCache.Options());
			Mapper = new IncrementalMapper(Cache);
			Reconstruction = new Reconstruction();
			Mapper.BeginReconstruction(Reconstruction);
			Options.InitMinNumInliers = 10;
			Options.AbsPoseMinNumInliers = 10;
			Options.AbsPoseMinInlierRatio = 0.1;
		}

		public void TearDown()
		{
			if (Mapper.Reconstruction is not null)
			{
				Mapper.EndReconstruction(discard: false);
			}
		}

		public void FindAndRegisterInitialPair()
		{
			Require(Mapper.FindInitialImagePair(Options, ref ImageId1, ref ImageId2, ref Cam2FromCam1));
			Mapper.RegisterInitialImagePair(Options, ImageId1, ImageId2, Cam2FromCam1);
		}

		public void TriangulateInitialPair()
		{
			Mapper.TriangulateImage(TriOptions, ImageId1);
			Mapper.TriangulateImage(TriOptions, ImageId2);
		}

		public void RegisterAllRemainingImages()
		{
			while (true)
			{
				bool anyImageRegistered = false;
				List<uint> nextImageIds = Mapper.FindNextImages(Options);
				foreach (uint imageId in nextImageIds)
				{
					if (Mapper.RegisterNextImage(Options, imageId))
					{
						Mapper.TriangulateImage(TriOptions, imageId);
						anyImageRegistered = true;
					}
				}

				if (!anyImageRegistered)
				{
					break;
				}
			}
		}

		public void BeginWithSynthesizedReconstruction()
		{
			if (Mapper.Reconstruction is not null)
			{
				Mapper.EndReconstruction(discard: false);
			}

			Reconstruction = GtReconstruction.Clone();
			Mapper.BeginReconstruction(Reconstruction);
		}

		public bool IsFrameRegistered(uint frameId) => Reconstruction.RegFrameIds.Contains(frameId);

		public long CountFramePoints3D(uint frameId)
		{
			long numPoints3D = 0;
			foreach (DataId dataId in Reconstruction.Frame(frameId).ImageIds())
			{
				numPoints3D += Reconstruction.Image((uint)dataId.Id).NumPoints3D;
			}

			return numPoints3D;
		}

		public int CountRegisteredFramesWithZeroPoints3D() =>
			Reconstruction.RegFrameIds.Count(frameId => CountFramePoints3D(frameId) == 0);

		public uint FindRegisteredFrameWithPoints3D()
		{
			foreach (uint frameId in Reconstruction.RegFrameIds)
			{
				if (CountFramePoints3D(frameId) > 0)
				{
					return frameId;
				}
			}

			return InvalidFrameId;
		}

		public void DeleteAllObservationsInFrame(uint frameId)
		{
			var observationsToDelete = new List<(uint ImageId, uint Point2DIdx)>();
			foreach (DataId dataId in Reconstruction.Frame(frameId).ImageIds())
			{
				Image image = Reconstruction.Image((uint)dataId.Id);
				for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
				{
					if (image.Points2D[(int)point2DIdx].HasPoint3D)
					{
						observationsToDelete.Add(((uint)dataId.Id, point2DIdx));
					}
				}
			}

			foreach ((uint imageId, uint point2DIdx) in observationsToDelete)
			{
				if (Reconstruction.Image(imageId).Points2D[(int)point2DIdx].HasPoint3D)
				{
					Mapper.ObservationManager.DeleteObservation(imageId, point2DIdx);
				}
			}
		}
	}
}
