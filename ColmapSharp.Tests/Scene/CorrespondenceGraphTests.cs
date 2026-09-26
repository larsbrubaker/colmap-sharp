// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CorrespondenceGraphTests: colmap/scene/correspondence_graph_test.cc ported 1:1, testing
// ColmapSharp/Scene/CorrespondenceGraph.cs. TEST(Suite, Name) becomes Suite_Name; the
// TEST_P cases of CorrespondenceGraphFinalizeTest, instantiated with Bool() as
// NotFinalized / Finalized, become CorrespondenceGraphFinalizeTest_<Name>(finalize) with
// [Arguments(false)] and [Arguments(true)]. Finalize() is FinalizeGraph(); FeatureMatches
// equality is SequenceEqual; UnorderedElementsAre compares as sets.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so tests that draw start
// with RandomUtils.SetPRNGSeed(0) and draw everything before their first await (the PRNG is
// per thread). UpdateTwoViewGeometry draws its one pose earlier than the C++ does; it is the
// only draw, so its value is unchanged.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;
using static ColmapSharp.Util.Types;

using Correspondence = ColmapSharp.Scene.CorrespondenceGraph.Correspondence;

namespace ColmapSharp.Tests.Scene;

public class CorrespondenceGraphTests
{
	private static int CountNumTransitiveCorrespondences(CorrespondenceGraph graph, uint imageId, uint point2DIdx, int transitivity)
	{
		var corrs = new List<Correspondence>();
		graph.ExtractTransitiveCorrespondences(imageId, point2DIdx, transitivity, corrs);
		return corrs.Count;
	}

	private static Rigid3d RandomRigid3d()
	{
		return new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
	}

	private static List<FeatureMatch> Matches(params (uint, uint)[] pairs)
	{
		return [.. pairs.Select(p => new FeatureMatch(p.Item1, p.Item2))];
	}

	[Test]
	public async Task Correspondence_Print()
	{
		var correspondence = new Correspondence(1, 2);
		await Assert.That(correspondence.ToString()).IsEqualTo("Correspondence(image_id=1, point2D_idx=2)");
	}

	[Test]
	public async Task CorrespondenceGraph_Empty()
	{
		var correspondenceGraph = new CorrespondenceGraph();
		using (Assert.Multiple())
		{
			await Assert.That(correspondenceGraph.NumImages).IsEqualTo(0);
			await Assert.That(correspondenceGraph.NumImagePairs).IsEqualTo(0);
			await Assert.That(correspondenceGraph.NumMatchesBetweenAllImages().Count).IsEqualTo(0);
			await Assert.That(correspondenceGraph.ImagePairs().Count).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CorrespondenceGraph_Print()
	{
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		correspondenceGraph.AddTwoViewGeometry(0, 1, new TwoViewGeometry());
		await Assert.That(correspondenceGraph.ToString()).IsEqualTo("CorrespondenceGraph(num_images=2, num_image_pairs=1)");
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task CorrespondenceGraphFinalizeTest_TwoView(bool finalize)
	{
		RandomUtils.SetPRNGSeed(0);
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		bool exists0 = correspondenceGraph.ExistsImage(0);
		bool exists1 = correspondenceGraph.ExistsImage(1);
		bool exists2 = correspondenceGraph.ExistsImage(2);
		int numImages = correspondenceGraph.NumImages;
		var twoViewGeometry01 = new TwoViewGeometry
		{
			Cam2FromCam1 = RandomRigid3d(),
			InlierMatches = Matches((0, 0), (1, 2), (3, 7), (4, 8)),
		};
		correspondenceGraph.AddTwoViewGeometry(0, 1, twoViewGeometry01);
		if (finalize)
		{
			correspondenceGraph.FinalizeGraph();
		}

		using (Assert.Multiple())
		{
			await Assert.That(exists0).IsTrue();
			await Assert.That(exists1).IsTrue();
			await Assert.That(exists2).IsFalse();
			await Assert.That(numImages).IsEqualTo(2);

			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(0)).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(4u);
			ulong pairId = ImagePairToPairId(0, 1);
			await Assert.That(correspondenceGraph.NumMatchesBetweenAllImages().Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.NumMatchesBetweenAllImages()[pairId]).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.ImagePairs().ToHashSet().SetEquals([pairId])).IsTrue();
			await Assert.That(correspondenceGraph.ImagePairs().Count).IsEqualTo(1);
			TwoViewGeometry twoViewGeometry01Stored = correspondenceGraph.ExtractTwoViewGeometry(0, 1, extractInlierMatches: true);
			await Assert.That(Rigid3dNear(twoViewGeometry01Stored.Cam2FromCam1!.Value, twoViewGeometry01.Cam2FromCam1!.Value, rtol: 1e-6, ttol: 1e-6)).IsTrue();
			await Assert.That(twoViewGeometry01Stored.InlierMatches.SequenceEqual(twoViewGeometry01.InlierMatches)).IsTrue();
			var matches01Stored = new List<FeatureMatch>();
			correspondenceGraph.ExtractMatchesBetweenImages(0, 1, matches01Stored);
			await Assert.That(matches01Stored.SequenceEqual(twoViewGeometry01.InlierMatches)).IsTrue();
			TwoViewGeometry twoViewGeometry10 = twoViewGeometry01.Clone();
			twoViewGeometry10.Invert();
			TwoViewGeometry twoViewGeometry10Stored = correspondenceGraph.ExtractTwoViewGeometry(1, 0, extractInlierMatches: true);
			await Assert.That(Rigid3dNear(twoViewGeometry10Stored.Cam2FromCam1!.Value, twoViewGeometry10.Cam2FromCam1!.Value, rtol: 1e-6, ttol: 1e-6)).IsTrue();
			await Assert.That(twoViewGeometry10Stored.InlierMatches.SequenceEqual(twoViewGeometry10.InlierMatches)).IsTrue();
			var matches10Stored = new List<FeatureMatch>();
			correspondenceGraph.ExtractMatchesBetweenImages(1, 0, matches10Stored);
			await Assert.That(matches10Stored.SequenceEqual(twoViewGeometry10.InlierMatches)).IsTrue();
			TwoViewGeometry twoViewGeometry01WithoutMatchesStored = correspondenceGraph.ExtractTwoViewGeometry(0, 1, extractInlierMatches: false);
			await Assert.That(twoViewGeometry01WithoutMatchesStored.InlierMatches).IsEmpty();

			var corrs = new List<Correspondence>();

			correspondenceGraph.ExtractCorrespondences(0, 0, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(0, 0)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(0, 0)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(0u);

			correspondenceGraph.ExtractCorrespondences(1, 0, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(1, 0)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(1, 0)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(0u);

			correspondenceGraph.ExtractCorrespondences(0, 1, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(0, 1)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(0, 1)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(2u);

			correspondenceGraph.ExtractCorrespondences(1, 2, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(1, 2)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(1, 2)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(1u);

			correspondenceGraph.ExtractCorrespondences(0, 4, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(0, 4)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(0, 4)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(8u);

			correspondenceGraph.ExtractCorrespondences(0, 3, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(0, 3)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(7u);

			correspondenceGraph.ExtractCorrespondences(1, 7, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(1, 7)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(1, 7)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(3u);

			correspondenceGraph.ExtractCorrespondences(1, 8, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(correspondenceGraph.HasCorrespondences(1, 8)).IsTrue();
			await Assert.That(correspondenceGraph.IsTwoViewObservation(1, 8)).IsTrue();
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(4u);

			for (uint i = 0; i < 10; ++i)
			{
				await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 0, i, 0)).IsEqualTo(0);
				correspondenceGraph.ExtractCorrespondences(0, i, corrs);
				await Assert.That(corrs.Count).IsEqualTo(CountNumTransitiveCorrespondences(correspondenceGraph, 0, i, 2));
				await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 1, i, 0)).IsEqualTo(0);
				correspondenceGraph.ExtractCorrespondences(1, i, corrs);
				await Assert.That(corrs.Count).IsEqualTo(CountNumTransitiveCorrespondences(correspondenceGraph, 1, i, 2));
			}

			var matches01 = new List<FeatureMatch>();
			correspondenceGraph.ExtractMatchesBetweenImages(0, 1, matches01);
			await Assert.That(matches01.SequenceEqual(twoViewGeometry01.InlierMatches)).IsTrue();
			var matches10 = new List<FeatureMatch>();
			correspondenceGraph.ExtractMatchesBetweenImages(1, 0, matches10);
			await Assert.That(matches10.SequenceEqual(twoViewGeometry10.InlierMatches)).IsTrue();
			await Assert.That(correspondenceGraph.NumObservationsForImage(0)).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(1)).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(0)).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(4u);
		}
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task CorrespondenceGraphFinalizeTest_ThreeView(bool finalize)
	{
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		correspondenceGraph.AddImage(2, 10);
		correspondenceGraph.AddTwoViewGeometry(0, 1, new TwoViewGeometry { InlierMatches = Matches((0, 0)) });
		correspondenceGraph.AddTwoViewGeometry(0, 2, new TwoViewGeometry { InlierMatches = Matches((0, 0)) });
		correspondenceGraph.AddTwoViewGeometry(1, 2, new TwoViewGeometry { InlierMatches = Matches((0, 0), (5, 5)) });
		if (finalize)
		{
			correspondenceGraph.FinalizeGraph();
		}

		using (Assert.Multiple())
		{
			await Assert.That(correspondenceGraph.NumObservationsForImage(0)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(1)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumObservationsForImage(2)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(0)).IsEqualTo(2u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(3u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(2)).IsEqualTo(3u);
			ulong pairId01 = ImagePairToPairId(0, 1);
			ulong pairId02 = ImagePairToPairId(0, 2);
			ulong pairId12 = ImagePairToPairId(1, 2);
			Dictionary<ulong, uint> numMatches = correspondenceGraph.NumMatchesBetweenAllImages();
			await Assert.That(numMatches.Count).IsEqualTo(3);
			await Assert.That(numMatches[pairId01]).IsEqualTo(1u);
			await Assert.That(numMatches[pairId02]).IsEqualTo(1u);
			await Assert.That(numMatches[pairId12]).IsEqualTo(2u);
			List<ulong> imagePairs = correspondenceGraph.ImagePairs();
			await Assert.That(imagePairs.Count).IsEqualTo(3);
			await Assert.That(imagePairs.ToHashSet().SetEquals([pairId01, pairId02, pairId12])).IsTrue();

			var corrs = new List<Correspondence>();
			correspondenceGraph.ExtractCorrespondences(0, 0, corrs);
			await Assert.That(corrs.Count).IsEqualTo(2);
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(0u);
			await Assert.That(corrs[1].ImageId).IsEqualTo(2u);
			await Assert.That(corrs[1].Point2DIdx).IsEqualTo(0u);

			correspondenceGraph.ExtractCorrespondences(1, 0, corrs);
			await Assert.That(corrs.Count).IsEqualTo(2);
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(0u);
			await Assert.That(corrs[1].ImageId).IsEqualTo(2u);
			await Assert.That(corrs[1].Point2DIdx).IsEqualTo(0u);

			correspondenceGraph.ExtractCorrespondences(2, 0, corrs);
			await Assert.That(corrs.Count).IsEqualTo(2);
			await Assert.That(corrs[0].ImageId).IsEqualTo(0u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(0u);
			await Assert.That(corrs[1].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[1].Point2DIdx).IsEqualTo(0u);

			correspondenceGraph.ExtractCorrespondences(1, 5, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(corrs[0].ImageId).IsEqualTo(2u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(5u);

			correspondenceGraph.ExtractCorrespondences(2, 5, corrs);
			await Assert.That(corrs.Count).IsEqualTo(1);
			await Assert.That(corrs[0].ImageId).IsEqualTo(1u);
			await Assert.That(corrs[0].Point2DIdx).IsEqualTo(5u);

			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 0, 0, 2)).IsEqualTo(2);
			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 1, 0, 2)).IsEqualTo(2);
			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 2, 0, 2)).IsEqualTo(2);
			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 0, 0, 3)).IsEqualTo(2);
			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 1, 0, 3)).IsEqualTo(2);
			await Assert.That(CountNumTransitiveCorrespondences(correspondenceGraph, 2, 0, 3)).IsEqualTo(2);
		}
	}

	[Test]
	public async Task CorrespondenceGraph_OutOfBounds()
	{
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 4);
		var twoViewGeometry = new TwoViewGeometry { InlierMatches = Matches((9, 3), (10, 3), (9, 4)) };
		correspondenceGraph.AddTwoViewGeometry(0, 1, twoViewGeometry);
		using (Assert.Multiple())
		{
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(0)).IsEqualTo(1u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(1u);
			ulong pairId = ImagePairToPairId(0, 1);
			await Assert.That(correspondenceGraph.NumMatchesBetweenAllImages()[pairId]).IsEqualTo(1u);
		}
	}

	[Test]
	public async Task CorrespondenceGraph_Duplicate()
	{
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		var twoViewGeometry = new TwoViewGeometry { InlierMatches = Matches((0, 0), (1, 1), (1, 1), (3, 3), (3, 4)) };
		correspondenceGraph.AddTwoViewGeometry(0, 1, twoViewGeometry);
		using (Assert.Multiple())
		{
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(0)).IsEqualTo(4u);
			await Assert.That(correspondenceGraph.NumCorrespondencesForImage(1)).IsEqualTo(4u);
			ulong pairId = ImagePairToPairId(0, 1);
			await Assert.That(correspondenceGraph.NumMatchesBetweenAllImages()[pairId]).IsEqualTo(4u);
		}
	}

	[Test]
	public async Task CorrespondenceGraph_UpdateTwoViewGeometry()
	{
		RandomUtils.SetPRNGSeed(0);
		Rigid3d updatedPose = RandomRigid3d();
		var correspondenceGraph = new CorrespondenceGraph();
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			InlierMatches = Matches((0, 0), (1, 2), (3, 7)),
		};
		correspondenceGraph.AddTwoViewGeometry(0, 1, twoViewGeometry);
		correspondenceGraph.FinalizeGraph();

		using (Assert.Multiple())
		{
			// Verify initial state has no relative pose.
			TwoViewGeometry extracted = correspondenceGraph.ExtractTwoViewGeometry(0, 1, false);
			await Assert.That(extracted.Cam2FromCam1.HasValue).IsFalse();
			await Assert.That(extracted.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);

			// Update with a decomposed relative pose.
			TwoViewGeometry updated = extracted.Clone();
			updated.Cam2FromCam1 = updatedPose;
			correspondenceGraph.UpdateTwoViewGeometry(0, 1, updated);

			// Verify the updated geometry is returned.
			TwoViewGeometry afterUpdate = correspondenceGraph.ExtractTwoViewGeometry(0, 1, false);
			await Assert.That(afterUpdate.Cam2FromCam1.HasValue).IsTrue();
			await Assert.That(Rigid3dNear(afterUpdate.Cam2FromCam1!.Value, updated.Cam2FromCam1!.Value, 1e-6, 1e-6)).IsTrue();
			await Assert.That(afterUpdate.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);

			// Verify matches are preserved (stored separately from geometry).
			var matches = new List<FeatureMatch>();
			correspondenceGraph.ExtractMatchesBetweenImages(0, 1, matches);
			await Assert.That(matches.SequenceEqual(twoViewGeometry.InlierMatches)).IsTrue();
		}
	}

	[Test]
	public async Task CorrespondenceGraph_UpdateTwoViewGeometrySwapped()
	{
		RandomUtils.SetPRNGSeed(0);
		var correspondenceGraph = new CorrespondenceGraph();
		// Use image IDs where ShouldSwapImagePair(1, 0) is true, i.e. id1 > id2.
		correspondenceGraph.AddImage(0, 10);
		correspondenceGraph.AddImage(1, 10);
		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			InlierMatches = Matches((0, 0), (1, 2)),
		};
		correspondenceGraph.AddTwoViewGeometry(0, 1, twoViewGeometry);
		correspondenceGraph.FinalizeGraph();

		// Update using the swapped order (1, 0).
		Rigid3d cam1FromCam0 = RandomRigid3d();
		var updated = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			Cam2FromCam1 = cam1FromCam0,
		};
		correspondenceGraph.UpdateTwoViewGeometry(1, 0, updated);

		// Extract in the same swapped order and verify it matches.
		TwoViewGeometry extracted10 = correspondenceGraph.ExtractTwoViewGeometry(1, 0, false);
		// Extract in canonical order and verify it's the inverse.
		TwoViewGeometry extracted01 = correspondenceGraph.ExtractTwoViewGeometry(0, 1, false);
		using (Assert.Multiple())
		{
			await Assert.That(Rigid3dNear(extracted10.Cam2FromCam1!.Value, cam1FromCam0, 1e-6, 1e-6)).IsTrue();
			await Assert.That(Rigid3dNear(extracted01.Cam2FromCam1!.Value, cam1FromCam0.Inverse(), 1e-6, 1e-6)).IsTrue();
		}
	}
}
