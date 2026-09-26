// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalTriangulatorTests: colmap/sfm/incremental_triangulator_test.cc ported 1:1, one
// method per gtest TEST(Suite, Name) named Suite_Name, testing
// ColmapSharp/Sfm/IncrementalTriangulator*.cs.
//
// Tier C (outcome): point and observation counts after triangulating noise-free synthetic
// scenes, exactly the counts C++ expects.
//
// Translation notes: the in-memory SQLite database is InMemoryDatabase and DatabaseCache is
// loaded from it, as in the other ported tests. PrngTestIsolation seeds the PRNG with 0
// before every test, as COLMAP's gtest_main does; the PRNG is per thread, so each test does
// all of its synthesis and triangulation before its first await. FlatHashSet{...} arguments
// are collection expressions. `Points3D().begin()` is the first point in the port's
// ascending id order; any point satisfies the C++ test.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sfm;

public class IncrementalTriangulatorTests
{
	// Returns NumPoints3D afterwards, which C++ EXPECTs to be 0 inside the helper.
	private static int DeleteAllPoints3D(Reconstruction reconstruction)
	{
		foreach (ulong point3DId in reconstruction.Point3DIds())
		{
			reconstruction.DeletePoint3D(point3DId);
		}

		return reconstruction.NumPoints3D;
	}

	private static void DeleteOneObservationFromEachTrack(Reconstruction reconstruction)
	{
		foreach (Point3D point3D in reconstruction.Points3D.Values.ToList())
		{
			Check(point3D.Track.Length > 0);
			TrackElement first = point3D.Track.Element(0);
			reconstruction.DeleteObservation(first.ImageId, first.Point2DIdx);
		}
	}

	private static void SplitPoint3D(Reconstruction reconstruction, ulong point3DId)
	{
		Point3D point3D = reconstruction.Point3D(point3DId);
		Check(point3D.Track.Length >= 4);
		var splitTrack = new Track();
		for (int i = point3D.Track.Length / 2; i < point3D.Track.Length; ++i)
		{
			TrackElement trackEl = point3D.Track.Element(i);
			splitTrack.AddElement(trackEl);
			reconstruction.Image(trackEl.ImageId).ResetPoint3DForPoint2D(trackEl.Point2DIdx);
		}

		int half = point3D.Track.Length / 2;
		point3D.Track.Elements.RemoveRange(half, point3D.Track.Length - half);
		ulong newPoint3DId = reconstruction.AddPoint3D(point3D.Xyz, splitTrack);
		foreach (TrackElement trackEl in splitTrack.Elements)
		{
			reconstruction.Image(trackEl.ImageId).SetPoint3DForPoint2D(trackEl.Point2DIdx, newPoint3DId);
		}
	}

	// ASSERT_* inside the C++ helpers: fail the test when a precondition does not hold.
	private static void Check(bool condition)
	{
		if (!condition)
		{
			throw new InvalidOperationException("Test precondition failed");
		}
	}

	private static (Reconstruction Reconstruction, CorrespondenceGraph Graph) Synthesize(int numFramesPerRig, int numPoints3D)
	{
		var reconstruction = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFramesPerRig,
			NumPoints3D = numPoints3D,
		};
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(syntheticOptions, reconstruction, database);
		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());
		return (reconstruction, cache.CorrespondenceGraph);
	}

	[Test]
	public async Task IncrementalTriangulator_Print()
	{
		var reconstruction = new Reconstruction();
		var triangulator = new IncrementalTriangulator(new CorrespondenceGraph(), reconstruction);
		await Assert.That(triangulator.ToString()).IsEqualTo(
			"IncrementalTriangulator(reconstruction=Reconstruction(num_rigs=0, "
			+ "num_cameras=0, num_frames=0, num_reg_frames=0, num_images=0, "
			+ "num_points3D=0), correspondence_graph=CorrespondenceGraph(num_images=0, "
			+ "num_image_pairs=0))");
	}

	[Test]
	public async Task IncrementalTriangulator_ModifiedPoints3D()
	{
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(3, 50);

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(0);

		using IEnumerator<ulong> points3DIt = reconstruction.Points3D.Keys.GetEnumerator();
		points3DIt.MoveNext();
		ulong point3DId1 = points3DIt.Current;
		points3DIt.MoveNext();
		ulong point3DId2 = points3DIt.Current;

		triangulator.AddModifiedPoint3D(point3DId1);
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(1);
		await Assert.That(triangulator.GetModifiedPoints3D().Contains(point3DId1)).IsTrue();

		triangulator.AddModifiedPoint3D(point3DId2);
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(2);
		await Assert.That(triangulator.GetModifiedPoints3D().Contains(point3DId2)).IsTrue();

		triangulator.AddModifiedPoint3D(point3DId1);
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(2);

		triangulator.ClearModifiedPoints3D();
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(0);
	}

	[Test]
	public async Task IncrementalTriangulator_ModifiedPoints3DRemovesNonExistent()
	{
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(3, 10);

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		HashSet<ulong> point3DIds = reconstruction.Point3DIds();
		await Assert.That(point3DIds.Count).IsGreaterThanOrEqualTo(1);
		ulong point3DId = point3DIds.First();

		triangulator.AddModifiedPoint3D(point3DId);
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(1);
		reconstruction.DeletePoint3D(point3DId);
		await Assert.That(triangulator.GetModifiedPoints3D().Count).IsEqualTo(0);
	}

	[Test]
	public async Task IncrementalTriangulator_TriangulateImage()
	{
		const int kNumPoints3D = 20;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(5, kNumPoints3D);

		int numPointsAfterDelete = DeleteAllPoints3D(reconstruction);

		var triangulator = new IncrementalTriangulator(graph, reconstruction);
		int totalTris = 0;
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			totalTris += triangulator.TriangulateImage(new IncrementalTriangulator.Options(), imageId);
		}

		await Assert.That(numPointsAfterDelete).IsEqualTo(0);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
		await Assert.That(totalTris).IsEqualTo(kNumPoints3D * reconstruction.NumRegImages);
	}

	[Test]
	public async Task IncrementalTriangulator_CompleteImage()
	{
		const int kNumPoints3D = 20;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(5, kNumPoints3D);

		DeleteOneObservationFromEachTrack(reconstruction);
		long numObservationsAfterDelete = reconstruction.ComputeNumObservations();

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		triangulator.CompleteImage(new IncrementalTriangulator.Options(), reconstruction.RegImageIds()[0]);

		await Assert.That(numObservationsAfterDelete)
			.IsEqualTo((long)reconstruction.NumPoints3D * (reconstruction.NumRegImages - 1));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
		await Assert.That(reconstruction.ComputeNumObservations())
			.IsEqualTo((long)kNumPoints3D * (reconstruction.NumRegImages - 1));
	}

	[Test]
	public async Task IncrementalTriangulator_CompleteTracks()
	{
		const int kNumPoints3D = 20;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(3, kNumPoints3D);

		DeleteOneObservationFromEachTrack(reconstruction);
		long numObservationsAfterDelete = reconstruction.ComputeNumObservations();

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		int numCompletions = triangulator.CompleteTracks(
			new IncrementalTriangulator.Options(), [reconstruction.Points3D.Keys.First()]);

		await Assert.That(numObservationsAfterDelete)
			.IsEqualTo((long)reconstruction.NumPoints3D * (reconstruction.NumRegImages - 1));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
		await Assert.That(numCompletions).IsEqualTo(1);
		await Assert.That(reconstruction.ComputeNumObservations())
			.IsEqualTo(((long)kNumPoints3D * reconstruction.NumRegImages) - kNumPoints3D + 1);
	}

	[Test]
	public async Task IncrementalTriangulator_CompleteAllTracks()
	{
		const int kNumPoints3D = 20;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(3, kNumPoints3D);

		DeleteOneObservationFromEachTrack(reconstruction);
		long numObservationsAfterDelete = reconstruction.ComputeNumObservations();

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		int numCompletions = triangulator.CompleteAllTracks(new IncrementalTriangulator.Options());

		await Assert.That(numObservationsAfterDelete)
			.IsEqualTo((long)reconstruction.NumPoints3D * (reconstruction.NumRegImages - 1));
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
		await Assert.That(numCompletions).IsEqualTo(kNumPoints3D);
		await Assert.That(reconstruction.ComputeNumObservations())
			.IsEqualTo((long)kNumPoints3D * reconstruction.NumRegImages);
	}

	[Test]
	public async Task IncrementalTriangulator_MergeTracks()
	{
		const int kNumPoints3D = 5;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(10, kNumPoints3D);

		using IEnumerator<ulong> points3DIt = reconstruction.Points3D.Keys.GetEnumerator();
		points3DIt.MoveNext();
		ulong point3DId1 = points3DIt.Current;
		points3DIt.MoveNext();
		ulong point3DId2 = points3DIt.Current;

		SplitPoint3D(reconstruction, point3DId1);
		SplitPoint3D(reconstruction, point3DId2);
		int numPointsAfterSplit = reconstruction.NumPoints3D;

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		int numMerged = triangulator.MergeTracks(new IncrementalTriangulator.Options(), [point3DId1]);

		await Assert.That(numPointsAfterSplit).IsEqualTo(kNumPoints3D + 2);
		await Assert.That(numMerged).IsEqualTo(reconstruction.NumRegImages);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D + 1);
	}

	[Test]
	public async Task IncrementalTriangulator_MergeAllTracks()
	{
		const int kNumPoints3D = 5;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(10, kNumPoints3D);

		using IEnumerator<ulong> points3DIt = reconstruction.Points3D.Keys.GetEnumerator();
		points3DIt.MoveNext();
		ulong point3DId1 = points3DIt.Current;
		points3DIt.MoveNext();
		ulong point3DId2 = points3DIt.Current;

		SplitPoint3D(reconstruction, point3DId1);
		SplitPoint3D(reconstruction, point3DId2);
		int numPointsAfterSplit = reconstruction.NumPoints3D;

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		int numMerged = triangulator.MergeAllTracks(new IncrementalTriangulator.Options());

		await Assert.That(numPointsAfterSplit).IsEqualTo(kNumPoints3D + 2);
		await Assert.That(numMerged).IsEqualTo(2 * reconstruction.NumRegImages);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
	}

	[Test]
	public async Task IncrementalTriangulator_Retriangulate()
	{
		const int kNumPoints3D = 20;
		(Reconstruction reconstruction, CorrespondenceGraph graph) = Synthesize(5, kNumPoints3D);

		int numPointsAfterDelete = DeleteAllPoints3D(reconstruction);

		var triangulator = new IncrementalTriangulator(graph, reconstruction);

		int numTris = triangulator.Retriangulate(new IncrementalTriangulator.Options());

		await Assert.That(numPointsAfterDelete).IsEqualTo(0);
		await Assert.That(numTris).IsEqualTo(kNumPoints3D * reconstruction.NumRegImages);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(kNumPoints3D);
	}
}
