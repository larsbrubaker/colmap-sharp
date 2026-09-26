// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Synthetic.Matches: the match synthesis half of colmap/scene/synthetic.cc (the anonymous
// namespace helpers AddOutlierMatches .. SynthesizeSparseMatches). SynthesizeDataset
// (Synthetic.cs) calls one of the three Synthesize*Matches when it has a database. Each pair
// gets its true inlier matches (2D points that observe the same 3D point), optional random
// outliers, and a two-view geometry with the exact E (and F) of the synthetic poses.
//
// Order: the chained pairs are collected in a NodeHashMap in COLMAP and written in its hash
// order; here they go in ascending pair id order (docs/CPP_DIVERGENCES.md entry 31). Since
// every pair's matches are shuffled with the global PRNG, that changes which pair gets which
// shuffle, not how many draws are made. Exhaustive and sparse pairs are ordered the same way
// in both (registered-image order and std::set order).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

public static partial class Synthetic
{
	private static void AddOutlierMatches(double inlierRatio, uint numPoints2D1, uint numPoints2D2, List<FeatureMatch> matches)
	{
		int numOutliers = (int)(matches.Count * (1.0 - inlierRatio));
		for (int i = 0; i < numOutliers; ++i)
		{
			// COLMAP draws the second index from [0, n2 - 2], not n2 - 1; kept as is.
			uint idx1 = RandomUtils.RandomUniformInteger(0u, unchecked(numPoints2D1 - 1));
			uint idx2 = RandomUtils.RandomUniformInteger(0u, unchecked(numPoints2D2 - 2));
			matches.Add(new FeatureMatch(idx1, idx2));
		}

		LibcxxRandom.Shuffle(matches, Check.NotNull(RandomUtils.Prng));
	}

	private static List<ulong> ExtractExhaustiveImagePairs(Reconstruction reconstruction)
	{
		int numRegImages = reconstruction.NumRegImages;
		int numExhaustivePairs = numRegImages * (numRegImages - 1) / 2;
		var imagePairs = new List<ulong>(numExhaustivePairs);
		List<uint> regImageIds = reconstruction.RegImageIds();
		foreach (uint imageId1 in regImageIds)
		{
			foreach (uint imageId2 in regImageIds)
			{
				if (imageId1 >= imageId2)
				{
					continue;
				}

				imagePairs.Add(ImagePairToPairId(imageId1, imageId2));
			}
		}

		Check.Eq(imagePairs.Count, numExhaustivePairs);
		return imagePairs;
	}

	// Fill the configuration, essential matrix, and (for perspective pairs) fundamental
	// matrix of a synthetic two-view geometry.
	private static void SetTwoViewGeometryModel(Camera camera1, Camera camera2, Rigid3d cam2FromCam1, TwoViewGeometry twoViewGeometry)
	{
		twoViewGeometry.E = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
		if (camera1.IsSpherical || camera2.IsSpherical)
		{
			// Omnidirectional cameras (e.g. EQUIRECTANGULAR) have no pinhole calibration
			// matrix, so the fundamental matrix is undefined; they are always calibrated and
			// use the bearing-based essential-matrix configuration.
			twoViewGeometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
			return;
		}

		bool isCalibrated = camera1.HasPriorFocalLength && camera2.HasPriorFocalLength;
		twoViewGeometry.Config = isCalibrated
			? TwoViewGeometry.ConfigurationType.Calibrated
			: TwoViewGeometry.ConfigurationType.Uncalibrated;
		twoViewGeometry.F = EssentialMatrix.FundamentalFromEssentialMatrix(
			camera2.CalibrationMatrix(), twoViewGeometry.E.Value, camera1.CalibrationMatrix());
	}

	private static TwoViewGeometry BuildTwoViewGeometry(bool hasRelativePose, Reconstruction reconstruction, ulong pairId)
	{
		var (imageId1, imageId2) = PairIdToImagePair(pairId);
		Image image1 = reconstruction.Image(imageId1);
		Image image2 = reconstruction.Image(imageId2);

		var twoViewGeometry = new TwoViewGeometry();
		Rigid3d cam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse();
		if (hasRelativePose)
		{
			twoViewGeometry.Cam2FromCam1 = cam2FromCam1;
		}

		SetTwoViewGeometryModel(image1.CameraPtr, image2.CameraPtr, cam2FromCam1, twoViewGeometry);

		List<Point2D> points2D1 = image1.Points2D;
		List<Point2D> points2D2 = image2.Points2D;
		for (int point2DIdx1 = 0; point2DIdx1 < points2D1.Count; ++point2DIdx1)
		{
			Point2D point2D1 = points2D1[point2DIdx1];
			if (!point2D1.HasPoint3D)
			{
				continue;
			}

			for (int point2DIdx2 = 0; point2DIdx2 < points2D2.Count; ++point2DIdx2)
			{
				if (point2D1.Point3DId == points2D2[point2DIdx2].Point3DId)
				{
					twoViewGeometry.InlierMatches.Add(new FeatureMatch((uint)point2DIdx1, (uint)point2DIdx2));
					break;
				}
			}
		}

		return twoViewGeometry;
	}

	private static void WriteTwoViewGeometryToDatabase(
		ulong pairId,
		TwoViewGeometry twoViewGeometry,
		double inlierMatchRatio,
		Reconstruction reconstruction,
		Database database)
	{
		var (imageId1, imageId2) = PairIdToImagePair(pairId);
		Image image1 = reconstruction.Image(imageId1);
		Image image2 = reconstruction.Image(imageId2);

		var matches = new List<FeatureMatch>(twoViewGeometry.InlierMatches);
		AddOutlierMatches(inlierMatchRatio, image1.NumPoints2D, image2.NumPoints2D, matches);

		if (!database.ExistsMatches(imageId1, imageId2))
		{
			database.WriteMatches(imageId1, imageId2, matches);
		}

		if (!database.ExistsTwoViewGeometry(imageId1, imageId2))
		{
			database.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		}
	}

	private static void SynthesizeExhaustiveMatches(double inlierMatchRatio, bool hasRelativePose, Reconstruction reconstruction, Database database)
	{
		foreach (ulong pairId in ExtractExhaustiveImagePairs(reconstruction))
		{
			TwoViewGeometry twoViewGeometry = BuildTwoViewGeometry(hasRelativePose, reconstruction, pairId);
			WriteTwoViewGeometryToDatabase(pairId, twoViewGeometry, inlierMatchRatio, reconstruction, database);
		}
	}

	private static void SynthesizeChainedMatches(double inlierMatchRatio, bool hasRelativePose, Reconstruction reconstruction, Database database)
	{
		// Ascending pair id order; COLMAP's NodeHashMap iterates in hash order (entry 31).
		var twoViewGeometries = new SortedDictionary<ulong, TwoViewGeometry>();
		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			// std::sort by image id; the order of equal image ids cannot reach the output
			// since only consecutive image ids form a pair and an image observes a point once.
			var trackElements = new List<TrackElement>(point3D.Track.Elements);
			trackElements.Sort((left, right) => left.ImageId.CompareTo(right.ImageId));
			for (int i = 1; i < trackElements.Count; ++i)
			{
				TrackElement prevTrackEl = trackElements[i - 1];
				TrackElement currTrackEl = trackElements[i];
				if (currTrackEl.ImageId != prevTrackEl.ImageId + 1)
				{
					continue;
				}

				ulong pairId = ImagePairToPairId(prevTrackEl.ImageId, currTrackEl.ImageId);
				if (!twoViewGeometries.TryGetValue(pairId, out TwoViewGeometry? twoViewGeometry))
				{
					twoViewGeometry = new TwoViewGeometry();
					twoViewGeometries.Add(pairId, twoViewGeometry);
				}

				twoViewGeometry.InlierMatches.Add(ShouldSwapImagePair(prevTrackEl.ImageId, currTrackEl.ImageId)
					? new FeatureMatch(currTrackEl.Point2DIdx, prevTrackEl.Point2DIdx)
					: new FeatureMatch(prevTrackEl.Point2DIdx, currTrackEl.Point2DIdx));
			}
		}

		foreach (var (pairId, twoViewGeometry) in twoViewGeometries)
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			Image image1 = reconstruction.Image(imageId1);
			Image image2 = reconstruction.Image(imageId2);
			Rigid3d cam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse();
			if (hasRelativePose)
			{
				twoViewGeometry.Cam2FromCam1 = cam2FromCam1;
			}

			SetTwoViewGeometryModel(image1.CameraPtr, image2.CameraPtr, cam2FromCam1, twoViewGeometry);
			WriteTwoViewGeometryToDatabase(pairId, twoViewGeometry, inlierMatchRatio, reconstruction, database);
		}
	}

	private static bool IsViewGraphConnected(SortedSet<uint> images, SortedSet<ulong> imagePairs)
	{
		if (images.Count <= 1)
		{
			return true;
		}

		if (imagePairs.Count == 0)
		{
			return false;
		}

		// Use UnionFind to check connectivity.
		var uf = new UnionFind<uint>();
		uf.Reserve(images.Count);
		foreach (ulong pairId in imagePairs)
		{
			var (imageId1, imageId2) = PairIdToImagePair(pairId);
			uf.Union(imageId1, imageId2);
		}

		// Check that all images have the same root.
		uint root = uf.Find(images.Min);
		foreach (uint node in images)
		{
			if (uf.Find(node) != root)
			{
				return false;
			}
		}

		return true;
	}

	private static void SynthesizeSparseMatches(
		double inlierMatchRatio,
		bool hasRelativePose,
		double sparsity,
		Reconstruction reconstruction,
		Database database)
	{
		Check.Ge(sparsity, 0.0);
		Check.Le(sparsity, 1.0);

		if (sparsity == 0.0)
		{
			SynthesizeExhaustiveMatches(inlierMatchRatio, hasRelativePose, reconstruction, database);
			return;
		}

		if (sparsity == 1.0)
		{
			return;
		}

		List<ulong> remainingImagePairs = ExtractExhaustiveImagePairs(reconstruction);
		var allImageIds = new SortedSet<uint>(reconstruction.RegImageIds());

		int numEdgesTotal = remainingImagePairs.Count;
		int numEdgesToRemove = (int)(sparsity * numEdgesTotal);

		// Try to remove edges randomly while maintaining connectivity.
		LibcxxRandom.Shuffle(remainingImagePairs, Check.NotNull(RandomUtils.Prng));
		var remainingEdgesSet = new SortedSet<ulong>(remainingImagePairs);
		int edgesRemoved = 0;
		foreach (ulong pairId in remainingImagePairs)
		{
			if (edgesRemoved >= numEdgesToRemove)
			{
				break;
			}

			remainingEdgesSet.Remove(pairId);
			if (IsViewGraphConnected(allImageIds, remainingEdgesSet))
			{
				edgesRemoved++;
			}
			else
			{
				remainingEdgesSet.Add(pairId);
			}
		}

		foreach (ulong pairId in remainingEdgesSet)
		{
			TwoViewGeometry twoViewGeometry = BuildTwoViewGeometry(hasRelativePose, reconstruction, pairId);
			WriteTwoViewGeometryToDatabase(pairId, twoViewGeometry, inlierMatchRatio, reconstruction, database);
		}
	}
}
