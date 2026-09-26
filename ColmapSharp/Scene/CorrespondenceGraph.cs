// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CorrespondenceGraph: port of colmap/scene/correspondence_graph.h and .cc. For every image
// point it stores the points in other images it was matched to (verified inlier matches
// from TwoViewGeometry.cs), plus one TwoViewGeometry per image pair. The incremental mapper
// triangulates and merges tracks by walking it. Tests:
// ColmapSharp.Tests/Scene/CorrespondenceGraphTests.cs (correspondence_graph_test.cc 1:1).
//
// Tier A (exact): pure bookkeeping, and every per-point correspondence list is in the same
// order as COLMAP's (the order in which matches were added).
//
// Translation notes:
// - Hash-container order (docs/CPP_DIVERGENCES.md, entry 14): COLMAP keeps images and image
//   pairs in NodeHashMap / FlatHashMap, whose iteration order depends on the build's hash
//   backend (std or Boost). The only outputs that expose that order are ImagePairs() and
//   NumMatchesBetweenAllImages(). Here both are Dictionaries that are never removed from,
//   so they enumerate in insertion order: the order AddTwoViewGeometry was called in.
//   FinalizeGraph also walks the images, but each image is flattened independently, so its
//   order is unobservable.
// - CorrespondenceRange (a [beg, end) pointer pair) becomes a ReadOnlySpan over the
//   correspondence storage; it is valid until the graph is next modified.
// - Invalid and duplicate matches are dropped as in COLMAP. COLMAP also prints a
//   LOG(WARNING) for each; there is no logging sink in ColmapSharp yet, so they are dropped
//   silently (the counts reflect them exactly as in COLMAP).
// - A missing image throws KeyNotFoundException (COLMAP's std::out_of_range from map::at),
//   with COLMAP's message where it words one.

using System.Globalization;
using System.Runtime.InteropServices;

using ColmapSharp.Feature;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::CorrespondenceGraph: the feature correspondences between all images.
/// </summary>
public sealed class CorrespondenceGraph
{
	private readonly Dictionary<uint, ImageData> images = [];

	// Never removed from, so it enumerates in insertion order (header).
	private readonly Dictionary<ulong, ImagePair> imagePairs = [];

	private bool finalized;

	/// <summary>Number of added images.</summary>
	public int NumImages => images.Count;

	/// <summary>Number of added image pairs.</summary>
	public int NumImagePairs => imagePairs.Count;

	/// <summary>
	/// Finalizes the graph: checks the per-image observation counts and flattens every
	/// image's correspondences into one compact array. Can be called only once. Port of
	/// CorrespondenceGraph::Finalize, renamed because C# reserves Finalize for destructors.
	/// </summary>
	public void FinalizeGraph()
	{
		Check.That(!finalized);
		finalized = true;

		foreach (ImageData image in images.Values)
		{
			// Verify incremental num_observations tracking is consistent.
			int numTotalCorrs = 0;
			uint expectedNumObservations = 0;
			foreach (List<Correspondence> corr in image.Corrs)
			{
				numTotalCorrs += corr.Count;
				if (corr.Count > 0)
				{
					expectedNumObservations += 1;
				}
			}

			Check.Eq(image.NumObservations, expectedNumObservations);

			// Reshuffle correspondences into the flattened array.
			int numPoints2D = image.Corrs.Count;
			image.FlatCorrs = new Correspondence[numTotalCorrs];
			image.FlatCorrBegs = new int[numPoints2D + 1];
			int size = 0;
			for (int point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
			{
				image.FlatCorrBegs[point2DIdx] = size;
				List<Correspondence> corrs = image.Corrs[point2DIdx];
				corrs.CopyTo(image.FlatCorrs, size);
				size += corrs.Count;
			}

			image.FlatCorrBegs[numPoints2D] = size;

			// Deallocate original data.
			image.Corrs = [];
		}
	}

	/// <summary>
	/// The number of observations in an image: image points with at least one
	/// correspondence.
	/// </summary>
	public uint NumObservationsForImage(uint imageId) => GetImageOrThrow(imageId).NumObservations;

	/// <summary>The number of correspondences of an image to all other images.</summary>
	public uint NumCorrespondencesForImage(uint imageId) => GetImageOrThrow(imageId).NumCorrespondences;

	/// <summary>The number of matches between a pair of images (0 if the pair was never added).</summary>
	public uint NumMatchesBetweenImages(uint imageId1, uint imageId2)
	{
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		return imagePairs.TryGetValue(pairId, out ImagePair? pair) ? pair.NumMatches : 0;
	}

	/// <summary>The number of matches of every image pair, in the order the pairs were added.</summary>
	public Dictionary<ulong, uint> NumMatchesBetweenAllImages()
	{
		var numMatchesBetweenImages = new Dictionary<ulong, uint>(imagePairs.Count);
		foreach ((ulong pairId, ImagePair pair) in imagePairs)
		{
			numMatchesBetweenImages.Add(pairId, pair.NumMatches);
		}

		return numMatchesBetweenImages;
	}

	/// <summary>Whether the image was added.</summary>
	public bool ExistsImage(uint imageId) => images.ContainsKey(imageId);

	/// <summary>All image pair ids, in the order the pairs were added.</summary>
	public List<ulong> ImagePairs() => [.. imagePairs.Keys];

	/// <summary>Adds an image with <paramref name="numPoints"/> 2D points. It must not exist yet.</summary>
	public void AddImage(uint imageId, int numPoints)
	{
		Check.That(!ExistsImage(imageId));
		var image = new ImageData();
		for (int i = 0; i < numPoints; i++)
		{
			image.Corrs.Add([]);
		}

		images.Add(imageId, image);
	}

	/// <summary>
	/// Adds the two-view geometry and inlier matches between two added images. Matches whose
	/// point indices are out of range, and duplicate matches, are ignored. Self-matches
	/// (<paramref name="imageId1"/> == <paramref name="imageId2"/>) are ignored entirely.
	/// The graph keeps its own copy of the geometry, without the matches.
	/// </summary>
	public void AddTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		// Avoid self-matches - should only happen, if user provides custom matches.
		if (imageId1 == imageId2)
		{
			return;
		}

		// Corresponding images.
		ImageData image1 = GetImage(imageId1);
		ImageData image2 = GetImage(imageId2);

		List<FeatureMatch> inlierMatches = twoViewGeometry.InlierMatches;

		// Store number of correspondences for each image to find good initial pair.
		image1.NumCorrespondences += (uint)inlierMatches.Count;
		image2.NumCorrespondences += (uint)inlierMatches.Count;

		// Set the number of all correspondences for this image pair. Further below, we make
		// sure that only unique correspondences are counted.
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		var imagePair = new ImagePair();
		if (!imagePairs.TryAdd(pairId, imagePair))
		{
			// The message is only formatted on failure; this runs once per image pair.
			Check.That(
				false,
				string.Create(
					CultureInfo.InvariantCulture,
					$"Two view geometry for image pair was already added: image_id1={imageId1}, image_id2={imageId2}"),
				"inserted");
		}

		imagePair.NumMatches = (uint)inlierMatches.Count;

		// Store all matches in the correspondence graph data structure. This uses more memory
		// than storing the raw match matrices, but is significantly more efficient when
		// updating the correspondences in case an observation is triangulated.
		foreach (FeatureMatch match in inlierMatches)
		{
			bool validIdx1 = match.Point2DIdx1 < (uint)image1.Corrs.Count;
			bool validIdx2 = match.Point2DIdx2 < (uint)image2.Corrs.Count;

			if (validIdx1 && validIdx2)
			{
				List<Correspondence> corrs1 = image1.Corrs[(int)match.Point2DIdx1];
				List<Correspondence> corrs2 = image2.Corrs[(int)match.Point2DIdx2];

				// Valid correspondences are added bidirectionally, so checking from only one
				// side is sufficient to detect duplicated matches.
				// A plain loop, not List.Exists: a lambda would allocate a closure per match.
				bool duplicate = false;
				foreach (Correspondence corr in CollectionsMarshal.AsSpan(corrs1))
				{
					if (corr.ImageId == imageId2 && corr.Point2DIdx == match.Point2DIdx2)
					{
						duplicate = true;
						break;
					}
				}

				if (duplicate)
				{
					image1.NumCorrespondences -= 1;
					image2.NumCorrespondences -= 1;
					imagePair.NumMatches -= 1;
				}
				else
				{
					corrs1.Add(new Correspondence(imageId2, match.Point2DIdx2));
					// First correspondence makes this point an observation.
					if (corrs1.Count == 1)
					{
						image1.NumObservations += 1;
					}

					corrs2.Add(new Correspondence(imageId1, match.Point2DIdx1));
					if (corrs2.Count == 1)
					{
						image2.NumObservations += 1;
					}
				}
			}
			else
			{
				image1.NumCorrespondences -= 1;
				image2.NumCorrespondences -= 1;
				imagePair.NumMatches -= 1;
			}
		}

		imagePair.TwoViewGeometry = StoredGeometry(imageId1, imageId2, twoViewGeometry);
	}

	/// <summary>
	/// The correspondences of image point <paramref name="point2DIdx"/> to all other images.
	/// Valid until the graph is next modified.
	/// </summary>
	public ReadOnlySpan<Correspondence> FindCorrespondences(uint imageId, uint point2DIdx)
	{
		ImageData image = GetImage(imageId);
		if (!finalized)
		{
			// List's indexer throws when out of range, as vector::at does.
			return CollectionsMarshal.AsSpan(image.Corrs[checked((int)point2DIdx)]);
		}

		int beg = image.FlatCorrBegs[checked((int)point2DIdx)];
		int end = image.FlatCorrBegs[checked((int)point2DIdx + 1)];
		return image.FlatCorrs.AsSpan(beg, end - beg);
	}

	/// <summary>Copies the correspondences of an image point into <paramref name="corrs"/> (cleared first).</summary>
	public void ExtractCorrespondences(uint imageId, uint point2DIdx, List<Correspondence> corrs)
	{
		ReadOnlySpan<Correspondence> range = FindCorrespondences(imageId, point2DIdx);
		corrs.Clear();
		corrs.AddRange(range);
	}

	/// <summary>
	/// Transitively collects the correspondences of an image point: first its direct
	/// correspondences, then theirs, and so forth up to <paramref name="transitivity"/>
	/// levels or until no new ones are found. The result has no duplicates and does not
	/// contain the given observation itself.
	/// </summary>
	public void ExtractTransitiveCorrespondences(uint imageId, uint point2DIdx, int transitivity, List<Correspondence> corrs)
	{
		if (transitivity == 1)
		{
			ExtractCorrespondences(imageId, point2DIdx, corrs);
			return;
		}

		corrs.Clear();
		if (!HasCorrespondences(imageId, point2DIdx))
		{
			return;
		}

		// Push the requested image point on the queue to visit. Removed again below.
		corrs.Add(new Correspondence(imageId, point2DIdx));

		// Visited observations, for deduplication only; the output order is the discovery
		// order of `corrs`, never the set's.
		var imageCorrs = new HashSet<(uint, uint)>(PairHash.Instance) { (imageId, point2DIdx) };

		int corrQueueBeg = 0;
		int corrQueueEnd = 1;

		for (int t = 0; t < transitivity; ++t)
		{
			// Collect correspondences at transitive level t to all correspondences that were
			// collected at transitive level t - 1.
			for (int i = corrQueueBeg; i < corrQueueEnd; ++i)
			{
				Correspondence refCorr = corrs[i];
				foreach (Correspondence corr in FindCorrespondences(refCorr.ImageId, refCorr.Point2DIdx))
				{
					// Check if correspondence already collected, otherwise collect.
					if (imageCorrs.Add((corr.ImageId, corr.Point2DIdx)))
					{
						corrs.Add(corr);
					}
				}
			}

			// Move on to the next block of correspondences at the next transitive level.
			corrQueueBeg = corrQueueEnd;
			corrQueueEnd = corrs.Count;

			// No new correspondences collected in the last transitivity level.
			if (corrQueueBeg == corrQueueEnd)
			{
				break;
			}
		}

		// Remove the first element, which is the given observation, by swapping it with the
		// last collected correspondence.
		if (corrs.Count > 1)
		{
			corrs[0] = corrs[^1];
		}

		corrs.RemoveAt(corrs.Count - 1);
	}

	/// <summary>
	/// All matches between two images into <paramref name="matches"/> (cleared first),
	/// ordered by the point index in <paramref name="imageId1"/>.
	/// </summary>
	public void ExtractMatchesBetweenImages(uint imageId1, uint imageId2, List<FeatureMatch> matches)
	{
		matches.Clear();

		uint numCorrespondences = NumMatchesBetweenImages(imageId1, imageId2);
		if (numCorrespondences == 0)
		{
			return;
		}

		matches.EnsureCapacity((int)numCorrespondences);

		ImageData image1 = GetImage(imageId1);
		int numPoints2D1 = finalized ? image1.FlatCorrBegs.Length - 1 : image1.Corrs.Count;
		for (uint point2DIdx1 = 0; point2DIdx1 < (uint)numPoints2D1; ++point2DIdx1)
		{
			foreach (Correspondence corr in FindCorrespondences(imageId1, point2DIdx1))
			{
				if (corr.ImageId == imageId2)
				{
					matches.Add(new FeatureMatch(point2DIdx1, corr.Point2DIdx));
				}
			}
		}
	}

	/// <summary>
	/// The two-view geometry of an added pair, inverted if the ids are given in the other
	/// order, optionally with its inlier matches.
	/// </summary>
	public TwoViewGeometry ExtractTwoViewGeometry(uint imageId1, uint imageId2, bool extractInlierMatches)
	{
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		Check.That(imagePairs.TryGetValue(pairId, out ImagePair? imagePair));
		TwoViewGeometry twoViewGeometry = imagePair.TwoViewGeometry.Clone();
		if (Types.ShouldSwapImagePair(imageId1, imageId2))
		{
			twoViewGeometry.Invert();
		}

		// Extract after inversion, as they are extracted in the correct order.
		if (extractInlierMatches)
		{
			ExtractMatchesBetweenImages(imageId1, imageId2, twoViewGeometry.InlierMatches);
		}

		return twoViewGeometry;
	}

	/// <summary>
	/// Replaces the two-view geometry of an existing image pair. The inlier matches of the
	/// given geometry are ignored (matches are stored separately).
	/// </summary>
	public void UpdateTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		Check.That(imagePairs.TryGetValue(pairId, out ImagePair? imagePair));
		imagePair.TwoViewGeometry = StoredGeometry(imageId1, imageId2, twoViewGeometry);
	}

	/// <summary>Whether the image point has any correspondences.</summary>
	public bool HasCorrespondences(uint imageId, uint point2DIdx) => !FindCorrespondences(imageId, point2DIdx).IsEmpty;

	/// <summary>
	/// Whether the observation is part of a two-view track: it has exactly one
	/// correspondence, and that correspondence has exactly one correspondence (this one).
	/// </summary>
	public bool IsTwoViewObservation(uint imageId, uint point2DIdx)
	{
		ReadOnlySpan<Correspondence> range = FindCorrespondences(imageId, point2DIdx);
		if (range.Length != 1)
		{
			return false;
		}

		return FindCorrespondences(range[0].ImageId, range[0].Point2DIdx).Length == 1;
	}

	/// <summary>COLMAP's operator&lt;&lt;: "CorrespondenceGraph(num_images=n, num_image_pairs=m)".</summary>
	public override string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"CorrespondenceGraph(num_images={NumImages}, num_image_pairs={NumImagePairs})");
	}

	// The geometry as stored: a copy without matches (COLMAP takes it by value and clears
	// them), in the canonical smaller-id-first orientation.
	private static TwoViewGeometry StoredGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		TwoViewGeometry stored = twoViewGeometry.Clone();
		stored.InlierMatches = [];
		if (Types.ShouldSwapImagePair(imageId1, imageId2))
		{
			stored.Invert();
		}

		return stored;
	}

	// images_.at(image_id).
	private ImageData GetImage(uint imageId)
	{
		if (!images.TryGetValue(imageId, out ImageData? image))
		{
			throw new KeyNotFoundException(string.Create(CultureInfo.InvariantCulture, $"Image with ID {imageId} does not exist in the correspondence graph"));
		}

		return image;
	}

	// The accessors that word COLMAP's out_of_range message.
	private ImageData GetImageOrThrow(uint imageId)
	{
		if (!images.TryGetValue(imageId, out ImageData? image))
		{
			throw new KeyNotFoundException(string.Create(CultureInfo.InvariantCulture, $"Image with ID {imageId} does not exist"));
		}

		return image;
	}

	/// <summary>
	/// Port of CorrespondenceGraph::Correspondence: a corresponding point
	/// <see cref="Point2DIdx"/> in image <see cref="ImageId"/>. Defaults to invalid ids
	/// (stored plus one, as FeatureMatch.cs explains).
	/// </summary>
	public struct Correspondence : IEquatable<Correspondence>
	{
		private uint imageIdPlusOne;
		private uint point2DIdxPlusOne;

		/// <summary>A correspondence with InvalidImageId and InvalidPoint2DIdx.</summary>
		public Correspondence()
		{
		}

		/// <summary>The point <paramref name="point2DIdx"/> in image <paramref name="imageId"/>.</summary>
		public Correspondence(uint imageId, uint point2DIdx)
		{
			ImageId = imageId;
			Point2DIdx = point2DIdx;
		}

		/// <summary>The identifier of the corresponding image.</summary>
		public uint ImageId
		{
			readonly get => unchecked(imageIdPlusOne - 1);
			set => imageIdPlusOne = unchecked(value + 1);
		}

		/// <summary>The index of the corresponding point in the corresponding image.</summary>
		public uint Point2DIdx
		{
			readonly get => unchecked(point2DIdxPlusOne - 1);
			set => point2DIdxPlusOne = unchecked(value + 1);
		}

		/// <summary>Equality of both ids (C#-only; COLMAP's struct has no operator==).</summary>
		public static bool operator ==(Correspondence left, Correspondence right) => left.Equals(right);

		/// <summary>Inequality of either id.</summary>
		public static bool operator !=(Correspondence left, Correspondence right) => !left.Equals(right);

		/// <inheritdoc/>
		public readonly bool Equals(Correspondence other)
		{
			return imageIdPlusOne == other.imageIdPlusOne && point2DIdxPlusOne == other.point2DIdxPlusOne;
		}

		/// <inheritdoc/>
		public override readonly bool Equals(object? obj) => obj is Correspondence other && Equals(other);

		/// <inheritdoc/>
		public override readonly int GetHashCode() => PairHash.Instance.GetHashCode((ImageId, Point2DIdx));

		/// <summary>COLMAP's operator&lt;&lt;: "Correspondence(image_id=i, point2D_idx=p)".</summary>
		public override readonly string ToString()
		{
			return string.Create(CultureInfo.InvariantCulture, $"Correspondence(image_id={ImageId}, point2D_idx={Point2DIdx})");
		}
	}

	// CorrespondenceGraph::Image.
	private sealed class ImageData
	{
		// Number of 2D points with at least one correspondence to another image.
		public uint NumObservations;

		// Total number of correspondences to other images; used to find a good initial pair
		// connected to many images.
		public uint NumCorrespondences;

		// Correspondences to other images per image point. Filled before FinalizeGraph, empty
		// afterwards.
		public List<List<Correspondence>> Corrs = [];

		// Flattened correspondences after FinalizeGraph, empty before.
		public Correspondence[] FlatCorrs = [];

		// For each point, the start of its correspondences in FlatCorrs; the end of point i is
		// the start of point i + 1. Length num_points2D + 1, the last entry is FlatCorrs.Length.
		public int[] FlatCorrBegs = [];
	}

	// CorrespondenceGraph::ImagePair.
	private sealed class ImagePair
	{
		// The number of inlier matches between the pair of images.
		public uint NumMatches;

		// The two-view geometry of the image pair, without matches.
		public TwoViewGeometry TwoViewGeometry = new();
	}
}
