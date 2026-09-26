// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SequentialPairGenerator: port of colmap::SequentialPairGenerator
// (colmap/controllers/pairing.h/.cc) - pairs every image with its next `overlap` images in
// name order (or the images 2^i ahead with quadratic overlap), for video-like captures,
// optionally expanded to every image of the same and neighboring rig frames. Part of the
// pair generator family of Controllers/PairGenerator.cs; options in PairingOptions.cs.
// Tests: ColmapSharp.Tests/Controllers/PairingTests.cs (pairing_test.cc 1:1).
//
// Tier A (exact). Image order: COLMAP std::sorts the images by name, and names are unique
// (the database enforces it), so the unstable sort has no ties.
//
// Not supported: loop detection (options.LoopDetection) needs COLMAP's vocabulary-tree
// retrieval (colmap/retrieval), which is out of scope; the constructor throws
// NotSupportedException when it is enabled.

using System.Text;

using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>
/// Port of colmap::SequentialPairGenerator: pairs each image with its sequential (or
/// quadratic) successors in image name order, and optionally with its rig frame mates.
/// </summary>
public sealed class SequentialPairGenerator : PairGenerator
{
	private readonly SequentialPairingOptions options;
	private readonly FeatureMatcherCache cache;
	private readonly List<uint> imageIds;

	// Optional mapping from frames to images and vice versa (COLMAP's NodeHashMaps; only
	// looked up, never iterated, and each frame's list is in the frame's data id order).
	private readonly Dictionary<uint, List<uint>> frameToImageIds = [];
	private readonly Dictionary<uint, uint> imageToFrameId = [];
	private int imageIdx;

	/// <summary>A generator over the images of <paramref name="cache"/>.</summary>
	public SequentialPairGenerator(SequentialPairingOptions options, FeatureMatcherCache cache)
	{
		this.options = options;
		this.cache = Check.NotNull(cache);
		Check.That(options.Check());
		if (options.LoopDetection)
		{
			throw new NotSupportedException(
				"Sequential matching with loop detection needs a vocabulary tree, which ColmapSharp does not "
				+ "support. Turn off loop detection, or use exhaustive or spatial matching to find loops.");
		}

		imageIds = GetOrderedImageIds();

		if (options.ExpandRigImages)
		{
			foreach (uint frameId in cache.GetFrameIds())
			{
				Frame frame = cache.GetFrame(frameId);
				if (!frameToImageIds.TryGetValue(frameId, out List<uint>? frameImageIds))
				{
					frameImageIds = [];
					frameToImageIds.Add(frameId, frameImageIds);
				}

				foreach (var dataId in frame.ImageIds())
				{
					frameImageIds.Add((uint)dataId.Id);
					imageToFrameId[(uint)dataId.Id] = frameId;
				}
			}
		}
	}

	/// <summary>A generator over the images of <paramref name="database"/>.</summary>
	public SequentialPairGenerator(SequentialPairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset() => imageIdx = 0;

	/// <inheritdoc/>
	public override bool HasFinished() => imageIdx >= imageIds.Count;

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		var imagePairs = new List<(uint, uint)>();
		if (imageIdx >= imageIds.Count)
		{
			return imagePairs;
		}

		uint imageId1 = imageIds[imageIdx];

		// If image is part of a rig, then pair the other images in the same frame.
		if (options.ExpandRigImages && imageToFrameId.TryGetValue(imageId1, out uint frameId1))
		{
			foreach (uint frameImageId2 in frameToImageIds[frameId1])
			{
				if (imageId1 != frameImageId2)
				{
					imagePairs.Add((imageId1, frameImageId2));
				}
			}
		}

		for (int i = 0; i < options.Overlap; ++i)
		{
			// 1ull << i: a shift of 64 or more is UB in C++; COLMAP breaks out of the loop long
			// before that for any real image count, and so does this (the index overflows the
			// image count first).
			long imageIdx2 = options.QuadraticOverlap
				? (i < 63 ? imageIdx + (1L << i) : long.MaxValue)
				: imageIdx + i + 1L;
			if (imageIdx2 >= imageIds.Count)
			{
				break;
			}

			uint imageId2 = imageIds[(int)imageIdx2];
			if (!IsValidSequentialNeighbor(imageId1, imageId2))
			{
				continue;
			}

			imagePairs.Add((imageId1, imageId2));
			MaybeExpandRigImages(imagePairs, imageId1, imageId2);
		}

		++imageIdx;
		return imagePairs;
	}

	private void MaybeExpandRigImages(List<(uint, uint)> imagePairs, uint imageId1, uint imageId2)
	{
		if (!options.ExpandRigImages)
		{
			return;
		}

		if (imageToFrameId.TryGetValue(imageId2, out uint frameId2))
		{
			// Pair with all images in second frame.
			foreach (uint frameImageId2 in frameToImageIds[frameId2])
			{
				if (imageId1 != frameImageId2 && imageId2 != frameImageId2)
				{
					imagePairs.Add((imageId1, frameImageId2));
				}
			}
		}
	}

	private bool IsValidSequentialNeighbor(uint imageId1, uint imageId2)
	{
		if (!options.ExpandRigImages)
		{
			return true;
		}

		if (!imageToFrameId.TryGetValue(imageId1, out uint frameId1)
			|| !imageToFrameId.TryGetValue(imageId2, out uint frameId2))
		{
			return true;
		}

		if (frameToImageIds[frameId1].Count == 1 && frameToImageIds[frameId2].Count == 1)
		{
			return true;
		}

		// Rig images are sorted by their sensor-prefixed names. Crossing from the end of one
		// sensor's sequence to the beginning of the next would create a false temporal
		// neighbor. Same-frame sensor pairs are added in Next(), and temporal pairs are
		// expanded to the other sensors by MaybeExpandRigImages().
		return cache.GetImage(imageId1).CameraId == cache.GetImage(imageId2).CameraId;
	}

	// COLMAP std::sorts the images by name with std::string's operator<, a byte-wise
	// compare of the UTF-8 names, so that is the order here too (UTF-16 ordinal order
	// differs for supplementary-plane characters).
	private List<uint> GetOrderedImageIds()
	{
		var orderedImages = new List<(byte[] Name, uint ImageId)>();
		foreach (uint imageId in cache.GetImageIds())
		{
			orderedImages.Add((Encoding.UTF8.GetBytes(cache.GetImage(imageId).Name), imageId));
		}

		orderedImages.Sort((image1, image2) => image1.Name.AsSpan().SequenceCompareTo(image2.Name));

		var orderedImageIds = new List<uint>(orderedImages.Count);
		foreach ((_, uint imageId) in orderedImages)
		{
			orderedImageIds.Add(imageId);
		}

		return orderedImageIds;
	}
}
