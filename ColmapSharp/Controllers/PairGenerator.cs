// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PairGenerator: port of the PairGenerator interface of colmap/controllers/pairing.h/.cc and
// the generators that only enumerate ids: ExhaustivePairGenerator, TransitivePairGenerator,
// ImportedPairGenerator (with ReadImagePairsText) and ExistingMatchedPairGenerator. The
// sequential and spatial generators are SequentialPairGenerator.cs and
// SpatialPairGenerator.cs; their options are PairingOptions.cs. Every generator reads the
// database through a FeatureMatcherCache (Controllers/FeatureMatcherCache.cs). Tests:
// ColmapSharp.Tests/Controllers/PairingTests.cs (pairing_test.cc 1:1).
//
// Tier A (exact): the pair order decides the matching order, so it must be COLMAP's. It
// is: image ids come sorted from the cache, and the Database's pair reads
// (ReadNumMatches, ReadTwoViewGeometryNumInliers) come in pair-id order, the rowid order of
// COLMAP's SQLite SELECTs.
//
// Translation notes:
// - std::pair<image_t, image_t> is a (uint, uint) tuple; Next() returns a new list.
// - The constructors taking a std::shared_ptr<Database> take a Database and build a
//   FeatureMatcherCache of options.CacheSize() images, like COLMAP.
// - LOG(INFO) progress lines are not ported (there is no logging sink); LOG(ERROR) for an
//   unknown image name in a pair list is dropped the same way: the line is skipped.
// - VocabTreePairGenerator is not ported: vocabulary-tree retrieval (colmap/retrieval) is
//   out of scope.

using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::PairGenerator: yields the image pairs to match, block by block.</summary>
public abstract class PairGenerator
{
	/// <summary>Restarts the generation from the first block.</summary>
	public abstract void Reset();

	/// <summary>Whether every block has been returned.</summary>
	public abstract bool HasFinished();

	/// <summary>The next block of image pairs; empty once finished.</summary>
	public abstract List<(uint ImageId1, uint ImageId2)> Next();

	/// <summary>Port of PairGenerator::AllPairs: every remaining block, concatenated.</summary>
	public List<(uint ImageId1, uint ImageId2)> AllPairs()
	{
		var imagePairs = new List<(uint, uint)>();
		while (!HasFinished())
		{
			imagePairs.AddRange(Next());
		}

		return imagePairs;
	}
}

/// <summary>
/// Port of colmap::ExhaustivePairGenerator: every pair of images, in blocks of
/// BlockSize x BlockSize images.
/// </summary>
public sealed class ExhaustivePairGenerator : PairGenerator
{
	private readonly List<uint> imageIds;
	private readonly int blockSize;
	private int startIdx1;
	private int startIdx2;

	/// <summary>A generator over the images of <paramref name="cache"/>.</summary>
	public ExhaustivePairGenerator(ExhaustivePairingOptions options, FeatureMatcherCache cache)
	{
		imageIds = Check.NotNull(cache).GetImageIds();
		blockSize = options.BlockSize;
		Check.That(options.Check());
	}

	/// <summary>A generator over the images of <paramref name="database"/>.</summary>
	public ExhaustivePairGenerator(ExhaustivePairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset()
	{
		startIdx1 = 0;
		startIdx2 = 0;
	}

	/// <inheritdoc/>
	public override bool HasFinished() => startIdx1 >= imageIds.Count;

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		var imagePairs = new List<(uint, uint)>();
		if (HasFinished())
		{
			return imagePairs;
		}

		int endIdx1 = Math.Min(imageIds.Count, startIdx1 + blockSize) - 1;
		int endIdx2 = Math.Min(imageIds.Count, startIdx2 + blockSize) - 1;

		for (int idx1 = startIdx1; idx1 <= endIdx1; ++idx1)
		{
			for (int idx2 = startIdx2; idx2 <= endIdx2; ++idx2)
			{
				int blockId1 = idx1 % blockSize;
				int blockId2 = idx2 % blockSize;
				// Avoid duplicate pairs.
				if ((idx1 > idx2 && blockId1 <= blockId2) || (idx1 < idx2 && blockId1 < blockId2))
				{
					imagePairs.Add((imageIds[idx1], imageIds[idx2]));
				}
			}
		}

		startIdx2 += blockSize;
		if (startIdx2 >= imageIds.Count)
		{
			startIdx2 = 0;
			startIdx1 += blockSize;
		}

		return imagePairs;
	}
}

/// <summary>
/// Port of colmap::TransitivePairGenerator: in each iteration, pairs every two images that
/// share a verified neighbor but are not verified themselves.
/// </summary>
public sealed class TransitivePairGenerator : PairGenerator
{
	private readonly TransitivePairingOptions options;
	private readonly FeatureMatcherCache cache;
	private readonly List<(uint, uint)> imagePairs = [];
	private readonly HashSet<ulong> imagePairIds = [];
	private int currentIteration;

	/// <summary>A generator over the two-view geometries of <paramref name="cache"/>'s database.</summary>
	public TransitivePairGenerator(TransitivePairingOptions options, FeatureMatcherCache cache)
	{
		this.options = options;
		this.cache = Check.NotNull(cache);
		Check.That(options.Check());
	}

	/// <summary>A generator over the two-view geometries of <paramref name="database"/>.</summary>
	public TransitivePairGenerator(TransitivePairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset()
	{
		currentIteration = 0;
		imagePairs.Clear();
		imagePairIds.Clear();
	}

	/// <inheritdoc/>
	public override bool HasFinished() => currentIteration >= options.NumIterations && imagePairs.Count == 0;

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		if (imagePairs.Count != 0)
		{
			// Batches are taken from the back, as COLMAP pops them off its vector.
			var batch = new List<(uint, uint)>();
			while (imagePairs.Count != 0 && batch.Count < options.BatchSize)
			{
				batch.Add(imagePairs[^1]);
				imagePairs.RemoveAt(imagePairs.Count - 1);
			}

			return batch;
		}

		if (currentIteration >= options.NumIterations)
		{
			return [];
		}

		currentIteration++;

		List<(ulong PairId, int NumInliers)> existingPairIdsAndNumInliers = [];
		cache.AccessDatabase(database => existingPairIdsAndNumInliers = database.ReadTwoViewGeometryNumInliers());

		// std::map: images visited in id order, neighbors in pair-id read order.
		var adjacency = new SortedDictionary<uint, List<uint>>();
		foreach ((ulong pairId, _) in existingPairIdsAndNumInliers)
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			AdjacencyOf(adjacency, imageId1).Add(imageId2);
			AdjacencyOf(adjacency, imageId2).Add(imageId1);
			imagePairIds.Add(pairId);
		}

		foreach ((uint imageId1, List<uint> neighbors) in adjacency)
		{
			foreach (uint imageId2 in neighbors)
			{
				if (!adjacency.TryGetValue(imageId2, out List<uint>? neighbors2))
				{
					continue;
				}

				foreach (uint imageId3 in neighbors2)
				{
					if (imageId1 == imageId3)
					{
						continue;
					}

					ulong imagePairId = ImagePairToPairId(imageId1, imageId3);
					if (imagePairIds.Contains(imagePairId))
					{
						continue;
					}

					imagePairs.Add((Math.Min(imageId1, imageId3), Math.Max(imageId1, imageId3)));
					imagePairIds.Add(imagePairId);
				}
			}
		}

		return Next();
	}

	private static List<uint> AdjacencyOf(SortedDictionary<uint, List<uint>> adjacency, uint imageId)
	{
		if (!adjacency.TryGetValue(imageId, out List<uint>? neighbors))
		{
			neighbors = [];
			adjacency.Add(imageId, neighbors);
		}

		return neighbors;
	}
}

/// <summary>
/// Port of colmap::ImportedPairGenerator: the pairs listed in a text file, one
/// "name1 name2" pair per line, in blocks of BlockSize pairs.
/// </summary>
public sealed class ImportedPairGenerator : PairGenerator
{
	private readonly ImportedPairingOptions options;
	private readonly List<(uint, uint)> imagePairs;
	private int pairIdx;

	/// <summary>A generator over the pair list, resolving names through <paramref name="cache"/>.</summary>
	public ImportedPairGenerator(ImportedPairingOptions options, FeatureMatcherCache cache)
	{
		this.options = options;
		Check.That(options.Check());

		var imageNameToImageId = new Dictionary<string, uint>();
		foreach (uint imageId in cache.GetImageIds())
		{
			imageNameToImageId.TryAdd(cache.GetImage(imageId).Name, imageId);
		}

		imagePairs = ReadImagePairsText(options.MatchListPath, imageNameToImageId);
	}

	/// <summary>A generator over the pair list, resolving names through <paramref name="database"/>.</summary>
	public ImportedPairGenerator(ImportedPairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset() => pairIdx = 0;

	/// <inheritdoc/>
	public override bool HasFinished() => pairIdx >= imagePairs.Count;

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		var blockImagePairs = new List<(uint, uint)>();
		if (HasFinished())
		{
			return blockImagePairs;
		}

		int blockEnd = (int)Math.Min((long)pairIdx + options.BlockSize, imagePairs.Count);
		for (int j = pairIdx; j < blockEnd; ++j)
		{
			blockImagePairs.Add(imagePairs[j]);
		}

		pairIdx += options.BlockSize;
		return blockImagePairs;
	}

	// Port of the anonymous ReadImagePairsText: each non-empty, non-'#' line is split at its
	// first two spaces (std::getline with ' '); unknown names skip the line, and a pair
	// already listed (in either order) is kept only once, at its first occurrence.
	private static List<(uint, uint)> ReadImagePairsText(string path, Dictionary<string, uint> imageNameToImageId)
	{
		List<(string Text, bool ValidUtf8)> lines;
		using (FileStream stream = FileOpen.OpenRead(path))
		{
			lines = CppLineTokens.ReadLines(stream);
		}

		var imagePairs = new List<(uint, uint)>();
		var imagePairsSet = new HashSet<ulong>();
		foreach ((string text, _) in lines)
		{
			string line = CppLineTokens.Trim(text);
			if (line.Length == 0 || line[0] == '#')
			{
				continue;
			}

			int space1 = line.IndexOf(' ');
			string imageName1 = CppLineTokens.Trim(space1 < 0 ? line : line[..space1]);
			string imageName2 = "";
			if (space1 >= 0)
			{
				string rest = line[(space1 + 1)..];
				int space2 = rest.IndexOf(' ');
				imageName2 = CppLineTokens.Trim(space2 < 0 ? rest : rest[..space2]);
			}

			if (!imageNameToImageId.TryGetValue(imageName1, out uint imageId1)
				|| !imageNameToImageId.TryGetValue(imageName2, out uint imageId2))
			{
				continue;
			}

			if (imagePairsSet.Add(ImagePairToPairId(imageId1, imageId2)))
			{
				imagePairs.Add((imageId1, imageId2));
			}
		}

		return imagePairs;
	}
}

/// <summary>
/// Port of colmap::ExistingMatchedPairGenerator: the pairs that already have raw matches in
/// the database, in batches of BatchSize pairs.
/// </summary>
public sealed class ExistingMatchedPairGenerator : PairGenerator
{
	private readonly ExistingMatchedPairingOptions options;
	private readonly List<(uint, uint)> imagePairs = [];
	private int startIdx;

	/// <summary>A generator over the matched pairs of <paramref name="cache"/>'s database.</summary>
	public ExistingMatchedPairGenerator(ExistingMatchedPairingOptions options, FeatureMatcherCache cache)
	{
		this.options = options;
		Check.That(options.Check());
		cache.AccessDatabase(database =>
		{
			foreach ((ulong pairId, _) in database.ReadNumMatches())
			{
				imagePairs.Add(PairIdToImagePair(pairId));
			}
		});
	}

	/// <summary>A generator over the matched pairs of <paramref name="database"/>.</summary>
	public ExistingMatchedPairGenerator(ExistingMatchedPairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset() => startIdx = 0;

	/// <inheritdoc/>
	public override bool HasFinished() => startIdx >= imagePairs.Count;

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		if (HasFinished())
		{
			return [];
		}

		int endIdx = (int)Math.Min((long)startIdx + options.BatchSize, imagePairs.Count);
		List<(uint ImageId1, uint ImageId2)> batch = imagePairs.GetRange(startIdx, endIdx - startIdx);
		startIdx = endIdx;
		return batch;
	}
}
