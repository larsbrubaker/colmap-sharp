// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PairingOptions: the option structs of colmap/controllers/pairing.h and their Check()s
// from pairing.cc, one per pair generator (Controllers/PairGenerator.cs,
// SequentialPairGenerator.cs, SpatialPairGenerator.cs). CacheSize() is the number of images
// the generator's FeatureMatcherCache (Controllers/FeatureMatcherCache.cs) holds when the
// generator is built straight from a Database. Tests:
// ColmapSharp.Tests/Controllers/PairingTests.cs (pairing_test.cc).
//
// Tier A (exact). Check() is CHECK_OPTION_*: false on a violation.
//
// Not ported: VocabTreePairingOptions (and SequentialPairingOptions::VocabTreeOptions,
// vocab_tree_path, num_threads) - vocabulary-tree retrieval (colmap/retrieval) is out of
// scope. SequentialPairingOptions keeps the loop_detection_* values COLMAP's Check()
// validates, so Check() stays 1:1, but enabling LoopDetection makes the
// SequentialPairGenerator throw NotSupportedException. FeaturePairsMatchingOptions belongs
// to the feature_matching controller and is ported with it.

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::ExhaustivePairingOptions.</summary>
public sealed class ExhaustivePairingOptions
{
	/// <summary>Block size, i.e. number of images to simultaneously load into memory.</summary>
	public int BlockSize { get; set; } = 50;

	/// <summary>Port of ExhaustivePairingOptions::Check.</summary>
	public bool Check() => BlockSize > 1;

	/// <summary>
	/// Each block matches two sets of images with size block_size. To hold all images in
	/// the block, the cache thus needs to hold 2 * block_size.
	/// </summary>
	public int CacheSize() => 2 * BlockSize;
}

/// <summary>Port of colmap::SequentialPairingOptions.</summary>
public sealed class SequentialPairingOptions
{
	/// <summary>Number of overlapping image pairs.</summary>
	public int Overlap { get; set; } = 10;

	/// <summary>Whether to match images against their quadratic neighbors.</summary>
	public bool QuadraticOverlap { get; set; } = true;

	/// <summary>
	/// Whether to match an image against all images within the same rig frame and all images
	/// in neighboring rig frames. This assumes images are named rig/camera/imageNNNN.jpg, so
	/// that for overlap=1, rig1/camera1/image0001.jpg is matched against the other cameras'
	/// image0001.jpg (same frame) and every camera's image0002.jpg (neighboring frame). If no
	/// rigs/frames are configured in the database, this option is ignored.
	/// </summary>
	public bool ExpandRigImages { get; set; } = true;

	/// <summary>
	/// Whether to enable vocabulary tree based loop detection. Not supported: the retrieval
	/// module is not ported, and a generator built with this set throws.
	/// </summary>
	public bool LoopDetection { get; set; }

	/// <summary>The frequency at which loop detection is triggered, in number of images.</summary>
	public int LoopDetectionPeriod { get; set; } = 10;

	/// <summary>The number of images to retrieve in loop detection.</summary>
	public int LoopDetectionNumImages { get; set; } = 50;

	/// <summary>
	/// The minimum image index distance between a loop detection query and a retrieved
	/// image. 0 disables this restriction.
	/// </summary>
	public int LoopDetectionMinIndexDistance { get; set; }

	/// <summary>Number of nearest neighbors to retrieve per query feature.</summary>
	public int LoopDetectionNumNearestNeighbors { get; set; } = 1;

	/// <summary>Number of nearest-neighbor checks to use in retrieval.</summary>
	public int LoopDetectionNumChecks { get; set; } = 64;

	/// <summary>Port of SequentialPairingOptions::Check.</summary>
	public bool Check() =>
		Overlap > 0
		&& LoopDetectionPeriod > 0
		&& LoopDetectionNumImages > 0
		&& LoopDetectionMinIndexDistance >= 0
		&& LoopDetectionNumNearestNeighbors > 0
		&& LoopDetectionNumChecks > 0;

	/// <summary>The number of images the generator's feature cache holds.</summary>
	public int CacheSize() => Math.Max(5 * LoopDetectionNumImages, 5 * Overlap);
}

/// <summary>Port of colmap::SpatialPairingOptions.</summary>
public sealed class SpatialPairingOptions
{
	/// <summary>Whether to ignore the Z-component of the location prior.</summary>
	public bool IgnoreZ { get; set; } = true;

	/// <summary>The maximum number of nearest neighbors to match.</summary>
	public int MaxNumNeighbors { get; set; } = 50;

	/// <summary>
	/// The minimum number of nearest neighbors to match. Neighbors include those within
	/// MaxDistance or to satisfy MinNumNeighbors.
	/// </summary>
	public int MinNumNeighbors { get; set; }

	/// <summary>
	/// The maximum distance between the query and nearest neighbor. For GPS coordinates the
	/// unit is Euclidean distance in meters.
	/// </summary>
	public double MaxDistance { get; set; } = 100;

	/// <summary>Port of SpatialPairingOptions::Check.</summary>
	public bool Check() =>
		MaxDistance >= 0.0
		&& MaxNumNeighbors > 0
		&& MinNumNeighbors <= MaxNumNeighbors
		&& MinNumNeighbors >= 0
		&& (MaxDistance > 0.0 || MinNumNeighbors > 0);

	/// <summary>The number of images the generator's feature cache holds.</summary>
	public int CacheSize() => 5 * MaxNumNeighbors;
}

/// <summary>Port of colmap::TransitivePairingOptions.</summary>
public sealed class TransitivePairingOptions
{
	/// <summary>The maximum number of image pairs to process in one batch.</summary>
	public int BatchSize { get; set; } = 1000;

	/// <summary>The number of transitive closure iterations.</summary>
	public int NumIterations { get; set; } = 3;

	/// <summary>Port of TransitivePairingOptions::Check.</summary>
	public bool Check() => BatchSize > 0 && NumIterations > 0;

	/// <summary>The number of images the generator's feature cache holds.</summary>
	public int CacheSize() => 2 * BatchSize;
}

/// <summary>Port of colmap::ImportedPairingOptions.</summary>
public sealed class ImportedPairingOptions
{
	/// <summary>Number of image pairs to match in one batch.</summary>
	public int BlockSize { get; set; } = 1225;

	/// <summary>Path to the file with the image pairs ("name1 name2" per line).</summary>
	public string MatchListPath { get; set; } = "";

	/// <summary>Port of ImportedPairingOptions::Check.</summary>
	public bool Check() => BlockSize > 0;

	/// <summary>The number of images the generator's feature cache holds.</summary>
	public int CacheSize() => BlockSize;
}

/// <summary>Port of colmap::ExistingMatchedPairingOptions.</summary>
public sealed class ExistingMatchedPairingOptions
{
	/// <summary>The number of image pairs to match in one batch.</summary>
	public int BatchSize { get; set; } = 1000;

	/// <summary>Port of ExistingMatchedPairingOptions::Check.</summary>
	public bool Check() => BatchSize > 1;

	/// <summary>The number of images the generator's feature cache holds.</summary>
	public int CacheSize() => Math.Max(10, (int)(2 * Math.Sqrt(BatchSize)));
}
