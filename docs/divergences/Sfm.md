# Divergences from COLMAP: Sfm: incremental and global mapping

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 50. ObservationManager iterates its image pairs in insertion order and prints its graph

**What differs.** COLMAP's `ObservationManager` (sfm/observation_manager.cc) keeps the
per-pair statistics in an absl `FlatHashMap`, so `ImagePairs()` iterates in hash order, which
abseil seeds per process. The port (`Sfm/ObservationManager.cs`) keeps them in a `Dictionary`
that is only ever added to, so `ImagePairs` iterates in insertion order: the correspondence
graph's pair order (`CorrespondenceGraph.NumMatchesBetweenAllImages`), then the pairs
`AddImage` added, visiting existing images in the order the manager added them (ascending
image id at construction, entry 21, then `AddImage` order). Separately, COLMAP's `operator<<`
streams the correspondence graph's `shared_ptr`, i.e. its address; `ToString` prints the
graph's own `ToString` instead ("null" when there is none, as in C++).

**Why.** CLAUDE.md requires reproducible results, and the hash order carries no meaning. No
count inside the manager depends on it; the consumer that iterates `ImagePairs`
(`IncrementalTriangulator::Retriangulate`) visits pairs in that order, so the port's
retriangulation order is deterministic where COLMAP's varies from run to run. .NET has no
stable object address to print.

**Evidence.** `ObservationManagerTests` (observation_manager_test.cc 1:1, including `Print`,
which has no graph) pass.

## 51. IncrementalTriangulator visits points and pairs in a deterministic order

**What differs.** COLMAP's `IncrementalTriangulator` (sfm/incremental_triangulator.cc)
iterates hash containers whose order reaches its output: `CompleteAllTracks` and
`MergeAllTracks` walk `Reconstruction::Point3DIds()` (an unordered set), `Retriangulate` walks
`ObservationManager::ImagePairs()` (an absl flat map), and `CompleteTracks`/`MergeTracks` walk
the caller's `FlatHashSet`. Which point is merged or completed first decides which merges
succeed (a merged point's average position changes the next test) and which observations a
track claims; which pair is retriangulated first decides which pair continues or creates a
point. The port (`Sfm/IncrementalTriangulator*.cs`) walks `Point3DIds` in ascending id order
(entry 21), `ImagePairs` in insertion order (entry 50), and the caller's collection in its own
enumeration order. The modified-point set is a `HashSet`, whose order depends only on the
sequence of adds and removes, so it is the same on every run.

**Why.** CLAUDE.md requires reproducible results; abseil seeds its hash per process, so
COLMAP's order carries no meaning and varies run to run.

**Evidence.** `IncrementalTriangulatorTests` (incremental_triangulator_test.cc 1:1) pass;
their counts do not depend on the order on the noise-free synthetic scenes.

## 58. IncrementalMapper searches for the initial pair sequentially

**What differs.** COLMAP's `IncrementalMapperImpl::FindInitialImagePair`
(sfm/incremental_mapper_impl.cc) starts one thread-pool task per seed image; each task walks
its second images, claims every pair in the shared `init_image_pairs` set under a mutex, and
sets a shared `stop` flag when it finds a good pair. The caller returns the first successful
result in seed order. With more than one thread, a later seed's task can claim a pair (or be
stopped) before an earlier seed's task reaches it, so which pairs are tried, which are marked
as tried, and therefore which pair is returned depend on thread timing. The port
(`Sfm/IncrementalMapperImpl.cs`) runs the seeds in order and stops at the first success, which
is COLMAP's result with `num_threads = 1`.

**Why.** CLAUDE.md requires sequential and parallel runs to give the same result; the parallel
search is only a speed-up, and its race changes both the returned pair and the
`init_image_pairs` state later calls see.

**Evidence.** `IncrementalMapperTests` (incremental_mapper_test.cc 1:1) pass, including
`FullPipeline`, `EstimateInitialTwoViewGeometry` and `ResetInitializationStats`, which go
through this search.

## 59. IncrementalMapper breaks ranking ties by image id

**What differs.** COLMAP ranks images with `std::sort` (unstable) over inputs in hash-map
order: seed images in `FindFirstInitialImage` (prior focal length, then correspondences) and
second images in `FindSecondInitialImage` (over a `FlatHashMap` of correspondence counts),
the next images in `FindNextImages` (by rank, in two buckets), and the overlapping images in
`FindLocalBundle` (by shared observations, over a `FlatHashMap`). Among equal keys the order is
unspecified. The port (`Sfm/IncrementalMapperImpl.cs`) breaks every such tie by ascending
image id. `AdjustLocalBundle` collects its variable points in the enumeration order of the
caller's set (a `HashSet`, deterministic for the same sequence of operations) and hands that
order to the triangulator's `MergeTracks`/`CompleteTracks` (entry 51).

**Why.** Ties are common (integer correspondence counts, equal visibility scores), and the
chosen seed pair, registration order and local bundle all reach the reconstruction. CLAUDE.md
requires a deterministic order; COLMAP's order among ties carries no meaning and depends on
abseil's per-process hash seed.

**Evidence.** `IncrementalMapperTests` (incremental_mapper_test.cc 1:1) pass; `FullPipeline`
registers all 10 frames and matches the ground truth within COLMAP's ReconstructionNear
bounds (0.1 deg, 0.1).

## 100. GlobalMapper::EstablishTracks builds tracks in a deterministic order

**What differs.** COLMAP's `GlobalMapper::EstablishTracks` walks the pose graph's valid edges
and then the union-find's parents and the root-to-observations map, all hash maps
(`std::unordered_map` / `NodeHashMap`). That order decides which observation becomes each
track's root, the order of the elements inside each track and the 3D point id each track
gets. The port (`Sfm/GlobalMapper.cs`) walks the valid edges in ascending pair id and the
tracks in the order their first observation entered the union-find (insertion order,
entry 2), so the ids and element order are deterministic but can differ from COLMAP's. The
candidate tracks are the same. With the default limits every candidate is kept, so the kept
set is the same too; but tracks are selected by (length, id) descending, so when
`KeepMaxNumTracks` or `TrackRequiredTracksPerView` cuts inside a group of equally long
tracks, *which* of them are kept can differ from COLMAP.

**Why.** CLAUDE.md requires hash iteration order that reaches results to be deterministic;
the element order reaches global positioning and bundle adjustment as residual order.

**Evidence.** `GlobalMapperTests` (all four `GlobalMapper_*` cases) match the ground truth
through `ReconstructionNear` at COLMAP's bounds, with the exact observation count in the
noise-free cases.
