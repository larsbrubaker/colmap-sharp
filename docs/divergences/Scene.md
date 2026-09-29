# Divergences from COLMAP: Scene: reconstruction, database, synthesis, rigs and clustering

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 14. CorrespondenceGraph lists image pairs in insertion order

**What differs.** `CorrespondenceGraph.ImagePairs()` and `NumMatchesBetweenAllImages()`
enumerate image pairs in the order `AddTwoViewGeometry` added them. COLMAP iterates its
`FlatHashMap<image_pair_t, ImagePair>`, whose order is the hash table's: `std::unordered_map`
or `boost::unordered_flat_map`, whichever `COLMAP_HASH_MAP_BACKEND` the build picked
(`cmake/FindDependencies.cmake`), so it is not even the same across COLMAP builds.

**Why.** Same reasoning as entry 2: reproducing either backend's bucket layout would mean
porting it, and the order is not part of COLMAP's contract. `correspondence_graph_test.cc`
compares `ImagePairs()` with `UnorderedElementsAre` and reads `NumMatchesBetweenAllImages()`
by key. Insertion order is deterministic on every platform. Every per-point correspondence
list, `ExtractMatchesBetweenImages` and `ExtractTransitiveCorrespondences` are in COLMAP's
order already (they come from vectors, not hash iteration).

**Evidence.** `CorrespondenceGraphTests` (all 11 cases) pass. Code that walks the pairs in
this order (`PoseGraph.Load`, and through it the global mapper) sees them in load order; a
pycolmap fixture for such a pipeline is then compared at Tier C, not Tier A. `SceneClustering.Create` reorders the pairs itself (entry 120).

## 20. CSV number parsing uses .NET's invariant parser

**What differs.** `Util/Misc.CsvToDoubleVector` (COLMAP's `CSVToVector<double>`, used by
`Camera.SetParamsFromString`) parses each element with `double.TryParse(NumberStyles.Float,
InvariantCulture)` instead of COLMAP's `StringToDouble`, which reads with a classic-locale
`std::istringstream` and rejects trailing characters. Both accept plain decimal and
exponent notation ("1", "-0.5", "1e-3", ".5") and reject words ("invalid"). They can disagree
on edge spellings: .NET accepts "Infinity", "NaN" and "∞", which libc++'s stream parsing
handles differently, and .NET rejects hexadecimal floats ("0x1p3").

**Why.** Porting libc++'s `num_get` to match those spellings would be porting a C++ standard
library for inputs COLMAP never writes: every string that reaches this parser in COLMAP's
own formats is produced by `VectorToCSV` (decimal, `%g`-style), which both parsers read the
same way.

**Evidence.** `CameraTests.Camera_ParamsFromString` and `Camera_ParamsToString` pass 1:1,
which read and write parameter lists in exactly the notation both parsers agree on.

## 21. Reconstruction iterates its objects in ascending id order

**What differs.** COLMAP's `Reconstruction` keeps rigs, cameras, frames, images and 3D points
in `NodeHashMap`s, whose iteration order is unspecified (it depends on the hash-map backend
and its history). `Scene/Reconstruction*.cs` keeps them in `Util/IdMap.cs`, which iterates
in ascending id order. Where that order reaches an output, the result can differ from a
given COLMAP build:
- `Crop` hands out new 3D point ids in the order it visits the old points (here: ascending
  old id), and registers the copied frames in frame-id order, so `RegFrameIds` of a cropped
  reconstruction is ascending.
- `FindImageWithName` returns the smallest-id image when several share a name.
- `Rigs`/`Cameras`/`Frames`/`Images`/`Points3D` enumerate in ascending id order for every
  caller (later ports that iterate them inherit this order).
`RegFrameIds` itself is COLMAP's vector in registration order, unchanged. (The centroid of
`Normalize`/`ComputeCentroid` differs separately, by summation order: entry 15.)

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order
(its tests only use order-free properties), so ascending id is the natural deterministic
choice and needs no extra state.

**Evidence.** All ported `ReconstructionTests` (`Crop`, `Normalize`,
`ComputeBoundsAndCentroid`, ...) pass with COLMAP's expectations;
`ReconstructionTests.CSharpOnly_IterationIsInAscendingIdOrder` and `IdMapTests` pin the
order.

## 23. The in-memory database applies a failing write completely or not at all

**What differs.** `Scene/InMemoryDatabase` replaces COLMAP's SQLite database and enforces
the same constraints (UNIQUE indices, primary keys, foreign keys, the image id CHECK), but it
validates a whole write before storing anything. COLMAP's `WriteRig`/`UpdateRig` and
`WriteFrame`/`UpdateFrame` issue one INSERT for the rig or frame row and then one INSERT per
sensor or data id, each in SQLite's autocommit mode; when a later INSERT violates a
constraint (say, a sensor already in another rig), COLMAP throws after the earlier rows have
been committed, leaving a rig or frame with part of its sensors or data. Here the same call
throws and leaves the database unchanged. Consequences:
- On successful writes the assigned ids, what is stored and every read are the same.
- After such a partially failed `WriteRig`/`WriteFrame`, COLMAP has consumed the new id (the
  rig or frame row exists), so its next automatic id is one higher; here the id was never
  used and the next write gets it. A write whose first INSERT fails consumes no id in either.
- The same calls throw, but the exception types and messages differ. COLMAP throws a
  `std::runtime_error` with `sqlite3_errstr`'s text ("SQLite error: constraint failed");
  here it is an `InvalidOperationException` worded like SQLite's extended messages
  ("SQLite error: UNIQUE constraint failed: images.name", "... FOREIGN KEY constraint
  failed"). THROW_CHECK sites use `Util/Check` as everywhere else.

**Why.** The partially written state is an accident of statement-at-a-time execution, not
behavior any COLMAP caller relies on (all of them treat the exception as fatal), and
reproducing it would mean emulating SQLite's statement boundaries.

**Evidence.** `DatabaseTests` (database_test.cc 1:1) passes, including the cases that
expect a constraint violation to throw (`PosePrior`, `TwoViewGeometry`).

## 25. Reconstruction text reading takes whole tokens; image names must be UTF-8

**What differs.** Two things in `Scene/ReconstructionIO*.cs` and `Util/CppLineTokens.cs`:
- *Tokens.* COLMAP's text readers extract values with libc++'s `istream >>`, which stops at
  the first character that cannot continue a number and leaves the rest for the next
  extraction: `"12abc"` read into an integer gives 12, and the next read sees `"abc"`.
  `CppLineTokens` takes whitespace-separated tokens whole, so `"12abc"` (or `"1.5"` read as an
  integer) fails that read. The other text readers built on `CppLineTokens` (SIFT feature
  text, pair lists, the MVS workspace and PMVS files, ASCII PLY) take tokens the same way.
  Everything else follows libc++ as probed with a libc++ harness:
  unsigned reads accept a leading `-` and wrap (`"-1"` is the maximum) and fail above the
  maximum; double reads accept decimal and hexadecimal floats (`"0x1p3"` = 8), reject `inf`
  and `nan`, and fail on overflow and on inexact underflow below the normal range (strtod's
  ERANGE). One corner is approximated: a *decimal* token whose value is an exactly
  representable subnormal fails here but reads in libc++.
- *Image names.* COLMAP keeps an image name as a `std::string` of raw bytes, so any byte
  sequence (Latin-1, Shift-JIS, ...) loads and is written back unchanged. `Image.Name` is a
  C# string, so both readers decode names as strict UTF-8, and a name that is not valid UTF-8
  throws `InvalidDataException` naming the file and the image id and telling the user to
  re-save the name as UTF-8. Writers encode names as UTF-8, so UTF-8 names round-trip byte for
  byte.

**Why.** Every file COLMAP writes separates values with spaces and never glues a number to
other text, so whole-token parsing reads every COLMAP-written file exactly as COLMAP does;
reproducing num_get's partial-token behavior only changes how malformed files fail. For
names, silently decoding with replacement characters (or Latin-1) would load a model whose
names no longer match the image files on disk, which the user would only discover later; a
clear error at load time is the safer behavior, and COLMAP itself writes UTF-8 names on every
platform that produces them.

**Evidence.** `ReconstructionIOOracleTests` read and re-write pycolmap-written models byte
for byte. `ReconstructionIORobustnessTests` pin the token rules (the libc++ probe results),
the multibyte-name round trip (`"café/画像.jpg"`, binary and text) and the non-UTF-8 error.

## 31. SynthesizeDataset visits points, images and chained pairs in ascending id order

**What differs.** `Synthetic.SynthesizeDataset` / `SynthesizeNoise` / `SynthesizeImages`
(`ColmapSharp/Scene/Synthetic*.cs`) iterate the reconstruction's 3D points and images in
ascending id order (IdMap, entry 21), and the CHAINED match config writes its pairs in
ascending pair id order. COLMAP iterates `NodeHashMap`s there (Boost unordered hash order).
The PRNG draw sequence is identical; what differs is (a) the order of each image's projected
2D points before the shuffle, so which 3D point ends up at which 2D index (and the
`point2D_idx` of track elements); (b) which 3D point / pair / image receives which draws in
the track-length pruning, the chained-match shuffles, `SynthesizeNoise`'s 2D and 3D point
noise and `SynthesizeImages`' descriptor seeds for points without a 3D point.
Separately, frame translations and 2D projections differ from the macOS pycolmap wheel in
the last ulp: the wheel contracts the quaternion rotation `rotation * -proj_center` into FMAs
(CLAUDE.md, "No FMA"), ColmapSharp does not. The rotations themselves match bit for bit.

**Why.** Reproducing Boost.Unordered's bucket order would mean porting Boost's container
layout and hash mixing (not ported, entry 2/21); the order is not part of synthetic.h's
contract, and no COLMAP test depends on it. Contraction is a compiler choice (entry 1).

**Evidence.** `oracle/fixture_synthetic.py` records pycolmap 4.2.0's output for two seeded
option sets; `SyntheticOracleTests` matches camera parameters, ids, names, 3D point positions
and frame rotations exactly, the index and position of every 2D point without a 3D point
exactly (so the shuffle permutation is the same), the set of 2D indices holding a 3D point
exactly, sensor-from-rig poses to 1e-12, frame translations to 1e-14 (observed: at most
1.3e-15, about one ulp of |t| = 5) and every 3D point's projection to 1e-9 px (observed:
one ulp). A third fixture case prunes tracks (track_length) and runs `synthesize_noise`:
`CSharpOnly_TrackLengthAndNoiseMatchPycolmap` finds the same track lengths and 3D point ids,
the same frame poses after noise (1e-12), and the per-image and per-point noise chunks as the
same multiset, which shows the pruning consumed exactly COLMAP's number of draws and only
their assignment differs. A fourth case pins this at bundle-adjustment scale: seed 0, 1 rig,
1 camera, 100 frames, 2000 points with the "Nominal" noise of `bundle_adjustment_test.cc`
(0.5 px, 0.1, 0.5 deg, 0.1), recorded as per-image summaries (clean observations; each
image's 2D noise chunk) plus every 3D point's noise. There the reassignment is visible in a
bundle adjuster's initial cost (0.5 * sum of squared reprojection residuals): ColmapSharp's
noisy dataset gives 287550090.63, pycolmap's 287510459.99. A differential harness (scratch,
not checked in) showed the gap is entirely this entry: it took ColmapSharp's clean dataset,
installed pycolmap's point2D-to-3D-point layout, and replayed `SynthesizeNoise`'s draws
visiting images and 3D points in pycolmap's hash order (the `reconstruction.images` /
`points3D` iteration order). The result matched pycolmap's noisy reconstruction to 1.1e-13 px
(2D), 6.7e-16 (3D), 1.8e-15 (translations) and 8.7e-19 (rotations), and its cost was
287510459.9944969, the same as pycolmap's. All 17 `synthetic_test.cc` cases pass 1:1.

## 32. SynthesizeImages hands bitmaps to a sink instead of writing image files

**What differs.** COLMAP's `SynthesizeImages(options, reconstruction, image_path)` writes
each image to `image_path / image.Name()` via `Bitmap::Write` (OpenImageIO). ColmapSharp's
`Synthetic.SynthesizeImages(options, reconstruction, writeImage)` calls
`writeImage(image.Name, bitmap)` for each rendered image; the pixels are the same.

**Why.** Image file encoding/decoding is not ported: the host (MatterCAD) owns image I/O
(PORTING_PLAN.md, the skipped `bitmap_test.cc` file cases). A sink lets a caller write files
with its own encoder or keep the bitmaps in memory.

**Evidence.** `SyntheticTests.SynthesizeImages_Nominal` checks every image reaches the sink
with the camera's width and height (COLMAP's test reads the PNG back and checks the same).

## 33. DatabaseCache iterates its objects in ascending id order

**What differs.** COLMAP's `DatabaseCache` keeps rigs, cameras, frames and images in
`NodeHashMap`s. `Scene/DatabaseCache.cs` keeps them in `Util/IdMap.cs`, so `Rigs`/`Cameras`/
`Frames`/`Images` enumerate in ascending id order, like `Reconstruction` (entry 21). Where
that order reaches an output:
- `FindImageWithName` returns the smallest-id image when several share a name (only possible
  through `AddImage`; database names are unique).
- `Reconstruction.Load` visits the cache in that order, so when an existing object conflicts
  with the cache, the reported check failure is the one for the smallest id.
- `CreateFromCache` copies objects in that order; the copies land in IdMaps again, so the
  result is the same either way.
The correspondence graph is unaffected: `Load` adds pairs in the order the database returns
them and `CreateFromCache` in the source graph's insertion order (entry 14), both as in COLMAP.
`ReconstructionPruning.FindRedundantPoints3D` lists redundant ids in `Points3D` order (entry
21); its selection is order-free because the queue key `(gain, point3D_id)` is a total order.

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order here
(its tests use order-free properties), so ascending id is the natural choice.

**Evidence.** The 1:1 `DatabaseCacheTests` and `ReconstructionTests.
Reconstruction_TranscribeImageIdsToDatabase` pass with COLMAP's expectations;
`DatabaseCacheTests.CSharpOnly_CreateFromCacheKeepsWholeFrames` and
`ReconstructionPruningTests.CSharpOnly_SameTilePointIsRedundant` pin the filtering and the
selection tie-break.

## 34. ReconstructionManager.Write breaks point-count ties by index

**What differs.** COLMAP's `ReconstructionManager::Write` orders the models with `std::sort` by
descending `NumPoints3D` and writes them to `0/`, `1/`, ...; the order of models with the same
point count is unspecified. `Scene/ReconstructionManager.cs` sorts with an index tie-break, so
equal-sized models keep their manager order (a stable sort).

**Why.** CLAUDE.md's stable-sort rule: the tie order decides which directory a model lands in.
libc++'s `std::sort` uses insertion sort for short ranges, which also keeps equal elements in
order, so for the handful of models a mapper produces the result is the same as COLMAP's; only
a long run of equal counts could differ.

**Evidence.** `ReconstructionManagerTests.CSharpOnly_WriteOrdersByPointCountThenIndex` writes
models with point counts 1, 2, 1, 2 and reads back sources 1, 3, 0, 2.

## 38. PoseGraph orders equally large frame components by smallest frame id

**What differs.** COLMAP's `PoseGraph::ConnectedFrameComponents` collects the frames of the
valid edges in a `FlatHashSet`, finds the components (whose order follows that set, entry 2)
and orders them with an unstable `std::sort` by descending size, so the order of equally large
components is unspecified. `LargestConnectedFrameComponent` keeps the first strictly largest
component in the same hash order. `Scene/PoseGraph.cs` hands `FindConnectedComponents` the
frames in ascending id order and sorts stably, so equally large components come out by their
smallest frame id, and `LargestConnectedFrameComponent` returns that same first one.
`ConnectedImageIdsForFrameComponents` follows the same component order. The pose graph's own
edge map is a `Dictionary` (insertion order until an edge is deleted); no output depends on it.

**Why.** CLAUDE.md's stable-sort and hash-order rules: the tie order decides which component
global SfM treats as the largest. COLMAP relies on no particular order: `pose_graph_test.cc`
uses components of different sizes and compares sets.

**Evidence.** `PoseGraphTests` (pose_graph_test.cc 1:1) pass with COLMAP's expectations;
`PoseGraphTests.CSharpOnly_EqualSizeComponentsOrderedBySmallestFrameId` pins the tie order and
that the largest component equals the first one.

## 60. UndistortReconstruction keeps each camera's id

**What differs.** COLMAP's `UndistortReconstruction` (image/undistortion.cc) assigns
`UndistortCamera(...)` to the stored camera (`reconstruction->Camera(camera_id) = ...`).
`UndistortCamera` builds a default-constructed `Camera`, so every undistorted camera's
`camera_id` becomes `kInvalidCameraId` while its map key stays. The port
(`ImageProcessing/Undistortion.cs`) writes the undistorted model, size, parameters and prior
flag into the stored camera but keeps its `CameraId` (and so its `SensorId`).
`UndistortCamera` itself still returns a camera with the invalid id, as in COLMAP.

**Why.** Code here keys by `camera.CameraId` or `camera.SensorId` (Reconstruction.Crop,
Alignment's merge, rotation averaging, the default bundle adjuster, the triangulator's
bogus-params cache); with two or more cameras they would collide on the invalid id.
MatterCAD undistorts and then densifies the same in-memory reconstruction. COLMAP avoids the
problem because its undistorter writes the reconstruction to disk, and the writers take the
id from the map key, so file output is the same either way.

**Evidence.** `UndistortionTests.CSharpOnly_UndistortReconstruction_KeepsCameraIds` (two
cameras: each id equals its map key, rig lookups by `SensorId` succeed, and `Crop` keeps both
cameras) failed before the change and passes after it; the ported undistortion_test.cc cases
pass unchanged.

## 69. ExtractColorsForAllImages sums colors per image and reduces in image-id order

**What differs.** COLMAP's `Reconstruction::ExtractColorsForAllImages` adds each image's
interpolated colors into per-thread sums (whichever pool thread ran the image) and merges the
thread sums, so the floating-point summation order depends on scheduling and the thread
count. `Scene/Reconstruction.Colors.cs` fills one partial sum per image (in parallel) and
reduces them in ascending image-id order.

**Why.** CLAUDE.md requires sequential and parallel runs to give the same result. The sums
are of float colors in double, so the orders differ at most in the last bits of the mean,
which can move the rounded 8-bit color by one only at an exact .5 tie.

**Evidence.** `ReconstructionTests.Reconstruction_ExtractColorsForAllImages`
(reconstruction_test.cc 1:1) passes; the C#-only
`Reconstruction_ExtractColorsForAllImagesIndependentOfThreadCount` gets identical colors
with 1 and 8 threads.

## 80. ReadRigConfig requires exactly 4 rotation and 3 translation entries

**What differs.** COLMAP's `ReadRigConfig` (scene/rig.cc) writes the entries of
`cam_from_rig_rotation` and `cam_from_rig_translation` into a fixed-size `Eigen::Vector4d` and
`Rigid3d::translation()` by running index, without checking the count. The port
(`Scene/RigConfig.cs`) throws a "Check failed" error unless there are exactly 4 and 3 entries.

**Why.** Fewer entries leave COLMAP with uninitialized coefficients and more entries write out
of bounds; both are undefined behavior with no result to match. Every well-formed config reads
the same.

**Evidence.** `RigConfigTests` (rig_test.cc 1:1) pass, including `ReadRigConfig_Nominal`,
which reads a 4-entry rotation and a 3-entry translation.

## 81. ApplyRigConfig hands the reconstruction its rigs and frames in id order

**What differs.** COLMAP's `UpdateRigsAndFramesFromDatabase` (scene/rig.cc) collects the
reconstruction's rigs and frames in `NodeHashMap`s and passes them to
`Reconstruction::SetRigsAndFrames` in the maps' iteration order, and `AddFrame` registers posed
frames in that order, so it decides the order of `RegFrameIds()`. The port
(`Scene/RigConfig.Apply.cs`) collects them in sorted maps, so the frames are registered in
ascending frame id order.

**Why.** In COLMAP 4.2.0 `NodeHashMap` is `boost::unordered_node_map` or `std::unordered_map`,
depending on the build (util/hash_containers.h). Either way its iteration order is
implementation-defined: it depends on the build configuration, the hash and the standard
library, so there is no single order to match. CLAUDE.md requires a deterministic order. The
sets of rigs, frames and registered frames do not depend on it.

**Evidence.** `RigConfigTests.ApplyRigConfig_*` (rig_test.cc 1:1) pass with COLMAP's counts of
rigs, frames and registered frames. None of them checks the registration order.

## 93. Scene and reconstruction clustering break sort ties deterministically

**What differs.** Three `std::sort` calls leave equal keys in an unspecified order:
`SceneClustering::PartitionHierarchicalCluster` sorts the overlap-candidate edges of each
child by descending weight, `SceneClustering::PartitionFlatCluster` sorts each image's
related images by descending weight (scene_clustering.cc), and `EstablishStrongClusters`
(reconstruction_clustering.cc) sorts the union-find clusters by descending size after
collecting them from a `NodeHashMap`, so equal-size clusters are numbered in hash order. The
port (`Scene/SceneClustering.cs`, `Scene/ReconstructionClustering.cs`) keeps equal weights in
edge order (a stable sort) and numbers equal-size clusters by ascending smallest frame id.
When a weight tie straddles the overlap budget, a different overlap image can be chosen;
equal-size reconstruction clusters can get swapped ids.

**Why.** The translation rules require an explicit, deterministic tie-break wherever tie
order reaches an output. Edge order is what the macOS SDK's libc++ `std::sort`
(`__algorithm/sort.h`) gives for short lists: lengths 2-5 go through `__sort3/4/5`, which
never reorder equal elements under these `>` comparators; lengths 6-23 go through insertion
sort, which is stable; lengths of 24 or more go through pdqsort, which is not. (Older libc++
was already unstable from 7 elements for non-trivially-copyable types.) So an overlap list or
related-image list of 24+ entries with tied weights can come out in a different order than
COLMAP's; every such list in COLMAP's tests has at most 12 entries. Reproducing pdqsort and
absl's hash order would tie the result to one library version.

**Evidence.** All eight scene_clustering_test.cc cases and all nine
reconstruction_clustering_test.cc cases pass (SceneClusteringTests,
ReconstructionClusteringTests); a reviewer's scratch run over the scene tests with the same
stable order also matched every expected membership.

## 94. Flat scene clusters are ordered by size, then smallest image id

**What differs.** `SceneClustering::PartitionFlatCluster` sorts the child clusters with the
comparator `size(a) >= size(b) && min(a) < min(b)`, which is not a strict weak ordering
(`std::sort` then has undefined behavior) and dereferences `min_element` of an empty cluster.
The port orders the children by what COLMAP's comment says it intends: descending size, then
ascending smallest image id, with empty clusters last (stable among themselves). Only the
order of `GetRootCluster().ChildClusters` / `GetLeafClusters()` is affected; each child's
overlap images depend only on its own members. In practice the orders do differ: with 3-5
children libc++'s `__sort3/4/5` only swap when the comparator says so, so a child of size 2
with smallest id 1 stays ahead of a child of size 3 with smallest id 5 (neither compares
"less" than the other), while the port puts the larger child first.

**Why.** Undefined behavior cannot be ported; the comment states the intent.

**Evidence.** `SceneClusteringTests.CSharpOnly_FlatChildClustersOrderedBySizeThenSmallestId`
pins the order; `SceneClustering_ThreeFlatClusters` and `_ThreeFlatClustersTwoOverlap`
(1:1, order-insensitive as in COLMAP) pass.

## 120. SceneClustering.Create hands the image pairs over in Boost hash order

**What differs.** COLMAP's `SceneClustering::Create` builds the edge list by iterating the
`NodeHashMap` from `NumMatchesBetweenAllImages()`, so the edges come in hash order. The
port (`Scene/SceneClustering.cs`) models the order of COLMAP's default backend, which CMake
picks when Boost >= 1.84 is found (`cmake/FindDependencies.cmake`): `boost::unordered_node_map`
with the identity `std::hash<uint64_t>`. Boost marks that hash non-avalanching
(`hash_traits.hpp`), so its table mixes it with `mulx(h) = lo ^ hi` of `h * 0x9E3779B97F4A7C15`
(`detail/mulx.hpp`), puts the key in group `mix >> size_index` (`pow2_size_policy::position`
in `detail/foa/core.hpp`) and iterates groups in ascending order. The port orders the pairs
by that mixed value (then by pair id), instead of the correspondence graph's insertion order
(entry 14), which is ascending pair id. This is Boost's order up to the order of the keys
that share a group (Boost keeps insertion order within a group and moves overflow to later
groups), which depends on the table size and the insertion history; verified against the
Boost.Unordered 1.87.0 source.

**Why.** The edge order decides every weight tie: the vertex numbering of the graph cut
(entry 77) and which overlap images each child cluster gets (entry 93). COLMAP relies,
without saying so, on that order being unrelated to image ids. In ascending order every tie
breaks toward the same few low image ids, and a database numbers the images of one rig
frame consecutively. With `hierarchical_pipeline_test.cc`'s panoramic case, where all
1770 pairs have 100 matches, the root split's overlap was images 1-3 and 31-33: one frame
per side. A zero-baseline rig's images of one frame share a projection center, so the two
halves had only two distinct centers in common. `AlignReconstructionsViaReprojections`
cannot estimate a similarity from that, the final merge failed, and the run ended with two
reconstructions. The hash order is backend-specific (entry 14); modelling the default Boost
backend reproduces the property, and most of the order, and stays deterministic.

**Evidence.** Tier C. `SceneClusteringTests.CSharpOnly_CreateSpreadsTiedOverlapAcrossFrames`
(C#-only) fails with the insertion order (2 shared frames) and passes with the Boost order
(at least 3). `HierarchicalPipeline_WithoutNoiseAndPanoramicNonTrivialFrames` (1:1)
now merges into one reconstruction. The leaf reconstructions register 27-39 images from
12-13 image clusters, close to pycolmap 4.2.0's `hierarchical_mapping` on the same dataset
(27-33); with the insertion order they registered 15-18. The ported
`scene_clustering_test.cc` cases call `Partition` directly and are unaffected.
