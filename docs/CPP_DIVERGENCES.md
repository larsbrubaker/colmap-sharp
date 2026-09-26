# Deliberate divergences from COLMAP

Each entry: what differs, why, and the evidence. Numbered so code comments can cite them
(`docs/CPP_DIVERGENCES.md`, entry N). Remove an entry when the divergence is gone.

## 1. Real-valued random draws are not fused (no FMA), unlike the macOS pycolmap wheel

**What differs.** `RandomUtils.RandomUniformReal` (for `min != 0`), `RandomGaussian` and
`NormalDistribution` evaluate libc++'s `(b - a) * u + a`, `u*u + v*v` and
`x * stddev + mean` as separate multiply and add. Apple clang's default `-ffp-contract=on`
fuses these into single-rounding FMAs on arm64, so a libc++ build with default flags can
differ from ColmapSharp in the last ulp of those draws. Integer draws, `RandomUniformReal`
with `min == 0`, the shuffles and the mt19937 words are unaffected.

**Why.** CLAUDE.md's "No FMA" rule: ColmapSharp's results must be identical on every
platform and .NET JIT never contracts, while contraction in the C++ build is a compiler and
flags choice, not COLMAP behavior. Matching it would mean `Math.FusedMultiplyAdd` at exactly
the sites a particular clang version chose to fuse.

**Evidence.** `oracle/fixture_random.py` builds `oracle/random_harness.cc` with
`-ffp-contract=off` and with the default; the draws above differ between the two builds
(recorded as `contracted_cases` in `ColmapSharp.Tests/TestData/oracle/random.json`), and
the C# port matches the `-ffp-contract=off` build bit for bit (`RandomOracleTests`). The
pycolmap 4.2.0 wheel's `_core` disassembles (`otool -tv`) to about 36,000 `fmadd`/`fmsub`
family instructions, so it is built with contraction on and most likely produces the
contracted numbers. Downstream, a pycolmap fixture that depends on real-valued draws (e.g.
`synthesize_dataset` noise) can therefore differ in the last ulp and is compared at Tier B
or C, not Tier A.

## 2. Hash-container iteration order in UnionFind and connected components

**What differs.** `UnionFind.Parents` enumerates in insertion order. `FindConnectedComponents`
returns components ordered by their first node in the caller's node enumeration, each listing
its nodes in that order, and `FindLargestConnectedComponent` breaks ties between equally large
components the same way. COLMAP's orders come from `boost::unordered_node_map` /
`unordered_flat_set` (`colmap/util/hash_containers.h`).

**Why.** Reproducing them would mean porting Boost.Unordered's bucket layout and
`boost::hash` mixing for every key type, and Boost is not ported (`docs/LICENSE_AUDIT.md`).
The order is not part of COLMAP's contract: `connected_components_test.cc` compares with
`UnorderedElementsAre`, and `union_find_test.cc` never inspects it. Insertion order is
deterministic and the same on every platform.

**Evidence.** The component *sets*, and every root `Find` returns for the same call sequence,
are identical (ColmapSharp.Tests/Mathematics/UnionFindTests.cs, ConnectedComponentsTests.cs).
Callers that need a reproducible component order should pass nodes in a deterministic order
(for example sorted ids).

## 3. Spanning-tree ties between equal edge weights

**What differs.** `ComputeMaximumSpanningTree` / `ComputeMinimumSpanningTree` run Kruskal with
edges sorted by (cost, input index). COLMAP calls `boost::kruskal_minimum_spanning_tree`,
which pops edges from a `std::priority_queue`, so among equal costs it takes them in heap
order. With tied weights the two can choose different (equally optimal) trees.

**Why.** Boost is not ported; the input-index tie-break is the deterministic, documented rule.
The cost transform itself (`max_weight - w` in float for the maximum tree) is kept, so the
same weights tie on both sides.

**Evidence.** With distinct costs the minimum spanning tree is unique and the parents match
exactly (SpanningTreeTests).

## 4. Stoer-Wagner: which min cut, and which side is labeled 1

**What differs.** `ComputeMinGraphCutStoerWagner` is written from Stoer & Wagner (JACM 1997)
instead of calling `boost::stoer_wagner_min_cut`. The cut weight is the global minimum on
both sides, but when several cuts share it the one reported can differ, and the side labeled
1 is the set of vertices merged into the last vertex of the best phase (Boost's parity map
labels its own choice).

**Why.** Boost is not ported. COLMAP's contract is the minimum weight plus a 0/1 label per
vertex; `graph_cut_test.cc` checks only the weight and the label range.

**Evidence.** The ported tests pass, and GraphCutCrossCheckTests compares the weight and the
labeled cut against exhaustive search on 200 random graphs.

## 5. Boykov-Kolmogorov max-flow with float capacities

**What differs.** `MinSTGraphCut` is written from Boykov & Kolmogorov (PAMI 2004) instead of
calling `boost::boykov_kolmogorov_max_flow`, and it pushes each node's direct
source -> node -> sink flow before the search starts. Both change the order in which flow is
augmented. With integer capacities that order cannot show in the result; with float
capacities (`MinSTGraphCut<float>`, as Delaunay meshing uses it) the returned flow is summed
in a different order, so it can differ from COLMAP's by rounding, and a node whose residual
path to a terminal is only a rounding residue can be labeled on the other side of the cut.

**Why.** Boost is not ported (`docs/LICENSE_AUDIT.md`), and Kolmogorov's own maxflow library
is GPL/research-only, so neither is transcribed. Reproducing Boost's exact float rounding would
mean reproducing its exact augmentation order. The terminal-capacity handling is needed for
scale: storing terminal links as ordinary terminal out-edges made each augmentation rescan
them, which was quadratic on Delaunay-sized graphs.

**Evidence.** For integer capacities the result is Tier A: GraphCutCrossCheckTests compares
the flow and the labels with exhaustive search on 300 random graphs, and the sink-side labels
equal the unique minimal sink-side min cut, which is the same for every maximum flow and so
for Boost too. MinSTGraphCutScalingTests checks, for a 200k-node float grid, that the labeled
cut's capacity equals the returned flow within 1e-3 relative.

## 6. FMA contraction in the macOS arm64 pycolmap wheel (quaternion-vector rotation, small products, GPS)

**What differs.** `Quaterniond * Vector3d` (Eigen's quaternion-vector rotation) differs from
the pycolmap 4.2.0 macOS arm64 wheel by 1-2 ulps on about half of the inputs. Downstream,
so do the Rigid3d/Sim3d operations built on it (point transform, the translations of
composition and inverse, `TgtOriginInSrc`), `Rigid3d.AdjointInverse` and
`GetCovarianceForRigid3dInverse` (3x3 and 6x6 products), and the `GPSTransform`
ellipsoid/ECEF/ENU/UTM conversions (last bit of the ECEF-scale coordinates, up to
9.3e-10 m).

**Why.** Same cause as entry 1: the wheel is built with contraction on, and it evaluates
the cross products inside the rotation, `a1*b2 - a2*b1`, as `fma(a1, b2, -(a2*b1))`.
ColmapSharp never uses FMA in math paths (CLAUDE.md, "No FMA"), so its results are the
same on every platform. They are expected to match a C++ build that does not contract,
which has not been checked here.

**Evidence.** `oracle/linear_algebra_rotations.py` prints it: re-deriving `q * v` with
ColmapSharp's formula gives 69/138 mismatches against the wheel with plain cross products
and 0/138 with the cross products fused as above.
`RotationOracleTests.ToleranceFields("rotated")` pins the C# result at 1e-14 relative.
The other quaternion operations in that fixture are bit-identical
(`RotationOracleTests.ExactFields`). For the geometry (`oracle/geometry_transforms.py`,
`GeometryOracleTests`): the Rigid3d/Sim3d operations that do not rotate a vector are
bit-identical, and in `EllipsoidToECEF` only the z coordinate, `(N * (1 - e2) + alt) * sin_lat`,
differs; evaluating `N * (1 - e2) + alt` as one FMA takes it from 13/80 mismatches to
3/80 (the script prints this), while x and y (no multiply-add) match on every case. `GeometryOracleTests.ToleranceFields`
pins these at 1e-14 relative (1e-13 for the 6x6 covariance) and 1e-8 m for GPS coordinates.

## 7. sin(a/2) in the angle-axis to quaternion conversion

**What differs.** `AngleAxisd.ToQuaternion` can differ from the wheel by 1 ulp in the vector
part.

**Why.** .NET's `Math.Sin` calls the platform libm `sin`. On some inputs the wheel's
`sin(a/2)` rounds one ulp away from libm `sin`. The cause is not established. One
hypothesis is that the compiler fused the adjacent `sin` and `cos` of the same argument into
a `sincos` call that rounds differently. We do not emulate a compiler's choice of math
routine.

**Evidence.** `oracle/linear_algebra_rotations.py`: all fixture cases but one are
bit-identical with libm `sin`, and that one becomes identical when `sin(a/2)` moves one
ulp. `RotationOracleTests.ToleranceFields("from_axis_angle")` pins it at 1e-14 relative.

## 8. CameraDatabase iterates the sensor-width table in specs.cc order, not hash order

**What differs.** COLMAP's `camera_specs_t` is a `NodeHashMap` (a `std::unordered_map` or
`boost::unordered_node_map`, depending on the build), and `CameraDatabase::QuerySensorWidth`
iterates it, writing the output width on every match and stopping after the second
non-exact match per make. `CameraSpecs.InitializeCameraSpecs` returns a list in `specs.cc`
source order, so when a cleaned EXIF make matches more than one table make (a substring
match either way round, e.g. an empty make matches all of them), which widths are seen,
and so the width left behind and whether a unique match is found, can differ from a given
COLMAP build.

**Why.** Hash iteration order is unspecified and differs between standard libraries and
Boost, so there is no single COLMAP behavior to match; CLAUDE.md asks for deterministic
order here.

**Evidence.** `database_test.cc`'s cases (ported in `CameraDatabaseTests`) match a single
make and pass. Queries whose make matches one table make are unaffected.

## 9. Bitmap.Rescale is a managed resampler, not OpenImageIO's resize

**What differs.** COLMAP's `Bitmap::Rescale` calls `OIIO::ImageBufAlgo::resize` with a
"triangle" (kBilinear) or "box" (kBox) filter. `ColmapSharp/Sensor/BitmapResize.cs`
reimplements the model OIIO's output follows (filter widened by the downsampling ratio,
clamp-to-edge samples, separable, round to nearest). Bilinear results are within one gray
level of pycolmap's; box results agree except where a source pixel center lies exactly on
the box edge at a non-integer ratio, where OIIO's inclusion rule is not reproduced and a
destination pixel can average one source pixel more or fewer.

**Why.** OpenImageIO is native (docs/LICENSE_AUDIT.md). Its resize accumulates in float
with its own filter evaluation, so bit-exact output would need a port of OIIO's
resampling code (Apache-2.0, allowed but not done).

**Evidence.** `oracle/fixture_bitmap_rescale.py` records pycolmap 4.2.0's bilinear output on
seeded grey and RGB images, up and down; `BitmapRescaleOracleTests` checks every pixel
within one gray level. Impulse probes (a single lit pixel, 1-D and 2-D) match pycolmap
exactly for both filters at ratios 4, 8, 3, 1.5, 5/3, 2/3 and 3/5. The box tie case: 23 -> 10
pixels, destination pixel 5 (center 12.65) excludes source pixel 11 (center 11.5, distance
1.15 = half the box) in pycolmap, and a half-open box fails other cases, so the rule is
not simply half-open.

## 10. ExifReader leaves rationals with a zero denominator unset

**What differs.** An EXIF RATIONAL with denominator 0 (FocalLength, FocalPlaneXResolution,
GPSLatitude/Longitude, GPSAltitude) is not stored in the Bitmap's metadata by
`ColmapSharp/Sensor/ExifReader.cs`. OpenImageIO, through which COLMAP reads EXIF, most
likely stores the float quotient (inf, or NaN for 0/0), which COLMAP's getters would see.

**Why.** An inf/NaN focal length or GPS coordinate carries no information, and cameras
write 0/0 to mean "unknown", so "absent" is the faithful reading. It is visible only
through the getters: `ExifLatitude`/`ExifLongitude`/`ExifAltitude` return null here where
COLMAP could return NaN or inf, and `ExifFocalLength` returns null (or a later fallback's
value) where COLMAP could return inf or NaN from a zero-denominator FocalLength.

**Evidence.** Not verified against OIIO: no oracle fixture carries a zero-denominator tag.
The reader's behavior is stated in its file header.

## 11. UTMToEllipsoid latitude can differ by 1 ulp

**What differs.** `GPSTransform.UTMToEllipsoid` returns a latitude one ulp away from the
pycolmap 4.2.0 macOS arm64 wheel on 2 of the 80 points in `geometry_transforms.json`;
longitude and altitude, and every other point, are bit-identical.

**Why.** The cause is not established. Re-deriving the conversion in Python with the same
libm (`math.sin`, `math.asin`, `math.cosh`, `math.sinh`) reproduces the C# result exactly,
and fusing the multiply-adds of the xi'/eta' series or of the latitude series into FMAs does
not remove the two mismatches, so it is neither a port bug nor the contraction of entry 6.
It may be another compiler choice in the wheel (as in entry 7). We do not emulate it.

**Evidence.** `GeometryOracleTests.ToleranceFields("gps", "utm_to_ellipsoid")` pins it at
1e-14 relative; the observed gap is 1 ulp (about 7e-15 deg). The Python re-derivation was a
scratch harness following gps.cc term for term.

## 12. FMA contraction in the camera models

**What differs.** Camera model projection (`CameraModelImgFromCam`) and ray unprojection
(`CameraModelCamRayFromImg`) of every perspective model, and `CameraModelCamFromImg` of the
fisheye, division, FOV and EUCM models, differ from the pycolmap 4.2.0 macOS arm64 wheel by
a few ulps on part of the inputs (at most 1.6e-14 relative to max(1, |value|) in the
fixture). Which calls succeed or fail never differs.

**Why.** Same cause as entries 1 and 6: the wheel is built with contraction on and fuses
multiply-adds that sit in one C++ statement, e.g. `*x = f * *x + c1` in every model's
`ImgFromCam` and `u * u + v * v + 1.0` in `CamRayFromImg`. ColmapSharp never uses FMA in
math paths (CLAUDE.md, "No FMA"). The iterative undistortion runs its distortion on
`ceres::Jet`, whose operators are separate function calls that clang does not contract, and
it matches the wheel bit for bit.

**Evidence.** `oracle/camera_models.py` prints it: re-deriving SIMPLE_RADIAL's projected x
with ColmapSharp's formula matches the wheel on 64/75 and 60/75 points of the two parameter
sets, and on 75/75 with only `f * x + c1` fused; PINHOLE's ray z matches on 98/101 plain and
101/101 with `u*u + v*v` fused as `fma(u, u, v*v)`.
`CameraModelOracleTests` requires bit-identical `CamFromImg` for the models whose
unprojection is the iterative undistortion or a plain pinhole, and for all of
EQUIRECTANGULAR, and pins everything else at 2e-14 relative to max(1, |value|).

**Related, not observed here.** C++ `EquirectangularCameraModel` evaluates
`2.0 * EIGEN_PI * (...)` with EIGEN_PI a `long double` literal, so on x86-64 Linux (80-bit
long double) its results may differ from both the macOS wheel (where long double is double)
and ColmapSharp, which uses `Math.PI` in double.

## 13. Sparse Cholesky: simplicial LLT/LDLT with our AMD instead of CHOLMOD and Eigen

**What differs.** `SparseCholeskyWithFallbackSolver`'s first stage is CHOLMOD's supernodal
LLT in COLMAP and the simplicial LLT of `LinearAlgebra/SimplicialCholesky.cs` here; its
fallback and `LeastAbsoluteDeviationSolver`'s `SimplicialLLT` are Eigen's simplicial
factorizations in COLMAP and ours here. `SolverType.SupernodalCholmodLLT` keeps its name but
selects the managed fallback solver. The fill-reducing ordering is our AMD
(`LinearAlgebra/AmdOrdering.cs`, from the Amestoy-Davis-Duff paper, without dense-row
deferral, initial supervariable detection or mass elimination), so the permutation, and with
it the order of floating-point operations and the last bits of every solution, differ from
CHOLMOD's and Eigen's. Accept/reject decisions follow the same rules (LLT stops at the first
pivot <= 0, LDLT at an exactly zero pivot), but a pivot within rounding of zero can land on
the other side under a different ordering.

Also, once the fallback solver has switched to LDLT, `AnalyzePattern`/`Compute` with a new
pattern re-analyzes the LDLT stage here. COLMAP re-analyzes only the supernodal stage, so its
next `ldlt_.factorize` runs on a stale symbolic analysis (an upstream bug with undefined
results in Eigen); `ComputeNewPatternAfterFallback` pins the fix.

**Why.** CHOLMOD and CSparse are GPL/LGPL and Eigen is MPL-2.0 (`docs/LICENSE_AUDIT.md`);
none of them can be ported. A supernodal code is a performance refinement over the
simplicial one, not a different result, and can replace it later behind the same API.

**Evidence.** `sparse_cholesky_test.cc` and `least_absolute_deviations_test.cc` pass 1:1
(`SparseCholeskyWithFallbackSolverTests`, `LeastAbsoluteDeviationsTests`), including the
singular, indefinite and ridge cases whose outcome depends on the pivot rules. The C#-only
`SimplicialCholeskyTests` pin the solutions to the dense LLT within 1e-10 relative error.

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

**Evidence.** `CorrespondenceGraphTests` (all 11 cases) pass. Code ported later that walks
the pairs in this order (COLMAP's `PoseGraph::Load`, and through it the global mapper) will
see them in load order; a pycolmap fixture for such a pipeline is then compared at Tier C,
not Tier A.

## 16. PROSAC's out-of-range sample index fails a Check instead of reading past the data

**What differs.** COLMAP's `ProgressiveSampler` (ported faithfully in
`Optim/ProgressiveSampler.cs`) makes index `n` the mandatory element of a progressive sample,
and `n` can equal `total_num_samples`: on the first sample when `num_samples ==
total_num_samples`, and in general on the sample where the growth schedule reaches the last
element. COLMAP's `Sampler::SampleXY` then reads `X[total_num_samples]`, past the end of the
`std::vector` (undefined behavior: garbage or a crash). `SampleX`/`SampleXY` in
`Optim/Sampler.cs` check every sampled index against the data length and throw COLMAP's
"Check failed" `ArgumentException` instead.

**Why.** Undefined behavior has no C# equivalent to match, and silently clamping or shifting
the index would change the sampler's Tier A sequence that `progressive_sampler_test.cc` pins.
COLMAP 4.2.0 instantiates `ProgressiveSampler` nowhere outside its own test, so no pipeline
reaches this path; failing loudly keeps any future user from getting a model estimated from
out-of-bounds memory.

**Evidence.** `ProgressiveSamplerTests` (1:1) still pass unchanged. The C#-only
`RansacTests.CSharpOnly_ProgressiveSamplerIndexPastEndFailsCheck` runs RANSAC with PROSAC on
exactly `kMinNumSamples` pairs and expects the Check failure.

## 17. RANSAC and LO-RANSAC always run their trial loop serially

**What differs.** With `RANSACOptions::num_threads > 1` (or -1), COLMAP built with OpenMP
runs the trial loop on several threads, each with its own sampler seeded
`random_seed + thread index`, sharing an atomic trial counter and a mutex-guarded best model.
`Optim/Ransac.cs` and `Optim/LoRansac.cs` validate `num_threads` exactly as COLMAP does
(`Check()`, and "Parallel RANSAC only supports RandomSampler" for any other sampler with more
than one effective thread) and then run the loop once on the calling thread, which is what
COLMAP itself does when built without OpenMP ("the block runs once serially").

**Why.** COLMAP's parallel result depends on thread scheduling (which thread claims which
trial index, and which thread's model reaches the shared best first), so it is not
reproducible even against itself; CLAUDE.md requires sequential and parallel runs to give the
same result. The serial loop with the thread-0 seed is COLMAP's own `num_threads == 1`
behavior and its non-OpenMP build's behavior for every `num_threads`. A deterministic parallel
scheme (for example, fixed per-trial seeds) would be a different algorithm from both COLMAP
builds and is left for when profiling shows RANSAC is a bottleneck.

**Evidence.** `RansacTests.RANSAC_ParallelSimilarityTransform` and
`LoRansacTests.LORANSAC_ParallelSimilarityTransform` (1:1, `num_threads = 4`) pass with
COLMAP's expectations. The C#-only `RansacTests.CSharpOnly_ParallelRequiresRandomSampler`
pins the kept validation.
