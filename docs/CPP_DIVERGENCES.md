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

## 15. ComputeBoundingBoxAndCentroid sorts instead of std::nth_element

**What differs.** `Geometry/Normalization.cs` fully sorts each coordinate list where COLMAP
partitions it with two `std::nth_element` calls. The bounding box (the elements at the two
percentile positions) is the same value, and so is the multiset of elements the centroid
averages, but the order they are summed in differs: COLMAP's is whatever libc++'s
`nth_element` leaves between the two positions. The centroid can therefore differ from
COLMAP's in the last bits.

**Why.** The element order after `nth_element` is an unspecified implementation detail of the
C++ standard library; reproducing it would mean porting libc++'s introselect for one
rounding-level effect. Sorting satisfies every `nth_element` postcondition.

**Evidence.** `normalization_test.cc` passes 1:1 (`NormalizationTests`), including the exact
bounding boxes and the 1e-6 centroid checks.

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

## 18. Solver cost and gradient are summed in residual-block order for any thread count

**What differs.** Ceres' `ProgramEvaluator` accumulates the cost and the gradient per
thread and then adds the per-thread partial sums, so with `num_threads > 1` the last bits of
the cost, the gradient (and through them every accept/reject decision and tolerance test)
depend on how `ParallelFor` split the residual blocks. `Solver/ProgramEvaluator.cs`
evaluates the residual blocks in parallel but stores each block's cost in its own slot and
sums the slots in residual-block order, and computes the gradient as J'r after the parallel
pass (`BlockSparseMatrix.LeftMultiplyAndAccumulate`, row block by row block). The result
equals Ceres' single-threaded summation order for the cost; the gradient's grouping of
additions differs from Ceres' even at one thread (Ceres adds each residual block's
J_i' r_i into the gradient as it goes, which is the same order for the block-sparse
Jacobian and a column-wise order for the dense one).

**Why.** CLAUDE.md's threading rule: sequential and parallel runs must give the same
result. Ceres' per-thread reduction is scheduling-dependent, so no fixed order could
reproduce it anyway; bundle adjustment is Tier C.

**Evidence.** `BundleAdjustmentProblemTests.ThreadCount_DoesNotChangeTheResult` solves the
same problem with 1 and 4 threads for every linear solver and gets bit-identical parameters,
final cost and iteration counts. `Gradient_MatchesFiniteDifferencesOfTheCost` checks the
gradient against the cost it is the derivative of.

## 19. TinySphereManifold's tangent basis uses Hughes & Moller, not Eigen's unitOrthogonal()

**What differs.** COLMAP's `SphereManifold<3>` (`estimators/cost_functions/tiny_manifold.h`)
builds the tangent basis at x as `b1 = x_hat.unitOrthogonal()`, `b2 = x_hat.cross(b1)`.
`Estimators/CostFunctions/TinyManifold.cs` builds `b1` with the construction of Hughes &
Moller, "Building an Orthonormal Basis from a Unit Vector" (JGT 4(4), 1999): zero the
component of smallest magnitude (the first on a tie), swap the other two and negate one,
normalize. `b2` is the same cross product. The two constructions can pick a different unit
vector in the tangent plane, so the tangent coordinates of a step, and with them TinySolver's
Jacobi scaling and iterates, can differ from COLMAP's while spanning the same plane.

**Why.** Eigen documents only that `unitOrthogonal()` returns some unit vector orthogonal to
the input (its choice for 3-vectors is an implementation detail of MPL-2.0 code we may not
transcribe, CLAUDE.md contract 2). Any orthonormal tangent basis gives a valid retraction and
Plus Jacobian; TinySolver is Tier C, so what must agree is the converged result, which does
not depend on the basis.

**Evidence.** `TinyManifoldTests.SphereManifold_PlusStaysOnUnitSphere` and
`SphereManifold_PlusJacobianMatchesFiniteDiff` (1:1: orthogonality to x within 1e-12 and the
finite-difference Jacobian within 1e-6) and `TinySolverTests.TinySolver_ManifoldConvergesAndStaysOnManifold`
(1:1: the sphere fit converges to target.normalized() within 1e-6) pass.

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

## 22. SPARSE_NORMAL_CHOLESKY factors with the simplicial LLT and its own AMD ordering

**What differs.** Ceres 2.2 (as COLMAP builds it, with SuiteSparse) reorders the reduced
program's parameter blocks with CAMD on the Jacobian's block structure, forms J'J in that
order and factors it with CHOLMOD; with EIGEN_SPARSE it leaves the blocks in order and lets
Eigen's AMD permute the matrix. `Solver/LinearSolvers.cs` keeps the problem's block order,
forms the lower triangle of J'J + D^2 from the block structure, and factors it with
`LinearAlgebra/SimplicialCholesky.cs` (LLT, AMD from `AmdOrdering.cs`), the same stand-in as
entry 13. The steps agree to rounding; the last bits differ, and so can a borderline
accept/reject decision late in a solve.

**Why.** CHOLMOD is GPL/LGPL and Eigen MPL-2.0 (`docs/LICENSE_AUDIT.md`); entry 13 has the
reasoning. A factorization failure is reported as Ceres' Eigen back end reports it
("Eigen failure. Unable to find numeric factorization.", a rejected step).

**Evidence.** `CeresExampleTests.Powell` and `CurveFitting` reach the tutorial's printed
optimum, cost, iteration count and number of rejected steps with SPARSE_NORMAL_CHOLESKY as
with DENSE_QR, and `BundleAdjustmentProblemTests.RecoversGroundTruth` recovers the ground
truth with all three linear solvers.

## 24. The 7-point fundamental solver takes its null space from unpivoted Householder QR

**What differs.** `FundamentalMatrixSevenPointEstimator::Estimate` gets the 2D null space of
its 7 x 9 constraint system from columns 7 and 8 of the full Q of
`A.fullPivHouseholderQr()` (A is 9 x 7). `Estimators/Solvers/FundamentalMatrixEstimators.cs`
takes the same two columns from an unpivoted Householder QR of A
(`LinearAlgebra/Householder.FactorInPlace` / `ApplyQ`, allocation-free). Both pairs are
orthonormal bases of the same space: in any factorization A = Q R, the first seven columns
of Q contain A's column space, so the last two are orthogonal to every constraint, whatever
the pivoting and whatever A's rank. Only the basis inside that plane differs.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR (which also swaps
rows) has no published convention precise enough to reproduce Eigen's Q column for column.
The basis does not reach the solutions: the solver finds the rank-2 members of the pencil
{a f1 + b f2}, normalized to unit norm, and that set of matrices (up to sign, which the
Sampson error and COLMAP's F / F(2, 2) comparisons ignore) does not depend on the basis. The
one place it shows is the parametrization lambda f1 + f2 with f1 = q7 - q8: when
|det(f1)| < 1e-16 COLMAP returns no model, and which member of the pencil f1 is depends on
the basis. For a generic sample that is a measure-zero event on either side, so the two can
disagree only on a sample that sits on that numeric knife edge, and then one of them emits
no model for that RANSAC hypothesis.

**Evidence.** `FundamentalMatrixTests.FundamentalSevenPointEstimator_Reference` (COLMAP's
Matlab reference values, 1e-6) and `FundamentalSevenPointEstimator_Nominal` (100 random
problems, at least one model equal to the true F up to scale) pass 1:1.

## 26. re3q3's random change of variables uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3` (P4Pf's three-quadratics solver,
`Estimators/Solvers/PoseLib/Re3q3.cs`) retries a near-degenerate system (all three
elimination determinants below 1e-10) after a random affine change of variables. PoseLib
draws that from Eigen's `Quaternion::UnitRandom()` and `setRandom()`, which read the C
library's global `std::rand()`. The port draws a uniform rotation (Shoemake, Graphics Gems
III) and a shift in [-1, 1]^3 from a fresh `Mt19937` seeded with 0 on every call, through
`LibcxxRandom.UniformReal`.

**Why.** `std::rand()` is a process-global stream whose sequence depends on the C library
and on every earlier call in the process, so COLMAP's result in this branch is not
reproducible even against itself; porting it would add hidden global state shared across
threads, which CLAUDE.md rules out (sequential and parallel runs must agree). Eigen's
`UnitRandom` code is MPL-2.0 and is not ported, so the rotation follows the published method
it cites. A fixed per-call seed keeps the result a pure function of the input. Only the
degenerate branch is affected; the regular path is a straight port.

**Evidence.** The branch is only reached for near-singular inputs; the 1:1
`AbsolutePoseTests.AbsolutePose_P4PF*` cases (regular path) pass with COLMAP's tolerances.
A change of variables followed by `refine_3q3` against the original coefficients targets the
same system, so the solution set should agree up to solver tolerance whichever random matrix
is drawn. The C#-only `Re3q3Tests.CSharpOnly_DegenerateEliminationUsesRandomVarChange`
(x^2 = 1, y^2 = 4, z^2 = 9: every elimination determinant is 0) finds all eight solutions and
each satisfies the original system to 1e-8. Not compared against C++ PoseLib.
