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

## 25. Reconstruction text reading takes whole tokens; image names must be UTF-8

**What differs.** Two things in `Scene/ReconstructionIO*.cs` and `Scene/CppLineTokens.cs`:
- *Tokens.* COLMAP's text readers extract values with libc++'s `istream >>`, which stops at
  the first character that cannot continue a number and leaves the rest for the next
  extraction: `"12abc"` read into an integer gives 12, and the next read sees `"abc"`.
  `CppLineTokens` takes whitespace-separated tokens whole, so `"12abc"` (or `"1.5"` read as an
  integer) fails that read. Everything else follows libc++ as probed with a libc++ harness:
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
each satisfies the original system to 1e-8. Cross-checked against C++ PoseLib fa7280f built
with clang++ and Eigen 3.4 headers in a scratch harness: for std::srand seeds 1..20 it also
returns 8 solutions with max residual below 3e-9, and for x^2 = 1, y^2 = 4, z = x + y (a
linear third equation) it returns 0 solutions for every seed, as the port does (upstream
behavior, pinned by `Re3q3Tests.CSharpOnly_LinearEquationReturnsNoSolutionsLikePoseLib`).

## 27. The five-point solver takes its null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_5pt` (the minimal case of COLMAP's
`EssentialMatrixFivePointEstimator`) takes the 4D null space of its five epipolar
constraints from the last four columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 5 matrix).
`Estimators/Solvers/PoseLib/Relpose5pt.cs` takes the same four columns from an unpivoted
Householder QR (`LinearAlgebra/Householder.FactorInPlace` / `ApplyQ`, allocation-free). Both
are orthonormal bases of the same space (in any A = Q R the first five columns of Q contain
A's column space); only the basis inside it differs, and so do the intermediate polynomial
coefficients (`compute_trace_constraints`, the degree-10 determinant polynomial and its
roots, which parametrize the solutions in basis coordinates).

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entry 24 is the same
choice for the 7-point fundamental solver). The solution set does not depend on the basis:
the solver returns every unit-norm E = x N0 + y N1 + z N2 + N3 in the null space that meets
the cubic constraints, and a change of orthonormal basis maps that set onto itself (up to
the sign of each E, which the Sampson errors and COLMAP's tests ignore). What can differ is
numerical conditioning: the back substitution's 1e-6 test for switching to the three-row QR
solve, and the Sturm bracketing tolerance, act on basis-dependent quantities, so a solution
near those thresholds can come out with slightly different rounding, and a near-degenerate
root can be kept on one side and dropped on the other.

**Evidence.** `EssentialMatrixSolverTests.EssentialMatrixFivePointEstimatorTests_Nominal(5)`
(100 random minimal problems, at least one model within COLMAP's 5e-3 of the true E, with
Sampson residuals below 1e-5) passes 1:1, as do the over-determined (20, 1000) and LO-RANSAC
cases that sit on the same solver. Not compared against C++ PoseLib intermediates.

## 28. FromTwoVectors handles nearly opposite vectors with its own half-turn construction

**What differs.** `Quaterniond.FromTwoVectors` (the replacement for Eigen's
`Quaternion::FromTwoVectors`, used by `Synthetic.SynthesizeDataset` to aim frames) uses
Melax's shortest-arc formula like Eigen does in general, but when the two directions are
nearly opposite (1 + c < 1e-8, c the cosine between them) it composes a half turn about an
axis perpendicular to the first vector (built from the least-aligned coordinate axis) with the
well-conditioned short arc from the negated first vector to the second. Eigen switches branch
at a different threshold (1 + c < 1e-12) and picks its perpendicular axis another way, so for
1 + c < 1e-8 the returned rotation can differ from COLMAP's: for exactly opposite vectors any
half turn about a perpendicular axis is correct and the two libraries pick different ones; for
nearly opposite vectors both map the first direction onto the second, but COLMAP's general
formula there carries errors up to ~1e-8 that ours does not.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), so this branch is written here
from first principles; the threshold is where Melax's formula loses more than ~5e-9 relative
accuracy (s = sqrt(2 (1 + c)) with 1 + c known only to ~1e-16 absolute). The general branch,
where all practical inputs land, is unchanged.

**Evidence.** `QuaternionTests.CSharpOnly_FromTwoVectorsOpposite` checks exactly opposite
and nearly opposite inputs (1 + c from 0 to ~5e-9, every least-aligned axis) map the first
direction onto the second within 8e-16 with unit norm. `SyntheticOracleTests` still matches
pycolmap's frame rotations bit for bit (none of those inputs is nearly opposite; view
directions are uniform random, so 1 + c < 1e-8 has probability ~5e-9 per frame).

## 29. re3q3_rotation's pre-rotation uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3_rotation` (used by `gp3p`, which COLMAP's
`GP3PEstimator` calls; `Estimators/Solvers/PoseLib/Re3q3.cs` `SolveRotation`) rotates every
problem by a random rotation R0 before solving in Cayley parameters, so that no solution
lands on the Cayley transform's singularity (a rotation by pi). Unlike re3q3's change of
variables (entry 26), this draw happens on *every* call, not only for degenerate input.
PoseLib takes R0 from Eigen's `Quaternion::UnitRandom()`, i.e. from the process-global
`std::rand()`. The port draws it (Shoemake's uniform rotation) from a fresh `Mt19937` seeded
with 1 on every call.

**Why.** The same as entry 26: `std::rand()` is hidden global state whose sequence depends on
the C library and on every earlier call, so COLMAP's gp3p is not reproducible even against
itself, and porting it would make sequential and parallel runs disagree. Eigen's `UnitRandom`
is MPL-2.0 and not ported. The seed differs from entry 26's so the two draws are not the same
rotation. Consequence: for a fixed input the port always uses the same R0, so an input whose
true rotation is (close to) R0 composed with a rotation by pi is always ill-conditioned here,
where PoseLib would be only for an unlucky draw. That set has measure zero and RANSAC draws
other samples, so no ported test is affected.

**Evidence.** PoseLib fa7280f's gp3p, built with clang++ and Eigen 3.4 headers in a scratch
harness, returns the same four poses for `std::srand` seeds 1..5 on a fixed three-ray input,
agreeing to about 1e-10 across seeds; the port returns the same four poses to 1e-8
(`Gp3pTests.CSharpOnly_MatchesPoseLibSolutionSet`, order-insensitive, since the order depends
on R0). The 1:1 `GeneralizedAbsolutePoseTests.ParameterizedGP3PEstimatorTests_Nominal` cases
pass with COLMAP's tolerances.
each satisfies the original system to 1e-8. Not compared against C++ PoseLib.

## 30. The six-point focal relative pose solvers take their null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_6pt_shared_focal` and `relpose_6pt_onesided_focal`
(the minimal solvers of COLMAP's `RelativePoseSharedFocalEstimator` and
`RelativePoseOneSidedFocalEstimator`) take the 3D null space of their six epipolar
constraints from the last three columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 6 matrix).
`Estimators/Solvers/PoseLib/Relpose6ptSharedFocal.NullSpace3` (used by both) takes the same
three columns from an unpivoted Householder QR (`LinearAlgebra/Householder.FactorInPlace` /
`ApplyQ`, allocation-free). Both are orthonormal bases of the same space; only the basis
inside it differs, and with it the 280 (shared) or 190 (one-sided) template coefficients,
the action matrix and its eigenvalues, which parametrize F = N0 + x N1 + y N2 in basis
coordinates. The one-sided solver also orders its nine solutions by the eigenvalue order of
`LinearAlgebra/EigenSolver.cs` (Schur order), which may differ from Eigen's.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entries 24 and 27 are
the same choice for the 7-point and 5-point solvers). In exact arithmetic the solution set does
not depend on the basis (apart from solutions whose F has no N0 component, which the affine
parametrization cannot represent; a measure-zero configuration that moves with the basis).
In floating point it can: the shared-focal solver brackets the real roots of an
ill-conditioned degree-15 characteristic polynomial with Sturm sequences, and which
spurious (non-physical) real roots survive that bracketing depends on the basis, so extra
models can appear or vanish on either side. The output order is not part of the contract: COLMAP treats the models as
a set. What can differ is numerical conditioning near the solvers' thresholds (the Sturm
bracketing tolerance, the 1e-8 tests on imaginary parts and on q = 1/f^2), where a
near-degenerate root can be kept on one side and dropped on the other.

**Evidence.** `RelativePoseSharedFocalTests.RelativePoseSharedFocalEstimator_Nominal` and
`RelativePoseOneSidedFocalTests` (`_Nominal`, `_FullSphereCalibratedRays`) pass 1:1 with
COLMAP's tolerances and failure-rate bounds; with seeds 0 to 4 both failure-rate tests
observed 0 failures in 100 trials (COLMAP's bound is 3). A differential run against the
real PoseLib (the reviewer's harness, fed the same inputs to both sides): one-sided, 574 of
574 models identical; shared focal, 521 of 552 models match to 1e-6, and the true focal is
found in 295 of 300 problems on both sides, the differences being the spurious roots above.

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

## 35. The Schur solvers eliminate sequentially with dynamic-size kernels and their own sparse ordering

**What differs.** `Solver/SchurEliminator.cs`, `SchurComplementSolvers.cs`,
`ImplicitSchurComplement.cs`, `IterativeSchurSolver.cs` and `SchurOrdering.cs` port Ceres
2.2's DENSE_SCHUR, SPARSE_SCHUR and ITERATIVE_SCHUR with these differences:
1. The chunks (one E block's rows) are eliminated one after another in row order. Ceres runs
   them in parallel and serializes the updates of each reduced-camera-matrix cell and
   right-hand-side block with a mutex, so with `num_threads > 1` the order of additions into a
   cell depends on scheduling. The order here is Ceres' single-threaded order.
2. Only the dynamic-size eliminator is ported. Ceres picks a template specialization from the
   Jacobian's static block sizes (for COLMAP's BA typically `<2, 3, 6>`, and
   `SchurEliminatorForOneFBlock<2, 3, 6>` when there is a single camera block); those use
   Eigen's fixed-size products and invert 3x3 E'E blocks with Eigen's cofactor `inverse()`,
   where the port uses Ceres' naive loop kernels (`SmallBlas.cs`) and a Cholesky solve of the
   identity (Ceres' own dynamic-size path). Results agree to rounding, not bit for bit.
3. SPARSE_SCHUR factors the lower triangle of the reduced camera matrix with
   `LinearAlgebra/SimplicialCholesky.cs` (AMD on the scalar pattern). Ceres with EIGEN_SPARSE
   first reorders the F blocks by AMD on the block pattern of the Schur complement
   (`ReorderSchurComplementColumnsUsingEigen`) and factors in natural order; with SuiteSparse
   it uses CAMD/CHOLMOD. The factorization differs, as for SPARSE_NORMAL_CHOLESKY (entry 22).
4. The CG vector reductions (dot products, norms) are left-to-right sums; Ceres uses Eigen's
   (vectorized) `norm()`/`dot()` or per-thread partial sums.
5. A non-positive-definite E'E or diagonal preconditioner block becomes NaN (the LM step is
   then invalid and the radius shrinks), where Eigen's LLT leaves unspecified values.
6. Only the automatic elimination ordering is ported (one elimination group: the greedy
   independent set, `ComputeStableSchurOrdering`); a user `ParameterBlockOrdering` with
   several groups (only COLMAP's global positioner sets one) is not.

**Why.** CLAUDE.md's threading rule (sequential and parallel runs must give the same result;
Ceres' multithreaded order is not reproducible anyway), and Eigen and SuiteSparse are
replaced rather than ported (CLAUDE.md contract 2). Bundle adjustment is Tier C.

**Evidence.** The ported Ceres tests pass with Ceres' tolerances:
`SchurEliminatorTests` (reduced system and solution vs. the dense reference, 1e-14 relative),
`ImplicitSchurComplementTests.SchurMatrixValuesTest` (1e-14), `SchurComplementSolverTests`
and `IterativeSchurComplementSolverTests` (vs. DENSE_QR, 1e-10 and 1e-14),
`ConjugateGradientsSolverTests` (4 ULP). `BundleAdjustmentProblemTests` runs every Schur
solver: ground truth recovered (`RecoversGroundTruth`), DENSE_QR's optimum on noisy
observations (`EverySolver_ReachesTheSameOptimum`, cost to 1e-9 relative, parameters to
1e-6), and bit-identical results for 1 and 4 threads (`ThreadCount_DoesNotChangeTheResult`).
Not compared against C++ Ceres.

## 36. A user ParameterBlockOrdering keeps each group in insertion order

**What differs.** Ceres' `ParameterBlockOrdering` (`OrderedGroups<double*>`) stores each
group as a `std::set<double*>`, so `ApplyOrdering` lays out the blocks of one group in
increasing heap-address order. `Solver/ParameterBlockOrdering.cs` keeps each group in the
order its blocks were added (`AddElementToGroup`; moving a block to another group appends it
there). The program's parameter block order within a group, and so the Schur eliminator's
chunk order and the reduced camera system's column order, can differ from a given Ceres run.

**Why.** C# has no stable addresses, and in C++ the address order is itself an accident of
the allocator (COLMAP's global positioner adds points, frame centers and rig cameras that
live in separate hash-map nodes), so no run-to-run order exists to match. Insertion order is
deterministic and is the order the caller controls. Groups themselves are still visited in
increasing id order, and every Ceres check (independent first group, element count, unknown
blocks) is the same.

**Evidence.** `OrderedGroupsTests` (Ceres' `ordered_groups_test.cc`) and
`ReorderProgramTests.ApplyOrderingNormal` pass unchanged; neither depends on the order
inside a group. The effect on results is rounding-level (the order of additions into the
reduced system), within the Tier C outcome bar; `ProblemTests.UserOrdering_SchurSolverMatchesDenseQr`
checks the solutions against DENSE_QR.

## 37. Solve does not edit the caller's ordering, and ITERATIVE_SCHUR never falls back to CGNR

**What differs.** 1. Ceres' preprocessor removes the blocks it drops as constant from
`Solver::Options::linear_solver_ordering` itself (a `shared_ptr`, so the caller's object is
edited). `LeastSquaresSolver.Solve` edits a copy; `SolverOptions.LinearSolverOrdering` is
unchanged afterwards. 2. When that removal empties the first elimination group, Ceres swaps
SPARSE_SCHUR for SPARSE_NORMAL_CHOLESKY and DENSE_SCHUR for DENSE_QR (both done here, and
reported in `SolverSummary.LinearSolverTypeUsed`) and ITERATIVE_SCHUR for CGNR; CGNR is not
ported, so that last case ends the solve with FAILURE and a message saying so. 3. An empty
user ordering, or (for a Schur solver) one that holds only blocks the reduction removed,
trips a CHECK in Ceres' `MinNonZeroGroup` and aborts the process; here the solve ends with
FAILURE and a message naming the problem.

**Why.** 1. The in-place edit is a side effect on an input the caller may reuse (COLMAP
rebuilds its ordering before every solve, so it never sees it). 2. CGNR is excluded from the
port (PORTING_PLAN.md Phase 7); no COLMAP caller combines ITERATIVE_SCHUR with a user ordering. 3. A
library inside MatterCAD must not abort the host; the CHECK's condition is kept, only its
consequence differs.

**Evidence.** `ProblemTests.UserOrdering_ConstantFirstGroup_SwitchesSolver`: every block of the
first group constant, SPARSE_SCHUR solves as SPARSE_NORMAL_CHOLESKY and the ordering still
holds all six blocks. `ProblemTests.UserOrdering_Empty_Fails` and
`UserOrdering_OnlyConstantBlocks_Fails` cover case 3.

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

## 39. Bundle adjustment builds its problem in ascending id order

**What differs.** COLMAP's `DefaultBundleAdjuster` (estimators/bundle_adjustment_ceres.cc)
adds residuals by iterating the config's `FlatHashSet`s of images, variable points and
constant points, and `FixGaugeWithThreePoints` picks the three gauge points by iterating a
`FlatHashMap` of per-point observation counts. Abseil's hash containers iterate in an order
that is seeded per process, so COLMAP's residual order, and which three points fix the gauge,
can change from run to run. The port (`Estimators/BundleAdjustmentCeres.Default.cs`,
`BundleAdjustmentCeres.Gauge.cs`) visits the config's images and points in ascending id order
and the observation counts in first-seen order (the order residuals were added), so the
problem layout is the same on every run. The containers COLMAP already orders (`std::set` of
parameterized image and camera ids) are ordered the same way here.

**Why.** CLAUDE.md requires reproducible results; the hash order carries no meaning. The
residual order only changes floating-point summation order inside the solver (Tier C), and
the choice of gauge points only changes which three points stay fixed, never how many.

**Evidence.** `BundleAdjustmentCeresTests` (bundle_adjustment_ceres_test.cc) pass with
COLMAP's exact parameter counts, e.g. `FixGaugeWithThreePoints` and `TwoViewRig` (97 variable
points). `BundleAdjustmentOracleTests` matches pycolmap's final cost to 1e-6 relative and its
poses and points to 1e-6 on a 16-image scene with the TWO_CAMS_FROM_WORLD gauge.

## 40. ExtractTopScaleFeatures keeps equal-scale keypoints in input order

**What differs.** COLMAP's `ExtractTopScaleFeatures` (feature/utils.cc) selects the largest
scales with `std::partial_sort`, which leaves keypoints of equal scale in an unspecified,
implementation-defined order (libc++'s heap-based selection). The port
(`Feature/FeatureUtils.cs`) sorts by scale, largest first, with the input index as the
tie-break.

**Why.** Equal scales are common: the SIFT extractor emits one keypoint per orientation, and
all of them share the scale. Reproducing libc++'s heap order exactly would tie the result to
one standard library; the index tie-break is deterministic and stable. Only the order among
equal scales differs, and *which* keypoints are kept differs only when a tie straddles the
cut, where COLMAP's choice is arbitrary too.

**Evidence.** `FeatureUtilsTests.ExtractTopScaleFeatures_Nominal` (utils_test.cc 1:1, distinct
scales) passes unchanged.

## 41. The macOS arm64 wheel's VLFeat fuses multiply-adds; the SIFT port does not

**What differs.** COLMAP compiles VLFeat's C with the platform compiler's default
floating-point contraction. On macOS arm64, Apple clang's default is `-ffp-contract=on`, so
expressions such as the Gaussian convolution's `acc += v * c` (imopv.c), the descriptor's
bin weights and the fast atan2/resqrt polynomials become fused multiply-adds that round once.
The port (`Feature/VLFeat/`) evaluates every `a*b + c` as a rounded product plus a rounded
sum (CLAUDE.md "No FMA"), which is what VLFeat computes with contraction off and on any
target without FMA contraction. The scale space therefore differs from the wheel's in the
last bits, and those differences propagate to keypoint positions, scales and orientations
(about 1e-3 at most on the fixtures) and, rarely, to a descriptor byte (by one gray level).
A detection threshold could in principle flip on some image, adding or dropping a keypoint.

**Why.** Emulating contraction would mean choosing, per expression, whatever the C compiler
chose - which is compiler- and flag-specific (x86 builds, Linux wheels and older compilers
do not fuse) - and `Math.FusedMultiplyAdd` is banned in math paths so results are identical
on every platform. VLFeat's source defines the algorithm; the unfused evaluation is its
literal semantics.

**Evidence.** `oracle/sift_harness.c` drives cpp-reference's VLFeat the way COLMAP does. Built
with `-ffp-contract=off` (and without SSE2, like COLMAP's arm64 build), its keypoints,
orientations and raw float descriptors match the port bit for bit on all six fixture cases
(`VlSiftFilterTests.CSharpOnly_MatchesUnfusedVLFeatExactly`: upsampling by one and two
octaves, direct copy, downsampling, upright). Built with the default flags, the same harness
prints the first keypoint of the sift_test.cc square image at x = 98.7666931, exactly the
wheel's value, while the unfused build prints 98.7667007, exactly the port's. Against the
wheel (`SiftOracleTests.CSharpOnly_MatchesPycolmapWithinFmaTolerance`, nine option sets) the
keypoint counts and order agree exactly, values within 1.03e-3, descriptor bytes within 1 on
at most 0.05% of the bytes.

The covariant extractor (`Feature/CovariantSift.cs`, VLFeat covdet) has the same split:
`sift_harness.c`'s covdet mode, unfused, matches `VlCovDetTests` bit for bit on eight cases
(every feature frame and score, every raw descriptor of every DSP scale), while against the
wheel (`CovariantSiftOracleTests`) the frames agree within 4e-4. Fusing also reaches two
discontinuities there: on the symmetric square two orientations of one feature score equally,
so which comes first flips; and on the texture with domain-size pooling one keypoint (of 184)
gets a descriptor up to 4 gray levels off, because the patch extractor picks its scale-space
level with a floor of a log2 that lands on the other side of an integer for one pooled scale.
`CSharpOnly_KnownDivergences` pins that case.

## 42. FeatureDescriptorIndex is an exact nearest-neighbor search, not faiss's IVF index

**What differs.** COLMAP's `FeatureDescriptorIndex` (feature/index.cc) wraps faiss: an exact
`IndexFlatL2` for fewer than 512 indexed descriptors, and for 512 or more an inverted-file
index (`IndexIVFScalarQuantizer` with `QT_8bit_direct` for SIFT, `IndexIVFFlat` otherwise;
4 sqrt(N) k-means centroids, searched with `nprobe = 8`). The IVF search is approximate: it
only visits the 8 nearest clusters, so it can miss a query's true nearest or second-nearest
neighbour. The port (`Feature/FeatureDescriptorIndex.cs`) always returns the exact k nearest
neighbours, ordered by (squared distance, index). SIFT descriptors are compared as integers
(exact); other descriptor types sum (a - b)^2 in float, left to right, where faiss's flat
index uses the ||a||^2 + ||b||^2 - 2ab expansion.

**Why.** faiss is native code (CLAUDE.md contract 1). Porting it, k-means training included,
would reproduce a randomized approximation whose exact output also depends on faiss's
OpenMP partitioning and BLAS. The exact search is the result the IVF index approximates, and
it makes the default (index) matching path agree with COLMAP's own brute-force matcher, which
is what `SiftCPUFeatureMatcherFaissVsBruteForce` asserts. On an image pair with 512+
features, the SIFT matches can therefore differ from COLMAP's default CPU matcher wherever
faiss's approximation missed a neighbour (COLMAP returns fewer or different matches there);
they equal COLMAP's `cpu_brute_force_matcher` matches except where acos-of-dot and sqrt-of-L2
round a threshold comparison differently, which COLMAP's two paths also do. Ties in distance
are ordered by index; faiss leaves them unspecified, and the ratio test rejects a tie for the
best neighbour either way.

**Evidence.** `FeatureDescriptorIndexTests` (index_test.cc 1:1, including 1000 descriptors,
where COLMAP uses IVF) and `SiftMatcherTests.SiftCPUFeatureMatcherFaissVsBruteForce_Nominal`
(sift_test.cc 1:1) pass. Speed: an 8192 x 8192 SIFT pair with cross-check takes about 114 ms
through the index and 51 ms brute force on a 10-core Apple M-series machine (Release build).

## 43. SIFT never reuses a previous image's gradient, and extraction can be cancelled

**What differs.** COLMAP's SiftCPUFeatureExtractor keeps one VLFeat filter while the image
size stays the same, and VLFeat resets the octave its gradient cache belongs to (`grad_o`)
only in `vl_sift_new`, not in `vl_sift_process_first_octave`. When the previous image's last
octave with keypoints is the new image's first octave, the new image's orientations and
descriptors in that octave are computed from the previous image's gradient. That always
happens with `num_octaves = 1`, and with the default options whenever the previous image's
keypoints all sit in the finest octave (a low-contrast, fine-textured photo). The port
(`Feature/VLFeat/VlSiftFilter.cs`) resets the cache at the start of every image, so a reused
extractor gives exactly what a fresh one gives.

Separately, `FeatureExtractor.Extract` takes a `CancellationToken` (COLMAP's has none). The SIFT
extractor checks it before building the scale space, before every Gaussian level and DoG
extremum scan of every octave, and before each DoG level's orientations and descriptors. A
cancel throws `OperationCanceledException` and leaves the outputs untouched. The extractor
then drops its filter, so the next image is extracted exactly as by a fresh extractor.
Uncancelled results are unchanged.

**Why.** The stale gradient is an upstream bug: it makes COLMAP's features for an image depend
on which image the same worker extracted before it, so on image order and on how images are
spread over extraction threads. MatterCAD needs results that don't depend on either.
Cancellation is needed because one 12 MP extraction takes seconds and the user must be able
to stop a reconstruction (CLAUDE.md "Cancellation and progress").

**Evidence.** `oracle/fixture_sift_reuse.py` writes `sift_reuse.json`: pycolmap's features of
a texture image B from a fresh extractor and from one reused after a fine-noise image A of the
same size whose 23 keypoints are all in octave -1. The reused extractor gives 155 keypoints
where the fresh one gives 154 (default options), and 30 where it gives 29 (`num_octaves = 1`).
`SiftExtractorReuseTests` (C#-only) requires the port's reused extractor to equal its fresh
one bit for bit, and to match pycolmap's fresh output within the FMA tolerance of entry 41;
before the reset the port reproduced pycolmap's 155 and 30. `SiftCancellationTests` (C#-only):
a pre-cancelled token throws at the first check before any allocation; a cancel at a check in
the middle of an image throws at that check; runs with a live token, with no token, and on the
same extractor after a cancel give bit-identical features.

## 44. Rotation averaging lays out its linear system in ascending id order

**What differs.** COLMAP's `RotationAveragingProblem` allocates frame parameters in the
iteration order of a `FlatHashSet<frame_t>`, camera (unknown cam_from_rig) parameters in the
order of a `NodeHashMap<camera_t, int>`, and constraint rows in the order of
`PoseGraph::ValidEdges()` (a `NodeHashMap`); `ComputeResiduals` and the IRLS weights walk the
pair constraints in `NodeHashMap<image_pair_t, …>` order, which is also the order of the
jitter draws near the ±π boundary of the 1-DOF residual. The gauge is fixed at the *first*
active frame in hash order (the first gravity-aligned one, if any).
`Estimators/RotationAveragingProblem*.cs` allocates frames and cameras in ascending id order,
rows in ascending pair id order, visits constraints in that order, and fixes the gauge at the
smallest-id (gravity-aligned, if any) active frame.

The driver code (`Estimators/RotationEstimator.cs`, `Estimators/RotationAveraging.cs`) does the
same where COLMAP walks hash containers: the maximum spanning tree numbers the active images in
ascending id order (COLMAP: `FlatHashSet` order, and node 0 is the tree's root, which is held
at the identity rotation) and adds edges in ascending pair id order (which breaks weight ties);
the stratified gravity subset copies its edges in ascending pair id order;
`InitializeRigRotationsFromImages` walks frames in ascending id order, which fixes the order in
which quaternion samples are summed; `CreateExpandedReconstruction` numbers its singleton rigs
in ascending rig id (then sensor) order and its new frames in ascending frame id (then data id)
order.

**Why.** CLAUDE.md's hash-order rule: the layout changes the floating-point grouping of the
normal equations and the Cholesky ordering, the jitter draw order, and the gauge (the solution
is only defined up to a global rotation, so a different fixed frame gives a globally rotated
result). COLMAP relies on none of these: its tests compare gauge-invariant relative rotations.

**Evidence.** `RotationAveragingTests` (rotation_averaging_test.cc 1:1, all 15 cases) pass
with COLMAP's tolerances, including `DeterministicRandomSeed` and the 1e-12 degree
`RidgeRegularizationDoesNotBiasSolution`; the `CSharpOnly_*` cases recover the synthetic scenes' relative
rotations within COLMAP's 1e-2 degree bar, and `CSharpOnly_DeterministicWithSeed` pins
bit-identical repeat solves.

## 45. Covariance factors a dense Jacobian instead of Ceres' sparse QR

**What differs.** COLMAP's pose refinements (`RefineAbsolutePose`,
`RefineGeneralizedAbsolutePose`) call `ceres::Covariance` with default options: SPARSE_QR,
which factors the problem's sparse tangent-space Jacobian with a column-pivoting sparse QR
(SuiteSparseQR or Eigen's `SparseQR` with COLAMD ordering) and reads (J'J)^-1 from R. The
port (`Solver/Covariance.cs`) builds the same Jacobian densely (loss function applied,
constant blocks and blocks no residual uses without columns, as in Ceres' ComputeCovarianceSparsity) and factors it with the dense column-pivoting Householder
QR of `LinearAlgebra/ColPivHouseholderQR.cs`; rank deficiency (by that QR's rank threshold)
makes `Compute` fail, as in Ceres. The covariance is the same matrix up to round-off.

**Why.** SuiteSparseQR is GPL/LGPL and Eigen's sparse QR is MPL-2.0 (CLAUDE.md contract 2).
COLMAP's covariance callers are single poses (plus a camera) against their observations, a
few columns, where dense is exact enough and fast. The Phase 8 BA covariance
(`estimators/covariance`) is COLMAP's own Schur-based code and does not use this.

**Evidence.** `CovarianceTests` (the SPARSE_QR legs of Ceres' covariance_test.cc
NormalBehavior, ManifoldInTangentSpace and ManifoldInTangentSpaceWithConstantBlocks, at
Ceres' 1e-5 tolerance) and the covariance checks of `PoseEstimationTests` pass.

## 46. IsPanoramicRig compares rig camera origins to the smallest camera index

**What differs.** `IsPanoramicRig` (estimators/generalized_pose.cc) decides whether all
cameras used by a set of correspondences share one optical center: it puts the camera
indices in a `FlatHashSet`, takes the set's first element as the reference, and checks
every other camera's origin with `isApprox(reference, 1e-6)`. The set's iteration order is
unspecified. The port (`Estimators/GeneralizedPoseEstimation.cs`) takes the smallest camera
index as the reference.

**Why.** Hash iteration order is not reproducible across containers (CLAUDE.md, hash
container rule). The choice only matters when origins agree to about 1e-6 relative with some
reference but not another (isApprox is not transitive), i.e. never for a real rig, where
cameras are either co-centered (panoramic) or centimeters apart.

**Evidence.** `GeneralizedPoseEstimationTests` (generalized_pose_test.cc 1:1), whose
EstimateGeneralizedRelativePose_Nominal covers panoramic and non-panoramic rigs, passes.

## 47. Gravity refinement visits error-prone frames and their neighbor pairs in id order

**What.** `GravityRefiner.RefineGravity` (Estimators/GravityRefinement.cs) walks the
error-prone frames, each frame's neighbor pairs and the image adjacency sets in ascending id
order. COLMAP walks absl `FlatHashSet`s, whose order is unspecified.

**Why.** CLAUDE.md requires deterministic iteration. The order reaches the result because
an accepted frame's refined gravity is written into the shared pose prior, which later frames
read as a neighbor gravity; the neighbor order also sets the residual order of each small
solve. There is no order to match: COLMAP's is a property of absl's hash seed.

**Evidence.** `GravityRefinementTests` (gravity_refinement_test.cc 1:1) pass at COLMAP's
1e-2 degree tolerance with 30% gravity outliers.

## 48. Global positioning keeps frame centers and cameras-in-rig in id order

**What.** `GlobalPositioner` (Estimators/GlobalPositioning.cs) keeps its frame centers,
cameras-in-rig and points sorted by id, where COLMAP uses `NodeHashMap`s. The order decides
which frame receives which random initial center (drawn in frame-id order here) and the
element order within the Schur ordering groups.

**Why.** CLAUDE.md requires deterministic iteration. With a fixed seed COLMAP's own random
starts depend on its hash order, so they cannot be reproduced; the solve converges to the
same positions up to the gauge.

**Evidence.** `GlobalPositioningTests`: `GlobalPositioning_Nominal`,
`GlobalPositioning_MultiCameraRig` and `RefineSensorFromRigFalsePreservesRig` (all 1:1) pass
with COLMAP's `ReconstructionNear` bounds.

## 49. Alignment visits hash containers in a fixed order

**What differs.** Three places in `colmap/estimators/alignment.cc` iterate hash containers
where the order reaches a result; `Estimators/Alignment*.cs` fixes each order:
- `MergeReconstructions` copies the source images missing from the target in the iteration
  order of a `FlatHashSet`, which decides the order their frames are registered in the target
  (`RegFrameIds`). The port copies them in the source's `RegImageIds()` order (registration
  order). The source points are merged in `Points3D` order (ascending id, entry 21), which
  decides the ids the new target points get.
- `AlignReconstructionsViaPoints` pairs each source point with the target point seen most
  often along its track via `std::max_element` over a `FlatHashMap`, so a tie goes to
  whichever id the hash map yields first. The port breaks ties by first appearance along the
  source track.
- `AlignReconstructionToOrigRigScales` sums the per-rig scales in the order of the caller's
  rig map (a `NodeHashMap` in C++, the caller's `IReadOnlyDictionary` here), so the mean scale
  can differ from a given COLMAP build in the last ulp. Within a rig the sensors are visited in
  ascending id order in both (`std::map`).

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order: its
tests check only counts (`MergeReconstructions`) and the recovered transform within 1e-6.

**Evidence.** `AlignmentTests` (alignment_test.cc 1:1) pass: `MergeReconstructions`,
`AlignReconstructionsViaPoints` and `AlignReconstructionToOrigRigScales` with COLMAP's
expectations and tolerances.

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

## 52. SpatialPairGenerator ranks neighbors by coordinate-difference distances, not faiss's

**What differs.** COLMAP's `SpatialPairGenerator` (controllers/pairing.cc) finds each
image's nearest position priors with faiss's brute-force `IndexFlatL2`, on float positions
centered on their mean. The port (`Controllers/SpatialPairGenerator.cs`) runs a managed
brute-force search over the same float positions, ranking by the float squared distance
summed from coordinate differences, ties by the smaller index. faiss does the same for fewer
than 20 query positions (`distance_compute_blas_threshold`), but from 20 on it computes
`||x||^2 + ||y||^2 - 2 x.y` with a BLAS matrix product, whose rounding differs, so for
larger datasets two neighbors at (nearly) equal distance can come out in the other order,
and a neighbor right at `max_distance` can fall on the other side of the cut-off. COLMAP
also runs the search on `num_threads` OpenMP threads; the port has no such option. The
mean position subtracted before the float cast is summed sequentially per column, while
Eigen's `colwise().mean()` may accumulate in vectorized blocks (e.g. four partial sums), so
the double mean can differ by an ulp, which can move a float position by one ulp.

**Why.** faiss is native code (docs/LICENSE_AUDIT.md). The difference form is also the more
accurate one: the BLAS form cancels catastrophically for nearby points, which is exactly
what spatial matching looks at. The set of pairs only changes at exact ties or at the
distance cut-off.

**Evidence.** `PairingTests` (the SpatialPairGenerator cases of pairing_test.cc, all with
fewer than 20 positions) pass with COLMAP's expected pair order.

## 53. Problem.GetParameterBlocks lists blocks in insertion order

**What differs.** Ceres' `Problem::GetParameterBlocks` walks the problem's `std::map` keyed by
the blocks' addresses, so it lists them in memory-address order. The port
(`Solver/Problem.cs`) lists them in insertion order. The one caller,
`GetOtherParams` in `Estimators/Covariance.cs` (colmap/estimators/covariance.cc), uses the order
to lay out the "other" blocks (camera intrinsics, sensor_from_rig poses) after the poses in
the Schur complement and in `L_inv`.

**Why.** .NET arrays have no stable address, and address order is an accident of the
allocator in C++ as well. The layout changes only the elimination order inside the sparse
factorization, not the covariances beyond round-off.

**Evidence.** `BACovarianceTests` (covariance_test.cc 1:1) checks `GetOtherParamsCov` against
ceres::Covariance at COLMAP's 1e-8 tolerance.

## 54. BA covariance factors with our sparse products and Cholesky instead of Eigen's

**What differs.** COLMAP's `EstimateBACovariance` (colmap/estimators/covariance.cc) forms the
Schur complement with `Eigen::SparseMatrix` products, eliminates the other parameters with
`Eigen::SimplicialLLT`, and factors the result with `Eigen::SimplicialLDLT` (AMD ordering)
before inverting L densely. The port (`Estimators/Covariance.Schur.cs`) does the same steps
with `LinearAlgebra/SparseMatrixCsc.cs` and `LinearAlgebra/SimplicialCholesky.cs`, whose AMD
ordering and summation order differ from Eigen's. The damping, the rank test (|D| > 1e-6),
the D^-1/2 row scaling with `DBL_MIN` floor and the permutation are COLMAP's. The results
agree to round-off.

**Why.** Eigen is MPL-2.0 (CLAUDE.md contract 2), so its sparse module is replaced, not ported.

**Evidence.** `BACovarianceTests` (covariance_test.cc 1:1, all seven instantiations) passes at
COLMAP's 1e-8 element tolerance against ceres::Covariance; the largest observed difference is
about 2e-12 on pose covariances of about 8e-5.

## 57. The covariant SIFT extractor orders equal (octave, level) features stably

**What differs.** COLMAP's `CovariantSiftCPUFeatureExtractor` (feature/sift.cc) sorts the
covdet features by (octave, level), both descending, with `std::sort`, which leaves features
with equal keys in an implementation-defined order (libc++'s introsort; it insertion-sorts,
stably, only short ranges). The port (`Feature/CovariantSift.cs`) sorts with the covdet index
as the tie-break, so equal keys keep detection order. The keypoints and descriptors are the
same set, but their order within an (octave, level) group can differ once there are more
features than libc++ insertion-sorts. It also changes one keypoint when `max_num_features`
truncates: COLMAP pushes the first keypoint of the next group before it stops, and which
keypoint is first there is the tie order.

**Why.** Reproducing libc++'s `std::sort` exactly would tie the result to one standard
library version (its thresholds and partitioning changed between LLVM releases); the stable
order is deterministic, as for entry 40.

**Evidence.** `CovariantSiftOracleTests`: on the texture fixtures (184 and 191 features)
every keypoint pairs with a pycolmap keypoint within the FMA drift of entry 41, but not index
by index. With `max_num_features = 20` both give 26 keypoints; the first 25 pair up and the
26th differs (`CSharpOnly_KnownDivergences`). The five covariant rows of
`SiftTests.SiftCpuExtraction_Nominal` (sift_test.cc 1:1) pass.

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

## 62. Reading a truncated MVS .bin file throws

**What differs.** `Mvs/Mat.cs` `Read` (and so `DepthMap`/`NormalMap`, and
`ConsistencyGraph`'s payload reader) throws `EndOfStreamException` when the file holds fewer
elements than its `width&height&depth&` header announces. COLMAP's `Mat<float>::Read` reads
element by element with `ReadBinaryLittleEndian`, whose stream silently fails at end of file,
leaving the rest of the buffer with whatever the stream read produced (unspecified values)
and no error.

**Why.** A real upstream robustness bug: a half-written depth map (a crash or a cancelled
PatchMatch run, which MatterCAD's cancel button makes routine) would otherwise be fused as
garbage depths. Well-formed files read identically.

**Evidence.** `MatTests.Mat_ReadInvalid` (C#-only) pins the throw; `Mat_WriteReadByteExact`
and `ConsistencyGraph_WriteByteExact` pin that complete files are byte-identical to COLMAP's
format.

## 63. The MVS projection matrices can differ from COLMAP's in the last float bits

**What differs.** Two things in `Mvs/Image.cs`, both Tier B:
- `ComposeInverseProjectionMatrix` inverts the float 4x4 `[K [R | T]; 0 0 0 1]` with the
  textbook adjugate (Laplace expansion over 2x2 minors) where COLMAP calls Eigen's
  `Matrix4f::inverse()`. Both are exact up to float rounding, but the rounding of the
  operations differs, so `GetInvP` can differ from COLMAP in the last bits.
- The fixed-size float products (`P = K [R | T]`, `RotatePose`, `ComputeRelativePose`,
  `ComputeProjectionCenter`) are summed in Eigen's coefficient order as separate multiplies
  and adds. Eigen 3.4 on aarch64 may evaluate them with a fused multiply-add (`pmadd`), and
  compilers with `-ffp-contract=on` may fuse them too, so last-bit differences are possible
  from FMA contraction on the C++ side (as in entry 1). K, R and T themselves, the sizes and
  Rescale's K scaling are exact.

**Why.** Eigen is excluded (docs/LICENSE_AUDIT.md); its 4x4 inverse cannot be transcribed.
CLAUDE.md's "No FMA" rule keeps the products unfused so results are the same on every
platform.

**Evidence.** `MvsImageTests.Image_InverseProjectionMatrix` (C#-only): `P * InvP` is the
identity within 8 float epsilons of the summed products for a realistic camera. The ported
image_test.cc cases pass with gtest's 4-ulp EXPECT_FLOAT_EQ. No oracle fixture pins the
products bit for bit. PatchMatch (the only consumer) is Tier C.

## 64. GetMaxOverlappingImages orders equal shared-point counts by image index

**What differs.** `mvs::Model::GetMaxOverlappingImages` (model.cc) sorts each image's
overlapping images by shared-point count, descending, with `std::partial_sort` or
`std::sort`, which leave images with equal counts in an implementation-defined order. The
port (`Mvs/Model.cs`) breaks ties by ascending image index, the order the candidates come
out of COLMAP's `std::map`. When a count tie straddles the `num_images` cut-off, the chosen
source images can differ from COLMAP's.

**Why.** Reproducing libc++'s heap-based `partial_sort` and introsort exactly would tie the
result to one standard library version; the index order is deterministic (as entries 40
and 57).

**Evidence.** `ModelTests.Model_GetMaxOverlappingImagesTies` (C#-only) pins the order;
`Model_GetMaxOverlappingImages` (model_test.cc 1:1) passes.

## 77. ComputeNormalizedMinGraphCut partitions with our own multilevel bisection, not METIS

**What differs.** COLMAP's `ComputeNormalizedMinGraphCut` (math/graph_cut.cc) calls
`METIS_PartGraphKway` with default options. The port (`Mathematics/GraphCut.cs`) builds the
same CSR graph (vertex indices by first appearance, parallel edges kept) and hands it to
`Mathematics/MultilevelPartitioner.cs`, written here from the published multilevel scheme
(Hendrickson & Leland 1995; Karypis & Kumar, SIAM J. Sci. Comput. 1998; Fiduccia &
Mattheyses 1982): heavy-edge-matching coarsening (followed, when over 10% of the vertices
stay unmatched, by pairing unmatched vertices that share a neighbor and unmatched isolated
vertices, so stars, hub images and many small components still coarsen), greedy graph
growing from eight seeds on the coarsest graph (a vertex too heavy to fit is skipped), FM refinement at every level, and recursive bisection for k parts
(floor(k/2) parts on the first side). METIS's k-way path instead refines all k parts at once
with its own greedy k-way refinement and randomizes its visit orders with GKlib's RNG. So the
labels, which part gets which number, and the exact cut can differ from COLMAP's. Each
bisection allows 3% over its target weight (METIS's k-way `ufactor` default of 30), rounded
up to a whole vertex so that small graphs can always be split. Every tie is broken by vertex
index, so the output is deterministic. Self-loops are ignored (they never cross a cut).

**Why.** Porting the reached METIS subset (coarsening, recursive-bisection initial
partitioning, 2-way and k-way FM refinement, balancing, GKlib's priority queues and RNG) is
well over the ~3k-line budget set for this step, and a close match would also need GKlib's
random stream reproduced exactly. COLMAP's contract, and all its callers need (scene
clustering), is a balanced partition with a small cut; graph_cut_test.cc checks the label
range, that both parts are used, and the component split of a disconnected graph. No METIS
code was read or transcribed, so METIS's notice is not needed.

**Evidence.** Tier C. The four ported `GraphCut_ComputeNormalizedMinGraphCut*` cases in
GraphCutTests pass. NormalizedMinGraphCutTests (C#-only) checks that planted clusters (2-5
dense clusters in a ring of light edges, ten random draws each) come out one part per
cluster, that random graphs of 100-400 vertices split into 1, 2, 3, 5 and 8 parts give
non-empty parts within 10% (plus three vertices) of n/k and identical output on a second
call, that a 100 x 40 unit grid is bisected with 43 cut edges against an optimum of 40, and
that the partitioner's step count grows less than 6x from n = 5k to 20k on a star, a
20-hub graph and disconnected pairs (a quadratic step would give 16x). In Release on an
Apple arm64 laptop, k = 2 takes 42-162 ms for stars of 10k-40k leaves, 21-58 ms for
10k-40k leaves on 20 hubs, 2-22 ms for 10k-40k disconnected pairs, and 0.11 s / 2.4 s for
random graphs of 10k / 100k vertices with 5 edges per vertex.

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
