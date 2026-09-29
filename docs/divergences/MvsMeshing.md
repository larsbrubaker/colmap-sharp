# Divergences from COLMAP: Mvs, meshing: simplification, Poisson, Delaunay and texturing

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 72. SimplifyMesh collapse costs can differ from COLMAP's in the last bits

**What differs.** COLMAP's `ComputeEdgeCollapse` (mvs/mesh_simplification.cc) solves the
4x4 system for the optimal position with Eigen's closed-form 4x4 `determinant()` and
`inverse()` (vectorized, and possibly contracted into FMAs by Apple clang). The port
(`Mvs/MeshSimplifier.Candidates.cs`) uses the Laplace expansion over 2x2 minors of
`LinearAlgebra/Matrix4d.cs`. When the system is non-singular, which happens at the corners
and along the boundary when boundary_weight > 0 and on curved surfaces, the position and
cost can differ in the last bits. Two costs that are equal in C++ can then differ here (or
the reverse), so the heap can pop them in a different order and the meshes diverge from
that collapse on. The heap is a port of libc++'s (`Mvs/CollapseHeap.cs`) with COLMAP's
cost-only comparator, so ties that are exact on both sides pop in the same order. Tier C.

**Why.** Eigen is not ported (docs/LICENSE_AUDIT.md), and reproducing its SIMD evaluation
order and the compiler's contraction choices is not possible from C#. See CLAUDE.md, "No FMA".

**Evidence.** `MeshSimplificationOracleTests` (fixture `oracle/fixture_mesh_simplification.py`):
- On flat grids with boundary_weight 0, every system is singular. The fallback uses only
  quadric evaluations, and every cost is exactly 0. The output is byte-identical to
  pycolmap's in 8 of 8 cases (grids 4 to 100 at ratios 0.1 to 0.5), and those tests assert
  exact equality.
- On a curved 30x30 grid under six option sets (boundary_weight 0, 1000 and 1e6, and
  max_error 0, 1e-6 and 1e-3), the output was byte-identical when generated. The tests
  allow 1e-5 on positions.
- On a curved 1,002,528-face grid, the faces are identical, and one of the 50,518 vertices
  differs by 9.3e-10 (one float ulp).

## 73. SimplifyMesh visits boundary edges in vertex-index order

**What differs.** COLMAP collects the edges that have a single face into a `NodeHashMap`
(`boost::unordered_node_map` with its `PairHash`). It adds each edge's boundary quadric to
the edge's two vertices in the map's iteration order, and that order sets the order of the
floating-point sums into each vertex's quadric. The port (`Mvs/MeshSimplifier.cs`,
`AddBoundaryQuadrics`) visits the edges by (smaller, larger) vertex index, taking them from
the vertex adjacency instead of a hash map.

**Why.** COLMAP's order is deterministic too, but reproducing it would mean porting Boost's
table layout and growth policy (hash-to-bucket mapping and group iteration). The effect is
limited to the last bits of a boundary vertex's quadric, since each boundary vertex receives
only two or three boundary quadrics. With boundary_weight 0 no boundary quadrics are added,
so on flat grids the effect is provably zero.

**Evidence.** The curved-grid oracle cases with boundary_weight 1000 and 1e6, and the color
cases (boundary_weight 1000), match pycolmap's positions and faces byte for byte
(`MeshSimplificationOracleTests`).

## 74. PoissonRecon arithmetic is strict IEEE, not -ffast-math

**What differs.** COLMAP compiles its vendored PoissonRecon with `-funroll-loops -ffast-math`
(`src/thirdparty/PoissonRecon/CMakeLists.txt`, non-MSVC). Fast-math lets the compiler
reassociate sums, contract multiply-adds, replace divisions by reciprocal multiplies and
assume no NaN/Inf, so the numbers the pycolmap wheel's `poisson_meshing` produces depend on
that compiler's choices. The port (`ColmapSharp/Mvs/PoissonRecon/`) performs every operation
as written in the C++ source, in source order, with no contraction.

**Why.** CLAUDE.md's "No FMA" rule and cross-platform determinism: fast-math output is not a
property of PoissonRecon but of one compiler build, and cannot be reproduced from managed
code. Matching the source's arithmetic is the reproducible target.

**Evidence.** The building blocks are pinned bit for bit against the same templates built
with `-ffp-contract=off` and without fast-math (`oracle/poisson_bspline_harness.cc`,
`PoissonBSplineOracleTests`). The meshing pipeline as a whole is Tier C against pycolmap
(poisson_meshing_test.cc is itself an outcome test).

## 75. The Poisson normal transform uses a correctly rounded pow(x, 1./3), not libm's

**What differs.** PoissonRecon scales the normal transform by `pow( fabs( det ) , 1./Dim )`
(`TransformedInputOrientedSampleStream`, Reconstructors.streams.h), taking `pow` from the
platform libm. The port computes the correctly rounded double of x^fl(1/3) in managed code
(`Mvs/PoissonRecon/PowOneThird.cs`: a double-double Newton step on the cube root plus the
x^-(1/3 - fl(1/3)) correction). Apple's libm, which the macOS pycolmap wheel uses, is not
correctly rounded for every input, so on those inputs the two differ in the last bit of the
double (and possibly of the float the C++ casts it to).

**Why.** .NET's `Math.Pow` is the host's libm (and browser-wasm has its own), so using it
would make the unit-cube normals depend on the platform. A correctly rounded result is the
one value every platform can reproduce, and it equals Apple's wherever Apple's is correctly
rounded.

**Evidence.** `PoissonTreeOracleTests.PowOneThird_IsCorrectlyRoundedAndMatchesLibmWhereLibmIs`:
over 4000 float arguments spanning 2^-80..2^80, the port matches the fixture's 60-digit
correctly rounded values in all cases, and Apple's libm in all but the 7 where libm is off by
one ulp. For COLMAP's inputs the argument is the determinant of the unit-cube scaling,
typically far from those cases, and the tree-stage fixtures match bit for bit.

## 76. Poisson meshing runs in memory; the file-to-file call is a wrapper

**What differs.** COLMAP's `PoissonMeshing(options, input_path, output_path)` hands PoissonRecon
a PLY file and gets a PLY file back (`RunPoissonRecon` / `RunSurfaceTrimmer` parse command
lines and stream the files). The port's core takes the points, normals and optional colors
as float arrays (`Mvs/PoissonRecon/PoissonSampleSet.Build` is the first stage) and will
return an in-memory mesh with per-vertex colors and optional density. A thin wrapper with
COLMAP's file-to-file signature (reading and writing PLY through `Util/Ply*`) serves the
ported poisson_meshing_test.cc cases. The stages take a `CancellationToken` and report
`IProgress<PoissonProgress>` (tree build, density, splat, solve per depth, level set, trim),
which the C++ has no equivalent for.

**Why.** MatterCAD reconstructs from memory and must show progress and let the user cancel
(CLAUDE.md); writing temporary PLY files would add I/O and a browser-wasm file-system
dependency for nothing. The numbers are unchanged: the samples are consumed in the same
order the PLY stream would deliver them, as float (PoissonRecon's Real).

**Evidence.** `PoissonTreeOracleTests` feeds the harness's points through the in-memory API
and matches PoissonRecon's tree and accumulated samples bit for bit.

## 90. Texture mapping's occlusion test runs on a BVH instead of CGAL's AABB tree

**What differs.** COLMAP's `OcclusionTester` (mvs/texture_mapping.cc) builds a CGAL
`AABB_tree` over the mesh (float kernel) and asks `any_intersection` for the segment from
the camera center to kEps short of a face corner. The port (`Mvs/TriangleBvh.cs`, used by
`Mvs/TextureMapping.Views.cs`) is a binned-SAH bounding volume hierarchy with a
Möller–Trumbore segment/triangle test in double precision. Differences:
- **The face itself.** CGAL returns *an arbitrary* intersected primitive; if that happens to
  be the face being tested, COLMAP answers "not occluded" even when another triangle also
  blocks the segment. The port skips the face's own triangle and keeps looking, so a face
  is occluded exactly when some *other* triangle blocks it. COLMAP's answer depends on
  CGAL's traversal order; the port's does not.
- **Edges and corners** count as hits (inclusive barycentric bounds). CGAL decides these
  with float predicates, so for segments passing within rounding of a triangle edge the two
  can disagree.
- **Coplanar / parallel segments** never hit. CGAL reports a coplanar overlap as a segment,
  which COLMAP ignores (it only uses point intersections), so the outcome is the same; the
  port also treats a segment within a relative 1e-12 of parallel as a miss.
- **Hit distance.** COLMAP compares the distance of CGAL's float intersection point with
  `dist - kEps`. The port computes the point in double, rounds it to float, and measures its
  distance in float the same way, so only the point's rounding can differ.
- **Always on.** COLMAP skips occlusion entirely (with a warning) when built without CGAL.
  The port always tests occlusion, i.e. it behaves like the CGAL-enabled build.

**Why.** CGAL is GPL (docs/LICENSE_AUDIT.md) and was not read. Ignoring the face's own
triangle is the evident intent of COLMAP's `hit_face == face_idx` check and makes the result
independent of traversal order and thread count.

**Evidence.** `TriangleBvhTests` (C#-only): hits, misses, excluded ids, the distance bound,
edges, coplanar segments, an empty tree, coincident centroids, agreement with a brute-force
OR over single-triangle trees on 3000 random triangles, occluded faces under a raised quad
left untextured, and bit-identical results for one thread and all threads. The 15 ported
texture_mapping_test.cc cases pass.

## 91. Texture mapping breaks view-label and atlas-packing ties by index

**What differs.** Two tie orders in mvs/texture_mapping.cc depend on unspecified orders:
- `SelectViews`' smoothing counts neighbor labels in a `NodeHashMap` and takes a label whose
  count strictly beats the best so far while iterating the map, so among equally common
  labels the winner follows hash order. The port visits labels in ascending image index,
  so the lowest index wins a tie.
- `PackAtlas` sorts the patch rectangles by height with `std::sort`, which leaves equal
  heights in an implementation-defined order; the port keeps them in region order.

**Why.** Deterministic and independent of the standard library and hash seed (CLAUDE.md,
translation rules; as entries 40, 57 and 64).

**Evidence.** `MeshTextureMapping_NeighborSmoothing` (texture_mapping_test.cc 1:1) passes;
`TriangleBvhTests.MeshTextureMapping_SameResultForAnyThreadCount` pins run-to-run and
thread-count stability.

## 92. Texture mapping keeps one bit per (face, image) instead of a dense score table

**What differs.** COLMAP's `SelectViews` stores a double score for every (face, image) pair
(800 MB for a million faces and a hundred images) and then reads it twice: for each face's
best view (first strictly larger score in image order) and, during smoothing, whether a
face's score in a neighbor's view is positive. The port computes the best view inside the
scoring loop, with the same strict comparison in the same order, and keeps only a "score >
0" bit per pair. Also, the color-correction system is factorized once and solved
for all three channels (COLMAP rebuilds and refactorizes the identical matrix per channel),
and the camera centers are computed once per image instead of per face. Scoring runs image
by image (faces in parallel) instead of face by face, and COLMAP's per-corner occlusion
query is answered from one query per (vertex, image): `TriangleBvh.CountHitsUpToTwo`
returns clear, one blocker (with its face) or several, and a face is occluded at that corner
exactly when some triangle other than itself blocks it, as in entry 90.

**Why.** Memory and time at the scale MatterCAD needs; the selected views, UVs and atlas are
the same as computing it COLMAP's way.

**Evidence.** The 15 ported texture_mapping_test.cc cases, including
`MeshTextureMapping_SingleFaceTwoViews` (best view) and `MeshTextureMapping_NeighborSmoothing`
(positive-score gate), pass. `TriangleBvhTests.AnyHit_MatchesBruteForceOnRandomTriangles`
checks that the counting query gives AnyHit's answer for any excluded id. On a 1M-face,
100-image benchmark the views, UVs and atlas hash identically before and after the
per-vertex sharing.

## 103. Delaunay meshing uses our own tetrahedralization, not CGAL's

**What differs.** COLMAP's `mvs/delaunay_meshing.cc` triangulates the points with
`CGAL::Delaunay_triangulation_3<Epick, Fast_location>`. The port uses
`Geometry/Delaunay/DelaunayTriangulation3` (Bowyer-Watson with a visibility walk, BRIO +
Hilbert range insertion, Shewchuk-filtered exact predicates). With default options COLMAP
builds it through `CreateSubSampledDelaunayTriangulation`, one point at a time in a shuffled
order (the range constructor is used only when `max_proj_dist` is 0). For points in general
position the Delaunay triangulation is unique, so both build the same cells. They differ in:
- **Cospherical points.** CGAL documents that its Delaunay triangulation resolves five or more
  cospherical points by symbolic perturbation, so its result does not depend on insertion
  order. Ours uses a strict in-sphere conflict test, so which of the valid triangulations is
  built depends on the insertion order (it is still deterministic for a given order).
- **Locate on a facet, edge or vertex.** When a query point lies exactly on a shared facet,
  edge or vertex, several cells contain it; CGAL's locate and our walk can return different
  ones (ours depends on the hint and the walk's pseudo-random facet order). This changes
  which cell's vertices the subsampling compares against, and where the ray caster starts.
- **Enumeration order and handles.** Cells, facets and edges come in a different order and
  with different handles (entry 109 covers where that reaches the output).
- **Constructions.** Circumcenters and segment/triangle intersection points are inexact
  double constructions on both sides and may differ in the last bits.
  `TryIntersectSegmentTriangle`, standing in for `CGAL::intersection(Segment_3, Triangle_3)`
  assigned to a point, reports no intersection for a segment lying in the triangle's plane;
  CGAL returns a segment there (which COLMAP's ray caster rejects too) except when the two
  touch in a single point, where CGAL returns that point.

**Why.** CGAL is GPL and excluded (docs/LICENSE_AUDIT.md), so its insertion order, locate
tie-breaking and cell storage cannot be matched by reading it. Implementing symbolic
perturbation (Devillers and Teillaud, "Perturbations for Delaunay and weighted Delaunay 3D
triangulations", CGTA 2011) would remove the order dependence for cospherical input; it is
not done yet.

**Evidence.** `DelaunayTriangulation3Tests` and `DelaunayDegenerateTests` (C#-only): the
empty-circumsphere property on random, grid (coplanar and cospherical) and exactly
cospherical inputs, the Euler characteristic, duplicate handling, and incremental versus
range insertion producing the same cell set in general position. Downstream, Delaunay
meshing is compared at Tier C.

## 104. Subsampled Delaunay triangulation inserts every point until the points span 3D

**What differs.** COLMAP's `CreateSubSampledDelaunayTriangulation` inserts points
unconditionally only while `number_of_vertices() < 4`; after that it locates each point and
skips it when it reprojects close to the vertices of its cell. If the first four or more
(shuffled) points are coplanar, CGAL's triangulation is still 2D and `locate` returns a
2D face whose `vertex(3)` COLMAP's loop then reads. `DelaunayMeshingInput` keeps inserting
unconditionally while `DelaunayTriangulation3.Dimension < 3`, and applies COLMAP's test from
the first 3D cell on.

**Why.** Our triangulation has no 2D cells to locate in (it holds points pending until four
are affinely independent, entry 103), and COLMAP's 2D path reads a vertex slot a 2D face does
not have. With non-degenerate input the two agree after the first four points.

**Evidence.** `DelaunayMeshingTests.CSharpOnly_SubsampledTriangulationInsertsEveryPointWhileFlat`:
50 coplanar points all become vertices and the triangulation stays 2D.

## 105. Sparse Delaunay meshing numbers the points in ascending point3D id

**What differs.** COLMAP's `CopyFromSparseReconstruction` numbers the input points in the
iteration order of `reconstruction.Points3D()`, an `unordered_map`. `DelaunayMeshingInput.
FromSparseReconstruction` numbers them in ascending point3D id. The numbering feeds the
shuffle in `CreateSubSampledDelaunayTriangulation`, so which points are kept can differ.

**Why.** CLAUDE.md's rule for hash-container order: libc++'s bucket order is not
reproducible from .NET, and ascending id is deterministic.

**Evidence.** `DelaunayMeshingTests.CSharpOnly_InputFromSparseReconstruction` pins the
ascending-id order; the meshing result is compared at Tier C.

## 106. Poisson splatting runs sequentially in sample order

**What differs.** PoissonRecon splats the samples' normals (`setInterpolatedDataField`) in a
`ThreadPool::ParallelFor` over samples, adding into shared per-node sums with atomic float
adds and into per-thread depth/weight sums. COLMAP's `PoissonMeshing` runs PoissonRecon on
every hardware thread by default (`num_threads = -1`), so the order of those float additions,
and with it the last bits of the normal field, varies from run to run; it also decides which
thread creates a node first and so the node numbering. The port
(`Mvs/PoissonRecon/PoissonSplat.cs`) splats sequentially in sample order, which is what
PoissonRecon does with one thread.

**Why.** CLAUDE.md requires deterministic results, and there is no fixed multi-threaded
order to match. The single-threaded order is the one reproducible reference.

**Evidence.** `PoissonTreeOracleTests.DensityStage_MatchesHarness` (density3, density5,
density6) matches a single-threaded run of the vendored C++ bit for bit: the normal field,
the sample depth/weight sums, the colour field and the node numbering.

## 109. Delaunay meshing assembles the graph and the surface in cell and facet order

**What differs.** COLMAP numbers the s-t graph nodes and adds the edges while iterating a
`NodeHashMap<Cell_handle, DelaunayCellData>`, walks the finite facets in CGAL's order, and
collects the surface vertices in a `FlatHashSet<Vertex_handle>` whose iteration order becomes
the output vertex order. `DelaunayMeshing` (Mvs/DelaunayMeshing.cs) numbers nodes and adds
edges in cell-handle order, walks `FiniteFacets()` in handle order, and numbers surface
vertices in order of first appearance. Node and edge order can change which minimum cut
Boykov-Kolmogorov returns when several cuts have equal cost, and the vertex and face order of
the output mesh differ.

**Why.** Hash-container order (pointer hashes in COLMAP) cannot be reproduced and is not
stable even between COLMAP runs; CLAUDE.md asks for a deterministic order.

**Evidence.** `DelaunayMeshingSceneCSharpOnlyTests.CSharpOnly_ThreadCountDoesNotChangeTheMesh`
(identical meshes) and the ported `DelaunayMeshingTests`.

## 110. Delaunay meshing sums per-image weights in image order for any thread count

**What differs.** COLMAP integrates images on a thread pool and adds each image's cell
weights to the graph in the order the jobs finish, so with more than one thread the float sums,
and so the cut, can vary from run to run. The port integrates batches of images in parallel,
each into its own map, and adds them in image order, which is what COLMAP does with one thread.
Each worker also locates points with its own `LocateCursor`, so no result depends on another
thread's walk.

**Why.** CLAUDE.md: sequential and parallel runs must give the same result.

**Evidence.** `DelaunayMeshingSceneCSharpOnlyTests.CSharpOnly_ThreadCountDoesNotChangeTheMesh`:
1 and 5 threads give identical vertex and face lists.

## 111. When the ray leaves the hull at a point, the sink vote goes to the infinite cell behind it

**What differs.** For each observation COLMAP locates the cell just behind the point
(`point + epsilon` along the viewing ray), intersects the viewing ray with all four facets of
that cell (`triangulation.triangle(cell, i)`), and puts the sink vote in the cell across the
farthest hit facet. That cell is infinite exactly when the viewing ray leaves the convex hull
at the point: a hull vertex seen through the point set from the far side (the see-through
synthetic scenes of `delaunay_meshing_test.cc`, where every point is visible in every image),
or a ray grazing the silhouette. A hull vertex seen from outside, facing the camera, has a
finite cell behind it. Three facets of an infinite cell run through CGAL's infinite vertex,
and COLMAP intersects them anyway, with whatever point CGAL stores for that vertex; the value
is not documented. The port (Mvs/DelaunayMeshing.Integrate.cs):
- if the cast segment entered the hull before reaching the point (it crossed at least one
  facet), gives the sink vote to the infinite cell behind the point (no edge weight);
- otherwise (the ray stayed outside the hull, e.g. grazing), adds nothing behind the point;
- applies COLMAP's rule unchanged whenever the cell behind the point is finite.

**Why.** Our infinite vertex has no geometric position, and CGAL's is not observable without
reading CGAL. Black-box checking was not possible: the pinned pycolmap 4.2.0 wheel is built
without CGAL (`sparse_delaunay_meshing` / `dense_delaunay_meshing` are compiled only under
`COLMAP_CGAL_ENABLED` in `pycolmap/pipeline/meshing.cc`, and the wheel has neither). Skipping
the infinite facets altogether gives the see-through scenes almost no sink votes, the cut has
no surface, and all three ported tests fail. The gate matters for object-only captures: an
ungated vote on grazing rays pushes cells outside the object towards "inside" and loses
surface (1733 faces instead of 2054 on the 1500-point, noise-0.01 object-only sphere). With
the gate the review's occluded object-only scenes gave the same meshes as both "skip" and
"infinite vertex at the origin", and the see-through scenes still give outward faces.

**Object-only captures.** When every observed point is a hull vertex (an exact convex object
seen only from outside), no viewing ray enters the hull, no cell gets a source vote, and the
cut has no surface - in COLMAP as here (entry 112 gives the readable error). Noisy or concave
real captures put points inside the hull and do produce closed, outward surfaces.

**Uncertainty:** where the ray leaves the hull at the point, which hull facets are cut can
differ from COLMAP, whose result there depends on an undocumented CGAL value.

**Evidence.** The three ported `DelaunayMeshingTests` pass; `DelaunayMeshingSceneCSharpOnlyTests`
`CSharpOnly_ObjectOnlyNoisySphereGivesAClosedOutwardSurface` (closed, outward, and the
1500-point face count that the ungated rule lost) and `CSharpOnly_SphereMeshIsOrientedOutward`.

## 112. Delaunay meshing explains an empty cut instead of failing a Check

**What differs.** When the graph cut labels every cell the same, COLMAP reaches
`Percentile` with no surface facets and fails `THROW_CHECK(!elems.empty())`. `DelaunayMeshing`
throws `InvalidOperationException` with `DelaunayMeshing.NoSurfaceMessage` ("Delaunay meshing
found no surface: too few views see into the scene's free space ... Try Poisson meshing or add
views.") at the same point. Only the exception type and message differ.

**Why.** MatterCAD shows the message to the user, and the common cause (a smooth convex object
seen only from outside, entry 111) has a clear remedy.

**Evidence.** `DelaunayMeshingSceneCSharpOnlyTests.CSharpOnly_NoSurfaceThrowsAReadableError`.

## 113. Reading a truncated binary PLY mesh throws inside the texcoord lists

**What differs.** COLMAP's `ReadPlyMesh` (util/ply.cc) checks `file.good()` after every vertex,
face count and face index it reads from a binary file, but not after a face's texcoord count
or its six UVs. What a truncation there does in COLMAP depends on where the file ends:
- Cut before the count: the count is left indeterminate (the failed read does not write it),
  so it is usually not 6 and the file fails with "Expected 6 texture coordinates per
  triangular face".
- Cut inside the UVs of a face that is not the last: the missing UVs are indeterminate, and
  the next face's count read fails with "Unexpected end of PLY file at face i + 1".
- Cut inside the UVs of the last face: nothing reads after them, so the mesh loads with
  indeterminate UVs and no error.
`Util/Ply.Mesh.cs` checks the count and every UV read and throws "Unexpected end of PLY file
at face i" for the face whose texcoords are cut, in all three cases. Every other truncation
throws in both, with the same message.

**Why.** The same upstream robustness gap as entry 62: in the last case a half-written mesh
would load with garbage texture coordinates, and in the other two the error names the wrong
cause or the wrong face. Well-formed files read identically.

**Evidence.** Reading of ply.cc (the unchecked `file.read` calls for `num_texcoords` and
`uv`). `PlyTests.CSharpOnly_TruncatedBinaryTexcoordsThrow` (C#-only) cuts a one-face textured
binary mesh before the texcoord count and after two of the six UVs (the last-face case), and
expects "Unexpected end of PLY file at face 0" both times; the ported ply_test.cc cases, which
read complete files, pass unchanged.

## 116. PoissonRecon's log( float ) is the double logarithm rounded to float

**What differs.** `_getSampleDepthAndWeight` calls `log` on float ratios, which resolves to
the float overload (`logf`) of the platform libm. The port computes
`(float)Math.Log((double)x)` (`PoissonSplat.LogF`). `Math.Log` is the host's libm as well, but
its error is far below a float ulp, so the result is the correctly rounded float except when
the exact logarithm lies within about 2^-52 relative of a float rounding boundary (about 2^-29
of a float ulp), where the double's own rounding can decide which way the float rounds.

**Why.** .NET's `MathF.Log` maps to the host's `logf`, whose accuracy varies by platform and
on browser-wasm; the rounded double logarithm gives the same float on every platform in all
but those rare cases.

**Evidence.** `PoissonTreeOracleTests.LogF_MatchesLibm`: over 4000 float arguments spanning
2^-40..2^40 (`TestData/oracle/poisson_libm.json`, from `oracle/poisson_libm_harness.cc`), the
result equals Apple's `logf` (the oracle's) bit for bit. End to end, one of the rare cases shows:
`PoissonMeshingOracleTests.PoissonMeshing_FileMatchesUpstreamExactly` (the file wrapper against
upstream's own PoissonRecon and SurfaceTrimmer, `oracle/poisson_meshing_harness.cc`) matches
byte for byte except, in its depth6trim case, output vertex 898's density `value`, one float ulp
from upstream's; the review that found it traced the ulp to this logarithm. The test pins that
one exception.

## 123. Poisson system assembly adds its shared sums sequentially in node order

**What differs.** PoissonRecon runs its system assembly with `ThreadPool::ParallelFor` and
atomic float adds into shared entries in several places: `_addFEMConstraints` (each normal
scattering into the constraints around its parent), `_addInterpolationConstraints` and
`_updateRestrictedInterpolationConstraints` (each interpolation point adding into the
constraints of the functions supported on it), and the conjugate-gradient solve at the base
of `_solveRegularMG` (`SolveCG` sums its dot products in per-thread partial sums), as does
Solve's iso-value (per-thread double sums of the weighted sample values, added in thread
order). With
COLMAP's default of every hardware thread,
the order of those float additions, and with it the last bits of the right-hand side and so of
the solution, varies from run to run. The port (`Mvs/PoissonRecon/PoissonFemConstraints.cs`,
`PoissonMultigrid.cs`, `PoissonSystem.cs`, `PoissonSparseMatrix.cs`,
`PoissonImplicitEvaluator.cs`) makes those adds in
sorted node (or row) order, which is what
PoissonRecon does with one thread (`ThreadPool::NONE`). The multi-colored Gauss-Seidel
relaxation (`PoissonSystem.GaussSeidel.cs`) is not affected: two rows of one color never share
an unknown, so their order cannot change the result, and the port relaxes them in index order.
The level-set extractor's iso-vertices (`PoissonLevelSetExtractor.IsoVertices.cs` for slice
edges, `PoissonLevelSetExtractor.XSliceIsoVertices.cs` for slab edges) are the same case for
numbering rather than sums: upstream sets a slice's or slab's vertices in a `ParallelFor` over
its leaves, numbers them with the vertex stream's atomic counter and keeps per-thread key lists,
so the vertex order varies with threads; the port visits the leaves in sorted order with one
list, as a single-threaded run does. The iso-edges (`PoissonLevelSetExtractor.IsoEdges.cs`) keep
per-thread lists too (the vertex pairs and the face iso-edges pushed to coarser slices and
slabs), merged in thread order at finalize, so the order of the iso-edges inside one face-edge
map entry varies with threads; the port records them in leaf order with one list. The polygons
(`PoissonLevelSetExtractor.Polygons.cs`) are the same: upstream's `SetLevelSet` triangulates a
slab's leaves in a `ParallelFor`, so the order of the triangles and the numbering of the
barycenter vertices it writes vary with threads; the port writes them in sorted leaf order.

**Why.** CLAUDE.md requires deterministic results, and there is no fixed multi-threaded order
to match. The single-threaded order is the one reproducible reference (the same choice as
entry 106 for splatting).

**Evidence.** `PoissonTreeOracleTests.SystemConstraints_MatchHarness` (system3, system5,
system6, system8) matches a single-threaded run of the vendored C++ bit for bit: the FEM and
interpolation constraints, the restricted interpolation constraints, the Gauss-Seidel
solutions and the base-depth multigrid solve; `PoissonTreeOracleTests.PostSolveStages_MatchHarness`
(levelset3, levelset5, levelset6, levelset8) does the same for the iso-value sums, and
`PoissonTreeOracleTests.LevelSetSliceIsoVertices_MatchHarness` and
`PoissonTreeOracleTests.LevelSetIsoVertices_MatchHarness` for the vertex order, and
`PoissonTreeOracleTests.LevelSetIsoEdges_MatchHarness` for the iso-edge lists, and
`PoissonTreeOracleTests.LevelSetPolygons_MatchHarness` for the triangles and barycenters.
## 125. A polygon's barycenter vertex gets the mean of its loop's depths

**What differs.** When `AddIsoPolygons` (`FEMTree.LevelSet.3D.inl`) splits a coplanar loop
around a new barycenter vertex, upstream builds that vertex as `Vertex c; c *= 0;`, adds the
loop's vertices and divides by their count. The position, gradient and color members are
zero-initialized `Point`s, but the depth is a plain `float` that the default constructor
leaves uninitialized, and an uninitialized (possibly NaN) value times zero is not reliably
zero. `PoissonLevelSetExtractor.Barycenter` starts the depth at zero, so the barycenter's depth
(the density COLMAP's `--density` writes) is the mean of the loop's depths. Upstream's value
is undefined: an optimized build gives every barycenter a NaN depth.

**Why.** Reading the uninitialized float is undefined behavior, so there is no upstream value
to match. Averaging the depth like every other member is what `c *= 0` evidently intends, and
it keeps the density finite. That matters: `PoissonMeshing` passes `--density` whenever
`trim > 0` (the default is 10) and the surface trimmer then cuts by that density, so a NaN
density on a barycenter vertex would make trimming depend on how NaN compares, rather than on
the surface.

**Evidence.** `oracle/poisson_levelset6_harness.cc` built with the fixture script's flags
(`-std=c++17 -O1 -ffp-contract=off`, Apple clang 21, Darwin arm64) reproduces the checked-in
`poisson_levelset6.json` bit for bit, with finite barycenter depths (levelset6's first is
`0x1.2ab9fp+2`); `PoissonTreeOracleTests.LevelSetPolygons_MatchHarness` pins the port to it.
The same harness at `-O2` prints NaN for every barycenter depth (all 40 in levelset6, all 92
in levelset8) and is otherwise unchanged. At `-O0` every harness is bit-identical to `-O1`
(the harnesses call `__sincosf_stret` explicitly so their inputs don't depend on the
optimization level), so PoissonRecon itself does not depend on it apart from this depth. The
oracle stays at the harness's
standard `-O1`, and a Tier C end-to-end mesh fixture from optimized pycolmap must not expect
bit-exact barycenter vertices or their densities.

## 130. PoissonMeshing carries only red, green and blue from the input PLY

**What differs.** `RunPoissonRecon` reads a PLY input's vertex header and turns every property
other than x, y, z and nx, ny, nz into auxiliary data (`VertexFactory::DynamicFactory`), which
it splats, extracts and writes back under the same names and on-disk types. The file wrapper
`PoissonMeshing.Run(options, inputPath, outputPath)` carries exactly `uchar red`, `uchar green`
and `uchar blue` (the mesh colors) and drops any other extra vertex property; values are read
with COLMAP's own `Ply.ReadPly`.

**Why.** COLMAP's `poisson_mesher` input is the `fused.ply` that stereo fusion writes, whose
extra properties are exactly those three uchar colors, so for COLMAP's own pipeline the output
is the same. Carrying arbitrary properties would need a per-channel type through the extractor,
the trimmer and the writer that no COLMAP input exercises.

**Evidence.** `PoissonMeshingOracleTests` feeds pycolmap and the port a fused.ply-layout input
and requires the identical output PLY property layout (x, y, z, [value,] red, green, blue).

## 131. PoissonRecon's command-line flags do not leak between PoissonMeshing calls

**What differs.** PoissonRecon.cpp and SurfaceTrimmer.cpp keep their command-line parameters in
namespace-scope globals, and `CmdLineParse` only ever sets a flag's `set` state, never clears
it. So within one process a `--density` (trim > 0) or `--fullDepth` (depth < 5) from an earlier
`PoissonMeshing` call stays in force for every later call: a later untrimmed run still writes
the density `value` property, and a later depth >= 5 run keeps the earlier, smaller full depth.
The port derives every setting from the call's own options.

**Why.** The leak is an accident of the command-line wrapper, not a documented behavior; one
call's result should not depend on earlier calls. A single `colmap poisson_mesher` process
makes one call, which the port matches.

**Evidence.** Running pycolmap 4.2.0's `poisson_meshing` in one Python process for depth 5 with
trim 3.6, then depth 4 with point weight 0 and trim 0, wrote a `property float value` in the
second output; the same second call in a fresh process did not. Likewise depth 6 with trim 3.5
after the depth 4 call gave 2736 vertices, and 4062 in a fresh process (the port gives 4065;
the difference is the trim-boundary noise PoissonMeshingOracleTests allows).
`oracle/fixture_poisson_meshing.py` therefore runs each case in its own process.

## 132. PoissonMeshing trims an empty mesh to an empty mesh instead of crashing

**What differs.** With trim > 0 and an input that yields no surface (a PLY with no points),
upstream's `RunSurfaceTrimmer` crashes the process. The port's `PoissonSurfaceTrimmer.Trim`
returns an empty mesh, so `PoissonMeshing.Run(options, inputPath, outputPath)` returns true and
writes the header-only PLY PoissonRecon itself wrote (x, y, z, value, and red, green, blue when
the input has them; `element vertex 0`, `element face 0`), and the in-memory `Run` returns a
mesh with no vertices and no triangles.

**Why.** An upstream bug. SurfaceTrimmer.cpp's `Execute` seeds the value range it prints under
`--verbose` with `min = max = vertices[0].template get<1>();` before checking the vertex count;
on an empty `std::vector` that reads through its null data pointer. The range feeds only that
log line, so the port does not compute it, and an empty mesh passes through the trim steps
unchanged. Crashing the host application on an empty point cloud is never the intended result.

**Evidence.** `oracle/poisson_meshing_harness.cc` built as `oracle/fixture_poisson_meshing.py`
builds it, run as `harness 5 1.0 3.5 empty.ply out.ply` on a colored PLY with no points, exits
with 139 (SIGSEGV) after PoissonRecon has written its 248-byte header-only `out.ply`, so the
fixture's `EXACT_ONLY_CASES` run the empty input untrimmed only.
`PoissonMeshingTests.PoissonMeshing_EmptyInputTrimmed_WritesEmptyMesh` pins the port's output:
exactly those 248 bytes, and an empty in-memory mesh.
