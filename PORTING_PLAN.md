# COLMAP → C# porting plan

Open work only; history lives in git. Remove a phase when it is done and green, and prune
stale notes as you go. The contract and translation rules are in `CLAUDE.md`, and license
decisions are in `docs/LICENSE_AUDIT.md`.

**Reference:** COLMAP 4.2.0 (`REFERENCE`); oracle `pycolmap==4.2.0`.
**Size:** ~100k lines of C++ in scope (tests, UI and CLI excluded), 142 `*_test.cc` files,
~1,640 gtest cases.
**Goal for MatterCAD:** photos → camera poses + sparse cloud → dense depth → textured mesh,
all cancellable with progress.

## Out of scope (do not re-litigate)

GUI (`ui/`), CLI executables (`exe/`), CUDA/HIP kernels as such (the PatchMatch *algorithm* is
ported to CPU), SiftGPU, ONNX learned features (ALIKED, LightGlue, LoMa, AnyCalib),
Caspar GPU BA, LSD line detection and `estimators/coordinate_frame` (AGPL), CGAL code paths,
SQLite database files, `download` support, `retrieval/` vocabulary-tree matching (revisit
only if exhaustive/sequential/spatial matching proves too slow for MatterCAD photo sets).

## Phases (dependency order)

Each phase ends with its ported tests green. Test names follow COLMAP's.

### Phase 3 — Sensor
Skipped `bitmap_test.cc` cases (the host decodes and encodes images, so OIIO file I/O is not
ported; C# `Bitmap` is a reference type, so C++ move semantics have no counterpart):
`MoveConstructEmpty`, `MoveConstruct`, `MoveAssignEmpty`, `MoveAssign`, `ReadWriteAsRGB`,
`ReadWriteUnicodePath`, `ReadWriteAsGrey`, `ReadWriteAsGreyNonLinear`,
`ReadWriteLinearColorspace`, `WriteJpegWithQuality`, `WriteInvalidFormat`, `ReadNonImageFile`,
`ReadNonExistentFile`, `ReadUnsupportedChannels`, all `ParameterizedBitmapFormatTests`, and
the PNG round-trip tails of `CloneAsRGB` / `CloneAsGrey`. `Bitmap.Rescale` is Tier B (managed
resampler matching OIIO within 1 grey level, `docs/CPP_DIVERGENCES.md`); porting OIIO's
resize (Apache-2.0) would make it exact if a fixture ever needs that.

### Phase 4 — Scene
- `rig` (scene/rig: `ReadRigConfig`/`ApplyRigConfig` + `rig_test.cc`).
- `Reconstruction.ExtractColors*` (the host decodes images; take a Bitmap provider).
- `scene_clustering`, `reconstruction_clustering`.
- Deferred tests: `reconstruction_test.cc` ExtractColorsForAllImages;
  `reconstruction_matchers_test.cc` Near (needs alignment); an IdMap test that reaches
  `Compact()`.
- Before the pipeline phases, decide how the library surfaces COLMAP's `LOG(WARNING)`
  messages (dropped for now).

Skipped `util/types_test.cc` cases: `Span.SizeAndEmpty`, `FilterView.Empty/All/None/Nominal/
RangeExpression` — COLMAP's `span`/`filter_view` are replaced by `System.Span<T>` and LINQ, so
there is no ColmapSharp code under test.
Skipped `scene/database_test.cc` cases: `OpenFile`, `OpenCloseFile`, `OpenFileWithNonASCIIPath`
— SQLite database files are out of scope (the database is `InMemoryDatabase`).

### Phase 6 — Minimal solvers and estimators
The four TinySolver callers (essential / fundamental refinement, relpose shared and one-sided
focal) use `Optim/TinySolver.cs` with `TinyProductManifold<…, TinyEuclideanManifold1>`.
Estimators implement `IEstimator<TX,TY,TModel>` (+ `ILocalEstimator` for LO-RANSAC) as
`readonly struct`s — see `Optim/Estimator.cs` and `Estimators/Solvers/SimilarityTransform.cs`.
`LoRansac.Estimate` hides (does not override) `Ransac.Estimate`: call it on the LoRansac type.
Open estimators: `two_view_geometry`'s `EstimateRigTwoViewGeometries` (+ its `Nominal` test;
`generalized_pose` is now on main), `alignment`,
`rotation_averaging`, `global_positioning`,
`gravity_refinement`, `view_graph_calibration`.

### Phase 7 — Nonlinear least-squares solver (Ceres replacement)
Done: Jet/autodiff, losses, manifolds, Problem, LM trust region, DENSE_QR,
DENSE_NORMAL_CHOLESKY, SPARSE_NORMAL_CHOLESKY, DENSE_SCHUR, SPARSE_SCHUR, ITERATIVE_SCHUR
(Jacobi/SchurJacobi), automatic and user (multi-group) Schur ordering, parameter bounds with
Ceres' projected line search, `Problem.Evaluate` (`Solver/`). Not ported on purpose: `Summary.FullReport`
(log-only), Dogleg, inner iterations, SuiteSparse/LAPACK/Accelerate/NESDIS/SPSE/SUBSET/CGNR
variants and their Ceres tests, Ceres' line search minimizer and its tests, static-size Schur specializations
(SchurEliminatorForOneFBlock), unstable independent-set ordering, visibility-clustering
preconditioners, Ceres' bit-packed cell-key tests.
Performance (before BA at scale), keeping results bit-identical and thread-count independent:
the Schur eliminator is ~88% of SPARSE_SCHUR (0.32 s/solve at 200 cams/20k pts, Release):
fixed-size kernels for the 2/3/6 BA shapes, precomputed cell offsets (no Dictionary lookups),
two-phase parallel elimination (per-chunk slots, then per-camera block rows in chunk order),
cached E'E inverses for back substitution, parallel implicit products for ITERATIVE_SCHUR,
blocked LLT for DENSE_SCHUR; supernodal/blocked simplicial Cholesky for large reduced systems.
Cancellation is checked between iterations only (like COLMAP). Optional: Vector128 lanes in
`Jet` (IEEE-exact, no FMA) need a CLAUDE.md rule clarification first.

### Phase 8 — Bundle adjustment
Done: `ceres::Covariance` subset (dense QR, `Solver/Covariance.cs`; divergence 45), default Ceres bundle adjuster (`Estimators/BundleAdjustment*.cs`), cost functions, 7-value
pose blocks. Open: `CreatePosePriorBundleAdjuster` / `PosePriorBundleAdjuster` (needs
`estimators/alignment`'s `AlignReconstructionToPosePriors`) with PosePriorBundleAdjusterBackendTest.Nominal
and the five PosePriorBundleAdjuster.* cases; BundleAdjusterBackendTest.Nominal,
.NominalMultiCameraRigConstantSensorFromRig and DefaultBundleAdjuster.NominalMultiCameraRig (their
`ReconstructionNear` matcher needs alignment); `covariance` (+ test).
Skipped: CeresBundleAdjustmentOptions.FallsBackToCpuWithoutCudaDevice (CUDA), the CASPAR
instantiations of the backend suites and `bundle_adjustment_caspar_test.cc` (GPU backend out of
scope). Performance: ~2.6× slower than native Ceres on a 100-image synthetic scene (7.6 s vs
2.9 s) — the Schur eliminator speedups in Phase 7 close most of this.

### Phase 9 — Features
`feature/sift` covariant extractor (`CovariantSiftCPUFeatureExtractor`: affine shape, DSP-SIFT
/ domain-size pooling, force_covariant — needs VLFeat `covdet.c`, `vl_sift_calc_raw_descriptor`,
`vl_imgradient_polar_f`; then the five pending `SiftCpuExtraction` rows CovariantSift,
CovariantAffineSift, CovariantAffineSiftUpright, CovariantDSPSift, CovariantAffineDSPSift).
Skipped (SiftGPU excluded): `sift_test.cc` ExtractSiftFeaturesGPU.Nominal,
CreateSiftGPUMatcherOpenGL/CUDA.Nominal, MatchSiftFeaturesGPU.{Nominal,TypeMismatch},
MatchSiftFeaturesCPUvsGPU.Nominal, MatchGuidedSiftFeaturesGPU.* (7),
MatchGuidedSiftFeaturesCPUvsGPUGuided.EssentialMatrix. Skipped (ONNX learned features out of
scope): `matcher_test.cc` Check for the 9 non-SIFT-bruteforce types and the aliked lines of
Copy/CopyAssignment; `extractor_test.cc` Move and MoveAssignment (they only move the LoMa
shared_ptr), Check for ALIKED_N16ROT/ALIKED_N32/LOMA_B/LOMA_B128, and the aliked/loma lines of
Copy/CopyAssignment. The feature index is exact (divergence 42), not faiss.
Memory: VLFeat's scale space for a 6400×4800 upsampled first octave is multi-GB (as in COLMAP);
MatterCAD (esp. wasm32) must cap `max_image_size` accordingly.

### Phase 10 — Incremental SfM
`sfm/observation_manager`, `incremental_triangulator`, `incremental_mapper(_impl)`,
`controllers/incremental_pipeline`, `controllers/bundle_adjustment`.
End-to-end Tier C fixtures: small real photo sets reconstructed by pycolmap vs. us.

### Phase 11 — Pipeline controllers
`controllers/pairing` (exhaustive, sequential, spatial), `feature_extraction`,
`feature_matching(_utils)`, `matcher_cache`, `image_reader`, `undistorters`,
`automatic_reconstruction` (minus CGAL/GPU branches), cancellation + progress surface.

### Phase 12 — Dense reconstruction (MVS)
`image/undistortion`, `image/warp`, `mvs/mat`, `image`, `depth_map`, `normal_map`,
`model`, `workspace`, `consistency_graph`, `patch_match` + a managed CPU port of
`patch_match_cuda.cu` (parallel over pixels/rows, deterministic), `fusion`,
`poisson_meshing` (PoissonRecon MIT port, add notice), `delaunay_meshing` (tetrahedralization
via MIConvexHull (MIT) or MatterCAD's own; graph-cut surface extraction is COLMAP's own code),
`mesh_simplification`, `texture_mapping` (in scope: MatterCAD shows the textured model so
the user can relate it to their photos; CGAL's AABB tree for occlusion is replaced by a
managed BVH written here).

### Phase 13 — Global and hierarchical mapping
- `ComputeNormalizedMinGraphCut` (graph_cut.cc, METIS k-way) and its 4 graph_cut_test.cc
  cases (`ComputeNormalizedMinGraphCut*`): only scene_clustering uses it. Port METIS
  (Apache-2.0) or write a multilevel partitioner with FM refinement (a divergence entry).
`sfm/global_mapper`, `controllers/global_pipeline`, `hierarchical_pipeline`,
`rotation_averaging` controller.

### Phase 14 — MatterCAD integration
Reference `ColmapSharp` from MatterCAD, host-side image decoding into the library's pixel
buffer, a "photos → mesh" design operation with progress and cancel.

