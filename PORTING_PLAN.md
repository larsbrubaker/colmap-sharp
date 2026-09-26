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
`rig` (scene/rig: `ReadRigConfig`/`ApplyRigConfig` + `rig_test.cc`; needs database,
reconstruction and synthetic), `pose_graph` (+ `pose_graph_test.cc`; its component functions take a
`Reconstruction`), `reconstruction` remainder (`Load`, `TranscribeImageIdsToDatabase`,
Read/Write*, ConvertToPLY/ImportPLY, ExtractColors*; deferred `reconstruction_test.cc` cases
ConstructCopy, AssignCopy, Print, SetRigsAndFramesResetsNumRegImages,
DeleteAllPoints2DAndPoints3D, TearDown, SetRigsAndFrames, TranscribeImageIdsToDatabase,
ConvertToPLY, ImportPLYFromVector, ReadWriteTextRoundtrip, ReadWriteBinaryRoundtrip,
ReadAutoDetectFormat, ExtractColorsForAllImages; `reconstruction_matchers_test.cc` Eq/Near;
add an IdMap test that reaches `Compact()`), `reconstruction_io` (COLMAP binary and text
formats, so pycolmap-written models become fixtures), `reconstruction_manager`,
`reconstruction_pruning`, `synthetic` (the synthetic dataset generator that most downstream
tests use), `database` (in-memory store with COLMAP's API), `database_cache`,
`scene_clustering`, `reconstruction_clustering`. Before the pipeline phases, decide how the
library surfaces COLMAP's `LOG(WARNING)` messages (dropped for now).

Skipped `util/types_test.cc` cases: `Span.SizeAndEmpty`, `FilterView.Empty/All/None/Nominal/
RangeExpression` — COLMAP's `span`/`filter_view` are replaced by `System.Span<T>` and LINQ, so
there is no ColmapSharp code under test.

### Phase 6 — Minimal solvers and estimators
The four TinySolver callers (essential / fundamental refinement, relpose shared and one-sided
focal) use `Optim/TinySolver.cs` with `TinyProductManifold<…, TinyEuclideanManifold1>`.
Estimators implement `IEstimator<TX,TY,TModel>` (+ `ILocalEstimator` for LO-RANSAC) as
`readonly struct`s — see `Optim/Estimator.cs` and `Estimators/Solvers/SimilarityTransform.cs`.
`LoRansac.Estimate` hides (does not override) `Ransac.Estimate`: call it on the LoRansac type.
`estimators/solvers/*` (P3P, EPnP, 5-pt/7-pt/8-pt, homography, affine,
generalized pose, focal solvers; the PoseLib parts are BSD-3, add its notice),
`two_view_geometry`, `pose`, `generalized_pose`, `triangulation`, `alignment`,
`fundamental_matrix_degensac`, `rotation_averaging`, `global_positioning`,
`gravity_refinement`, `view_graph_calibration`.

### Phase 7 — Nonlinear least-squares solver (Ceres replacement)
Done: `Jet<TGrad>` (widths 1–33), `IScalar<T>`, rotation helpers, loss functions, manifolds,
`AutoDiffCostFunction` (views via `ArraySegment`). Open: problem / residual-block model with
parameter blocks, constant blocks and manifolds; Levenberg–Marquardt trust region; linear
solvers (dense QR/normal Cholesky, sparse normal Cholesky, dense and sparse Schur, iterative
Schur with PCG + Jacobi/Schur-Jacobi); multi-threaded evaluation writing per-thread slots;
solver summary. Ported from Ceres 2.2.0 (BSD-3; notice already present). This is the riskiest
phase; validate against pycolmap BA on synthetic scenes (Tier C).
The simplicial sparse Cholesky (`LinearAlgebra/SimplicialCholesky.cs`) factors a 1000-camera
Schur-like pattern in ~3.5 s (scalar, ~1.3 GFLOP/s): add a supernodal or 6x6-blocked numeric
phase behind the same API before BA at that scale. Optional later speedup: Vector128 lanes
in `Jet` (IEEE-exact, no FMA) need a CLAUDE.md rule clarification first.

### Phase 8 — Bundle adjustment
`estimators/bundle_adjustment*` (the non-GPU paths), `cost_functions/*`,
`ceres_loss_function`, `covariance`.

### Phase 9 — Features
`feature/sift` (VLFeat CPU SIFT port, BSD-2, add its notice; DSP-SIFT and domain-size
pooling options), `feature/types`, `feature/utils`, `feature/matcher` (brute force + managed
kd-tree for approximate NN, cross-check, ratio test, guided matching), `feature/extractor`,
`feature/index`.

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

