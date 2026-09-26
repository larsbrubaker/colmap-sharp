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

### Phase 0 — Scaffold and oracle
- A first fixture script under `oracle/` that writes inputs and outputs to
  `ColmapSharp.Tests/TestData/oracle/`, plus a C# test that reads them, to prove the
  oracle loop end to end.

### Phase 1 — Math and linear algebra foundation
- `LinearAlgebra/`: fixed-size `Vector2d/3d/4d`, `Matrix2d/3d/3x4d/4d`, `Quaterniond`,
  `AngleAxis`; dynamic `VectorXd`/`MatrixXd`; decompositions: Householder QR, Jacobi and
  Golub–Kahan SVD, symmetric eigen, LU with partial pivoting, LLᵀ/LDLᵀ. Written from
  textbook algorithms — **Eigen is MPL-2.0 and must not be transcribed.**
- `Mathematics/` (COLMAP's `math/`): `random` (mt19937 + distributions, exact),
  `polynomial` (companion-matrix and Durand–Kerner roots), `union_find`,
  `connected_components`, `spanning_tree`, `graph_cut` (max-flow; COLMAP's own BSD code),
  `matrix.h` helpers.
- **Random targets libc++.** `oracle/probe_random.py` shows the pycolmap 4.2.0 wheel
  (`cp314-macosx_14_0_arm64`) links `/usr/lib/libc++.1.dylib` and imports 333 `std::__1`
  symbols and no `__cxx11`/`GLIBCXX` ones. pycolmap binds only `set_random_seed`, and
  `RandomUniformInteger/Real/Gaussian` are header templates inlined into `_core`, so the
  distributions are LLVM libc++'s: `uniform_int_distribution` via
  `__independent_bits_engine` with rejection, `uniform_real_distribution` via
  `generate_canonical`, `normal_distribution` via the Marsaglia polar method with a cached
  second value. Port those from libc++'s `<__random/*>` headers (Apache-2.0 WITH
  LLVM-exception: permissive, but add a `docs/LICENSE_AUDIT.md` row and a
  `THIRD_PARTY_NOTICES.md` section in the same change) and say so in `Random.cs`. Oracle
  fixtures that depend on the PRNG must come from the macOS wheel; the Linux manylinux
  wheel uses libstdc++ and would give different numbers.
- Tests: the remaining `math/*_test.cc`, plus C#-only decomposition tests against oracle
  fixtures (numpy in the oracle venv is fine for pure linear algebra checks).

### Phase 2 — Geometry
`geometry/`: `rigid3`, `sim3`, `pose`, `essential_matrix`, `homography_matrix`,
`triangulation`, `normalization`, `bbox`, `gps`, `pose_prior`.

### Phase 3 — Sensor
`sensor/models` (all camera models; Tier A, bit-exact projection/unprojection, including the
iterative undistortion), `rig`, `specs` (camera sensor-width DB), `bitmap` as a managed pixel
buffer (host decodes images; managed EXIF reader for focal length/make/model), `database`.

### Phase 4 — Scene
`point2d`, `point3d`, `track`, `camera`, `frame`, `image`, `rig`, `correspondence_graph`,
`two_view_geometry`, `pose_graph`, `projection`, `visibility_pyramid`, `reconstruction`,
`reconstruction_io` (COLMAP binary and text formats, so pycolmap-written models become
fixtures), `reconstruction_manager`, `reconstruction_pruning`, `synthetic` (the synthetic
dataset generator that most downstream tests use), `database` (in-memory store with
COLMAP's API), `database_cache`, `scene_clustering`, `reconstruction_clustering`.

### Phase 5 — Optimization primitives
`optim/`: samplers (random, progressive, combination), `ransac`, `loransac`,
`support_measurement`, `sprt`, `least_absolute_deviations`, `tiny_solver`,
`sparse_cholesky` (own implementation with AMD ordering; CHOLMOD is LGPL).

### Phase 6 — Minimal solvers and estimators
`estimators/solvers/*` (P3P, EPnP, 5-pt/7-pt/8-pt, homography, affine, similarity,
generalized pose, focal solvers; the PoseLib parts are BSD-3, add its notice),
`two_view_geometry`, `pose`, `generalized_pose`, `triangulation`, `alignment`,
`fundamental_matrix_degensac`, `rotation_averaging`, `global_positioning`,
`gravity_refinement`, `view_graph_calibration`.

### Phase 7 — Nonlinear least-squares solver (Ceres replacement)
`Solver/`: `Jet<N>` forward-mode autodiff, cost function / residual block model,
parameter blocks with manifolds (quaternion, sphere, subset), loss functions (trivial,
Huber, soft-L1, Cauchy, Tolerant…), Levenberg–Marquardt trust region, Schur complement with
dense and sparse Cholesky and iterative (PCG with Jacobi/Schur-Jacobi preconditioner) linear
solvers, solver summary. Ported from Ceres (BSD-3, add its notice), subset only. This is the
riskiest phase; validate against pycolmap BA on synthetic scenes (Tier C).

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
`sfm/global_mapper`, `controllers/global_pipeline`, `hierarchical_pipeline`,
`rotation_averaging` controller.

### Phase 14 — MatterCAD integration
Reference `ColmapSharp` from MatterCAD, host-side image decoding into the library's pixel
buffer, a "photos → mesh" design operation with progress and cancel.

