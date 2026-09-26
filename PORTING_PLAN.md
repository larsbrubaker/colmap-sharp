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

### Phase 1 — Math and linear algebra foundation
- `LinearAlgebra/`: SVD (JacobiSVD semantics: 3x3, 4x4, Nx9, Nx6, 6x3..6x5, 12x12, MatrixXd
  with full U), symmetric eigen (4x4 SelfAdjointEigenSolver), general EigenSolver (companion
  matrix roots; 4x4 with complex eigenvectors), FullPivLU rank (3xN). Inventory in the
  `MatrixXd.cs` header. Written from textbook algorithms — **Eigen is MPL-2.0 and must not
  be transcribed.**
- `Mathematics/` (COLMAP's `math/`):
  `polynomial` (companion-matrix and Durand–Kerner roots).
- `random_eigen.h` + `random_eigen_test.cc` (then replace the draw helper in
  `MatrixUtilsTests` with it).
- Tests: the remaining `math/*_test.cc`, plus C#-only decomposition tests against oracle
  fixtures (numpy in the oracle venv is fine for pure linear algebra checks).

### Phase 2 — Geometry
`geometry/`: `pose`, `essential_matrix`, `homography_matrix`, `triangulation`,
`normalization`, `pose_prior`; `rigid3_matchers_test.cc`, `sim3_matchers_test.cc`.

### Phase 3 — Sensor
`sensor/models_jacobian.h` (`ImgFromCamWithJac`, `CameraModelImgFromCamWithJac`,
`CamRayFromImgJacobian`) + `models_jacobian_test.cc`; `rig`.

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
`point2d`, `point3d`, `track`, `camera`, `frame`, `image`, `rig`, `correspondence_graph`,
`two_view_geometry`, `pose_graph`, `projection`, `visibility_pyramid`, `reconstruction`,
`reconstruction_io` (COLMAP binary and text formats, so pycolmap-written models become
fixtures), `reconstruction_manager`, `reconstruction_pruning`, `synthetic` (the synthetic
dataset generator that most downstream tests use), `database` (in-memory store with
COLMAP's API), `database_cache`, `scene_clustering`, `reconstruction_clustering`.

### Phase 5 — Optimization primitives
`optim/`: `ransac`, `loransac` (generic over `ISampler<TSelf>` / `ISupportMeasurer<TSupport>`;
decide PROSAC's out-of-range index when k == N, cast `num_inliers - i` to ulong as C++ does;
constrain samplers to `class` or drop the struct claim in Sampler.cs), `least_absolute_deviations`, `tiny_solver`,
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
Build on `Solver/Scalar.cs`'s `IScalar<T>` (camera models are already generic over it):
`Jet<TGrad>` with `TGrad` an `[InlineArray(N)]` gradient struct exposing a static `Length`
(no const generics in C#), `in` parameters for large N (benchmark N=6/9/15), Ceres's
mixed scalar operators (`T*double`, `double/T`, … — not the same as `FromDouble(s) op x`),
Exp/Log/Pow/Acos/Asin/Hypot/IsFinite/Min/Max, and generic k-th-variable seeding.

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

