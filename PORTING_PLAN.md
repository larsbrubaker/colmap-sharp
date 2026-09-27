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

## Open work

Each step ends with its ported tests green. Test names follow COLMAP's.

### Phase 11 — Pipeline controllers
`automatic_reconstruction` (minus CGAL/GPU branches), including the step that turns
`MeshTextureMapping`'s result into a textured mesh.

### Phase 12 — Dense reconstruction (MVS)
- `poisson_meshing`: PoissonRecon port in progress. Done: B-splines, octree, density,
  splatting, finalize, interpolation info, FEM constraint/system integrators and
  restriction/prolongation, FEM and interpolation constraints, system matrix rows, point-constraint transfers and the
  sliced Gauss-Seidel, and the whole linear solve (`_solveRegularMG`, SolveCG, cascadic
  solveSystem). iso-value, level-set corner evaluation. HyperCube tables, slice/slab cell indices and
  corner values/MC indices. Remaining: level-set extraction (iso-vertices, iso-edges/polygons, driver and output), the trimmer, the
  public API and `poisson_meshing_test.cc`.

### Verification
- End-to-end Tier C fixtures: small real photo sets reconstructed by pycolmap vs. us (also the
  first chance to reach the structure-based → structure-less registration fallback, which no
  synthetic scene triggers).

### Performance (keep results bit-identical and thread-count independent)
- Schur solvers (fixed-size kernels, precomputed cell offsets and cached E'E inverses are in;
  SchurDeterminismTests pins the output hash, which must not change): two-phase parallel
  elimination (per-chunk E'F and inverse slots, then per-camera block rows replayed in chunk
  order, in batches of chunks to bound memory), parallel implicit products for
  ITERATIVE_SCHUR, blocked LLT for DENSE_SCHUR.
- Fusion is single-threaded (divergence 87): per-image parallel precompute of per-pixel
  world points/normals first, then a speculative band-parallel traversal with in-order commit,
  which reproduces the one-thread result.
- Optional: Vector128 lanes in `Jet` (allowed under CLAUDE.md's lane rule).
- Evaluate a faithful port of libc++ `std::sort` (sort3/4/5, insertion sort below 24, pdqsort
  above) so tie-sensitive sorts match COLMAP instead of carrying divergence entries.

## Skipped tests
Every COLMAP test not ported, with the reason.

- **Bitmap file I/O** (the host decodes and encodes images, so OIIO is not ported; C# `Bitmap`
  is a reference type, so move semantics have no counterpart) — `bitmap_test.cc`:
  `MoveConstructEmpty`, `MoveConstruct`, `MoveAssignEmpty`, `MoveAssign`, `ReadWriteAsRGB`,
  `ReadWriteUnicodePath`, `ReadWriteAsGrey`, `ReadWriteAsGreyNonLinear`,
  `ReadWriteLinearColorspace`, `WriteJpegWithQuality`, `WriteInvalidFormat`,
  `ReadNonImageFile`, `ReadNonExistentFile`, `ReadUnsupportedChannels`, all
  `ParameterizedBitmapFormatTests`, and the PNG round-trip tails of `CloneAsRGB` /
  `CloneAsGrey`.
- **Replaced by .NET types** — `util/types_test.cc`: `Span.SizeAndEmpty`,
  `FilterView.Empty/All/None/Nominal/RangeExpression` (`System.Span<T>` and LINQ).
- **SQLite files** (the database is `InMemoryDatabase`) — `scene/database_test.cc`:
  `OpenFile`, `OpenCloseFile`, `OpenFileWithNonASCIIPath`.
- **Ceres internals not ported** (SuiteSparse/LAPACK/Accelerate/NESDIS/SPSE/SUBSET/CGNR
  variants, Dogleg, inner iterations, line search minimizer, bit-packed cell keys): their
  Ceres tests.
- **CUDA / GPU** — `CeresBundleAdjustmentOptions.FallsBackToCpuWithoutCudaDevice`, the CASPAR
  instantiations of the BA backend suites, `bundle_adjustment_caspar_test.cc`, the `caspar`
  lines of `incremental_pipeline_test.cc` IncrementalPipelineOptions
  PropagatesExplicitMaxNumIterations / DefaultMaxNumIterationsUsesBackendDefaults, and
  pycolmap `mvs_test.py`'s PatchMatch cases.
- **SiftGPU** — `sift_test.cc`: ExtractSiftFeaturesGPU.Nominal,
  CreateSiftGPUMatcherOpenGL/CUDA.Nominal, MatchSiftFeaturesGPU.{Nominal,TypeMismatch},
  MatchSiftFeaturesCPUvsGPU.Nominal, MatchGuidedSiftFeaturesGPU.* (7),
  MatchGuidedSiftFeaturesCPUvsGPUGuided.EssentialMatrix.
- **ONNX learned features** — `matcher_test.cc` Check for the 9 non-SIFT-bruteforce types and
  the aliked lines of Copy/CopyAssignment; `extractor_test.cc` Move and MoveAssignment,
  Check for ALIKED_N16ROT/ALIKED_N32/LOMA_B/LOMA_B128, and the aliked/loma lines of
  Copy/CopyAssignment.
- **Vocabulary-tree retrieval** — `pairing_test.cc` VocabTreePairGenerator.Nominal,
  VocabTreePairGenerator.DoesNotDeadlockOnFailedQuery,
  SequentialPairGenerator.LoopDetectionMinIndexDistance; `feature_matching_test.cc`
  CreateVocabTreeFeatureMatcher.Nominal.

## Notes for MatterCAD
- VLFeat's scale space for a 6400×4800 upsampled first octave is multi-GB (as in COLMAP);
  cap `max_image_size`, especially on wasm32.
- `Bitmap.Rescale` is Tier B (managed resampler within 1 grey level of OIIO); porting OIIO's
  resize (Apache-2.0) would make it exact if a fixture ever needs that.

### Phase 14 — MatterCAD integration (not started; needs Lars)
Reference `ColmapSharp` from MatterCAD, host-side image decoding into the library's pixel
buffer, a "photos → mesh" design operation with progress and cancel.
