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
`MeshTextureMapping`'s result into a textured mesh. Starts once `poisson_meshing` is done.

### Phase 12 — Dense reconstruction (MVS)
- `poisson_meshing` (PoissonRecon port, `Mvs/PoissonRecon/`). Everything through the linear
  solve, the iso-value, the level set's iso-vertices, iso-edges and polygons, the Extract
  driver, model-space output (`PoissonMeshOutput`) and `PoissonSurfaceTrimmer` is done and bit-exact against the
  vendored C++ (harnesses `oracle/poisson_levelset{,2,3,4,5,6}_harness.cc` and
  `poisson_{extract,trim}_harness.cc`, built at `-O1`). Remaining (harness fixtures ≤ ~1 MB):
  1. Coverage: the vertex-pair push loops (`PoissonLevelSetExtractor.IsoEdges.cs` slice and
     slab pushes) are reached (vertexpairs2) but not pinned — deleting either still passes, and
     a 320-input C++ search found no input where a walk uses a pushed pair. Likely needs a leaf
     two levels coarser than its neighbors across a doubly crossed edge, or a proof that grading
     rules it out.
  2. Harness hardening: pin `MakeInput`'s sin/cos in `oracle/poisson_harness.h` to
     `__sincosf_stret` so fixtures don't depend on `-O` level (verified byte-identical at
     `-O1`; see divergence 125's evidence).
  3. Trimmer coverage: `SurfaceTrimmer_Crafted_MatchesHarness` pins libc++ order only for
     `componentEdges`; walking `componentHalfEdges` (`PoissonSurfaceTrimmer.Islands.cs:83`) or
     `componentBoundaryHalfEdges` (:97) in insertion order still passes. Needs a crafted mesh
     with more neighbor-rich merges.
  4. The public in-memory `PoissonMeshing` API plus the file wrapper, and
     `poisson_meshing_test.cc` 1:1, with a Tier C pycolmap fixture. Colors follow the input's
     extra PLY properties, not `options.color` (PoissonRecon ignores `--colors` for .ply).
     Review items: `PoissonMeshOutput.FromLevelSet` must throw unless `Extract` completed
     (cancelled or never-run extractors now give a partial/empty mesh), and the extractor's
     step methods (`InitSlice`, `IsoSurface`, ...) should become internal so `Extract`'s
     run-once guard can't be bypassed. Chain `PoissonMeshOutput.FromLevelSet` →
     `PoissonSurfaceTrimmer.Trim` when `trim != 0`, converting the trim as COLMAP does
     (`std::to_string` then `atof` then float: `(float)double.Parse(trim.ToString("F6",
     InvariantCulture))`).

### Verification
- End-to-end Tier C fixtures: small real photo sets reconstructed by pycolmap vs. us (also the
  first chance to reach the structure-based → structure-less registration fallback, which no
  synthetic scene triggers).

### Performance (keep results bit-identical and thread-count independent)
- Schur solvers (SchurDeterminismTests pins the output hash, which must not change; fixed-size
  kernels, cell offsets, cached inverses and two-phase parallel elimination are in). The
  parallel elimination only reaches ~1.8x at 8 threads (phase B does ~2x the sequential CPU
  work, mostly memory traffic over the E'F slots). One-time setup per structure
  (SparseSchurComplementSolver.InitStorage's SortedSet, GetCellLayout's lists) costs about as
  much as ten eliminations at 200 cams/20k pts. Still open: parallel implicit products for
  ITERATIVE_SCHUR, blocked LLT for DENSE_SCHUR.
- Fusion is single-threaded (divergence 87). `FusionOracleTests`'
  `CSharpOnly_FusedOutputIsBitIdenticalForAnyThreadCount` pins the output hash. A per-image
  precompute of world points/normals was measured and gives nothing (the per-pixel math is
  <100 ms of a ~3.3 s traversal at 10×1600×1200) for 24 B/pixel, so don't retry it. Profile
  of that run: medians (`MathUtils.NthElement`, branch mispredicts on short lists) ~30%, the
  pixel walk ~70% (~145M queue entries, memory-bound). Tried and not faster: array queue,
  skipping already-fused pushes, sort-based medians. Open levers: a branchless
  `Vector128` rank-count median (~0.6 s; needs the lane-rule test, falls back to quickselect
  on ±0/NaN), and the structural fix, a speculative band-parallel traversal with in-order
  commit.
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
