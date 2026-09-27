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
`AutomaticReconstructionController` is ported (`Controllers/AutomaticReconstruction*.cs`,
divergence 134), with a C#-only texturing step per dense model (divergence 135). Remaining:
- A fast dense test with prebuilt depth/normal maps in `stereo/` (skipping PatchMatch) that
  covers fusion with real masks, the re-undistort-when-store-empty rule and the
  advancing-front early return; today only the ~13 s textured end-to-end test covers dense.
- Poisson meshing inside the controller on a cloud dense enough to survive COLMAP's default
  trim 10 (small clouds trim to nothing, in pycolmap too); MatterCAD will need to choose trim
  for small photo sets.
- Progress restarts once within the dense stage (undistortion, then PatchMatch).
- Resume validates only a mesh's PLY header; a mesh or `fused.ply` cut off mid-body (crash
  while writing) would still make texturing throw on the next run.

### Phase 12 — Dense reconstruction (MVS)
- `poisson_meshing` (PoissonRecon port, `Mvs/PoissonRecon/`). Everything through the linear
  solve, the iso-value, the level set's iso-vertices, iso-edges and polygons, the Extract
  driver, model-space output (`PoissonMeshOutput`), `PoissonSurfaceTrimmer` and the public
  `PoissonMeshing` API (`Mvs/PoissonMeshing.cs`, `poisson_meshing_test.cc` ported) are done
  and bit-exact against the vendored C++ (harnesses `oracle/poisson_levelset{,2,3,4,5,6}_harness.cc`,
  `poisson_{extract,trim,meshing}_harness.cc`, built at `-O1`). Remaining coverage gap: the vertex-pair push loops (`PoissonLevelSetExtractor.IsoEdges.cs` slice and
     slab pushes) are reached (vertexpairs2) but not pinned — deleting either still passes, and
     a 320-input C++ search found no input where a walk uses a pushed pair. Likely needs a leaf
     two levels coarser than its neighbors across a doubly crossed edge, or a proof that grading
     rules it out.

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
- PatchMatch (Release, LOW, 1000 px, one reference + 4 sources): ~27 s CPU single-threaded,
  6-9 s wall on 10 cores; ~3.8 ns per window tap, ~3x off the per-core floor. 69% is
  `ComputeFourLockstep` (gather-bound `Sample4`), 12% the scalar best-cost recompute. CPU
  levers (~1.5-2x total, bit-identical): lockstep the 4-source recompute, halve `Sample4`'s
  gathers (ushort texel pairs, hoisted `inside` test), index rotations instead of copying.
  The big lever is a GPU port (below).
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

### Phase 13 — GPU PatchMatch (WebGPU through a host-supplied compute seam)
Done: the seam (`ColmapSharp/Compute/`), WGSL kernels (`Mvs/Shaders/`), planner, orchestrator
(`PatchMatchGpu`), CPU twin (`Mvs/Testing/ReferenceComputeDevice`, bit-identical to
`PatchMatchCpu`), public API (`PatchMatch.RunAsync(device)`, `PatchMatchController.ComputeDevice`,
`AutomaticReconstructionOptions.ComputeDevice`; divergence 136). On an Apple M5 (MatterCAD
`Tests/ColmapGpuTests`, branch `gpu-compute-adapter`, not merged) the GPU run passes every Tier C
check. The cooperative `sweep_band` (workgroup per column) makes it 1.7–2.8× faster than the
10-core CPU (1000×750: 6.1 s vs 10.5 s); on Metal only photometric sums are bit-identical to the
serial kernel (wgpu compiles with fast math). `AutomaticReconstructionController.RunAsync` lets a
non-blocking (browser) device run PatchMatch, and `Mvs/Testing/PatchMatchGpuConformance` checks
any device (RNG/conversion probes, kernel probes, four Tier C runs incl. a banded one). Remaining:
- D1b speed: workgroup-shared reference window, band re-sizing for the cooperative kernel,
  32-lane workgroups, splitting the final per-source NCC.
- Merge MatterCAD's `gpu-compute-adapter` (needs agg-sharp `cb97c4cc` pushed first — Lars).
- C3 app wiring / C4 browser smoke check wait for Phase 14.
- `sweep_band`/`filter_pixels` per-kernel GPU probes (full-run agreement covers them today).

### Phase 14 — MatterCAD integration (not started; needs Lars)
Reference `ColmapSharp` from MatterCAD, host-side image decoding into the library's pixel
buffer, a "photos → mesh" design operation with progress and cancel.
