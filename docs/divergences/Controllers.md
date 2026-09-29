# Divergences from COLMAP: Controllers: pipelines, readers, undistorters and the automatic reconstruction

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

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

## 66. The incremental pipeline has no Caspar (GPU) bundle adjustment options

**What differs.** COLMAP's `IncrementalPipelineOptions::LocalBundleAdjustment()` and
`GlobalBundleAdjustment()` also fill `BundleAdjustmentOptions::caspar` (the GPU backend's
`solver_iter_max` and `gpu_index`). ColmapSharp's `BundleAdjustmentOptions` has no Caspar
member, so `Controllers/IncrementalPipelineOptions.cs` fills only the Ceres options. `Check`
rejects a Caspar backend exactly as a COLMAP build without `CASPAR_ENABLED` does, and
`EffBa{Local,Global}MaxNumIterations` still return Caspar's default (200,
`IncrementalPipelineOptions.CasparDefaultSolverIterMax`) for a Caspar configuration.
The same holds for `GlobalMapperOptions::BundleAdjustment()` (`Sfm/GlobalMapperOptions.cs`),
which in COLMAP also sets `caspar->gpu_index`; here it sets only the Ceres `gpu_index`.

**Why.** Caspar is a GPU solver, out of scope for a managed library (PORTING_PLAN.md).

**Evidence.** The `caspar` assertions of incremental_pipeline_test.cc's
`PropagatesExplicitMaxNumIterations` and `DefaultMaxNumIterationsUsesBackendDefaults` are the
only lines not ported; their Ceres lines and all of `EffBaMaxNumIterations` pass.

## 67. BundleAdjustmentController takes BundleAdjustmentOptions instead of an OptionManager

**What differs.** COLMAP's `BundleAdjustmentController(const OptionManager&, ...)` reads only
`*options.bundle_adjustment`. The CLI's `OptionManager` is not ported, so
`Controllers/BundleAdjustmentController.cs` takes that `BundleAdjustmentOptions` directly.

**Why.** `OptionManager` is the command-line option registry (CLI executables are out of
scope); the controller's behavior depends on nothing else in it.

**Evidence.** A default `OptionManager` constructs a default `BundleAdjustmentOptions`
(option_manager.cc); `BundleAdjustmentControllerTests` (bundle_adjustment_test.cc 1:1) pass
with `new BundleAdjustmentOptions()`.

## 68. The incremental and global pipelines read point colors through a host callback, not image_path

**What differs.** COLMAP's `IncrementalPipelineOptions::image_path` and
`GlobalPipelineOptions::image_path` (with the unused copy in `GlobalMapperOptions`) name a folder that
`Reconstruction::ExtractColorsForImage` reads `image_path / image.Name()` from with
`Bitmap::Read`. ColmapSharp's options have `ReadImage`, a `Func<string, Bitmap?>` from image
name to the decoded image, and `Reconstruction.ExtractColorsForImage(imageId, bitmap)` takes
the bitmap (converted to RGB like `Bitmap::Read(as_rgb=true)`). A null `ReadImage` or a null
result behaves like COLMAP's failed read (the points stay black; COLMAP also logs a warning).

**Why.** The library does not decode image files (the host does), and MatterCAD may
hold the photos in memory rather than in a folder.

**Evidence.** With the default (null) `ReadImage` the pipeline behaves like COLMAP's default
empty `image_path`, which is what incremental_pipeline_test.cc runs; the C#-only
`IncrementalPipeline_ExtractsColorsFromReadImage` checks every image is requested and every
point takes the image color. `GlobalPipeline` extracts colors only when `ReadImage` is set,
as COLMAP extracts them only for a non-empty `image_path`.

## 70. Feature matching never waits for pairs it did not queue

**What differs.** COLMAP's `FeatureMatcherController::Match` and
`GeometricVerifierController::Verify` (controllers/feature_matching_utils.cc) count an output
for every pair that passes the skip checks, then `Pop` that many results. Two kinds of pair
are counted but never queued for a worker that exists: in `Verify`, a pair with neither
matches nor inlier matches; in `Match` with `skip_geometric_verification` (and no guided
matching), a pair whose raw matches already exist (it goes to the verifier queue, which then
has no verifiers). COLMAP blocks forever on such a batch. The port
(`Controllers/FeatureMatchingUtils.cs`) skips the first kind (still deleting its stale
two-view geometry, as COLMAP does first) and writes the second back unverified (its matches
and an empty two-view geometry).

**Why.** A hang is an upstream bug, not behavior to reproduce. The existing-matches
generator COLMAP uses for verification only yields matched pairs, but `Verify` is public and a
caller's own pair list with unmatched pairs does hit the first case. Every pair the port
handles is written exactly as a non-hanging COLMAP run would write it.

**Evidence.** Reading of feature_matching_utils.cc (the `num_outputs` counts against the
`Push` calls). `FeatureMatchingUtilsTests` (feature_matching_utils_test.cc 1:1) pass.

## 71. Unseeded RANSAC in the matching controllers starts every pair from the default seed

**What differs.** With `ransac_options.random_seed = -1` (the default), COLMAP's verifier
threads draw from their thread-local PRNG, seeded once when the thread first uses it, and the
stream continues from pair to pair, so a pair's RANSAC samples depend on which thread
verified it and what that thread verified before. The port runs each verification (and each
frame pair of rig verification) on a PRNG freshly seeded with `kDefaultPRNGSeed`, so every
pair sees the stream of a fresh COLMAP thread, and then restores the worker's own PRNG:
`Parallel.For` also runs iterations on the calling thread, whose generator COLMAP's separate
worker threads never touch. With one thread, COLMAP's first pair matches exactly; later pairs
draw different samples, a Tier C difference. The feature-pairs importer is single-threaded in
COLMAP too, on its own thread; the port seeds once at the start of the import (that thread's
stream, which then continues across pairs as in COLMAP) and restores the caller's PRNG at the
end.

**Why.** CLAUDE.md requires sequential and parallel runs to give the same result. COLMAP's
own multi-threaded results are timing dependent, so there is no fixed stream to reproduce.

**Evidence.** `FeatureMatchingUtilsTests.CSharpOnly_MatchIsThreadCountIndependent`: on
outlier-laden (60% inlier) matches with keypoint noise, 1 and 4 threads write identical
two-view geometries; with the reseed removed, the test fails.
`CSharpOnly_MatchAndVerifyLeaveCallerPrngUntouched`: the caller's next draw after `Match` and
`Verify` is the same as without them, with 1 and 4 threads. `FeatureMatchingTests` pass.

## 82. ImageReader reads images through a host-supplied source, not the file system

**What differs.** COLMAP's `ImageReaderOptions` names folders (`image_path`, `mask_path`) and
a file (`camera_mask_path`), lists them with `GetRecursiveFileList`, and decodes with
`Bitmap::Read` (OpenImageIO). The port (`Controllers/ImageReader.cs`,
`Controllers/ImageSource.cs`) takes an `IImageSource` for the images and one for the masks
(names plus a decode callback), and an already-decoded `Bitmap` for the camera mask. The
reader keeps `Bitmap::Read`'s final grey/RGB conversion. A camera mask the host cannot
decode is the host's error to report, where COLMAP logs "Failed to read invalid mask file"
and continues without a mask.

**Why.** OpenImageIO is native and excluded (docs/LICENSE_AUDIT.md), and MatterCAD (which
also runs in the browser) owns image decoding and EXIF extraction (`Sensor/ExifReader.cs`).
Everything after decoding (camera and rig assignment, focal length from EXIF, GPS and gravity
priors, mask lookup by `name.png` then `stem.png`) is unchanged.

**Evidence.** `ImageReaderTests` (image_reader_test.cc 1:1, with the written test files as
an `InMemoryImageSource` of the same bitmaps and unreadable files as null entries) pass.

## 83. Feature extraction commits images in reader order

**What differs.** COLMAP's feature extractor controller runs extractor threads between
job queues and writes each image as it leaves the extractors, so with more than one thread
the image, frame and pose prior ids follow completion order and vary from run to run. The
port (`Controllers/FeatureExtraction.cs`) extracts a batch of images in parallel and writes
the batch in reader order, so the ids always equal COLMAP's single-threaded run.

**Why.** CLAUDE.md requires sequential and parallel runs to give the same result.

**Evidence.** `FeatureExtractionTests.CSharpOnly_ThreadCountIndependentAndReportsProgress`:
1 and 3 threads give identical image ids, keypoints and descriptors.

## 98. The undistorters hand their images to a host sink, and "copy" re-hands the decoded image

**What differs.** COLMAP's undistorter controllers (`controllers/undistorters.cc`) read
images from `image_path` with `Bitmap::Read` and write them with `Bitmap::Write`
(OpenImageIO). The ports (`Controllers/ColmapUndistorter.cs`, `PmvsUndistorter.cs`,
`StandaloneImageUndistorter.cs`) read through an `IImageSource` (entry 82) and write through
an `IBitmapSink` (`Controllers/BitmapSink.cs`) under the same output paths; the requested
JPEG quality travels as the bitmap's "Compression" metadata, as in COLMAP. Where COLMAP
`FileCopy`s an image that needs no undistortion (already undistorted, or a non-perspective
camera without `max_image_size`), the port hands the host's decoded image to the sink
unchanged, so a file-writing host re-encodes it instead of copying its bytes. The
`copy_type` option (copy / hard link / symlink) of `COLMAPUndistorter` and
`StandaloneImageUndistorter` is therefore not ported. Text outputs (configs, scripts,
projection matrices, bundle and visibility files) and directories are written to disk as in
COLMAP. `ColmapUndistorter.UndistortedReconstruction` (C#-only) exposes the reconstruction it
writes to `sparse/`.

**Why.** OpenImageIO is native and excluded (docs/LICENSE_AUDIT.md); MatterCAD (also in the
browser) owns image decoding and encoding. `InMemoryBitmapStore` is both the sink and the MVS
`IBitmapSource`, so MatterCAD can go from undistortion to `Mvs.Workspace` with no image files.

**Evidence.** `UndistortersTests` (undistorters_test.cc 1:1, image existence checked in the
store) pass; `CSharpOnly_ColmapTextOutputsMatchPycolmap` and
`CSharpOnly_PmvsTextOutputsMatchPycolmap` match pycolmap 4.2.0's text files byte for byte;
`CSharpOnly_UndistortedWorkspaceFeedsMvsWithoutFiles` runs the in-memory hand-off.

## 99. The undistorters run images in batches, so a stop finishes the batch in flight

**What differs.** COLMAP queues every image on a `ThreadPool`, then waits on the futures in
order, checking `CheckIfStopped` before each; on a stop, `ThreadPool::Stop` drops the tasks no
thread has picked up, and the ones already running finish. The port (`Undistorters.RunTasks`)
starts `num_threads` images at a time with `Parallel.For` and checks `CheckIfStopped` before
consuming each image's result, starting the next batch only when not stopped. Which images
were already written when a stop is seen can therefore differ; what COLMAP guarantees (no
sparse model, configs or scripts after a stop, and no exception) is the same. The stop check
also returns true when the controller's `BaseController.CancellationToken` is cancelled,
the library's stand-in for COLMAP's Ctrl-C (`ScopedSignalHandler`). PMVS's `option-all`
writes `Environment.ProcessorCount` where COLMAP writes `std::thread::hardware_concurrency()`.

**Why.** CLAUDE.md maps ThreadPool to `Parallel.For` with per-slot writes and asks for
sequential and parallel runs to agree, and for cancellation through `CancellationToken`.

**Evidence.** `UndistortersTests.COLMAPUndistorter_StopsPendingWork`,
`CSharpOnly_CancellationTokenStopsWithoutThrowing`, `CSharpOnly_ThreadCountDoesNotChangeImages`
and `BaseControllerTests.BaseController_CancellationTokenStops` pass.

## 101. RotationAveragingPipeline checks for a stop between stages

**What differs.** COLMAP's `RotationAveragingPipeline::Run` never calls `CheckIfStopped`. The
port (`Controllers/RotationAveragingPipeline.cs`) checks `BaseController.CheckIfStopped`
(the host's `CancellationToken` or stop function) before seeding rotations from gravity
priors, before gravity refinement and before rotation averaging, and returns when a stop was
requested. A stop found after seeding first un-poses the seeded frames (whose translation is
NaN until rotation averaging), so a stopped run never leaves half-posed frames. Without a
stop request it runs exactly COLMAP's steps.

**Why.** CLAUDE.md requires long-running work to be cancellable from MatterCAD.

**Evidence.** `RotationAveragingPipelineTests.CSharpOnly_CancellationStopsBeforeRotationAveraging`,
`CSharpOnly_StopBeforeGravitySeedingSeedsNothing` (the first stop check runs before any frame
is posed) and `CSharpOnly_StopAfterGravitySeedingUnposesSeededFrames`; the three ported cases
run without a stop request and pass at COLMAP's tolerances.

## 102. RotationAveragingPipeline seeds gravity rotations through the prior's corr_data_id

**What differs.** COLMAP's `RotationAveragingPipeline::Run` looks up the image of a gravity
prior with `reconstruction_->Image(pose_prior.pose_prior_id)`. A pose prior's id is its own
row id; the image it belongs to is `corr_data_id`, which every other consumer uses, gated on
a camera sensor (`rotation_averaging.cc`, `gravity_refinement.cc`). The port
(`Controllers/RotationAveragingPipeline.cs`) uses `CorrDataId.Id` for camera priors and skips
other priors.

**Why.** A real upstream bug: whenever prior ids and image ids differ, COLMAP seeds the wrong
frame or fails on an image id that does not exist. They coincide in COLMAP's tests, where
every image has one prior written in image order.

**Evidence.** `RotationAveragingPipelineTests.CSharpOnly_GravityPriorsSeedTheirCorrespondingImage`
renumbers the priors to 101..105: the pose_prior_id lookup failed with "Image with ID 101 does
not exist"; with the fix all five images are posed within 1e-2 degrees of the ground truth.

## 107. GlobalPipeline orders equally large reconstructions by component

**What differs.** COLMAP's `GlobalPipeline::Run` sorts the new reconstructions by registered
frame count with `std::sort`, which is unstable, so reconstructions with equal counts end up
in an order that depends on the standard library. The port
(`Controllers/GlobalPipeline.cs`) sorts stably, so equally large reconstructions keep the
order in which they were mapped: the input view-graph components in order (largest first,
equally large ones by smallest frame id), and within each, the sub-components rotation
filtering splits it into, in the same order.

**Why.** CLAUDE.md requires ties whose order reaches an output to be broken explicitly.

**Evidence.** `GlobalPipelineTests.CSharpOnly_EqualSizeReconstructionsKeepComponentOrder`
pins the order of two equally large components; `GlobalPipeline_MultiComponents` and the other
multi-component cases (1:1, order-insensitive as in COLMAP) pass.

## 108. HierarchicalPipeline can be cancelled and then returns before merging

**What differs.** COLMAP's `HierarchicalPipeline::Run` never calls `CheckIfStopped`, and the
`IncrementalPipeline` it builds per cluster gets no stop function. The port
(`Controllers/HierarchicalPipeline.cs`) hands `BaseController`'s `CancellationToken` and stop
function to every cluster's `IncrementalPipeline`, so a stop request ends the running
clusters' mapping and makes the remaining clusters return at once. After the clusters, a
stopped run returns without merging and leaves the caller's `ReconstructionManager`
untouched. Without a stop request it runs exactly COLMAP's steps.

**Why.** CLAUDE.md requires long-running work to be cancellable from MatterCAD. Merging
the partial cluster reconstructions of a cancelled run would spend more time on a result
nobody asked for, and could fail COLMAP's final "at least one registered image" check.

**Evidence.** `HierarchicalPipelineTests.CSharpOnly_CancellationStopsBeforeMerging` (C#-only)
cancels after the first cluster and sees no merge and an empty manager; the four ported
cases run without a stop request and pass at COLMAP's bounds.

## 121. HierarchicalPipeline reconstructs every cluster from a fresh PRNG

**What differs.** COLMAP's `HierarchicalPipeline::Run` reconstructs the clusters on a new
`ThreadPool`. Each worker thread's `thread_local` PRNG starts from the default seed on its
first draw and continues across every cluster that worker picks up, so which cluster sees
which part of the stream depends on the schedule. The port (`Controllers/HierarchicalPipeline.cs`)
runs `Parallel.ForEach`, which uses the calling thread and reused pool threads. It clears the
thread's PRNG before each cluster, so every cluster starts from the default seed, and it
restores the thread's own PRNG afterwards. The merge then draws from the calling thread's
PRNG, as it does on COLMAP's main thread.

**Why.** Without this, a cluster continued whatever the calling thread or an earlier task
on the pool thread had drawn, so the result depended on the test order and the scheduler;
CLAUDE.md requires sequential and parallel runs to give the same result. Each cluster's
stream is now the one the first cluster on a fresh COLMAP worker gets. This is the same
choice as entry 71 for feature matching.

**Evidence.** `HierarchicalPipelineTests.CSharpOnly_ResultIgnoresClusterSchedule` (C#-only)
gets bit-identical poses from one worker and from eight, and fails without the fresh PRNG.
Before the fix, `HierarchicalPipeline_WithoutNoise` (1:1) passed alone but missed its 5e-4
projection-center bound (5.2e-4) in the full suite.

## 134. AutomaticReconstructionController runs dense stereo and Delaunay meshing on the CPU, and has no vocabulary tree

**What differs.** COLMAP's controller (automatic_reconstruction.cc) is shaped by its build flags;
the port behaves like this:
- `dense` defaults to true and PatchMatch stereo, fusion and meshing run. COLMAP defaults
  `dense` to true only with CUDA and MVS, and without CUDA returns after image undistortion
  ("Skipping patch match stereo because CUDA is not available"). PatchMatch runs on the CPU,
  or on the host's `ComputeDevice` when one is set and supports blocking waits (entry 136);
  a device that does not (the browser) is not used by this synchronous controller, which
  warns and runs PatchMatch on the CPU.
- Delaunay meshing runs (a CGAL-free port, `Mvs/DelaunayMeshing.cs`), as in a CGAL build.
  Advancing-front meshing logs "Skipping advancing front meshing because CGAL is not available"
  and returns, as in a build without CGAL.
- `vocab_tree_path` does not exist: individual and internet data always match exhaustively
  (COLMAP switches to vocabulary-tree matching at 200 images when a tree is given), and video
  data matches sequentially with `loop_detection` off (COLMAP turns it on with its downloadable
  tree). `use_gpu`, `gpu_index` and `ba_backend` do not exist either.
- `image_path` / `mask_path` are host image sources, `database.db` is a `Database` (an
  `InMemoryDatabase` unless the host passes one), `sparse/project.ini` is not written, and the
  undistorted images stay in memory, so a model is undistorted again when its images are not in
  the controller's store even though `dense/<i>` exists.
- Video data keeps the host's `ImageNames` selection, a deliberate fix of an upstream ordering
  bug. COLMAP sets `image_reader`/`mapper` `image_names` (automatic_reconstruction.cc:67-69) and
  then calls `ModifyForVideoData`, whose `ResetOptions(false)` (option_manager.cc:103-105, 1221,
  1231) rebuilds `image_reader` and `mapper`, while `BaseOptionManager::ResetOptions` restores
  only the project/database/image *paths*. So a COLMAP video run silently processes every image,
  unlike individual and internet data, which keep the selection.

**Why.** CUDA/GPU, CGAL, vocabulary-tree retrieval, SQLite files, OpenImageIO decoding and the
CLI option registry are out of scope (`PORTING_PLAN.md`, `docs/LICENSE_AUDIT.md`), while the
PatchMatch algorithm and Delaunay meshing are ported, so running them is what a COLMAP build with
those features does. Sequential matching with loop detection throws here
(`SequentialPairGenerator.cs`), so leaving COLMAP's `loop_detection = true` would make every video
run fail. COLMAP's clearing of `image_names` for video data is an ordering bug (the names are
set before the preset that wipes them) that silently drops the user's image selection, so the
port restores the selection after the preset, matching individual and internet data.

**Evidence.** `AutomaticReconstructionTests.ParameterizedAutomaticReconstructionTests_Nominal`
(automatic_reconstruction_test.cc 1:1, dense off as in COLMAP's test) passes for the incremental
and hierarchical mappers. The dense path is pinned by the C#-only
`AutomaticReconstructionTests.CSharpOnly_DenseTexturedSceneGivesFusedPointsAndMesh`: a
ray-traced textured scene goes photos in through CPU PatchMatch, fusion and Delaunay meshing to
a non-empty `fused.ply` and `meshed-delaunay.ply`, reporting progress only under the
controller's stage names. `CSharpOnly_MaskSourceRoutesByReservedKey` pins that the fusion's
masks and the undistorted images cannot be confused, even for a workspace at `masks/ws`.
`CSharpOnly_VideoDataKeepsImageNamesSelection` pins the video-data fix: with four of six
consecutive views selected, only those four reach the database and are registered (the test
fails when the restore after `ModifyForVideoData` is removed).

## 135. AutomaticReconstructionController textures each dense mesh

**What differs.** COLMAP's automatic reconstruction ends at the mesh; texturing is the separate
`mesh_texturer` command (`RunMeshTexturer`, colmap/exe/mvs.cc). The port adds
`AutomaticReconstructionOptions.Texture` (default true) and, per dense model after meshing, runs
that command's body (`Controllers/AutomaticReconstruction.Texture.cs`) with
`workspace_path = dense/<i>`, `input_path` = the Poisson or Delaunay mesh just written,
`output_path = dense/<i>/<mesh name>-textured` (e.g. `meshed-delaunay-textured`), `output_type`
BIN and default `MeshTextureMappingOptions` (only `num_threads` follows the controller's, like
every other stage). Where that command reads image files and encodes `texture.png`, the port
reads the controller's in-memory undistorted images and hands the atlas to the host's
`AutomaticReconstructionOptions.TextureSink` under `.../texture.png` (not at all when the atlas
is empty); `mesh.ply` (per-corner UVs, `comment TextureFile texture.png`) is written as COLMAP
writes it. Every result is also kept in memory in
`AutomaticReconstructionController.TexturedMeshes`, and progress is reported under a
`"Texturing"` stage. Because that result lives in memory, a model whose `fused.ply` and mesh
already exist is still undistorted again and textured once per controller (PatchMatch, fusion
and meshing stay skipped); with `Texture` off, the skip is COLMAP's.

Resuming after an interrupted mesher also differs, deliberately. `PoissonMeshing` creates its
output file before reconstructing (COLMAP's `THROW_CHECK_PATH_OPEN`, kept as is in
`Mvs/PoissonMeshing.cs`), and COLMAP ignores its `false` result, so a failed or cancelled run
leaves an empty `meshed-poisson.ply` beside a complete `fused.ply`. COLMAP's next run skips that
model for good on `ExistsFile`; here texturing would fail to read it and abort every resume.
So the controller deletes the mesh file when a mesher throws, is cancelled or returns `false`
(and then skips texturing that model, going on to the next as COLMAP does), and on a resume a
mesh file without a complete PLY header naming vertex `x`, `y`, `z` and a face element counts
as missing: it is deleted and the model is re-meshed from its kept `fused.ply`. Texturing
progress always ends at 100%, also when `MeshTextureMapping` returns early (empty mesh, no
views).

**Why.** MatterCAD's photo-to-mesh result carries the photos' colors as a texture, so the
automatic pipeline has to end in a textured mesh; running COLMAP's own texturer step with its
own defaults keeps the result what `colmap automatic_reconstructor` followed by
`colmap mesh_texturer` gives. Image encoding is left to the host throughout the library
(entry 98).

**Evidence.** `AutomaticReconstructionTests.CSharpOnly_DenseTexturedSceneGivesFusedPointsAndMesh`
runs photos in to a textured Delaunay mesh: one textured model whose UVs number six floats per
face and lie in [0, 1], more than half the faces assigned a view, a non-empty atlas that is the
same bitmap the sink received, a `mesh.ply` whose header names `texture.png` and whose UVs match,
and the `"Texturing"` stage last in the progress stages.
`AutomaticReconstructionTests.CSharpOnly_ResumeReusesDenseResultsAndRecoversPartialMesh` pins
the resume rules on one finished workspace: with `Texture` on, a new controller leaves
`fused.ply` and the mesh untouched (same length and write time), reports no fusion or meshing
stage and textures the model; with `Texture` off it reports nothing past the dense heading and
textures nothing; over a zero-byte mesh it re-meshes from the untouched `fused.ply` and
textures; cancelling during Poisson meshing leaves no `meshed-poisson.ply`; and a Poisson run
whose trim empties the mesh still ends texturing progress at 1000 of 1000.

## 137. AutomaticReconstructionController reseeds the PRNG before Delaunay meshing

**What differs.** `CreateSubSampledDelaunayTriangulation` inserts the points in the order of a
`Shuffle` drawn from the calling thread's PRNG (`Mvs/DelaunayMeshingInput.cs`, unchanged). COLMAP's
controller calls `DenseDelaunayMeshing` on its own thread without seeding, so the shuffle starts
from whatever that thread drew before: after a fresh sparse reconstruction, the state the mapper's
RANSAC left (`RANSAC` calls `SetPRNGSeed(random_seed)` and then samples on the controller thread,
since `mapper->Run()` runs inline); on a resume that reads `sparse/` back, a fresh thread; and for
model `i > 0`, whatever model `i - 1`'s meshing left, or not, if that model was skipped. The port's
controller calls `RandomUtils.SetPRNGSeed()` (seed `DefaultPRNGSeed`, 0, what a fresh COLMAP thread
uses) right before each model's Delaunay meshing (`Controllers/AutomaticReconstruction.Dense.cs`),
so a model's mesh depends only on its undistorted sparse model, `fused.ply` and `fused.ply.vis`.
Here the dependence was worse than in COLMAP: the controller runs synchronously on the host's
thread, often a reused pool thread, so the state also depended on unrelated earlier work in the
process.

**Why.** A resumed workspace that re-meshes a model must give the same mesh as the run that built
it, and the same inputs must give the same mesh whatever ran before in the process. The seed
matches COLMAP's resumed-run behavior, so only the first-run case changes.

**Evidence.** In `AutomaticReconstructionTests.CSharpOnly_ResumeReusesDenseResultsAndRecoversPartialMesh`
the first run and the resume that re-meshes over a zero-byte mesh fed Delaunay meshing identical
inputs (instrumented: same hash over every point position, visibility count, image pose float and
point index; 1882 points, 4 images) but different shuffles (first eight indices
`1420,1170,73,1630,...` vs `1061,1594,1533,740,...`), so `meshed-delaunay.ply` differed while
`fused.ply`, `fused.ply.vis` and the depth/normal maps were byte-identical. The test now asserts
the re-mesh is byte-identical to the first run's mesh, and fails without the reseed.

## 138. AutomaticReconstructionController runs the sparse mapper from a fresh PRNG

**What differs.** COLMAP's `AutomaticReconstructionController` is a `Thread`: `Run` executes on
a new thread, and `mapper->Run()` runs inline there. With the default `random_seed` of -1,
RANSAC does not seed, so the mapper draws from that thread's `thread_local` PRNG, which starts
from the default seed on its first draw (extraction and matching ran on their own threads). The
port's controller runs on the caller's thread, often a reused pool thread after an `await`, so
the mapper continued whatever that thread drew before. `RunSparseMapper`
(`Controllers/AutomaticReconstruction.cs`) now clears the thread's PRNG before building and
running the mapper, and restores the caller's PRNG afterwards, the same choice as entries 71
and 121. The global mapper's relative pose re-estimation runs on worker threads in COLMAP too;
entry 139 covers it.

**Why.** The same photos must give the same model whatever ran before in the process;
CLAUDE.md requires sequential and parallel runs to agree. The mapper now sees the stream a
fresh COLMAP controller thread gives it, so only runs that inherited a used PRNG change.

**Evidence.** `demo/ColmapDemo.Tests/SessionGpuFallbackTests` runs two sessions one after the
other on pool threads; instrumented, each mapper left its thread with a used PRNG, and about one
suite run in four the second session's sparse points differed
(sum of x 191.4505 vs 191.6660, same 197 points), so the mesh had 30624 faces instead of 31843. `AutomaticReconstructionTests.CSharpOnly_UnseededSparseModelIgnoresTheCallersPrng`
(C#-only) runs the sparse stages twice on one thread after seeding it differently; `cameras.bin`
differed without the fresh PRNG, and all sparse files are byte-identical with it.

## 140. AutomaticReconstructionController builds dense/<i> from sparse/<i>

**What differs.** `RunSparseMapper` (colmap/controllers/automatic_reconstruction.cc) writes the
mapper's models through `ReconstructionManager::Write`, which puts them in `sparse/<i>` sorted by
descending 3D-point count, but leaves them in memory in mapper build order. `RunDenseMapper`
then names `dense/<i>` after the in-memory index, so on a first run with two or more models
`dense/<i>` can be built from a different model than `sparse/<i>`. A later run over the same
workspace (COLMAP's own resume, which reads `sparse/<i>` back in sorted order) then pairs model
i with another model's `dense/<i>`: it skips that model's dense work, or runs PatchMatch over the
other model's undistorted images and depth maps and fuses and meshes the mix. The resume also
sorts the `sparse/` directory names as strings, so with 11 or more models `sparse/10` becomes
model 2. The port reorders the models in memory right after `Write`, the same way (descending
point count, ties in index order as in entry 34), and reads a resume's `sparse/<i>` by the number
i (`Controllers/AutomaticReconstruction.cs`). So `dense/<i>`, `sparse/<i>`,
`ReconstructionManager.Get(i)` and `TexturedMeshes[].ModelIdx` (entry 135) all name the same
model, the one with the most points first, on a first run and on every resume. With one model
nothing changes.

**Why.** A resumed workspace must not mix two models' files, and a host that retries the dense
stages over an existing `sparse/` (MatterCAD's demo does, after a GPU failure) must get the same
models at the same indices as the first run.

**Evidence.** `AutomaticReconstructionTests.CSharpOnly_DenseModelsFollowSparseOrderOnFirstRunAndResume`
(C#-only) hands the controller two models in build order [34 points, 103 points]; before the fix
`sparse/` held [103, 34] while `dense/0` and `dense/1` held the undistorted models of [34, 103].
With the fix both the first run and a resume build `dense/<i>` from `sparse/<i>`, and every
dense product (depth and normal maps, fused.ply, fused.ply.vis, the mesh and its texture) is
byte-identical between the two runs. `CSharpOnly_ResumeReadsSparseModelsInNumericOrder`
(C#-only) reads 11 models back; before the fix they came in the order
`0, 1, 10, 2, ..., 9`.

## 141. AutomaticReconstructionOptions can keep bundle adjustment off a known camera

**What differs.** COLMAP's `AutomaticReconstructionController::Options` has no way to stop the
mapper refining the focal length or the extra parameters; the mapper always runs with
`ba_refine_focal_length = ba_refine_extra_params = true` there. The port adds
`AutomaticReconstructionOptions.BaRefineFocalLength` and `BaRefineExtraParams` (both default
true, so the default run is COLMAP's) and copies them onto the mapper's options after the presets.

**Why.** The reconstruction benchmark (`Mvs/Testing/Benchmark`) found that on a small object in
a narrow view (about 70 px across, a 14 degree field of view) bundle adjustment trades focal
length against depth: the focal collapsed from 288 to 120 and relative rotations came out 20-30%
low, and pycolmap 4.2.0 does the same (focal 96.6). With the true focal given in `CameraParams`
and refinement off, the rotations match the truth. A phone's EXIF focal or a calibrated camera
(MatterCAD's planned calibration target) is that case.

The options only reach the incremental and hierarchical mappers. Under `MapperType.Global`
the controller's constructor throws an `ArgumentException` when either is false: the global
path runs `ViewGraphCalibration`, which re-estimates every focal length, and
`GlobalMapperOptions`' bundle adjustment has no switch for intrinsics, so the setting would be
silently ignored. Honouring it there would mean skipping view-graph calibration and threading
the refine flags through `GlobalPipelineOptions`, a larger change than the known-camera case
needs today.

**Evidence.** `AutomaticReconstructionTests.CSharpOnly_BaRefineOptionsReachTheMapper` and
`CSharpOnly_BaRefineOffIsRefusedUnderGlobalMapper`; the benchmark's known-intrinsics case in
`benchmarks/baseline.json`.

## 143. Video data can add KLT tracks as keypoints and matches

**What differs.** COLMAP has no video tracker. With `AutomaticReconstructionOptions.VideoTracking`
on (off by default, so the default run is COLMAP's) and video data with SIFT, the matching stage
runs `SequenceTracker` over the frames in name order, picks keyframes by parallax
(`Feature/Tracking/TrackMatcher.cs`), appends each keyframe's track keypoints with fixed-scale
SIFT descriptors after its own SIFT features, runs the usual sequential matching, then adds
each keyframe pair's track matches to the pair's descriptor matches and re-runs two-view
verification on the union (`Controllers/VideoTrackMatching.cs`).

**Why.** Frames SIFT cannot place (dark or low-texture views) still carry trackable corners.

**Evidence** (`ColmapSharp.Benchmarks --quick`, 40 frames 480x360, mapper seeds 1 and 2, video
data; mean over the seeds). With the true masks (`--masks`): realistic TexturedSphere registers
34/40 without tracks and 40/40 with them, median rotation error 1.31 -> 1.05 degrees, position
1.63% -> 0.90%; with known intrinsics 34 -> 40 frames but rotation 0.38 -> 0.92 degrees (the
median now includes the six frames SIFT could not place). DarkObject stays at 0/40. Without masks
the tracker also follows the still wall: the sphere's rotation error rises from 1.25 to 6.8
degrees and the focal collapses (|ln f/f0| 0.27 -> 0.99), and DarkObject "registers" 26-27/40
frames that cannot be aligned to the truth (wall-only geometry). So it is off by default, and
meant for use with object masks.

**Pose error on the frames both runs place** (true masks, seeds 1 and 2, the 34 frames SIFT
alone registers): self-calibrating, the tracks help (median rotation 1.16/1.47 -> 0.83/0.83
degrees); with known intrinsics they hurt (0.33/0.42 -> 0.61/0.60). The cause is KLT drift: a
track's position against the ray-cast truth is off by a median 0.75 px after 1-2 frames and
2.2 px after 6-8. None of the knobs tried restores the known-intrinsics case: skipping pairs
with many SIFT inliers (MaxSiftInliers 15-100) 0.56-0.61, all-zero track descriptors
(DescribeTracks off) 0.55, capping pair spans (MaxPairFrameGap 8) no change, cutting tracks
into 7-frame pieces (MaxTrackLength) 0.49/0.51 with descriptors, 0.66 without. All stay off by
default.
