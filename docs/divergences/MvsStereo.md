# Divergences from COLMAP: Mvs, stereo: MVS files, PatchMatch and fusion

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

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

## 84. PatchMatchController's "__auto__" source images order equal counts by image index

**What differs.** For a `__auto__, N` line of patch-match.cfg,
`PatchMatchController::ReadProblems` (patch_match.cc) ranks the reference image's
overlapping images by shared-point count with `std::partial_sort`, which leaves equal
counts in an implementation-defined order. The port (`Mvs/PatchMatchController.cs`) breaks
ties by ascending image index, the order the candidates come out of COLMAP's `std::map`.
When a tie straddles the cut-off at N, the chosen source images can differ from COLMAP's,
and tied images can be listed in a different order (the order of the source images is
the layer order of the PatchMatch kernel, so it reaches the depth maps through the
Monte Carlo sampling of source images).

**Why.** The same rule and reasoning as entry 64: libc++'s heap-based `partial_sort` is
not part of COLMAP's contract, and the index order is deterministic on every platform.

**Evidence.** `PatchMatchControllerTests.ReadProblems_AutoTiesGoToLowerImageIndex` and
`ReadProblems_AutoRanksBySharedPoints` (C#-only; COLMAP has no patch_match_test.cc) pin
the order, with the shared-point counts they rely on asserted first.

## 85. PatchMatch problems list their source images in configured order

**What differs.** `PatchMatchController::ProcessProblem` collects a problem's used images
(reference plus sources) in a `FlatHashSet<int>` (Boost `unordered_flat_set`) and rebuilds
the source image list by iterating it, so the sources the kernel sees come in hash order.
The port (`PatchMatchController.SetUpProblem`) iterates the reference image, then the
configured source images in their configured order, each once. The source order is the
kernel's layer order, which changes which source image a Monte Carlo draw picks, so the
computed depth and normal maps can differ (PatchMatch is Tier C, so this is within its
bar).

**Why.** Reproducing Boost.Unordered's bucket layout and hash mixing is out of reach (Boost
is not ported, entry 2), and the configured order is deterministic and what a reader of
patch-match.cfg expects.

**Evidence.** `PatchMatchControllerTests.SetUpProblem_Photometric` and
`SetUpProblem_MissingFiles` (C#-only) pin the order.

## 86. PatchMatch runs on the CPU: no GPU index, and its own random numbers

**What differs.** COLMAP's PatchMatch stereo runs only on CUDA: `PatchMatchOptions` has a
`gpu_index` string ("-1" for all devices), `PatchMatch::Check` requires exactly one index
>= -1, and `PatchMatchController` runs one problem per GPU (`ReadGpuIndices`). The port has
no CUDA and no device index: `gpu_index` and `ReadGpuIndices` are not ported, and
`PatchMatch.Check` skips the index checks. PatchMatch runs on the CPU, where `num_threads`
bounds the parallelism, unless the host passes a WebGPU compute device (entry 136).

**Why.** The library is pure managed code (CLAUDE.md, contract 1); the CUDA kernel
(`patch_match_cuda.cu`) is ported to CPU code, and the optional GPU path runs on the one
device the host supplies, so a device index has no meaning.

**Evidence.** `PatchMatchTests` (C#-only) pins every other `PatchMatchOptions::Check` bound
and `PatchMatch::Check` condition.

**Addendum: random numbers.** COLMAP draws PatchMatch's random numbers from cuRAND XORWOW
states kept in a map (`GpuMatPRNG`), each seeded with `curand_init(id, 0, 0)` where `id` is
the thread's position in the CUDA launch grid, and rotated with the other maps after each
sweep. `FillWithRandomNumbers` (the initial depths) and `InitNormalMap` draw from one state
per pixel. The sweep (`SweepFromTopToBottom`, patch_match_cuda.cu) instead takes one state per
column of the current rotated frame (`rand_state_map.Get(0, col)`), draws from it in order
down the whole column, and stores it back at row 0. The port draws from
`Mvs/PatchMatchRandom.cs`, a counter-based generator (SplitMix64's mixer over a Weyl
sequence) whose every draw is a function of (seed, pixel of the original reference image,
phase, draw index); the phase is the initial depth, the initial normal, or sweep s of
iteration i, and the sweep keys each pixel's draws on that pixel rather than continuing a
per-column stream. Draws map to (0, 1] with curand_uniform's formula. The random values, and
so the depth and normal maps, differ from any GPU run; PatchMatch is Tier C, and cuRAND's
sequence depends on the CUDA launch layout, so no run of COLMAP is a fixed target either.
What the port adds is independence from threads: the same seed gives the same draws for any
thread count or scheduling, and no generator state has to be stored or rotated. Evidence:
`PatchMatchInputsTests.PatchMatchRandom_*` and `ToOriginalPixel_UndoesMatRotate`
(C#-only); `GpuMatTests` (gpu_mat_test.cu 1:1) fill its matrices from this generator instead
of `GpuMatPRNG`.

## 87. StereoFusion traverses on one thread

**What differs.** COLMAP's `mvs::StereoFusion::Run` (fusion.cc) splits each image into 10-row
tasks on a `ThreadPool` of `num_threads` threads (when the workspace is pre-loaded). The tasks
read and write the shared fused-pixel masks without synchronization, and each thread appends to
its own point list, concatenated by thread id at the end. Which task claims a pixel first, and
so which points are produced and in what order, depends on scheduling. The port
(`Mvs/Fusion.cs`) runs the traversal on one thread in row order.
`StereoFusionOptions.NumThreads` only sets the workspace loading parallelism.

**Why.** CLAUDE.md requires the same result for every thread count. The one-thread traversal
is exactly COLMAP's own behavior with `num_threads = 1` and with `use_cache = true` (which
always fuses on one thread), so the result is one COLMAP can produce. A parallel version that
reproduces the one-thread run bit for bit is possible (for example, row bands traversed
speculatively in parallel and committed in row order, redoing a band whose traversal touched
pixels an earlier band claimed); that is future work if fusion time matters.

**Evidence.** `FusionOracleTests.StereoFusion_MatchesPycolmap` (C#-only): against pycolmap
4.2.0's `stereo_fusion` with `num_threads = 1` on the checked-in workspace, the same 154 points
in the same order, colors exact, positions within 1e-6 (a couple of float ulps).
`StereoFusion_MatchesPycolmapWithOptions` does the same for 129 points with noisy normals, a
bounding box, `max_image_size`, masks and short traversals.
`StereoFusion_CachedWorkspaceMatchesLoaded` pins identical output for the cached and
pre-loaded workspaces.

## 88. StereoFusion lists each point's visible images in ascending index order

**What differs.** COLMAP collects a fused point's image indices in a `FlatHashSet<int>`
(`std::unordered_set` in the default build) and copies it into the visibility vector, so the
order within each list is the hash set's iteration order. The port lists them in ascending
image index. `GetFusedPointsVisibility` and the `.vis` file (`WritePointsVisibility`) carry the
same sets; only the order within a list can differ.

**Why.** Hash iteration order is implementation-defined (libc++, libstdc++ and the Boost
backend all differ); CLAUDE.md asks for a deterministic order. Consumers that read the lists
(the meshers, not yet ported) should be checked for order sensitivity when they are ported.

**Evidence.** `FusionOracleTests.StereoFusion_MatchesPycolmap` compares each point's list with
pycolmap's, sorted.

## 89. StereoFusion throws on a mask it cannot decode

**What differs.** In `StereoFusion::InitFusedPixelMask` (fusion.cc), a mask file that exists but
fails `Bitmap::Read` is ignored: the image is fused as if it had no mask. The port
(`Mvs/Fusion.cs`) lets the host's `IBitmapSource.Read` exception propagate out of `Run`.

**Why.** A user who supplied masks expects the masked background to stay out of the point
cloud; silently fusing it produces a wrong result with no hint why. An error that names the
mask tells them what to fix. `IBitmapSource.Read` is also specified to throw on an unreadable
image, so there is no "false" return to mirror.

**Evidence.** `FusionTests.CSharpOnly_UnreadableMaskThrows` (C#-only): a source whose `Exists`
reports the mask and whose `Read` throws makes `Run` throw that exception.

## 95. PatchMatch samples source images with exact float bilinear weights

**What differs.** `PatchMatchCuda::InitSourceImages` binds the source images to a layered
CUDA texture with linear filtering. NVIDIA hardware computes the bilinear weights in 9-bit
fixed point (8 fractional bits), so a sample is quantized to steps of 1/256 of a texel. The
port (`Mvs/PatchMatchTextures.cs`) interpolates with exact float weights, using the formula of
COLMAP's own gfx9 emulation, `SampleLayeredBilinear` (patch_match_cuda.cu), with the same
texel-centre offset (+0.5) and zero border. Photo-consistency costs can differ from a GPU run
in their low bits.

**Why.** There is no texture unit on the CPU; exact weights are what COLMAP itself uses on
AMD gfx9, and the difference is below the NCC's sensitivity. PatchMatch is Tier C.

**Evidence.** `PatchMatchInputsTests.SourceImages_BilinearWithZeroBorder` (C#-only) pins
texel centres, midpoints, the zero border and padding of smaller layers.

## 96. PatchMatch float math uses .NET's MathF and no contraction

**What differs.** COLMAP's PatchMatch kernels (`gpu_mat_ref_image.cu`'s prefilter and
`patch_match_cuda.cu`) call CUDA's `expf`, `sqrtf`, `rsqrtf`, `sinf`/`cosf` and `erff`, and
nvcc contracts `a * b + c` into FMA by default. The port keeps every expression in float in
COLMAP's order but evaluates it with .NET's `MathF` functions (`rsqrt(x)` becomes
`1 / MathF.Sqrt(x)`) and never fuses a multiply-add (CLAUDE.md). `erff`, which .NET lacks, is
evaluated in double from Abramowitz and Stegun 7.1.6 and rounded to float
(`Mvs/PatchMatchLikelihood.cs`). CUDA's float `min`/`max` are `fminf`/`fmaxf`, which return
the other operand when one is NaN; the port keeps that (`PatchMatchKernel.CudaMin`/
`CudaMax`), because degenerate geometry relies on it to yield a bounded cost. Results can
differ from a GPU run in the last bits; downstream, PatchMatch's hypotheses can then diverge.

**Why.** CUDA's device math library is not available, and fused operations would make
results differ across CPUs. The rule keeps the port identical on every platform. PatchMatch
is Tier C.

**Evidence.** `PatchMatchInputsTests.RefImageFilter_MatchesBruteForce` (C#-only) checks the
prefilter against a double-precision evaluation to 1e-5;
`PatchMatchKernelTests.Likelihood_ErfMatchesKnownValues` checks `ErfF` against A&S's table;
`PatchMatchKernelTests.GeomConsistencyCost_DegenerateBackProjectionCostsTheMaximum` and the
`ComputeIncProb(float.NaN)` assertion in `Likelihood_MessagesAndPriors` pin the
`fminf`/`fmaxf` behavior.

## 97. PatchMatch computes window radii 21 to 32

**What differs.** `PatchMatchOptions::Check` accepts `window_radius` up to 32
(`kMaxPatchMatchWindowRadius`), but `PatchMatchCuda::Run` only instantiates its templated
kernels for radii 1 to 20; for 21 to 32 it logs "Window size ... not supported" and computes
nothing, leaving the depth and normal maps as initialized. The port's kernel takes the radius
at run time (`Mvs/PatchMatchKernel.Photometric.cs`), so every radius `Check` accepts is
computed.

**Why.** The limit is an artefact of CUDA template instantiation, and silently returning
random maps for an accepted option is a bug; computing is what the option asks for. Radii
1 to 20 behave as in COLMAP.

**Evidence.** `PatchMatchTests.PatchMatchOptions_CheckBounds` pins the accepted range;
`PatchMatchKernelTests.PhotoConsistency_ComputesRadiiBeyondCudaTemplates` computes the NCC
at radii 25 and 32.

## 122. PatchMatchController runs problems one at a time and aborts the one in flight on stop

**What differs.** `PatchMatchController::Run` (patch_match.cc) runs one problem per GPU on
a thread pool and checks `CheckIfStopped()` only at the start of `ProcessProblem`, so a stop
lets every in-flight problem finish and write its maps. The port
(`Mvs/PatchMatchController.cs`) runs the problems one after another, each PatchMatch run
parallel over columns on the CPU or over pixels on the host's compute device (entry 136),
and passes the CancellationToken into `PatchMatch.RunAsync`: a stop aborts the running
problem, which writes no maps, and `Run`/`RunAsync` return normally. The problems already
finished keep their maps; a later run redoes the aborted one (its outputs do not exist, so
it is not skipped).

**Why.** There is at most one device to run problems on (entries 86 and 136): one CPU
PatchMatch run already uses every core, and the host's GPU is not safe to share between
concurrent runs (`IComputeDevice` is not thread-safe). On the CPU a single 2 MP problem
takes minutes, and a user who cancels should not have to wait for it; skipping its outputs
keeps the workspace free of half-computed maps.

**Evidence.** `PatchMatchControllerTests.Run_CancelledMidProblemWritesNothingForIt` and
`Run_StopsWithoutErrorWhenCancelled` (C#-only).

## 136. PatchMatch can run on a host-provided WebGPU device

**What differs.** COLMAP's PatchMatch runs on CUDA devices chosen by `gpu_index`
(`patch_match_cuda.cu`). The port adds an optional GPU path that looks nothing like it from
the outside:
- **Host-supplied seam.** The library never references a graphics API. The host passes an
  `IComputeDevice` (`Compute/IComputeDevice.cs`, a small WebGPU-shaped slice: buffers, WGSL
  kernels, bind groups, recorded dispatches, async flush and readback) to
  `PatchMatch.RunAsync`, to `PatchMatchController.ComputeDevice` or to
  `AutomaticReconstructionOptions.ComputeDevice`. Without one, PatchMatch runs on the CPU as
  before (entry 86).
- **WGSL port of the CUDA kernel structure.** `Mvs/Shaders/*.wgsl` follow
  `patch_match_cuda.cu`'s kernels, with the entry points `init_random`, `initial_cost`,
  `backward_messages`, `sweep_band`, `filter_pixels`, `rotate_planes` and `rotate_normals`,
  recorded by `Mvs/PatchMatchGpu*.cs` in `PatchMatchCpu.Run`'s schedule and flushed once per
  sweep.
- **Filter as a separate pass.** COLMAP filters inside the last sweep; the port runs the
  filter (`filter_pixels`) after the last sweep's bands, and the CPU does the same, which was
  proved bit-identical to the fused sweep before the split (`PatchMatchSweepBandTests`' golden
  hashes).
- **Banded sweeps.** A column sweep is a `backward_messages` pass followed by row bands of
  `sweep_band`, so one dispatch stays within the device's limits; `PatchMatchGpuPlan` picks
  the band height. The CPU runs the same banded schedule, again pinned by
  `PatchMatchSweepBandTests`.
- **CPU fallback with a reason.** Before creating any buffer, `PatchMatchGpuPlan.TryCreate`
  checks every buffer, binding and dispatch against the device's limits. When one does not
  fit, PatchMatch runs on the CPU and says why in a sentence ending "Using the CPU."
  (`PatchMatch.FallbackReason`, a warning from the controller, and " (CPU)" instead of
  " (GPU)" in the controller's progress messages); `PatchMatch.Backend` records which ran. A
  GPU error during a run is not a fallback reason: it propagates. The device entry points are
  `PatchMatch.RunAsync` and `PatchMatchController.RunAsync` (`PatchMatch.Run` stays CPU-only).
  `PatchMatchController.Run` refuses a device that cannot be waited on synchronously (the
  browser) with `InvalidOperationException`, pointing at `RunAsync`. The automatic
  reconstruction has both entries too: `AutomaticReconstructionController.RunAsync` runs the
  same stages and awaits PatchMatch on any device, while its synchronous `Run` warns about a
  non-blocking device and runs PatchMatch on the CPU. Hosts in the browser must call `RunAsync`.
- **Random numbers.** The GPU draws the same counter-based random numbers as the CPU
  (`PatchMatchRandom`, entry 86), transliterated to 32-bit WGSL integer arithmetic; they are
  bit-exact by construction (`PatchMatchShaderRngTransliterationTests` pins the
  transliteration).
- **Everything else is Tier C on a real GPU.** WGSL's `exp`, `sin`, `cos` and `sqrt` f32
  builtins need not be correctly rounded, a GPU compiler may contract `a*b + c` into a fused
  multiply-add, and a GPU may flush subnormals; the kernels guard the NaN, infinity and
  subnormal cases whose outcome would otherwise change control flow (`patch_match_common.wgsl`),
  but depth and normal values can differ from the CPU's in the last bits and, through the
  sweeps, beyond. On Metal, wgpu compiles shaders with the default `MTLCompileOptions`
  (wgpu-hal's `metal/device.rs`), so fast math is on and the compiler may also reassociate
  float sums, differently in each kernel. That is why `sweep_band`'s two schemes (cooperative,
  a workgroup per column, the default; serial, an invocation per column) agree bit for bit on
  photometric runs, where each hypothesis's cost gains one term per sample, but only within
  tight bounds on geometric runs, where it gains two (`c += ncc; c += geom`).

**Why.** CUDA is native code and not available in the browser; WebGPU is what MatterCAD can
reach on every platform, including browser-wasm, and keeping the device behind a host-provided
seam keeps the library pure managed code (CLAUDE.md, contract 1). CPU PatchMatch takes minutes
per megapixel image, so a GPU path is what makes dense reconstruction usable. Mirroring the
CPU schedule exactly, rather than COLMAP's fused sweep, is what lets the GPU pipeline be
proved against the CPU on a CPU twin, and the fallback keeps a device that is too small from
failing a reconstruction.

**Evidence.** `ReferenceComputeDevice` (`Mvs/Testing/`) runs every WGSL kernel's CPU twin over
the GPU's buffers. `PatchMatchGpuTwinTests` pins the GPU pipeline on it bit-identical to
`PatchMatchCpu` (photometric, geometric and filtered runs, with one and with several bands).
`PatchMatchBackendTests` pins the public entry point: `RunAsync` on the twin reports `Gpu` and
gives the CPU's maps bit for bit; a device whose limits cannot hold the problem reports `Cpu`
with the planner's reason and records no device call; a non-blocking device runs through
`RunAsync`. `PatchMatchControllerTests.RunAsync_WithTwinDevice_WritesTheSameMapsAsTheCpu`,
`RunAsync_DeviceTooSmall_FallsBackToCpuAndSaysSo` (maps, " (CPU)" message and the logged
reason), `RunAsync_WithDevice_CancelledMidProblemWritesNothingForIt` and
`Run_WithNonBlockingDevice_ThrowsBeforeWritingAnything` pin the controller, and
`AutomaticReconstructionTests.CSharpOnly_ComputeDeviceGivesTheSameDenseResults` pins that the
automatic reconstruction writes the same depth maps, `fused.ply` and mesh with the twin as without,
and that `Run` ignores a non-blocking device with a warning;
`AutomaticReconstructionTests.CSharpOnly_RunAsyncGivesTheSameResultsAsRunOnAnyDevice` pins that
`RunAsync` writes the same bytes as `Run` and uses a non-blocking twin with the same outputs and no
warning, and `CSharpOnly_RunAsyncStopsMidDenseLikeRun` that both stop mid-dense alike.

On a real GPU (Apple M5, Metal), MatterCAD's `Tests/ColmapGpuTests` (`PatchMatchRandomGpuTests`,
`PatchMatchGpuKernelProbeTests`, `PatchMatchGpuRunTests`) measure:
- RNG: 4800 draws and 10,085 u32 -> f32 conversions bit-exact against `PatchMatchRandom`.
- `init_random`: depth within 1 ULP of the twin.
- `initial_cost`: within 1e-4 on 100% of entries (largest difference 3.77e-5).
- Full runs pass the same truth checks as the CPU runs (`PatchMatchRunTests`).
- The Tier C bounds of a GPU run against the CPU run: at least 95% of the pixels valid in both
  within 1% relative depth; median normal angle below 2 degrees; valid-pixel counts within 2%;
  consistency-graph membership agreement (Jaccard) at least 95%; and a repeated GPU run
  bit-identical to the first. Measured: 100% of pixels, 0 degrees and identical counts and
  graphs at 48x36, 40x30 and 26x19; at 320x240, 99.81% of pixels within the depth bound, a
  0.06% valid-count difference and 99.84% graph agreement.
