# Divergences from COLMAP: Feature: SIFT and matching

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 40. ExtractTopScaleFeatures keeps equal-scale keypoints in input order

**What differs.** COLMAP's `ExtractTopScaleFeatures` (feature/utils.cc) selects the largest
scales with `std::partial_sort`, which leaves keypoints of equal scale in an unspecified,
implementation-defined order (libc++'s heap-based selection). The port
(`Feature/FeatureUtils.cs`) sorts by scale, largest first, with the input index as the
tie-break.

**Why.** Equal scales are common: the SIFT extractor emits one keypoint per orientation, and
all of them share the scale. Reproducing libc++'s heap order exactly would tie the result to
one standard library; the index tie-break is deterministic and stable. Only the order among
equal scales differs, and *which* keypoints are kept differs only when a tie straddles the
cut, where COLMAP's choice is arbitrary too.

**Evidence.** `FeatureUtilsTests.ExtractTopScaleFeatures_Nominal` (utils_test.cc 1:1, distinct
scales) passes unchanged.

## 41. The macOS arm64 wheel's VLFeat fuses multiply-adds; the SIFT port does not

**What differs.** COLMAP compiles VLFeat's C with the platform compiler's default
floating-point contraction. On macOS arm64, Apple clang's default is `-ffp-contract=on`, so
expressions such as the Gaussian convolution's `acc += v * c` (imopv.c), the descriptor's
bin weights and the fast atan2/resqrt polynomials become fused multiply-adds that round once.
The port (`Feature/VLFeat/`) evaluates every `a*b + c` as a rounded product plus a rounded
sum (CLAUDE.md "No FMA"), which is what VLFeat computes with contraction off and on any
target without FMA contraction. The scale space therefore differs from the wheel's in the
last bits, and those differences propagate to keypoint positions, scales and orientations
(about 1e-3 at most on the fixtures) and, rarely, to a descriptor byte (by one gray level).
A detection threshold could in principle flip on some image, adding or dropping a keypoint.

**Why.** Emulating contraction would mean choosing, per expression, whatever the C compiler
chose - which is compiler- and flag-specific (x86 builds, Linux wheels and older compilers
do not fuse) - and `Math.FusedMultiplyAdd` is banned in math paths so results are identical
on every platform. VLFeat's source defines the algorithm; the unfused evaluation is its
literal semantics.

**Evidence.** `oracle/sift_harness.c` drives cpp-reference's VLFeat the way COLMAP does. Built
with `-ffp-contract=off` (and without SSE2, like COLMAP's arm64 build), its keypoints,
orientations and raw float descriptors match the port bit for bit on all six fixture cases
(`VlSiftFilterTests.CSharpOnly_MatchesUnfusedVLFeatExactly`: upsampling by one and two
octaves, direct copy, downsampling, upright). Built with the default flags, the same harness
prints the first keypoint of the sift_test.cc square image at x = 98.7666931, exactly the
wheel's value, while the unfused build prints 98.7667007, exactly the port's. Against the
wheel (`SiftOracleTests.CSharpOnly_MatchesPycolmapWithinFmaTolerance`, nine option sets) the
keypoint counts and order agree exactly, values within 1.03e-3, descriptor bytes within 1 on
at most 0.05% of the bytes.

The covariant extractor (`Feature/CovariantSift.cs`, VLFeat covdet) has the same split:
`sift_harness.c`'s covdet mode, unfused, matches `VlCovDetTests` bit for bit on eight cases
(every feature frame and score, every raw descriptor of every DSP scale), while against the
wheel (`CovariantSiftOracleTests`) the frames agree within 4e-4. Fusing also reaches two
discontinuities there: on the symmetric square two orientations of one feature score equally,
so which comes first flips; and on the texture with domain-size pooling one keypoint (of 184)
gets a descriptor up to 4 gray levels off, because the patch extractor picks its scale-space
level with a floor of a log2 that lands on the other side of an integer for one pooled scale.
`CSharpOnly_KnownDivergences` pins that case.

## 42. FeatureDescriptorIndex is an exact nearest-neighbor search, not faiss's IVF index

**What differs.** COLMAP's `FeatureDescriptorIndex` (feature/index.cc) wraps faiss: an exact
`IndexFlatL2` for fewer than 512 indexed descriptors, and for 512 or more an inverted-file
index (`IndexIVFScalarQuantizer` with `QT_8bit_direct` for SIFT, `IndexIVFFlat` otherwise;
4 sqrt(N) k-means centroids, searched with `nprobe = 8`). The IVF search is approximate: it
only visits the 8 nearest clusters, so it can miss a query's true nearest or second-nearest
neighbour. The port (`Feature/FeatureDescriptorIndex.cs`) always returns the exact k nearest
neighbours, ordered by (squared distance, index). SIFT descriptors are compared as integers
(exact); other descriptor types sum (a - b)^2 in float, left to right, where faiss's flat
index uses the ||a||^2 + ||b||^2 - 2ab expansion.

**Why.** faiss is native code (CLAUDE.md contract 1). Porting it, k-means training included,
would reproduce a randomized approximation whose exact output also depends on faiss's
OpenMP partitioning and BLAS. The exact search is the result the IVF index approximates, and
it makes the default (index) matching path agree with COLMAP's own brute-force matcher, which
is what `SiftCPUFeatureMatcherFaissVsBruteForce` asserts. On an image pair with 512+
features, the SIFT matches can therefore differ from COLMAP's default CPU matcher wherever
faiss's approximation missed a neighbour (COLMAP returns fewer or different matches there);
they equal COLMAP's `cpu_brute_force_matcher` matches except where acos-of-dot and sqrt-of-L2
round a threshold comparison differently, which COLMAP's two paths also do. Ties in distance
are ordered by index; faiss leaves them unspecified, and the ratio test rejects a tie for the
best neighbour either way.

**Evidence.** `FeatureDescriptorIndexTests` (index_test.cc 1:1, including 1000 descriptors,
where COLMAP uses IVF) and `SiftMatcherTests.SiftCPUFeatureMatcherFaissVsBruteForce_Nominal`
(sift_test.cc 1:1) pass. Speed: an 8192 x 8192 SIFT pair with cross-check takes about 114 ms
through the index and 51 ms brute force on a 10-core Apple M-series machine (Release build).

## 43. SIFT never reuses a previous image's gradient, and extraction can be cancelled

**What differs.** COLMAP's SiftCPUFeatureExtractor keeps one VLFeat filter while the image
size stays the same, and VLFeat resets the octave its gradient cache belongs to (`grad_o`)
only in `vl_sift_new`, not in `vl_sift_process_first_octave`. When the previous image's last
octave with keypoints is the new image's first octave, the new image's orientations and
descriptors in that octave are computed from the previous image's gradient. That always
happens with `num_octaves = 1`, and with the default options whenever the previous image's
keypoints all sit in the finest octave (a low-contrast, fine-textured photo). The port
(`Feature/VLFeat/VlSiftFilter.cs`) resets the cache at the start of every image, so a reused
extractor gives exactly what a fresh one gives.

Separately, `FeatureExtractor.Extract` takes a `CancellationToken` (COLMAP's has none). The SIFT
extractor checks it before building the scale space, before every Gaussian level and DoG
extremum scan of every octave, and before each DoG level's orientations and descriptors. A
cancel throws `OperationCanceledException` and leaves the outputs untouched. The extractor
then drops its filter, so the next image is extracted exactly as by a fresh extractor.
Uncancelled results are unchanged.

**Why.** The stale gradient is an upstream bug: it makes COLMAP's features for an image depend
on which image the same worker extracted before it, so on image order and on how images are
spread over extraction threads. MatterCAD needs results that don't depend on either.
Cancellation is needed because one 12 MP extraction takes seconds and the user must be able
to stop a reconstruction (CLAUDE.md "Cancellation and progress").

**Evidence.** `oracle/fixture_sift_reuse.py` writes `sift_reuse.json`: pycolmap's features of
a texture image B from a fresh extractor and from one reused after a fine-noise image A of the
same size whose 23 keypoints are all in octave -1. The reused extractor gives 155 keypoints
where the fresh one gives 154 (default options), and 30 where it gives 29 (`num_octaves = 1`).
`SiftExtractorReuseTests` (C#-only) requires the port's reused extractor to equal its fresh
one bit for bit, and to match pycolmap's fresh output within the FMA tolerance of entry 41;
before the reset the port reproduced pycolmap's 155 and 30. `SiftCancellationTests` (C#-only):
a pre-cancelled token throws at the first check before any allocation; a cancel at a check in
the middle of an image throws at that check; runs with a live token, with no token, and on the
same extractor after a cancel give bit-identical features.

## 57. The covariant SIFT extractor orders equal (octave, level) features stably

**What differs.** COLMAP's `CovariantSiftCPUFeatureExtractor` (feature/sift.cc) sorts the
covdet features by (octave, level), both descending, with `std::sort`, which leaves features
with equal keys in an implementation-defined order (libc++'s introsort; it insertion-sorts,
stably, only short ranges). The port (`Feature/CovariantSift.cs`) sorts with the covdet index
as the tie-break, so equal keys keep detection order. The keypoints and descriptors are the
same set, but their order within an (octave, level) group can differ once there are more
features than libc++ insertion-sorts. It also changes one keypoint when `max_num_features`
truncates: COLMAP pushes the first keypoint of the next group before it stops, and which
keypoint is first there is the tie order.

**Why.** Reproducing libc++'s `std::sort` exactly would tie the result to one standard
library version (its thresholds and partitioning changed between LLVM releases); the stable
order is deterministic, as for entry 40.

**Evidence.** `CovariantSiftOracleTests`: on the texture fixtures (184 and 191 features)
every keypoint pairs with a pycolmap keypoint within the FMA drift of entry 41, but not index
by index. With `max_num_features = 20` both give 26 keypoints; the first 25 pair up and the
26th differs (`CSharpOnly_KnownDivergences`). The five covariant rows of
`SiftTests.SiftCpuExtraction_Nominal` (sift_test.cc 1:1) pass.
