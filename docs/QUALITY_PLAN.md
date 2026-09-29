# Reconstruction quality plan — beyond COLMAP

Open work only; history lives in git. Goal (CLAUDE.md): the best photo/video → mesh
reconstruction we can build, measured on a benchmark. The driving case is a phone video of a
black matte mouse spinning on its cable in front of a grey wall. It places 24 of 40 frames and
gives an underside-only shell. Measured so far: contrast enhancement (CLAHE on Lab L*, a lower
SIFT peak threshold) adds keypoints but not registered frames. Denser sampling does help
(40 → 24, 80 → 49, 120 → 74 + a separate 10-frame model). Unplaced frames come in contiguous
blocks (the top and side sweeps), and the split model shares no images with the main one, so
`MergeReconstructions` can't join them.

## Where code goes

- Better behavior of ported code changes in place, with a `docs/CPP_DIVERGENCES.md` entry that
  carries the benchmark numbers.
- New algorithms get their own folders: `Segmentation/`, `Feature/Tracking/`, `Sfm/Silhouette/`,
  `Mvs/Silhouette/`. Their headers say "Not a COLMAP port" and cite the paper, or the permissive
  source and its license (plus a row in `docs/LICENSE_AUDIT.md` and a `THIRD_PARTY_NOTICES.md`
  entry when code was read).
- Single-object assumptions (silhouettes, visual hull, turntable) sit behind
  `AutomaticReconstructionOptions.Subject = Scene | Object`. The demo defaults videos to `Object`.
- Reuse what is already there:
  - masks end to end (`MaskFeatures`, `WorkspaceBitmapSource`, `StereoFusionOptions.MaskPath`);
  - `MinSTGraphCut` (BK, written from the paper);
  - `TriangleBvh`;
  - `Alignment*` (Sim3, merge);
  - `VlSiftFilter.CalcKeypointDescriptor`;
  - `Database` writes;
  - the controller's injected `Database`;
  - the `IComputeDevice` seam.

## Stages (one implementer deliverable each; "accept" is the benchmark bar)

0. **Benchmark harness.**
   - 0a: a synthetic object-scene generator in `Mvs/Testing/SyntheticObjectScene.cs`, lifted from
     the ray tracer in `AutomaticReconstructionTests.CSharpOnly.cs`. Scenes: a dark, near-textureless
     superellipsoid "mouse" with a label patch and seam grooves on a plain grey wall, turning about a
     vertical axis on a twisting-pendulum θ(t) with a slightly jittered static camera; plus a
     textured sphere and a textureless box. It outputs the images, true cameras, mesh and masks,
     deterministically.
   - 0b: metrics and a runner (`ColmapSharp.Benchmarks` console project plus a fast test):
     - registered fraction;
     - pose error after Sim3;
     - Chamfer, and F-score at τ (Tanks and Temples);
     - silhouette IoU as the proxy for real captures;
     - runtime per stage.
     Results are written as JSON, with `benchmarks/baseline.json` checked in and a regression
     tolerance. Report each metric over several mapper seeds (mean and worst): on the CLAHE
     mouse frames both our mapper and pycolmap split into two models on roughly 1 in 10–20
     seeds, so a single seed can mislead.
   - 0c: real captures (the mouse at f40/f80/f120, stored as a release asset with a checksummed
     manifest, not in git). A baseline matrix: Low / High / Extreme (affine + DSP + guided, all
     already ported), Individual vs Video, single camera, CLAHE. Results go in
     `docs/BENCHMARK.md`, and the demo's defaults may change from them.
1. **Silhouettes.**
   - 1a: automatic segmentation in `Segmentation/`:
     - background model: the per-pixel temporal median, with an Otsu threshold on L* as the
       fallback;
     - a trimap;
     - a GrabCut-style GMM plus Potts refinement via `MinSTGraphCut`;
     - opening sized to the object's width, to drop the cable;
     - the largest component, with holes filled.
     It outputs an `IImageSource` of masks. Accept: IoU ≥ 0.97 against the true masks.
   - 1b: drop keypoints on specular highlights; CLAHE only inside the mask. Accept: the inlier
     ratio rises and the registered count doesn't fall.
2. **Video tracking.**
   - 2a: pyramidal KLT in `Feature/Tracking/` (Lucas–Kanade, Shi–Tomasi, Bouguet; OpenCV
     `lkpyramid.cpp` Apache-2.0 as reading), with a forward–backward check and replenishment
     inside the mask.
   - 2b: keyframes chosen by parallax; tracked points get SIFT descriptors; matches go into the
     database, then verification and mapping.
   - 2c: sequential and loop matching over keyframes, shared bounded intrinsics, and full-rate
     small decoding in the demo.
   - Accept: > 90% of keyframes placed on the mouse with no split models; < 1° pose error on 0a.
3. **Silhouette-carved surface.**
   - 3a: an octree visual hull (k-disagreement tolerant) with marching cubes.
   - 3b: hull samples fill the gaps in the fused cloud before Poisson; faces projecting outside
     the silhouettes are removed, which replaces trim guessing.
   - 3c: per-pixel hull depth bounds for PatchMatch, CPU and WGSL.
   - Accept: silhouette IoU > 0.95 and a closed mesh on the mouse; completeness F-score +30 on 0a.
4. **Silhouette pose for the frames that are still unplaced.**
   - 4a: initialize by SLERP between neighbours in time.
   - 4b: silhouette-coherence refinement (Hernández, Schmitt and Cipolla 2007), then global BA.
   - 4c: join split models via shared tracks or silhouette-consistent Sim3.
   - Accept: 100% of the mouse frames placed.
5. **Turntable prior.** Fit the rotation axis from the registered poses; use it for 4a and as a
   soft BA pose prior.
6. **Graph-cut meshing with silhouette free space** (Labatut 2009, Jancosek–Pajdla 2011 from the
   paper only; OpenMVS is AGPL). Keep it only if it beats 3b on concavities.
7. **Dense on matte surfaces:** ACMM multi-scale geometric consistency and ACMP planar priors
   (both MIT), mirrored in the WGSL kernels.
8. **Features, benchmark-gated:** AKAZE (BSD-3), then XFeat (Apache-2.0 code and weights) on a
   managed conv runtime plus WGSL; ALIKED + LightGlue only if XFeat wins.
9. **Edges and lines:** EDLines (MIT) + LBD (Apache-2.0), for boxes and CAD parts.
10. **GPU kernels** through `IComputeDevice`: carving, KLT, the background median. Each has a CPU
    twin, following the `ReferenceComputeDevice` pattern.

## Licenses

- **Excluded:**
  - LSD (AGPL);
  - SuperPoint / SuperGlue (Magic Leap non-commercial);
  - DUSt3R / MASt3R (CC BY-NC-SA);
  - OpenMVS (AGPL);
  - Kolmogorov's BK code (research-only; ours is written from the paper).
- **To verify before use:** KAZE's original code, ACMMP, and the weights for MobileSAM,
  EfficientSAM and U²-Net.
- Everything else named above is BSD, MIT or Apache-2.0.
