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

- Better behavior of ported code changes in place, with a divergence that
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
   - 0a is in: `Mvs/Testing/SyntheticObjectScene` (DarkObject, TexturedSphere, TexturelessBox;
     frames, one camera, object-fixed CamFromWorld, mesh, true masks).
   - 0b is in (`Mvs/Testing/Benchmark`, `ColmapSharp.Benchmarks`, `benchmarks/baseline.json`,
     ~30 min). Findings: the realistic sphere places 85% but self-calibration collapses the
     focal (ratio ~0.73, F ~0.01); with true intrinsics F is 0.46 and rotation error 0.38°.
     DarkObject and TexturelessBox place 0 of 40 at 480x360 (pycolmap too).
   - 0d (Lars's plan): 3D-printed reference models, so real captures have a true mesh.
     - Design a set that spans the hard cases:
       - the same shape in black matte, white and multicolour filament;
       - concavities and thin parts;
       - a smooth featureless blob;
       - a mouse-like body.
     - Score: align the reconstruction to the source mesh (Sim3, then ICP, since real captures
       have no true poses), then report accuracy, completeness and F-score as for synthetic
       scenes.
     - Check each print against its mesh at a few measured points (calipers), so print error
       (~0.1–0.2 mm, shrink) is known and the F-score τ sits above it.
     - Watch for layer lines: they add texture a real object may lack. Include a sanded or
       painted copy of one shape.
   - 0c: real captures (the mouse at f40/f80/f120, stored as a release asset with a checksummed
     manifest, not in git). A baseline matrix: Low / High / Extreme (affine + DSP + guided, all
     already ported), Individual vs Video, single camera, CLAHE. Results go in
     `docs/BENCHMARK.md`, and the demo's defaults may change from them.
1. **Silhouettes.**
   - 1a and 1a+ are in (`Segmentation/SilhouetteSegmenter`, `TemporalConsistency`; masks meet
     IoU ≥ 0.975 on the 0a DarkObject). Open: turn `TemporalWindow = 2` on for video once the
     benchmark confirms it; moment alignment is weak on thin side views (f120 007), so switch
     it to KLT flow after 2a; the smaller f120 096 notch remains; a close-up object filling the
     frame is reported as not found. GPU twin for stage 10: `BackgroundModel.Build`.
   - 1b: drop keypoints on specular highlights; CLAHE only inside the mask. Accept: the inlier
     ratio rises and the registered count doesn't fall.
2. **Video tracking.**
   - 2a is in (`Feature/Tracking`; gain/offset compensation; sphere reprojection 0.50 px). Open:
     on the mouse's top/side views (f120 038–058) masked tracks mostly end after 1–2 frames and
     mask erosion is not the main cause; count why tracks end before 2b builds on them.
   - 2b: keyframes chosen by parallax; tracked points get SIFT descriptors; matches go into the
     database, then verification and mapping.
   - 2c: sequential and loop matching over keyframes, shared bounded intrinsics, and full-rate
     small decoding in the demo.
   - Accept: > 90% of keyframes placed on the mouse with no split models; < 1° pose error on 0a.
3. **Silhouette-carved surface.**
   - 3a is in (`Mvs/Silhouette/VisualHull`; DarkObject F 0.934; mouse IoU 0.945 k=0 / 0.966 k=1).
     Open: default k from the benchmark; centroid fans make meshes ~1.8x larger.
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

## Robustness to unskilled captures (standing goal)

Users will film shaky, fast, unstructured videos they believe are good. The pipeline should
still do its best with them and tell the user plainly what went wrong. Capture tips and
training help, but they are not the fix.
- **Benchmark:** a "bad capture" category.
  - The real clips come from Lars's deliberately poor videos.
  - The synthetic ones come from 0a with degradations added: motion blur, auto-exposure
    swings, zoom and focus changes, rolling shutter, fast turns, standing still, pure
    rotation with no parallax, and clutter.
- **Frame selection:**
  - drop blurred and duplicate frames (a sharpness score);
  - choose keyframes by parallax rather than a fixed rate;
  - detect stretches of pure rotation with no parallax.
- **Photometric:** normalize exposure between frames before matching and texturing.
- **Camera:** handle intrinsics that change within a clip (zoom or focus breathing). Group
  frames by focal length instead of forcing one camera. Model rolling shutter later.
- **Feedback in the demo and MatterCAD:**
  - say which parts of the object were never seen;
  - say which parts of the clip were blurred or had no parallax;
  - eventually, live coverage guidance while capturing.

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
