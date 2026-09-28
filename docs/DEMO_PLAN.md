# ColmapSharp demo app — plan

Open work only; history lives in git. An AGG (agg-sharp) app that turns photos, a video or a
webcam capture into a downloadable mesh, running natively on macOS and as WASM on GitHub Pages
(`https://larsbrubaker.github.io/colmap-sharp/`). It is the end-to-end test bed beyond the unit
tests. Precedents: agg-sharp's `examples/AggSharpDemo` (three-project shape, Pages workflow) and
`rust-apps/colmap-rust` (same app idea in Rust, live on Pages).

## Shape (decided)

- Lives in this repo under `demo/`, with agg-sharp as a git submodule at `demo/agg-sharp`. The
  library (`ColmapSharp/`) never references agg-sharp; only the demo does.
- Three projects, copying AggSharpDemo: `demo/ColmapDemo` (shared app, no platform references),
  `demo/ColmapDemo.Mac` (Exe + PlatformMac), `demo/ColmapDemo.Browser` (Blazor WASM + PlatformBrowser,
  both agg `.targets` imports, `index.html` with `<base href="./">`).
- Generic platform capabilities go to agg-sharp (MatterCAD's litmus test: would a DemoRunner in the
  browser need it?): browser file drop, video frame extraction (browser `<video>`/canvas, Mac
  AVFoundation), webcam capture (browser `getUserMedia`; Mac later). Reconstruction-specific code
  stays in the demo or the library.
- GPU: the `IComputeDevice` adapter over agg-sharp's `WebGpuRenderDevice` (today in MatterCAD's
  `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`) moves into the demo's shared project; dense
  PatchMatch runs on the GPU in both heads, CPU fallback with the planner's reason shown.
- Browser runs .NET interpreted and single-threaded (AOT crashes on .NET 10.0.11), so the app
  defaults to small inputs (low max image size, few images) and the library's `RunAsync` yields
  between units of work so the page stays responsive.

## Phases (each step: tests green, reviewed, merged)

1. **Skeleton + Mac pipeline.** Submodule, three projects building; UI: drop/open images, image
   strip, Run, progress per stage, 3D viewport (agg `SceneDrawContext`, trackball) showing sparse
   points then the mesh, Download (PLY/OBJ). Library side is done (`Util/ObjWriter`, `RunAsync` yields between
   units). Demo: PLY → preview mesh bridge; when the texture atlas is empty write an
   untextured OBJ (the controller still names `texture.png`, as COLMAP does).
2. **Browser + Pages.** Browser head builds and links (`-p:LinkEmdawnWebGpu=true`), `.github/
   workflows/pages.yml` publishes `demo/ColmapDemo.Browser`, GPU compute device via `CreateAsync`
   (first real browser run of GPU PatchMatch — the C4 smoke check), download as a file (zip for
   mesh + texture), a Playwright/headless smoke check of the published page. Check that the
   default `Task.Yield` actually lets the page paint; if not, supply a requestAnimationFrame-
   backed `YieldAsync`, otherwise make that hook internal. Long single calls still hold the
   frame: extraction, matching, the sparse mapper (longest), undistortion, CPU PatchMatch per
   problem, fusion, meshing, texturing.
3. **Browser drop + video files.** agg-sharp: page-level drag-and-drop into `FileDropDispatcher`;
   video → frames (browser `<video>` seek + canvas; Mac AVFoundation `AVAssetImageGenerator`).
   Demo: frame sampling (every Nth / target count), `DataType.Video`.
4. **Webcam (browser).** agg-sharp: `getUserMedia` preview + still capture + timed capture.
   Demo: capture flow (take photos around an object, or record → frames).
5. **Distribution.** Mac `.app` bundle zip on GitHub Releases (unsigned; document Gatekeeper),
   README "run from source" steps.
6. **Mac webcam** (AVFoundation capture).

## Known risks

- Browser CPU stages (SIFT, matching, bundle adjustment) run 20–50× slower than desktop; long
  single steps still hold the frame even with yields between units.
- ImageSharp (agg-sharp's decoder) uses the Six Labors Split License — check before shipping
  binaries.
- Nesting agg-sharp inside colmap-sharp means a recursive MatterCAD clone pulls a second
  agg-sharp; MatterCAD should not recurse into `Submodules/colmap-sharp/demo/agg-sharp`.
