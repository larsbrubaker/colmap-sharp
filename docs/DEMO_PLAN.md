# ColmapSharp demo app — plan

Open work only; history lives in git. An AGG (agg-sharp) app that turns photos, a video or a
webcam capture into a downloadable mesh, running natively on macOS and Windows and as WASM on GitHub Pages
(`https://larsbrubaker.github.io/colmap-sharp/`). It is the end-to-end test bed beyond the unit
tests. Precedents: agg-sharp's `examples/AggSharpDemo` (three-project shape, Pages workflow) and
`rust-apps/colmap-rust` (same app idea in Rust, live on Pages).

## Shape (decided)

- Lives in this repo under `demo/`, with agg-sharp as a git submodule at `demo/agg-sharp`. The
  library (`ColmapSharp/`) never references agg-sharp; only the demo does.
- Projects copying AggSharpDemo's shape: `demo/ColmapDemo` (shared app, no platform references),
  `demo/ColmapDemo.Mac` (Exe + PlatformMac), `demo/ColmapDemo.Windows` (Exe + PlatformWin32,
  `net10.0-windows`), `demo/ColmapDemo.Browser` (Blazor WASM + PlatformBrowser,
  both agg `.targets` imports, `index.html` with `<base href="./">`).
- Generic platform capabilities go to agg-sharp (MatterCAD's litmus test: would a DemoRunner in the
  browser need it?): browser file drop, video frame extraction (browser `<video>`/canvas, Mac
  AVFoundation), webcam capture (browser `getUserMedia`; Mac later). Reconstruction-specific code
  stays in the demo or the library.
- GPU: the `IComputeDevice` adapter over agg-sharp's `WebGpuRenderDevice` (today in MatterCAD's
  `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`) moves into the demo's shared project; dense
  PatchMatch runs on the GPU in every head, CPU fallback with the planner's reason shown.
- Browser runs .NET interpreted and single-threaded (AOT crashes on .NET 10.0.11), so the app
  defaults to small inputs (low max image size, few images) and the library's `RunAsync` yields
  between units of work so the page stays responsive.

## Phases (each step: tests green, reviewed, merged)

1. **Mac polish.** Phase 1 is in (Mac window runs photos → textured mesh, GPU PatchMatch, Save
   OBJ/PLY with `<name>.png`). Not yet exercised in a live window: Cancel, Save, trackball.
2. **Browser + Pages.** Browser head builds and links (`-p:LinkEmdawnWebGpu=true`), `.github/
   workflows/pages.yml` publishes `demo/ColmapDemo.Browser`, GPU compute device via `CreateAsync`
   (first real browser run of GPU PatchMatch — the C4 smoke check), download as a file (zip for
   mesh + texture), a Playwright/headless smoke check of the published page. Check that the
   default `Task.Yield` actually lets the page paint; if not, supply a requestAnimationFrame-
   backed `YieldAsync`, otherwise make that hook internal. Long single calls still hold the
   frame: extraction, matching, the sparse mapper (longest), undistortion, CPU PatchMatch per
   problem, fusion, meshing, texturing.
3. **Browser video + Windows decode speed.** agg-sharp: an `IVideoFrameReader` for the browser
   (`<video>` seek + canvas, or WebCodecs); until then a video dropped or picked in the browser
   gets a note pointing at photos or the desktop app (`ColmapDemoApp.BrowserNoVideoNote`), and a
   reader landing there also needs that note dropped from the browser head. When it lands, a
   window close mid-read must still release the staged video (`BrowserFileStaging.Release`),
   since `ReceiveDropAsync`'s release never runs once the closed window's pump stops. Browser drop is in
   but has only been exercised by unit tests: check a real drop of photos and of a video on the
   published page (by hand or Playwright). Windows: hardware decode (D3D11 device manager) is the
   remaining read-speed lever. A phone clip of a black matte mouse
   spinning on its cable placed 24 of 40 frames, the same frames pycolmap places: a capture limit
   (only the textured underside matches), not a port bug. Denser sampling adds frames of the sides
   already seen, not the unseen ones; the demo's capture tips cover what does help.
4. **Webcam (browser).** agg-sharp: `getUserMedia` preview + still capture + timed capture.
   Demo: capture flow (take photos around an object, or record → frames).
5. **Distribution.** Mac `.app` bundle zip on GitHub Releases (unsigned; document Gatekeeper),
   README "run from source" steps.
6. **Mac webcam** (AVFoundation capture).

## Known risks

- Browser CPU stages (SIFT, matching, bundle adjustment) run 20–50× slower than desktop; long
  single steps still hold the frame even with yields between units.
- Nesting agg-sharp inside colmap-sharp means a recursive MatterCAD clone pulls a second
  agg-sharp; MatterCAD should not recurse into `Submodules/colmap-sharp/demo/agg-sharp`.
