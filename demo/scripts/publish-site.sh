#!/usr/bin/env bash
#
# Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
#
# Publishes the demo's browser head (demo/ColmapDemo.Browser) exactly as the GitHub Pages workflow
# (.github/workflows/pages.yml) does - the workflow calls this script, so a local run and CI build the
# same thing. Prints the folder to serve as its last line. The shape of agg-sharp's
# scripts/publish-demo-site.sh.
#
# Needs the wasm-tools workload (`dotnet workload install wasm-tools`): LinkEmdawnWebGpu=true is the
# emcc relink that makes the page paint. The first link seeds a ~215 MB Emscripten cache under
# demo/agg-sharp/WebGpu/Browser/emscripten-cache (see WebGpu/build/WebGpuBrowser.targets); later ones
# reuse it.
#
# Serve the output from any static server, at a domain root or under a subpath - index.html uses a
# relative <base href>, so /colmap-sharp/ on GitHub Pages works the same as / locally.
# demo/scripts/check-site.py serves it under /colmap-sharp/ and checks it in headless Chrome.

set -euo pipefail

demo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="$demo/ColmapDemo.Browser/ColmapDemo.Browser.csproj"
site="$demo/ColmapDemo.Browser/bin/Release/net10.0/publish/wwwroot"

# Release, because it runs the trimmer: the site is a download.
dotnet publish "$project" -c Release -p:LinkEmdawnWebGpu=true >&2

# Jekyll drops every path that starts with an underscore - Blazor's whole _framework folder. The
# Actions deployment does not run Jekyll, but a branch-based Pages source would.
touch "$site/.nojekyll"

echo "$site"
