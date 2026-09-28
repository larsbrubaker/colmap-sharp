#!/usr/bin/env python3
#
# Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
"""Proves the published demo site paints and, with --autorun, runs the whole pipeline in headless Chrome.

  demo/scripts/publish-site.sh                          # build the site the Pages workflow deploys
  demo/scripts/check-site.py --out /tmp/site.png        # serve it at /colmap-sharp/, wait for paint, screenshot
  demo/scripts/check-site.py --out /tmp/run.png --autorun               # ...then run the six bundled samples
  demo/scripts/check-site.py --out /tmp/run.png --autorun --task-yield  # the same, yielding with Task.Yield
  demo/scripts/check-site.py --out /tmp/run.png --autorun --cpu         # the same, PatchMatch on the CPU

The Chrome, CDP, static server and paint-wait plumbing is agg-sharp's (demo/agg-sharp/scripts/
check-demo-site.py and run-browser-goldens.py), imported rather than copied; only the export it asks is
this head's (ColmapDemoBrowserProgram.PaintState). --autorun opens the page with ?demo=autorun
(ColmapDemo.Browser/BrowserDevHook.cs) and polls RunState until the run ends. Each poll is a script
evaluation on the page's one thread, so the gap between two answered polls is how long the page was
unresponsive; the paint count across the run says whether it kept drawing.
"""

import argparse
import base64
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import time

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "agg_check_demo_site", os.path.join(_here, "..", "agg-sharp", "scripts", "check-demo-site.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)
goldens = agg.goldens


def export_script(method):
    return ("(async () => {"
            " if (typeof getDotnetRuntime !== 'function') return 'loading';"
            " const runtime = getDotnetRuntime(0);"
            " if (!runtime) return 'loading';"
            " const app = await runtime.getAssemblyExports('ColmapDemo.Browser.dll');"
            f" return app.ColmapDemo.ColmapDemoBrowserProgram.{method}();"
            "})()")


# agg's await_paint asks the page through this module-level script; point it at this head's export.
agg.PAINT_STATE_SCRIPT = export_script("PaintState")
RUN_STATE_SCRIPT = export_script("RunState")

RUN_TIMEOUT_SECONDS = 1800


def screenshot(page, path):
    png = base64.b64decode(page.call("Page.captureScreenshot", {"format": "png"})["result"]["data"])
    with open(path, "wb") as handle:
        handle.write(png)
    return png


def await_run(page):
    """Polls RunState until the run ends. Returns (final state, longest gap between answers, paints before, after)."""
    # agg's devtools socket gives up on a read after 30 s; a stage can hold the page longer than that.
    page.sock.settimeout(RUN_TIMEOUT_SECONDS)
    started = time.time()
    paints_before = int(re.search(r"paints (\d+)", page.evaluate(agg.PAINT_STATE_SCRIPT, await_promise=True)).group(1))
    last_answer = time.time()
    longest_gap = 0.0
    seen_running = False
    state = ""
    while time.time() - started < RUN_TIMEOUT_SECONDS:
        # A long single step of the pipeline holds the thread, and this evaluation waits for it.
        state = page.evaluate(RUN_STATE_SCRIPT, await_promise=True, timeout=RUN_TIMEOUT_SECONDS) or ""
        now = time.time()
        longest_gap = max(longest_gap, now - last_answer)
        last_answer = now
        seen_running = seen_running or state.startswith("running")
        if seen_running and state.startswith("idle"):
            break
        errors = agg.page_errors(page)
        if errors:
            raise RuntimeError("the page failed:\n  " + "\n  ".join(errors))
        time.sleep(0.2)
    else:
        raise TimeoutError(f"the run did not end in {RUN_TIMEOUT_SECONDS}s. Last state: {state!r}")
    paints_after = int(re.search(r"paints (\d+)", page.evaluate(agg.PAINT_STATE_SCRIPT, await_promise=True)).group(1))
    return state, longest_gap, paints_before, paints_after, time.time() - started


def run(arguments):
    site = os.path.abspath(arguments.site or os.path.join(
        _here, "..", "ColmapDemo.Browser", "bin", "Release", "net10.0", "publish", "wwwroot"))
    if not os.path.exists(os.path.join(site, "index.html")):
        raise RuntimeError(f"no published site at '{site}' - run demo/scripts/publish-site.sh first")

    subpath = arguments.subpath.strip("/")
    serve_root = site
    if subpath:
        serve_root = tempfile.mkdtemp(prefix="colmap-demo-site-")
        os.symlink(site, os.path.join(serve_root, subpath))

    server = goldens.StaticServer(serve_root)
    chrome = goldens.Chrome(os.path.join(tempfile.gettempdir(), "colmap-demo-site-chrome.log"))
    page = None
    url = server.url + (subpath + "/" if subpath else "")
    if arguments.autorun:
        url += "?demo=autorun" + ("&yield=task" if arguments.task_yield else "") + ("&gpu=off" if arguments.cpu else "")
    try:
        page = chrome.open_page()
        page.call("Emulation.setDeviceMetricsOverride", {
            "width": agg.PAGE_WIDTH, "height": agg.PAGE_HEIGHT, "deviceScaleFactor": 1, "mobile": False})
        page.call("Page.navigate", {"url": url}, timeout=60)
        try:
            state, painted_seconds = agg.await_paint(page)
            print(f"painted ({state}) in {painted_seconds:.1f}s")
            if arguments.autorun:
                final, gap, before, after, seconds = await_run(page)
                print(f"run ended in {seconds:.1f}s: {final}")
                print(f"  paints during the run: {after - before}; longest unresponsive stretch: {gap:.1f}s")
                if "Done:" not in final:
                    raise RuntimeError("the run did not produce a mesh")
        except Exception as failure:
            screenshot(page, arguments.out)
            print(f"FAIL: {url}: {failure}\n  Screenshot: {arguments.out}")
            return 1

        page.evaluate(agg.TWO_FRAMES_SCRIPT, await_promise=True)
        screenshot(page, arguments.out)
        errors = agg.page_errors(page)
        if errors:
            print("FAIL:" + "".join("\n  " + e for e in errors))
            return 1
        print(f"PASS: {url}. Screenshot: {arguments.out}")
        return 0
    finally:
        if page is not None:
            for line in page.console_lines():
                if "COLMAP_DEMO" in line or "rror" in line:
                    print("  page: " + line.split("\n")[0][:200])
        chrome.close()
        server.close()
        if serve_root != site:
            shutil.rmtree(serve_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="where to write the screenshot PNG")
    parser.add_argument("--site", help="the published wwwroot (default: publish-site.sh's output)")
    parser.add_argument("--subpath", default="colmap-sharp",
                        help="serve the site under this path, as GitHub Pages does (default colmap-sharp; '' for the root)")
    parser.add_argument("--autorun", action="store_true", help="run the pipeline on the bundled sample photos")
    parser.add_argument("--task-yield", action="store_true", help="with --autorun: yield with Task.Yield, not setTimeout")
    parser.add_argument("--cpu", action="store_true", help="with --autorun: no GPU device, PatchMatch on the CPU")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
