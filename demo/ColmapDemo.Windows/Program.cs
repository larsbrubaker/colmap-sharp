// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The Windows head's entry point: a window around ColmapDemoApp (demo/ColmapDemo), the twin of
// ColmapDemo.Mac/Program.cs and the shape of agg-sharp's examples/DemoRunner. agg's
// AGG_SMOKE_FRAMES / AGG_SMOKE_SCREENSHOT environment variables work here unchanged (the WinForms
// host reads them), so a smoke run can render a few frames, save a PNG and exit. DevAutoRun (in
// the shared project) adds COLMAP_DEMO_* variables that preload photos, run, and screenshot the result.

using System;
using ColmapDemo.Compute;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public static class ColmapDemoWindowsProgram
	{
		// STA because WinForms drag-drop is OLE: on an MTA main thread the host turns AllowDrop off
		// (agg-sharp's WinformsDragDrop) and dropped photos would never arrive.
		[STAThread]
		public static void Main(string[] args)
		{
			// No provider setup: AggContext's per-OS defaults pick the WinForms host on Windows, and this
			// project references PlatformWin32 so that host is in the output folder to be found.
			// Per-monitor DPI aware (V2, as MatterCAD's HostStartup.Windows.ConfigureHighDpi), before anything
			// reads AggContext.OsInformation or opens a window: Windows fixes the mode at the first window, and
			// an unaware process is told 96 DPI and shown a blurry bitmap-scaled window. The managed call
			// rather than MatterCAD's P/Invoke, since WinForms already wraps it.
			System.Windows.Forms.Application.SetHighDpiMode(System.Windows.Forms.HighDpiMode.PerMonitorV2);

			// Widgets at the display's scale (1.5 at 150%) and a window sized to match; see DemoDisplayScale.
			SystemWindow systemWindow = DemoDisplayScale.CreateWindow("ColmapSharp — photos to mesh");
			// PatchMatch runs on the GPU when wgpu gives us a device; otherwise on the CPU, and the
			// panel says why so a slow run is not a mystery.
			WebGpuComputeDevice gpu = null;
			string computeNote;
			try
			{
				gpu = WebGpuComputeDevice.Create(raiseComputeLimits: true);
				computeNote = "Depth maps run on the GPU.";
			}
			catch (Exception e)
			{
				computeNote = "No GPU (" + e.Message + "); depth maps run on the CPU.";
			}

			// The WinForms host delivers dropped files: a drag raises mouse moves and the drop a mouse up,
			// each carrying MouseEventArgs.DragFiles, the same contract as the AppKit host.
			var app = new ColmapDemoApp(fileDropSupported: true, gpu, computeNote);
			systemWindow.AddChild(app);
			DevAutoRun.Attach(systemWindow, app);
			systemWindow.ShowAsSystemWindow();
		}
	}
}
