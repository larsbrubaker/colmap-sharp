// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The Windows head's entry point: a window around ColmapDemoApp (demo/ColmapDemo), the twin of
// ColmapDemo.Mac/Program.cs and the shape of agg-sharp's examples/DemoRunner. agg's
// AGG_SMOKE_FRAMES / AGG_SMOKE_SCREENSHOT environment variables work here unchanged (the WinForms
// host reads them), so a smoke run can render a few frames, save a PNG and exit. DevAutoRun (in
// the shared project) adds COLMAP_DEMO_* variables that preload photos, run, and screenshot the result.
// The project is WinExe (no console window of its own), so Main first reconnects Console to the
// terminal it was started from, if any, so those runs still print their results.

using System;
using System.IO;
using System.Runtime.InteropServices;
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
			AttachParentConsole();

			// No provider setup: AggContext's per-OS defaults pick the WinForms host on Windows, and this
			// project references PlatformWin32 so that host is in the output folder to be found.
			// No DPI-awareness call either: the WinForms host opts into per-monitor V2 itself before its
			// information provider reads the DPI (agg's WindowsDpiAwareness).
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

			// Dragged to a monitor with another scale, the app is rebuilt there (between runs).
			DemoDisplayScale.Follow(systemWindow, app);
			DevAutoRun.Attach(systemWindow, app);
			systemWindow.ShowAsSystemWindow();
		}

		private const int AttachParentProcess = -1;
		private const int StdOutputHandle = -11;

		[DllImport("kernel32.dll", SetLastError = true)]
		private static extern bool AttachConsole(int processId);

		[DllImport("kernel32.dll", SetLastError = true)]
		private static extern IntPtr GetStdHandle(int stdHandle);

		/// <summary>
		/// Routes Console output to the caller's terminal, if there is one. A WinExe gets no console,
		/// but the dev runs (COLMAP_DEMO_*, AGG_SMOKE_*) report through Console. Must run before
		/// anything touches Console, which caches its writers on first use.
		/// </summary>
		private static void AttachParentConsole()
		{
			// Output redirected to a file or pipe (e.g. `> log 2>&1` in Git Bash) is already inherited
			// as a valid handle; Console writes there on its own, and attaching would steal it.
			IntPtr stdout = GetStdHandle(StdOutputHandle);
			if (stdout != IntPtr.Zero && stdout != new IntPtr(-1))
			{
				return;
			}

			// Launched from Visual Studio or Explorer there is no parent console: attach fails and the
			// app stays console-free, which is the point of WinExe.
			if (!AttachConsole(AttachParentProcess))
			{
				return;
			}

			// The process's standard handles were captured at startup (empty), so bind Console to the
			// newly attached console explicitly. The shell has already printed its prompt (cmd and
			// PowerShell don't wait for a GUI exe), so the output interleaves with it; Git Bash waits.
			var consoleOut = new StreamWriter(new FileStream("CONOUT$", FileMode.Open, FileAccess.Write)) { AutoFlush = true };
			Console.SetOut(consoleOut);
			Console.SetError(consoleOut);
		}
	}
}
