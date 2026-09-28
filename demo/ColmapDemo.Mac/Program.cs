// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The mac head's entry point: a window around ColmapDemoApp (demo/ColmapDemo), the twin of
// ColmapDemo.Browser/Program.cs and the shape of agg-sharp's AggSharpDemo.Mac/Program.cs.
// agg's AGG_SMOKE_FRAMES / AGG_SMOKE_SCREENSHOT environment variables work here unchanged
// (the AppKit host reads them), so a smoke run can render a few frames, save a PNG and exit.

using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public static class ColmapDemoMacProgram
	{
		public static void Main(string[] args)
		{
			// No provider setup: AggContext's per-OS defaults pick the AppKit host on a Mac, and this
			// project references PlatformMac so that host is in the output folder to be found.
			var systemWindow = new SystemWindow(1200, 800)
			{
				Title = "ColmapSharp — photos to mesh",
			};
			// The AppKit host delivers dropped files (agg's FileDropDispatcher).
			systemWindow.AddChild(new ColmapDemoApp(fileDropSupported: true));
			systemWindow.ShowAsSystemWindow();
		}
	}
}
