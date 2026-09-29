// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DemoDisplayScale: how big the demo's widgets are drawn, and the window they open in. Every
// head (ColmapDemo.Mac, ColmapDemo.Windows, ColmapDemo.Browser) calls CreateWindow before it
// builds ColmapDemoApp, so the panel, its fonts and its margins are laid out at the display's
// scale from the first frame, and Follow once the app is in the window, so a move to a display
// with another scale rebuilds it there (ColmapDemoApp.Rescale.cs). The policy itself - scale =
// display scale, a design-size window clamped to the desktop, one rebuild per real change - is
// agg's UiScale, shared with MatterCAD and the AggSharpDemo; the demo has no text-size multiplier.

using System;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	/// <summary>
	/// Sets <see cref="GuiWidget.DeviceScale"/> from the display the demo opens on, sizes its window to
	/// match, and keeps both right as the window moves between displays.
	/// </summary>
	/// <remarks>
	/// agg lays widgets out in device pixels and the platform hosts only report the display's scale;
	/// without this a Retina Mac gets one device pixel per design unit: a 1200 x 800 pixel window that
	/// is 600 x 400 points on screen, with every font and button at half its physical size.
	/// </remarks>
	public static class DemoDisplayScale
	{
		/// <summary>The window's width at the design scale, in points (design units).</summary>
		public const int DesignWindowWidth = 1200;

		/// <summary>The window's height at the design scale, in points (design units).</summary>
		public const int DesignWindowHeight = 800;

		/// <summary>
		/// Sets <see cref="GuiWidget.DeviceScale"/> for the display the OS reports (the Mac main screen's
		/// backing scale, the Windows monitor DPI over 96, the browser's devicePixelRatio) and returns a
		/// window sized for it. Call before building <see cref="ColmapDemoApp"/>: widget sizes and fonts
		/// are fixed when a widget is built.
		/// </summary>
		public static SystemWindow CreateWindow(string title)
		{
			UiScale.ApplyAtStartup();
			(int width, int height) = UiScale.ScaledWindowSize(DesignWindowWidth, DesignWindowHeight);
			Console.WriteLine($"COLMAP_DEMO display scale {GuiWidget.DeviceScale}, window {width} x {height} px");
			return new SystemWindow(width, height) { Title = title };
		}

		/// <summary>
		/// Rebuilds <paramref name="app"/> at the new scale when <paramref name="window"/> moves to a display
		/// with another scale. A move during a run or while a video is being read waits for it to end:
		/// their progress reports write into the widgets a rebuild would replace.
		/// </summary>
		/// <returns>Disposing it stops following; closing the window does too.</returns>
		public static IDisposable Follow(SystemWindow window, ColmapDemoApp app)
		{
			return UiScale.Follow(
				window,
				() =>
				{
					app.RebuildUi();
					Console.WriteLine($"COLMAP_DEMO display scale now {GuiWidget.DeviceScale}");
				},
				canRebuildNow: () => !app.IsRunning && !app.IsReadingVideo);
		}
	}
}
