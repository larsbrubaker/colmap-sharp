// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DemoDisplayScale: how big the demo's widgets are drawn, and the window they open in. Every
// head (ColmapDemo.Mac, ColmapDemo.Windows, ColmapDemo.Browser) calls CreateWindow before it
// builds ColmapDemoApp, so the panel, its fonts and its margins are laid out at the display's
// scale from the first frame. The policy is MatterCAD's (MatterCADLib/ApplicationView/
// UserInterfaceScaling.cs) without the user text-size multiplier, and agg-sharp's
// AggSharpDemoApp(followDisplayScale: true) does the same at startup.

using System;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	/// <summary>
	/// Sets <see cref="GuiWidget.DeviceScale"/> from the display the demo opens on and sizes its window
	/// to match.
	/// </summary>
	/// <remarks>
	/// <para>
	/// agg lays widgets out in device pixels and the platform hosts only report the display's scale
	/// (<c>OsInformation.DisplayScale</c>, <see cref="SystemWindow.DisplayScale"/>); turning that into
	/// <see cref="GuiWidget.DeviceScale"/> is the application's call. Without it a Retina Mac gets one
	/// device pixel per design unit: a 1200 x 800 pixel window that is 600 x 400 points on screen, with
	/// every font and button at half its physical size.
	/// </para>
	/// <para>
	/// Startup only. A window dragged to a display with another scale keeps the scale it opened at
	/// (drawn sharp, but physically larger or smaller) until the next launch: following it live means
	/// rebuilding the panel, which holds the photo list and the last run's result.
	/// </para>
	/// </remarks>
	public static class DemoDisplayScale
	{
		/// <summary>The window's width at the design scale, in points (design units).</summary>
		public const int DesignWindowWidth = 1200;

		/// <summary>The window's height at the design scale, in points (design units).</summary>
		public const int DesignWindowHeight = 800;

		/// <summary>
		/// A display scale that can size a UI. A monitor hot-plug can be caught mid-transition reporting
		/// 0, and a UI laid out at scale 0 has nothing in it, so anything not finite and positive is 1.
		/// </summary>
		public static double Usable(double displayScale)
		{
			return double.IsNaN(displayScale) || double.IsInfinity(displayScale) || displayScale <= 0 ? 1 : displayScale;
		}

		/// <summary>
		/// The window size in device pixels for <paramref name="deviceScale"/>: the design size scaled, then
		/// clamped to <paramref name="desktopSize"/> (device pixels; zero when the host cannot measure it),
		/// so a 150% Windows laptop does not open a window taller than its screen.
		/// </summary>
		public static (int Width, int Height) WindowSize(double deviceScale, Point2D desktopSize)
		{
			int width = (int)Math.Round(DesignWindowWidth * deviceScale);
			int height = (int)Math.Round(DesignWindowHeight * deviceScale);
			if (desktopSize.x > 0)
			{
				width = Math.Min(width, desktopSize.x);
			}

			if (desktopSize.y > 0)
			{
				height = Math.Min(height, desktopSize.y);
			}

			return (width, height);
		}

		/// <summary>
		/// Sets <see cref="GuiWidget.DeviceScale"/> to <paramref name="displayScale"/> (device pixels per
		/// point) and returns a window sized for it. Call before building <see cref="ColmapDemoApp"/>: widget
		/// sizes and fonts are fixed when a widget is built.
		/// </summary>
		public static SystemWindow CreateWindow(string title, double displayScale, Point2D desktopSize)
		{
			GuiWidget.DeviceScale = Usable(displayScale);
			(int width, int height) = WindowSize(GuiWidget.DeviceScale, desktopSize);
			Console.WriteLine($"COLMAP_DEMO display scale {GuiWidget.DeviceScale}, window {width} x {height} px");
			return new SystemWindow(width, height) { Title = title };
		}

		/// <summary>
		/// <see cref="CreateWindow(string, double, Point2D)"/> for the display the OS reports: the Mac main
		/// screen's backing scale, the Windows system DPI over 96, the browser's devicePixelRatio.
		/// </summary>
		public static SystemWindow CreateWindow(string title)
		{
			IOsInformationProvider os = AggContext.OsInformation;
			return CreateWindow(title, os?.DisplayScale ?? 1, os?.DesktopSize ?? new Point2D(0, 0));
		}
	}
}
