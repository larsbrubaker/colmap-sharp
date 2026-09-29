// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DevAutoRun: a developer hook for unattended runs of the desktop heads (ColmapDemo.Mac and
// ColmapDemo.Windows each call Attach from their Program.cs), driven by environment variables, so
// a live run (the real window, the real GPU, the real UI-thread marshalling) can be checked and
// screenshotted without a person clicking. Unset, it does nothing. The browser head does not call
// it: a page has no environment variables to read.
//
//   COLMAP_DEMO_PHOTOS=<dir>              preload the photos in <dir> (sorted by name)
//   COLMAP_DEMO_AUTORUN=1                 press Run once the window is up
//   COLMAP_DEMO_SCREENSHOT_SPARSE=<png>   screenshot when the sparse points appear
//   COLMAP_DEMO_SCREENSHOT=<png>          screenshot when the run ends, then close the window
//
// agg's AGG_SMOKE_* counts frames from startup, which cannot wait for a run of unknown length, so
// this captures on the app's own events through SystemWindow.CaptureScreenshotAsync instead.

using System;
using System.IO;
using System.Linq;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public static class DevAutoRun
	{
		/// <summary>Wires the hook to <paramref name="app"/> in <paramref name="window"/>, if asked for.</summary>
		public static void Attach(SystemWindow window, ColmapDemoApp app)
		{
			string photoDir = Environment.GetEnvironmentVariable("COLMAP_DEMO_PHOTOS");
			if (string.IsNullOrEmpty(photoDir))
			{
				return;
			}

			app.AddPhotos(Directory.GetFiles(photoDir).OrderBy(p => p, StringComparer.Ordinal));

			string sparseShot = Environment.GetEnvironmentVariable("COLMAP_DEMO_SCREENSHOT_SPARSE");
			if (!string.IsNullOrEmpty(sparseShot))
			{
				app.SparseShown += () => UiThread.RunOnIdle(async () =>
				{
					await window.CaptureScreenshotAsync(sparseShot);
					Console.WriteLine($"COLMAP_DEMO sparse screenshot: {sparseShot}");
				});
			}

			string finalShot = Environment.GetEnvironmentVariable("COLMAP_DEMO_SCREENSHOT");
			app.RunFinished += result =>
			{
				Console.WriteLine(result == null
					? "COLMAP_DEMO run ended without a result"
					: $"COLMAP_DEMO run done: {result.Mesh?.Faces.Count ?? 0} faces, textured={result.IsTextured}");
				if (!string.IsNullOrEmpty(finalShot))
				{
					UiThread.RunOnIdle(async () =>
					{
						await window.CaptureScreenshotAsync(finalShot);
						Console.WriteLine($"COLMAP_DEMO final screenshot: {finalShot}");
						window.CloseOnIdle();
					});
				}
			};

			if (Environment.GetEnvironmentVariable("COLMAP_DEMO_AUTORUN") == "1")
			{
				UiThread.RunOnIdle(app.StartRun);
			}
		}
	}
}
