// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DevAutoRun: a developer hook for unattended runs of the desktop heads (ColmapDemo.Mac and
// ColmapDemo.Windows each call Attach from their Program.cs), driven by environment variables, so
// a live run (the real window, the real GPU, the real UI-thread marshalling) can be checked and
// screenshotted without a person clicking. Unset, it does nothing. The browser head does not call
// it: a page has no environment variables to read.
//
//   COLMAP_DEMO_PHOTOS=<dir>              preload the photos and videos in <dir> (sorted by name)
//   COLMAP_DEMO_VIDEO=<file>              add a video (cut into frames, as a drop would)
//   COLMAP_DEMO_AUTORUN=1                 press Run once the window is up (and the videos are read)
//   COLMAP_DEMO_SCREENSHOT_SPARSE=<png>   screenshot when the sparse points appear
//   COLMAP_DEMO_SCREENSHOT=<png>          screenshot when the run ends, then close the window
//   COLMAP_DEMO_SETTINGS_OPEN=1           open the Settings panel at startup (for a screenshot of it)
//
// agg's AGG_SMOKE_* counts frames from startup, which cannot wait for a run of unknown length, so
// this captures on the app's own events through SystemWindow.CaptureScreenshotAsync instead.

using System;
using System.Collections.Generic;
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
			// Before the photo check: a screenshot of the panel needs no photos.
			if (Environment.GetEnvironmentVariable("COLMAP_DEMO_SETTINGS_OPEN") == "1")
			{
				app.SettingsOpen = true;
			}

			string photoDir = Environment.GetEnvironmentVariable("COLMAP_DEMO_PHOTOS");
			string video = Environment.GetEnvironmentVariable("COLMAP_DEMO_VIDEO");
			if (string.IsNullOrEmpty(photoDir) && string.IsNullOrEmpty(video))
			{
				return;
			}

			// The folder's videos and COLMAP_DEMO_VIDEO are read as one batch, awaited before Run: a second
			// add while one is being read would only be refused ("still being read").
			var videos = new List<string>();
			if (!string.IsNullOrEmpty(photoDir))
			{
				var files = Directory.GetFiles(photoDir).OrderBy(p => p, StringComparer.Ordinal).ToList();
				videos.AddRange(files.Where(ColmapDemoApp.IsVideoPath));
				app.AddPhotos(files.Where(p => !ColmapDemoApp.IsVideoPath(p)));
			}

			if (!string.IsNullOrEmpty(video))
			{
				videos.Add(video);
			}

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
					: $"COLMAP_DEMO run done: {result.Mesh?.Faces.Count ?? 0} faces, textured={result.IsTextured}; {result.Placement?.StatusText}");
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

			bool autoRun = Environment.GetEnvironmentVariable("COLMAP_DEMO_AUTORUN") == "1";
			UiThread.RunOnIdle(async () =>
			{
				// On the UI thread, once the window is up: the videos' frames are read before Run.
				if (videos.Count > 0)
				{
					await app.AddVideosAsync(videos);
				}

				if (autoRun)
				{
					if (app.ErrorText.Length > 0)
					{
						Console.WriteLine("COLMAP_DEMO video not added: " + app.ErrorText);
					}

					app.StartRun();

					// Too few photos (a video that could not be read): say so and end the unattended run
					// rather than leave a window waiting for a run that never started.
					if (!app.IsRunning)
					{
						Console.WriteLine($"COLMAP_DEMO run did not start ({app.PhotoPaths.Count} photos)");
						if (!string.IsNullOrEmpty(finalShot))
						{
							await window.CaptureScreenshotAsync(finalShot);
							window.CloseOnIdle();
						}
					}
				}
			});
		}
	}
}
