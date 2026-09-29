// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Video: a dropped or picked video becomes photos. Its frames are read off the UI
// thread (VideoFrameSampler.cs picks which and writes them as PNGs into a temp folder per video),
// with "Reading video frames 12/40…" in the status line and Cancel stopping it; then the frames
// join the photo list as one line ("40 frames from clip.mp4") and a run treats them like any photo,
// except that a run of one video's frames alone gives them one shared camera (ColmapDemoApp.Placement.cs).
// Where agg has no video reader (Mac and browser today) the panel says so in the reader's own
// words. The frame folders are deleted by Clear and when the window closes (the one being read
// too: its read never gets back to the UI thread once the window is gone).
// The rest of the panel is in ColmapDemoApp.cs; Run and Cancel in ColmapDemoApp.Run.cs.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		// Each added video and the temp folder its frames are in.
		private readonly List<(string Video, string Folder)> videos = new List<(string Video, string Folder)>();

		// Each added video's frames (paths in the photo list), for telling a one-video run apart.
		private readonly List<IReadOnlyList<string>> videoFrameSets = new List<IReadOnlyList<string>>();

		private CancellationTokenSource videoCancel;

		// The folder of the video being read, until its read is back on the UI thread; OnClosed deletes
		// it, since a closed window's pump never runs the read's own cleanup.
		private string readingFolder;

		private const string StillReadingMessage = "A video is still being read. Add more once it is done.";

		/// <summary>How many frames each added video is cut into.</summary>
		public int TargetFramesPerVideo { get; set; } = VideoFrameSampler.DefaultTargetFrames;

		/// <summary>Where each video's frame folder is made.</summary>
		public string VideoFramesRoot { get; set; } = Path.Combine(Path.GetTempPath(), "ColmapDemo");

		/// <summary>Whether a video's frames are being read (Run waits; Cancel stops it).</summary>
		public bool IsReadingVideo => this.videoCancel != null;

		/// <summary>The error line's text when it is shown, otherwise empty.</summary>
		public string ErrorText => this.errorLine.Visible ? this.errorLine.Text : string.Empty;

		/// <summary>The lines of the photo list, one per photo or video.</summary>
		public IReadOnlyList<string> PhotoListLines => this.photoList.Children.Select(c => c.Text).ToList();

		/// <summary>Whether <paramref name="path"/> has one of the video extensions the demo takes (case-insensitive).</summary>
		public static bool IsVideoPath(string path)
		{
			string extension = Path.GetExtension(path);
			return VideoExtensions.Any(e => string.Equals(e, extension, StringComparison.OrdinalIgnoreCase));
		}

		/// <summary>
		/// Cuts each of <paramref name="paths"/> into frames and adds them to the photo list. Call on the
		/// UI thread; the reading happens off it and the list changes back on it. Never throws: a video
		/// that cannot be read says why in the panel, and the rest go on. Completes when all are read,
		/// failed, or cancelled.
		/// </summary>
		public async Task AddVideosAsync(IReadOnlyList<string> paths)
		{
			if (this.IsRunning || paths.Count == 0)
			{
				return;
			}

			if (this.IsReadingVideo)
			{
				this.ShowError(StillReadingMessage);
				return;
			}

			IVideoFrameReader reader = AggContext.VideoFrames;
			if (!reader.IsSupported)
			{
				this.ShowError(reader.UnsupportedReason ?? UnsupportedVideoFrameReader.DefaultReason);
				return;
			}

			this.errorLine.Visible = false;
			var cancel = new CancellationTokenSource();
			this.videoCancel = cancel;
			this.UpdatePhotoCount();
			try
			{
				foreach (string path in paths)
				{
					if (cancel.IsCancellationRequested)
					{
						// Cancelled after the last video's frames were all in: that one stays, the rest are not read.
						this.statusLine.Text = $"Stopped before reading {Path.GetFileName(path)}.";
						break;
					}

					if (this.videos.Any(v => v.Video == path))
					{
						continue;
					}

					await this.AddVideoAsync(reader, path, cancel.Token);
				}
			}
			finally
			{
				// Every path through AddVideoAsync ends back on the UI thread.
				cancel.Dispose();
				this.videoCancel = null;

				// An add refused while this read was going is moot now that it is done.
				if (this.errorLine.Visible && this.errorLine.Text == StillReadingMessage)
				{
					this.errorLine.Visible = false;
				}

				this.UpdatePhotoCount();
			}
		}

		private async Task AddVideoAsync(IVideoFrameReader reader, string path, CancellationToken cancel)
		{
			string name = Path.GetFileName(path);
			var row = this.AddListLine($"Reading {name}…");
			this.statusLine.Text = $"Reading {name}…";
			string folder = Path.Combine(this.VideoFramesRoot, "video-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + "-" + Guid.NewGuid().ToString("N")[..8]);
			var clock = Stopwatch.StartNew();

			// Reported on the reading thread; only the latest count matters, so a late one is dropped.
			Action<int, int> progress = (done, total) => UiThread.RunOnIdle(() =>
			{
				if (this.IsReadingVideo && !cancel.IsCancellationRequested)
				{
					this.statusLine.Text = $"Reading video frames {done}/{total}…";
				}
			});
			Func<Task<IReadOnlyList<string>>> read = () => VideoFrameSampler.ExtractFramesAsync(reader, path, folder, this.TargetFramesPerVideo, progress, cancel);

			IReadOnlyList<string> frames = null;
			Exception error = null;
			this.readingFolder = folder;
			try
			{
				// Decoding and PNG encoding are seconds of work: off the UI thread, except in the browser,
				// which has no other thread (and no reader yet).
				frames = this.RunOnUiThread ? await read() : await Task.Run(read);
			}
			catch (Exception e)
			{
				error = e;
			}

			await UiThread.SwitchToUiThreadAsync();
			this.readingFolder = null;
			if (frames != null)
			{
				Console.WriteLine($"COLMAP_DEMO video {name}: {frames.Count} frames in {clock.Elapsed.TotalSeconds:F1} s");
				this.videos.Add((path, folder));
				foreach (string frame in frames)
				{
					this.photoPaths.Add(frame);
				}

				// The count first: a phone's file name is longer than the panel is wide.
				row.Text = $"{frames.Count} {(frames.Count == 1 ? "frame" : "frames")} from {name}";
				this.AddListEntry(row, frames);
				this.videoFrameSets.Add(frames);
				this.statusLine.Text = string.Empty;
				this.UpdatePhotoCount();
				return;
			}

			row.Close();
			DeleteFolder(folder);
			if (error is OperationCanceledException)
			{
				this.statusLine.Text = $"Stopped reading {name}.";
			}
			else
			{
				Console.WriteLine($"COLMAP_DEMO video {name} failed: {error}");
				this.statusLine.Text = string.Empty;

				// The reader's own messages are written for the user; anything else gets the video's name.
				this.ShowError(error is VideoFrameReaderException ? error.Message : $"Could not read {name}: {FirstLine(error.Message)}");
			}
		}

		/// <summary>Deletes every video's frame folder; the frames' paths leave the list with them.</summary>
		private void DeleteVideoFrames()
		{
			foreach (var video in this.videos)
			{
				DeleteFolder(video.Folder);
			}

			this.videos.Clear();
			this.videoFrameSets.Clear();
		}

		private void ShowError(string message)
		{
			this.errorLine.Text = message;
			this.errorLine.Visible = true;
		}

		private static void DeleteFolder(string folder)
		{
			try
			{
				if (Directory.Exists(folder))
				{
					Directory.Delete(folder, recursive: true);
				}
			}
			catch (Exception e)
			{
				// A frame still open elsewhere; the OS temp cleaner gets it later.
				Console.Error.WriteLine($"Could not delete the video frames {folder}: {e.Message}");
			}
		}

		/// <summary>Stops a video being read, and deletes the frame folders with the window.</summary>
		public override void OnClosed(EventArgs e)
		{
			this.videoCancel?.Cancel();

			// Best effort: the reader may be writing a frame into it right now (DeleteFolder only logs
			// that), and a frame written after this is left for the OS temp cleaner.
			if (this.readingFolder != null)
			{
				DeleteFolder(this.readingFolder);
				this.readingFolder = null;
			}

			this.DeleteVideoFrames();
			base.OnClosed(e);
		}
	}
}
