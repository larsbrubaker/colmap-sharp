// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of the demo's video intake: VideoFrameSampler.PickFrameTimes (which moments of a video
// become photos), ColmapDemoApp.IsVideoPath, and adding a video to the panel through a fake
// IVideoFrameReader (demo/ColmapDemo/ColmapDemoApp.Video.cs): the frames listed, and a read that is
// cancelled, fails, is added to while going, or has its window closed under it. The panel's list
// changes are marshalled back through agg's idle queue, which the test pumps on a thread of its own
// (RunOnUiPump) since there is no window.
// NotInParallel with CancelTests: both drive agg's global idle queue, and a pump here would run
// the cancelled run's completion there.

using MatterHackers.Agg;
using MatterHackers.Agg.Image;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

namespace ColmapDemo.Tests;

[NotInParallel("AggUiThread")]
public class VideoIntakeTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public Point2D DesktopSize => new Point2D(1920, 1080);

		public long PhysicalMemory => 8L << 30;
	}

	/// <summary>A video of <see cref="Info"/> whose every frame is a small grey image.</summary>
	private sealed class FakeReader : IVideoFrameReader
	{
		public VideoInfo Info { get; init; } = new VideoInfo(TimeSpan.FromSeconds(8.6), 36, 64, 90, 30);

		public List<TimeSpan> RequestedTimes { get; } = new List<TimeSpan>();

		/// <summary>
		/// Awaited on the reading thread with the video and a frame index before that frame is handed
		/// over, and with the frame count once all are; it can cancel, throw, or poke the panel.
		/// </summary>
		public Func<string, int, Task> BeforeFrame { get; init; }

		public bool IsSupported => true;

		public string UnsupportedReason => null;

		public Task<VideoInfo> GetInfoAsync(string path, CancellationToken cancellationToken) => Task.FromResult(this.Info);

		public async Task ReadFramesAsync(string path, IReadOnlyList<TimeSpan> times, Func<int, ImageBuffer, Task> onFrame, CancellationToken cancellationToken)
		{
			this.RequestedTimes.AddRange(times);
			for (int i = 0; i < times.Count; i++)
			{
				if (this.BeforeFrame != null)
				{
					await this.BeforeFrame(path, i);
				}

				var image = new ImageBuffer(this.Info.Width, this.Info.Height);
				image.NewGraphics2D().Clear(new Color(i * 5, 128, 64));
				await onFrame(i, image);
			}

			if (this.BeforeFrame != null)
			{
				await this.BeforeFrame(path, times.Count);
			}
		}
	}

	[Test]
	public async Task PickFrameTimesSpacesTheTargetCountEvenly()
	{
		IReadOnlyList<TimeSpan> times = VideoFrameSampler.PickFrameTimes(TimeSpan.FromSeconds(8), 30, 40);

		await Assert.That(times.Count).IsEqualTo(40);
		await Assert.That(times[0]).IsEqualTo(TimeSpan.FromSeconds(0.1));
		await Assert.That(times[39]).IsEqualTo(TimeSpan.FromSeconds(7.9));
		for (int i = 1; i < times.Count; i++)
		{
			await Assert.That(times[i] - times[i - 1]).IsEqualTo(TimeSpan.FromSeconds(0.2));
		}
	}

	[Test]
	public async Task PickFrameTimesNeverTakesMoreFramesThanTheVideoHas()
	{
		// 1 s at 10 fps has 10 frames; each pick lands in its own frame's display interval.
		IReadOnlyList<TimeSpan> times = VideoFrameSampler.PickFrameTimes(TimeSpan.FromSeconds(1), 10, 40);

		await Assert.That(times.Count).IsEqualTo(10);
		await Assert.That(times.Select(t => (int)Math.Floor(t.TotalSeconds * 10)).Distinct().Count()).IsEqualTo(10);
	}

	[Test]
	public async Task PickFrameTimesTakesFewerFramesFromAVeryShortVideo()
	{
		// 0.5 s at 60 fps has 30 frames, but frames closer than MinSpacing are near-copies.
		await Assert.That(VideoFrameSampler.PickFrameTimes(TimeSpan.FromSeconds(0.5), 60, 40).Count).IsEqualTo(5);

		// Unknown frame rate: only the spacing limits it.
		await Assert.That(VideoFrameSampler.PickFrameTimes(TimeSpan.FromSeconds(2), 0, 40).Count).IsEqualTo(20);

		// A single-frame clip still gives its one frame, in the middle.
		IReadOnlyList<TimeSpan> one = VideoFrameSampler.PickFrameTimes(TimeSpan.FromSeconds(0.04), 25, 40);
		await Assert.That(one.Count).IsEqualTo(1);
		await Assert.That(one[0]).IsEqualTo(TimeSpan.FromSeconds(0.02));

		await Assert.That(VideoFrameSampler.PickFrameTimes(TimeSpan.Zero, 30, 40).Count).IsEqualTo(0);
	}

	[Test]
	public async Task IsVideoPathTakesVideoExtensionsInAnyCase()
	{
		await Assert.That(ColmapDemoApp.IsVideoPath(@"C:\clips\PXL_1.TS.mp4")).IsTrue();
		await Assert.That(ColmapDemoApp.IsVideoPath("/Users/me/IMG_0001.MOV")).IsTrue();
		foreach (string extension in new[] { ".m4v", ".avi", ".wmv", ".mkv" })
		{
			await Assert.That(ColmapDemoApp.IsVideoPath("clip" + extension)).IsTrue();
		}

		await Assert.That(ColmapDemoApp.IsVideoPath("photo.jpg")).IsFalse();
		await Assert.That(ColmapDemoApp.IsVideoPath("notes.txt")).IsFalse();
		await Assert.That(ColmapDemoApp.IsPhotoPath("clip.mp4")).IsFalse();
	}

	[Test]
	public async Task AddingAVideoListsItsFramesAsPhotosAndClearDeletesThem()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		var reader = new FakeReader();
		try
		{
			AggContext.VideoFrames = reader;
			var app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };
			string video = Path.Combine(root, "clip.MP4");

			await RunOnUiPump(() => app.AddVideosAsync(new[] { video }));

			await Assert.That(app.PhotoPaths.Count).IsEqualTo(12);
			await Assert.That(reader.RequestedTimes).IsEquivalentTo(VideoFrameSampler.PickFrameTimes(reader.Info.Duration, reader.Info.FramesPerSecond, 12));
			foreach (string frame in app.PhotoPaths)
			{
				await Assert.That(File.Exists(frame)).IsTrue();
				await Assert.That(ColmapDemoApp.IsPhotoPath(frame)).IsTrue();
			}

			// The frames decode like any photo, at the size the reader delivered.
			ColmapSharp.Sensor.Bitmap first = PhotoDecoder.Decode(app.PhotoPaths[0], 0);
			await Assert.That(first.Width).IsEqualTo(36);
			await Assert.That(first.Height).IsEqualTo(64);

			await Assert.That(app.PhotoListLines).IsEquivalentTo(new[] { "12 frames from clip.MP4" });
			await Assert.That(app.IsReadingVideo).IsFalse();
			await Assert.That(app.ErrorText).IsEqualTo(string.Empty);

			string folder = Path.GetDirectoryName(app.PhotoPaths[0]);
			app.FindDescendant("Clear Photos Button").InvokeClick();
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(0);
			await Assert.That(Directory.Exists(folder)).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task AVideoWhereThereIsNoReaderSaysSoInThePanel()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		try
		{
			AggContext.VideoFrames = new UnsupportedVideoFrameReader();
			var app = new ColmapDemoApp(fileDropSupported: false);

			app.AddPhotos(new[] { "clip.mov" });

			await Assert.That(app.ErrorText).IsEqualTo(UnsupportedVideoFrameReader.DefaultReason);
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(0);
			await Assert.That(app.PhotoListLines.Count).IsEqualTo(0);
			await Assert.That(app.IsReadingVideo).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
		}
	}

	[Test]
	public async Task CancellingMidReadDeletesItsFramesAndSaysSo()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		ColmapDemoApp app = null;
		var reader = new FakeReader
		{
			BeforeFrame = async (path, index) =>
			{
				if (index == 3)
				{
					await OnUiThread(() => app.RequestCancel());
				}
			},
		};
		try
		{
			AggContext.VideoFrames = reader;
			app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };

			await RunOnUiPump(() =>
			{
				app.AddPhotos(new[] { "a.png", "b.png", "c.png" }.Select(n => Path.Combine(root, n)));
				return app.AddVideosAsync(new[] { Path.Combine(root, "clip.mp4") });
			});

			await Assert.That(Directory.GetDirectories(root)).IsEmpty();
			await Assert.That(app.PhotoListLines).IsEquivalentTo(new[] { "a.png", "b.png", "c.png" });
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(3);
			await Assert.That(app.StatusText).IsEqualTo("Stopped reading clip.mp4.");
			await Assert.That(app.ErrorText).IsEqualTo(string.Empty);
			await Assert.That(app.IsReadingVideo).IsFalse();
			await Assert.That(app.FindDescendant("Run Button").Enabled).IsTrue();
			await Assert.That(app.CanCancel).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task CancellingBetweenVideosKeepsTheReadOneAndSaysWhichWasNot()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		ColmapDemoApp app = null;
		var reader = new FakeReader
		{
			// Once the first video's last frame is in: its read still succeeds, the second is never started.
			BeforeFrame = async (path, index) =>
			{
				if (Path.GetFileName(path) == "first.mp4" && index == 12)
				{
					await OnUiThread(() => app.RequestCancel());
				}
			},
		};
		try
		{
			AggContext.VideoFrames = reader;
			app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };

			await RunOnUiPump(() => app.AddVideosAsync(new[] { Path.Combine(root, "first.mp4"), Path.Combine(root, "second.mp4") }));

			await Assert.That(app.PhotoListLines).IsEquivalentTo(new[] { "12 frames from first.mp4" });
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(12);
			await Assert.That(reader.RequestedTimes.Count).IsEqualTo(12);
			await Assert.That(app.StatusText).IsEqualTo("Stopped before reading second.mp4.");
			await Assert.That(app.IsReadingVideo).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task AReaderErrorIsShownInItsOwnWordsAndItsFramesDeleted()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		const string message = "clip.mp4 stops being readable at 0:03.";
		var reader = new FakeReader
		{
			BeforeFrame = (path, index) => index == 3 ? throw new VideoFrameReaderException(message) : Task.CompletedTask,
		};
		try
		{
			AggContext.VideoFrames = reader;
			var app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };

			await RunOnUiPump(() => app.AddVideosAsync(new[] { Path.Combine(root, "clip.mp4") }));

			await Assert.That(app.ErrorText).IsEqualTo(message);
			await Assert.That(Directory.GetDirectories(root)).IsEmpty();
			await Assert.That(app.PhotoListLines.Count).IsEqualTo(0);
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(0);
			await Assert.That(app.StatusText).IsEqualTo(string.Empty);
			await Assert.That(app.IsReadingVideo).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task AnAddWhileAVideoIsBeingReadIsRefusedUntilItIsDone()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		ColmapDemoApp app = null;
		string errorDuringRead = null;
		var reader = new FakeReader
		{
			BeforeFrame = async (path, index) =>
			{
				if (index == 3)
				{
					await OnUiThread(() =>
					{
						_ = app.AddVideosAsync(new[] { Path.Combine(root, "other.mp4") });
						errorDuringRead = app.ErrorText;
					});
				}
			},
		};
		try
		{
			AggContext.VideoFrames = reader;
			app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };

			await RunOnUiPump(() => app.AddVideosAsync(new[] { Path.Combine(root, "clip.mp4") }));

			await Assert.That(errorDuringRead).IsEqualTo("A video is still being read. Add more once it is done.");

			// The read that was going succeeded, so the note about waiting for it is gone.
			await Assert.That(app.ErrorText).IsEqualTo(string.Empty);
			await Assert.That(app.PhotoListLines).IsEquivalentTo(new[] { "12 frames from clip.mp4" });
			await Assert.That(reader.RequestedTimes.Count).IsEqualTo(12);
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task ClosingTheWindowMidReadDeletesTheFramesBeingRead()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string root = Directory.CreateTempSubdirectory("ColmapDemoVideoTests").FullName;
		ColmapDemoApp app = null;
		string[] foldersAfterClose = null;
		var reader = new FakeReader
		{
			BeforeFrame = async (path, index) =>
			{
				if (index == 3)
				{
					// Checked at the close itself: a closed window's pump never runs the read's own
					// cleanup, which this test's pump would otherwise go on to run.
					await OnUiThread(() =>
					{
						app.Close();
						foldersAfterClose = Directory.GetDirectories(root);
					});
				}
			},
		};
		try
		{
			AggContext.VideoFrames = reader;
			app = new ColmapDemoApp(fileDropSupported: false) { VideoFramesRoot = root, TargetFramesPerVideo = 12 };

			await RunOnUiPump(() => app.AddVideosAsync(new[] { Path.Combine(root, "clip.mp4") }));

			await Assert.That(foldersAfterClose).IsEmpty();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			Directory.Delete(root, recursive: true);
		}
	}

	// Stands in for the window's pump: one dedicated thread, marked the UI thread once, starts the
	// work and then drains agg's idle queue until it is done, so the intake's switches back to the
	// UI thread always land here and never inline on the reading thread.
	private static Task RunOnUiPump(Func<Task> start)
	{
		var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
		var pump = new Thread(() =>
		{
			try
			{
				UiThread.MarkCurrentThreadAsUiThread();
				Task work = start();
				while (!work.IsCompleted)
				{
					UiThread.InvokePendingActions();
					Thread.Sleep(1);
				}

				// Late progress reports, so none is left for the next test's pump.
				UiThread.InvokePendingActions();
				if (work.IsFaulted)
				{
					done.SetException(work.Exception.InnerExceptions);
				}
				else if (work.IsCanceled)
				{
					done.SetCanceled();
				}
				else
				{
					done.SetResult();
				}
			}
			catch (Exception e)
			{
				done.SetException(e);
			}
		})
		{
			IsBackground = true,
			Name = "VideoIntakeTests UI pump",
		};
		pump.Start();
		return done.Task;
	}

	// From the reading thread: runs action on the pump and completes once it has.
	private static Task OnUiThread(Action action)
	{
		var ran = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
		UiThread.RunOnIdle(() =>
		{
			try
			{
				action();
				ran.SetResult();
			}
			catch (Exception e)
			{
				ran.SetException(e);
			}
		});
		return ran.Task;
	}
}
