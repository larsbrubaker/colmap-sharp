// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VideoFrameSampler: the demo's policy for turning a video into photos - which moments to take
// (PickFrameTimes) and writing each decoded frame as a PNG (ExtractFramesAsync). The decoding
// itself is agg-sharp's (AggContext.VideoFrames: Media Foundation on Windows, AVFoundation on the mac); the panel side -
// progress, cancel, the photo list - is in ColmapDemoApp.Video.cs.
// Tests: demo/ColmapDemo.Tests/VideoIntakeTests.cs.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using MatterHackers.Agg.Image;
using MatterHackers.Agg.Platform;

namespace ColmapDemo
{
	/// <summary>Which frames of a video become photos, and writing them out.</summary>
	public static class VideoFrameSampler
	{
		/// <summary>
		/// How many frames a video is cut into by default. Enough views for the mapper to chain a
		/// walk-around together, few enough that reading them (about a second per frame for a phone's
		/// HEVC today) and the run stay short.
		/// </summary>
		public const int DefaultTargetFrames = 40;

		/// <summary>
		/// The closest two frames are taken. Frames closer than this barely move the camera, so a
		/// very short video gives fewer frames rather than near-copies that add time and no baseline.
		/// </summary>
		public static readonly TimeSpan MinSpacing = TimeSpan.FromSeconds(0.1);

		/// <summary>
		/// Picks up to <paramref name="targetFrames"/> evenly spaced times in a video of
		/// <paramref name="duration"/>: the middles of equal slices, so the very first and last
		/// moments (often a shaky start and stop) are skipped by half a slice. Never more times than
		/// the video has frames at <paramref name="framesPerSecond"/> (0 when unknown), nor closer
		/// together than <see cref="MinSpacing"/>; an empty list for an empty video.
		/// </summary>
		public static IReadOnlyList<TimeSpan> PickFrameTimes(TimeSpan duration, double framesPerSecond, int targetFrames = DefaultTargetFrames)
		{
			// Some MKVs and fragmented MP4s report no duration; the user then sees "<name> has no frames to read."
			if (duration <= TimeSpan.Zero || targetFrames <= 0)
			{
				return Array.Empty<TimeSpan>();
			}

			long count = targetFrames;
			if (framesPerSecond > 0)
			{
				count = Math.Min(count, Math.Max(1, (long)Math.Floor(duration.TotalSeconds * framesPerSecond)));
			}

			count = Math.Min(count, Math.Max(1, duration.Ticks / MinSpacing.Ticks));

			var times = new TimeSpan[count];
			for (long i = 0; i < count; i++)
			{
				// Integer ticks, so the spacing is exact and the last time stays inside the video.
				times[i] = TimeSpan.FromTicks(duration.Ticks * (2 * i + 1) / (2 * count));
			}

			return times;
		}

		/// <summary>
		/// Reads <paramref name="videoPath"/>'s frames at <see cref="PickFrameTimes"/> through
		/// <paramref name="reader"/> and writes each as a PNG in <paramref name="folder"/> (created),
		/// named after the video with its frame number. <paramref name="progress"/> gets (done, total)
		/// on the reading thread before the first frame and after each. Returns the PNG paths in time
		/// order. Throws <see cref="VideoFrameReaderException"/> (a user-facing message) when the video
		/// cannot be read, and <see cref="OperationCanceledException"/> when cancelled; either way the
		/// caller owns deleting <paramref name="folder"/>.
		/// </summary>
		public static async Task<IReadOnlyList<string>> ExtractFramesAsync(
			IVideoFrameReader reader,
			string videoPath,
			string folder,
			int targetFrames,
			Action<int, int> progress,
			CancellationToken cancel)
		{
			VideoInfo info = await reader.GetInfoAsync(videoPath, cancel).ConfigureAwait(false);
			IReadOnlyList<TimeSpan> times = PickFrameTimes(info.Duration, info.FramesPerSecond, targetFrames);
			if (times.Count == 0)
			{
				throw new VideoFrameReaderException($"{Path.GetFileName(videoPath)} has no frames to read.");
			}

			// A cancel that came while the reader opened the video (the window closing) must not leave a
			// folder behind that nothing will delete.
			cancel.ThrowIfCancellationRequested();
			Directory.CreateDirectory(folder);
			string stem = Path.GetFileNameWithoutExtension(videoPath);
			var paths = new string[times.Count];
			progress?.Invoke(0, times.Count);
			await reader.ReadFramesAsync(
				videoPath,
				times,
				(index, image) =>
				{
					cancel.ThrowIfCancellationRequested();

					// PNG: lossless, and the run shrinks it to the max image size when it decodes it, as
					// it does any photo. A stream always overwrites and a failure is reported.
					string path = Path.Combine(folder, $"{stem}_{index + 1:D3}.png");
					using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
					{
						if (!ImageIO.SaveImageData(stream, ".png", image))
						{
							throw new IOException($"Could not write the frame {path}.");
						}
					}

					paths[index] = path;
					progress?.Invoke(index + 1, times.Count);
					return Task.CompletedTask;
				},
				cancel).ConfigureAwait(false);

			if (Array.IndexOf(paths, null) >= 0)
			{
				throw new VideoFrameReaderException($"Some frames of {Path.GetFileName(videoPath)} could not be read.");
			}

			return paths;
		}
	}
}
