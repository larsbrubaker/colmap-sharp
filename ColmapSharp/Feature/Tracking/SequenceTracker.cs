// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SequenceTracker: KLT feature tracks through a time-ordered video (docs/QUALITY_PLAN.md stage
// 2a). Not a COLMAP port; COLMAP has no video tracker. The scheme is the classic KLT tracker of
// Tomasi and Kanade (CMU-CS-91-132, 1991) with Shi-Tomasi corners (CVPR 1994), Bouguet's
// pyramidal Lucas-Kanade (Intel 2000; Lucas and Kanade, IJCAI 1981) and the forward-backward
// check of Kalal, Mikolajczyk and Matas (ICPR 2010), all in this folder and written from the
// papers only.
//
// Frames are pushed one at a time (AddFrame) or streamed (Run); only the previous and the
// current pyramid are held, so memory does not grow with the video. Each live point is tracked
// from the previous frame into the new one (PyramidalLucasKanade); a point that fails any check
// ends its track. When the live count falls below ReplenishFraction of TargetCount, new
// Shi-Tomasi corners are added (inside the frame's mask, if one is given, and at least
// MinDistance from the live points) up to TargetCount, each starting a new track.
//
// The mask is first eroded by MaskMargin pixels (by default the tracking window's radius), so a
// corner's whole window lies on the object: a window straddling the silhouette mixes the
// object's motion with the still background's and drifts, measured at a median 1.2 px per frame
// on the benchmark's TexturedSphere without the erosion. With EndTracksLeavingMask, a tracked
// point that lands outside the new frame's eroded mask also ends: on a turntable capture a point
// that slides off the object onto the background would otherwise carry on as a track that agrees
// with no rigid scene, and on a turning object the points nearing the silhouette are the most
// foreshortened ones.
//
// Positions are COLMAP's continuous image coordinates (pixel centers at +0.5). Track ids count
// up from 0 in the order tracks start; within a frame new tracks start strongest corner first.
// The result depends only on the frames, masks and options, not on the thread count.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Feature.Tracking;

/// <summary>One observation of a track: the frame index (0-based, in push order) and the position.</summary>
public readonly record struct TrackObservation(int Frame, double X, double Y);

/// <summary>A feature followed through consecutive frames.</summary>
public sealed class FeatureTrack
{
	internal FeatureTrack(int id) => Id = id;

	/// <summary>Unique id, counting up from 0 in the order tracks start.</summary>
	public int Id { get; }

	/// <summary>The observations, one per consecutive frame from the first.</summary>
	public List<TrackObservation> Observations { get; } = [];

	/// <summary>How many frames the track spans.</summary>
	public int Length => Observations.Count;
}

/// <summary>
/// Why the tracks that reached a frame ended there, one count per reason (the KLT statuses,
/// plus tracks that landed outside the frame's eroded mask). Diagnostics only.
/// </summary>
public readonly record struct TrackEndCounts(
	int ForwardBackwardFailed, int LowTexture, int HighResidual, int OutOfImage, int LeftMask)
{
	/// <summary>All tracks that ended at this frame.</summary>
	public int Total => ForwardBackwardFailed + LowTexture + HighResidual + OutOfImage + LeftMask;
}

/// <summary>Options for <see cref="SequenceTracker"/>.</summary>
public sealed class SequenceTrackerOptions
{
	// Illumination compensation on: measured on the benchmark's TexturedSphere (an object
	// turning under a fixed light), it cuts the per-pair error against the ray-cast truth from
	// 0.91 to 0.34 px median and the tracks' reprojection error from 1.38 to 0.50 px.
	/// <summary>How the points move from frame to frame (illumination compensation on).</summary>
	public KltOptions Klt { get; set; } = new() { CompensateIllumination = true };

	/// <summary>How new corners are found. Its MaxCorners is ignored; see <see cref="TargetCount"/>.</summary>
	public GoodFeaturesOptions Features { get; set; } = new();

	/// <summary>How many live points to aim for.</summary>
	public int TargetCount { get; set; } = 500;

	/// <summary>Detect new corners once the live count drops below this fraction of TargetCount.</summary>
	public double ReplenishFraction { get; set; } = 0.9;

	/// <summary>End a track whose tracked point lands outside the new frame's eroded mask (when a mask is given).</summary>
	public bool EndTracksLeavingMask { get; set; } = true;

	/// <summary>
	/// How far (pixels, square neighbourhood) to erode a given mask before using it; negative
	/// means the tracking window's radius, so a window never straddles the mask's edge.
	/// </summary>
	public int MaskMargin { get; set; } = -1;
}

/// <summary>
/// Follows Shi-Tomasi corners through a video with pyramidal Lucas-Kanade. Push frames in time
/// order with <see cref="AddFrame"/>; <see cref="Tracks"/> holds every track so far.
/// </summary>
public sealed class SequenceTracker
{
	private readonly SequenceTrackerOptions options;
	private readonly List<FeatureTrack> tracks = [];
	private readonly List<TrackEndCounts> endCounts = [];
	private List<FeatureTrack> live = [];
	private ImagePyramid? previous;
	private int frameCount;

	/// <summary>Creates a tracker with <paramref name="options"/> (defaults if null).</summary>
	public SequenceTracker(SequenceTrackerOptions? options = null)
	{
		this.options = options ?? new SequenceTrackerOptions();
		Check.That(this.options.TargetCount > 0);
		Check.That(this.options.ReplenishFraction > 0 && this.options.ReplenishFraction <= 1);
	}

	/// <summary>Every track started so far, in id order (ended and live).</summary>
	public IReadOnlyList<FeatureTrack> Tracks => tracks;

	/// <summary>Per added frame, why tracks ended on reaching it (all zero for the first frame).</summary>
	public IReadOnlyList<TrackEndCounts> EndCounts => endCounts;

	/// <summary>How many tracks are still being followed.</summary>
	public int LiveCount => live.Count;

	/// <summary>How many frames have been added.</summary>
	public int FrameCount => frameCount;

	/// <summary>
	/// Tracks the whole of <paramref name="frames"/>, with an optional mask per frame
	/// (8-bit grey, 255 = object; a null entry means no mask for that frame). Reports the number
	/// of frames done to <paramref name="progress"/>.
	/// </summary>
	public static IReadOnlyList<FeatureTrack> Run(
		IEnumerable<Bitmap> frames,
		IEnumerable<Bitmap?>? masks = null,
		SequenceTrackerOptions? options = null,
		IProgress<int>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var tracker = new SequenceTracker(options);
		using IEnumerator<Bitmap?>? maskEnum = masks?.GetEnumerator();
		foreach (Bitmap frame in frames)
		{
			cancellationToken.ThrowIfCancellationRequested();
			Bitmap? mask = null;
			if (maskEnum != null)
			{
				Check.That(maskEnum.MoveNext(), "fewer masks than frames");
				mask = maskEnum.Current;
			}

			tracker.AddFrame(frame, mask);
			progress?.Report(tracker.FrameCount);
		}

		return tracker.Tracks;
	}

	/// <summary>
	/// Adds the next frame: tracks the live points into it, ends the lost ones, and tops the
	/// live set back up with new corners if it has fallen too low. <paramref name="mask"/>
	/// (optional, 8-bit grey, same size) limits where new corners go and, with
	/// EndTracksLeavingMask, where tracked points may land.
	/// </summary>
	public void AddFrame(Bitmap frame, Bitmap? mask = null)
	{
		var pyramid = ImagePyramid.Build(frame, options.Klt.MaxLevel);
		if (previous != null)
		{
			Check.Eq(pyramid.Width, previous.Width);
			Check.Eq(pyramid.Height, previous.Height);
		}

		if (mask != null)
		{
			Check.That(mask.IsGrey && mask.Width == frame.Width && mask.Height == frame.Height);
			int margin = options.MaskMargin >= 0 ? options.MaskMargin : options.Klt.WindowSize / 2;
			mask = margin > 0 ? Erode(mask, margin) : mask;
		}

		int frameIndex = frameCount++;
		int fb = 0, low = 0, residual = 0, outside = 0, leftMask = 0;
		if (previous != null && live.Count > 0)
		{
			var points = new Vector2d[live.Count];
			for (int i = 0; i < live.Count; ++i)
			{
				TrackObservation last = live[i].Observations[^1];
				points[i] = new Vector2d(last.X, last.Y);
			}

			KltResult[] results = PyramidalLucasKanade.Track(previous, pyramid, points, options.Klt);
			var survivors = new List<FeatureTrack>(live.Count);
			for (int i = 0; i < live.Count; ++i)
			{
				KltResult r = results[i];
				switch (r.Status)
				{
					case KltStatus.ForwardBackwardFailed: fb++; continue;
					case KltStatus.LowTexture: low++; continue;
					case KltStatus.HighResidual: residual++; continue;
					case KltStatus.OutOfImage: outside++; continue;
				}

				if (options.EndTracksLeavingMask && mask != null && !InsideMask(mask, r.Position))
				{
					leftMask++;
					continue;
				}

				live[i].Observations.Add(new TrackObservation(frameIndex, r.Position.X, r.Position.Y));
				survivors.Add(live[i]);
			}

			live = survivors;
		}

		endCounts.Add(new TrackEndCounts(fb, low, residual, outside, leftMask));

		if (live.Count < options.ReplenishFraction * options.TargetCount)
		{
			Replenish(pyramid, mask, frameIndex);
		}

		previous = pyramid;
	}

	private void Replenish(ImagePyramid pyramid, Bitmap? mask, int frameIndex)
	{
		var existing = new Vector2d[live.Count];
		for (int i = 0; i < live.Count; ++i)
		{
			TrackObservation last = live[i].Observations[^1];
			existing[i] = new Vector2d(last.X, last.Y);
		}

		GoodFeaturesOptions f = options.Features;
		var detect = new GoodFeaturesOptions
		{
			MaxCorners = options.TargetCount - live.Count,
			QualityLevel = f.QualityLevel,
			MinDistance = f.MinDistance,
			BlockSize = f.BlockSize,
			BorderMargin = f.BorderMargin,
		};
		foreach (Vector2d p in GoodFeaturesToTrack.Detect(pyramid, detect, mask, existing))
		{
			var track = new FeatureTrack(tracks.Count);
			track.Observations.Add(new TrackObservation(frameIndex, p.X, p.Y));
			tracks.Add(track);
			live.Add(track);
		}
	}

	// Square-neighbourhood erosion (separable running minimum over 2r+1 pixels, outside = 0),
	// thresholded to 0/255.
	private static Bitmap Erode(Bitmap mask, int r)
	{
		int w = mask.Width, h = mask.Height;
		byte[] src = mask.RowMajorData;
		var rows = new byte[w * h];
		for (int y = 0; y < h; ++y)
		{
			int run = 0;
			// run = how many consecutive "on" pixels end at x; x - r is kept when the run
			// covers [x - 2r, x], i.e. its whole window. Pixels past the edge count as off.
			for (int x = 0; x < w + r; ++x)
			{
				run = x < w && src[y * w + x] >= 128 ? run + 1 : 0;
				int c = x - r;
				if (c >= 0)
				{
					rows[y * w + c] = (byte)(run >= 2 * r + 1 ? 255 : 0);
				}
			}
		}

		var eroded = new Bitmap(w, h, asRgb: false);
		byte[] dst = eroded.RowMajorData;
		for (int x = 0; x < w; ++x)
		{
			int run = 0;
			for (int y = 0; y < h + r; ++y)
			{
				run = y < h && rows[y * w + x] == 255 ? run + 1 : 0;
				int c = y - r;
				if (c >= 0)
				{
					dst[c * w + x] = (byte)(run >= 2 * r + 1 ? 255 : 0);
				}
			}
		}

		return eroded;
	}

	// The pixel containing continuous position p (pixel centers at +0.5).
	private static bool InsideMask(Bitmap mask, Vector2d p)
	{
		int x = (int)Math.Floor(p.X), y = (int)Math.Floor(p.Y);
		return x >= 0 && y >= 0 && x < mask.Width && y < mask.Height && mask.RowMajorData[y * mask.Width + x] >= 128;
	}
}
