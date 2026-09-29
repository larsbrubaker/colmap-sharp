// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePoseRegistration: places video frames that feature matching could not, from their
// silhouettes (docs/QUALITY_PLAN.md, stages 4a and 4b). Not a COLMAP port; COLMAP registers an
// image only through 2D-3D correspondences. The case it exists for: a stretch of a video where
// the object turns fast and shows little texture (the mouse's top/side sweep), so no frame in it
// matches, yet every frame has a clean mask.
//
// Per pass:
// 1. Carve a visual hull (Mvs/Silhouette/VisualHull) from every registered frame that has a mask.
// 2. For each unplaced frame, start from its neighbours in time (InitializePose, stage 4a):
//    rotation SLERP and camera-centre lerp between the nearest registered frames before and
//    after; with only one side, the motion of the two nearest on that side continued by at most
//    MaxExtrapolationRatio of their spacing.
// 3. Refine the pose against the hull (SilhouettePoseRefiner, stage 4b).
// 4. Accept it if the refined silhouette IoU reaches MinIou and refinement rotated the start by
//    at most MaxRotationFromInitDegrees, and then, in time order, only if carving with it does
//    not un-explain the frames registered by features (SilhouetteConsistencyGate.cs); register
//    it with that pose and no 2D-3D points.
// When the registered cameras fit a turntable circle (SilhouetteTurntable.cs, stage 5), steps 2-4
// run instead as SilhouettePoseRegistration.Turntable.cs: frames go one at a time nearest-first,
// with starts from a search round the whole turn.
// Passes repeat (MaxPasses) while a pass places something: the next hull includes the frames just
// placed, which tightens it where they saw new angles, and rejected frames start again from
// interpolations that now include the new neighbours.
//
// A frame's start uses only the frames registered at the start of its pass, so the frames of a
// pass are independent: they refine in parallel (each writing its own slot) and are registered
// afterwards in time order, which makes parallel and sequential runs identical.
//
// What a placed frame is, downstream (read this before wiring it into a pipeline):
// - It has a pose and no 2D points or 3D observations. PatchMatch picks source views by shared
//   3D points, so it ignores these frames; they help the visual hull and texturing only.
// - A later mapper or bundle-adjustment pass de-registers frames with no observations, so this
//   must run after the last mapper/BA pass.
// - A frame that is not yet an image is added with AddImageWithTrivialFrame, which needs the
//   rig with the camera's id to hold exactly that camera (a trivial rig).
//
// The world frame is whatever the reconstruction's is. For an object on a turntable or in a hand
// with a still camera, the registered poses are object-fixed, which is what makes lerping the
// camera centre meaningful.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>Registers unplaced frames of a video into a reconstruction by silhouette coherence.</summary>
public static partial class SilhouettePoseRegistration
{
	/// <summary>
	/// Places every frame of <paramref name="frames"/> (time order) that is not registered in
	/// <paramref name="reconstruction"/>, registering the accepted ones in place. A frame not in
	/// the reconstruction at all is added as a new image on the camera of its nearest registered
	/// neighbour in time. <paramref name="initialHull"/>, if given, is used for the first pass
	/// instead of carving one. <paramref name="progress"/> gets the fraction of passes done.
	/// </summary>
	public static SilhouettePoseResult Register(
		Reconstruction reconstruction,
		IReadOnlyList<SilhouetteFrame> frames,
		SilhouettePoseOptions options,
		VisualHull? initialHull = null,
		IProgress<double>? progress = null,
		CancellationToken cancellationToken = default)
	{
		int n = frames.Count;
		var reports = new SilhouettePoseFrameReport?[n];
		bool[] wasUnplaced = new bool[n];
		for (int i = 0; i < n; i++)
		{
			wasUnplaced[i] = PoseOf(reconstruction, frames[i].Name) is null;
		}

		SilhouetteConsistencyGate? gate = null;
		if (options.ConsistencyGate)
		{
			List<VisualHullView> references = HullViews(reconstruction, frames);
			if (references.Count > 0)
			{
				gate = new SilhouetteConsistencyGate(
					references, HullBox(reconstruction, options), options.HullOptions,
					options.ConsistencyMedianTolerance, options.ConsistencyMaxTolerance, cancellationToken);
			}
		}

		TurntableFit? turntable = null;
		if (options.Turntable)
		{
			var registered = new List<Rigid3d>();
			foreach (SilhouetteFrame frame in frames)
			{
				if (PoseOf(reconstruction, frame.Name) is Rigid3d pose)
				{
					registered.Add(pose);
				}
			}

			turntable = SilhouetteTurntable.Fit(registered);
		}

		VisualHull? hull = null;
		int passesRun = 0;
		for (int pass = 0; pass < options.MaxPasses; pass++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			var poses = new Rigid3d?[n];
			var todo = new List<int>();
			for (int i = 0; i < n; i++)
			{
				poses[i] = PoseOf(reconstruction, frames[i].Name);
				if (poses[i] is null)
				{
					todo.Add(i);
				}
			}

			if (todo.Count == 0 || todo.Count == n)
			{
				break;
			}

			hull = pass == 0 && initialHull is not null ? initialHull : BuildHull(reconstruction, frames, options, cancellationToken);
			var model = SilhouetteHullModel.FromHull(hull);
			passesRun++;
			if (turntable is { Valid: true })
			{
				int placedHere = TurntablePass(
					reconstruction, frames, reports, model, turntable, gate, pass, options, cancellationToken);
				progress?.Report((pass + 1.0) / options.MaxPasses);
				if (placedHere == 0)
				{
					break;
				}

				continue;
			}

			var outcomes = new (Rigid3d Pose, uint CameraId, SilhouettePoseFrameReport Report)?[todo.Count];
			void Attempt(int t)
			{
				cancellationToken.ThrowIfCancellationRequested();
				int i = todo[t];
				outcomes[t] = AttemptFrame(reconstruction, frames, poses, i, model, pass, options, cancellationToken);
			}

			if (options.Parallel)
			{
				System.Threading.Tasks.Parallel.For(
					0, todo.Count, new ParallelOptions { CancellationToken = cancellationToken }, Attempt);
			}
			else
			{
				for (int t = 0; t < todo.Count; t++)
				{
					Attempt(t);
				}
			}

			int placed = 0;
			for (int t = 0; t < todo.Count; t++)
			{
				int i = todo[t];
				if (outcomes[t] is not { } outcome)
				{
					continue;
				}

				SilhouettePoseFrameReport report = outcome.Report;
				if (report.Placed && gate is not null)
				{
					var view = new VisualHullView(reconstruction.Camera(outcome.CameraId), outcome.Pose, frames[i].Mask);
					if (!gate.TryAccept(view, out double medianDrop, out double maxDrop, cancellationToken))
					{
						report = report with
						{
							Placed = false,
							Reason = $"carving with it lowers the registered frames' hull IoU (median {medianDrop:F4}, worst {maxDrop:F4})",
						};
					}
				}

				reports[i] = report;
				if (report.Placed)
				{
					Place(reconstruction, frames[i].Name, outcome.CameraId, outcome.Pose);
					placed++;
				}
			}

			progress?.Report((pass + 1.0) / options.MaxPasses);
			if (placed == 0)
			{
				break;
			}
		}

		var result = new List<SilhouettePoseFrameReport>();
		for (int i = 0; i < n; i++)
		{
			if (wasUnplaced[i])
			{
				result.Add(reports[i] ?? new SilhouettePoseFrameReport(
					frames[i].Name, 0, false, double.NaN, double.NaN, 0, double.NaN, "no registered frame to start from"));
			}
		}

		return new SilhouettePoseResult(result, passesRun, hull, turntable);
	}

	/// <summary>
	/// The starting pose of frame <paramref name="index"/> from the known poses of its time
	/// neighbours (null where unknown), or null when no frame is known: SLERP of the rotation and
	/// lerp of the camera centre between the nearest known before and after; with one side only,
	/// the motion of the two nearest known on that side, continued by at most
	/// <paramref name="maxExtrapolationRatio"/> times their spacing (a single known frame is
	/// copied). When those two differ by less than <paramref name="minExtrapolationSpanDegrees"/>,
	/// only the frame adjacent to the nearer starts, as its copy; others get null.
	/// </summary>
	public static Rigid3d? InitializePose(
		IReadOnlyList<Rigid3d?> poses, int index, double maxExtrapolationRatio, double minExtrapolationSpanDegrees = 0)
	{
		int before = -1, after = -1;
		for (int i = index - 1; i >= 0 && before < 0; i--)
		{
			before = poses[i] is null ? -1 : i;
		}

		for (int i = index + 1; i < poses.Count && after < 0; i++)
		{
			after = poses[i] is null ? -1 : i;
		}

		if (before >= 0 && after >= 0)
		{
			double t = (index - before) / (double)(after - before);
			return Blend(poses[before]!.Value, poses[after]!.Value, t);
		}

		int near = before >= 0 ? before : after;
		if (near < 0)
		{
			return null;
		}

		int step = before >= 0 ? -1 : 1;
		int far = -1;
		for (int i = near + step; i >= 0 && i < poses.Count && far < 0; i += step)
		{
			far = poses[i] is null ? -1 : i;
		}

		if (far < 0)
		{
			return poses[near];
		}

		// Too short a baseline to trust its direction: only the adjacent frame starts, as a copy.
		double span = poses[near]!.Value.Rotation.AngularDistance(poses[far]!.Value.Rotation) * 180 / Math.PI;
		if (span < minExtrapolationSpanDegrees)
		{
			return Math.Abs(index - near) == 1 ? poses[near] : null;
		}

		// Blend from far to near with t > 1 continues the motion past near.
		double ratio = Math.Min(Math.Abs(index - near) / (double)Math.Abs(near - far), maxExtrapolationRatio);
		return Blend(poses[far]!.Value, poses[near]!.Value, 1 + ratio);
	}

	// Rotation interpolated (t in [0, 1]) or extrapolated (t > 1) along the geodesic from a to b,
	// camera centre along the line between them.
	private static Rigid3d Blend(Rigid3d a, Rigid3d b, double t)
	{
		Quaterniond qa = a.Rotation, qb = b.Rotation;
		if (qa.Dot(qb) < 0)
		{
			qb = new Quaterniond(-qb.W, -qb.X, -qb.Y, -qb.Z);
		}

		AngleAxisd relative = AngleAxisd.FromQuaternion(qb * qa.Inverse());
		Quaterniond q = relative.Angle == 0
			? qa
			: (Quaterniond.FromAngleAxis(new AngleAxisd(relative.Angle * t, relative.Axis)) * qa).Normalized();
		Vector3d ca = a.TgtOriginInSrc(), cb = b.TgtOriginInSrc();
		Vector3d center = ca + (cb - ca) * t;
		return new Rigid3d(q, -(q * center));
	}

	// The prior's sigma grows with the span the start was interpolated across (half of it,
	// so every-other-frame interpolation has scale 1); an extrapolation counts double its
	// distance. Measured on the DarkObject gap: even the frames next to a 20-frame gap's ends
	// start 5 degrees off, so distance to the nearest frame alone would over-trust them.
	private static double PriorScale(Rigid3d?[] poses, int index)
	{
		int before = index - 1, after = index + 1;
		while (before >= 0 && poses[before] is null)
		{
			before--;
		}

		while (after < poses.Length && poses[after] is null)
		{
			after++;
		}

		double scale = before >= 0 && after < poses.Length
			? (after - before) / 2.0
			: before >= 0 ? index - before : after - index;
		return Math.Max(1.0, scale);
	}

	private static (Rigid3d Pose, uint CameraId, SilhouettePoseFrameReport Report)? AttemptFrame(
		Reconstruction reconstruction,
		IReadOnlyList<SilhouetteFrame> frames,
		Rigid3d?[] poses,
		int index,
		SilhouetteHullModel model,
		int pass,
		SilhouettePoseOptions options,
		CancellationToken cancellationToken)
	{
		string name = frames[index].Name;
		Rigid3d? start = InitializePose(poses, index, options.MaxExtrapolationRatio, options.MinExtrapolationSpanDegrees);
		Camera? camera = CameraFor(reconstruction, frames, poses, index);
		if (start is null || camera is null)
		{
			return (default, 0, new SilhouettePoseFrameReport(
				name, pass, false, double.NaN, double.NaN, 0, double.NaN, "no registered frame to start from"));
		}

		SilhouettePoseRefinement refined = SilhouettePoseRefiner.Refine(
			camera, frames[index].Mask, start.Value, model, options, PriorScale(poses, index), cancellationToken);
		double rotation = refined.CamFromWorld.Rotation.AngularDistance(start.Value.Rotation) * 180 / Math.PI;
		string reason = refined.Iou < options.MinIou
			? $"silhouette IoU {refined.Iou:F3} below {options.MinIou:F3}"
			: rotation > options.MaxRotationFromInitDegrees
				? $"rotated {rotation:F1} degrees from its neighbours' motion"
				: "";
		var report = new SilhouettePoseFrameReport(
			name, pass, reason.Length == 0, refined.InitialIou, refined.Iou, refined.Iterations, rotation, reason, start, refined.CamFromWorld);
		return (refined.CamFromWorld, camera.CameraId, report);
	}

	// The frame's own camera if it is in the reconstruction, else its nearest registered time
	// neighbour's (earlier first on a tie).
	private static Camera? CameraFor(Reconstruction reconstruction, IReadOnlyList<SilhouetteFrame> frames, Rigid3d?[] poses, int index)
	{
		if (reconstruction.FindImageWithName(frames[index].Name) is Image own)
		{
			return own.CameraPtr;
		}

		for (int d = 1; d < frames.Count; d++)
		{
			foreach (int j in new[] { index - d, index + d })
			{
				if (j >= 0 && j < frames.Count && poses[j] is not null
					&& reconstruction.FindImageWithName(frames[j].Name) is Image image)
				{
					return image.CameraPtr;
				}
			}
		}

		return null;
	}

	private static Rigid3d? PoseOf(Reconstruction reconstruction, string name) =>
		reconstruction.FindImageWithName(name) is Image image && image.HasPose
			? image.CamFromWorld()
			: null;

	private static VisualHull BuildHull(
		Reconstruction reconstruction, IReadOnlyList<SilhouetteFrame> frames, SilhouettePoseOptions options, CancellationToken cancellationToken)
	{
		List<VisualHullView> views = HullViews(reconstruction, frames);
		return VisualHull.Build(views, HullBox(reconstruction, options), options.HullOptions, null, cancellationToken);
	}

	// Every registered frame that has a mask, as a carving view, in time order.
	private static List<VisualHullView> HullViews(Reconstruction reconstruction, IReadOnlyList<SilhouetteFrame> frames)
	{
		var views = new List<VisualHullView>();
		foreach (SilhouetteFrame frame in frames)
		{
			if (PoseOf(reconstruction, frame.Name) is Rigid3d pose)
			{
				views.Add(new VisualHullView(reconstruction.FindImageWithName(frame.Name)!.CameraPtr, pose, frame.Mask));
			}
		}

		return views;
	}

	private static AlignedBox3d HullBox(Reconstruction reconstruction, SilhouettePoseOptions options)
	{
		if (options.HullBox is AlignedBox3d given)
		{
			return given;
		}

		Check.That(reconstruction.Points3D.Count > 0, "No hull box given and the reconstruction has no 3D points to bound one");
		return VisualHullBounds.FromPoints(reconstruction.Points3D.Values.Select(p => p.Xyz).ToList());
	}

	// Registers `name` with `camFromWorld`: its existing image's frame if it has one, else a new
	// image (and trivial frame) on `cameraId`.
	private static void Place(Reconstruction reconstruction, string name, uint cameraId, Rigid3d camFromWorld)
	{
		if (reconstruction.FindImageWithName(name) is Image existing)
		{
			Frame frame = existing.FramePtr;
			frame.SetCamFromWorld(existing.CameraId, camFromWorld);
			reconstruction.RegisterFrame(frame.FrameId);
			return;
		}

		uint id = 1;
		foreach (uint imageId in reconstruction.Images.Keys)
		{
			id = Math.Max(id, imageId + 1);
		}

		foreach (uint frameId in reconstruction.Frames.Keys)
		{
			id = Math.Max(id, frameId + 1);
		}

		var image = new Image { ImageId = id, Name = name };
		image.SetCameraId(cameraId);
		reconstruction.AddImageWithTrivialFrame(image, camFromWorld);
	}
}
