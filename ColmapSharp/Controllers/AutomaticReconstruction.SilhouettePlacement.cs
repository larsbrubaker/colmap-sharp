// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionController, silhouette placement part: what
// AutomaticReconstructionOptions.SilhouettePlacement adds to an Object-mode run (divergence 144).
// Not a COLMAP port; COLMAP registers an image only through 2D-3D correspondences.
//
// At the end of the sparse stage (RunStages), after the last mapper and bundle adjustment pass,
// each sparse model goes through Sfm/Silhouette/SilhouettePoseRegistration.Register with:
// - frames: every selected photo (ImageNames, or all of Images) that has an object mask, in name
//   order, which is the order they were filmed in (the options must say the frames are
//   time-ordered: FramesAreTimeOrdered, or Data = Video); the model's registered frames among
//   them are the anchors, the rest are the frames to place. Masks are Object mode's
//   (AutomaticReconstruction.Object.cs: the host's or the segmenter's), scaled to the camera.
// - the hull: carved by Register from the model's registered frames, with the same options and
//   box as Object mode's meshing hull (ObjectHullOptions, VisualHullBounds round the sparse
//   points in point id order), so a frame is placed against the shape that is later meshed.
// The updated model is written back over sparse/<i>, so the dense stage (undistortion,
// PatchMatch, fusion), the meshing hull and texturing all see the placed frames, and a resume
// reads them back.
//
// Resume marker: sparse/<i>/silhouette-placement.txt (SilhouettePlacementMarkerName), written
// after the model, holds the summary line. A model directory that has it is not placed again,
// so a resumed run neither repeats the work nor tries the rejected frames a second time against
// a hull that now includes the placed ones (which could place more and change the model under
// an existing dense/<i>). A run stopped before the marker is written places the model again
// from the sparse/<i> on disk; one stopped while sparse/<i> is being rewritten leaves that model
// as the file writes left it, as any stop during Reconstruction.Write would.
//
// Downstream, a placed frame has a pose and no 2D-3D points: PatchMatch finds no source images
// for it and skips it with a warning (PatchMatchController), fusion skips its missing depth map
// (Fusion.cs), and the meshing hull and texturing use it like any registered frame.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm.Silhouette;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

public sealed partial class AutomaticReconstructionController
{
	/// <summary>
	/// The file in sparse/&lt;i&gt; that marks a model as already through silhouette placement
	/// (its content is the placement's summary line). A model with it is not placed again.
	/// </summary>
	public const string SilhouettePlacementMarkerName = "silhouette-placement.txt";

	/// <summary>The Message of the progress report that starts silhouette placement (Stage = SparseStage).</summary>
	public const string SilhouettePlacementMessage = "Placing more photos from their outlines";

	// The directory each model was read from or written to, by model index (RunSparseMapper).
	private IReadOnlyList<string> sparseModelDirs = [];

	// Placement needs Object mode's masks and hull, and runs at the end of the sparse stage.
	// Refuse any other combination up front rather than silently skipping it.
	private static void ThrowIfSilhouettePlacementCannotRun(AutomaticReconstructionOptions options)
	{
		if (!options.SilhouettePlacement)
		{
			return;
		}

		if (options.Subject != AutomaticReconstructionOptions.SubjectType.Object)
		{
			throw new ArgumentException(
				"Placing photos from their outlines (SilhouettePlacement) works only when capturing a single object. "
				+ "Set Subject to Object, or turn SilhouettePlacement off.",
				nameof(options));
		}

		// Starting poses are interpolated between neighbours in filming order.
		if (!AreFramesTimeOrdered(options))
		{
			throw new ArgumentException(
				"Placing photos from their outlines (SilhouettePlacement) needs the frames of one video, named in the order they were filmed. "
				+ "Set FramesAreTimeOrdered to true (or Data to Video), or turn SilhouettePlacement off.",
				nameof(options));
		}

		if (!options.Sparse)
		{
			throw new ArgumentException(
				"Placing photos from their outlines (SilhouettePlacement) runs as part of sparse reconstruction. "
				+ "Turn Sparse on, or turn SilhouettePlacement off.",
				nameof(options));
		}
	}

	private void RunSilhouettePlacement()
	{
		for (int i = 0; i < reconstructionManager.Size && i < sparseModelDirs.Count; ++i)
		{
			if (CheckIfStopped())
			{
				return;
			}

			string modelDir = sparseModelDirs[i];
			string markerPath = Path.Combine(modelDir, SilhouettePlacementMarkerName);
			if (File.Exists(markerPath))
			{
				continue;
			}

			Progress?.Report(new ControllerProgress(SparseStage, 0, 0, SilhouettePlacementMessage));
			string summary = PlaceBySilhouette(reconstructionManager.Get(i));
			if (reconstructionManager.Size > 1)
			{
				summary = $"Model {i + 1}: {summary}";
			}

			Progress?.Report(new ControllerProgress(SparseStage, 0, 0, summary));

			// The model first, then the marker, so a stop in between places it again on resume.
			CancellationToken.ThrowIfCancellationRequested();
			reconstructionManager.Get(i).Write(modelDir);
			File.WriteAllText(markerPath, summary + Environment.NewLine);
		}
	}

	// Places the unregistered frames of `reconstruction` in place; returns the plain summary.
	private string PlaceBySilhouette(Reconstruction reconstruction)
	{
		List<SilhouetteFrame> frames = PlacementFrames(reconstruction);
		int registered = frames.Count(f => reconstruction.FindImageWithName(f.Name) is { HasPose: true });
		int unplaced = frames.Count - registered;
		if (unplaced == 0)
		{
			return "Every photo with an outline was already placed.";
		}

		if (reconstruction.NumPoints3D == 0 || registered <= ObjectHullTolerance)
		{
			return $"Couldn't place more photos from their outlines: too few photos were placed to build the object's shape ({unplaced} not placed).";
		}

		List<Vector3d> points = [.. reconstruction.Points3D.OrderBy(p => p.Key).Select(p => p.Value.Xyz)];
		var poseOptions = new SilhouettePoseOptions
		{
			HullOptions = ObjectHullOptions(),
			HullBox = VisualHullBounds.FromPoints(points),
		};
		SilhouettePoseResult result = SilhouettePoseRegistration.Register(
			reconstruction, frames, poseOptions, null, null, CancellationToken);
		int placed = result.PlacedCount;
		return $"Placed {placed} more {(placed == 1 ? "photo" : "photos")} from their outlines ({unplaced - placed} couldn't be placed)";
	}

	// Every selected photo with a non-empty mask, in name order, its mask at its camera's size:
	// its own camera when it is an image of the model, else that of the nearest registered photo
	// in name order (the camera SilhouettePoseRegistration adds it on).
	private List<SilhouetteFrame> PlacementFrames(Reconstruction reconstruction)
	{
		IImageSource? maskSource = ResolveMasks(SparseStage);
		var frames = new List<SilhouetteFrame>();
		if (maskSource is null)
		{
			return frames;
		}

		IReadOnlyList<string> selected = options.ImageNames.Count == 0 ? options.Images!.ListNames() : options.ImageNames;
		List<string> names = [.. selected.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)];
		var named = new List<(string Name, Bitmap Mask)>();
		foreach (string name in names)
		{
			Bitmap? mask = maskSource.Read(SilhouetteSegmenter.MaskName(name));
			if (mask is not null && !mask.IsEmpty)
			{
				named.Add((name, mask));
			}
		}

		for (int k = 0; k < named.Count; k++)
		{
			Camera? camera = PlacementCamera(reconstruction, named, k);
			if (camera is null)
			{
				continue;
			}

			Bitmap mask = named[k].Mask;
			if (mask.Width != camera.Width || mask.Height != camera.Height || mask.Channels != 1)
			{
				// A copy: the source may hand out its own bitmap.
				mask = mask.CloneAsGrey();
				mask.Rescale((int)camera.Width, (int)camera.Height);
			}

			frames.Add(new SilhouetteFrame(named[k].Name, mask));
		}

		return frames;
	}

	private static Camera? PlacementCamera(Reconstruction reconstruction, List<(string Name, Bitmap Mask)> named, int k)
	{
		if (reconstruction.FindImageWithName(named[k].Name) is Scene.Image own)
		{
			return own.CameraPtr;
		}

		for (int d = 1; d < named.Count; d++)
		{
			foreach (int j in new[] { k - d, k + d })
			{
				if (j >= 0 && j < named.Count && reconstruction.FindImageWithName(named[j].Name) is { HasPose: true } image)
				{
					return image.CameraPtr;
				}
			}
		}

		return null;
	}
}
