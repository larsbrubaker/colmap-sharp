// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionController, object part: what AutomaticReconstructionOptions.Subject =
// Object adds to the pipeline (docs/QUALITY_PLAN.md, stage 3b; divergence 142). Not a COLMAP
// port; COLMAP has no single-object mode. With Scene (the default) none of this runs.
//
// - Masks: the host's AutomaticReconstructionOptions.Masks, or else SilhouetteSegmenter's
//   (TemporalWindow 2 for video, whose neighbouring frames vote on each other's masks), from the
//   selected images only (ImageNames, when set). They reach feature extraction
//   (ImageReaderOptions.Masks) and fusion (StereoFusionOptions.MaskPath) through the same
//   plumbing as host masks. Segmentation reports its progress under the stage that first needs
//   the masks (feature extraction, or the dense stage on a resumed workspace), with Message =
//   SegmentationMessage.
// - After fusion, each model gets a visual hull (Mvs/Silhouette/VisualHull.cs) from its
//   registered images' cameras and masks, in the box VisualHullBounds puts round the sparse
//   points, at ObjectHullResolution voxels with k = ObjectHullTolerance.
// - Poisson meshes fused.ply plus hull samples where the cloud has a gap
//   (HullSurfaceFusion.GapSamples, written to fused-hull.ply), without its density trim, so its
//   surface is watertight. HullSurfaceFusion.CleanUp then drops the pieces wholly outside the
//   silhouettes and pulls the rest inside the hull, without opening the surface. Delaunay
//   meshing reads fused.ply.vis, which has no entries for hull samples, so it meshes fused.ply
//   alone and gets only the clean-up. Advancing-front meshing is skipped (CGAL), as in a scene.
// - When the dense stages give no mesh (no fused points, Poisson failed, advancing front, or
//   nothing is left) the hull mesh itself is the model's mesh, so the user always gets a closed
//   shape.
// - The mesher writes to <mesh name>.partial.ply, and only the finished object mesh is renamed
//   onto the mesh path, so a run stopped in between leaves no mesh a resume would take for a
//   finished one (divergence 135's rule).
// - With no registered model there are no cameras and so no hull: the reconstruction fails the
//   way a scene's does (no model), and object mode adds nothing.
// Texturing then runs on whichever mesh resulted (AutomaticReconstruction.Texture.cs).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;
using ColmapSharp.Segmentation;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

public sealed partial class AutomaticReconstructionController
{
	/// <summary>Voxels along the hull box's longest side in object mode.</summary>
	public const int ObjectHullResolution = 128;

	/// <summary>
	/// The hull's disagreement tolerance k in object mode, and the clean-up's: 1 measured best on
	/// the real mouse capture (silhouette IoU 0.966 against 0.945 with k = 0), where a single bad
	/// mask or pose would otherwise bite the object.
	/// </summary>
	public const int ObjectHullTolerance = 1;

	/// <summary>
	/// How far (in hull voxels) a hull vertex must be from every fused point before it is added as
	/// a gap sample, and how close to the hull a mesh piece must come to count as the object rather
	/// than a floater: a few voxels, so that the hull's staircase and smoothing (under a voxel) and
	/// the fused cloud's noise never put a hull sample beside a measured point.
	/// </summary>
	public const double ObjectGapVoxels = 3;

	/// <summary>
	/// The deepest Poisson octree object mode uses: four times the hull's resolution
	/// (2^9 = 512 = 4 x 128). The hull samples cover the whole surface, so unlike a scene's partial
	/// cloud every level refines everywhere; at the demo's depth 11 a 320x240 sphere spent over
	/// seven minutes in Poisson's density estimation, and 512 cells across the object are already
	/// finer than the photos resolve it.
	/// </summary>
	public const int ObjectMaxPoissonDepth = 9;

	/// <summary>The file, next to fused.ply, that holds the fused points plus the hull's gap samples.</summary>
	public const string HullFilledCloudName = "fused-hull.ply";

	/// <summary>The file, next to fused.ply, that holds each model's visual hull mesh.</summary>
	public const string VisualHullMeshName = "visual-hull.ply";

	/// <summary>The Message of the progress reports of object mode's silhouette segmentation.</summary>
	public const string SegmentationMessage = "Finding the object in each photo";

	private IImageSource? masks;

	private bool IsObject => options.Subject == AutomaticReconstructionOptions.SubjectType.Object;

	// The hull options object mode builds with. The clean-up reads the same smoothing and iso
	// level, so "outside the hull" agrees with the surface vertices are pulled onto.
	private static VisualHullOptions ObjectHullOptions() => new()
	{
		Resolution = ObjectHullResolution,
		DisagreementTolerance = ObjectHullTolerance,
	};

	// The masks the stages use: the host's, or in object mode the segmenter's (computed once, its
	// progress reported under progressStage).
	private IImageSource? ResolveMasks(string progressStage)
	{
		if (masks is not null || options.Masks is not null || !IsObject)
		{
			return masks ??= options.Masks;
		}

		var segmentation = new SegmentationOptions
		{
			TemporalWindow = options.Data == AutomaticReconstructionOptions.DataType.Video ? 2 : 0,
		};

		// Only the selected images, when the host selected some: the others are never read.
		IImageSource images = options.ImageNames.Count == 0 ? options.Images! : new SelectedImages(options.Images!, options.ImageNames);
		int total = images.ListNames().Count;
		masks = SilhouetteSegmenter.SegmentToMaskSource(
			images,
			segmentation,
			Forward<int>(done => new ControllerProgress(progressStage, done, total, SegmentationMessage)),
			CancellationToken);
		return masks;
	}

	// The host's image source restricted to the selected names (ImageNames).
	private sealed class SelectedImages(IImageSource images, IReadOnlyList<string> names) : IImageSource
	{
		private readonly HashSet<string> selected = new(names, StringComparer.Ordinal);

		public IReadOnlyList<string> ListNames() => [.. images.ListNames().Where(selected.Contains)];

		public bool Exists(string name) => selected.Contains(name) && images.Exists(name);

		public Bitmap? Read(string name) => selected.Contains(name) ? images.Read(name) : null;
	}

	// A model's hull, the views it was carved from, and the occupancy its surface was taken from.
	private sealed record ObjectHull(VisualHull Hull, List<VisualHullView> Views, OccupancyGrid SurfaceGrid, double IsoLevel);

	// Object mode's meshing of model i, in place of RunMeshing: hull, gap filling, Poisson (or
	// Delaunay) into a partial file, clean-up or the hull fallback, then the rename onto
	// meshingPath. Returns false where RunDenseMapper's loop returns (once stopped).
	private bool RunObjectMeshing(int modelIdx, string densePath, string fusedPath, string meshingPath)
	{
		if (CheckIfStopped())
		{
			return false;
		}

		if (File.Exists(meshingPath))
		{
			return true;
		}

		ObjectHull? hull = BuildObjectHull(modelIdx, densePath);
		if (hull is null)
		{
			return RunMeshing(densePath, fusedPath, meshingPath);
		}

		string partialPath = Path.Combine(densePath, Path.GetFileNameWithoutExtension(meshingPath) + ".partial.ply");
		File.Delete(partialPath);
		try
		{
			if (options.Mesher == AutomaticReconstructionOptions.MesherType.AdvancingFront)
			{
				Log.Warning("Skipping advancing front meshing because CGAL is not available");
			}
			else
			{
				string meshInput = options.Mesher == AutomaticReconstructionOptions.MesherType.Poisson
					? WriteHullFilledCloud(densePath, fusedPath, hull.Hull)
					: fusedPath;
				if (!RunMeshing(densePath, meshInput, partialPath, ObjectPoissonOptions()) && CheckIfStopped())
				{
					File.Delete(partialPath);
					return false;
				}
			}

			CancellationToken.ThrowIfCancellationRequested();
			FinishObjectMesh(partialPath, hull);
			CancellationToken.ThrowIfCancellationRequested();
			File.Move(partialPath, meshingPath, overwrite: true);
		}
		catch
		{
			File.Delete(partialPath);
			throw;
		}

		return true;
	}

	// The silhouette views of model i's registered images that have a mask, in image id order.
	private List<VisualHullView> HullViews(Reconstruction reconstruction)
	{
		IImageSource? source = ResolveMasks(DenseStage);
		var views = new List<VisualHullView>();
		if (source is null)
		{
			return views;
		}

		List<uint> ids = reconstruction.RegImageIds();
		ids.Sort();
		foreach (uint id in ids)
		{
			Scene.Image image = reconstruction.Image(id);
			Camera camera = reconstruction.Camera(image.CameraId);
			Bitmap? mask = source.Read(SilhouetteSegmenter.MaskName(image.Name));
			if (mask is null || mask.IsEmpty)
			{
				continue;
			}

			if (mask.Width != camera.Width || mask.Height != camera.Height)
			{
				// A copy: the source may hand out its own bitmap.
				mask = mask.CloneAsGrey();
				mask.Rescale((int)camera.Width, (int)camera.Height);
			}

			views.Add(new VisualHullView(camera, image.CamFromWorld(), mask));
		}

		return views;
	}

	// Model i's visual hull, written to dense/<i>/visual-hull.ply, or null (with a warning) when
	// the model has no 3D points or too few masked views to carve.
	private ObjectHull? BuildObjectHull(int modelIdx, string densePath)
	{
		Reconstruction reconstruction = reconstructionManager.Get(modelIdx);
		List<VisualHullView> views = HullViews(reconstruction);
		if (reconstruction.NumPoints3D == 0 || views.Count <= ObjectHullTolerance)
		{
			Log.Warning("Couldn't find the object's outline in enough photos, so its shape comes from the matched points alone. "
				+ "Make sure the object stands out from the background.");
			return null;
		}

		List<Vector3d> points = [.. reconstruction.Points3D.OrderBy(p => p.Key).Select(p => p.Value.Xyz)];
		VisualHullOptions hullOptions = ObjectHullOptions();
		VisualHull hull = VisualHull.Build(views, VisualHullBounds.FromPoints(points), hullOptions, null, CancellationToken);
		if (hull.Mesh.Faces.Count == 0)
		{
			Log.Warning("The object's outlines in the photos don't agree on a shape, so its shape comes from the matched points alone. "
				+ "Make sure the object stands out from the background and stays whole in every photo.");
			return null;
		}

		Ply.WriteBinaryPlyMesh(Path.Combine(densePath, VisualHullMeshName), new PlyTexturedMesh(hull.Mesh));
		return new ObjectHull(hull, views, hull.Grid.Smoothed(hullOptions.SmoothingPasses), hullOptions.IsoLevel);
	}

	// fused.ply plus the hull's gap samples, written to fused-hull.ply; returns that path.
	private static string WriteHullFilledCloud(string densePath, string fusedPath, VisualHull hull)
	{
		List<PlyPoint> fused = Ply.ReadPly(fusedPath);
		List<Vector3d> cloud = [.. fused.Select(p => new Vector3d(p.X, p.Y, p.Z))];
		List<PlyPoint> samples = HullSurfaceFusion.GapSamples(hull.Mesh, cloud, ObjectGapVoxels * hull.Grid.VoxelSize);
		string path = Path.Combine(densePath, HullFilledCloudName);
		Ply.WriteBinaryPlyPoints(path, [.. fused, .. samples]);
		return path;
	}

	// The Poisson options of object mode: the silhouette clean-up replaces the density trim, and
	// the depth is capped (ObjectMaxPoissonDepth).
	private PoissonMeshingOptions ObjectPoissonOptions()
	{
		PoissonMeshingOptions source = optionManager.PoissonMeshing;
		return new PoissonMeshingOptions
		{
			PointWeight = source.PointWeight,
			Depth = Math.Min(source.Depth, ObjectMaxPoissonDepth),
			Color = source.Color,
			Trim = 0,
			NumThreads = source.NumThreads,
		};
	}

	// After meshing in object mode: clean the mesh at meshPath up (HullSurfaceFusion.CleanUp) in
	// place, or, when there is no mesh or nothing is left, write the hull mesh there instead.
	private static void FinishObjectMesh(string meshPath, ObjectHull hull)
	{
		PlyMesh? mesh = File.Exists(meshPath) && Ply.HasPlyMeshFaces(meshPath) ? Ply.ReadPlyMesh(meshPath).Mesh : null;
		if (mesh is not null && mesh.Faces.Count > 0)
		{
			PlyMesh cleaned = HullSurfaceFusion.CleanUp(
				mesh, hull.Hull.Mesh, hull.SurfaceGrid, hull.IsoLevel, hull.Views, ObjectHullTolerance, ObjectGapVoxels * hull.Hull.Grid.VoxelSize);
			if (cleaned.Faces.Count > 0)
			{
				Ply.WriteBinaryPlyMesh(meshPath, new PlyTexturedMesh(cleaned));
				return;
			}
		}

		Log.Warning("Couldn't build a detailed surface from the photos, so the result is the object's outline shape. "
			+ "More photos from more angles, with the object in sharp focus, give a more detailed result.");
		Ply.WriteBinaryPlyMesh(meshPath, new PlyTexturedMesh(hull.Hull.Mesh));
	}
}
