// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteConsistencyGate: the last check before SilhouettePoseRegistration.cs registers a
// silhouette-placed frame (docs/QUALITY_PLAN.md, stage 4b). Not a COLMAP port. It is the global
// form of silhouette coherence (Hernandez, Schmitt and Cipolla, PAMI 2007): a view at a wrong
// pose carves away object that the correctly placed views see, so their silhouettes stop being
// explained by the hull. The reference views are the feature-registered frames, whose poses
// are trusted; their hull IoU is measured once without any silhouette-placed frame, and a
// candidate passes only if carving with it (and every candidate accepted before it) keeps the
// median drop within MedianTolerance and every single frame's drop within MaxTolerance.
// TryAccept adds a passing candidate to the accepted set.
//
// Measured on the mouse (f40): without the gate, frames placed by silhouette alone lowered the
// feature-registered frames' hull IoU from 0.91-0.98 to 0.87-0.94 while their own IoU looked fine.
//
// Cost: with DisagreementTolerance 0 the hull is the per-voxel minimum over views, so the gate
// keeps the accepted-so-far occupancy grid and a candidate only takes the minimum with its own
// mask value at each still-occupied voxel center (HullMaskView.PointValue; a view that does not
// see a voxel leaves it alone), which is what VisualHull.Carve gives for the same views. Each
// check is then one pass over the grid plus a surface extraction, not a full carve of every view.
// With a tolerance above 0 the order statistic does not combine that way, and each check carves
// all views again.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>Accepts a candidate view only if it keeps the reference views' silhouettes explained.</summary>
public sealed class SilhouetteConsistencyGate
{
	private readonly IReadOnlyList<VisualHullView> _references;
	private readonly List<VisualHullView> _accepted = [];
	private readonly AlignedBox3d _box;
	private readonly VisualHullOptions _hullOptions;
	private readonly double[] _baseline;
	private OccupancyGrid _grid;

	/// <summary>
	/// A gate over <paramref name="references"/> (trusted poses), carving in
	/// <paramref name="box"/> with <paramref name="hullOptions"/>. Measures the baseline IoUs.
	/// </summary>
	public SilhouetteConsistencyGate(
		IReadOnlyList<VisualHullView> references,
		AlignedBox3d box,
		VisualHullOptions hullOptions,
		double medianTolerance = 0.005,
		double maxTolerance = 0.02,
		CancellationToken cancellationToken = default)
	{
		Check.That(references.Count > 0, "The gate needs reference views");
		_references = references;
		_box = box;
		_hullOptions = hullOptions;
		MedianTolerance = medianTolerance;
		MaxTolerance = maxTolerance;
		_grid = VisualHull.Carve(references, box, hullOptions, null, cancellationToken);
		_baseline = ReferenceIous(_grid);
	}

	/// <summary>The largest allowed drop of the median reference IoU.</summary>
	public double MedianTolerance { get; }

	/// <summary>The largest allowed drop of any single reference IoU.</summary>
	public double MaxTolerance { get; }

	/// <summary>The reference views' hull IoU with no extra view.</summary>
	public IReadOnlyList<double> Baseline => _baseline;

	/// <summary>The candidates accepted so far, in order.</summary>
	public IReadOnlyList<VisualHullView> Accepted => _accepted;

	/// <summary>
	/// Whether carving with the accepted views plus <paramref name="candidate"/> keeps the
	/// reference IoUs within tolerance of the baseline; if so the candidate joins the accepted
	/// set. <paramref name="medianDrop"/> and <paramref name="maxDrop"/> are the measured drops
	/// (negative: the IoU rose).
	/// </summary>
	public bool TryAccept(
		VisualHullView candidate,
		out double medianDrop,
		out double maxDrop,
		CancellationToken cancellationToken = default)
	{
		OccupancyGrid grid = _hullOptions.DisagreementTolerance == 0
			? WithView(_grid, candidate, _hullOptions.OffImageIsOutside)
			: VisualHull.Carve([.. _references, .. _accepted, candidate], _box, _hullOptions, null, cancellationToken);
		double[] ious = ReferenceIous(grid);
		var drops = new double[ious.Length];
		for (int i = 0; i < ious.Length; i++)
		{
			drops[i] = _baseline[i] - ious[i];
		}

		maxDrop = drops.Max();
		double[] sorted = [.. drops];
		Array.Sort(sorted);
		medianDrop = sorted.Length % 2 == 1
			? sorted[sorted.Length / 2]
			: 0.5 * (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]);
		if (medianDrop > MedianTolerance || maxDrop > MaxTolerance)
		{
			return false;
		}

		_grid = grid;
		_accepted.Add(candidate);
		return true;
	}

	// A copy of `grid` carved by one more view (k = 0: the per-voxel minimum).
	private static OccupancyGrid WithView(OccupancyGrid grid, VisualHullView view, bool offImageIsOutside)
	{
		var result = new OccupancyGrid(grid.Origin, grid.VoxelSize, grid.NX, grid.NY, grid.NZ);
		var mask = new HullMaskView(view, offImageIsOutside);
		for (int k = 0; k < grid.NZ; k++)
		{
			for (int j = 0; j < grid.NY; j++)
			{
				for (int i = 0; i < grid.NX; i++)
				{
					int index = grid.Index(i, j, k);
					float value = grid.Values[index];
					if (value > 0 && mask.PointValue(grid.Center(i, j, k)) is double seen)
					{
						value = Math.Min(value, (float)seen);
					}

					result.Values[index] = value;
				}
			}
		}

		return result;
	}

	// The references' IoU against the surface of `grid`, extracted as VisualHull.Build does.
	private double[] ReferenceIous(OccupancyGrid grid)
	{
		PlyMesh mesh = HullMarchingCubes.Extract(grid.Smoothed(_hullOptions.SmoothingPasses), _hullOptions.IsoLevel);
		var model = SilhouetteHullModel.FromMesh(mesh);
		var ious = new double[_references.Count];
		for (int i = 0; i < ious.Length; i++)
		{
			VisualHullView view = _references[i];
			ious[i] = SilhouettePoseRefiner.Iou(view.Camera, view.Mask, view.CamFromWorld, model);
		}

		return ious;
	}
}
