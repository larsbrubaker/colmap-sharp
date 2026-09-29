// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHull: an octree-carved visual hull from silhouettes, with a marching-cubes surface
// (docs/QUALITY_PLAN.md, stage 3a). Not a COLMAP port; COLMAP has no visual hull. Written from
// Laurentini 1994 (the visual hull), Szeliski 1993 (octree carving) and Kutulakos and Seitz 2000
// (carving); full citations in VisualHullOptions.cs. The mask lookups are HullMaskView.cs, the
// output grid OccupancyGrid.cs, the surface HullMarchingCubes.cs, the box helper
// VisualHullBounds.cs.
//
// The occupancy of a point, with k = DisagreementTolerance: of the views that see the point (in
// front of the camera and on the image) take the mask value at its projection (bilinear, in
// [0, 1]); the occupancy is the (k+1)-th smallest of those values, and 0 when k or fewer views
// see it. With k = 0 that is the minimum, the classic intersection of silhouette cones. With
// k = 1 a single view that says "outside" is outvoted. Two costs follow, both documented on the
// options: a view does not constrain what it cannot see, so parts of the box few views see stay
// filled and the hull inflates toward the box when views crop the scene (OffImageIsOutside
// counts off-image as outside instead, for captures that keep the object in frame); and with
// k >= 1 a part seen by only k views is carved even where they say inside.
//
// Carving evaluates that function only near the surface. The box is cut into blocks, each the
// root of an octree (Szeliski 1993). A cell is classified per view (HullMaskView.Classify):
// - outside if more than k views see all of it outside their masks;
// - outside if k or fewer views could see any of it;
// - inside if at least k+1 views see all of it inside their masks and at most k views are
//   anything but Inside or Unseen;
// - otherwise it splits in two along each axis, down to single voxels, which are evaluated at
//   their centers. For a pinhole camera the cell tests are conservative, so the result equals
//   evaluating every voxel center (VisualHullCarvingTests checks this against CarveByPoints); with
//   lens distortion the one-pixel footprint growth covers the bowing of projected cell edges in
//   the measured cases, which the same test also checks.
// A view's classification is inherited: once Inside, Outside or Unseen for a cell, it is so for
// every child, so only the Mixed views are re-tested below it.
//
// Threading: the blocks are carved in parallel, each writing only its own voxels, so sequential
// and parallel runs give identical grids. The surface is extracted sequentially.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>
/// A visual hull: the occupancy grid at the finest level and its closed, outward-wound
/// iso-surface mesh (world coordinates).
/// </summary>
public sealed class VisualHull
{
	// Octree roots are blocks of this many voxels per side, which gives enough parallel work
	// (512 blocks at 128^3) while keeping each root's first classification cheap.
	private const int BlockSize = 16;

	private VisualHull(OccupancyGrid grid, PlyMesh mesh)
	{
		Grid = grid;
		Mesh = mesh;
	}

	/// <summary>The unsmoothed occupancy at the finest level, values in [0, 1].</summary>
	public OccupancyGrid Grid { get; }

	/// <summary>The iso-surface of the smoothed occupancy: closed and wound outward.</summary>
	public PlyMesh Mesh { get; }

	/// <summary>
	/// Carves the visual hull of <paramref name="views"/> inside <paramref name="box"/> and
	/// extracts its surface. <paramref name="progress"/> gets the carved fraction in [0, 1].
	/// </summary>
	public static VisualHull Build(
		IReadOnlyList<VisualHullView> views,
		AlignedBox3d box,
		VisualHullOptions options,
		IProgress<double>? progress = null,
		CancellationToken cancellationToken = default)
	{
		OccupancyGrid grid = Carve(views, box, options, progress, cancellationToken);
		OccupancyGrid smoothed = grid.Smoothed(options.SmoothingPasses);
		cancellationToken.ThrowIfCancellationRequested();
		PlyMesh mesh = HullMarchingCubes.Extract(smoothed, options.IsoLevel);
		return new VisualHull(grid, mesh);
	}

	/// <summary>The carving alone: the occupancy grid of <see cref="Build"/>, without a surface.</summary>
	public static OccupancyGrid Carve(
		IReadOnlyList<VisualHullView> views,
		AlignedBox3d box,
		VisualHullOptions options,
		IProgress<double>? progress = null,
		CancellationToken cancellationToken = default)
	{
		(OccupancyGrid grid, HullMaskView[] masks) = Prepare(views, box, options);
		int nx = grid.NX, ny = grid.NY, nz = grid.NZ;
		int bx = (nx + BlockSize - 1) / BlockSize, by = (ny + BlockSize - 1) / BlockSize;
		int bz = (nz + BlockSize - 1) / BlockSize;
		int blocks = bx * by * bz;
		int done = 0;
		void CarveBlock(int b)
		{
			cancellationToken.ThrowIfCancellationRequested();
			int i0 = b % bx * BlockSize, j0 = b / bx % by * BlockSize, k0 = b / (bx * by) * BlockSize;
			// One carver per block: its buffers are the worker's own.
			new Carver(grid, masks, options.DisagreementTolerance).CarveRoot(
				i0, Math.Min(nx, i0 + BlockSize), j0, Math.Min(ny, j0 + BlockSize), k0, Math.Min(nz, k0 + BlockSize));
			progress?.Report((double)Interlocked.Increment(ref done) / blocks);
		}

		if (options.Parallel)
		{
			System.Threading.Tasks.Parallel.For(
				0, blocks, new ParallelOptions { CancellationToken = cancellationToken }, CarveBlock);
		}
		else
		{
			for (int b = 0; b < blocks; b++)
			{
				CarveBlock(b);
			}
		}

		return grid;
	}

	/// <summary>
	/// The reference for <see cref="Carve"/>: every voxel center evaluated on its own, no octree.
	/// Slow; for tests.
	/// </summary>
	internal static OccupancyGrid CarveByPoints(IReadOnlyList<VisualHullView> views, AlignedBox3d box, VisualHullOptions options)
	{
		(OccupancyGrid grid, HullMaskView[] masks) = Prepare(views, box, options);
		var carver = new Carver(grid, masks, options.DisagreementTolerance);
		var all = new int[masks.Length];
		for (int v = 0; v < all.Length; v++)
		{
			all[v] = v;
		}

		for (int k = 0; k < grid.NZ; k++)
		{
			for (int j = 0; j < grid.NY; j++)
			{
				for (int i = 0; i < grid.NX; i++)
				{
					grid.Values[grid.Index(i, j, k)] = carver.LeafValue(grid.Center(i, j, k), all, all.Length, 0, 0);
				}
			}
		}

		return grid;
	}

	private static (OccupancyGrid Grid, HullMaskView[] Masks) Prepare(
		IReadOnlyList<VisualHullView> views, AlignedBox3d box, VisualHullOptions options)
	{
		Check.NotNull(views);
		Check.NotNull(options);
		Check.That(options.Resolution > 0);
		Check.That(options.DisagreementTolerance >= 0);
		Vector3d extent = box.Diagonal();
		Check.That(extent.X > 0 && extent.Y > 0 && extent.Z > 0, "The bounding box is empty");
		double voxel = Math.Max(extent.X, Math.Max(extent.Y, extent.Z)) / options.Resolution;
		int nx = Math.Max(1, (int)Math.Ceiling(extent.X / voxel - 1e-9));
		int ny = Math.Max(1, (int)Math.Ceiling(extent.Y / voxel - 1e-9));
		int nz = Math.Max(1, (int)Math.Ceiling(extent.Z / voxel - 1e-9));
		var masks = new HullMaskView[views.Count];
		for (int v = 0; v < masks.Length; v++)
		{
			masks[v] = new HullMaskView(views[v], options.OffImageIsOutside);
		}

		return (new OccupancyGrid(box.Min, voxel, nx, ny, nz), masks);
	}

	/// <summary>The recursive octree carving over one grid; stateless apart from the grid it fills.</summary>
	private sealed class Carver(OccupancyGrid grid, HullMaskView[] masks, int tolerance)
	{
		// Scratch reused across the recursion: the Mixed views of each octree level, and the leaf's
		// mask values. A carver serves one worker, so nothing here is shared.
		private readonly List<int[]> levels = [];
		private readonly double[] leafValues = new double[masks.Length];

		public void CarveRoot(int i0, int i1, int j0, int j1, int k0, int k1)
		{
			int[] all = Level(0);
			for (int v = 0; v < all.Length; v++)
			{
				all[v] = v;
			}

			Carve(i0, i1, j0, j1, k0, k1, 1, all.Length, 0, 0);
		}

		// The first `activeCount` entries of Level(depth - 1) are the views still Mixed for the
		// parent; `inside` and `outside` count the views that saw the whole parent inside or
		// outside their masks (Unseen views are dropped).
		private void Carve(int i0, int i1, int j0, int j1, int k0, int k1, int depth, int activeCount, int inside, int outside)
		{
			int[] active = Level(depth - 1);
			if (i1 - i0 == 1 && j1 - j0 == 1 && k1 - k0 == 1)
			{
				grid.Values[grid.Index(i0, j0, k0)] = LeafValue(grid.Center(i0, j0, k0), active, activeCount, inside, outside);
				return;
			}

			double h = grid.VoxelSize;
			Vector3d o = grid.Origin;
			var min = new Vector3d(o.X + i0 * h, o.Y + j0 * h, o.Z + k0 * h);
			var max = new Vector3d(o.X + i1 * h, o.Y + j1 * h, o.Z + k1 * h);
			int[] mixed = Level(depth);
			int mixedCount = 0;
			for (int a = 0; a < activeCount; a++)
			{
				int v = active[a];
				switch (masks[v].Classify(min, max))
				{
					case CellFootprint.Inside:
						inside++;
						break;
					case CellFootprint.Outside:
						outside++;
						break;
					case CellFootprint.Mixed:
						mixed[mixedCount++] = v;
						break;
				}
			}

			bool carved = outside > tolerance || inside + outside + mixedCount <= tolerance;
			bool full = !carved && inside > tolerance && outside + mixedCount <= tolerance;
			if (carved || full)
			{
				Fill(i0, i1, j0, j1, k0, k1, full ? 1f : 0f);
				return;
			}

			// Children only read Level(depth) and write deeper levels, so siblings share it safely.
			int im = (i0 + i1 + 1) / 2, jm = (j0 + j1 + 1) / 2, km = (k0 + k1 + 1) / 2;
			for (int c = 0; c < 8; c++)
			{
				int ci0 = (c & 1) == 0 ? i0 : im, ci1 = (c & 1) == 0 ? im : i1;
				int cj0 = (c & 2) == 0 ? j0 : jm, cj1 = (c & 2) == 0 ? jm : j1;
				int ck0 = (c & 4) == 0 ? k0 : km, ck1 = (c & 4) == 0 ? km : k1;
				if (ci0 < ci1 && cj0 < cj1 && ck0 < ck1)
				{
					Carve(ci0, ci1, cj0, cj1, ck0, ck1, depth + 1, mixedCount, inside, outside);
				}
			}
		}

		// The (k+1)-th smallest mask value over the views that see `center`, 0 when k or fewer do.
		// `inside` and `outside` views contribute known ones and zeros.
		public float LeafValue(Vector3d center, int[] active, int activeCount, int inside, int outside)
		{
			int count = 0;
			for (int a = 0; a < activeCount; a++)
			{
				if (masks[active[a]].PointValue(center) is double value)
				{
					leafValues[count++] = value;
				}
			}

			if (inside + outside + count <= tolerance || outside > tolerance)
			{
				return 0f;
			}

			// Sorted order: `outside` zeros, then the active values, then `inside` ones.
			Array.Sort(leafValues, 0, count);
			int rank = tolerance - outside;
			return rank < count ? (float)leafValues[rank] : 1f;
		}

		private int[] Level(int depth)
		{
			while (levels.Count <= depth)
			{
				levels.Add(new int[masks.Length]);
			}

			return levels[depth];
		}

		private void Fill(int i0, int i1, int j0, int j1, int k0, int k1, float value)
		{
			for (int k = k0; k < k1; k++)
			{
				for (int j = j0; j < j1; j++)
				{
					Array.Fill(grid.Values, value, grid.Index(i0, j, k), i1 - i0);
				}
			}
		}
	}
}
