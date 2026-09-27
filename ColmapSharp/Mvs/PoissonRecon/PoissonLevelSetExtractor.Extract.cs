// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor.Extract: the slab driver of FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... >::Extract, as COLMAP's poisson_meshing runs it (the whole
// tree: slabDepth 0, slab [0, 1), no back or front boundary, so InteriorSlab holds for every
// slab and SetSlabBounds gives the slab's own bounds). It walks the finest depth's slabs back to
// front, calling the steps in PoissonLevelSetExtractor.cs (InitSlice, InitSlab,
// SetSliceValues), .IsoVertices.cs and .XSliceIsoVertices.cs (the iso-vertices), .IsoEdges.cs
// (the iso-edges and the finalize steps) and .Polygons.cs (IsoSurface) in Extract's order.
// What it leaves in Vertices and Polygons is what upstream writes to the vertex and polygon
// streams; PoissonMeshOutput maps it to the model and to COLMAP's output vertices. Tier A
// against oracle/poisson_extract_harness.cc (upstream's own Extract) and, stage by stage,
// oracle/poisson_levelset4_harness.cc through oracle/poisson_levelset6_harness.cc, whose tests
// run this driver stopped early (LevelSetExtractStages) with dump hooks.
//
// Translation notes: upstream's timing statistics (Stats) and its "bad average roots" warning
// are not ported; BadRootCount holds the count. Cancellation is checked and progress reported
// once per slab at the finest depth, where upstream has neither.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>How far <see cref="PoissonLevelSetExtractor.Extract(CancellationToken, IProgress{double}?)"/> runs; the stage tests stop early.</summary>
internal enum LevelSetExtractStages
{
	/// <summary>The iso-vertices and the edge part of the finalize steps only.</summary>
	IsoVertices,

	/// <summary>Everything but IsoSurface (the polygons).</summary>
	IsoEdges,

	/// <summary>Upstream's whole Extract.</summary>
	Polygons,
}

/// <summary>Callbacks the stage tests use to dump the extractor's state as Extract runs.</summary>
internal sealed class LevelSetExtractHooks
{
	/// <summary>Where Extract stops (see <see cref="LevelSetExtractStages"/>).</summary>
	public LevelSetExtractStages Stages { get; init; } = LevelSetExtractStages.Polygons;

	/// <summary>Called with sliceAtMaxDepth after each slice plane is finalized.</summary>
	public Action<int>? SliceFinalized { get; init; }

	/// <summary>Called with slabAtMaxDepth after each slab is finalized, before its polygons.</summary>
	public Action<int>? SlabFinalized { get; init; }

	/// <summary>Called with slabAtMaxDepth after each slab's polygons.</summary>
	public Action<int>? SlabExtracted { get; init; }
}

public sealed partial class PoissonLevelSetExtractor
{
	private bool extracted;

	/// <summary>
	/// Whether each vertex's Depth is a density weight (the extractor was given Extract's
	/// densityWeights, which Reconstructors.h passes when --density asks for it).
	/// </summary>
	public bool HasDensity => density != null;

	/// <summary>The auxiliary data channels per vertex (3 for colors; 0 without data).</summary>
	public int DataChannels => zeroData.Length;

	/// <summary>
	/// Extracts the whole level set into <see cref="Vertices"/> and <see cref="Polygons"/>
	/// (unit-cube coordinates). Port of <c>_LevelSetExtractor&lt; ... , 3 , ... &gt;::Extract</c>'s
	/// slab loop with COLMAP's settings (see the file header). Runs once per extractor.
	/// </summary>
	/// <param name="cancellationToken">Checked before each slab at the finest depth.</param>
	/// <param name="progress">Receives the fraction of the finest depth's slabs done, after each.</param>
	public void Extract(CancellationToken cancellationToken = default, IProgress<double>? progress = null) =>
		Extract(cancellationToken, progress, null);

	/// <summary>
	/// <see cref="Extract(CancellationToken, IProgress{double}?)"/>, stopped after
	/// <paramref name="hooks"/>' stages and calling its callbacks (the stage tests' dump points).
	/// </summary>
	internal void Extract(CancellationToken cancellationToken, IProgress<double>? progress, LevelSetExtractHooks? hooks)
	{
		if (extracted)
		{
			throw new InvalidOperationException("The level set has already been extracted; create a new extractor to extract again.");
		}

		extracted = true;
		LevelSetExtractStages stages = hooks?.Stages ?? LevelSetExtractStages.Polygons;
		bool edges = stages >= LevelSetExtractStages.IsoEdges;

		void Finalize(int sliceAtMaxDepth)
		{
			if (edges)
			{
				FinalizeSlice(sliceAtMaxDepth);
			}
			else
			{
				FinalizeSliceEdges(sliceAtMaxDepth);
			}

			hooks?.SliceFinalized?.Invoke(sliceAtMaxDepth);
		}

		// InitSlab here too, in case the slice wants to push iso-vertices down to the slab.
		InitSlice(0);
		InitSlab(0, true);
		SetSliceValues(0);
		SetSliceIsoVertices(0);
		if (edges)
		{
			SetSliceIsoEdges(0);
		}

		Finalize(0);

		// Iterate over the slabs at the finest level.
		int slabs = 1 << MaxDepth;
		for (int slab = 0; slab < slabs; slab++)
		{
			cancellationToken.ThrowIfCancellationRequested();

			// Need to init the slab before setting the slice (in case things need to be pushed
			// down from the slice to the slab).
			InitSlice(slab + 1);
			if (slab != 0)
			{
				InitSlab(slab, false);
			}

			// Need to set the front slice corner values before computing iso-vertices.
			SetSliceValues(slab + 1);

			// Compute the iso-vertices first, so that slice's iso-vertices are set last.
			SetSlabIsoVertices(slab);
			SetSliceIsoVertices(slab + 1);
			if (edges)
			{
				SetSliceIsoEdges(slab + 1);

				// Now compute the iso-edges.
				SetSlabIsoEdges(slab);
			}

			Finalize(slab + 1);
			if (edges)
			{
				FinalizeSlab(slab);
			}
			else
			{
				FinalizeSlabEdges(slab);
			}

			hooks?.SlabFinalized?.Invoke(slab);
			if (stages == LevelSetExtractStages.Polygons)
			{
				IsoSurface(slab);
			}

			hooks?.SlabExtracted?.Invoke(slab);
			progress?.Report((double)(slab + 1) / slabs);
		}
	}
}
