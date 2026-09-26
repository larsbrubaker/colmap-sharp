// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshing: port of colmap/mvs/delaunay_meshing.cc's DelaunayMeshing,
// SparseDelaunayMeshing and DenseDelaunayMeshing - surface reconstruction after P. Labatut,
// J-P. Pons and R. Keriven, "Robust and efficient surface reconstruction from range data",
// Computer Graphics Forum 2009. The points are tetrahedralized (DelaunayMeshingInput), every
// viewing ray votes that the cells it passes are empty and the cell behind its point is
// full (DelaunayMeshing.Integrate.cs), and a min s-t cut (Mathematics/MinSTGraphCut) labels
// the cells; the surface is the facets between the two labels.
//
// The API is in memory (no file paths): SparseDelaunayMeshing takes a Reconstruction,
// DenseDelaunayMeshing takes the undistorted sparse model plus StereoFusion's fused points
// and visibility, and both return the PlyMesh COLMAP would write. Both take a
// CancellationToken and report progress (fraction of images integrated).
// Neighbors: DelaunayMeshingOptions.cs, DelaunayMeshingInput.cs, DelaunayMeshingWeights.cs.
// Tests: ColmapSharp.Tests/Mvs/DelaunayMeshingTests.cs (delaunay_meshing_test.cc 1:1) and
// DelaunayMeshingTests.CSharpOnly.cs.
//
// Tier C (outcome). Deliberate differences (docs/CPP_DIVERGENCES.md): the tetrahedralization
// is ours (103), the graph and the surface are assembled in cell/facet handle order instead
// of hash-map order (109), per-image weights are summed in image order whatever the thread
// count (110), when the space just behind an observed point is outside the convex hull (an
// infinite cell) that cell gets the point's sink vote only if the ray passed through the hull
// (111), and a cut with no surface throws a message that explains it (112).
//
// Input whose points never span 3D (all coplanar or collinear) has no cells to cut: Run then
// fails a Check (Locate needs Dimension == 3), where COLMAP would work on CGAL's 2D
// triangulation.

using System.Runtime.InteropServices;

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// Delaunay meshing of sparse and dense reconstructions (COLMAP's SparseDelaunayMeshing,
/// DenseDelaunayMeshing and DelaunayMeshing).
/// </summary>
public static partial class DelaunayMeshing
{
	/// <summary>The message of the exception thrown when the graph cut leaves no surface.</summary>
	public const string NoSurfaceMessage =
		"Delaunay meshing found no surface: too few views see into the scene's free space (common for "
		+ "smooth, convex objects captured from outside). Try Poisson meshing or add views.";

	/// <summary>Meshes the 3D points of a sparse reconstruction, seen from its registered images.</summary>
	public static PlyMesh SparseDelaunayMeshing(DelaunayMeshingOptions options, Reconstruction reconstruction,
		IProgress<double>? progress = null, CancellationToken cancellationToken = default) =>
		Run(options, DelaunayMeshingInput.FromSparseReconstruction(reconstruction), progress, cancellationToken);

	/// <summary>
	/// Meshes fused dense points. <paramref name="reconstruction"/> is the undistorted sparse
	/// model of the dense workspace, and visibility[i] lists the registered-image indices that
	/// see fused point i (StereoFusion's fused points and visibility).
	/// </summary>
	public static PlyMesh DenseDelaunayMeshing(DelaunayMeshingOptions options, Reconstruction reconstruction,
		IReadOnlyList<PlyPoint> fusedPoints, IReadOnlyList<IReadOnlyList<int>> visibility,
		IProgress<double>? progress = null, CancellationToken cancellationToken = default) =>
		Run(options, DelaunayMeshingInput.FromDense(reconstruction, fusedPoints, visibility), progress, cancellationToken);

	/// <summary>Port of the DelaunayMeshing(options, input_data) function.</summary>
	public static PlyMesh Run(DelaunayMeshingOptions options, DelaunayMeshingInput input,
		IProgress<double>? progress = null, CancellationToken cancellationToken = default)
	{
		Check.That(options.Check());

		// Create a delaunay triangulation of all input points.
		var triangulation = input.CreateSubSampledDelaunayTriangulation(
			(float)options.MaxProjDist, (float)options.MaxDepthDist, cancellationToken);

		// Helper class to efficiently trace rays through the triangulation.
		var rayCaster = new DelaunayTriangulationRayCaster(triangulation);

		// Helper class to efficiently compute edge weights in the s-t graph.
		var edgeWeightComputer = new DelaunayMeshingEdgeWeightComputer(
			triangulation, options.VisibilitySigma, options.DistanceSigmaFactor);

		// Initialize the s-t graph with cells as nodes and oriented facets as edges. Nodes
		// are numbered in cell handle order (entry 109).
		var cellIndex = new int[triangulation.CellCapacity];
		Array.Fill(cellIndex, -1);
		var cells = new List<int>(triangulation.NumberOfCells);
		foreach (int cell in triangulation.AllCells())
		{
			cellIndex[cell] = cells.Count;
			cells.Add(cell);
		}

		var weights = new CellWeights[cells.Count];
		IntegrateImages(options, input, triangulation, rayCaster, edgeWeightComputer, cellIndex, weights,
			progress, cancellationToken);

		cancellationToken.ThrowIfCancellationRequested();
		var graphCut = BuildGraph(options, triangulation, cells, cellIndex, weights);
		graphCut.Compute();

		cancellationToken.ThrowIfCancellationRequested();
		return ExtractSurface(options, triangulation, cellIndex, graphCut);
	}

	// Setup the min-cut (max-flow) graph optimization. Each oriented facet in the Delaunay
	// triangulation corresponds to a directed edge and each cell corresponds to a node.
	private static MinSTGraphCut<float> BuildGraph(DelaunayMeshingOptions options, DelaunayTriangulation3 triangulation,
		List<int> cells, int[] cellIndex, CellWeights[] weights)
	{
		var graphCut = new MinSTGraphCut<float>(cells.Count);
		for (int index = 0; index < cells.Count; ++index)
		{
			int cell = cells[index];
			ref var data = ref weights[index];
			graphCut.AddNode(index, data.SourceWeight, data.SinkWeight);

			// Iterate all facets of the current cell to accumulate edge weight.
			for (int i = 0; i < 4; ++i)
			{
				// Extract the mirrored facet of the current cell (opposite orientation).
				var (mirrorCell, mirrorI) = triangulation.MirrorFacet(cell, i);
				int mirrorIndex = cellIndex[mirrorCell];

				// Avoid duplicate edges in graph.
				if (index < mirrorIndex)
				{
					continue;
				}

				// Implementation of geometry visualized in Figure 9 in P. Labatut, J-P. Pons,
				// and R. Keriven. "Robust and efficient surface reconstruction from range
				// data." Computer graphics forum, 2009.
				double edgeShapeWeight = options.QualityRegularization
					* (1.0 - Math.Min(ComputeCosFacetCellAngle(triangulation, cell, i),
						ComputeCosFacetCellAngle(triangulation, mirrorCell, mirrorI)));

				float forwardEdgeWeight = (float)(data.EdgeWeight(i) + edgeShapeWeight);
				float backwardEdgeWeight = (float)(weights[mirrorIndex].EdgeWeight(mirrorI) + edgeShapeWeight);

				graphCut.AddEdge(index, mirrorIndex, forwardEdgeWeight, backwardEdgeWeight);
			}
		}

		return graphCut;
	}

	/// <summary>
	/// Port of ComputeCosFacetCellAngle: the cosine between the facet normal (into the cell)
	/// and the direction from the facet's first vertex to the cell's circumcenter (Figure 9
	/// of Labatut et al. 2009); 1 for an infinite cell, 0.5 for degenerate geometry.
	/// </summary>
	internal static double ComputeCosFacetCellAngle(DelaunayTriangulation3 triangulation, int cell, int i)
	{
		if (triangulation.IsInfinite(cell))
		{
			return 1.0;
		}

		var (a, b, c) = triangulation.FacetVertices(cell, i);
		var t0 = triangulation.Point(a);
		var t1 = triangulation.Point(b);
		var t2 = triangulation.Point(c);

		double e1x = t1.X - t0.X, e1y = t1.Y - t0.Y, e1z = t1.Z - t0.Z;
		double e2x = t2.X - t0.X, e2y = t2.Y - t0.Y, e2z = t2.Z - t0.Z;
		double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
		double facetNormalLengthSquared = nx * nx + ny * ny + nz * nz;
		if (facetNormalLengthSquared == 0.0)
		{
			return 0.5;
		}

		var center = triangulation.Circumcenter(cell);
		double cx = center.X - t0.X, cy = center.Y - t0.Y, cz = center.Z - t0.Z;

		// COLMAP stores this one in a float.
		float coTangentLengthSquared = (float)(cx * cx + cy * cy + cz * cz);
		if (coTangentLengthSquared == 0.0)
		{
			return 0.5;
		}

		return (nx * cx + ny * cy + nz * cz) / Math.Sqrt(facetNormalLengthSquared * coTangentLengthSquared);
	}

	// Extract the surface facets as the oriented min-cut of the graph.
	private static PlyMesh ExtractSurface(DelaunayMeshingOptions options, DelaunayTriangulation3 triangulation,
		int[] cellIndex, MinSTGraphCut<float> graphCut)
	{
		// COLMAP collects the vertices in a FlatHashSet; here they are numbered in order of
		// first appearance (entry 109).
		var surfaceVertexIndices = new Dictionary<int, int>();
		var mesh = new PlyMesh();
		var surfaceFacets = new List<(int Cell, int Index)>();
		var surfaceFacetSideLengths = new List<float>();

		foreach (var (cell, i) in triangulation.FiniteFacets())
		{
			int mirrorCell = triangulation.Neighbor(cell, i);

			// Obtain labeling after the graph-cut.
			bool cellIsSource = graphCut.IsConnectedToSource(cellIndex[cell]);
			bool mirrorCellIsSource = graphCut.IsConnectedToSource(cellIndex[mirrorCell]);

			// The surface is equal to the location of the cut, which is at the transition
			// between source and sink nodes.
			if (cellIsSource == mirrorCellIsSource)
			{
				continue;
			}

			// Remember all unique vertices of the surface mesh.
			var (a, b, c) = triangulation.FacetVertices(cell, i);
			AddSurfaceVertex(triangulation, mesh, surfaceVertexIndices, a);
			AddSurfaceVertex(triangulation, mesh, surfaceVertexIndices, b);
			AddSurfaceVertex(triangulation, mesh, surfaceVertexIndices, c);

			// Determine maximum side length of facet.
			var p0 = triangulation.Point(a);
			var p1 = triangulation.Point(b);
			var p2 = triangulation.Point(c);
			float maxSquaredSideLength = (float)Math.Max(SquaredDistance(p0, p1),
				Math.Max(SquaredDistance(p0, p2), SquaredDistance(p1, p2)));
			surfaceFacetSideLengths.Add(MathF.Sqrt(maxSquaredSideLength));

			// Remember surface mesh facet and make sure it is oriented correctly: the facet
			// seen from the source (outside) cell, whose vertex order makes the normal point
			// into that cell, i.e. out of the surface.
			surfaceFacets.Add(cellIsSource ? (cell, i) : triangulation.MirrorFacet(cell, i));
		}

		// COLMAP fails a Check inside Percentile here; say what it means instead (entry 112).
		if (surfaceFacetSideLengths.Count == 0)
		{
			throw new InvalidOperationException(NoSurfaceMessage);
		}

		var sideLengthsCopy = surfaceFacetSideLengths.ToArray();
		float maxFacetSideLength = (float)(options.MaxSideLengthFactor
			* MathUtils.Percentile<float>(sideLengthsCopy, options.MaxSideLengthPercentile));

		mesh.Faces.Capacity = surfaceFacets.Count;
		for (int k = 0; k < surfaceFacets.Count; ++k)
		{
			// Note that skipping some of the facets here means that there will be some unused
			// vertices in the final mesh.
			if (surfaceFacetSideLengths[k] > maxFacetSideLength)
			{
				continue;
			}

			var (a, b, c) = triangulation.FacetVertices(surfaceFacets[k].Cell, surfaceFacets[k].Index);
			mesh.Faces.Add(new PlyMeshFace(surfaceVertexIndices[a], surfaceVertexIndices[b], surfaceVertexIndices[c]));
		}

		return mesh;
	}

	private static void AddSurfaceVertex(DelaunayTriangulation3 triangulation, PlyMesh mesh,
		Dictionary<int, int> surfaceVertexIndices, int vertex)
	{
		ref int index = ref CollectionsMarshal.GetValueRefOrAddDefault(surfaceVertexIndices, vertex, out bool exists);
		if (!exists)
		{
			index = mesh.Vertices.Count;
			var point = triangulation.Point(vertex);
			mesh.Vertices.Add(new PlyMeshVertex((float)point.X, (float)point.Y, (float)point.Z));
		}
	}

	private static double SquaredDistance(Vector3d a, Vector3d b)
	{
		double dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
		return dx * dx + dy * dy + dz * dz;
	}
}
