// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshing.Integrate: the per-image integration of colmap/mvs/delaunay_meshing.cc's
// DelaunayMeshing (the IntegreateImage lambda and the loop that accumulates its results) and
// DelaunayCellData. See DelaunayMeshing.cs for the overview.
//
// Threading: COLMAP integrates images on a thread pool and adds each image's weights into the
// graph in completion order, so with several threads the float sums (and so the cut) can vary
// between runs. Here batches of images are integrated in parallel, each image into its own
// dictionary, and the batch is added in image order - the order of COLMAP's single-threaded
// run, whatever the thread count (entry 110). Each image locates with its own cursors, so a
// ray's result never depends on another thread.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

public static partial class DelaunayMeshing
{
	/// <summary>Port of DelaunayCellData minus its index: accumulated s-t weights of one cell.</summary>
	internal struct CellWeights
	{
		public float SourceWeight;
		public float SinkWeight;
		public float EdgeWeight0;
		public float EdgeWeight1;
		public float EdgeWeight2;
		public float EdgeWeight3;

		public readonly float EdgeWeight(int i) => i switch
		{
			0 => EdgeWeight0,
			1 => EdgeWeight1,
			2 => EdgeWeight2,
			_ => EdgeWeight3,
		};

		[UnscopedRef]
		public ref float EdgeWeightRef(int i)
		{
			switch (i)
			{
				case 0:
					return ref EdgeWeight0;
				case 1:
					return ref EdgeWeight1;
				case 2:
					return ref EdgeWeight2;
				default:
					return ref EdgeWeight3;
			}
		}
	}

	private static void IntegrateImages(DelaunayMeshingOptions options, DelaunayMeshingInput input,
		DelaunayTriangulation3 triangulation, DelaunayTriangulationRayCaster rayCaster,
		DelaunayMeshingEdgeWeightComputer edgeWeightComputer, int[] cellIndex, CellWeights[] weights,
		IProgress<double>? progress, CancellationToken cancellationToken)
	{
		int numThreads = Threading.GetEffectiveNumThreads(options.NumThreads);
		int numImages = input.Images.Count;
		int batchSize = 2 * numThreads;
		var results = new Dictionary<int, CellWeights>[batchSize];
		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = numThreads,
			CancellationToken = cancellationToken,
		};

		for (int batchStart = 0; batchStart < numImages; batchStart += batchSize)
		{
			int count = Math.Min(batchSize, numImages - batchStart);
			Parallel.For(0, count, parallelOptions, k =>
				results[k] = IntegrateImage(input, batchStart + k, triangulation, rayCaster, edgeWeightComputer,
					cancellationToken));

			// Accumulate the weights of the images into the global graph, in image order.
			for (int k = 0; k < count; ++k)
			{
				foreach (var (cell, cellData) in results[k])
				{
					ref var global = ref weights[cellIndex[cell]];
					global.SinkWeight += cellData.SinkWeight;
					global.SourceWeight += cellData.SourceWeight;
					for (int j = 0; j < 4; ++j)
					{
						global.EdgeWeightRef(j) += cellData.EdgeWeight(j);
					}
				}

				results[k] = null!;
				progress?.Report((double)(batchStart + k + 1) / numImages);
			}
		}
	}

	// Accumulated weights for one image only (COLMAP's IntegreateImage).
	private static Dictionary<int, CellWeights> IntegrateImage(DelaunayMeshingInput input, int imageIdx,
		DelaunayTriangulation3 triangulation, DelaunayTriangulationRayCaster rayCaster,
		DelaunayMeshingEdgeWeightComputer edgeWeightComputer, CancellationToken cancellationToken)
	{
		var imageCellGraphData = new Dictionary<int, CellWeights>();

		// Image that is integrated into s-t graph.
		var image = input.Images[imageIdx];
		var imagePosition = DelaunayMeshingInput.ToDouble(image.CamInWorld);

		// Intersections between viewing rays and Delaunay triangulation.
		var intersections = new List<DelaunayTriangulationRayCaster.Intersection>();
		// Every ray starts at the camera, so one cursor stays at the camera's cell; the other
		// follows the observed points (and is moved next to each ray's end before locating
		// the point behind it).
		var cameraCursor = default(DelaunayTriangulation3.LocateCursor);
		var pointCursor = default(DelaunayTriangulation3.LocateCursor);

		// Iterate through all image observations and integrate them into the graph.
		for (int n = 0; n < image.PointIdxs.Count; ++n)
		{
			if ((n & 1023) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			var point = input.Points[image.PointIdxs[n]];

			// Likelihood of the point observation. The square is a uint32 product in COLMAP.
			double alpha = edgeWeightComputer.ComputeVisibilityProb(
				unchecked(point.NumVisibleImages * point.NumVisibleImages));

			var pointPosition = DelaunayMeshingInput.ToDouble(point.Position);
			double vx = pointPosition.X - imagePosition.X;
			double vy = pointPosition.Y - imagePosition.Y;
			double vz = pointPosition.Z - imagePosition.Z;
			double viewingLength = Math.Sqrt(vx * vx + vy * vy + vz * vz);
			double epsilonScale = 0.001 * edgeWeightComputer.DistanceSigma;
			double ex = epsilonScale * (vx / viewingLength);
			double ey = epsilonScale * (vy / viewingLength);
			double ez = epsilonScale * (vz / viewingLength);

			// Find intersected facets between image and point.
			rayCaster.CastRaySegment(imagePosition,
				new Vector3d(pointPosition.X - ex, pointPosition.Y - ey, pointPosition.Z - ez), intersections, ref cameraCursor);

			// Accumulate source weights for cell containing image.
			if (intersections.Count > 0)
			{
				ref var data = ref CollectionsMarshal.GetValueRefOrAddDefault(imageCellGraphData, intersections[0].Cell, out _);
				data.SourceWeight = (float)(data.SourceWeight + alpha);
			}

			// Accumulate edge weights from image to point.
			foreach (var intersection in intersections)
			{
				ref var data = ref CollectionsMarshal.GetValueRefOrAddDefault(imageCellGraphData, intersection.Cell, out _);
				ref float edge = ref data.EdgeWeightRef(intersection.Index);
				edge = (float)(edge + alpha * edgeWeightComputer.ComputeDistanceProb(intersection.TargetDistanceSquared));
			}

			// Accumulate edge weights from point to extended point and accumulate sink weight
			// of the cell inside the surface: find the first facet that is intersected by the
			// extended ray behind the observed point. Then accumulate the edge weight of that
			// facet and accumulate the sink weight of the cell behind that facet.
			var behindPosition = new Vector3d(pointPosition.X + ex, pointPosition.Y + ey, pointPosition.Z + ez);
			if (intersections.Count > 0)
			{
				// The ray stops just before the point, in the cell past its last crossing.
				var last = intersections[^1];
				pointCursor.StartNear(triangulation.Neighbor(last.Cell, last.Index));
			}

			int behindPointCell = triangulation.Locate(behindPosition, ref pointCursor);

			// The cell behind the point is infinite when the viewing ray leaves the convex hull
			// at the point: a hull vertex seen through the point set from the far side, or a
			// ray grazing the silhouette. (A hull vertex seen from outside, facing the camera,
			// has a finite cell behind it.) Three facets of an infinite cell run through the
			// infinite vertex and have no geometric triangle; COLMAP intersects them anyway,
			// using whatever point CGAL stores for that vertex. Here (entry 111): when the ray
			// did pass through the hull before reaching the point, the infinite cell behind
			// it gets the sink vote; when it did not (it stayed outside, e.g. grazing), the
			// observation adds nothing behind the point.
			if (triangulation.IsInfinite(behindPointCell))
			{
				if (intersections.Count > 0)
				{
					ref var outside = ref CollectionsMarshal.GetValueRefOrAddDefault(imageCellGraphData, behindPointCell, out _);
					outside.SinkWeight = (float)(outside.SinkWeight + alpha);
				}

				continue;
			}

			int behindNeighborIdx = -1;
			double behindDistanceSquared = 0.0;
			for (int neighborIdx = 0; neighborIdx < 4; ++neighborIdx)
			{
				var (a, b, c) = triangulation.FacetVertices(behindPointCell, neighborIdx);
				if (DelaunayTriangulation3.TryIntersectRayTriangle(imagePosition, pointPosition,
					triangulation.Point(a), triangulation.Point(b), triangulation.Point(c), out var interPoint))
				{
					double dx = interPoint.X - pointPosition.X;
					double dy = interPoint.Y - pointPosition.Y;
					double dz = interPoint.Z - pointPosition.Z;
					double distanceSquared = dx * dx + dy * dy + dz * dz;
					if (distanceSquared > behindDistanceSquared)
					{
						behindDistanceSquared = distanceSquared;
						behindNeighborIdx = neighborIdx;
					}
				}
			}

			if (behindNeighborIdx >= 0)
			{
				ref var data = ref CollectionsMarshal.GetValueRefOrAddDefault(imageCellGraphData, behindPointCell, out _);
				ref float edge = ref data.EdgeWeightRef(behindNeighborIdx);
				edge = (float)(edge + alpha * edgeWeightComputer.ComputeDistanceProb(behindDistanceSquared));

				int insideCell = triangulation.Neighbor(behindPointCell, behindNeighborIdx);
				ref var inside = ref CollectionsMarshal.GetValueRefOrAddDefault(imageCellGraphData, insideCell, out _);
				inside.SinkWeight = (float)(inside.SinkWeight + alpha);
			}
		}

		return imageCellGraphData;
	}
}
