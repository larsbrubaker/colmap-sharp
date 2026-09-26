// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Normalization: colmap/geometry/normalization.h/.cc - the robust (percentile-clipped)
// bounding box and centroid of a point set, used to normalize reconstructions, and
// Hartley's isotropic normalization of image points, used by the DLT estimators. Tests:
// ColmapSharp.Tests/Geometry/NormalizationTests.cs (normalization_test.cc 1:1).
//
// ComputeBoundingBoxAndCentroid: COLMAP partitions each coordinate list with
// std::nth_element and then sums the elements between the two percentile positions. The
// set of elements in that range is fully determined, but the order nth_element leaves them
// in is libc++'s implementation detail, and it decides the rounding of the centroid sum.
// Here each list is fully sorted instead (every nth_element postcondition holds for a
// sorted list), so the bounding box is exact and the centroid can differ from COLMAP's by
// rounding: Tier B (docs/CPP_DIVERGENCES.md, entry 15). CenterAndNormalizeImagePoints is
// scalar code, Tier A (the test pins its matrix bit for bit).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/normalization.h.
/// </summary>
public static class Normalization
{
	/// <summary>
	/// The bounding box of the coordinates between the <paramref name="minPercentile"/> and
	/// <paramref name="maxPercentile"/> positions (each axis separately), and the mean of
	/// the coordinates in that range. The coordinate lists are copied, not modified.
	/// Port of colmap::ComputeBoundingBoxAndCentroid.
	/// </summary>
	public static (AlignedBox3d Bbox, Vector3d Centroid) ComputeBoundingBoxAndCentroid(
		double minPercentile,
		double maxPercentile,
		IReadOnlyList<double> coordsX,
		IReadOnlyList<double> coordsY,
		IReadOnlyList<double> coordsZ)
	{
		Check.That(coordsX.Count > 0);
		Check.Eq(coordsX.Count, coordsY.Count);
		Check.Eq(coordsX.Count, coordsZ.Count);
		Check.Ge(minPercentile, 0.0);
		Check.Le(minPercentile, 1.0);
		Check.Ge(maxPercentile, 0.0);
		Check.Le(maxPercentile, 1.0);
		Check.Le(minPercentile, maxPercentile);

		int endIdx = coordsX.Count - 1;
		int minIdx = (int)Math.Min((ulong)endIdx, (ulong)Math.Floor(minPercentile * endIdx));
		int maxIdx = (int)Math.Min((ulong)endIdx, (ulong)Math.Ceiling(maxPercentile * endIdx));

		double[] xs = SortedCopy(coordsX);
		double[] ys = SortedCopy(coordsY);
		double[] zs = SortedCopy(coordsZ);

		var bboxMin = new Vector3d(xs[minIdx], ys[minIdx], zs[minIdx]);
		var bboxMax = new Vector3d(xs[maxIdx], ys[maxIdx], zs[maxIdx]);

		double cx = 0;
		double cy = 0;
		double cz = 0;
		double normalization = 1.0 / (maxIdx - minIdx + 1);
		for (int i = minIdx; i <= maxIdx; ++i)
		{
			cx += normalization * xs[i];
			cy += normalization * ys[i];
			cz += normalization * zs[i];
		}

		return (new AlignedBox3d(bboxMin, bboxMax), new Vector3d(cx, cy, cz));
	}

	/// <summary>
	/// Translates the points so their centroid is the origin and scales them so their
	/// root-mean-square distance from it is sqrt(2) (Hartley normalization). Returns the
	/// normalized points and the 3x3 transform that maps original to normalized points.
	/// Port of colmap::CenterAndNormalizeImagePoints.
	/// </summary>
	public static (Vector2d[] NormedPoints, Matrix3d NormedFromOrig) CenterAndNormalizeImagePoints(
		IReadOnlyList<Vector2d> points)
	{
		Vector2d[] source = points as Vector2d[] ?? points.ToArray();
		var normedPoints = new Vector2d[source.Length];
		Matrix3d normedFromOrig = CenterAndNormalizeImagePoints(source, normedPoints);
		return (normedPoints, normedFromOrig);
	}

	/// <summary>
	/// <see cref="CenterAndNormalizeImagePoints(IReadOnlyList{Vector2d})"/> writing the
	/// normalized points into a caller-provided span of the same length (allocation-free, for
	/// solvers that run once per RANSAC hypothesis). Returns normed_from_orig.
	/// </summary>
	public static Matrix3d CenterAndNormalizeImagePoints(ReadOnlySpan<Vector2d> points, Span<Vector2d> normedPoints)
	{
		int numPoints = points.Length;
		Check.Gt(numPoints, 0);
		Check.Eq(normedPoints.Length, numPoints);

		// Calculate centroid.
		Vector2d centroid = Vector2d.Zero;
		foreach (Vector2d point in points)
		{
			centroid += point;
		}

		centroid /= numPoints;

		// Root mean square distance to centroid of all points.
		double rmsMeanDist = 0;
		foreach (Vector2d point in points)
		{
			rmsMeanDist += (point - centroid).SquaredNorm;
		}

		rmsMeanDist = Math.Sqrt(rmsMeanDist / numPoints);

		// Compose normalization matrix.
		double normFactor = Math.Sqrt(2.0) / rmsMeanDist;
		var normedFromOrig = new Matrix3d(
			normFactor, 0, -normFactor * centroid.X,
			0, normFactor, -normFactor * centroid.Y,
			0, 0, 1);

		// Apply normalization matrix.
		for (int i = 0; i < numPoints; ++i)
		{
			normedPoints[i] = (normedFromOrig * points[i].Homogeneous()).HNormalized();
		}

		return normedFromOrig;
	}

	private static double[] SortedCopy(IReadOnlyList<double> values)
	{
		double[] copy = values.ToArray();
		Array.Sort(copy);
		return copy;
	}
}
