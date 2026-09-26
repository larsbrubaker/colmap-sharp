// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// NormalizationTests: colmap/geometry/normalization_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/Normalization.cs. The exact (EXPECT_EQ) checks are exact here too.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class NormalizationTests
{
	[Test]
	public async Task ComputeBoundingBoxAndCentroid_SingleCoord()
	{
		(AlignedBox3d bbox, Vector3d centroid) = Normalization.ComputeBoundingBoxAndCentroid(0, 1, [1], [2], [3]);
		using (Assert.Multiple())
		{
			await Assert.That(bbox.Min == new Vector3d(1, 2, 3)).IsTrue();
			await Assert.That(bbox.Max == new Vector3d(1, 2, 3)).IsTrue();
			await Assert.That(centroid == new Vector3d(1, 2, 3)).IsTrue();
		}
	}

	[Test]
	public async Task ComputeBoundingBoxAndCentroid_TwoCoords()
	{
		(AlignedBox3d bbox, Vector3d centroid) = Normalization.ComputeBoundingBoxAndCentroid(0, 1, [2, -1], [3, -2], [4, -3]);
		using (Assert.Multiple())
		{
			await Assert.That(bbox.Min == new Vector3d(-1, -2, -3)).IsTrue();
			await Assert.That(bbox.Max == new Vector3d(2, 3, 4)).IsTrue();
			await Assert.That(centroid == new Vector3d(0.5, 0.5, 0.5)).IsTrue();
		}
	}

	[Test]
	public async Task ComputeBoundingBoxAndCentroid_ThreeCoords()
	{
		(AlignedBox3d bbox, Vector3d centroid) =
			Normalization.ComputeBoundingBoxAndCentroid(0, 1, [2, -1, 5], [3, -2, 5], [4, -3, 5]);
		using (Assert.Multiple())
		{
			await Assert.That(bbox.Min == new Vector3d(-1, -2, -3)).IsTrue();
			await Assert.That(bbox.Max == new Vector3d(5, 5, 5)).IsTrue();
			await Assert.That(EigenMatrixNear(centroid, new Vector3d(2, 2, 2), 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task ComputeBoundingBoxAndCentroid_FiveCoords()
	{
		(AlignedBox3d bbox1, Vector3d centroid1) = Normalization.ComputeBoundingBoxAndCentroid(
			0, 1, [2, -1, 5, 100, -100], [3, -2, 5, 100, -100], [4, -3, 5, 100, -100]);
		(AlignedBox3d bbox2, Vector3d centroid2) = Normalization.ComputeBoundingBoxAndCentroid(
			0.3, 0.7, [2, -1, 5, 100, -100], [3, -2, 5, 100, -100], [4, -3, 5, 100, -100]);
		using (Assert.Multiple())
		{
			await Assert.That(bbox1.Min == new Vector3d(-100, -100, -100)).IsTrue();
			await Assert.That(bbox1.Max == new Vector3d(100, 100, 100)).IsTrue();
			await Assert.That(EigenMatrixNear(centroid1, new Vector3d(1.2, 1.2, 1.2), 1e-6)).IsTrue();
			await Assert.That(bbox2.Min == new Vector3d(-1, -2, -3)).IsTrue();
			await Assert.That(bbox2.Max == new Vector3d(5, 5, 5)).IsTrue();
			await Assert.That(EigenMatrixNear(centroid2, new Vector3d(2, 2, 2), 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task CenterAndNormalizeImagePoints_Nominal()
	{
		const int NumPoints = 11;
		var points = new List<Vector2d>();
		for (int i = 0; i < NumPoints; ++i)
		{
			points.Add(new Vector2d(i, i));
		}

		(Vector2d[] normedPoints, Matrix3d matrix) = Normalization.CenterAndNormalizeImagePoints(points);

		Vector2d meanPoint = Vector2d.Zero;
		foreach (Vector2d point in normedPoints)
		{
			meanPoint += point;
		}

		using (Assert.Multiple())
		{
			await Assert.That(matrix[0, 0]).IsEqualTo(0.31622776601683794);
			await Assert.That(matrix[1, 1]).IsEqualTo(0.31622776601683794);
			await Assert.That(matrix[0, 2]).IsEqualTo(-1.5811388300841898);
			await Assert.That(matrix[1, 2]).IsEqualTo(-1.5811388300841898);
			await Assert.That(Math.Abs(meanPoint.X)).IsLessThan(1e-6);
			await Assert.That(Math.Abs(meanPoint.Y)).IsLessThan(1e-6);
		}
	}
}
