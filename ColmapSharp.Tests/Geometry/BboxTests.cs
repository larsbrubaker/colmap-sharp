// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BboxTests: colmap/geometry/bbox_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/Bbox.cs and LinearAlgebra/AlignedBox3d.cs. Tier A: the arithmetic
// is exact on these inputs, but the checks keep COLMAP's 1e-10 tolerance.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class BboxTests
{
	[Test]
	public async Task ComputeEqualPartsBboxes_Split1x1x2()
	{
		var bbox = new AlignedBox3d(new Vector3d(0, 0, 0), new Vector3d(1, 1, 1));
		var split = (1, 1, 1);

		List<AlignedBox3d> bboxes = Bbox.ComputeEqualPartsBboxes(bbox, split);

		await Assert.That(bboxes.Count).IsEqualTo(1);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(bboxes[0].Min, new Vector3d(0.0, 0.0, 0.0), 1e-10)).IsTrue();
			await Assert.That(EigenMatrixNear(bboxes[0].Max, new Vector3d(1.0, 1.0, 1.0), 1e-10)).IsTrue();
		}
	}

	[Test]
	public async Task ComputeEqualPartsBboxes_Split2x2x2()
	{
		var bbox = new AlignedBox3d(new Vector3d(0, 0, 0), new Vector3d(2, 2, 2));
		var split = (2, 2, 2);

		List<AlignedBox3d> bboxes = Bbox.ComputeEqualPartsBboxes(bbox, split);

		await Assert.That(bboxes.Count).IsEqualTo(8);

		var covered = new AlignedBox3d();
		foreach (AlignedBox3d subBbox in bboxes)
		{
			covered = covered.Extend(subBbox);
		}

		using (Assert.Multiple())
		{
			foreach (AlignedBox3d subBbox in bboxes)
			{
				Vector3d diag = subBbox.Diagonal();
				await Assert.That(EigenMatrixNear(diag, new Vector3d(1.0, 1.0, 1.0), 1e-10)).IsTrue();
			}

			await Assert.That(EigenMatrixNear(covered.Min, bbox.Min, 1e-10)).IsTrue();
			await Assert.That(EigenMatrixNear(covered.Max, bbox.Max, 1e-10)).IsTrue();
		}
	}

	[Test]
	public async Task ComputeEqualPartsBboxes_Asymmetric()
	{
		var bbox = new AlignedBox3d(new Vector3d(0, 0, 0), new Vector3d(6, 4, 2));
		var split = (3, 2, 1);

		List<AlignedBox3d> bboxes = Bbox.ComputeEqualPartsBboxes(bbox, split);

		await Assert.That(bboxes.Count).IsEqualTo(6);
		using (Assert.Multiple())
		{
			foreach (AlignedBox3d subBbox in bboxes)
			{
				Vector3d diag = subBbox.Diagonal();
				await Assert.That(EigenMatrixNear(diag, new Vector3d(2.0, 2.0, 2.0), 1e-10)).IsTrue();
			}
		}
	}

	[Test]
	public async Task ComputeEqualPartsBboxes_WithOffset()
	{
		var bbox = new AlignedBox3d(new Vector3d(10, 20, 30), new Vector3d(20, 30, 40));
		var split = (2, 2, 2);

		List<AlignedBox3d> bboxes = Bbox.ComputeEqualPartsBboxes(bbox, split);

		await Assert.That(bboxes.Count).IsEqualTo(8);

		// Check that sub-boxes cover the original box
		var covered = new AlignedBox3d();
		foreach (AlignedBox3d subBbox in bboxes)
		{
			covered = covered.Extend(subBbox);
		}

		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(covered.Min, bbox.Min, 1e-10)).IsTrue();
			await Assert.That(EigenMatrixNear(covered.Max, bbox.Max, 1e-10)).IsTrue();
		}
	}
}
