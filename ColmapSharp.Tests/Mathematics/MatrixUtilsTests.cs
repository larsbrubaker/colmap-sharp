// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MatrixUtilsTests: colmap/math/matrix_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/MatrixUtils.cs.
// Named MatrixUtilsTests rather than MatrixTests so it does not collide with the C#-only
// LinearAlgebra.MatrixTests in a --treenode-filter.
// Tier B (goes through Householder QR): the tolerances are COLMAP's.
// RandomEigenMatrixd<4, 4>() is RandomEigen.RandomEigenMatrixXd(4, 4) (same draw order).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class MatrixUtilsTests
{
	[Test]
	public async Task DecomposeMatrixRQ_Nominal()
	{
		for (int i = 0; i < 10; ++i)
		{
			MatrixXd a = RandomEigen.RandomEigenMatrixXd(4, 4);

			MatrixUtils.DecomposeMatrixRQ(a, out MatrixXd r, out MatrixXd q);

			MatrixXd rq = r * q;
			using (Assert.Multiple())
			{
				await Assert.That(r.BottomRows(4).IsUpperTriangular()).IsTrue();
				await Assert.That(q.IsUnitary()).IsTrue();
				await Assert.That(q.Determinant()).IsEqualTo(1.0).Within(1e-6);
				for (int k = 0; k < a.Size; k++)
				{
					await Assert.That(a.AsSpan()[k]).IsEqualTo(rq.AsSpan()[k]).Within(1e-6);
				}
			}
		}
	}
}
