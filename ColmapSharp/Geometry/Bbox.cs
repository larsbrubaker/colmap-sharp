// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Bbox: colmap/geometry/bbox.h and bbox.cc - splitting a bounding box into a grid of equal
// sub-boxes (used by the MVS and clustering code to tile a scene). The box type is
// LinearAlgebra/AlignedBox3d.cs. Tests: ColmapSharp.Tests/Geometry/BboxTests.cs
// (bbox_test.cc 1:1).
//
// Tier A (exact): the same arithmetic as COLMAP, min + i * size per coefficient.
// Translation note: Eigen::Vector3i split becomes a (X, Y, Z) int tuple; there is no
// integer vector type in LinearAlgebra yet and nothing else needs one.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/bbox.h.
/// </summary>
public static class Bbox
{
	/// <summary>
	/// Divide a bounding box into equal-sized sub-boxes. <paramref name="split"/> is the
	/// number of splits along each axis. Returns the sub-boxes covering the original box,
	/// x fastest, then y, then z. Port of colmap::ComputeEqualPartsBboxes.
	/// </summary>
	public static List<AlignedBox3d> ComputeEqualPartsBboxes(AlignedBox3d bbox, (int X, int Y, int Z) split)
	{
		Check.Gt(split.X, 0);
		Check.Gt(split.Y, 0);
		Check.Gt(split.Z, 0);

		Vector3d extent = bbox.Diagonal();
		var size = new Vector3d(extent.X / split.X, extent.Y / split.Y, extent.Z / split.Z);

		var bboxes = new List<AlignedBox3d>(split.X * split.Y * split.Z);
		for (int k = 0; k < split.Z; ++k)
		{
			for (int j = 0; j < split.Y; ++j)
			{
				for (int i = 0; i < split.X; ++i)
				{
					var min = new Vector3d(
						bbox.Min.X + i * size.X,
						bbox.Min.Y + j * size.Y,
						bbox.Min.Z + k * size.Z);
					bboxes.Add(new AlignedBox3d(min, min + size));
				}
			}
		}

		return bboxes;
	}
}
