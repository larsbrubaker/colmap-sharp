// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction.Ply: Reconstruction::ConvertToPLY and the two ImportPLY overloads of
// colmap/scene/reconstruction.cc, which move 3D points to and from Util/Ply.cs's PlyPoint
// list. reconstruction_io's ExportPLY, which writes ConvertToPLY's result, is not ported
// yet. Tests:
// ColmapSharp.Tests/Scene/ReconstructionTests.Ply.cs.
//
// ConvertToPLY lists the points in ascending id order, where COLMAP lists them in its hash
// map's order (divergence 21).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class Reconstruction
{
	/// <summary>
	/// The 3D points as PLY points: position narrowed to float and color, with zero normals.
	/// </summary>
	public List<PlyPoint> ConvertToPLY()
	{
		var plyPoints = new List<PlyPoint>(_points3D.Count);
		foreach (Point3D point3D in _points3D.Values)
		{
			plyPoints.Add(new PlyPoint
			{
				X = (float)point3D.Xyz.X,
				Y = (float)point3D.Xyz.Y,
				Z = (float)point3D.Xyz.Z,
				R = point3D.Color.X,
				G = point3D.Color.Y,
				B = point3D.Color.Z,
			});
		}

		return plyPoints;
	}

	/// <summary>
	/// Replaces all 3D points with the points of the PLY file at <paramref name="path"/>, each
	/// with an empty track. As in COLMAP, only the point map is cleared: images keep their
	/// 2D-3D links and ids keep counting up from the old maximum.
	/// </summary>
	public void ImportPLY(string path)
	{
		_points3D.Clear();
		List<PlyPoint> plyPoints = Ply.ReadPly(path);
		AddPlyPoints(plyPoints);
	}

	/// <summary>Replaces all 3D points with <paramref name="plyPoints"/>, each with an empty track.</summary>
	public void ImportPLY(IReadOnlyList<PlyPoint> plyPoints)
	{
		_points3D.Clear();
		AddPlyPoints(plyPoints);
	}

	private void AddPlyPoints(IReadOnlyList<PlyPoint> plyPoints)
	{
		foreach (PlyPoint plyPoint in plyPoints)
		{
			AddOwnedPoint3D(
				new Vector3d(plyPoint.X, plyPoint.Y, plyPoint.Z),
				new Track(),
				new Vector3ub(plyPoint.R, plyPoint.G, plyPoint.B));
		}
	}
}
