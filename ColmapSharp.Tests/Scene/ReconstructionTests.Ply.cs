// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests.Ply: reconstruction_test.cc's ConvertToPLY and ImportPLYFromVector,
// testing ColmapSharp/Scene/Reconstruction.Ply.cs. std::find_if over the points becomes a
// LINQ search with the same predicate and tolerance.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	[Test]
	public async Task Reconstruction_ConvertToPLY()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		reconstruction.AddPoint3D(new Vector3d(1, 2, 3), new Track(), new Vector3ub(255, 0, 0));
		reconstruction.AddPoint3D(new Vector3d(4, 5, 6), new Track(), new Vector3ub(0, 255, 0));

		List<PlyPoint> plyPoints = reconstruction.ConvertToPLY();
		await Assert.That(plyPoints.Count).IsEqualTo(2);

		// Decompose into individual EXPECT calls by locating both expected points.
		int p1 = plyPoints.FindIndex(p => Math.Abs(p.X - 1.0) < 1e-6);
		int p2 = plyPoints.FindIndex(p => Math.Abs(p.X - 4.0) < 1e-6);

		await Assert.That(p1).IsNotEqualTo(-1);
		await Assert.That(p2).IsNotEqualTo(-1);

		await Assert.That(plyPoints[p1].Y).IsEqualTo(2.0f).Within(1e-6f);
		await Assert.That(plyPoints[p1].Z).IsEqualTo(3.0f).Within(1e-6f);
		await Assert.That(plyPoints[p1].R).IsEqualTo((byte)255);
		await Assert.That(plyPoints[p1].G).IsEqualTo((byte)0);
		await Assert.That(plyPoints[p1].B).IsEqualTo((byte)0);

		await Assert.That(plyPoints[p2].Y).IsEqualTo(5.0f).Within(1e-6f);
		await Assert.That(plyPoints[p2].Z).IsEqualTo(6.0f).Within(1e-6f);
		await Assert.That(plyPoints[p2].R).IsEqualTo((byte)0);
		await Assert.That(plyPoints[p2].G).IsEqualTo((byte)255);
		await Assert.That(plyPoints[p2].B).IsEqualTo((byte)0);
	}

	[Test]
	public async Task Reconstruction_ImportPLYFromVector()
	{
		var reconstruction = new Reconstruction();
		var plyPoints = new List<PlyPoint>
		{
			new() { X = 1, Y = 2, Z = 3, R = 100, G = 150, B = 200 },
			new() { X = 4, Y = 5, Z = 6, R = 50, G = 60, B = 70 },
		};

		reconstruction.ImportPLY(plyPoints);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(2);

		// Verify the points were imported correctly by locating both expected points.
		Point3D? p1 = reconstruction.Points3D.Values.FirstOrDefault(p => Math.Abs(p.Xyz.X - 1.0) < 1e-6);
		Point3D? p2 = reconstruction.Points3D.Values.FirstOrDefault(p => Math.Abs(p.Xyz.X - 4.0) < 1e-6);

		await Assert.That(p1).IsNotNull();
		await Assert.That(p2).IsNotNull();

		await Assert.That(p1!.Xyz.Y).IsEqualTo(2.0).Within(1e-6);
		await Assert.That(p1.Xyz.Z).IsEqualTo(3.0).Within(1e-6);
		await Assert.That(p1.Color).IsEqualTo(new Vector3ub(100, 150, 200));

		await Assert.That(p2!.Xyz.Y).IsEqualTo(5.0).Within(1e-6);
		await Assert.That(p2.Xyz.Z).IsEqualTo(6.0).Within(1e-6);
		await Assert.That(p2.Color).IsEqualTo(new Vector3ub(50, 60, 70));
	}

	[Test]
	public async Task CSharpOnly_ImportPLYFromFileReplacesPoints()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-ply-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		string path = Path.Combine(dir, "points.ply");
		Ply.WriteBinaryPlyPoints(path, [new PlyPoint { X = 7, Y = 8, Z = 9, R = 1, G = 2, B = 3 }]);

		var reconstruction = new Reconstruction();
		reconstruction.AddPoint3D(new Vector3d(0, 0, 0), new Track());
		reconstruction.ImportPLY(path);

		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(1);
		Point3D point = reconstruction.Points3D.Values.Single();
		await Assert.That(point.Xyz).IsEqualTo(new Vector3d(7, 8, 9));
		await Assert.That(point.Color).IsEqualTo(new Vector3ub(1, 2, 3));
		await Assert.That(point.Track.Length).IsEqualTo(0);
	}
}
