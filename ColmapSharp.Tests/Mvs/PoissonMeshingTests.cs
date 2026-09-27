// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoissonMeshingTests: colmap/mvs/poisson_meshing_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Mvs/PoissonMeshing.cs through its file
// wrapper. Tier C (outcome): COLMAP's cases only check that a readable mesh comes out. C#-only
// cases are in PoissonMeshingTests.CSharpOnly.cs; the match against pycolmap is
// PoissonMeshingOracleTests.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PoissonMeshingTests
{
	private static void WriteRandomPlyPoints(string path, int numPoints = 100)
	{
		var plyPoints = new List<PlyPoint>(numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			Vector3d point3D = RandomEigen.RandomEigenVector3d();
			plyPoints.Add(new PlyPoint
			{
				X = (float)point3D.X,
				Y = (float)point3D.Y,
				Z = (float)point3D.Z,
				Nx = 0.0f,
				Ny = 0.0f,
				Nz = 1.0f,
				R = 0,
				G = 64,
				B = 128,
			});
		}

		Ply.WriteBinaryPlyPoints(path, plyPoints, writeNormal: true, writeRgb: true);
	}

	[Test]
	public async Task PoissonMeshing_Integration()
	{
		string testDir = MvsTestUtils.CreateTestDir();
		string inputPath = Path.Combine(testDir, "points.ply");
		string outputPath = Path.Combine(testDir, "mesh.ply");
		WriteRandomPlyPoints(inputPath);

		var options = new PoissonMeshingOptions
		{
			PointWeight = 1.0,
			Depth = 3,  // Use smaller depth for faster test
			Trim = 0.0, // Disable trimming
			NumThreads = 1,
		};

		await Assert.That(PoissonMeshing.Run(options, inputPath, outputPath)).IsTrue();

		await Assert.That(File.Exists(outputPath)).IsTrue();
		List<PlyPoint> meshVertices = Ply.ReadPly(outputPath);
		await Assert.That(meshVertices.Count).IsGreaterThanOrEqualTo(3);
	}

	[Test]
	public async Task PoissonMeshing_WithTrimming()
	{
		string testDir = MvsTestUtils.CreateTestDir();
		string inputPath = Path.Combine(testDir, "points.ply");
		string outputPath = Path.Combine(testDir, "mesh.ply");
		WriteRandomPlyPoints(inputPath);

		var options = new PoissonMeshingOptions
		{
			PointWeight = 1.0,
			Depth = 3,
			Trim = 5.0,
			NumThreads = 1,
		};

		await Assert.That(PoissonMeshing.Run(options, inputPath, outputPath)).IsTrue();
		await Assert.That(File.Exists(outputPath)).IsTrue();
		List<PlyPoint> meshVertices = Ply.ReadPly(outputPath);

		// With random data and trimming, we can't make strong assumptions about
		// the number of vertices, but reading the file ensures valid PLY format.
		await Assert.That(meshVertices.Count).IsGreaterThanOrEqualTo(0);
	}
}
