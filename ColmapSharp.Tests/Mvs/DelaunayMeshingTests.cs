// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshingTests: colmap/mvs/delaunay_meshing_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mvs/DelaunayMeshing.cs. Tier C: the
// assertions are COLMAP's (a mesh with at least 3 vertices is written and read back).
//
// The C# API is in memory, so where COLMAP reads the sparse model from sparse_path the
// synthetic Reconstruction is passed directly. The mesh still goes through
// WriteBinaryPlyMesh and ReadPly like the C++ test, and the dense test still writes
// fused.ply and fused.ply.vis and reads them back to build the input.

using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class DelaunayMeshingTests
{
	private static Reconstruction CreateSyntheticReconstruction(int numFrames = 5, int numPoints3D = 100)
	{
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFrames,
			NumPoints3D = numPoints3D,
		};
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(options, reconstruction);
		return reconstruction;
	}

	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-delaunay-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	[Test]
	public async Task SparseDelaunayMeshing_Integration()
	{
		string testDir = CreateTestDir();
		string outputPath = Path.Combine(testDir, "mesh.ply");
		var reconstruction = CreateSyntheticReconstruction();

		var options = new DelaunayMeshingOptions { NumThreads = 1 };
		var mesh = DelaunayMeshing.SparseDelaunayMeshing(options, reconstruction);
		Ply.WriteBinaryPlyMesh(outputPath, new PlyTexturedMesh(mesh));

		await Assert.That(File.Exists(outputPath)).IsTrue();
		var meshVertices = Ply.ReadPly(outputPath);
		await Assert.That(meshVertices.Count).IsGreaterThanOrEqualTo(3);
	}

	[Test]
	public async Task SparseDelaunayMeshing_NonSubsampled()
	{
		string testDir = CreateTestDir();
		string outputPath = Path.Combine(testDir, "mesh.ply");
		var reconstruction = CreateSyntheticReconstruction();

		// Setting max_proj_dist=0 exercises the non-subsampled
		// CreateDelaunayTriangulation() path instead of
		// CreateSubSampledDelaunayTriangulation().
		var options = new DelaunayMeshingOptions { MaxProjDist = 0, NumThreads = 1 };
		var mesh = DelaunayMeshing.SparseDelaunayMeshing(options, reconstruction);
		Ply.WriteBinaryPlyMesh(outputPath, new PlyTexturedMesh(mesh));

		await Assert.That(File.Exists(outputPath)).IsTrue();
		var meshVertices = Ply.ReadPly(outputPath);
		await Assert.That(meshVertices.Count).IsGreaterThanOrEqualTo(3);
	}

	[Test]
	public async Task DenseDelaunayMeshing_Integration()
	{
		string testDir = CreateTestDir();
		string outputPath = Path.Combine(testDir, "mesh.ply");
		var reconstruction = CreateSyntheticReconstruction(3, 50);

		// Create fused.ply from reconstruction points
		var plyPoints = new List<PlyPoint>(reconstruction.NumPoints3D);
		foreach (var (_, point3D) in reconstruction.Points3D)
		{
			plyPoints.Add(new PlyPoint
			{
				X = (float)point3D.Xyz.X,
				Y = (float)point3D.Xyz.Y,
				Z = (float)point3D.Xyz.Z,
				Nx = 0.0f,
				Ny = 0.0f,
				Nz = 1.0f,
				R = 128,
				G = 128,
				B = 128,
			});
		}

		string fusedPath = Path.Combine(testDir, "fused.ply");
		Ply.WriteBinaryPlyPoints(fusedPath, plyPoints, writeNormal: true, writeRgb: true);

		// Create fused.ply.vis: for each point, list visible image indices.
		// Each point is visible in all images to give sufficient multi-view
		// information for the graph-cut optimization.
		string visPath = Path.Combine(testDir, "fused.ply.vis");
		int numVisible = reconstruction.NumRegImages;
		var visibility = new List<IReadOnlyList<int>>();
		for (int i = 0; i < plyPoints.Count; ++i)
		{
			visibility.Add(Enumerable.Range(0, numVisible).ToArray());
		}

		StereoFusion.WritePointsVisibility(visPath, visibility);

		var options = new DelaunayMeshingOptions { NumThreads = 1 };
		var readPoints = Ply.ReadPly(fusedPath);
		var readVisibility = StereoFusion.ReadPointsVisibility(visPath, readPoints.Count);
		var mesh = DelaunayMeshing.DenseDelaunayMeshing(options, reconstruction, readPoints, readVisibility);
		Ply.WriteBinaryPlyMesh(outputPath, new PlyTexturedMesh(mesh));

		await Assert.That(File.Exists(outputPath)).IsTrue();
		var meshVertices = Ply.ReadPly(outputPath);
		await Assert.That(meshVertices.Count).IsGreaterThanOrEqualTo(3);
	}
}
