// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of MeshBridge (demo/ColmapDemo/MeshBridge.cs) and ReconstructionSession.SaveMesh: the
// library's PlyMesh reaches agg with its geometry, per-face colors or per-face texture UVs, and a
// saved mesh is the set of files a viewer needs (OBJ + MTL + a PNG named after the OBJ when textured).

using ColmapSharp.Util;
using MatterHackers.Agg;
using MatterHackers.Agg.Image;
using MatterHackers.PolygonMesh;

namespace ColmapDemo.Tests;

public class MeshBridgeTests
{
	private static PlyMesh TwoTriangles()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0, 0, 0, 30, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(1, 0, 0, 60, 90, 0));
		mesh.Vertices.Add(new PlyMeshVertex(0, 1, 0, 0, 0, 150));
		mesh.Vertices.Add(new PlyMeshVertex(1, 1, 2, 255, 255, 255));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 2));
		mesh.Faces.Add(new PlyMeshFace(1, 3, 2));
		return mesh;
	}

	private static readonly float[] Uvs = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1];

	[Test]
	public async Task UntexturedMeshKeepsGeometryAndAveragesVertexColorsPerFace()
	{
		Mesh mesh = MeshBridge.ToAggMesh(TwoTriangles());

		await Assert.That(mesh.Vertices.Count).IsEqualTo(4);
		await Assert.That(mesh.Faces.Count).IsEqualTo(2);
		await Assert.That(mesh.Faces[1].v0).IsEqualTo(1);
		await Assert.That(mesh.Faces[1].v1).IsEqualTo(3);
		await Assert.That(mesh.Faces[1].v2).IsEqualTo(2);
		await Assert.That((double)mesh.Vertices[3].Z).IsEqualTo(2.0);
		await Assert.That(mesh.FaceTextures.Count).IsEqualTo(0);
		await Assert.That(mesh.FaceColors[0]).IsEqualTo(new Color(30, 30, 50));
	}

	[Test]
	public async Task TexturedMeshGetsOneUvTriplePerFaceFromTheAtlas()
	{
		var atlas = new ImageBuffer(4, 4);

		Mesh mesh = MeshBridge.ToAggMesh(TwoTriangles(), atlas, Uvs);

		await Assert.That(mesh.FaceTextures.Count).IsEqualTo(2);
		FaceTextureData second = mesh.FaceTextures[1];
		await Assert.That(second.image).IsSameReferenceAs(atlas);
		await Assert.That((double)second.uv0.X).IsEqualTo(1.0);
		await Assert.That((double)second.uv1.Y).IsEqualTo(1.0);
		await Assert.That((double)second.uv2.X).IsEqualTo(0.0);
	}

	[Test]
	public async Task EmptyAtlasFallsBackToFaceColors()
	{
		Mesh mesh = MeshBridge.ToAggMesh(TwoTriangles(), new ImageBuffer(), Uvs);

		await Assert.That(mesh.FaceTextures.Count).IsEqualTo(0);
		await Assert.That(mesh.FaceColors).IsNotNull();
	}

	[Test]
	public async Task SavingATexturedMeshWritesObjMtlAndTexture()
	{
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		var result = new SessionResult { Mesh = TwoTriangles(), FaceUvs = Uvs, Atlas = new ImageBuffer(4, 4) };

		ReconstructionSession.SaveMesh(result, Path.Combine(dir, "model.obj"));

		await Assert.That(File.Exists(Path.Combine(dir, "model.obj"))).IsTrue();
		await Assert.That(File.ReadAllText(Path.Combine(dir, "model.mtl"))).Contains("map_Kd model.png");
		await Assert.That(File.Exists(Path.Combine(dir, "model.png"))).IsTrue();
		Directory.Delete(dir, recursive: true);
	}

	[Test]
	public async Task SavingAnUntexturedMeshWritesAColoredObjOnly()
	{
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		var result = new SessionResult { Mesh = TwoTriangles() };

		ReconstructionSession.SaveMesh(result, Path.Combine(dir, "model.obj"));

		string obj = File.ReadAllText(Path.Combine(dir, "model.obj"));
		await Assert.That(obj).DoesNotContain("mtllib");
		await Assert.That(obj.Split('\n').Count(l => l.StartsWith("v "))).IsEqualTo(4);
		await Assert.That(File.Exists(Path.Combine(dir, "model.png"))).IsFalse();
		Directory.Delete(dir, recursive: true);
	}

	[Test]
	public async Task SavingAsPlyReadsBack()
	{
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		var result = new SessionResult { Mesh = TwoTriangles() };

		ReconstructionSession.SaveMesh(result, Path.Combine(dir, "model.ply"));

		PlyTexturedMesh back = Ply.ReadPlyMesh(Path.Combine(dir, "model.ply"));
		await Assert.That(back.Mesh.Faces.Count).IsEqualTo(2);
		await Assert.That(back.Mesh.Vertices.Count).IsEqualTo(4);
		Directory.Delete(dir, recursive: true);
	}
}
