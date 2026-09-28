// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of ReconstructionSession's housekeeping (demo/ColmapDemo/ReconstructionSession.cs): the
// saved texture follows the OBJ's name and is overwritten on a re-save, and a run's temp
// workspace is deleted however the run ends unless the developer asked to keep it.

using ColmapSharp.Controllers;
using ColmapSharp.Util;
using MatterHackers.Agg;
using MatterHackers.Agg.Image;

namespace ColmapDemo.Tests;

public class SessionTests
{
	private static PlyMesh Triangle()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(1, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(0, 1, 0));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 2));
		return mesh;
	}

	private static SessionResult Textured(Color color)
	{
		var atlas = new ImageBuffer(2, 2);
		atlas.NewGraphics2D().Clear(color);
		return new SessionResult { Mesh = Triangle(), FaceUvs = [0, 0, 1, 0, 0, 1], Atlas = atlas };
	}

	private static Color PixelOf(string png) => ImageIO.LoadImage(png).GetPixel(0, 0);

	[Test]
	public async Task ResavingOverwritesTheTexture()
	{
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		string obj = Path.Combine(dir, "model.obj");

		ReconstructionSession.SaveMesh(Textured(Color.Red), obj);
		ReconstructionSession.SaveMesh(Textured(Color.Blue), obj);

		await Assert.That(PixelOf(Path.Combine(dir, "model.png"))).IsEqualTo(Color.Blue);
		Directory.Delete(dir, recursive: true);
	}

	[Test]
	public async Task TwoMeshesInOneFolderKeepTheirOwnTextures()
	{
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;

		ReconstructionSession.SaveMesh(Textured(Color.Red), Path.Combine(dir, "a.obj"));
		ReconstructionSession.SaveMesh(Textured(Color.Blue), Path.Combine(dir, "b.obj"));

		await Assert.That(File.ReadAllText(Path.Combine(dir, "a.mtl"))).Contains("map_Kd a.png");
		await Assert.That(File.ReadAllText(Path.Combine(dir, "b.mtl"))).Contains("map_Kd b.png");
		await Assert.That(PixelOf(Path.Combine(dir, "a.png"))).IsEqualTo(Color.Red);
		await Assert.That(PixelOf(Path.Combine(dir, "b.png"))).IsEqualTo(Color.Blue);
		Directory.Delete(dir, recursive: true);
	}

	[Test]
	[Arguments(false, false, 0)]
	[Arguments(true, false, 0)]
	[Arguments(false, true, 1)]
	public async Task TheWorkspaceIsDeletedUnlessKept(bool cancelled, bool keep, int expectedLeft)
	{
		string root = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		var session = new ReconstructionSession(new SessionSettings { WorkspaceRoot = root, KeepWorkspace = keep });
		using var cancel = new CancellationTokenSource();
		if (cancelled)
		{
			cancel.Cancel();
		}

		// No photos: the run ends at once (without a model, or cancelled), which is all this needs.
		try
		{
			await session.RunAsync(new InMemoryImageSource(), cancel.Token);
		}
		catch (OperationCanceledException)
		{
		}

		await Assert.That(Directory.GetDirectories(root).Length).IsEqualTo(expectedLeft);
		Directory.Delete(root, recursive: true);
	}
}
