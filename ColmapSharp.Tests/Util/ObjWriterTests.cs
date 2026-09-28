// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ObjWriterTests: C#-only tests of ColmapSharp/Util/ObjWriter.cs, not a port - COLMAP has no OBJ
// writer. They pin the exact text of a tiny untextured and textured mesh (element counts,
// 1-based "f v" and "f v/vt" indices, per-corner "vt" lines with COLMAP's UVs unflipped, vertex
// colors in [0, 1]), the MTL, that floats round-trip and ignore a comma-decimal current culture,
// and that the path convenience writes the MTL next to the OBJ.

using System.Globalization;
using System.Text;

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class ObjWriterTests
{
	// Two triangles over four vertices sharing an edge, each corner with its own UV.
	private static PlyTexturedMesh TinyTexturedMesh() => new(new PlyMesh
	{
		Vertices =
		[
			new PlyMeshVertex(0, 0, 0, 255, 0, 51),
			new PlyMeshVertex(1.5f, 0, 0),
			new PlyMeshVertex(0, 0.1f, -2),
			new PlyMeshVertex(1, 1, 1e-7f),
		],
		Faces = [new PlyMeshFace(0, 1, 2), new PlyMeshFace(2, 1, 3)],
	})
	{
		FaceUvs = [0, 0, 0.5f, 0, 0, 0.25f, 0, 0.25f, 0.5f, 0, 1, 1],
		TextureFile = "texture.png",
	};

	private static string ToText(Action<Stream> write)
	{
		using var stream = new MemoryStream();
		write(stream);
		return Encoding.UTF8.GetString(stream.ToArray());
	}

	[Test]
	public async Task CSharpOnly_WritesUntexturedMeshWithColors()
	{
		PlyMesh mesh = TinyTexturedMesh().Mesh;
		string text = ToText(s => ObjWriter.WriteObj(s, mesh));

		await Assert.That(text).IsEqualTo(
			"# Written by ColmapSharp\n" +
			"v 0 0 0 1 0 0.2\n" +
			"v 1.5 0 0 0.78431374 0.78431374 0.78431374\n" +
			"v 0 0.1 -2 0.78431374 0.78431374 0.78431374\n" +
			"v 1 1 1E-07 0.78431374 0.78431374 0.78431374\n" +
			"f 1 2 3\n" +
			"f 3 2 4\n");

		string plain = ToText(s => ObjWriter.WriteObj(s, mesh, writeColors: false));
		await Assert.That(plain.Split('\n')[1]).IsEqualTo("v 0 0 0");
	}

	[Test]
	public async Task CSharpOnly_WritesTexturedMeshAndMtl()
	{
		PlyTexturedMesh mesh = TinyTexturedMesh();
		string obj = ToText(s => ObjWriter.WriteTexturedObj(s, mesh, "mesh.mtl"));
		string mtl = ToText(s => ObjWriter.WriteMtl(s, mesh.TextureFile));

		await Assert.That(obj).IsEqualTo(
			"# Written by ColmapSharp\n" +
			"mtllib mesh.mtl\n" +
			"v 0 0 0\n" +
			"v 1.5 0 0\n" +
			"v 0 0.1 -2\n" +
			"v 1 1 1E-07\n" +
			"vt 0 0\n" +
			"vt 0.5 0\n" +
			"vt 0 0.25\n" +
			"vt 0 0.25\n" +
			"vt 0.5 0\n" +
			"vt 1 1\n" +
			"usemtl texture\n" +
			"f 1/1 2/2 3/3\n" +
			"f 3/4 2/5 4/6\n");

		string[] lines = obj.Split('\n');
		await Assert.That(lines.Count(l => l.StartsWith("v ", StringComparison.Ordinal))).IsEqualTo(mesh.Mesh.Vertices.Count);
		await Assert.That(lines.Count(l => l.StartsWith("vt ", StringComparison.Ordinal))).IsEqualTo(mesh.Mesh.Faces.Count * 3);
		await Assert.That(lines.Count(l => l.StartsWith("f ", StringComparison.Ordinal))).IsEqualTo(mesh.Mesh.Faces.Count);

		await Assert.That(mtl).IsEqualTo(
			"# Written by ColmapSharp\n" +
			"newmtl texture\n" +
			"Ka 1 1 1\n" +
			"Kd 1 1 1\n" +
			"Ks 0 0 0\n" +
			"d 1\n" +
			"illum 1\n" +
			"map_Kd texture.png\n");
	}

	[Test]
	public async Task CSharpOnly_FloatsRoundTripAndIgnoreCommaDecimalCulture()
	{
		var mesh = new PlyMesh
		{
			Vertices = [new PlyMeshVertex(0.1f, -123456.79f, 3.4028235e38f), new PlyMeshVertex(1.17549435e-38f, 1f / 3, 2.5f)],
			Faces = [],
		};

		CultureInfo originalCulture = CultureInfo.CurrentCulture;
		var commaDecimal = (CultureInfo)CultureInfo.InvariantCulture.Clone();
		commaDecimal.NumberFormat.NumberDecimalSeparator = ",";
		string text;
		string formatted;
		try
		{
			CultureInfo.CurrentCulture = commaDecimal;
			formatted = 2.5f.ToString();
			text = ToText(s => ObjWriter.WriteObj(s, mesh, writeColors: false));
		}
		finally
		{
			CultureInfo.CurrentCulture = originalCulture;
		}

		// The culture really was comma-decimal.
		await Assert.That(formatted).IsEqualTo("2,5");
		await Assert.That(text.Contains(',', StringComparison.Ordinal)).IsFalse();

		string[] vertexLines = text.Split('\n').Where(l => l.StartsWith("v ", StringComparison.Ordinal)).ToArray();
		for (int i = 0; i < mesh.Vertices.Count; ++i)
		{
			float[] values = vertexLines[i][2..].Split(' ').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
			await Assert.That(values[0]).IsEqualTo(mesh.Vertices[i].X);
			await Assert.That(values[1]).IsEqualTo(mesh.Vertices[i].Y);
			await Assert.That(values[2]).IsEqualTo(mesh.Vertices[i].Z);
		}
	}

	[Test]
	public async Task CSharpOnly_PathWritesMtlNextToObj()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-obj-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		string objText, mtlText;
		try
		{
			ObjWriter.WriteTexturedObj(Path.Combine(dir, "mesh.obj"), TinyTexturedMesh());
			objText = File.ReadAllText(Path.Combine(dir, "mesh.obj"));
			mtlText = File.ReadAllText(Path.Combine(dir, "mesh.mtl"));
		}
		finally
		{
			Directory.Delete(dir, recursive: true);
		}

		await Assert.That(objText.Split('\n')[1]).IsEqualTo("mtllib mesh.mtl");
		await Assert.That(mtlText.EndsWith("map_Kd texture.png\n", StringComparison.Ordinal)).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_RejectsMeshWithoutTextureOrWithWrongUvCount()
	{
		PlyTexturedMesh untextured = new(TinyTexturedMesh().Mesh);
		PlyTexturedMesh shortUvs = TinyTexturedMesh();
		shortUvs.FaceUvs.RemoveAt(0);

		await Assert.That(() => ToText(s => ObjWriter.WriteTexturedObj(s, untextured, "mesh.mtl"))).Throws<ArgumentException>();
		await Assert.That(() => ToText(s => ObjWriter.WriteTexturedObj(s, shortUvs, "mesh.mtl"))).Throws<ArgumentException>();
	}
}
