// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ObjWriter: C#-only, not a port - COLMAP writes meshes only as PLY (Ply.Write.cs). Wavefront
// OBJ is what most viewers and modelling tools open with a texture, so a host (the demo app,
// MatterCAD) can hand a reconstruction's mesh to them: an untextured PlyMesh becomes an OBJ with
// optional vertex colors, and a PlyTexturedMesh (AutomaticReconstruction.Texture.cs writes one per
// model) becomes an OBJ plus an MTL whose map_Kd names the mesh's TextureFile. The library does
// not encode the texture image; the host writes it next to the MTL, as it does for mesh.ply.
// Tests: ColmapSharp.Tests/Util/ObjWriterTests.cs.
//
// Format choices:
// - Numbers use the invariant culture and float's shortest round-trip form, so reading the file
//   back gives the same floats on any machine.
// - Vertex colors are the widespread "v x y z r g b" extension with each channel in [0, 1].
// - COLMAP's UVs are per face corner (PlyTexturedMesh.FaceUvs, 6 floats per face), so each
//   corner gets its own "vt" line: face i's corners are vt 3i+1..3i+3 (OBJ indices are 1-based).
// - No V flip: COLMAP's ComputeFaceUVs already stores v = 1 - atlas_y / atlas_height, i.e. the
//   texture's bottom-left is (0, 0), which is OBJ's convention as well as PLY's.

using System.Globalization;
using System.Text;

namespace ColmapSharp.Util;

/// <summary>Writes meshes as Wavefront OBJ (and MTL for a textured mesh). C#-only.</summary>
public static class ObjWriter
{
	/// <summary>The one material a textured OBJ uses; the MTL defines it.</summary>
	public const string MaterialName = "texture";

	private static readonly UTF8Encoding Utf8NoBom = new(false);

	/// <summary>
	/// Writes <paramref name="mesh"/> to an OBJ file: positions, with each vertex's color after it
	/// when <paramref name="writeColors"/>, and its triangles.
	/// </summary>
	public static void WriteObj(string path, PlyMesh mesh, bool writeColors = true)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteObj(file, mesh, writeColors);
	}

	/// <summary>
	/// Writes <paramref name="mesh"/> as OBJ to a stream: positions, with each vertex's color after
	/// it when <paramref name="writeColors"/>, and its triangles. The stream is left open.
	/// </summary>
	public static void WriteObj(Stream stream, PlyMesh mesh, bool writeColors = true)
	{
		using var writer = new StreamWriter(stream, Utf8NoBom, 1 << 16, leaveOpen: true);
		writer.Write("# Written by ColmapSharp\n");
		WriteVertices(writer, mesh, writeColors);

		var line = new StringBuilder();
		foreach (PlyMeshFace face in mesh.Faces)
		{
			line.Clear();
			line.Append("f ").Append(I(face.VertexIdx1 + 1L))
				.Append(' ').Append(I(face.VertexIdx2 + 1L))
				.Append(' ').Append(I(face.VertexIdx3 + 1L)).Append('\n');
			writer.Write(line);
		}
	}

	/// <summary>
	/// Writes a textured mesh to <paramref name="objPath"/> and its material to an MTL file of the
	/// same name next to it (mesh.obj gets mesh.mtl). The MTL's map_Kd is the mesh's
	/// <see cref="PlyTexturedMesh.TextureFile"/>, relative to the MTL's folder, so the texture image
	/// must be written there under that name. Throws <see cref="ArgumentException"/> for a mesh
	/// without a texture; write its <see cref="PlyTexturedMesh.Mesh"/> with WriteObj instead.
	/// </summary>
	public static void WriteTexturedObj(string objPath, PlyTexturedMesh mesh)
	{
		string mtlPath = Path.ChangeExtension(objPath, ".mtl");
		CheckTextured(mesh);
		using (FileStream obj = FileOpen.OpenWrite(objPath))
		{
			WriteTexturedObj(obj, mesh, Path.GetFileName(mtlPath));
		}

		using FileStream mtl = FileOpen.OpenWrite(mtlPath);
		WriteMtl(mtl, mesh.TextureFile);
	}

	/// <summary>
	/// Writes a textured mesh as OBJ to a stream, naming <paramref name="mtlFileName"/> as its
	/// material library (write that with <see cref="WriteMtl"/>): positions, one texture coordinate
	/// per face corner, and faces as "f v/vt". The stream is left open. Throws
	/// <see cref="ArgumentException"/> for a mesh without a texture.
	/// </summary>
	public static void WriteTexturedObj(Stream stream, PlyTexturedMesh mesh, string mtlFileName)
	{
		CheckTextured(mesh);
		using var writer = new StreamWriter(stream, Utf8NoBom, 1 << 16, leaveOpen: true);
		writer.Write("# Written by ColmapSharp\n");
		writer.Write("mtllib " + mtlFileName + "\n");
		WriteVertices(writer, mesh.Mesh, writeColors: false);

		var line = new StringBuilder();
		List<float> uvs = mesh.FaceUvs;
		for (int i = 0; i < uvs.Count; i += 2)
		{
			line.Clear();
			line.Append("vt ").Append(F(uvs[i])).Append(' ').Append(F(uvs[i + 1])).Append('\n');
			writer.Write(line);
		}

		writer.Write("usemtl " + MaterialName + "\n");
		List<PlyMeshFace> faces = mesh.Mesh.Faces;
		for (int i = 0; i < faces.Count; ++i)
		{
			PlyMeshFace face = faces[i];
			long firstUv = (3L * i) + 1;
			line.Clear();
			line.Append("f ").Append(I(face.VertexIdx1 + 1L)).Append('/').Append(I(firstUv))
				.Append(' ').Append(I(face.VertexIdx2 + 1L)).Append('/').Append(I(firstUv + 1))
				.Append(' ').Append(I(face.VertexIdx3 + 1L)).Append('/').Append(I(firstUv + 2)).Append('\n');
			writer.Write(line);
		}
	}

	/// <summary>
	/// Writes the MTL of a textured OBJ to a stream: one material, <see cref="MaterialName"/>, whose
	/// diffuse map is <paramref name="textureFile"/> (a path relative to the MTL). White ambient and
	/// diffuse and no specular, so viewers show the texture's colors unchanged. The stream is left
	/// open.
	/// </summary>
	public static void WriteMtl(Stream stream, string textureFile)
	{
		using var writer = new StreamWriter(stream, Utf8NoBom, 1 << 10, leaveOpen: true);
		writer.Write("# Written by ColmapSharp\n");
		writer.Write("newmtl " + MaterialName + "\n");
		writer.Write("Ka 1 1 1\n");
		writer.Write("Kd 1 1 1\n");
		writer.Write("Ks 0 0 0\n");
		writer.Write("d 1\n");
		writer.Write("illum 1\n");
		writer.Write("map_Kd " + textureFile + "\n");
	}

	private static void WriteVertices(StreamWriter writer, PlyMesh mesh, bool writeColors)
	{
		var line = new StringBuilder();
		foreach (PlyMeshVertex vertex in mesh.Vertices)
		{
			line.Clear();
			line.Append("v ").Append(F(vertex.X)).Append(' ').Append(F(vertex.Y)).Append(' ').Append(F(vertex.Z));
			if (writeColors)
			{
				line.Append(' ').Append(F(vertex.R / 255f))
					.Append(' ').Append(F(vertex.G / 255f))
					.Append(' ').Append(F(vertex.B / 255f));
			}

			line.Append('\n');
			writer.Write(line);
		}
	}

	private static void CheckTextured(PlyTexturedMesh mesh)
	{
		if (mesh.FaceUvs.Count == 0 || mesh.TextureFile.Length == 0)
		{
			throw new ArgumentException(
				"The mesh has no texture (no UVs or no texture file); write mesh.Mesh with ObjWriter.WriteObj instead.",
				nameof(mesh));
		}

		if (mesh.FaceUvs.Count != (long)mesh.Mesh.Faces.Count * 6)
		{
			throw new ArgumentException(
				$"Expected 6 UV coordinates per face ({mesh.Mesh.Faces.Count} faces) but the mesh has {mesh.FaceUvs.Count}.",
				nameof(mesh));
		}
	}

	// Shortest string that parses back to the same float, in the invariant culture.
	private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);

	private static string I(long value) => value.ToString(CultureInfo.InvariantCulture);
}
