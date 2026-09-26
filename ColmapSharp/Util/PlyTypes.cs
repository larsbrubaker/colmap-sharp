// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PlyTypes: the data types of colmap/util/ply.h - a point of a point cloud (PlyPoint) and a
// triangle mesh (PlyMeshVertex, PlyMeshFace, PlyMesh, PlyTexturedMesh). Ply.cs reads them
// from PLY files and Ply.Write.cs writes them. These are also the shapes MatterCAD reads a
// reconstruction's sparse/dense points and meshes through. Tests:
// ColmapSharp.Tests/Util/PlyTests.cs (ply_test.cc 1:1).
//
// A face's vertex indices are int here (size_t in C++): a List index is an int, and COLMAP
// writes them to binary files as int anyway.

namespace ColmapSharp.Util;

/// <summary>Port of colmap::PlyPoint: a point with a normal and an RGB color, all zero by default.</summary>
public struct PlyPoint
{
	/// <summary>Position x.</summary>
	public float X;
	/// <summary>Position y.</summary>
	public float Y;
	/// <summary>Position z.</summary>
	public float Z;
	/// <summary>Normal x.</summary>
	public float Nx;
	/// <summary>Normal y.</summary>
	public float Ny;
	/// <summary>Normal z.</summary>
	public float Nz;
	/// <summary>Red.</summary>
	public byte R;
	/// <summary>Green.</summary>
	public byte G;
	/// <summary>Blue.</summary>
	public byte B;
}

/// <summary>Port of colmap::PlyMeshVertex: a mesh vertex position and color (gray by default).</summary>
public struct PlyMeshVertex
{
	// The default color is gray.
	private const byte DefaultColor = 200;

	/// <summary>A vertex at the origin with the default gray color.</summary>
	public PlyMeshVertex()
		: this(0, 0, 0)
	{
	}

	/// <summary>A vertex at (x, y, z) with the default gray color.</summary>
	public PlyMeshVertex(float x, float y, float z)
		: this(x, y, z, DefaultColor, DefaultColor, DefaultColor)
	{
	}

	/// <summary>A vertex at (x, y, z) with color (r, g, b).</summary>
	public PlyMeshVertex(float x, float y, float z, byte r, byte g, byte b)
	{
		X = x;
		Y = y;
		Z = z;
		R = r;
		G = g;
		B = b;
	}

	/// <summary>Position x.</summary>
	public float X;
	/// <summary>Position y.</summary>
	public float Y;
	/// <summary>Position z.</summary>
	public float Z;
	/// <summary>Red.</summary>
	public byte R;
	/// <summary>Green.</summary>
	public byte G;
	/// <summary>Blue.</summary>
	public byte B;
}

/// <summary>Port of colmap::PlyMeshFace: a triangle as three indices into the vertex list.</summary>
public struct PlyMeshFace
{
	/// <summary>A triangle over the three vertex indices.</summary>
	public PlyMeshFace(int vertexIdx1, int vertexIdx2, int vertexIdx3)
	{
		VertexIdx1 = vertexIdx1;
		VertexIdx2 = vertexIdx2;
		VertexIdx3 = vertexIdx3;
	}

	/// <summary>Index of the first vertex.</summary>
	public int VertexIdx1;
	/// <summary>Index of the second vertex.</summary>
	public int VertexIdx2;
	/// <summary>Index of the third vertex.</summary>
	public int VertexIdx3;
}

/// <summary>Port of colmap::PlyMesh: a triangle mesh.</summary>
public sealed class PlyMesh
{
	/// <summary>The vertices.</summary>
	public List<PlyMeshVertex> Vertices { get; set; } = [];

	/// <summary>The triangles.</summary>
	public List<PlyMeshFace> Faces { get; set; } = [];
}

/// <summary>
/// Port of colmap::PlyTexturedMesh: a mesh with optional per-face texture coordinates and the
/// texture image they refer to.
/// </summary>
public sealed class PlyTexturedMesh
{
	/// <summary>An empty mesh with no texture.</summary>
	public PlyTexturedMesh()
	{
	}

	/// <summary>C++'s <c>PlyTexturedMesh{mesh}</c>: <paramref name="mesh"/> without a texture.</summary>
	public PlyTexturedMesh(PlyMesh mesh)
	{
		Mesh = mesh;
	}

	/// <summary>The triangle mesh.</summary>
	public PlyMesh Mesh { get; set; } = new();

	/// <summary>Per-face UV coordinates: 6 floats per face (u1,v1, u2,v2, u3,v3), or empty.</summary>
	public List<float> FaceUvs { get; set; } = [];

	/// <summary>The texture image file name, "comment TextureFile ..." in the header; empty for none.</summary>
	public string TextureFile { get; set; } = "";
}
