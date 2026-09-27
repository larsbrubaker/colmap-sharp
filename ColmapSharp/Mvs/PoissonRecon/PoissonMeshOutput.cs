// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonMeshOutput: the extracted surface as COLMAP's poisson_meshing writes it, in model
// coordinates. Ports the output side of Reconstructors.h's Implicit::extractLevelSet
// (TransformedOutputLevelSetVertexStream's position transform by Solve's unitCubeToModel,
// PoissonSampleSet.UnitCubeToModel here) and of PoissonRecon.cpp's WriteMesh for COLMAP's
// options: Reconstructors.streams.h's OutputVertexInfo keeps the position, the density weight
// as the PLY "value" property only when --density is passed (COLMAP passes it when it trims),
// and the auxiliary data. COLMAP's input is a PLY, so the auxiliary data is a DynamicFactory
// over the input's extra properties - COLMAP's fused.ply has red, green and blue as uchar - and
// PLY::Write converts each float channel to uchar the way PlyFile.inl's get_stored_item and
// write_binary_item do (PlyUChar). The triangles are PoissonLevelSetExtractor.Polygons in
// write order. Tier A against oracle/poisson_extract_harness.cc.
//
// The gradient transform of TransformedOutputLevelSetVertexStream is not ported: COLMAP never
// passes --normals, so the gradients are neither computed (gradientNormals off) nor written.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// An extracted Poisson surface in model coordinates: the vertices (position, optional
/// density value, optional color bytes) and the triangles.
/// </summary>
public sealed class PoissonMeshOutput
{
	private PoissonMeshOutput(int vertexCount, int colorChannels, float[] positions, float[]? values, byte[]? colors, int[] triangles)
	{
		VertexCount = vertexCount;
		ColorChannels = colorChannels;
		Positions = positions;
		Values = values;
		Colors = colors;
		Triangles = triangles;
	}

	/// <summary>The number of vertices.</summary>
	public int VertexCount { get; }

	/// <summary>The color channels per vertex (3, red, green, blue, in COLMAP; 0 without colors).</summary>
	public int ColorChannels { get; }

	/// <summary>The vertex positions in model coordinates, x, y, z per vertex.</summary>
	public float[] Positions { get; }

	/// <summary>
	/// The density value per vertex (the PLY "value" property SurfaceTrimmer trims by), or null
	/// when the extraction had no density.
	/// </summary>
	public float[]? Values { get; }

	/// <summary>The color bytes, <see cref="ColorChannels"/> per vertex, or null without colors.</summary>
	public byte[]? Colors { get; }

	/// <summary>The triangles, three vertex indices each, in output order.</summary>
	public int[] Triangles { get; }

	/// <summary>
	/// The output mesh of an extractor that has run Extract: each vertex moved to the model by
	/// <paramref name="unitCubeToModel"/> (Solve's unitCubeToModel, a 4x4 homogeneous transform)
	/// with its density value and color bytes when present. Port of extractLevelSet's
	/// TransformedOutputLevelSetVertexStream and WriteMesh's vertex conversion.
	/// </summary>
	public static PoissonMeshOutput FromLevelSet(PoissonLevelSetExtractor extractor, PoissonXForm unitCubeToModel)
	{
		ArgumentNullException.ThrowIfNull(extractor);
		ArgumentNullException.ThrowIfNull(unitCubeToModel);
		if (unitCubeToModel.Dim != 4)
		{
			throw new ArgumentException($"unitCubeToModel must be 4x4 (homogeneous 3D); it is {unitCubeToModel.Dim}x{unitCubeToModel.Dim}.", nameof(unitCubeToModel));
		}

		List<LevelSetVertex> vertices = extractor.Vertices;
		int n = vertices.Count;
		int channels = extractor.DataChannels;
		float[] positions = new float[3 * n];
		float[]? values = extractor.HasDensity ? new float[n] : null;
		byte[]? colors = channels > 0 ? new byte[channels * n] : null;
		Span<float> p = stackalloc float[3];
		for (int i = 0; i < n; i++)
		{
			LevelSetVertex v = vertices[i];
			p[0] = v.X;
			p[1] = v.Y;
			p[2] = v.Z;
			unitCubeToModel.TransformPoint(p, positions.AsSpan(3 * i, 3));
			if (values != null)
			{
				values[i] = v.Depth;
			}

			if (colors != null)
			{
				for (int c = 0; c < channels; c++)
				{
					colors[(channels * i) + c] = PlyUChar(v.Data[c]);
				}
			}
		}

		int[] triangles = new int[3 * extractor.Polygons.Count];
		for (int t = 0; t < extractor.Polygons.Count; t++)
		{
			int[] polygon = extractor.Polygons[t];
			if (polygon.Length != 3)
			{
				throw new InvalidOperationException($"Polygon {t} has {polygon.Length} vertices; the extraction writes triangles only.");
			}

			polygon.CopyTo(triangles, 3 * t);
		}

		return new PoissonMeshOutput(n, channels, positions, values, colors, triangles);
	}

	/// <summary>
	/// A float channel written to a uchar PLY property: widened to double, converted to unsigned
	/// int, and its low byte kept. Port of PlyFile.inl's get_stored_item (PLY_FLOAT) followed by
	/// write_binary_item (PLY_UCHAR). So it truncates rather than rounds, and 256 and above wrap
	/// (256.5 gives 0). The double-to-unsigned conversion is undefined upstream outside
	/// [0, 2^32); .NET saturates it (negatives and NaN to 0, 2^32 and above to uint.MaxValue),
	/// which is what arm64's conversion, and so the macOS reference build, does.
	/// </summary>
	public static byte PlyUChar(float value) => unchecked((byte)(uint)(double)value);
}
