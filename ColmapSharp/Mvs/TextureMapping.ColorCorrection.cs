// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TextureMapping.ColorCorrection: ApplyGlobalColorCorrection from
// colmap/mvs/texture_mapping.cc (the global seam leveling of Waechter et al. 2014). One
// additive color offset per (region, vertex) is solved so that colors agree across seams
// between regions textured from different views, then the offsets are interpolated over each
// face's baked atlas pixels. Runs between BakeTexture and InpaintAtlas
// (TextureMapping.Atlas.cs).
//
// The sparse system goes through LinearAlgebra/SimplicialCholesky.cs (LDLT, AMD ordering) in
// place of Eigen::SimplicialLDLT: Tier B, same math in a different operation order. The
// system matrix is the same for all three channels (only the right-hand side differs), so it
// is factorized once and solved three times; COLMAP rebuilds and refactorizes the identical
// matrix per channel, which gives the same factor.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

namespace ColmapSharp.Mvs;

public static partial class TextureMapping
{
	/// <summary>Port of ApplyGlobalColorCorrection.</summary>
	internal static void ApplyGlobalColorCorrection(
		Bitmap atlas,
		MeshGeometry geometry,
		List<FaceRegion> regions,
		RegionProjection[] projections,
		AtlasLayout layout,
		IReadOnlyList<Image> images,
		int[] adjacency,
		int[] viewPerFace,
		bool[] bakedMask,
		MeshTextureMappingOptions options,
		CancellationToken cancellationToken)
	{
		int numFaces = geometry.NumFaces;
		int[] f = geometry.Faces;
		float[] vtx = geometry.Vertices;

		// Seam edges: (face_l, face_r, vert_a, vert_b) for neighboring faces with different
		// views, face_l < face_r.
		var seamEdges = new List<(int FaceL, int FaceR, int VertA, int VertB)>();
		for (int fi = 0; fi < numFaces; fi++)
		{
			if (viewPerFace[fi] < 0)
			{
				continue;
			}

			for (int e = 0; e < 3; e++)
			{
				int va = f[3 * fi + e];
				int vb = f[3 * fi + (e + 1) % 3];
				ulong ekey = EdgeKey(va, vb);
				for (int s = 3 * fi; s < 3 * fi + 3; s++)
				{
					int ni = adjacency[s];
					if (ni < 0 || ni <= fi || viewPerFace[ni] < 0 || viewPerFace[ni] == viewPerFace[fi])
					{
						continue;
					}

					bool sharesEdge = false;
					for (int ne = 0; ne < 3; ne++)
					{
						if (EdgeKey(f[3 * ni + ne], f[3 * ni + (ne + 1) % 3]) == ekey)
						{
							sharesEdge = true;
							break;
						}
					}

					if (sharesEdge)
					{
						seamEdges.Add((fi, ni, va, vb));
					}
				}
			}
		}

		if (seamEdges.Count == 0)
		{
			return;
		}

		// Per-region vertex-to-variable mapping, variables numbered in first-seen order.
		var regionVertMaps = new Dictionary<int, int>[regions.Count];
		var faceToRegion = new int[numFaces];
		Array.Fill(faceToRegion, -1);
		int totalVars = 0;
		for (int ri = 0; ri < regions.Count; ri++)
		{
			var map = new Dictionary<int, int>();
			regionVertMaps[ri] = map;
			foreach (int fi in regions[ri].FaceIds)
			{
				faceToRegion[fi] = ri;
				for (int k = 0; k < 3; k++)
				{
					if (map.TryAdd(f[3 * fi + k], totalVars))
					{
						totalVars++;
					}
				}
			}
		}

		if (totalVars == 0)
		{
			return;
		}

		cancellationToken.ThrowIfCancellationRequested();

		// The seam terms: both variables and both views' colors at the shared vertex. COLMAP
		// recomputes these per channel; they do not depend on the channel.
		var terms = new List<(int VarL, int VarR, BitmapColor<float> ColorL, BitmapColor<float> ColorR)>();
		foreach ((int faceL, int faceR, int vertA, int vertB) in seamEdges)
		{
			int riL = faceToRegion[faceL];
			int riR = faceToRegion[faceR];
			if (riL < 0 || riR < 0)
			{
				continue;
			}

			foreach (int sv in (ReadOnlySpan<int>)[vertA, vertB])
			{
				if (!regionVertMaps[riL].TryGetValue(sv, out int varL) || !regionVertMaps[riR].TryGetValue(sv, out int varR))
				{
					continue;
				}

				Image imgL = images[regions[riL].ViewId];
				Image imgR = images[regions[riR].ViewId];
				ProjectPoint(imgL.GetP(), vtx[3 * sv], vtx[3 * sv + 1], vtx[3 * sv + 2], out float lx, out float ly);
				ProjectPoint(imgR.GetP(), vtx[3 * sv], vtx[3 * sv + 1], vtx[3 * sv + 2], out float rx, out float ry);
				BitmapColor<float>? colorL = imgL.GetBitmap().InterpolateBilinear(lx, ly);
				BitmapColor<float>? colorR = imgR.GetBitmap().InterpolateBilinear(rx, ry);
				if (colorL is null || colorR is null)
				{
					continue;
				}

				terms.Add((varL, varR, colorL.Value, colorR.Value));
			}
		}

		// The system matrix: +1/-1 blocks per seam term, plus beta on the diagonal.
		double beta = options.ColorCorrectionRegularization;
		var triplets = new List<SparseTriplet>(4 * terms.Count + totalVars);
		foreach ((int varL, int varR, _, _) in terms)
		{
			triplets.Add(new SparseTriplet(varL, varL, 1.0));
			triplets.Add(new SparseTriplet(varR, varR, 1.0));
			triplets.Add(new SparseTriplet(varL, varR, -1.0));
			triplets.Add(new SparseTriplet(varR, varL, -1.0));
		}

		for (int i = 0; i < totalVars; i++)
		{
			triplets.Add(new SparseTriplet(i, i, beta));
		}

		cancellationToken.ThrowIfCancellationRequested();
		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(totalVars, totalVars, triplets);

		// SimplicialCholesky takes no token, so cancellation is checked around the
		// factorization and before each solve.
		var solver = new SimplicialCholesky(SimplicialCholeskyKind.LDLT);
		solver.AnalyzePattern(a);
		cancellationToken.ThrowIfCancellationRequested();
		solver.Factorize(a);
		if (solver.Info != ComputationInfo.Success)
		{
			// COLMAP logs "Color correction: failed to factorize system" and leaves the
			// atlas uncorrected.
			return;
		}

		var offsets = new double[3 * totalVars];
		for (int ch = 0; ch < 3; ch++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			var rhs = new VectorXd(totalVars);
			Span<double> r = rhs.AsSpan();
			foreach ((int varL, int varR, BitmapColor<float> colorL, BitmapColor<float> colorR) in terms)
			{
				double fL = Channel(colorL, ch);
				double fR = Channel(colorR, ch);
				r[varL] += fR - fL;
				r[varR] += fL - fR;
			}

			VectorXd x = solver.Solve(rhs);
			ReadOnlySpan<double> xs = x.AsSpan();
			for (int i = 0; i < totalVars; i++)
			{
				offsets[3 * i + ch] = xs[i];
			}
		}

		ApplyOffsets(atlas, geometry, regions, projections, layout, regionVertMaps, offsets, bakedMask, cancellationToken);
	}

	private static double Channel(BitmapColor<float> color, int ch) => ch == 0 ? color.R : ch == 1 ? color.G : color.B;

	// The last part of ApplyGlobalColorCorrection: interpolates the corner offsets over each
	// face's baked pixels and adds them, clamped to [0, 255] and truncated.
	private static void ApplyOffsets(
		Bitmap atlas,
		MeshGeometry geometry,
		List<FaceRegion> regions,
		RegionProjection[] projections,
		AtlasLayout layout,
		Dictionary<int, int>[] regionVertMaps,
		double[] offsets,
		bool[] bakedMask,
		CancellationToken cancellationToken)
	{
		int[] f = geometry.Faces;
		int aw = layout.AtlasWidth;
		int ah = layout.AtlasHeight;
		Span<float> atlasVerts = stackalloc float[6];
		Span<double> vertOffsets = stackalloc double[9];
		for (int ri = 0; ri < regions.Count; ri++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			FaceRegion region = regions[ri];
			RegionProjection rp = projections[ri];
			PackRect placement = layout.Placements[ri];
			for (int i = 0; i < region.FaceIds.Count; i++)
			{
				int fi = region.FaceIds[i];
				for (int k = 0; k < 3; k++)
				{
					int varId = regionVertMaps[ri][f[3 * fi + k]];
					for (int c = 0; c < 3; c++)
					{
						vertOffsets[3 * k + c] = offsets[3 * varId + c];
					}
				}

				ComputeAtlasVerts(rp, placement, i, atlasVerts);
				AtlasPixelBox(atlasVerts, 0, aw, ah, out int minPx, out int minPy, out int maxPx, out int maxPy);
				for (int py = minPy; py <= maxPy; py++)
				{
					for (int px = minPx; px <= maxPx; px++)
					{
						if (!bakedMask[(long)py * aw + px])
						{
							continue;
						}

						(float bu, float bv, float bw) = Barycentric(
							px + 0.5f, py + 0.5f, atlasVerts[0], atlasVerts[1], atlasVerts[2], atlasVerts[3], atlasVerts[4], atlasVerts[5]);
						if (bu < -0.01f || bv < -0.01f || bw < -0.01f)
						{
							continue;
						}

						BitmapColor<byte> color = atlas.GetPixel(px, py) ?? new BitmapColor<byte>(0);
						byte red = OffsetChannel(color.R, bu * vertOffsets[0] + bv * vertOffsets[3] + bw * vertOffsets[6]);
						byte green = OffsetChannel(color.G, bu * vertOffsets[1] + bv * vertOffsets[4] + bw * vertOffsets[7]);
						byte blue = OffsetChannel(color.B, bu * vertOffsets[2] + bv * vertOffsets[5] + bw * vertOffsets[8]);
						atlas.SetPixel(px, py, new BitmapColor<byte>(red, green, blue));
					}
				}
			}
		}
	}

	// static_cast<uint8_t>(std::max(0.0, std::min(255.0, value + offset))), keeping std::min /
	// std::max's argument order (a NaN offset becomes 255).
	private static byte OffsetChannel(byte value, double offset)
	{
		double x = value + offset;
		double upper = x < 255.0 ? x : 255.0;
		double clamped = 0.0 < upper ? upper : 0.0;
		return (byte)clamped;
	}
}
