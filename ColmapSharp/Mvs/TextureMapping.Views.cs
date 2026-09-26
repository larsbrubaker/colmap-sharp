// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TextureMapping.Views: view selection for texture mapping - COLMAP's OcclusionTester and
// SelectViews (colmap/mvs/texture_mapping.cc). Scores each (face, image) pair by the face's
// projected area, keeps the best image per face, and smooths the labels over face
// neighbors. The occlusion test runs on TriangleBvh.cs instead of CGAL's AABB tree
// (docs/CPP_DIVERGENCES.md, entry 90); the smoothing breaks label ties by image index
// (entry 91). Neighbors: TextureMapping.cs (entry point, adjacency).
//
// Memory: COLMAP keeps a dense double score per (face, image), 800 MB for a million faces
// and a hundred images. Only two things are read from it afterwards - the best image per
// face and whether a score is positive - so this port computes the best image inside the
// scoring loop (same strict ">" in image order) and keeps one bit per (face, image) for
// "score > 0". The selected views are identical.
//
// Time: COLMAP runs up to three occlusion queries per (face, image). The segment for a corner
// depends only on the vertex and the camera, so each vertex is queried once per image and
// the result (clear, one blocker, or several) is shared by all faces around it; that answers
// COLMAP's per-face question exactly (OcclusionTester.OccludesFace).

using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

public static partial class TextureMapping
{
	// Faces scored per parallel work item.
	private const int FacesPerChunk = 1024;

	/// <summary>Port of the OcclusionTester struct, over <see cref="TriangleBvh"/>.</summary>
	internal sealed class OcclusionTester
	{
		private readonly TriangleBvh tree;

		/// <summary>Port of OcclusionTester::Build.</summary>
		public OcclusionTester(MeshGeometry geometry)
		{
			float[] v = geometry.Vertices;
			int[] f = geometry.Faces;
			var corners = new List<float>(9 * geometry.NumFaces);
			var faceIds = new List<int>(geometry.NumFaces);
			for (int i = 0; i < geometry.NumFaces; i++)
			{
				int a = 3 * f[3 * i], b = 3 * f[3 * i + 1], c = 3 * f[3 * i + 2];
				float e1x = v[b] - v[a], e1y = v[b + 1] - v[a + 1], e1z = v[b + 2] - v[a + 2];
				float e2x = v[c] - v[a], e2y = v[c + 1] - v[a + 1], e2z = v[c + 2] - v[a + 2];
				float cx = e1y * e2z - e1z * e2y;
				float cy = e1z * e2x - e1x * e2z;
				float cz = e1x * e2y - e1y * e2x;

				// Skip degenerate (zero-area) triangles, as COLMAP does (they trip CGAL's
				// assertions there).
				if (cx * cx + cy * cy + cz * cz < 1e-20f)
				{
					continue;
				}

				for (int k = 0; k < 3; k++)
				{
					corners.Add(v[a + k]);
				}

				for (int k = 0; k < 3; k++)
				{
					corners.Add(v[b + k]);
				}

				for (int k = 0; k < 3; k++)
				{
					corners.Add(v[c + k]);
				}

				faceIds.Add(i);
			}

			tree = new TriangleBvh(corners.ToArray(), faceIds.ToArray());
		}

		/// <summary>No triangle blocks the vertex (see <see cref="BlockerState"/>).</summary>
		public const int Clear = -1;

		/// <summary>Two or more triangles block the vertex.</summary>
		public const int ManyBlockers = -2;

		/// <summary>
		/// What lies between the camera center and a mesh vertex, ignoring hits within kEps of
		/// the vertex: <see cref="Clear"/>, <see cref="ManyBlockers"/>, or the one blocking
		/// face's index. COLMAP's OcclusionTester::IsOccluded(camera, vertex, face) asks the
		/// same segment query once per face corner; its answer for any face follows from this
		/// state (<see cref="OccludesFace"/>), so a vertex is queried once per image and the
		/// answer is shared by all the faces around it.
		/// </summary>
		public int BlockerState(ReadOnlySpan<float> cameraCenter, float vx, float vy, float vz)
		{
			const float kEps = 1e-4f;
			float dx = vx - cameraCenter[0], dy = vy - cameraCenter[1], dz = vz - cameraCenter[2];
			float dist = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
			if (dist < kEps)
			{
				return Clear;
			}

			float s = kEps / dist;
			float tx = vx - s * dx, ty = vy - s * dy, tz = vz - s * dz;
			int hits = tree.CountHitsUpToTwo(cameraCenter[0], cameraCenter[1], cameraCenter[2], tx, ty, tz, dist - kEps, out int onlyFace);
			return hits == 0 ? Clear : hits == 1 ? onlyFace : ManyBlockers;
		}

		/// <summary>
		/// Port of OcclusionTester::IsOccluded's answer from a <see cref="BlockerState"/>: some
		/// triangle other than <paramref name="faceIdx"/> itself blocks the corner.
		/// </summary>
		public static bool OccludesFace(int state, int faceIdx) =>
			state == ManyBlockers || (state >= 0 && state != faceIdx);
	}

	// Per-image data read in the scoring loop, computed once instead of per face.
	private readonly struct ViewData
	{
		public ViewData(Image image)
		{
			P = image.GetP().ToArray();
			Center = new float[3];
			MvsGeometry.ComputeProjectionCenter(image.GetR(), image.GetT(), Center);
			Width = image.GetWidth();
			Height = image.GetHeight();
		}

		public float[] P { get; }

		public float[] Center { get; }

		public int Width { get; }

		public int Height { get; }
	}

	/// <summary>
	/// Port of SelectViews: the image index per face (-1 when no image sees it), after
	/// <see cref="MeshTextureMappingOptions.ViewSelectionSmoothingIterations"/> rounds of
	/// neighbor smoothing.
	/// </summary>
	internal static int[] SelectViews(
		MeshGeometry geometry,
		float[] faceNormals,
		IReadOnlyList<Image> images,
		int[] adjacency,
		MeshTextureMappingOptions options,
		IProgress<double>? progress,
		CancellationToken cancellationToken)
	{
		int numFaces = geometry.NumFaces;
		int numImages = images.Count;
		var viewPerFace = new int[numFaces];
		Array.Fill(viewPerFace, -1);
		if (numFaces == 0 || numImages == 0)
		{
			return viewPerFace;
		}

		var tester = new OcclusionTester(geometry);
		var views = new ViewData[numImages];
		for (int ii = 0; ii < numImages; ii++)
		{
			views[ii] = new ViewData(images[ii]);
		}

		// positive[fi * words + ii / 64] bit ii % 64: the (face, image) score is > 0.
		int words = (numImages + 63) / 64;
		var positive = new ulong[(long)numFaces * words];

		int numChunks = (numFaces + FacesPerChunk - 1) / FacesPerChunk;
		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(options.NumThreads),
			CancellationToken = cancellationToken,
		};

		// Images are scored one after another (faces in parallel), so each face still sees
		// the images in index order and keeps the first strictly larger score. The occlusion
		// state of each vertex is cached for the current image: entry = (image + 1) << 32 |
		// (uint)state. Threads that race on a vertex compute the same state, and a 64-bit
		// write is atomic, so the cache cannot change any answer.
		var bestScore = new double[numFaces];
		Array.Fill(bestScore, -1.0);
		var vertexStates = new long[geometry.Vertices.Length / 3];
		for (int ii = 0; ii < numImages; ii++)
		{
			int image = ii;
			Parallel.For(0, numChunks, parallelOptions, chunk =>
			{
				int end = Math.Min(numFaces, (chunk + 1) * FacesPerChunk);
				for (int fi = chunk * FacesPerChunk; fi < end; fi++)
				{
					double area = ScoreFaceView(geometry, faceNormals, views[image], image, tester, vertexStates, options, fi);
					if (area > bestScore[fi])
					{
						bestScore[fi] = area;
						viewPerFace[fi] = image;
					}

					if (area > 0)
					{
						positive[(long)fi * words + (image >> 6)] |= 1UL << (image & 63);
					}
				}
			});

			// Reported from this thread after each image, so the fractions only grow.
			progress?.Report(ProgressPrepared + (ProgressViewsSelected - ProgressPrepared) * (ii + 1) / numImages);
		}

		for (int iter = 0; iter < options.ViewSelectionSmoothingIterations; iter++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			int[] current = viewPerFace;
			var newViews = new int[numFaces];
			Parallel.For(0, numChunks, parallelOptions, chunk =>
			{
				int end = Math.Min(numFaces, (chunk + 1) * FacesPerChunk);
				for (int fi = chunk * FacesPerChunk; fi < end; fi++)
				{
					newViews[fi] = SmoothedLabel(current, adjacency, positive, words, fi);
				}
			});
			viewPerFace = newViews;
		}

		return viewPerFace;
	}

	// The body of SelectViews' scoring loop for one (face, image) pair: the projected area,
	// or -1 when the image does not see the face (COLMAP's score table entry).
	private static double ScoreFaceView(
		MeshGeometry geometry,
		float[] faceNormals,
		ViewData view,
		int image,
		OcclusionTester tester,
		long[] vertexStates,
		MeshTextureMappingOptions options,
		int fi)
	{
		float nx = faceNormals[3 * fi], ny = faceNormals[3 * fi + 1], nz = faceNormals[3 * fi + 2];
		if (nx * nx + ny * ny + nz * nz < 1e-10f)
		{
			return -1.0;
		}

		float[] v = geometry.Vertices;
		int[] f = geometry.Faces;
		Span<float> verts = stackalloc float[9];
		for (int k = 0; k < 3; k++)
		{
			int vi = 3 * f[3 * fi + k];
			verts[3 * k] = v[vi];
			verts[3 * k + 1] = v[vi + 1];
			verts[3 * k + 2] = v[vi + 2];
		}

		float centroidX = (verts[0] + verts[3] + verts[6]) / 3.0f;
		float centroidY = (verts[1] + verts[4] + verts[7]) / 3.0f;
		float centroidZ = (verts[2] + verts[5] + verts[8]) / 3.0f;
		float[] center = view.Center;

		// Eigen's normalized(): divide by the norm, a zero vector stays zero.
		float dx = center[0] - centroidX, dy = center[1] - centroidY, dz = center[2] - centroidZ;
		float n2 = dx * dx + dy * dy + dz * dz;
		if (n2 > 0)
		{
			float norm = MathF.Sqrt(n2);
			dx /= norm;
			dy /= norm;
			dz /= norm;
		}

		float cosAngle = nx * dx + ny * dy + nz * dz;
		if (cosAngle < (float)options.MinCosNormalAngle)
		{
			return -1.0;
		}

		Span<float> proj = stackalloc float[6];
		int visibleCount = 0;
		for (int k = 0; k < 3; k++)
		{
			float x = verts[3 * k], y = verts[3 * k + 1], z = verts[3 * k + 2];
			if (ProjectPointDepth(view.P, x, y, z) <= 0)
			{
				// Behind the camera.
				return -1.0;
			}

			ProjectPoint(view.P, x, y, z, out proj[2 * k], out proj[2 * k + 1]);
			if (proj[2 * k] >= 0 && proj[2 * k] < view.Width && proj[2 * k + 1] >= 0 && proj[2 * k + 1] < view.Height)
			{
				++visibleCount;
			}
		}

		if (visibleCount < options.MinVisibleVertices)
		{
			return -1.0;
		}

		long stamp = (long)(image + 1) << 32;
		for (int k = 0; k < 3; k++)
		{
			int vertex = f[3 * fi + k];
			long entry = Volatile.Read(ref vertexStates[vertex]);
			int state;
			if ((entry & ~0xFFFFFFFFL) == stamp)
			{
				state = (int)entry;
			}
			else
			{
				state = tester.BlockerState(center, verts[3 * k], verts[3 * k + 1], verts[3 * k + 2]);
				Volatile.Write(ref vertexStates[vertex], stamp | (uint)state);
			}

			if (OcclusionTester.OccludesFace(state, fi))
			{
				return -1.0;
			}
		}

		float e1x = proj[2] - proj[0], e1y = proj[3] - proj[1];
		float e2x = proj[4] - proj[0], e2y = proj[5] - proj[1];
		return Math.Abs((double)e1x * e2y - (double)e1y * e2x);
	}

	// One face of one smoothing iteration: the label most common among the neighbors,
	// provided the face scores positively in it and it beats the current label's count.
	// COLMAP iterates a hash map of label counts, so which of two equally common labels wins
	// depends on hash order; here labels are visited in ascending image index
	// (docs/CPP_DIVERGENCES.md, entry 91).
	private static int SmoothedLabel(int[] viewPerFace, int[] adjacency, ulong[] positive, int words, int fi)
	{
		int current = viewPerFace[fi];
		if (current < 0)
		{
			return current;
		}

		// Up to three neighbor labels with their counts.
		Span<int> labels = stackalloc int[3];
		Span<int> counts = stackalloc int[3];
		int distinct = 0;
		for (int s = 3 * fi; s < 3 * fi + 3; s++)
		{
			int ni = adjacency[s];
			if (ni < 0 || viewPerFace[ni] < 0)
			{
				continue;
			}

			int label = viewPerFace[ni];
			int j = 0;
			while (j < distinct && labels[j] != label)
			{
				j++;
			}

			if (j == distinct)
			{
				labels[distinct] = label;
				counts[distinct++] = 0;
			}

			counts[j]++;
		}

		int bestLabel = current;
		int bestCount = 0;
		for (int j = 0; j < distinct; j++)
		{
			if (labels[j] == current)
			{
				bestCount = counts[j];
			}
		}

		// Visit labels in ascending order.
		int previous = -1;
		for (int visited = 0; visited < distinct; visited++)
		{
			int pick = -1;
			for (int j = 0; j < distinct; j++)
			{
				if (labels[j] > previous && (pick < 0 || labels[j] < labels[pick]))
				{
					pick = j;
				}
			}

			previous = labels[pick];
			int label = labels[pick];
			bool labelPositive = (positive[(long)fi * words + (label >> 6)] & (1UL << (label & 63))) != 0;
			if (counts[pick] > bestCount && labelPositive)
			{
				bestCount = counts[pick];
				bestLabel = label;
			}
		}

		return bestLabel;
	}
}
