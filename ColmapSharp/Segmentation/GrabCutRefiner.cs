// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GrabCutRefiner: refines a trimap into a foreground mask by iterated graph cuts over colour
// models. Not a COLMAP port. Written from C. Rother, V. Kolmogorov and A. Blake, "GrabCut -
// Interactive Foreground Extraction using Iterated Graph Cuts", SIGGRAPH 2004: the energy
// E = sum_n D_n + V, with D_n = -log p(z_n | GMM of alpha_n) (ColorGmm) and the contrast
// sensitive Potts term V = gamma * exp(-beta |z_m - z_n|^2) / dist(m, n) over 8-neighbours,
// beta = 1 / (2 <|z_m - z_n|^2>). Colours are Lab, not the paper's RGB; beta normalises the
// contrast either way. Each iteration assigns components, refits both models and cuts, as the
// paper's iterative minimisation does. The cut is the existing MinSTGraphCut (Boykov-Kolmogorov,
// written from its paper). Only the unknown band of the trimap becomes graph nodes: an edge to
// a sure pixel is a fixed cost, so it folds into the node's terminal capacity.

using ColmapSharp.Mathematics;

namespace ColmapSharp.Segmentation;

internal static class GrabCutRefiner
{
	/// <summary>Trimap label: certainly background.</summary>
	public const byte SureBackground = 0;

	/// <summary>Trimap label: certainly foreground.</summary>
	public const byte SureForeground = 1;

	/// <summary>Trimap label: to be decided by the cut.</summary>
	public const byte Unknown = 2;

	// The four forward neighbour offsets of the 8-neighbourhood (each pair is visited once).
	private static readonly (int Dx, int Dy)[] ForwardNeighbours = [(1, 0), (0, 1), (1, 1), (-1, 1)];

	/// <summary>
	/// Returns the foreground mask. <paramref name="alpha"/> is the initial labelling of the
	/// unknown pixels; sure pixels keep their trimap label. The background colour model learns
	/// only from background-labelled pixels that <paramref name="backgroundModelAllowed"/> admits
	/// and that connect to the image border through background: enclosed wall-coloured regions
	/// (a lit grey underside, white label text) are exactly what it must not learn as wall.
	/// </summary>
	public static bool[] Refine(LabImage image, byte[] trimap, bool[] alpha, bool[] backgroundModelAllowed, SegmentationOptions options, CancellationToken cancellationToken)
	{
		int width = image.Width, height = image.Height, count = width * height;
		var labels = new bool[count];
		var nodeOf = new int[count];
		int numNodes = 0;
		for (int p = 0; p < count; ++p)
		{
			labels[p] = trimap[p] == SureForeground || (trimap[p] == Unknown && alpha[p]);
			nodeOf[p] = trimap[p] == Unknown ? numNodes++ : -1;
		}

		if (numNodes == 0)
		{
			return labels;
		}

		double beta = ContrastBeta(image);
		for (int iteration = 0; iteration < options.GrabCutIterations; ++iteration)
		{
			cancellationToken.ThrowIfCancellationRequested();
			int[] fgIndices = Indices(labels, true);
			int[] bgIndices = BackgroundModelIndices(labels, backgroundModelAllowed, width, height);
			if (fgIndices.Length == 0 || bgIndices.Length == 0)
			{
				break;
			}

			ColorGmm fg = FitAndReassign(image, fgIndices, options);
			ColorGmm bg = FitAndReassign(image, bgIndices, options);
			Cut(image, trimap, nodeOf, numNodes, labels, fg, bg, beta, options.Smoothness);
		}

		return labels;
	}

	// GrabCut steps 1 and 2: seed the model with k-means on the current labelling, then assign
	// each pixel its most likely component and re-estimate from that assignment. Re-seeding
	// every iteration (the paper carries the models over) keeps a model valid when the cut
	// moved many pixels across; at a few iterations the cost is small.
	private static ColorGmm FitAndReassign(LabImage image, int[] indices, SegmentationOptions options)
	{
		ColorGmm gmm = ColorGmm.Fit(image, indices, options.GmmComponents, options.KMeansIterations, options.CovarianceRegularization);
		var assignment = new int[indices.Length];
		for (int i = 0; i < indices.Length; ++i)
		{
			int p = indices[i];
			assignment[i] = gmm.MostLikelyComponent(image.L[p], image.A[p], image.B[p]);
		}

		gmm.Estimate(image, indices, assignment);
		return gmm;
	}

	private static void Cut(LabImage image, byte[] trimap, int[] nodeOf, int numNodes, bool[] labels, ColorGmm fg, ColorGmm bg, double beta, double gamma)
	{
		int width = image.Width, height = image.Height;
		var sourceCapacity = new double[numNodes];
		var sinkCapacity = new double[numNodes];
		var graph = new MinSTGraphCut<double>(numNodes);

		// Source side = foreground. Cutting a node's source link (labelling it background)
		// costs its background data term, and the sink link its foreground term.
		for (int p = 0; p < nodeOf.Length; ++p)
		{
			int node = nodeOf[p];
			if (node < 0)
			{
				continue;
			}

			double dFg = fg.NegativeLogLikelihood(image.L[p], image.A[p], image.B[p]);
			double dBg = bg.NegativeLogLikelihood(image.L[p], image.A[p], image.B[p]);
			// Only the difference matters; shift so both capacities are non-negative.
			double shift = Math.Min(dFg, dBg);
			sourceCapacity[node] = dBg - shift;
			sinkCapacity[node] = dFg - shift;
		}

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int p = y * width + x;
				foreach ((int dx, int dy) in ForwardNeighbours)
				{
					int nx = x + dx, ny = y + dy;
					if (nx < 0 || nx >= width || ny >= height)
					{
						continue;
					}

					int q = ny * width + nx;
					int np = nodeOf[p], nq = nodeOf[q];
					if (np < 0 && nq < 0)
					{
						continue;
					}

					double weight = PairWeight(image, p, q, beta, gamma, dx != 0 && dy != 0);
					if (np >= 0 && nq >= 0)
					{
						graph.AddEdge(np, nq, weight, weight);
					}
					else
					{
						// One end is sure: the unknown end pays when it takes the other label.
						int node = np >= 0 ? np : nq;
						int sure = np >= 0 ? q : p;
						if (trimap[sure] == SureForeground)
						{
							sourceCapacity[node] += weight;
						}
						else
						{
							sinkCapacity[node] += weight;
						}
					}
				}
			}
		}

		for (int node = 0; node < numNodes; ++node)
		{
			graph.AddNode(node, sourceCapacity[node], sinkCapacity[node]);
		}

		graph.Compute();
		for (int p = 0; p < nodeOf.Length; ++p)
		{
			if (nodeOf[p] >= 0)
			{
				labels[p] = graph.IsConnectedToSource(nodeOf[p]);
			}
		}
	}

	private static double PairWeight(LabImage image, int p, int q, double beta, double gamma, bool diagonal)
	{
		double dl = image.L[p] - image.L[q];
		double da = image.A[p] - image.A[q];
		double db = image.B[p] - image.B[q];
		double weight = gamma * Math.Exp(-beta * (dl * dl + da * da + db * db));
		return diagonal ? weight / Math.Sqrt(2.0) : weight;
	}

	// beta = 1 / (2 <|z_m - z_n|^2>) over all 8-neighbour pairs (0 for a flat image).
	private static double ContrastBeta(LabImage image)
	{
		int width = image.Width, height = image.Height;
		double sum = 0.0;
		long pairs = 0;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int p = y * width + x;
				foreach ((int dx, int dy) in ForwardNeighbours)
				{
					int nx = x + dx, ny = y + dy;
					if (nx < 0 || nx >= width || ny >= height)
					{
						continue;
					}

					int q = ny * width + nx;
					double dl = image.L[p] - image.L[q];
					double da = image.A[p] - image.A[q];
					double db = image.B[p] - image.B[q];
					sum += dl * dl + da * da + db * db;
					++pairs;
				}
			}
		}

		return pairs == 0 || sum <= 0.0 ? 0.0 : 1.0 / (2.0 * sum / pairs);
	}

	// Background pixels outside the hull that reach the border; all background pixels when
	// none do (the object fills the frame).
	internal static int[] BackgroundModelIndices(bool[] labels, bool[] allowed, int width, int height)
	{
		bool[] enclosedOrForeground = BinaryMorphology.FillHoles(labels, width, height);
		var result = new List<int>();
		for (int p = 0; p < labels.Length; ++p)
		{
			if (!enclosedOrForeground[p] && allowed[p])
			{
				result.Add(p);
			}
		}

		return result.Count > 0 ? [.. result] : Indices(labels, false);
	}

	private static int[] Indices(bool[] labels, bool value)
	{
		var result = new List<int>();
		for (int p = 0; p < labels.Length; ++p)
		{
			if (labels[p] == value)
			{
				result.Add(p);
			}
		}

		return [.. result];
	}
}
