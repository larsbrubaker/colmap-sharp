// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SurfaceMetrics: how close a reconstructed surface is to the true object (docs/QUALITY_PLAN.md,
// stage 0b). Not a COLMAP port. The measures are those of the Tanks and Temples benchmark
// (A. Knapitsch, J. Park, Q.-Y. Zhou and V. Koltun, "Tanks and Temples: Benchmarking
// Large-Scale Scene Reconstruction", ACM TOG 36(4), 2017):
// - accuracy: distances from the reconstruction to the truth (the Chamfer half "recon -> truth");
// - completeness: distances from the truth to the reconstruction ("truth -> recon");
// - precision / recall at a threshold tau: the fraction of each within tau, and their harmonic
//   mean, the F-score.
// All distances are reported as a percentage of the truth's bounding-box diagonal so scenes of
// any scale compare.
//
// The reconstruction is a mesh (sampled uniformly by area, BenchmarkMesh.SampleSurface) or, when
// a run gave no mesh, the fused points themselves; it must already be in the truth's frame
// (BenchmarkEvaluator maps it with the pose Sim3). Truth -> recon distances use
// TriangleBvh.ClosestPoint on a mesh and PointKdTree on a cloud.
//
// The wall: only the object is truth, but the scenes put it in front of a wall, and a
// reconstruction may keep bits of it. A reconstructed sample counts toward accuracy and
// precision only when it lies inside the truth's bounding box grown by OutlierMargin (a
// fraction of the diagonal) on every side. Anything outside that box is at least the margin
// away from the object, so it is background, not a badly placed object surface; it is excluded
// and reported separately as ExcludedFraction, so a reconstruction that is mostly wall still
// shows. Floating junk close to the object (inside the grown box) counts fully against
// accuracy. Completeness is unaffected: it looks only from the truth.
//
// The margin is 25% of the diagonal, not smaller, so a wrongly scaled object still counts: an
// elongated object (the mouse) reconstructed 1.3 times too large grows by 15% of its length at
// each end of its long axis, which a 10% margin would have excluded as background
// (BenchmarkMetricsTests.CSharpOnly_OversizedReconstructionLosesPrecision).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>Options of <see cref="SurfaceMetrics.Compute"/>.</summary>
public sealed class SurfaceMetricOptions
{
	/// <summary>The F-score threshold tau as a fraction of the truth's diagonal (Tanks and Temples style).</summary>
	public double TauFraction { get; set; } = 0.01;

	/// <summary>Points sampled on each mesh.</summary>
	public int SampleCount { get; set; } = 20000;

	/// <summary>
	/// The truth's bounding box is grown by this fraction of its diagonal on every side; recon
	/// samples outside it are background (the wall) and excluded from accuracy.
	/// </summary>
	public double OutlierMarginFraction { get; set; } = 0.25;

	/// <summary>Seed of the surface samplers.</summary>
	public uint Seed { get; set; } = 1;
}

/// <summary>Accuracy, completeness and F-score of one reconstructed surface.</summary>
public sealed record SurfaceScores
{
	/// <summary>Mean recon -> truth distance, % of the diagonal (NaN when no sample counted).</summary>
	public double AccuracyPct { get; init; }

	/// <summary>Mean truth -> recon distance, % of the diagonal (NaN when there is no reconstruction).</summary>
	public double CompletenessPct { get; init; }

	/// <summary>Fraction of counted recon samples within tau of the truth.</summary>
	public double Precision { get; init; }

	/// <summary>Fraction of truth samples within tau of the reconstruction.</summary>
	public double Recall { get; init; }

	/// <summary>Harmonic mean of precision and recall (0 when both are 0).</summary>
	public double FScore { get; init; }

	/// <summary>The threshold tau used, % of the diagonal.</summary>
	public double TauPct { get; init; }

	/// <summary>Fraction of recon samples outside the grown box (background, not counted).</summary>
	public double ExcludedFraction { get; init; }

	/// <summary>Number of recon samples (before exclusion).</summary>
	public int ReconSamples { get; init; }
}

/// <summary>Tanks-and-Temples-style surface scores against a truth mesh.</summary>
public static class SurfaceMetrics
{
	/// <summary>
	/// Scores <paramref name="reconMesh"/> (or, when it is null, <paramref name="reconPoints"/>)
	/// against <paramref name="truth"/>. Both must be in the truth's frame. With neither, or an
	/// empty one, precision, recall and F are 0 and the distances NaN.
	/// </summary>
	public static SurfaceScores Compute(
		BenchmarkMesh truth,
		BenchmarkMesh? reconMesh,
		IReadOnlyList<Vector3d>? reconPoints,
		SurfaceMetricOptions options)
	{
		Check.NotNull(truth);
		Check.NotNull(options);
		double diagonal = truth.Diagonal();
		Check.That(diagonal > 0, "The truth mesh has no extent");
		double tau = options.TauFraction * diagonal;

		Vector3d[] reconSamples = reconMesh is not null
			? reconMesh.SampleSurface(options.SampleCount, options.Seed + 1)
			: reconPoints is not null ? [.. reconPoints] : [];
		Vector3d[] truthSamples = truth.SampleSurface(options.SampleCount, options.Seed);

		// Accuracy: recon -> truth, over the samples inside the grown box.
		TriangleBvh truthBvh = truth.BuildBvh();
		(Vector3d min, Vector3d max) = truth.Bounds();
		double margin = options.OutlierMarginFraction * diagonal;
		var reconDistances = new double[reconSamples.Length];
		var counted = new bool[reconSamples.Length];
		Parallel.For(0, reconSamples.Length, i =>
		{
			Vector3d p = reconSamples[i];
			counted[i] = Inside(p, min, max, margin);
			reconDistances[i] = counted[i] ? truthBvh.ClosestPoint(p, out _, out _) : 0;
		});

		double accuracySum = 0;
		int numCounted = 0;
		int numPrecise = 0;
		for (int i = 0; i < reconSamples.Length; i++)
		{
			if (!counted[i])
			{
				continue;
			}

			numCounted++;
			accuracySum += reconDistances[i];
			if (reconDistances[i] <= tau)
			{
				numPrecise++;
			}
		}

		// Completeness: truth -> recon.
		var truthDistances = new double[truthSamples.Length];
		if (reconSamples.Length > 0)
		{
			if (reconMesh is not null)
			{
				TriangleBvh reconBvh = reconMesh.BuildBvh();
				Parallel.For(0, truthSamples.Length, i => truthDistances[i] = reconBvh.ClosestPoint(truthSamples[i], out _, out _));
			}
			else
			{
				var tree = new PointKdTree(reconSamples);
				Parallel.For(0, truthSamples.Length, i => truthDistances[i] = tree.NearestDistance(truthSamples[i]));
			}
		}

		double completenessSum = 0;
		int numRecalled = 0;
		for (int i = 0; i < truthSamples.Length; i++)
		{
			completenessSum += truthDistances[i];
			if (reconSamples.Length > 0 && truthDistances[i] <= tau)
			{
				numRecalled++;
			}
		}

		double precision = numCounted == 0 ? 0 : numPrecise / (double)numCounted;
		double recall = truthSamples.Length == 0 ? 0 : numRecalled / (double)truthSamples.Length;
		return new SurfaceScores
		{
			AccuracyPct = numCounted == 0 ? double.NaN : 100 * accuracySum / numCounted / diagonal,
			CompletenessPct = reconSamples.Length == 0 ? double.NaN : 100 * completenessSum / truthSamples.Length / diagonal,
			Precision = precision,
			Recall = recall,
			FScore = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0,
			TauPct = 100 * options.TauFraction,
			ExcludedFraction = reconSamples.Length == 0 ? 0 : (reconSamples.Length - numCounted) / (double)reconSamples.Length,
			ReconSamples = reconSamples.Length,
		};
	}

	private static bool Inside(Vector3d p, Vector3d min, Vector3d max, double margin) =>
		p.X >= min.X - margin && p.X <= max.X + margin
		&& p.Y >= min.Y - margin && p.Y <= max.Y + margin
		&& p.Z >= min.Z - margin && p.Z <= max.Z + margin;
}
