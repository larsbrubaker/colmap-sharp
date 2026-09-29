// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkSummary: how the reconstruction benchmark (docs/QUALITY_PLAN.md, stage 0b) folds the
// runs of one case (one scene under several mapper seeds) into per-metric numbers, and how it
// decides that a new report regressed against a baseline. Not a COLMAP port. The
// ColmapSharp.Benchmarks runner writes the summaries into its JSON report and calls Regressions
// for --compare; they live here so the test suite can pin them.
//
// Summary, per metric: the mean over the runs where it is finite, the worst run in the
// metric's own direction (lowest F-score, highest error) or NaN when any run could not compute
// it (a failed run is the worst case), and the number of such failed runs. The mean alone would
// hide failures: a seed that stops aligning drops out of the mean and can make it look better.
//
// Regressions, per case present in both reports (matched by case name, scene and scene seed)
// and per metric of the baseline:
// - more failed runs than the baseline;
// - a worst that was a number and is now null (NaN);
// - a mean worse than the baseline's by more than max(tolerance, tolerance x |baseline|), or a
//   mean that became null.
// Timings are not compared (they depend on the machine). Reports whose "config" blocks differ
// (frames, size, quality, tau, masks, scene kinds, motion, seeds) are not comparable at all, and
// Regressions refuses them.

using System.Globalization;
using System.Text.Json.Nodes;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>One metric of one case over its runs.</summary>
public readonly record struct MetricSummary(string Name, double Mean, double Worst, int Failed, bool HigherIsBetter);

/// <summary>Per-case summaries and the regression check of the benchmark.</summary>
public static class BenchmarkSummary
{
	/// <summary>Mean, worst and failed count of each metric over <paramref name="runs"/>, in the metrics' fixed order.</summary>
	public static List<MetricSummary> Summarize(IReadOnlyList<BenchmarkMetrics> runs)
	{
		var summary = new List<MetricSummary>();
		if (runs.Count == 0)
		{
			return summary;
		}

		IReadOnlyList<BenchmarkValue> first = runs[0].Values();
		for (int m = 0; m < first.Count; m++)
		{
			double[] values = [.. runs.Select(r => r.Values()[m].Value)];
			double[] finite = [.. values.Where(double.IsFinite)];
			double mean = finite.Length == 0 ? double.NaN : finite.Sum() / finite.Length;
			double worst = finite.Length < values.Length ? double.NaN
				: first[m].HigherIsBetter ? finite.Min() : finite.Max();
			summary.Add(new MetricSummary(first[m].Name, mean, worst, values.Length - finite.Length, first[m].HigherIsBetter));
		}

		return summary;
	}

	/// <summary>
	/// The regressions of <paramref name="currentJson"/> against <paramref name="baselineJson"/>
	/// (see the file header), one line each. Throws <see cref="InvalidOperationException"/> when
	/// the two reports' config blocks differ.
	/// </summary>
	public static List<string> Regressions(string baselineJson, string currentJson, double tolerance)
	{
		JsonNode baseline = JsonNode.Parse(baselineJson)!;
		JsonNode current = JsonNode.Parse(currentJson)!;
		if (!JsonNode.DeepEquals(baseline["config"], current["config"]))
		{
			throw new InvalidOperationException(
				"The reports were made with different settings and cannot be compared. Baseline config: "
				+ baseline["config"]?.ToJsonString() + "; current config: " + current["config"]?.ToJsonString());
		}

		// Which direction is better, by metric name (the names and directions do not depend on
		// the values).
		Dictionary<string, bool> higherIsBetter = new BenchmarkMetrics().Values().ToDictionary(v => v.Name, v => v.HigherIsBetter);
		Dictionary<string, JsonObject> currentCases = Cases(current);
		var regressions = new List<string>();
		foreach ((string key, JsonObject baseSummary) in Cases(baseline))
		{
			if (!currentCases.TryGetValue(key, out JsonObject? summary))
			{
				continue;
			}

			foreach ((string metric, JsonNode? node) in baseSummary)
			{
				if (!higherIsBetter.TryGetValue(metric, out bool higher))
				{
					continue;
				}

				JsonNode? now = summary[metric];
				int baseFailed = Int(node?["failed"]);
				int failed = Int(now?["failed"]);
				if (failed > baseFailed)
				{
					regressions.Add($"{key} {metric}: failed runs {baseFailed} -> {failed}");
				}

				if (Number(node?["worst"]) is not null && Number(now?["worst"]) is null)
				{
					regressions.Add($"{key} {metric}: worst {Format(Number(node?["worst"]))} -> null");
				}

				if (Number(node?["mean"]) is not double b)
				{
					continue;
				}

				double? mean = Number(now?["mean"]);
				double allowed = Math.Max(tolerance, tolerance * Math.Abs(b));
				if (mean is not double m || (higher ? m < b - allowed : m > b + allowed))
				{
					regressions.Add($"{key} {metric}: mean {Format(b)} -> {Format(mean)}");
				}
			}
		}

		return regressions;
	}

	/// <summary>The key that matches a case between two reports.</summary>
	public static string CaseKey(string caseName, string scene, uint sceneSeed) =>
		string.Create(CultureInfo.InvariantCulture, $"{caseName} {scene} #{sceneSeed}");

	private static Dictionary<string, JsonObject> Cases(JsonNode report)
	{
		var cases = new Dictionary<string, JsonObject>();
		foreach (JsonNode? c in report["cases"]!.AsArray())
		{
			string key = CaseKey(c!["case"]!.GetValue<string>(), c["scene"]!.GetValue<string>(), c["scene_seed"]!.GetValue<uint>());
			cases[key] = c["summary"]!.AsObject();
		}

		return cases;
	}

	private static double? Number(JsonNode? node) => node?.GetValue<double>();

	private static int Int(JsonNode? node) => node?.GetValue<int>() ?? 0;

	private static string Format(double? value) => value?.ToString("G6", CultureInfo.InvariantCulture) ?? "null";
}
