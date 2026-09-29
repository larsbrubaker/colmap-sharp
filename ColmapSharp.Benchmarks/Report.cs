// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Report: the benchmark runner's output (Program.cs). Per case (case name, scene kind and scene
// seed) it writes every metric's summary over the mapper seeds - mean, worst and the number of
// failed runs (BenchmarkSummary.Summarize, which also defines them) - plus the raw runs. JSON
// writes non-finite numbers as null. The config block records every setting that changes the
// numbers, since --compare refuses reports whose configs differ. Also the plain-text table
// printed at the end of a run.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

using ColmapSharp.Mvs.Testing.Benchmark;

namespace ColmapSharp.Benchmarks;

internal static class Report
{
	// Columns of the text table: metric name and header.
	private static readonly (string Metric, string Header)[] TableColumns =
	[
		("registered_fraction", "reg"),
		("num_models", "models"),
		("focal_log_error", "|ln f/f0|"),
		("rotation_error_median_deg", "rot med°"),
		("position_error_median_pct", "pos med%"),
		("accuracy_pct", "acc%"),
		("completeness_pct", "comp%"),
		("f_score", "F"),
		("silhouette_iou_mean", "IoU"),
	];

	/// <summary>The case's metric summaries plus its total time (mean and slowest run).</summary>
	public static List<MetricSummary> Summarize(CaseResult c)
	{
		List<MetricSummary> summary = BenchmarkSummary.Summarize([.. c.Runs.Select(r => r.Result.Metrics)]);
		double[] totals = [.. c.Runs.Select(r => r.Result.TotalSeconds)];
		summary.Add(new MetricSummary("total_seconds", totals.Average(), totals.Max(), 0, false));
		return summary;
	}

	public static string ToJson(BenchmarkConfig config, IReadOnlyList<CaseResult> cases)
	{
		var root = new JsonObject
		{
			["config"] = new JsonObject
			{
				["kinds"] = Strings(config.Kinds.Select(k => k.ToString())),
				["fast_kinds"] = Strings(config.FastKinds.Select(k => k.ToString())),
				["known_kinds"] = Strings(config.KnownKinds.Select(k => k.ToString())),
				["frames"] = config.Frames,
				["motion_duration"] = config.MotionDuration,
				["width"] = config.Width,
				["height"] = config.Height,
				["quality"] = config.Quality.ToString(),
				["data"] = config.Data.ToString(),
				["scene_seeds"] = new JsonArray([.. config.SceneSeeds.Select(s => (JsonNode)s)]),
				["mapper_seeds"] = new JsonArray([.. config.MapperSeeds.Select(s => (JsonNode)s)]),
				["camera_model"] = "SIMPLE_PINHOLE",
				["tau_fraction"] = config.TauFraction,
				["true_masks"] = config.UseTrueMasks,
				["video_tracking"] = config.VideoTracking,
				["dense"] = config.Dense,
			},
		};

		var caseArray = new JsonArray();
		foreach (CaseResult c in cases)
		{
			var summary = new JsonObject();
			foreach (MetricSummary m in Summarize(c))
			{
				summary[m.Name] = new JsonObject { ["mean"] = Number(m.Mean), ["worst"] = Number(m.Worst), ["failed"] = m.Failed };
			}

			var runs = new JsonArray();
			foreach (RunResult run in c.Runs)
			{
				var metrics = new JsonObject { ["focal_ratio"] = Number(run.Result.Metrics.FocalRatio) };
				foreach (BenchmarkValue v in run.Result.Metrics.Values())
				{
					metrics[v.Name] = Number(v.Value);
				}

				var stages = new JsonObject();
				foreach ((string stage, double seconds) in run.Result.StageSeconds)
				{
					stages[stage] = Number(Math.Round(seconds, 3));
				}

				runs.Add(new JsonObject
				{
					["mapper_seed"] = run.MapperSeed,
					["surface_source"] = run.Result.Metrics.SurfaceSource,
					["metrics"] = metrics,
					["stage_seconds"] = stages,
					["total_seconds"] = Number(Math.Round(run.Result.TotalSeconds, 3)),
				});
			}

			caseArray.Add(new JsonObject
			{
				["case"] = c.CaseName,
				["scene"] = c.Kind.ToString(),
				["scene_seed"] = c.SceneSeed,
				["motion_duration"] = c.MotionDuration,
				["known_intrinsics"] = c.KnownIntrinsics,
				["summary"] = summary,
				["runs"] = runs,
			});
		}

		root["cases"] = caseArray;
		return root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n";
	}

	public static string ToTable(BenchmarkConfig config, IReadOnlyList<CaseResult> cases)
	{
		var text = new StringBuilder();
		text.AppendLine(string.Create(CultureInfo.InvariantCulture,
			$"{config.Frames} frames {config.Width}x{config.Height}, {config.Quality}, realistic motion {config.MotionDuration}, mapper seeds {string.Join(",", config.MapperSeeds)}"));
		text.AppendLine("mean / worst over the seeds; '-' is a metric no run could compute, a worst of '-' means some run failed");
		text.Append($"{"case",-30}");
		foreach ((_, string header) in TableColumns)
		{
			text.Append($"{header,16}");
		}

		text.AppendLine($"{"time s",16}");
		foreach (CaseResult c in cases)
		{
			List<MetricSummary> summary = Summarize(c);
			text.Append($"{c.CaseName + " " + c.Kind + " #" + c.SceneSeed,-30}");
			foreach ((string metric, _) in TableColumns)
			{
				MetricSummary m = summary.First(s => s.Name == metric);
				text.Append($"{Format(m.Mean) + " / " + Format(m.Worst),16}");
			}

			MetricSummary t = summary.First(s => s.Name == "total_seconds");
			text.AppendLine($"{Format(t.Mean) + " / " + Format(t.Worst),16}");
		}

		return text.ToString();
	}

	private static JsonArray Strings(IEnumerable<string> values) => new([.. values.Select(v => (JsonNode)v)]);

	private static JsonNode? Number(double value) => double.IsFinite(value) ? JsonValue.Create(value) : null;

	private static string Format(double value) =>
		!double.IsFinite(value) ? "-" : Math.Abs(value) >= 10 ? value.ToString("F1", CultureInfo.InvariantCulture) : value.ToString("F3", CultureInfo.InvariantCulture);
}
