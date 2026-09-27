// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuConformanceReport: what PatchMatchGpuConformance.RunAsync
// (PatchMatchGpuConformance.cs) found about a compute device - each check with its measured
// value, threshold and outcome, the CPU and GPU timings, and a JSON form a host can log or send
// back from a browser smoke run. Not a COLMAP port. The JSON is written with Utf8JsonWriter by
// hand rather than JsonSerializer, so the library stays trim- and AOT-clean (no reflection).

using System.Globalization;
using System.Text;
using System.Text.Json;

namespace ColmapSharp.Mvs.Testing;

/// <summary>How one conformance check came out.</summary>
public enum PatchMatchGpuConformanceOutcome
{
	/// <summary>The measured value met the threshold.</summary>
	Passed,

	/// <summary>The measured value missed the threshold, or the stage it belongs to failed to run.</summary>
	Failed,

	/// <summary>
	/// The check does not apply to this device; the detail says why. Only the CPU twin
	/// (<see cref="ReferenceComputeDevice"/>) has such checks: it runs only the PatchMatch
	/// kernels and draws its random numbers through PatchMatchRandom itself.
	/// </summary>
	NotApplicable,
}

/// <summary>One conformance check: what was measured, against what bound, and the outcome.</summary>
/// <param name="Name">A stable identifier, "stage.measure" (for example "run.photometric.depth_agreement").</param>
/// <param name="Measured">The measured value; NaN when the stage did not produce one.</param>
/// <param name="Comparison">How <paramref name="Measured"/> must relate to <paramref name="Threshold"/>: "&gt;=", "&gt;", "&lt;=", "&lt;" or "==".</param>
/// <param name="Threshold">The bound.</param>
/// <param name="Outcome">Whether the check passed.</param>
/// <param name="Detail">A human-readable note: the first mismatch, the error, or why the check does not apply.</param>
public sealed record PatchMatchGpuConformanceCheck(
	string Name,
	double Measured,
	string Comparison,
	double Threshold,
	PatchMatchGpuConformanceOutcome Outcome,
	string Detail);

/// <summary>The wall-clock time of one full-run configuration on the CPU and on the device.</summary>
/// <param name="Config">The configuration ("photometric", "geometric" or "filter").</param>
/// <param name="CpuMilliseconds">PatchMatchCpu's run.</param>
/// <param name="GpuMilliseconds">The device's first run, kernel compiles included.</param>
/// <param name="GpuRepeatMilliseconds">The device's second run of the same problem.</param>
public sealed record PatchMatchGpuConformanceTiming(string Config, double CpuMilliseconds, double GpuMilliseconds, double GpuRepeatMilliseconds);

/// <summary>The result of <see cref="PatchMatchGpuConformance.RunAsync"/>.</summary>
public sealed class PatchMatchGpuConformanceReport
{
	internal PatchMatchGpuConformanceReport(IReadOnlyList<PatchMatchGpuConformanceCheck> checks, IReadOnlyList<PatchMatchGpuConformanceTiming> timings)
	{
		Checks = checks;
		Timings = timings;
	}

	/// <summary>Every check, in the order run.</summary>
	public IReadOnlyList<PatchMatchGpuConformanceCheck> Checks { get; }

	/// <summary>The timings of the full-run configurations that ran.</summary>
	public IReadOnlyList<PatchMatchGpuConformanceTiming> Timings { get; }

	/// <summary>True when no check failed (a not-applicable check does not count against it).</summary>
	public bool AllPassed => Checks.All(c => c.Outcome != PatchMatchGpuConformanceOutcome.Failed);

	/// <summary>The checks that failed.</summary>
	public IEnumerable<PatchMatchGpuConformanceCheck> Failures => Checks.Where(c => c.Outcome == PatchMatchGpuConformanceOutcome.Failed);

	/// <summary>The report as indented JSON: allPassed, the checks and the timings.</summary>
	public string ToJson()
	{
		using var stream = new MemoryStream();
		using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
		{
			writer.WriteStartObject();
			writer.WriteBoolean("allPassed", AllPassed);
			writer.WriteStartArray("checks");
			foreach (PatchMatchGpuConformanceCheck check in Checks)
			{
				writer.WriteStartObject();
				writer.WriteString("name", check.Name);
				WriteNumber(writer, "measured", check.Measured);
				writer.WriteString("comparison", check.Comparison);
				WriteNumber(writer, "threshold", check.Threshold);
				writer.WriteString("outcome", OutcomeName(check.Outcome));
				writer.WriteString("detail", check.Detail);
				writer.WriteEndObject();
			}

			writer.WriteEndArray();
			writer.WriteStartArray("timings");
			foreach (PatchMatchGpuConformanceTiming timing in Timings)
			{
				writer.WriteStartObject();
				writer.WriteString("config", timing.Config);
				WriteNumber(writer, "cpuMs", timing.CpuMilliseconds);
				WriteNumber(writer, "gpuMs", timing.GpuMilliseconds);
				WriteNumber(writer, "gpuRepeatMs", timing.GpuRepeatMilliseconds);
				writer.WriteEndObject();
			}

			writer.WriteEndArray();
			writer.WriteEndObject();
		}

		return Encoding.UTF8.GetString(stream.ToArray());
	}

	/// <summary>One line per check and timing, for a test log.</summary>
	public override string ToString()
	{
		var text = new StringBuilder();
		text.Append(AllPassed ? "PatchMatch GPU conformance: all checks passed" : "PatchMatch GPU conformance: FAILED").AppendLine();
		foreach (PatchMatchGpuConformanceCheck check in Checks)
		{
			text.Append(CultureInfo.InvariantCulture, $"  [{OutcomeName(check.Outcome)}] {check.Name}: {check.Measured:G6} {check.Comparison} {check.Threshold:G6}");
			if (check.Detail.Length > 0)
			{
				text.Append(" (").Append(check.Detail).Append(')');
			}

			text.AppendLine();
		}

		foreach (PatchMatchGpuConformanceTiming timing in Timings)
		{
			text.Append(CultureInfo.InvariantCulture, $"  time {timing.Config}: CPU {timing.CpuMilliseconds:F1} ms, GPU {timing.GpuMilliseconds:F1} ms (compiles included), GPU repeat {timing.GpuRepeatMilliseconds:F1} ms").AppendLine();
		}

		return text.ToString();
	}

	private static string OutcomeName(PatchMatchGpuConformanceOutcome outcome) => outcome switch
	{
		PatchMatchGpuConformanceOutcome.Passed => "passed",
		PatchMatchGpuConformanceOutcome.Failed => "failed",
		_ => "not_applicable",
	};

	// JSON has no NaN or infinity; a value a stage never produced is written as null.
	private static void WriteNumber(Utf8JsonWriter writer, string name, double value)
	{
		if (double.IsFinite(value))
		{
			writer.WriteNumber(name, value);
		}
		else
		{
			writer.WriteNull(name);
		}
	}
}
