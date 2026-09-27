// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuConformance: a host's check that its IComputeDevice (Compute/IComputeDevice.cs)
// runs GPU PatchMatch (PatchMatchGpu*.cs, Mvs/Shaders/) correctly, without depending on
// colmap-sharp's test assembly - MatterCAD's desktop GPU tests call it, and a browser smoke check
// can too. Not a COLMAP port (COLMAP's PatchMatch is CUDA-only). It runs, on the synthetic scene
// of PatchMatchSyntheticScene.cs:
// 1. probes (PatchMatchGpuConformance.Probes.cs): the WGSL random numbers and u32-to-f32
//    conversion bit for bit against PatchMatchRandom, and init_random / initial_cost against
//    the CPU twin (ReferenceComputeDevice);
// 2. full runs in three configurations - photometric (48x36, 3 iterations), geometric with
//    filtering (40x30, 1 iteration, seeded with the true maps) and photometric with filtering
//    (26x19, 1 iteration), PatchMatchRunTests' configurations - each checked three ways: the
//    truth checks PatchMatchRunTests makes of a CPU run; Tier C agreement with PatchMatchCpu on
//    the same problem and seed; and a repeat on the same device that must be bit-identical.
// A real GPU is not bit-exact with the CPU (its sqrt, exp and division round differently), so
// the agreement is bounded by the thresholds below, not exact; on the CPU twin, which computes
// through the CPU code, every agreement measure is perfect. The results come back as a
// PatchMatchGpuConformanceReport (PatchMatchGpuConformanceReport.cs), never as an exception: a
// stage that throws is reported as failed and the rest still run. Only cancellation propagates.

using System.Diagnostics;

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs.Testing;

/// <summary>Checks that an <see cref="IComputeDevice"/> runs GPU PatchMatch correctly.</summary>
public static partial class PatchMatchGpuConformance
{
	/// <summary>Of the pixels valid (nonzero depth) in both the device and the CPU run, at least this fraction ...</summary>
	public const double MinDepthAgreementFraction = 0.95;

	/// <summary>... has a device depth within this relative difference of the CPU's.</summary>
	public const double MaxRelativeDepthDifference = 0.01;

	/// <summary>The median angle between device and CPU normals of pixels valid in both, in degrees, is below this.</summary>
	public const double MaxMedianNormalAngleDegrees = 2;

	/// <summary>The device's valid-pixel count is within this fraction of the CPU's.</summary>
	public const double MaxValidCountDifference = 0.02;

	/// <summary>
	/// The consistency graphs agree at least this much: (pixel, source) memberships in both over
	/// those in either (Jaccard), so pixels neither run marks consistent cannot inflate it.
	/// </summary>
	public const double MinGraphAgreement = 0.95;

	/// <summary>init_random's depths differ from the CPU twin's by at most this many units in the last place.</summary>
	public const int MaxInitDepthUlps = 1;

	/// <summary>initial_cost's costs agree with the CPU twin's within this absolute difference ...</summary>
	public const float MaxInitialCostDifference = 1e-4f;

	/// <summary>... on at least this fraction of the (pixel, source) entries.</summary>
	public const double MinInitialCostAgreement = 0.999;

	/// <summary>Truth checks: more than this fraction of interior depths is within the tolerance of the truth.</summary>
	public const double MinTruthDepthFraction = 0.9;

	/// <summary>The photometric truth check's relative depth tolerance (PatchMatchRunTests).</summary>
	public const float PhotometricTruthDepthTolerance = 0.02f;

	/// <summary>The geometric truth check's relative depth tolerance (PatchMatchRunTests).</summary>
	public const float GeometricTruthDepthTolerance = 0.01f;

	/// <summary>The photometric truth check: the median interior normal is within this angle of the truth, in degrees.</summary>
	public const float MaxTrueNormalAngleDegrees = 15;

	/// <summary>Pixels this close to the edge are left out of the truth checks.</summary>
	public const int TruthBorder = 4;

	/// <summary>The PRNG seed of every run, device and CPU alike.</summary>
	public const ulong Seed = 0x5EED_1234_ABCDUL;

	private const string TwinProbeReason =
		"the CPU twin runs only the PatchMatch kernels and draws through PatchMatchRandom itself; the kernel probe and full runs cover it";

	/// <summary>
	/// Runs every check on <paramref name="device"/> and reports the results. Takes seconds on a
	/// GPU; the CPU runs it compares against dominate.
	/// </summary>
	/// <param name="device">The device to check.</param>
	/// <param name="ct">Cancels between and within stages.</param>
	public static async Task<PatchMatchGpuConformanceReport> RunAsync(IComputeDevice device, CancellationToken ct = default)
	{
		ArgumentNullException.ThrowIfNull(device);
		var checks = new List<PatchMatchGpuConformanceCheck>();
		var timings = new List<PatchMatchGpuConformanceTiming>();
		bool isTwin = device is ReferenceComputeDevice;

		await RandomProbeAsync(device, isTwin, checks, ct).ConfigureAwait(false);
		await ConversionProbeAsync(device, isTwin, checks, ct).ConfigureAwait(false);
		await KernelProbeAsync(device, checks, ct).ConfigureAwait(false);
		foreach (string config in new[] { "photometric", "geometric", "filter" })
		{
			ct.ThrowIfCancellationRequested();
			await FullRunAsync(device, config, checks, timings, ct).ConfigureAwait(false);
		}

		return new PatchMatchGpuConformanceReport(checks, timings);
	}

	/// <summary>One configuration: the CPU run, two device runs, and the truth, agreement and repeat checks.</summary>
	private static async Task FullRunAsync(IComputeDevice device, string config, List<PatchMatchGpuConformanceCheck> checks, List<PatchMatchGpuConformanceTiming> timings, CancellationToken ct)
	{
		string prefix = "run." + config + ".";
		(int width, int height, PatchMatchOptions options, bool geometric) = config switch
		{
			"photometric" => (48, 36, PatchMatchSyntheticScene.Options(3, geomConsistency: false, filter: false), false),
			"geometric" => (40, 30, PatchMatchSyntheticScene.Options(1, geomConsistency: true, filter: true), true),
			_ => (26, 19, PatchMatchSyntheticScene.Options(1, geomConsistency: false, filter: true), false),
		};

		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = PatchMatchSyntheticScene.Scene(width, height);
		PatchMatch.Problem problem = geometric
			? PatchMatchSyntheticScene.Problem(images, truth, normals)
			: PatchMatchSyntheticScene.Problem(images);

		var cpuTime = Stopwatch.StartNew();
		var cpu = new PatchMatchCpu(options, problem, Seed);
		cpu.Run(ct);
		cpuTime.Stop();

		var progress = new List<double>();
		PatchMatchGpu gpu;
		PatchMatchGpu repeat;
		var gpuTime = new Stopwatch();
		var repeatTime = new Stopwatch();
		try
		{
			gpuTime.Start();
			gpu = new PatchMatchGpu(options, problem, Seed);
			await gpu.RunAsync(device, ct, new SynchronousProgress(progress)).ConfigureAwait(false);
			gpuTime.Stop();
			repeatTime.Start();
			repeat = new PatchMatchGpu(options, problem, Seed);
			await repeat.RunAsync(device, ct).ConfigureAwait(false);
			repeatTime.Stop();
		}
		catch (Exception e) when (e is not OperationCanceledException)
		{
			checks.Add(StageFailed(prefix + "ran", e, threshold: 1));
			return;
		}

		checks.Add(Compare(prefix + "ran", 1, "==", 1));
		timings.Add(new PatchMatchGpuConformanceTiming(config, cpuTime.Elapsed.TotalMilliseconds, gpuTime.Elapsed.TotalMilliseconds, repeatTime.Elapsed.TotalMilliseconds));
		AddTruthChecks(prefix, config, width, height, options, truth[0], gpu, progress, checks);
		AddAgreementChecks(prefix, gpu, cpu, checks);
		AddRepeatChecks(prefix, gpu, repeat, checks);
	}

	/// <summary>The checks PatchMatchRunTests makes of a CPU run of the same configuration.</summary>
	private static void AddTruthChecks(string prefix, string config, int width, int height, PatchMatchOptions options, DepthMap truth, PatchMatchGpu gpu, List<double> progress, List<PatchMatchGpuConformanceCheck> checks)
	{
		prefix += "truth.";
		DepthMap depthMap = gpu.GetDepthMap();
		checks.Add(Compare(prefix + "size_matches", depthMap.GetWidth() == width && depthMap.GetHeight() == height ? 1 : 0, "==", 1,
			$"{depthMap.GetWidth()}x{depthMap.GetHeight()}, expected {width}x{height}"));

		// One report per sweep, ending at 1.
		checks.Add(Compare(prefix + "progress_reports", progress.Count, "==", 4 * options.NumIterations));
		checks.Add(Compare(prefix + "progress_final", progress.Count > 0 ? progress[^1] : double.NaN, "==", 1));

		List<int> graph = gpu.GetConsistentImageIdxs();
		var consistent = new bool[height, width];
		int foreign = 0;
		int consistentPixels = 0;
		foreach ((int row, int col, int imageIdx) in PatchMatchSyntheticScene.Memberships(graph))
		{
			consistentPixels += consistent[row, col] ? 0 : 1;
			consistent[row, col] = true;
			foreign += imageIdx == 1 || imageIdx == 2 ? 0 : 1;
		}

		checks.Add(Compare(prefix + "graph_foreign_sources", foreign, "==", 0, "memberships naming an image other than sources 1 and 2"));

		switch (config)
		{
			case "photometric":
				checks.Add(Compare(prefix + "depth_min", depthMap.GetDepthMin(), "==", 2));
				checks.Add(Compare(prefix + "depth_within_tolerance", PatchMatchSyntheticScene.FractionWithin(depthMap, truth, PhotometricTruthDepthTolerance, TruthBorder), ">", MinTruthDepthFraction,
					$"fraction of interior depths within {PhotometricTruthDepthTolerance} (relative) of the truth"));
				checks.Add(Compare(prefix + "median_normal_cosine", PatchMatchSyntheticScene.MedianTrueNormalCosine(gpu.GetNormalMap(), TruthBorder), ">", MathF.Cos(MaxTrueNormalAngleDegrees * MathF.PI / 180),
					$"cosine to the true normal; the bound is cos {MaxTrueNormalAngleDegrees} deg"));
				checks.Add(Compare(prefix + "graph_memberships", graph.Count, "==", 0, "no filtering, so no consistency graph"));
				checks.Add(Compare(prefix + "sel_prob_depth", gpu.GetSelProbMap().GetDepth(), "==", 2));
				break;

			case "geometric":
				checks.Add(Compare(prefix + "depth_within_tolerance", PatchMatchSyntheticScene.FractionWithin(depthMap, truth, GeometricTruthDepthTolerance, TruthBorder), ">", MinTruthDepthFraction,
					$"fraction of interior depths within {GeometricTruthDepthTolerance} (relative) of the truth"));
				checks.Add(Compare(prefix + "consistent_pixels", consistentPixels, ">", width * height / 2));
				string center = string.Join(",", PatchMatchSyntheticScene.Memberships(graph).Where(m => m.Row == 15 && m.Col == 20).Select(m => m.ImageIdx));
				checks.Add(Compare(prefix + "center_sources", center == "1,2" ? 1 : 0, "==", 1, $"sources consistent at (15, 20): [{center}], expected [1,2]"));
				break;

			default:
				// PatchMatchRunTests claims nothing of this run beyond determinism (the repeat
				// checks); guard against a vacuous pass.
				checks.Add(Compare(prefix + "consistent_pixels", consistentPixels, ">", 0));
				break;
		}

		if (options.Filter)
		{
			int unfilteredDepths = 0;
			for (int row = 0; row < height; ++row)
			{
				for (int col = 0; col < width; ++col)
				{
					unfilteredDepths += !consistent[row, col] && depthMap.Get(row, col) != 0 ? 1 : 0;
				}
			}

			checks.Add(Compare(prefix + "filtered_nonzero_depths", unfilteredDepths, "==", 0, "pixels with no consistent source must have depth 0"));
		}
	}

	/// <summary>The Tier C agreement between the device run and the CPU run of the same problem and seed.</summary>
	private static void AddAgreementChecks(string prefix, PatchMatchGpu gpu, PatchMatchCpu cpu, List<PatchMatchGpuConformanceCheck> checks)
	{
		prefix += "cpu.";
		float[] gpuDepth = gpu.GetDepthMap().Data;
		float[] cpuDepth = cpu.GetDepthMap().Data;
		float[] gpuNormals = gpu.GetNormalMap().Data;
		float[] cpuNormals = cpu.GetNormalMap().Data;
		int planeSize = gpuDepth.Length;

		int gpuValid = gpuDepth.Count(d => d != 0);
		int cpuValid = cpuDepth.Count(d => d != 0);
		int both = 0;
		int depthAgree = 0;
		var angles = new List<double>();
		for (int i = 0; i < planeSize; ++i)
		{
			if (gpuDepth[i] == 0 || cpuDepth[i] == 0)
			{
				continue;
			}

			both++;
			if (Math.Abs(gpuDepth[i] - cpuDepth[i]) < MaxRelativeDepthDifference * Math.Abs(cpuDepth[i]))
			{
				depthAgree++;
			}

			// The angle as 2 atan2(|a - b|, |a + b|), not acos(a . b): the normals are unit
			// length only to float precision, so acos puts identical normals about 0.02 degrees
			// apart, and it is ill-conditioned at small angles, which is where the bound is.
			double diff2 = 0;
			double sum2 = 0;
			for (int c = 0; c < 3; ++c)
			{
				double a = gpuNormals[c * planeSize + i];
				double b = cpuNormals[c * planeSize + i];
				diff2 += (a - b) * (a - b);
				sum2 += (a + b) * (a + b);
			}

			angles.Add(2 * Math.Atan2(Math.Sqrt(diff2), Math.Sqrt(sum2)) * 180 / Math.PI);
		}

		angles.Sort();
		var gpuGraph = PatchMatchSyntheticScene.Memberships(gpu.GetConsistentImageIdxs()).ToHashSet();
		var cpuGraph = PatchMatchSyntheticScene.Memberships(cpu.GetConsistentImageIdxs()).ToHashSet();
		int graphUnion = gpuGraph.Union(cpuGraph).Count();

		checks.Add(Compare(prefix + "pixels_valid_in_both", both, ">", 0));
		checks.Add(Compare(prefix + "depth_agreement", both == 0 ? double.NaN : depthAgree / (double)both, ">=", MinDepthAgreementFraction,
			$"fraction of {both} pixels valid in both within {MaxRelativeDepthDifference} (relative) of the CPU depth"));
		checks.Add(Compare(prefix + "median_normal_angle_degrees", angles.Count == 0 ? double.NaN : angles[angles.Count / 2], "<", MaxMedianNormalAngleDegrees));
		checks.Add(Compare(prefix + "valid_count_difference", Math.Abs(gpuValid - cpuValid) / (double)Math.Max(cpuValid, 1), "<=", MaxValidCountDifference,
			$"device {gpuValid} valid pixels, CPU {cpuValid}"));
		checks.Add(Compare(prefix + "graph_agreement", graphUnion == 0 ? 1 : gpuGraph.Intersect(cpuGraph).Count() / (double)graphUnion, ">=", MinGraphAgreement,
			$"Jaccard over {graphUnion} (pixel, source) memberships"));
	}

	/// <summary>Same device, same inputs, same schedule: the second run must repeat the first bit for bit.</summary>
	private static void AddRepeatChecks(string prefix, PatchMatchGpu gpu, PatchMatchGpu repeat, List<PatchMatchGpuConformanceCheck> checks)
	{
		prefix += "repeat.";
		checks.Add(DifferingBits(prefix + "depth_differing_values", repeat.GetDepthMap().Data, gpu.GetDepthMap().Data));
		checks.Add(DifferingBits(prefix + "normal_differing_values", repeat.GetNormalMap().Data, gpu.GetNormalMap().Data));
		checks.Add(DifferingBits(prefix + "sel_prob_differing_values", repeat.GetSelProbMap().Data, gpu.GetSelProbMap().Data));
		bool sameGraph = repeat.GetConsistentImageIdxs().SequenceEqual(gpu.GetConsistentImageIdxs());
		checks.Add(Compare(prefix + "graph_identical", sameGraph ? 1 : 0, "==", 1));
	}

	// Bitwise, so NaN payloads and signed zeros count; the detail names the first difference.
	private static PatchMatchGpuConformanceCheck DifferingBits(string name, float[] actual, float[] expected)
	{
		if (actual.Length != expected.Length)
		{
			return Compare(name, double.NaN, "==", 0, $"lengths differ: {actual.Length} vs {expected.Length}");
		}

		int differing = 0;
		string detail = "";
		for (int i = 0; i < actual.Length; ++i)
		{
			if (BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i]) && differing++ == 0)
			{
				detail = $"first at {i}: {actual[i]:R} vs {expected[i]:R}";
			}
		}

		return Compare(name, differing, "==", 0, detail);
	}

	/// <summary>A check whose outcome is <paramref name="measured"/> <paramref name="comparison"/> <paramref name="threshold"/> (NaN fails).</summary>
	private static PatchMatchGpuConformanceCheck Compare(string name, double measured, string comparison, double threshold, string detail = "")
	{
		bool passed = comparison switch
		{
			">=" => measured >= threshold,
			">" => measured > threshold,
			"<=" => measured <= threshold,
			"<" => measured < threshold,
			"==" => measured == threshold,
			_ => throw new ArgumentOutOfRangeException(nameof(comparison), comparison, "Not a comparison."),
		};
		return new PatchMatchGpuConformanceCheck(name, measured, comparison, threshold,
			passed ? PatchMatchGpuConformanceOutcome.Passed : PatchMatchGpuConformanceOutcome.Failed, detail);
	}

	// A check whose stage threw: no measured value, failed, with the error as the detail.
	private static PatchMatchGpuConformanceCheck StageFailed(string name, Exception error, double threshold = 0) =>
		new(name, double.NaN, "==", threshold, PatchMatchGpuConformanceOutcome.Failed, $"the stage threw {error.GetType().Name}: {error.Message}");

	private static PatchMatchGpuConformanceCheck NotApplicable(string name, string reason) =>
		new(name, double.NaN, "==", 0, PatchMatchGpuConformanceOutcome.NotApplicable, reason);

	// Progress<T> posts to the thread pool; this records reports as they happen.
	private sealed class SynchronousProgress(List<double> reports) : IProgress<double>
	{
		public void Report(double value) => reports.Add(value);
	}
}
