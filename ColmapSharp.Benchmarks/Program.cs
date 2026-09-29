// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapSharp.Benchmarks: the reconstruction benchmark runner (docs/QUALITY_PLAN.md, stage 0b).
// Not a COLMAP port. For every case (BenchmarkConfig.cs) it renders a SyntheticObjectScene,
// reconstructs it with the full AutomaticReconstructionController under several mapper seeds
// (SyntheticBenchmark.Run), and reports each metric's mean and worst case over the seeds: on
// real captures one seed in 10-20 splits the model, so a single seed can mislead. It writes the
// raw runs and the summary as JSON and prints a short table.
//
// With --compare it also checks the summary against an earlier JSON (benchmarks/baseline.json)
// and exits with 1 on a regression (BenchmarkSummary.Regressions: more failed runs, a worst gone
// null, or a mean worse by more than the tolerance), or 2 when the reports' settings differ.
// Without --out the JSON is still built, and compared, but not written.
//
// Usage: dotnet run -c Release --project ColmapSharp.Benchmarks -- [options]; --help lists them.

using System.Globalization;

using ColmapSharp.Benchmarks;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;

BenchmarkConfig config;
try
{
	config = BenchmarkConfig.Parse(args);
}
catch (ArgumentException e)
{
	Console.Error.WriteLine(e.Message);
	Console.Error.WriteLine(BenchmarkConfig.Usage);
	return 2;
}

if (config.ShowHelp)
{
	Console.WriteLine(BenchmarkConfig.Usage);
	return 0;
}

// Every kind gets a realistic case; the fast kinds also get one over the whole pendulum, and the
// known kinds one with the true intrinsics (BenchmarkConfig's header).
var matrix = config.Kinds.Select(k => ("realistic", k, config.MotionDuration, false))
	.Concat(config.FastKinds.Select(k => ("fast", k, 1.0, false)))
	.Concat(config.KnownKinds.Select(k => ("known", k, config.MotionDuration, true)));
var cases = new List<CaseResult>();
foreach ((string caseName, SyntheticObjectKind kind, double motion, bool knownIntrinsics) in matrix)
{
	foreach (uint sceneSeed in config.SceneSeeds)
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(kind, config.Frames, config.Width, config.Height, sceneSeed, motionDuration: motion);
		var runs = new List<RunResult>();
		foreach (int mapperSeed in config.MapperSeeds)
		{
			var options = new SyntheticBenchmarkOptions
			{
				Quality = config.Quality,
				Data = config.Data,
				MapperSeed = mapperSeed,
				UseTrueMasks = config.UseTrueMasks,
				KnownIntrinsics = knownIntrinsics,
			};
			options.Surface.TauFraction = config.TauFraction;
			Console.Error.Write($"{caseName} {kind} scene seed {sceneSeed} mapper seed {mapperSeed} ... ");
			SyntheticBenchmarkResult result = SyntheticBenchmark.Run(scene, options);
			Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
				$"{result.TotalSeconds:F1} s, {result.Metrics.NumRegistered}/{result.Metrics.NumFrames} registered, F {result.Metrics.Surface.FScore:F3}"));
			runs.Add(new RunResult(mapperSeed, result));
		}

		cases.Add(new CaseResult(caseName, kind, sceneSeed, motion, knownIntrinsics, runs));
	}
}

string json = Report.ToJson(config, cases);
if (config.OutputPath is not null)
{
	string? dir = Path.GetDirectoryName(Path.GetFullPath(config.OutputPath));
	if (dir is not null)
	{
		Directory.CreateDirectory(dir);
	}

	File.WriteAllText(config.OutputPath, json);
	Console.Error.WriteLine($"Wrote {config.OutputPath}");
}

Console.WriteLine(Report.ToTable(config, cases));

if (config.ComparePath is not null)
{
	List<string> regressions;
	try
	{
		regressions = BenchmarkSummary.Regressions(File.ReadAllText(config.ComparePath), json, config.Tolerance);
	}
	catch (InvalidOperationException e)
	{
		Console.Error.WriteLine(e.Message);
		return 2;
	}

	foreach (string line in regressions)
	{
		Console.WriteLine("REGRESSION " + line);
	}

	Console.WriteLine(regressions.Count == 0 ? "No regressions against " + config.ComparePath : $"{regressions.Count} regression(s)");
	return regressions.Count == 0 ? 0 : 1;
}

return 0;
