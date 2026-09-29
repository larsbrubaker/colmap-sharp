// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkConfig: the benchmark runner's command line (Program.cs). The defaults are the
// checked-in baseline's matrix (benchmarks/baseline.json): every scene kind, one scene seed,
// three mapper seeds, 480x360 frames at Low quality; about 30 minutes on a laptop CPU, nearly
// all of it the sphere's dense stages. Also holds the per-run and per-case result records the
// report is built from.
//
// Cases:
// - "realistic" (every kind): the frames span 27% of the scene's pendulum (motionDuration
//   0.27), so at 40 frames the object turns about 5 degrees between frames and 10 at most, like
//   the real mouse clip (5-10). At 480x360 the object is about 140 px across. The dark object
//   and the box register no frame at 240x180 (4-17 keypoints a frame; pycolmap 4.2.0 registers
//   none there either) nor, so far, at 480x360.
// - "fast" (--fast-kinds, TexturedSphere by default): the whole pendulum, 15 degrees a frame
//   and up to 37, the harder variant.
// - "known" (--known-kinds, TexturedSphere and DarkObject by default): realistic motion with
//   the true intrinsics given and bundle adjustment's focal refinement off
//   (SyntheticBenchmarkOptions.KnownIntrinsics), as with a calibrated camera or a trusted EXIF
//   focal. The self-calibrating cases collapse the focal on this small object (divergence 141).
//
// --quick is the everyday check (benchmarks/baseline-quick.json): sparse only (no dense stages,
// so the surface and silhouette metrics are 0 or null and only registration, focal and pose
// are measured), the realistic and known cases of TexturedSphere and DarkObject, no fast case,
// and mapper seeds 1 and 2. Options after --quick still override it. Its report records
// "dense": false, so --compare refuses to set it against the full baseline.

using System.Globalization;

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;

namespace ColmapSharp.Benchmarks;

/// <summary>One reconstruction of one scene under one mapper seed.</summary>
internal sealed record RunResult(int MapperSeed, SyntheticBenchmarkResult Result);

/// <summary>All mapper seeds of one (scene kind, scene seed).</summary>
internal sealed record CaseResult(string CaseName, SyntheticObjectKind Kind, uint SceneSeed, double MotionDuration, bool KnownIntrinsics, IReadOnlyList<RunResult> Runs);

/// <summary>The runner's settings.</summary>
internal sealed class BenchmarkConfig
{
	public const string Usage = """
		ColmapSharp.Benchmarks [options]
		  --kinds <list>         scene kinds, comma separated (default: DarkObject,TexturedSphere,TexturelessBox)
		  --frames <n>           frames per scene (default 40)
		  --motion <d>           fraction of the pendulum the realistic cases span (default 0.27)
		  --fast-kinds <list>    scene kinds that also get a "fast" case over the whole pendulum
		                         (default TexturedSphere; "none" for no fast cases)
		  --known-kinds <list>   scene kinds that also get a "known" (true intrinsics, no focal
		                         refinement) case (default TexturedSphere,DarkObject; "none")
		  --width <px>           frame width (default 480)
		  --height <px>          frame height (default 360)
		  --quality <q>          Low, Medium, High or Extreme (default Low)
		  --data <d>             Individual or Video (default Individual)
		  --scene-seeds <list>   scene seeds, comma separated (default 1)
		  --mapper-seeds <list>  mapper seeds, comma separated (default 1,2,3)
		  --tau <fraction>       F-score threshold as a fraction of the object's diagonal (default 0.01)
		  --masks                give the reconstruction the true masks
		  --video-tracking       video data with KLT tracks added to the matches (implies --data Video);
		                         compare with a --data Video run of the same cases
		  --out <path>           write the JSON report here
		  --compare <path>       compare with an earlier JSON report; exit 1 on a regression
		  --tolerance <t>        allowed relative worsening of an error metric's mean,
		                         max(0.01, t * |baseline|) (default 0.05); scores in [0, 1] may
		                         drop 0.02, and any lost frame or extra model is flagged
		  --quick                everyday check in a few minutes: sparse only, TexturedSphere and
		                         DarkObject (realistic and known), no fast case, mapper seeds 1,2;
		                         compare with benchmarks/baseline-quick.json
		  --help                 show this text
		""";

	public bool ShowHelp { get; private set; }

	public List<SyntheticObjectKind> Kinds { get; private set; } = [.. Enum.GetValues<SyntheticObjectKind>()];

	public int Frames { get; private set; } = 40;

	public double MotionDuration { get; private set; } = 0.27;

	public List<SyntheticObjectKind> FastKinds { get; private set; } = [SyntheticObjectKind.TexturedSphere];

	public List<SyntheticObjectKind> KnownKinds { get; private set; } = [SyntheticObjectKind.TexturedSphere, SyntheticObjectKind.DarkObject];

	public int Width { get; private set; } = 480;

	public int Height { get; private set; } = 360;

	public AutomaticReconstructionOptions.QualityLevel Quality { get; private set; } = AutomaticReconstructionOptions.QualityLevel.Low;

	public AutomaticReconstructionOptions.DataType Data { get; private set; } = AutomaticReconstructionOptions.DataType.Individual;

	public List<uint> SceneSeeds { get; private set; } = [1];

	public List<int> MapperSeeds { get; private set; } = [1, 2, 3];

	/// <summary>Whether to run the dense stages (depth maps, fusion, meshing); --quick turns them off.</summary>
	public bool Dense { get; private set; } = true;

	public double TauFraction { get; private set; } = new SurfaceMetricOptions().TauFraction;

	public bool UseTrueMasks { get; private set; }

	/// <summary>Whether the reconstruction adds KLT video tracks to its matches (--video-tracking).</summary>
	public bool VideoTracking { get; private set; }

	public string? OutputPath { get; private set; }

	public string? ComparePath { get; private set; }

	public double Tolerance { get; private set; } = 0.05;

	public static BenchmarkConfig Parse(string[] args)
	{
		var config = new BenchmarkConfig();
		for (int i = 0; i < args.Length; i++)
		{
			string name = args[i];
			string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{name} needs a value");
			switch (name)
			{
				case "--help":
				case "-h":
					config.ShowHelp = true;
					break;
				case "--quick":
					config.Dense = false;
					config.Kinds = [SyntheticObjectKind.TexturedSphere, SyntheticObjectKind.DarkObject];
					config.FastKinds = [];
					config.KnownKinds = [SyntheticObjectKind.TexturedSphere, SyntheticObjectKind.DarkObject];
					config.MapperSeeds = [1, 2];
					break;
				case "--kinds":
					config.Kinds = [.. List(Next()).Select(Enum.Parse<SyntheticObjectKind>)];
					break;
				case "--frames":
					config.Frames = Int(Next());
					break;
				case "--motion":
					config.MotionDuration = double.Parse(Next(), CultureInfo.InvariantCulture);
					break;
				case "--fast-kinds":
					string fast = Next();
					config.FastKinds = fast == "none" ? [] : [.. List(fast).Select(Enum.Parse<SyntheticObjectKind>)];
					break;
				case "--known-kinds":
					string known = Next();
					config.KnownKinds = known == "none" ? [] : [.. List(known).Select(Enum.Parse<SyntheticObjectKind>)];
					break;
				case "--width":
					config.Width = Int(Next());
					break;
				case "--height":
					config.Height = Int(Next());
					break;
				case "--quality":
					config.Quality = Enum.Parse<AutomaticReconstructionOptions.QualityLevel>(Next(), ignoreCase: true);
					break;
				case "--data":
					config.Data = Enum.Parse<AutomaticReconstructionOptions.DataType>(Next(), ignoreCase: true);
					break;
				case "--scene-seeds":
					config.SceneSeeds = [.. List(Next()).Select(s => uint.Parse(s, CultureInfo.InvariantCulture))];
					break;
				case "--mapper-seeds":
					config.MapperSeeds = [.. List(Next()).Select(Int)];
					break;
				case "--tau":
					config.TauFraction = double.Parse(Next(), CultureInfo.InvariantCulture);
					break;
				case "--video-tracking":
					config.VideoTracking = true;
					config.Data = AutomaticReconstructionOptions.DataType.Video;
					break;
				case "--masks":
					config.UseTrueMasks = true;
					break;
				case "--out":
					config.OutputPath = Next();
					break;
				case "--compare":
					config.ComparePath = Next();
					break;
				case "--tolerance":
					config.Tolerance = double.Parse(Next(), CultureInfo.InvariantCulture);
					break;
				default:
					throw new ArgumentException($"Unknown option {name}");
			}
		}

		return config;
	}

	private static string[] List(string value) => value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

	private static int Int(string value) => int.Parse(value, CultureInfo.InvariantCulture);
}
