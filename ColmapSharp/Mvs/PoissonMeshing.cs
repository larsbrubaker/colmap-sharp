// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause) and Kazhdan's PoissonRecon as vendored by COLMAP (MIT), see
// THIRD_PARTY_NOTICES.md.
//
// PoissonMeshing: port of colmap::mvs::PoissonMeshing (mvs/poisson_meshing.cc) - screened
// Poisson surface reconstruction of a fused point cloud. COLMAP runs PoissonRecon.cpp's
// RunPoissonRecon with --in --out --pointWeight --depth [--fullDepth] [--colors] [--density],
// then, when trim != 0, SurfaceTrimmer.cpp's RunSurfaceTrimmer on the file PoissonRecon wrote.
// Run(options, positions, normals, colors) is that chain in memory, with every PoissonRecon
// setting COLMAP does not pass at its PoissonRecon.cpp default (PoissonSolutionParameters):
// Poisson::Solver::Solve's stages from Reconstructors.h (PoissonSampleSet, PoissonDensity,
// PoissonSplat, PoissonInterpolation, PoissonFinalize, PoissonFemConstraints, PoissonSystem,
// PoissonImplicitEvaluator), then extractLevelSet (PoissonLevelSetExtractor, PoissonMeshOutput)
// and PoissonSurfaceTrimmer. Run(options, inputPath, outputPath) is the file wrapper, reading
// and writing PLY as PoissonRecon does (PoissonMeshing.Ply.cs). Each stage is Tier A against
// the vendored C++ (the PoissonTreeOracleTests harnesses); the whole chain is Tier A against
// upstream's own code built the same way (see below) and Tier C against the pycolmap wheel
// (PoissonMeshingOracleTests), since the optimized wheel may round a few floats differently
// (docs/CPP_DIVERGENCES.md, entry 125).
//
// Translation notes:
// - COLMAP formats pointWeight and trim with std::to_string and PoissonRecon parses them back
//   with atof into a float; CppToStringAsFloat reproduces that exactly.
// - PoissonRecon ignores --colors for PLY input: it carries the input's extra vertex properties
//   through as auxiliary data, so the mesh has colors exactly when the input does (see
//   PoissonMeshingOptions.Color). --density is passed when trim > 0, so the density "value"
//   property is in the output only then.
// - num_threads only sets PoissonRecon's thread pool; the port runs sequentially. Its result
//   equals upstream's run with num_threads = 1. Upstream with more threads (COLMAP's default
//   num_threads = -1 uses every hardware thread) sums some floats and numbers some nodes and
//   vertices in thread order, so its output varies from run to run in the last bits
//   (docs/CPP_DIVERGENCES.md, entries 106 and 123).
// - The file wrapper is Tier A against upstream's own RunPoissonRecon and RunSurfaceTrimmer
//   built at -O1 (PoissonMeshingOracleTests.Exact.cs): the output PLY is byte-identical, except
//   where PoissonSplat.LogF's rounding differs from Apple's logf (entry 116).
// - COLMAP opens (and truncates) the output for writing before it starts (THROW_CHECK_PATH_OPEN),
//   so an unwritable output path throws; the file wrapper does the same. It then catches a
//   failing reconstruction, logs a warning and returns false; the file wrapper does the same,
//   except that cancellation propagates as OperationCanceledException.
// - Whether the mesh has colors follows the input's vertex properties, not its point count: a
//   PLY with red, green and blue and no points still writes a header with those properties.

using System.Globalization;
using System.Numerics;

using ColmapSharp.Mvs.PoissonRecon;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Screened Poisson surface reconstruction of an oriented point cloud (port of colmap::mvs::PoissonMeshing).</summary>
public static partial class PoissonMeshing
{
	// Where each stage's progress falls in the overall fraction.
	private const double TreeEnd = 0.10, PrepareEnd = 0.25, SolveEnd = 0.60, IsoValueEnd = 0.65, ExtractEnd = 0.95;

	/// <summary>
	/// Reconstructs a surface from oriented points as COLMAP's PoissonMeshing does, in memory.
	/// <paramref name="positions"/> and <paramref name="normals"/> hold x, y, z per point;
	/// <paramref name="colors"/> is empty (an uncolored mesh) or holds red, green, blue bytes per
	/// point (a colored mesh, as fused.ply's colors give). The mesh has the density values when
	/// options.Trim is positive (COLMAP's --density), and is trimmed by them when Trim is nonzero.
	/// </summary>
	/// <param name="options">The options; they must pass Check.</param>
	/// <param name="positions">x, y, z per point.</param>
	/// <param name="normals">nx, ny, nz per point (need not be unit length).</param>
	/// <param name="colors">Empty, or red, green, blue per point.</param>
	/// <param name="cancellationToken">Checked between stages and inside the long ones.</param>
	/// <param name="progress">Receives the overall fraction done, ending at 1.</param>
	public static PoissonMeshOutput Run(
		PoissonMeshingOptions options,
		ReadOnlySpan<float> positions,
		ReadOnlySpan<float> normals,
		ReadOnlySpan<byte> colors = default,
		CancellationToken cancellationToken = default,
		IProgress<double>? progress = null) =>
		Run(options, positions, normals, colors, hasColors: !colors.IsEmpty, cancellationToken, progress);

	// The in-memory Run with the colors' presence explicit, so that an input with color
	// properties but no points still gives a mesh with (no) colors, as PoissonRecon's
	// DynamicFactory is built from the PLY header, not from the points.
	private static PoissonMeshOutput Run(
		PoissonMeshingOptions options,
		ReadOnlySpan<float> positions,
		ReadOnlySpan<float> normals,
		ReadOnlySpan<byte> colors,
		bool hasColors,
		CancellationToken cancellationToken,
		IProgress<double>? progress)
	{
		ArgumentNullException.ThrowIfNull(options);
		Check.That(options.Check());
		int pointCount = positions.Length / 3;
		if (hasColors ? colors.Length != 3 * pointCount : !colors.IsEmpty)
		{
			throw new ArgumentException("colors must be empty or hold red, green and blue bytes per point.", nameof(colors));
		}

		// DynamicFactory< float > reads each uchar channel as a float.
		int auxPerPoint = hasColors ? 3 : 0;
		float[] aux = new float[colors.Length];
		for (int i = 0; i < colors.Length; i++)
		{
			aux[i] = colors[i];
		}

		// RunPoissonRecon's arguments: --pointWeight and --depth always, --fullDepth depth when
		// depth < 5 ("Full depth cannot exceed system depth"), the rest PoissonRecon.cpp's defaults.
		var parameters = new PoissonSolutionParameters
		{
			PointWeight = CppToStringAsFloat(options.PointWeight),
			Depth = (uint)options.Depth,
			FullDepth = options.Depth < 5 ? (uint)options.Depth : 5u,
		};

		IProgress<PoissonProgress>? stageProgress = progress == null ? null : new StageProgress(progress);
		PoissonSampleSet set = PoissonSampleSet.Build(positions, normals, aux, auxPerPoint, parameters, cancellationToken, stageProgress);
		progress?.Report(TreeEnd);
		cancellationToken.ThrowIfCancellationRequested();

		(SortedTreeNodes sorted, DensityEstimator density, SparseNodeData? data, float[] solution) =
			Solve(set, parameters, cancellationToken, stageProgress, progress);

		FemTree tree = set.Tree;
		float isoValue = new PoissonImplicitEvaluator(tree, sorted, PoissonFemConstraints.TestSignature, solution)
			.IsoValue(set, stageProgress, cancellationToken).Value;
		progress?.Report(IsoValueEnd);

		// --density (trim > 0) makes the extraction carry the density weight as "value".
		double extractEnd = options.Trim != 0 ? ExtractEnd : 1.0;
		var extractor = new PoissonLevelSetExtractor(tree, sorted, PoissonFemConstraints.TestSignature, solution, isoValue, options.Trim > 0 ? density : null, data);
		extractor.Extract(cancellationToken, progress == null ? null : new BandProgress(progress, IsoValueEnd, extractEnd));
		PoissonMeshOutput mesh = PoissonMeshOutput.FromLevelSet(extractor, set.UnitCubeToModel);

		if (options.Trim != 0)
		{
			cancellationToken.ThrowIfCancellationRequested();
			mesh = PoissonSurfaceTrimmer.Trim(mesh, CppToStringAsFloat(options.Trim), cancellationToken, progress == null ? null : new BandProgress(progress, ExtractEnd, 1.0));
		}

		progress?.Report(1.0);
		return mesh;
	}

	/// <summary>
	/// Port of <c>colmap::mvs::PoissonMeshing( options , input_path , output_path )</c>: reads the
	/// oriented points of the PLY at <paramref name="inputPath"/> (it must have normals), and
	/// writes the reconstructed mesh as a binary PLY to <paramref name="outputPath"/> with the
	/// properties PoissonRecon (and SurfaceTrimmer) write: x, y, z, the density "value" when
	/// options.Trim is positive, and red, green, blue when the input has them. Invalid options or
	/// paths throw, as does an output path that cannot be opened for writing (COLMAP's
	/// THROW_CHECK_PATH_OPEN, which also creates or truncates the file before the
	/// reconstruction starts). Returns false, after logging a warning, when the reconstruction
	/// fails.
	/// </summary>
	public static bool Run(
		PoissonMeshingOptions options,
		string inputPath,
		string outputPath,
		CancellationToken cancellationToken = default,
		IProgress<double>? progress = null)
	{
		ArgumentNullException.ThrowIfNull(options);
		Check.That(options.Check());
		Check.That(FileUtils.HasFileExtension(inputPath, ".ply"), inputPath);
		Check.That(File.Exists(inputPath), inputPath);
		Check.That(FileUtils.HasFileExtension(outputPath, ".ply"), outputPath);

		// THROW_CHECK_PATH_OPEN( output_path ): outside the try, so a directory or a missing
		// parent directory throws rather than returning false.
		using (FileOpen.OpenWrite(outputPath))
		{
		}

		try
		{
			PoissonInputPoints input = ReadInputPoints(inputPath);
			PoissonMeshOutput mesh = Run(options, input.Positions, input.Normals, input.Colors, input.HasColors, cancellationToken, progress);
			WriteMeshPly(outputPath, mesh);
			return true;
		}
		catch (OperationCanceledException)
		{
			throw;
		}
		catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException or InvalidOperationException or NotSupportedException)
		{
			Log.Warning($"PoissonRecon failed with exception: {e.Message}");
			return false;
		}
	}

	/// <summary>
	/// A double passed to PoissonRecon on its command line: formatted by std::to_string (printf's
	/// "%f", six decimals, correctly rounded with ties to even) and parsed back by
	/// CmdLineParameter&lt;float&gt; (<c>float( atof( str ) )</c>). So 10.0 stays 10, but
	/// 1e-7 becomes 0 and 0.0078125 becomes 0.007812f.
	/// </summary>
	internal static float CppToStringAsFloat(double value) =>
		(float)double.Parse(CppToString(value), NumberStyles.Float, CultureInfo.InvariantCulture);

	/// <summary>std::to_string( double ): printf's "%f" of the exact binary value.</summary>
	internal static string CppToString(double value)
	{
		if (double.IsNaN(value))
		{
			return double.IsNegative(value) ? "-nan" : "nan";
		}

		if (double.IsInfinity(value))
		{
			return value < 0 ? "-inf" : "inf";
		}

		long bits = BitConverter.DoubleToInt64Bits(value);
		bool negative = bits < 0;
		int exponent = (int)((bits >> 52) & 0x7FF);
		long mantissa = bits & 0xFFFFFFFFFFFFFL;
		if (exponent == 0)
		{
			exponent = 1;
		}
		else
		{
			mantissa |= 1L << 52;
		}

		// |value| * 10^6 = mantissa * 10^6 * 2^(exponent - 1075), rounded half to even.
		int shift = exponent - 1075;
		BigInteger scaled = new BigInteger(mantissa) * 1_000_000;
		BigInteger micro;
		if (shift >= 0)
		{
			micro = scaled << shift;
		}
		else
		{
			BigInteger denominator = BigInteger.One << -shift;
			micro = BigInteger.DivRem(scaled, denominator, out BigInteger remainder);
			int half = (remainder * 2).CompareTo(denominator);
			if (half > 0 || (half == 0 && !micro.IsEven))
			{
				micro += 1;
			}
		}

		string digits = micro.ToString(CultureInfo.InvariantCulture).PadLeft(7, '0');
		string text = digits[..^6] + "." + digits[^6..];
		return negative ? "-" + text : text;
	}

	// Solve's stages after reading the samples, as Reconstructors.h's Poisson::Solver::Solve runs
	// them with COLMAP's settings (no envelope, no value interpolation, approximate
	// interpolation): the density, the normal field, the color field, the interpolation
	// constraints when pointWeight > 0, finalizing, the constraints and the solve.
	private static (SortedTreeNodes Sorted, DensityEstimator Density, SparseNodeData? Data, float[] Solution) Solve(
		PoissonSampleSet set,
		PoissonSolutionParameters parameters,
		CancellationToken cancellationToken,
		IProgress<PoissonProgress>? stageProgress,
		IProgress<double>? progress)
	{
		FemTree tree = set.Tree;
		int depth = (int)parameters.Depth, baseDepth = (int)parameters.BaseDepth, solveDepth = (int)parameters.SolveDepth;
		tree.ResetNodeIndices(0);

		// setDensityEstimator< 1 , Reconstructor::WeightDegree >( samples , kernelDepth , samplesPerNode ).
		DensityEstimator density = PoissonDensity.SetDensityEstimator(set, 1, 2, (int)parameters.KernelDepth, parameters.SamplesPerNode);
		cancellationToken.ThrowIfCancellationRequested();

		// The normal field, negated.
		SparseNodeData normals = PoissonSplat.SetNormalField(set, density, baseDepth, depth, parameters.LowDepthCutOff, out var pointDepthAndWeight, cancellationToken, stageProgress);

		// The auxiliary data (colors), scaled per level, only when the input has some.
		SparseNodeData? data = set.AuxPerPoint > 0 ? PoissonSplat.SetAuxField(set, parameters.PerLevelDataScaleFactor) : null;
		cancellationToken.ThrowIfCancellationRequested();

		// The approximate point interpolation constraints, at targetValue 0.5.
		SparseNodeData? interpolation = null;
		float pointWeight = 0;
		if (parameters.PointWeight > 0)
		{
			pointWeight = parameters.PointWeight * PoissonInterpolation.AverageSampleWeight(pointDepthAndWeight.WeightSum, pointDepthAndWeight.TotalWeight);
			interpolation = PoissonInterpolation.Build(set, 0.5f, pointWeight, depth, 1);
		}

		// finalizeForMultigrid< MaxDegree = 2 , 1 >, re-keying ( normals , density , aux ).
		SortedTreeNodes sorted = PoissonFinalize.FinalizeForMultigrid(tree, 2, baseDepth, (int)parameters.FullDepth, normals, out _, interpolation, normals, density, data);
		cancellationToken.ThrowIfCancellationRequested();

		// The FEM (divergence) constraints at solveDepth = params.depth, then the point constraints.
		var constraints = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, PoissonFemConstraints.CreateDivergenceIntegrator(), normals, constraints, depth);
		if (interpolation != null)
		{
			PoissonFemConstraints.AddInterpolationConstraints(tree, sorted, interpolation, constraints, depth);
		}

		progress?.Report(PrepareEnd);
		cancellationToken.ThrowIfCancellationRequested();

		// solveSystem with FEMIntegrator::System( { 0 , 1 } ); only the point interpolation
		// constrains the DC term (ConstrainsDCTerm of a null iInfo is false).
		var system = new FemSystemIntegrator(PoissonFemConstraints.TestSignature, 1, 0.0, 1.0);
		var solver = new PoissonSystem(tree, sorted, system, new PoissonPointEvaluator(PoissonFemConstraints.TestSignature, 1, solveDepth), interpolation, pointWeight);
		IProgress<int>? depthProgress = progress == null ? null : new DepthProgress(progress, solveDepth);
		float[] solution = solver.Solve(constraints, baseDepth, baseDepth, solveDepth, (int)parameters.Iters, (int)parameters.BaseVCycles, parameters.CgSolverAccuracy, constrainsDCTerm: interpolation != null, depthProgress, cancellationToken);
		progress?.Report(SolveEnd);
		return (sorted, density, data, solution);
	}

	// Maps a fraction onto [start, end] of the overall progress.
	private sealed class BandProgress(IProgress<double> overall, double start, double end) : IProgress<double>
	{
		public void Report(double value) => overall.Report(start + ((end - start) * value));
	}

	// Maps the solver's solved depth onto the solve band.
	private sealed class DepthProgress(IProgress<double> overall, int solveDepth) : IProgress<int>
	{
		public void Report(int value) => overall.Report(PrepareEnd + ((SolveEnd - PrepareEnd) * Math.Clamp((value + 1.0) / (solveDepth + 1.0), 0, 1)));
	}

	// Maps the stages' own reports (tree build, splatting, iso-value) onto their bands.
	private sealed class StageProgress(IProgress<double> overall) : IProgress<PoissonProgress>
	{
		public void Report(PoissonProgress value)
		{
			(double start, double end) = value.Stage switch
			{
				PoissonStage.TreeBuild => (0.0, TreeEnd),
				PoissonStage.IsoValue => (SolveEnd, IsoValueEnd),
				_ => (TreeEnd, PrepareEnd),
			};
			overall.Report(start + ((end - start) * Math.Clamp(value.Fraction, 0, 1)));
		}
	}
}
