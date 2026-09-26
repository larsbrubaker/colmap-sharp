// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeres: colmap/estimators/bundle_adjustment_ceres.h and the options /
// summary half of bundle_adjustment_ceres.cc: CeresBundleAdjustmentOptions (COLMAP's solver
// defaults, CreateLossFunction, CreateSolverOptions with the linear solver chosen by image
// count and the single-thread cut-off for small problems), CeresBundleAdjustmentSummary,
// CeresPosePriorBundleAdjustmentOptions, the abstract CeresBundleAdjuster and the factories.
// The problem setup (residuals, parameterization, gauge fixing) is
// BundleAdjustmentCeres.Default.cs; the pose prior adjuster is BundleAdjustmentCeres.PosePrior.cs.
// The solver is Solver/LeastSquaresSolver.cs.
// Tests: ColmapSharp.Tests/Estimators/BundleAdjustmentCeresTests.cs
// (bundle_adjustment_ceres_test.cc) and BundleAdjustmentTests.cs.
//
// Tier C (outcome): the solve goes through the managed Ceres replacement, so final poses and
// points agree with COLMAP to convergence tolerance, not bit for bit. The problem layout
// (which blocks exist, their manifolds, what is constant) is exact, and the tests pin it via
// num_residuals_reduced / num_effective_parameters_reduced.
//
// Translation notes:
// - The GPU paths (use_gpu, gpu_index, the *_gpu_solver thresholds, SolveWithGpuFallback's
//   retry) keep their options so configurations round-trip, but ColmapSharp has no GPU
//   solver: CreateSolverOptions always takes COLMAP's "compiled without CUDA" branch.
// - Ceres' sparse_linear_algebra_library_type is always available here (the managed sparse
//   Cholesky), so the SPARSE_SCHUR branch is always eligible (COLMAP's has_sparse).
// - The result is independent of NumThreads: the solver is deterministic across thread
//   counts (Solver/), so the single-thread cut-off only affects speed.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::CeresBundleAdjustmentOptions::LossFunctionType.</summary>
public enum BundleAdjustmentLossFunctionType
{
	/// <summary>Plain least squares.</summary>
	Trivial,

	/// <summary>ceres::SoftLOneLoss.</summary>
	SoftL1,

	/// <summary>ceres::CauchyLoss.</summary>
	Cauchy,

	/// <summary>ceres::HuberLoss.</summary>
	Huber,
}

/// <summary>Port of colmap::CeresBundleAdjustmentOptions.</summary>
public sealed class CeresBundleAdjustmentOptions
{
	/// <summary>COLMAP's defaults, including the solver options it overrides.</summary>
	public CeresBundleAdjustmentOptions()
	{
		SolverOptions.FunctionTolerance = 0.0;
		SolverOptions.GradientTolerance = 1e-4;
		SolverOptions.ParameterTolerance = 0.0;
		SolverOptions.MaxNumIterations = 100;
		SolverOptions.MaxLinearSolverIterations = 200;
		SolverOptions.MaxNumConsecutiveInvalidSteps = 10;
		SolverOptions.MaxConsecutiveNonmonotonicSteps = 10;
		SolverOptions.NumThreads = -1;
	}

	/// <summary>Loss function type.</summary>
	public BundleAdjustmentLossFunctionType LossFunctionType { get; set; } = BundleAdjustmentLossFunctionType.Trivial;

	/// <summary>Scaling factor determines residual at which robustification takes place.</summary>
	public double LossFunctionScale { get; set; } = 1.0;

	/// <summary>Whether to use a GPU solver. ColmapSharp has none; kept for option parity.</summary>
	public bool UseGpu { get; set; }

	/// <summary>GPU index (unused, see UseGpu).</summary>
	public string GpuIndex { get; set; } = "-1";

	/// <summary>The Ceres solver options.</summary>
	public SolverOptions SolverOptions { get; private set; } = new();

	/// <summary>Heuristic threshold to switch from CPU to GPU based solvers (unused).</summary>
	public int MinNumImagesGpuSolver { get; set; } = 50;

	/// <summary>
	/// Heuristic threshold on the minimum number of residuals to enable multi-threading. Note
	/// that single-threaded is typically better for small bundle adjustment problems due to
	/// the overhead of threading.
	/// </summary>
	public int MinNumResidualsForCpuMultiThreading { get; set; } = 50000;

	/// <summary>Largest image count solved with DENSE_SCHUR.</summary>
	public int MaxNumImagesDirectDenseCpuSolver { get; set; } = 50;

	/// <summary>Largest image count solved with SPARSE_SCHUR; above it ITERATIVE_SCHUR.</summary>
	public int MaxNumImagesDirectSparseCpuSolver { get; set; } = 1000;

	/// <summary>GPU counterpart of MaxNumImagesDirectDenseCpuSolver (unused).</summary>
	public int MaxNumImagesDirectDenseGpuSolver { get; set; } = 200;

	/// <summary>GPU counterpart of MaxNumImagesDirectSparseCpuSolver (unused).</summary>
	public int MaxNumImagesDirectSparseGpuSolver { get; set; } = 4000;

	/// <summary>
	/// Whether to automatically select solver type based on problem size. When false, uses
	/// the linear solver and preconditioner types from SolverOptions directly.
	/// </summary>
	public bool AutoSelectSolverType { get; set; } = true;

	/// <summary>Creates the loss function for these options.</summary>
	public LossFunction CreateLossFunction() => CreateLossFunction(LossFunctionType, LossFunctionScale);

	/// <summary>Port of the file-local colmap::CreateLossFunction.</summary>
	public static LossFunction CreateLossFunction(BundleAdjustmentLossFunctionType type, double scale) => type switch
	{
		BundleAdjustmentLossFunctionType.Trivial => new TrivialLoss(),
		BundleAdjustmentLossFunctionType.SoftL1 => new SoftLOneLoss(scale),
		BundleAdjustmentLossFunctionType.Cauchy => new CauchyLoss(scale),
		BundleAdjustmentLossFunctionType.Huber => new HuberLoss(scale),
		_ => throw new ArgumentOutOfRangeException(nameof(type)),
	};

	/// <summary>Creates solver options tailored to the config and problem.</summary>
	public SolverOptions CreateSolverOptions(BundleAdjustmentConfig config, Problem problem)
	{
		SolverOptions customSolverOptions = SolverOptions.Clone();
		int numImages = config.NumImages;

		// Auto-select solver type based on problem size, unless disabled.
		if (AutoSelectSolverType)
		{
			if (numImages <= MaxNumImagesDirectDenseCpuSolver)
			{
				customSolverOptions.LinearSolverType = LinearSolverType.DenseSchur;
			}
			else if (numImages <= MaxNumImagesDirectSparseCpuSolver)
			{
				customSolverOptions.LinearSolverType = LinearSolverType.SparseSchur;
			}
			else
			{
				// Indirect sparse (preconditioned CG) solver.
				customSolverOptions.LinearSolverType = LinearSolverType.IterativeSchur;
				customSolverOptions.PreconditionerType = PreconditionerType.SchurJacobi;
			}
		}

		customSolverOptions.NumThreads = problem.NumResiduals < MinNumResidualsForCpuMultiThreading
			? 1
			: GetEffectiveNumThreads(customSolverOptions.NumThreads);

		string? solverError = customSolverOptions.Validate();
		Util.Check.That(solverError is null, solverError);
		return customSolverOptions;
	}

	/// <summary>Port of CeresBundleAdjustmentOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check() =>
		LossFunctionScale >= 0
		&& MaxNumImagesDirectDenseCpuSolver < MaxNumImagesDirectSparseCpuSolver
		&& MaxNumImagesDirectDenseGpuSolver < MaxNumImagesDirectSparseGpuSolver;

	/// <summary>A deep copy, including the solver options.</summary>
	public CeresBundleAdjustmentOptions Clone()
	{
		var copy = (CeresBundleAdjustmentOptions)MemberwiseClone();
		copy.SolverOptions = SolverOptions.Clone();
		return copy;
	}

	// Port of colmap::GetEffectiveNumThreads (util/threading.cc): a non-positive count means
	// all hardware threads, and the result is at least 1.
	private static int GetEffectiveNumThreads(int numThreads) =>
		Math.Max(numThreads <= 0 ? Environment.ProcessorCount : numThreads, 1);
}

/// <summary>Port of colmap::CeresBundleAdjustmentSummary: the summary plus the full solver summary.</summary>
public sealed class CeresBundleAdjustmentSummary : BundleAdjustmentSummary
{
	private CeresBundleAdjustmentSummary(SolverSummary ceresSummary)
	{
		CeresSummary = ceresSummary;
	}

	/// <summary>The solver's summary.</summary>
	public SolverSummary CeresSummary { get; }

	/// <summary>Port of CeresBundleAdjustmentSummary::Create.</summary>
	public static CeresBundleAdjustmentSummary Create(SolverSummary ceresSummary)
	{
		ArgumentNullException.ThrowIfNull(ceresSummary);
		return new CeresBundleAdjustmentSummary(ceresSummary)
		{
			TerminationType = ceresSummary.TerminationType switch
			{
				Solver.TerminationType.Convergence => BundleAdjustmentTerminationType.Convergence,
				Solver.TerminationType.NoConvergence => BundleAdjustmentTerminationType.NoConvergence,
				Solver.TerminationType.Failure => BundleAdjustmentTerminationType.Failure,
				Solver.TerminationType.UserSuccess => BundleAdjustmentTerminationType.UserSuccess,
				Solver.TerminationType.UserFailure => BundleAdjustmentTerminationType.UserFailure,
				_ => throw new InvalidOperationException($"Unknown Ceres termination type: {ceresSummary.TerminationType}"),
			},
			NumResiduals = ceresSummary.NumResidualsReduced,
		};
	}

	/// <inheritdoc/>
	public override string BriefReport() => CeresSummary.BriefReport();
}

/// <summary>Port of colmap::CeresPosePriorBundleAdjustmentOptions.</summary>
public sealed class CeresPosePriorBundleAdjustmentOptions
{
	/// <summary>Loss function for prior position loss.</summary>
	public BundleAdjustmentLossFunctionType PriorPositionLossFunctionType { get; set; } = BundleAdjustmentLossFunctionType.Trivial;

	/// <summary>Threshold on the residual for the robust loss.</summary>
	public double PriorPositionLossScale { get; set; } = Math.Sqrt(MathUtils.ChiSquare95ThreeDof);

	/// <summary>Port of CeresPosePriorBundleAdjustmentOptions::Check.</summary>
	public bool Check() => PriorPositionLossScale > 0;

	/// <summary>A copy.</summary>
	public CeresPosePriorBundleAdjustmentOptions Clone() => (CeresPosePriorBundleAdjustmentOptions)MemberwiseClone();
}

/// <summary>Port of colmap::CeresBundleAdjuster: a bundle adjuster with access to its problem.</summary>
public abstract class CeresBundleAdjuster : BundleAdjuster
{
	/// <inheritdoc/>
	protected CeresBundleAdjuster(BundleAdjustmentOptions options, BundleAdjustmentConfig config)
		: base(options, config)
	{
	}

	/// <summary>The underlying least-squares problem.</summary>
	public abstract Problem Problem { get; }
}

/// <summary>The Ceres bundle adjuster factories of bundle_adjustment_ceres.h.</summary>
public static class CeresBundleAdjusters
{
	/// <summary>Port of colmap::CreateDefaultCeresBundleAdjuster.</summary>
	public static CeresBundleAdjuster CreateDefaultCeresBundleAdjuster(
		BundleAdjustmentOptions options, BundleAdjustmentConfig config, Reconstruction reconstruction) =>
		new DefaultBundleAdjuster(options, config, reconstruction);

	/// <summary>Port of colmap::CreatePosePriorCeresBundleAdjuster (BundleAdjustmentCeres.PosePrior.cs).</summary>
	public static CeresBundleAdjuster CreatePosePriorCeresBundleAdjuster(
		BundleAdjustmentOptions options,
		PosePriorBundleAdjustmentOptions priorOptions,
		BundleAdjustmentConfig config,
		IEnumerable<Geometry.PosePrior> posePriors,
		Reconstruction reconstruction) =>
		new PosePriorBundleAdjuster(options, priorOptions, config, posePriors, reconstruction);

	// SolveWithGpuFallback without the GPU retry (no GPU here): the solver options for this
	// problem, COLMAP's CancellationCallback when check_if_stopped is set, and the solve.
	internal static SolverSummary Solve(
		BundleAdjustmentOptions options, BundleAdjustmentConfig config, Problem problem, CancellationToken cancellationToken)
	{
		SolverOptions solverOptions = options.Ceres!.CreateSolverOptions(config, problem);
		if (options.CheckIfStopped is not null)
		{
			solverOptions.Callbacks.Add(new CancellationCallback(options.CheckIfStopped));
		}

		return LeastSquaresSolver.Solve(solverOptions, problem, cancellationToken);
	}

	/// <summary>
	/// Port of CreateSummaryAndLogFailure: wraps the solver summary and logs an error when
	/// the solution is not usable.
	/// </summary>
	internal static CeresBundleAdjustmentSummary CreateSummaryAndLogFailure(SolverSummary ceresSummary, string context)
	{
		CeresBundleAdjustmentSummary summary = CeresBundleAdjustmentSummary.Create(ceresSummary);
		if (!summary.IsSolutionUsable())
		{
			Util.Log.Error($"{context} failed: {ceresSummary.Message}");
		}

		return summary;
	}

	private sealed class CancellationCallback(Func<bool> checkIfStopped) : IIterationCallback
	{
		public CallbackReturnType Invoke(IterationSummary summary) =>
			checkIfStopped() ? CallbackReturnType.SolverTerminateSuccessfully : CallbackReturnType.SolverContinue;
	}
}
