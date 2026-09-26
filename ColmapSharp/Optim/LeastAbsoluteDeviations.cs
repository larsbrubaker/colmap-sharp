// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// LeastAbsoluteDeviationSolver: colmap/optim/least_absolute_deviations.h and .cc. Solves
// min ||A x - b||_1 by ADMM over the factored normal equations A^T A; rotation averaging
// and global positioning build on it. The normal equations are factored by
// LinearAlgebra/SimplicialCholesky.cs (SimplicialLLT) or by SparseCholeskyWithFallbackSolver
// (SparseCholesky.cs). Tests: ColmapSharp.Tests/Optim/LeastAbsoluteDeviationsTests.cs
// (least_absolute_deviations_test.cc 1:1).
//
// Tier C: an iterative solver over a Tier B factorization; COLMAP's tests check the
// converged solution within tolerances.
//
// Translation notes:
// - C++ keeps references to the caller's options and A; this port copies the options and
//   keeps a reference to A (SparseMatrixCsc's pattern is immutable, its values are not, so
//   callers must not change A's values between construction and Solve, as in C++).
// - std::runtime_error (underdetermined system) maps to InvalidOperationException.
// - Solve writes into the caller's x (the Eigen::VectorXd* out-parameter), which must have
//   A.Cols entries; C++ would resize it.
// - SolverType.SupernodalCholmodLLT keeps COLMAP's name but selects the managed
//   SparseCholeskyWithFallbackSolver (CHOLMOD is not ported, docs/CPP_DIVERGENCES.md
//   entry 13).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Least absolute deviations (LAD) fitting via ADMM by solving the problem
///
///        min || A x - b ||_1
///
/// This implementation is based on the paper "Distributed Optimization and Statistical
/// Learning via the Alternating Direction Method of Multipliers" by Boyd et al. and the
/// Matlab implementation at
/// https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html
/// Port of colmap::LeastAbsoluteDeviationSolver.
/// </summary>
public sealed class LeastAbsoluteDeviationSolver
{
	/// <summary>Port of LeastAbsoluteDeviationSolver::Options.</summary>
	public sealed class Options
	{
		/// <summary>Linear solver for the normal equations.</summary>
		public enum SolverType
		{
			/// <summary>Simplicial LLT (Eigen::SimplicialLLT in COLMAP).</summary>
			SimplicialLLT,

			/// <summary>LLT with LDLT fallback (CHOLMOD supernodal LLT in COLMAP).</summary>
			SupernodalCholmodLLT,
		}

		/// <summary>Augmented Lagrangian parameter.</summary>
		public double Rho { get; set; } = 1.0;

		/// <summary>Over-relaxation parameter, typical values are between 1.0 and 1.8.</summary>
		public double Alpha { get; set; } = 1.0;

		/// <summary>Maximum solver iterations.</summary>
		public int MaxNumIterations { get; set; } = 1000;

		/// <summary>Absolute solution threshold, as suggested by Boyd et al.</summary>
		public double AbsoluteTolerance { get; set; } = 1e-4;

		/// <summary>Relative solution threshold, as suggested by Boyd et al.</summary>
		public double RelativeTolerance { get; set; } = 1e-2;

		/// <summary>
		/// Tikhonov ridge added to the diagonal of the normal equations A^T A before
		/// factorization. Set to a small positive value (e.g., 1e-12) for poorly conditioned
		/// but mathematically positive definite systems. Default 0 disables regularization.
		/// </summary>
		public double RidgeRegularization { get; set; }

		/// <summary>Which linear solver factors A^T A.</summary>
		public SolverType Solver { get; set; } = SolverType.SimplicialLLT;

		/// <summary>A copy of these options.</summary>
		public Options Clone() => (Options)MemberwiseClone();
	}

	private readonly Options _options;
	private readonly SparseMatrixCsc _a;
	private readonly SimplicialCholesky? _simplicial;
	private readonly SparseCholeskyWithFallbackSolver? _fallback;

	/// <summary>Validates the options and factors A^T A (plus the ridge).</summary>
	public LeastAbsoluteDeviationSolver(Options options, SparseMatrixCsc a)
	{
		_options = options.Clone();
		_a = a;
		Check.Ge(_options.RidgeRegularization, 0.0);
		Check.Gt(_options.Rho, 0.0);
		Check.Gt(_options.Alpha, 0.0);
		Check.Gt(_options.MaxNumIterations, 0);
		Check.Ge(_options.AbsoluteTolerance, 0.0);
		Check.Ge(_options.RelativeTolerance, 0.0);
		if (a.Rows < a.Cols)
		{
			throw new InvalidOperationException("Underdetermined systems not supported.");
		}

		SparseMatrixCsc ata = NormalEquations(a, _options.RidgeRegularization);
		if (_options.Solver == Options.SolverType.SimplicialLLT)
		{
			_simplicial = new SimplicialCholesky(SimplicialCholeskyKind.LLT);
			_simplicial.Compute(ata);
			Valid = _simplicial.Info == ComputationInfo.Success;
		}
		else
		{
			_fallback = new SparseCholeskyWithFallbackSolver();
			Valid = _fallback.Compute(ata);
		}
	}

	/// <summary>
	/// False if the factorization of A^T A failed during construction (e.g., the system is
	/// rank deficient or numerically not positive definite), in which case Solve always
	/// returns false without producing NaN output.
	/// </summary>
	public bool Valid { get; }

	/// <summary>
	/// Runs ADMM from scratch and writes the solution into <paramref name="x"/> (length
	/// A.Cols). Returns false, leaving x unchanged, when the factorization failed.
	/// </summary>
	public bool Solve(VectorXd b, VectorXd x)
	{
		Check.That(x is not null);
		Check.Eq(x.Length, _a.Cols);
		Check.Eq(b.Length, _a.Rows);
		if (!Valid)
		{
			return false;
		}

		int rows = _a.Rows;
		var z = VectorXd.Zero(rows);
		var u = VectorXd.Zero(rows);
		double bNorm = b.Norm();
		double epsPriThreshold = Math.Sqrt(rows) * _options.AbsoluteTolerance;
		double epsDualThreshold = Math.Sqrt(_a.Cols) * _options.AbsoluteTolerance;
		double alpha = _options.Alpha;
		double rho = _options.Rho;

		for (int i = 0; i < _options.MaxNumIterations; i++)
		{
			if (!SolveNormal(_a.TransposeMultiply(b + z - u), out VectorXd solution))
			{
				return false;
			}

			solution.AsSpan().CopyTo(x.AsSpan());
			VectorXd ax = _a.Multiply(x);
			VectorXd axHat = alpha * ax + (1 - alpha) * (z + b);

			VectorXd zOld = z;
			z = Shrinkage(axHat - b + u, 1 / rho);
			u = u + (axHat - z - b);

			double rNorm = (ax - z - b).Norm();
			double sNorm = (-rho * _a.TransposeMultiply(z - zOld)).Norm();
			double epsPri = epsPriThreshold
				+ _options.RelativeTolerance * Math.Max(bNorm, Math.Max(ax.Norm(), z.Norm()));
			double epsDual = epsDualThreshold
				+ _options.RelativeTolerance * (rho * _a.TransposeMultiply(u)).Norm();

			if (rNorm < epsPri && sNorm < epsDual)
			{
				break;
			}
		}

		return true;
	}

	private bool SolveNormal(VectorXd rhs, out VectorXd solution)
	{
		if (_simplicial is not null)
		{
			solution = _simplicial.Solve(rhs);
			return true;
		}

		return _fallback!.Solve(rhs, out solution);
	}

	private static VectorXd Shrinkage(VectorXd a, double kappa)
	{
		var result = new VectorXd(a.Length);
		ReadOnlySpan<double> src = a.AsSpan();
		Span<double> dst = result.AsSpan();
		for (int i = 0; i < src.Length; i++)
		{
			dst[i] = Math.Min(src[i] + kappa, 0) + Math.Max(src[i] - kappa, 0);
		}

		return result;
	}

	private static SparseMatrixCsc NormalEquations(SparseMatrixCsc a, double ridgeRegularization)
	{
		SparseMatrixCsc ata = a.TransposeTimesSelf();
		if (ridgeRegularization > 0)
		{
			// Eigen's coeffRef inserts a diagonal entry for an all-zero column of A;
			// AddToDiagonal does the same.
			ata = ata.AddToDiagonal(ridgeRegularization);
		}

		return ata;
	}
}
