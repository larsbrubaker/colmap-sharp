// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SparseCholeskyWithFallbackSolver: colmap/optim/sparse_cholesky.h and sparse_cholesky.cc.
// Tries an LL^T factorization first and falls back to LDL^T once LL^T reports the matrix is
// not positive definite. Used by LeastAbsoluteDeviations.cs (and later rotation averaging).
// Tests: ColmapSharp.Tests/Optim/SparseCholeskyTests.cs (sparse_cholesky_test.cc 1:1).
//
// COLMAP's first stage is CHOLMOD's supernodal LLT (GPL, not ported); here it is the
// simplicial LLT of LinearAlgebra/SimplicialCholesky.cs with AMD ordering, which accepts
// and rejects the same matrices (both stop at the first pivot <= 0) but groups the
// floating-point work differently (divergence 13). The fallback stage
// is Eigen::SimplicialLDLT in COLMAP and SimplicialCholesky's LDLT here. Tier B.
//
// Translation notes:
// - Solve's Eigen::VectorXd* out-parameter becomes `out VectorXd`. After a failed
//   factorization Eigen's solve() has no defined result; here x is a zero vector and Solve
//   returns false.
// - COLMAP's LOG(WARNING) on fallback goes to Util/Log.cs; the fallback is also visible
//   through UsesLdlt.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Sparse Cholesky solver that tries LLT first (fastest) and falls back to the
/// numerically more tolerant LDLT once LLT reports the matrix is not positive definite.
/// Port of colmap::SparseCholeskyWithFallbackSolver.
/// </summary>
public sealed class SparseCholeskyWithFallbackSolver
{
	private readonly SimplicialCholesky _llt = new(SimplicialCholeskyKind.LLT);
	private readonly SimplicialCholesky _ldlt = new(SimplicialCholeskyKind.LDLT);
	private bool _useLdlt;

	/// <summary>True once LLT failed and the solver switched to LDLT for good.</summary>
	public bool UsesLdlt => _useLdlt;

	/// <summary>One-shot factorization (analyze + factorize).</summary>
	public bool Compute(SparseMatrixCsc a)
	{
		AnalyzePattern(a);
		return Factorize(a);
	}

	/// <summary>
	/// For iterative reuse with matrices of identical sparsity but changing values, call
	/// AnalyzePattern once and then Factorize per iteration.
	/// </summary>
	public void AnalyzePattern(SparseMatrixCsc a)
	{
		_llt.AnalyzePattern(a);

		// LDLT pattern is analyzed lazily on first fallback, since the common case never
		// needs it. Once the solver has fallen back it stays on LDLT, so a new pattern must
		// reach the LDLT stage here. (COLMAP only re-analyzes supernodal_ here, and Eigen's
		// factorize() on a stale analysis has undefined results; this port re-analyzes
		// instead of reproducing that.)
		if (_useLdlt)
		{
			_ldlt.AnalyzePattern(a);
		}
	}

	/// <summary>Numeric factorization; returns false when both stages fail.</summary>
	public bool Factorize(SparseMatrixCsc a)
	{
		if (!_useLdlt)
		{
			if (_llt.Factorize(a))
			{
				return true;
			}

			Log.Warning("Supernodal Cholesky factorization failed; falling back to simplicial LDLT for ill-conditioned system.");
			_ldlt.AnalyzePattern(a);
			_useLdlt = true;
		}

		return _ldlt.Factorize(a);
	}

	/// <summary>Solves A x = b with the current factorization.</summary>
	public bool Solve(VectorXd b, out VectorXd x)
	{
		SimplicialCholesky active = _useLdlt ? _ldlt : _llt;
		if (active.Info != ComputationInfo.Success)
		{
			x = new VectorXd(b.Length);
			return false;
		}

		x = active.Solve(b);
		return true;
	}
}
