// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SimplicialCholesky: sparse Cholesky factorization P A P^T = L L^T or L D L^T of a
// symmetric matrix in SparseMatrixCsc form, the replacement for Eigen::SimplicialLLT,
// Eigen::SimplicialLDLT and CHOLMOD's supernodal LLT behind COLMAP's optim/sparse_cholesky
// (SparseCholeskyWithFallbackSolver, Optim/SparseCholesky.cs) and
// optim/least_absolute_deviations. Neighbors: SparseMatrixCsc.cs (storage) and
// AmdOrdering.cs (the fill-reducing permutation P).
//
// Algorithms, written from their published descriptions (no Eigen, CSparse, LDL or CHOLMOD
// code was read or ported; those are MPL/LGPL/GPL, docs/LICENSE_AUDIT.md):
// - Elimination tree: J. W. H. Liu, "The Role of Elimination Trees in Sparse
//   Factorization", SIAM J. Matrix Anal. Appl. 11(1), 1990 (ancestor path compression).
// - Row structure of L as the union of etree paths from the nonzeros of each column of the
//   upper triangle, and the up-looking factorization that solves one sparse triangular
//   system per row: Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006, ch. 4,
//   used as a description only. Column counts come from walking the same row structures
//   symbolically, which costs O(nnz(L)).
//
// AnalyzePattern does the ordering and all symbolic work once; Factorize only scatters the
// new values through a precomputed map and runs the numeric pass, so repeated
// factorizations of one pattern (every Levenberg-Marquardt or IRLS iteration) reuse it.
//
// Like Eigen's sparse Cholesky, only one triangle of the input is read (Lower by default).
// Failure follows Eigen's documented info() contract: LLT reports NumericalIssue at the
// first pivot <= 0; LDLT accepts negative pivots (indefinite matrices) and fails only on a
// pivot that is exactly zero. Tier B: same math, different operation order than Eigen.

namespace ColmapSharp.LinearAlgebra;

/// <summary>Which factorization a <see cref="SimplicialCholesky"/> computes.</summary>
public enum SimplicialCholeskyKind
{
	/// <summary>P A P^T = L L^T (Eigen::SimplicialLLT); needs a positive-definite matrix.</summary>
	LLT,

	/// <summary>P A P^T = L D L^T with unit L (Eigen::SimplicialLDLT); tolerates indefinite matrices.</summary>
	LDLT,
}

/// <summary>Fill-reducing ordering applied before the factorization.</summary>
public enum SparseOrdering
{
	/// <summary>Approximate minimum degree (Eigen's default AMDOrdering).</summary>
	Amd,

	/// <summary>No permutation (Eigen::NaturalOrdering).</summary>
	Natural,
}

/// <summary>
/// Simplicial (column-by-column) sparse Cholesky with reusable symbolic analysis.
/// Replacement for Eigen::SimplicialLLT / Eigen::SimplicialLDLT.
/// </summary>
public sealed class SimplicialCholesky
{
	private readonly SimplicialCholeskyKind _kind;
	private readonly SparseOrdering _ordering;
	private readonly SymmetricPart _part;

	// Symbolic analysis.
	private int _n = -1;
	private SparseMatrixCsc? _analyzed;
	private int[] _patternColPtr = [];
	private int[] _patternRowIdx = [];
	private int[] _perm = []; // new index -> original index
	private int[] _cColPtr = []; // upper triangle of P A P^T
	private int[] _cRowIdx = [];
	private int[] _valueMap = []; // input entry -> slot in _cValues, or -1 when not read
	private int[] _parent = [];
	private int[] _lColPtr = [];
	private int[] _lRowIdx = [];

	// Numeric factorization.
	private double[] _cValues = [];
	private double[] _lValues = []; // strictly lower part of L, column by column
	private double[] _diag = []; // L's diagonal (LLT) or D (LDLT)
	private bool _factorized;

	/// <summary>Creates an unanalyzed solver.</summary>
	public SimplicialCholesky(
		SimplicialCholeskyKind kind = SimplicialCholeskyKind.LDLT,
		SparseOrdering ordering = SparseOrdering.Amd,
		SymmetricPart part = SymmetricPart.Lower)
	{
		_kind = kind;
		_ordering = ordering;
		_part = part;
	}

	/// <summary>Result of the last Factorize; NumericalIssue before any factorization.</summary>
	public ComputationInfo Info { get; private set; } = ComputationInfo.NumericalIssue;

	/// <summary>The fill-reducing permutation: Permutation[k] is the original index of row/column k.</summary>
	public ReadOnlySpan<int> Permutation => _perm;

	/// <summary>Stored entries of L, diagonal included (one per column), from the symbolic analysis.</summary>
	public long NonZerosL => _n < 0 ? 0 : (long)_lColPtr[_n] + _n;

	/// <summary>
	/// Diagnostic for tests: list entries the AMD ordering visited in the last AnalyzePattern
	/// (0 for the natural ordering). Load-independent, unlike wall-clock time.
	/// </summary>
	internal long OrderingWork { get; private set; }

	/// <summary>
	/// Diagnostic for tests: inner-loop steps of the last numeric factorization, one per
	/// elimination-tree node visited while scattering a row and one per entry of L read while
	/// updating it. Load-independent, unlike wall-clock time.
	/// </summary>
	internal long FactorizationWork { get; private set; }

	/// <summary>Analyzes the pattern and factorizes (Eigen's compute()).</summary>
	public SimplicialCholesky Compute(SparseMatrixCsc a)
	{
		AnalyzePattern(a);
		Factorize(a);
		return this;
	}

	/// <summary>
	/// Orders the matrix and computes the structure of L. Only the pattern of the configured
	/// triangle of <paramref name="a"/> is used; values are ignored.
	/// </summary>
	public void AnalyzePattern(SparseMatrixCsc a)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"Sparse Cholesky needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;
		_n = n;
		_factorized = false;
		Info = ComputationInfo.NumericalIssue;
		_analyzed = a;
		_patternColPtr = a.ColPtr.ToArray();
		_patternRowIdx = a.RowIndices.ToArray();

		SparseMatrixCsc read = a.TriangularPart(_part);
		if (_ordering == SparseOrdering.Amd)
		{
			_perm = AmdOrdering.Compute(read, out long orderingWork);
			OrderingWork = orderingWork;
		}
		else
		{
			OrderingWork = 0;
			_perm = new int[n];
			for (int k = 0; k < n; k++)
			{
				_perm[k] = k;
			}
		}

		var inverse = new int[n];
		for (int k = 0; k < n; k++)
		{
			inverse[_perm[k]] = k;
		}

		BuildPermutedUpper(a, inverse);
		_parent = EliminationTree(n, _cColPtr, _cRowIdx);
		BuildStructureOfL();
		_cValues = new double[_cRowIdx.Length];
		_lValues = new double[_lRowIdx.Length];
		_diag = new double[n];
	}

	/// <summary>
	/// Numeric factorization of a matrix with the analyzed pattern. Returns false (and sets
	/// Info to NumericalIssue) on a pivot the factorization kind cannot accept.
	/// </summary>
	public bool Factorize(SparseMatrixCsc a)
	{
		if (_n < 0)
		{
			throw new InvalidOperationException("AnalyzePattern must run before Factorize.");
		}

		// A SparseMatrixCsc's pattern is immutable, so the very matrix that was analyzed (the
		// solver's per-iteration case) needs no O(nnz) pattern comparison.
		if (!ReferenceEquals(a, _analyzed)
			&& (a.Rows != _n || a.Cols != _n || !a.ColPtr.SequenceEqual(_patternColPtr)
				|| !a.RowIndices.SequenceEqual(_patternRowIdx)))
		{
			throw new ArgumentException("Factorize needs a matrix with the pattern given to AnalyzePattern.", nameof(a));
		}

		Array.Clear(_cValues);
		ReadOnlySpan<double> values = a.Values;
		for (int p = 0; p < values.Length; p++)
		{
			int slot = _valueMap[p];
			if (slot >= 0)
			{
				_cValues[slot] = values[p];
			}
		}

		FactorizationWork = 0;
		_factorized = _kind == SimplicialCholeskyKind.LLT ? FactorizeLlt() : FactorizeLdlt();
		Info = _factorized ? ComputationInfo.Success : ComputationInfo.NumericalIssue;
		return _factorized;
	}

	/// <summary>Solves A x = b with the current factorization.</summary>
	public VectorXd Solve(VectorXd b)
	{
		if (!_factorized)
		{
			throw new InvalidOperationException("Solve needs a successful factorization.");
		}

		if (b.Length != _n)
		{
			throw new ArgumentException($"Right-hand side has length {b.Length}, expected {_n}.", nameof(b));
		}

		var y = new double[_n];
		ReadOnlySpan<double> bs = b.AsSpan();
		for (int k = 0; k < _n; k++)
		{
			y[k] = bs[_perm[k]];
		}

		bool llt = _kind == SimplicialCholeskyKind.LLT;

		// Forward substitution with L (unit diagonal for LDLT), then D^-1 for LDLT.
		for (int j = 0; j < _n; j++)
		{
			if (llt)
			{
				y[j] /= _diag[j];
			}

			double yj = y[j];
			for (int p = _lColPtr[j]; p < _lColPtr[j + 1]; p++)
			{
				y[_lRowIdx[p]] -= _lValues[p] * yj;
			}
		}

		if (!llt)
		{
			for (int j = 0; j < _n; j++)
			{
				y[j] /= _diag[j];
			}
		}

		// Backward substitution with L^T.
		for (int j = _n - 1; j >= 0; j--)
		{
			double yj = y[j];
			for (int p = _lColPtr[j]; p < _lColPtr[j + 1]; p++)
			{
				yj -= _lValues[p] * y[_lRowIdx[p]];
			}

			y[j] = llt ? yj / _diag[j] : yj;
		}

		var x = new VectorXd(_n);
		Span<double> xs = x.AsSpan();
		for (int k = 0; k < _n; k++)
		{
			xs[_perm[k]] = y[k];
		}

		return x;
	}

	/// <summary>
	/// D of P A P^T = L D L^T (Eigen's SimplicialLDLT::vectorD()). LDLT only; needs a
	/// successful factorization.
	/// </summary>
	public VectorXd VectorD()
	{
		if (!_factorized || _kind != SimplicialCholeskyKind.LDLT)
		{
			throw new InvalidOperationException("VectorD needs a successful LDLT factorization.");
		}

		return new VectorXd(_diag);
	}

	/// <summary>
	/// The factor L of the permuted matrix (Eigen's matrixL()), diagonal stored: ones for
	/// LDLT, L's diagonal for LLT. Needs a successful factorization.
	/// </summary>
	public SparseMatrixCsc MatrixL()
	{
		if (!_factorized)
		{
			throw new InvalidOperationException("MatrixL needs a successful factorization.");
		}

		// Each column's strict-lower rows were appended in increasing row order by the
		// up-looking factorization, so putting the diagonal first keeps the rows sorted.
		var colPtr = new int[_n + 1];
		var rowIdx = new int[_lRowIdx.Length + _n];
		var values = new double[_lRowIdx.Length + _n];
		int q = 0;
		for (int j = 0; j < _n; j++)
		{
			rowIdx[q] = j;
			values[q++] = _kind == SimplicialCholeskyKind.LLT ? _diag[j] : 1.0;
			for (int p = _lColPtr[j]; p < _lColPtr[j + 1]; p++)
			{
				rowIdx[q] = _lRowIdx[p];
				values[q++] = _lValues[p];
			}

			colPtr[j + 1] = q;
		}

		return SparseMatrixCsc.FromCsc(_n, _n, colPtr, rowIdx, values);
	}

	/// <summary>
	/// Builds the upper triangle of C = P A P^T in CSC form from the configured triangle of A,
	/// and the map from A's entries to C's value slots.
	/// </summary>
	private void BuildPermutedUpper(SparseMatrixCsc a, int[] inverse)
	{
		ReadOnlySpan<int> colPtr = a.ColPtr;
		ReadOnlySpan<int> rowIdx = a.RowIndices;
		var count = new int[_n + 1];
		for (int j = 0; j < _n; j++)
		{
			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				int i = rowIdx[p];
				if (IsRead(i, j))
				{
					count[Math.Max(inverse[i], inverse[j]) + 1]++;
				}
			}
		}

		for (int k = 0; k < _n; k++)
		{
			count[k + 1] += count[k];
		}

		_cColPtr = (int[])count.Clone();
		_cRowIdx = new int[count[_n]];
		_valueMap = new int[a.NonZeros];
		for (int j = 0; j < _n; j++)
		{
			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				int i = rowIdx[p];
				if (!IsRead(i, j))
				{
					_valueMap[p] = -1;
					continue;
				}

				int pi = inverse[i], pj = inverse[j];
				int slot = count[Math.Max(pi, pj)]++;
				_cRowIdx[slot] = Math.Min(pi, pj);
				_valueMap[p] = slot;
			}
		}
	}

	private bool IsRead(int i, int j) => _part == SymmetricPart.Lower ? i >= j : i <= j;

	/// <summary>
	/// Elimination tree of a symmetric matrix given by its upper triangle (Liu 1990):
	/// parent[i] is the row index of the first off-diagonal nonzero in column i of L, or -1.
	/// </summary>
	private static int[] EliminationTree(int n, int[] colPtr, int[] rowIdx)
	{
		var parent = new int[n];
		var ancestor = new int[n];
		for (int k = 0; k < n; k++)
		{
			parent[k] = -1;
			ancestor[k] = -1;
			for (int p = colPtr[k]; p < colPtr[k + 1]; p++)
			{
				// Climb from i to the root of its current subtree, pointing every node on the
				// way at k (path compression); that root's parent is k.
				int r = rowIdx[p];
				while (r >= 0 && r < k)
				{
					int up = ancestor[r];
					ancestor[r] = k;
					if (up < 0)
					{
						parent[r] = k;
					}

					r = up;
				}
			}
		}

		return parent;
	}

	/// <summary>
	/// Column pointers of the strictly lower part of L: row k of L is the set of etree nodes
	/// reachable upward from the nonzeros of column k of C, stopping at k.
	/// </summary>
	private void BuildStructureOfL()
	{
		var counts = new int[_n + 1];
		var flag = new int[_n];
		Array.Fill(flag, -1);
		for (int k = 0; k < _n; k++)
		{
			flag[k] = k;
			for (int p = _cColPtr[k]; p < _cColPtr[k + 1]; p++)
			{
				for (int i = _cRowIdx[p]; flag[i] != k; i = _parent[i])
				{
					flag[i] = k;
					counts[i + 1]++;
				}
			}
		}

		for (int k = 0; k < _n; k++)
		{
			counts[k + 1] += counts[k];
		}

		_lColPtr = counts;
		_lRowIdx = new int[counts[_n]];
	}

	/// <summary>
	/// Reach of column k of C in the elimination tree, in topological order (every node
	/// before its ancestors), written to pattern[top..n). Also scatters the column into y.
	/// </summary>
	private int ScatterRow(int k, double[] y, int[] flag, int[] pattern, int[] stack)
	{
		int top = _n;
		flag[k] = k;
		for (int p = _cColPtr[k]; p < _cColPtr[k + 1]; p++)
		{
			int i = _cRowIdx[p];
			y[i] += _cValues[p];
			int len = 0;
			for (; flag[i] != k; i = _parent[i])
			{
				FactorizationWork++;
				stack[len++] = i;
				flag[i] = k;
			}

			while (len > 0)
			{
				pattern[--top] = stack[--len];
			}
		}

		return top;
	}

	/// <summary>Up-looking L D L^T: row k of L solves L(0:k,0:k) D y = C(0:k, k).</summary>
	private bool FactorizeLdlt()
	{
		var y = new double[_n];
		var flag = new int[_n];
		Array.Fill(flag, -1);
		var pattern = new int[_n];
		var stack = new int[_n];
		var fill = new int[_n];
		for (int k = 0; k < _n; k++)
		{
			int top = ScatterRow(k, y, flag, pattern, stack);
			double d = y[k];
			y[k] = 0.0;
			for (; top < _n; top++)
			{
				int j = pattern[top];
				double yj = y[j];
				y[j] = 0.0;
				int end = _lColPtr[j] + fill[j];
				FactorizationWork += fill[j];
				for (int p = _lColPtr[j]; p < end; p++)
				{
					y[_lRowIdx[p]] -= _lValues[p] * yj;
				}

				double lkj = yj / _diag[j];
				d -= lkj * yj;
				_lRowIdx[end] = k;
				_lValues[end] = lkj;
				fill[j]++;
			}

			if (d == 0.0)
			{
				return false;
			}

			_diag[k] = d;
		}

		return true;
	}

	/// <summary>Up-looking L L^T: row k of L solves L(0:k,0:k) l = C(0:k, k).</summary>
	private bool FactorizeLlt()
	{
		var y = new double[_n];
		var flag = new int[_n];
		Array.Fill(flag, -1);
		var pattern = new int[_n];
		var stack = new int[_n];
		var fill = new int[_n];
		for (int k = 0; k < _n; k++)
		{
			int top = ScatterRow(k, y, flag, pattern, stack);
			double d = y[k];
			y[k] = 0.0;
			for (; top < _n; top++)
			{
				int j = pattern[top];
				double lkj = y[j] / _diag[j];
				y[j] = 0.0;
				int end = _lColPtr[j] + fill[j];
				FactorizationWork += fill[j];
				for (int p = _lColPtr[j]; p < end; p++)
				{
					y[_lRowIdx[p]] -= _lValues[p] * lkj;
				}

				d -= lkj * lkj;
				_lRowIdx[end] = k;
				_lValues[end] = lkj;
				fill[j]++;
			}

			if (d <= 0.0)
			{
				return false;
			}

			_diag[k] = Math.Sqrt(d);
		}

		return true;
	}
}
