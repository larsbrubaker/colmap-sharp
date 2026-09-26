// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColPivHouseholderQR: rank-revealing QR with column pivoting, A P = Q R, the replacement
// for Eigen::ColPivHouseholderQR / matrix.colPivHouseholderQr(). COLMAP uses rank() (the
// P3P/EPnP degeneracy check in estimators/solvers/absolute_pose.cc, the fixed-point
// gauge check in bundle_adjustment_ceres.cc) and solve() (absolute_pose.cc's 4-vector
// least-squares step).
//
// Algorithm: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 5.4.1
// (Householder QR with column pivoting: at step j the remaining column of largest norm is
// swapped in), with the reflector convention in Householder.cs. The partial column norms
// are downdated after each step and recomputed when cancellation makes the downdate
// unreliable, following Drmač and Bujanović, "On the failure of rank-revealing QR
// factorization software - a case study", LAPACK Working Note 176 (2008), the same scheme
// LAPACK's dlaqp2 documents. Written from those sources; Eigen (MPL-2.0) is not ported.
// Ties pick the first column of largest norm.
//
// Rank: Eigen's documented default. A pivot |R(i,i)| counts as nonzero when it is strictly
// greater than threshold * maxPivot, with threshold = min(rows, cols) * machine epsilon and
// maxPivot the largest |R(i,i)|. Solve returns the basic solution: the components past the
// rank are set to zero, then undone through the permutation. Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Householder QR with column pivoting, A P = Q R. Replacement for
/// Eigen::ColPivHouseholderQR&lt;MatrixXd&gt;.
/// </summary>
public sealed class ColPivHouseholderQR
{
	private readonly MatrixXd _qr;
	private readonly double[] _hCoeffs;

	// Column j of A P is column _permutation[j] of A.
	private readonly int[] _permutation;
	private readonly double _maxPivot;

	/// <summary>Factorizes a matrix of any shape. The input is not modified.</summary>
	public ColPivHouseholderQR(MatrixXd a)
	{
		_qr = a.Clone();
		int m = a.Rows;
		int n = a.Cols;
		int k = Math.Min(m, n);
		_hCoeffs = new double[k];
		_permutation = new int[n];

		// partialNorms: current norm of column c's rows j..m-1; referenceNorms: the norm at
		// the last full recomputation (LAWN 176's vn1 / vn2).
		var partialNorms = new double[n];
		var referenceNorms = new double[n];
		for (int c = 0; c < n; c++)
		{
			_permutation[c] = c;
			partialNorms[c] = ColumnNorm(_qr.ColumnSpan(c));
			referenceNorms[c] = partialNorms[c];
		}

		double tolerance = Math.Sqrt(LinearAlgebraConstants.MachineEpsilon);
		for (int j = 0; j < k; j++)
		{
			int pivot = j;
			for (int c = j + 1; c < n; c++)
			{
				if (partialNorms[c] > partialNorms[pivot])
				{
					pivot = c;
				}
			}

			if (pivot != j)
			{
				Span<double> left = _qr.ColumnSpan(j);
				Span<double> right = _qr.ColumnSpan(pivot);
				for (int r = 0; r < m; r++)
				{
					(left[r], right[r]) = (right[r], left[r]);
				}

				(_permutation[j], _permutation[pivot]) = (_permutation[pivot], _permutation[j]);
				(partialNorms[j], partialNorms[pivot]) = (partialNorms[pivot], partialNorms[j]);
				(referenceNorms[j], referenceNorms[pivot]) = (referenceNorms[pivot], referenceNorms[j]);
			}

			Span<double> column = _qr.ColumnSpan(j)[j..];
			double tau = Householder.MakeInPlace(column);
			_hCoeffs[j] = tau;
			_maxPivot = Math.Max(_maxPivot, Math.Abs(column[0]));
			ReadOnlySpan<double> essential = column[1..];
			for (int c = j + 1; c < n; c++)
			{
				Span<double> target = _qr.ColumnSpan(c);
				Householder.ApplyLeft(essential, tau, target[j..]);

				// Downdate the norm of rows j+1.. by removing R(j, c).
				if (partialNorms[c] != 0)
				{
					double ratio = Math.Abs(target[j]) / partialNorms[c];
					double temp = Math.Max(0, 1 - ratio * ratio);
					double normRatio = partialNorms[c] / referenceNorms[c];
					if (temp * normRatio * normRatio <= tolerance)
					{
						partialNorms[c] = ColumnNorm(target[(j + 1)..]);
						referenceNorms[c] = partialNorms[c];
					}
					else
					{
						partialNorms[c] *= Math.Sqrt(temp);
					}
				}
			}
		}
	}

	/// <summary>The packed factor (Eigen's matrixQR()). A copy.</summary>
	public MatrixXd MatrixQR() => _qr.Clone();

	/// <summary>The reflector coefficients tau (Eigen's hCoeffs()). A copy.</summary>
	public VectorXd HCoeffs() => new(_hCoeffs);

	/// <summary>The upper-triangular (trapezoidal) factor R, m x n.</summary>
	public MatrixXd MatrixR() => Householder.UpperPart(_qr);

	/// <summary>The full orthogonal factor Q (m x m).</summary>
	public MatrixXd HouseholderQ() => Householder.AccumulateQ(_qr, _hCoeffs);

	/// <summary>Column j of A P is column ColsPermutationIndices()[j] of A. A copy.</summary>
	public int[] ColsPermutationIndices() => (int[])_permutation.Clone();

	/// <summary>The permutation matrix P with A P = Q R (Eigen's colsPermutation()).</summary>
	public MatrixXd ColsPermutation()
	{
		int n = _permutation.Length;
		var p = new MatrixXd(n, n);
		for (int j = 0; j < n; j++)
		{
			p[_permutation[j], j] = 1;
		}

		return p;
	}

	/// <summary>
	/// Numerical rank: the number of |R(i,i)| strictly greater than
	/// min(rows, cols) * epsilon * the largest |R(i,i)|.
	/// </summary>
	public int Rank()
	{
		double threshold = _hCoeffs.Length * LinearAlgebraConstants.MachineEpsilon * _maxPivot;
		int rank = 0;
		for (int i = 0; i < _hCoeffs.Length; i++)
		{
			if (Math.Abs(_qr[i, i]) > threshold)
			{
				rank++;
			}
		}

		return rank;
	}

	/// <summary>
	/// Basic least-squares solution of A x = b: solves the leading rank x rank triangle of R
	/// and sets the remaining components to zero.
	/// </summary>
	public VectorXd Solve(VectorXd b)
	{
		if (b.Length != _qr.Rows)
		{
			throw new ArgumentException($"Right-hand side has {b.Length} rows, expected {_qr.Rows}.", nameof(b));
		}

		int rank = Rank();
		var y = b.Clone();
		Householder.ApplyQTranspose(_qr, _hCoeffs, y.AsSpan());
		Householder.BackSubstitute(_qr, rank, y.AsSpan());
		var x = new VectorXd(_qr.Cols);
		for (int i = 0; i < rank; i++)
		{
			x[_permutation[i]] = y[i];
		}

		return x;
	}

	private static double ColumnNorm(ReadOnlySpan<double> column) => Math.Sqrt(VectorXd.Dot(column, column));
}
