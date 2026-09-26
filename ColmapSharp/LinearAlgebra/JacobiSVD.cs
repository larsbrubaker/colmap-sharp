// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// JacobiSVD: singular value decomposition A = U diag(s) V^T of a matrix of any shape, the
// replacement for Eigen::JacobiSVD<MatrixXd> and the fixed/partly-dynamic shapes COLMAP
// uses (Dynamic x 9 and x 6 for the DLT systems, 6x3..6x5 and 12x12 in EPnP, the 4 x N
// quaternion average in geometry/pose.cc). 3x3 and 4x4 callers in hot paths use the
// allocation-free Svd3d / Svd4d (SvdFixed.cs) instead; both run JacobiSvdKernel.
//
// Rectangular input is reduced first (Golub & Van Loan, "Matrix Computations", 4th ed.,
// §8.6.3, the R-SVD idea): for rows > cols, A P = Q R with ColPivHouseholderQR, the kernel
// decomposes the square top of R = Ur S Vr^T, U = Q blockdiag(Ur, I) and V = P Vr, so the
// full U falls out of Q without any basis completion. The column pivoting is Eigen's
// documented default preconditioner (ColPivHouseholderQRPreconditioner) and it matters:
// it makes R graded (decreasing diagonal), which two-sided Jacobi resolves to high
// relative accuracy (Demmel and Veselić 1992, cited in JacobiSvdKernel.cs). Unpivoted QR
// lost the null vector of badly column-scaled DLT systems (homography_matrix_test's
// NumericalStability, pixel coordinates of 1e6 next to a column of ones). Q is never formed: the stored reflectors are
// applied to just the requested columns of blockdiag(Ur, I), so a thin U of an N x 3
// system costs O(N) memory, not O(N^2). rows < cols decomposes A^T and
// swaps the factors. Written from those sources; Eigen (MPL-2.0) is not ported.
//
// Semantics follow Eigen's documentation: singular values non-negative and decreasing;
// U and V computed only when requested, thin (min(rows, cols) columns) or full; rank()
// and solve() use the threshold max(1, diagSize) * epsilon relative to the largest singular
// value; solve() returns the minimum-norm least-squares solution with the singular values
// at or below the threshold treated as zero. Singular vector signs are arbitrary (see
// JacobiSvdKernel.cs). Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>Which singular vectors JacobiSVD computes (Eigen's DecompositionOptions).</summary>
[Flags]
public enum SvdOptions
{
	/// <summary>Singular values only.</summary>
	None = 0,

	/// <summary>The first min(rows, cols) columns of U.</summary>
	ComputeThinU = 1,

	/// <summary>All rows x rows of U.</summary>
	ComputeFullU = 2,

	/// <summary>The first min(rows, cols) columns of V.</summary>
	ComputeThinV = 4,

	/// <summary>All cols x cols of V.</summary>
	ComputeFullV = 8,
}

/// <summary>
/// Singular value decomposition A = U diag(s) V^T by two-sided Jacobi. Replacement for
/// Eigen::JacobiSVD.
/// </summary>
public sealed class JacobiSVD
{
	private readonly MatrixXd? _u;
	private readonly MatrixXd? _v;
	private readonly double[] _singularValues;

	/// <summary>Decomposes <paramref name="a"/> (not modified).</summary>
	public JacobiSVD(MatrixXd a, SvdOptions options = SvdOptions.None)
	{
		if ((options & SvdOptions.ComputeThinU) != 0 && (options & SvdOptions.ComputeFullU) != 0
			|| (options & SvdOptions.ComputeThinV) != 0 && (options & SvdOptions.ComputeFullV) != 0)
		{
			throw new ArgumentException("Request either the thin or the full factor, not both.", nameof(options));
		}

		Rows = a.Rows;
		Cols = a.Cols;
		if (a.Rows < a.Cols)
		{
			var transposed = new JacobiSVD(a.Transpose(), SwapUV(options));
			_u = transposed._v;
			_v = transposed._u;
			_singularValues = transposed._singularValues;
			Info = transposed.Info;
			Sweeps = transposed.Sweeps;
			return;
		}

		int n = a.Cols;
		int m = a.Rows;
		bool wantU = (options & (SvdOptions.ComputeThinU | SvdOptions.ComputeFullU)) != 0;
		bool wantV = (options & (SvdOptions.ComputeThinV | SvdOptions.ComputeFullV)) != 0;
		_singularValues = new double[n];

		ColPivHouseholderQR? qr = null;
		MatrixXd square;
		if (m > n)
		{
			qr = new ColPivHouseholderQR(a);
			square = qr.MatrixR().TopRows(n);
		}
		else
		{
			square = a.Clone();
		}

		var ur = wantU ? new MatrixXd(n, n) : null;
		var v = wantV ? new MatrixXd(n, n) : null;
		Info = JacobiSvdKernel.Decompose(
			square.AsSpan(), n, ur is null ? default : ur.AsSpan(), v is null ? default : v.AsSpan(), _singularValues, out int sweeps);
		Sweeps = sweeps;
		if (v is not null && qr is not null)
		{
			// V = P Vr: row j of Vr belongs to column ColsPermutationIndices()[j] of A.
			int[] permutation = qr.ColsPermutationIndices();
			var permuted = new MatrixXd(n, n);
			for (int j = 0; j < n; j++)
			{
				for (int c = 0; c < n; c++)
				{
					permuted[permutation[j], c] = v[j, c];
				}
			}

			v = permuted;
		}

		_v = v;

		if (ur is not null)
		{
			int uCols = (options & SvdOptions.ComputeFullU) != 0 ? m : n;
			if (qr is null)
			{
				_u = ur;
			}
			else
			{
				// U = Q blockdiag(Ur, I). Column c is Q applied to [Ur(:, c); 0] for c < n
				// and to the unit vector e_c after that, so the reflectors are applied to
				// just the requested columns and the thin U never forms the m x m Q.
				MatrixXd packed = qr.MatrixQR();
				double[] tau = qr.HCoeffs().AsSpan().ToArray();
				var u = new MatrixXd(m, uCols);
				for (int c = 0; c < uCols; c++)
				{
					Span<double> column = u.ColumnSpan(c);
					if (c < n)
					{
						ur.ColumnSpan(c).CopyTo(column);
					}
					else
					{
						column[c] = 1;
					}

					Householder.ApplyQ(packed, tau, column);
				}

				_u = u;
			}
		}
	}

	/// <summary>Rows of the decomposed matrix.</summary>
	public int Rows { get; }

	/// <summary>Columns of the decomposed matrix.</summary>
	public int Cols { get; }

	/// <summary>
	/// Jacobi sweeps the decomposition ran, the last of which rotated nothing (0 for a zero
	/// or non-finite matrix). A diagnostic for convergence tests.
	/// </summary>
	internal int Sweeps { get; }

	/// <summary>Success, NoConvergence, or InvalidInput when the matrix had a non-finite entry.</summary>
	public ComputationInfo Info { get; }

	/// <summary>The min(rows, cols) singular values, non-negative and decreasing. A copy.</summary>
	public VectorXd SingularValues() => new(_singularValues);

	/// <summary>U (rows x rows when full, rows x min(rows, cols) when thin). A copy.</summary>
	public MatrixXd MatrixU() => (_u ?? throw new InvalidOperationException("U was not requested (SvdOptions).")).Clone();

	/// <summary>V (cols x cols when full, cols x min(rows, cols) when thin). A copy.</summary>
	public MatrixXd MatrixV() => (_v ?? throw new InvalidOperationException("V was not requested (SvdOptions).")).Clone();

	/// <summary>
	/// Numerical rank: the number of singular values strictly greater than
	/// max(1, min(rows, cols)) * epsilon * the largest singular value.
	/// </summary>
	public int Rank() => JacobiSvdKernel.Rank(_singularValues);

	/// <summary>
	/// Minimum-norm least-squares solution of A x = b; singular values at or below the rank
	/// threshold are treated as zero. Needs U and V (thin or full).
	/// </summary>
	public VectorXd Solve(VectorXd b)
	{
		if (_u is null || _v is null)
		{
			throw new InvalidOperationException("Solve needs U and V (SvdOptions).");
		}

		if (b.Length != Rows)
		{
			throw new ArgumentException($"Right-hand side has {b.Length} rows, expected {Rows}.", nameof(b));
		}

		int rank = Rank();
		var x = new VectorXd(Cols);
		for (int i = 0; i < rank; i++)
		{
			double coefficient = 0;
			for (int r = 0; r < Rows; r++)
			{
				coefficient += _u[r, i] * b[r];
			}

			coefficient /= _singularValues[i];
			for (int c = 0; c < Cols; c++)
			{
				x[c] += _v[c, i] * coefficient;
			}
		}

		return x;
	}

	private static SvdOptions SwapUV(SvdOptions options)
	{
		SvdOptions swapped = SvdOptions.None;
		if ((options & SvdOptions.ComputeThinU) != 0)
		{
			swapped |= SvdOptions.ComputeThinV;
		}

		if ((options & SvdOptions.ComputeFullU) != 0)
		{
			swapped |= SvdOptions.ComputeFullV;
		}

		if ((options & SvdOptions.ComputeThinV) != 0)
		{
			swapped |= SvdOptions.ComputeThinU;
		}

		if ((options & SvdOptions.ComputeFullV) != 0)
		{
			swapped |= SvdOptions.ComputeFullU;
		}

		return swapped;
	}
}
