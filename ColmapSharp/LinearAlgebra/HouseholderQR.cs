// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HouseholderQR: A = Q R by Householder reflections, the replacement for
// Eigen::HouseholderQR / matrix.householderQr(). COLMAP uses householderQ() to get an
// orthonormal null-space basis (the 8-point fundamental and essential solvers,
// geometry/pose.cc GravityAlignedRotation) and math/matrix.h's DecomposeMatrixRQ builds on
// it (ColmapSharp/Mathematics/MatrixUtils.cs).
//
// Algorithm: unblocked Householder QR, Golub & Van Loan, "Matrix Computations", 4th ed.,
// Algorithm 5.2.1, with the reflector convention in Householder.cs (LAPACK dgeqr2's).
// Written from the book; Eigen (MPL-2.0) is not ported. Any shape is accepted; min(m, n)
// reflectors are formed, the last one of a square matrix being the identity (tau = 0).
// Tier B: Eigen switches to a blocked update for large matrices, so results agree within
// rounding; the signs of Q's columns and R's rows follow the documented convention and are
// pinned against numpy (LAPACK) in DecompositionOracleTests.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Householder QR, A = Q R. Replacement for Eigen::HouseholderQR&lt;MatrixXd&gt;.
/// </summary>
public sealed class HouseholderQR
{
	private readonly MatrixXd _qr;
	private readonly double[] _hCoeffs;

	/// <summary>Factorizes a matrix of any shape. The input is not modified.</summary>
	public HouseholderQR(MatrixXd a)
	{
		_qr = a.Clone();
		_hCoeffs = new double[Math.Min(a.Rows, a.Cols)];
		Householder.FactorInPlace(_qr.AsSpan(), a.Rows, a.Cols, _hCoeffs);
	}

	/// <summary>
	/// The packed factor (Eigen's matrixQR()): R on and above the diagonal, the essential
	/// parts of the reflectors below it. A copy.
	/// </summary>
	public MatrixXd MatrixQR() => _qr.Clone();

	/// <summary>The reflector coefficients tau (Eigen's hCoeffs()). A copy.</summary>
	public VectorXd HCoeffs() => new(_hCoeffs);

	/// <summary>The upper-triangular (trapezoidal) factor R, m x n.</summary>
	public MatrixXd MatrixR() => Householder.UpperPart(_qr);

	/// <summary>The full orthogonal factor Q (m x m), Eigen's householderQ() evaluated.</summary>
	public MatrixXd HouseholderQ() => Householder.AccumulateQ(_qr, _hCoeffs);

	/// <summary>
	/// Least-squares solution of A x = b for a matrix with at least as many rows as
	/// columns and full column rank (the exact solution when A is square).
	/// </summary>
	public VectorXd Solve(VectorXd b)
	{
		if (_qr.Rows < _qr.Cols)
		{
			throw new InvalidOperationException("HouseholderQR.Solve needs rows >= cols.");
		}

		if (b.Length != _qr.Rows)
		{
			throw new ArgumentException($"Right-hand side has {b.Length} rows, expected {_qr.Rows}.", nameof(b));
		}

		var y = b.Clone();
		Householder.ApplyQTranspose(_qr, _hCoeffs, y.AsSpan());
		Householder.BackSubstitute(_qr, _qr.Cols, y.AsSpan());
		return y.Head(_qr.Cols);
	}
}
