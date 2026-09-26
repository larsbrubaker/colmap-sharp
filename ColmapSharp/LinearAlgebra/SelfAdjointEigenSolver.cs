// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SelfAdjointEigenSolver: eigen-decomposition A = V diag(lambda) V^T of a real symmetric
// matrix, the replacement for Eigen::SelfAdjointEigenSolver. COLMAP uses it on the 4x4
// normal matrix in geometry/triangulation.cc (eigenvectors().col(0), the eigenvector of the
// smallest eigenvalue).
//
// Algorithm: cyclic Jacobi, Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm
// 8.5.3, with the symmetric Schur rotation of Algorithm 8.5.1. Rotations are accumulated
// into V, so the eigenvectors are orthonormal to working precision. Sweeps run until the
// off-diagonal Frobenius norm is at most epsilon times the matrix's Frobenius norm (or a
// sweep rotates nothing), on the input divided by its largest |entry| so the norms cannot
// under- or overflow; Info is NoConvergence if the sweep cap is reached first. Jacobi is chosen over tridiagonal QL for its high relative
// accuracy on the small matrices COLMAP decomposes. Written from G&VL; Eigen (MPL-2.0)
// is not ported.
//
// Semantics follow Eigen's documentation: only the lower triangle of the input is read;
// eigenvalues are returned in increasing order (ties keep their diagonal order); the
// eigenvector columns are normalized. Eigenvector signs are arbitrary, as in Eigen. Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Eigenvalues and eigenvectors of a real symmetric matrix by cyclic Jacobi. Replacement
/// for Eigen::SelfAdjointEigenSolver.
/// </summary>
public sealed class SelfAdjointEigenSolver
{
	private const int MaxSweeps = 100;

	private readonly double[] _eigenvalues;
	private readonly MatrixXd? _eigenvectors;

	/// <summary>
	/// Decomposes the symmetric matrix whose lower triangle is <paramref name="a"/>'s
	/// (not modified).
	/// </summary>
	public SelfAdjointEigenSolver(MatrixXd a, bool computeEigenvectors = true)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"Matrix must be square, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;
		var w = new MatrixXd(n, n);
		bool finite = true;
		for (int j = 0; j < n; j++)
		{
			for (int i = j; i < n; i++)
			{
				w[i, j] = a[i, j];
				w[j, i] = a[i, j];
				finite &= double.IsFinite(a[i, j]);
			}
		}

		_eigenvalues = new double[n];
		MatrixXd v = MatrixXd.Identity(n);
		if (!finite)
		{
			Array.Fill(_eigenvalues, double.NaN);
			Info = ComputationInfo.InvalidInput;
			return;
		}

		// Work on A / max|a_ij| so the squared norms below neither underflow (entries near
		// 1e-170) nor overflow (near 1e160); the eigenvalues are scaled back at the end.
		double scale = 0;
		foreach (double value in w.AsSpan())
		{
			scale = Math.Max(scale, Math.Abs(value));
		}

		if (scale == 0)
		{
			scale = 1;
		}

		Span<double> data = w.AsSpan();
		for (int i = 0; i < data.Length; i++)
		{
			data[i] /= scale;
		}

		Info = ComputationInfo.NoConvergence;
		double tolerance = LinearAlgebraConstants.MachineEpsilon * w.Norm();
		for (int sweep = 0; sweep < MaxSweeps; sweep++)
		{
			if (OffDiagonalNorm(w) <= tolerance)
			{
				Info = ComputationInfo.Success;
				break;
			}

			bool rotated = false;
			for (int p = 0; p < n - 1; p++)
			{
				for (int q = p + 1; q < n; q++)
				{
					if (w[p, q] != 0)
					{
						rotated = true;
						Rotate(w, v, p, q);
					}
				}
			}

			if (!rotated)
			{
				Info = ComputationInfo.Success;
				break;
			}
		}

		for (int i = 0; i < n; i++)
		{
			_eigenvalues[i] = w[i, i] * scale;
		}

		// Stable insertion sort into increasing order, moving the eigenvector columns.
		for (int i = 1; i < n; i++)
		{
			for (int k = i; k > 0 && _eigenvalues[k] < _eigenvalues[k - 1]; k--)
			{
				(_eigenvalues[k], _eigenvalues[k - 1]) = (_eigenvalues[k - 1], _eigenvalues[k]);
				JacobiSvdKernel.SwapColumns(v.AsSpan(), n, k, k - 1);
			}
		}

		_eigenvectors = computeEigenvectors ? v : null;
	}

	/// <summary>Success, or InvalidInput when the lower triangle had a non-finite entry.</summary>
	public ComputationInfo Info { get; }

	/// <summary>The eigenvalues in increasing order. A copy.</summary>
	public VectorXd Eigenvalues() => new(_eigenvalues);

	/// <summary>The normalized eigenvectors as columns, in eigenvalue order. A copy.</summary>
	public MatrixXd Eigenvectors() =>
		(_eigenvectors ?? throw new InvalidOperationException("Eigenvectors were not requested.")).Clone();

	private static double OffDiagonalNorm(MatrixXd w)
	{
		double sum = 0;
		for (int j = 0; j < w.Cols; j++)
		{
			for (int i = 0; i < w.Rows; i++)
			{
				if (i != j)
				{
					sum += w[i, j] * w[i, j];
				}
			}
		}

		return Math.Sqrt(sum);
	}

	/// <summary>
	/// G&amp;VL Algorithm 8.5.1: the rotation J = [c s; -s c] in the (p, q) plane with
	/// (J^T W J)(p, q) = 0, applied as W &lt;- J^T W J and V &lt;- V J.
	/// </summary>
	private static void Rotate(MatrixXd w, MatrixXd v, int p, int q)
	{
		double tau = (w[q, q] - w[p, p]) / (2 * w[p, q]);
		double t = (tau >= 0 ? 1.0 : -1.0) / (Math.Abs(tau) + Math.Sqrt(1 + tau * tau));
		double c = 1 / Math.Sqrt(1 + t * t);
		double s = t * c;
		int n = w.Rows;
		for (int k = 0; k < n; k++)
		{
			double x = w[k, p];
			double y = w[k, q];
			w[k, p] = c * x - s * y;
			w[k, q] = s * x + c * y;
		}

		for (int k = 0; k < n; k++)
		{
			double x = w[p, k];
			double y = w[q, k];
			w[p, k] = c * x - s * y;
			w[q, k] = s * x + c * y;
		}

		w[p, q] = 0;
		w[q, p] = 0;
		for (int k = 0; k < n; k++)
		{
			double x = v[k, p];
			double y = v[k, q];
			v[k, p] = c * x - s * y;
			v[k, q] = s * x + c * y;
		}
	}
}
