// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// EigenSolver: eigenvalues (and optionally eigenvectors) of a general real square matrix,
// the replacement for Eigen::EigenSolver. COLMAP uses it for polynomial roots
// (math/polynomial.cc, the companion matrix, eigenvalues only) and for the 4x4 in
// estimators/solvers/generalized_relative_pose.cc (eigenvalues and complex eigenvectors).
//
// Algorithm (Golub & Van Loan, "Matrix Computations", 4th ed.):
// 1. Householder reduction to upper Hessenberg form H = Z^T A Z (Algorithm 7.4.2).
// 2. Real Schur form T = Z^T A Z by the shifted Francis double-shift QR step (Algorithm
//    7.5.1) with the deflation of §7.5.1 (a subdiagonal entry is negligible when it is at
//    most epsilon times the sum of its two diagonal neighbors) and Wilkinson's ad hoc
//    exceptional shift after 10 and 20 iterations without deflation (Wilkinson and
//    Reinsch, "Handbook for Automatic Computation II", 1971, procedure hqr). A deflated
//    2x2 block with real eigenvalues is split by a rotation (its first column rotated onto
//    an eigenvector), so T is quasi-triangular with 1x1 real blocks and 2x2 blocks for
//    complex pairs only (EigenSolver.Vectors.cs has the eigenvector back-substitution).
// Written from those sources; Eigen (MPL-2.0) is not ported.
//
// Semantics follow Eigen's documentation: eigenvalues come in the order of the diagonal
// blocks of T (not sorted); a complex pair appears as (re + i im, re - i im) with im > 0
// first; eigenvectors are the columns of a complex matrix, each normalized to unit norm.
// The eigenvector of a real eigenvalue is real; the complex phase of a complex eigenvector
// (and every sign) is this implementation's, like Eigen's arbitrary. Info is NoConvergence
// when the QR iteration exceeds 30 iterations per eigenvalue. The input is divided by its
// largest |entry| first and the eigenvalues multiplied back. Tier B.
//
// Complex-eigenvector phase: generalized_relative_pose.cc (GR6P/GR8P) reads V.real() of
// every eigenvector of its 4x4, including those of complex eigenvalues, whose real part
// depends on the phase. Ours is fixed by the back-substitution in EigenSolver.Vectors.cs
// (the block's eigenvector starts as (b, lambda - a), then the vector is scaled to unit
// norm without rotating its phase), which need not be Eigen's. Real eigenvalues are
// unaffected (their vectors are real, and hnormalized() removes the scale and sign). A
// divergence entry belongs with the GR6P port if its outputs for complex eigenvalues matter.

using System.Numerics;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Eigenvalues and eigenvectors of a general real matrix via Hessenberg reduction and the
/// Francis double-shift QR algorithm. Replacement for Eigen::EigenSolver.
/// </summary>
public sealed partial class EigenSolver
{
	private const int MaxIterationsPerEigenvalue = 30;

	private readonly Complex[] _eigenvalues;
	private readonly Complex[,]? _eigenvectors;

	/// <summary>Decomposes the square matrix <paramref name="a"/> (not modified).</summary>
	public EigenSolver(MatrixXd a, bool computeEigenvectors = true)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"Matrix must be square, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;
		_eigenvalues = new Complex[n];
		foreach (double value in a.AsSpan())
		{
			if (!double.IsFinite(value))
			{
				Array.Fill(_eigenvalues, new Complex(double.NaN, double.NaN));
				Info = ComputationInfo.InvalidInput;
				return;
			}
		}

		// Work on A / max|a_ij| so the Francis step's products cannot overflow (or
		// underflow); eigenvalues scale back, eigenvectors are unaffected.
		MatrixXd t = a.Clone();
		double scale = 0;
		foreach (double value in t.AsSpan())
		{
			scale = Math.Max(scale, Math.Abs(value));
		}

		if (scale == 0)
		{
			scale = 1;
		}

		Span<double> data = t.AsSpan();
		for (int i = 0; i < data.Length; i++)
		{
			data[i] /= scale;
		}

		MatrixXd z = MatrixXd.Identity(n);
		ReduceToHessenberg(t, z);
		if (!ComputeRealSchur(t, z))
		{
			Array.Fill(_eigenvalues, new Complex(double.NaN, double.NaN));
			Info = ComputationInfo.NoConvergence;
			return;
		}

		Info = ComputationInfo.Success;
		for (int i = 0; i < n; i++)
		{
			if (i == n - 1 || t[i + 1, i] == 0)
			{
				_eigenvalues[i] = new Complex(t[i, i], 0);
			}
			else
			{
				(double re, double im) = ComplexBlockEigenvalue(t, i);
				_eigenvalues[i] = new Complex(re, im);
				_eigenvalues[i + 1] = new Complex(re, -im);
				i++;
			}
		}

		if (computeEigenvectors)
		{
			_eigenvectors = ComputeEigenvectors(t, z, _eigenvalues);
		}

		for (int i = 0; i < n; i++)
		{
			_eigenvalues[i] *= scale;
		}
	}

	/// <summary>Success, InvalidInput (non-finite entry) or NoConvergence.</summary>
	public ComputationInfo Info { get; }

	/// <summary>The eigenvalues in Schur-block order. A copy.</summary>
	public Complex[] Eigenvalues() => (Complex[])_eigenvalues.Clone();

	/// <summary>The unit-norm eigenvectors as columns, [row, column]. A copy.</summary>
	public Complex[,] Eigenvectors() =>
		(Complex[,])(_eigenvectors ?? throw new InvalidOperationException("Eigenvectors were not requested.")).Clone();

	/// <summary>
	/// The eigenvalue re + i im (im &gt; 0) of the complex 2x2 block at (i, i): with
	/// p = (a - d) / 2, the eigenvalues are d + p +- sqrt(p^2 + b c).
	/// </summary>
	private static (double Re, double Im) ComplexBlockEigenvalue(MatrixXd t, int i)
	{
		double p = 0.5 * (t[i, i] - t[i + 1, i + 1]);
		double q = p * p + t[i, i + 1] * t[i + 1, i];
		return (t[i + 1, i + 1] + p, Math.Sqrt(Math.Abs(q)));
	}

	/// <summary>G&amp;VL Algorithm 7.4.2: H = Z^T A Z upper Hessenberg, Z accumulated.</summary>
	private static void ReduceToHessenberg(MatrixXd h, MatrixXd z)
	{
		int n = h.Rows;
		Span<double> x = n > 2 ? new double[n] : default;
		for (int k = 0; k < n - 2; k++)
		{
			int length = n - k - 1;
			Span<double> v = x[..length];
			for (int i = 0; i < length; i++)
			{
				v[i] = h[k + 1 + i, k];
			}

			double tau = Householder.MakeInPlace(v);
			if (tau == 0)
			{
				continue;
			}

			double beta = v[0];
			v[0] = 1;
			ApplyReflectorLeft(h, v, tau, k + 1, k);
			ApplyReflectorRight(h, v, tau, k + 1, 0, n);
			ApplyReflectorRight(z, v, tau, k + 1, 0, n);
			h[k + 1, k] = beta;
			for (int i = k + 2; i < n; i++)
			{
				h[i, k] = 0;
			}
		}
	}

	/// <summary>(I - tau v v^T) applied to rows first.. of columns firstCol..n-1.</summary>
	private static void ApplyReflectorLeft(MatrixXd m, ReadOnlySpan<double> v, double tau, int first, int firstCol)
	{
		for (int c = firstCol; c < m.Cols; c++)
		{
			double dot = 0;
			for (int i = 0; i < v.Length; i++)
			{
				dot += v[i] * m[first + i, c];
			}

			dot *= tau;
			for (int i = 0; i < v.Length; i++)
			{
				m[first + i, c] -= dot * v[i];
			}
		}
	}

	/// <summary>(I - tau v v^T) applied from the right to columns first.. of rows rowStart..rowEnd-1.</summary>
	private static void ApplyReflectorRight(MatrixXd m, ReadOnlySpan<double> v, double tau, int first, int rowStart, int rowEnd)
	{
		for (int r = rowStart; r < rowEnd; r++)
		{
			double dot = 0;
			for (int i = 0; i < v.Length; i++)
			{
				dot += m[r, first + i] * v[i];
			}

			dot *= tau;
			for (int i = 0; i < v.Length; i++)
			{
				m[r, first + i] -= dot * v[i];
			}
		}
	}

	/// <summary>
	/// Francis QR iteration on the Hessenberg matrix t, accumulating into z, until t is
	/// quasi-triangular with standardized real 2x2 blocks split. False on no convergence.
	/// </summary>
	private static bool ComputeRealSchur(MatrixXd t, MatrixXd z)
	{
		int n = t.Rows;
		double norm = t.Norm();
		int hi = n - 1;
		int iterations = 0;
		int totalIterations = 0;
		Span<double> v = stackalloc double[3];
		while (hi >= 0)
		{
			int lo = hi;
			while (lo > 0)
			{
				double s = Math.Abs(t[lo - 1, lo - 1]) + Math.Abs(t[lo, lo]);
				if (s == 0)
				{
					s = norm;
				}

				if (Math.Abs(t[lo, lo - 1]) <= LinearAlgebraConstants.MachineEpsilon * s)
				{
					t[lo, lo - 1] = 0;
					break;
				}

				lo--;
			}

			if (lo == hi)
			{
				hi--;
				iterations = 0;
			}
			else if (lo == hi - 1)
			{
				SplitRealBlock(t, z, lo);
				hi -= 2;
				iterations = 0;
			}
			else
			{
				iterations++;
				totalIterations++;
				if (totalIterations > MaxIterationsPerEigenvalue * n)
				{
					return false;
				}

				FrancisStep(t, z, lo, hi, iterations, v);
			}
		}

		return true;
	}

	/// <summary>
	/// If the 2x2 block at (m, m) has real eigenvalues, rotates it to upper triangular form:
	/// the rotation's first column is the eigenvector (lambda - d, c) of the eigenvalue
	/// lambda = d + p + sign(p) sqrt(p^2 + b c), the one farther from d.
	/// </summary>
	private static void SplitRealBlock(MatrixXd t, MatrixXd z, int m)
	{
		double p = 0.5 * (t[m, m] - t[m + 1, m + 1]);
		double q = p * p + t[m, m + 1] * t[m + 1, m];
		if (q < 0)
		{
			return;
		}

		double root = Math.Sqrt(q);
		double shifted = p >= 0 ? p + root : p - root;
		double c0 = shifted;
		double c1 = t[m + 1, m];
		double length = Math.Sqrt(c0 * c0 + c1 * c1);
		if (length == 0)
		{
			return;
		}

		double c = c0 / length;
		double s = c1 / length;
		int n = t.Rows;

		// T <- G^T T G, Z <- Z G with G = [c -s; s c].
		for (int j = m; j < n; j++)
		{
			double x = t[m, j];
			double y = t[m + 1, j];
			t[m, j] = c * x + s * y;
			t[m + 1, j] = -s * x + c * y;
		}

		for (int i = 0; i <= m + 1; i++)
		{
			double x = t[i, m];
			double y = t[i, m + 1];
			t[i, m] = c * x + s * y;
			t[i, m + 1] = -s * x + c * y;
		}

		for (int i = 0; i < n; i++)
		{
			double x = z[i, m];
			double y = z[i, m + 1];
			z[i, m] = c * x + s * y;
			z[i, m + 1] = -s * x + c * y;
		}

		t[m + 1, m] = 0;
	}

	/// <summary>
	/// G&amp;VL Algorithm 7.5.1 on the unreduced window lo..hi (at least 3 x 3): one implicit
	/// double-shift QR step by bulge chasing with 3-element Householder reflectors.
	/// </summary>
	private static void FrancisStep(MatrixXd t, MatrixXd z, int lo, int hi, int iterations, Span<double> v)
	{
		int n = t.Rows;
		double shiftSum;
		double shiftProduct;
		if (iterations == 10 || iterations == 20)
		{
			// Wilkinson's exceptional shift breaks cycles of the standard shift.
			double s = Math.Abs(t[hi, hi - 1]) + Math.Abs(t[hi - 1, hi - 2]);
			shiftSum = 1.5 * s;
			shiftProduct = s * s;
		}
		else
		{
			shiftSum = t[hi - 1, hi - 1] + t[hi, hi];
			shiftProduct = t[hi - 1, hi - 1] * t[hi, hi] - t[hi - 1, hi] * t[hi, hi - 1];
		}

		double x = t[lo, lo] * t[lo, lo] + t[lo, lo + 1] * t[lo + 1, lo] - shiftSum * t[lo, lo] + shiftProduct;
		double y = t[lo + 1, lo] * (t[lo, lo] + t[lo + 1, lo + 1] - shiftSum);
		double w = t[lo + 1, lo] * t[lo + 2, lo + 1];
		for (int k = lo; k <= hi - 1; k++)
		{
			int length = k < hi - 1 ? 3 : 2;
			Span<double> reflector = v[..length];
			reflector[0] = x;
			reflector[1] = y;
			if (length == 3)
			{
				reflector[2] = w;
			}

			double tau = Householder.MakeInPlace(reflector);
			if (tau != 0)
			{
				double beta = reflector[0];
				reflector[0] = 1;
				int firstCol = Math.Max(lo, k - 1);
				ApplyReflectorLeft(t, reflector, tau, k, firstCol);
				ApplyReflectorRight(t, reflector, tau, k, 0, Math.Min(k + 3, hi) + 1);
				ApplyReflectorRight(z, reflector, tau, k, 0, n);
				if (k > lo)
				{
					t[k, k - 1] = beta;
					t[k + 1, k - 1] = 0;
					if (length == 3)
					{
						t[k + 2, k - 1] = 0;
					}
				}
			}

			if (k < hi - 1)
			{
				x = t[k + 1, k];
				y = t[k + 2, k];
				if (k < hi - 2)
				{
					w = t[k + 3, k];
				}
			}
		}
	}
}
