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
// Performance: the kernels work on the flat column-major buffers without bounds checks,
// and an eigenvalues-only solve skips Z and confines the QR sweeps to the active window
// (LAPACK's wantt = wantz = false). Neither changes a single rounding: every entry the
// eigenvalues depend on gets the same operations in the same order (GR6P's 64 x 64 action
// matrix, PoseLib/GenRelpose6pt.cs, is the hot caller).
//
// Semantics follow Eigen's documentation: eigenvalues come in the order of the diagonal
// blocks of T (not sorted); a complex pair appears as (re + i im, re - i im) with im > 0
// first; eigenvectors are the columns of a complex matrix, each normalized to unit norm.
// The eigenvector of a real eigenvalue is real; the complex phase of a complex eigenvector
// (and every sign) is this implementation's, like Eigen's arbitrary. Info is NoConvergence
// when the QR iteration exceeds 30 iterations per eigenvalue. The input is divided by its
// largest |entry| first and the eigenvalues multiplied back. Tier B.
//
// Complex-eigenvector phase: ours is fixed by the back-substitution in
// EigenSolver.Vectors.cs (the block's eigenvector starts as (b, lambda - a), then the vector
// is scaled to unit norm without rotating its phase), which need not be Eigen's. No COLMAP
// caller can see it: GR8P (generalized_relative_pose.cc) reads V.real() of its 4x4 G, but G
// is symmetric, so every eigenvalue is real, its vector is real, and hnormalized() removes
// the scale and sign; GR6P's PoseLib solver uses eigenvalues only (see
// Estimators/Solvers/GeneralizedRelativePose.GR8P.cs and PoseLib/GenRelpose6pt.cs).

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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

		// Eigenvalues only: Z is never needed, and the Schur form only has to be right on the
		// diagonal blocks, so the QR sweeps touch just the active window (LAPACK dhseqr's
		// wantt = wantz = false). Every entry inside the window gets the same operations in
		// the same order either way, so the eigenvalues are bit-identical to the full run.
		MatrixXd? z = computeEigenvectors ? MatrixXd.Identity(n) : null;
		Span<double> tData = t.AsSpan();
		Span<double> zData = z is null ? default : z.AsSpan();
		ReduceToHessenberg(tData, n, zData);
		if (!ComputeRealSchur(tData, n, zData, computeEigenvectors))
		{
			Array.Fill(_eigenvalues, new Complex(double.NaN, double.NaN));
			Info = ComputationInfo.NoConvergence;
			return;
		}

		Info = ComputationInfo.Success;
		for (int i = 0; i < n; i++)
		{
			if (i == n - 1 || tData[i * n + i + 1] == 0)
			{
				_eigenvalues[i] = new Complex(tData[i * n + i], 0);
			}
			else
			{
				(double re, double im) = ComplexBlockEigenvalue(tData, n, i);
				_eigenvalues[i] = new Complex(re, im);
				_eigenvalues[i + 1] = new Complex(re, -im);
				i++;
			}
		}

		if (computeEigenvectors)
		{
			_eigenvectors = ComputeEigenvectors(t, z!, _eigenvalues);
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
	private static (double Re, double Im) ComplexBlockEigenvalue(ReadOnlySpan<double> t, int n, int i)
	{
		double a = t[i * n + i];
		double b = t[(i + 1) * n + i];
		double c = t[i * n + i + 1];
		double d = t[(i + 1) * n + i + 1];
		double p = 0.5 * (a - d);
		double q = p * p + b * c;
		return (d + p, Math.Sqrt(Math.Abs(q)));
	}

	/// <summary>
	/// G&amp;VL Algorithm 7.4.2: H = Z^T A Z upper Hessenberg on the column-major n x n h,
	/// Z accumulated unless <paramref name="z"/> is empty.
	/// </summary>
	private static void ReduceToHessenberg(Span<double> h, int n, Span<double> z)
	{
		Span<double> x = n > 2 ? new double[n] : default;
		Span<double> work = n > 2 ? new double[n] : default;
		for (int k = 0; k < n - 2; k++)
		{
			int length = n - k - 1;
			Span<double> v = x[..length];
			h.Slice(k * n + k + 1, length).CopyTo(v);

			double tau = Householder.MakeInPlace(v);
			if (tau == 0)
			{
				continue;
			}

			double beta = v[0];
			v[0] = 1;
			ApplyReflectorLeft(h, n, v, tau, k + 1, k, n);
			ApplyReflectorRight(h, n, v, tau, k + 1, 0, n, work);
			if (!z.IsEmpty)
			{
				ApplyReflectorRight(z, n, v, tau, k + 1, 0, n, work);
			}

			h[k * n + k + 1] = beta;
			h.Slice(k * n + k + 2, n - k - 2).Clear();
		}
	}

	/// <summary>
	/// (I - tau v v^T) applied to rows first.. of columns firstCol..colEnd-1 of the
	/// column-major n-row m.
	/// </summary>
	private static void ApplyReflectorLeft(
		Span<double> m, int n, ReadOnlySpan<double> v, double tau, int first, int firstCol, int colEnd)
	{
		int length = v.Length;
		ref double vRef = ref MemoryMarshal.GetReference(v);
		int c = firstCol;

		// Four columns at a time: four independent dot-product chains instead of one, each
		// still summed left to right, so every column gets exactly the plain loop's result.
		for (; c + 4 <= colEnd; c += 4)
		{
			ref double s0 = ref MemoryMarshal.GetReference(m.Slice(c * n + first, length));
			ref double s1 = ref MemoryMarshal.GetReference(m.Slice((c + 1) * n + first, length));
			ref double s2 = ref MemoryMarshal.GetReference(m.Slice((c + 2) * n + first, length));
			ref double s3 = ref MemoryMarshal.GetReference(m.Slice((c + 3) * n + first, length));
			double dot0 = 0;
			double dot1 = 0;
			double dot2 = 0;
			double dot3 = 0;
			for (int i = 0; i < length; i++)
			{
				double vi = Unsafe.Add(ref vRef, i);
				dot0 += vi * Unsafe.Add(ref s0, i);
				dot1 += vi * Unsafe.Add(ref s1, i);
				dot2 += vi * Unsafe.Add(ref s2, i);
				dot3 += vi * Unsafe.Add(ref s3, i);
			}

			dot0 *= tau;
			dot1 *= tau;
			dot2 *= tau;
			dot3 *= tau;
			for (int i = 0; i < length; i++)
			{
				double vi = Unsafe.Add(ref vRef, i);
				Unsafe.Add(ref s0, i) -= dot0 * vi;
				Unsafe.Add(ref s1, i) -= dot1 * vi;
				Unsafe.Add(ref s2, i) -= dot2 * vi;
				Unsafe.Add(ref s3, i) -= dot3 * vi;
			}
		}

		for (; c < colEnd; c++)
		{
			ref double segment = ref MemoryMarshal.GetReference(m.Slice(c * n + first, length));
			double dot = 0;
			for (int i = 0; i < length; i++)
			{
				dot += Unsafe.Add(ref vRef, i) * Unsafe.Add(ref segment, i);
			}

			dot *= tau;
			for (int i = 0; i < length; i++)
			{
				Unsafe.Add(ref segment, i) -= dot * Unsafe.Add(ref vRef, i);
			}
		}
	}

	/// <summary>
	/// (I - tau v v^T) applied from the right to columns first.. of rows rowStart..rowEnd-1
	/// of the column-major n-row m. Each row's dot product is the same left-to-right sum
	/// over v as a row-by-row loop; the rows are just advanced together, one contiguous
	/// column at a time, with <paramref name="work"/> (at least rowEnd - rowStart long)
	/// holding the running sums.
	/// </summary>
	private static void ApplyReflectorRight(
		Span<double> m, int n, ReadOnlySpan<double> v, double tau, int first, int rowStart, int rowEnd, Span<double> work)
	{
		int rows = rowEnd - rowStart;
		if (rows <= 0)
		{
			return;
		}

		Span<double> dots = work[..rows];
		dots.Clear();
		ref double dotRef = ref MemoryMarshal.GetReference(dots);

		// Columns go in groups of four so each running sum is loaded and stored once per
		// group; within a row the terms are still added in column order.
		int i = 0;
		for (; i + 4 <= v.Length; i += 4)
		{
			double v0 = v[i];
			double v1 = v[i + 1];
			double v2 = v[i + 2];
			double v3 = v[i + 3];
			ref double k0 = ref MemoryMarshal.GetReference(m.Slice((first + i) * n + rowStart, rows));
			ref double k1 = ref MemoryMarshal.GetReference(m.Slice((first + i + 1) * n + rowStart, rows));
			ref double k2 = ref MemoryMarshal.GetReference(m.Slice((first + i + 2) * n + rowStart, rows));
			ref double k3 = ref MemoryMarshal.GetReference(m.Slice((first + i + 3) * n + rowStart, rows));
			for (int r = 0; r < rows; r++)
			{
				double dot = Unsafe.Add(ref dotRef, r);
				dot += Unsafe.Add(ref k0, r) * v0;
				dot += Unsafe.Add(ref k1, r) * v1;
				dot += Unsafe.Add(ref k2, r) * v2;
				dot += Unsafe.Add(ref k3, r) * v3;
				Unsafe.Add(ref dotRef, r) = dot;
			}
		}

		for (; i < v.Length; i++)
		{
			double vi = v[i];
			ref double column = ref MemoryMarshal.GetReference(m.Slice((first + i) * n + rowStart, rows));
			for (int r = 0; r < rows; r++)
			{
				Unsafe.Add(ref dotRef, r) += Unsafe.Add(ref column, r) * vi;
			}
		}

		for (int r = 0; r < rows; r++)
		{
			Unsafe.Add(ref dotRef, r) *= tau;
		}

		for (i = 0; i < v.Length; i++)
		{
			double vi = v[i];
			ref double column = ref MemoryMarshal.GetReference(m.Slice((first + i) * n + rowStart, rows));
			for (int r = 0; r < rows; r++)
			{
				Unsafe.Add(ref column, r) -= Unsafe.Add(ref dotRef, r) * vi;
			}
		}
	}

	/// <summary>
	/// Francis QR iteration on the column-major Hessenberg matrix t, accumulating into z,
	/// until t is quasi-triangular with standardized real 2x2 blocks split. When
	/// <paramref name="full"/> is false only the active window is updated (z is empty), which
	/// leaves the diagonal blocks - all the eigenvalues read - exactly as in the full run.
	/// False on no convergence.
	/// </summary>
	private static bool ComputeRealSchur(Span<double> t, int n, Span<double> z, bool full)
	{
		double norm = Math.Sqrt(VectorXd.Dot(t, t));
		int hi = n - 1;
		int iterations = 0;
		int totalIterations = 0;
		Span<double> v = stackalloc double[3];
		Span<double> work = new double[n];
		while (hi >= 0)
		{
			int lo = hi;
			while (lo > 0)
			{
				double s = Math.Abs(t[(lo - 1) * n + lo - 1]) + Math.Abs(t[lo * n + lo]);
				if (s == 0)
				{
					s = norm;
				}

				if (Math.Abs(t[(lo - 1) * n + lo]) <= LinearAlgebraConstants.MachineEpsilon * s)
				{
					t[(lo - 1) * n + lo] = 0;
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
				SplitRealBlock(t, n, z, lo, full);
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

				FrancisStep(t, n, z, lo, hi, iterations, v, full, work);
			}
		}

		return true;
	}

	/// <summary>
	/// If the 2x2 block at (m, m) has real eigenvalues, rotates it to upper triangular form:
	/// the rotation's first column is the eigenvector (lambda - d, c) of the eigenvalue
	/// lambda = d + p + sign(p) sqrt(p^2 + b c), the one farther from d. When
	/// <paramref name="full"/> is false only the block itself is rotated.
	/// </summary>
	private static void SplitRealBlock(Span<double> t, int n, Span<double> z, int m, bool full)
	{
		int c0Index = m * n;
		int c1Index = (m + 1) * n;
		double p = 0.5 * (t[c0Index + m] - t[c1Index + m + 1]);
		double q = p * p + t[c1Index + m] * t[c0Index + m + 1];
		if (q < 0)
		{
			return;
		}

		double root = Math.Sqrt(q);
		double shifted = p >= 0 ? p + root : p - root;
		double c0 = shifted;
		double c1 = t[c0Index + m + 1];
		double length = Math.Sqrt(c0 * c0 + c1 * c1);
		if (length == 0)
		{
			return;
		}

		double c = c0 / length;
		double s = c1 / length;

		// T <- G^T T G, Z <- Z G with G = [c -s; s c].
		int colEnd = full ? n : m + 2;
		for (int j = m; j < colEnd; j++)
		{
			double x = t[j * n + m];
			double y = t[j * n + m + 1];
			t[j * n + m] = c * x + s * y;
			t[j * n + m + 1] = -s * x + c * y;
		}

		for (int i = full ? 0 : m; i <= m + 1; i++)
		{
			double x = t[c0Index + i];
			double y = t[c1Index + i];
			t[c0Index + i] = c * x + s * y;
			t[c1Index + i] = -s * x + c * y;
		}

		if (!z.IsEmpty)
		{
			for (int i = 0; i < n; i++)
			{
				double x = z[c0Index + i];
				double y = z[c1Index + i];
				z[c0Index + i] = c * x + s * y;
				z[c1Index + i] = -s * x + c * y;
			}
		}

		t[c0Index + m + 1] = 0;
	}

	/// <summary>
	/// G&amp;VL Algorithm 7.5.1 on the unreduced window lo..hi (at least 3 x 3): one implicit
	/// double-shift QR step by bulge chasing with 3-element Householder reflectors. When
	/// <paramref name="full"/> is false the reflectors are applied inside the window only.
	/// </summary>
	private static void FrancisStep(
		Span<double> t, int n, Span<double> z, int lo, int hi, int iterations, Span<double> v, bool full, Span<double> work)
	{
		// t(r, c) is t[c * n + r].
		double shiftSum;
		double shiftProduct;
		if (iterations == 10 || iterations == 20)
		{
			// Wilkinson's exceptional shift breaks cycles of the standard shift.
			double s = Math.Abs(t[(hi - 1) * n + hi]) + Math.Abs(t[(hi - 2) * n + hi - 1]);
			shiftSum = 1.5 * s;
			shiftProduct = s * s;
		}
		else
		{
			shiftSum = t[(hi - 1) * n + hi - 1] + t[hi * n + hi];
			shiftProduct = t[(hi - 1) * n + hi - 1] * t[hi * n + hi] - t[hi * n + hi - 1] * t[(hi - 1) * n + hi];
		}

		double tLoLo = t[lo * n + lo];
		double x = tLoLo * tLoLo + t[(lo + 1) * n + lo] * t[lo * n + lo + 1] - shiftSum * tLoLo + shiftProduct;
		double y = t[lo * n + lo + 1] * (tLoLo + t[(lo + 1) * n + lo + 1] - shiftSum);
		double w = t[lo * n + lo + 1] * t[(lo + 1) * n + lo + 2];
		int colEnd = full ? n : hi + 1;
		int rowStart = full ? 0 : lo;
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
				int rowEnd = Math.Min(k + 3, hi) + 1;
				if (length == 3)
				{
					ApplyReflector3Left(t, n, reflector[1], reflector[2], tau, k, firstCol, colEnd);
					ApplyReflector3Right(t, n, reflector[1], reflector[2], tau, k, rowStart, rowEnd);
				}
				else
				{
					ApplyReflectorLeft(t, n, reflector, tau, k, firstCol, colEnd);
					ApplyReflectorRight(t, n, reflector, tau, k, rowStart, rowEnd, work);
				}

				if (!z.IsEmpty)
				{
					ApplyReflectorRight(z, n, reflector, tau, k, 0, n, work);
				}

				if (k > lo)
				{
					t[(k - 1) * n + k] = beta;
					t[(k - 1) * n + k + 1] = 0;
					if (length == 3)
					{
						t[(k - 1) * n + k + 2] = 0;
					}
				}
			}

			if (k < hi - 1)
			{
				x = t[k * n + k + 1];
				y = t[k * n + k + 2];
				if (k < hi - 2)
				{
					w = t[k * n + k + 3];
				}
			}
		}
	}

	/// <summary>
	/// <see cref="ApplyReflectorLeft"/> for the Francis step's 3-element reflector
	/// v = (1, v1, v2), unrolled: the same dot product (0 + a + v1 b + v2 c, left to right)
	/// and updates, without the per-column loop overhead that dominates at this size.
	/// </summary>
	private static void ApplyReflector3Left(
		Span<double> m, int n, double v1, double v2, double tau, int first, int firstCol, int colEnd)
	{
		if (colEnd <= firstCol)
		{
			return;
		}

		// The slice bounds-checks the whole column range once (first + 2 < n).
		ref double origin = ref MemoryMarshal.GetReference(m.Slice(firstCol * n, (colEnd - firstCol) * n));
		for (int c = 0; c < colEnd - firstCol; c++)
		{
			ref double a = ref Unsafe.Add(ref origin, c * n + first);
			ref double b = ref Unsafe.Add(ref a, 1);
			ref double d = ref Unsafe.Add(ref a, 2);
			double dot = 0;
			dot += 1.0 * a;
			dot += v1 * b;
			dot += v2 * d;
			dot *= tau;
			a -= dot * 1.0;
			b -= dot * v1;
			d -= dot * v2;
		}
	}

	/// <summary>
	/// <see cref="ApplyReflectorRight"/> for the Francis step's 3-element reflector
	/// v = (1, v1, v2) on columns first..first+2, unrolled with the same operation order.
	/// </summary>
	private static void ApplyReflector3Right(
		Span<double> m, int n, double v1, double v2, double tau, int first, int rowStart, int rowEnd)
	{
		int rows = rowEnd - rowStart;
		if (rows <= 0)
		{
			return;
		}

		ref double c0 = ref MemoryMarshal.GetReference(m.Slice(first * n + rowStart, rows));
		ref double c1 = ref MemoryMarshal.GetReference(m.Slice((first + 1) * n + rowStart, rows));
		ref double c2 = ref MemoryMarshal.GetReference(m.Slice((first + 2) * n + rowStart, rows));
		for (int r = 0; r < rows; r++)
		{
			ref double a = ref Unsafe.Add(ref c0, r);
			ref double b = ref Unsafe.Add(ref c1, r);
			ref double d = ref Unsafe.Add(ref c2, r);
			double dot = 0;
			dot += a * 1.0;
			dot += b * v1;
			dot += d * v2;
			dot *= tau;
			a -= dot * 1.0;
			b -= dot * v1;
			d -= dot * v2;
		}
	}
}
