// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// JacobiSvdKernel: the square two-sided Jacobi SVD that JacobiSVD (any shape, MatrixXd) and
// the allocation-free fixed-size wrappers Svd3d / Svd4d share. It works in place on
// column-major spans so the fixed-size callers can run it on stackalloc buffers.
//
// Algorithm: two-sided (Kogbetliantz) cyclic Jacobi, Golub & Van Loan, "Matrix
// Computations", 4th ed., §8.6.3. Each 2x2 step follows Brent, Luk and Van Loan,
// "Computation of the singular value decomposition using mesh-connected processors",
// J. VLSI Comput. Syst. 1 (1985), which diagonalizes B = [w x; y z] by a left rotation
// through theta and a right rotation through phi found from their sum and difference.
// Derivation used here: split B into a scaled rotation plus a scaled reflection,
//   B = r1 Rot(a1) + r2 Ref(a2),  Rot(a) = [cos -sin; sin cos],  Ref(a) = [cos sin; sin -cos],
// with r1 Rot(a1) = [(w+z)/2, (x-y)/2; (y-x)/2, (w+z)/2] and
//      r2 Ref(a2) = [(w-z)/2, (x+y)/2; (x+y)/2, (z-w)/2].
// Since Rot(theta)^T Rot(a) Rot(phi) = Rot(a - theta + phi) and
// Rot(theta)^T Ref(a) Rot(phi) = Ref(a - theta - phi), choosing theta = (a1 + a2) / 2 and
// phi = (a2 - a1) / 2 gives Rot(theta)^T B Rot(phi) = diag(r1 + r2, r1 - r2). The angles
// come from atan2, which neither overflows nor underflows. Every rotation is accumulated
// into U and V, so both are orthogonal to working precision whatever the rank, which is
// what COLMAP's null-space reads (the last column of V) rely on. A 2x2 block with an exactly
// zero column (row) is instead diagonalized by the left (right) rotation alone, the right
// (left) one being exactly the identity, so exact null vectors from zero columns stay exact
// (TryOneSidedRotation).
//
// Convergence: the relative test of Demmel and Veselić, "Jacobi's method is more accurate
// than QR", SIAM J. Matrix Anal. Appl. 13 (1992): the pair (i, j) is rotated while
// max(|a(i,j)|, |a(j,i)|) > max(epsilon * sqrt(|a(i,i)|) * sqrt(|a(j,j)|), epsilon * ||A||_F),
// and the iteration stops after a sweep that rotates nothing. The absolute floor
// epsilon * ||A||_F is Golub & Van Loan's stopping level off(A) <= epsilon ||A||_F (§8.5.3)
// applied per pair, in the spirit of the tolerances of Drmač and Veselić, "New fast and
// accurate Jacobi SVD algorithm" I/II (LAPACK Working Notes 169/170, 2007): without it a
// zero diagonal (rank-deficient input: essential matrices, noise-free DLT systems, zero
// columns) would keep rotating on roundoff. Exactly rank-deficient inputs converge in a
// few sweeps (SpectralTests pins the counts) (Info NoConvergence if a sweep cap is reached
// first, which does not happen on finite input in practice). The input is scaled by the
// power of two nearest its Frobenius norm (computed without overflow), which is exact, and
// the singular values scaled back.
// Written from those sources; Eigen (MPL-2.0) is not ported.
//
// Conventions (Eigen's documented JacobiSVD semantics): singular values non-negative and
// sorted in decreasing order. A negative diagonal entry flips the sign of the matching U
// column. Ties keep their original diagonal position (stable insertion sort). The signs of
// singular vector pairs are whatever the rotations produce; like Eigen's, they are
// arbitrary, and COLMAP only uses them up to sign. Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Two-sided Jacobi SVD of a square column-major matrix, in place on spans.
/// </summary>
internal static class JacobiSvdKernel
{
	private const int MaxSweeps = 100;

	/// <summary>
	/// Decomposes the n x n matrix in <paramref name="a"/> (destroyed) as U diag(s) V^T.
	/// <paramref name="u"/> and <paramref name="v"/> receive n x n column-major factors
	/// when non-empty; <paramref name="s"/> receives the n singular values, descending.
	/// Returns Success, NoConvergence, or InvalidInput for a non-finite entry (outputs are
	/// then NaN).
	/// </summary>
	public static ComputationInfo Decompose(Span<double> a, int n, Span<double> u, Span<double> v, Span<double> s) =>
		Decompose(a, n, u, v, s, out _);

	/// <summary>Decompose, also reporting the number of sweeps run (the last one rotates nothing).</summary>
	public static ComputationInfo Decompose(Span<double> a, int n, Span<double> u, Span<double> v, Span<double> s, out int sweeps)
	{
		sweeps = 0;
		if (!u.IsEmpty)
		{
			SetIdentity(u, n);
		}

		if (!v.IsEmpty)
		{
			SetIdentity(v, n);
		}

		foreach (double value in a)
		{
			if (!double.IsFinite(value))
			{
				s.Fill(double.NaN);
				u.Fill(double.NaN);
				v.Fill(double.NaN);
				return ComputationInfo.InvalidInput;
			}
		}

		double norm = FrobeniusNorm(a);
		if (norm == 0)
		{
			s[..n].Clear();
			return ComputationInfo.Success;
		}

		// Scale by the power of two nearest the norm: exact, so an input that needs no
		// rotation (a diagonal matrix) returns its |entries| bit for bit.
		int exponent = Math.ILogB(norm);
		for (int i = 0; i < a.Length; i++)
		{
			a[i] = Math.ScaleB(a[i], -exponent);
		}

		double floor = LinearAlgebraConstants.MachineEpsilon * FrobeniusNorm(a);
		ComputationInfo info = ComputationInfo.NoConvergence;
		for (int sweep = 0; sweep < MaxSweeps; sweep++)
		{
			sweeps = sweep + 1;
			bool rotated = false;
			for (int i = 0; i < n - 1; i++)
			{
				for (int j = i + 1; j < n; j++)
				{
					double off = Math.Max(Math.Abs(a[j * n + i]), Math.Abs(a[i * n + j]));
					double diagonalScale = Math.Sqrt(Math.Abs(a[i * n + i])) * Math.Sqrt(Math.Abs(a[j * n + j]));
					if (off > Math.Max(LinearAlgebraConstants.MachineEpsilon * diagonalScale, floor))
					{
						rotated = true;
						RotatePair(a, n, i, j, u, v);
					}
				}
			}

			if (!rotated)
			{
				info = ComputationInfo.Success;
				break;
			}
		}

		for (int i = 0; i < n; i++)
		{
			double d = a[i * n + i];
			s[i] = Math.ScaleB(Math.Abs(d), exponent);
			if (d < 0 && !u.IsEmpty)
			{
				Span<double> column = u.Slice(i * n, n);
				for (int k = 0; k < n; k++)
				{
					column[k] = -column[k];
				}
			}
		}

		SortDescending(n, u, v, s);
		return info;
	}

	/// <summary>
	/// The Brent-Luk-Van Loan step on rows/columns i &lt; j (derivation in the file header):
	/// A &lt;- Rot(theta)^T A Rot(phi), U &lt;- U Rot(theta), V &lt;- V Rot(phi).
	/// </summary>
	private static void RotatePair(Span<double> a, int n, int i, int j, Span<double> u, Span<double> v)
	{
		double w = a[i * n + i];
		double x = a[j * n + i];
		double y = a[i * n + j];
		double z = a[j * n + j];
		double ct;
		double st;
		double cp;
		double sp;
		if (!TryOneSidedRotation(w, x, y, z, out ct, out st, out cp, out sp))
		{
			double rotationAngle = Math.Atan2(0.5 * y - 0.5 * x, 0.5 * w + 0.5 * z);
			double reflectionAngle = Math.Atan2(0.5 * x + 0.5 * y, 0.5 * w - 0.5 * z);
			double theta = 0.5 * (rotationAngle + reflectionAngle);
			double phi = 0.5 * (reflectionAngle - rotationAngle);
			ct = Math.Cos(theta);
			st = Math.Sin(theta);
			cp = Math.Cos(phi);
			sp = Math.Sin(phi);
		}

		// Rows i, j <- Rot(theta)^T rows.
		for (int k = 0; k < n; k++)
		{
			double ri = a[k * n + i];
			double rj = a[k * n + j];
			a[k * n + i] = ct * ri + st * rj;
			a[k * n + j] = -st * ri + ct * rj;
		}

		// Columns i, j <- columns Rot(phi).
		RotateColumns(a, n, i, j, cp, sp);

		// The pair is diagonal in exact arithmetic; drop the rounding residue.
		a[j * n + i] = 0;
		a[i * n + j] = 0;

		// A = Rot(theta) A' Rot(phi)^T.
		if (!u.IsEmpty)
		{
			RotateColumns(u, n, i, j, ct, st);
		}

		if (!v.IsEmpty)
		{
			RotateColumns(v, n, i, j, cp, sp);
		}
	}

	/// <summary>
	/// The 2x2 block B = [w x; y z] with an exactly zero column (or row) is diagonalized by
	/// a left (right) rotation alone: the rotation that folds the other column (row) onto
	/// one axis. The right (left) rotation is then exactly the identity, so a zero column
	/// of A is never mixed into V (a zero row never into U), and the null vector it stands
	/// for stays an exact unit vector. The general two-angle step cannot promise that:
	/// cos(pi/2) is 6.1e-17, not 0, and that residue lands in coordinates COLMAP tests for
	/// exact zero (TriangulatePoint rejects parallel rays by V(3,3) == 0,
	/// triangulation_test.cc TriangulatePoint.ParallelRays). Returns false when neither
	/// case applies.
	/// </summary>
	private static bool TryOneSidedRotation(
		double w, double x, double y, double z, out double ct, out double st, out double cp, out double sp)
	{
		ct = 1;
		st = 0;
		cp = 1;
		sp = 0;
		if (x == 0 && z == 0)
		{
			// Column j is zero: Rot(theta)^T (w, y) = (r, 0).
			double r = Hypot(w, y);
			ct = w / r;
			st = y / r;
			return true;
		}

		if (w == 0 && y == 0)
		{
			// Column i is zero: Rot(theta)^T (x, z) = (0, r).
			double r = Hypot(x, z);
			ct = z / r;
			st = -x / r;
			return true;
		}

		if (y == 0 && z == 0)
		{
			// Row j is zero: (w, x) Rot(phi) = (r, 0).
			double r = Hypot(w, x);
			cp = w / r;
			sp = x / r;
			return true;
		}

		if (w == 0 && x == 0)
		{
			// Row i is zero: (y, z) Rot(phi) = (0, r).
			double r = Hypot(y, z);
			cp = z / r;
			sp = -y / r;
			return true;
		}

		return false;
	}

	// sqrt(a^2 + b^2) of a nonzero pair without overflow or underflow.
	private static double Hypot(double a, double b)
	{
		double largest = Math.Max(Math.Abs(a), Math.Abs(b));
		double sa = a / largest;
		double sb = b / largest;
		return largest * Math.Sqrt(sa * sa + sb * sb);
	}

	/// <summary>Columns i, j of an n x n matrix times Rot = [c -s; s c].</summary>
	private static void RotateColumns(Span<double> m, int n, int i, int j, double c, double s)
	{
		Span<double> ci = m.Slice(i * n, n);
		Span<double> cj = m.Slice(j * n, n);
		for (int k = 0; k < n; k++)
		{
			double p = ci[k];
			double q = cj[k];
			ci[k] = c * p + s * q;
			cj[k] = -s * p + c * q;
		}
	}

	/// <summary>
	/// Frobenius norm without overflow or underflow: the entries are divided by the largest
	/// magnitude before squaring.
	/// </summary>
	internal static double FrobeniusNorm(ReadOnlySpan<double> values)
	{
		double largest = 0;
		foreach (double value in values)
		{
			largest = Math.Max(largest, Math.Abs(value));
		}

		if (largest == 0)
		{
			return 0;
		}

		double sum = 0;
		foreach (double value in values)
		{
			double scaled = value / largest;
			sum += scaled * scaled;
		}

		return largest * Math.Sqrt(sum);
	}

	/// <summary>
	/// Insertion sort of the singular values into decreasing order, moving the matching
	/// U and V columns. Insertion sort is stable, so equal values keep their order.
	/// </summary>
	private static void SortDescending(int n, Span<double> u, Span<double> v, Span<double> s)
	{
		for (int i = 1; i < n; i++)
		{
			for (int k = i; k > 0 && s[k] > s[k - 1]; k--)
			{
				(s[k], s[k - 1]) = (s[k - 1], s[k]);
				SwapColumns(u, n, k, k - 1);
				SwapColumns(v, n, k, k - 1);
			}
		}
	}

	internal static void SwapColumns(Span<double> m, int n, int a, int b)
	{
		if (m.IsEmpty)
		{
			return;
		}

		for (int k = 0; k < n; k++)
		{
			(m[a * n + k], m[b * n + k]) = (m[b * n + k], m[a * n + k]);
		}
	}

	private static void SetIdentity(Span<double> m, int n)
	{
		m[..(n * n)].Clear();
		for (int i = 0; i < n; i++)
		{
			m[i * n + i] = 1;
		}
	}

	/// <summary>
	/// Eigen's documented default rank rule: a singular value counts as nonzero when it is
	/// strictly greater than max(1, diagSize) * epsilon * the largest singular value.
	/// </summary>
	public static int Rank(ReadOnlySpan<double> s)
	{
		if (s.IsEmpty || s[0] == 0)
		{
			return 0;
		}

		double threshold = s[0] * Math.Max(1, s.Length) * LinearAlgebraConstants.MachineEpsilon;
		int rank = 0;
		foreach (double value in s)
		{
			if (value > threshold)
			{
				rank++;
			}
		}

		return rank;
	}
}
