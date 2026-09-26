// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// Sturm: PoseLib/misc/sturm.h - real-root isolation of a univariate polynomial by Sturm
// sequences and bisection, finished with Ridders' method and Newton steps. Used by the
// minimal solvers whose problem reduces to one polynomial: Re3q3.cs (degree 8, for P4Pf),
// Relpose5pt.cs (degree 10) and Relpose6ptSharedFocal.cs (degree 15).
//
// Translation notes:
// - PoseLib makes the degree N a template parameter so the buffers live on the stack; here N
//   is an argument and the buffers are stackalloc'd, so there is still no heap allocation.
// - signchanges<N> counts sign changes with a popcount over a bit mask for N < 32; that is
//   the same count as the plain loop PoseLib uses for N >= 32, which is what is ported.
// - charpoly_danilevsky_piv (the characteristic polynomial by Danilevsky's method with
//   pivoting) is ported for Relpose6ptSharedFocal.cs, which brackets the eigenvalues of its
//   action matrix as polynomial roots. The matrix is a column-major span of n x n values.
// - MAX_STURM_RECURSION_DEPTH_LIMIT keeps PoseLib's default, 300.

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>Sturm-sequence root bracketing. Port of poselib::sturm.</summary>
public static class Sturm
{
	private const int MaxRecursionDepth = 300;

	/// <summary>
	/// Find the real roots of the degree-<paramref name="n"/> polynomial
	/// sum coeffs[i] x^i. Returns the number of roots written to <paramref name="roots"/>
	/// (which must hold n values), or 0 when the leading coefficient is exactly zero.
	/// Port of poselib::sturm::bisect_sturm&lt;N&gt;.
	/// </summary>
	public static int BisectSturm(int n, ReadOnlySpan<double> coeffs, Span<double> roots, double tol = 1e-10)
	{
		if (n == 0)
		{
			return 0;
		}

		if (n == 1)
		{
			if (coeffs[1] == 0.0)
			{
				return 0;
			}

			roots[0] = -coeffs[0] / coeffs[1];
			return 1;
		}

		if (coeffs[n] == 0.0)
		{
			return 0;
		}

		Span<double> fvec = stackalloc double[2 * n + 1];
		Span<double> svec = stackalloc double[3 * n];

		// fvec is the polynomial and its first derivative.
		coeffs[..(n + 1)].CopyTo(fvec);

		// Normalize w.r.t. leading coeff
		double cInv = 1.0 / fvec[n];
		for (int i = 0; i < n; ++i)
		{
			fvec[i] *= cInv;
		}

		fvec[n] = 1.0;

		// Compute the derivative with normalized coefficients
		for (int i = 0; i < n - 1; ++i)
		{
			fvec[n + 1 + i] = fvec[i + 1] * ((i + 1) / (double)n);
		}

		fvec[2 * n] = 1.0;

		// Compute sturm sequences
		BuildSturmSeq(n, fvec, svec);

		// All real roots are in the interval [-r0, r0]
		double r0 = GetBounds(n, fvec);
		double a = -r0;
		double b = r0;

		int sa = SignChanges(n, svec, a);
		int sb = SignChanges(n, svec, b);

		int nRoots = sa - sb;
		if (nRoots == 0)
		{
			return 0;
		}

		nRoots = 0;
		IsolateRoots(n, fvec, svec, a, b, sa, sb, roots, ref nRoots, tol, 0);

		return nRoots;
	}

	// Constructs the quotients needed for evaluating the sturm sequence.
	private static void BuildSturmSeq(int n, ReadOnlySpan<double> fvec, Span<double> svec)
	{
		Span<double> f = stackalloc double[3 * n];
		fvec[..(2 * n + 1)].CopyTo(f);

		// f1, f2, f3 are offsets into f, juggled as PoseLib juggles its pointers.
		int f1 = 0;
		int f2 = n + 1;
		int f3 = f2 + n;

		for (int i = 0; i < n - 1; ++i)
		{
			double q1 = f[f1 + n - i] * f[f2 + n - 1 - i];
			double q0 = f[f1 + n - 1 - i] * f[f2 + n - 1 - i] - f[f1 + n - i] * f[f2 + n - 2 - i];

			f[f3] = f[f1] - q0 * f[f2];
			for (int j = 1; j < n - 1 - i; ++j)
			{
				f[f3 + j] = f[f1 + j] - q1 * f[f2 + j - 1] - q0 * f[f2 + j];
			}

			double c = -Math.Abs(f[f3 + n - 2 - i]);
			double ci = 1.0 / c;
			for (int j = 0; j < n - 1 - i; ++j)
			{
				f[f3 + j] = f[f3 + j] * ci;
			}

			// juggle pointers (f1,f2,f3) -> (f2,f3,f1)
			int tmp = f1;
			f1 = f2;
			f2 = f3;
			f3 = tmp;

			svec[3 * i] = q0;
			svec[3 * i + 1] = q1;
			svec[3 * i + 2] = c;
		}

		svec[3 * n - 3] = f[f1];
		svec[3 * n - 2] = f[f1 + 1];
		svec[3 * n - 1] = f[f2];
	}

	// Evaluates polynomial using Horner's method.
	// Assumes that f[N] = 1.0
	private static double Polyval(int n, ReadOnlySpan<double> f, double x)
	{
		double fx = x + f[n - 1];
		for (int i = n - 2; i >= 0; --i)
		{
			fx = x * fx + f[i];
		}

		return fx;
	}

	// Evaluates the sturm sequence and counts the number of sign changes
	private static int SignChanges(int n, ReadOnlySpan<double> svec, double x)
	{
		Span<double> f = stackalloc double[n + 1];
		f[n] = svec[3 * n - 1];
		f[n - 1] = svec[3 * n - 3] + x * svec[3 * n - 2];

		for (int i = n - 2; i >= 0; --i)
		{
			f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
		}

		int count = 0;
		bool neg1 = f[0] < 0;
		for (int i = 0; i < n; ++i)
		{
			bool neg2 = f[i + 1] < 0;
			if (neg1 ^ neg2)
			{
				++count;
			}

			neg1 = neg2;
		}

		return count;
	}

	// Computes the Cauchy bound on the real roots.
	// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
	private static double GetBounds(int n, ReadOnlySpan<double> fvec)
	{
		double max = 0;
		for (int i = 0; i < n; ++i)
		{
			max = Math.Max(max, Math.Abs(fvec[i]));
		}

		return 1.0 + max;
	}

	// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
	private static void RiddersMethodNewton(int n, ReadOnlySpan<double> fvec, double a, double b, Span<double> roots, ref int nRoots, double tol)
	{
		double fa = Polyval(n, fvec, a);
		double fb = Polyval(n, fvec, b);

		if (!((fa < 0) ^ (fb < 0)))
		{
			return;
		}

		const double TolNewton = 1e-3;

		for (int iter = 0; iter < 30; ++iter)
		{
			if (Math.Abs(a - b) < TolNewton)
			{
				break;
			}

			double c = (a + b) * 0.5;
			double fc = Polyval(n, fvec, c);
			double s = Math.Sqrt(fc * fc - fa * fb);
			// C++ `if (!s)`: true only for s == 0 (a NaN s is truthy and carries on).
			if (s == 0)
			{
				break;
			}

			double d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
			double fd = Polyval(n, fvec, d);

			if (fd >= 0 ? (fc < 0) : (fc > 0))
			{
				a = c;
				fa = fc;
				b = d;
				fb = fd;
			}
			else if (fd >= 0 ? (fa < 0) : (fa > 0))
			{
				b = d;
				fb = fd;
			}
			else
			{
				a = d;
				fa = fd;
			}
		}

		// We switch to Newton's method once we are close to the root
		double x = (a + b) * 0.5;

		ReadOnlySpan<double> fpvec = fvec[(n + 1)..];
		for (int iter = 0; iter < 10; ++iter)
		{
			double fx = Polyval(n, fvec, x);
			if (Math.Abs(fx) < tol)
			{
				break;
			}

			double fpx = n * Polyval(n - 1, fpvec, x);
			double dx = fx / fpx;
			x -= dx;
			if (Math.Abs(dx) < tol)
			{
				break;
			}
		}

		roots[nRoots++] = x;
	}

	private static void IsolateRoots(
		int n, ReadOnlySpan<double> fvec, ReadOnlySpan<double> svec, double a, double b, int sa, int sb,
		Span<double> roots, ref int nRoots, double tol, int depth)
	{
		if (depth > MaxRecursionDepth)
		{
			return;
		}

		if (b - a < tol)
		{
			roots[nRoots++] = b;
			return;
		}

		int nRts = sa - sb;

		if (nRts > 1)
		{
			double c = (a + b) * 0.5;
			int sc = SignChanges(n, svec, c);
			IsolateRoots(n, fvec, svec, a, c, sa, sc, roots, ref nRoots, tol, depth + 1);
			IsolateRoots(n, fvec, svec, c, b, sc, sb, roots, ref nRoots, tol, depth + 1);
		}
		else if (nRts == 1)
		{
			RiddersMethodNewton(n, fvec, a, b, roots, ref nRoots, tol);
		}
	}

	/// <summary>
	/// Writes the monic characteristic polynomial of the n x n column-major matrix
	/// <paramref name="a"/> (destroyed) to <paramref name="p"/> (n + 1 coefficients, constant
	/// first), by Danilevsky's reduction to Frobenius form with row/column pivoting.
	/// Port of poselib::sturm::charpoly_danilevsky_piv.
	/// </summary>
	public static void CharpolyDanilevskyPiv(Span<double> a, int n, Span<double> p)
	{
		Span<double> v = stackalloc double[n];
		Span<double> vinv = stackalloc double[n];
		Span<double> aCol = stackalloc double[n];
		Span<double> newRow = stackalloc double[n];
		for (int i = n - 1; i > 0; i--)
		{
			int pivInd = i - 1;
			double piv = Math.Abs(a[i + n * (i - 1)]);

			// Find largest pivot
			for (int j = 0; j < i - 1; j++)
			{
				if (Math.Abs(a[i + n * j]) > piv)
				{
					piv = Math.Abs(a[i + n * j]);
					pivInd = j;
				}
			}

			if (pivInd != i - 1)
			{
				// Perform permutation
				for (int c = 0; c < n; c++)
				{
					(a[(i - 1) + n * c], a[pivInd + n * c]) = (a[pivInd + n * c], a[(i - 1) + n * c]);
				}

				for (int r = 0; r < n; r++)
				{
					(a[r + n * (i - 1)], a[r + n * pivInd]) = (a[r + n * pivInd], a[r + n * (i - 1)]);
				}
			}

			piv = a[i + n * (i - 1)];

			// A.row(i - 1) = v^T * A with v = A.row(i).
			for (int c = 0; c < n; c++)
			{
				v[c] = a[i + n * c];
			}

			for (int c = 0; c < n; c++)
			{
				double sum = 0.0;
				for (int k = 0; k < n; k++)
				{
					sum += v[k] * a[k + n * c];
				}

				newRow[c] = sum;
			}

			for (int c = 0; c < n; c++)
			{
				a[(i - 1) + n * c] = newRow[c];
			}

			for (int k = 0; k < n; k++)
			{
				vinv[k] = -1.0 * v[k];
			}

			vinv[i - 1] = 1;
			for (int k = 0; k < n; k++)
			{
				vinv[k] /= piv;
			}

			vinv[i - 1] -= 1;
			for (int r = 0; r < n; r++)
			{
				aCol[r] = a[r + n * (i - 1)];
			}

			for (int j = 0; j <= i; j++)
			{
				for (int c = 0; c < n; c++)
				{
					a[j + n * c] = a[j + n * c] + aCol[j] * vinv[c];
				}
			}

			for (int c = 0; c < n; c++)
			{
				a[i + n * c] = 0.0;
			}

			a[i + n * (i - 1)] = 1;
		}

		p[n] = 1;
		for (int i = 0; i < n; i++)
		{
			p[i] = -a[n * (n - i - 1)];
		}
	}
}
