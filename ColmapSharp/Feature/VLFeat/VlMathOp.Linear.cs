// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlMathOp.Linear: the small dense linear algebra of thirdparty/VLFeat/mathop.c that the
// covariant detector uses - vl_svd2 with its vl_lapack_dlasv2 (Bai and Demmel's 2x2
// triangular SVD, as written in VLFeat's own C, not LAPACK's Fortran), vl_gaussian_elimination
// and vl_solve_linear_system_2/3 - plus mathop.h's vl_log2_d and vl_ceil_d. Part of VlMathOp
// (VlMathOp.cs has the SIFT pieces). Callers: VlCovDet*.cs.
//
// Tier A (exact): double throughout, same operation order as the C. Matrices are VLFeat's
// column-major 2x2 / 3x3 arrays.

namespace ColmapSharp.Feature.VLFeat;

/// <content>2x2 SVD and Gaussian elimination.</content>
public static partial class VlMathOp
{
	/// <summary>VLFeat's VL_ERR_OK.</summary>
	public const int ErrOk = 0;

	/// <summary>VLFeat's VL_ERR_OVERFLOW (returned by a singular elimination).</summary>
	public const int ErrOverflow = 1;

	/// <summary>Port of vl_log2_d (__builtin_log2 under GCC/clang).</summary>
	public static double Log2D(double x) => Math.Log2(x);

	/// <summary>Port of vl_ceil_d (__builtin_ceil under GCC/clang).</summary>
	public static double CeilD(double x) => Math.Ceiling(x);

	/// <summary>
	/// Port of vl_svd2: M = U * S * V' for a column-major 2x2 matrix. S[0] and S[3] are the
	/// signed singular values (largest modulus first), S[1] = S[2] = 0.
	/// </summary>
	public static void Svd2(Span<double> s, Span<double> u, Span<double> v, ReadOnlySpan<double> m)
	{
		double m11 = m[0];
		double m21 = m[1];
		double m12 = m[2];
		double m22 = m[3];
		double cu1 = m11;
		double su1 = m21;
		double norm = Math.Sqrt((cu1 * cu1) + (su1 * su1));
		cu1 /= norm;
		su1 /= norm;

		double f = (cu1 * m11) + (su1 * m21);
		double g = (cu1 * m12) + (su1 * m22);
		double h = (-su1 * m12) + (cu1 * m22);

		LapackDlasv2(out double smin, out double smax, out double sv2, out double cv2, out double su2, out double cu2, f, g, h);

		s[0] = smax;
		s[1] = 0;
		s[2] = 0;
		s[3] = smin;

		if (!u.IsEmpty)
		{
			u[0] = (cu2 * cu1) - (su2 * su1);
			u[1] = (su2 * cu1) + (cu2 * su1);
			u[2] = (-cu2 * su1) - (su2 * cu1);
			u[3] = (-su2 * su1) + (cu2 * cu1);
		}

		if (!v.IsEmpty)
		{
			v[0] = cv2;
			v[1] = sv2;
			v[2] = -sv2;
			v[3] = cv2;
		}
	}

	/// <summary>
	/// Port of vl_lapack_dlasv2: SVD of the upper triangular [f g; 0 h] (Z. Bai and J. Demmel,
	/// "Computing the Generalized Singular Value Decomposition", SIAM J. Sci. Comput. 14(6),
	/// 1993), in VLFeat's formulation.
	/// </summary>
	public static void LapackDlasv2(
		out double smin, out double smax, out double sv, out double cv, out double su, out double cu,
		double f, double g, double h)
	{
		double svt = 0, cvt = 0, sut = 0, cut = 0;
		double ft = f, gt = g, ht = h;
		double fa = Math.Abs(f), ga = Math.Abs(g), ha = Math.Abs(h);
		int pmax = 1;
		bool swap = false;
		bool glarge = false;
		double tmp;
		smin = 0;
		smax = 0;

		// Make fa >= ha.
		if (fa < ha)
		{
			pmax = 3;
			tmp = ft;
			ft = ht;
			ht = tmp;
			tmp = fa;
			fa = ha;
			ha = tmp;
			swap = true;
		}

		if (ga == 0.0)
		{
			// Diagonal.
			smin = ha;
			smax = fa;
			cut = 1.0;
			sut = 0.0;
			cvt = 1.0;
			svt = 0.0;
		}
		else
		{
			if (ga > fa)
			{
				// g is the largest entry.
				pmax = 2;
				if ((fa / ga) < EpsilonD)
				{
					// g is very large.
					glarge = true;
					smax = ga;
					smin = ha > 1.0 ? fa / (ga / ha) : (fa / ga) * ha;
					cut = 1.0;
					sut = ht / gt;
					cvt = 1.0;
					svt = ft / gt;
				}
			}

			if (!glarge)
			{
				// Normal case.
				double fmh = fa - ha;
				double d = fmh == fa ? 1.0 : fmh / fa;
				double q = gt / ft;
				double sq = 2.0 - d;
				double dd = d * d;
				double qq = q * q;
				double ss = sq * sq;
				double spq = Math.Sqrt(ss + qq);
				double dpq = d == 0.0 ? Math.Abs(q) : Math.Sqrt(dd + qq);
				double a = 0.5 * (spq + dpq);
				smin = ha / a;
				smax = fa * a;
				if (qq == 0.0)
				{
					// qq underflow.
					tmp = d == 0.0
						? Sign(ft) * 2 * Sign(gt)
						: (gt / (Sign(ft) * fmh)) + (q / sq);
				}
				else
				{
					tmp = ((q / (spq + sq)) + (q / (dpq + d))) * (1.0 + a);
				}

				double tt = Math.Sqrt((tmp * tmp) + 4.0);
				cvt = 2.0 / tt;
				svt = tmp / tt;
				cut = (cvt + (svt * q)) / a;
				sut = (ht / ft) * svt / a;
			}
		}

		if (swap)
		{
			cu = svt;
			su = cvt;
			cv = sut;
			sv = cut;
		}
		else
		{
			cu = cut;
			su = sut;
			cv = cvt;
			sv = svt;
		}

		// Correct the signs of smax and smin.
		int tsign = pmax switch
		{
			1 => Sign(cv) * Sign(cu) * Sign(f),
			2 => Sign(sv) * Sign(cu) * Sign(g),
			_ => Sign(sv) * Sign(su) * Sign(h),
		};
		smax = ISign(tsign) * smax;
		smin = ISign(tsign * Sign(f) * Sign(h)) * smin;
	}

	/// <summary>
	/// Port of vl_gaussian_elimination: Gaussian elimination with partial pivoting of the
	/// column-major <paramref name="numRows"/> x <paramref name="numColumns"/> matrix [A, b] in
	/// place. Returns <see cref="ErrOverflow"/>, leaving the matrix half-processed as VLFeat
	/// does, when a pivot is below 1e-10.
	/// </summary>
	public static int GaussianElimination(Span<double> a, int numRows, int numColumns)
	{
		for (int j = 0; j < numRows; ++j)
		{
			double maxa = 0;
			double maxabsa = 0;
			int maxi = -1;

			// Look for the maximally stable pivot.
			for (int i = j; i < numRows; ++i)
			{
				double v = a[i + (j * numRows)];
				double absa = Math.Abs(v);
				if (absa > maxabsa)
				{
					maxa = v;
					maxabsa = absa;
					maxi = i;
				}
			}

			// If singular give up.
			if (maxabsa < 1e-10)
			{
				return ErrOverflow;
			}

			// Swap the j-th row with the i-th row and normalize the j-th row.
			for (int jj = j; jj < numColumns; ++jj)
			{
				double tmp = a[maxi + (jj * numRows)];
				a[maxi + (jj * numRows)] = a[j + (jj * numRows)];
				a[j + (jj * numRows)] = tmp;
				a[j + (jj * numRows)] /= maxa;
			}

			// Elimination.
			for (int ii = j + 1; ii < numRows; ++ii)
			{
				double x = a[ii + (j * numRows)];
				for (int jj = j; jj < numColumns; ++jj)
				{
					a[ii + (jj * numRows)] -= x * a[j + (jj * numRows)];
				}
			}
		}

		// Backward substitution.
		for (int i = numRows - 1; i > 0; --i)
		{
			// Substitute in all rows above.
			for (int ii = i - 1; ii >= 0; --ii)
			{
				double x = a[ii + (i * numRows)];
				for (int j = numRows; j < numColumns; ++j)
				{
					a[ii + (j * numRows)] -= x * a[i + (j * numRows)];
				}
			}
		}

		return ErrOk;
	}

	/// <summary>Port of vl_solve_linear_system_3: x = A \ b for a column-major 3x3 A.</summary>
	public static int SolveLinearSystem3(Span<double> x, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
	{
		Span<double> m = stackalloc double[12];
		a.Slice(0, 9).CopyTo(m);
		m[9] = b[0];
		m[10] = b[1];
		m[11] = b[2];
		int err = GaussianElimination(m, 3, 4);
		x[0] = m[9];
		x[1] = m[10];
		x[2] = m[11];
		return err;
	}

	/// <summary>Port of vl_solve_linear_system_2: x = A \ b for a column-major 2x2 A.</summary>
	public static int SolveLinearSystem2(Span<double> x, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
	{
		Span<double> m = stackalloc double[6];
		a.Slice(0, 4).CopyTo(m);
		m[4] = b[0];
		m[5] = b[1];
		int err = GaussianElimination(m, 2, 3);
		x[0] = m[4];
		x[1] = m[5];
		return err;
	}

	// mathop.c's sign(x) macro: -1 for negative, +1 otherwise (so +1 for zero and NaN).
	private static int Sign(double x) => x < 0.0 ? -1 : +1;

	// mathop.c's isign(i) macro.
	private static int ISign(int i) => i < 0 ? -1 : +1;
}
