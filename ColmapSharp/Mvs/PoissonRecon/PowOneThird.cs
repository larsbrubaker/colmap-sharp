// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PowOneThird: a managed, correctly rounded pow( x , 1./3 ) for the PoissonRecon port.
// TransformedInputOrientedSampleStream scales the normal transform by
// pow( fabs( det ) , 1./Dim ), which the C++ takes from the platform libm. .NET's Math.Pow
// is also the platform libm (and wasm has its own), so the port computes the value itself:
// the correctly rounded double of x^p for p = fl(1/3) = 1/3 - 2^-54/3, which is what Apple's
// libm (the oracle's) returns whenever its result is correctly rounded. See
// docs/CPP_DIVERGENCES.md, entry 75.
//
// Method (written here): x^p = cbrt(x) * x^-d with d = 1/3 - p = 2^-54/3. A double guess y0
// of cbrt(x) is refined by one Newton step whose residual y0^3 - x is formed exactly in
// double-double (Dekker's splitting product, no fused multiply-add), giving
// cbrt(x) = y0 + c with |c| of order one ulp; x^-d = 1 - d ln(x) to far below an ulp. The
// result is the single rounding of y0 + (c - y0 d ln x). Only an exact tie at half an ulp
// could round differently from the true value, which does not occur for finite inputs short
// of pathological cases.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>Correctly rounded <c>pow( x , 1./3 )</c> (see the file header).</summary>
public static class PowOneThird
{
	// d = 1/3 - fl(1/3) = 2^-54 / 3 (fl(1/3) = 6004799503160661 * 2^-54).
	private static readonly double Delta = Math.ScaleB(1.0 / 3.0, -54);

	/// <summary>The correctly rounded value of x^(1./3) for x &gt;= 0 (NaN for negative x, like pow).</summary>
	public static double Pow(double x)
	{
		if (double.IsNaN(x) || x < 0)
		{
			return double.NaN;
		}

		if (x == 0 || double.IsPositiveInfinity(x))
		{
			return x;
		}

		// Scale into a range where the splitting products cannot overflow or underflow; scaling
		// by 2^(3k) scales the cube root by exactly 2^k.
		int k = 0;
		int exponent = Math.ILogB(x);
		if (exponent > 600 || exponent < -600)
		{
			k = exponent / 3;
			x = Math.ScaleB(x, -3 * k);
		}

		double y0 = Math.Cbrt(x);

		// r = y0^3 - x in double-double.
		TwoProduct(y0, y0, out double s, out double e);
		TwoProduct(s, y0, out double t, out double f);
		double g = e * y0;
		double r = (t - x) + (f + g);
		double c = -r / (3.0 * y0 * y0);

		// x^p = (y0 + c) * (1 - d ln x); ln x here is of the unscaled x.
		double lnX = Math.Log(x) + (k * 3.0 * Math.Log(2.0));
		double result = y0 + (c - (y0 * Delta * lnX));
		return k == 0 ? result : Math.ScaleB(result, k);
	}

	// Dekker's exact product: a * b = p + err.
	private static void TwoProduct(double a, double b, out double p, out double err)
	{
		p = a * b;
		Split(a, out double ah, out double al);
		Split(b, out double bh, out double bl);
		err = (((ah * bh) - p) + (ah * bl) + (al * bh)) + (al * bl);
	}

	private static void Split(double a, out double hi, out double lo)
	{
		double c = 134217729.0 * a; // 2^27 + 1
		hi = c - (c - a);
		lo = a - hi;
	}
}
