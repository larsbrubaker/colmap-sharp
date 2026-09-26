// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlMathOp: the pieces of thirdparty/VLFeat/mathop.h that SIFT uses - VL_PI/VL_EPSILON_*
// constants, vl_mod_2pi_f, vl_floor_f/_d, vl_fast_atan2_f, vl_fast_resqrt_f and
// vl_fast_sqrt_f - plus sift.c's private fast_expn table. Neighbors: VlImOpv.cs (the column
// convolution) and VlSiftFilter*.cs (the SIFT filter that calls these).
//
// Tier A (exact). These are VLFeat's own approximations, computed in the same precision as
// the C (float stays float, the magic-constant inverse square root reinterprets the same bits).
// Mixed float/double expressions keep the C promotion points; the call sites spell the casts
// out. COLMAP builds VLFeat without SSE2 on arm64 (thirdparty/VLFeat/CMakeLists.txt), so the
// scalar code here is the code the macOS wheel runs.

namespace ColmapSharp.Feature.VLFeat;

/// <summary>Port of the SIFT-relevant parts of VLFeat's mathop.h.</summary>
public static partial class VlMathOp
{
	/// <summary>VL_PI.</summary>
	public const double Pi = 3.141592653589793;

	/// <summary>VL_LOG_OF_2.</summary>
	public const double LogOf2 = 0.693147180559945;

	/// <summary>VL_EPSILON_F (2^-23).</summary>
	public const float EpsilonF = 1.19209290E-07F;

	/// <summary>VL_EPSILON_D (2^-52).</summary>
	public const double EpsilonD = 2.220446049250313e-16;

	/// <summary>Port of vl_mod_2pi_f: x modulo 2*pi for arguments near [0, 2*pi).</summary>
	public static float Mod2PiF(float x)
	{
		const float TwoPi = (float)(2 * Pi);
		while (x > TwoPi)
		{
			x -= TwoPi;
		}

		while (x < 0.0F)
		{
			x += TwoPi;
		}

		return x;
	}

	/// <summary>Port of vl_floor_f (floor, then convert to integer).</summary>
	public static long FloorF(float x)
	{
		long xi = (long)x;
		if (x >= 0 || (float)xi == x)
		{
			return xi;
		}

		return xi - 1;
	}

	/// <summary>Port of vl_floor_d.</summary>
	public static long FloorD(double x)
	{
		long xi = (long)x;
		if (x >= 0 || (double)xi == x)
		{
			return xi;
		}

		return xi - 1;
	}

	/// <summary>Port of vl_fast_atan2_f: a cubic approximation of atan2 (max error ~0.004 rad).</summary>
	public static float FastAtan2F(float y, float x)
	{
		float angle, r;
		const float C3 = 0.1821F;
		const float C1 = 0.9675F;
		float absY = MathF.Abs(y) + EpsilonF;

		if (x >= 0)
		{
			r = (x - absY) / (x + absY);
			angle = (float)(Pi / 4);
		}
		else
		{
			r = (x + absY) / (absY - x);
			angle = (float)(3 * Pi / 4);
		}

		angle += ((C3 * r * r) - C1) * r;
		return (y < 0) ? -angle : angle;
	}

	/// <summary>Port of vl_fast_resqrt_f: the 0x5f3759df inverse square root with two Newton steps.</summary>
	public static float FastResqrtF(float x)
	{
		float xhalf = 0.5f * x;
		int i = BitConverter.SingleToInt32Bits(x);
		i = 0x5f3759df - (i >> 1);
		float u = BitConverter.Int32BitsToSingle(i);
		u = u * (1.5f - (xhalf * u * u));
		u = u * (1.5f - (xhalf * u * u));
		return u;
	}

	/// <summary>Port of vl_fast_sqrt_f: x * resqrt(x), zero below 1e-8.</summary>
	public static float FastSqrtF(float x)
	{
		// `x < 1e-8` compares the float promoted to double with the double literal.
		return ((double)x < 1e-8) ? 0 : x * FastResqrtF(x);
	}

	// sift.c: EXPN_SZ, EXPN_MAX and expn_tab.
	private const int ExpnSize = 256;
	private const double ExpnMax = 25.0;
	private static readonly double[] ExpnTable = CreateExpnTable();

	/// <summary>
	/// Port of sift.c's fast_expn: exp(-x) by linear interpolation in a 257-entry table over
	/// [0, 25]; zero beyond.
	/// </summary>
	public static double FastExpN(double x)
	{
		if (x > ExpnMax)
		{
			return 0.0;
		}

		x *= ExpnSize / ExpnMax;
		int i = (int)FloorD(x);
		double r = x - i;
		double a = ExpnTable[i];
		double b = ExpnTable[i + 1];
		return a + (r * (b - a));
	}

	// Port of sift.c's fast_expn_init.
	private static double[] CreateExpnTable()
	{
		var table = new double[ExpnSize + 1];
		for (int k = 0; k < ExpnSize + 1; ++k)
		{
			table[k] = Math.Exp(-(double)k * (ExpnMax / ExpnSize));
		}

		return table;
	}
}
