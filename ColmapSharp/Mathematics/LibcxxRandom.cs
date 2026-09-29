// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from LLVM libc++ (Apache-2.0 WITH LLVM-exception, see THIRD_PARTY_NOTICES.md):
// <__random/uniform_int_distribution.h> (including __independent_bits_engine),
// <__random/generate_canonical.h>, <__random/uniform_real_distribution.h>,
// <__random/normal_distribution.h>, <__random/log2.h> and <__algorithm/shuffle.h>, as
// shipped in the macOS SDK that Apple clang 21 uses. Modified: translated to C# and
// specialized to std::mt19937 as the engine (Apache-2.0 section 4(b) notice).
//
// LibcxxRandom: the C++ standard leaves the algorithms of std::uniform_int_distribution,
// uniform_real_distribution, normal_distribution and std::shuffle to the implementation, so
// the same mt19937 stream gives different numbers under libc++ and libstdc++. COLMAP's
// oracle (the macOS pycolmap wheel) links libc++ (PORTING_PLAN.md, Phase 1;
// oracle/probe_random.py), so these are libc++'s algorithms, and seeded COLMAP code
// (RANSAC sampling, synthetic datasets) draws the same numbers here. RandomUtils.cs wraps
// them in COLMAP's thread-local PRNG API; Mt19937.cs is the engine.
//
// Tier A (exact), pinned by RandomTests against oracle/fixture_random.py's libc++ harness.
// The integer paths are exact everywhere. The real paths are plain IEEE +,-,*,/ and sqrt,
// which are exact too, with one caveat: NormalDistribution calls log, which is the platform
// libm's in both C++ and .NET (on macOS both call the system libm, so they agree bit for
// bit; elsewhere a last-ulp difference is possible, divergence 114).
//
// Constants below are libc++'s template parameters evaluated for std::mt19937 on macOS,
// where mt19937::result_type (uint_fast32_t) is a 32-bit unsigned int: the engine range
// R = max - min + 1 = 2^32, so __log2 gives m = 32 bits per engine call.

using System.Numerics;

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// libc++'s <c>&lt;random&gt;</c> distributions and <c>std::shuffle</c>, driven by <see cref="Mt19937"/>.
/// </summary>
internal static class LibcxxRandom
{
	// __log2<uint64_t, 2^32>: bits delivered by one mt19937 call.
	private const int EngineBits = 32;

	/// <summary>
	/// <c>std::uniform_int_distribution&lt;T&gt;(a, b)(g)</c>: uniform in [a, b], both inclusive.
	/// Types of at most 32 bits work in uint32 and wider ones in uint64, as libc++ does.
	/// </summary>
	public static T UniformInt<T>(Mt19937 g, T a, T b)
		where T : IBinaryInteger<T>
	{
		int size = a.GetByteCount();
		Check.That(size <= 8, "uniform_int_distribution supports integers of at most 64 bits");
		if (size <= sizeof(uint))
		{
			uint rp = unchecked(uint.CreateTruncating(b) - uint.CreateTruncating(a) + 1u);
			if (rp == 1)
			{
				return a;
			}

			const int dt = 32;
			if (rp == 0)
			{
				// The full 2^32 range: one engine word.
				return T.CreateTruncating(IndependentBits32(g, dt));
			}

			int w = dt - BitOperations.LeadingZeroCount(rp) - 1;
			if ((rp & (uint.MaxValue >> (dt - w))) != 0)
			{
				++w;
			}

			uint u;
			do
			{
				u = IndependentBits32(g, w);
			}
			while (u >= rp);
			return T.CreateTruncating(unchecked(u + uint.CreateTruncating(a)));
		}
		else
		{
			ulong rp = unchecked(ulong.CreateTruncating(b) - ulong.CreateTruncating(a) + 1ul);
			if (rp == 1)
			{
				return a;
			}

			const int dt = 64;
			if (rp == 0)
			{
				return T.CreateTruncating(IndependentBits64(g, dt));
			}

			int w = dt - BitOperations.LeadingZeroCount(rp) - 1;
			if ((rp & (ulong.MaxValue >> (dt - w))) != 0)
			{
				++w;
			}

			ulong u;
			do
			{
				u = IndependentBits64(g, w);
			}
			while (u >= rp);
			return T.CreateTruncating(unchecked(u + ulong.CreateTruncating(a)));
		}
	}

	// __independent_bits_engine<mt19937, uint32_t>(g, w)(). The working type is uint32, in
	// which R = 2^32 wraps to 0, so libc++ takes its __eval(false_type) branch: one engine
	// word, masked to the low w bits (n = 1, w0 = w for every w <= 32).
	private static uint IndependentBits32(Mt19937 g, int w)
	{
		uint mask0 = w > 0 ? uint.MaxValue >> (EngineBits - w) : 0u;
		return g.Next() & mask0;
	}

	// __independent_bits_engine<mt19937, uint64_t>(g, w)(). The working type is uint64 and
	// R = 2^32 != 0, so this is libc++'s constructor plus __eval(true_type), kept literal:
	// the rejection bounds y0/y1 never reject a 32-bit word here, but the shape is the spec.
	private static ulong IndependentBits64(Mt19937 g, int w)
	{
		const ulong r = 1ul << EngineBits;
		const int wdt = 64;
		const int edt = 32;

		int n = (w / EngineBits) + (w % EngineBits != 0 ? 1 : 0);
		int w0 = w / n;
		ulong y0 = w0 < wdt ? (r >> w0) << w0 : 0ul;
		if (r - y0 > y0 / (ulong)n)
		{
			++n;
			w0 = w / n;
			y0 = w0 < wdt ? (r >> w0) << w0 : 0ul;
		}

		int n0 = n - (w % n);
		ulong y1 = w0 < wdt - 1 ? (r >> (w0 + 1)) << (w0 + 1) : 0ul;
		uint mask0 = w0 > 0 ? uint.MaxValue >> (edt - w0) : 0u;
		uint mask1 = w0 < edt - 1 ? uint.MaxValue >> (edt - (w0 + 1)) : uint.MaxValue;

		ulong sp = 0;
		for (int k = 0; k < n0; ++k)
		{
			uint u;
			do
			{
				u = g.Next() - Mt19937.Min;
			}
			while (u >= y0);
			sp = w0 < wdt ? sp << w0 : 0ul;
			sp += u & mask0;
		}

		for (int k = n0; k < n; ++k)
		{
			uint u;
			do
			{
				u = g.Next() - Mt19937.Min;
			}
			while (u >= y1);
			sp = w0 < wdt - 1 ? sp << (w0 + 1) : 0ul;
			sp += u & mask1;
		}

		return sp;
	}

	/// <summary>
	/// <c>std::generate_canonical&lt;T, digits&gt;(g)</c>: a value in [0, 1) built from
	/// ceil(digits / 32) engine words (1 for float, 2 for double). Like libc++, float can
	/// round up to exactly 1.
	/// </summary>
	public static T GenerateCanonical<T>(Mt19937 g)
		where T : IBinaryFloatingPointIeee754<T>
	{
		int dt = T.Zero.GetSignificandBitLength();
		Check.That(dt == 24 || dt == 53, "libc++'s <random> distributions are ported for float and double only");
		int b = dt;
		int k = (b / EngineBits) + (b % EngineBits != 0 ? 1 : 0) + (b == 0 ? 1 : 0);
		T rp = T.CreateTruncating(Mt19937.Max - Mt19937.Min) + T.One;
		T @base = rp;
		T sp = T.CreateTruncating(g.Next() - Mt19937.Min);
		for (int i = 1; i < k; ++i, @base *= rp)
		{
			sp += T.CreateTruncating(g.Next() - Mt19937.Min) * @base;
		}

		return sp / @base;
	}

	/// <summary><c>std::uniform_real_distribution&lt;T&gt;(a, b)(g)</c>: uniform in [a, b).</summary>
	public static T UniformReal<T>(Mt19937 g, T a, T b)
		where T : IBinaryFloatingPointIeee754<T>
	{
		return ((b - a) * GenerateCanonical<T>(g)) + a;
	}

	/// <summary>
	/// <c>std::shuffle(list.begin(), list.end(), g)</c>: libc++'s Fisher-Yates pass, which
	/// draws from <c>uniform_int_distribution&lt;ptrdiff_t&gt;</c> (64-bit) and skips
	/// self-swaps.
	/// </summary>
	public static void Shuffle<T>(IList<T> list, Mt19937 g)
	{
		long d = list.Count;
		if (d > 1)
		{
			int first = 0;
			int last = list.Count - 1;
			for (--d; first < last; ++first, --d)
			{
				long i = UniformInt(g, 0L, d);
				if (i != 0)
				{
					int other = first + (int)i;
					(list[first], list[other]) = (list[other], list[first]);
				}
			}
		}
	}
}

/// <summary>
/// <c>std::normal_distribution&lt;T&gt;</c> as libc++ implements it: the Marsaglia polar method,
/// which makes two normal values per accepted pair and caches the second for the next call.
/// </summary>
internal sealed class NormalDistribution<T>
	where T : IBinaryFloatingPointIeee754<T>
{
	private T cached = T.Zero;
	private bool cachedIsHot;

	/// <summary>Creates a distribution with the given mean and standard deviation.</summary>
	public NormalDistribution(T mean, T stddev)
	{
		Mean = mean;
		StdDev = stddev;
	}

	/// <summary>The distribution's mean.</summary>
	public T Mean { get; }

	/// <summary>The distribution's standard deviation.</summary>
	public T StdDev { get; }

	/// <summary><c>normal_distribution::operator()(g)</c>.</summary>
	public T Next(Mt19937 g)
	{
		T up;
		if (cachedIsHot)
		{
			cachedIsHot = false;
			up = cached;
		}
		else
		{
			T u;
			T v;
			T s;
			do
			{
				u = LibcxxRandom.UniformReal(g, -T.One, T.One);
				v = LibcxxRandom.UniformReal(g, -T.One, T.One);
				s = (u * u) + (v * v);
			}
			while (s > T.One || s == T.Zero);

			T fp = T.Sqrt(T.CreateTruncating(-2) * T.Log(s) / s);
			cached = v * fp;
			cachedIsHot = true;
			up = u * fp;
		}

		return (up * StdDev) + Mean;
	}
}
