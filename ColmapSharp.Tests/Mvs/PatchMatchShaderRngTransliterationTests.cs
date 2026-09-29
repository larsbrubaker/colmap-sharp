// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchShaderRngTransliterationTests: C#-only (COLMAP's random numbers come from cuRAND;
// PatchMatchRandom replaces them, divergence 86). A TRANSLITERATION CHECK,
// NOT PRODUCTION CODE: WgslTransliteration below is a line-by-line C# copy of the integer
// functions in ColmapSharp/Mvs/Shaders/patch_match_common.wgsl (u64 on two u32 words, the
// 16-bit-limb multiply, the SplitMix64 mixer, the stream, and the round-to-nearest-even
// u32 -> f32 conversion), run against the production ColmapSharp.Mvs.PatchMatchRandom and C#'s
// own (float)uint conversion. It shows the limb algorithm is right; it cannot show the GPU
// runs the WGSL the same way. That is the real-GPU RNG probe's job (PORTING_PLAN.md Phase 13,
// the conformance kit). Edit the copy whenever the WGSL changes, and keep each line matching.
// Tier A: every result is compared bit for bit.

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchShaderRngTransliterationTests
{
	[Test]
	public async Task RandomStreams_MatchPatchMatchRandom()
	{
		// Seeds and pixels chosen to exercise carries and high words: zero, the default seed,
		// all-ones words, negative rows/cols/phases, and pseudo-random ones.
		var seeds = new List<ulong> { 0, PatchMatchRandom.DefaultSeed, 1, 0xFFFFFFFFUL, 0x100000000UL, ulong.MaxValue, 0x6A09E667F3BCC908UL };
		var picker = new Random(20260927);
		for (int i = 0; i < 8; ++i)
		{
			seeds.Add((ulong)picker.NextInt64() ^ ((ulong)picker.Next() << 40));
		}

		int mismatches = 0;
		int compared = 0;
		foreach (ulong seed in seeds)
		{
			for (int key = 0; key < 400; ++key)
			{
				int row = key < 200 ? key % 23 - 3 : picker.Next(int.MinValue, int.MaxValue);
				int col = key < 200 ? key / 23 : picker.Next(int.MinValue, int.MaxValue);
				int phase = key % 7 - 2;
				var expected = new PatchMatchRandom(seed, row, col, phase);
				WgslTransliteration.Random actual = WgslTransliteration.RandomInit(((uint)seed, (uint)(seed >> 32)), row, col, phase);
				for (int draw = 0; draw < 6; ++draw)
				{
					float e = expected.NextUniform();
					float a = WgslTransliteration.RandomNextUniform(ref actual);
					compared++;
					if (BitConverter.SingleToUInt32Bits(e) != BitConverter.SingleToUInt32Bits(a))
					{
						mismatches++;
					}
				}
			}
		}

		// 15 seeds x 400 keys x 6 draws.
		await Assert.That(compared).IsEqualTo(36000);
		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	public async Task Mul64_MatchesUnsignedWrappingMultiply()
	{
		var picker = new Random(7);
		var cases = new List<(ulong A, ulong B)>
		{
			(ulong.MaxValue, ulong.MaxValue), (0xFFFFFFFFUL, 0xFFFFFFFFUL), (0x9E3779B97F4A7C15UL, 3), (0, ulong.MaxValue),
			(0x0000FFFF0000FFFFUL, 0xFFFF0000FFFF0000UL),
		};
		for (int i = 0; i < 20000; ++i)
		{
			cases.Add(((ulong)picker.NextInt64() * 2 + (ulong)picker.Next(2), (ulong)picker.NextInt64() * 2 + (ulong)picker.Next(2)));
		}

		int mismatches = 0;
		foreach ((ulong a, ulong b) in cases)
		{
			(uint lo, uint hi) = WgslTransliteration.U64Mul(((uint)a, (uint)(a >> 32)), ((uint)b, (uint)(b >> 32)));
			if ((((ulong)hi << 32) | lo) != unchecked(a * b))
			{
				mismatches++;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	public async Task U32ToF32_MatchesCSharpConversion()
	{
		// Every value below 2^25 (exact, then the first rounding binade with its ties), every
		// value near each power of two and each tie above it, and a random sample.
		var values = new List<uint>();
		for (uint v = 0; v < (1u << 25); ++v)
		{
			values.Add(v);
		}

		for (int bit = 24; bit < 32; ++bit)
		{
			uint power = 1u << bit;
			uint half = 1u << (bit - 24);
			for (int d = -300; d <= 300; ++d)
			{
				values.Add(unchecked(power + (uint)d));
				values.Add(unchecked(power + half * 3 + (uint)d));
				values.Add(unchecked(power + half + (uint)d));
			}
		}

		for (uint d = 0; d < 300; ++d)
		{
			values.Add(uint.MaxValue - d);
		}

		var picker = new Random(11);
		for (int i = 0; i < 1_000_000; ++i)
		{
			values.Add((uint)picker.NextInt64(0, 1L << 32));
		}

		int mismatches = 0;
		foreach (uint v in values)
		{
			if (BitConverter.SingleToUInt32Bits(WgslTransliteration.U32ToF32Rne(v)) != BitConverter.SingleToUInt32Bits((float)v))
			{
				mismatches++;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	/// <summary>
	/// Line-by-line copy of patch_match_common.wgsl's integer functions. vec2&lt;u32&gt; is
	/// (X, Y) = (low word, high word); WGSL's u32 arithmetic wraps, as unchecked uint does.
	/// </summary>
	private static class WgslTransliteration
	{
		public static readonly (uint X, uint Y) Golden = (0x7F4A7C15u, 0x9E3779B9u);

		public struct Random
		{
			public (uint X, uint Y) Key;
			public (uint X, uint Y) Counter;
		}

		public static (uint X, uint Y) U64Add((uint X, uint Y) a, (uint X, uint Y) b)
		{
			uint lo = unchecked(a.X + b.X);
			uint carry = lo < a.X ? 1u : 0u;
			return (lo, unchecked(a.Y + b.Y + carry));
		}

		public static (uint X, uint Y) U32MulWide(uint a, uint b)
		{
			uint a0 = a & 0xFFFFu;
			uint a1 = a >> 16;
			uint b0 = b & 0xFFFFu;
			uint b1 = b >> 16;
			uint p00 = a0 * b0;
			uint p01 = a0 * b1;
			uint p10 = a1 * b0;
			uint p11 = a1 * b1;
			uint mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
			uint lo = (p00 & 0xFFFFu) | (mid << 16);
			uint hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
			return (lo, hi);
		}

		public static (uint X, uint Y) U64Mul((uint X, uint Y) a, (uint X, uint Y) b)
		{
			(uint X, uint Y) w = U32MulWide(a.X, b.X);
			return (w.X, unchecked(w.Y + a.X * b.Y + a.Y * b.X));
		}

		public static (uint X, uint Y) U64Shr((uint X, uint Y) a, int n)
			=> ((a.X >> n) | (a.Y << (32 - n)), a.Y >> n);

		public static (uint X, uint Y) Xor((uint X, uint Y) a, (uint X, uint Y) b) => (a.X ^ b.X, a.Y ^ b.Y);

		public static (uint X, uint Y) RandomMix((uint X, uint Y) z0)
		{
			(uint X, uint Y) z = z0;
			z = U64Mul(Xor(z, U64Shr(z, 30)), (0x1CE4E5B9u, 0xBF58476Du));
			z = U64Mul(Xor(z, U64Shr(z, 27)), (0x133111EBu, 0x94D049BBu));
			return Xor(z, U64Shr(z, 31));
		}

		public static Random RandomInit((uint X, uint Y) seed, int row, int col, int phase)
		{
			(uint X, uint Y) pixel = ((uint)col, (uint)row);
			(uint X, uint Y) phase64 = ((uint)phase, 0u);
			(uint X, uint Y) key = RandomMix(Xor(RandomMix(Xor(RandomMix(U64Add(seed, Golden)), pixel)), phase64));
			return new Random { Key = key, Counter = (0u, 0u) };
		}

		public static uint RandomNextBits(ref Random state)
		{
			state.Counter = U64Add(state.Counter, (1u, 0u));
			return RandomMix(U64Add(state.Key, U64Mul(state.Counter, Golden))).Y;
		}

		public static float RandomNextUniform(ref Random state)
		{
			uint bits = RandomNextBits(ref state);
			return U32ToF32Rne(bits) * BitConverter.UInt32BitsToSingle(0x2F800000u) + BitConverter.UInt32BitsToSingle(0x2F000000u);
		}

		public static float U32ToF32Rne(uint v)
		{
			if (v == 0u)
			{
				return 0.0f;
			}

			uint msb = 31u - (uint)System.Numerics.BitOperations.LeadingZeroCount(v);
			uint exponent = (msb + 127u) << 23;
			if (msb <= 23u)
			{
				return BitConverter.UInt32BitsToSingle(exponent | ((v << (int)(23u - msb)) & 0x7FFFFFu));
			}

			uint shift = msb - 23u;
			uint significand = v >> (int)shift;
			uint remainder = v & ((1u << (int)shift) - 1u);
			uint half = 1u << (int)(shift - 1u);
			if (remainder > half || (remainder == half && (significand & 1u) == 1u))
			{
				significand = significand + 1u;
			}

			return BitConverter.UInt32BitsToSingle(exponent + (significand - 0x800000u));
		}
	}
}
