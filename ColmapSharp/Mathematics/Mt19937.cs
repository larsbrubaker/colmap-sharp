// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Mt19937: the 32-bit Mersenne Twister, std::mt19937 as the C++ standard defines it
// ([rand.eng.mers] and [rand.predef]: w=32, n=624, m=397, r=31, a=0x9908b0df, u=11,
// d=0xffffffff, s=7, b=0x9d2c5680, t=15, c=0xefc60000, l=18, f=1812433253). It is the engine
// behind COLMAP's thread-local PRNG (colmap/math/random.h, ported in RandomUtils.cs); the
// distributions that turn its words into numbers are in LibcxxRandom.cs.
//
// Written from the standard's definition (Matsumoto & Nishimura, 1998), not from any
// library's source. The standard pins the output sequence (the 10000th draw of a
// default-seeded engine is 4123659995), so every conforming library - and this class -
// produce the same words for the same seed. Tier A (exact); pinned by RandomTests against
// the libc++ oracle fixture.

namespace ColmapSharp.Mathematics;

/// <summary>
/// The 32-bit Mersenne Twister engine, bit-identical to C++'s <c>std::mt19937</c>.
/// Not thread-safe: each thread uses its own instance (see <see cref="RandomUtils.Prng"/>).
/// </summary>
public sealed class Mt19937
{
	/// <summary>Smallest value <see cref="Next"/> returns (std::mt19937::min()).</summary>
	public const uint Min = 0;

	/// <summary>Largest value <see cref="Next"/> returns (std::mt19937::max()).</summary>
	public const uint Max = uint.MaxValue;

	/// <summary>std::mt19937::default_seed.</summary>
	public const uint DefaultSeed = 5489;

	private const int N = 624;
	private const int M = 397;
	private const uint MatrixA = 0x9908b0df;
	private const uint UpperMask = 0x80000000;
	private const uint LowerMask = 0x7fffffff;

	private readonly uint[] state = new uint[N];
	private int index;

	/// <summary>Creates an engine seeded like <c>std::mt19937(seed)</c>.</summary>
	public Mt19937(uint seed = DefaultSeed)
	{
		Seed(seed);
	}

	/// <summary>Re-seeds the engine like <c>std::mt19937::seed(value)</c>.</summary>
	public void Seed(uint seed)
	{
		state[0] = seed;
		for (int i = 1; i < N; i++)
		{
			uint previous = state[i - 1];
			state[i] = unchecked(1812433253u * (previous ^ (previous >> 30)) + (uint)i);
		}

		index = N;
	}

	/// <summary>Returns the next 32-bit word (<c>std::mt19937::operator()</c>).</summary>
	public uint Next()
	{
		if (index >= N)
		{
			Twist();
		}

		uint y = state[index++];
		y ^= y >> 11;
		y ^= (y << 7) & 0x9d2c5680;
		y ^= (y << 15) & 0xefc60000;
		y ^= y >> 18;
		return y;
	}

	// Regenerates all N words at once. libc++ twists one word per draw instead; the words
	// come out the same either way, because word i only depends on words i+1 and i+M of the
	// previous generation, which the in-place loop has not overwritten yet when it needs them
	// (or has, for i+M >= N, exactly as the recurrence requires).
	private void Twist()
	{
		for (int i = 0; i < N; i++)
		{
			uint y = (state[i] & UpperMask) | (state[(i + 1) % N] & LowerMask);
			uint next = state[(i + M) % N] ^ (y >> 1);
			if ((y & 1) != 0)
			{
				next ^= MatrixA;
			}

			state[i] = next;
		}

		index = 0;
	}
}
