// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchRandom: the random numbers of the CPU PatchMatch kernel, replacing COLMAP's
// cuRAND state per pixel (gpu_mat_prng.cu's curand_init(id, 0, 0) and curand_uniform in
// patch_match_cuda.cu). Written for colmap-sharp; not a COLMAP port
// (divergence 86).
//
// Counter-based: every draw is a pure function of (seed, pixel in the original, unrotated
// reference image, phase, draw index), where the phase names the step that draws (the
// initial depth, the initial normal, or sweep s of iteration i). No state is stored per
// pixel and none has to follow the maps through PatchMatch's 90-degree rotations, and the
// result cannot depend on how pixels are split among threads.
//
// The generator is SplitMix64's output function (Steele, Lea and Flood, "Fast splittable
// pseudorandom number generators", OOPSLA 2014; the constants are the public-domain
// reference ones) applied to a key and a Weyl sequence over the draw index. The key hashes
// the seed, pixel and phase through the same mixer. A 32-bit output maps to a float in
// (0, 1] exactly as cuRAND's curand_uniform does: x * 2^-32 + 2^-33.

namespace ColmapSharp.Mvs;

/// <summary>
/// A stream of uniform floats in (0, 1] for one pixel and one phase of PatchMatch. A value
/// type: each pixel's stream lives on the stack of the loop that draws from it.
/// </summary>
public struct PatchMatchRandom
{
	/// <summary>The seed PatchMatch uses unless told otherwise.</summary>
	public const ulong DefaultSeed = 0;

	/// <summary>Phase of the initial random depth map (FillWithRandomNumbers in COLMAP).</summary>
	public const int InitDepthPhase = -2;

	/// <summary>Phase of the initial random normal map (InitNormalMap in COLMAP).</summary>
	public const int InitNormalPhase = -1;

	private const ulong Golden = 0x9E3779B97F4A7C15UL;

	// cuRAND's CURAND_2POW32_INV, whose literal 2.3283064e-10f rounds to exactly 2^-32 in
	// float; curand_uniform adds half of it as the offset.
	private const float TwoPow32Inv = 2.3283064e-10f;

	private readonly ulong key;
	private ulong counter;

	/// <summary>
	/// The stream of pixel (<paramref name="row"/>, <paramref name="col"/>) of the original
	/// (unrotated) reference image in <paramref name="phase"/>, under <paramref name="seed"/>.
	/// </summary>
	public PatchMatchRandom(ulong seed, int row, int col, int phase)
	{
		ulong pixel = ((ulong)(uint)row << 32) | (uint)col;
		key = Mix(Mix(Mix(unchecked(seed + Golden)) ^ pixel) ^ (ulong)(uint)phase);
		counter = 0;
	}

	/// <summary>The phase of sweep <paramref name="sweep"/> (0..3) of iteration <paramref name="iteration"/>.</summary>
	public static int SweepPhase(int iteration, int sweep) => iteration * 4 + sweep;

	/// <summary>The next uniform float in (0, 1], like curand_uniform.</summary>
	public float NextUniform()
	{
		counter++;
		uint bits = (uint)(Mix(unchecked(key + counter * Golden)) >> 32);
		return bits * TwoPow32Inv + TwoPow32Inv / 2.0f;
	}

	/// <summary>SplitMix64's output mixer (a bijection on 64-bit values).</summary>
	private static ulong Mix(ulong z)
	{
		unchecked
		{
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
			return z ^ (z >> 31);
		}
	}
}
