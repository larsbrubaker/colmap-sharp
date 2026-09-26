// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Threading: the free functions of colmap/util/threading.h that the port needs. COLMAP's
// ThreadPool itself becomes Parallel.For at each call site (CLAUDE.md, translation rules).

namespace ColmapSharp.Util;

/// <summary>Helpers from colmap/util/threading.h.</summary>
public static class Threading
{
	/// <summary>
	/// Port of colmap::GetEffectiveNumThreads: a non-positive count means all hardware
	/// threads, and the result is at least 1.
	/// </summary>
	public static int GetEffectiveNumThreads(int numThreads)
	{
		int numEffectiveThreads = numThreads;
		if (numThreads <= 0)
		{
			numEffectiveThreads = Environment.ProcessorCount;
		}

		if (numEffectiveThreads <= 0)
		{
			numEffectiveThreads = 1;
		}

		return numEffectiveThreads;
	}
}
