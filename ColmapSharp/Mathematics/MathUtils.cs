// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MathUtils: the scalar helpers of colmap/math/math.h and colmap/math/math.cc -
// sign, clamp, degree/radian conversion, percentile/median/MAD, mean/variance/stddev,
// NextCombination, sigmoid scaling, NChooseK and TruncateCast. It is the first file of
// the Mathematics module (COLMAP's math/, renamed so the namespace does not shadow
// System.Math); random.h, polynomial.h, union_find.h and the graph helpers land next to it
// as their own files. Tests: ColmapSharp.Tests/Mathematics/MathTests.cs (math_test.cc 1:1).
//
// Tier A (exact) for the integer and plain-arithmetic helpers: sign, clamp, degree/radian
// conversion, mean/variance/stddev, NextCombination, NChooseK and TruncateCast are
// bit-identical to COLMAP for the same input. Two exceptions:
// - Sigmoid/ScaleSigmoid go through exp, which is the platform libm's in C++ and .NET's
//   Math.Exp here; they can differ in the last ulp, so they are Tier B (math_test.cc
//   compares them with a tolerance).
// - Percentile/Median/MedianAbsoluteDeviation are exact for ordinary input, but with NaN
//   in the data, or -0.0 and +0.0 tied at the selected rank, this quickselect may pick a
//   different element than libc++'s nth_element (a NaN, or the other signed zero).
//
// Translation notes:
// - C++ templates over arithmetic T become .NET generic math (INumber<T> and friends).
//   Conversions to double go through double.CreateChecked, which for every integer and
//   float type COLMAP instantiates is the same value static_cast<double> gives.
// - Percentile/Median reorder the span in place like COLMAP's std::nth_element does. The
//   exact order left behind differs from libc++'s (it is unspecified there too); the
//   returned value does not (NaN and signed-zero ties aside, above), because it depends
//   only on the order statistics.
// - THROW_CHECK* go through ColmapSharp.Util.Check, which reproduces COLMAP's messages.

using System.Numerics;

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of the free functions in colmap/math/math.h and math.cc.
/// </summary>
public static class MathUtils
{
	/// <summary>
	/// 95% quantile of chi-square distribution with 3 degrees of freedom.
	/// Port of colmap::kChiSquare95ThreeDof.
	/// </summary>
	public const double ChiSquare95ThreeDof = 7.814727903251179;

	/// <summary>
	/// Return 1 if number is positive (including 0), -1 if negative.
	/// COLMAP leaves NaN undefined; here it returns -1.
	/// </summary>
	public static int SignOfNumber<T>(T val)
		where T : INumber<T>
	{
		return val >= T.Zero ? 1 : -1;
	}

	/// <summary>
	/// Clamp the given value to a low and maximum value. Evaluated exactly as
	/// std::max(low, std::min(value, high)), so NaN and inverted bounds behave as in COLMAP.
	/// </summary>
	public static T Clamp<T>(T value, T low, T high)
		where T : IComparisonOperators<T, T, bool>
	{
		// std::min(a, b) is (b < a) ? b : a; std::max(a, b) is (a < b) ? b : a.
		T min = high < value ? high : value;
		return low < min ? min : low;
	}

	/// <summary>Convert angle in degree to radians.</summary>
	public static float DegToRad(float deg)
	{
		return deg * 0.0174532925199432954743716805978692718781530857086181640625f;
	}

	/// <summary>Convert angle in degree to radians.</summary>
	public static double DegToRad(double deg)
	{
		return deg * 0.0174532925199432954743716805978692718781530857086181640625;
	}

	/// <summary>Convert angle in radians to degree.</summary>
	public static float RadToDeg(float rad)
	{
		return rad * 57.29577951308232286464772187173366546630859375f;
	}

	/// <summary>Convert angle in radians to degree.</summary>
	public static double RadToDeg(double rad)
	{
		return rad * 57.29577951308232286464772187173366546630859375;
	}

	/// <summary>
	/// Compute the n-th percentile. Reorders elements in-place.
	/// Performs linear interpolation between values.
	/// </summary>
	/// <exception cref="ArgumentException">The span is empty or p is outside [0, 100].</exception>
	public static double Percentile<T>(Span<T> elems, double p)
		where T : INumber<T>
	{
		Check.That(!elems.IsEmpty);
		Check.Ge(p, 0);
		Check.Le(p, 100);

		double idxDouble = p / 100.0 * (elems.Length - 1);
		double leftIdxDouble = Math.Floor(idxDouble);
		int leftIdx = (int)leftIdxDouble;
		double rightIdxDouble = Math.Ceiling(idxDouble);
		int rightIdx = (int)rightIdxDouble;
		NthElement(elems, rightIdx);
		double right = double.CreateChecked(elems[rightIdx]);
		if (leftIdx == rightIdx)
		{
			return right;
		}

		// After nth_element everything before rightIdx is <= it, so the largest of those is
		// the (rightIdx - 1)-th order statistic.
		T leftValue = elems[0];
		for (int i = 1; i < rightIdx; i++)
		{
			// std::max_element keeps the first of equal maxima; only the value matters here.
			if (leftValue < elems[i])
			{
				leftValue = elems[i];
			}
		}

		double left = double.CreateChecked(leftValue);
		return (rightIdxDouble - idxDouble) * left + (idxDouble - leftIdxDouble) * right;
	}

	/// <summary>
	/// Determine median value. Reorders elements in-place.
	/// Performs linear interpolation between mid values.
	/// </summary>
	public static double Median<T>(Span<T> elems)
		where T : INumber<T>
	{
		return Percentile(elems, 50);
	}

	/// <summary>
	/// Determine median absolute deviation (MAD). Reorders elements in-place.
	/// </summary>
	/// <returns>The median and the MAD.</returns>
	public static (double Median, double Mad) MedianAbsoluteDeviation<T>(Span<T> elems)
		where T : INumber<T>
	{
		double median = Median(elems);
		var absDeviations = new double[elems.Length];
		for (int i = 0; i < elems.Length; i++)
		{
			absDeviations[i] = Math.Abs(double.CreateChecked(elems[i]) - median);
		}

		return (median, Median<double>(absDeviations));
	}

	/// <summary>Determine mean value in a vector.</summary>
	/// <exception cref="ArgumentException">The span is empty.</exception>
	public static double Mean<T>(ReadOnlySpan<T> elems)
		where T : INumber<T>
	{
		Check.That(!elems.IsEmpty);

		double sum = 0;
		foreach (T el in elems)
		{
			sum += double.CreateChecked(el);
		}

		return sum / elems.Length;
	}

	/// <summary>Determine sample variance in a vector.</summary>
	public static double Variance<T>(ReadOnlySpan<T> elems)
		where T : INumber<T>
	{
		double mean = Mean(elems);
		double var = 0;
		foreach (T el in elems)
		{
			double diff = double.CreateChecked(el) - mean;
			var += diff * diff;
		}

		// A single element divides by zero and gives NaN, as in COLMAP.
		return var / (elems.Length - 1);
	}

	/// <summary>Determine sample standard deviation in a vector.</summary>
	public static double StdDev<T>(ReadOnlySpan<T> elems)
		where T : INumber<T>
	{
		return Math.Sqrt(Variance(elems));
	}

	/// <summary>
	/// Generate N-choose-K combinations. The range is <paramref name="elems"/>, and the
	/// current combination is elems[0, middle). Port of colmap::NextCombination(first,
	/// middle, last). Elements must be in sorted order.
	/// </summary>
	/// <returns>False once the sequence wraps back to the first combination.</returns>
	public static bool NextCombination<T>(Span<T> elems, int middle)
		where T : IComparisonOperators<T, T, bool>
	{
		return NextCombination(elems, 0, middle, middle, elems.Length);
	}

	/// <summary>Sigmoid function.</summary>
	public static T Sigmoid<T>(T x)
		where T : IFloatingPointIeee754<T>
	{
		return Sigmoid(x, T.One);
	}

	/// <summary>Sigmoid function.</summary>
	public static T Sigmoid<T>(T x, T alpha)
		where T : IFloatingPointIeee754<T>
	{
		return T.One / (T.One + T.Exp(-x * alpha));
	}

	/// <summary>Scale values according to sigmoid transform with alpha = 1 and x0 = 10.</summary>
	public static T ScaleSigmoid<T>(T x)
		where T : IFloatingPointIeee754<T>
	{
		return ScaleSigmoid(x, T.One, T.CreateChecked(10));
	}

	/// <summary>
	/// Scale values according to sigmoid transform.
	///
	///   x \in [0, 1] -> x \in [-x0, x0] -> sigmoid(x, alpha) -> x \in [0, 1]
	/// </summary>
	/// <param name="x">Value to be scaled in the range [0, 1].</param>
	/// <param name="alpha">Exponential sigmoid factor.</param>
	/// <param name="x0">Spread that determines the range x is scaled to.</param>
	/// <returns>The scaled value in the range [0, 1].</returns>
	public static T ScaleSigmoid<T>(T x, T alpha, T x0)
		where T : IFloatingPointIeee754<T>
	{
		T t0 = Sigmoid(-x0, alpha);
		T t1 = Sigmoid(x0, alpha);
		x = (Sigmoid(T.CreateChecked(2) * x0 * x - x0, alpha) - t0) / (t1 - t0);
		return x;
	}

	/// <summary>
	/// Binomial coefficient or all combinations, defined as n! / ((n - k)! k!).
	/// </summary>
	public static ulong NChooseK(ulong n, ulong k)
	{
		// Implementation based on: https://blog.plover.com/math/choose.html
		if (n == 0 || n < k)
		{
			return 0;
		}

		ulong r = 1;
		for (ulong d = 1; d <= k; ++d)
		{
			// uint64_t arithmetic wraps in C++; unchecked keeps that for huge inputs.
			r = unchecked(r * n--);
			r /= d;
		}

		return r;
	}

	/// <summary>
	/// Cast value from one type to another and truncate instead of overflow, if the
	/// input value is out of range of the output data type.
	/// </summary>
	/// <remarks>
	/// Integer targets only. C++ clamps against std::numeric_limits&lt;T2&gt;::min(), which
	/// for a floating-point T2 is the smallest positive normal (not the lowest value), so a
	/// float target would clamp every negative input up to about 1e-38. .NET's MinValue is
	/// the lowest value, so allowing float targets would silently diverge. Every COLMAP call
	/// site is TruncateCast&lt;float, uint8_t&gt; (or another integer target), so the
	/// constraint costs nothing and makes the divergence unrepresentable.
	/// </remarks>
	public static T2 TruncateCast<T1, T2>(T1 value)
		where T1 : INumber<T1>
		where T2 : IBinaryInteger<T2>, IMinMaxValue<T2>
	{
		// CreateTruncating is static_cast's semantics for every pairing COLMAP uses.
		T1 max = T1.CreateTruncating(T2.MaxValue);
		T1 min = T1.CreateTruncating(T2.MinValue);
		T1 lower = min < value ? value : min;   // std::max(min, value)
		T1 clamped = max < lower ? max : lower; // std::min(max, lower)
		return T2.CreateTruncating(clamped);
	}

	// Port of colmap::internal::NextCombination, with iterators as indices into one span.
	private static bool NextCombination<T>(Span<T> s, int first1, int last1, int first2, int last2)
		where T : IComparisonOperators<T, T, bool>
	{
		if ((first1 == last1) || (first2 == last2))
		{
			return false;
		}

		int m1 = last1 - 1;
		int m2 = last2 - 1;
		while (m1 != first1 && s[m1] >= s[m2])
		{
			--m1;
		}

		bool result = (m1 == first1) && s[first1] >= s[m2];
		if (!result)
		{
			while (first2 != m2 && s[m1] >= s[first2])
			{
				++first2;
			}

			first1 = m1;
			(s[first1], s[first2]) = (s[first2], s[first1]);
			++first1;
			++first2;
		}

		if ((first1 != last1) && (first2 != last2))
		{
			m1 = last1;
			m2 = first2;
			while ((m1 != first1) && (m2 != last2))
			{
				--m1;
				(s[m1], s[m2]) = (s[m2], s[m1]);
				++m2;
			}

			s[first1..m1].Reverse();
			s[first1..last1].Reverse();
			s[m2..last2].Reverse();
			s[first2..last2].Reverse();
		}

		return !result;
	}

	// std::nth_element: afterwards s[n] holds the value a full sort would put there, with
	// nothing greater before it and nothing smaller after it. Quickselect with a
	// median-of-three pivot, falling back to a sort past 2*log2(n) rounds (introselect) so
	// adversarial input stays O(n log n).
	private static void NthElement<T>(Span<T> s, int n)
		where T : INumber<T>
	{
		int lo = 0;
		int hi = s.Length - 1;
		int depthLimit = 2 * (BitOperations.Log2((uint)s.Length) + 1);
		while (hi > lo)
		{
			if (depthLimit-- == 0)
			{
				s[lo..(hi + 1)].Sort();
				return;
			}

			int mid = lo + ((hi - lo) / 2);
			if (s[mid] < s[lo])
			{
				(s[mid], s[lo]) = (s[lo], s[mid]);
			}

			if (s[hi] < s[lo])
			{
				(s[hi], s[lo]) = (s[lo], s[hi]);
			}

			if (s[hi] < s[mid])
			{
				(s[hi], s[mid]) = (s[mid], s[hi]);
			}

			T pivot = s[mid];
			int i = lo;
			int j = hi;
			while (i <= j)
			{
				while (s[i] < pivot)
				{
					i++;
				}

				while (pivot < s[j])
				{
					j--;
				}

				if (i <= j)
				{
					(s[i], s[j]) = (s[j], s[i]);
					i++;
					j--;
				}
			}

			// Now s[lo..j] <= pivot <= s[i..hi], and anything strictly between j and i
			// equals the pivot and is already in its final place.
			if (n <= j)
			{
				hi = j;
			}
			else if (n >= i)
			{
				lo = i;
			}
			else
			{
				return;
			}
		}
	}
}
