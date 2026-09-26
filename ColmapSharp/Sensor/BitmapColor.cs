// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BitmapColor: the BitmapColor<T> template and internal::BitmapColorCast of
// colmap/sensor/bitmap.h, the pixel value type of Bitmap (Bitmap.cs). COLMAP instantiates
// it for uint8_t (stored pixels) and float (interpolated pixels).
// Tests: ColmapSharp.Tests/Sensor/BitmapColorTests.cs (the BitmapColor cases of bitmap_test.cc).
//
// Tier A. Translation notes:
// - The C++ template over arithmetic T becomes .NET generic math.
// - operator== compares with T's ==, like C++ (so NaN != NaN), not record-struct equality.
// - Cast reproduces BitmapColorCast exactly: std::round (half away from zero), then a clamp
//   to [numeric_limits<D>::min(), numeric_limits<D>::max()] converted to T and evaluated
//   with std::min/std::max's comparison order. For a floating D, numeric_limits::min() is
//   the smallest positive normal, not -max (so COLMAP maps 0 to FLT_MIN); that quirk is kept.
// - ToString matches operator<<: "RGB(r, g, b)", integers as integers and floating point in
//   an ostream's default format (6 significant digits).

using System.Globalization;
using System.Numerics;

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::BitmapColor&lt;T&gt;: an RGB triple. Grey pixels use r (and set all three).
/// </summary>
public readonly struct BitmapColor<T> : IEquatable<BitmapColor<T>>
	where T : unmanaged, INumber<T>
{
	/// <summary>Red (or grey) component.</summary>
	public readonly T R;

	/// <summary>Green component.</summary>
	public readonly T G;

	/// <summary>Blue component.</summary>
	public readonly T B;

	/// <summary>Grey color: all three components set to <paramref name="gray"/>.</summary>
	public BitmapColor(T gray)
	{
		R = gray;
		G = gray;
		B = gray;
	}

	/// <summary>RGB color.</summary>
	public BitmapColor(T r, T g, T b)
	{
		R = r;
		G = g;
		B = b;
	}

	/// <summary>
	/// Port of BitmapColor::Cast&lt;D&gt;: rounds each component and clamps it to D's range.
	/// </summary>
	public BitmapColor<D> Cast<D>()
		where D : unmanaged, INumber<D>, IMinMaxValue<D>
	{
		return new BitmapColor<D>(CastComponent<D>(R), CastComponent<D>(G), CastComponent<D>(B));
	}

	/// <summary>Port of internal::BitmapColorCast&lt;T, D&gt;.</summary>
	private static D CastComponent<D>(T value)
		where D : unmanaged, INumber<D>, IMinMaxValue<D>
	{
		// std::round(value) for an integer T converts to double first; for float it is
		// roundf. Rounding a float in double is exact, so one double path serves both.
		double rounded = Math.Round(double.CreateTruncating(value), MidpointRounding.AwayFromZero);
		double low = double.CreateTruncating(T.CreateTruncating(NumericLimitsMin<D>()));
		double high = double.CreateTruncating(T.CreateTruncating(double.CreateTruncating(D.MaxValue)));
		// std::max(a, b) is (a < b) ? b : a and std::min(a, b) is (b < a) ? b : a, which
		// decides what a NaN turns into (the lower limit).
		double lowered = low < rounded ? rounded : low;
		double clamped = lowered < high ? lowered : high;
		return D.CreateTruncating(clamped);
	}

	/// <summary>
	/// std::numeric_limits&lt;D&gt;::min(): the lowest value for integers, the smallest
	/// positive normal for floating point.
	/// </summary>
	private static double NumericLimitsMin<D>()
		where D : unmanaged, INumber<D>, IMinMaxValue<D>
	{
		if (typeof(D) == typeof(float))
		{
			return 1.17549435E-38f;
		}
		if (typeof(D) == typeof(double))
		{
			return 2.2250738585072014E-308;
		}
		return double.CreateTruncating(D.MinValue);
	}

	/// <inheritdoc/>
	public bool Equals(BitmapColor<T> other) => R == other.R && G == other.G && B == other.B;

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is BitmapColor<T> other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(R, G, B);

	/// <summary>Port of BitmapColor::operator==.</summary>
	public static bool operator ==(BitmapColor<T> lhs, BitmapColor<T> rhs) => lhs.Equals(rhs);

	/// <summary>Port of BitmapColor::operator!=.</summary>
	public static bool operator !=(BitmapColor<T> lhs, BitmapColor<T> rhs) =>
		lhs.R != rhs.R || lhs.G != rhs.G || lhs.B != rhs.B;

	/// <summary>Port of operator&lt;&lt;(ostream, BitmapColor): "RGB(r, g, b)".</summary>
	public override string ToString() => $"RGB({Format(R)}, {Format(G)}, {Format(B)})";

	private static string Format(T value)
	{
		if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
		{
			// An ostream's default floating-point output: %g with precision 6.
			return double.CreateTruncating(value).ToString("G6", CultureInfo.InvariantCulture);
		}
		return value.ToString(null, CultureInfo.InvariantCulture);
	}
}
