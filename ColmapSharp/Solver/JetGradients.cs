// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The gradient storage for Jet<TGrad> (Jet.cs). ceres::Jet<double, N> takes N as a template
// argument; C# has no const generics, so N is carried by a type instead: an [InlineArray(N)]
// struct of N doubles that reports N through a static abstract Length. The JIT specializes
// Jet<TGrad> per gradient type, so Length is a constant in every loop and the storage lives
// inline in the Jet (no heap array, no allocation per operation).
//
// Which sizes exist: every width from 1 to 33. AutoDiffCostFunction (AutoDiffCostFunction.cs)
// can differentiate in chunks of TGrad.Length variables, the way
// ceres::DynamicAutoDiffCostFunction strides, but a single full-width pass is fastest and is
// what Ceres' AutoDiffCostFunction does, so each exact width a COLMAP cost function has is
// provided: 33 is the widest (RigReprojErrorCostFunctor: point 3 + two poses 7 + the
// 16-parameter RAD_TAN_THIN_PRISM_FISHEYE camera), and models_jacobian_test.cc needs
// num_params + 3 for every camera model. The structs are generated boilerplate, one per width.

using System.Runtime.CompilerServices;

namespace ColmapSharp.Solver;

/// <summary>
/// Fixed-size gradient storage for <see cref="Jet{TGrad}"/>: an inline array of
/// <see cref="Length"/> doubles, ceres::Jet's <c>N</c>.
/// </summary>
public interface IJetGradient
{
	/// <summary>The number of derivative slots, Ceres' <c>N</c>.</summary>
	static abstract int Length { get; }
}

/// <summary>1 derivative slot.</summary>
[InlineArray(1)]
public struct Grad1 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 1;
}

/// <summary>2 derivative slots.</summary>
[InlineArray(2)]
public struct Grad2 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 2;
}

/// <summary>3 derivative slots.</summary>
[InlineArray(3)]
public struct Grad3 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 3;
}

/// <summary>4 derivative slots.</summary>
[InlineArray(4)]
public struct Grad4 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 4;
}

/// <summary>5 derivative slots.</summary>
[InlineArray(5)]
public struct Grad5 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 5;
}

/// <summary>6 derivative slots.</summary>
[InlineArray(6)]
public struct Grad6 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 6;
}

/// <summary>7 derivative slots.</summary>
[InlineArray(7)]
public struct Grad7 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 7;
}

/// <summary>8 derivative slots.</summary>
[InlineArray(8)]
public struct Grad8 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 8;
}

/// <summary>9 derivative slots.</summary>
[InlineArray(9)]
public struct Grad9 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 9;
}

/// <summary>10 derivative slots.</summary>
[InlineArray(10)]
public struct Grad10 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 10;
}

/// <summary>11 derivative slots.</summary>
[InlineArray(11)]
public struct Grad11 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 11;
}

/// <summary>12 derivative slots.</summary>
[InlineArray(12)]
public struct Grad12 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 12;
}

/// <summary>13 derivative slots.</summary>
[InlineArray(13)]
public struct Grad13 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 13;
}

/// <summary>14 derivative slots.</summary>
[InlineArray(14)]
public struct Grad14 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 14;
}

/// <summary>15 derivative slots.</summary>
[InlineArray(15)]
public struct Grad15 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 15;
}

/// <summary>16 derivative slots.</summary>
[InlineArray(16)]
public struct Grad16 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 16;
}

/// <summary>17 derivative slots.</summary>
[InlineArray(17)]
public struct Grad17 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 17;
}

/// <summary>18 derivative slots.</summary>
[InlineArray(18)]
public struct Grad18 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 18;
}

/// <summary>19 derivative slots.</summary>
[InlineArray(19)]
public struct Grad19 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 19;
}

/// <summary>20 derivative slots.</summary>
[InlineArray(20)]
public struct Grad20 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 20;
}

/// <summary>21 derivative slots.</summary>
[InlineArray(21)]
public struct Grad21 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 21;
}

/// <summary>22 derivative slots.</summary>
[InlineArray(22)]
public struct Grad22 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 22;
}

/// <summary>23 derivative slots.</summary>
[InlineArray(23)]
public struct Grad23 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 23;
}

/// <summary>24 derivative slots.</summary>
[InlineArray(24)]
public struct Grad24 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 24;
}

/// <summary>25 derivative slots.</summary>
[InlineArray(25)]
public struct Grad25 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 25;
}

/// <summary>26 derivative slots.</summary>
[InlineArray(26)]
public struct Grad26 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 26;
}

/// <summary>27 derivative slots.</summary>
[InlineArray(27)]
public struct Grad27 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 27;
}

/// <summary>28 derivative slots.</summary>
[InlineArray(28)]
public struct Grad28 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 28;
}

/// <summary>29 derivative slots.</summary>
[InlineArray(29)]
public struct Grad29 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 29;
}

/// <summary>30 derivative slots.</summary>
[InlineArray(30)]
public struct Grad30 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 30;
}

/// <summary>31 derivative slots.</summary>
[InlineArray(31)]
public struct Grad31 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 31;
}

/// <summary>32 derivative slots.</summary>
[InlineArray(32)]
public struct Grad32 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 32;
}

/// <summary>33 derivative slots.</summary>
[InlineArray(33)]
public struct Grad33 : IJetGradient
{
	private double element0;

	/// <inheritdoc/>
	public static int Length => 33;
}
