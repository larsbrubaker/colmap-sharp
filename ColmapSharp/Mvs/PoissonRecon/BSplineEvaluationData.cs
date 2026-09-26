// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineEvaluationData: BSplineEvaluationData<FEMSig> from thirdparty/PoissonRecon/
// BSplineData.h and .inl - point values and integrals of one basis function, the index
// compression the FEM tree uses (functions away from the boundary are all translates of one
// another, so tables store only IndexSize distinct rows: the Pad + OffsetStart ones at the
// left boundary, one interior representative, and those at the right boundary), the
// two-scale up-sampling coefficients, and the tabulated evaluators (values at cell centers
// and corners, at the same depth or one finer) built on those indices.
// BSplineIntegrationData uses the same index scheme for its inner-product tables.
//
// Tier A: values come from BSplineComponents' polynomials with the C++ operation order.
//
// Translation notes: the class template over FEMSig becomes an object built for a signature
// (For caches one per signature), and the nested evaluator structs become classes holding
// flat arrays. The template parameter D of the evaluators (derivatives tabulated) is a
// constructor argument.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Evaluation of the basis functions of one FEM signature. Port of
/// <c>BSplineEvaluationData&lt;FEMSig&gt;</c>.
/// </summary>
public sealed class BSplineEvaluationData
{
	private static readonly BSplineEvaluationData[] Cache = BuildCache();

	private BSplineEvaluationData(int femSignature)
	{
		FemSig = femSignature;
		Degree = FemSignature.Degree(femSignature);
		Boundary = FemSignature.Boundary(femSignature);
		Sizes = BSplineSupportSizes.For(Degree);
		Pad = PadOf(femSignature);
		OffsetStart = -Sizes.SupportStart;
		OffsetStop = Sizes.SupportEnd + (Degree & 1);
		IndexSize = OffsetStart + OffsetStop + 1 + 2 * Pad;
	}

	/// <summary>The FEM signature.</summary>
	public int FemSig { get; }

	/// <summary>The B-spline degree.</summary>
	public int Degree { get; }

	/// <summary>The boundary type.</summary>
	public BoundaryType Boundary { get; }

	/// <summary>The support sizes of the degree.</summary>
	public BSplineSupportSizes Sizes { get; }

	/// <summary>
	/// Functions beyond each end of [0, Nodes): SupportEnd for free bases (they reach in from
	/// outside), -1 for odd-degree Dirichlet (the boundary functions vanish), else 0.
	/// </summary>
	public int Pad { get; }

	/// <summary>The number of left-boundary offsets tabulated individually.</summary>
	public int OffsetStart { get; }

	/// <summary>The number of right-boundary offsets tabulated individually.</summary>
	public int OffsetStop { get; }

	/// <summary>The number of distinct table rows (boundary functions plus one interior representative).</summary>
	public int IndexSize { get; }

	/// <summary>The evaluation data for a signature (degree 0 to <see cref="FemSignature.MaxDegree"/>).</summary>
	public static BSplineEvaluationData For(int femSignature) => Cache[FemSignature.CheckSignature(femSignature)];

	/// <summary>The first valid offset at a depth. Port of <c>Begin</c>.</summary>
	public static int Begin(int femSignature, int depth) => -PadOf(femSignature);

	/// <summary>One past the last valid offset at a depth. Port of <c>End</c>.</summary>
	public static int End(int femSignature, int depth) =>
		(1 << depth) + (FemSignature.Degree(femSignature) & 1) + PadOf(femSignature);

	/// <summary>Port of <c>Begin</c>.</summary>
	public int BeginAt(int depth) => -Pad;

	/// <summary>Port of <c>End</c>.</summary>
	public int EndAt(int depth) => (1 << depth) + (Degree & 1) + Pad;

	/// <summary>True if no basis function has this offset at this depth. Port of <c>OutOfBounds</c>.</summary>
	public bool OutOfBounds(int depth, int offset) => offset < BeginAt(depth) || offset >= EndAt(depth);

	/// <summary>The table row of an offset. Port of <c>OffsetToIndex</c>.</summary>
	public int OffsetToIndex(int depth, int offset)
	{
		int dim = Sizes.Nodes(depth);
		if (offset < OffsetStart)
		{
			return Pad + offset;
		}
		else if (offset >= dim - OffsetStop)
		{
			return Pad + OffsetStart + 1 + offset - (dim - OffsetStop);
		}
		else
		{
			return Pad + OffsetStart;
		}
	}

	/// <summary>A representative offset of a table row. Port of <c>IndexToOffset</c>.</summary>
	public int IndexToOffset(int depth, int idx) =>
		idx - Pad <= OffsetStart ? idx - Pad : Sizes.Nodes(depth) + Pad - IndexSize + idx;

	/// <summary>
	/// The d-th derivative at s in [0,1] of the function at (depth, off); 0 outside [0,1] or the
	/// support. Port of <c>Value</c>.
	/// </summary>
	public double Value(int depth, int off, double s, int d)
	{
		if (s < 0 || s > 1)
		{
			return 0.0;
		}

		int res = 1 << depth;
		if (OutOfBounds(depth, off))
		{
			return 0;
		}

		var components = new BSplineComponents(FemSig, Degree, depth, off);

		// [NOTE] This is an ugly way to ensure that when s=1 we evaluate using a B-Spline component within the valid range.
		int ii = StdMinMax.StdMax(0, StdMinMax.StdMin(res - 1, (int)Math.Floor(s * res))) - off;
		if (ii < Sizes.SupportStart || ii > Sizes.SupportEnd)
		{
			return 0;
		}

		return d <= Degree ? components[ii - Sizes.SupportStart][d].Evaluate(s) : 0;
	}

	/// <summary>
	/// The integral over [b, e] (clipped to [0,1]) of the d-th derivative of the function at
	/// (depth, off). Port of <c>Integral</c>.
	/// </summary>
	public double Integral(int depth, int off, double b, double e, int d)
	{
		double integral = 0;

		// Check for valid integration bounds
		if (OutOfBounds(depth, off))
		{
			return 0;
		}

		if (b >= e || b >= 1 || e <= 0)
		{
			return 0;
		}

		if (b < 0)
		{
			b = 0;
		}

		if (e > 1)
		{
			e = 1;
		}

		int res = 1 << depth;
		double supportBegin = (double)(off + Sizes.SupportStart) / res;
		double supportEnd = (double)(off + 1 + Sizes.SupportEnd) / res;
		if (b >= supportEnd || e <= supportBegin)
		{
			return 0;
		}

		var components = new BSplineComponents(FemSig, Degree, depth, off);
		for (int i = Sizes.SupportStart; i <= Sizes.SupportEnd; i++)
		{
			// The index of the current cell
			int c = off + i;

			// The bounds of the current cell
			double cellBegin = StdMinMax.StdMax(b, (double)c / res);
			double cellEnd = StdMinMax.StdMin(e, (double)(c + 1) / res);
			if (cellBegin < cellEnd)
			{
				integral += d <= Degree ? components[i - Sizes.SupportStart][d].Integral(cellBegin, cellEnd) : 0;
			}
		}

		return integral;
	}

	/// <summary>
	/// The weights (over 2^Degree) with which the function at (depth, offset) is a combination
	/// of the functions 2*offset + [UpSampleStart, UpSampleEnd] one depth finer, with boundary
	/// reflection. Port of <c>BSplineUpSamplingCoefficients( depth , offset )</c>; element j is
	/// <c>operator[]( j )</c>.
	/// </summary>
	public double[] UpSamplingCoefficients(int depth, int offset)
	{
		// [ 1/8 1/2 3/4 1/2 1/8]
		// [ 1 , 1 ] ->  [ 3/4 , 1/2 , 1/8 ] + [ 1/8 , 1/2 , 3/4 ] = [ 7/8 , 1 , 7/8 ]
		int upSampleSize = Sizes.UpSampleSize;
		int dim = Sizes.Nodes(depth);
		int fineDim = Sizes.Nodes(depth + 1);
		offset = BSplineData.RemapOffset(FemSig, depth, offset, out bool reflect);
		int multiplier = (Boundary == BoundaryType.Dirichlet && reflect) ? -1 : 1;

		// The short-circuit keeps offset % (dim-1) from being evaluated when dim-1 is zero.
		bool useReflected = Boundary != BoundaryType.Free && (Sizes.Inset != 0 || offset % (dim - 1) != 0);
		Span<int> b = stackalloc int[upSampleSize];
		PoissonPolynomial.BinomialCoefficients(Degree + 1, b);
		var coefficients = new int[upSampleSize];

		// Get the array of coefficients, relative to the origin (the C++ offsets a pointer by this).
		int baseOffset = -(2 * offset + Sizes.UpSampleStart);
		for (int i = Sizes.UpSampleStart; i <= Sizes.UpSampleEnd; i++)
		{
			int fineOffset = 2 * offset + i;
			fineOffset = BSplineData.RemapOffset(FemSig, depth + 1, fineOffset, out reflect);
			if (useReflected || !reflect)
			{
				int fineMultiplier = multiplier * ((Boundary == BoundaryType.Dirichlet && reflect) ? -1 : 1);
				coefficients[fineOffset + baseOffset] += b[i - Sizes.UpSampleStart] * fineMultiplier;
			}

			// If we are not inset and we are at the boundary, use the reflection as well
			if (Boundary != BoundaryType.Free && Sizes.Inset == 0 && offset % (dim - 1) != 0 && fineOffset % (fineDim - 1) == 0)
			{
				fineOffset = BSplineData.RemapOffset(FemSig, depth + 1, fineOffset, out reflect);
				int fineMultiplier = multiplier * ((Boundary == BoundaryType.Dirichlet && reflect) ? -1 : 1);
				if (Boundary == BoundaryType.Dirichlet)
				{
					fineMultiplier *= -1;
				}

				coefficients[fineOffset + baseOffset] += b[i - Sizes.UpSampleStart] * fineMultiplier;
			}
		}

		var result = new double[upSampleSize];
		for (int j = 0; j < upSampleSize; j++)
		{
			result[j] = (double)coefficients[j] / (1 << Degree);
		}

		return result;
	}

	private static int PadOf(int femSignature)
	{
		int degree = FemSignature.Degree(femSignature);
		BoundaryType boundary = FemSignature.Boundary(femSignature);
		if (boundary == BoundaryType.Free)
		{
			return BSplineSupportSizes.For(degree).SupportEnd;
		}

		return ((degree & 1) != 0 && boundary == BoundaryType.Dirichlet) ? -1 : 0;
	}

	private static BSplineEvaluationData[] BuildCache()
	{
		var cache = new BSplineEvaluationData[(FemSignature.MaxDegree + 1) * FemSignature.BoundaryCount];
		for (int sig = 0; sig < cache.Length; sig++)
		{
			cache[sig] = new BSplineEvaluationData(sig);
		}

		return cache;
	}
}
