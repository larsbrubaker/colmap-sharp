// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineData: BSplineData<FEMSig,D> from thirdparty/PoissonRecon/BSplineData.h and .inl -
// the explicit polynomials of one B-spline basis function (BSplineComponents: one polynomial
// per cell of its support, plus D derivatives), the boundary remapping of offsets
// (RemapOffset), and the per-depth SparseBSplineEvaluator the FEM tree uses to evaluate
// basis functions at points: only the functions touching the two boundaries differ, so it
// stores Degree + 1 of them at each end and one shifted interior function. Used by
// BSplineEvaluationData (Value/Integral) and, in later slices, FEMTree's point evaluation.
//
// Tier A: the polynomials are built with the same operations in the same order as the C++
// (oracle/poisson_bspline_harness.cc).
//
// Translation notes: the template parameters FEMSig and D (the number of stored derivatives)
// are constructor arguments. Every stored polynomial has degree Degree, as in the C++, where
// derivatives are assigned into Polynomial<Degree> slots (zero-padded); this keeps the
// Horner evaluation the same operation sequence.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The polynomial pieces of one basis function, left to right over its support, with
/// derivatives. Port of <c>BSplineData&lt;FEMSig,D&gt;::BSplineComponents</c>.
/// </summary>
public sealed class BSplineComponents
{
	// _polys[cell][derivative]
	private readonly PoissonPolynomial[][] polys;

	/// <summary>The components of the function at (depth, offset) for a signature, with D derivatives.</summary>
	public BSplineComponents(int femSignature, int derivatives, int depth, int offset)
	{
		int degree = FemSignature.Degree(femSignature);
		int res = 1 << depth;
		var elements = new BSplineElements(degree, res, offset, FemSignature.Boundary(femSignature));
		BSplineSupportSizes sizes = BSplineSupportSizes.For(degree);

		// The first index is the position, the second is the element type
		var components = new PoissonPolynomial[degree + 1, degree + 1];

		// Generate the elements that can appear in the base function corresponding to the base function at (depth,offset) = (0,0)
		for (int d = 0; d <= degree; d++)
		{
			for (int dd = 0; dd <= degree; dd++)
			{
				components[d, dd] = PoissonPolynomial.BSplineComponent(degree, degree - dd).Shift(-((degree + 1) / 2) + d);
			}
		}

		// Now adjust to the desired depth and offset
		double width = 1.0 / res;
		for (int d = 0; d <= degree; d++)
		{
			for (int dd = 0; dd <= degree; dd++)
			{
				components[d, dd] = components[d, dd].Scale(width).Shift(width * offset);
			}
		}

		// Now write in the polynomials
		polys = new PoissonPolynomial[degree + 1][];
		for (int d = 0; d <= degree; d++)
		{
			polys[d] = new PoissonPolynomial[derivatives + 1];
			int idx = offset + sizes.SupportStart + d;
			polys[d][0] = new PoissonPolynomial(degree);
			if (idx >= 0 && idx < res)
			{
				for (int dd = 0; dd <= degree; dd++)
				{
					polys[d][0].AddInPlace(components[d, dd].Multiply((double)elements[idx][dd]).Divide(elements.Denominator));
				}
			}
		}

		for (int d = 1; d <= derivatives; d++)
		{
			for (int dd = 0; dd <= degree; dd++)
			{
				polys[dd][d] = polys[dd][d - 1].Derivative().WithDegree(degree);
			}
		}
	}

	/// <summary>
	/// The polynomials (value, then derivatives) on the <paramref name="cell"/>-th cell of the
	/// support, counted from SupportStart. Port of <c>operator[]</c>.
	/// </summary>
	public PoissonPolynomial[] this[int cell] => polys[cell];
}

/// <summary>
/// Per-depth evaluators for one basis signature. Port of <c>BSplineData&lt;FEMSig,D&gt;</c>.
/// </summary>
public sealed class BSplineData
{
	private readonly SparseBSplineEvaluator[] evaluators;

	/// <summary>Evaluators for depths 0..maxDepth. Port of <c>BSplineData( int maxDepth )</c>.</summary>
	public BSplineData(int femSignature, int derivatives, int maxDepth)
	{
		FemSig = femSignature;
		Derivatives = derivatives;
		evaluators = new SparseBSplineEvaluator[maxDepth + 1];
		for (int d = 0; d <= maxDepth; d++)
		{
			evaluators[d] = new SparseBSplineEvaluator(femSignature, derivatives, d);
		}
	}

	/// <summary>The FEM signature of the basis.</summary>
	public int FemSig { get; }

	/// <summary>The number of derivatives stored (template parameter D).</summary>
	public int Derivatives { get; }

	/// <summary>The evaluator for a depth.</summary>
	public SparseBSplineEvaluator this[int depth] => evaluators[depth];

	/// <summary>
	/// Maps an offset outside the valid range back into it by reflection across the boundary
	/// (periodic with period 2*(dim-1+I)); <paramref name="reflect"/> reports whether it was
	/// mirrored. Free bases are returned unchanged. Port of <c>BSplineData::RemapOffset</c>.
	/// </summary>
	public static int RemapOffset(int femSignature, int depth, int offset, out bool reflect)
	{
		int degree = FemSignature.Degree(femSignature);
		int inset = (degree & 1) != 0 ? 0 : 1;
		if (FemSignature.Boundary(femSignature) == BoundaryType.Free)
		{
			reflect = false;
			return offset;
		}

		int neumann = FemSignature.Of(degree, BoundaryType.Neumann);
		int dim = BSplineEvaluationData.End(neumann, depth) - BSplineEvaluationData.Begin(neumann, depth);
		offset = Modulo(offset, 2 * (dim - 1 + inset));
		reflect = offset >= dim;
		return reflect ? 2 * (dim - 1 + inset) - (offset + inset) : offset;
	}

	// PR_MODULO: a non-negative remainder.
	private static int Modulo(int a, int b) => a < 0 ? (b - ((-a) % b)) % b : a % b;
}

/// <summary>
/// Evaluates the basis functions of one depth at points. Port of
/// <c>BSplineData::SparseBSplineEvaluator</c>.
/// </summary>
public sealed class SparseBSplineEvaluator
{
	private readonly BSplineComponents[] preComponents;
	private readonly BSplineComponents[] postComponents;
	private readonly BSplineComponents centerComponents;
	private readonly int preStart;
	private readonly int preEnd;
	private readonly int postStart;
	private readonly int postEnd;
	private readonly int centerIndex;
	private readonly int depth;
	private readonly double width;
	private readonly int leftSupportRadius;

	/// <summary>Port of <c>SparseBSplineEvaluator::init( depth )</c>.</summary>
	public SparseBSplineEvaluator(int femSignature, int derivatives, int depth)
	{
		int degree = FemSignature.Degree(femSignature);
		BSplineSupportSizes sizes = BSplineSupportSizes.For(degree);
		leftSupportRadius = -sizes.SupportStart;
		this.depth = depth;
		width = 1.0 / (1 << depth);

		// _preStart + BSplineSupportSizes< _Degree >::SupportEnd >=0
		preStart = -sizes.SupportEnd;

		// _postStart + BSplineSupportSizes< _Degree >::SupportEnd <= (1<<depth)-1
		postStart = (1 << depth) - 1 - sizes.SupportEnd;
		preEnd = preStart + degree + 1;
		postEnd = postStart + degree + 1;
		centerIndex = ((preStart + degree + 1) + (postStart - 1)) / 2;
		centerComponents = new BSplineComponents(femSignature, derivatives, depth, centerIndex);
		preComponents = new BSplineComponents[degree + 1];
		postComponents = new BSplineComponents[degree + 1];
		for (int i = 0; i <= degree; i++)
		{
			preComponents[i] = new BSplineComponents(femSignature, derivatives, depth, preStart + i);
			postComponents[i] = new BSplineComponents(femSignature, derivatives, depth, postStart + i);
		}
	}

	/// <summary>The d-th derivative of function fIdx at p. Port of <c>value( p , fIdx , d )</c>.</summary>
	public double Value(double p, int fIdx, int d) => Value(p, (int)(p * (1 << depth)), fIdx, d);

	/// <summary>
	/// The d-th derivative of function fIdx at p, which lies in cell pIdx. Port of
	/// <c>value( p , pIdx , fIdx , d )</c>.
	/// </summary>
	public double Value(double p, int pIdx, int fIdx, int d)
	{
		if (fIdx < preStart)
		{
			return 0;
		}
		else if (fIdx < preEnd)
		{
			return preComponents[fIdx - preStart][pIdx - fIdx + leftSupportRadius][d].Evaluate(p);
		}
		else if (fIdx < postStart)
		{
			return centerComponents[pIdx - fIdx + leftSupportRadius][d].Evaluate(p + width * (centerIndex - fIdx));
		}
		else if (fIdx < postEnd)
		{
			return postComponents[fIdx - postStart][pIdx - fIdx + leftSupportRadius][d].Evaluate(p);
		}
		else
		{
			return 0;
		}
	}

	/// <summary>
	/// The polynomials (value and derivatives) of function fIdx on the cell containing p, with p
	/// shifted into the stored function's frame. Port of <c>polynomialsAndOffset( p , fIdx )</c>.
	/// </summary>
	public PoissonPolynomial[] PolynomialsAndOffset(ref double p, int fIdx) =>
		PolynomialsAndOffset(ref p, (int)(p * (1 << depth)), fIdx);

	/// <summary>Port of <c>polynomialsAndOffset( p , pIdx , fIdx )</c>.</summary>
	public PoissonPolynomial[] PolynomialsAndOffset(ref double p, int pIdx, int fIdx)
	{
		if (fIdx < preEnd)
		{
			return preComponents[fIdx - preStart][pIdx - fIdx + leftSupportRadius];
		}
		else if (fIdx < postStart)
		{
			p += width * (centerIndex - fIdx);
			return centerComponents[pIdx - fIdx + leftSupportRadius];
		}
		else
		{
			return postComponents[fIdx - postStart][pIdx - fIdx + leftSupportRadius];
		}
	}
}
