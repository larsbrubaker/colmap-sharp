// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// RestrictionProlongation: FEMIntegrator::RestrictionProlongation< Sigs > with
// BaseFEMIntegrator::RestrictionProlongation's stencils (thirdparty/PoissonRecon/FEMTree.h and
// FEMTree.System.inl) for isotropic 3D signatures: the two-scale coefficient of a fine function
// in a coarse one (the product of the per-axis BSplineUpSampleEvaluator values), the
// up-sample stencil (the fine functions a centered coarse function refines into) and the
// down-sample stencils (per child corner, the coarse functions a fine function restricts to).
// The multigrid solver (next slices) prolongs coarse solutions and restricts residuals with
// them. Tier A against oracle/poisson_fem_harness.cc (the "restrictionprolongation" cases).
//
// Translation notes:
// - The coefficient multiplies from the last axis: ((1 * v2) * v1) * v0, as the C++ recursion
//   _coefficient does.
// - Stencils are flat double arrays, axis 0 outermost; the down-sample stencils are indexed by
//   corner ((c0 * 2) + c1) * 2 + c2 with the corner bits reversed onto the axes, as the C++'s
//   outer loop does (the same layout as FemConstraintIntegrator.SetParentChildStencils).

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Two-scale coefficients between adjacent depths and their stencils. Port of PoissonRecon's
/// <c>FEMIntegrator::RestrictionProlongation</c> for isotropic 3D signatures.
/// </summary>
public sealed class RestrictionProlongation
{
	private const int Dim = 3;
	private readonly BSplineSupportSizes sizes;
	private BSplineUpSampleEvaluator? upSampler;

	/// <summary>Restriction and prolongation for basis <paramref name="signature"/> (all axes).</summary>
	public RestrictionProlongation(int signature)
	{
		Signature = FemSignature.CheckSignature(signature);
		sizes = BSplineSupportSizes.For(FemSignature.Degree(signature));
	}

	/// <summary>The basis' FEM signature.</summary>
	public int Signature { get; }

	/// <summary>The fine depth (Base::highDepth).</summary>
	public int HighDepth { get; private set; }

	/// <summary>The up-sample stencil's width per axis (BSplineSupportSizes::UpSampleSize).</summary>
	public int UpSampleSize => sizes.UpSampleSize;

	/// <summary>The down-sample stencils' width per axis (-DownSample0Start + DownSample1End + 1).</summary>
	public int DownSampleSize => -sizes.DownSample0Start + sizes.DownSample1End + 1;

	/// <summary>
	/// Tabulates the two-scale coefficients from <paramref name="highDepth"/> - 1 to
	/// <paramref name="highDepth"/>. Port of <c>init( depth )</c>. All three axes share one
	/// evaluator, since the signature is isotropic.
	/// </summary>
	public void Init(int highDepth)
	{
		HighDepth = highDepth;
		upSampler = new BSplineUpSampleEvaluator(Signature, highDepth - 1);
	}

	/// <summary>
	/// The weight of fine function cOff in coarse function pOff. Port of <c>upSampleCoefficient</c>.
	/// </summary>
	public double UpSampleCoefficient(ReadOnlySpan<int> pOff, ReadOnlySpan<int> cOff)
	{
		BSplineUpSampleEvaluator evaluator = upSampler ?? throw new InvalidOperationException("Call Init before evaluating.");
		double coefficient = 1.0;
		for (int d = Dim - 1; d >= 0; d--)
		{
			coefficient = coefficient * evaluator.Value(pOff[d], cOff[d]);
		}

		return coefficient;
	}

	/// <summary>
	/// The coefficients of the fine functions highCenter + UpSampleStart + i in the centered coarse
	/// function highCenter / 2. Port of <c>setStencil( UpSampleStencil )</c>.
	/// </summary>
	public double[] SetStencil()
	{
		int n = UpSampleSize;
		int highCenter = (1 << HighDepth) >> 1;
		var stencil = new double[n * n * n];
		Span<int> pOff = [highCenter / 2, highCenter / 2, highCenter / 2];
		Span<int> cOff = stackalloc int[Dim];
		for (int i0 = 0; i0 < n; i0++)
		{
			cOff[0] = i0 + highCenter + sizes.UpSampleStart;
			for (int i1 = 0; i1 < n; i1++)
			{
				cOff[1] = i1 + highCenter + sizes.UpSampleStart;
				for (int i2 = 0; i2 < n; i2++)
				{
					cOff[2] = i2 + highCenter + sizes.UpSampleStart;
					stencil[((i0 * n) + i1) * n + i2] = UpSampleCoefficient(pOff, cOff);
				}
			}
		}

		return stencil;
	}

	/// <summary>
	/// Per child corner, the coefficients of the fine function at that corner of the brood in the
	/// coarse functions cOff / 2 + DownSample0Start + i. Port of <c>setStencils( DownSampleStencils )</c>.
	/// </summary>
	public double[][] SetStencils()
	{
		int n = DownSampleSize;

		// [NOTE] We want the center to be at the first node of the brood, which is not the case when childDepth is 1.
		int highCenter = ((1 << HighDepth) >> 1 >> 1) << 1;
		var stencils = new double[8][];
		Span<int> cOff = stackalloc int[Dim];
		Span<int> pOff = stackalloc int[Dim];
		for (int o0 = 0; o0 < 2; o0++)
		{
			cOff[2] = o0 + highCenter;
			for (int o1 = 0; o1 < 2; o1++)
			{
				cOff[1] = o1 + highCenter;
				for (int o2 = 0; o2 < 2; o2++)
				{
					cOff[0] = o2 + highCenter;
					var stencil = new double[n * n * n];
					for (int i0 = 0; i0 < n; i0++)
					{
						pOff[0] = cOff[0] / 2 + i0 + sizes.DownSample0Start;
						for (int i1 = 0; i1 < n; i1++)
						{
							pOff[1] = cOff[1] / 2 + i1 + sizes.DownSample0Start;
							for (int i2 = 0; i2 < n; i2++)
							{
								pOff[2] = cOff[2] / 2 + i2 + sizes.DownSample0Start;
								stencil[((i0 * n) + i1) * n + i2] = UpSampleCoefficient(pOff, cOff);
							}
						}
					}

					stencils[((o0 * 2) + o1) * 2 + o2] = stencil;
				}
			}
		}

		return stencils;
	}
}
