// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// FemSystemIntegrator: FEMIntegrator::System< Sigs , IsotropicUIntPack< 3 , D > > with
// BaseFEMIntegrator::System's stencils (thirdparty/PoissonRecon/FEMTree.h and
// FEMTree.System.inl): the bilinear form of the FEM system, sum over k of w[k] times the inner
// product of the k-th derivatives of two basis functions, for same-depth (cc) and parent-child
// (pc) pairs, and the per-depth stencils the multigrid solver (next slices) reads. Solve's
// system is F( { 0 , 1 } ) on the degree-1 Neumann basis: the gradient inner product.
// Tier A against oracle/poisson_fem_harness.cc (the "femsystem" cases).
//
// Translation notes:
// - Upstream the system holds a ScalarConstraint (a Constraint with CDim 1 whose test and
//   constraint functions are the same); the port holds a FemConstraintIntegrator the same way,
//   so the integrals and stencils share its arithmetic.
// - ScalarConstraint's constructor adds w[k] once per ordered sequence of k axis derivatives
//   (so d^2/dxdy and d^2/dydx each contribute), recursively in the C++'s order.
// - The stencils are the IterateFirst forms the solver uses: setStencil<false> and
//   setStencils<true>. With equal test and constraint degrees both overlap starts are the same,
//   so they equal FemConstraintIntegrator's SetStencil and SetParentChildStencils.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The FEM system's bilinear form on one basis and its stencils. Port of PoissonRecon's
/// <c>FEMIntegrator::System</c> for isotropic 3D signatures.
/// </summary>
public sealed class FemSystemIntegrator
{
	private const int Dim = 3;
	private readonly FemConstraintIntegrator integrator;
	private readonly double[] one = new double[1];

	/// <summary>
	/// A system on basis <paramref name="signature"/> with derivatives up to
	/// <paramref name="derivatives"/> per axis, weighting the k-th derivatives by
	/// <paramref name="weights"/>[k]. Port of <c>System( std::initializer_list&lt;double&gt; w )</c>.
	/// </summary>
	public FemSystemIntegrator(int signature, int derivatives, params double[] weights)
	{
		integrator = new FemConstraintIntegrator(signature, derivatives, signature, derivatives, 1);
		if (weights.Length == 0)
		{
			return;
		}

		// DMax = min( TDerivatives , CDerivatives ); weights past it are ignored.
		double[] w = new double[derivatives + 1];
		for (int k = 0; k < w.Length && k < weights.Length; k++)
		{
			w[k] = weights[k];
		}

		Span<int> counts = stackalloc int[Dim];
		SetDerivativeWeights(counts, w, 0, Math.Min(derivatives + 1, weights.Length) - 1);
	}

	/// <summary>The basis' FEM signature (all axes).</summary>
	public int Signature => integrator.TestSignature;

	/// <summary>The depth of the finer functions (Base::highDepth).</summary>
	public int HighDepth => integrator.HighDepth;

	/// <summary>
	/// True when constants are in the system's kernel (the zeroth-derivative weight is 0). Port
	/// of <c>vanishesOnConstants</c>.
	/// </summary>
	public bool VanishesOnConstants => integrator.Weights[0, 0, 0] == 0;

	/// <summary>Tabulates the integrals for functions at <paramref name="depth"/>. Port of <c>init( depth )</c>.</summary>
	public void Init(int depth) => integrator.Init(depth);

	/// <summary>Same-depth form of functions off1 and off2. Port of <c>ccIntegrate</c>.</summary>
	public double CcIntegrate(ReadOnlySpan<int> off1, ReadOnlySpan<int> off2)
	{
		integrator.CcIntegrate(off1, off2, one);
		return one[0];
	}

	/// <summary>Form of function off1 (one depth coarser) and function off2. Port of <c>pcIntegrate</c>.</summary>
	public double PcIntegrate(ReadOnlySpan<int> off1, ReadOnlySpan<int> off2)
	{
		integrator.PcIntegrate(off1, off2, one);
		return one[0];
	}

	/// <summary>
	/// The same-depth stencil around the centered function (OverlapSize^3 entries, axis 0
	/// outermost). Port of <c>setStencil&lt;false&gt;( CCStencil )</c>.
	/// </summary>
	public double[] SetStencil() => integrator.SetStencil();

	/// <summary>
	/// The parent-child stencils, one per child corner (see
	/// <see cref="FemConstraintIntegrator.SetParentChildStencils"/>). Port of
	/// <c>setStencils&lt;true&gt;( PCStencils )</c>.
	/// </summary>
	public double[][] SetParentChildStencils() => integrator.SetParentChildStencils();

	// ScalarConstraint's SetDerivativeWeights: weights[0][idx][idx] += w[0], then recurse once
	// per axis with that axis' derivative count raised.
	private void SetDerivativeWeights(Span<int> counts, double[] w, int wStart, int d)
	{
		int index = FemConstraintIntegrator.DerivativeIndex(integrator.TestDerivatives, counts[0], counts[1], counts[2]);
		integrator.Weights[0, index, index] += w[wStart];
		if (d > 0)
		{
			for (int dd = 0; dd < Dim; dd++)
			{
				counts[dd]++;
				SetDerivativeWeights(counts, w, wStart + 1, d - 1);
				counts[dd]--;
			}
		}
	}
}
