// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// FemConstraintIntegrator: FEMIntegrator::Constraint< TSigs , TDerivatives , CSigs ,
// CDerivatives , CDim > with BaseFEMIntegrator::Constraint's stencils (thirdparty/PoissonRecon/
// FEMTree.h and FEMTree.System.inl) for isotropic 3D signatures: the integral of weighted
// products of derivatives of a test function and a constraint function, as a CDim-vector, for
// same-depth (cc), parent-child (pc) and child-parent (cp) pairs, built from the 1D
// BSplineIntegrator tables as tensor products; and the per-depth stencils (the integrals around
// a centered function, which are the same for every interior function) the constraint and
// system assembly (next slices) read instead of integrating per node. Solve's use is the
// divergence of the normal field: test signature 5 (degree 1, Neumann) with one derivative,
// constraint signature 7 (degree 2, Dirichlet) with none, CDim 3, weights[d][e_d][0] = 1.
// Tier A against oracle/poisson_fem_harness.cc (the "femconstraint" cases).
//
// Translation notes:
// - Derivative multi-indices follow TensorDerivatives< UIntPack< D , D , D > >: index
//   ((d0 * (D+1)) + d1) * (D+1) + d2.
// - The 1D factors multiply from the last axis: dot0 * (dot1 * (dot2 * 1.0)), as the C++
//   recursion _integral does, and _integrate accumulates weight * product per (test, constraint)
//   derivative pair with a positive weight, in the C++'s pair order.
// - Before the first init( depth ) with depth > 0 the parent-child tables are the
//   C++'s default-constructed (all zero) ones and the port returns 0; init( 0 ) keeps the last.
// - Stencils are flat double arrays (CDim values per window entry, axis 0 outermost), per the
//   performance direction for the solver slices.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Weighted integrals of test-function and constraint-function derivatives, and their stencils.
/// Port of PoissonRecon's <c>FEMIntegrator::Constraint</c> for isotropic 3D signatures.
/// </summary>
public sealed class FemConstraintIntegrator
{
	private const int Dim = 3;
	private readonly List<(int D1, int D2, List<(int C, double Weight)> Indices)> weightedIndices = [];
	private BSplineIntegrator? cc;
	private BSplineIntegrator? pc;
	private BSplineIntegrator? cp;

	/// <summary>A constraint with zero weights. Set <see cref="Weights"/>, then call <see cref="Init"/>.</summary>
	public FemConstraintIntegrator(int testSignature, int testDerivatives, int constraintSignature, int constraintDerivatives, int cDim)
	{
		TestSignature = FemSignature.CheckSignature(testSignature);
		ConstraintSignature = FemSignature.CheckSignature(constraintSignature);
		TestDerivatives = testDerivatives;
		ConstraintDerivatives = constraintDerivatives;
		CDim = cDim;
		TestDerivativeSize = Cube(testDerivatives + 1);
		ConstraintDerivativeSize = Cube(constraintDerivatives + 1);
		Weights = new double[cDim, TestDerivativeSize, ConstraintDerivativeSize];
		BSplineOverlapSizes overlap = BSplineOverlapSizes.For(FemSignature.Degree(testSignature), FemSignature.Degree(constraintSignature));
		OverlapSize = overlap.OverlapSize;
	}

	/// <summary>The test functions' FEM signature (all axes).</summary>
	public int TestSignature { get; }

	/// <summary>The constraint functions' FEM signature (all axes).</summary>
	public int ConstraintSignature { get; }

	/// <summary>The derivatives taken of the test functions, per axis.</summary>
	public int TestDerivatives { get; }

	/// <summary>The derivatives taken of the constraint functions, per axis.</summary>
	public int ConstraintDerivatives { get; }

	/// <summary>The number of result components.</summary>
	public int CDim { get; }

	/// <summary>TensorDerivatives::Size of the test derivatives.</summary>
	public int TestDerivativeSize { get; }

	/// <summary>TensorDerivatives::Size of the constraint derivatives.</summary>
	public int ConstraintDerivativeSize { get; }

	/// <summary>The same-depth stencil width per axis (BSplineOverlapSizes&lt;TDegree,CDegree&gt;::OverlapSize).</summary>
	public int OverlapSize { get; }

	/// <summary>The depth of the test functions (Base::highDepth).</summary>
	public int HighDepth { get; private set; }

	/// <summary>weights[c, d1, d2]: the weight of test derivative d1 against constraint derivative d2 in component c.</summary>
	public double[,,] Weights { get; }

	/// <summary>The TensorDerivatives index of per-axis derivative counts.</summary>
	public static int DerivativeIndex(int derivatives, int d0, int d1, int d2) =>
		((d0 * (derivatives + 1)) + d1) * (derivatives + 1) + d2;

	/// <summary>
	/// Tabulates the 1D integrals for test functions at <paramref name="depth"/> and collects the
	/// positive weights. Port of <c>init( depth )</c> followed by <c>init()</c>.
	/// </summary>
	public void Init(int depth)
	{
		HighDepth = depth;
		cc = new BSplineIntegrator(TestSignature, ConstraintSignature, TestDerivatives, ConstraintDerivatives, depth, child: false);
		// The C++ sets the parent-child tables only for depth > 0 and otherwise keeps the previous
		// ones (all zero on a fresh integrator); the assembly loops count depths down, so depth 0
		// reuses depth 1's tables.
		if (depth > 0)
		{
			pc = new BSplineIntegrator(TestSignature, ConstraintSignature, TestDerivatives, ConstraintDerivatives, depth - 1, child: true);
			cp = new BSplineIntegrator(ConstraintSignature, TestSignature, ConstraintDerivatives, TestDerivatives, depth - 1, child: true);
		}
		weightedIndices.Clear();
		for (int d1 = 0; d1 < TestDerivativeSize; d1++)
		{
			for (int d2 = 0; d2 < ConstraintDerivativeSize; d2++)
			{
				var indices = new List<(int, double)>();
				for (int c = 0; c < CDim; c++)
				{
					if (Weights[c, d1, d2] > 0)
					{
						indices.Add((c, Weights[c, d1, d2]));
					}
				}

				if (indices.Count > 0)
				{
					weightedIndices.Add((d1, d2, indices));
				}
			}
		}
	}

	/// <summary>Same-depth integral of test function off1 against constraint function off2. Port of <c>ccIntegrate</c>.</summary>
	public void CcIntegrate(ReadOnlySpan<int> off1, ReadOnlySpan<int> off2, Span<double> result) => Integrate(0, off1, off2, result);

	/// <summary>Test function off1 (one depth coarser) against constraint function off2. Port of <c>pcIntegrate</c>.</summary>
	public void PcIntegrate(ReadOnlySpan<int> off1, ReadOnlySpan<int> off2, Span<double> result) => Integrate(1, off1, off2, result);

	/// <summary>Test function off1 against constraint function off2 (one depth coarser). Port of <c>cpIntegrate</c>.</summary>
	public void CpIntegrate(ReadOnlySpan<int> off1, ReadOnlySpan<int> off2, Span<double> result) => Integrate(2, off1, off2, result);

	/// <summary>
	/// The same-depth stencil around the centered test function: entry ((i0 * n) + i1) * n + i2
	/// (n = OverlapSize) holds the CDim integrals against constraint function center +
	/// overlapStart + i. Port of <c>setStencil&lt;false&gt;( CCStencil )</c>.
	/// </summary>
	public double[] SetStencil()
	{
		int n = OverlapSize;
		int center = (1 << HighDepth) >> 1;
		int overlapStart = BSplineOverlapSizes.For(FemSignature.Degree(TestSignature), FemSignature.Degree(ConstraintSignature)).OverlapStart;
		var stencil = new double[n * n * n * CDim];
		Span<int> femOffset = [center, center, center];
		Span<int> cOffset = stackalloc int[Dim];
		for (int i0 = 0; i0 < n; i0++)
		{
			cOffset[0] = i0 + center + overlapStart;
			for (int i1 = 0; i1 < n; i1++)
			{
				cOffset[1] = i1 + center + overlapStart;
				for (int i2 = 0; i2 < n; i2++)
				{
					cOffset[2] = i2 + center + overlapStart;
					CcIntegrate(femOffset, cOffset, stencil.AsSpan((((i0 * n) + i1) * n + i2) * CDim, CDim));
				}
			}
		}

		return stencil;
	}

	/// <summary>
	/// The parent-child stencils, one per child corner c (index ((c0 * 2) + c1) * 2 + c2, with
	/// corner bits reversed onto the axes as the C++'s outer loop does): entry i holds the CDim
	/// integrals of test function center/2 + overlapStart + i against the child at the corner.
	/// Port of <c>setStencils&lt;true&gt;( PCStencils )</c>.
	/// </summary>
	public double[][] SetParentChildStencils()
	{
		int n = OverlapSize;

		// [NOTE] We want the center to be at the first node of the brood, which is not the case when childDepth is 1.
		int center = ((1 << HighDepth) >> 1 >> 1) << 1;
		int overlapStart = BSplineOverlapSizes.For(FemSignature.Degree(ConstraintSignature), FemSignature.Degree(TestSignature)).OverlapStart;
		var stencils = new double[8][];
		Span<int> fineCenter = stackalloc int[Dim];
		Span<int> femOffset = stackalloc int[Dim];
		for (int o0 = 0; o0 < 2; o0++)
		{
			fineCenter[2] = o0 + center;
			for (int o1 = 0; o1 < 2; o1++)
			{
				fineCenter[1] = o1 + center;
				for (int o2 = 0; o2 < 2; o2++)
				{
					fineCenter[0] = o2 + center;
					var stencil = new double[n * n * n * CDim];
					for (int i0 = 0; i0 < n; i0++)
					{
						femOffset[0] = i0 + (center / 2) + overlapStart;
						for (int i1 = 0; i1 < n; i1++)
						{
							femOffset[1] = i1 + (center / 2) + overlapStart;
							for (int i2 = 0; i2 < n; i2++)
							{
								femOffset[2] = i2 + (center / 2) + overlapStart;
								PcIntegrate(femOffset, fineCenter, stencil.AsSpan((((i0 * n) + i1) * n + i2) * CDim, CDim));
							}
						}
					}

					stencils[((o0 * 2) + o1) * 2 + o2] = stencil;
				}
			}
		}

		return stencils;
	}

	private static int Cube(int x) => x * x * x;

	// Port of _integrate / _integral: 0 = child-child, 1 = parent-child, 2 = child-parent.
	private void Integrate(int type, ReadOnlySpan<int> off1, ReadOnlySpan<int> off2, Span<double> result)
	{
		result.Clear();
		int tn = TestDerivatives + 1;
		int cn = ConstraintDerivatives + 1;
		foreach (var (d1, d2, indices) in weightedIndices)
		{
			double integral = 1.0;
			for (int d = Dim - 1; d >= 0; d--)
			{
				int td = d switch { 0 => d1 / (tn * tn), 1 => (d1 / tn) % tn, _ => d1 % tn };
				int cd = d switch { 0 => d2 / (cn * cn), 1 => (d2 / cn) % cn, _ => d2 % cn };
				double dot = type switch
				{
					0 => cc!.Dot(off1[d], off2[d], td, cd),
					1 => pc?.Dot(off1[d], off2[d], td, cd) ?? 0.0,
					_ => cp?.Dot(off2[d], off1[d], cd, td) ?? 0.0,
				};
				integral = dot * integral;
			}

			foreach (var (c, weight) in indices)
			{
				result[c] += weight * integral;
			}
		}
	}
}
