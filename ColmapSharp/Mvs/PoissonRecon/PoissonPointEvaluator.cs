// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonPointEvaluator: FEMIntegrator::PointEvaluatorState (with
// BaseFEMIntegrator::PointEvaluatorState's dValues and partialDotDValues) and
// PointEvaluator::initEvaluationState( p , depth , offset , state ) from
// thirdparty/PoissonRecon/FEMTree.h, for isotropic 3D signatures and value-only evaluation
// (PointD = 0, all Solve uses): per axis, the values at a point of the basis functions whose
// support contains the point's cell, tabulated once with BSplineData, then multiplied into
// tensor-product values. PoissonFemConstraints (interpolation constraints) and PoissonSystem
// (the interpolation part of the matrix rows and prolonged constraints) evaluate through it.
// Tier A (oracle/poisson_system_harness.cc).
//
// Translation notes: one state per caller, reused across points (the C++ builds a fresh one
// per point; every entry read is rewritten by Init). Products are formed from the last axis:
// v0 * ( v1 * ( v2 * 1. ) ), as the C++ recursion _value does.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The per-axis basis-function values at one point. Port of PoissonRecon's
/// <c>PointEvaluatorState</c> with value-only evaluation.
/// </summary>
public sealed class PoissonPointEvaluator
{
	private readonly BSplineData bSplineData;
	private readonly BSplineSupportSizes support;
	private readonly int[] pointOffset = new int[3];

	// _oneDValues[axis][s + SupportEnd]: the value at the point of function pointOffset + s.
	private readonly double[] values;

	/// <summary>
	/// An evaluator for basis <paramref name="signature"/> at depths 0..<paramref name="maxDepth"/>,
	/// storing <paramref name="derivatives"/> derivatives. Port of <c>PointEvaluator( maxDepth )</c>.
	/// </summary>
	public PoissonPointEvaluator(int signature, int derivatives, int maxDepth)
	{
		bSplineData = new BSplineData(signature, derivatives, maxDepth);
		support = BSplineSupportSizes.For(FemSignature.Degree(signature));
		values = new double[3 * support.SupportSize];
	}

	/// <summary>The number of functions tabulated per axis (BSplineSupportSizes::SupportSize).</summary>
	public int SupportSize => support.SupportSize;

	/// <summary>
	/// Tabulates, per axis, the functions offset + s (s from -SupportEnd to -SupportStart) at
	/// position p (a float point, widened), the offset being the cell containing p. Port of
	/// <c>initEvaluationState( p , depth , offset , state )</c>.
	/// </summary>
	public void Init(int depth, float x, float y, float z, int ox, int oy, int oz)
	{
		pointOffset[0] = ox;
		pointOffset[1] = oy;
		pointOffset[2] = oz;
		SparseBSplineEvaluator evaluator = bSplineData[depth];
		int width = support.SupportSize;
		for (int k = 0; k < 3; k++)
		{
			double coordinate = k == 0 ? x : k == 1 ? y : z;
			for (int s = -support.SupportEnd; s <= -support.SupportStart; s++)
			{
				double p = coordinate;
				PoissonPolynomial[] components = evaluator.PolynomialsAndOffset(ref p, pointOffset[k], pointOffset[k] + s);
				values[k * width + s + support.SupportEnd] = components[0].Evaluate(p);
			}
		}
	}

	/// <summary>The tabulated value of function pointOffset + i - SupportEnd on an axis. Port of <c>values&lt;axis&gt;()[i][0]</c>.</summary>
	public double AxisValue(int axis, int i) => values[axis * support.SupportSize + i];

	/// <summary>The value at the point of the function at (x, y, z) (0 outside the tabulated span). Port of <c>value( offset , 0 )</c>.</summary>
	public double Value(int x, int y, int z) => OneD(0, x) * (OneD(1, y) * (OneD(2, z) * 1.0));

	/// <summary>The product over the first two axes. Port of <c>subValue( offset , 0 )</c>.</summary>
	public double SubValue(int x, int y) => OneD(0, x) * (OneD(1, y) * 1.0);

	// _OneDValues::value( off - pointOffset , 0 ).
	private double OneD(int axis, int offset)
	{
		int dOff = offset - pointOffset[axis];
		return dOff >= -support.SupportEnd && dOff <= -support.SupportStart ? values[axis * support.SupportSize + dOff + support.SupportEnd] : 0;
	}
}
