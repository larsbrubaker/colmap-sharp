// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineEvaluators: the tabulated evaluators of BSplineEvaluationData<FEMSig>
// (thirdparty/PoissonRecon/BSplineData.h and .inl) - CenterEvaluator::Evaluator and
// ::ChildEvaluator (values at cell centers, same depth or one finer), CornerEvaluator::
// Evaluator and ::ChildEvaluator (values at cell corners; the top derivative, which jumps at
// corners, is averaged from both sides) and UpSampleEvaluator (the two-scale coefficients).
// Each table has IndexSize rows (BSplineEvaluationData's boundary index compression) and is
// filled from BSplineEvaluationData.Value / UpSamplingCoefficients, so it is Tier A with them.
// The FEM tree's evaluation and multigrid (later slices) read these.
//
// Translation notes: the C++ fixed-size 3D arrays _ccValues[D+1][IndexSize][Size] become one
// flat array indexed ((d * IndexSize) + row) * Size + column; `set( depth )` is the constructor.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Which points a <see cref="BSplineTabulatedEvaluator"/> samples: cell centers or cell
/// corners, at the function's own depth or one finer.
/// </summary>
public enum BSplineSamplePoints
{
	/// <summary>Port of <c>CenterEvaluator&lt;D&gt;::Evaluator</c>.</summary>
	Center,

	/// <summary>Port of <c>CenterEvaluator&lt;D&gt;::ChildEvaluator</c>.</summary>
	ChildCenter,

	/// <summary>Port of <c>CornerEvaluator&lt;D&gt;::Evaluator</c>.</summary>
	Corner,

	/// <summary>Port of <c>CornerEvaluator&lt;D&gt;::ChildEvaluator</c>.</summary>
	ChildCorner,
}

/// <summary>
/// Values (and derivatives up to D) of the basis functions of a depth, tabulated at cell
/// centers or corners. Port of BSplineEvaluationData's Center/Corner Evaluator and
/// ChildEvaluator.
/// </summary>
public sealed class BSplineTabulatedEvaluator
{
	private readonly BSplineEvaluationData data;
	private readonly double[] values;
	private readonly int start;
	private readonly int size;
	private readonly bool child;
	private readonly bool corner;

	/// <summary>
	/// Tabulates for functions at <paramref name="depth"/> (the parent depth for the child
	/// variants). Port of <c>SetCenterEvaluator</c>, <c>SetChildCenterEvaluator</c>,
	/// <c>SetCornerEvaluator</c> and <c>SetChildCornerEvaluator</c>.
	/// </summary>
	public BSplineTabulatedEvaluator(int femSignature, int derivatives, BSplineSamplePoints points, int depth)
	{
		data = BSplineEvaluationData.For(femSignature);
		Derivatives = derivatives;
		Depth = depth;
		child = points is BSplineSamplePoints.ChildCenter or BSplineSamplePoints.ChildCorner;
		corner = points is BSplineSamplePoints.Corner or BSplineSamplePoints.ChildCorner;
		BSplineSupportSizes s = data.Sizes;
		(start, int end) = points switch
		{
			BSplineSamplePoints.Center => (s.SupportStart, s.SupportEnd),
			BSplineSamplePoints.ChildCenter => (s.ChildSupportStart, s.ChildSupportEnd),
			BSplineSamplePoints.Corner => (s.BCornerStart, s.BCornerEnd),
			_ => (s.ChildBCornerStart, s.ChildBCornerEnd),
		};
		size = end - start + 1;
		values = new double[(derivatives + 1) * data.IndexSize * size];

		int res = 1 << (child ? depth + 1 : depth);
		int scale = child ? 2 : 1;
		for (int i = 0; i < data.IndexSize; i++)
		{
			for (int j = start; j <= end; j++)
			{
				int ii = data.IndexToOffset(depth, i);
				int jj = j - start;
				if (!corner)
				{
					double sc = 0.5 + scale * ii + j;
					for (int d1 = 0; d1 <= derivatives; d1++)
					{
						values[Slot(d1, i, jj)] = data.Value(depth, ii, sc / res, d1);
					}

					continue;
				}

				double sp = scale * ii + j;
				for (int d1 = 0; d1 <= derivatives; d1++)
				{
					if (d1 == data.Degree)
					{
						// The top derivative is piecewise constant and jumps at corners: average the two sides.
						if (j == start)
						{
							values[Slot(d1, i, jj)] = data.Value(depth, ii, (sp + 0.5) / res, d1) / 2;
						}
						else if (j == end)
						{
							values[Slot(d1, i, jj)] = data.Value(depth, ii, (sp - 0.5) / res, d1) / 2;
						}
						else
						{
							values[Slot(d1, i, jj)] = (data.Value(depth, ii, (sp - 0.5) / res, d1) + data.Value(depth, ii, (sp + 0.5) / res, d1)) / 2;
						}
					}
					else
					{
						values[Slot(d1, i, jj)] = data.Value(depth, ii, sp / res, d1);
					}
				}
			}
		}
	}

	/// <summary>The number of derivatives tabulated (template parameter D).</summary>
	public int Derivatives { get; }

	/// <summary>The functions' depth (the parent depth for child evaluators).</summary>
	public int Depth { get; }

	/// <summary>
	/// The d-th derivative of function fIdx at sample cIdx (a center or corner index at the
	/// sampled depth); 0 outside the tabulated support. Port of <c>value( fIdx , cIdx , d )</c>.
	/// </summary>
	public double Value(int fIdx, int cIdx, int d)
	{
		// The C++ array is sized by the template's D; fail loudly rather than read another entry.
		if (d < 0 || d > Derivatives)
		{
			throw new ArgumentOutOfRangeException(nameof(d), $"Derivative {d} exceeds the tabulated {Derivatives}.");
		}

		int dd = cIdx - (child ? 2 * fIdx : fIdx);
		int res = (1 << (child ? Depth + 1 : Depth)) + (corner ? 1 : 0);
		if (cIdx < 0 || cIdx >= res || data.OutOfBounds(Depth, fIdx) || dd < start || dd >= start + size)
		{
			return 0;
		}

		return values[Slot(d, data.OffsetToIndex(Depth, fIdx), dd - start)];
	}

	private int Slot(int d, int row, int column) => ((d * data.IndexSize) + row) * size + column;
}

/// <summary>
/// The two-scale coefficients from a depth to the next, tabulated. Port of
/// <c>BSplineEvaluationData::UpSampleEvaluator</c>.
/// </summary>
public sealed class BSplineUpSampleEvaluator
{
	private readonly BSplineEvaluationData data;
	private readonly double[] values;

	/// <summary>Port of <c>SetUpSampleEvaluator( evaluator , lowDepth )</c>.</summary>
	public BSplineUpSampleEvaluator(int femSignature, int lowDepth)
	{
		data = BSplineEvaluationData.For(femSignature);
		LowDepth = lowDepth;
		int size = data.Sizes.UpSampleSize;
		values = new double[data.IndexSize * size];
		for (int i = 0; i < data.IndexSize; i++)
		{
			int ii = data.IndexToOffset(lowDepth, i);
			double[] b = data.UpSamplingCoefficients(lowDepth, ii);
			for (int j = 0; j < size; j++)
			{
				values[i * size + j] = b[j];
			}
		}
	}

	/// <summary>The coarse depth.</summary>
	public int LowDepth { get; }

	/// <summary>
	/// The weight of fine function cIdx in coarse function pIdx. Port of <c>value( pIdx , cIdx )</c>.
	/// </summary>
	public double Value(int pIdx, int cIdx)
	{
		BSplineSupportSizes s = data.Sizes;
		int dd = cIdx - 2 * pIdx;
		if (data.OutOfBounds(LowDepth + 1, cIdx) || data.OutOfBounds(LowDepth, pIdx) || dd < s.UpSampleStart || dd > s.UpSampleEnd)
		{
			return 0;
		}

		return values[data.OffsetToIndex(LowDepth, pIdx) * s.UpSampleSize + dd - s.UpSampleStart];
	}
}
