// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineIntegrationData: BSplineIntegrationData<FEMSig1,FEMSig2> from thirdparty/
// PoissonRecon/BSplineData.h and .inl - inner products of (derivatives of) two basis
// functions, possibly at different depths (Dot), and the tables the FEM system assembly uses
// (FunctionIntegrator::Integrator for same-depth pairs, ::ChildIntegrator for a function and
// the functions one depth finer). Dot is exact integer bookkeeping on BSplineElements plus
// one weighted sum over the tabulated piece integrals, so the tables are Tier A.
//
// Translation notes: the template parameters D1 and D2 (derivatives taken) are arguments.
// The C++ fills the integrator tables through a recursive IntegratorSetter over all
// (d1 <= D1, d2 <= D2); every entry is independent, so a double loop fills the same values.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Inner products between the bases of two FEM signatures. Port of
/// <c>BSplineIntegrationData&lt;FEMSig1,FEMSig2&gt;</c>.
/// </summary>
public sealed class BSplineIntegrationData
{
	/// <summary>Index bookkeeping for a pair of signatures.</summary>
	public BSplineIntegrationData(int femSignature1, int femSignature2)
	{
		FemSig1 = femSignature1;
		FemSig2 = femSignature2;
		Degree1 = FemSignature.Degree(femSignature1);
		Degree2 = FemSignature.Degree(femSignature2);
		Overlap = BSplineOverlapSizes.For(Degree1, Degree2);
		Pad1 = BSplineEvaluationData.For(femSignature1).Pad;
		OffsetStart = -Overlap.OverlapSupportStart;
		OffsetStop = Overlap.OverlapSupportEnd + (Degree1 & 1);
		IndexSize = OffsetStart + OffsetStop + 1 + 2 * Pad1;
	}

	/// <summary>The template parameter <c>FEMSig1</c>.</summary>
	public int FemSig1 { get; }

	/// <summary>The template parameter <c>FEMSig2</c>.</summary>
	public int FemSig2 { get; }

	/// <summary>The template parameter <c>Degree1</c>.</summary>
	public int Degree1 { get; }

	/// <summary>The template parameter <c>Degree2</c>.</summary>
	public int Degree2 { get; }

	/// <summary>The overlap sizes of the two degrees.</summary>
	public BSplineOverlapSizes Overlap { get; }

	/// <summary>The number of distinct table rows for the first function's offset.</summary>
	public int IndexSize { get; }

	/// <summary>Port of <c>OffsetStart</c> (inclusive bound or size, in function offsets).</summary>
	public int OffsetStart { get; }

	/// <summary>Port of <c>OffsetStop</c> (inclusive bound or size, in function offsets).</summary>
	public int OffsetStop { get; }

	private int Pad1 { get; }

	/// <summary>The table row of the first function's offset. Port of <c>OffsetToIndex</c>.</summary>
	public int OffsetToIndex(int depth, int offset)
	{
		int dim = BSplineSupportSizes.For(Degree1).Nodes(depth);
		if (offset < OffsetStart)
		{
			return Pad1 + offset;
		}
		else if (offset >= dim - OffsetStop)
		{
			return Pad1 + OffsetStart + 1 + offset - (dim - OffsetStop);
		}
		else
		{
			return Pad1 + OffsetStart;
		}
	}

	/// <summary>A representative offset of a table row. Port of <c>IndexToOffset</c>.</summary>
	public int IndexToOffset(int depth, int idx) =>
		idx - Pad1 <= OffsetStart ? idx - Pad1 : BSplineSupportSizes.For(Degree1).Nodes(depth) + Pad1 - IndexSize + idx;

	/// <summary>
	/// The offsets whose overlapping neighbors are all interiorly supported. Port of
	/// <c>InteriorOverlappedSpan</c>.
	/// </summary>
	public void InteriorOverlappedSpan(int depth, out int begin, out int end)
	{
		BSplineSupportSizes s2 = BSplineSupportSizes.For(Degree2);
		begin = -Overlap.OverlapStart - s2.SupportStart;
		end = (1 << depth) - Overlap.OverlapEnd - s2.SupportEnd;
	}

	/// <summary>
	/// \int D^d1 B1_{depth1,off1} * D^d2 B2_{depth2,off2} over [0,1], with boundary conditions.
	/// Port of <c>Dot&lt;D1,D2&gt;( depth1 , off1 , depth2 , off2 )</c>.
	/// </summary>
	public double Dot(int d1, int d2, int depth1, int off1, int depth2, int off2)
	{
		if (d1 > Degree1)
		{
			throw new InvalidOperationException($"Taking more derivatives than the degree: {d1} > {Degree1}");
		}

		if (d2 > Degree2)
		{
			throw new InvalidOperationException($"Taking more derivatives than the degree: {d2} > {Degree2}");
		}

		int degree1 = Degree1 - d1;
		int degree2 = Degree2 - d2;
		int depth = StdMinMax.StdMax(depth1, depth2);
		BSplineSupportSizes s1 = BSplineSupportSizes.For(Degree1);
		BSplineSupportSizes s2 = BSplineSupportSizes.For(Degree2);
		BSplineElements b1;
		BSplineElements b2;
		if (s1.IsInteriorlySupported(depth1, off1) && s2.IsInteriorlySupported(depth2, off2))
		{
			// Away from the boundary, integrate on a small window around the coarser function.
			if (depth1 < depth2)
			{
				int res = 1 - s1.SupportStart + s1.SupportEnd;
				s1.InteriorSupportedSpan(depth1, out int begin1, out _);
				b1 = new BSplineElements(Degree1, res, begin1, BoundaryType.Free);
				for (int d = depth1; d < depth2; d++)
				{
					b1 = b1.UpSample();
					res <<= 1;
				}

				b2 = new BSplineElements(Degree2, res, off2 - ((off1 - begin1) << (depth2 - depth1)), BoundaryType.Free);
			}
			else
			{
				int res = 1 - s2.SupportStart + s2.SupportEnd;
				s2.InteriorSupportedSpan(depth2, out int begin2, out _);
				b2 = new BSplineElements(Degree2, res, begin2, BoundaryType.Free);
				for (int d = depth2; d < depth1; d++)
				{
					b2 = b2.UpSample();
					res <<= 1;
				}

				b1 = new BSplineElements(Degree1, res, off1 - ((off2 - begin2) << (depth1 - depth2)), BoundaryType.Free);
			}
		}
		else
		{
			b1 = new BSplineElements(Degree1, 1 << depth1, off1, FemSignature.Boundary(FemSig1));
			b2 = new BSplineElements(Degree2, 1 << depth2, off2, FemSignature.Boundary(FemSig2));
			while (depth1 < depth)
			{
				b1 = b1.UpSample();
				depth1++;
			}

			while (depth2 < depth)
			{
				b2 = b2.UpSample();
				depth2++;
			}
		}

		BSplineElements db1 = b1.Differentiate(d1);
		BSplineElements db2 = b2.Differentiate(d2);

		int start1 = -1, end1 = -1, start2 = -1, end2 = -1;
		for (int i = 0; i < b1.Count; i++)
		{
			for (int j = 0; j <= Degree1; j++)
			{
				if (b1[i][j] != 0 && start1 == -1)
				{
					start1 = i;
				}

				if (b1[i][j] != 0)
				{
					end1 = i + 1;
				}
			}

			for (int j = 0; j <= Degree2; j++)
			{
				if (b2[i][j] != 0 && start2 == -1)
				{
					start2 = i;
				}

				if (b2[i][j] != 0)
				{
					end2 = i + 1;
				}
			}
		}

		if (start1 == end1 || start2 == end2 || start1 >= end2 || start2 >= end1)
		{
			return 0.0;
		}

		int start = StdMinMax.StdMax(start1, start2);
		int end = StdMinMax.StdMin(end1, end2);
		var sums = new int[degree1 + 1, degree2 + 1];

		// Iterate over the support
		for (int i = start; i < end; i++)
		{
			// Iterate over all pairs of elements within a node
			for (int j = 0; j <= degree1; j++)
			{
				for (int k = 0; k <= degree2; k++)
				{
					// Accumulate the product of the coefficients
					sums[j, k] += db1[i][j] * db2[i][k];
				}
			}
		}

		double dot = 0;
		double[,] integrals = BSplineElements.ElementIntegrals(degree1, degree2);
		for (int j = 0; j <= degree1; j++)
		{
			for (int k = 0; k <= degree2; k++)
			{
				dot += integrals[j, k] * sums[j, k];
			}
		}

		dot /= b1.Denominator;
		dot /= b2.Denominator;
		return (d1 == 0 && d2 == 0) ? dot / (1 << depth) : dot * (1 << (depth * (d1 + d2 - 1)));
	}
}

/// <summary>
/// Tabulated inner products of basis pairs at one depth (or a depth and the next), for all
/// derivative pairs up to (D1, D2). Port of <c>FunctionIntegrator::Integrator</c> and
/// <c>FunctionIntegrator::ChildIntegrator</c>.
/// </summary>
public sealed class BSplineIntegrator
{
	private readonly BSplineIntegrationData data;
	private readonly double[] integrals;
	private readonly int start;
	private readonly int size;
	private readonly bool child;
	private readonly int maxD1;
	private readonly int maxD2;

	/// <summary>
	/// Tabulates the inner products for functions at <paramref name="depth"/> (the parent depth
	/// when <paramref name="child"/>). Port of <c>SetIntegrator</c> / <c>SetChildIntegrator</c>.
	/// </summary>
	public BSplineIntegrator(int femSignature1, int femSignature2, int maxD1, int maxD2, int depth, bool child)
	{
		data = new BSplineIntegrationData(femSignature1, femSignature2);
		Depth = depth;
		this.child = child;
		this.maxD1 = maxD1;
		this.maxD2 = maxD2;
		start = child ? data.Overlap.ChildOverlapStart : data.Overlap.OverlapStart;
		int end = child ? data.Overlap.ChildOverlapEnd : data.Overlap.OverlapEnd;
		size = end - start + 1;
		integrals = new double[(maxD1 + 1) * (maxD2 + 1) * data.IndexSize * size];
		for (int d1 = 0; d1 <= maxD1; d1++)
		{
			for (int d2 = 0; d2 <= maxD2; d2++)
			{
				for (int i = 0; i < data.IndexSize; i++)
				{
					for (int j = start; j <= end; j++)
					{
						int ii = data.IndexToOffset(depth, i);
						integrals[Slot(d1, d2, i, j - start)] = child
							? data.Dot(d1, d2, depth, ii, depth + 1, 2 * ii + j)
							: data.Dot(d1, d2, depth, ii, depth, ii + j);
					}
				}
			}
		}
	}

	/// <summary>The first function's depth.</summary>
	public int Depth { get; }

	/// <summary>
	/// The inner product of D^d1 of function off1 with D^d2 of function off2 (one depth finer
	/// for a child integrator). Port of <c>dot( off1 , off2 , d1 , d2 )</c>.
	/// </summary>
	public double Dot(int off1, int off2, int d1, int d2)
	{
		// The C++ arrays are sized by the template's D1/D2; an out-of-range derivative would
		// silently read another table entry here, so it fails loudly instead.
		if (d1 < 0 || d1 > maxD1 || d2 < 0 || d2 > maxD2)
		{
			throw new ArgumentOutOfRangeException(nameof(d1), $"Derivatives ({d1}, {d2}) exceed the tabulated ({maxD1}, {maxD2}).");
		}

		int d = off2 - (child ? 2 * off1 : off1);
		if (BSplineEvaluationData.For(data.FemSig1).OutOfBounds(Depth, off1)
			|| BSplineEvaluationData.For(data.FemSig2).OutOfBounds(child ? Depth + 1 : Depth, off2)
			|| d < start || d >= start + size)
		{
			return 0;
		}

		return integrals[Slot(d1, d2, data.OffsetToIndex(Depth, off1), d - start)];
	}

	private int Slot(int d1, int d2, int row, int column) =>
		(((d1 * (maxD2 + 1)) + d2) * data.IndexSize + row) * size + column;
}
