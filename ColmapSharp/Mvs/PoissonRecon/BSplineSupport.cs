// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineSupport: the index bookkeeping at the top of thirdparty/PoissonRecon/BSplineData.h -
// BoundaryType, the FEM signature (degree and boundary type packed into one integer, with
// DerivativeBoundary), BSplineSupportSizes (which neighbors a B-spline of a given degree
// touches, at its own depth and one finer) and BSplineOverlapSizes (which pairs of B-splines
// overlap). Pure integer arithmetic, Tier A. BSplineElements, BSplineEvaluationData and
// BSplineIntegrationData read these bounds.
//
// Translation notes: the C++ computes these as static constants of class templates over the
// degree (and PR_BSPLINE_SET_BOUNDS macros); here they are fields of a small immutable object
// per degree (pair), cached so lookups do not allocate. C++ integer division truncates toward
// zero, as C#'s does, so the expressions are copied unchanged. The PR_*_HALF macros are
// FloorOfHalf/CeilOfHalf below.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>Boundary conditions of a finite-element basis. Port of PoissonRecon's <c>BoundaryType</c>.</summary>
public enum BoundaryType
{
	/// <summary>No boundary condition: the basis extends past the unit interval.</summary>
	Free = 0,

	/// <summary>Zero value at the boundary (odd reflection).</summary>
	Dirichlet = 1,

	/// <summary>Zero derivative at the boundary (even reflection).</summary>
	Neumann = 2,
}

/// <summary>
/// The FEM signature: one integer combining a B-spline degree and a boundary type. Port of
/// <c>FEMDegreeAndBType</c>, <c>FEMSignature</c> and <c>DerivativeBoundary</c>.
/// </summary>
public static class FemSignature
{
	/// <summary>The number of boundary types (BOUNDARY_COUNT).</summary>
	public const int BoundaryCount = 3;

	/// <summary>The signature of the degree-0, free-boundary basis (FEMTrivialSignature).</summary>
	public const int Trivial = 0;

	/// <summary>
	/// The highest B-spline degree the port supports. COLMAP's path instantiates degrees 0
	/// (colors), 1 (the FEM basis) and 2 (normals and density); the oracle covers exactly
	/// those, so higher degrees are rejected rather than left unverified.
	/// </summary>
	public const int MaxDegree = 2;

	/// <summary>Returns <paramref name="degree"/>, or throws if it is outside 0..MaxDegree.</summary>
	public static int CheckDegree(int degree)
	{
		if (degree < 0 || degree > MaxDegree)
		{
			throw new ArgumentOutOfRangeException(nameof(degree), $"B-spline degree {degree} is not supported (0 to {MaxDegree}).");
		}

		return degree;
	}

	/// <summary>Returns <paramref name="signature"/>, or throws if its degree is outside 0..MaxDegree.</summary>
	public static int CheckSignature(int signature)
	{
		if (signature < 0 || signature >= (MaxDegree + 1) * BoundaryCount)
		{
			throw new ArgumentOutOfRangeException(nameof(signature), $"FEM signature {signature} is not supported (degrees 0 to {MaxDegree}).");
		}

		return signature;
	}

	/// <summary>Port of <c>FEMDegreeAndBType&lt;Degree,BType&gt;::Signature</c>.</summary>
	public static int Of(int degree, BoundaryType boundary) => degree * BoundaryCount + (int)boundary;

	/// <summary>The degree packed in a signature. Port of <c>FEMSignature::Degree</c>.</summary>
	public static int Degree(int signature) => signature / BoundaryCount;

	/// <summary>The boundary type packed in a signature. Port of <c>FEMSignature::BType</c>.</summary>
	public static BoundaryType Boundary(int signature) => (BoundaryType)(signature % BoundaryCount);

	/// <summary>
	/// The boundary type of the d-th derivative of a basis with the given boundary type:
	/// differentiating swaps Dirichlet and Neumann. Port of <c>DerivativeBoundary&lt;BType,D&gt;</c>.
	/// </summary>
	public static BoundaryType DerivativeBoundary(BoundaryType boundary, int d)
	{
		for (int i = 0; i < d; i++)
		{
			boundary = boundary switch
			{
				BoundaryType.Dirichlet => BoundaryType.Neumann,
				BoundaryType.Neumann => BoundaryType.Dirichlet,
				_ => BoundaryType.Free,
			};
		}

		return boundary;
	}

	/// <summary>
	/// The signature of the d-th derivative basis. Port of <c>FEMSignature::DSignature&lt;D&gt;</c>.
	/// </summary>
	public static int DSignature(int signature, int d = 1)
	{
		int degree = Degree(signature);
		if (degree < d)
		{
			throw new ArgumentOutOfRangeException(nameof(d), $"Cannot take {d} derivatives of a degree-{degree} basis.");
		}

		return Of(degree - d, DerivativeBoundary(Boundary(signature), d));
	}

	/// <summary>False only for Dirichlet bases. Port of <c>HasPartitionOfUnity</c>.</summary>
	public static bool HasPartitionOfUnity(BoundaryType boundary) => boundary != BoundaryType.Dirichlet;
}

/// <summary>
/// The support of a degree-<see cref="Degree"/> B-spline, in cells relative to its index.
/// Port of <c>BSplineSupportSizes&lt;Degree&gt;</c>; every bound is inclusive, [Start, End].
/// </summary>
public sealed class BSplineSupportSizes
{
	private static readonly BSplineSupportSizes[] Cache = [new(0), new(1), new(2)];

	private BSplineSupportSizes(int degree)
	{
		Degree = degree;
		Inset = (degree & 1) != 0 ? 0 : 1;
		SupportStart = -((degree + 1) / 2);
		SupportEnd = degree / 2;
		ChildSupportStart = 2 * SupportStart;
		ChildSupportEnd = 2 * (SupportEnd + 1) - 1;
		CornerStart = SupportStart + 1;
		CornerEnd = SupportEnd;
		ChildCornerStart = 2 * SupportStart + 1;
		ChildCornerEnd = 2 * SupportEnd + 1;
		BCornerStart = CornerStart - 1;
		BCornerEnd = CornerEnd + 1;
		ChildBCornerStart = ChildCornerStart - 1;
		ChildBCornerEnd = ChildCornerEnd + 1;
		UpSampleStart = -(degree + 1 - Inset) / 2;
		UpSampleEnd = (degree + 1 + Inset) / 2;
		DownSample0Start = CeilOfHalf(0 - (degree + 1 + Inset) / 2);
		DownSample0End = FloorOfHalf(0 + (degree + 1 - Inset) / 2);
		DownSample1Start = CeilOfHalf(1 - (degree + 1 + Inset) / 2);
		DownSample1End = FloorOfHalf(1 + (degree + 1 - Inset) / 2);
	}

	/// <summary>The B-spline degree.</summary>
	public int Degree { get; }

	/// <summary>
	/// 1 for even degrees (dual basis, centered in a cell), 0 for odd (primal basis, centered on
	/// a cell's left end).
	/// </summary>
	public int Inset { get; }

	/// <summary>Port of <c>SupportStart</c> (inclusive bound or size, in function offsets).</summary>
	public int SupportStart { get; }

	/// <summary>Port of <c>SupportEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int SupportEnd { get; }

	/// <summary>Port of <c>SupportSize</c> (inclusive bound or size, in function offsets).</summary>
	public int SupportSize => SupportEnd - SupportStart + 1;

	/// <summary>Port of <c>ChildSupportStart</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildSupportStart { get; }

	/// <summary>Port of <c>ChildSupportEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildSupportEnd { get; }

	/// <summary>Port of <c>ChildSupportSize</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildSupportSize => ChildSupportEnd - ChildSupportStart + 1;

	/// <summary>Port of <c>CornerStart</c> (inclusive bound or size, in function offsets).</summary>
	public int CornerStart { get; }

	/// <summary>Port of <c>CornerEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int CornerEnd { get; }

	/// <summary>Port of <c>CornerSize</c> (inclusive bound or size, in function offsets).</summary>
	public int CornerSize => CornerEnd - CornerStart + 1;

	/// <summary>Port of <c>ChildCornerStart</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildCornerStart { get; }

	/// <summary>Port of <c>ChildCornerEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildCornerEnd { get; }

	/// <summary>Port of <c>ChildCornerSize</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildCornerSize => ChildCornerEnd - ChildCornerStart + 1;

	/// <summary>Port of <c>BCornerStart</c> (inclusive bound or size, in function offsets).</summary>
	public int BCornerStart { get; }

	/// <summary>Port of <c>BCornerEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int BCornerEnd { get; }

	/// <summary>Port of <c>BCornerSize</c> (inclusive bound or size, in function offsets).</summary>
	public int BCornerSize => BCornerEnd - BCornerStart + 1;

	/// <summary>Port of <c>ChildBCornerStart</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildBCornerStart { get; }

	/// <summary>Port of <c>ChildBCornerEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildBCornerEnd { get; }

	/// <summary>Port of <c>ChildBCornerSize</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildBCornerSize => ChildBCornerEnd - ChildBCornerStart + 1;

	/// <summary>The finer-level indices 2*I + [UpSampleStart, UpSampleEnd] a coarse function I refines into.</summary>
	public int UpSampleStart { get; }

	/// <summary>Port of <c>UpSampleEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int UpSampleEnd { get; }

	/// <summary>Port of <c>UpSampleSize</c> (inclusive bound or size, in function offsets).</summary>
	public int UpSampleSize => UpSampleEnd - UpSampleStart + 1;

	/// <summary>Port of <c>DownSample0Start</c> (inclusive bound or size, in function offsets).</summary>
	public int DownSample0Start { get; }

	/// <summary>Port of <c>DownSample0End</c> (inclusive bound or size, in function offsets).</summary>
	public int DownSample0End { get; }

	/// <summary>Port of <c>DownSample1Start</c> (inclusive bound or size, in function offsets).</summary>
	public int DownSample1Start { get; }

	/// <summary>Port of <c>DownSample1End</c> (inclusive bound or size, in function offsets).</summary>
	public int DownSample1End { get; }

	/// <summary>DownSampleStart[parity of the child index]. Port of <c>DownSampleStart[]</c>.</summary>
	public int DownSampleStart(int parity) => parity == 0 ? DownSample0Start : DownSample1Start;

	/// <summary>Port of <c>DownSampleEnd[]</c>.</summary>
	public int DownSampleEnd(int parity) => parity == 0 ? DownSample0End : DownSample1End;

	/// <summary>Port of <c>DownSampleSize[]</c>.</summary>
	public int DownSampleSize(int parity) => DownSampleEnd(parity) - DownSampleStart(parity) + 1;

	/// <summary>The support sizes for a degree (0 to <see cref="FemSignature.MaxDegree"/>).</summary>
	public static BSplineSupportSizes For(int degree) => Cache[FemSignature.CheckDegree(degree)];

	/// <summary>The number of functions at a depth: 2^depth, plus one for odd degrees. Port of <c>Nodes</c>.</summary>
	public int Nodes(int depth) => (1 << depth) + (Degree & 1);

	/// <summary>
	/// The offsets whose support lies inside [0, 2^depth). Port of <c>InteriorSupportedSpan</c>.
	/// </summary>
	public void InteriorSupportedSpan(int depth, out int begin, out int end)
	{
		begin = -SupportStart;
		end = (1 << depth) - SupportEnd;
	}

	/// <summary>True if the support lies inside [0, 2^depth). Port of <c>IsInteriorlySupported</c>.</summary>
	public bool IsInteriorlySupported(int depth, int offset) =>
		offset + SupportStart >= 0 && offset + SupportEnd < (1 << depth);

	// PR__FLOOR_OF_HALF/PR__CEIL_OF_HALF extended to negative x as PR_FLOOR_OF_HALF/PR_CEIL_OF_HALF.
	internal static int FloorOfHalf(int x) => x < 0 ? -((-x + 1) >> 1) : x >> 1;

	internal static int CeilOfHalf(int x) => x < 0 ? -((-x) >> 1) : (x + 1) >> 1;
}

/// <summary>
/// Overlap bounds between B-splines of degrees <see cref="Degree1"/> and <see cref="Degree2"/>.
/// Port of <c>BSplineOverlapSizes&lt;Degree1,Degree2&gt;</c>; every bound is inclusive.
/// </summary>
public sealed class BSplineOverlapSizes
{
	private static readonly BSplineOverlapSizes[,] Cache = BuildCache();

	private BSplineOverlapSizes(int degree1, int degree2)
	{
		Degree1 = degree1;
		Degree2 = degree2;
		BSplineSupportSizes e1 = BSplineSupportSizes.For(degree1);
		BSplineSupportSizes e2 = BSplineSupportSizes.For(degree2);
		OverlapStart = e1.SupportStart - e2.SupportEnd;
		OverlapEnd = e1.SupportEnd - e2.SupportStart;
		ChildOverlapStart = e1.ChildSupportStart - e2.SupportEnd;
		ChildOverlapEnd = e1.ChildSupportEnd - e2.SupportStart;
		OverlapSupportStart = OverlapStart + e2.SupportStart;
		OverlapSupportEnd = OverlapEnd + e2.SupportEnd;
		ChildOverlapSupportStart = ChildOverlapStart + e2.SupportStart;
		ChildOverlapSupportEnd = ChildOverlapEnd + e2.SupportEnd;

		int wide = (2 * degree2 + degree1 + 3 + 2 * e2.Inset - e1.Inset) / 2;
		int narrow = (2 * degree2 + degree1 + 3 - 2 * e2.Inset + e1.Inset) / 2;
		// PR_SMALLEST_INTEGER_LARGER_THAN_HALF(x) = CEIL_OF_HALF(x+1),
		// PR_LARGEST_INTEGER_SMALLER_THAN_HALF(x) = FLOOR_OF_HALF(x-1).
		ParentOverlap0Start = BSplineSupportSizes.CeilOfHalf(0 - wide + 1);
		ParentOverlap0End = BSplineSupportSizes.FloorOfHalf(0 + narrow - 1);
		ParentOverlap1Start = BSplineSupportSizes.CeilOfHalf(1 - wide + 1);
		ParentOverlap1End = BSplineSupportSizes.FloorOfHalf(1 + narrow - 1);
	}

	/// <summary>The template parameter <c>Degree1</c>.</summary>
	public int Degree1 { get; }

	/// <summary>The template parameter <c>Degree2</c>.</summary>
	public int Degree2 { get; }

	/// <summary>Port of <c>OverlapStart</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapStart { get; }

	/// <summary>Port of <c>OverlapEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapEnd { get; }

	/// <summary>Port of <c>OverlapSize</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapSize => OverlapEnd - OverlapStart + 1;

	/// <summary>Port of <c>ChildOverlapStart</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapStart { get; }

	/// <summary>Port of <c>ChildOverlapEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapEnd { get; }

	/// <summary>Port of <c>ChildOverlapSize</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapSize => ChildOverlapEnd - ChildOverlapStart + 1;

	/// <summary>Port of <c>OverlapSupportStart</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapSupportStart { get; }

	/// <summary>Port of <c>OverlapSupportEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapSupportEnd { get; }

	/// <summary>Port of <c>OverlapSupportSize</c> (inclusive bound or size, in function offsets).</summary>
	public int OverlapSupportSize => OverlapSupportEnd - OverlapSupportStart + 1;

	/// <summary>Port of <c>ChildOverlapSupportStart</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapSupportStart { get; }

	/// <summary>Port of <c>ChildOverlapSupportEnd</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapSupportEnd { get; }

	/// <summary>Port of <c>ChildOverlapSupportSize</c> (inclusive bound or size, in function offsets).</summary>
	public int ChildOverlapSupportSize => ChildOverlapSupportEnd - ChildOverlapSupportStart + 1;

	/// <summary>Port of <c>ParentOverlap0Start</c> (inclusive bound or size, in function offsets).</summary>
	public int ParentOverlap0Start { get; }

	/// <summary>Port of <c>ParentOverlap0End</c> (inclusive bound or size, in function offsets).</summary>
	public int ParentOverlap0End { get; }

	/// <summary>Port of <c>ParentOverlap1Start</c> (inclusive bound or size, in function offsets).</summary>
	public int ParentOverlap1Start { get; }

	/// <summary>Port of <c>ParentOverlap1End</c> (inclusive bound or size, in function offsets).</summary>
	public int ParentOverlap1End { get; }

	/// <summary>Port of <c>ParentOverlapStart[]</c>, indexed by the child's parity.</summary>
	public int ParentOverlapStart(int parity) => parity == 0 ? ParentOverlap0Start : ParentOverlap1Start;

	/// <summary>Port of <c>ParentOverlapEnd[]</c>.</summary>
	public int ParentOverlapEnd(int parity) => parity == 0 ? ParentOverlap0End : ParentOverlap1End;

	/// <summary>Port of <c>ParentOverlapSize[]</c>.</summary>
	public int ParentOverlapSize(int parity) => ParentOverlapEnd(parity) - ParentOverlapStart(parity) + 1;

	/// <summary>The overlap sizes for a pair of degrees (0 to <see cref="FemSignature.MaxDegree"/> each).</summary>
	public static BSplineOverlapSizes For(int degree1, int degree2) => Cache[FemSignature.CheckDegree(degree1), FemSignature.CheckDegree(degree2)];

	private static BSplineOverlapSizes[,] BuildCache()
	{
		var cache = new BSplineOverlapSizes[FemSignature.MaxDegree + 1, FemSignature.MaxDegree + 1];
		for (int i = 0; i <= FemSignature.MaxDegree; i++)
		{
			for (int j = 0; j <= FemSignature.MaxDegree; j++)
			{
				cache[i, j] = new BSplineOverlapSizes(i, j);
			}
		}

		return cache;
	}
}
