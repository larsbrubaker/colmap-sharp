// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// BSplineElements: BSplineElements, Differentiator and SetBSplineElementIntegrals from
// thirdparty/PoissonRecon/BSplineData.h and BSplineData.inl. A BSplineElements is a function
// on [0,1] split into `res` cells, where each cell holds integer weights of the degree + 1
// B-spline pieces that can live there, over a common integer denominator. Boundary
// conditions are folded in by reflecting (Neumann: even, Dirichlet: odd) and wrapping the
// B-spline's pieces back into range. BSplineIntegrationData.Dot integrates products of
// these exactly (integer sums times the tabulated piece integrals), and BSplineData builds
// its per-offset polynomials from them.
//
// Integer bookkeeping plus one table of doubles (SetBSplineElementIntegrals), in the C++
// order: Tier A.
//
// Translation notes: the C++ struct derives from std::vector<BSplineElementCoefficients>;
// here the coefficients are one row per cell (Coefficients[cell][piece]). The ordering of
// pieces within a cell is "/" "-" "\" - the opposite of PoissonPolynomial.BSplineComponent's
// index, as the C++ warns. The copy-then-upSample idiom (`b = b1 ; b.upSample( b1 )`)
// becomes `b1 = b1.UpSample()`.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A piecewise-polynomial function on [0,1] stored as integer B-spline-piece weights per cell.
/// Port of PoissonRecon's <c>BSplineElements&lt;Degree&gt;</c>.
/// </summary>
public sealed class BSplineElements
{
	/// <summary>The zero function with <paramref name="res"/> cells.</summary>
	public BSplineElements(int degree, int res)
	{
		Degree = degree;
		Denominator = 1;
		Coefficients = new int[res][];
		for (int i = 0; i < res; i++)
		{
			Coefficients[i] = new int[degree + 1];
		}
	}

	/// <summary>
	/// The B-spline with the given offset on a grid of <paramref name="res"/> cells, with the
	/// boundary folded in. Port of <c>BSplineElements( int res , int offset , BoundaryType bType )</c>.
	/// </summary>
	public BSplineElements(int degree, int res, int offset, BoundaryType boundary)
		: this(degree, res)
	{
		// If we have primal dirichlet constraints, the boundary functions are necessarily zero
		if (Primal && boundary == BoundaryType.Dirichlet && offset % res == 0)
		{
			return;
		}

		// Construct the B-Spline
		for (int i = 0; i <= Degree; i++)
		{
			int idx = -Off + offset + i;
			if (idx >= 0 && idx < res)
			{
				Coefficients[idx][i] = 1;
			}
		}

		if (boundary != BoundaryType.Free)
		{
			// Fold in the periodic instances (which cancels the negation)
			AddPeriodic(true, RotateLeft(offset, res), false);
			AddPeriodic(false, RotateRight(offset, res), false);

			// Recursively fold in the boundaries
			if (Primal && offset % res == 0)
			{
				return;
			}

			// Fold in the reflected instance (which may require negation)
			AddPeriodic(true, ReflectLeft(offset, res), boundary == BoundaryType.Dirichlet);
			AddPeriodic(false, ReflectRight(offset, res), boundary == BoundaryType.Dirichlet);
		}
	}

	/// <summary>The polynomial degree of every piece.</summary>
	public int Degree { get; }

	/// <summary>Coefficients[cell][piece]: the integer weight of each B-spline piece in each cell.</summary>
	public int[][] Coefficients { get; }

	/// <summary>The common denominator of all coefficients (a power of two after up-sampling).</summary>
	public int Denominator { get; private set; }

	/// <summary>The number of cells.</summary>
	public int Count => Coefficients.Length;

	/// <summary>Row accessor matching the C++ <c>(*this)[cell][piece]</c>.</summary>
	public int[] this[int cell] => Coefficients[cell];

	// Odd degrees are primal (functions centered on cell corners).
	private bool Primal => (Degree & 1) == 1;

	private int Off => (Degree + 1) / 2;

	/// <summary>
	/// The same function on a grid twice as fine (two-scale relation with binomial weights; the
	/// denominator gains a factor 2^Degree). Port of <c>upSample</c>.
	/// </summary>
	public BSplineElements UpSample()
	{
		BSplineSupportSizes sizes = BSplineSupportSizes.For(Degree);
		Span<int> bCoefficients = stackalloc int[sizes.UpSampleSize];
		PoissonPolynomial.BinomialCoefficients(Degree + 1, bCoefficients);
		var high = new BSplineElements(Degree, Count * 2);

		// [NOTE] We have flipped the order of the B-spline elements
		for (int i = 0; i < Count; i++)
		{
			for (int j = 0; j <= Degree; j++)
			{
				// At index I , B-spline element J corresponds to a B-spline centered at:
				//		I - SupportStart - J
				int idx = i - sizes.SupportStart - j;
				for (int k = sizes.UpSampleStart; k <= sizes.UpSampleEnd; k++)
				{
					// Index idx at the coarser resolution gets up-sampled into indices:
					//		2*idx + [UpSampleStart,UpSampleEnd]
					// at the finer resolution
					int fineIdx = 2 * idx + k;

					// Compute the index of the B-spline element relative to 2*i and 2*i+1
					int j1 = -fineIdx + 2 * i - sizes.SupportStart;
					int j2 = -fineIdx + 2 * i + 1 - sizes.SupportStart;
					if (j1 >= 0 && j1 <= Degree)
					{
						high.Coefficients[2 * i + 0][j1] += Coefficients[i][j] * bCoefficients[k - sizes.UpSampleStart];
					}

					if (j2 >= 0 && j2 <= Degree)
					{
						high.Coefficients[2 * i + 1][j2] += Coefficients[i][j] * bCoefficients[k - sizes.UpSampleStart];
					}
				}
			}
		}

		high.Denominator = Denominator << Degree;
		return high;
	}

	/// <summary>
	/// The <paramref name="d"/>-th derivative, as elements of degree Degree - d (in units of the
	/// cell width; callers rescale). Port of <c>differentiate&lt;D&gt;</c> / <c>Differentiator</c>.
	/// </summary>
	public BSplineElements Differentiate(int d)
	{
		if (d > Degree)
		{
			throw new ArgumentOutOfRangeException(nameof(d), $"Cannot take {d} derivatives of degree-{Degree} elements.");
		}

		BSplineElements current = this;
		for (int step = 0; step < d; step++)
		{
			var lower = new BSplineElements(current.Degree - 1, current.Count);
			for (int i = 0; i < current.Count; i++)
			{
				for (int j = 0; j <= current.Degree; j++)
				{
					if (j - 1 >= 0)
					{
						lower.Coefficients[i][j - 1] -= current.Coefficients[i][j];
					}

					if (j < current.Degree)
					{
						lower.Coefficients[i][j] += current.Coefficients[i][j];
					}
				}
			}

			lower.Denominator = current.Denominator;
			current = lower;
		}

		return d == 0 ? Copy() : current;
	}

	/// <summary>
	/// integrals[i][j] = \int_0^1 of piece (degree1 - i) of the degree1 B-spline times piece
	/// (degree2 - j) of the degree2 B-spline. Port of <c>SetBSplineElementIntegrals</c>.
	/// </summary>
	public static double[,] ElementIntegrals(int degree1, int degree2)
	{
		var integrals = new double[degree1 + 1, degree2 + 1];
		for (int i = 0; i <= degree1; i++)
		{
			PoissonPolynomial p1 = PoissonPolynomial.BSplineComponent(degree1, degree1 - i);
			for (int j = 0; j <= degree2; j++)
			{
				PoissonPolynomial p2 = PoissonPolynomial.BSplineComponent(degree2, degree2 - j);
				integrals[i, j] = p1.Multiply(p2).Integral(0, 1);
			}
		}

		return integrals;
	}

	private BSplineElements Copy()
	{
		var copy = new BSplineElements(Degree, Count) { Denominator = Denominator };
		for (int i = 0; i < Count; i++)
		{
			Coefficients[i].CopyTo(copy.Coefficients[i], 0);
		}

		return copy;
	}

	private int ReflectLeft(int offset, int res) => Primal ? -offset : -1 - offset;

	private int ReflectRight(int offset, int res) => Primal ? 2 * res - offset : 2 * res - 1 - offset;

	private static int RotateLeft(int offset, int res) => offset - 2 * res;

	private static int RotateRight(int offset, int res) => offset + 2 * res;

	// Port of _addPeriodic<Left>: adds the B-spline at `offset` (possibly negated), and keeps
	// wrapping by 2*res in the same direction while any of its pieces still land in range.
	private void AddPeriodic(bool left, int offset, bool negate)
	{
		int res = Count;
		while (true)
		{
			bool set = false;

			// Add in the corresponding B-spline elements (possibly negated)
			for (int i = 0; i <= Degree; i++)
			{
				int idx = -Off + offset + i;
				if (idx >= 0 && idx < res)
				{
					Coefficients[idx][i] += negate ? -1 : 1;
					set = true;
				}
			}

			// If there is a change for additional overlap, give it a go
			if (!set)
			{
				return;
			}

			offset = left ? RotateLeft(offset, res) : RotateRight(offset, res);
		}
	}
}
