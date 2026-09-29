// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Written for ColmapSharp (the counterpart of Ceres Solver 2.2.0's fixed-size small_blas.h
// template instantiations, BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Fixed inner dimension kernels for the shapes bundle adjustment feeds the Schur eliminator
// (SchurEliminator.cs) through SmallBlas.cs: residual blocks of 2 rows, E blocks (points) of
// size 3. The k loop of SmallBlas' naive kernels is unrolled here, but each element is still
// 0.0 + a_0 b_0 + a_1 b_1 (+ a_2 b_2) summed in that order into a temporary and then stored
// with the operation, so the results are bit-identical to the naive loops (pinned by
// SmallBlasFixedTests). Ceres' own fixed-size path uses Eigen's kernels, whose order may
// differ (divergence 35), so matching the naive loops is the contract.
//
// The lengths are checked once per call; the loops then index without per-element bounds
// checks.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ColmapSharp.Solver;

/// <summary>Unrolled inner-dimension 2 and 3 products for <see cref="SmallBlas"/>.</summary>
internal static class SmallBlasFixed
{
	/// <summary>C op= A' B for A (2 x colsA) and B (2 x colsB); C(row, col) is at index0 + row * stride + col.</summary>
	public static void MatrixTransposeMatrixMultiply2(
		ReadOnlySpan<double> a, int colsA, ReadOnlySpan<double> b, int colsB, Span<double> c, int index0, int stride, int operation)
	{
		if (colsA <= 0 || colsB <= 0)
		{
			return;
		}

		CheckLengths(a.Length >= 2 * colsA, b.Length >= 2 * colsB, c.Length >= index0 + ((colsA - 1) * stride) + colsB, index0);
		ref double ar = ref MemoryMarshal.GetReference(a);
		ref double br = ref MemoryMarshal.GetReference(b);
		ref double cr = ref MemoryMarshal.GetReference(c);
		for (int row = 0; row < colsA; row++)
		{
			double a0 = Unsafe.Add(ref ar, row);
			double a1 = Unsafe.Add(ref ar, colsA + row);
			ref double cRow = ref Unsafe.Add(ref cr, index0 + (row * stride));
			for (int col = 0; col < colsB; col++)
			{
				double tmp = 0.0;
				tmp += a0 * Unsafe.Add(ref br, col);
				tmp += a1 * Unsafe.Add(ref br, colsB + col);
				Store(ref Unsafe.Add(ref cRow, col), tmp, operation);
			}
		}
	}

	/// <summary>C op= A' B for A (3 x colsA) and B (3 x colsB); C(row, col) is at index0 + row * stride + col.</summary>
	public static void MatrixTransposeMatrixMultiply3(
		ReadOnlySpan<double> a, int colsA, ReadOnlySpan<double> b, int colsB, Span<double> c, int index0, int stride, int operation)
	{
		if (colsA <= 0 || colsB <= 0)
		{
			return;
		}

		CheckLengths(a.Length >= 3 * colsA, b.Length >= 3 * colsB, c.Length >= index0 + ((colsA - 1) * stride) + colsB, index0);
		ref double ar = ref MemoryMarshal.GetReference(a);
		ref double br = ref MemoryMarshal.GetReference(b);
		ref double cr = ref MemoryMarshal.GetReference(c);
		for (int row = 0; row < colsA; row++)
		{
			double a0 = Unsafe.Add(ref ar, row);
			double a1 = Unsafe.Add(ref ar, colsA + row);
			double a2 = Unsafe.Add(ref ar, (2 * colsA) + row);
			ref double cRow = ref Unsafe.Add(ref cr, index0 + (row * stride));
			for (int col = 0; col < colsB; col++)
			{
				double tmp = 0.0;
				tmp += a0 * Unsafe.Add(ref br, col);
				tmp += a1 * Unsafe.Add(ref br, colsB + col);
				tmp += a2 * Unsafe.Add(ref br, (2 * colsB) + col);
				Store(ref Unsafe.Add(ref cRow, col), tmp, operation);
			}
		}
	}

	/// <summary>C op= A B for A (rowsA x 3) and B (3 x colsB); C(row, col) is at index0 + row * stride + col.</summary>
	public static void MatrixMatrixMultiplyInner3(
		ReadOnlySpan<double> a, int rowsA, ReadOnlySpan<double> b, int colsB, Span<double> c, int index0, int stride, int operation)
	{
		if (rowsA <= 0 || colsB <= 0)
		{
			return;
		}

		CheckLengths(a.Length >= 3 * rowsA, b.Length >= 3 * colsB, c.Length >= index0 + ((rowsA - 1) * stride) + colsB, index0);
		ref double ar = ref MemoryMarshal.GetReference(a);
		ref double br = ref MemoryMarshal.GetReference(b);
		ref double cr = ref MemoryMarshal.GetReference(c);
		for (int row = 0; row < rowsA; row++)
		{
			double a0 = Unsafe.Add(ref ar, 3 * row);
			double a1 = Unsafe.Add(ref ar, (3 * row) + 1);
			double a2 = Unsafe.Add(ref ar, (3 * row) + 2);
			ref double cRow = ref Unsafe.Add(ref cr, index0 + (row * stride));
			for (int col = 0; col < colsB; col++)
			{
				double tmp = 0.0;
				tmp += a0 * Unsafe.Add(ref br, col);
				tmp += a1 * Unsafe.Add(ref br, colsB + col);
				tmp += a2 * Unsafe.Add(ref br, (2 * colsB) + col);
				Store(ref Unsafe.Add(ref cRow, col), tmp, operation);
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Store(ref double c, double tmp, int operation)
	{
		if (operation > 0)
		{
			c += tmp;
		}
		else if (operation < 0)
		{
			c -= tmp;
		}
		else
		{
			c = tmp;
		}
	}

	// The spans must cover every element the loops touch, since they index unchecked.
	private static void CheckLengths(bool a, bool b, bool c, int index0)
	{
		if (!a || !b || !c || index0 < 0)
		{
			throw new ArgumentOutOfRangeException(nameof(a), "A small BLAS operand is shorter than its dimensions.");
		}
	}
}
