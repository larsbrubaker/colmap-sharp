// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SiftMatchKernelsTests: C#-only tests (no COLMAP counterpart) for
// ColmapSharp/Feature/SiftMatchKernels.cs. COLMAP fills the whole N x M distance matrix and
// scans rows and the transpose; the port fuses that into one parallel integer scan. These
// tests pin the fused scan to a direct transcription of COLMAP's matrix scan
// (FindBestMatchesOneWayBruteForce / FindBestMatchesOneWayIndex loops) on tie-heavy random
// descriptors, for several thread counts, and the SIMD dot product to the scalar definition.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftMatchKernelsTests
{
	// Values from a tiny range make equal distances (ties) common.
	private static RowMajorMatrix<byte> RandomDescriptors(Random random, int rows, int maxValue)
	{
		var m = new RowMajorMatrix<byte>(rows, 128);
		for (int i = 0; i < m.Size; ++i)
		{
			m.Data[i] = (byte)random.Next(maxValue + 1);
		}

		return m;
	}

	private static int ScalarDot(RowMajorMatrix<byte> a, int ra, RowMajorMatrix<byte> b, int rb)
	{
		int sum = 0;
		for (int c = 0; c < 128; ++c)
		{
			sum += a[ra, c] * b[rb, c];
		}

		return sum;
	}

	private static int ScalarL2(RowMajorMatrix<byte> a, int ra, RowMajorMatrix<byte> b, int rb)
	{
		int sum = 0;
		for (int c = 0; c < 128; ++c)
		{
			int d = a[ra, c] - b[rb, c];
			sum += d * d;
		}

		return sum;
	}

	// COLMAP's loop over one row of the matrix: (best idx, best, second) as floats.
	private static (int Idx, float Best, float Second) ColmapRowScan(float[] row, bool useL2)
	{
		int bestIdx = -1;
		float best = useL2 ? float.MaxValue : 0;
		float second = best;
		for (int i = 0; i < row.Length; ++i)
		{
			float v = row[i];
			bool better = useL2 ? v < best : v > best;
			bool betterSecond = useL2 ? v < second : v > second;
			if (better)
			{
				bestIdx = i;
				second = best;
				best = v;
			}
			else if (betterSecond)
			{
				second = v;
			}
		}

		return (bestIdx, best, second);
	}

	private static async Task ExpectSame(Top2 fused, (int Idx, float Best, float Second) colmap, bool useL2)
	{
		await Assert.That(fused.BestIdx).IsEqualTo(colmap.Idx);
		if (fused.BestIdx == -1)
		{
			return;
		}

		float best = useL2 ? -fused.Best : fused.Best;
		float second = useL2 ? (fused.Second == -int.MaxValue ? float.MaxValue : -fused.Second) : fused.Second;
		await Assert.That(best).IsEqualTo(colmap.Best);
		await Assert.That(second).IsEqualTo(colmap.Second);
	}

	[Test]
	[Arguments(false, 1)]
	[Arguments(false, 3)]
	[Arguments(false, 8)]
	[Arguments(true, 1)]
	[Arguments(true, 5)]
	public async Task FusedScan_MatchesColmapMatrixScan(bool useL2, int numThreads)
	{
		var random = new Random(42);
		RowMajorMatrix<byte> d1 = RandomDescriptors(random, 300, 2);
		RowMajorMatrix<byte> d2 = RandomDescriptors(random, 170, 2);

		// Some exact duplicate rows so the best itself ties.
		d1.Row(3).CopyTo(d1.Row(7));
		d2.Row(5).CopyTo(d2.Row(9));
		d2.Row(0).Clear();

		// A guided filter that rejects a pattern of pairs, to cover the rejected-pair value.
		Func<int, int, bool> filter = (i1, i2) => (i1 + (2 * i2)) % 5 == 0;

		var rows = new Top2[d1.Rows];
		var cols = new Top2[d2.Rows];
		SiftMatchKernels.ScanDistances(d1, d2, useL2, filter, numThreads, rows, cols);

		float Distance(int i1, int i2)
		{
			if (filter(i1, i2))
			{
				return useL2 ? SiftMatchKernels.SqSiftDescriptorNorm : 0;
			}

			return useL2 ? ScalarL2(d1, i1, d2, i2) : ScalarDot(d1, i1, d2, i2);
		}

		for (int i1 = 0; i1 < d1.Rows; ++i1)
		{
			var row = new float[d2.Rows];
			for (int i2 = 0; i2 < d2.Rows; ++i2)
			{
				row[i2] = Distance(i1, i2);
			}

			await ExpectSame(rows[i1], ColmapRowScan(row, useL2), useL2);
		}

		for (int i2 = 0; i2 < d2.Rows; ++i2)
		{
			var col = new float[d1.Rows];
			for (int i1 = 0; i1 < d1.Rows; ++i1)
			{
				col[i1] = Distance(i1, i2);
			}

			await ExpectSame(cols[i2], ColmapRowScan(col, useL2), useL2);
		}
	}

	[Test]
	public async Task DotProduct_MatchesScalarDefinition()
	{
		var random = new Random(7);
		RowMajorMatrix<byte> a = RandomDescriptors(random, 64, 255);
		RowMajorMatrix<byte> b = RandomDescriptors(random, 64, 255);

		// All-255 rows: the largest possible dot product.
		a.Row(0).Fill(255);
		b.Row(0).Fill(255);
		for (int i = 0; i < a.Rows; ++i)
		{
			for (int j = 0; j < b.Rows; ++j)
			{
				int simd = SiftMatchKernels.DotProduct(ref a.Data[i * 128], ref b.Data[j * 128]);
				await Assert.That(simd).IsEqualTo(ScalarDot(a, i, b, j));
			}
		}
	}
}
