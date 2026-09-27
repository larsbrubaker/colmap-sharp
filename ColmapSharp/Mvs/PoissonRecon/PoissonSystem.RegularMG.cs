// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSystem.RegularMG: the solve at the base depth (thirdparty/PoissonRecon/
// FEMTree.System.inl: _solveRegularMG), where the tree is complete, so the system is solved by
// a classic V-cycle over depths baseDepth..0 of Galerkin matrices (R * M * P, built with
// PoissonSparseMatrix and PoissonMultigrid.DownSampleMatrix): Gauss-Seidel down to depth 1,
// conjugate gradients at depth 0 (PoissonSparseMatrix.SolveCG), Gauss-Seidel back up. The
// cascadic solver (PoissonSystem.Solve) runs it once at the base depth before relaxing the
// finer depths with SolveSystemGS. Tier A against oracle/poisson_system_harness.cc (the
// "regularmg" cases).
//
// Translation notes: residual norms (computeNorms) are not computed, as in SolveSystemGS. The
// C++ leaves the coarse X and B arrays uninitialized; each is written before it is read.

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonSystem
{
	/// <summary>
	/// Runs <paramref name="vCycles"/> V-cycles on the complete depths 0..<paramref name="baseDepth"/>,
	/// updating the base-depth coefficients in <paramref name="solution"/> (indexed by node index)
	/// against <paramref name="constraints"/> (indexed by node index), with <paramref name="iters"/>
	/// Gauss-Seidel sweeps per level up to <paramref name="maxSolveDepth"/> and a CG solve at
	/// depth 0 to <paramref name="cgAccuracy"/>. <paramref name="constrainsDCTerm"/> is the
	/// interpolation info's constrainsDCTerm (true for Solve's point constraints). Port of
	/// <c>_solveRegularMG</c> without residual norms.
	/// </summary>
	public void SolveRegularMG(int baseDepth, int maxSolveDepth, float[] solution, float[] constraints, int vCycles, int iters, double cgAccuracy, bool constrainsDCTerm)
	{
		if (maxSolveDepth > baseDepth)
		{
			throw new InvalidOperationException($"Regular MG depth cannot exceed base depth: {maxSolveDepth} <= {baseDepth}");
		}

		var p = new PoissonSparseMatrix[baseDepth];
		var r = new PoissonSparseMatrix[baseDepth];
		var m = new PoissonSparseMatrix[baseDepth + 1];
		var diagonals = new float[baseDepth + 1][];
		var b = new float[baseDepth + 1][];
		var x = new float[baseDepth + 1][];
		var mx = new float[baseDepth + 1][];
		var colors = new List<int>[baseDepth + 1][];

		// systemMatrix( baseDepth ): the slice rows without a prolonged solution.
		int baseBegin = PoissonMultigrid.Begin(tree, sorted, baseDepth);
		f.Init(baseDepth);
		var rows = new PoissonSystemMatrix(neighbors.Length);
		GetSliceMatrixAndProlongationConstraints(rows, null, baseDepth, baseBegin, PoissonMultigrid.End(tree, sorted, baseDepth), null, null, f.SetStencil(), f.SetParentChildStencils());
		m[baseDepth] = PoissonSparseMatrix.From(rows);
		for (int d = baseDepth; d > 0; d--)
		{
			r[d - 1] = PoissonMultigrid.DownSampleMatrix(tree, sorted, f.Signature, d, baseDepth);
			p[d - 1] = r[d - 1].Transpose(m[d].Rows);
			m[d - 1] = r[d - 1].Multiply(m[d]).Multiply(p[d - 1]);
		}

		for (int d = 0; d <= baseDepth; d++)
		{
			int dim = m[d].Rows;
			diagonals[d] = new float[dim];
			mx[d] = new float[dim];
			m[d].SetDiagonalR(diagonals[d]);
			colors[d] = NewColors();
			SetMultiColorIndices(PoissonMultigrid.Begin(tree, sorted, d), PoissonMultigrid.End(tree, sorted, d), colors[d]);
			if (d < baseDepth)
			{
				x[d] = new float[dim];
				b[d] = new float[dim];
			}
		}

		// The base level reads and writes the caller's arrays in place.
		x[baseDepth] = solution;
		b[baseDepth] = constraints;
		int Offset(int d) => d == baseDepth ? baseBegin : 0;

		for (int v = 0; v < vCycles; v++)
		{
			// Restriction
			for (int d = baseDepth; d > 0; d--)
			{
				if (d <= maxSolveDepth)
				{
					for (int i = 0; i < iters; i++)
					{
						m[d].GsIteration(colors[d], diagonals[d], b[d], Offset(d), x[d], Offset(d), forward: true);
					}
				}

				m[d].Multiply(x[d], Offset(d), mx[d], 0);
				for (int i = 0; i < m[d].Rows; i++)
				{
					mx[d][i] = b[d][Offset(d) + i] - mx[d][i];
				}

				r[d - 1].Multiply(mx[d], 0, b[d - 1], 0);
				Array.Clear(x[d - 1], 0, m[d - 1].Rows);
			}

			// Base
			{
				const int d = 0;
				int nonZeroRows = 0;
				for (int i = 0; i < m[d].Rows; i++)
				{
					if (m[d].RowSize(i) != 0)
					{
						nonZeroRows++;
					}
				}

				// The C++ sizes the full grid at the base depth, not at depth 0.
				BSplineEvaluationData evaluation = BSplineEvaluationData.For(f.Signature);
				long width = evaluation.EndAt(baseDepth) - evaluation.BeginAt(baseDepth);
				long totalDim = width * width * width;
				bool hasPartitionOfUnity = FemSignature.HasPartitionOfUnity(FemSignature.Boundary(f.Signature));
				bool addDCTerm = nonZeroRows == totalDim && !constrainsDCTerm && hasPartitionOfUnity && f.VanishesOnConstants;
				m[d].SolveCG(addDCTerm, b[d], Offset(d), nonZeroRows, x[d], Offset(d), (float)cgAccuracy);
			}

			// Prolongation
			for (int d = 1; d <= baseDepth; d++)
			{
				p[d - 1].Multiply(x[d - 1], 0, x[d], Offset(d), add: true);
				for (int i = 0; i < m[d - 1].Rows; i++)
				{
					x[d - 1][i] *= 0;
				}

				if (d <= maxSolveDepth)
				{
					for (int i = 0; i < iters; i++)
					{
						m[d].GsIteration(colors[d], diagonals[d], b[d], Offset(d), x[d], Offset(d), forward: false);
					}
				}
			}
		}
	}

	private static List<int>[] NewColors()
	{
		var colors = new List<int>[ColorModulusPerAxis * ColorModulusPerAxis * ColorModulusPerAxis];
		for (int c = 0; c < colors.Length; c++)
		{
			colors[c] = [];
		}

		return colors;
	}
}
