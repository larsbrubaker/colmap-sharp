// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests.System (C#-only, not a COLMAP test): the system assembly of
// Poisson::Solver::Solve after finalizeForMultigrid - the normal-divergence constraints
// (PoissonFemConstraints, with PoissonMultigrid's restriction and prolongation), the point
// interpolation constraints, and the solver's per-depth matrix rows and prolongation
// constraints and point-constraint transfers (PoissonSystem, PoissonSystemMatrix) - against the
// "system*" runs of oracle/poisson_system_harness.cc (TestData/oracle/poisson_system.json).
// Tier A, bit-identical floats. Each run replays the harness's input through the stages the
// other PoissonTreeOracleTests files check step by step, then compares the stages here.

using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public partial class PoissonTreeOracleTests
{
	// oracle/poisson_system_harness.cc's system assembly.
	private const string SystemFixture = "poisson_system.json";

	[Test]
	[Arguments("system3", 3)]
	[Arguments("system5", 5)]
	[Arguments("system6", 6)]
	[Arguments("system8", 8)]
	public async Task SystemConstraints_MatchHarness(string name, int depth)
	{
		JsonElement cases = OracleFixture.Load(SystemFixture).GetProperty("cases");
		var produced = new Cases(name);
		(FemTree tree, SortedTreeNodes sorted, Prepared p) = Finalize(cases, name, depth, produced);

		// Solve: addFEMConstraints( F , *normalInfo , constraints , solveDepth ), and the harness's
		// extra run with a shallower maxDepth.
		FemConstraintIntegrator f = PoissonFemConstraints.CreateDivergenceIntegrator();
		var shallow = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, f, p.Normals, shallow, depth - 2);
		produced.F("femconstraintsshallow", shallow.Select(v => (double)v).ToList());
		var constraints = new float[sorted.Size];
		PoissonFemConstraints.AddFemConstraints(tree, sorted, f, p.Normals, constraints, depth);
		produced.F("femconstraints", constraints.Select(v => (double)v).ToList());

		// Solve: addInterpolationConstraints( constraints , solveDepth , iInfo ).
		PoissonFemConstraints.AddInterpolationConstraints(tree, sorted, p.Interpolation, constraints, depth);
		produced.F("interpolationconstraints", constraints.Select(v => (double)v).ToList());
		AddSliceCases(produced, tree, sorted, p, depth, constraints);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	// The harness's solver-assembly block: per depth of at most SliceLimit nodes, the matrix rows,
	// prolongation constraints and inverse diagonals of
	// _getSliceMatrixAndProlongationConstraints over a synthetic prolonged solution, and
	// _getProlongedMatrixRowSize of every valid node; then the point-constraint transfers and the
	// sliced Gauss-Seidel relaxation.
	private static void AddSliceCases(Cases produced, FemTree tree, SortedTreeNodes sorted, Prepared p, int solveDepth, float[] systemConstraints)
	{
		const int SliceLimit = 12000;
		var f = new FemSystemIntegrator(PoissonFemConstraints.TestSignature, 1, 0.0, 1.0);
		var bsData = new PoissonPointEvaluator(PoissonFemConstraints.TestSignature, 1, solveDepth);
		var system = new PoissonSystem(tree, sorted, f, bsData, p.Interpolation, p.PointWeight);
		int maxDepth = PoissonMultigrid.MaxDepth(tree);
		var prolonged = new float[PoissonMultigrid.End(tree, sorted, maxDepth - 1)];
		for (int i = 0; i < prolonged.Length; i++)
		{
			prolonged[i] = (float)((long)(i * 37L % 101) - 50) / 64f;
		}

		var depths = new List<double>();
		var rowSizes = new List<double>();
		var columns = new List<double>();
		var values = new List<double>();
		var constraints = new List<double>();
		var diagonal = new List<double>();
		var prolongedRowSizes = new List<double>();
		var matrix = new PoissonSystemMatrix(27);
		var parentWindow = new int[27];
		for (int d = 1; d <= maxDepth; d++)
		{
			int begin = PoissonMultigrid.Begin(tree, sorted, d), end = PoissonMultigrid.End(tree, sorted, d);
			if (end - begin > SliceLimit)
			{
				continue;
			}

			depths.Add(d);
			f.Init(d);
			double[] cc = f.SetStencil();
			double[][] pc = f.SetParentChildStencils();
			var diag = new float[end - begin];
			var cons = new float[end - begin];
			system.GetSliceMatrixAndProlongationConstraints(matrix, diag, d, begin, end, prolonged, cons, cc, pc);
			for (int i = 0; i < end - begin; i++)
			{
				rowSizes.Add(matrix.RowSize(i));
				for (int j = 0; j < matrix.RowSize(i); j++)
				{
					columns.Add(matrix.Column(i, j));
					values.Add(matrix.Value(i, j));
				}

				constraints.Add(cons[i]);
				diagonal.Add(PoissonMultigrid.IsValidFem1Node(tree, sorted.TreeNodes[i + begin]) ? diag[i] : 0.0);
			}

			var key = new NeighborKey(tree, 1, 1, resetOnMissing: false);
			key.Set(d + tree.DepthOffset);
			for (int i = begin; i < end; i++)
			{
				int node = sorted.TreeNodes[i];
				if (!PoissonMultigrid.IsValidFem1Node(tree, node))
				{
					continue;
				}

				key.GetNeighbors(1, 1, tree.Parent(node), parentWindow);
				prolongedRowSizes.Add(system.GetProlongedMatrixRowSize(node, parentWindow));
			}
		}

		produced.I("slicedepths", depths);
		produced.I("slicerowsizes", rowSizes);
		produced.I("slicecolumns", columns);
		produced.F("slicevalues", values);
		produced.F("sliceconstraints", constraints);
		produced.F("slicediagonal", diagonal);
		produced.I("prolongedrowsizes", prolongedRowSizes);

		// The point-constraint transfers, as the harness runs them.
		for (int d = 1; d <= maxDepth; d++)
		{
			system.SetPointValuesFromProlongedSolution(d, prolonged);
		}

		var pointValues = new List<double>();
		tree.ProcessNodes(tree.Root, node =>
		{
			int slot = p.Interpolation.Index(tree.NodeIndex(node));
			if (slot != -1)
			{
				pointValues.AddRange([tree.NodeIndex(node), p.Interpolation.Value(slot, 4)]);
			}
		});
		produced.F("prolongedpointvalues", pointValues);
		var solution = new float[PoissonMultigrid.End(tree, sorted, maxDepth)];
		for (int i = 0; i < solution.Length; i++)
		{
			solution[i] = (float)((long)(i * 53L % 97) - 48) / 32f;
		}

		var restricted = new float[PoissonMultigrid.End(tree, sorted, maxDepth - 1)];
		for (int d = 1; d <= maxDepth; d++)
		{
			system.UpdateRestrictedInterpolationConstraints(d, solution, restricted);
		}

		produced.F("restrictedinterpolation", restricted.Select(v => (double)v).ToList());

		// The sliced Gauss-Seidel runs, from a zero solution.
		List<double> Gs(int iters, bool coarseToFine, int sliceBlockSize)
		{
			var x = new float[PoissonMultigrid.End(tree, sorted, maxDepth)];
			var result = new List<double>();
			for (int d = 1; d <= maxDepth; d++)
			{
				int begin = PoissonMultigrid.Begin(tree, sorted, d), end = PoissonMultigrid.End(tree, sorted, d);
				if (end - begin > SliceLimit)
				{
					continue;
				}

				f.Init(d);
				system.SolveSystemGS(d, x, prolonged, systemConstraints, iters, coarseToFine, sliceBlockSize);
				for (int i = begin; i < end; i++)
				{
					result.Add(x[i]);
				}
			}

			return result;
		}

		produced.F("gsprolongation", Gs(8, coarseToFine: true, 1));
		produced.F("gsblocked", Gs(3, coarseToFine: false, 2));
		AddSparseCases(produced, tree, sorted, system, f, (int)p.Parameters.BaseDepth, SliceLimit, systemConstraints);
	}

	// The harness's base-depth multigrid algebra: DownSampleMatrix, its transpose, the two
	// products and the reciprocal diagonal of the system matrix, per small depth.
	private static void AddSparseCases(Cases produced, FemTree tree, SortedTreeNodes sorted, PoissonSystem system, FemSystemIntegrator f, int baseDepth, int sliceLimit, float[] systemConstraints)
	{
		var shape = new List<double>();
		var entries = new List<double>();
		var products = new List<double>();
		var diagonals = new List<double>();
		void DumpMatrix(PoissonSparseMatrix m)
		{
			shape.Add(m.Rows);
			for (int i = 0; i < m.Rows; i++)
			{
				shape.Add(m.RowSize(i));
				for (int j = 0; j < m.RowSize(i); j++)
				{
					shape.Add(m.Column(i, j));
					entries.Add(m.Value(i, j));
				}
			}
		}

		int maxDepth = PoissonMultigrid.MaxDepth(tree);
		var sliceMatrix = new PoissonSystemMatrix(27);
		for (int d = 1; d <= maxDepth; d++)
		{
			int highBegin = PoissonMultigrid.Begin(tree, sorted, d), highEnd = PoissonMultigrid.End(tree, sorted, d);
			int high = highEnd - highBegin, low = PoissonMultigrid.End(tree, sorted, d - 1) - PoissonMultigrid.Begin(tree, sorted, d - 1);
			if (high > sliceLimit)
			{
				continue;
			}

			PoissonSparseMatrix r = PoissonMultigrid.DownSampleMatrix(tree, sorted, PoissonFemConstraints.TestSignature, d, baseDepth);
			PoissonSparseMatrix pm = r.Transpose(high);
			DumpMatrix(r);
			DumpMatrix(pm);
			var x = new float[high];
			var y = new float[low];
			var z = new float[high];
			for (int i = 0; i < high; i++)
			{
				x[i] = (float)((long)(i * 29L % 83) - 41) / 16f;
				z[i] = (float)((long)(i * 7L % 13) - 6) / 8f;
			}

			r.Multiply(x, 0, y, 0);
			pm.Multiply(y, 0, z, 0, add: true);
			products.AddRange(y.Select(v => (double)v));
			products.AddRange(z.Select(v => (double)v));

			// systemMatrix( d ): the slice rows without a prolonged solution.
			f.Init(d);
			system.GetSliceMatrixAndProlongationConstraints(sliceMatrix, null, d, highBegin, highEnd, null, null, f.SetStencil(), f.SetParentChildStencils());
			var diagonal = new float[high];
			PoissonSparseMatrix.From(sliceMatrix).SetDiagonalR(diagonal);
			diagonals.AddRange(diagonal.Select(v => (double)v));
		}

		produced.I("sparseshape", shape);
		produced.F("sparseentries", entries);
		produced.F("sparseproducts", products);
		produced.F("sparsediagonals", diagonals);

		// _solveRegularMG's Galerkin chain from systemMatrix( baseDepth ).
		var galerkinShape = new List<double> { baseDepth };
		var galerkinEntries = new List<double>();
		f.Init(baseDepth);
		system.GetSliceMatrixAndProlongationConstraints(sliceMatrix, null, baseDepth, PoissonMultigrid.Begin(tree, sorted, baseDepth), PoissonMultigrid.End(tree, sorted, baseDepth), null, null, f.SetStencil(), f.SetParentChildStencils());
		PoissonSparseMatrix m = PoissonSparseMatrix.From(sliceMatrix);
		for (int d = baseDepth; d > 0; d--)
		{
			PoissonSparseMatrix r = PoissonMultigrid.DownSampleMatrix(tree, sorted, PoissonFemConstraints.TestSignature, d, baseDepth);
			m = r.Multiply(m).Multiply(r.Transpose(m.Rows));
			galerkinShape.Add(m.Rows);
			for (int i = 0; i < m.Rows; i++)
			{
				galerkinShape.Add(m.RowSize(i));
				for (int j = 0; j < m.RowSize(i); j++)
				{
					galerkinShape.Add(m.Column(i, j));
					galerkinEntries.Add(m.Value(i, j));
				}
			}
		}

		produced.I("galerkinshape", galerkinShape);
		produced.F("galerkinentries", galerkinEntries);

		// _solveRegularMG from a zero solution, as the harness runs it.
		List<double> RegularMG(int maxSolveDepth, int vCycles, int iters)
		{
			var x = new float[PoissonMultigrid.End(tree, sorted, maxDepth)];
			system.SolveRegularMG(baseDepth, maxSolveDepth, x, systemConstraints, vCycles, iters, (double)1e-3f, constrainsDCTerm: true);
			var result = new List<double>();
			for (int i = PoissonMultigrid.Begin(tree, sorted, baseDepth); i < PoissonMultigrid.End(tree, sorted, baseDepth); i++)
			{
				result.Add(x[i]);
			}

			return result;
		}

		produced.F("regularmg", RegularMG(baseDepth, 1, 8));
		produced.F("regularmgshallow", RegularMG(baseDepth - 1, 2, 3));
	}

	// The harness's Run up to and including finalizeForMultigrid, emitting its sorted slices.
	private static (FemTree Tree, SortedTreeNodes Sorted, Prepared P) Finalize(JsonElement cases, string name, int depth, Cases produced)
	{
		Prepared p = Prepare(cases, name, depth, new Cases(name));
		FemTree tree = p.Set.Tree;
		SortedTreeNodes sorted = PoissonFinalize.FinalizeForMultigrid(
			tree, MaxDegree, (int)p.Parameters.BaseDepth, (int)p.Parameters.FullDepth, p.Normals, out _, p.Interpolation, p.Normals, p.Density, p.Colors);
		var slices = new List<double>();
		for (int d = 0; d < sorted.Levels; d++)
		{
			slices.AddRange([sorted.Begin(d), sorted.End(d)]);
		}

		produced.I("sortedslices", slices);
		return (tree, sorted, p);
	}
}
