// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSystem.Solve: FEMTree::solveSystem (thirdparty/PoissonRecon/FEMTree.System.inl) with
// the SolverInfo Poisson::Solver::Solve uses (Reconstructors.h): cascadic, one V-cycle, a
// cleared initial solution, cgDepth 0, one slice per Gauss-Seidel block, SOR weights 1, one
// base V-cycle. With those settings the restriction phase does nothing (no restricted
// constraints are kept), and the prolongation phase walks the depths coarse to fine: the base
// depth is solved by SolveRegularMG, every finer depth by SolveSystemGS against what the
// prolonged coarser solution already meets, and after each depth the prolonged solution is
// up-sampled (PoissonMultigrid.UpSample), the new depth's solution added, and the point
// constraints' dual values reset from it (SetPointValuesFromProlongedSolution). Tier A against
// oracle/poisson_system_harness.cc (the "solve*" cases).
//
// Translation notes: the C++ prints per-depth statistics when verbose; the port reports the
// solved depth through IProgress and checks the CancellationToken before each depth instead.

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonSystem
{
	/// <summary>
	/// Solves the system for the coefficients of every sorted node (the returned array, indexed
	/// by node index) against <paramref name="constraints"/> (indexed by node index), from depth
	/// max(<paramref name="minSolveDepth"/>, <paramref name="baseDepth"/>) to
	/// min(<paramref name="maxSolveDepth"/>, tree max depth), with Solve's solver settings and
	/// <paramref name="iters"/> Gauss-Seidel sweeps per depth (Solve's params.iters, 8) and
	/// <paramref name="baseVCycles"/> V-cycles at the base depth (params.baseVCycles, 1).
	/// <paramref name="cgAccuracy"/> is Solve's params.cgSolverAccuracy (the float 1e-3).
	/// <paramref name="progress"/> receives each depth once it is solved. Port of
	/// <c>solveSystem( Sigs , F , constraints , baseDepth , solveDepth , sInfo , iInfo )</c>.
	/// </summary>
	public float[] Solve(float[] constraints, int baseDepth, int minSolveDepth, int maxSolveDepth, int iters, int baseVCycles, double cgAccuracy, bool constrainsDCTerm, IProgress<int>? progress = null, CancellationToken cancellationToken = default)
	{
		int treeMaxDepth = PoissonMultigrid.MaxDepth(tree);
		var solution = new float[PoissonMultigrid.End(tree, sorted, treeMaxDepth)];
		maxSolveDepth = StdMinMax.StdMin(maxSolveDepth, treeMaxDepth);
		if (minSolveDepth > maxSolveDepth)
		{
			return solution;
		}

		minSolveDepth = StdMinMax.StdMax(minSolveDepth, baseDepth);

		// Mark all nodes that define valid finite elements
		PoissonMultigrid.SetFem1ValidityFlags(tree, sorted, f.Signature);
		int coarseEnd = PoissonMultigrid.End(tree, sorted, treeMaxDepth - 1);
		var residualConstraints = new float[coarseEnd];
		var prolongedSolution = new float[coarseEnd];
		var rp = new RestrictionProlongation(f.Signature);

		// The prolongation phase of the single cascadic V-cycle.
		int lastDepth = StdMinMax.StdMax(baseDepth, maxSolveDepth);
		for (int d = minSolveDepth; d <= lastDepth; d++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			f.Init(d);
			int begin = PoissonMultigrid.Begin(tree, sorted, d), end = PoissonMultigrid.End(tree, sorted, d);

			// SetResidualConstraints: copy the constraints (there are no restricted ones to subtract).
			if (d < treeMaxDepth)
			{
				Array.Copy(constraints, begin, residualConstraints, begin, end - begin);
			}

			float[] depthConstraints = d == treeMaxDepth ? constraints : residualConstraints;
			if (d == baseDepth)
			{
				if (baseVCycles != 0)
				{
					SolveRegularMG(baseDepth, StdMinMax.StdMin(baseDepth, maxSolveDepth), solution, depthConstraints, baseVCycles, iters, cgAccuracy, constrainsDCTerm);
				}
			}
			else
			{
				SolveSystemGS(d, solution, prolongedSolution, depthConstraints, iters, coarseToFine: true, sliceBlockSize: 1);
			}

			UpdateProlongation(d, baseDepth, treeMaxDepth, solution, prolongedSolution, rp);
			progress?.Report(d);
		}

		return solution;
	}

	// solveSystem's UpdateProlongation lambda.
	private void UpdateProlongation(int depth, int baseDepth, int treeMaxDepth, float[] solution, float[] prolongedSolution, RestrictionProlongation rp)
	{
		if (depth == treeMaxDepth)
		{
			return;
		}

		int begin = PoissonMultigrid.Begin(tree, sorted, depth), end = PoissonMultigrid.End(tree, sorted, depth);
		if (depth == baseDepth)
		{
			Array.Copy(solution, begin, prolongedSolution, begin, end - begin);
		}
		else if (depth > baseDepth)
		{
			// Clear the prolonged solution @(depth), up-sample the prolonged solution @(depth-1)
			// into it, and add in the solution @(depth)
			Array.Clear(prolongedSolution, begin, end - begin);
			f.Init(depth);
			PoissonMultigrid.UpSample(tree, sorted, rp, depth, prolongedSolution, 0, prolongedSolution, 0, 1);
			for (int i = begin; i < end; i++)
			{
				prolongedSolution[i] += solution[i];
			}
		}

		if (depth + 1 > baseDepth && depth + 1 <= treeMaxDepth)
		{
			SetPointValuesFromProlongedSolution(depth + 1, prolongedSolution);
		}
	}
}
