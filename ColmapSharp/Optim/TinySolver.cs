// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP colmap/optim/tiny_solver.h, which is COLMAP's customized copy of
// ceres::TinySolver (Ceres Solver, BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// A tiny dense Levenberg-Marquardt solver for small problems with low latency and low
// overhead: the essential/fundamental-matrix and relative-pose-with-focal refinements of
// estimators/solvers/ run it once per RANSAC model. COLMAP's copy differs from upstream
// Ceres in three ways, all kept here: (1) an optional manifold decouples the ambient
// parameter block from the tangent step (Plus to step, PlusJacobian to project the ambient
// Jacobian); (2) cost-function and linear-solver failures are handled (the step is rejected,
// or COST_FUNCTION_FAILED is reported at the start); (3) the parameter and tangent sizes are
// fixed per problem, only the residual count varies.
//
// Algorithm: K. Madsen, H. Nielsen, O. Tingleff, "Methods for Non-linear Least Squares
// Problems", 2004.
//
// Translation notes:
// - C++'s compile-time sizes become ITinySolverFunction.NumParameters (static abstract, the
//   C++ enum NUM_PARAMETERS) and the manifold's AmbientSize/TangentSize. The residual count
//   comes from the function instance (NumResiduals(), which C++ needs only when dynamic).
// - Function and manifold are generic type parameters: implement them as structs so the JIT
//   specializes the solver with direct calls, as the C++ templates do.
// - The Eigen LDLT linear solver is LinearAlgebra/LDLT.cs, with its Info checked as C++
//   checks info(). Buffers are allocated once per residual count and reused across solves;
//   LDLT itself allocates its factor per iteration (a tangent-size square, at most 8x8 for
//   COLMAP's callers), which is negligible next to the residual evaluation.
// - The manifold policies (Euclidean here, the rest in
//   Estimators/CostFunctions/TinyManifold.cs) and the autodiff adapter
//   (Solver/TinySolverAutoDiffFunction.cs) are this file's neighbors.
//
// Result tier: C (iterative). Scalar steps match the C++ line for line, but the Eigen
// products and LDLT are replaced, so iterates agree to rounding, not bit for bit.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// The cost function TinySolver minimizes: C++'s TinySolver "Function" concept
/// (Scalar = double, NUM_PARAMETERS, NUM_RESIDUALS / NumResiduals(), operator()).
/// </summary>
public interface ITinySolverFunction
{
	/// <summary>Size of the ambient parameter block, C++'s NUM_PARAMETERS.</summary>
	static abstract int NumParameters { get; }

	/// <summary>Number of residuals, C++'s NUM_RESIDUALS or NumResiduals().</summary>
	int NumResiduals { get; }

	/// <summary>
	/// Evaluates the residuals at <paramref name="parameters"/> (NumParameters long) and,
	/// unless <paramref name="jacobian"/> is empty (C++'s nullptr), the column-major
	/// NumResiduals x NumParameters Jacobian with respect to the ambient parameters.
	/// Returns false if the function cannot be evaluated there.
	/// </summary>
	bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian);
}

/// <summary>
/// A manifold policy for TinySolver (the concept described in COLMAP's
/// estimators/cost_functions/tiny_manifold.h). Stateless and evaluated on doubles only.
/// </summary>
public interface ITinyManifold
{
	/// <summary>Size of the parameter block, C++'s kAmbientSize.</summary>
	int AmbientSize { get; }

	/// <summary>Degrees of freedom, C++'s kTangentSize.</summary>
	int TangentSize { get; }

	/// <summary>True iff Plus is plain addition, C++'s kIsEuclidean.</summary>
	bool IsEuclidean { get; }

	/// <summary>xPlusDelta = Plus(x, delta), of sizes [ambient, tangent, ambient].</summary>
	void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta);

	/// <summary>The row-major AmbientSize x TangentSize Jacobian of Plus at delta = 0.</summary>
	void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian);
}

/// <summary>
/// Port of colmap::EuclideanManifold&lt;N&gt;: the identity manifold, tangent space equal to
/// the ambient space and Plus plain addition. It reproduces plain ceres::TinySolver.
/// </summary>
public readonly struct TinyEuclideanManifold : ITinyManifold
{
	/// <summary>Creates the Euclidean manifold R^<paramref name="size"/>.</summary>
	public TinyEuclideanManifold(int size)
	{
		Check.Gt(size, 0);
		Size = size;
	}

	/// <summary>N, the dimension.</summary>
	public int Size { get; }

	/// <inheritdoc/>
	public int AmbientSize => Size;

	/// <inheritdoc/>
	public int TangentSize => Size;

	/// <inheritdoc/>
	public bool IsEuclidean => true;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		for (int i = 0; i < Size; ++i)
		{
			xPlusDelta[i] = x[i] + delta[i];
		}
	}

	/// <summary>The identity. Unused by the solver's Euclidean fast path, as in COLMAP.</summary>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		jacobian[..(Size * Size)].Clear();
		for (int i = 0; i < Size; ++i)
		{
			jacobian[i * Size + i] = 1;
		}
	}
}

/// <summary>
/// C++'s EuclideanManifold&lt;1&gt; as a stateless struct, so a product that contains it
/// (e.g. pose plus log-focal in the relative-pose-with-focal solvers) is correct when
/// default-constructed. <c>default(TinyEuclideanManifold)</c> has size 0 and must not be
/// used that way.
/// </summary>
public readonly struct TinyEuclideanManifold1 : ITinyManifold
{
	/// <inheritdoc/>
	public int AmbientSize => 1;

	/// <inheritdoc/>
	public int TangentSize => 1;

	/// <inheritdoc/>
	public bool IsEuclidean => true;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta) =>
		xPlusDelta[0] = x[0] + delta[0];

	/// <summary>The 1x1 identity.</summary>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian) => jacobian[0] = 1;
}

/// <summary>TinySolver's termination reason, C++'s TinySolver::Status.</summary>
public enum TinySolverStatus
{
	/// <summary>max_norm |J'(x) * f(x)| &lt; gradient_tolerance.</summary>
	GradientTooSmall,

	/// <summary>||dx|| &lt;= parameter_tolerance * (||x|| + parameter_tolerance).</summary>
	RelativeStepSizeTooSmall,

	/// <summary>cost_threshold &gt; ||f(x)||^2 / 2.</summary>
	CostTooSmall,

	/// <summary>num_iterations &gt;= max_num_iterations.</summary>
	HitMaxIterations,

	/// <summary>(new_cost - old_cost) &lt; function_tolerance * old_cost.</summary>
	CostChangeTooSmall,

	/// <summary>
	/// The cost function failed to evaluate at the initial point (or, unexpectedly, at an
	/// accepted point), so no meaningful step could be taken.
	/// </summary>
	CostFunctionFailed,
}

/// <summary>TinySolver's options, C++'s TinySolver::Options.</summary>
public sealed class TinySolverOptions
{
	/// <summary>Iteration cap.</summary>
	public int MaxNumIterations { get; set; } = 50;

	/// <summary>max_norm |J'(x) * f(x)| &lt; gradient_tolerance stops.</summary>
	public double GradientTolerance { get; set; } = 1e-10;

	/// <summary>||dx|| &lt;= parameter_tolerance * (||x|| + parameter_tolerance) stops.</summary>
	public double ParameterTolerance { get; set; } = 1e-8;

	/// <summary>(new_cost - old_cost) &lt; function_tolerance * old_cost stops.</summary>
	public double FunctionTolerance { get; set; } = 1e-6;

	/// <summary>cost_threshold &gt; ||f(x)||^2 / 2 stops. Default: machine epsilon.</summary>
	public double CostThreshold { get; set; } = 2.220446049250313E-16;

	/// <summary>The first trust region radius; the LM damping starts at its inverse.</summary>
	public double InitialTrustRegionRadius { get; set; } = 1e4;

	/// <summary>Validates the options (C++'s Options::Check()).</summary>
	public void Check()
	{
		Util.Check.Gt(MaxNumIterations, 0);
		Util.Check.Ge(GradientTolerance, 0.0);
		Util.Check.Ge(ParameterTolerance, 0.0);
		Util.Check.Ge(FunctionTolerance, 0.0);
		Util.Check.Ge(CostThreshold, 0.0);
		Util.Check.Gt(InitialTrustRegionRadius, 0.0);
	}
}

/// <summary>TinySolver's result, C++'s TinySolver::Summary.</summary>
public readonly record struct TinySolverSummary(
	double InitialCost,
	double FinalCost,
	double GradientMaxNorm,
	int Iterations,
	TinySolverStatus Status);

/// <summary>
/// Port of colmap::TinySolver: small dense Levenberg-Marquardt over a manifold. One instance
/// can solve many problems; its buffers are reused. Not thread safe.
/// </summary>
public sealed class TinySolver<TFunction, TManifold>
	where TFunction : ITinySolverFunction
	where TManifold : struct, ITinyManifold
{
	private readonly TManifold manifold;
	private readonly int numParameters;
	private readonly int numTangent;

	private readonly double[] xNew;
	private readonly double[] dx;
	private readonly double[] g;
	private readonly double[] jacobiScaling;
	private readonly double[] plusJacobian;
	private readonly MatrixXd jtj;
	private readonly MatrixXd jtjRegularized;
	private readonly VectorXd gVector;

	private double[] residuals = [];
	private double[] fxNew = [];

	// Column-major NumResiduals x NumParameters, written by the function when the manifold
	// is not Euclidean, then projected into `jacobian`.
	private double[] jacobianAmbient = [];

	// Column-major NumResiduals x NumTangent: the tangent-space Jacobian.
	private double[] jacobian = [];

	private double cost;
	private int iterations;
	private double gradientMaxNorm;

	/// <summary>Creates a solver for the given manifold (C++ default-constructs it).</summary>
	public TinySolver(TManifold manifold)
	{
		Check.Eq(TFunction.NumParameters, manifold.AmbientSize, "Function::NUM_PARAMETERS must match Manifold::kAmbientSize.");
		this.manifold = manifold;
		numParameters = TFunction.NumParameters;
		numTangent = manifold.TangentSize;
		Check.Gt(numTangent, 0);
		xNew = new double[numParameters];
		dx = new double[numTangent];
		g = new double[numTangent];
		jacobiScaling = new double[numTangent];
		plusJacobian = new double[numParameters * numTangent];
		jtj = new MatrixXd(numTangent, numTangent);
		jtjRegularized = new MatrixXd(numTangent, numTangent);
		gVector = new VectorXd(numTangent);
	}

	/// <summary>
	/// Minimizes 1/2 ||f(x)||^2 starting from <paramref name="xAndMin"/>, which receives the
	/// result. Options default to <see cref="TinySolverOptions"/>'s defaults.
	/// </summary>
	public TinySolverSummary Solve(in TFunction function, Span<double> xAndMin, TinySolverOptions? options = null)
	{
		options ??= new TinySolverOptions();
		Check.Eq(xAndMin.Length, numParameters);
		options.Check();
		Initialize(function);
		Span<double> x = xAndMin;
		iterations = 0;
		gradientMaxNorm = -1;

		// Bail out cleanly if the cost function cannot be evaluated at the initial point;
		// there is nothing meaningful the solver can do in that case.
		if (!Update(function, x))
		{
			return new TinySolverSummary(-1, -1, gradientMaxNorm, iterations, TinySolverStatus.CostFunctionFailed);
		}

		double initialCost = cost;
		if (gradientMaxNorm < options.GradientTolerance)
		{
			return new TinySolverSummary(initialCost, cost, gradientMaxNorm, iterations, TinySolverStatus.GradientTooSmall);
		}

		if (cost < options.CostThreshold)
		{
			return new TinySolverSummary(initialCost, cost, gradientMaxNorm, iterations, TinySolverStatus.CostTooSmall);
		}

		TinySolverStatus status = TinySolverStatus.HitMaxIterations;
		double u = 1.0 / options.InitialTrustRegionRadius;
		double v = 2;
		Span<double> lmStep = stackalloc double[numTangent];

		for (iterations = 1; iterations < options.MaxNumIterations; iterations++)
		{
			const double minDiagonal = 1e-6;
			const double maxDiagonal = 1e32;
			jtj.AsSpan().CopyTo(jtjRegularized.AsSpan());
			for (int i = 0; i < numTangent; ++i)
			{
				jtjRegularized[i, i] += u * Math.Min(Math.Max(jtj[i, i], minDiagonal), maxDiagonal);
			}

			var linearSolver = new LDLT(jtjRegularized);

			// If the factorization failed, the regularized normal equations were not
			// solvable. Treat this like a rejected step: shrink the trust region (which
			// increases the regularization) and try again.
			if (linearSolver.Info != ComputationInfo.Success)
			{
				u *= v;
				v *= 2;
				continue;
			}

			linearSolver.Solve(gVector).AsSpan().CopyTo(lmStep);
			for (int i = 0; i < numTangent; ++i)
			{
				dx[i] = jacobiScaling[i] * lmStep[i];
			}

			// Adding parameter_tolerance to x.norm() ensures that this works if x is near
			// zero.
			double parameterTolerance = options.ParameterTolerance * (Norm(x) + options.ParameterTolerance);
			if (Norm(dx) < parameterTolerance)
			{
				status = TinySolverStatus.RelativeStepSizeTooSmall;
				break;
			}

			manifold.Plus(x, dx, xNew);

			// If the cost function fails to evaluate at the trial point, reject the step and
			// shrink the trust region rather than acting on garbage.
			if (!function.Evaluate(xNew, fxNew, []))
			{
				u *= v;
				v *= 2;
				continue;
			}

			double costChange = 2 * cost - SquaredNorm(fxNew);
			double modelCostChange = ModelCostChange(lmStep);

			// rho is the ratio of the actual reduction in error to the reduction in error
			// that would be obtained if the problem was linear. See [1] for details.
			double rho = costChange / modelCostChange;
			if (rho > 0)
			{
				// Accept the Levenberg-Marquardt step because the linear model fits well.
				xNew.CopyTo(x);

				if (Math.Abs(costChange) < options.FunctionTolerance)
				{
					cost = SquaredNorm(fxNew) / 2;
					status = TinySolverStatus.CostChangeTooSmall;
					break;
				}

				// The cost function already evaluated successfully at xNew == x above, so
				// re-evaluating (now also for the Jacobian) is not expected to fail; guard
				// against it regardless.
				if (!Update(function, x))
				{
					status = TinySolverStatus.CostFunctionFailed;
					break;
				}

				if (gradientMaxNorm < options.GradientTolerance)
				{
					status = TinySolverStatus.GradientTooSmall;
					break;
				}

				if (cost < options.CostThreshold)
				{
					status = TinySolverStatus.CostTooSmall;
					break;
				}

				double tmp = 2 * rho - 1;
				u *= Math.Max(1 / 3.0, 1 - tmp * tmp * tmp);
				v = 2;
			}
			else
			{
				// Reject the update because either the normal equations failed to solve or
				// the local linear model was not good (rho < 0).

				// Additionally if the cost change is too small, then terminate.
				if (Math.Abs(costChange) < options.FunctionTolerance)
				{
					status = TinySolverStatus.CostChangeTooSmall;
					break;
				}

				// Reduce the size of the trust region.
				u *= v;
				v *= 2;
			}
		}

		return new TinySolverSummary(initialCost, cost, gradientMaxNorm, iterations, status);
	}

	// Evaluates the residuals and the Jacobian with respect to the ambient parameters, then
	// projects it into the tangent space at x: J_tangent = J_ambient * dPlus(x, delta)/ddelta
	// at delta = 0. Then forms the Jacobi-scaled normal equations.
	private bool Update(in TFunction function, ReadOnlySpan<double> x)
	{
		int n = residuals.Length;
		if (manifold.IsEuclidean)
		{
			// The tangent space equals the ambient space, so the cost function writes its
			// Jacobian straight into `jacobian` (no projection, no copy).
			if (!function.Evaluate(x, residuals, jacobian))
			{
				return false;
			}
		}
		else
		{
			if (!function.Evaluate(x, residuals, jacobianAmbient))
			{
				return false;
			}

			manifold.PlusJacobian(x, plusJacobian);
			for (int t = 0; t < numTangent; ++t)
			{
				for (int r = 0; r < n; ++r)
				{
					double sum = 0;
					for (int p = 0; p < numParameters; ++p)
					{
						sum += jacobianAmbient[p * n + r] * plusJacobian[p * numTangent + t];
					}

					jacobian[t * n + r] = sum;
				}
			}
		}

		for (int r = 0; r < n; ++r)
		{
			residuals[r] = -residuals[r];
		}

		// On the first iteration, compute a diagonal (Jacobi) scaling matrix, which we store
		// as a vector: jacobi_scaling = 1 / (1 + ||column of J||). 1 is added to the
		// denominator to regularize small diagonal entries.
		if (iterations == 0)
		{
			for (int t = 0; t < numTangent; ++t)
			{
				jacobiScaling[t] = 1.0 / (1.0 + Norm(jacobian.AsSpan(t * n, n)));
			}
		}

		// This explicitly computes the normal equations, which is numerically unstable.
		// Nevertheless, it is often good enough and is fast.
		for (int t = 0; t < numTangent; ++t)
		{
			Span<double> column = jacobian.AsSpan(t * n, n);
			double scale = jacobiScaling[t];
			for (int r = 0; r < n; ++r)
			{
				column[r] *= scale;
			}
		}

		gradientMaxNorm = 0;
		for (int a = 0; a < numTangent; ++a)
		{
			ReadOnlySpan<double> columnA = jacobian.AsSpan(a * n, n);
			for (int b = 0; b <= a; ++b)
			{
				double dot = Dot(columnA, jacobian.AsSpan(b * n, n));
				jtj[a, b] = dot;
				jtj[b, a] = dot;
			}

			g[a] = Dot(columnA, residuals);
			gVector[a] = g[a];
			gradientMaxNorm = a == 0 ? Math.Abs(g[a]) : Math.Max(gradientMaxNorm, Math.Abs(g[a]));
		}

		cost = SquaredNorm(residuals) / 2;
		return true;
	}

	// lm_step . (2 g - jtj lm_step), the cost reduction the linear model predicts.
	private double ModelCostChange(ReadOnlySpan<double> lmStep)
	{
		double sum = 0;
		for (int i = 0; i < numTangent; ++i)
		{
			double jtjStep = 0;
			for (int j = 0; j < numTangent; ++j)
			{
				jtjStep += jtj[i, j] * lmStep[j];
			}

			sum += lmStep[i] * (2 * g[i] - jtjStep);
		}

		return sum;
	}

	// Only the number of residuals varies between problems; the parameter and tangent
	// dimensions are fixed, so nothing else needs (re)allocation.
	private void Initialize(in TFunction function)
	{
		int numResiduals = function.NumResiduals;
		Check.Ge(numResiduals, 0);
		if (residuals.Length != numResiduals)
		{
			residuals = new double[numResiduals];
			fxNew = new double[numResiduals];
			jacobianAmbient = manifold.IsEuclidean ? [] : new double[numResiduals * numParameters];
			jacobian = new double[numResiduals * numTangent];
		}
	}

	private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
	{
		double sum = 0;
		for (int i = 0; i < a.Length; ++i)
		{
			sum += a[i] * b[i];
		}

		return sum;
	}

	private static double SquaredNorm(ReadOnlySpan<double> a) => Dot(a, a);

	private static double Norm(ReadOnlySpan<double> a) => Math.Sqrt(SquaredNorm(a));
}

/// <summary>
/// TinySolver over the Euclidean manifold of the function's parameters: C++'s
/// <c>TinySolver&lt;Function&gt;</c> with its default manifold.
/// </summary>
public static class TinySolver
{
	/// <summary>A solver for <typeparamref name="TFunction"/> on R^NumParameters.</summary>
	public static TinySolver<TFunction, TinyEuclideanManifold> Create<TFunction>()
		where TFunction : ITinySolverFunction
		=> new(new TinyEuclideanManifold(TFunction.NumParameters));
}
