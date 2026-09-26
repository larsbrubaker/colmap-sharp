// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CostFunctionUtils: colmap/estimators/cost_functions/utils.h - CreateAutoDiffCostFunction,
// the AutoDiffCostFunctor base (here the ISizedAutoDiffFunctor interface), and the generic
// functors NormalPriorCostFunctor, NormalErrorCostFunctor and CovarianceWeightedCostFunctor.
// Every autodiff functor in this folder that COLMAP derives from AutoDiffCostFunctor
// implements ISizedAutoDiffFunctor and is wrapped by Solver/AutoDiffCostFunction.cs through
// CreateAutoDiffCostFunction. Tests: ColmapSharp.Tests/Estimators/CostFunctions/UtilsTests.cs.
//
// Translation notes:
// - C++ carries the residual count and block sizes as template arguments
//   (AutoDiffCostFunctor<Derived, kNumResiduals, kParameterDims...>). C# has no integer
//   generics, so a functor reports them through ISizedAutoDiffFunctor, and
//   CreateAutoDiffCostFunction picks the Jet width (the total parameter count, as Ceres'
//   fixed-size AutoDiffCostFunction does) with one switch at construction. Evaluation is
//   then fully specialized on the functor and width, with no per-call dispatch.
// - NormalPriorCostFunctor<N> / NormalErrorCostFunctor<N> take N at run time.
// - CovarianceWeightedCostFunctor applies left_sqrt_info = (cov^-1).llt().matrixL()^T,
//   which is upper triangular, in place: row i only reads residuals j >= i, which are
//   still unmodified when rows are processed top down. Eigen's dense product adds the
//   exact zeros below the diagonal as well, which changes nothing but the sign of a zero
//   result (or a NaN from 0 * inf), so there is no temporary vector of T.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// An autodiff functor that knows its residual count and parameter block sizes: the C# form
/// of colmap::AutoDiffCostFunctor's <c>kNumResiduals</c> and <c>kParameterDims</c>.
/// </summary>
public interface ISizedAutoDiffFunctor : IAutoDiffFunctor
{
	/// <summary>C++ <c>kNumResiduals</c>.</summary>
	int NumResiduals { get; }

	/// <summary>C++ <c>kParameterDims</c>.</summary>
	int[] ParameterBlockSizes { get; }
}

/// <summary>Port of the free functions of colmap/estimators/cost_functions/utils.h.</summary>
public static class CostFunctionUtils
{
	/// <summary>
	/// colmap::CreateAutoDiffCostFunction: wraps the functor in an AutoDiffCostFunction
	/// whose Jet width is the functor's total parameter count, so one pass computes every
	/// Jacobian (Ceres' fixed-size AutoDiffCostFunction).
	/// </summary>
	public static CostFunction CreateAutoDiffCostFunction<TFunctor>(TFunctor functor)
		where TFunctor : ISizedAutoDiffFunctor
	{
		int[] sizes = functor.ParameterBlockSizes;
		int numResiduals = functor.NumResiduals;
		int total = 0;
		foreach (int size in sizes)
		{
			total += size;
		}

		return total switch
		{
			1 => new AutoDiffCostFunction<TFunctor, Grad1>(functor, numResiduals, sizes),
			2 => new AutoDiffCostFunction<TFunctor, Grad2>(functor, numResiduals, sizes),
			3 => new AutoDiffCostFunction<TFunctor, Grad3>(functor, numResiduals, sizes),
			4 => new AutoDiffCostFunction<TFunctor, Grad4>(functor, numResiduals, sizes),
			5 => new AutoDiffCostFunction<TFunctor, Grad5>(functor, numResiduals, sizes),
			6 => new AutoDiffCostFunction<TFunctor, Grad6>(functor, numResiduals, sizes),
			7 => new AutoDiffCostFunction<TFunctor, Grad7>(functor, numResiduals, sizes),
			8 => new AutoDiffCostFunction<TFunctor, Grad8>(functor, numResiduals, sizes),
			9 => new AutoDiffCostFunction<TFunctor, Grad9>(functor, numResiduals, sizes),
			10 => new AutoDiffCostFunction<TFunctor, Grad10>(functor, numResiduals, sizes),
			11 => new AutoDiffCostFunction<TFunctor, Grad11>(functor, numResiduals, sizes),
			12 => new AutoDiffCostFunction<TFunctor, Grad12>(functor, numResiduals, sizes),
			13 => new AutoDiffCostFunction<TFunctor, Grad13>(functor, numResiduals, sizes),
			14 => new AutoDiffCostFunction<TFunctor, Grad14>(functor, numResiduals, sizes),
			15 => new AutoDiffCostFunction<TFunctor, Grad15>(functor, numResiduals, sizes),
			16 => new AutoDiffCostFunction<TFunctor, Grad16>(functor, numResiduals, sizes),
			17 => new AutoDiffCostFunction<TFunctor, Grad17>(functor, numResiduals, sizes),
			18 => new AutoDiffCostFunction<TFunctor, Grad18>(functor, numResiduals, sizes),
			19 => new AutoDiffCostFunction<TFunctor, Grad19>(functor, numResiduals, sizes),
			20 => new AutoDiffCostFunction<TFunctor, Grad20>(functor, numResiduals, sizes),
			21 => new AutoDiffCostFunction<TFunctor, Grad21>(functor, numResiduals, sizes),
			22 => new AutoDiffCostFunction<TFunctor, Grad22>(functor, numResiduals, sizes),
			23 => new AutoDiffCostFunction<TFunctor, Grad23>(functor, numResiduals, sizes),
			24 => new AutoDiffCostFunction<TFunctor, Grad24>(functor, numResiduals, sizes),
			25 => new AutoDiffCostFunction<TFunctor, Grad25>(functor, numResiduals, sizes),
			26 => new AutoDiffCostFunction<TFunctor, Grad26>(functor, numResiduals, sizes),
			27 => new AutoDiffCostFunction<TFunctor, Grad27>(functor, numResiduals, sizes),
			28 => new AutoDiffCostFunction<TFunctor, Grad28>(functor, numResiduals, sizes),
			29 => new AutoDiffCostFunction<TFunctor, Grad29>(functor, numResiduals, sizes),
			30 => new AutoDiffCostFunction<TFunctor, Grad30>(functor, numResiduals, sizes),
			31 => new AutoDiffCostFunction<TFunctor, Grad31>(functor, numResiduals, sizes),
			32 => new AutoDiffCostFunction<TFunctor, Grad32>(functor, numResiduals, sizes),
			33 => new AutoDiffCostFunction<TFunctor, Grad33>(functor, numResiduals, sizes),
			_ => throw new ArgumentOutOfRangeException(
				nameof(functor), total, "No Jet width is defined for this many parameters (Solver/JetGradients.cs)."),
		};
	}

}

/// <summary>
/// Cost functor for a single parameter block against a fixed prior:
/// residual = param - prior. Port of colmap::NormalPriorCostFunctor&lt;N&gt;.
/// </summary>
public readonly struct NormalPriorCostFunctor : ISizedAutoDiffFunctor
{
	private readonly double[] _prior;
	private readonly int[] _sizes;

	/// <summary>A functor for a block of <c>prior.Length</c> parameters.</summary>
	public NormalPriorCostFunctor(ReadOnlySpan<double> prior)
	{
		_prior = prior.ToArray();
		_sizes = [prior.Length];
	}

	/// <inheritdoc/>
	public int NumResiduals => _prior.Length;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => _sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(ReadOnlySpan<double> prior) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new NormalPriorCostFunctor(prior));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		for (int i = 0; i < _prior.Length; i++)
		{
			residuals[i] = parameters[i] - _prior[i];
		}

		return true;
	}
}

/// <summary>
/// Cost functor for the difference of two parameter blocks of equal size:
/// residual = param0 - param1. Port of colmap::NormalErrorCostFunctor&lt;N&gt;.
/// </summary>
public readonly struct NormalErrorCostFunctor : ISizedAutoDiffFunctor
{
	private readonly int _size;
	private readonly int[] _sizes;

	/// <summary>A functor for two blocks of <paramref name="size"/> parameters each.</summary>
	public NormalErrorCostFunctor(int size)
	{
		Check.Gt(size, 0);
		_size = size;
		_sizes = [size, size];
	}

	/// <inheritdoc/>
	public int NumResiduals => _size;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => _sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(int size) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new NormalErrorCostFunctor(size));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		for (int i = 0; i < _size; i++)
		{
			residuals[i] = parameters[i] - parameters[_size + i];
		}

		return true;
	}
}

/// <summary>
/// Whitens the residuals of another functor with a covariance: residuals are multiplied on
/// the left by the upper-triangular square root of the information matrix,
/// (cov^-1).llt().matrixL()^T. For example, to weight a reprojection error with an image
/// measurement covariance:
/// <c>CovarianceWeightedCostFunctor.Create(point2DCov, new ReprojErrorCostFunctor&lt;M&gt;(point2D))</c>.
/// Port of colmap::CovarianceWeightedCostFunctor.
/// </summary>
public readonly struct CovarianceWeightedCostFunctor<TFunctor> : ISizedAutoDiffFunctor
	where TFunctor : ISizedAutoDiffFunctor
{
	// Row-major NumResiduals x NumResiduals, upper triangular.
	private readonly double[] _leftSqrtInfo;
	private readonly TFunctor _cost;

	/// <summary>Wraps <paramref name="cost"/>, whose residual count must match the covariance size.</summary>
	public CovarianceWeightedCostFunctor(MatrixXd cov, TFunctor cost)
	{
		int n = cost.NumResiduals;
		Check.Eq(cov.Rows, n);
		Check.Eq(cov.Cols, n);
		_cost = cost;
		_leftSqrtInfo = LeftSqrtInformation(cov);
	}

	/// <inheritdoc/>
	public int NumResiduals => _cost.NumResiduals;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => _cost.ParameterBlockSizes;

	/// <summary>The autodiff cost function of <paramref name="cost"/> weighted by <paramref name="cov"/>.</summary>
	public static CostFunction Create(MatrixXd cov, TFunctor cost) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new CovarianceWeightedCostFunctor<TFunctor>(cov, cost));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		if (!_cost.Evaluate(parameters, residuals))
		{
			return false;
		}

		int n = _cost.NumResiduals;
		for (int i = 0; i < n; i++)
		{
			T sum = _leftSqrtInfo[i * n + i] * residuals[i];
			for (int j = i + 1; j < n; j++)
			{
				sum += _leftSqrtInfo[i * n + j] * residuals[j];
			}

			residuals[i] = sum;
		}

		return true;
	}

	// cov.inverse().llt().matrixL().transpose(), row-major. Eigen inverts fixed sizes up to
	// 4x4 by cofactors and larger ones by partial-pivot LU; the Matrix2d/3d/4d inverses are
	// the cofactor versions and MatrixXd.Inverse the LU one.
	private static double[] LeftSqrtInformation(MatrixXd cov)
	{
		int n = cov.Rows;
		MatrixXd information = n switch
		{
			2 => FromMatrix2d(new Matrix2d(cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]).Inverse()),
			3 => FromMatrix3d(ToMatrix3d(cov).Inverse()),
			4 => FromMatrix4d(ToMatrix4d(cov).Inverse()),
			_ => cov.Inverse(),
		};
		MatrixXd l = new LLT(information).MatrixL();
		var result = new double[n * n];
		for (int i = 0; i < n; i++)
		{
			for (int j = i; j < n; j++)
			{
				result[i * n + j] = l[j, i];
			}
		}

		return result;
	}

	private static Matrix3d ToMatrix3d(MatrixXd m) => new(
		m[0, 0], m[0, 1], m[0, 2],
		m[1, 0], m[1, 1], m[1, 2],
		m[2, 0], m[2, 1], m[2, 2]);

	private static Matrix4d ToMatrix4d(MatrixXd m)
	{
		Span<double> columnMajor = stackalloc double[16];
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				columnMajor[j * 4 + i] = m[i, j];
			}
		}

		return Matrix4d.FromColumnMajor(columnMajor);
	}

	private static MatrixXd FromMatrix2d(Matrix2d m) => MatrixXd.FromRowMajor(2, 2, [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]);

	private static MatrixXd FromMatrix3d(Matrix3d m)
	{
		Span<double> columnMajor = stackalloc double[9];
		m.CopyToColumnMajor(columnMajor);
		return MatrixXd.FromColumnMajor(3, 3, columnMajor);
	}

	private static MatrixXd FromMatrix4d(Matrix4d m)
	{
		var result = new MatrixXd(4, 4);
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				result[i, j] = m[i, j];
			}
		}

		return result;
	}
}

/// <summary>Non-generic entry point for <see cref="CovarianceWeightedCostFunctor{TFunctor}"/>.</summary>
public static class CovarianceWeightedCostFunctor
{
	/// <summary>
	/// colmap::CovarianceWeightedCostFunctor&lt;CostFunctor&gt;::Create: the autodiff cost
	/// function of <paramref name="cost"/> whitened by <paramref name="cov"/>.
	/// </summary>
	public static CostFunction Create<TFunctor>(MatrixXd cov, TFunctor cost)
		where TFunctor : ISizedAutoDiffFunctor =>
		CovarianceWeightedCostFunctor<TFunctor>.Create(cov, cost);
}
