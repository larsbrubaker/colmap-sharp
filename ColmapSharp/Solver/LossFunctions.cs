// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/loss_function.h and
// internal/ceres/loss_function.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Robust loss functions rho(s) applied to a residual block's squared norm s = |f(x)|^2, so
// the solver minimizes 1/2 sum_i rho(|f_i|^2). COLMAP builds Trivial, SoftLOne, Cauchy and
// Huber for bundle adjustment (estimators/bundle_adjustment_ceres.cc CreateLossFunction),
// Cauchy for view-graph calibration, Arctan for gravity refinement and a Scaled Huber/Cauchy
// for global positioning; Tolerant, Tukey and Composed are Ceres' remaining losses and come
// along because they are a few lines each and share the test. The (future) solver's
// residual-block corrector consumes Evaluate's rho[0..2].
// Ceres' ownership flags (TAKE_OWNERSHIP / DO_NOT_TAKE_OWNERSHIP) have no C# meaning and
// are dropped; LossFunctionWrapper (a mutable indirection for changing a loss mid-solve)
// is not used by COLMAP and is not ported.

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::LossFunction: a robustifier rho(s) of the squared residual norm s.
/// </summary>
public abstract class LossFunction
{
	/// <summary>
	/// Writes rho(s), rho'(s) and rho''(s) to <paramref name="rho"/>[0..2]. s is
	/// non-negative; rho(0) = 0, rho'(0) = 1 and rho'(s) &gt; 0 for every loss here except
	/// Tukey's (whose outlier region has rho' = 0).
	/// </summary>
	public abstract void Evaluate(double s, Span<double> rho);
}

/// <summary>ceres::TrivialLoss: rho(s) = s, plain least squares.</summary>
public sealed class TrivialLoss : LossFunction
{
	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		rho[0] = s;
		rho[1] = 1.0;
		rho[2] = 0.0;
	}
}

/// <summary>
/// ceres::HuberLoss: rho(s) = s for s &lt;= a^2, 2 a sqrt(s) - a^2 beyond; quadratic near
/// zero and linear in the residual for outliers.
/// </summary>
public sealed class HuberLoss : LossFunction
{
	private readonly double a;
	private readonly double b;

	/// <summary>Creates the loss with scale a (the residual size where it turns linear).</summary>
	public HuberLoss(double a)
	{
		this.a = a;
		b = a * a;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		if (s > b)
		{
			// Outlier region. 'r' is always positive.
			double r = Math.Sqrt(s);
			rho[0] = 2.0 * a * r - b;
			rho[1] = Math.Max(LossConstants.MinNormal, a / r);
			rho[2] = -rho[1] / (2.0 * s);
		}
		else
		{
			// Inlier region.
			rho[0] = s;
			rho[1] = 1.0;
			rho[2] = 0.0;
		}
	}
}

/// <summary>ceres::SoftLOneLoss: rho(s) = 2 a^2 (sqrt(1 + s / a^2) - 1).</summary>
public sealed class SoftLOneLoss : LossFunction
{
	private readonly double b;
	private readonly double c;

	/// <summary>Creates the loss with scale a.</summary>
	public SoftLOneLoss(double a)
	{
		b = a * a;
		c = 1 / b;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		double sum = 1.0 + s * c;
		double tmp = Math.Sqrt(sum);
		// 'sum' and 'tmp' are always positive, assuming that 's' is.
		rho[0] = 2.0 * b * (tmp - 1.0);
		rho[1] = Math.Max(LossConstants.MinNormal, 1.0 / tmp);
		rho[2] = -(c * rho[1]) / (2.0 * sum);
	}
}

/// <summary>ceres::CauchyLoss: rho(s) = a^2 log(1 + s / a^2).</summary>
public sealed class CauchyLoss : LossFunction
{
	private readonly double b;
	private readonly double c;

	/// <summary>Creates the loss with scale a.</summary>
	public CauchyLoss(double a)
	{
		b = a * a;
		c = 1 / b;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		double sum = 1.0 + s * c;
		double inv = 1.0 / sum;
		// 'sum' and 'inv' are always positive, assuming that 's' is.
		rho[0] = b * Math.Log(sum);
		rho[1] = Math.Max(LossConstants.MinNormal, inv);
		rho[2] = -c * (inv * inv);
	}
}

/// <summary>
/// ceres::ArctanLoss: rho(s) = a atan2(s, a), which caps the cost of any residual at
/// a pi / 2.
/// </summary>
public sealed class ArctanLoss : LossFunction
{
	private readonly double a;
	private readonly double b;

	/// <summary>Creates the loss with scale a.</summary>
	public ArctanLoss(double a)
	{
		this.a = a;
		b = 1 / (a * a);
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		double sum = 1 + s * s * b;
		double inv = 1 / sum;
		// 'sum' and 'inv' are always positive.
		rho[0] = a * Math.Atan2(s, a);
		rho[1] = Math.Max(LossConstants.MinNormal, inv);
		rho[2] = -2.0 * s * b * (inv * inv);
	}
}

/// <summary>
/// ceres::TolerantLoss: rho(s) = b log(1 + e^((s - a) / b)) - b log(1 + e^(-a / b)), which
/// costs almost nothing below a and grows linearly above it.
/// </summary>
public sealed class TolerantLoss : LossFunction
{
	private readonly double a;
	private readonly double b;
	private readonly double c;

	/// <summary>Creates the loss with tolerance a (&gt;= 0) and transition width b (&gt; 0).</summary>
	public TolerantLoss(double a, double b)
	{
		Check.Ge(a, 0.0);
		Check.Gt(b, 0.0);
		this.a = a;
		this.b = b;
		c = b * Math.Log(1.0 + Math.Exp(-a / b));
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		double x = (s - a) / b;
		// The basic equation is rho[0] = b ln(1 + e^x). However, if e^x is too large, it will
		// overflow. Since numerically 1 + e^x == e^x when x is greater than about ln(2^53)
		// for doubles, beyond this threshold x substitutes for ln(1 + e^x) as a numerically
		// equivalent approximation.
		const double kLog2Pow53 = 36.7;
		if (x > kLog2Pow53)
		{
			rho[0] = s - a - c;
			rho[1] = 1.0;
			rho[2] = 0.0;
		}
		else
		{
			double eX = Math.Exp(x);
			rho[0] = b * Math.Log(1.0 + eX) - c;
			rho[1] = Math.Max(LossConstants.MinNormal, eX / (1.0 + eX));
			rho[2] = 0.5 / (b * (1.0 + Math.Cosh(x)));
		}
	}
}

/// <summary>
/// ceres::TukeyLoss: Tukey's biweight, rho(s) = a^2 / 3 (1 - (1 - s / a^2)^3) for
/// s &lt;= a^2 and the constant a^2 / 3 beyond, so outliers contribute no gradient.
/// </summary>
public sealed class TukeyLoss : LossFunction
{
	private readonly double aSquared;

	/// <summary>Creates the loss with scale a.</summary>
	public TukeyLoss(double a)
	{
		aSquared = a * a;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		if (s <= aSquared)
		{
			// Inlier region.
			double value = 1.0 - s / aSquared;
			double valueSq = value * value;
			rho[0] = aSquared / 3.0 * (1.0 - valueSq * value);
			rho[1] = valueSq;
			rho[2] = -2.0 / aSquared * value;
		}
		else
		{
			// Outlier region.
			rho[0] = aSquared / 3.0;
			rho[1] = 0.0;
			rho[2] = 0.0;
		}
	}
}

/// <summary>ceres::ComposedLoss: rho(s) = f(g(s)).</summary>
public sealed class ComposedLoss : LossFunction
{
	private readonly LossFunction f;
	private readonly LossFunction g;

	/// <summary>Creates f(g(s)).</summary>
	public ComposedLoss(LossFunction f, LossFunction g)
	{
		ArgumentNullException.ThrowIfNull(f);
		ArgumentNullException.ThrowIfNull(g);
		this.f = f;
		this.g = g;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> rho)
	{
		Span<double> rhoF = stackalloc double[3];
		Span<double> rhoG = stackalloc double[3];
		g.Evaluate(s, rhoG);
		f.Evaluate(rhoG[0], rhoF);
		rho[0] = rhoF[0];
		// f'(g(s)) * g'(s).
		rho[1] = rhoF[1] * rhoG[1];
		// f''(g(s)) * g'(s) * g'(s) + f'(g(s)) * g''(s).
		rho[2] = rhoF[2] * rhoG[1] * rhoG[1] + rhoF[1] * rhoG[2];
	}
}

/// <summary>
/// ceres::ScaledLoss: a * rho(s), or a * s when no inner loss is given. COLMAP weights
/// residual groups against each other with it (global positioning).
/// </summary>
public sealed class ScaledLoss : LossFunction
{
	private readonly LossFunction? rho;
	private readonly double a;

	/// <summary>Creates a * rho(s); a null <paramref name="rho"/> means the trivial loss.</summary>
	public ScaledLoss(LossFunction? rho, double a)
	{
		this.rho = rho;
		this.a = a;
	}

	/// <inheritdoc/>
	public override void Evaluate(double s, Span<double> result)
	{
		if (rho is null)
		{
			result[0] = a * s;
			result[1] = a;
			result[2] = 0.0;
		}
		else
		{
			rho.Evaluate(s, result);
			result[0] *= a;
			result[1] *= a;
			result[2] *= a;
		}
	}
}

/// <summary>Constants the loss functions share.</summary>
internal static class LossConstants
{
	/// <summary>
	/// std::numeric_limits&lt;double&gt;::min(), the smallest positive normal double (not C#'s
	/// double.MinValue, the most negative, nor double.Epsilon, the smallest subnormal).
	/// Ceres floors rho' at it so the corrector never divides by zero.
	/// </summary>
	public const double MinNormal = 2.2250738585072014E-308;
}
