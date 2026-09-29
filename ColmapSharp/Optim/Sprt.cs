// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sprt: colmap/optim/sprt.h and sprt.cc - the Sequential Probability Ratio Test of
// Matas et al. 2005, which lets randomized RANSAC reject a bad model after checking only a
// few residuals. Standalone: it sits next to the support measurers (SupportMeasurement.cs)
// as another way to evaluate residuals. Tests: ColmapSharp.Tests/Optim/SprtTests.cs
// (sprt_test.cc 1:1).
//
// Tier: Evaluate is exact given the decision threshold (it multiplies the same doubles in
// the same order). The threshold itself goes through log, which is the platform libm's in
// C++ and Math.Log here, so it can differ from COLMAP's in the last ulp (Tier B,
// divergence 114); only a likelihood ratio landing within that ulp of the
// threshold could change a decision.
//
// Translation notes:
// - Options is a struct because COLMAP copies it by value into the SPRT (`options_ =
//   options`); its defaults come from the parameterless constructor.
// - The size_t* out-parameters become `out int`.

namespace ColmapSharp.Optim;

/// <summary>
/// Sequential Probability Ratio Test as proposed in
///
///   "Randomized RANSAC with Sequential Probability Ratio Test",
///   Matas et al., 2005
///
/// Port of colmap::SPRT.
/// </summary>
public sealed class Sprt
{
	/// <summary>Port of colmap::SPRT::Options.</summary>
	public struct Options
	{
		/// <summary>Probability of rejecting a good model.</summary>
		public double Delta = 0.01;

		/// <summary>A priori assumed minimum inlier ratio.</summary>
		public double Epsilon = 0.1;

		/// <summary>
		/// The ratio of the time it takes to estimate a model from a random sample over the
		/// time it takes to decide whether one data sample is an inlier or not. Matas et al.
		/// propose 200 for the 7-point algorithm.
		/// </summary>
		public double EvalTimeRatio = 200;

		/// <summary>
		/// Number of models per random sample, that have to be verified. E.g. 1-3 for the
		/// 7-point fundamental matrix algorithm, or 1-10 for the 5-point essential matrix
		/// algorithm.
		/// </summary>
		public int NumModelsPerSample = 1;

		/// <summary>Options with COLMAP's default values.</summary>
		public Options()
		{
		}
	}

	private Options options;
	private double deltaEpsilon;
	private double delta1Epsilon1;
	private double decisionThreshold;

	/// <summary>Create a test with the given options.</summary>
	public Sprt(Options options)
	{
		Update(options);
	}

	/// <summary>Replace the options and recompute the decision threshold.</summary>
	public void Update(Options options)
	{
		this.options = options;
		deltaEpsilon = options.Delta / options.Epsilon;
		delta1Epsilon1 = (1 - options.Delta) / (1 - options.Epsilon);
		UpdateDecisionThreshold();
	}

	/// <summary>
	/// Run the test over <paramref name="residuals"/> in order, counting an absolute residual
	/// of at most <paramref name="maxResidual"/> as an inlier.
	/// </summary>
	/// <param name="residuals">The model's residuals.</param>
	/// <param name="maxResidual">Inlier threshold on the absolute residual.</param>
	/// <param name="numInliers">Inliers among the residuals evaluated.</param>
	/// <param name="numEvalSamples">How many residuals were evaluated before deciding.</param>
	/// <returns>True if the model is accepted, false if rejected early.</returns>
	public bool Evaluate(
		ReadOnlySpan<double> residuals,
		double maxResidual,
		out int numInliers,
		out int numEvalSamples)
	{
		numInliers = 0;

		double likelihoodRatio = 1;

		for (int i = 0; i < residuals.Length; ++i)
		{
			if (Math.Abs(residuals[i]) <= maxResidual)
			{
				numInliers += 1;
				likelihoodRatio *= deltaEpsilon;
			}
			else
			{
				likelihoodRatio *= delta1Epsilon1;
			}

			if (likelihoodRatio > decisionThreshold)
			{
				numEvalSamples = i + 1;
				return false;
			}
		}

		numEvalSamples = residuals.Length;

		return true;
	}

	private void UpdateDecisionThreshold()
	{
		// Equation 2
		double c = (1 - options.Delta) * Math.Log((1 - options.Delta) / (1 - options.Epsilon))
			+ options.Delta * Math.Log(options.Delta / options.Epsilon);

		// Equation 6
		double a0 = options.EvalTimeRatio * c / options.NumModelsPerSample + 1;

		double a = a0;

		const double Eps = 1.5e-8;

		// Compute A using the recursive relation
		//    A* = lim(n->inf) A
		// The series typically converges within 4 iterations

		for (int i = 0; i < 100; ++i)
		{
			double a1 = a0 + Math.Log(a);

			if (Math.Abs(a1 - a) < Eps)
			{
				break;
			}

			a = a1;
		}

		decisionThreshold = a;
	}
}
