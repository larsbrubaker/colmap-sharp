// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SprtTests: colmap/optim/sprt_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, same expected values. Tests ColmapSharp/Optim/Sprt.cs. The checks are
// integer outcomes (counts and accept/reject), so they pin Evaluate exactly.

using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class SprtTests
{
	[Test]
	public async Task SPRT_EvaluateAllInliers()
	{
		var options = new Sprt.Options();
		options.Delta = 0.05;
		options.Epsilon = 0.5;
		var sprt = new Sprt(options);

		// All residuals are small (inliers)
		double[] residuals = Enumerable.Repeat(0.1, 100).ToArray();
		bool accepted = sprt.Evaluate(residuals, 1.0, out int numInliers, out int numEvalSamples);

		await Assert.That(accepted).IsTrue();
		await Assert.That(numInliers).IsEqualTo(100);
		await Assert.That(numEvalSamples).IsEqualTo(100);
	}

	[Test]
	public async Task SPRT_EvaluateAllOutliers()
	{
		var options = new Sprt.Options();
		options.Delta = 0.05;
		options.Epsilon = 0.5;
		var sprt = new Sprt(options);

		// All residuals are large (outliers) - should trigger early rejection
		double[] residuals = Enumerable.Repeat(10.0, 100).ToArray();
		bool accepted = sprt.Evaluate(residuals, 1.0, out int numInliers, out int numEvalSamples);

		await Assert.That(accepted).IsFalse();
		await Assert.That(numInliers).IsEqualTo(0);
		await Assert.That(numEvalSamples).IsLessThan(100);
	}

	[Test]
	public async Task SPRT_EvaluateMixedEarlyReject()
	{
		var options = new Sprt.Options();
		options.Delta = 0.05;
		options.Epsilon = 0.9;
		var sprt = new Sprt(options);

		// Mostly outliers - should reject early
		double[] residuals = Enumerable.Repeat(10.0, 1000).ToArray();
		// Sprinkle a few inliers
		residuals[0] = 0.1;
		residuals[10] = 0.1;

		bool accepted = sprt.Evaluate(residuals, 1.0, out int numInliers, out int numEvalSamples);

		await Assert.That(accepted).IsFalse();
		// With epsilon=0.9 and delta=0.05, the likelihood ratio exceeds the decision
		// threshold after processing the inlier at index 0 and 4 subsequent outliers.
		await Assert.That(numInliers).IsEqualTo(1);
		await Assert.That(numEvalSamples).IsEqualTo(5);
	}

	[Test]
	public async Task SPRT_EvaluateEmpty()
	{
		var options = new Sprt.Options();
		var sprt = new Sprt(options);

		double[] residuals = Array.Empty<double>();
		bool accepted = sprt.Evaluate(residuals, 1.0, out int numInliers, out int numEvalSamples);

		await Assert.That(accepted).IsTrue();
		await Assert.That(numInliers).IsEqualTo(0);
		await Assert.That(numEvalSamples).IsEqualTo(0);
	}
}
