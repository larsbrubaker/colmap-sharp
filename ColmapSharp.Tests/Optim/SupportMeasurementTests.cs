// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SupportMeasurementTests: colmap/optim/support_measurement_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks. Tests
// ColmapSharp/Optim/SupportMeasurement.cs. Exact comparisons, as in COLMAP (Tier A).

using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class SupportMeasurementTests
{
	[Test]
	public async Task InlierSupportMeasurer_Nominal()
	{
		var support1 = new InlierSupportMeasurer.Support();
		await Assert.That(support1.NumInliers).IsEqualTo(0);
		await Assert.That(support1.ResidualSum).IsEqualTo(double.MaxValue);
		var measurer = new InlierSupportMeasurer();
		double[] residuals = { -1.0, 0.0, 1.0, 2.0 };
		support1 = measurer.Evaluate(residuals, 1.0);
		await Assert.That(support1.NumInliers).IsEqualTo(3);
		await Assert.That(support1.ResidualSum).IsEqualTo(0.0);
		var support2 = new InlierSupportMeasurer.Support();
		support2.NumInliers = 2;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum = support1.ResidualSum;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.NumInliers = support1.NumInliers;
		support2.ResidualSum += 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum -= 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum -= 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsTrue();
	}

	[Test]
	public async Task UniqueInlierSupportMeasurer_Nominal()
	{
		var support1 = new UniqueInlierSupportMeasurer.Support();
		await Assert.That(support1.NumInliers).IsEqualTo(0);
		await Assert.That(support1.NumUniqueInliers).IsEqualTo(0);
		await Assert.That(support1.ResidualSum).IsEqualTo(double.MaxValue);

		var measurer = new UniqueInlierSupportMeasurer(new ulong[] { 1, 2, 2, 3 });
		double[] residuals = { -1.0, 0.0, 1.0, 2.0 };
		support1 = measurer.Evaluate(residuals, 1.0);
		await Assert.That(support1.NumInliers).IsEqualTo(3);
		await Assert.That(support1.NumUniqueInliers).IsEqualTo(2);
		await Assert.That(support1.ResidualSum).IsEqualTo(0.0);

		var support2 = new UniqueInlierSupportMeasurer.Support();
		support2.NumUniqueInliers = support1.NumUniqueInliers - 1;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.NumInliers = support1.NumInliers + 1;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.NumInliers = support1.NumInliers;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum = support1.ResidualSum - 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum = support1.ResidualSum;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.NumUniqueInliers = support1.NumUniqueInliers;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.ResidualSum = support1.ResidualSum - 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsTrue();
		support2.NumInliers = support1.NumInliers + 1;
		support2.ResidualSum = support1.ResidualSum + 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsTrue();
		support2.NumUniqueInliers = support1.NumUniqueInliers + 1;
		support2.NumInliers = support1.NumInliers - 1;
		support2.ResidualSum = support1.ResidualSum + 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsTrue();
	}

	[Test]
	public async Task MEstimatorSupportMeasurer_Nominal()
	{
		var support1 = new MEstimatorSupportMeasurer.Support();
		await Assert.That(support1.NumInliers).IsEqualTo(0);
		await Assert.That(support1.Score).IsEqualTo(double.MaxValue);
		var measurer = new MEstimatorSupportMeasurer();
		double[] residuals = { -1.0, 0.0, 1.0, 2.0 };
		support1 = measurer.Evaluate(residuals, 1.0);
		await Assert.That(support1.NumInliers).IsEqualTo(3);
		await Assert.That(support1.Score).IsEqualTo(1.0);
		MEstimatorSupportMeasurer.Support support2 = support1;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.NumInliers -= 1;
		support2.Score += 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsTrue();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.Score -= 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsFalse();
		support2.Score -= 0.01;
		await Assert.That(measurer.IsLeftBetter(support1, support2)).IsFalse();
		await Assert.That(measurer.IsLeftBetter(support2, support1)).IsTrue();
	}
}
