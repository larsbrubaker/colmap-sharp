// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TranslationTransformTests: colmap/estimators/solvers/translation_transform_test.cc ported
// 1:1 (TEST(TranslationTransform, Estimate) as TranslationTransform_Estimate), same checks
// and tolerances. Tests ColmapSharp/Estimators/Solvers/TranslationTransform.cs.
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and the test draws everything before its first await.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class TranslationTransformTests
{
	[Test]
	public async Task TranslationTransform_Estimate()
	{
		const int kNumPoints = 100;

		var src = new Vector2d[kNumPoints];
		for (int i = 0; i < kNumPoints; ++i)
		{
			double x = RandomUtils.RandomUniformReal(-1000.0, 1000.0);
			double y = RandomUtils.RandomUniformReal(-1000.0, 1000.0);
			src[i] = new Vector2d(x, y);
		}

		double tx = RandomUtils.RandomUniformReal(-1000.0, 1000.0);
		double ty = RandomUtils.RandomUniformReal(-1000.0, 1000.0);
		var translation = new Vector2d(tx, ty);

		var dst = new Vector2d[kNumPoints];
		for (int i = 0; i < kNumPoints; ++i)
		{
			dst[i] = src[i] + translation;
		}

		var estimator = new TranslationTransformEstimator2d();
		var models = new List<Vector2d>();
		estimator.Estimate(src, dst, models);

		await Assert.That(models.Count).IsEqualTo(1);
		Vector2d estimatedTranslation = models[0];

		var residuals = new double[kNumPoints];
		estimator.Residuals(src, dst, estimatedTranslation, residuals);

		using (Assert.Multiple())
		{
			await Assert.That(estimatedTranslation.X).IsEqualTo(translation.X).Within(1e-6);
			await Assert.That(estimatedTranslation.Y).IsEqualTo(translation.Y).Within(1e-6);
			for (int i = 0; i < residuals.Length; ++i)
			{
				await Assert.That(residuals[i] < 1e-6).IsTrue();
			}
		}
	}
}
