// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// VisibilityPyramidTests: colmap/scene/visibility_pyramid_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Scene/VisibilityPyramid.cs.
// The Eigen::VectorXi of per-level scores is an int array; scores.sum() and
// scores.tail(n - 1).sum() are the sums over all levels and over all but the first.

using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class VisibilityPyramidTests
{
	[Test]
	public async Task VisibilityPyramid_Default()
	{
		var pyramid = new VisibilityPyramid();
		using (Assert.Multiple())
		{
			await Assert.That(pyramid.NumLevels).IsEqualTo(0);
			await Assert.That(pyramid.Width).IsEqualTo(0);
			await Assert.That(pyramid.Height).IsEqualTo(0);
			await Assert.That(pyramid.Score).IsEqualTo(0UL);
		}
	}

	[Test]
	public async Task VisibilityPyramid_Score()
	{
		using (Assert.Multiple())
		{
			for (int numLevels = 1; numLevels < 8; ++numLevels)
			{
				int[] scores = new int[numLevels];
				ulong maxScore = 0;
				for (int i = 1; i <= numLevels; ++i)
				{
					scores[i - 1] = (1 << i) * (1 << i);
					maxScore += (ulong)(scores[i - 1] * scores[i - 1]);
				}

				ulong sum = (ulong)scores.Sum();
				ulong tailSum = (ulong)scores.Skip(1).Sum();

				var pyramid = new VisibilityPyramid(numLevels, 4, 4);
				await Assert.That(pyramid.NumLevels).IsEqualTo(numLevels);
				await Assert.That(pyramid.Width).IsEqualTo(4);
				await Assert.That(pyramid.Height).IsEqualTo(4);
				await Assert.That(pyramid.Score).IsEqualTo(0UL);
				await Assert.That(pyramid.MaxScore).IsEqualTo(maxScore);

				await Assert.That(pyramid.Score).IsEqualTo(0UL);
				pyramid.SetPoint(0, 0);
				await Assert.That(pyramid.Score).IsEqualTo(sum);
				pyramid.SetPoint(0, 0);
				await Assert.That(pyramid.Score).IsEqualTo(sum);
				pyramid.SetPoint(0, 1);
				await Assert.That(pyramid.Score).IsEqualTo(sum + tailSum);
				pyramid.SetPoint(0, 1);
				pyramid.SetPoint(0, 1);
				pyramid.SetPoint(1, 0);
				await Assert.That(pyramid.Score).IsEqualTo(sum + 2 * tailSum);
				pyramid.SetPoint(1, 0);
				pyramid.SetPoint(1, 1);
				await Assert.That(pyramid.Score).IsEqualTo(sum + 3 * tailSum);
				pyramid.ResetPoint(0, 0);
				await Assert.That(pyramid.Score).IsEqualTo(sum + 3 * tailSum);
				pyramid.ResetPoint(0, 0);
				await Assert.That(pyramid.Score).IsEqualTo(sum + 2 * tailSum);
				pyramid.SetPoint(0, 2);
				await Assert.That(pyramid.Score).IsEqualTo(2 * sum + 2 * tailSum);
			}
		}
	}
}
