// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureTypesTests: colmap/feature/types_test.cc, so far the FeatureMatches.Nominal case,
// testing ColmapSharp/Feature/FeatureMatch.cs. The keypoint, descriptor and matrix-conversion
// cases (FeatureKeypoints.Nominal, FeatureKeypoint.Rot90, FeatureDescriptors.*,
// KeypointsMatrixConversion.Roundtrip, MatchesMatrixConversion.Roundtrip) arrive with the
// rest of feature/types.h in the feature phase.
//
// `FeatureMatches matches(1)` value-initializes one match; the C# equivalent is an array of
// one default element, which FeatureMatch decodes as two invalid indices like COLMAP.

using ColmapSharp.Feature;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Feature;

public class FeatureTypesTests
{
	[Test]
	public async Task FeatureMatches_Nominal()
	{
		var match = new FeatureMatch();
		FeatureMatch sameMatch = match;
		var matches = new List<FeatureMatch>(new FeatureMatch[1]);
		using (Assert.Multiple())
		{
			await Assert.That(match.Point2DIdx1).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(match.Point2DIdx2).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(matches.Count).IsEqualTo(1);
			await Assert.That(matches[0].Point2DIdx1).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(matches[0].Point2DIdx2).IsEqualTo(InvalidPoint2DIdx);

			await Assert.That(match == sameMatch).IsTrue();
			await Assert.That(match != new FeatureMatch(0, 1)).IsTrue();
		}
	}
}
