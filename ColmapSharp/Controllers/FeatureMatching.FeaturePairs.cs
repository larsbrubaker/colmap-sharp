// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatching.FeaturePairs: the FeaturePairsFeatureMatcher of
// colmap/controllers/feature_matching.cc (CreateFeaturePairsFeatureMatcher), which imports
// feature matches from a text file and optionally verifies them. The rest of the controller
// is in FeatureMatching.cs. Tests: FeatureMatchingTests.CreateFeaturePairsFeatureMatcher_Nominal.
//
// The file is read with std::getline / StringTrim / `>>` semantics through
// Util/CppLineTokens.cs. COLMAP's try/catch around `>>` never fires (the streams do not
// throw). A token that does not parse as an index stores 0; a missing one (end of line)
// stores nothing, so the field keeps kInvalidPoint2DIdx, as libc++ does.
// COLMAP runs the import on its own thread, whose PRNG an unseeded RANSAC draws from and
// continues across pairs. Here the import starts from the default seed, as that fresh thread
// would, and the calling thread's PRNG is restored afterwards.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

public static partial class FeatureMatching
{
	/// <summary>The <see cref="ControllerProgress.Stage"/> of <see cref="MatchFeaturePairs"/>.</summary>
	public const string ImportMatchesStage = "Importing matches";

	/// <summary>
	/// Port of CreateFeaturePairsFeatureMatcher: imports feature matches from a text file of
	/// blocks "image_name1 image_name2", then one "idx1 idx2" line per match, then an empty
	/// line.
	/// </summary>
	public static void MatchFeaturePairs(
		Database database,
		FeaturePairsMatchingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		// FeaturePairsFeatureMatcher's own thread: one PRNG stream for the whole import.
		FeatureMatcherController.WithFreshPrng(geometryOptions, () =>
		{
			ImportFeaturePairs(database, pairingOptions, matchingOptions, geometryOptions, progress, cancellationToken);
			return 0;
		});
	}

	// FeaturePairsFeatureMatcher::Run.
	private static void ImportFeaturePairs(
		Database database,
		FeaturePairsMatchingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress,
		CancellationToken cancellationToken)
	{
		var cache = new FeatureMatcherCache(100, database);
		Check.That(pairingOptions.Check());
		Check.That(matchingOptions.Check());
		Check.That(geometryOptions.Check());

		var imageNameToImage = new Dictionary<string, Image>(StringComparer.Ordinal);
		foreach (uint imageId in cache.GetImageIds())
		{
			Image image = cache.GetImage(imageId);
			imageNameToImage.TryAdd(image.Name, image);
		}

		List<(string Text, bool ValidUtf8)> lines;
		using (FileStream stream = FileOpen.OpenRead(pairingOptions.MatchListPath))
		{
			lines = CppLineTokens.ReadLines(stream);
		}

		int lineIdx = 0;
		int numPairs = 0;
		while (lineIdx < lines.Count)
		{
			cancellationToken.ThrowIfCancellationRequested();

			string line = CppLineTokens.Trim(lines[lineIdx++].Text);
			if (line.Length == 0)
			{
				continue;
			}

			var lineTokens = new CppLineTokens(line);
			if (!lineTokens.TryReadString(out string imageName1))
			{
				imageName1 = "";
			}

			if (!lineTokens.TryReadString(out string imageName2))
			{
				imageName2 = "";
			}

			if (!imageNameToImage.TryGetValue(imageName1, out Image? image1))
			{
				Log.Warning($"SKIP: Image {imageName1} not found in database.");
				break;
			}

			if (!imageNameToImage.TryGetValue(imageName2, out Image? image2))
			{
				Log.Warning($"SKIP: Image {imageName2} not found in database.");
				break;
			}

			bool skipPair = database.ExistsTwoViewGeometry(image1.ImageId, image2.ImageId);

			var matches = new List<FeatureMatch>();
			while (lineIdx < lines.Count)
			{
				line = CppLineTokens.Trim(lines[lineIdx++].Text);
				if (line.Length == 0)
				{
					break;
				}

				matches.Add(ReadMatch(new CppLineTokens(line)));
			}

			numPairs += 1;
			progress?.Report(new ControllerProgress(ImportMatchesStage, numPairs, 0, $"{imageName1} - {imageName2}"));
			if (skipPair)
			{
				continue;
			}

			Camera camera1 = cache.GetCamera(image1.CameraId);
			Camera camera2 = cache.GetCamera(image2.CameraId);

			TwoViewGeometry twoViewGeometry;
			if (pairingOptions.VerifyMatches)
			{
				database.WriteMatches(image1.ImageId, image2.ImageId, matches);

				twoViewGeometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
					camera1,
					FeatureUtils.FeatureKeypointsToPointsVector(cache.GetKeypoints(image1.ImageId)),
					camera2,
					FeatureUtils.FeatureKeypointsToPointsVector(cache.GetKeypoints(image2.ImageId)),
					matches,
					geometryOptions);
			}
			else
			{
				twoViewGeometry = new TwoViewGeometry
				{
					Config = camera1.HasPriorFocalLength && camera2.HasPriorFocalLength
						? TwoViewGeometry.ConfigurationType.Calibrated
						: TwoViewGeometry.ConfigurationType.Uncalibrated,
					InlierMatches = matches,
				};
			}

			database.WriteTwoViewGeometry(image1.ImageId, image2.ImageId, twoViewGeometry);
		}
	}

	// `line_stream >> match.point2D_idx1 >> match.point2D_idx2` on a default FeatureMatch
	// (both kInvalidPoint2DIdx). A token that fails to parse stores 0 and sets failbit; at the
	// end of the line the sentry fails first and nothing is stored. After a failure the next
	// extraction does nothing.
	private static FeatureMatch ReadMatch(CppLineTokens tokens)
	{
		var match = new FeatureMatch();
		if (!tokens.HasToken)
		{
			return match;
		}

		if (!tokens.TryReadUInt32(out uint idx1))
		{
			match.Point2DIdx1 = 0;
			return match;
		}

		match.Point2DIdx1 = idx1;
		if (tokens.HasToken)
		{
			match.Point2DIdx2 = tokens.TryReadUInt32(out uint idx2) ? idx2 : 0;
		}

		return match;
	}
}
