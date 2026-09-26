// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftFeaturesText: port of colmap::LoadSiftFeaturesFromTextFile (feature/sift.h and .cc),
// which reads SIFT features from a text file ("num_features dim" header, then one
// "x y scale orientation d0 ... d127" line per feature). Kept apart from Sift.cs, which holds
// the extractor. The feature importer (Controllers/FeatureExtraction.Import.cs) is its only
// caller; its test (FeatureExtractionTests.CreateFeatureImporterController_Nominal) covers it.
//
// Tier A (exact). The stream reads (`std::getline`, `line_stream >> value` in the classic
// locale) go through Util/CppLineTokens.cs, so a token parses as libc++ parses it (strtof for
// floats), except that a token is taken whole (docs/CPP_DIVERGENCES.md entry 25).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of the SIFT text-file reader of colmap/feature/sift.h.</summary>
public static class SiftFeaturesText
{
	/// <summary>
	/// Port of LoadSiftFeaturesFromTextFile: reads the features of <paramref name="path"/>
	/// into <paramref name="keypoints"/> (replaced) and <paramref name="descriptors"/>.
	/// Throws ArgumentException (COLMAP's THROW_CHECK) on a malformed file.
	/// </summary>
	public static void LoadSiftFeaturesFromTextFile(
		string path, List<FeatureKeypoint> keypoints, FeatureDescriptors descriptors)
	{
		Check.NotNull(keypoints);
		Check.NotNull(descriptors);

		List<(string Text, bool ValidUtf8)> lines;
		using (FileStream stream = FileOpen.OpenRead(path))
		{
			lines = CppLineTokens.ReadLines(stream);
		}

		// std::getline erases the line first, so a line past the end of the file reads empty.
		int lineIdx = 0;
		string NextLine() => lineIdx < lines.Count ? lines[lineIdx++].Text : "";

		var headerLineStream = new CppLineTokens(NextLine());
		ulong dim = 0;
		bool headerRead = headerLineStream.TryReadUInt32(out uint numFeatures) && headerLineStream.TryReadUInt64(out dim);
		Check.That(headerRead, expression: "header_line_stream >> num_features >> dim");

		Check.Eq(dim, (ulong)SiftCpuFeatureExtractor.SiftDescriptorDim,
			"SIFT features must have kSiftDescriptorDim dimensions");

		keypoints.Clear();
		var data = new RowMajorMatrix<byte>(checked((int)numFeatures), (int)dim);

		for (int i = 0; i < (int)numFeatures; ++i)
		{
			var featureLineStream = new CppLineTokens(NextLine());

			float y = 0, scale = 0, orientation = 0;
			bool keypointRead =
				featureLineStream.TryReadFloat(out float x) &&
				featureLineStream.TryReadFloat(out y) &&
				featureLineStream.TryReadFloat(out scale) &&
				featureLineStream.TryReadFloat(out orientation);
			Check.That(keypointRead, expression: "feature_line_stream >> x >> y >> scale >> orientation");

			keypoints.Add(new FeatureKeypoint(x, y, scale, orientation));

			// Descriptor
			for (int j = 0; j < (int)dim; ++j)
			{
				Check.That(featureLineStream.TryReadFloat(out float value), expression: "feature_line_stream >> value");
				Check.Ge(value, 0f);
				Check.Le(value, 255f);
				data[i, j] = MathUtils.TruncateCast<float, byte>(value);
			}
		}

		descriptors.Type = FeatureExtractorType.Sift;
		descriptors.Data = data;
	}
}
