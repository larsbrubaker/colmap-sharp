// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureExtraction.Import: the FeatureImporterController of
// colmap/controllers/feature_extraction.cc (CreateFeatureImporterController), which imports
// SIFT features from text files instead of extracting them: each image needs a file with the
// same name plus ".txt" in the import directory (read by Feature/SiftFeaturesText.cs). The
// extractor is in FeatureExtraction.cs. Tests:
// FeatureExtractionTests.CreateFeatureImporterController_Nominal.
//
// Tier A (exact). As for the extractor, the controller Thread becomes a synchronous method
// with an IProgress (the "Processing file [i/n]" LOG(INFO) line) and a CancellationToken;
// "SKIP: No features found" goes into the progress message.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

public static partial class FeatureExtraction
{
	/// <summary>The <see cref="ControllerProgress.Stage"/> of <see cref="ImportFeatures"/>.</summary>
	public const string ImportStage = "Feature import";

	/// <summary>
	/// Port of CreateFeatureImporterController followed by running it: reads the images of
	/// <paramref name="readerOptions"/> (writing their cameras and rigs) and imports each
	/// image's SIFT features from "<paramref name="importPath"/>/&lt;image name&gt;.txt".
	/// Currently hard-coded to support SIFT features, like COLMAP.
	/// </summary>
	public static void ImportFeatures(
		Database database,
		ImageReaderOptions readerOptions,
		string importPath,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		if (!Directory.Exists(importPath))
		{
			Log.Error("Import directory does not exist.");
			return;
		}

		var imageReader = new ImageReader(readerOptions, database);

		while (imageReader.NextIndex < imageReader.NumImages)
		{
			cancellationToken.ThrowIfCancellationRequested();

			int fileIndex = imageReader.NextIndex + 1;

			// Load image data and possibly save camera to database.
			if (imageReader.Next(out ImageReaderData read, readMask: false) != ImageReader.Status.Success)
			{
				progress?.Report(new ControllerProgress(ImportStage, fileIndex, imageReader.NumImages, read.Image.Name));
				continue;
			}

			Image image = read.Image;
			string path = Path.Combine(importPath, image.Name + ".txt");

			if (File.Exists(path))
			{
				var keypoints = new List<FeatureKeypoint>();
				var descriptors = new FeatureDescriptors();
				SiftFeaturesText.LoadSiftFeaturesFromTextFile(path, keypoints, descriptors);

				using (var databaseTransaction = new DatabaseTransaction(database))
				{
					if (image.ImageId == InvalidImageId)
					{
						image.ImageId = database.WriteImage(image);

						PosePrior posePrior = read.PosePrior;
						if (posePrior.HasPosition() || posePrior.HasGravity())
						{
							posePrior.CorrDataId = image.DataId;
							posePrior.PosePriorId = database.WritePosePrior(posePrior);
						}

						var frame = new Frame();
						frame.SetRigId(read.Rig.RigId);
						frame.AddDataId(image.DataId);
						database.WriteFrame(frame);
					}

					if (!database.ExistsKeypoints(image.ImageId))
					{
						database.WriteKeypoints(image.ImageId, keypoints);
					}

					if (!database.ExistsDescriptors(image.ImageId))
					{
						database.WriteDescriptors(image.ImageId, descriptors);
					}
				}

				progress?.Report(new ControllerProgress(ImportStage, fileIndex, imageReader.NumImages, image.Name));
			}
			else
			{
				progress?.Report(new ControllerProgress(
					ImportStage, fileIndex, imageReader.NumImages, $"SKIP: No features found at {path}"));
			}
		}
	}
}
