// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureExtraction: port of colmap/controllers/feature_extraction.h and .cc - the feature
// extractor controller. It reads every image through ImageReader (ImageReader.cs), which
// writes cameras and rigs, then down-scales, (gravity-)rotates, extracts features
// (Feature/FeatureExtractor.cs), maps keypoints back to the camera's resolution, applies
// the camera mask and the per-image mask, and writes the image, its frame, pose prior,
// keypoints and descriptors to the Database. Tests:
// ColmapSharp.Tests/Controllers/FeatureExtractionTests.cs (feature_extraction_test.cc).
//
// Tier A (exact) given the extractor's output: the bookkeeping around the extractor is
// deterministic, and each image is extracted independently.
//
// Translation notes:
// - COLMAP's reader -> resizer threads -> extractor threads -> writer thread pipeline
//   (JobQueues of size 1) becomes batches: the reader fills a batch of num_threads images
//   sequentially, Parallel.For resizes and extracts them (one extractor per worker, like one
//   per extractor thread; SIFT extraction does not depend on what an extractor processed
//   before, docs/CPP_DIVERGENCES.md entry 43), and the batch is written in reader order.
//   COLMAP's writer commits in extractor completion order, so with several threads its image
//   ids depend on timing; here they always equal COLMAP's single-threaded ids
//   (docs/CPP_DIVERGENCES.md entry 83).
// - The GPU path (SiftGPU, CUDA) is excluded (docs/LICENSE_AUDIT.md); FeatureExtractionOptions
//   has no use_gpu, so every run takes COLMAP's CPU branch.
// - The controller Thread becomes a synchronous method taking the Database (COLMAP opens it
//   from a path), a CancellationToken (Thread::Stop) and an IProgress (the "Processed file
//   [i/n]" LOG(INFO) block). Cancellation throws OperationCanceledException between
//   batches; images of finished batches stay written, as COLMAP's writer keeps them.
// - camera_mask_path becomes ImageReaderOptions.CameraMask, decoded by the host (entry 82);
//   it is converted to grey like Bitmap::Read(path, as_rgb=false).
// - FeatureImporterController (import from text files) is in FeatureExtraction.Import.cs.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap/controllers/feature_extraction.h.</summary>
public static partial class FeatureExtraction
{
	/// <summary>The <see cref="ControllerProgress.Stage"/> of <see cref="ExtractFeatures"/>.</summary>
	public const string ExtractionStage = "Feature extraction";

	// Port of the ImageData job of feature_extraction.cc.
	private sealed class ImageData
	{
		public ImageReader.Status Status = ImageReader.Status.Failure;
		public required ImageReaderData Read;
		public List<FeatureKeypoint> Keypoints = [];
		public FeatureDescriptors Descriptors = new();
		public bool HasMask;
	}

	/// <summary>
	/// Port of CreateFeatureExtractorController followed by running it: reads the images of
	/// <paramref name="readerOptions"/>, extracts features, and writes them to
	/// <paramref name="database"/>. Images whose features are already in the database are
	/// skipped.
	/// </summary>
	public static void ExtractFeatures(
		Database database,
		ImageReaderOptions readerOptions,
		FeatureExtractionOptions extractionOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var imageReader = new ImageReader(readerOptions, database);
		Check.That(readerOptions.Check());
		Check.That(extractionOptions.Check());

		Bitmap? cameraMask = null;
		if (readerOptions.CameraMask is not null)
		{
			cameraMask = readerOptions.CameraMask.IsRGB
				? readerOptions.CameraMask.CloneAsGrey()
				: readerOptions.CameraMask;
		}

		int numThreads = Threading.GetEffectiveNumThreads(extractionOptions.NumThreads);
		Check.Gt(numThreads, 0);

		int maxImageSize = extractionOptions.EffMaxImageSize();
		Check.Gt(maxImageSize, 0);

		FeatureExtractionOptions workerExtractionOptions = extractionOptions.Clone();
		var defaultExtractionOptions = new FeatureExtractionOptions();
		if (extractionOptions.NumThreads == -1 &&
			extractionOptions.Type == FeatureExtractorType.Sift &&
			extractionOptions.MaxImageSize == defaultExtractionOptions.MaxImageSize &&
			extractionOptions.Sift.FirstOctave == defaultExtractionOptions.Sift.FirstOctave)
		{
			Log.Warning(
				"Your current options use the maximum number of threads on the machine to extract " +
				"features. Extracting SIFT features on the CPU can consume a lot of RAM per thread for " +
				"large images. Consider reducing the maximum image size and/or the first octave or " +
				"manually limit the number of extraction threads. Ignore this warning, if your machine " +
				"has sufficient memory for the current settings.");
		}

		int numExtractors;
		switch (extractionOptions.Type)
		{
			case FeatureExtractorType.Sift:
				// Prevent nested threading, as we multi-thread at the controller level as SIFT
				// extraction doesn't require much RAM per extractor.
				numExtractors = numThreads;
				workerExtractionOptions.NumThreads = 1;
				break;
			case FeatureExtractorType.AlikedN16Rot:
			case FeatureExtractorType.AlikedN32:
			case FeatureExtractorType.LomaB:
			case FeatureExtractorType.LomaB128:
				// Use a single extractor with parallelization per image because ALIKED/LoMa
				// require a lot of RAM per extractor.
				numExtractors = 1;
				workerExtractionOptions.NumThreads = numThreads;
				break;
			default:
				throw FeatureExtractionOptions.UnknownType(extractionOptions.Type);
		}

		Check.Gt(numExtractors, 0);
		Check.That(workerExtractionOptions.Check());

		int numImages = imageReader.NumImages;
		int numWritten = 0;
		var batch = new List<ImageData>(numExtractors);
		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = numExtractors,
			CancellationToken = cancellationToken,
		};

		while (imageReader.NextIndex < numImages)
		{
			cancellationToken.ThrowIfCancellationRequested();

			batch.Clear();
			while (batch.Count < numExtractors && imageReader.NextIndex < numImages)
			{
				ImageReader.Status status = imageReader.Next(out ImageReaderData read);
				var imageData = new ImageData { Status = status, Read = read, HasMask = read.Mask is not null };
				if (status != ImageReader.Status.Success)
				{
					// Release the memory, since it is not used afterwards.
					read.Bitmap = new Bitmap();
					read.Mask = null;
				}

				batch.Add(imageData);
			}

			Parallel.For(
				0,
				batch.Count,
				parallelOptions,
				() => FeatureExtractor.Create(workerExtractionOptions),
				(i, _, extractor) =>
				{
					ProcessImage(batch[i], extractor, maxImageSize, cameraMask, cancellationToken);
					return extractor;
				},
				_ => { });

			foreach (ImageData imageData in batch)
			{
				WriteImage(database, imageData);
				numWritten += 1;
				progress?.Report(new ControllerProgress(ExtractionStage, numWritten, numImages, imageData.Read.Image.Name));
			}
		}
	}

	// ImageResizerThread + FeatureExtractorThread for one image.
	private static void ProcessImage(
		ImageData imageData,
		FeatureExtractor extractor,
		int maxImageSize,
		Bitmap? cameraMask,
		CancellationToken cancellationToken)
	{
		ImageReaderData read = imageData.Read;
		if (imageData.Status == ImageReader.Status.Success)
		{
			read.Bitmap.Thumbnail(maxImageSize);

			Bitmap bitmap = read.Bitmap;
			int origWidth = bitmap.Width;
			int origHeight = bitmap.Height;
			int rot90 = read.PosePrior.HasGravity() ? PosePrior.ComputeRot90FromGravity(read.PosePrior.Gravity) : 0;
			if (rot90 > 0)
			{
				bitmap.Rot90(rot90);
			}

			if (extractor.Extract(bitmap, imageData.Keypoints, imageData.Descriptors, cancellationToken))
			{
				if (rot90 > 0)
				{
					int w = bitmap.Width;
					int h = bitmap.Height;
					for (int i = 0; i < imageData.Keypoints.Count; ++i)
					{
						FeatureKeypoint kp = imageData.Keypoints[i];
						kp.Rot90(4 - rot90, w, h);
						imageData.Keypoints[i] = kp;
					}
				}

				ScaleKeypoints(origWidth, origHeight, read.Camera.Width, read.Camera.Height, imageData.Keypoints);
				if (cameraMask is not null)
				{
					MaskFeatures(cameraMask, imageData.Keypoints, imageData.Descriptors);
				}

				if (read.Mask is not null)
				{
					MaskFeatures(read.Mask, imageData.Keypoints, imageData.Descriptors);
				}
			}
			else
			{
				imageData.Status = ImageReader.Status.Failure;
			}
		}

		// Release the memory, since it is not used afterwards.
		read.Bitmap = new Bitmap();
		read.Mask = null;
	}

	// FeatureWriterThread for one image.
	private static void WriteImage(Database database, ImageData imageData)
	{
		ImageReaderData read = imageData.Read;
		if (imageData.Status != ImageReader.Status.Success)
		{
			Log.Warning($"{read.Image.Name} {ImageReader.StatusToString(imageData.Status)}");
			return;
		}

		using var databaseTransaction = new DatabaseTransaction(database);

		Image image = read.Image;
		if (image.ImageId == InvalidImageId)
		{
			image.ImageId = database.WriteImage(image);

			PosePrior posePrior = read.PosePrior;
			if (posePrior.HasPosition() || posePrior.HasGravity())
			{
				posePrior.CorrDataId = image.DataId;
				posePrior.PosePriorId = database.WritePosePrior(posePrior);
				read.PosePrior = posePrior;
			}

			var frame = new Frame();
			frame.SetRigId(read.Rig.RigId);
			frame.AddDataId(image.DataId);
			database.WriteFrame(frame);
		}

		if (!database.ExistsKeypoints(image.ImageId))
		{
			database.WriteKeypoints(image.ImageId, imageData.Keypoints);
		}

		if (!database.ExistsDescriptors(image.ImageId))
		{
			database.WriteDescriptors(image.ImageId, imageData.Descriptors);
		}
	}

	// Port of ScaleKeypoints: maps keypoints from the (down-scaled) bitmap to the camera's
	// resolution.
	private static void ScaleKeypoints(int bitmapWidth, int bitmapHeight, int cameraWidth, int cameraHeight, List<FeatureKeypoint> keypoints)
	{
		if (bitmapWidth != cameraWidth || bitmapHeight != cameraHeight)
		{
			float scaleX = (float)cameraWidth / bitmapWidth;
			float scaleY = (float)cameraHeight / bitmapHeight;
			for (int i = 0; i < keypoints.Count; ++i)
			{
				FeatureKeypoint keypoint = keypoints[i];
				keypoint.Rescale(scaleX, scaleY);
				keypoints[i] = keypoint;
			}
		}
	}

	// Port of MaskFeatures: drops the features on black (or outside) mask pixels, keeping the
	// order of the rest.
	private static void MaskFeatures(Bitmap mask, List<FeatureKeypoint> keypoints, FeatureDescriptors descriptors)
	{
		RowMajorMatrix<byte> data = descriptors.Data;
		int cols = data.Cols;
		int outIndex = 0;
		for (int i = 0; i < keypoints.Count; ++i)
		{
			BitmapColor<byte>? color = mask.GetPixel((int)keypoints[i].X, (int)keypoints[i].Y);
			if (color is null || color.Value.R == 0)
			{
				// Delete this keypoint by not copying it to the output.
			}
			else
			{
				// Retain this keypoint by copying it to the output index (in case this index
				// differs from its current position).
				if (outIndex != i)
				{
					keypoints[outIndex] = keypoints[i];
					data.Row(i).CopyTo(data.Row(outIndex));
				}

				outIndex += 1;
			}
		}

		keypoints.RemoveRange(outIndex, keypoints.Count - outIndex);
		var resized = new RowMajorMatrix<byte>(outIndex, cols);
		Array.Copy(data.Data, resized.Data, outIndex * cols);
		descriptors.Data = resized;
	}
}
