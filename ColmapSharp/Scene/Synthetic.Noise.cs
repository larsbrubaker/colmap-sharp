// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Synthetic.Noise: SynthesizeNoise and SynthesizeImages of colmap/scene/synthetic.cc, the two
// functions that work on a dataset SynthesizeDataset (Synthetic.cs) made.
//
// SynthesizeImages differs in its output: COLMAP writes each image to image_path/name with
// OpenImageIO, and image file I/O is the host's job here (PORTING_PLAN.md, bitmap_test.cc
// skips), so each rendered Bitmap goes to a caller-supplied sink with its image name instead
// (divergence 32).
//
// Order: images and 3D points are visited in ascending id order; COLMAP visits them in hash
// order (divergence 31), which decides which one receives which draws.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public static partial class Synthetic
{
	/// <summary>
	/// Adds Gaussian noise to the frame poses, 2D points (and the database keypoints), 3D
	/// points and the database pose priors. Port of colmap::SynthesizeNoise.
	/// </summary>
	public static void SynthesizeNoise(SyntheticNoiseOptions options, Reconstruction reconstruction, Database? database = null)
	{
		Check.Ge(options.RigFromWorldTranslationStddev, 0.0);
		Check.Ge(options.RigFromWorldRotationStddev, 0.0);
		Check.Ge(options.Point3DStddev, 0.0);
		Check.Ge(options.Point2DStddev, 0.0);
		Check.Ge(options.PriorPositionStddev, 0.0);
		Check.Ge(options.PriorGravityStddev, 0.0);

		foreach (uint frameId in reconstruction.RegFrameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			Rigid3d rigFromWorld = frame.RigFromWorld();
			Quaterniond rotation = rigFromWorld.Rotation;
			Vector3d translation = rigFromWorld.Translation;

			if (options.RigFromWorldRotationStddev > 0.0)
			{
				double angle = Math.Clamp(RandomUtils.RandomGaussian(0.0, options.RigFromWorldRotationStddev), -180.0, 180.0);
				rotation *= Quaterniond.FromAngleAxis(new AngleAxisd(MathUtils.DegToRad(angle), new Vector3d(0, 0, 1)));
			}

			if (options.RigFromWorldTranslationStddev > 0.0)
			{
				translation += RandomGaussianVector3d(options.RigFromWorldTranslationStddev);
			}

			frame.SetRigFromWorld(new Rigid3d(rotation, translation));
		}

		if (options.Point2DStddev > 0.0)
		{
			foreach (uint imageId in reconstruction.Images.Keys)
			{
				AddPoint2DNoise(options.Point2DStddev, reconstruction.Image(imageId), database);
			}
		}

		if (options.Point3DStddev > 0.0)
		{
			foreach (Point3D point3D in reconstruction.Points3D.Values)
			{
				point3D.Xyz += RandomGaussianVector3d(options.Point3DStddev);
			}
		}

		if (database != null && (options.PriorPositionStddev > 0.0 || options.PriorGravityStddev > 0.0))
		{
			foreach (PosePrior readPosePrior in database.ReadAllPosePriors())
			{
				PosePrior posePrior = readPosePrior;
				AddPosePriorNoise(options, ref posePrior);
				database.UpdatePosePrior(posePrior);
			}
		}

		reconstruction.UpdatePoint3DErrors();
	}

	/// <summary>
	/// Renders an image per reconstruction image: a dark patch with a pattern unique to the
	/// 3D point (or unique on its own) around every 2D point, and a small bright peak at its
	/// center. Notice that this approach does not result in perfect feature detections and
	/// matches due to overlapping patches, etc. Port of colmap::SynthesizeImages, except that
	/// each image goes to <paramref name="writeImage"/> (image name, bitmap) instead of a file
	/// (see the file header).
	/// </summary>
	public static void SynthesizeImages(SyntheticImageOptions options, Reconstruction reconstruction, Action<string, Bitmap> writeImage)
	{
		Check.Gt(options.FeaturePatchRadius, 0);
		Check.Lt(options.FeaturePeakRadius, options.FeaturePatchRadius);
		Check.Gt(options.FeaturePatchMaxBrightness, 0);
		Check.Lt(options.FeaturePatchMaxBrightness, 255);

		double patchRadius = Math.Sqrt(2 * options.FeaturePatchRadius * options.FeaturePatchRadius);

		int totalNumDescriptors = 0;
		foreach (Image image in reconstruction.Images.Values)
		{
			Camera camera = image.CameraPtr;
			var bitmap = new Bitmap(camera.Width, camera.Height, asRgb: true);
			bitmap.Fill(new BitmapColor<byte>(0, 0, 0));

			foreach (Point2D point2D in image.Points2D)
			{
				// std::round: halfway cases away from zero.
				int x = (int)Math.Round(point2D.Xy.X, MidpointRounding.AwayFromZero);
				int y = (int)Math.Round(point2D.Xy.Y, MidpointRounding.AwayFromZero);
				if (x < 0 || y < 0 || x >= camera.Width || y >= camera.Height)
				{
					continue;
				}

				// mt19937's seed is 32 bits (libc++ uint_fast32_t): the id is truncated.
				uint seed = point2D.HasPoint3D
					? unchecked((uint)point2D.Point3DId)
					: unchecked((uint)(reconstruction.NumPoints3D + ++totalNumDescriptors));
				var featureGenerator = new Mt19937(seed);
				DrawFeature(options, bitmap, x, y, patchRadius, featureGenerator);
			}

			writeImage(image.Name, bitmap);
		}
	}

	private static void DrawFeature(SyntheticImageOptions options, Bitmap bitmap, int x, int y, double patchRadius, Mt19937 featureGenerator)
	{
		// Draw a circular patch around the feature with a unique pattern with the aim of
		// producing a unique feature descriptor. Make the pattern a bit darker than the peak,
		// so the keypoint is detected at the center.
		int patchMinX = Math.Max(x - options.FeaturePatchRadius, 0);
		int patchMaxX = Math.Min(x + options.FeaturePatchRadius, bitmap.Width);
		int patchMinY = Math.Max(y - options.FeaturePatchRadius, 0);
		int patchMaxY = Math.Min(y + options.FeaturePatchRadius, bitmap.Height);
		for (int py = patchMinY; py < patchMaxY; ++py)
		{
			for (int px = patchMinX; px < patchMaxX; ++px)
			{
				double radius = Math.Sqrt((px - x) * (px - x) + (py - y) * (py - y));
				if (radius > options.FeaturePatchRadius)
				{
					continue;
				}

				// Adjust the brightness so it fades out to the edge of the patch. The double
				// bound converts to the distribution's int by truncation, as in C++.
				int maxBrightness = (int)((1.0 - radius / patchRadius) * options.FeaturePatchMaxBrightness);
				byte brightness = (byte)LibcxxRandom.UniformInt(featureGenerator, 0, maxBrightness);
				bitmap.SetPixel(px, py, new BitmapColor<byte>(brightness));
			}
		}

		// Draw a small, bright peak around the feature for keypoint detection.
		int peakMinX = Math.Max(x - options.FeaturePeakRadius, 0);
		int peakMaxX = Math.Min(x + options.FeaturePeakRadius, bitmap.Width);
		int peakMinY = Math.Max(y - options.FeaturePeakRadius, 0);
		int peakMaxY = Math.Min(y + options.FeaturePeakRadius, bitmap.Height);
		// Three draws in argument order (clang evaluates left to right).
		byte r = (byte)LibcxxRandom.UniformInt(featureGenerator, options.FeaturePatchMaxBrightness, 255);
		byte g = (byte)LibcxxRandom.UniformInt(featureGenerator, options.FeaturePatchMaxBrightness, 255);
		byte b = (byte)LibcxxRandom.UniformInt(featureGenerator, options.FeaturePatchMaxBrightness, 255);
		var peakColor = new BitmapColor<byte>(r, g, b);
		for (int py = peakMinY; py < peakMaxY; ++py)
		{
			for (int px = peakMinX; px < peakMaxX; ++px)
			{
				bitmap.SetPixel(px, py, peakColor);
			}
		}
	}

	private static void AddPoint2DNoise(double stddev, Image image, Database? database)
	{
		List<Point2D> points2D = image.Points2D;
		for (int i = 0; i < points2D.Count; ++i)
		{
			Point2D point2D = points2D[i];
			double dx = RandomUtils.RandomGaussian(0.0, stddev);
			double dy = RandomUtils.RandomGaussian(0.0, stddev);
			point2D.Xy += new Vector2d(dx, dy);
			points2D[i] = point2D;
		}

		if (database != null)
		{
			List<Feature.FeatureKeypoint> keypoints = database.ReadKeypoints(image.ImageId);
			for (int point2DIdx = 0; point2DIdx < keypoints.Count; ++point2DIdx)
			{
				Feature.FeatureKeypoint keypoint = keypoints[point2DIdx];
				keypoint.X = (float)points2D[point2DIdx].Xy.X;
				keypoint.Y = (float)points2D[point2DIdx].Xy.Y;
				keypoints[point2DIdx] = keypoint;
			}

			database.UpdateKeypoints(image.ImageId, keypoints);
		}
	}

	private static void AddPosePriorNoise(SyntheticNoiseOptions options, ref PosePrior posePrior)
	{
		if (options.PriorPositionStddev > 0.0)
		{
			bool priorInWgs84 = posePrior.CoordinateSystem == PosePriorCoordinateSystem.Wgs84;
			if (priorInWgs84)
			{
				PosePriorPositionWgs84ToCartesian(ref posePrior);
			}

			posePrior.Position += RandomGaussianVector3d(options.PriorPositionStddev);
			if (!posePrior.HasPositionCov())
			{
				posePrior.PositionCovariance = Matrix3d.Zero;
			}

			posePrior.PositionCovariance += options.PriorPositionStddev * options.PriorPositionStddev * Matrix3d.Identity;
			if (priorInWgs84)
			{
				PosePriorPositionCartesianToWgs84(ref posePrior);
			}
		}

		if (options.PriorGravityStddev > 0.0)
		{
			double angle = RandomUtils.RandomGaussian(0.0, MathUtils.DegToRad(options.PriorGravityStddev));
			Vector3d axis = posePrior.Gravity.Cross(RandomEigen.RandomEigenVector3d()).Normalized();
			// Eigen's AngleAxis * vector goes through the rotation matrix.
			posePrior.Gravity = (new AngleAxisd(angle, axis).ToRotationMatrix() * posePrior.Gravity).Normalized();
		}
	}

	// Eigen::Vector3d(RandomGaussian(0, s), RandomGaussian(0, s), RandomGaussian(0, s)),
	// drawn x, y, z (clang's left-to-right argument order).
	private static Vector3d RandomGaussianVector3d(double stddev)
	{
		double x = RandomUtils.RandomGaussian(0.0, stddev);
		double y = RandomUtils.RandomGaussian(0.0, stddev);
		double z = RandomUtils.RandomGaussian(0.0, stddev);
		return new Vector3d(x, y, z);
	}
}
