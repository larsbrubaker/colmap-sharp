// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Alignment: colmap/estimators/alignment.h and .cc - aligning a reconstruction to image
// locations or pose priors, aligning two reconstructions to each other (via reprojections,
// projection centers or shared 3D points), and measuring per-image alignment errors
// (ImageAlignmentError, AlignmentErrorSummary). Merging reconstructions and restoring the
// metric rig scale are in Alignment.Merge.cs; the LO-RANSAC estimator of
// AlignReconstructionsViaReprojections is in Alignment.Estimator.cs. The similarity
// transforms come from Solvers/SimilarityTransform.cs. Tests:
// ColmapSharp.Tests/Estimators/AlignmentTests.cs (alignment_test.cc 1:1); the test-side
// ReconstructionNear matcher (reconstruction_matchers.h) is built on
// AlignReconstructionsViaProjCenters and ComputeImageAlignmentError.
//
// Tier C (outcome) for the robust alignments (LO-RANSAC), Tier B for ComputeImageAlignmentError
// and AlignmentErrorSummary (scalar, but through quaternion angular distance and
// nth_element-reordered sums).
//
// Translation notes:
// - `Sim3d* tgt_from_src` becomes `ref Sim3d`, left unchanged on failure as in C++ (as
//   SimilarityTransform.EstimateSim3dRobust does). AlignReconstructionToLocations accepts a
//   null pointer in C++; no caller passes one, so there is no pointer-free overload.
// - COLMAP's LOG(WARNING)/LOG(ERROR) lines go to Util/Log.cs; LOG(INFO)/VLOG lines are dropped.
// - AlignReconstructionsViaPoints picks, per source point, the target point seen most often
//   along its track; std::max_element over a hash map breaks ties by hash order, the port by
//   first appearance along the track (divergence 49).

using System.Runtime.InteropServices;

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Image alignment errors in the target coordinate frame. Port of
/// colmap::ImageAlignmentError.
/// </summary>
public sealed class ImageAlignmentError
{
	/// <summary>The image's name.</summary>
	public string ImageName { get; set; } = "";

	/// <summary>Angle between the two camera orientations, in degrees (-1 until computed).</summary>
	public double RotationErrorDeg { get; set; } = -1;

	/// <summary>Distance between the two projection centers (-1 until computed).</summary>
	public double ProjCenterError { get; set; } = -1;
}

/// <summary>
/// Summary of alignment errors for image poses. Port of colmap::AlignmentErrorSummary.
/// </summary>
public sealed class AlignmentErrorSummary
{
	/// <summary>Statistics of one kind of error. Port of AlignmentErrorSummary::Statistics.</summary>
	public readonly record struct Statistics(double Min, double Max, double Mean, double Median, double P90, double P99);

	/// <summary>Statistics of the rotation errors, in degrees.</summary>
	public Statistics RotationErrorsDeg { get; private set; }

	/// <summary>Statistics of the projection center errors.</summary>
	public Statistics ProjCenterErrors { get; private set; }

	/// <summary>
	/// Summarize <paramref name="errors"/>; all statistics are zero when it is empty. Port
	/// of AlignmentErrorSummary::Compute.
	/// </summary>
	public static AlignmentErrorSummary Compute(IReadOnlyList<ImageAlignmentError> errors)
	{
		var summary = new AlignmentErrorSummary();
		if (errors.Count == 0)
		{
			return summary;
		}

		var rotationErrorsDeg = new double[errors.Count];
		var projCenterErrors = new double[errors.Count];
		for (int i = 0; i < errors.Count; ++i)
		{
			rotationErrorsDeg[i] = errors[i].RotationErrorDeg;
			projCenterErrors[i] = errors[i].ProjCenterError;
		}

		summary.RotationErrorsDeg = ComputeStatistics(rotationErrorsDeg);
		summary.ProjCenterErrors = ComputeStatistics(projCenterErrors);
		return summary;
	}

	// COLMAP's order of calls: Percentile and Median reorder the values in place (as in
	// C++), so the mean is summed over the order the first two percentiles left.
	private static Statistics ComputeStatistics(double[] values)
	{
		double min = MathUtils.Percentile<double>(values, 0);
		double max = MathUtils.Percentile<double>(values, 100);
		double mean = MathUtils.Mean<double>(values);
		double median = MathUtils.Median<double>(values);
		double p90 = MathUtils.Percentile<double>(values, 90);
		double p99 = MathUtils.Percentile<double>(values, 99);
		return new Statistics(min, max, mean, median, p90, p99);
	}
}

/// <summary>Port of colmap/estimators/alignment.h.</summary>
public static partial class Alignment
{
	/// <summary>
	/// Robustly align reconstruction to given image locations (projection centers). Port of
	/// colmap::AlignReconstructionToLocations.
	/// </summary>
	public static bool AlignReconstructionToLocations(
		Reconstruction srcReconstruction,
		IReadOnlyList<string> tgtImageNames,
		IReadOnlyList<Vector3d> tgtImageLocations,
		int minCommonImages,
		RansacOptions ransacOptions,
		ref Sim3d tgtFromSrc)
	{
		Check.Ge(minCommonImages, 3);
		Check.Eq(tgtImageNames.Count, tgtImageLocations.Count);

		// Find out which images are contained in the reconstruction and get the positions
		// of their camera centers.
		var commonImageIds = new HashSet<uint>();
		var src = new List<Vector3d>();
		var dst = new List<Vector3d>();
		for (int i = 0; i < tgtImageNames.Count; ++i)
		{
			Image? srcImage = srcReconstruction.FindImageWithName(tgtImageNames[i]);
			if (srcImage is null)
			{
				continue;
			}

			if (!srcImage.HasPose)
			{
				continue;
			}

			// Ignore duplicate images.
			if (!commonImageIds.Add(srcImage.ImageId))
			{
				continue;
			}

			src.Add(srcImage.ProjectionCenter());
			dst.Add(tgtImageLocations[i]);
		}

		// Only compute the alignment if there are enough correspondences.
		if (commonImageIds.Count < minCommonImages)
		{
			return false;
		}

		var estimate = new Sim3d();
		var report = SimilarityTransform.EstimateSim3dRobust(CollectionsMarshal.AsSpan(src), CollectionsMarshal.AsSpan(dst), ransacOptions, ref estimate);

		if (report.Support.NumInliers < minCommonImages)
		{
			return false;
		}

		tgtFromSrc = estimate;
		return true;
	}

	/// <summary>
	/// Robustly align reconstruction to given pose priors. If max_error is not set in the
	/// RANSAC options, derive it from the median position covariance. Port of
	/// colmap::AlignReconstructionToPosePriors.
	/// </summary>
	public static bool AlignReconstructionToPosePriors(
		Reconstruction srcReconstruction,
		IReadOnlyList<PosePrior> tgtPosePriors,
		RansacOptions ransacOptions,
		double priorPositionFallbackStddev,
		ref Sim3d tgtFromSrc)
	{
		Check.Gt(priorPositionFallbackStddev, 0.0);

		var src = new List<Vector3d>(tgtPosePriors.Count);
		var tgt = new List<Vector3d>(tgtPosePriors.Count);
		var rmsVars = new List<double>(tgtPosePriors.Count);

		var tgtImageToPosePrior = new Dictionary<uint, PosePrior>();
		foreach (PosePrior posePrior in tgtPosePriors)
		{
			if (posePrior.CorrDataId.SensorId.Type == SensorType.Camera && posePrior.HasPosition())
			{
				Check.That(
					tgtImageToPosePrior.TryAdd((uint)posePrior.CorrDataId.Id, posePrior),
					$"Duplicate pose prior for image {posePrior.CorrDataId.Id}");
			}
		}

		foreach (uint imageId in srcReconstruction.RegImageIds())
		{
			if (tgtImageToPosePrior.TryGetValue(imageId, out PosePrior posePrior))
			{
				Image image = srcReconstruction.Image(imageId);
				src.Add(image.ProjectionCenter());
				tgt.Add(posePrior.Position);
				double trace = posePrior.PositionCovariance.Trace();
				if (trace > 0.0)
				{
					rmsVars.Add(trace / 3.0);
				}
			}
		}

		if (src.Count < 3)
		{
			Log.Warning("Not enough valid pose priors for alignment");
			return false;
		}

		if (ransacOptions.MaxError <= 0)
		{
			if (rmsVars.Count == 0)
			{
				Log.Warning("No pose priors with valid covariance found.");
				rmsVars.Add(priorPositionFallbackStddev * priorPositionFallbackStddev);
			}

			// Scale the median RMS variance by the 95% chi-square quantile for 3 DOF.
			ransacOptions.MaxError = Math.Sqrt(MathUtils.ChiSquare95ThreeDof * MathUtils.Median(CollectionsMarshal.AsSpan(rmsVars)));
		}

		return SimilarityTransform.EstimateSim3dRobust(CollectionsMarshal.AsSpan(src), CollectionsMarshal.AsSpan(tgt), ransacOptions, ref tgtFromSrc).Success;
	}

	/// <summary>
	/// Robustly compute alignment between reconstructions by finding images that are
	/// registered in both reconstructions. The alignment is then estimated robustly inside
	/// RANSAC from corresponding projection centers. An alignment is verified by reprojecting
	/// common 3D point observations. The min_inlier_observations threshold determines how
	/// many observations in a common image must reproject within the given threshold. Port
	/// of colmap::AlignReconstructionsViaReprojections.
	/// </summary>
	public static bool AlignReconstructionsViaReprojections(
		Reconstruction srcReconstruction,
		Reconstruction tgtReconstruction,
		double minInlierObservations,
		double maxReprojError,
		ref Sim3d tgtFromSrc)
	{
		Check.Ge(minInlierObservations, 0.0);
		Check.Le(minInlierObservations, 1.0);

		var ransacOptions = new RansacOptions
		{
			MaxError = 1.0 - minInlierObservations,
			MinInlierRatio = 0.2,
		};

		var ransac = new LoRansac<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator, Image, Image, Sim3d>(
			ransacOptions,
			new ReconstructionAlignmentEstimator(maxReprojError, srcReconstruction, tgtReconstruction),
			new ReconstructionAlignmentEstimator(maxReprojError, srcReconstruction, tgtReconstruction));

		List<(uint ImageId, uint OtherImageId)> commonImageIds = srcReconstruction.FindCommonRegImageIds(tgtReconstruction);

		if (commonImageIds.Count < 3)
		{
			return false;
		}

		var srcImages = new Image[commonImageIds.Count];
		var tgtImages = new Image[commonImageIds.Count];
		for (int i = 0; i < commonImageIds.Count; ++i)
		{
			srcImages[i] = srcReconstruction.Image(commonImageIds[i].ImageId);
			tgtImages[i] = tgtReconstruction.Image(commonImageIds[i].OtherImageId);
		}

		var report = ransac.Estimate(srcImages, tgtImages);

		if (report.Success)
		{
			tgtFromSrc = report.Model;
		}

		return report.Success;
	}

	/// <summary>
	/// Robustly compute alignment between reconstructions by finding images that are
	/// registered in both reconstructions. The alignment is then estimated robustly inside
	/// RANSAC from corresponding projection centers and by minimizing the Euclidean distance
	/// between them in world space. Port of colmap::AlignReconstructionsViaProjCenters.
	/// </summary>
	public static bool AlignReconstructionsViaProjCenters(
		Reconstruction srcReconstruction,
		Reconstruction tgtReconstruction,
		double maxProjCenterError,
		ref Sim3d tgtFromSrc)
	{
		Check.Gt(maxProjCenterError, 0.0);

		var refImageNames = new List<string>();
		var refProjCenters = new List<Vector3d>();
		foreach (Image image in tgtReconstruction.Images.Values)
		{
			if (image.HasPose)
			{
				refImageNames.Add(image.Name);
				refProjCenters.Add(image.ProjectionCenter());
			}
		}

		var ransacOptions = new RansacOptions { MaxError = maxProjCenterError };
		return AlignReconstructionToLocations(
			srcReconstruction,
			refImageNames,
			refProjCenters,
			minCommonImages: 3,
			ransacOptions,
			ref tgtFromSrc);
	}

	/// <summary>
	/// Robustly compute the alignment between reconstructions that share the same 2D points.
	/// It is estimated by minimizing the 3D distance between corresponding 3D points. Port
	/// of colmap::AlignReconstructionsViaPoints.
	/// </summary>
	public static bool AlignReconstructionsViaPoints(
		Reconstruction srcReconstruction,
		Reconstruction tgtReconstruction,
		int minCommonObservations,
		double maxError,
		double minInlierRatio,
		ref Sim3d tgtFromSrc)
	{
		Check.Gt(minCommonObservations, 0);
		Check.Gt(maxError, 0.0);
		Check.Ge(minInlierRatio, 0.0);
		Check.Le(minInlierRatio, 1.0);

		var srcXyz = new List<Vector3d>();
		var tgtXyz = new List<Vector3d>();

		// counts in first-seen order (the tie-break of max_element below), with an index.
		var counts = new List<(ulong Point3DId, int Count)>();
		var countIndex = new Dictionary<ulong, int>();

		// Associate 3D points using point2D_idx
		foreach (Point3D srcPoint3D in srcReconstruction.Points3D.Values)
		{
			counts.Clear();
			countIndex.Clear();

			// Count how often a 3D point in tgt is associated to this 3D point.
			foreach (TrackElement trackEl in srcPoint3D.Track.Elements)
			{
				Image tgtImage = tgtReconstruction.Image(trackEl.ImageId);
				if (!tgtImage.HasPose)
				{
					continue;
				}

				Point2D tgtPoint2D = tgtImage.Point2DAt(trackEl.Point2DIdx);
				if (tgtPoint2D.HasPoint3D)
				{
					// As in COLMAP, the first association counts as 0, so a count is the
					// number of associations minus one.
					if (countIndex.TryGetValue(tgtPoint2D.Point3DId, out int idx))
					{
						counts[idx] = (tgtPoint2D.Point3DId, counts[idx].Count + 1);
					}
					else
					{
						countIndex[tgtPoint2D.Point3DId] = counts.Count;
						counts.Add((tgtPoint2D.Point3DId, 0));
					}
				}
			}

			if (counts.Count == 0)
			{
				continue;
			}

			// The 3D point in tgt who is associated the most is selected
			(ulong Point3DId, int Count) bestPoint3D = counts[0];
			for (int i = 1; i < counts.Count; ++i)
			{
				if (bestPoint3D.Count < counts[i].Count)
				{
					bestPoint3D = counts[i];
				}
			}

			if (bestPoint3D.Count >= minCommonObservations)
			{
				srcXyz.Add(srcPoint3D.Xyz);
				tgtXyz.Add(tgtReconstruction.Point3D(bestPoint3D.Point3DId).Xyz);
			}
		}

		Check.Eq(srcXyz.Count, tgtXyz.Count);

		var ransacOptions = new RansacOptions
		{
			MaxError = maxError,
			MinInlierRatio = minInlierRatio,
		};
		return SimilarityTransform.EstimateSim3dRobust(CollectionsMarshal.AsSpan(srcXyz), CollectionsMarshal.AsSpan(tgtXyz), ransacOptions, ref tgtFromSrc).Success;
	}

	/// <summary>
	/// Compute image alignment errors in the target coordinate frame, one per image
	/// registered in both reconstructions (matched by name). Port of
	/// colmap::ComputeImageAlignmentError.
	/// </summary>
	public static List<ImageAlignmentError> ComputeImageAlignmentError(
		Reconstruction srcReconstruction,
		Reconstruction tgtReconstruction,
		Sim3d tgtFromSrc)
	{
		List<(uint ImageId, uint OtherImageId)> commonImageIds = srcReconstruction.FindCommonRegImageIds(tgtReconstruction);
		var errors = new List<ImageAlignmentError>(commonImageIds.Count);
		foreach (var (srcImageId, tgtImageId) in commonImageIds)
		{
			Image srcImage = srcReconstruction.Image(srcImageId);
			Rigid3d tgtWorldFromSrcCam = Pose.TransformCameraWorld(tgtFromSrc, srcImage.CamFromWorld()).Inverse();
			Rigid3d tgtWorldFromTgtCam = tgtReconstruction.Image(tgtImageId).CamFromWorld().Inverse();

			errors.Add(new ImageAlignmentError
			{
				ImageName = srcImage.Name,
				RotationErrorDeg = MathUtils.RadToDeg(tgtWorldFromSrcCam.Rotation.AngularDistance(tgtWorldFromTgtCam.Rotation)),
				ProjCenterError = (tgtWorldFromSrcCam.Translation - tgtWorldFromTgtCam.Translation).Norm,
			});
		}

		return errors;
	}
}
