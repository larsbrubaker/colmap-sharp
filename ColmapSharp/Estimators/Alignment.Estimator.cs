// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionAlignmentEstimator: the anonymous-namespace estimator of
// colmap/estimators/alignment.cc that AlignReconstructionsViaReprojections (Alignment.cs)
// runs LO-RANSAC with. A sample is a pair of the same image registered in the source and
// in the target reconstruction; the model is tgt_from_src, estimated from the images'
// projection centers (Solvers/SimilarityTransform.cs) and scored by how many of each
// image's common 3D points reproject in both directions. Tests: through
// ColmapSharp.Tests/Estimators/AlignmentTests.cs (alignment_test.cc).
//
// X_t and Y_t are `const Image*` in C++; here they are Image references. The struct only
// holds references and a threshold, so LO-RANSAC's per-worker copy is a cheap copy.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Estimates tgt_from_src from images registered in two reconstructions. Port of the
/// ReconstructionAlignmentEstimator of colmap/estimators/alignment.cc.
/// </summary>
internal readonly struct ReconstructionAlignmentEstimator
	: IEstimator<Image, Image, Sim3d>, ILocalEstimator<Image, Image, Sim3d>
{
	private readonly double _maxSquaredReprojError;
	private readonly Reconstruction _srcReconstruction;
	private readonly Reconstruction _tgtReconstruction;

	public ReconstructionAlignmentEstimator(double maxReprojError, Reconstruction srcReconstruction, Reconstruction tgtReconstruction)
	{
		Check.Ge(maxReprojError, 0.0);
		_maxSquaredReprojError = maxReprojError * maxReprojError;
		_srcReconstruction = Check.NotNull(srcReconstruction);
		_tgtReconstruction = Check.NotNull(tgtReconstruction);
	}

	/// <summary>kMinNumSamples.</summary>
	public static int MinNumSamples => 3;

	/// <summary>Estimate 3D similarity transform from corresponding projection centers.</summary>
	public void Estimate(ReadOnlySpan<Image> srcImages, ReadOnlySpan<Image> tgtImages, List<Sim3d> models)
	{
		Check.Ge(srcImages.Length, 3);
		Check.Ge(tgtImages.Length, 3);
		Check.Eq(srcImages.Length, tgtImages.Length);

		models.Clear();

		var projCenters1 = new Vector3d[srcImages.Length];
		var projCenters2 = new Vector3d[tgtImages.Length];
		for (int i = 0; i < srcImages.Length; ++i)
		{
			Check.Eq(srcImages[i].ImageId, tgtImages[i].ImageId);
			projCenters1[i] = srcImages[i].ProjectionCenter();
			projCenters2[i] = tgtImages[i].ProjectionCenter();
		}

		var tgtFromSrc = new Sim3d();
		if (!SimilarityTransform.EstimateSim3d(projCenters1, projCenters2, ref tgtFromSrc))
		{
			return;
		}

		models.Add(tgtFromSrc);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Image> srcImages, ReadOnlySpan<Image> tgtImages, in Sim3d initialModel, List<Sim3d> models)
	{
		Estimate(srcImages, tgtImages, models);
	}

	/// <summary>
	/// For each image, determine the ratio of 3D points that correctly project from one image
	/// to the other image and vice versa for the given tgt_from_src. The residual is then
	/// defined as 1 minus this ratio, i.e., an error threshold of 0.3 means that 70% of the
	/// points for that image must reproject within the given maximum reprojection error
	/// threshold. (The value handed to RANSAC is that residual squared.)
	/// </summary>
	public void Residuals(ReadOnlySpan<Image> srcImages, ReadOnlySpan<Image> tgtImages, in Sim3d tgtFromSrc, Span<double> residuals)
	{
		Check.Eq(srcImages.Length, tgtImages.Length);
		Check.Eq(residuals.Length, srcImages.Length);

		Sim3d srcFromTgt = tgtFromSrc.Inverse();

		for (int i = 0; i < srcImages.Length; ++i)
		{
			Image srcImage = srcImages[i];
			Image tgtImage = tgtImages[i];

			Check.Eq(srcImage.ImageId, tgtImage.ImageId);

			Camera srcCamera = srcImage.CameraPtr;
			Camera tgtCamera = tgtImage.CameraPtr;

			Matrix3x4d srcCamFromWorld = srcImage.CamFromWorld().ToMatrix();
			Matrix3x4d tgtCamFromWorld = tgtImage.CamFromWorld().ToMatrix();

			Check.Eq(srcImage.NumPoints2D, tgtImage.NumPoints2D);

			int numInliers = 0;
			int numCommonPoints = 0;

			for (uint point2DIdx = 0; point2DIdx < srcImage.NumPoints2D; ++point2DIdx)
			{
				// Check if both images have a 3D point.

				Point2D srcPoint2D = srcImage.Point2DAt(point2DIdx);
				if (!srcPoint2D.HasPoint3D)
				{
					continue;
				}

				Point2D tgtPoint2D = tgtImage.Point2DAt(point2DIdx);
				if (!tgtPoint2D.HasPoint3D)
				{
					continue;
				}

				numCommonPoints += 1;

				Vector3d srcPointInTgt = tgtFromSrc * _srcReconstruction.Point3D(srcPoint2D.Point3DId).Xyz;
				if (Projection.CalculateSquaredReprojectionError(tgtPoint2D.Xy, srcPointInTgt, tgtCamFromWorld, tgtCamera) > _maxSquaredReprojError)
				{
					continue;
				}

				Vector3d tgtPointInSrc = srcFromTgt * _tgtReconstruction.Point3D(tgtPoint2D.Point3DId).Xyz;
				if (Projection.CalculateSquaredReprojectionError(srcPoint2D.Xy, tgtPointInSrc, srcCamFromWorld, srcCamera) > _maxSquaredReprojError)
				{
					continue;
				}

				numInliers += 1;
			}

			if (numCommonPoints == 0)
			{
				residuals[i] = 1.0;
			}
			else
			{
				double negativeInlierRatio = 1.0 - (numInliers / (double)numCommonPoints);
				residuals[i] = negativeInlierRatio * negativeInlierRatio;
			}
		}
	}
}
