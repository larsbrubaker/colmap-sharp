// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation.Rig: EstimateRigTwoViewGeometries of
// colmap/estimators/two_view_geometry.cc. It pools the matches of every image pair between
// two rigs into one generalized relative pose problem (Estimators/GeneralizedPoseEstimation.cs,
// EstimateGeneralizedRelativePose), then splits the inliers back into per-image-pair
// TwoViewGeometry objects whose cam2_from_cam1 and E follow from the rig-to-rig pose and the
// rigs' calibration. TwoViewGeometryEstimation.cs holds the options and shared helpers.
// Test: ColmapSharp.Tests/Estimators/TwoViewGeometryEstimationTests.Rig.cs.
//
// Tier C (outcome): the pose comes out of LO-RANSAC.
//
// Translation notes:
// - Output order is COLMAP's: the pairs with inliers in ascending (image_id1, image_id2)
//   order (std::map there, SortedDictionary here), then the pairs without inliers in input
//   order. The duplicate-pair check (a FlatHashSet there) and the camera index map (a
//   NodeHashMap) are only looked up, so their iteration order never matters.
// - Cameras are shared by reference rather than copied; EstimateGeneralizedRelativePose only
//   reads them.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

public static partial class TwoViewGeometryEstimation
{
	/// <summary>
	/// Estimate the two-view geometries for all matched images between a pair of rigs.
	/// Port of colmap::EstimateRigTwoViewGeometries.
	/// </summary>
	/// <param name="rig1">First rig.</param>
	/// <param name="rig2">Second rig.</param>
	/// <param name="images">Images in first and second rig.</param>
	/// <param name="cameras">Cameras in first and second rig.</param>
	/// <param name="matches">Feature matches between first and second rig, per image pair
	/// (first image in rig1, second in rig2).</param>
	/// <param name="options">Two-view geometry estimation options.</param>
	/// <returns>Two-view geometries for all matched images; empty when the rig pose cannot be
	/// estimated or has too few inliers.</returns>
	public static List<((uint ImageId1, uint ImageId2) ImagePair, TwoViewGeometry Geometry)> EstimateRigTwoViewGeometries(
		Rig rig1,
		Rig rig2,
		IReadOnlyDictionary<uint, Image> images,
		IReadOnlyDictionary<uint, Camera> cameras,
		IReadOnlyList<((uint ImageId1, uint ImageId2) ImagePair, List<FeatureMatch> Matches)> matches,
		TwoViewGeometryOptions options)
	{
		var points1 = new List<Vector2d>();
		var points2 = new List<Vector2d>();
		var cameraIdxs1 = new List<int>();
		var cameraIdxs2 = new List<int>();
		var camsFromRig = new List<Rigid3d>(rig1.NumSensors + rig2.NumSensors);
		var camerasVec = new List<Camera>(camsFromRig.Capacity);
		var corrs = new List<(uint ImageId1, uint Point2DIdx1, uint ImageId2, uint Point2DIdx2)>();

		var cameraIdToRigAndCameraIdx = new Dictionary<uint, (uint RigId, int CameraIdx)>();
		int MaybeAddCamera(Rig rig, Camera camera)
		{
			if (cameraIdToRigAndCameraIdx.TryGetValue(camera.CameraId, out var existing))
			{
				Check.Eq(existing.RigId, rig.RigId, "The same camera is assigned to both rigs");
				return existing.CameraIdx;
			}

			int cameraIdx = camerasVec.Count;
			cameraIdToRigAndCameraIdx.Add(camera.CameraId, (rig.RigId, cameraIdx));
			camerasVec.Add(camera);
			camsFromRig.Add(rig.IsRefSensor(camera.SensorId) ? new Rigid3d() : rig.SensorFromRig(camera.SensorId));
			return cameraIdx;
		}

		var imagePairs = new HashSet<ulong>(matches.Count);
		foreach (((uint imageId1, uint imageId2), List<FeatureMatch> pairMatches) in matches)
		{
			Check.That(imagePairs.Add(Types.ImagePairToPairId(imageId1, imageId2)), "Duplicate image pair");

			Image image1 = images[imageId1];
			Camera camera1 = cameras[image1.CameraId];
			int cameraIdx1 = MaybeAddCamera(rig1, camera1);

			Image image2 = images[imageId2];
			Camera camera2 = cameras[image2.CameraId];
			int cameraIdx2 = MaybeAddCamera(rig2, camera2);

			foreach (FeatureMatch match in pairMatches)
			{
				points1.Add(image1.Points2D[(int)match.Point2DIdx1].Xy);
				points2.Add(image2.Points2D[(int)match.Point2DIdx2].Xy);
				cameraIdxs1.Add(cameraIdx1);
				cameraIdxs2.Add(cameraIdx2);
				corrs.Add((imageId1, match.Point2DIdx1, imageId2, match.Point2DIdx2));
			}
		}

		var twoViewGeometries = new List<((uint ImageId1, uint ImageId2) ImagePair, TwoViewGeometry Geometry)>();
		if (corrs.Count == 0)
		{
			return twoViewGeometries;
		}

		Rigid3d? maybeRig2FromRig1 = null;
		Rigid3d? maybePano2FromPano1 = null;
		if (!GeneralizedPoseEstimation.EstimateGeneralizedRelativePose(
				options.RansacOptions,
				points1,
				points2,
				cameraIdxs1,
				cameraIdxs2,
				camsFromRig,
				camerasVec,
				ref maybeRig2FromRig1,
				ref maybePano2FromPano1,
				out int numInliers,
				out bool[] inlierMask)
			|| ToSizeT(numInliers) < ToSizeT(options.MinNumInliers))
		{
			return twoViewGeometries;
		}

		var inlierMatches = new SortedDictionary<(uint ImageId1, uint ImageId2), List<FeatureMatch>>();
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (!inlierMask[i])
			{
				continue;
			}

			(uint imageId1, uint point2DIdx1, uint imageId2, uint point2DIdx2) = corrs[i];
			if (!inlierMatches.TryGetValue((imageId1, imageId2), out List<FeatureMatch>? pairInliers))
			{
				pairInliers = [];
				inlierMatches.Add((imageId1, imageId2), pairInliers);
			}

			pairInliers.Add(new FeatureMatch(point2DIdx1, point2DIdx2));
		}

		TwoViewGeometry.ConfigurationType config = maybeRig2FromRig1.HasValue
			? TwoViewGeometry.ConfigurationType.CalibratedRig
			: TwoViewGeometry.ConfigurationType.Calibrated;
		Rigid3d rig2FromRig1 = maybeRig2FromRig1 ?? maybePano2FromPano1!.Value;

		Rigid3d GetCamFromRig(Rig rig, uint imageId)
		{
			var cameraId = new SensorId(SensorType.Camera, images[imageId].CameraId);
			return rig.IsRefSensor(cameraId) ? new Rigid3d() : rig.SensorFromRig(cameraId);
		}

		foreach (((uint ImageId1, uint ImageId2) imagePair, List<FeatureMatch> pairMatches) in inlierMatches)
		{
			Rigid3d cam1FromRig1 = GetCamFromRig(rig1, imagePair.ImageId1);
			Rigid3d cam2FromRig2 = GetCamFromRig(rig2, imagePair.ImageId2);
			Rigid3d cam2FromCam1 = cam2FromRig2 * rig2FromRig1 * cam1FromRig1.Inverse();

			// TODO(jsch): For panoramic rigs, we could further distinguish between
			// panoramic/planar configurations by estimating a homography matrix.
			twoViewGeometries.Add((imagePair, new TwoViewGeometry
			{
				Config = config,
				InlierMatches = pairMatches,
				Cam2FromCam1 = cam2FromCam1,
				E = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1),
			}));
		}

		// Ensure that each matched input pair has a corresponding two-view geometry, even if it
		// has no inliers.
		foreach (((uint ImageId1, uint ImageId2) imagePair, _) in matches)
		{
			if (!inlierMatches.ContainsKey(imagePair))
			{
				twoViewGeometries.Add((imagePair, new TwoViewGeometry()));
			}
		}

		return twoViewGeometries;
	}
}
