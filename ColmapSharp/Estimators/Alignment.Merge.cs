// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Alignment.Merge: MergeReconstructions (with its CopyRegisteredImage helper) and
// AlignReconstructionToOrigRigScales of colmap/estimators/alignment.cc. The alignment they
// start from is in Alignment.cs. Tests: ColmapSharp.Tests/Estimators/AlignmentTests.cs
// (alignment_test.cc: MergeReconstructions, MergeReconstructionsInconsistentImageNames,
// AlignReconstructionToOrigRigScales).
//
// Deterministic order (docs/CPP_DIVERGENCES.md, entry 49): COLMAP copies the missing images
// in the iteration order of a FlatHashSet, which decides the order their frames are
// registered in the target (RegFrameIds). The port copies them in the source's
// RegImageIds order. The source points are merged in Points3D order (ascending id, entry
// 21), which decides the ids the new target points get.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

public static partial class Alignment
{
	/// <summary>
	/// Aligns the source to the target reconstruction and merges cameras, images, points3D
	/// into the target using the alignment. Returns false on failure, in which case the
	/// target is unchanged. Port of colmap::MergeReconstructions.
	/// </summary>
	public static bool MergeReconstructions(double maxReprojError, Reconstruction srcReconstruction, Reconstruction tgtReconstruction)
	{
		var tgtFromSrc = new Sim3d();
		if (!AlignReconstructionsViaReprojections(
			srcReconstruction,
			tgtReconstruction,
			minInlierObservations: 0.3,
			maxReprojError,
			ref tgtFromSrc))
		{
			return false;
		}

		// Find common and missing images in the two reconstructions. Images are matched by
		// image id, which assumes that both reconstructions share a consistent
		// image_id<->name mapping (i.e. were derived from the same database). If this
		// assumption is violated -- e.g. the reconstructions were built from independent
		// databases that both number their images 1..N -- then distinct physical images end
		// up with colliding ids. Detect the inconsistency via the image name and fail loudly
		// instead.
		var commonImageIds = new HashSet<uint>();
		var missingImageIds = new HashSet<uint>();
		var missingImageIdsInOrder = new List<uint>();
		foreach (uint imageId in srcReconstruction.RegImageIds())
		{
			if (tgtReconstruction.ExistsImage(imageId))
			{
				string srcName = srcReconstruction.Image(imageId).Name;
				string tgtName = tgtReconstruction.Image(imageId).Name;
				if (srcName != tgtName)
				{
					Log.Error(
						$"Cannot merge reconstructions: image_id={imageId} refers to \"{srcName}\" in the source reconstruction "
						+ $"but \"{tgtName}\" in the target. MergeReconstructions requires both reconstructions to share a "
						+ "consistent image_id<->name mapping (i.e., be derived from the same database).");
					return false;
				}

				commonImageIds.Add(imageId);
			}
			else if (missingImageIds.Add(imageId))
			{
				missingImageIdsInOrder.Add(imageId);
			}
		}

		// Register the missing images in this src_reconstruction.
		foreach (uint imageId in missingImageIdsInOrder)
		{
			CopyRegisteredImage(imageId, tgtFromSrc, srcReconstruction, tgtReconstruction);
		}

		// Merge the two point clouds using the following two rules:
		//    - copy points to this src_reconstruction with non-conflicting tracks,
		//      i.e. points that do not have an already triangulated observation
		//      in this src_reconstruction.
		//    - merge tracks that are unambiguous, i.e. only merge points in the two
		//      reconstructions if they have a one-to-one mapping.
		// Note that in both cases no cheirality or reprojection test is performed.

		foreach (Point3D point3D in srcReconstruction.Points3D.Values)
		{
			var newTrack = new Track();
			var oldTrack = new Track();
			var oldPoint3DIds = new HashSet<ulong>();
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				if (commonImageIds.Contains(trackEl.ImageId))
				{
					Point2D point2D = tgtReconstruction.Image(trackEl.ImageId).Point2DAt(trackEl.Point2DIdx);
					if (point2D.HasPoint3D)
					{
						oldTrack.AddElement(trackEl);
						oldPoint3DIds.Add(point2D.Point3DId);
					}
					else
					{
						newTrack.AddElement(trackEl);
					}
				}
				else if (missingImageIds.Contains(trackEl.ImageId))
				{
					tgtReconstruction.Image(trackEl.ImageId).ResetPoint3DForPoint2D(trackEl.Point2DIdx);
					newTrack.AddElement(trackEl);
				}
			}

			bool createNewPoint = newTrack.Length >= 2;
			bool mergeNewAndOldPoint = (newTrack.Length + oldTrack.Length) >= 2 && oldPoint3DIds.Count == 1;
			if (createNewPoint || mergeNewAndOldPoint)
			{
				Vector3d xyz = tgtFromSrc * point3D.Xyz;
				ulong point3DId = tgtReconstruction.AddPoint3D(xyz, newTrack, point3D.Color);
				if (oldPoint3DIds.Count == 1)
				{
					tgtReconstruction.MergePoints3D(point3DId, oldPoint3DIds.First());
				}
			}
		}

		return true;
	}

	/// <summary>
	/// Align reconstruction to the original metric scales in rig extrinsics. Returns false
	/// if there is no available non-panoramic rig in the alignment process. Port of
	/// colmap::AlignReconstructionToOrigRigScales.
	/// </summary>
	public static bool AlignReconstructionToOrigRigScales(IReadOnlyDictionary<uint, Rig> origRigs, Reconstruction reconstruction)
	{
		double scaleSum = 0;
		int scaleCount = 0;
		foreach (var (rigId, origRig) in origRigs)
		{
			double scaleSumRig = 0;
			int scaleCountRig = 0;
			foreach (var (sensorId, sensorFromOrigRig) in origRig.NonRefSensors)
			{
				if (sensorFromOrigRig is not Rigid3d sensorFromOrigRigValue)
				{
					continue;
				}

				// Here we do not include rigs that are panoramic.
				double sensorFromOrigRigNorm = sensorFromOrigRigValue.Translation.Norm;
				if (sensorFromOrigRigNorm < 1e-6)
				{
					continue;
				}

				Check.That(reconstruction.Rig(rigId).HasSensorFromRig(sensorId));
				double scale = reconstruction.Rig(rigId).SensorFromRig(sensorId).Translation.Norm / sensorFromOrigRigNorm;
				scaleSumRig += scale;
				++scaleCountRig;
			}

			if (scaleCountRig > 0)
			{
				scaleSum += scaleSumRig / scaleCountRig;
				++scaleCount;
			}
		}

		if (scaleCount == 0)
		{
			return false;
		}

		var newFromOldWorld = new Sim3d { Scale = scaleCount / scaleSum };
		reconstruction.Transform(newFromOldWorld);
		return true;
	}

	private static void CopyRegisteredImage(uint imageId, Sim3d tgtFromSrc, Reconstruction srcReconstruction, Reconstruction tgtReconstruction)
	{
		Image srcImage = srcReconstruction.Image(imageId);
		if (!tgtReconstruction.ExistsCamera(srcImage.CameraId))
		{
			tgtReconstruction.AddCamera(srcReconstruction.Camera(srcImage.CameraId));
		}

		if (!tgtReconstruction.ExistsRig(srcImage.FramePtr.RigId))
		{
			tgtReconstruction.AddRig(srcReconstruction.Rig(srcImage.FramePtr.RigId));
		}

		if (!tgtReconstruction.ExistsFrame(srcImage.FrameId))
		{
			Frame tgtFrame = srcReconstruction.Frame(srcImage.FrameId).Clone();
			tgtFrame.ResetRigPtr();
			tgtReconstruction.AddFrame(tgtFrame);
			Rigid3d camFromTgtWorld = Pose.TransformCameraWorld(tgtFromSrc, srcImage.CamFromWorld());
			tgtReconstruction.Frame(srcImage.FrameId).SetCamFromWorld(srcImage.CameraId, camFromTgtWorld);
		}

		Image tgtImage = srcImage.Clone();
		tgtImage.ResetCameraPtr();
		tgtImage.ResetFramePtr();
		tgtReconstruction.AddImage(tgtImage);
	}
}
