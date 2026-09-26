// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedPoseEstimationTests: colmap/estimators/generalized_pose_test.cc ported 1:1.
// TEST(Suite, Name) becomes Suite_Name; same synthetic rigs, outlier injection and
// tolerances. Tests ColmapSharp/Estimators/GeneralizedPoseEstimation.cs. Tier C: RANSAC and
// the refinement are judged by outcome at COLMAP's tolerances.
//
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does, and draws everything before
// its first await (the PRNG is per thread). std::shuffle is libc++'s (LibcxxRandom.Shuffle).
// The C++ fixtures group observations in FlatHashMaps; here Dictionary insertion order is
// used, which only reorders the correspondences.

using ColmapSharp.Estimators;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Assertions.Enums;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public class GeneralizedPoseEstimationTests
{
	private sealed class GeneralizedAbsolutePoseProblem
	{
		public Rigid3d GtRigFromWorld { get; set; }

		public List<Vector2d> Points2D { get; } = [];

		public List<Vector3d> Points3D { get; } = [];

		public List<ulong> Point3DIds { get; } = [];

		public List<int> CameraIdxs { get; } = [];

		public List<Rigid3d> CamsFromRig { get; } = [];

		public List<Camera> Cameras { get; } = [];
	}

	private static GeneralizedAbsolutePoseProblem BuildGeneralizedAbsolutePoseProblem()
	{
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 2,
			NumFramesPerRig = 1,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(options, reconstruction);

		var problem = new GeneralizedAbsolutePoseProblem();
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		problem.GtRigFromWorld = new Rigid3d(rotation, RandomEigen.RandomEigenVector3d());
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			foreach (Point2D point2D in image.Points2D)
			{
				if (point2D.HasPoint3D)
				{
					problem.Points2D.Add(point2D.Xy);
					problem.Points3D.Add(reconstruction.Point3D(point2D.Point3DId).Xyz);
					problem.Point3DIds.Add(point2D.Point3DId);
					problem.CameraIdxs.Add(problem.Cameras.Count);
				}
			}

			problem.Cameras.Add(image.CameraPtr.Clone());
			problem.CamsFromRig.Add(image.CamFromWorld() * problem.GtRigFromWorld.Inverse());
		}

		return problem;
	}

	private static Rigid3d RandomRigFromGtRig()
	{
		const double RotationNoiseDegree = 1;
		const double TranslationNoise = 0.1;
		Vector3d axis = RandomEigen.RandomEigenVector3d().Normalized();
		var rotation = Quaterniond.FromAngleAxis(new AngleAxisd(MathUtils.DegToRad(RotationNoiseDegree), axis));
		return new Rigid3d(rotation, RandomEigen.RandomEigenVector3d() * TranslationNoise);
	}

	private static List<Camera> CloneCameras(List<Camera> cameras) => cameras.Select(c => c.Clone()).ToList();

	[Test]
	public async Task EstimateGeneralizedAbsolutePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		GeneralizedAbsolutePoseProblem problem = BuildGeneralizedAbsolutePoseProblem();
		int numPoints = problem.Points2D.Count;

		const double GtInlierRatio = 0.8;
		const double OutlierDistance = 50;
		int gtNumInliers = Math.Max((int)(GtInlierRatio * numPoints), GP3PEstimator.MinNumSamples);
		var shuffledIdxs = Enumerable.Range(0, numPoints).ToList();
		LibcxxRandom.Shuffle(shuffledIdxs, RandomUtils.Prng!);

		var uniqueInlierIds = new HashSet<ulong>();
		for (int i = 0; i < gtNumInliers; ++i)
		{
			uniqueInlierIds.Add(problem.Point3DIds[shuffledIdxs[i]]);
		}

		bool[] gtInlierMask = Enumerable.Repeat(true, numPoints).ToArray();
		for (int i = gtNumInliers; i < numPoints; ++i)
		{
			problem.Points2D[shuffledIdxs[i]] += RandomEigen.RandomEigenVector2d().Normalized() * OutlierDistance;
			gtInlierMask[shuffledIdxs[i]] = false;
		}

		var ransacOptions = new RansacOptions
		{
			MaxError = 2,
			MinInlierRatio = GtInlierRatio / 2,
			Confidence = 0.99999,
		};

		var rigFromWorld = new Rigid3d();
		bool success = GeneralizedPoseEstimation.EstimateGeneralizedAbsolutePose(
			ransacOptions,
			problem.Points2D,
			problem.Points3D,
			problem.CameraIdxs,
			problem.CamsFromRig,
			problem.Cameras,
			ref rigFromWorld,
			out int numInliers,
			out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(numInliers).IsEqualTo(uniqueInlierIds.Count);
		await Assert.That(inlierMask).IsEquivalentTo(gtInlierMask, CollectionOrdering.Matching);
		await Assert.That(Rigid3dNear(rigFromWorld, problem.GtRigFromWorld, 1e-6, 1e-6)).IsTrue();
	}

	[Test]
	public async Task RefineGeneralizedAbsolutePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		GeneralizedAbsolutePoseProblem problem = BuildGeneralizedAbsolutePoseProblem();
		bool[] gtInlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		Rigid3d rigFromGtRig = RandomRigFromGtRig();
		Rigid3d rigFromWorld = rigFromGtRig * problem.GtRigFromWorld;

		var options = new AbsolutePoseRefinementOptions { RefineFocalLength = false, RefineExtraParams = false };
		bool success = GeneralizedPoseEstimation.RefineGeneralizedAbsolutePose(
			options,
			gtInlierMask,
			problem.Points2D,
			problem.Points3D,
			problem.CameraIdxs,
			problem.CamsFromRig,
			ref rigFromWorld,
			problem.Cameras,
			out Matrix6d rigFromWorldCov);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(rigFromWorld, problem.GtRigFromWorld, 1e-6, 1e-6)).IsTrue();
		await Assert.That(rigFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(rigFromWorldCov != Matrix6d.Zero).IsTrue();
	}

	[Test]
	public async Task RefineGeneralizedAbsolutePose_PositionPrior()
	{
		RandomUtils.SetPRNGSeed(0);
		GeneralizedAbsolutePoseProblem problem = BuildGeneralizedAbsolutePoseProblem();
		// Isolate the position-prior-only refinement path without reprojection terms.
		bool[] inlierMask = new bool[problem.Points2D.Count];

		var options = new AbsolutePoseRefinementOptions
		{
			UsePositionPrior = true,
			PositionPriorInWorld = new Vector3d(1.0, 2.0, 3.0),
			PositionPriorCovariance = Matrix3d.Identity,
		};
		var rigFromWorld = new Rigid3d(
			Quaterniond.FromAngleAxis(new AngleAxisd(0.2, Vector3d.UnitY)), new Vector3d(0.3, -0.5, 0.7));
		double ComputePositionError(Rigid3d toCheck) =>
			(toCheck.Inverse().Translation - options.PositionPriorInWorld).Norm;
		double initialError = ComputePositionError(rigFromWorld);
		bool success = GeneralizedPoseEstimation.RefineGeneralizedAbsolutePose(
			options,
			inlierMask,
			problem.Points2D,
			problem.Points3D,
			problem.CameraIdxs,
			problem.CamsFromRig,
			ref rigFromWorld,
			problem.Cameras);

		await Assert.That(success).IsTrue();
		await Assert.That(ComputePositionError(rigFromWorld)).IsLessThan(initialError);
		await Assert.That(rigFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
	}

	[Test]
	public async Task RefineGeneralizedAbsolutePose_PositionPriorCovariance()
	{
		RandomUtils.SetPRNGSeed(0);
		GeneralizedAbsolutePoseProblem problem = BuildGeneralizedAbsolutePoseProblem();
		bool[] inlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		var weakPriorOptions = new AbsolutePoseRefinementOptions
		{
			UsePositionPrior = true,
			PositionPriorInWorld = problem.GtRigFromWorld.Inverse().Translation + new Vector3d(1.0, -0.7, 0.5),
			// Large covariance = weak prior (high uncertainty).
			PositionPriorCovariance = Matrix3d.Identity,
		};

		AbsolutePoseRefinementOptions strongPriorOptions = weakPriorOptions.Clone();
		// Small covariance = strong prior (low uncertainty).
		strongPriorOptions.PositionPriorCovariance = 0.01 * Matrix3d.Identity;

		Rigid3d rigFromGtRig = RandomRigFromGtRig();
		Rigid3d initialRigFromWorld = rigFromGtRig * problem.GtRigFromWorld;

		List<Camera> weakPriorCameras = CloneCameras(problem.Cameras);
		List<Camera> strongPriorCameras = CloneCameras(problem.Cameras);
		Rigid3d weakPriorRigFromWorld = initialRigFromWorld;
		Rigid3d strongPriorRigFromWorld = initialRigFromWorld;

		bool weakSuccess = GeneralizedPoseEstimation.RefineGeneralizedAbsolutePose(
			weakPriorOptions, inlierMask, problem.Points2D, problem.Points3D, problem.CameraIdxs,
			problem.CamsFromRig, ref weakPriorRigFromWorld, weakPriorCameras);
		bool strongSuccess = GeneralizedPoseEstimation.RefineGeneralizedAbsolutePose(
			strongPriorOptions, inlierMask, problem.Points2D, problem.Points3D, problem.CameraIdxs,
			problem.CamsFromRig, ref strongPriorRigFromWorld, strongPriorCameras);

		await Assert.That(weakSuccess).IsTrue();
		await Assert.That(weakPriorRigFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(strongSuccess).IsTrue();
		await Assert.That(strongPriorRigFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);

		double ComputePositionError(Rigid3d toCheck) =>
			(toCheck.Inverse().Translation - weakPriorOptions.PositionPriorInWorld).Norm;
		await Assert.That(ComputePositionError(strongPriorRigFromWorld)).IsLessThan(ComputePositionError(weakPriorRigFromWorld));
	}

	private sealed class GeneralizedRelativePoseProblem
	{
		public Rigid3d GtRig2FromRig1 { get; set; }

		public List<Vector2d> Points2D1 { get; } = [];

		public List<Vector2d> Points2D2 { get; } = [];

		public List<int> CameraIdxs1 { get; } = [];

		public List<int> CameraIdxs2 { get; } = [];

		public List<Rigid3d> CamsFromRig { get; } = [];

		public List<Camera> Cameras { get; } = [];
	}

	private static GeneralizedRelativePoseProblem BuildGeneralizedRelativePoseProblem(
		int numCamerasPerRig1, int numCamerasPerRig2, double sensorFromRigTranslationStddev)
	{
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = Math.Max(numCamerasPerRig1, numCamerasPerRig2),
			NumFramesPerRig = 1,
			NumPoints3D = 100,
			SensorFromRigTranslationStddev = sensorFromRigTranslationStddev,
			SensorFromRigRotationStddev = 10,
		};
		Synthetic.SynthesizeDataset(options, reconstruction);

		Frame frame1 = reconstruction.Frame(1);
		Frame frame2 = reconstruction.Frame(2);
		ColmapSharp.Util.Check.That(frame1.RigId != frame2.RigId, "frame1.RigId() != frame2.RigId()");

		var problem = new GeneralizedRelativePoseProblem
		{
			GtRig2FromRig1 = frame2.RigFromWorld() * frame1.RigFromWorld().Inverse(),
		};

		var observations2 = new Dictionary<ulong, List<(Image Image, int Point2DIdx)>>();
		foreach (DataId dataId in frame2.ImageIds())
		{
			Image image = reconstruction.Image((uint)dataId.Id);
			for (int point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				Point2D point2D = image.Points2D[point2DIdx];
				if (point2D.HasPoint3D)
				{
					if (!observations2.TryGetValue(point2D.Point3DId, out var list))
					{
						observations2[point2D.Point3DId] = list = [];
					}

					list.Add((image, point2DIdx));
				}
			}

			if (--numCamerasPerRig2 == 0)
			{
				break;
			}
		}

		var cameraIdToIdx = new Dictionary<uint, int>();
		int MaybeAddAndGetCamera(Image image)
		{
			if (!cameraIdToIdx.TryGetValue(image.CameraId, out int idx))
			{
				idx = problem.Cameras.Count;
				cameraIdToIdx[image.CameraId] = idx;
				problem.Cameras.Add(image.CameraPtr.Clone());
				Rig rig = image.FramePtr.RigPtr;
				problem.CamsFromRig.Add(
					rig.IsRefSensor(image.CameraPtr.SensorId) ? new Rigid3d() : rig.SensorFromRig(image.CameraPtr.SensorId));
			}

			return idx;
		}

		foreach (DataId dataId in frame1.ImageIds())
		{
			Image image1 = reconstruction.Image((uint)dataId.Id);
			for (int point2DIdx1 = 0; point2DIdx1 < image1.NumPoints2D; ++point2DIdx1)
			{
				Point2D point2D1 = image1.Points2D[point2DIdx1];
				if (!observations2.TryGetValue(point2D1.Point3DId, out var observations))
				{
					continue;
				}

				foreach ((Image image2, int point2DIdx2) in observations)
				{
					problem.Points2D1.Add(point2D1.Xy);
					problem.Points2D2.Add(image2.Points2D[point2DIdx2].Xy);
					problem.CameraIdxs1.Add(MaybeAddAndGetCamera(image1));
					problem.CameraIdxs2.Add(MaybeAddAndGetCamera(image2));
				}
			}

			if (--numCamerasPerRig1 == 0)
			{
				break;
			}
		}

		return problem;
	}

	[Test]
	public async Task EstimateGeneralizedRelativePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		foreach (int numCamerasPerRig1 in new[] { 1, 2, 3 })
		{
			foreach (int numCamerasPerRig2 in new[] { 1, 2, 3 })
			{
				// A meaningful inter-camera baseline is needed to recover metric scale
				// reliably in non-panoramic configurations.
				foreach (double sensorFromRigTranslationStddev in new[] { 0.0, 0.2 })
				{
					string what = $"rig1={numCamerasPerRig1} rig2={numCamerasPerRig2} stddev={sensorFromRigTranslationStddev}";
					GeneralizedRelativePoseProblem problem = BuildGeneralizedRelativePoseProblem(
						numCamerasPerRig1, numCamerasPerRig2, sensorFromRigTranslationStddev);

					var ransacOptions = new RansacOptions { MaxError = 1 };

					Rigid3d? rig2FromRig1 = null;
					Rigid3d? pano2FromPano1 = null;
					bool success = GeneralizedPoseEstimation.EstimateGeneralizedRelativePose(
						ransacOptions,
						problem.Points2D1,
						problem.Points2D2,
						problem.CameraIdxs1,
						problem.CameraIdxs2,
						problem.CamsFromRig,
						problem.Cameras,
						ref rig2FromRig1,
						ref pano2FromPano1,
						out int numInliers,
						out bool[] inlierMask);
					log.True(success, $"{what}: success");
					log.Equal(numInliers, problem.Points2D1.Count, $"{what}: num_inliers");
					log.True(inlierMask.All(x => x), $"{what}: all inliers");
					if ((numCamerasPerRig1 == 1 && numCamerasPerRig2 == 1) || sensorFromRigTranslationStddev == 0)
					{
						// Panoramic pairs do not allow for recovery of translation scale.
						if (!log.True(rig2FromRig1 is null, $"{what}: no rig2_from_rig1")
							|| !log.True(pano2FromPano1 is not null, $"{what}: pano2_from_pano1"))
						{
							continue;
						}

						var expected = new Rigid3d(
							problem.GtRig2FromRig1.Rotation, problem.GtRig2FromRig1.Translation.Normalized());
						log.True(Rigid3dNear(pano2FromPano1!.Value, expected, 1e-6, 1e-6), $"{what}: pano pose");
					}
					else
					{
						if (!log.True(rig2FromRig1 is not null, $"{what}: rig2_from_rig1")
							|| !log.True(pano2FromPano1 is null, $"{what}: no pano2_from_pano1"))
						{
							continue;
						}

						log.True(Rigid3dNear(rig2FromRig1!.Value, problem.GtRig2FromRig1, 1e-3, 2e-3), $"{what}: rig pose");
					}
				}
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	private sealed class StructureLessAbsolutePoseProblem
	{
		public Rigid3d GtCamFromWorld { get; set; }

		public List<Vector2d> WorldPoints2D { get; } = [];

		public List<Vector2d> QueryPoints2D { get; } = [];

		public List<int> WorldCameraIdxs { get; } = [];

		public List<Rigid3d> WorldCamsFromWorld { get; } = [];

		public List<Camera> WorldCameras { get; } = [];

		public Camera QueryCamera { get; set; } = null!;
	}

	private static StructureLessAbsolutePoseProblem BuildStructureLessAbsolutePoseProblem(int numWorldCams)
	{
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = numWorldCams + 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 100,
		};
		Synthetic.SynthesizeDataset(options, reconstruction);

		uint queryImageId = reconstruction.RegImageIds()[0];
		Image queryImage = reconstruction.Image(queryImageId);

		var problem = new StructureLessAbsolutePoseProblem
		{
			GtCamFromWorld = queryImage.CamFromWorld(),
			QueryCamera = queryImage.CameraPtr.Clone(),
		};

		// Build mapping of world cameras
		var worldImageIdToCameraIdx = new Dictionary<uint, int>();
		var worldObs = new Dictionary<ulong, List<(Image Image, int Point2DIdx)>>();

		foreach (uint worldImageId in reconstruction.RegImageIds())
		{
			if (worldImageId == queryImageId)
			{
				continue;
			}

			Image worldImage = reconstruction.Image(worldImageId);

			if (!worldImageIdToCameraIdx.ContainsKey(worldImage.ImageId))
			{
				worldImageIdToCameraIdx[worldImage.ImageId] = problem.WorldCameras.Count;
				problem.WorldCameras.Add(worldImage.CameraPtr.Clone());
				problem.WorldCamsFromWorld.Add(worldImage.CamFromWorld());
			}

			for (int point2DIdx = 0; point2DIdx < worldImage.NumPoints2D; ++point2DIdx)
			{
				Point2D point2D = worldImage.Points2D[point2DIdx];
				if (point2D.HasPoint3D)
				{
					if (!worldObs.TryGetValue(point2D.Point3DId, out var list))
					{
						worldObs[point2D.Point3DId] = list = [];
					}

					list.Add((worldImage, point2DIdx));
				}
			}
		}

		for (int point2DIdx = 0; point2DIdx < queryImage.NumPoints2D; ++point2DIdx)
		{
			Point2D queryPoint2D = queryImage.Points2D[point2DIdx];
			if (!queryPoint2D.HasPoint3D)
			{
				continue;
			}

			if (!worldObs.TryGetValue(queryPoint2D.Point3DId, out var observations))
			{
				continue;
			}

			foreach ((Image worldImage, int worldPoint2DIdx) in observations)
			{
				problem.WorldPoints2D.Add(worldImage.Points2D[worldPoint2DIdx].Xy);
				problem.QueryPoints2D.Add(queryPoint2D.Xy);
				problem.WorldCameraIdxs.Add(worldImageIdToCameraIdx[worldImage.ImageId]);
			}
		}

		return problem;
	}

	[Test]
	public async Task EstimateStructureLessAbsolutePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		StructureLessAbsolutePoseProblem problem = BuildStructureLessAbsolutePoseProblem(numWorldCams: 5);

		var options = new StructureLessAbsolutePoseEstimationOptions();
		var camFromWorld = new Rigid3d();
		bool success = GeneralizedPoseEstimation.EstimateStructureLessAbsolutePose(
			options,
			problem.QueryPoints2D,
			problem.WorldPoints2D,
			problem.WorldCameraIdxs,
			problem.WorldCamsFromWorld,
			problem.WorldCameras,
			problem.QueryCamera,
			ref camFromWorld,
			out int numInliers,
			out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(numInliers).IsEqualTo(problem.WorldPoints2D.Count);
		await Assert.That(inlierMask.Length).IsEqualTo(problem.WorldPoints2D.Count);
		await Assert.That(Rigid3dNear(camFromWorld, problem.GtCamFromWorld, 1e-6, 1e-6)).IsTrue();
	}

	[Test]
	public async Task EstimateStructureLessAbsolutePose_WithOutliers()
	{
		RandomUtils.SetPRNGSeed(0);
		StructureLessAbsolutePoseProblem problem = BuildStructureLessAbsolutePoseProblem(numWorldCams: 10);

		// Add outliers by perturbing some query observations.
		const double OutlierRatio = 0.3;
		int numOutliers = (int)(OutlierRatio * problem.QueryPoints2D.Count);
		var shuffledIdxs = Enumerable.Range(0, problem.QueryPoints2D.Count).ToList();
		LibcxxRandom.Shuffle(shuffledIdxs, RandomUtils.Prng!);
		for (int i = 0; i < numOutliers; ++i)
		{
			problem.QueryPoints2D[shuffledIdxs[i]] += new Vector2d(1000, 1000);
		}

		var options = new StructureLessAbsolutePoseEstimationOptions();
		options.RansacOptions.MaxError = 1.0;  // pixels
		var camFromWorld = new Rigid3d();
		bool success = GeneralizedPoseEstimation.EstimateStructureLessAbsolutePose(
			options,
			problem.QueryPoints2D,
			problem.WorldPoints2D,
			problem.WorldCameraIdxs,
			problem.WorldCamsFromWorld,
			problem.WorldCameras,
			problem.QueryCamera,
			ref camFromWorld,
			out int numInliers,
			out bool[] _);

		await Assert.That(success).IsTrue();
		await Assert.That((double)numInliers).IsGreaterThan(problem.WorldPoints2D.Count * (1 - OutlierRatio) * 0.9);
		await Assert.That(Rigid3dNear(camFromWorld, problem.GtCamFromWorld, 5e-3, 5e-3)).IsTrue();
	}

	[Test]
	public async Task EstimateStructureLessAbsolutePose_PanoramicWorldCameras()
	{
		RandomUtils.SetPRNGSeed(0);
		StructureLessAbsolutePoseProblem problem = BuildStructureLessAbsolutePoseProblem(numWorldCams: 1);

		var options = new StructureLessAbsolutePoseEstimationOptions();
		var camFromWorld = new Rigid3d();
		bool success = GeneralizedPoseEstimation.EstimateStructureLessAbsolutePose(
			options,
			problem.QueryPoints2D,
			problem.WorldPoints2D,
			problem.WorldCameraIdxs,
			problem.WorldCamsFromWorld,
			problem.WorldCameras,
			problem.QueryCamera,
			ref camFromWorld,
			out int _,
			out bool[] _);

		await Assert.That(success).IsFalse();
	}

	/// <summary>
	/// C#-only: with no correspondences EstimateStructureLessAbsolutePose returns false, like
	/// the other estimators here (COLMAP's version dereferences an empty set).
	/// </summary>
	[Test]
	public async Task CSharpOnly_EstimateStructureLessAbsolutePose_Empty()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100, 200, 200);
		var camFromWorld = new Rigid3d();
		bool success = GeneralizedPoseEstimation.EstimateStructureLessAbsolutePose(
			new StructureLessAbsolutePoseEstimationOptions(),
			[],
			[],
			[],
			[new Rigid3d()],
			[camera],
			camera,
			ref camFromWorld,
			out int numInliers,
			out bool[] inlierMask);

		await Assert.That(success).IsFalse();
		await Assert.That(numInliers).IsEqualTo(0);
		await Assert.That(inlierMask).IsEmpty();
	}
}
