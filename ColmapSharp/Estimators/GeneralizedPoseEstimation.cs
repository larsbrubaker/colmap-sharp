// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedPoseEstimation: colmap/estimators/generalized_pose.h/.cc - pose estimation for
// multi-camera rigs (generalized cameras): absolute pose by GP3P RANSAC, relative pose by
// GR6P/GR8P LO-RANSAC (or, for two panoramic rigs, the central EstimateRelativePose of
// PoseEstimation.cs), structure-less absolute pose (Zheng and Wu 2013), and the nonlinear
// refinement of a rig's absolute pose. The single-camera versions and the option classes are
// in PoseEstimation.cs. Tests: ColmapSharp.Tests/Estimators/GeneralizedPoseEstimationTests.cs
// (generalized_pose_test.cc 1:1).
//
// Translation notes:
// - std::optional<Rigid3d>* outputs are ref Rigid3d? (the function only assigns on success,
//   as in C++). The cameras are reference types, updated in place like std::vector<Camera>*.
// - IsPanoramicRig compares every rig camera's origin to that of the first element of a
//   FlatHashSet of the camera indices; the set's order is unspecified, so the smallest index
//   is taken as the first here (divergence 46).

using System.Runtime.CompilerServices;

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

using static ColmapSharp.Estimators.CostFunctions.ManifoldHelpers;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::StructureLessAbsolutePoseEstimationOptions.</summary>
public sealed class StructureLessAbsolutePoseEstimationOptions
{
	/// <summary>Creates the options with COLMAP's defaults.</summary>
	public StructureLessAbsolutePoseEstimationOptions()
	{
		RansacOptions.MaxError = 6.0;
		// Use high confidence to avoid preemptive termination o RANSAC
		// - too early termination may lead to bad registration.
		RansacOptions.MinNumTrials = 100;
		RansacOptions.MaxNumTrials = 10000;
		RansacOptions.Confidence = 0.99999;
	}

	/// <summary>Options used for RANSAC.</summary>
	public RansacOptions RansacOptions = new();

	/// <summary>Checks the options.</summary>
	public void Check() => RansacOptions.Check();
}

/// <summary>Port of colmap/estimators/generalized_pose.h (the free functions).</summary>
public static class GeneralizedPoseEstimation
{
	/// <summary>
	/// Estimate generalized absolute pose from 2D-3D correspondences; camera_idxs gives the
	/// rig camera of each correspondence. Returns whether the pose is estimated successfully.
	/// Port of colmap::EstimateGeneralizedAbsolutePose.
	/// </summary>
	public static bool EstimateGeneralizedAbsolutePose(
		RansacOptions options,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		IReadOnlyList<int> cameraIdxs,
		IReadOnlyList<Rigid3d> camsFromRig,
		IReadOnlyList<Camera> cameras,
		ref Rigid3d rigFromWorld,
		out int numInliers,
		out bool[] inlierMask)
	{
		Check.Eq(points2D.Count, points3D.Count);
		Check.Eq(points2D.Count, cameraIdxs.Count);
		ThrowCheckCameras(cameraIdxs, camsFromRig, cameras);
		options.Check();
		numInliers = 0;
		inlierMask = [];
		if (points2D.Count == 0)
		{
			return false;
		}

		// Precompute cam_from_rig matrices for fast residual computation
		var camsFromRigMatrices = new Matrix3x4d[camsFromRig.Count];
		for (int i = 0; i < camsFromRig.Count; i++)
		{
			camsFromRigMatrices[i] = camsFromRig[i].ToMatrix();
		}

		var rigPoints2D = new GP3PObservation[points2D.Count];
		for (int i = 0; i < points2D.Count; i++)
		{
			int cameraIdx = cameraIdxs[i];
			rigPoints2D[i] = new GP3PObservation(
				camsFromRigMatrices[cameraIdx], cameras[cameraIdx].CamRayFromImg(points2D[i]) ?? Vector3d.Zero);
		}

		// Associate unique ids to each 3D point.
		// Needed for UniqueInlierSupportMeasurer to avoid counting the same
		// 3D point multiple times due to FoV overlap in rig.
		ulong[] uniquePoint3DIds = ComputeUniquePointIds(points3D);

		// Average of the errors over the cameras, weighted by the number of
		// correspondences
		RansacOptions optionsCopy = options;
		optionsCopy.MaxError = ComputeMaxErrorInCamera(cameraIdxs, cameras, options.MaxError);

		var ransac = new Ransac<GP3PEstimator, GP3PObservation, Vector3d, Rigid3d, UniqueInlierSupportMeasurer,
			UniqueInlierSupportMeasurer.Support, RandomSampler>(
			optionsCopy,
			new GP3PEstimator(GP3PEstimator.ResidualType.ReprojectionError),
			new UniqueInlierSupportMeasurer(uniquePoint3DIds));
		RansacReport<Rigid3d, UniqueInlierSupportMeasurer.Support> report = ransac.Estimate(rigPoints2D, points3D.ToArray());
		if (!report.Success)
		{
			return false;
		}

		rigFromWorld = report.Model;
		numInliers = report.Support.NumUniqueInliers;
		inlierMask = report.InlierMask;

		return true;
	}

	/// <summary>
	/// Estimate generalized relative pose from 2D-2D correspondences. Sets
	/// <paramref name="rig2FromRig1"/> if at least one of the rigs is non-panoramic, else
	/// <paramref name="pano2FromPano1"/> (translation up to scale). Returns whether the pose
	/// is estimated successfully. Port of colmap::EstimateGeneralizedRelativePose.
	/// </summary>
	public static bool EstimateGeneralizedRelativePose(
		RansacOptions ransacOptions,
		IReadOnlyList<Vector2d> points2D1,
		IReadOnlyList<Vector2d> points2D2,
		IReadOnlyList<int> cameraIdxs1,
		IReadOnlyList<int> cameraIdxs2,
		IReadOnlyList<Rigid3d> camsFromRig,
		IReadOnlyList<Camera> cameras,
		ref Rigid3d? rig2FromRig1,
		ref Rigid3d? pano2FromPano1,
		out int numInliers,
		out bool[] inlierMask)
	{
		Check.Eq(points2D1.Count, points2D2.Count);
		ThrowCheckCameras(cameraIdxs1, camsFromRig, cameras);
		ThrowCheckCameras(cameraIdxs2, camsFromRig, cameras);
		ransacOptions.Check();
		numInliers = 0;
		inlierMask = [];

		int numPoints = points2D1.Count;
		if (numPoints == 0)
		{
			return false;
		}

		// Both branches below score with the pixel-unit tangent Sampson error, so the
		// RANSAC threshold is the plain pixel ransac_options throughout. No
		// per-camera conversion to normalized/angular units is needed.
		if (IsPanoramicRig(cameraIdxs1, camsFromRig) && IsPanoramicRig(cameraIdxs2, camsFromRig))
		{
			// EstimateRelativePose treats the panoramic rig as one central camera, so
			// each ray carries its unprojection Jacobian, rotated into the rig frame by
			// the same rotation as the ray. Unprojectable points are zeroed, which the
			// tangent Sampson residual reports as infinite (rejected).
			var camRays1WithJac = new CamRayWithJac[numPoints];
			var camRays2WithJac = new CamRayWithJac[numPoints];
			for (int i = 0; i < numPoints; ++i)
			{
				camRays1WithJac[i] = RayInRig(cameras[cameraIdxs1[i]], camsFromRig[cameraIdxs1[i]], points2D1[i]);
				camRays2WithJac[i] = RayInRig(cameras[cameraIdxs2[i]], camsFromRig[cameraIdxs2[i]], points2D2[i]);
			}

			var cam2FromCam1 = new Rigid3d();
			if (PoseEstimation.EstimateRelativePose(
				ransacOptions, camRays1WithJac, camRays2WithJac, ref cam2FromCam1, out numInliers, out inlierMask))
			{
				pano2FromPano1 = cam2FromCam1;
				return true;
			}

			return false;
		}

		var points1 = new GrnpObservation[numPoints];
		var points2 = new GrnpObservation[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			points1[i] = new GrnpObservation(
				camsFromRig[cameraIdxs1[i]], cameras[cameraIdxs1[i]].CamRayFromImgWithJac(points2D1[i]) ?? CamRayWithJac.Zero);
			points2[i] = new GrnpObservation(
				camsFromRig[cameraIdxs2[i]], cameras[cameraIdxs2[i]].CamRayFromImgWithJac(points2D2[i]) ?? CamRayWithJac.Zero);
		}

		var ransac = new LoRansac<GR6PEstimator, GR8PEstimator, GrnpObservation, GrnpObservation, Rigid3d>(
			ransacOptions, default, default);
		RansacReport<Rigid3d, InlierSupportMeasurer.Support> report = ransac.Estimate(points1, points2);
		if (!report.Success)
		{
			return false;
		}

		rig2FromRig1 = report.Model;
		numInliers = report.Support.NumInliers;
		inlierMask = report.InlierMask;

		return true;
	}

	/// <summary>
	/// Refine generalized absolute pose (optionally focal lengths) from 2D-3D
	/// correspondences. Returns whether the solution is usable.
	/// Port of colmap::RefineGeneralizedAbsolutePose.
	/// </summary>
	public static bool RefineGeneralizedAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		IReadOnlyList<int> cameraIdxs,
		IReadOnlyList<Rigid3d> camsFromRig,
		ref Rigid3d rigFromWorld,
		IReadOnlyList<Camera> cameras) =>
		RefineGeneralizedAbsolutePose(
			options, inlierMask, points2D, points3D, cameraIdxs, camsFromRig, ref rigFromWorld, cameras, null);

	/// <summary>
	/// RefineGeneralizedAbsolutePose that also estimates the 6x6 covariance of the rotation
	/// (as axis-angle, in tangent space) and translation terms of rig_from_world.
	/// </summary>
	public static bool RefineGeneralizedAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		IReadOnlyList<int> cameraIdxs,
		IReadOnlyList<Rigid3d> camsFromRig,
		ref Rigid3d rigFromWorld,
		IReadOnlyList<Camera> cameras,
		out Matrix6d rigFromWorldCov)
	{
		var covariance = new StrongBox<Matrix6d>();
		bool result = RefineGeneralizedAbsolutePose(
			options, inlierMask, points2D, points3D, cameraIdxs, camsFromRig, ref rigFromWorld, cameras, covariance);
		rigFromWorldCov = covariance.Value;
		return result;
	}

	private static bool RefineGeneralizedAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		IReadOnlyList<int> cameraIdxs,
		IReadOnlyList<Rigid3d> camsFromRig,
		ref Rigid3d rigFromWorld,
		IReadOnlyList<Camera> cameras,
		StrongBox<Matrix6d>? rigFromWorldCov)
	{
		Check.Eq(points2D.Count, inlierMask.Count);
		Check.Eq(points2D.Count, points3D.Count);
		Check.Eq(points2D.Count, cameraIdxs.Count);
		Check.Eq(camsFromRig.Count, cameras.Count);
		// COLMAP dereferences std::min_element / max_element here, which is undefined for an
		// empty list; fail with a clear message instead.
		Check.That(cameraIdxs.Count > 0, "RefineGeneralizedAbsolutePose needs at least one correspondence (camera_idxs is empty).");
		Check.Ge(cameraIdxs.Min(), 0);
		Check.Lt(cameraIdxs.Max(), cameras.Count);
		options.Check();

		var lossFunction = new CauchyLoss(options.LossFunctionScale);

		var cameraCounts = new int[cameras.Count];

		// Cost function assumes unit quaternion.
		var rigFromWorldStorage = new Rigid3dStorage(
			new Rigid3d(rigFromWorld.Rotation.Normalized(), rigFromWorld.Translation));
		double[] rigFromWorldParams = rigFromWorldStorage.Params;

		var point3DParams = new double[points3D.Count][];
		var camFromRigParams = new double[camsFromRig.Count][];
		for (int i = 0; i < camsFromRig.Count; ++i)
		{
			camFromRigParams[i] = new Rigid3dStorage(camsFromRig[i]).Params;
		}

		var problem = new Problem();

		for (int i = 0; i < points2D.Count; ++i)
		{
			// Skip outlier observations
			if (!inlierMask[i])
			{
				continue;
			}

			int cameraIdx = cameraIdxs[i];
			cameraCounts[cameraIdx] += 1;

			point3DParams[i] = [points3D[i].X, points3D[i].Y, points3D[i].Z];
			problem.AddResidualBlock(
				CameraCostFunctions.CreateRigReprojErrorCostFunction(cameras[cameraIdx].ModelId, points2D[i]),
				lossFunction,
				point3DParams[i],
				camFromRigParams[cameraIdx],
				rigFromWorldParams,
				cameras[cameraIdx].Params);
			problem.SetParameterBlockConstant(point3DParams[i]);
		}

		if (options.UsePositionPrior)
		{
			problem.AddResidualBlock(
				CovarianceWeightedCostFunctor.Create(
					MatrixXd.From(options.PositionPriorCovariance),
					new AbsolutePosePositionPriorCostFunctor(options.PositionPriorInWorld)),
				null,
				rigFromWorldParams);
		}

		if (problem.NumResiduals > 0)
		{
			// Camera parameterization.
			for (int i = 0; i < cameras.Count; i++)
			{
				if (cameraCounts[i] == 0)
				{
					continue;
				}

				Camera camera = cameras[i];

				// We don't optimize the rig parameters (it's likely under-constrained).
				problem.SetParameterBlockConstant(camFromRigParams[i]);

				if (!options.RefineFocalLength && !options.RefineExtraParams)
				{
					problem.SetParameterBlockConstant(camera.Params);
				}
				else
				{
					// Always set the principal point as fixed.
					var constCameraParams = new List<int>(camera.PrincipalPointIdxs.ToArray());
					if (!options.RefineFocalLength)
					{
						constCameraParams.AddRange(camera.FocalLengthIdxs.ToArray());
					}

					if (!options.RefineExtraParams)
					{
						constCameraParams.AddRange(camera.ExtraParamsIdxs.ToArray());
					}

					if (constCameraParams.Count == camera.Params.Length)
					{
						problem.SetParameterBlockConstant(camera.Params);
					}
					else
					{
						SetManifold(problem, camera.Params, CreateSubsetManifold(camera.Params.Length, constCameraParams));
					}
				}
			}

			SetManifold(
				problem,
				rigFromWorldParams,
				CreateProductManifold(CreateEigenQuaternionManifold(), CreateEuclideanManifold(3)));
		}

		var solverOptions = new SolverOptions
		{
			GradientTolerance = options.GradientTolerance,
			MaxNumIterations = options.MaxNumIterations,
			LinearSolverType = LinearSolverType.DenseQr,
			// The overhead of creating threads is too large.
			NumThreads = 1,
		};

		SolverSummary summary = LeastSquaresSolver.Solve(solverOptions, problem);
		rigFromWorld = rigFromWorldStorage.Value;

		if (problem.NumResiduals > 0 && rigFromWorldCov is not null)
		{
			var covariance = new Covariance();
			ArraySegment<double>[] parameterBlocks = [rigFromWorldParams];
			if (!covariance.Compute(parameterBlocks, problem))
			{
				return false;
			}

			rigFromWorldCov.Value = Matrix6d.FromColumnMajor(
				covariance.GetCovarianceMatrixInTangentSpace(parameterBlocks).AsSpan());
		}

		return summary.IsSolutionUsable;
	}

	/// <summary>
	/// Estimate absolute camera pose using 2D-2D correspondences. The 2D-2D correspondences
	/// are assumed to be structureless, i.e. the 3D points are not known. Based on
	/// "Structure from Motion Using Structure-less Resection", Zheng and Wu, 2013.
	/// Port of colmap::EstimateStructureLessAbsolutePose.
	/// </summary>
	public static bool EstimateStructureLessAbsolutePose(
		StructureLessAbsolutePoseEstimationOptions options,
		IReadOnlyList<Vector2d> queryPoints2D,
		IReadOnlyList<Vector2d> worldPoints2D,
		IReadOnlyList<int> worldCameraIdxs,
		IReadOnlyList<Rigid3d> worldCamsFromWorld,
		IReadOnlyList<Camera> worldCameras,
		Camera queryCamera,
		ref Rigid3d queryCamFromWorld,
		out int numInliers,
		out bool[] inlierMask)
	{
		Check.Eq(worldPoints2D.Count, queryPoints2D.Count);
		Check.Eq(worldPoints2D.Count, worldCameraIdxs.Count);
		Check.Eq(worldCamsFromWorld.Count, worldCameras.Count);
		ThrowCheckCameras(worldCameraIdxs, worldCamsFromWorld, worldCameras);
		options.Check();
		numInliers = 0;
		inlierMask = [];

		// Not in COLMAP, where IsPanoramicRig would dereference begin() of an empty set
		// (undefined); return false like the other estimators do for no correspondences.
		if (worldPoints2D.Count == 0)
		{
			return false;
		}

		if (IsPanoramicRig(worldCameraIdxs, worldCamsFromWorld))
		{
			return false;
		}

		int numPoints = worldPoints2D.Count;
		var worldObs = new GrnpObservation[numPoints];
		var queryObs = new GrnpObservation[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			int worldCameraIdx = worldCameraIdxs[i];
			worldObs[i] = new GrnpObservation(
				worldCamsFromWorld[worldCameraIdx],
				worldCameras[worldCameraIdx].CamRayFromImgWithJac(worldPoints2D[i]) ?? CamRayWithJac.Zero);
			queryObs[i] = new GrnpObservation(
				new Rigid3d(), queryCamera.CamRayFromImgWithJac(queryPoints2D[i]) ?? CamRayWithJac.Zero);
		}

		// GR6P/GR8P score with the pixel-unit tangent Sampson error, so the RANSAC
		// threshold is the plain pixel max_error. No per-camera conversion needed.
		var ransac = new LoRansac<GR6PEstimator, GR8PEstimator, GrnpObservation, GrnpObservation, Rigid3d>(
			options.RansacOptions, default, default);
		RansacReport<Rigid3d, InlierSupportMeasurer.Support> report = ransac.Estimate(worldObs, queryObs);
		if (!report.Success)
		{
			return false;
		}

		queryCamFromWorld = report.Model;
		numInliers = report.Support.NumInliers;
		inlierMask = report.InlierMask;

		return true;
	}

	private static void ThrowCheckCameras(IReadOnlyList<int> cameraIdxs, IReadOnlyList<Rigid3d> camsFromRig, IReadOnlyList<Camera> cameras)
	{
		Check.That(cameras.Count > 0, "!cameras.empty()");
		Check.Eq(camsFromRig.Count, cameras.Count);
		// std::minmax_element on an empty range dereferences end() in C++ (undefined); only
		// non-empty index lists are checked here.
		if (cameraIdxs.Count > 0)
		{
			Check.Ge(cameraIdxs.Min(), 0);
			Check.Lt(cameraIdxs.Max(), cameras.Count);
		}
	}

	private static bool IsPanoramicRig(IReadOnlyList<int> cameraIdxs, IReadOnlyList<Rigid3d> camsFromRig)
	{
		int[] cameraIdxSet = cameraIdxs.Distinct().Order().ToArray();
		Vector3d firstOriginInRig = camsFromRig[cameraIdxSet[0]].TgtOriginInSrc();
		for (int k = 1; k < cameraIdxSet.Length; ++k)
		{
			Vector3d otherOriginInRig = camsFromRig[cameraIdxSet[k]].TgtOriginInSrc();
			if (!firstOriginInRig.IsApprox(otherOriginInRig, 1e-6))
			{
				return false;
			}
		}

		return true;
	}

	private static double ComputeMaxErrorInCamera(IReadOnlyList<int> cameraIdxs, IReadOnlyList<Camera> cameras, double maxErrorPx)
	{
		Check.Gt(maxErrorPx, 0.0);
		double maxErrorCam = 0.0;
		foreach (int cameraIdx in cameraIdxs)
		{
			maxErrorCam += cameras[cameraIdx].CamFromImgThreshold(maxErrorPx);
		}

		return maxErrorCam / cameraIdxs.Count;
	}

	// Lexicographic x, y, z order.
	private static bool LowerVector3d(Vector3d v1, Vector3d v2)
	{
		if (v1.X < v2.X)
		{
			return true;
		}
		else if (v1.X == v2.X)
		{
			if (v1.Y < v2.Y)
			{
				return true;
			}
			else if (v1.Y == v2.Y)
			{
				return v1.Z < v2.Z;
			}
		}

		return false;
	}

	private static ulong[] ComputeUniquePointIds(IReadOnlyList<Vector3d> points3D)
	{
		// std::sort is unstable; exactly equal points land in one group either way, and a
		// group's id is its first sorted position, so the index tie-break changes nothing.
		int[] point3DIds = Enumerable.Range(0, points3D.Count).ToArray();
		Array.Sort(point3DIds, (i, j) =>
			LowerVector3d(points3D[i], points3D[j]) ? -1 : LowerVector3d(points3D[j], points3D[i]) ? 1 : i.CompareTo(j));

		int uniqueIt = 0;
		var uniquePoint3DIds = new ulong[points3D.Count];
		for (int currentIt = 0; currentIt < point3DIds.Length; currentIt++)
		{
			if (!points3D[point3DIds[uniqueIt]].IsApprox(points3D[point3DIds[currentIt]], 1e-5))
			{
				uniqueIt = currentIt;
			}

			uniquePoint3DIds[point3DIds[currentIt]] = (ulong)uniqueIt;
		}

		return uniquePoint3DIds;
	}

	// A ray and its unprojection Jacobian rotated from the camera into the rig frame, or the
	// zero sentinel if the point does not unproject.
	private static CamRayWithJac RayInRig(Camera camera, Rigid3d camFromRig, Vector2d point2D)
	{
		Matrix3d rigFromCam = camFromRig.Rotation.Inverse().ToRotationMatrix();
		CamRayWithJac? rj = camera.CamRayFromImgWithJac(point2D);
		return rj is { } value
			? new CamRayWithJac(rigFromCam * value.Ray, rigFromCam * value.Jacobian)
			: CamRayWithJac.Zero;
	}
}
