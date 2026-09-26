// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseEstimation: colmap/estimators/pose.h/.cc - robust absolute and relative pose
// estimation (RANSAC / LO-RANSAC over the minimal solvers in Estimators/Solvers/) and their
// nonlinear refinement with the Ceres replacement (Solver/, covariance in
// Solver/Covariance.cs) and the cost functions in Estimators/CostFunctions/. Tests:
// ColmapSharp.Tests/Estimators/PoseEstimationTests.cs (pose_test.cc). The multi-camera (rig)
// counterparts are in GeneralizedPoseEstimation.cs.
//
// Translation notes:
// - Output pointers become ref/out parameters; the Camera is a reference type and is updated
//   in place as in COLMAP (its Params array is the camera parameter block).
// - A pose being refined lives in a 7-value Rigid3dStorage-style array
//   [qx, qy, qz, qw, tx, ty, tz] for the duration of the solve (colmap::Rigid3d::params).
// - The optional cam_from_world_cov pointer is an overload with an out Matrix6d.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

using System.Runtime.CompilerServices;

using static ColmapSharp.Estimators.CostFunctions.ManifoldHelpers;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::AbsolutePoseEstimationOptions.</summary>
public sealed class AbsolutePoseEstimationOptions
{
	/// <summary>Creates the options with COLMAP's defaults.</summary>
	public AbsolutePoseEstimationOptions()
	{
		RansacOptions.MaxError = 12.0;
		// Use high confidence to avoid preemptive termination of P3P RANSAC
		// - too early termination may lead to bad registration.
		RansacOptions.MinNumTrials = 100;
		RansacOptions.MaxNumTrials = 10000;
		RansacOptions.Confidence = 0.99999;
	}

	/// <summary>Whether to estimate the focal length.</summary>
	public bool EstimateFocalLength { get; set; }

	/// <summary>Options used for P3P RANSAC.</summary>
	public RansacOptions RansacOptions = new();

	/// <summary>Checks the options.</summary>
	public void Check() => RansacOptions.Check();
}

/// <summary>Port of colmap::AbsolutePoseRefinementOptions.</summary>
public sealed class AbsolutePoseRefinementOptions
{
	/// <summary>Convergence criterion.</summary>
	public double GradientTolerance { get; set; } = 1.0;

	/// <summary>Maximum number of solver iterations.</summary>
	public int MaxNumIterations { get; set; } = 100;

	/// <summary>Scaling factor determines at which residual robustification takes place.</summary>
	public double LossFunctionScale { get; set; } = 1.0;

	/// <summary>Whether to refine the focal length parameter group.</summary>
	public bool RefineFocalLength { get; set; }

	/// <summary>Whether to refine the extra parameter group.</summary>
	public bool RefineExtraParams { get; set; }

	/// <summary>Whether to print final summary (log-only in COLMAP; unused here).</summary>
	public bool PrintSummary { get; set; }

	/// <summary>Whether to add a soft position prior on the camera center in world coordinates.</summary>
	public bool UsePositionPrior { get; set; }

	/// <summary>Prior on camera/rig center in world coordinates.</summary>
	public Vector3d PositionPriorInWorld { get; set; } = Vector3d.Zero;

	/// <summary>
	/// Covariance of the position prior in world coordinates (3x3, SPD). Smaller values
	/// indicate higher confidence in the prior position. Defaults to identity (isotropic,
	/// sigma = 1m).
	/// </summary>
	public Matrix3d PositionPriorCovariance { get; set; } = Matrix3d.Identity;

	/// <summary>A copy of these options.</summary>
	public AbsolutePoseRefinementOptions Clone() => (AbsolutePoseRefinementOptions)MemberwiseClone();

	/// <summary>Checks the options.</summary>
	public void Check()
	{
		Util.Check.Ge(GradientTolerance, 0.0);
		Util.Check.Ge(MaxNumIterations, 0);
		Util.Check.Ge(LossFunctionScale, 0.0);
	}
}

/// <summary>Port of colmap/estimators/pose.h (the free functions).</summary>
public static class PoseEstimation
{
	/// <summary>
	/// Estimate absolute pose (optionally focal length) from 2D-3D correspondences.
	/// Focal length estimation is performed using discrete sampling around the focal length
	/// of the given camera. The focal length that results in the maximal number of inliers
	/// is assigned to the given camera. Returns whether the pose is estimated successfully.
	/// Port of colmap::EstimateAbsolutePose.
	/// </summary>
	public static bool EstimateAbsolutePose(
		AbsolutePoseEstimationOptions options,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		ref Rigid3d camFromWorld,
		Camera camera,
		out int numInliers,
		out bool[] inlierMask)
	{
		Check.Eq(points2D.Count, points3D.Count);
		options.Check();

		numInliers = 0;
		inlierMask = [];

		var points3DArray = points3D.ToArray();
		if (options.EstimateFocalLength)
		{
			// TODO(jsch): Implement non-minimal solver for LORANSAC refinement.
			// Experiments showed marginal difference between RANSAC/LORANSAC for PNPF
			// after refining the estimates of this function using RefineAbsolutePose.
			var principalPoint = new Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY());
			var points2DCentered = new Vector2d[points2D.Count];
			for (int i = 0; i < points2D.Count; ++i)
			{
				points2DCentered[i] = points2D[i] - principalPoint;
			}

			int[] focalLengthIdxs = camera.FocalLengthIdxs.ToArray();
			var ransac = new Ransac<P4PFEstimator, Vector2d, Vector3d, P4PFModel>(
				options.RansacOptions, new P4PFEstimator(shareFocalLength: focalLengthIdxs.Length == 1));
			RansacReport<P4PFModel, InlierSupportMeasurer.Support> report = ransac.Estimate(points2DCentered, points3DArray);
			if (report.Success)
			{
				camFromWorld = new Rigid3d(
					Quaterniond.FromRotationMatrix(report.Model.CamFromWorld.LeftCols3()), report.Model.CamFromWorld.Col(3));
				for (int k = 0; k < focalLengthIdxs.Length; ++k)
				{
					camera.Params[focalLengthIdxs[k]] = k == 0 ? report.Model.FocalLengths.X : report.Model.FocalLengths.Y;
				}

				numInliers = checked((int)report.Support.NumInliers);
				inlierMask = report.InlierMask;
				return true;
			}
		}
		else
		{
			var points2DWithRays = new Point2DWithRay[points2D.Count];
			for (int i = 0; i < points2D.Count; ++i)
			{
				points2DWithRays[i] = new Point2DWithRay(points2D[i], camera.CamRayFromImg(points2D[i]) ?? Vector3d.Zero);
			}

			ImgFromCamFunc imgFromCamFunc = camPoint => camera.ImgFromCam(camPoint);
			var ransac = new LoRansac<P3PEstimator, EPNPEstimator, Point2DWithRay, Vector3d, Matrix3x4d>(
				options.RansacOptions, new P3PEstimator(imgFromCamFunc), new EPNPEstimator(imgFromCamFunc));
			RansacReport<Matrix3x4d, InlierSupportMeasurer.Support> report = ransac.Estimate(points2DWithRays, points3DArray);
			if (report.Success)
			{
				camFromWorld = new Rigid3d(Quaterniond.FromRotationMatrix(report.Model.LeftCols3()), report.Model.Col(3));
				numInliers = checked((int)report.Support.NumInliers);
				inlierMask = report.InlierMask;
				return true;
			}
		}

		return false;
	}

	/// <summary>
	/// Estimate relative pose from 2D-2D correspondences, scored with the pixel-unit tangent
	/// Sampson error. The first camera is at the origin; the second camera's pose is the
	/// world-to-camera transformation. Returns whether the pose is estimated successfully.
	/// Port of colmap::EstimateRelativePose.
	/// </summary>
	public static bool EstimateRelativePose(
		RansacOptions ransacOptions,
		IReadOnlyList<CamRayWithJac> camRays1WithJac,
		IReadOnlyList<CamRayWithJac> camRays2WithJac,
		ref Rigid3d cam2FromCam1,
		out int numInliers,
		out bool[] inlierMask)
	{
		Check.Eq(camRays1WithJac.Count, camRays2WithJac.Count);
		numInliers = 0;
		inlierMask = [];

		var ransac = new LoRansac<EssentialMatrixTangentSampsonEstimator, EssentialMatrixTangentSampsonEstimator,
			CamRayWithJac, CamRayWithJac, Matrix3d>(ransacOptions, default, default);
		RansacReport<Matrix3d, InlierSupportMeasurer.Support> report =
			ransac.Estimate(camRays1WithJac.ToArray(), camRays2WithJac.ToArray());

		if (!report.Success)
		{
			return false;
		}

		var inlierCamRays1 = new List<Vector3d>((int)report.Support.NumInliers);
		var inlierCamRays2 = new List<Vector3d>((int)report.Support.NumInliers);
		for (int i = 0; i < camRays1WithJac.Count; ++i)
		{
			if (report.InlierMask[i])
			{
				inlierCamRays1.Add(camRays1WithJac[i].Ray);
				inlierCamRays2.Add(camRays2WithJac[i].Ray);
			}
		}

		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(report.Model, inlierCamRays1, inlierCamRays2, out cam2FromCam1, validIndices);

		Quaterniond q = cam2FromCam1.Rotation;
		Vector3d t = cam2FromCam1.Translation;
		if (double.IsNaN(q.X) || double.IsNaN(q.Y) || double.IsNaN(q.Z) || double.IsNaN(q.W)
			|| double.IsNaN(t.X) || double.IsNaN(t.Y) || double.IsNaN(t.Z))
		{
			return false;
		}

		numInliers = checked((int)report.Support.NumInliers);
		inlierMask = report.InlierMask;

		return validIndices.Count > 0;
	}

	/// <summary>
	/// Refine absolute pose (optionally focal length) from 2D-3D correspondences. Returns
	/// whether the solution is usable. Port of colmap::RefineAbsolutePose.
	/// </summary>
	public static bool RefineAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		ref Rigid3d camFromWorld,
		Camera camera) =>
		RefineAbsolutePose(options, inlierMask, points2D, points3D, ref camFromWorld, camera, null);

	/// <summary>
	/// <see cref="RefineAbsolutePose(AbsolutePoseRefinementOptions, IReadOnlyList{bool}, IReadOnlyList{Vector2d}, IReadOnlyList{Vector3d}, ref Rigid3d, Camera)"/>
	/// that also estimates the 6x6 covariance of the rotation (as axis-angle, in tangent
	/// space) and translation terms.
	/// </summary>
	public static bool RefineAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		ref Rigid3d camFromWorld,
		Camera camera,
		out Matrix6d camFromWorldCov)
	{
		var covariance = new StrongBox<Matrix6d>();
		bool result = RefineAbsolutePose(options, inlierMask, points2D, points3D, ref camFromWorld, camera, covariance);
		camFromWorldCov = covariance.Value;
		return result;
	}

	private static bool RefineAbsolutePose(
		AbsolutePoseRefinementOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<Vector2d> points2D,
		IReadOnlyList<Vector3d> points3D,
		ref Rigid3d camFromWorld,
		Camera camera,
		StrongBox<Matrix6d>? camFromWorldCov)
	{
		Check.Eq(inlierMask.Count, points2D.Count);
		Check.Eq(points2D.Count, points3D.Count);
		options.Check();

		var lossFunction = new CauchyLoss(options.LossFunctionScale);

		// CostFunction assumes unit quaternions.
		var poseStorage = new Rigid3dStorage(
			new Rigid3d(camFromWorld.Rotation.Normalized(), camFromWorld.Translation));
		double[] poseParams = poseStorage.Params;

		var problem = new Problem();

		for (int i = 0; i < points2D.Count; ++i)
		{
			// Skip outlier observations
			if (!inlierMask[i])
			{
				continue;
			}

			problem.AddResidualBlock(
				CameraCostFunctions.CreateReprojErrorConstantPoint3DCostFunction(camera.ModelId, points2D[i], points3D[i]),
				lossFunction,
				poseParams,
				camera.Params);
		}

		if (options.UsePositionPrior)
		{
			problem.AddResidualBlock(
				CovarianceWeightedCostFunctor.Create(
					MatrixXd.From(options.PositionPriorCovariance),
					new AbsolutePosePositionPriorCostFunctor(options.PositionPriorInWorld)),
				null,
				poseParams);
		}

		if (problem.NumResiduals > 0)
		{
			if (problem.HasParameterBlock(camera.Params))
			{
				// Camera parameterization.
				if (!options.RefineFocalLength && !options.RefineExtraParams)
				{
					problem.SetParameterBlockConstant(camera.Params);
				}
				else
				{
					// Always set the principal point as fixed.
					var cameraParamsConst = new List<int>(camera.PrincipalPointIdxs.ToArray());
					if (!options.RefineFocalLength)
					{
						cameraParamsConst.AddRange(camera.FocalLengthIdxs.ToArray());
					}

					if (!options.RefineExtraParams)
					{
						cameraParamsConst.AddRange(camera.ExtraParamsIdxs.ToArray());
					}

					if (cameraParamsConst.Count == camera.Params.Length)
					{
						problem.SetParameterBlockConstant(camera.Params);
					}
					else
					{
						SetManifold(problem, camera.Params, CreateSubsetManifold(camera.Params.Length, cameraParamsConst));
					}
				}
			}

			SetManifold(
				problem,
				poseParams,
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
		camFromWorld = poseStorage.Value;

		if (!summary.IsSolutionUsable)
		{
			return false;
		}

		if (problem.NumResiduals > 0 && camFromWorldCov is not null)
		{
			var covariance = new Covariance();
			ArraySegment<double>[] parameterBlocks = [poseParams];
			if (!covariance.Compute(parameterBlocks, problem))
			{
				return false;
			}

			// The rotation covariance is estimated in the tangent space of the
			// quaternion, which corresponds to the 3-DoF axis-angle local
			// parameterization.
			camFromWorldCov.Value = Matrix6d.FromColumnMajor(
				covariance.GetCovarianceMatrixInTangentSpace(parameterBlocks).AsSpan());
		}

		return true;
	}

	/// <summary>
	/// Refine relative pose of two cameras: minimizes the pixel-unit tangent Sampson error
	/// over the masked-in correspondences as plain least squares (no robust loss). The first
	/// camera is [I | 0]; the translation is refined up to scale (stays a unit vector).
	/// Returns whether the solution is usable. Port of colmap::RefineRelativePose.
	/// </summary>
	public static bool RefineRelativePose(
		SolverOptions options,
		IReadOnlyList<bool> inlierMask,
		IReadOnlyList<CamRayWithJac> camRays1WithJac,
		IReadOnlyList<CamRayWithJac> camRays2WithJac,
		ref Rigid3d cam2FromCam1)
	{
		Check.Eq(camRays1WithJac.Count, camRays2WithJac.Count);
		Check.Eq(camRays1WithJac.Count, inlierMask.Count);

		// CostFunction assumes unit quaternions.
		var poseStorage = new Rigid3dStorage(
			new Rigid3d(cam2FromCam1.Rotation.Normalized(), cam2FromCam1.Translation));
		double[] poseParams = poseStorage.Params;

		// No robust loss: the observations are already RANSAC-gated inliers, so the
		// refinement is plain least squares.
		var problem = new Problem();

		for (int i = 0; i < camRays1WithJac.Count; ++i)
		{
			// Skip outliers and unprojectable (zero) rays, which the residual scores as
			// a perfect fit rather than rejecting.
			if (!inlierMask[i] || IsZero(camRays1WithJac[i].Ray) || IsZero(camRays2WithJac[i].Ray))
			{
				continue;
			}

			problem.AddResidualBlock(
				TangentSampsonErrorCostFunctor.Create(camRays1WithJac[i], camRays2WithJac[i]), null, poseParams);
		}

		// COLMAP calls SetManifold unconditionally; Ceres then fails (LOG(FATAL)) on a
		// block that is not in the problem, so only an empty problem can differ.
		SetManifold(problem, poseParams, CreateProductManifold(CreateEigenQuaternionManifold(), CreateSphereManifold(3)));

		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);
		cam2FromCam1 = poseStorage.Value;

		return summary.IsSolutionUsable;
	}

	/// <summary>
	/// Refine essential matrix: decomposes it into rotation and translation and refines the
	/// relative pose with <see cref="RefineRelativePose"/>. Returns whether the solution is
	/// usable. Port of colmap::RefineEssentialMatrix.
	/// </summary>
	public static bool RefineEssentialMatrix(
		SolverOptions options,
		IReadOnlyList<CamRayWithJac> camRays1WithJac,
		IReadOnlyList<CamRayWithJac> camRays2WithJac,
		IReadOnlyList<bool> inlierMask,
		ref Matrix3d e)
	{
		Check.Eq(camRays1WithJac.Count, camRays2WithJac.Count);
		Check.Eq(camRays1WithJac.Count, inlierMask.Count);

		// Collect inliers. PoseFromEssentialMatrix needs the bare bearings. The
		// refinement additionally needs their unprojection Jacobians.
		var inlierCamRays1WithJac = new List<CamRayWithJac>();
		var inlierCamRays2WithJac = new List<CamRayWithJac>();
		var inlierRays1 = new List<Vector3d>();
		var inlierRays2 = new List<Vector3d>();
		for (int i = 0; i < inlierMask.Count; ++i)
		{
			if (inlierMask[i])
			{
				inlierCamRays1WithJac.Add(camRays1WithJac[i]);
				inlierCamRays2WithJac.Add(camRays2WithJac[i]);
				inlierRays1.Add(camRays1WithJac[i].Ray);
				inlierRays2.Add(camRays2WithJac[i].Ray);
			}
		}

		// Extract relative pose from essential matrix.
		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(e, inlierRays1, inlierRays2, out Rigid3d cam2FromCam1, validIndices);

		if (validIndices.Count == 0)
		{
			return false;
		}

		// Refine over all inliers (robustness came from the RANSAC selection).
		var allInliers = new bool[inlierCamRays1WithJac.Count];
		Array.Fill(allInliers, true);
		if (!RefineRelativePose(options, allInliers, inlierCamRays1WithJac, inlierCamRays2WithJac, ref cam2FromCam1))
		{
			return false;
		}

		e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		return true;
	}

	// Eigen's isZero(): every coefficient within the default dummy precision (1e-12) of 0.
	private static bool IsZero(Vector3d v) =>
		Math.Abs(v.X) <= LinearAlgebraConstants.DummyPrecision
		&& Math.Abs(v.Y) <= LinearAlgebraConstants.DummyPrecision
		&& Math.Abs(v.Z) <= LinearAlgebraConstants.DummyPrecision;
}
