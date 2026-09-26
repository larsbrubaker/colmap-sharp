// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Covariance: colmap/estimators/covariance.h and .cc - covariances of the parameters of a
// bundle-adjustment problem (points, poses and "other" blocks such as camera intrinsics),
// computed with the Schur complement trick instead of ceres::Covariance
// (Solver/Covariance.cs), which would factor the whole Jacobian. This file holds the public
// API (BACovariance, BACovarianceOptions, EstimateBACovariance and COLMAP's internal::
// parameter listings); Covariance.Schur.cs holds the elimination and the factorization.
// Tests: ColmapSharp.Tests/Estimators/BACovarianceTests.cs (covariance_test.cc 1:1). Tier B:
// the covariances agree with ceres::Covariance (dense QR here) within COLMAP's 1e-8.
//
// Translation notes:
// - COLMAP keys parameter blocks by raw pointer; here a block is identified by its
//   ArraySegment's (array, offset), as in Solver/Problem.cs. A default segment (null array)
//   plays the part of nullptr.
// - std::optional<MatrixXd> is a nullable MatrixXd.
// - LOG(WARNING)/VLOG output is dropped (the library has no log sink yet); the functions
//   return null where COLMAP warns and returns nullopt.
// - GetOtherParams lists blocks in Problem.GetParameterBlocks order, which is insertion order
//   here and address order in Ceres (docs/CPP_DIVERGENCES.md, entry 53).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::internal::PoseParam: a variable cam_from_world block of an image.</summary>
/// <param name="ImageId">The image the pose belongs to.</param>
/// <param name="CamFromWorld">The pose's parameter block ([qx, qy, qz, qw, tx, ty, tz]).</param>
public readonly record struct PoseParam(uint ImageId, ArraySegment<double> CamFromWorld);

/// <summary>Port of colmap::internal::PointParam: a variable xyz block of a 3D point.</summary>
/// <param name="Point3DId">The 3D point.</param>
/// <param name="Xyz">The point's parameter block.</param>
public readonly record struct PointParam(ulong Point3DId, ArraySegment<double> Xyz);

/// <summary>Which parameters BACovarianceOptions computes covariances for.</summary>
public enum BACovarianceParams
{
	/// <summary>Poses only.</summary>
	Poses,

	/// <summary>Points only.</summary>
	Points,

	/// <summary>Poses and points.</summary>
	PosesAndPoints,

	/// <summary>Poses, points and every other variable block.</summary>
	All,
}

/// <summary>Port of colmap::BACovarianceOptions.</summary>
public sealed class BACovarianceOptions
{
	/// <summary>For which parameters to compute the covariance.</summary>
	public BACovarianceParams Params { get; set; } = BACovarianceParams.All;

	/// <summary>
	/// Damping factor for the Hessian in the Schur complement solver. Enables to robustly
	/// deal with poorly conditioned parameters.
	/// </summary>
	public double Damping { get; set; } = 1e-8;

	/// <summary>
	/// WARNING: COLMAP will remove this option in a future release. For custom bundle
	/// adjustment problems, this enables to specify a custom set of pose parameter blocks to
	/// consider. These pose blocks need not be part of the reconstruction, but they must
	/// follow the standard requirement for applying the Schur complement trick.
	/// </summary>
	public List<PoseParam> ExperimentalCustomPoses { get; } = [];
}

/// <summary>
/// Port of colmap::BACovariance: the covariances computed by EstimateBACovariance. Point
/// covariances are stored per point; pose and other covariances are read from
/// L_inv, where L_inv^T L_inv is the inverse of the Schur complement on pose/other parameters.
/// </summary>
public sealed class BACovariance
{
	private readonly Dictionary<ulong, MatrixXd> pointCovs;
	private readonly Dictionary<uint, (int Start, int Size)> poseLStartSize;
	private readonly Dictionary<(double[] Array, int Offset), (int Start, int Size)> otherLStartSize;
	private readonly MatrixXd lInv;

	/// <summary>Creates the result from its parts (COLMAP's constructor).</summary>
	public BACovariance(
		Dictionary<ulong, MatrixXd> pointCovs,
		Dictionary<uint, (int Start, int Size)> poseLStartSize,
		Dictionary<(double[] Array, int Offset), (int Start, int Size)> otherLStartSize,
		MatrixXd lInv)
	{
		this.pointCovs = pointCovs;
		this.poseLStartSize = poseLStartSize;
		this.otherLStartSize = otherLStartSize;
		this.lInv = lInv;
	}

	/// <summary>
	/// Covariance for 3D points, conditioned on all other variables set constant. If some
	/// dimensions are kept constant, the respective rows/columns are omitted. Returns null if
	/// the 3D point is not a variable in the problem.
	/// </summary>
	public MatrixXd? GetPointCov(ulong point3DId) =>
		pointCovs.TryGetValue(point3DId, out MatrixXd? cov) ? cov.Clone() : null;

	/// <summary>
	/// Tangent space covariance in the order [rotation, translation]. If some dimensions are
	/// kept constant, the respective rows/columns are omitted. Returns null if the image is
	/// not a variable in the problem.
	/// </summary>
	public MatrixXd? GetCamCovFromWorld(uint imageId)
	{
		if (!poseLStartSize.TryGetValue(imageId, out (int Start, int Size) block))
		{
			return null;
		}

		return ExtractCovFromLInverse(lInv, block.Start, block.Start, block.Size, block.Size);
	}

	/// <summary>Cross covariance of two poses' tangent spaces; null if either is not a variable.</summary>
	public MatrixXd? GetCamCrossCovFromWorld(uint imageId1, uint imageId2)
	{
		if (!poseLStartSize.TryGetValue(imageId1, out (int Start, int Size) block1)
			|| !poseLStartSize.TryGetValue(imageId2, out (int Start, int Size) block2))
		{
			return null;
		}

		return ExtractCovFromLInverse(lInv, block1.Start, block2.Start, block1.Size, block2.Size);
	}

	/// <summary>
	/// Relative pose covariance in the order [rotation, translation]. Returns null if some
	/// dimensions are kept constant for either of the two poses. This does not mean that one
	/// cannot get relative pose covariance for such case, but requires custom logic to fill
	/// in zero block in the covariance matrix.
	/// </summary>
	public MatrixXd? GetCam2CovFromCam1(uint imageId1, Rigid3d cam1FromWorld, uint imageId2, Rigid3d cam2FromWorld)
	{
		MatrixXd? cov11 = GetCamCovFromWorld(imageId1);
		if (cov11 is null || cov11.Rows != 6)
		{
			// COLMAP warns when the pose is (partially) constant.
			return null;
		}

		MatrixXd? cov22 = GetCamCovFromWorld(imageId2);
		if (cov22 is null || cov22.Rows != 6)
		{
			return null;
		}

		MatrixXd cov12 = Check.NotNull(GetCamCrossCovFromWorld(imageId1, imageId2));
		var cov = new MatrixXd(12, 12);
		cov.SetBlock(0, 0, cov11);
		cov.SetBlock(6, 6, cov22);
		cov.SetBlock(0, 6, cov12);
		cov.SetBlock(6, 0, cov12.Transpose());
		return MatrixXd.From(Rigid3d.GetCovarianceForRelativeRigid3d(cam1FromWorld, cam2FromWorld, cov));
	}

	/// <summary>
	/// Tangent space covariance for any other variable parameter block in the problem. If
	/// some dimensions are kept constant, the respective rows/columns are omitted. Returns
	/// null if the parameter block is not a variable in the problem.
	/// </summary>
	public MatrixXd? GetOtherParamsCov(ArraySegment<double> parameters)
	{
		if (parameters.Array is null
			|| !otherLStartSize.TryGetValue((parameters.Array, parameters.Offset), out (int Start, int Size) block))
		{
			return null;
		}

		return ExtractCovFromLInverse(lInv, block.Start, block.Start, block.Size, block.Size);
	}

	/// <summary>
	/// L_inv.block(0, rowStart, n, rowBlockSize)^T * L_inv.block(0, colStart, n, colBlockSize).
	/// </summary>
	private static MatrixXd ExtractCovFromLInverse(MatrixXd lInv, int rowStart, int colStart, int rowBlockSize, int colBlockSize)
	{
		var cov = new MatrixXd(rowBlockSize, colBlockSize);
		for (int j = 0; j < colBlockSize; j++)
		{
			Span<double> right = lInv.ColumnSpan(colStart + j);
			for (int i = 0; i < rowBlockSize; i++)
			{
				Span<double> left = lInv.ColumnSpan(rowStart + i);
				double sum = 0.0;
				for (int k = 0; k < left.Length; k++)
				{
					sum += left[k] * right[k];
				}

				cov[i, j] = sum;
			}
		}

		return cov;
	}
}

/// <summary>Port of colmap::EstimateBACovariance and the internal:: parameter listings.</summary>
public static partial class BACovarianceEstimation
{
	/// <summary>
	/// Computes covariances for the parameters in a bundle adjustment problem. The problem
	/// must have a structure suitable for the Schur complement trick. This is the case for
	/// the standard configuration of bundle adjustment problems, but be careful if you modify
	/// the underlying problem with custom residuals. Returns null if the estimation was not
	/// successful.
	/// </summary>
	public static BACovariance? EstimateBACovariance(
		BACovarianceOptions options, Reconstruction reconstruction, CeresBundleAdjuster bundleAdjuster)
	{
		ArgumentNullException.ThrowIfNull(bundleAdjuster);
		return EstimateBACovarianceFromProblem(options, reconstruction, Check.NotNull(bundleAdjuster.Problem));
	}

	/// <summary>EstimateBACovariance on a caller-built problem.</summary>
	public static BACovariance? EstimateBACovarianceFromProblem(
		BACovarianceOptions options, Reconstruction reconstruction, Problem problem)
	{
		ArgumentNullException.ThrowIfNull(options);
		bool estimatePointCovs = options.Params is BACovarianceParams.Points
			or BACovarianceParams.PosesAndPoints or BACovarianceParams.All;
		bool estimatePoseCovs = options.Params is BACovarianceParams.Poses
			or BACovarianceParams.PosesAndPoints or BACovarianceParams.All;
		bool estimateOtherCovs = options.Params == BACovarianceParams.All;

		List<PointParam> points = GetPointParams(reconstruction, problem);
		List<PoseParam> poses = options.ExperimentalCustomPoses.Count == 0
			? GetPoseParams(reconstruction, problem)
			: options.ExperimentalCustomPoses;
		List<ArraySegment<double>> others = GetOtherParams(problem, poses, points);

		int pointNumParams = 0;
		int poseNumParams = 0;
		int otherNumParams = 0;
		var poseLStartSize = new Dictionary<uint, (int Start, int Size)>();
		var otherLStartSize = new Dictionary<(double[] Array, int Offset), (int Start, int Size)>();
		foreach (PointParam point in points)
		{
			pointNumParams += problem.ParameterBlockTangentSize(point.Xyz);
		}

		if (estimatePoseCovs || estimateOtherCovs)
		{
			foreach (PoseParam pose in poses)
			{
				int numParams = problem.ParameterBlockTangentSize(pose.CamFromWorld);

				// emplace: a repeated image id keeps its first entry.
				poseLStartSize.TryAdd(pose.ImageId, (poseNumParams, numParams));
				poseNumParams += numParams;
			}

			foreach (ArraySegment<double> other in others)
			{
				int numParams = problem.ParameterBlockTangentSize(other);
				otherLStartSize.TryAdd((other.Array!, other.Offset), (poseNumParams + otherNumParams, numParams));
				otherNumParams += numParams;
			}
		}

		var pointCovs = new Dictionary<ulong, MatrixXd>();
		if (!ComputeSchurComplement(
				estimatePointCovs,
				estimatePoseCovs,
				estimateOtherCovs,
				options.Damping,
				pointNumParams,
				points,
				poses,
				others,
				problem,
				pointCovs,
				out SparseMatrixCsc? s))
		{
			return null;
		}

		if (!estimatePoseCovs && !estimateOtherCovs)
		{
			return new BACovariance(pointCovs, [], [], new MatrixXd(0, 0));
		}

		if (!estimateOtherCovs)
		{
			if (!SchurEliminateOtherParams(options.Damping, poseNumParams, otherNumParams, ref s!))
			{
				return null;
			}
		}

		if (!ComputeLInverse(s!, out MatrixXd? lInv))
		{
			return null;
		}

		return new BACovariance(pointCovs, poseLStartSize, otherLStartSize, lInv!);
	}

	/// <summary>
	/// Port of colmap::internal::GetPoseParams: the variable rig_from_world blocks of the
	/// reconstruction's images that are in the problem, in image id order. Only trivial frames
	/// (the image's camera is the rig's reference sensor) are supported, as in COLMAP.
	/// </summary>
	public static List<PoseParam> GetPoseParams(Reconstruction reconstruction, Problem problem)
	{
		ArgumentNullException.ThrowIfNull(reconstruction);
		ArgumentNullException.ThrowIfNull(problem);
		var parameters = new List<PoseParam>(reconstruction.NumImages);
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			// TODO(jsch): Add support for non-trivial frames.
			Check.That(image.IsRefInFrame);
			double[] camFromWorld = image.FramePtr.RigFromWorldStorage.Params;
			if (problem.HasParameterBlock(camFromWorld) && !problem.IsParameterBlockConstant(camFromWorld))
			{
				parameters.Add(new PoseParam(imageId, camFromWorld));
			}
		}

		return parameters;
	}

	/// <summary>
	/// Port of colmap::internal::GetPointParams: the variable xyz blocks of the
	/// reconstruction's points that are in the problem, in point id order.
	/// </summary>
	public static List<PointParam> GetPointParams(Reconstruction reconstruction, Problem problem)
	{
		ArgumentNullException.ThrowIfNull(reconstruction);
		ArgumentNullException.ThrowIfNull(problem);
		var parameters = new List<PointParam>(reconstruction.NumPoints3D);
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			double[] xyz = point3D.XyzParams;
			if (problem.HasParameterBlock(xyz) && !problem.IsParameterBlockConstant(xyz))
			{
				parameters.Add(new PointParam(point3DId, xyz));
			}
		}

		return parameters;
	}

	/// <summary>
	/// Port of colmap::internal::GetOtherParams: the problem's variable blocks that are
	/// neither one of <paramref name="poses"/> nor one of <paramref name="points"/>.
	/// </summary>
	public static List<ArraySegment<double>> GetOtherParams(
		Problem problem, IReadOnlyList<PoseParam> poses, IReadOnlyList<PointParam> points)
	{
		ArgumentNullException.ThrowIfNull(problem);
		var poseAndPointParams = new HashSet<(double[]? Array, int Offset)>();
		foreach (PoseParam pose in poses)
		{
			poseAndPointParams.Add((pose.CamFromWorld.Array, pose.CamFromWorld.Offset));
		}

		foreach (PointParam point in points)
		{
			poseAndPointParams.Add((point.Xyz.Array, point.Xyz.Offset));
		}

		var parameters = new List<ArraySegment<double>>();
		foreach (ArraySegment<double> param in problem.GetParameterBlocks())
		{
			if (!problem.IsParameterBlockConstant(param) && !poseAndPointParams.Contains((param.Array, param.Offset)))
			{
				parameters.Add(param);
			}
		}

		return parameters;
	}
}
