// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RelativePoseSharedFocal: colmap/estimators/solvers/relpose_shared_focal.h and .cc - the
// relative pose estimator for two views sharing one unknown focal length, on
// principal-point-centered pixels. The minimal solve is PoseLib's relpose_6pt_shared_focal
// (PoseLib/Relpose6ptSharedFocal.cs); the LO-RANSAC refiner is Optim/TinySolver.cs on
// Estimators/CostFunctions/TinyRelativePoseSampsonError.cs (TinyFocalSampsonErrorCostFunctor,
// autodiff) over the pose-plus-log-focal manifold; residuals are the pixel Sampson error
// under F = K^-1 E K^-1. IsFocalIdentifiable screens out the poses whose focal cannot be
// recovered. Neighbor: RelativePoseOneSidedFocal.cs (one view calibrated). Tests:
// ColmapSharp.Tests/Estimators/Solvers/RelativePoseSharedFocalTests.cs
// (relpose_shared_focal_test.cc).
//
// Tier B for Estimate and Residuals, Tier C for Refine.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Relative pose plus a shared focal length from principal-point-centered image point pairs
/// (u - cx, v - cy). Port of colmap::RelativePoseSharedFocalEstimator.
/// </summary>
public readonly struct RelativePoseSharedFocalEstimator
	: IEstimator<Vector2d, Vector2d, RelativePoseSharedFocalEstimator.Model>,
	ILocalEstimator<Vector2d, Vector2d, RelativePoseSharedFocalEstimator.Model>
{
	/// <summary>The estimated model: a calibrated essential matrix plus the shared focal (C++'s M_t).</summary>
	public record struct Model(Matrix3d E, double Focal)
	{
		/// <summary>C++'s default: E = identity, focal = 0.</summary>
		public Model()
			: this(Matrix3d.Identity, 0.0)
		{
		}
	}

	// ProductManifold<EigenQuaternionManifold, SphereManifold<3>, EuclideanManifold<1>>.
	private static readonly TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold, TinyEuclideanManifold1>
		RelativePoseSharedFocalManifold = new(default, default, default);

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 6;

	/// <summary>
	/// Estimate relative pose and shared focal from &gt;= 6 centered point pairs by wrapping
	/// poselib::relpose_6pt_shared_focal (uses the first six points). Clears
	/// <paramref name="models"/> first.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, List<Model> models)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Ge(points1.Length, MinNumSamples);
		models.Clear();

		// The minimal solver is formulated for unit bearing vectors, so raw pixel coordinates
		// make it numerically unstable: after appending the homogeneous 1 and normalizing, that
		// third component becomes negligible and the focal length is recovered from it.
		// Isotropically rescale the (already principal-point-centered) points to unit
		// magnitude first; the recovered focal is expressed in the rescaled units and undone
		// below. This mirrors the normalization PoseLib applies before running the solver.
		double scale = ComputeScaleForNormalization(points1, points2);
		if (!(scale > 0.0))
		{
			return;
		}

		double invScale = 1.0 / scale;

		var rays1 = new Vector3d[points1.Length];
		var rays2 = new Vector3d[points2.Length];
		for (int i = 0; i < points1.Length; ++i)
		{
			rays1[i] = (invScale * points1[i]).Homogeneous().Normalized();
			rays2[i] = (invScale * points2[i]).Homogeneous().Normalized();
		}

		var imagePairs = new List<ImagePair>();
		Relpose6ptSharedFocal.Solve(rays1, rays2, imagePairs);

		foreach (ImagePair imagePair in imagePairs)
		{
			// Undo the isotropic rescaling on the recovered focal; the pose (and thus the
			// essential matrix) is scale-invariant. The solver may return degenerate
			// solutions, so guard the focal defensively.
			double focal = imagePair.Camera1.Focal() * scale;
			if (!(focal > 0.0))
			{
				continue;
			}

			Rigid3d cam2FromCam1 = PoseLibUtils.ConvertPoseLibPoseToRigid3d(imagePair.Pose);
			models.Add(new Model(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), focal));
		}
	}

	/// <summary>The local estimate refines a copy of the current best model (C++'s Refine hook).</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Model initialModel, List<Model> models)
	{
		Model model = initialModel;
		if (Refine(points1, points2, ref model))
		{
			models.Add(model);
		}
	}

	/// <summary>
	/// Nonlinear local optimization of the joint 6-DoF pose + shared focal, in place,
	/// starting from <paramref name="model"/>. This is the entry point used by LO-RANSAC for
	/// local optimization. Returns true and overwrites the model with the refined one on
	/// success. On a degenerate decomposition (or non-positive focal) it returns false and
	/// leaves the model unchanged.
	/// </summary>
	public static bool Refine(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, ref Model model)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Ge(points1.Length, MinNumSamples);

		if (!(model.Focal > 0.0))
		{
			return false;
		}

		// Decompose the current essential matrix into a relative pose, resolving the
		// four-fold ambiguity by cheirality over the focal-calibrated sample rays.
		Vector3d[] camRays1 = CalibratedRays(points1, model.Focal);
		Vector3d[] camRays2 = CalibratedRays(points2, model.Focal);
		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(model.E, camRays1, camRays2, out Rigid3d cam2FromCam1, validIndices);
		if (validIndices.Count == 0)
		{
			// Degenerate configuration: leave the initial model unchanged.
			return false;
		}

		// Nonlinear pixel-space Sampson refinement of the joint 6-DoF (5-DoF pose plus shared
		// focal) via TinySolver (autodiff), applying the shared-focal relative pose manifold.
		// Plain least squares: the points are assumed to be the inlier set, so robustness
		// comes from the RANSAC inlier selection.
		var functor = new TinyFocalSampsonErrorCostFunctor(points1.ToArray(), points2.ToArray());
		TinySolverAutoDiffFunction<TinyFocalSampsonErrorCostFunctor, Grad8> f = functor.CreateAutoDiffFunction();
		var solver = new TinySolver<TinySolverAutoDiffFunction<TinyFocalSampsonErrorCostFunctor, Grad8>,
			TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold, TinyEuclideanManifold1>>(
			RelativePoseSharedFocalManifold);
		var options = new TinySolverOptions { MaxNumIterations = 25 };

		Vector4d q = cam2FromCam1.Rotation.Normalized().Coeffs;
		Vector3d t = cam2FromCam1.Translation.Normalized();
		Span<double> x = [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z, Math.Log(model.Focal)];
		solver.Solve(f, x, options);

		// Keep the refined estimate only if the solve stayed finite and left a non-degenerate
		// baseline; otherwise fall back to the decomposed pose and seed focal.
		if (TryReadRefinedModel(x, out Matrix3d refinedE, out double refinedFocal))
		{
			model = new Model(refinedE, refinedFocal);
		}

		return true;
	}

	/// <summary>
	/// Squared pixel-space Sampson error of each centered point pair under the fundamental
	/// matrix implied by the model (F = Kinv * E * Kinv); double.MaxValue for every pair when
	/// the focal is not positive.
	/// </summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Model model, Span<double> residuals)
	{
		Check.Eq(points1.Length, points2.Length);
		if (!(model.Focal > 0.0))
		{
			residuals[..points1.Length].Fill(double.MaxValue);
			return;
		}

		Matrix3d f = FundamentalFromEssentialSharedFocal(model.E, model.Focal);
		FundamentalMatrixResiduals.SquaredSampsonError(points1, points2, f, residuals[..points1.Length]);
	}

	/// <summary>
	/// Whether the focal recovered for the given pose should be treated as reliable, i.e.
	/// whether the pose is clear of the singular family (parallel optical axes, or coplanar
	/// axes meeting with both centers equidistant from their intersection). Tested as two
	/// predicates: the axes are sufficiently skew, or, failing that, sufficiently far from
	/// isosceles.
	/// </summary>
	public static bool IsFocalIdentifiable(Rigid3d cam2FromCam1)
	{
		// Minimum sine of the angle between the baseline and the plane of the two optical
		// axes for the axes to count as skew. Skew axes cannot be singular, so this admits
		// them outright. Turntable and object-scan capture, the common near-coplanar case,
		// falls below it and is deferred to the second predicate.
		const double kMinAxesSkew = 0.05;
		// Minimum relative difference between the distances of the two camera centers from
		// the intersection of their optical axes, for near-coplanar axes to count as clear of
		// the isosceles singularity. Also rejects near-parallel axes, whose distances diverge
		// together. Both thresholds sit at the knee of a synthetic focal-accuracy sweep.
		const double kMinIsoscelesDeviation = 0.05;

		// Skew optical axes neither intersect nor are parallel, so they can never be singular
		// and the second predicate need not be consulted.
		if (AxesSkewness(cam2FromCam1) > kMinAxesSkew)
		{
			return true;
		}

		// The axes are (near-)coplanar: singular only if they additionally are
		// (near-)parallel, or intersect with the two centers (near-)equidistant from the
		// intersection point. A zero baseline (pure rotation) lands here with both distances
		// zero and is rejected.
		return IsoscelesDeviation(cam2FromCam1) > kMinIsoscelesDeviation;
	}

	/// <summary>
	/// Reads the refined E and focal back from a finished solve over
	/// [qx, qy, qz, qw, tx, ty, tz, log_f]. Returns false (keep the decomposed pose and seed
	/// focal) when the solve did not stay finite or collapsed the baseline. Shared with
	/// RelativePoseOneSidedFocal.cs.
	/// </summary>
	internal static bool TryReadRefinedModel(ReadOnlySpan<double> x, out Matrix3d e, out double focal)
	{
		bool allFinite = true;
		foreach (double value in x)
		{
			allFinite &= double.IsFinite(value);
		}

		var translation = new Vector3d(x[4], x[5], x[6]);
		if (allFinite && translation.SquaredNorm > 0)
		{
			var cam2FromCam1 = new Rigid3d(new Quaterniond(x[3], x[0], x[1], x[2]).Normalized(), translation);
			e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			focal = Math.Exp(x[7]);
			return true;
		}

		e = default;
		focal = 0.0;
		return false;
	}

	/// <summary>
	/// Unit bearings (x / f, y / f, 1) normalized, of centered image points. Shared with
	/// RelativePoseOneSidedFocal.cs.
	/// </summary>
	internal static Vector3d[] CalibratedRays(ReadOnlySpan<Vector2d> points, double focal)
	{
		double invF = 1.0 / focal;
		var rays = new Vector3d[points.Length];
		for (int i = 0; i < points.Length; ++i)
		{
			rays[i] = new Vector3d(points[i].X * invF, points[i].Y * invF, 1.0).Normalized();
		}

		return rays;
	}

	private static Matrix3d FundamentalFromEssentialSharedFocal(in Matrix3d e, double focal)
	{
		double invF = 1.0 / focal;
		Matrix3d kInv = Matrix3d.FromDiagonal(new Vector3d(invF, invF, 1.0));
		return kInv * e * kInv;
	}

	private static double AxesSkewness(Rigid3d cam2FromCam1)
	{
		var axis1 = new Vector3d(0, 0, 1);
		Vector3d axis2 = cam2FromCam1.Rotation.Inverse() * new Vector3d(0, 0, 1);
		Vector3d baselineDir = cam2FromCam1.TgtOriginInSrc().Normalized();
		if (!(double.IsFinite(baselineDir.X) && double.IsFinite(baselineDir.Y) && double.IsFinite(baselineDir.Z)))
		{
			return 0.0;  // Pure rotation: no baseline direction; fully degenerate.
		}

		return Math.Abs(baselineDir.Dot(axis1.Cross(axis2)));
	}

	private static double IsoscelesDeviation(Rigid3d cam2FromCam1)
	{
		Vector3d center2 = cam2FromCam1.TgtOriginInSrc();
		var axis1 = new Vector3d(0, 0, 1);
		Vector3d axis2 = cam2FromCam1.Rotation.Inverse() * new Vector3d(0, 0, 1);

		// Closest approach of the lines d1 * axis1 and center2 + d2 * axis2.
		double cosAxes = axis1.Dot(axis2);
		double sinSqAxes = 1.0 - cosAxes * cosAxes;
		if (sinSqAxes == 0.0)
		{
			return 0.0;  // Exactly parallel axes: singular.
		}

		double proj1 = center2.Dot(axis1);
		double proj2 = center2.Dot(axis2);
		double dist1 = (proj1 - cosAxes * proj2) / sinSqAxes;
		double dist2 = (cosAxes * proj1 - proj2) / sinSqAxes;

		double distSum = Math.Abs(dist1) + Math.Abs(dist2);
		if (distSum == 0.0)
		{
			return 0.0;  // Both centers at the intersection point.
		}

		return Math.Abs(dist1 - dist2) / distSum;
	}

	private static double ComputeScaleForNormalization(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2)
	{
		if (points1.Length == 0)
		{
			return 0.0;
		}

		double sum = 0.0;
		for (int i = 0; i < points1.Length; ++i)
		{
			sum += points1[i].Norm + points2[i].Norm;
		}

		return sum / (2.0 * points1.Length);
	}
}
