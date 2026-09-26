// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RelativePoseOneSidedFocal: colmap/estimators/solvers/relpose_one_sided_focal.h and .cc -
// the relative pose estimator for an uncalibrated first view (one unknown focal) and a
// calibrated second view seen through bearing rays with unprojection Jacobians, so any
// central camera model (fisheye, spherical) works for the second view. The minimal solve is
// PoseLib's relpose_6pt_onesided_focal (PoseLib/Relpose6ptOnesidedFocal.cs, full template);
// the LO-RANSAC refiner is Optim/TinySolver.cs on
// TinyOneSidedFocalTangentSampsonErrorCostFunctor (analytic Jacobian,
// Estimators/CostFunctions/TinyRelativePoseSampsonError.cs); residuals are the tangent
// Sampson error in pixels under M = E K1^-1. Neighbor: RelativePoseSharedFocal.cs, whose
// ray and read-back helpers it shares. Tests:
// ColmapSharp.Tests/Estimators/Solvers/RelativePoseOneSidedFocalTests.cs
// (relpose_one_sided_focal_test.cc).
//
// Tier B for Estimate and Residuals, Tier C for Refine.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Relative pose plus the first view's focal from principal-point-centered points of the
/// uncalibrated view and calibrated rays (with Jacobians) of the second. Port of
/// colmap::RelativePoseOneSidedFocalEstimator.
/// </summary>
public readonly struct RelativePoseOneSidedFocalEstimator
	: IEstimator<Vector2d, CamRayWithJac, RelativePoseOneSidedFocalEstimator.Model>,
	ILocalEstimator<Vector2d, CamRayWithJac, RelativePoseOneSidedFocalEstimator.Model>
{
	/// <summary>
	/// The estimated model: a calibrated essential matrix plus the recovered focal length of
	/// the first (uncalibrated) view (C++'s M_t).
	/// </summary>
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
		RelativePoseOneSidedFocalManifold = new(default, default, default);

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 6;

	/// <summary>
	/// Estimate relative pose and the first view's focal from &gt;= 6 correspondences by
	/// wrapping poselib::relpose_6pt_onesided_focal (uses the first six). Clears
	/// <paramref name="models"/> first.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> imgPoints1, ReadOnlySpan<CamRayWithJac> camRays2WithJac, List<Model> models)
	{
		Check.Eq(imgPoints1.Length, camRays2WithJac.Length);
		Check.Ge(imgPoints1.Length, MinNumSamples);
		models.Clear();

		// Unlike poselib::relpose_6pt_shared_focal, this solver conditions its own input: it
		// isotropically rescales the uncalibrated points internally and undoes that scaling on
		// the recovered focal, which therefore comes back in the pixel units of imgPoints1. So
		// pass the centered points through as plain homogeneous coordinates and do not
		// pre-normalize them here.
		var imgPoints1Homogeneous = new Vector3d[imgPoints1.Length];
		var camRays2 = new Vector3d[camRays2WithJac.Length];
		for (int i = 0; i < imgPoints1.Length; ++i)
		{
			imgPoints1Homogeneous[i] = imgPoints1[i].Homogeneous();
			// Already unit bearings, which keeps the nullspace computation well conditioned.
			// Renormalizing here would turn a zeroed entry, left behind by a failed
			// unprojection, into NaN and poison the whole sample.
			camRays2[i] = camRays2WithJac[i].Ray;
		}

		// Use the full elimination template (use_elim = false), which solves for the focal
		// directly as an unknown. The default compact template instead recovers it from the
		// fundamental matrix via a semi-calibrated Kruppa step, which is numerically fragile
		// and loses accuracy even on exact points, propagating into E. The larger template
		// costs only marginally more runtime, as both variants end in the same
		// eigendecomposition, which dominates.
		var imagePairs = new List<ImagePair>();
		Relpose6ptOnesidedFocal.Solve(imgPoints1Homogeneous, camRays2, imagePairs);

		foreach (ImagePair imagePair in imagePairs)
		{
			// The solver may return degenerate solutions, so guard the focal defensively.
			double focal = imagePair.Camera1.Focal();
			if (!(focal > 0.0))
			{
				continue;
			}

			Rigid3d cam2FromCam1 = PoseLibUtils.ConvertPoseLibPoseToRigid3d(imagePair.Pose);
			models.Add(new Model(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), focal));
		}
	}

	/// <summary>The local estimate refines a copy of the current best model (C++'s Refine hook).</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> imgPoints1, ReadOnlySpan<CamRayWithJac> camRays2WithJac, in Model initialModel, List<Model> models)
	{
		Model model = initialModel;
		if (Refine(imgPoints1, camRays2WithJac, ref model))
		{
			models.Add(model);
		}
	}

	/// <summary>
	/// Nonlinear local optimization of the joint 6-DoF pose plus the unknown focal, in place,
	/// starting from <paramref name="model"/>. This is the entry point used by LO-RANSAC for
	/// local optimization. Returns true and overwrites the model with the refined one on
	/// success. On a degenerate decomposition (or non-positive focal) it returns false and
	/// leaves the model unchanged.
	/// </summary>
	public static bool Refine(ReadOnlySpan<Vector2d> imgPoints1, ReadOnlySpan<CamRayWithJac> camRays2WithJac, ref Model model)
	{
		Check.Eq(imgPoints1.Length, camRays2WithJac.Length);
		Check.Ge(imgPoints1.Length, MinNumSamples);

		if (!(model.Focal > 0.0))
		{
			return false;
		}

		// Decompose the current essential matrix into a relative pose, resolving the four-fold
		// ambiguity by cheirality. The first view's rays follow from the current focal
		// estimate; the second view's are already calibrated. Only the bearings are
		// materialized, since that is all PoseFromEssentialMatrix needs.
		Vector3d[] camRays1 = RelativePoseSharedFocalEstimator.CalibratedRays(imgPoints1, model.Focal);
		var camRays2 = new Vector3d[camRays2WithJac.Length];
		for (int i = 0; i < camRays2WithJac.Length; ++i)
		{
			camRays2[i] = camRays2WithJac[i].Ray;
		}

		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(model.E, camRays1, camRays2, out Rigid3d cam2FromCam1, validIndices);
		if (validIndices.Count == 0)
		{
			// Degenerate configuration: leave the initial model unchanged.
			return false;
		}

		// Nonlinear refinement of the joint 6-DoF (5-DoF pose plus the unknown focal) via
		// TinySolver (analytic Jacobians), applying the one-sided focal relative pose
		// manifold. The functor minimizes the same tangent Sampson error that Residuals()
		// scores, so local optimization improves exactly the quantity LO-RANSAC measures.
		// Plain least squares: the points are assumed to be the inlier set, so robustness
		// comes from the RANSAC inlier selection.
		var f = new TinyOneSidedFocalTangentSampsonErrorCostFunctor(imgPoints1.ToArray(), camRays2WithJac.ToArray());
		var solver = new TinySolver<TinyOneSidedFocalTangentSampsonErrorCostFunctor,
			TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold, TinyEuclideanManifold1>>(
			RelativePoseOneSidedFocalManifold);
		var options = new TinySolverOptions { MaxNumIterations = 25 };

		Vector4d q = cam2FromCam1.Rotation.Normalized().Coeffs;
		Vector3d t = cam2FromCam1.Translation.Normalized();
		Span<double> x = [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z, Math.Log(model.Focal)];
		solver.Solve(f, x, options);

		// Keep the refined estimate only if the solve stayed finite and left a non-degenerate
		// baseline; otherwise fall back to the decomposed pose and seed focal.
		if (RelativePoseSharedFocalEstimator.TryReadRefinedModel(x, out Matrix3d refinedE, out double refinedFocal))
		{
			model = new Model(refinedE, refinedFocal);
		}

		return true;
	}

	/// <summary>
	/// Squared tangent Sampson error, in pixels, of each correspondence under
	/// M = E * K1inv; double.MaxValue when the focal is not positive or a correspondence is
	/// degenerate (e.g. a zeroed ray left by a failed unprojection).
	/// </summary>
	public void Residuals(ReadOnlySpan<Vector2d> imgPoints1, ReadOnlySpan<CamRayWithJac> camRays2WithJac, in Model model, Span<double> residuals)
	{
		Check.Eq(imgPoints1.Length, camRays2WithJac.Length);
		if (!(model.Focal > 0.0))
		{
			residuals[..imgPoints1.Length].Fill(double.MaxValue);
			return;
		}

		Matrix3d m = model.E * Matrix3d.FromDiagonal(new Vector3d(1.0 / model.Focal, 1.0 / model.Focal, 1.0));
		Matrix3d mTranspose = m.Transpose();
		for (int i = 0; i < imgPoints1.Length; ++i)
		{
			Vector3d camRay2 = camRays2WithJac[i].Ray;
			Vector3d mPoint1 = m * imgPoints1[i].Homogeneous();
			double num = camRay2.Dot(mPoint1);
			// Constraint gradients in each view's own pixels. The first view needs no Jacobian,
			// as its d(x, y, 1)/d(x, y) merely selects the first two rows. This is
			// ComputeSquaredTangentSampsonError specialized to that structure, which is worth
			// inlining here because RANSAC evaluates it per hypothesis per correspondence.
			Vector3d mtRay2 = mTranspose * camRay2;
			double denom = (mtRay2.X * mtRay2.X + mtRay2.Y * mtRay2.Y)
				+ EssentialMatrix.SquaredPixelGradientNorm(camRays2WithJac[i].Jacobian, mPoint1);
			if (!(denom > 0))
			{
				// Degenerate, e.g. a zeroed entry left by a failed unprojection.
				residuals[i] = double.MaxValue;
				continue;
			}

			residuals[i] = num * num / denom;
		}
	}
}
