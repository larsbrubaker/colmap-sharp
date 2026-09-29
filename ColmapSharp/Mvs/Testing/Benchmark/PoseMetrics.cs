// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoseMetrics: camera pose error of a reconstruction against the truth (docs/QUALITY_PLAN.md,
// stage 0b). Not a COLMAP port. SfM recovers the scene only up to a similarity, so the estimate
// is first mapped into the truth frame by a Sim3, then compared camera by camera.
//
// Why the Sim3 is not fitted to camera centers alone (as COLMAP's AlignReconstructionTo* do):
// the benchmark's cameras sit on a short arc around a small object, often only a few of them,
// and a similarity fitted to few, clustered or nearly collinear centers leaves the rotation
// about their common line almost free. On a 16-frame sphere run the centers-only fit reported
// 28 degrees of rotation error where every camera's own orientation agreed with the truth
// within 5. So the alignment is split:
// - rotation: each camera gives an estimate R_true^T R_est of the truth-from-estimate rotation;
//   their chordal L2 mean (the rotation nearest their sum in the Frobenius norm, from an SVD;
//   R. Hartley, J. Trumpf, Y. Dai and H. Li, "Rotation Averaging", IJCV 103, 2013, section 5.3)
//   is taken, then taken again over the cameras within max(3 x median, 1 degree) of the first
//   mean, so one wildly wrong camera does not pull it;
// - scale and translation: least squares on the centers with that rotation fixed (closed form),
//   refitted once over the cameras whose residual is within max(3 x median, 1e-9 x diagonal).
// Deterministic: no random sampling.
//
// After alignment, per camera:
// - rotation error: the angle between the true camera rotation and the estimated one expressed
//   in the truth frame (R_est * R_sim^-1), in degrees;
// - position error: the distance between the true center and the aligned estimated center, as a
//   percentage of the diagonal.
// The same Sim3 maps the reconstructed mesh into the truth frame for SurfaceMetrics.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>Pose error of a set of cameras after Sim3 alignment.</summary>
public sealed record PoseScores
{
	/// <summary>Whether an alignment was found (at least three cameras, not degenerate).</summary>
	public bool Aligned { get; init; }

	/// <summary>Maps the reconstruction's frame into the truth's.</summary>
	public Sim3d TruthFromEstimate { get; init; } = new();

	/// <summary>Number of cameras compared.</summary>
	public int NumPoses { get; init; }

	/// <summary>Median rotation error, degrees (NaN when not aligned).</summary>
	public double MedianRotationDeg { get; init; } = double.NaN;

	/// <summary>Largest rotation error, degrees (NaN when not aligned).</summary>
	public double MaxRotationDeg { get; init; } = double.NaN;

	/// <summary>Median camera position error, % of the diagonal (NaN when not aligned).</summary>
	public double MedianPositionPct { get; init; } = double.NaN;

	/// <summary>Largest camera position error, % of the diagonal (NaN when not aligned).</summary>
	public double MaxPositionPct { get; init; } = double.NaN;
}

/// <summary>Camera pose error after a Sim3 alignment (rotation from orientations, scale and translation from centers).</summary>
public static class PoseMetrics
{
	/// <summary>
	/// Aligns <paramref name="estimated"/> to <paramref name="truth"/> (the same cameras, in the
	/// same order, as CamFromWorld) and measures the remaining error. Distances are in % of
	/// <paramref name="diagonal"/>. Needs at least three cameras.
	/// </summary>
	public static PoseScores Compute(IReadOnlyList<Rigid3d> estimated, IReadOnlyList<Rigid3d> truth, double diagonal)
	{
		Check.Eq(estimated.Count, truth.Count);
		Check.That(diagonal > 0);
		int n = estimated.Count;
		if (n < 3 || !TryAlign(estimated, truth, diagonal, out Sim3d truthFromEstimate))
		{
			return new PoseScores { NumPoses = n };
		}

		Quaterniond estimateFromTruth = truthFromEstimate.Rotation.Inverse();
		var rotationDeg = new double[n];
		var positionPct = new double[n];
		for (int i = 0; i < n; i++)
		{
			Quaterniond estimatedInTruth = estimated[i].Rotation * estimateFromTruth;
			rotationDeg[i] = truth[i].Rotation.AngularDistance(estimatedInTruth) * (180 / Math.PI);
			positionPct[i] = 100 * ((truthFromEstimate * estimated[i].TgtOriginInSrc()) - truth[i].TgtOriginInSrc()).Norm / diagonal;
		}

		return new PoseScores
		{
			Aligned = true,
			TruthFromEstimate = truthFromEstimate,
			NumPoses = n,
			MedianRotationDeg = Median(rotationDeg),
			MaxRotationDeg = rotationDeg.Max(),
			MedianPositionPct = Median(positionPct),
			MaxPositionPct = positionPct.Max(),
		};
	}

	/// <summary>
	/// The truth-from-estimate similarity: rotation from the orientations, then scale and
	/// translation from the centers (see the file header). False when it is degenerate (all
	/// estimated centers coincide, or the scale comes out non-positive).
	/// </summary>
	internal static bool TryAlign(IReadOnlyList<Rigid3d> estimated, IReadOnlyList<Rigid3d> truth, double diagonal, out Sim3d truthFromEstimate)
	{
		truthFromEstimate = new Sim3d();
		int n = estimated.Count;
		var perCamera = new Quaterniond[n];
		for (int i = 0; i < n; i++)
		{
			perCamera[i] = truth[i].Rotation.Inverse() * estimated[i].Rotation;
		}

		bool[] all = [.. Enumerable.Repeat(true, n)];
		if (!TryChordalMean(perCamera, all, out Quaterniond rotation))
		{
			return false;
		}

		Quaterniond first = rotation;
		double[] angles = [.. perCamera.Select(q => q.AngularDistance(first))];
		double angleCut = Math.Max(3 * Median(angles), Math.PI / 180);
		if (!TryChordalMean(perCamera, [.. angles.Select(a => a <= angleCut)], out rotation))
		{
			return false;
		}

		var src = new Vector3d[n];
		var tgt = new Vector3d[n];
		for (int i = 0; i < n; i++)
		{
			src[i] = rotation * estimated[i].TgtOriginInSrc();
			tgt[i] = truth[i].TgtOriginInSrc();
		}

		if (!TryScaleTranslation(src, tgt, all, out double scale, out Vector3d translation))
		{
			return false;
		}

		var residuals = new double[n];
		for (int i = 0; i < n; i++)
		{
			residuals[i] = ((scale * src[i]) + translation - tgt[i]).Norm;
		}

		double residualCut = Math.Max(3 * Median(residuals), 1e-9 * diagonal);
		bool[] inliers = [.. residuals.Select(r => r <= residualCut)];
		if (inliers.Count(x => x) >= 3 && !TryScaleTranslation(src, tgt, inliers, out scale, out translation))
		{
			return false;
		}

		if (!(scale > 0))
		{
			return false;
		}

		truthFromEstimate = new Sim3d(scale, rotation, translation);
		return true;
	}

	/// <summary>The median (the mean of the middle two for an even count).</summary>
	internal static double Median(double[] values)
	{
		double[] sorted = [.. values];
		Array.Sort(sorted);
		int m = sorted.Length / 2;
		return sorted.Length % 2 == 1 ? sorted[m] : 0.5 * (sorted[m - 1] + sorted[m]);
	}

	// The chordal L2 mean of the selected rotations, as a quaternion.
	private static bool TryChordalMean(Quaterniond[] rotations, bool[] use, out Quaterniond mean)
	{
		mean = Quaterniond.Identity;
		if (!TryChordalMeanMatrix(rotations, use, out Matrix3d matrix))
		{
			return false;
		}

		mean = Quaterniond.FromRotationMatrix(matrix).Normalized();
		return true;
	}

	/// <summary>
	/// The chordal L2 mean of the selected rotations as a matrix: U diag(1, 1, det(U V^T)) V^T
	/// for the SVD U S V^T of their summed rotation matrices. The det(U V^T) factor keeps the
	/// result a proper rotation (det +1) when the sum's nearest orthogonal matrix is a
	/// reflection (e.g. Rx(pi) + Ry(pi) + Rz(pi) = -I); without it U V^T would be -I.
	/// </summary>
	internal static bool TryChordalMeanMatrix(Quaterniond[] rotations, bool[] use, out Matrix3d mean)
	{
		mean = Matrix3d.Identity;
		Matrix3d sum = Matrix3d.Zero;
		int count = 0;
		for (int i = 0; i < rotations.Length; i++)
		{
			if (use[i])
			{
				sum += rotations[i].ToRotationMatrix();
				count++;
			}
		}

		if (count == 0)
		{
			return false;
		}

		Svd3d svd = Svd3d.Compute(sum);
		if (svd.Info != ComputationInfo.Success)
		{
			return false;
		}

		Matrix3d u = svd.MatrixU, vt = svd.MatrixV.Transpose();
		double d = (u * vt).Determinant() < 0 ? -1 : 1;
		mean = u * Matrix3d.FromDiagonal(new Vector3d(1, 1, d)) * vt;
		return true;
	}

	// Least squares of |s src + t - tgt|^2 over the selected pairs (the rotation is already in
	// src): s = sum((src - mean_src) . (tgt - mean_tgt)) / sum(|src - mean_src|^2).
	private static bool TryScaleTranslation(Vector3d[] src, Vector3d[] tgt, bool[] use, out double scale, out Vector3d translation)
	{
		scale = 0;
		translation = Vector3d.Zero;
		Vector3d meanSrc = Vector3d.Zero, meanTgt = Vector3d.Zero;
		int count = 0;
		for (int i = 0; i < src.Length; i++)
		{
			if (use[i])
			{
				meanSrc += src[i];
				meanTgt += tgt[i];
				count++;
			}
		}

		if (count == 0)
		{
			return false;
		}

		meanSrc /= count;
		meanTgt /= count;
		double num = 0, den = 0;
		for (int i = 0; i < src.Length; i++)
		{
			if (use[i])
			{
				Vector3d a = src[i] - meanSrc;
				num += a.Dot(tgt[i] - meanTgt);
				den += a.SquaredNorm;
			}
		}

		if (!(den > 0))
		{
			return false;
		}

		scale = num / den;
		translation = meanTgt - (scale * meanSrc);
		return true;
	}
}
