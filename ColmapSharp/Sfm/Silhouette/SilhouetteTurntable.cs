// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouetteTurntable: the turntable prior for silhouette pose registration
// (docs/QUALITY_PLAN.md, stage 5). Not a COLMAP port. An object turning in front of a near-static
// camera puts the camera centres, in the object-fixed world, on a circle about the rotation axis.
// Fit fits that circle to the registered frames (plane by the smallest-eigenvalue direction of the
// centres' scatter, then Kasa's algebraic circle fit in the plane with one 3-MAD trim) and reports
// how turntable-like the motion is; a poor fit disables the prior with a reason.
//
// PoseAt gives a turntable pose at angle theta: a reference registered frame turned about the
// axis (through the circle centre) from its own angle to theta, which keeps that frame's tilt
// and viewing direction relative to the axis. SearchStarts scores every ThetaStepDegrees at a
// coarse pyramid level by silhouette IoU against the hull and returns the best local peaks, so
// a frame whose neighbours give no usable start (the mouse's top/side sweep) gets candidates
// from all the way round, including the 180-degree twin a symmetric silhouette always has.
// SilhouettePoseRegistration.Turntable.cs refines the candidates and picks between them.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>A fitted turntable circle, or why there is none.</summary>
/// <param name="Valid">Whether the motion is turntable-like enough to use.</param>
/// <param name="Reason">Why it is not; empty when valid.</param>
/// <param name="Axis">Unit rotation axis (the circle's plane normal).</param>
/// <param name="Center">The circle centre.</param>
/// <param name="Radius">The circle radius.</param>
/// <param name="RadialResidual">RMS of (distance to centre - radius) / radius over the kept centres.</param>
/// <param name="PlanarResidual">RMS distance off the plane / radius over the kept centres.</param>
/// <param name="ArcDegrees">The angle the kept centres span round the circle.</param>
public sealed record TurntableFit(
	bool Valid, string Reason, Vector3d Axis, Vector3d Center, double Radius, double RadialResidual, double PlanarResidual, double ArcDegrees)
{
	/// <summary>Unit vector in the plane: angle 0.</summary>
	public Vector3d U => AnyPerpendicular(Axis);

	/// <summary>Unit vector in the plane: angle 90 degrees.</summary>
	public Vector3d V => Axis.Cross(U);

	/// <summary>The angle (radians) of a camera centre round the circle.</summary>
	public double AngleOf(Vector3d center)
	{
		Vector3d d = center - Center;
		return Math.Atan2(d.Dot(V), d.Dot(U));
	}

	/// <summary>
	/// <paramref name="reference"/> (a registered camFromWorld) turned about the axis so its
	/// camera centre sits at angle <paramref name="theta"/> (radians).
	/// </summary>
	public Rigid3d PoseAt(Rigid3d reference, double theta)
	{
		double turn = theta - AngleOf(reference.TgtOriginInSrc());
		Quaterniond q = Quaterniond.FromAngleAxis(new AngleAxisd(turn, Axis));
		// T: X -> q (X - c) + c moves the reference camera to theta; the new camera sees
		// X the way the reference saw T^-1 X.
		var worldFromTurned = new Rigid3d(q, Center - q * Center);
		return reference * worldFromTurned.Inverse();
	}

	internal static Vector3d AnyPerpendicular(Vector3d axis)
	{
		Vector3d seed = Math.Abs(axis.X) < 0.9 ? new Vector3d(1, 0, 0) : new Vector3d(0, 1, 0);
		return axis.Cross(seed).Normalized();
	}
}

/// <summary>Fitting and searching the turntable prior.</summary>
public static class SilhouetteTurntable
{
	/// <summary>Fits are rejected above this radial or planar residual (fraction of the radius).</summary>
	public const double MaxResidual = 0.15;

	/// <summary>Fits are rejected when the registered centres span less than this arc (degrees).</summary>
	public const double MinArcDegrees = 20;

	/// <summary>Fits the turntable circle to camera centres (camFromWorld of registered frames).</summary>
	public static TurntableFit Fit(IReadOnlyList<Rigid3d> camFromWorld)
	{
		var centers = camFromWorld.Select(p => p.TgtOriginInSrc()).ToList();
		var none = new Vector3d(0, 0, 1);
		if (centers.Count < 4)
		{
			return new TurntableFit(false, $"only {centers.Count} registered frames (need 4)", none, default, 0, double.NaN, double.NaN, 0);
		}

		(Vector3d axis, Vector3d center, double radius) = FitOnce(centers);
		List<double> radial = centers.Select(c => Math.Abs(InPlane(c, axis, center).Norm - radius)).ToList();
		double[] sorted = [.. radial.OrderBy(r => r)];
		double mad = sorted[sorted.Length / 2];
		var kept = centers.Where((c, i) => radial[i] <= Math.Max(3 * mad, 1e-12 * radius)).ToList();
		if (kept.Count >= 4 && kept.Count < centers.Count)
		{
			(axis, center, radius) = FitOnce(kept);
		}

		double radialRms = Math.Sqrt(kept.Average(c => Square((InPlane(c, axis, center).Norm - radius) / radius)));
		double planarRms = Math.Sqrt(kept.Average(c => Square((c - center).Dot(axis) / radius)));
		var fit = new TurntableFit(true, "", axis, center, radius, radialRms, planarRms, 0);
		double[] angles = [.. kept.Select(fit.AngleOf).OrderBy(a => a)];
		double largestGap = 2 * Math.PI - (angles[^1] - angles[0]);
		for (int i = 1; i < angles.Length; i++)
		{
			largestGap = Math.Max(largestGap, angles[i] - angles[i - 1]);
		}

		double arc = (2 * Math.PI - largestGap) * 180 / Math.PI;
		fit = fit with { ArcDegrees = arc };
		string reason =
			!double.IsFinite(radius) || radius <= 0 ? "the camera centres do not fit a circle"
			: radialRms > MaxResidual ? $"camera centres are {radialRms:P0} of the radius off the circle (limit {MaxResidual:P0})"
			: planarRms > MaxResidual ? $"camera centres are {planarRms:P0} of the radius off the plane (limit {MaxResidual:P0})"
			: arc < MinArcDegrees ? $"the registered frames span only {arc:F0} degrees of the turn"
			: "";
		return fit with { Valid = reason.Length == 0, Reason = reason };
	}

	/// <summary>
	/// Candidate starts for a frame: the <paramref name="count"/> best local IoU peaks of the
	/// turntable pose round the full circle, turned from <paramref name="reference"/>, scored at
	/// pyramid level <paramref name="level"/>. Each is (pose, coarse IoU), best first.
	/// </summary>
	public static List<(Rigid3d Pose, double Iou)> SearchStarts(
		TurntableFit fit, Rigid3d reference, Camera camera, Bitmap mask, SilhouetteHullModel hull,
		double stepDegrees = 3, int level = 2, int count = 3)
	{
		int factor = 1 << level;
		BinaryImage levelMask = SilhouetteRaster.Downsample(mask, factor);
		int steps = (int)Math.Round(360 / stepDegrees);
		var scores = new double[steps];
		var poses = new Rigid3d[steps];
		for (int s = 0; s < steps; s++)
		{
			poses[s] = fit.PoseAt(reference, s * 2 * Math.PI / steps);
			scores[s] = SilhouettePoseRefiner.LevelIou(camera, levelMask, poses[s], hull, 1.0 / factor);
		}

		var peaks = new List<int>();
		for (int s = 0; s < steps; s++)
		{
			double before = scores[(s + steps - 1) % steps], after = scores[(s + 1) % steps];
			if (scores[s] > 0 && scores[s] >= before && scores[s] > after)
			{
				peaks.Add(s);
			}
		}

		// Best first; ties keep the lower angle (deterministic).
		return [.. peaks.OrderByDescending(s => scores[s]).ThenBy(s => s).Take(count).Select(s => (poses[s], scores[s]))];
	}

	private static (Vector3d Axis, Vector3d Center, double Radius) FitOnce(List<Vector3d> centers)
	{
		var mean = new Vector3d(0, 0, 0);
		foreach (Vector3d c in centers)
		{
			mean += c;
		}

		mean /= centers.Count;
		var scatter = new MatrixXd(3, 3);
		foreach (Vector3d c in centers)
		{
			Vector3d d = c - mean;
			for (int a = 0; a < 3; a++)
			{
				for (int b = 0; b < 3; b++)
				{
					scatter[a, b] += d[a] * d[b];
				}
			}
		}

		MatrixXd vectors = new SelfAdjointEigenSolver(scatter).Eigenvectors();
		var axis = new Vector3d(vectors[0, 0], vectors[1, 0], vectors[2, 0]).Normalized();
		Vector3d u = TurntableFit.AnyPerpendicular(axis), v = axis.Cross(u);

		// Kasa: x^2 + y^2 + D x + E y + F = 0, least squares on (D, E, F).
		var ata = new double[3, 3];
		var atb = new double[3];
		foreach (Vector3d c in centers)
		{
			double x = (c - mean).Dot(u), y = (c - mean).Dot(v);
			double[] row = [x, y, 1];
			double rhs = -(x * x + y * y);
			for (int a = 0; a < 3; a++)
			{
				atb[a] += row[a] * rhs;
				for (int b = 0; b < 3; b++)
				{
					ata[a, b] += row[a] * row[b];
				}
			}
		}

		double[]? solution = Solve3(ata, atb);
		if (solution is null)
		{
			return (axis, mean, double.PositiveInfinity);
		}

		double cx = -solution[0] / 2, cy = -solution[1] / 2;
		double radius = Math.Sqrt(Math.Max(0, cx * cx + cy * cy - solution[2]));
		return (axis, mean + u * cx + v * cy, radius);
	}

	private static Vector3d InPlane(Vector3d c, Vector3d axis, Vector3d center)
	{
		Vector3d d = c - center;
		return d - axis * d.Dot(axis);
	}

	private static double Square(double x) => x * x;

	// Cramer's rule; null when singular.
	private static double[]? Solve3(double[,] m, double[] b)
	{
		double Det(double[,] a) =>
			a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
			- a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
			+ a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]);
		double det = Det(m);
		if (Math.Abs(det) < 1e-300)
		{
			return null;
		}

		var x = new double[3];
		for (int col = 0; col < 3; col++)
		{
			var mc = (double[,])m.Clone();
			for (int r = 0; r < 3; r++)
			{
				mc[r, col] = b[r];
			}

			x[col] = Det(mc) / det;
		}

		return x;
	}
}
