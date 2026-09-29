// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePoseRefiner: refines one camera pose so the projected visual hull matches the frame's
// foreground mask (docs/QUALITY_PLAN.md, stage 4b). Not a COLMAP port. The idea, a pose is right
// when the object's silhouettes agree, is silhouette coherence (Hernandez, Schmitt and Cipolla,
// "Silhouette Coherence for Camera Calibration under Circular Motion", PAMI 2007), written from
// the paper; the cost here is a symmetric contour distance rather than the paper's area ratio,
// because a distance has a gradient far from the answer.
//
// The cost, per pyramid level (coarse to fine, level L at 1/2^L size):
// - Hull contour to mask: the hull mesh is rasterized at the current pose; each pixel of its
//   contour takes the nearest projected hull vertex as its contour generator (a 3D point).
//   Residual: the mask's signed distance transform at the generator's projection (zero on the
//   mask contour, positive outside it).
// - Mask contour to hull: each mask contour pixel is paired with the nearest projected
//   generator. Residual: the 2D offset between them. Pairs further than three times the median
//   pair distance (and at least 3 pixels) are dropped, which is what keeps a bitten mask from
//   dragging the pose into the bite.
// Both terms are normalized by their counts so neither side dominates, and are Huber weighted
// (HuberPixels).
// - Motion prior: the rotation and camera centre away from the start, over sigmas
//   (PriorRotationDegrees, PriorCenterFraction) scaled by how far the start was interpolated.
//   Measured on the DarkObject with every other frame registered: without it the pose slid
//   along a valley where the silhouette barely changes (a 2-4 degree swing with a matching
//   shift, hull IoU still 0.99) to a median 2.1 degrees / 7.4% of the diagonal, worse than the
//   0.4 degree start; with it, 0.3 degrees / 1.3%. The generators and pairs are fixed within a round, as in ICP, so a round's
// cost is smooth in the pose; each round then takes one Levenberg-Marquardt step (numeric
// central-difference Jacobian, Marquardt damping, 6x6 normal equations solved by Gaussian
// elimination with partial pivoting). The pose update is a left perturbation in the camera
// frame: x' = exp(w) x + depth * d, with depth the hull centroid's depth, so all six
// parameters are on a radian scale.
//
// Determinism: every loop is sequential with a fixed order, so a frame's result does not depend
// on what else runs in parallel (SilhouettePoseRegistration parallelizes over frames).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>Silhouette-coherence refinement of a single camera pose against a hull.</summary>
public static class SilhouettePoseRefiner
{
	// Residual for a generator that falls behind the camera: far outside any mask.
	private const double BehindCameraResidual = 50.0;

	// Convergence (see Refine): rounds without a relative cost improvement of StallImprovement.
	private const double StallImprovement = 0.01;
	private const int StallRounds = 3;

	// A step that moves the image by less than this many level pixels is below the contours'
	// quantization noise.
	private const double MinStepPixels = 0.05;

	// Numeric differentiation step on the radian-scaled parameters.
	private const double JacobianStep = 1e-6;

	/// <summary>
	/// Refines <paramref name="initialCamFromWorld"/> so <paramref name="hull"/>, seen by
	/// <paramref name="camera"/>, matches <paramref name="mask"/> (the camera's size).
	/// <paramref name="priorScale"/> multiplies the motion prior's sigmas: the further the start
	/// was interpolated, the less it is trusted.
	/// </summary>
	public static SilhouettePoseRefinement Refine(
		Camera camera,
		Bitmap mask,
		Rigid3d initialCamFromWorld,
		SilhouetteHullModel hull,
		SilhouettePoseOptions options,
		double priorScale = 1.0,
		CancellationToken cancellationToken = default)
	{
		Check.That(priorScale > 0);
		Check.That(mask.Width == camera.Width && mask.Height == camera.Height, "Mask and camera sizes differ");
		BinaryImage fullMask = SilhouetteRaster.Downsample(mask, 1);
		double initialIou = Iou(camera, fullMask, initialCamFromWorld, hull);
		Rigid3d pose = initialCamFromWorld;
		int iterations = 0;
		for (int level = options.PyramidLevels - 1; level >= 0; level--)
		{
			cancellationToken.ThrowIfCancellationRequested();
			var state = new LevelState(camera, mask, hull, level, options, initialCamFromWorld, priorScale);
			if (state.MaskContour.Count == 0)
			{
				continue;
			}

			double lambda = 1e-3;
			double bestCost = double.PositiveInfinity;
			int stalled = 0;
			for (int round = 0; round < options.MaxIterationsPerLevel; round++)
			{
				if (!state.Correspond(pose))
				{
					break;
				}

				(Rigid3d next, bool stepped, bool converged, double cost) = state.Step(pose, ref lambda);
				if (stepped)
				{
					pose = next;
					iterations++;
				}

				// Re-picking the contour correspondences each round moves the cost by a few
				// percent even at the optimum (contours are pixel-quantized), so the pose jitters
				// by a tenth of a pixel forever. Stop once a round's starting cost has not beaten
				// the best by StallImprovement for StallRounds rounds in a row.
				if (cost < bestCost * (1 - StallImprovement))
				{
					bestCost = cost;
					stalled = 0;
				}
				else if (++stalled >= StallRounds)
				{
					break;
				}

				if (converged || !stepped)
				{
					break;
				}
			}
		}

		return new SilhouettePoseRefinement(pose, initialIou, Iou(camera, fullMask, pose, hull), iterations);
	}

	/// <summary>The full-size silhouette IoU of <paramref name="hull"/> at <paramref name="camFromWorld"/> with <paramref name="mask"/>.</summary>
	public static double Iou(Camera camera, Bitmap mask, Rigid3d camFromWorld, SilhouetteHullModel hull) =>
		Iou(camera, SilhouetteRaster.Downsample(mask, 1), camFromWorld, hull);

	/// <summary>The IoU against a mask already at pyramid scale <paramref name="scale"/> (for coarse searches).</summary>
	internal static double LevelIou(Camera camera, BinaryImage levelMask, Rigid3d camFromWorld, SilhouetteHullModel hull, double scale)
	{
		(double[] px, double[] py) = Project(camera, camFromWorld, hull.Vertices, scale);
		return SilhouetteRaster.Iou(SilhouetteRaster.Fill(levelMask.Width, levelMask.Height, px, py, hull.Triangles), levelMask);
	}

	private static double Iou(Camera camera, BinaryImage mask, Rigid3d camFromWorld, SilhouetteHullModel hull)
	{
		(double[] px, double[] py) = Project(camera, camFromWorld, hull.Vertices, 1.0);
		return SilhouetteRaster.Iou(SilhouetteRaster.Fill(mask.Width, mask.Height, px, py, hull.Triangles), mask);
	}

	/// <summary><paramref name="pose"/> moved by the camera-frame perturbation <paramref name="xi"/> (rotation vector, then translation over depth).</summary>
	internal static Rigid3d Perturb(Rigid3d pose, ReadOnlySpan<double> xi, double depth)
	{
		var w = new Vector3d(xi[0], xi[1], xi[2]);
		double angle = w.Norm;
		Quaterniond dq = angle == 0 ? Quaterniond.Identity : Quaterniond.FromAngleAxis(new AngleAxisd(angle, w / angle));
		var dt = new Vector3d(xi[3] * depth, xi[4] * depth, xi[5] * depth);
		return new Rigid3d((dq * pose.Rotation).Normalized(), dq * pose.Translation + dt);
	}

	// Projections in level pixels (full-size pixels times scale); NaN where behind the camera.
	private static (double[] X, double[] Y) Project(Camera camera, Rigid3d camFromWorld, Vector3d[] points, double scale)
	{
		var px = new double[points.Length];
		var py = new double[points.Length];
		for (int i = 0; i < points.Length; i++)
		{
			Vector2d? p = camera.ImgFromCam(camFromWorld * points[i]);
			px[i] = p is Vector2d q ? q.X * scale : double.NaN;
			py[i] = p is Vector2d r ? r.Y * scale : double.NaN;
		}

		return (px, py);
	}

	// Takes every stride-th item so at most `max` remain, in order.
	private static List<T> Subsample<T>(List<T> items, int max)
	{
		if (items.Count <= max)
		{
			return items;
		}

		var result = new List<T>(max);
		for (int i = 0; i < max; i++)
		{
			result.Add(items[(int)((long)i * items.Count / max)]);
		}

		return result;
	}

	private static double HuberWeight(double r, double delta)
	{
		double a = Math.Abs(r);
		return a <= delta ? 1.0 : delta / a;
	}

	// One pyramid level: the mask's distance transform and contour, and the round's
	// correspondences.
	private sealed class LevelState
	{
		private readonly Camera _camera;
		private readonly SilhouetteHullModel _hull;
		private readonly SilhouettePoseOptions _options;
		private readonly double _scale;
		private readonly int _width;
		private readonly int _height;
		private readonly float[] _signedDistance;
		private readonly double _huber;
		private Vector3d[] _generators = [];
		private double[] _generatorWeights = [];
		private readonly List<(int Generator, double X, double Y, double Weight)> _pairs = [];
		private double _depth = 1.0;
		private readonly Rigid3d _prior;
		private readonly double _priorScale;
		private readonly Vector3d _priorCenter;
		private readonly double _priorDistance;

		public LevelState(Camera camera, Bitmap mask, SilhouetteHullModel hull, int level, SilhouettePoseOptions options, Rigid3d prior, double priorScale)
		{
			_priorScale = priorScale;
			_prior = prior;
			_priorCenter = prior.TgtOriginInSrc();
			_priorDistance = Math.Max(1e-9, (prior * hull.Centroid).Norm);
			_camera = camera;
			_hull = hull;
			_options = options;
			int factor = 1 << level;
			_scale = 1.0 / factor;
			BinaryImage levelMask = SilhouetteRaster.Downsample(mask, factor);
			_width = levelMask.Width;
			_height = levelMask.Height;
			_signedDistance = SilhouetteRaster.SignedDistance(levelMask);
			MaskContour = Subsample(SilhouetteRaster.Contour(levelMask), options.MaxContourPoints);
			_huber = options.HuberPixels;
		}

		public List<(double X, double Y)> MaskContour { get; }

		// Finds the hull's contour generators at `pose` and pairs the mask contour with them.
		// False when the hull has no contour in view.
		public bool Correspond(Rigid3d pose)
		{
			_depth = Math.Max(1e-9, (pose * _hull.Centroid).Z);
			(double[] px, double[] py) = Project(_camera, pose, _hull.Vertices, _scale);
			BinaryImage rendered = SilhouetteRaster.Fill(_width, _height, px, py, _hull.Triangles);
			List<(double X, double Y)> contour = SilhouetteRaster.Contour(rendered);
			if (contour.Count == 0)
			{
				return false;
			}

			var vertexGrid = new PointGrid(px, py, 4.0);
			var seen = new HashSet<int>();
			var generatorIds = new List<int>();
			foreach ((double x, double y) in contour)
			{
				int nearest = vertexGrid.Nearest(x, y, out _);
				// A generator that projects off the image or onto its border row or column is where
				// the image cuts the hull, not an object contour (Contour skips those pixels too);
				// its signed distance there would be a spurious residual.
				if (nearest >= 0 && px[nearest] >= 1 && py[nearest] >= 1 && px[nearest] < _width - 1 && py[nearest] < _height - 1
					&& seen.Add(nearest))
				{
					generatorIds.Add(nearest);
				}
			}

			generatorIds = Subsample(generatorIds, _options.MaxContourPoints);
			if (generatorIds.Count == 0)
			{
				return false;
			}

			_generators = new Vector3d[generatorIds.Count];
			_generatorWeights = new double[generatorIds.Count];
			var gx = new double[generatorIds.Count];
			var gy = new double[generatorIds.Count];
			for (int i = 0; i < generatorIds.Count; i++)
			{
				int v = generatorIds[i];
				_generators[i] = _hull.Vertices[v];
				gx[i] = px[v];
				gy[i] = py[v];
				double r = SilhouetteRaster.SampleBilinear(_signedDistance, _width, _height, gx[i], gy[i]);
				_generatorWeights[i] = HuberWeight(r, _huber) / generatorIds.Count;
			}

			var generatorGrid = new PointGrid(gx, gy, 4.0);
			var candidates = new List<(int Generator, double X, double Y, double Distance)>(MaskContour.Count);
			foreach ((double x, double y) in MaskContour)
			{
				int nearest = generatorGrid.Nearest(x, y, out double distance);
				if (nearest >= 0)
				{
					candidates.Add((nearest, x, y, distance));
				}
			}

			_pairs.Clear();
			if (candidates.Count > 0)
			{
				double[] distances = candidates.Select(c => c.Distance).ToArray();
				Array.Sort(distances);
				double gate = Math.Max(3.0, 3.0 * distances[distances.Length / 2]);
				int kept = candidates.Count(c => c.Distance <= gate);
				foreach ((int g, double x, double y, double d) in candidates)
				{
					if (d <= gate)
					{
						_pairs.Add((g, x, y, HuberWeight(d, _huber) / kept));
					}
				}
			}

			return true;
		}

		// One Levenberg-Marquardt step from `pose` on this round's correspondences.
		public (Rigid3d Pose, bool Stepped, bool Converged, double Cost) Step(Rigid3d pose, ref double lambda)
		{
			int m = _generators.Length + 2 * _pairs.Count + 6;
			var r0 = new double[m];
			Residuals(pose, [0, 0, 0, 0, 0, 0], r0);
			double cost0 = SquaredNorm(r0);
			var jacobian = new double[6, m];
			var plus = new double[m];
			var minus = new double[m];
			Span<double> xi = stackalloc double[6];
			for (int p = 0; p < 6; p++)
			{
				xi.Clear();
				xi[p] = JacobianStep;
				Residuals(pose, xi, plus);
				xi[p] = -JacobianStep;
				Residuals(pose, xi, minus);
				for (int i = 0; i < m; i++)
				{
					jacobian[p, i] = (plus[i] - minus[i]) / (2 * JacobianStep);
				}
			}

			var jtj = new double[6, 6];
			var jtr = new double[6];
			for (int a = 0; a < 6; a++)
			{
				for (int i = 0; i < m; i++)
				{
					jtr[a] += jacobian[a, i] * r0[i];
				}

				for (int b = 0; b < 6; b++)
				{
					double sum = 0;
					for (int i = 0; i < m; i++)
					{
						sum += jacobian[a, i] * jacobian[b, i];
					}

					jtj[a, b] = sum;
				}
			}

			var trial = new double[m];
			for (int attempt = 0; attempt < 8; attempt++)
			{
				double[]? delta = SolveDamped(jtj, jtr, lambda);
				if (delta is null)
				{
					lambda *= 10;
					continue;
				}

				Rigid3d candidate = Perturb(pose, delta, _depth);
				Residuals(candidate, [0, 0, 0, 0, 0, 0], trial);
				if (SquaredNorm(trial) < cost0)
				{
					lambda = Math.Max(1e-9, lambda / 3);
					double stepNorm = Math.Sqrt(delta.Sum(d => d * d));
					bool converged = stepNorm < MinStepPixels / (_scale * _camera.MeanFocalLength());
					return (candidate, true, converged, cost0);
				}

				lambda *= 4;
			}

			return (pose, false, true, cost0);
		}

		private void Residuals(Rigid3d pose, ReadOnlySpan<double> xi, double[] output)
		{
			Rigid3d q = Perturb(pose, xi, _depth);
			var gx = new double[_generators.Length];
			var gy = new double[_generators.Length];
			for (int i = 0; i < _generators.Length; i++)
			{
				Vector2d? p = _camera.ImgFromCam(q * _generators[i]);
				gx[i] = p is Vector2d a ? a.X * _scale : double.NaN;
				gy[i] = p is Vector2d b ? b.Y * _scale : double.NaN;
				double r = double.IsNaN(gx[i])
					? BehindCameraResidual
					: SilhouetteRaster.SampleBilinear(_signedDistance, _width, _height, gx[i], gy[i]);
				output[i] = Math.Sqrt(_generatorWeights[i]) * r;
			}

			int o = _generators.Length;
			for (int j = 0; j < _pairs.Count; j++)
			{
				(int g, double x, double y, double weight) = _pairs[j];
				double w = Math.Sqrt(weight);
				output[o + 2 * j] = double.IsNaN(gx[g]) ? w * BehindCameraResidual : w * (gx[g] - x);
				output[o + 2 * j + 1] = double.IsNaN(gy[g]) ? w * BehindCameraResidual : w * (gy[g] - y);
			}

			// The motion prior: rotation (radians) and camera centre (fraction of the start's
			// distance to the hull) away from the start, over their sigmas.
			o += 2 * _pairs.Count;
			double rotationSigma = _options.PriorRotationDegrees * _priorScale * Math.PI / 180;
			AngleAxisd turn = AngleAxisd.FromQuaternion(q.Rotation * _prior.Rotation.Inverse());
			Vector3d shift = (q.TgtOriginInSrc() - _priorCenter) / (_priorDistance * _options.PriorCenterFraction * _priorScale);
			for (int a = 0; a < 3; a++)
			{
				output[o + a] = double.IsFinite(rotationSigma) && rotationSigma > 0 ? turn.Angle * turn.Axis[a] / rotationSigma : 0;
				output[o + 3 + a] = double.IsFinite(shift[a]) ? shift[a] : 0;
			}
		}

		private static double SquaredNorm(double[] r)
		{
			double sum = 0;
			foreach (double v in r)
			{
				sum += v * v;
			}

			return sum;
		}

		// (JtJ + lambda diag(JtJ)) delta = -Jtr by Gaussian elimination with partial pivoting;
		// null when singular.
		private static double[]? SolveDamped(double[,] jtj, double[] jtr, double lambda)
		{
			var a = new double[6, 7];
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					a[i, j] = jtj[i, j];
				}

				a[i, i] += lambda * Math.Max(jtj[i, i], 1e-12);
				a[i, 6] = -jtr[i];
			}

			for (int col = 0; col < 6; col++)
			{
				int pivot = col;
				for (int row = col + 1; row < 6; row++)
				{
					if (Math.Abs(a[row, col]) > Math.Abs(a[pivot, col]))
					{
						pivot = row;
					}
				}

				if (Math.Abs(a[pivot, col]) < 1e-300)
				{
					return null;
				}

				for (int j = 0; j < 7; j++)
				{
					(a[col, j], a[pivot, j]) = (a[pivot, j], a[col, j]);
				}

				for (int row = col + 1; row < 6; row++)
				{
					double f = a[row, col] / a[col, col];
					for (int j = col; j < 7; j++)
					{
						a[row, j] -= f * a[col, j];
					}
				}
			}

			var x = new double[6];
			for (int i = 5; i >= 0; i--)
			{
				double sum = a[i, 6];
				for (int j = i + 1; j < 6; j++)
				{
					sum -= a[i, j] * x[j];
				}

				x[i] = sum / a[i, i];
			}

			return x.All(double.IsFinite) ? x : null;
		}
	}

	// A uniform bucket grid over 2D points for nearest-neighbour queries. NaN points are left out.
	// Ties go to the lower index, so queries are deterministic.
	private sealed class PointGrid
	{
		private readonly double[] _x;
		private readonly double[] _y;
		private readonly double _cell;
		private readonly Dictionary<long, List<int>> _buckets = [];
		private readonly int _minCellX;
		private readonly int _maxCellX;
		private readonly int _minCellY;
		private readonly int _maxCellY;

		public PointGrid(double[] x, double[] y, double cell)
		{
			_x = x;
			_y = y;
			_cell = cell;
			double minX = double.MaxValue, maxX = double.MinValue, minY = double.MaxValue, maxY = double.MinValue;
			for (int i = 0; i < x.Length; i++)
			{
				if (double.IsNaN(x[i]) || Math.Abs(x[i]) > 1e7 || Math.Abs(y[i]) > 1e7)
				{
					continue;
				}

				long key = Key((int)Math.Floor(x[i] / cell), (int)Math.Floor(y[i] / cell));
				if (!_buckets.TryGetValue(key, out List<int>? list))
				{
					_buckets[key] = list = [];
				}

				list.Add(i);
				minX = Math.Min(minX, x[i]);
				maxX = Math.Max(maxX, x[i]);
				minY = Math.Min(minY, y[i]);
				maxY = Math.Max(maxY, y[i]);
			}

			_minCellX = (int)Math.Floor(minX / cell);
			_maxCellX = (int)Math.Floor(maxX / cell);
			_minCellY = (int)Math.Floor(minY / cell);
			_maxCellY = (int)Math.Floor(maxY / cell);
		}

		// The nearest point's index (-1 if none). Searches square rings of cells outward and
		// stops once the ring is beyond the best distance found.
		public int Nearest(double x, double y, out double distance)
		{
			int cx = (int)Math.Floor(x / _cell), cy = (int)Math.Floor(y / _cell);
			int best = -1;
			double bestSq = double.MaxValue;
			if (_buckets.Count == 0)
			{
				distance = double.PositiveInfinity;
				return -1;
			}

			// Beyond this ring every occupied cell has been visited.
			int maxRing = Math.Max(
				Math.Max(Math.Abs(cx - _minCellX), Math.Abs(cx - _maxCellX)),
				Math.Max(Math.Abs(cy - _minCellY), Math.Abs(cy - _maxCellY)));
			for (int ring = 0; ring <= maxRing; ring++)
			{
				if (best >= 0 && (ring - 1) * _cell > Math.Sqrt(bestSq))
				{
					break;
				}

				// Only the ring's perimeter: the top and bottom rows in full, then the two side
				// columns between them (the whole ring once at ring 0).
				for (int j = cy - ring; j <= cy + ring; j++)
				{
					bool edgeRow = j == cy - ring || j == cy + ring;
					int step = edgeRow ? 1 : Math.Max(1, 2 * ring);
					for (int i = cx - ring; i <= cx + ring; i += step)
					{
						if (!_buckets.TryGetValue(Key(i, j), out List<int>? list))
						{
							continue;
						}

						foreach (int k in list)
						{
							double dx = _x[k] - x, dy = _y[k] - y, sq = dx * dx + dy * dy;
							if (sq < bestSq || (sq == bestSq && k < best))
							{
								bestSq = sq;
								best = k;
							}
						}
					}
				}
			}

			distance = best >= 0 ? Math.Sqrt(bestSq) : double.PositiveInfinity;
			return best;
		}

		private static long Key(int i, int j) => ((long)i << 32) ^ (uint)j;
	}
}
