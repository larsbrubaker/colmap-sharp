// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PyramidalLucasKanade: sparse pyramidal Lucas-Kanade point tracking between two frames, with a
// forward-backward check (docs/QUALITY_PLAN.md stage 2a). Not a COLMAP port; COLMAP has no
// feature tracker. Written from the papers only (no code was read):
// - B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application
//   to Stereo Vision", IJCAI 1981: Gauss-Newton on the window's squared intensity difference.
// - C. Tomasi and T. Kanade, "Detection and Tracking of Point Features", CMU-CS-91-132, 1991.
// - J. Shi and C. Tomasi, "Good Features to Track", CVPR 1994: reject a window whose structure
//   tensor's smaller eigenvalue is too small.
// - J.-Y. Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature Tracker", Intel 2000:
//   coarse to fine, the template's gradient and G matrix fixed per level (inverse compositional
//   in spirit), the guess doubled from level to level.
// - Z. Kalal, K. Mikolajczyk and J. Matas, "Forward-Backward Error: Automatic Detection of
//   Tracking Failures", ICPR 2010: track back from the result and reject a point that does not
//   come home.
// - H. Jin, P. Favaro and S. Soatto, "Real-Time Feature Tracking and Outlier Rejection with
//   Changes in Illumination", ICCV 2001: a gain and an offset between the windows, estimated
//   with the motion (KltOptions.CompensateIllumination).
//
// Coordinates: points in and out are COLMAP's continuous image coordinates (the center of pixel
// (i, j) is (i + 0.5, j + 0.5)). Internally a point is shifted by -0.5 into ImagePyramid's index
// coordinates, where the levels differ by a pure factor of 2.
//
// Determinism: each point is tracked on its own and writes only its own result slot, with every
// sum over its window accumulated in a fixed order in double, so a parallel run gives exactly the
// sequential result. No FMA.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Feature.Tracking;

/// <summary>Why a point was or was not tracked.</summary>
public enum KltStatus
{
	/// <summary>Tracked and passed every check.</summary>
	Tracked,

	/// <summary>The window at level 0 had too little texture (small minimum eigenvalue).</summary>
	LowTexture,

	/// <summary>The result lies outside the next frame.</summary>
	OutOfImage,

	/// <summary>The window's mean absolute intensity difference after tracking was too large.</summary>
	HighResidual,

	/// <summary>Tracking back from the result missed the starting point by too much.</summary>
	ForwardBackwardFailed,
}

/// <summary>Options for <see cref="PyramidalLucasKanade.Track"/>.</summary>
public sealed class KltOptions
{
	/// <summary>Side of the square tracking window in pixels (odd).</summary>
	public int WindowSize { get; set; } = 21;

	/// <summary>Pyramid levels above level 0 to use (0 = no pyramid).</summary>
	public int MaxLevel { get; set; } = 3;

	/// <summary>Gauss-Newton iterations per level at most.</summary>
	public int MaxIterations { get; set; } = 30;

	/// <summary>Stop iterating at a level once the update is shorter than this (pixels of that level).</summary>
	public double Epsilon { get; set; } = 0.01;

	/// <summary>
	/// Smallest accepted minimum eigenvalue of the window's structure tensor divided by the
	/// window area, in (grey levels per pixel) squared. Sensor noise of sigma 1.5 grey levels
	/// alone gives about 0.5.
	/// </summary>
	public double MinEigenThreshold { get; set; } = 0.25;

	/// <summary>Largest accepted mean absolute grey-level difference over the window at level 0.</summary>
	public double MaxMeanResidual { get; set; } = 25;

	/// <summary>
	/// Largest accepted forward-backward round-trip error in pixels; zero or less disables the
	/// check (and halves the work).
	/// </summary>
	public double ForwardBackwardThreshold { get; set; } = 1.0;

	/// <summary>
	/// Also estimate a gain and an offset between the windows (next = gain * prev + offset).
	/// An object turning under a fixed light carries its texture through the light's shading,
	/// so a surface point gets brighter or darker from frame to frame; plain Lucas-Kanade
	/// (and plain SSD) then pulls the point towards a spot of matching brightness instead of
	/// the moved texture. Only level 0 estimates them: a coarse level's few pixels cannot
	/// separate gain from motion (large motions were lost), and level 0 decides the result.
	/// Off by default because the two extra unknowns cost precision where the light does not
	/// change (a pure subpixel translation is recovered within about 0.07 px instead of 0.05);
	/// <see cref="SequenceTrackerOptions"/> turns it on for turning-object video.
	/// </summary>
	public bool CompensateIllumination { get; set; }

	/// <summary>Track points in parallel. The result is identical either way.</summary>
	public bool Parallel { get; set; } = true;
}

/// <summary>The result of tracking one point.</summary>
/// <param name="Position">Where the point is in the next frame (continuous image coordinates).</param>
/// <param name="Status">Whether it was tracked, and why not if not.</param>
/// <param name="Residual">Mean absolute grey-level difference over the window at level 0.</param>
/// <param name="ForwardBackwardError">Round-trip error in pixels (NaN if not measured).</param>
public readonly record struct KltResult(Vector2d Position, KltStatus Status, double Residual, double ForwardBackwardError)
{
	/// <summary>True when <see cref="Status"/> is <see cref="KltStatus.Tracked"/>.</summary>
	public bool IsTracked => Status == KltStatus.Tracked;
}

/// <summary>Pyramidal Lucas-Kanade tracking of sparse points. See the file header.</summary>
public static class PyramidalLucasKanade
{
	private const int MaxWindowSize = 51;

	/// <summary>
	/// Tracks <paramref name="points"/> (continuous image coordinates in the frame of
	/// <paramref name="prev"/>) into the frame of <paramref name="next"/>, then, if the
	/// options enable it, back again for the forward-backward check.
	/// <paramref name="initialGuesses"/>, if given, are where to start looking in the next frame.
	/// </summary>
	public static KltResult[] Track(
		ImagePyramid prev,
		ImagePyramid next,
		IReadOnlyList<Vector2d> points,
		KltOptions options,
		IReadOnlyList<Vector2d>? initialGuesses = null)
	{
		Check.That(options.WindowSize >= 3 && options.WindowSize % 2 == 1 && options.WindowSize <= MaxWindowSize);
		Check.That(options.MaxLevel >= 0 && options.MaxIterations >= 1);
		Check.Eq(prev.Width, next.Width);
		Check.Eq(prev.Height, next.Height);
		if (initialGuesses != null)
		{
			Check.Eq(initialGuesses.Count, points.Count);
		}

		int maxLevel = Math.Min(options.MaxLevel, Math.Min(prev.Levels.Count, next.Levels.Count) - 1);
		bool checkBack = options.ForwardBackwardThreshold > 0;
		var results = new KltResult[points.Count];
		void TrackOne(int i)
		{
			Vector2d start = points[i];
			KltResult forward = TrackPoint(prev, next, start, initialGuesses?[i] ?? start, maxLevel, options);
			if (!forward.IsTracked || !checkBack)
			{
				results[i] = forward with { ForwardBackwardError = double.NaN };
				return;
			}

			// The back track starts from the forward result with no motion guess (Kalal et al.
			// 2010). Starting it at `start` would only test that start is a local minimum, and a
			// forward track that settled on a false minimum would pass.
			KltResult back = TrackPoint(next, prev, forward.Position, forward.Position, maxLevel, options);
			double dx = back.Position.X - start.X, dy = back.Position.Y - start.Y;
			double fb = Math.Sqrt(dx * dx + dy * dy);
			bool ok = back.IsTracked && fb <= options.ForwardBackwardThreshold;
			results[i] = forward with
			{
				Status = ok ? KltStatus.Tracked : KltStatus.ForwardBackwardFailed,
				ForwardBackwardError = back.IsTracked ? fb : double.PositiveInfinity,
			};
		}

		if (options.Parallel)
		{
			System.Threading.Tasks.Parallel.For(0, points.Count, TrackOne);
		}
		else
		{
			for (int i = 0; i < points.Count; ++i)
			{
				TrackOne(i);
			}
		}

		return results;
	}

	// One point, coarse to fine, in index coordinates (continuous minus 0.5).
	private static KltResult TrackPoint(
		ImagePyramid prev, ImagePyramid next, Vector2d point, Vector2d guess, int maxLevel, KltOptions options)
	{
		int half = options.WindowSize / 2;
		int n = options.WindowSize * options.WindowSize;
		Span<float> tmpl = stackalloc float[n];
		Span<float> gx = stackalloc float[n];
		Span<float> gy = stackalloc float[n];
		Span<bool> templateInside = stackalloc bool[n];
		double ux = point.X - 0.5, uy = point.Y - 0.5;
		double scale = 1.0 / (1 << maxLevel);
		double nx = (guess.X - 0.5) * scale, ny = (guess.Y - 0.5) * scale;
		double epsSq = options.Epsilon * options.Epsilon;
		bool lowTexture = false;

		// The photometric model J = gain * T + offset (see CompensateIllumination), carried from
		// level to level: the low-pass keeps a gain and an offset as they are.
		double gain = 1, offset = 0;
		for (int level = maxLevel; level >= 0; --level)
		{
			PyramidLevel lp = prev.Levels[level];
			PyramidLevel ln = next.Levels[level];
			double levelScale = 1.0 / (1 << level);
			double px = ux * levelScale, py = uy * levelScale;

			// Template, its gradient and the structure tensor G, fixed for this level. Window
			// pixels outside the image are left out (their gradient is zeroed): on a coarse level
			// the window can cover much of the image, and clamped border values do not move with
			// the scene, so they would pull the point towards standing still.
			double gxx = 0, gxy = 0, gyy = 0;
			int k = 0, inside = 0;
			for (int dy = -half; dy <= half; ++dy)
			{
				for (int dx = -half; dx <= half; ++dx, ++k)
				{
					double sx = px + dx, sy = py + dy;
					templateInside[k] = sx >= 0 && sx <= lp.Width - 1 && sy >= 0 && sy <= lp.Height - 1;
					if (!templateInside[k])
					{
						tmpl[k] = 0;
						gx[k] = 0;
						gy[k] = 0;
						continue;
					}

					++inside;
					tmpl[k] = lp.Sample(lp.Intensity, sx, sy);
					float ix = lp.Sample(lp.GradX, sx, sy);
					float iy = lp.Sample(lp.GradY, sx, sy);
					gx[k] = ix;
					gy[k] = iy;
					gxx += ix * ix;
					gxy += ix * iy;
					gyy += iy * iy;
				}
			}

			double det = gxx * gyy - gxy * gxy;
			double halfDiff = (gxx - gyy) * 0.5;
			double minEig = ((gxx + gyy) * 0.5 - Math.Sqrt(halfDiff * halfDiff + gxy * gxy)) / Math.Max(inside, 1);
			if (minEig < options.MinEigenThreshold || det <= 0 || 4 * inside < n)
			{
				// Too flat at this level: keep the guess and try the finer one (Bouguet); at
				// level 0 the point is lost.
				if (level == 0)
				{
					lowTexture = true;
					break;
				}

				nx *= 2;
				ny *= 2;
				continue;
			}

			double prevDx = 0, prevDy = 0;
			for (int iter = 0; iter < options.MaxIterations; ++iter)
			{
				double deltaX, deltaY;
				if (options.CompensateIllumination && level == 0)
				{
					(deltaX, deltaY) = GainOffsetStep(ln, tmpl, gx, gy, templateInside, px, py, nx - px, ny - py, half, ref gain, ref offset);
				}
				else
				{
					// G and b over the same pixels: those whose template and target both lie
					// inside the images. Away from the border that is the whole window and G is
					// the one summed above; near it, G shrinks with b so the step stays a
					// least-squares step.
					double bx = 0, by = 0, hxx = 0, hxy = 0, hyy = 0;
					k = 0;
					double ox = nx - px, oy = ny - py;
					for (int dy = -half; dy <= half; ++dy)
					{
						for (int dx = -half; dx <= half; ++dx, ++k)
						{
							double jx = px + dx + ox, jy = py + dy + oy;
							if (!templateInside[k]
								|| !(jx >= 0 && jx <= ln.Width - 1 && jy >= 0 && jy <= ln.Height - 1))
							{
								continue;
							}

							float j = ln.Sample(ln.Intensity, jx, jy);
							double diff = tmpl[k] - j;
							float ix = gx[k], iy = gy[k];
							bx += diff * ix;
							by += diff * iy;
							hxx += ix * ix;
							hxy += ix * iy;
							hyy += iy * iy;
						}
					}

					double stepDet = hxx * hyy - hxy * hxy;
					if (!(stepDet > 0))
					{
						break;
					}

					deltaX = (hyy * bx - hxy * by) / stepDet;
					deltaY = (hxx * by - hxy * bx) / stepDet;
				}

				nx += deltaX;
				ny += deltaY;
				if (deltaX * deltaX + deltaY * deltaY <= epsSq)
				{
					break;
				}

				// Oscillating between two positions: settle in the middle.
				double sumX = deltaX + prevDx, sumY = deltaY + prevDy;
				if (iter > 0 && sumX * sumX + sumY * sumY <= epsSq)
				{
					nx -= deltaX * 0.5;
					ny -= deltaY * 0.5;
					break;
				}

				prevDx = deltaX;
				prevDy = deltaY;
			}

			if (level > 0)
			{
				nx *= 2;
				ny *= 2;
			}
		}

		var position = new Vector2d(nx + 0.5, ny + 0.5);
		if (lowTexture)
		{
			return new KltResult(position, KltStatus.LowTexture, double.NaN, double.NaN);
		}

		if (!(position.X >= 0 && position.X < prev.Width && position.Y >= 0 && position.Y < prev.Height))
		{
			return new KltResult(position, KltStatus.OutOfImage, double.NaN, double.NaN);
		}

		double residual = MeanResidual(prev.Levels[0], next.Levels[0], ux, uy, nx, ny, half, gain, offset);
		KltStatus status = residual <= options.MaxMeanResidual ? KltStatus.Tracked : KltStatus.HighResidual;
		return new KltResult(position, status, residual, double.NaN);
	}

	// One Gauss-Newton step of the illumination-compensated model J(u + d) = gain * T(u) + offset
	// (Jin, Favaro and Soatto 2001, with the template's gradient standing in for the next frame's,
	// as in the plain step: at the solution grad J = gain * grad T). The residual
	// e = J - gain * T - offset is linearized in (delta d, delta gain, delta offset) with the row
	// (gain * gx, gain * gy, -T, -1) and the 4x4 normal equations are solved directly. Updates
	// gain and offset in place and returns delta d. A window whose template is too flat to
	// separate gain from offset from motion (singular system) takes the plain translation step.
	private static (double X, double Y) GainOffsetStep(
		PyramidLevel ln,
		ReadOnlySpan<float> tmpl,
		ReadOnlySpan<float> gx,
		ReadOnlySpan<float> gy,
		ReadOnlySpan<bool> templateInside,
		double px,
		double py,
		double ox,
		double oy,
		int half,
		ref double gain,
		ref double offset)
	{
		Span<double> h = stackalloc double[16];
		Span<double> r = stackalloc double[4];
		Span<double> a = stackalloc double[4];
		h.Clear();
		r.Clear();
		int k = 0;
		for (int dy = -half; dy <= half; ++dy)
		{
			for (int dx = -half; dx <= half; ++dx, ++k)
			{
				double jx = px + dx + ox, jy = py + dy + oy;
				if (!templateInside[k]
					|| !(jx >= 0 && jx <= ln.Width - 1 && jy >= 0 && jy <= ln.Height - 1))
				{
					continue;
				}

				double t = tmpl[k];
				double e = ln.Sample(ln.Intensity, jx, jy) - gain * t - offset;
				a[0] = gain * gx[k];
				a[1] = gain * gy[k];
				a[2] = -t;
				a[3] = -1;
				for (int row = 0; row < 4; ++row)
				{
					r[row] -= a[row] * e;
					for (int col = 0; col < 4; ++col)
					{
						h[row * 4 + col] += a[row] * a[col];
					}
				}
			}
		}

		Span<double> step = stackalloc double[4];
		if (SolveSymmetric4(h, r, step))
		{
			gain += step[2];
			offset += step[3];
			return (step[0], step[1]);
		}

		// Translation only, gain and offset held.
		double det = h[0] * h[5] - h[1] * h[4];
		if (!(det > 0))
		{
			return (0, 0);
		}

		return ((h[5] * r[0] - h[1] * r[1]) / det, (h[0] * r[1] - h[4] * r[0]) / det);
	}

	// Solves the 4x4 system m x = b by Gaussian elimination with partial pivoting (m and b are
	// overwritten). False when a pivot is negligible against the matrix's scale.
	private static bool SolveSymmetric4(Span<double> m, Span<double> b, Span<double> x)
	{
		double scale = 0;
		for (int i = 0; i < 16; ++i)
		{
			scale = Math.Max(scale, Math.Abs(m[i]));
		}

		for (int col = 0; col < 4; ++col)
		{
			int pivot = col;
			for (int row = col + 1; row < 4; ++row)
			{
				if (Math.Abs(m[row * 4 + col]) > Math.Abs(m[pivot * 4 + col]))
				{
					pivot = row;
				}
			}

			if (!(Math.Abs(m[pivot * 4 + col]) > 1e-9 * scale))
			{
				return false;
			}

			if (pivot != col)
			{
				for (int c = 0; c < 4; ++c)
				{
					(m[col * 4 + c], m[pivot * 4 + c]) = (m[pivot * 4 + c], m[col * 4 + c]);
				}

				(b[col], b[pivot]) = (b[pivot], b[col]);
			}

			for (int row = col + 1; row < 4; ++row)
			{
				double f = m[row * 4 + col] / m[col * 4 + col];
				for (int c = col; c < 4; ++c)
				{
					m[row * 4 + c] -= f * m[col * 4 + c];
				}

				b[row] -= f * b[col];
			}
		}

		for (int row = 3; row >= 0; --row)
		{
			double sum = b[row];
			for (int c = row + 1; c < 4; ++c)
			{
				sum -= m[row * 4 + c] * x[c];
			}

			x[row] = sum / m[row * 4 + row];
		}

		return true;
	}

	// Mean |gain * T + offset - J| over the window at level 0 (gain 1 and offset 0 when the
	// illumination is not compensated), over the pixels inside both images - the same pixels the
	// tracking step uses: clamped border values do not move with the scene, so counting them
	// would flag a correctly tracked point near the border.
	private static double MeanResidual(
		PyramidLevel lp, PyramidLevel ln, double ux, double uy, double nx, double ny, int half, double gain, double offset)
	{
		double sum = 0;
		int count = 0;
		for (int dy = -half; dy <= half; ++dy)
		{
			for (int dx = -half; dx <= half; ++dx)
			{
				double sx = ux + dx, sy = uy + dy, jx = nx + dx, jy = ny + dy;
				if (!(sx >= 0 && sx <= lp.Width - 1 && sy >= 0 && sy <= lp.Height - 1
					&& jx >= 0 && jx <= ln.Width - 1 && jy >= 0 && jy <= ln.Height - 1))
				{
					continue;
				}

				sum += Math.Abs(gain * lp.Sample(lp.Intensity, sx, sy) + offset - ln.Sample(ln.Intensity, jx, jy));
				++count;
			}
		}

		return count == 0 ? double.PositiveInfinity : sum / count;
	}
}
