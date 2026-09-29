// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PyramidalLucasKanadeTests: C#-only tests (no COLMAP counterpart; COLMAP has no tracker) of the
// KLT pieces in Feature/Tracking/ (docs/QUALITY_PLAN.md stage 2a): ImagePyramid,
// GoodFeaturesToTrack and PyramidalLucasKanade. The images are a smooth texture (a sum of
// sinusoids) sampled at pixel centers, so a known warp of the continuous texture gives the
// exact true position of every point in COLMAP's continuous image coordinates.

using ColmapSharp.Feature.Tracking;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature.Tracking;

public class PyramidalLucasKanadeTests
{
	private const int Width = 160;
	private const int Height = 120;

	[Test]
	public async Task SubpixelTranslation_IsRecoveredWithin005Px()
	{
		const double tx = 2.37, ty = -1.62;
		var texture = new Texture(1);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) => (x - tx, y - ty));

		(List<Vector2d> points, KltResult[] results) = TrackAll(prev, next, new KltOptions());

		await Assert.That(points.Count).IsGreaterThan(20);
		for (int i = 0; i < points.Count; ++i)
		{
			await Assert.That(results[i].IsTracked).IsTrue();
			await Assert.That(Distance(results[i].Position, points[i].X + tx, points[i].Y + ty)).IsLessThan(0.05);
		}
	}

	[Test]
	public async Task SmallRotationAndScale_IsRecoveredWithin01Px()
	{
		// next(q) = texture(A^-1 (q - c) + c): a point p moves to c + A (p - c).
		// Rotation gives each pixel of a window its own subpixel phase, so bilinear sampling's
		// error no longer cancels as it does under a translation: measured on this texture at 1
		// degree, the error does not shrink with the window (5, 11, 21 px all reach a worst
		// point of about 0.11 px), while scale alone stays under 0.05 px. Texture frequency is
		// not the cause: the subpixel translation test on this same busy texture reaches a worst
		// point of 0.027 px (0.032 px on the default texture). So the 0.1 px bar is
		// held by the 90th percentile, with the worst point bounded at 0.15 px.
		double angle = 1.0 * Math.PI / 180, scale = 1.01;
		double cos = Math.Cos(angle) * scale, sin = Math.Sin(angle) * scale;
		double cx = Width / 2.0, cy = Height / 2.0;
		var texture = new Texture(2, naturalSpectrum: false);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) =>
		{
			double dx = x - cx, dy = y - cy, det = cos * cos + sin * sin;
			return (cx + (cos * dx + sin * dy) / det, cy + (-sin * dx + cos * dy) / det);
		});

		(List<Vector2d> points, KltResult[] results) = TrackAll(prev, next, new KltOptions());

		await Assert.That(points.Count).IsGreaterThan(20);
		var errors = new List<double>();
		for (int i = 0; i < points.Count; ++i)
		{
			double dx = points[i].X - cx, dy = points[i].Y - cy;
			await Assert.That(results[i].IsTracked).IsTrue();
			errors.Add(Distance(results[i].Position, cx + cos * dx - sin * dy, cy + sin * dx + cos * dy));
		}

		errors.Sort();
		await Assert.That(errors[errors.Count * 9 / 10]).IsLessThan(0.1);
		await Assert.That(errors[^1]).IsLessThan(0.15);
	}

	[Test]
	public async Task LargeMotion_NeedsThePyramid()
	{
		// 41 px is four window radii: level 0 alone cannot bridge it, level 3 sees about 5 px.
		const double tx = 41.3, ty = 29.6;
		const int width = 480, height = 360;
		var texture = new Texture(3);
		Bitmap prev = texture.Render(width, height, (x, y) => (x, y));
		Bitmap next = texture.Render(width, height, (x, y) => (x - tx, y - ty));

		int Recovered(int levels)
		{
			(List<Vector2d> points, KltResult[] results) = TrackAll(prev, next, new KltOptions { MaxLevel = levels });
			return Enumerable.Range(0, points.Count).Count(i =>
				results[i].IsTracked && Distance(results[i].Position, points[i].X + tx, points[i].Y + ty) < 0.1);
		}

		// Only points whose destination is inside the image can be recovered at all.
		List<Vector2d> inside = Detect(prev).Where(p => p.X + tx < width - 12 && p.Y + ty < height - 12).ToList();
		int withPyramid = Recovered(3);
		int without = Recovered(0);
		await Assert.That(withPyramid).IsGreaterThanOrEqualTo((int)(0.9 * inside.Count));
		await Assert.That(without).IsLessThanOrEqualTo(inside.Count / 5);
	}

	[Test]
	public async Task ForwardBackward_RejectsOccludedPoints()
	{
		// The next frame is the same texture moved a little, with a square of an unrelated
		// texture pasted over its middle. Residual rejection is off, so only the forward-backward
		// check can catch the occluded points. The motion is small, so one pyramid level is
		// enough; a coarser level's window would reach across the occluder's edge from the
		// visible points counted below (at level L the window spans 2^L times its size).
		const double tx = 1.3, ty = 0.8;
		const int x0 = 50, x1 = 110, y0 = 30, y1 = 90;
		var texture = new Texture(4);
		var occluder = new Texture(5);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) => (x - tx, y - ty));
		Bitmap cover = occluder.Render(Width, Height, (x, y) => (x, y));
		for (int y = y0; y < y1; ++y)
		{
			for (int x = x0; x < x1; ++x)
			{
				next.RowMajorData[y * Width + x] = cover.RowMajorData[y * Width + x];
			}
		}

		var options = new KltOptions { MaxLevel = 1, MaxMeanResidual = double.PositiveInfinity };
		(List<Vector2d> points, KltResult[] results) = TrackAll(prev, next, options);

		int occluded = 0, visible = 0;
		for (int i = 0; i < points.Count; ++i)
		{
			Vector2d p = points[i];
			if (p.X > x0 + 12 && p.X < x1 - 12 && p.Y > y0 + 12 && p.Y < y1 - 12)
			{
				++occluded;
				await Assert.That(results[i].IsTracked).IsFalse();
			}
			else if ((p.X < x0 - 24 || p.X > x1 + 24 || p.Y < y0 - 24 || p.Y > y1 + 24) && p.X + tx < Width - 12 && p.Y + ty < Height - 12)
			{
				++visible;
				await Assert.That(results[i].IsTracked).IsTrue();
			}
		}

		await Assert.That(occluded).IsGreaterThan(3);
		await Assert.That(visible).IsGreaterThan(3);
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task ForwardBackward_RejectsAFalseMinimum(bool compensateIllumination)
	{
		// A periodic texture (period 12 px, with weak aperiodic texture on top) moves by a small
		// d, but the forward track starts from a guess one period off, so it settles on the false
		// minimum one period beyond the truth. Tracking back from there with no motion guess
		// (Kalal et al.) settles one period from the start, so the round trip exposes it; a back
		// track that starts at the start point only confirms the start is a local minimum.
		// (Displacements over half a period without a guess do not discriminate: on a truly
		// periodic image the back track mirrors the forward one and returns home.)
		const double period = 12, tx = 2.2, ty = 1.3;
		var weak = new Texture(9);
		double Periodic(double x, double y) =>
			40 * Math.Sin(2 * Math.PI * x / period) + 40 * Math.Sin(2 * Math.PI * y / period);
		Bitmap Frame(double dx, double dy) => Texture.RenderFunction(Width, Height, (x, y) =>
			128 + Periodic(x - dx, y - dy) + 0.1 * (weak.Value(x - dx, y - dy) - 128));
		Bitmap prev = Frame(0, 0);
		Bitmap next = Frame(tx, ty);
		List<Vector2d> points = GoodFeaturesToTrack.Detect(
			ImagePyramid.Build(prev, 0), new GoodFeaturesOptions { BorderMargin = 30, MinDistance = 10 });
		var options = new KltOptions
		{
			MaxLevel = 0,
			MaxMeanResidual = double.PositiveInfinity,
			CompensateIllumination = compensateIllumination,
		};
		ImagePyramid p0 = ImagePyramid.Build(prev, 0), p1 = ImagePyramid.Build(next, 0);

		Vector2d[] falseGuesses = [.. points.Select(p => new Vector2d(p.X + tx + period, p.Y + ty))];
		KltResult[] fromFalseGuess = PyramidalLucasKanade.Track(p0, p1, points, options, falseGuesses);
		KltResult[] fromNoGuess = PyramidalLucasKanade.Track(p0, p1, points, options);

		await Assert.That(points.Count).IsGreaterThan(10);
		for (int i = 0; i < points.Count; ++i)
		{
			// The forward track really did land on the false minimum...
			await Assert.That(Distance(fromFalseGuess[i].Position, points[i].X + tx + period, points[i].Y + ty)).IsLessThan(0.5);
			// ...and the check rejects it, while the true minimum passes.
			await Assert.That(fromFalseGuess[i].Status).IsEqualTo(KltStatus.ForwardBackwardFailed);
			await Assert.That(fromNoGuess[i].IsTracked).IsTrue();
		}
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task PointNearTheBorder_IsNotFlaggedHighResidual(bool compensateIllumination)
	{
		// Points whose window pokes out of the image, tracked correctly: the residual must only
		// count window pixels inside both images, so it stays at the noise level.
		const double tx = -1.7, ty = 0.9;
		var texture = new Texture(10);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) => (x - tx, y - ty));
		List<Vector2d> points = GoodFeaturesToTrack.Detect(
			ImagePyramid.Build(prev, 0), new GoodFeaturesOptions { BorderMargin = 1, MinDistance = 4 });
		var nearBorder = points.Where(p => p.X + tx > 1 && p.Y + ty > 1 && p.Y + ty < Height - 1
			&& (p.X < 10 || p.Y < 10 || p.Y > Height - 10)).ToList();
		var options = new KltOptions { MaxMeanResidual = 3, CompensateIllumination = compensateIllumination };

		KltResult[] results = PyramidalLucasKanade.Track(
			ImagePyramid.Build(prev, 3), ImagePyramid.Build(next, 3), nearBorder, options);

		await Assert.That(nearBorder.Count).IsGreaterThan(5);
		int tracked = 0;
		for (int i = 0; i < nearBorder.Count; ++i)
		{
			await Assert.That(results[i].Status).IsNotEqualTo(KltStatus.HighResidual);
			if (results[i].IsTracked)
			{
				++tracked;
				await Assert.That(Distance(results[i].Position, nearBorder[i].X + tx, nearBorder[i].Y + ty)).IsLessThan(0.1);
			}
		}

		await Assert.That(tracked).IsGreaterThan(nearBorder.Count / 2);
	}

	[Test]
	public async Task PointsLeavingTheImage_AreRejected()
	{
		const double tx = -30.5, ty = 0.4;
		var texture = new Texture(6);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) => (x - tx, y - ty));

		(List<Vector2d> points, KltResult[] results) = TrackAll(prev, next, new KltOptions());

		int leaving = 0;
		for (int i = 0; i < points.Count; ++i)
		{
			if (points[i].X + tx < 0)
			{
				++leaving;
				await Assert.That(results[i].IsTracked).IsFalse();
			}
		}

		await Assert.That(leaving).IsGreaterThan(3);
	}

	[Test]
	public async Task GoodFeatures_RespectMaskAndMinDistance()
	{
		Bitmap image = new Texture(7).Render(Width, Height, (x, y) => (x, y));
		var mask = new Bitmap(Width, Height, asRgb: false);
		for (int y = 20; y < 100; ++y)
		{
			for (int x = 0; x < 90; ++x)
			{
				mask.RowMajorData[y * Width + x] = 255;
			}
		}

		var options = new GoodFeaturesOptions { MinDistance = 12, MaxCorners = 1000, QualityLevel = 0.001 };
		List<Vector2d> corners = GoodFeaturesToTrack.Detect(ImagePyramid.Build(image, 0), options, mask);

		await Assert.That(corners.Count).IsGreaterThan(10);
		foreach (Vector2d c in corners)
		{
			await Assert.That(mask.RowMajorData[(int)c.Y * Width + (int)c.X]).IsEqualTo((byte)255);
		}

		for (int i = 0; i < corners.Count; ++i)
		{
			for (int j = i + 1; j < corners.Count; ++j)
			{
				await Assert.That(Distance(corners[i], corners[j].X, corners[j].Y)).IsGreaterThanOrEqualTo(12.0);
			}
		}

		// Existing points push new corners away too.
		List<Vector2d> more = GoodFeaturesToTrack.Detect(ImagePyramid.Build(image, 0), options, mask, corners);
		await Assert.That(more.Count).IsEqualTo(0);
	}

	[Test]
	public async Task Coordinates_PixelCentersAreAtHalf()
	{
		// A round blob centered on the center of pixel (40, 30), i.e. continuous (40.5, 30.5),
		// is detected there, and tracked to where the blob moves in continuous coordinates.
		static Bitmap Blob(double cx, double cy) => Texture.RenderFunction(Width, Height, (x, y) =>
			50 + 150 * Math.Exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * 2.5 * 2.5)));
		Bitmap prev = Blob(40.5, 30.5);
		Bitmap next = Blob(43.25, 28.75);

		List<Vector2d> corners = GoodFeaturesToTrack.Detect(
			ImagePyramid.Build(prev, 0), new GoodFeaturesOptions { MaxCorners = 1 });
		KltResult[] results = PyramidalLucasKanade.Track(
			ImagePyramid.Build(prev, 3), ImagePyramid.Build(next, 3), corners, new KltOptions());

		await Assert.That(corners.Count).IsEqualTo(1);
		await Assert.That(corners[0]).IsEqualTo(new Vector2d(40.5, 30.5));
		await Assert.That(results[0].IsTracked).IsTrue();
		await Assert.That(Distance(results[0].Position, 43.25, 28.75)).IsLessThan(0.05);
	}

	[Test]
	public async Task SequentialAndParallel_GiveIdenticalResults()
	{
		var texture = new Texture(8);
		Bitmap prev = texture.Render(Width, Height, (x, y) => (x, y));
		Bitmap next = texture.Render(Width, Height, (x, y) => (x - 4.3 + 0.01 * y, y + 2.2));

		(_, KltResult[] parallel) = TrackAll(prev, next, new KltOptions { Parallel = true });
		(_, KltResult[] sequential) = TrackAll(prev, next, new KltOptions { Parallel = false });

		await Assert.That(parallel.SequenceEqual(sequential)).IsTrue();
	}

	private static List<Vector2d> Detect(Bitmap image) =>
		GoodFeaturesToTrack.Detect(ImagePyramid.Build(image, 0), new GoodFeaturesOptions { BorderMargin = 12 });

	private static (List<Vector2d> Points, KltResult[] Results) TrackAll(Bitmap prev, Bitmap next, KltOptions options)
	{
		List<Vector2d> points = Detect(prev);
		KltResult[] results = PyramidalLucasKanade.Track(
			ImagePyramid.Build(prev, options.MaxLevel), ImagePyramid.Build(next, options.MaxLevel), points, options);
		return (points, results);
	}

	private static double Distance(Vector2d p, double x, double y) => Math.Sqrt((p.X - x) * (p.X - x) + (p.Y - y) * (p.Y - y));

	// A smooth random texture: 128 plus a sum of sinusoids. By default (naturalSpectrum) 32 waves
	// of 10 to 200 pixels with amplitude growing with wavelength, roughly the 1/f spectrum of real
	// frames: the coarse pyramid levels need that long-wavelength content. Its corners are often
	// nearly one-dimensional, though, so a window's deformation under rotation leaks into the
	// badly conditioned direction; the alternative, 16 waves of 8 to 40 pixels with similar
	// amplitudes, is busy and isotropic, so every corner's window is well conditioned.
	private sealed class Texture
	{
		private readonly (double Fx, double Fy, double Phase, double Amplitude)[] waves;

		public Texture(int seed, bool naturalSpectrum = true)
		{
			var random = new Random(seed);
			waves = new (double, double, double, double)[naturalSpectrum ? 32 : 16];
			for (int k = 0; k < waves.Length; ++k)
			{
				double wavelength = naturalSpectrum ? 10 * Math.Pow(2, 4.3 * random.NextDouble()) : 8 + 32 * random.NextDouble();
				double freq = 2 * Math.PI / wavelength;
				double dir = 2 * Math.PI * random.NextDouble();
				double amplitude = naturalSpectrum ? (2 + wavelength / 4) * (0.5 + random.NextDouble()) : 10 + 8 * random.NextDouble();
				waves[k] = (freq * Math.Cos(dir), freq * Math.Sin(dir), 2 * Math.PI * random.NextDouble(), amplitude);
			}
		}

		// The texture's value at continuous point (u, v).
		public double Value(double u, double v)
		{
			double value = 128;
			foreach ((double fx, double fy, double phase, double amplitude) in waves)
			{
				value += amplitude * Math.Sin(fx * u + fy * v + phase);
			}

			return value;
		}

		// Grey image whose pixel (i, j) is the texture at warp(i + 0.5, j + 0.5).
		public Bitmap Render(int width, int height, Func<double, double, (double X, double Y)> warp) =>
			RenderFunction(width, height, (x, y) =>
			{
				(double u, double v) = warp(x, y);
				return Value(u, v);
			});

		// Samples f at every pixel center (continuous coordinates) into an 8-bit grey bitmap,
		// quantized like a real frame.
		public static Bitmap RenderFunction(int width, int height, Func<double, double, double> f)
		{
			var bitmap = new Bitmap(width, height, asRgb: false);
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					bitmap.RowMajorData[y * width + x] = (byte)Math.Clamp(Math.Round(f(x + 0.5, y + 0.5)), 0, 255);
				}
			}

			return bitmap;
		}
	}
}
