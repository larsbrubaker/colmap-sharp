// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticObjectScene (continued): the textured-sphere-on-a-textured-wall views that
// AutomaticReconstructionTests' dense C#-only tests run on, and the solid value noise that both
// that scene and the object scenes (SyntheticObjectScene.Shapes.cs) texture their surfaces
// with. Not a COLMAP port. This is the ray tracer that used to live in
// AutomaticReconstructionTests.CSharpOnly.cs, lifted here unchanged so the test and the
// benchmark generator share one copy: RenderTexturedSphereOnWall's pixels are bit-identical to
// the test's former RenderTexturedScene. Its arithmetic (double[] vectors, the operation order)
// is kept as it was on purpose, since those tests' results depend on the exact pixels.

using ColmapSharp.Sensor;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class SyntheticObjectScene
{
	/// <summary>
	/// Ray-traces view <paramref name="viewIdx"/> of <paramref name="numViews"/> (grey values in
	/// an RGB bitmap): cameras 4 units from the origin on an arc about the y axis (12 degrees
	/// apart, alternating slightly up and down), looking at a unit sphere at the origin in front
	/// of the wall z = 3. Both surfaces carry the same solid noise texture, sampled at the 3D hit
	/// point so it is the same surface in every view; 3x3 supersampling keeps the fine octaves
	/// from aliasing. Every pixel is photo-consistent, which is what PatchMatch needs.
	/// </summary>
	public static Bitmap RenderTexturedSphereOnWall(int viewIdx, int numViews, int width, int height)
	{
		double angle = (viewIdx - (numViews - 1) / 2.0) * 12.0 * Math.PI / 180.0;
		double[] center = [4 * Math.Sin(angle), viewIdx % 2 == 0 ? 0.3 : -0.3, -4 * Math.Cos(angle)];
		double[] forward = Normalized([-center[0], -center[1], -center[2]]);
		double[] right = Normalized(Cross([0, 1, 0], forward));
		double[] down = Cross(forward, right);
		double focal = 1.2 * Math.Max(width, height);

		var bitmap = new Bitmap(width, height, asRgb: true);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				double sum = 0;
				for (int sy = 0; sy < 3; ++sy)
				{
					for (int sx = 0; sx < 3; ++sx)
					{
						double u = (x + (sx + 0.5) / 3 - width / 2.0) / focal;
						double v = (y + (sy + 0.5) / 3 - height / 2.0) / focal;
						double[] dir = Normalized([
							forward[0] + u * right[0] + v * down[0],
							forward[1] + u * right[1] + v * down[1],
							forward[2] + u * right[2] + v * down[2]]);
						double[] hit = TraceSphereOrWall(center, dir);
						sum += SolidNoiseAlbedo(hit[0], hit[1], hit[2]);
					}
				}

				byte value = (byte)Math.Clamp(sum / 9 * 255, 0, 255);
				bitmap.SetPixel(x, y, new BitmapColor<byte>(value, value, value));
			}
		}

		return bitmap;
	}

	// The first hit of the ray with the unit sphere, else with the wall z = 3.
	private static double[] TraceSphereOrWall(double[] origin, double[] dir)
	{
		double b = Dot(origin, dir);
		double c = Dot(origin, origin) - 1;
		double discriminant = b * b - c;
		double t = discriminant >= 0 ? -b - Math.Sqrt(discriminant) : (3 - origin[2]) / dir[2];
		return [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]];
	}

	// Four octaves of value noise at a 3D point, in [0, 1], stretched for contrast. Base
	// frequency 4 per unit, so a unit sphere shows detail at every SIFT scale.
	private static double SolidNoiseAlbedo(double x, double y, double z)
	{
		double sum = 0;
		double weight = 0.5;
		double frequency = 4;
		for (int octave = 0; octave < 4; ++octave)
		{
			sum += weight * ValueNoise(x * frequency, y * frequency, z * frequency);
			weight *= 0.5;
			frequency *= 2;
		}

		return Math.Clamp((sum / 0.9375 - 0.5) * 2 + 0.5, 0, 1);
	}

	// Trilinear interpolation of lattice hashes with a smoothstep fade, in [0, 1].
	private static double ValueNoise(double x, double y, double z)
	{
		int xi = (int)Math.Floor(x), yi = (int)Math.Floor(y), zi = (int)Math.Floor(z);
		double fx = Smooth(x - xi), fy = Smooth(y - yi), fz = Smooth(z - zi);
		double Lerp(double a, double b, double t) => a + (b - a) * t;
		double Corner(int dx, int dy, int dz) => Hash(xi + dx, yi + dy, zi + dz);
		return Lerp(
			Lerp(Lerp(Corner(0, 0, 0), Corner(1, 0, 0), fx), Lerp(Corner(0, 1, 0), Corner(1, 1, 0), fx), fy),
			Lerp(Lerp(Corner(0, 0, 1), Corner(1, 0, 1), fx), Lerp(Corner(0, 1, 1), Corner(1, 1, 1), fx), fy),
			fz);
	}

	private static double Smooth(double t) => t * t * (3 - 2 * t);

	// A lattice value in [0, 1] from an integer hash (deterministic on every platform).
	private static double Hash(int x, int y, int z)
	{
		uint h = unchecked((uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u);
		h ^= h >> 13;
		h = unchecked(h * 0x5bd1e995u);
		h ^= h >> 15;
		return (h & 0xFFFFFF) / (double)0xFFFFFF;
	}

	private static double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

	private static double[] Cross(double[] a, double[] b) =>
		[a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

	private static double[] Normalized(double[] a)
	{
		double norm = Math.Sqrt(Dot(a, a));
		return [a[0] / norm, a[1] / norm, a[2] / norm];
	}
}
