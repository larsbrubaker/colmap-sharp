// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlScaleSpace: VLFeat's Gaussian scale space (thirdparty/VLFeat/scalespace.h and .c) - the
// geometry, the per-octave level storage, and vl_scalespace_put_image with its upsample /
// downsample / fill helpers. The covariant detector (VlCovDet*.cs) keeps its Gaussian and
// DoG ("cornerness") scale spaces in it. VLFeat's SIFT filter (VlSiftFilter*.cs) has its own,
// different, scale space and does not use this.
//
// Tier A (exact). Smoothing goes through vl_imsmooth_f (VlImOpv.Smooth.cs). Kept quirks:
// _vl_scalespace_fill_octave takes the incremental sigma with sqrtf (a float square root of
// the double difference); the default geometry's base scale uses octave resolution 3 whatever
// resolution the caller then sets.
//
// Translation notes: VLFeat's octaves are separately allocated float arrays; levels are
// (array, offset) pairs here. vl_scalespacegeometry_is_equal (which compares
// octaveFirstSubdivision against octaveLastSubdivision, an upstream typo) is not ported:
// the covariant extractor builds a new detector per image, so a scale space is never reused.

namespace ColmapSharp.Feature.VLFeat;

/// <summary>Port of VlScaleSpaceGeometry.</summary>
public struct VlScaleSpaceGeometry
{
	/// <summary>Image width.</summary>
	public int Width;

	/// <summary>Image height.</summary>
	public int Height;

	/// <summary>Index of the first octave.</summary>
	public int FirstOctave;

	/// <summary>Index of the last octave.</summary>
	public int LastOctave;

	/// <summary>Number of octave subdivisions.</summary>
	public int OctaveResolution;

	/// <summary>Index of the first octave subdivision.</summary>
	public int OctaveFirstSubdivision;

	/// <summary>Index of the last octave subdivision.</summary>
	public int OctaveLastSubdivision;

	/// <summary>Base smoothing (smoothing of octave 0, level 0).</summary>
	public double BaseScale;

	/// <summary>Nominal smoothing of the original image.</summary>
	public double NominalScale;

	/// <summary>Port of vl_scalespace_get_default_geometry.</summary>
	public static VlScaleSpaceGeometry Default(int width, int height)
	{
		Util.Check.That(width >= 1);
		Util.Check.That(height >= 1);
		var geom = new VlScaleSpaceGeometry
		{
			Width = width,
			Height = height,
			FirstOctave = 0,
			LastOctave = (int)Math.Max(Math.Floor(VlMathOp.Log2D(Math.Min(width, height))) - 3, 0),
			OctaveResolution = 3,
			OctaveFirstSubdivision = 0,
		};
		geom.OctaveLastSubdivision = geom.OctaveResolution - 1;
		geom.BaseScale = 1.6 * Math.Pow(2.0, 1.0 / geom.OctaveResolution);
		geom.NominalScale = 0.5;
		return geom;
	}
}

/// <summary>Port of VlScaleSpaceOctaveGeometry.</summary>
public readonly record struct VlScaleSpaceOctaveGeometry(int Width, int Height, double Step);

/// <summary>Port of VlScaleSpace and the vl_scalespace_* functions.</summary>
public sealed class VlScaleSpace
{
	private readonly float[][] octaves;

	/// <summary>Port of vl_scalespace_new_with_geometry.</summary>
	public VlScaleSpace(VlScaleSpaceGeometry geom)
	{
		// is_valid_geometry.
		Util.Check.That(geom.FirstOctave <= geom.LastOctave);
		Util.Check.That(geom.OctaveResolution >= 1);
		Util.Check.That(geom.OctaveFirstSubdivision <= geom.OctaveLastSubdivision);
		Util.Check.That(geom.BaseScale >= 0.0);
		Util.Check.That(geom.NominalScale >= 0.0);

		Geometry = geom;
		int numSublevels = geom.OctaveLastSubdivision - geom.OctaveFirstSubdivision + 1;
		octaves = new float[geom.LastOctave - geom.FirstOctave + 1][];
		for (int o = geom.FirstOctave; o <= geom.LastOctave; ++o)
		{
			VlScaleSpaceOctaveGeometry ogeom = GetOctaveGeometry(o);
			octaves[o - geom.FirstOctave] = new float[checked(ogeom.Width * ogeom.Height * numSublevels)];
		}
	}

	/// <summary>Port of vl_scalespace_get_geometry.</summary>
	public VlScaleSpaceGeometry Geometry { get; }

	/// <summary>Port of vl_scalespace_get_octave_geometry.</summary>
	public VlScaleSpaceOctaveGeometry GetOctaveGeometry(int o)
	{
		return new VlScaleSpaceOctaveGeometry(
			ShiftLeft(Geometry.Width, -o), ShiftLeft(Geometry.Height, -o), Math.Pow(2.0, o));
	}

	/// <summary>
	/// Port of vl_scalespace_get_level: the level (o, s) as the octave's array and the offset
	/// of the level in it.
	/// </summary>
	public (float[] Data, int Offset) GetLevel(int o, int s)
	{
		VlScaleSpaceOctaveGeometry ogeom = GetOctaveGeometry(o);
		Util.Check.That(o >= Geometry.FirstOctave && o <= Geometry.LastOctave);
		Util.Check.That(s >= Geometry.OctaveFirstSubdivision && s <= Geometry.OctaveLastSubdivision);
		return (octaves[o - Geometry.FirstOctave], ogeom.Width * ogeom.Height * (s - Geometry.OctaveFirstSubdivision));
	}

	/// <summary>Port of vl_scalespace_get_level_sigma.</summary>
	public double GetLevelSigma(int o, int s)
	{
		return Geometry.BaseScale * Math.Pow(2.0, o + ((double)s / Geometry.OctaveResolution));
	}

	/// <summary>
	/// Port of vl_scalespace_put_image: builds every level from the image (row-major floats of
	/// the geometry's size). The token is checked before each octave.
	/// </summary>
	public void PutImage(ReadOnlySpan<float> image, CancellationToken cancellationToken = default)
	{
		cancellationToken.ThrowIfCancellationRequested();
		StartOctaveFromImage(image, Geometry.FirstOctave);
		FillOctave(Geometry.FirstOctave);
		for (int o = Geometry.FirstOctave + 1; o <= Geometry.LastOctave; ++o)
		{
			cancellationToken.ThrowIfCancellationRequested();
			StartOctaveFromPreviousOctave(o);
			FillOctave(o);
		}
	}

	// VL_SHIFT_LEFT.
	internal static int ShiftLeft(int x, int n) => n >= 0 ? x << n : x >> -n;

	// Port of copy_and_upsample: doubles the resolution by bilinear interpolation, reading the
	// last row/column twice at the far border.
	private static void CopyAndUpsample(float[] destination, int d, float[] source, int s, int width, int height)
	{
		for (int y = 0; y < height; ++y)
		{
			int oy = (y < height - 1 ? 1 : 0) * width;
			float v10 = source[s];
			float v11 = source[s + oy];
			for (int x = 0; x < width; ++x)
			{
				int ox = x < width - 1 ? 1 : 0;
				float v00 = v10;
				float v01 = v11;
				v10 = source[s + ox];
				v11 = source[s + ox + oy];
				destination[d] = v00;
				destination[d + 1] = 0.5f * (v00 + v10);
				destination[d + (2 * width)] = 0.5f * (v00 + v01);
				destination[d + (2 * width) + 1] = 0.25f * (v00 + v01 + v10 + v11);
				d += 2;
				s++;
			}

			d += 2 * width;
		}
	}

	// Port of copy_and_downsample: keeps every 2^numOctaves-th pixel of every
	// 2^numOctaves-th row.
	private static void CopyAndDownsample(float[] destination, int d, ReadOnlySpan<float> source, int width, int height, int numOctaves)
	{
		if (numOctaves == 0)
		{
			source.Slice(0, width * height).CopyTo(destination.AsSpan(d));
			return;
		}

		int step = 1 << numOctaves;
		for (int y = 0; y < height; y += step)
		{
			int p = y * width;
			for (int x = 0; x < width - (step - 1); x += step)
			{
				destination[d++] = source[p];
				p += step;
			}
		}
	}

	// Port of _vl_scalespace_fill_octave.
	private void FillOctave(int o)
	{
		VlScaleSpaceOctaveGeometry ogeom = GetOctaveGeometry(o);
		for (int s = Geometry.OctaveFirstSubdivision + 1; s <= Geometry.OctaveLastSubdivision; ++s)
		{
			double sigma = GetLevelSigma(o, s);
			double previousSigma = GetLevelSigma(o, s - 1);

			// sqrtf: the double difference rounded to float, square-rooted in float.
			double deltaSigma = MathF.Sqrt((float)((sigma * sigma) - (previousSigma * previousSigma)));
			(float[] level, int levelOffset) = GetLevel(o, s);
			(float[] previous, int previousOffset) = GetLevel(o, s - 1);
			VlImOpv.ImSmoothF(
				level, levelOffset, ogeom.Width,
				previous, previousOffset, ogeom.Width, ogeom.Height, ogeom.Width,
				deltaSigma / ogeom.Step, deltaSigma / ogeom.Step);
		}
	}

	// Port of _vl_scalespace_start_octave_from_image.
	private void StartOctaveFromImage(ReadOnlySpan<float> image, int o)
	{
		// Copy the image to the first subdivision of octave o, upscaling or downscaling as
		// needed. For o < 0 the image first lands in octave 0, which therefore must exist.
		(float[] level, int levelOffset) = GetLevel(Math.Max(0, o), Geometry.OctaveFirstSubdivision);
		CopyAndDownsample(level, levelOffset, image, Geometry.Width, Geometry.Height, Math.Max(0, o));

		for (int op = -1; op >= o; --op)
		{
			VlScaleSpaceOctaveGeometry succGeom = GetOctaveGeometry(op + 1);
			(float[] succ, int succOffset) = GetLevel(op + 1, Geometry.OctaveFirstSubdivision);
			(level, levelOffset) = GetLevel(op, Geometry.OctaveFirstSubdivision);
			CopyAndUpsample(level, levelOffset, succ, succOffset, succGeom.Width, succGeom.Height);
		}

		// Adjust the smoothing of the first level just initialised, accounting for the fact
		// that the input image is assumed to be a nominal scale level.
		double sigma = GetLevelSigma(o, Geometry.OctaveFirstSubdivision);
		double imageSigma = Geometry.NominalScale;
		if (sigma > imageSigma)
		{
			VlScaleSpaceOctaveGeometry ogeom = GetOctaveGeometry(o);
			double deltaSigma = Math.Sqrt((sigma * sigma) - (imageSigma * imageSigma));
			(level, levelOffset) = GetLevel(o, Geometry.OctaveFirstSubdivision);
			VlImOpv.ImSmoothF(
				level, levelOffset, ogeom.Width,
				level, levelOffset, ogeom.Width, ogeom.Height, ogeom.Width,
				deltaSigma / ogeom.Step, deltaSigma / ogeom.Step);
		}
	}

	// Port of _vl_scalespace_start_octave_from_previous_octave.
	private void StartOctaveFromPreviousOctave(int o)
	{
		// From the previous octave pick the level closest to this octave's first subdivision.
		int prevLevelIndex = Math.Min(
			Geometry.OctaveFirstSubdivision + Geometry.OctaveResolution, Geometry.OctaveLastSubdivision);
		(float[] prev, int prevOffset) = GetLevel(o - 1, prevLevelIndex);
		(float[] level, int levelOffset) = GetLevel(o, Geometry.OctaveFirstSubdivision);
		VlScaleSpaceOctaveGeometry prevGeom = GetOctaveGeometry(o - 1);
		CopyAndDownsample(level, levelOffset, prev.AsSpan(prevOffset), prevGeom.Width, prevGeom.Height, 1);

		// Add remaining smoothing, if any.
		double sigma = GetLevelSigma(o, Geometry.OctaveFirstSubdivision);
		double prevSigma = GetLevelSigma(o - 1, prevLevelIndex);
		if (sigma > prevSigma)
		{
			VlScaleSpaceOctaveGeometry ogeom = GetOctaveGeometry(o);
			double deltaSigma = Math.Sqrt((sigma * sigma) - (prevSigma * prevSigma));
			VlImOpv.ImSmoothF(
				level, levelOffset, ogeom.Width,
				level, levelOffset, ogeom.Width, ogeom.Height, ogeom.Width,
				deltaSigma / ogeom.Step, deltaSigma / ogeom.Step);
		}
	}
}
