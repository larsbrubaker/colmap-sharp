// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PointExtent: the bounding-box transform of thirdparty/PoissonRecon/PointExtent.h/.inl
// (Extent, GetExtent and the two GetXForm overloads) for 3D points: the model-to-unit-cube
// transform that centers the points' axis-aligned box in [0,1]^3, scaled so the longest side
// times scaleFactor spans the cube. PoissonSampleSet applies it before building the octree.
// Tier A: float arithmetic in the C++ order.
//
// Specialization: Poisson::Solve calls GetXForm<Real,Dim,ExtendedAxes=true> with
// dir = alignDir = Dim-1 (PoissonRecon.cpp's default, which COLMAP keeps). For dir < Dim no
// rotation is applied and the frame of direction dir is (dir+1, dir+2, dir+3) mod Dim, which
// for dir = 2 is the identity, so only the three axis extents enter the result; the six
// diagonal "extended" directions are computed by the C++ but never read. This port computes
// the axis extents only and accepts dir in [0, 3). The axis dot products are evaluated as
// the C++ Point::Dot does (0 + p0*e0 + p1*e1 + p2*e2), which turns a -0 coordinate into +0.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The model-to-unit-cube transform of a point set. Port of PoissonRecon's
/// <c>PointExtent::GetXForm</c> for Dim = 3, ExtendedAxes = true and an axis direction.
/// </summary>
public static class PointExtent
{
	/// <summary>
	/// The 4x4 homogeneous transform taking the points (xyz triples) into the unit cube.
	/// Port of <c>GetXForm( stream , ... , scaleFactor , dir )</c>.
	/// </summary>
	public static PoissonXForm GetXForm(ReadOnlySpan<float> positions, float scaleFactor, int dir)
	{
		const int Dim = 3;
		if (dir < 0 || dir >= Dim)
		{
			throw new ArgumentOutOfRangeException(nameof(dir), "Only the axis alignment directions 0, 1 and 2 are supported.");
		}

		// Extent: [first, second] per axis direction, starting at [+inf, -inf].
		Span<float> first = [float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity];
		Span<float> second = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
		for (int i = 0; i + 2 < positions.Length; i += 3)
		{
			for (int d = 0; d < Dim; d++)
			{
				float dot = 0.0f;
				for (int k = 0; k < Dim; k++)
				{
					dot += positions[i + k] * (k == d ? 1.0f : 0.0f);
				}

				// std::min<Real> / std::max<Real>.
				first[d] = StdMinMax.StdMin(first[d], dot);
				second[d] = StdMinMax.StdMax(second[d], dot);
			}
		}

		// frame[d] = (dir + 1 + d) % Dim; R(r, c) = direction(frame[c])[r].
		Span<int> frame = stackalloc int[Dim];
		for (int d = 0; d < Dim; d++)
		{
			frame[d] = (dir + 1 + d) % Dim;
		}

		PoissonXForm r = PoissonXForm.Identity(Dim + 1);
		for (int c = 0; c < Dim; c++)
		{
			for (int row = 0; row < Dim; row++)
			{
				r[row, c] = frame[c] == row ? 1.0f : 0.0f;
			}
		}

		Span<float> min = stackalloc float[Dim];
		Span<float> max = stackalloc float[Dim];
		for (int d = 0; d < Dim; d++)
		{
			min[d] = first[frame[d]];
			max[d] = second[frame[d]];
		}

		return GetXForm(min, max, scaleFactor).Multiply(r);
	}

	// Port of GetXForm( min , max , scaleFactor , rotate=false ): rXForm * sXForm * tXForm with
	// rXForm the identity.
	private static PoissonXForm GetXForm(ReadOnlySpan<float> min, ReadOnlySpan<float> max, float scaleFactor)
	{
		const int Dim = 3;
		Span<float> center = stackalloc float[Dim];
		for (int d = 0; d < Dim; d++)
		{
			center[d] = (max[d] + min[d]) / 2;
		}

		float scale = max[0] - min[0];
		for (int d = 1; d < Dim; d++)
		{
			// std::max<Real>( scale , max[d]-min[d] )
			float extent = max[d] - min[d];
			scale = StdMinMax.StdMax(scale, extent);
		}

		scale *= scaleFactor;
		for (int i = 0; i < Dim; i++)
		{
			center[i] -= scale / 2;
		}

		PoissonXForm tXForm = PoissonXForm.Identity(Dim + 1);
		PoissonXForm sXForm = PoissonXForm.Identity(Dim + 1);
		PoissonXForm rXForm = PoissonXForm.Identity(Dim + 1);
		for (int i = 0; i < Dim; i++)
		{
			sXForm[i, i] = (float)(1.0 / scale);
			tXForm[Dim, i] = -center[i];
		}

		return rXForm.Multiply(sXForm).Multiply(tXForm);
	}
}
