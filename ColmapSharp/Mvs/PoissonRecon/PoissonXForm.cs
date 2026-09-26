// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonXForm: XForm<float,Dim> from thirdparty/PoissonRecon/Geometry.h - the small square
// float matrix PoissonRecon uses for the model-to-unit-cube transform (Dim = 4, homogeneous)
// and the normal transform (Dim = 3): products, transpose, cofactor determinant and inverse,
// and point/vector transforms. PointExtent builds the transform; PoissonSampleSet applies it
// to every input sample. Tier A: float arithmetic in the C++ order, signed zeros included
// (oracle/poisson_tree_harness.cc).
//
// Translation notes: coords[i][j] is stored flat at i * Dim + j. The C++ indexes coords[c][r]
// with the first index as the column when transforming points (q[i] += coords[j][i] * p[j]),
// so the translation of a homogeneous transform lives in coords[Dim-1][*]. Real is float
// throughout, as in COLMAP's build (USE_DOUBLE is not defined).

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>A Dim x Dim float matrix. Port of PoissonRecon's <c>XForm&lt;float,Dim&gt;</c>.</summary>
public sealed class PoissonXForm
{
	private readonly float[] coords;

	/// <summary>The zero matrix. Port of <c>XForm( void )</c>.</summary>
	public PoissonXForm(int dim)
	{
		if (dim < 1)
		{
			throw new ArgumentOutOfRangeException(nameof(dim), "XForm dimension must be positive.");
		}

		Dim = dim;
		coords = new float[dim * dim];
	}

	/// <summary>The dimension.</summary>
	public int Dim { get; }

	/// <summary>Element (i, j) = coords[i][j]. Port of <c>operator()( i , j )</c>.</summary>
	public float this[int i, int j]
	{
		get => coords[i * Dim + j];
		set => coords[i * Dim + j] = value;
	}

	/// <summary>The identity. Port of <c>Identity</c>.</summary>
	public static PoissonXForm Identity(int dim)
	{
		var x = new PoissonXForm(dim);
		for (int d = 0; d < dim; d++)
		{
			x[d, d] = 1.0f;
		}

		return x;
	}

	/// <summary>
	/// The upper-left (Dim-1) block. Port of <c>XForm&lt;Real,Dim&gt;( const XForm&lt;Real,Dim+1&gt;&amp; )</c>.
	/// </summary>
	public PoissonXForm UpperLeft()
	{
		var x = new PoissonXForm(Dim - 1);
		for (int i = 0; i < Dim - 1; i++)
		{
			for (int j = 0; j < Dim - 1; j++)
			{
				x[i, j] = this[i, j];
			}
		}

		return x;
	}

	/// <summary>
	/// this * m, with n.coords[i][j] = sum_k m.coords[i][k] * coords[k][j], summed from zero in
	/// k order. Port of <c>operator * ( const XForm&amp; )</c>.
	/// </summary>
	public PoissonXForm Multiply(PoissonXForm m)
	{
		var n = new PoissonXForm(Dim);
		for (int i = 0; i < Dim; i++)
		{
			for (int j = 0; j < Dim; j++)
			{
				for (int k = 0; k < Dim; k++)
				{
					n[i, j] += m[i, k] * this[k, j];
				}
			}
		}

		return n;
	}

	/// <summary>Every element times s. Port of <c>operator * ( Real s )</c>.</summary>
	public PoissonXForm Multiply(float s)
	{
		var n = new PoissonXForm(Dim);
		for (int k = 0; k < coords.Length; k++)
		{
			n.coords[k] = coords[k] * s;
		}

		return n;
	}

	/// <summary>Port of <c>transpose</c>.</summary>
	public PoissonXForm Transpose()
	{
		var x = new PoissonXForm(Dim);
		for (int i = 0; i < Dim; i++)
		{
			for (int j = 0; j < Dim; j++)
			{
				x[i, j] = this[j, i];
			}
		}

		return x;
	}

	/// <summary>Cofactor expansion along column 0 of coords. Port of <c>determinant</c>.</summary>
	public float Determinant()
	{
		if (Dim == 1)
		{
			return coords[0];
		}

		float det = 0.0f;
		for (int d = 0; d < Dim; d++)
		{
			if ((d & 1) != 0)
			{
				det -= this[d, 0] * SubDeterminant(d, 0);
			}
			else
			{
				det += this[d, 0] * SubDeterminant(d, 0);
			}
		}

		return det;
	}

	/// <summary>The determinant with row i and column j removed. Port of <c>subDeterminant</c>.</summary>
	public float SubDeterminant(int i, int j)
	{
		var x = new PoissonXForm(Dim - 1);
		Span<int> ii = stackalloc int[Dim - 1];
		Span<int> jj = stackalloc int[Dim - 1];
		for (int a = 0, i2 = 0, j2 = 0; a < Dim; a++)
		{
			if (a != i)
			{
				ii[i2++] = a;
			}

			if (a != j)
			{
				jj[j2++] = a;
			}
		}

		for (int i2 = 0; i2 < Dim - 1; i2++)
		{
			for (int j2 = 0; j2 < Dim - 1; j2++)
			{
				x[i2, j2] = this[ii[i2], jj[j2]];
			}
		}

		return x.Determinant();
	}

	/// <summary>The adjugate over the determinant. Port of <c>inverse</c>.</summary>
	public PoissonXForm Inverse()
	{
		var x = new PoissonXForm(Dim);
		if (Dim == 1)
		{
			// XForm<float,1>::inverse divides in double.
			x.coords[0] = (float)(1.0 / coords[0]);
			return x;
		}

		float d = Determinant();
		for (int i = 0; i < Dim; i++)
		{
			for (int j = 0; j < Dim; j++)
			{
				x[j, i] = (i + j) % 2 == 0 ? SubDeterminant(i, j) / d : -SubDeterminant(i, j) / d;
			}
		}

		return x;
	}

	/// <summary>
	/// Transforms the (Dim-1)-point p in homogeneous coordinates: q[i] = sum_j coords[j][i] p[j]
	/// + coords[Dim-1][i]. Port of <c>operator * ( const Point&lt;Real,Dim-1&gt;&amp; )</c>.
	/// </summary>
	public void TransformPoint(ReadOnlySpan<float> p, Span<float> q)
	{
		for (int i = 0; i < Dim - 1; i++)
		{
			float v = 0.0f;
			for (int j = 0; j < Dim - 1; j++)
			{
				v += this[j, i] * p[j];
			}

			v += this[Dim - 1, i];
			q[i] = v;
		}
	}

	/// <summary>
	/// q[i] = sum_j coords[j][i] v[j]. Port of <c>operator * ( const Point&lt;Real,Dim&gt;&amp; )</c>.
	/// </summary>
	public void TransformVector(ReadOnlySpan<float> v, Span<float> q)
	{
		for (int i = 0; i < Dim; i++)
		{
			float s = 0.0f;
			for (int j = 0; j < Dim; j++)
			{
				s += this[j, i] * v[j];
			}

			q[i] = s;
		}
	}
}
