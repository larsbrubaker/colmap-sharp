// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Svd3d / Svd4d: allocation-free SVDs of Matrix3d and Matrix4d, the replacement for
// Eigen::JacobiSVD<Matrix3d> / JacobiSVD<Matrix4d> with ComputeFullU | ComputeFullV, which
// COLMAP runs in hot paths (ComputeClosestRotationMatrix, essential/fundamental matrix
// decomposition and rank-2 enforcement, the P3P/EPnP/triangulation null spaces). They run
// the same JacobiSvdKernel as JacobiSVD on stackalloc buffers, so their results are
// bit-identical to JacobiSVD on the same matrix. Semantics and sign conventions:
// JacobiSvdKernel.cs. Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// SVD of a Matrix3d, M = U diag(SingularValues) V^T, with full U and V.
/// </summary>
public readonly struct Svd3d
{
	private Svd3d(Matrix3d u, Matrix3d v, Vector3d singularValues, ComputationInfo info)
	{
		MatrixU = u;
		MatrixV = v;
		SingularValues = singularValues;
		Info = info;
	}

	/// <summary>Left singular vectors (columns).</summary>
	public Matrix3d MatrixU { get; }

	/// <summary>Right singular vectors (columns).</summary>
	public Matrix3d MatrixV { get; }

	/// <summary>Singular values, non-negative and decreasing.</summary>
	public Vector3d SingularValues { get; }

	/// <summary>Success, or InvalidInput for a non-finite entry.</summary>
	public ComputationInfo Info { get; }

	/// <summary>Decomposes <paramref name="m"/>.</summary>
	public static Svd3d Compute(in Matrix3d m)
	{
		Span<double> a = stackalloc double[9];
		Span<double> u = stackalloc double[9];
		Span<double> v = stackalloc double[9];
		Span<double> s = stackalloc double[3];
		m.CopyToColumnMajor(a);
		ComputationInfo info = JacobiSvdKernel.Decompose(a, 3, u, v, s);
		return new Svd3d(Matrix3d.FromColumnMajor(u), Matrix3d.FromColumnMajor(v), new Vector3d(s[0], s[1], s[2]), info);
	}

	/// <summary>Numerical rank, JacobiSVD's rule (JacobiSvdKernel.Rank).</summary>
	public int Rank()
	{
		Span<double> s = [SingularValues.X, SingularValues.Y, SingularValues.Z];
		return JacobiSvdKernel.Rank(s);
	}
}

/// <summary>
/// SVD of a Matrix4d, M = U diag(SingularValues) V^T, with full U and V.
/// </summary>
public readonly struct Svd4d
{
	private Svd4d(Matrix4d u, Matrix4d v, Vector4d singularValues, ComputationInfo info)
	{
		MatrixU = u;
		MatrixV = v;
		SingularValues = singularValues;
		Info = info;
	}

	/// <summary>Left singular vectors (columns).</summary>
	public Matrix4d MatrixU { get; }

	/// <summary>Right singular vectors (columns).</summary>
	public Matrix4d MatrixV { get; }

	/// <summary>Singular values, non-negative and decreasing.</summary>
	public Vector4d SingularValues { get; }

	/// <summary>Success, or InvalidInput for a non-finite entry.</summary>
	public ComputationInfo Info { get; }

	/// <summary>Decomposes <paramref name="m"/>.</summary>
	public static Svd4d Compute(in Matrix4d m)
	{
		Span<double> a = stackalloc double[16];
		Span<double> u = stackalloc double[16];
		Span<double> v = stackalloc double[16];
		Span<double> s = stackalloc double[4];
		m.CopyToColumnMajor(a);
		ComputationInfo info = JacobiSvdKernel.Decompose(a, 4, u, v, s);
		return new Svd4d(Matrix4d.FromColumnMajor(u), Matrix4d.FromColumnMajor(v), new Vector4d(s[0], s[1], s[2], s[3]), info);
	}

	/// <summary>Numerical rank, JacobiSVD's rule (JacobiSvdKernel.Rank).</summary>
	public int Rank()
	{
		Span<double> s = [SingularValues.X, SingularValues.Y, SingularValues.Z, SingularValues.W];
		return JacobiSvdKernel.Rank(s);
	}
}
