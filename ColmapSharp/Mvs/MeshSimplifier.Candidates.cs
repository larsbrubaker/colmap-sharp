// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MeshSimplifier.Candidates: the per-edge math of colmap/mvs/mesh_simplification.cc -
// ComputeQuadricError, ComputeEdgeCollapse (optimal position, cost and color of merging two
// vertices, Garland & Heckbert 1997) and WouldCauseFlip. The mesh state and the collapse
// loop are in MeshSimplifier.cs.
//
// Tier C (outcome). COLMAP solves the 4x4 system with Eigen's closed-form determinant and
// inverse; here the determinant and the inverse's last column come from the Laplace
// expansion over 2x2 minors (Eberly, "The Laplace Expansion Theorem: Computing the
// Determinants and Inverses of Matrices", Geometric Tools 2008), exactly the expressions
// of LinearAlgebra/Matrix4d.cs, so costs and positions can differ from COLMAP's in the last
// bits.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

namespace ColmapSharp.Mvs;

internal sealed partial class MeshSimplifier
{
	private const double Epsilon = 1e-12;

	/// <summary>Port of ComputeQuadricError: v^T Q v for v = (pos, 1), Q a row-major 4x4.</summary>
	private static double ComputeQuadricError(ReadOnlySpan<double> q, Vector3d pos)
	{
		double x = pos.X, y = pos.Y, z = pos.Z;

		// (v^T Q) first, then the dot product with v, as the C++ expression associates.
		double r0 = x * q[0] + y * q[4] + z * q[8] + q[12];
		double r1 = x * q[1] + y * q[5] + z * q[9] + q[13];
		double r2 = x * q[2] + y * q[6] + z * q[10] + q[14];
		double r3 = x * q[3] + y * q[7] + z * q[11] + q[15];
		return r0 * x + r1 * y + r2 * z + r3;
	}

	// std::max(0.0, value): 0 unless 0 < value, so a NaN error becomes 0 as in C++.
	private static double NonNegative(double value) => 0.0 < value ? value : 0.0;

	/// <summary>
	/// Port of ComputeEdgeCollapse without the color: the cost and optimal position of
	/// merging <paramref name="v2"/> into <paramref name="v1"/>, and whether v1's own error is
	/// the lower one (which picks the color when colors are not interpolated).
	/// </summary>
	private double ComputeEdgeCollapse(int v1, int v2, out Vector3d optimalPosition, out bool v1HasLowerError)
	{
		Span<double> qBar = stackalloc double[16];
		ReadOnlySpan<double> q1 = quadrics.AsSpan(v1 * 16, 16);
		ReadOnlySpan<double> q2 = quadrics.AsSpan(v2 * 16, 16);
		for (int i = 0; i < 16; i++)
		{
			qBar[i] = q1[i] + q2[i];
		}

		// Per-vertex errors, used by the fallback position solve and the non-interpolated
		// color choice.
		double errV1 = ComputeQuadricError(qBar, positions[v1]);
		double errV2 = ComputeQuadricError(qBar, positions[v2]);
		v1HasLowerError = errV1 <= errV2;

		// Try to solve the 4x4 system for the optimal position (Garland & Heckbert): A is
		// Q_bar with its last row replaced by (0, 0, 0, 1), and the position is A^-1 (0,0,0,1),
		// i.e. the first three entries of the inverse's last column.
		if (TrySolveOptimalPosition(qBar, out optimalPosition))
		{
			return NonNegative(ComputeQuadricError(qBar, optimalPosition));
		}

		// Fallback: evaluate at v1, v2, and the midpoint; pick the minimum.
		Vector3d mid = 0.5 * (positions[v1] + positions[v2]);
		double errMid = ComputeQuadricError(qBar, mid);
		if (errV1 <= errV2 && errV1 <= errMid)
		{
			optimalPosition = positions[v1];
			return NonNegative(errV1);
		}

		if (errV2 <= errMid)
		{
			optimalPosition = positions[v2];
			return NonNegative(errV2);
		}

		optimalPosition = mid;
		return NonNegative(errMid);
	}

	// The determinant test and the solve, with the Laplace expansion of Matrix4d.Determinant
	// and Matrix4d.Inverse evaluated on A literally (the constant last row included, so the
	// result matches Matrix4d bit for bit even for non-finite quadrics).
	internal static bool TrySolveOptimalPosition(ReadOnlySpan<double> qBar, out Vector3d position)
	{
		double a00 = qBar[0], a01 = qBar[1], a02 = qBar[2], a03 = qBar[3];
		double a10 = qBar[4], a11 = qBar[5], a12 = qBar[6], a13 = qBar[7];
		double a20 = qBar[8], a21 = qBar[9], a22 = qBar[10], a23 = qBar[11];
		double a30 = 0, a31 = 0, a32 = 0, a33 = 1;
		double s0 = a00 * a11 - a10 * a01;
		double s1 = a00 * a12 - a10 * a02;
		double s2 = a00 * a13 - a10 * a03;
		double s3 = a01 * a12 - a11 * a02;
		double s4 = a01 * a13 - a11 * a03;
		double s5 = a02 * a13 - a12 * a03;
		double c5 = a22 * a33 - a32 * a23;
		double c4 = a21 * a33 - a31 * a23;
		double c3 = a21 * a32 - a31 * a22;
		double c2 = a20 * a33 - a30 * a23;
		double c1 = a20 * a32 - a30 * a22;
		double c0 = a20 * a31 - a30 * a21;
		double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
		if (!(Math.Abs(det) > Epsilon))
		{
			position = default;
			return false;
		}

		position = new Vector3d(
			(-a21 * s5 + a22 * s4 - a23 * s3) / det,
			(a20 * s5 - a22 * s2 + a23 * s1) / det,
			(-a20 * s4 + a21 * s2 - a23 * s0) / det);
		return true;
	}

	/// <summary>
	/// The color part of ComputeEdgeCollapse: interpolated along the edge at the optimal
	/// position's projection, or the color of the vertex with the lower error.
	/// </summary>
	private void ComputeOptimalColor(int v1, int v2, Vector3d optimalPosition, bool v1HasLowerError,
		out float r, out float g, out float b)
	{
		int c1 = v1 * 3;
		int c2 = v2 * 3;
		if (!interpolateColors)
		{
			int c = v1HasLowerError ? c1 : c2;
			r = colors[c];
			g = colors[c + 1];
			b = colors[c + 2];
			return;
		}

		Vector3d edgeDir = positions[v2] - positions[v1];
		double edgeLenSq = edgeDir.SquaredNorm;
		float t = 0.5f;
		if (edgeLenSq > 0)
		{
			t = (float)((optimalPosition - positions[v1]).Dot(edgeDir) / edgeLenSq);
			t = MathUtils.Clamp(t, 0.0f, 1.0f);
		}

		// Single precision, like Eigen::Vector3f: (1 - t) * c1 + t * c2 per channel.
		float s = 1.0f - t;
		r = s * colors[c1] + t * colors[c2];
		g = s * colors[c1 + 1] + t * colors[c2 + 1];
		b = s * colors[c1 + 2] + t * colors[c2 + 2];
	}

	/// <summary>
	/// Port of WouldCauseFlip: whether moving v1 or v2 to <paramref name="newPos"/> would flip
	/// the normal of one of its faces that it does not share with the other.
	/// </summary>
	private bool WouldCauseFlip(int v1, int v2, Vector3d newPos)
	{
		return WouldFlipVertex(v1, v2, newPos) || WouldFlipVertex(v2, v1, newPos);
	}

	private bool WouldFlipVertex(int vCheck, int vOther, Vector3d newPos)
	{
		foreach (int fi in adjacentFaces[vCheck])
		{
			if (faceRemoved[fi])
			{
				continue;
			}

			int f0 = faceIndices[fi * 3], f1 = faceIndices[fi * 3 + 1], f2 = faceIndices[fi * 3 + 2];
			if (f0 == vOther || f1 == vOther || f2 == vOther)
			{
				continue;
			}

			Vector3d p0 = positions[f0];
			Vector3d p1 = positions[f1];
			Vector3d p2 = positions[f2];
			Vector3d oldNormal = (p1 - p0).Cross(p2 - p0);

			Vector3d np0 = p0, np1 = p1, np2 = p2;
			if (f0 == vCheck)
			{
				np0 = newPos;
			}
			else if (f1 == vCheck)
			{
				np1 = newPos;
			}
			else if (f2 == vCheck)
			{
				np2 = newPos;
			}

			Vector3d newNormal = (np1 - np0).Cross(np2 - np0);
			if (oldNormal.Dot(newNormal) < 0)
			{
				return true;
			}
		}

		return false;
	}
}
