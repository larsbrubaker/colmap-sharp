// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FixedBuffers: the inline, allocation-free coefficient storage behind the fixed-size
// matrices in this folder (Matrix2d, Matrix3d, Matrix3x4d, Matrix4d). Not a port of
// anything; Eigen's fixed-size storage is replaced by .NET inline arrays, which are
// trim- and AOT-clean and keep each matrix a plain value type.
//
// Every matrix stores its coefficients column-major, the same memory order as Eigen's
// default, so a COLMAP pointer walk over matrix.data() maps to the same flat index here.

using System.Runtime.CompilerServices;

namespace ColmapSharp.LinearAlgebra;

[InlineArray(9)]
internal struct Buffer9
{
	private double _element0;
}

[InlineArray(12)]
internal struct Buffer12
{
	private double _element0;
}

[InlineArray(16)]
internal struct Buffer16
{
	private double _element0;
}
