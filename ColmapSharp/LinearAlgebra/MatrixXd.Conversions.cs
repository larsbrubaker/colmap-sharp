// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MatrixXd conversions to and from the fixed-size matrices (Matrix2d, Matrix3d,
// Matrix3x4d, Matrix4d), the analogue of assigning between Eigen::MatrixXd and a
// fixed-size Eigen::Matrix. Part of MatrixXd (see MatrixXd.cs). Both sides store
// column-major, so the conversions are plain buffer copies.

namespace ColmapSharp.LinearAlgebra;

public sealed partial class MatrixXd
{
	/// <summary>Converts a Matrix2d.</summary>
	public static MatrixXd From(Matrix2d m)
	{
		return FromColumnMajor(2, 2, [m[0, 0], m[1, 0], m[0, 1], m[1, 1]]);
	}

	/// <summary>Converts a Matrix3d.</summary>
	public static MatrixXd From(in Matrix3d m)
	{
		var r = new MatrixXd(3, 3);
		m.CopyToColumnMajor(r._data);
		return r;
	}

	/// <summary>Converts a Matrix3x4d.</summary>
	public static MatrixXd From(in Matrix3x4d m)
	{
		var r = new MatrixXd(3, 4);
		m.CopyToColumnMajor(r._data);
		return r;
	}

	/// <summary>Converts a Matrix4d.</summary>
	public static MatrixXd From(in Matrix4d m)
	{
		var r = new MatrixXd(4, 4);
		m.CopyToColumnMajor(r._data);
		return r;
	}

	/// <summary>Converts to a Matrix2d; the shape must be 2x2.</summary>
	public Matrix2d ToMatrix2d()
	{
		RequireShape(2, 2);
		return Matrix2d.FromColumnMajor(_data);
	}

	/// <summary>Converts to a Matrix3d; the shape must be 3x3.</summary>
	public Matrix3d ToMatrix3d()
	{
		RequireShape(3, 3);
		return Matrix3d.FromColumnMajor(_data);
	}

	/// <summary>Converts to a Matrix3x4d; the shape must be 3x4.</summary>
	public Matrix3x4d ToMatrix3x4d()
	{
		RequireShape(3, 4);
		return Matrix3x4d.FromColumnMajor(_data);
	}

	/// <summary>Converts to a Matrix4d; the shape must be 4x4.</summary>
	public Matrix4d ToMatrix4d()
	{
		RequireShape(4, 4);
		return Matrix4d.FromColumnMajor(_data);
	}

	private void RequireShape(int rows, int cols)
	{
		if (Rows != rows || Cols != cols)
		{
			throw new ArgumentException($"Expected a {rows}x{cols} matrix, got {Rows}x{Cols}.");
		}
	}
}
