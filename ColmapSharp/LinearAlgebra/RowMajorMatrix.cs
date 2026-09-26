// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RowMajorMatrix<T>: a dynamic-size, row-major matrix of plain values, the counterpart of
// COLMAP's Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> typedefs that
// carry feature data rather than linear algebra: FeatureDescriptorsData (uint8),
// FeatureDescriptorsFloatData and FeatureKeypointsBlob/Matrix (float), FeatureMatchesBlob
// and FeatureMatchesMatrix (uint32). Written here (not ported from Eigen); the only Eigen
// behavior it mirrors is the storage order: element (r, c) lives at Data[r * Cols + c], so
// `data.data()[i]` in COLMAP is Data[i] here. Numeric matrices stay in MatrixXd.
//
// Used by Feature/FeatureKeypoint.cs, Feature/FeatureDescriptors.cs and Scene/Database.cs.
// Tests: ColmapSharp.Tests/Feature/FeatureTypesTests.cs.
//
// Unlike Eigen, a new matrix is zero-filled rather than uninitialized; COLMAP's tests never
// read an uninitialized matrix, so no result depends on it.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// A dense row-major matrix of <typeparamref name="T"/> with value equality (same shape and
/// elements), used for feature descriptors, keypoint blobs and match blobs.
/// </summary>
public sealed class RowMajorMatrix<T> : IEquatable<RowMajorMatrix<T>>
	where T : unmanaged, IEquatable<T>
{
	/// <summary>A zero-filled rows x cols matrix.</summary>
	public RowMajorMatrix(int rows, int cols)
	{
		ArgumentOutOfRangeException.ThrowIfNegative(rows);
		ArgumentOutOfRangeException.ThrowIfNegative(cols);
		Rows = rows;
		Cols = cols;
		Data = new T[checked(rows * cols)];
	}

	/// <summary>A rows x cols matrix over <paramref name="data"/> (row-major, not copied).</summary>
	public RowMajorMatrix(int rows, int cols, T[] data)
	{
		ArgumentOutOfRangeException.ThrowIfNegative(rows);
		ArgumentOutOfRangeException.ThrowIfNegative(cols);
		ArgumentOutOfRangeException.ThrowIfNotEqual(data.Length, checked(rows * cols));
		Rows = rows;
		Cols = cols;
		Data = data;
	}

	/// <summary>An empty 0 x 0 matrix (a default-constructed dynamic Eigen matrix).</summary>
	public RowMajorMatrix()
		: this(0, 0)
	{
	}

	/// <summary>Number of rows.</summary>
	public int Rows { get; }

	/// <summary>Number of columns.</summary>
	public int Cols { get; }

	/// <summary>Number of elements (Eigen's size()).</summary>
	public int Size => Data.Length;

	/// <summary>The row-major element storage (Eigen's data()).</summary>
	public T[] Data { get; }

	/// <summary>Element (row, col).</summary>
	public ref T this[int row, int col]
	{
		get
		{
			if ((uint)row >= (uint)Rows || (uint)col >= (uint)Cols)
			{
				throw new ArgumentOutOfRangeException(nameof(row), $"({row}, {col}) outside {Rows}x{Cols}");
			}

			return ref Data[(row * Cols) + col];
		}
	}

	/// <summary>The elements of one row.</summary>
	public Span<T> Row(int row)
	{
		ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)row, (uint)Rows, nameof(row));
		return Data.AsSpan(row * Cols, Cols);
	}

	/// <summary>A deep copy.</summary>
	public RowMajorMatrix<T> Clone() => new(Rows, Cols, (T[])Data.Clone());

	/// <summary>Same shape and elements.</summary>
	public bool Equals(RowMajorMatrix<T>? other) =>
		other is not null && Rows == other.Rows && Cols == other.Cols && Data.AsSpan().SequenceEqual(other.Data);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => Equals(obj as RowMajorMatrix<T>);

	/// <summary>Hash of the shape (the elements are mutable).</summary>
	public override int GetHashCode() => HashCode.Combine(Rows, Cols);

	/// <summary>Same shape and elements.</summary>
	public static bool operator ==(RowMajorMatrix<T>? a, RowMajorMatrix<T>? b) => a is null ? b is null : a.Equals(b);

	/// <summary>Different shape or elements.</summary>
	public static bool operator !=(RowMajorMatrix<T>? a, RowMajorMatrix<T>? b) => !(a == b);

	/// <inheritdoc/>
	public override string ToString() => $"RowMajorMatrix<{typeof(T).Name}>({Rows}x{Cols})";
}
