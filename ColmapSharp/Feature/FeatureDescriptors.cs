// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureDescriptors: the descriptor half of colmap/feature/types.h and .cc -
// FeatureExtractorType, FeatureMatcherType, FeatureDescriptors (uint8 rows, one per
// keypoint, tagged with the extractor type) and FeatureDescriptorsFloat, with the
// conversions between them. Neighbors: FeatureKeypoint.cs and FeatureMatch.cs (the rest of
// feature/types.h); Scene/Database.cs stores descriptors. Tests:
// ColmapSharp.Tests/Feature/FeatureTypesTests.cs (feature/types_test.cc 1:1).
//
// Tier A (exact). SIFT descriptors convert by value (static_cast between uint8 and float);
// the learned-feature types (ALIKED, LoMa - their extractors are out of scope, but a
// database may still carry their descriptors) store float32 values as raw bytes, which
// COLMAP converts with memcpy. MemoryMarshal.Cast reinterprets the same bytes in the
// platform's byte order, exactly like memcpy.
//
// The enums use PascalCase members with COLMAP's numeric values; ToColmapString gives
// COLMAP's MAKE_ENUM_CLASS spelling for messages.

using System.Runtime.InteropServices;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap::FeatureExtractorType.</summary>
public enum FeatureExtractorType
{
	/// <summary>UNDEFINED.</summary>
	Undefined = -1,

	/// <summary>SIFT.</summary>
	Sift = 0,

	/// <summary>ALIKED_N16ROT.</summary>
	AlikedN16Rot = 1,

	/// <summary>ALIKED_N32.</summary>
	AlikedN32 = 2,

	/// <summary>LOMA_B.</summary>
	LomaB = 3,

	/// <summary>LOMA_B128.</summary>
	LomaB128 = 4,
}

/// <summary>Port of colmap::FeatureMatcherType.</summary>
public enum FeatureMatcherType
{
	/// <summary>UNDEFINED.</summary>
	Undefined = -1,

	/// <summary>SIFT_BRUTEFORCE.</summary>
	SiftBruteForce = 0,

	/// <summary>SIFT_LIGHTGLUE.</summary>
	SiftLightGlue = 1,

	/// <summary>ALIKED_BRUTEFORCE.</summary>
	AlikedBruteForce = 2,

	/// <summary>ALIKED_LIGHTGLUE.</summary>
	AlikedLightGlue = 3,

	/// <summary>LOMA_BRUTEFORCE.</summary>
	LomaBruteForce = 4,

	/// <summary>LOMA_B.</summary>
	LomaB = 5,

	/// <summary>LOMA_B128.</summary>
	LomaB128 = 6,

	/// <summary>LOMA_R.</summary>
	LomaR = 7,

	/// <summary>LOMA_L.</summary>
	LomaL = 8,

	/// <summary>LOMA_G.</summary>
	LomaG = 9,
}

/// <summary>COLMAP's enum-to-string spelling (MAKE_ENUM_CLASS) of the feature enums.</summary>
public static class FeatureTypeExtensions
{
	/// <summary>FeatureExtractorTypeToString.</summary>
	public static string ToColmapString(this FeatureExtractorType type) => type switch
	{
		FeatureExtractorType.Undefined => "UNDEFINED",
		FeatureExtractorType.Sift => "SIFT",
		FeatureExtractorType.AlikedN16Rot => "ALIKED_N16ROT",
		FeatureExtractorType.AlikedN32 => "ALIKED_N32",
		FeatureExtractorType.LomaB => "LOMA_B",
		FeatureExtractorType.LomaB128 => "LOMA_B128",
		_ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FeatureExtractorType"),
	};

	/// <summary>FeatureMatcherTypeToString.</summary>
	public static string ToColmapString(this FeatureMatcherType type) => type switch
	{
		FeatureMatcherType.Undefined => "UNDEFINED",
		FeatureMatcherType.SiftBruteForce => "SIFT_BRUTEFORCE",
		FeatureMatcherType.SiftLightGlue => "SIFT_LIGHTGLUE",
		FeatureMatcherType.AlikedBruteForce => "ALIKED_BRUTEFORCE",
		FeatureMatcherType.AlikedLightGlue => "ALIKED_LIGHTGLUE",
		FeatureMatcherType.LomaBruteForce => "LOMA_BRUTEFORCE",
		FeatureMatcherType.LomaB => "LOMA_B",
		FeatureMatcherType.LomaB128 => "LOMA_B128",
		FeatureMatcherType.LomaR => "LOMA_R",
		FeatureMatcherType.LomaL => "LOMA_L",
		FeatureMatcherType.LomaG => "LOMA_G",
		_ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FeatureMatcherType"),
	};
}

/// <summary>
/// Port of colmap::FeatureDescriptors: one uint8 descriptor per row (FeatureDescriptorsData)
/// with the extractor type that produced them.
/// </summary>
public sealed class FeatureDescriptors
{
	/// <summary>No descriptors, UNDEFINED type.</summary>
	public FeatureDescriptors()
	{
	}

	/// <summary>Descriptors of the given type over <paramref name="data"/> (not copied).</summary>
	public FeatureDescriptors(FeatureExtractorType type, RowMajorMatrix<byte> data)
	{
		Type = type;
		Data = data;
	}

	/// <summary>The extractor that produced the descriptors.</summary>
	public FeatureExtractorType Type { get; set; } = FeatureExtractorType.Undefined;

	/// <summary>The descriptors, one per row.</summary>
	public RowMajorMatrix<byte> Data { get; set; } = new();

	/// <summary>
	/// Port of FeatureDescriptors::FromFloat: SIFT values cast to uint8, learned-feature
	/// float32 values reinterpreted as bytes.
	/// </summary>
	public static FeatureDescriptors FromFloat(FeatureDescriptorsFloat floatDesc)
	{
		var result = new FeatureDescriptors { Type = floatDesc.Type };
		int rows = floatDesc.Data.Rows;
		int floatCols = floatDesc.Data.Cols;
		switch (floatDesc.Type)
		{
			case FeatureExtractorType.Sift:
			{
				// Cast each float value to uint8 (static_cast truncates toward zero).
				var data = new byte[floatDesc.Data.Size];
				for (int i = 0; i < data.Length; ++i)
				{
					data[i] = unchecked((byte)floatDesc.Data.Data[i]);
				}

				result.Data = new RowMajorMatrix<byte>(rows, floatCols, data);
				break;
			}

			case FeatureExtractorType.AlikedN16Rot:
			case FeatureExtractorType.AlikedN32:
			case FeatureExtractorType.LomaB:
			case FeatureExtractorType.LomaB128:
			{
				// Reinterpret float32 data as uint8 bytes.
				byte[] bytes = MemoryMarshal.AsBytes(floatDesc.Data.Data.AsSpan()).ToArray();
				result.Data = new RowMajorMatrix<byte>(rows, floatCols * sizeof(float), bytes);
				break;
			}

			default:
				throw new InvalidOperationException($"Unsupported feature type: {floatDesc.Type.ToColmapString()}");
		}

		return result;
	}

	/// <summary>Port of FeatureDescriptors::ToFloat.</summary>
	public FeatureDescriptorsFloat ToFloat() => FeatureDescriptorsFloat.FromBytes(this);

	/// <summary>A deep copy (C++ copy construction).</summary>
	public FeatureDescriptors Clone() => new(Type, Data.Clone());
}

/// <summary>
/// Port of colmap::FeatureDescriptorsFloat: descriptors as float32 rows
/// (FeatureDescriptorsFloatData) with their extractor type.
/// </summary>
public sealed class FeatureDescriptorsFloat
{
	/// <summary>No descriptors, UNDEFINED type.</summary>
	public FeatureDescriptorsFloat()
	{
	}

	/// <summary>Descriptors of the given type over <paramref name="data"/> (not copied).</summary>
	public FeatureDescriptorsFloat(FeatureExtractorType type, RowMajorMatrix<float> data)
	{
		Type = type;
		Data = data;
	}

	/// <summary>The extractor that produced the descriptors.</summary>
	public FeatureExtractorType Type { get; set; } = FeatureExtractorType.Undefined;

	/// <summary>The descriptors, one per row.</summary>
	public RowMajorMatrix<float> Data { get; set; } = new();

	/// <summary>
	/// Port of FeatureDescriptorsFloat::FromBytes: SIFT uint8 values cast to float,
	/// learned-feature bytes reinterpreted as float32.
	/// </summary>
	public static FeatureDescriptorsFloat FromBytes(FeatureDescriptors byteDesc)
	{
		var result = new FeatureDescriptorsFloat { Type = byteDesc.Type };
		int rows = byteDesc.Data.Rows;
		int uint8Cols = byteDesc.Data.Cols;
		switch (byteDesc.Type)
		{
			case FeatureExtractorType.Sift:
			{
				// Cast each uint8 value to float.
				var data = new float[byteDesc.Data.Size];
				for (int i = 0; i < data.Length; ++i)
				{
					data[i] = byteDesc.Data.Data[i];
				}

				result.Data = new RowMajorMatrix<float>(rows, uint8Cols, data);
				break;
			}

			case FeatureExtractorType.AlikedN16Rot:
			case FeatureExtractorType.AlikedN32:
			case FeatureExtractorType.LomaB:
			case FeatureExtractorType.LomaB128:
			{
				// Reinterpret uint8 bytes as float32 data.
				Check.Eq(uint8Cols % sizeof(float), 0);
				float[] floats = MemoryMarshal.Cast<byte, float>(byteDesc.Data.Data.AsSpan()).ToArray();
				result.Data = new RowMajorMatrix<float>(rows, uint8Cols / sizeof(float), floats);
				break;
			}

			default:
				throw new InvalidOperationException($"Unsupported feature type: {byteDesc.Type.ToColmapString()}");
		}

		return result;
	}

	/// <summary>Port of FeatureDescriptorsFloat::ToBytes.</summary>
	public FeatureDescriptors ToBytes() => FeatureDescriptors.FromFloat(this);
}
