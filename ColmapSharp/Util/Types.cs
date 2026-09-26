// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Types: the id vocabulary of colmap/util/types.h - rig, camera, image, frame, image pair,
// 2D point, 3D point, timestamp and pose prior ids with their kInvalid* sentinels, the
// sensor_t / data_t composite ids, the image-pair id packing (ImagePairToPairId,
// PairIdToImagePair) and PairHash / HashCombine. Everything in Scene/ builds on it (Track,
// Point2D, CorrespondenceGraph, ...). It lives in Util/ because COLMAP puts it in util/.
// Tests: ColmapSharp.Tests/Util/TypesTests.cs (util/types_test.cc 1:1, less the span and
// filter_view cases, which have no C# code to test; see that file's header).
//
// Tier A (exact): the pair-id packing and PairHash give the same numbers as COLMAP.
//
// Decisions:
// - Ids are plain integers, not strongly typed wrappers: image_t / camera_t / rig_t /
//   frame_t / point2D_t / pose_prior_t are uint, image_pair_t / point3D_t are ulong,
//   timestamp_t is long. COLMAP's hot loops index containers by id and do arithmetic on
//   them (pair packing, point2D_idx ranges); wrappers would add a conversion at every one
//   of those sites for little protection, since COLMAP itself relies only on the typedef
//   names. Parameter and field names carry the meaning (imageId, point2DIdx, ...).
// - The invalid sentinels are consts named without the k prefix (InvalidImageId). Bring
//   them into scope with `using static ColmapSharp.Util.Types;`.
// - COLMAP's span<T> and filter_view/filter_iterator are not ported: System.Span<T> and
//   LINQ's Where are the C# equivalents and callers use those directly.
// - sensor_t / data_t become SensorId / DataId record structs; their std::hash
//   specializations become the record structs' own hashing, since no result depends on the
//   hash values (hash-container iteration order is made deterministic where it leaks, see
//   docs/CPP_DIVERGENCES.md).

namespace ColmapSharp.Util;

/// <summary>
/// Port of the id typedefs, sentinels and image-pair helpers of colmap/util/types.h.
/// </summary>
public static class Types
{
	/// <summary>kInvalidRigId (rig_t is uint).</summary>
	public const uint InvalidRigId = uint.MaxValue;

	/// <summary>kInvalidCameraId (camera_t is uint).</summary>
	public const uint InvalidCameraId = uint.MaxValue;

	/// <summary>kInvalidImageId (image_t is uint).</summary>
	public const uint InvalidImageId = uint.MaxValue;

	/// <summary>
	/// kMaxNumImages: image ids must stay below int32 max so a pair of them packs into one
	/// image_pair_t.
	/// </summary>
	public const ulong MaxNumImages = int.MaxValue;

	/// <summary>kInvalidFrameId (frame_t is uint).</summary>
	public const uint InvalidFrameId = uint.MaxValue;

	/// <summary>kInvalidImagePairId (image_pair_t is ulong).</summary>
	public const ulong InvalidImagePairId = ulong.MaxValue;

	/// <summary>kInvalidPoint2DIdx (point2D_t is uint).</summary>
	public const uint InvalidPoint2DIdx = uint.MaxValue;

	/// <summary>kInvalidPoint3DId (point3D_t is ulong).</summary>
	public const ulong InvalidPoint3DId = ulong.MaxValue;

	/// <summary>kInvalidTimestamp (timestamp_t is long).</summary>
	public const long InvalidTimestamp = long.MinValue;

	/// <summary>kInvalidPosePriorId (pose_prior_t is uint).</summary>
	public const uint InvalidPosePriorId = uint.MaxValue;

	/// <summary>kInvalidSensorId.</summary>
	public static readonly SensorId InvalidSensorId = new(SensorType.Invalid, SensorId.InvalidId);

	/// <summary>kInvalidDataId.</summary>
	public static readonly DataId InvalidDataId = new(InvalidSensorId, DataId.InvalidId);

	/// <summary>
	/// Port of ShouldSwapImagePair: a pair is stored with the smaller image id first.
	/// </summary>
	public static bool ShouldSwapImagePair(uint imageId1, uint imageId2) => imageId1 > imageId2;

	/// <summary>
	/// Port of ImagePairToPairId: packs an unordered image pair into one id, smaller id first,
	/// so (a, b) and (b, a) give the same id. Throws if either id is not below MaxNumImages.
	/// </summary>
	public static ulong ImagePairToPairId(uint imageId1, uint imageId2)
	{
		ThrowIfGtMaxImages(imageId1);
		ThrowIfGtMaxImages(imageId2);
		if (ShouldSwapImagePair(imageId1, imageId2))
		{
			return MaxNumImages * imageId2 + imageId1;
		}

		return MaxNumImages * imageId1 + imageId2;
	}

	/// <summary>
	/// Port of PairIdToImagePair: the inverse of ImagePairToPairId, smaller id first.
	/// </summary>
	public static (uint ImageId1, uint ImageId2) PairIdToImagePair(ulong pairId)
	{
		uint imageId2 = (uint)(pairId % MaxNumImages);
		uint imageId1 = (uint)((pairId - imageId2) / MaxNumImages);
		ThrowIfGtMaxImages(imageId1);
		ThrowIfGtMaxImages(imageId2);
		return (imageId1, imageId2);
	}

	/// <summary>
	/// Port of HashCombine (boost's hash_combine formula on size_t, which is 64-bit on every
	/// platform COLMAP's oracle runs on).
	/// </summary>
	public static ulong HashCombine(ulong seed, ulong value)
	{
		return unchecked(seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2)));
	}

	// COLMAP throws std::runtime_error; InvalidOperationException is the closest .NET type
	// that is not an argument-validation error (COLMAP's THROW_CHECKs map to ArgumentException).
	private static void ThrowIfGtMaxImages(uint imageId)
	{
		if (imageId >= MaxNumImages)
		{
			throw new InvalidOperationException($"image_id={imageId} >= kMaxNumImages.");
		}
	}
}

/// <summary>Port of colmap::SensorType.</summary>
public enum SensorType
{
	/// <summary>INVALID.</summary>
	Invalid = -1,

	/// <summary>CAMERA.</summary>
	Camera = 0,

	/// <summary>IMU.</summary>
	Imu = 1,
}

/// <summary>
/// Port of colmap::sensor_t: a sensor identified by its type and its id within that type
/// (a camera_t for cameras). Ordered by (type, id) like COLMAP's operator&lt;.
/// </summary>
public readonly record struct SensorId(SensorType Type, uint Id) : IComparable<SensorId>
{
	/// <summary>sensor_t::kInvalidId.</summary>
	public const uint InvalidId = uint.MaxValue;

	/// <summary>The default sensor_t: (INVALID, kInvalidId).</summary>
	public SensorId()
		: this(SensorType.Invalid, InvalidId)
	{
	}

	/// <inheritdoc/>
	public int CompareTo(SensorId other)
	{
		int byType = Type.CompareTo(other.Type);
		return byType != 0 ? byType : Id.CompareTo(other.Id);
	}

	/// <summary>sensor_t::operator&lt;.</summary>
	public static bool operator <(SensorId left, SensorId right) => left.CompareTo(right) < 0;

	/// <summary>The reverse of operator&lt;.</summary>
	public static bool operator >(SensorId left, SensorId right) => left.CompareTo(right) > 0;
}

/// <summary>
/// Port of colmap::data_t: one measurement (an image_t for cameras) of a sensor. Ordered by
/// (sensor, id) like COLMAP's operator&lt;.
/// </summary>
public readonly record struct DataId : IComparable<DataId>
{
	/// <summary>
	/// data_t::kInvalidId. COLMAP declares it uint32 max although the id field is uint64,
	/// and its constructor takes a uint32 id; both are kept.
	/// </summary>
	public const uint InvalidId = uint.MaxValue;

	/// <summary>The default data_t: (kInvalidSensorId, kInvalidId).</summary>
	public DataId()
		: this(Types.InvalidSensorId, InvalidId)
	{
	}

	/// <summary>data_t(sensor_id, id).</summary>
	public DataId(SensorId sensorId, uint id)
	{
		SensorId = sensorId;
		Id = id;
	}

	/// <summary>The sensor that produced the measurement.</summary>
	public SensorId SensorId { get; init; }

	/// <summary>The measurement's id within the sensor.</summary>
	public ulong Id { get; init; }

	/// <inheritdoc/>
	public int CompareTo(DataId other)
	{
		int bySensor = SensorId.CompareTo(other.SensorId);
		return bySensor != 0 ? bySensor : Id.CompareTo(other.Id);
	}

	/// <summary>data_t::operator&lt;.</summary>
	public static bool operator <(DataId left, DataId right) => left.CompareTo(right) < 0;

	/// <summary>The reverse of operator&lt;.</summary>
	public static bool operator >(DataId left, DataId right) => left.CompareTo(right) > 0;
}

/// <summary>
/// Port of colmap::PairHash, the hash COLMAP uses for sets and maps keyed on id pairs.
/// Pairs of 32-bit integers pack into disjoint halves of the 64-bit hash (collision-free);
/// pairs of 64-bit integers go through HashCombine of their std::hash values, which libc++
/// defines as the value itself. As an equality comparer it mixes the 64-bit hash down to
/// .NET's 32-bit hash code (not bit-exact with anything; only Hash() is).
/// </summary>
public sealed class PairHash :
	IEqualityComparer<(uint, uint)>,
	IEqualityComparer<(int, int)>,
	IEqualityComparer<(ulong, ulong)>
{
	/// <summary>Shared instance.</summary>
	public static readonly PairHash Instance = new();

	/// <summary>PairHash on std::pair&lt;uint32_t, uint32_t&gt;.</summary>
	public static ulong Hash((uint First, uint Second) pair)
	{
		return ((ulong)pair.First << 32) | pair.Second;
	}

	/// <summary>
	/// PairHash on std::pair&lt;int32_t, int32_t&gt;: each half goes through its unsigned
	/// counterpart, so negatives keep exactly their low 32 bits.
	/// </summary>
	public static ulong Hash((int First, int Second) pair)
	{
		return Hash(((uint)pair.First, (uint)pair.Second));
	}

	/// <summary>PairHash on std::pair&lt;uint64_t, uint64_t&gt;.</summary>
	public static ulong Hash((ulong First, ulong Second) pair)
	{
		return Types.HashCombine(pair.First, pair.Second);
	}

	/// <inheritdoc/>
	public bool Equals((uint, uint) x, (uint, uint) y) => x == y;

	/// <inheritdoc/>
	public int GetHashCode((uint, uint) obj) => Fold(Hash(obj));

	/// <inheritdoc/>
	public bool Equals((int, int) x, (int, int) y) => x == y;

	/// <inheritdoc/>
	public int GetHashCode((int, int) obj) => Fold(Hash(obj));

	/// <inheritdoc/>
	public bool Equals((ulong, ulong) x, (ulong, ulong) y) => x == y;

	/// <inheritdoc/>
	public int GetHashCode((ulong, ulong) obj) => Fold(Hash(obj));

	// .NET wants 32 bits. XOR-folding the packed (a << 32 | b) would leave a ^ b, so (1, 2)
	// and (2, 1), and every pair with the same XOR, would collide; with millions of keys that
	// degrades a HashSet badly. Fibonacci hashing (multiply by 2^64 / phi, keep the high 32
	// bits) mixes every input bit into the result. Hash() itself stays COLMAP-exact.
	private static int Fold(ulong hash) => unchecked((int)((hash * 0x9E3779B97F4A7C15UL) >> 32));
}
