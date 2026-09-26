// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Track: port of colmap/scene/track.h and track.cc - TrackElement (one observation: image
// id and 2D point index) and Track (the list of observations of one 3D point). Point3D owns
// a Track; the reconstruction and the incremental mapper edit tracks as points are
// triangulated, merged and filtered. Tests: ColmapSharp.Tests/Scene/TrackTests.cs
// (track_test.cc 1:1).
//
// Tier A (exact): plain bookkeeping; ToString matches COLMAP's operator<< text.
//
// Translation notes:
// - TrackElement is a mutable struct like the C++ value type. COLMAP's default constructor
//   sets both ids invalid; a C# struct's default is all-zero bits, so each id is stored plus
//   one (unchecked) and default(TrackElement) == new TrackElement() (see FeatureMatch.cs).
// - Track is a class (it owns a list), so C++ copy-assignment becomes Clone(). Equality is
//   COLMAP's value equality (element-wise), via Equals and ==.
// - Element(idx) returns a ref into the list like C++'s TrackElement&; an out-of-range idx
//   throws, as std::vector::at does.
// - Reserve and Compress set List.Capacity exactly, which is what std::vector's reserve and
//   shrink_to_fit do (track_test.cc checks the capacities). Growth on Add follows .NET's
//   doubling from 4 rather than libc++'s doubling from 1; only capacity differs, never
//   contents.

using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::TrackElement: an observation of a 3D point, the 2D point
/// <see cref="Point2DIdx"/> in image <see cref="ImageId"/>. Defaults to invalid ids.
/// </summary>
public struct TrackElement : IEquatable<TrackElement>
{
	// Id + 1, so that the all-zero default decodes to the invalid ids (header).
	private uint imageIdPlusOne;
	private uint point2DIdxPlusOne;

	/// <summary>An element with InvalidImageId and InvalidPoint2DIdx.</summary>
	public TrackElement()
	{
	}

	/// <summary>The observation of 2D point <paramref name="point2DIdx"/> in <paramref name="imageId"/>.</summary>
	public TrackElement(uint imageId, uint point2DIdx)
	{
		ImageId = imageId;
		Point2DIdx = point2DIdx;
	}

	/// <summary>The image in which the track element is observed.</summary>
	public uint ImageId
	{
		readonly get => unchecked(imageIdPlusOne - 1);
		set => imageIdPlusOne = unchecked(value + 1);
	}

	/// <summary>The index of the observing 2D point in that image.</summary>
	public uint Point2DIdx
	{
		readonly get => unchecked(point2DIdxPlusOne - 1);
		set => point2DIdxPlusOne = unchecked(value + 1);
	}

	/// <summary>Equality of both ids.</summary>
	public static bool operator ==(TrackElement left, TrackElement right) => left.Equals(right);

	/// <summary>Inequality of either id.</summary>
	public static bool operator !=(TrackElement left, TrackElement right) => !left.Equals(right);

	/// <inheritdoc/>
	public readonly bool Equals(TrackElement other)
	{
		return imageIdPlusOne == other.imageIdPlusOne && point2DIdxPlusOne == other.point2DIdxPlusOne;
	}

	/// <inheritdoc/>
	public override readonly bool Equals(object? obj) => obj is TrackElement other && Equals(other);

	/// <inheritdoc/>
	public override readonly int GetHashCode() => PairHash.Instance.GetHashCode((ImageId, Point2DIdx));

	/// <summary>COLMAP's operator&lt;&lt;: "TrackElement(image_id=i, point2D_idx=p)".</summary>
	public override readonly string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"TrackElement(image_id={ImageId}, point2D_idx={Point2DIdx})");
	}
}

/// <summary>
/// Port of colmap::Track: the observations of one 3D point, in insertion order.
/// </summary>
public sealed class Track : IEquatable<Track>
{
	private List<TrackElement> elements = [];

	/// <summary>The number of track elements.</summary>
	public int Length => elements.Count;

	/// <summary>All elements, mutable in place.</summary>
	public List<TrackElement> Elements => elements;

	/// <summary>Replaces all elements with <paramref name="newElements"/>.</summary>
	public void SetElements(List<TrackElement> newElements)
	{
		elements = newElements;
	}

	/// <summary>The element at <paramref name="idx"/>; throws when out of range.</summary>
	public ref TrackElement Element(int idx)
	{
		// List's indexer does the bounds check that std::vector::at does.
		_ = elements[idx];
		return ref CollectionsMarshal.AsSpan(elements)[idx];
	}

	/// <summary>Overwrites the element at <paramref name="idx"/>; throws when out of range.</summary>
	public void SetElement(int idx, TrackElement element)
	{
		elements[idx] = element;
	}

	/// <summary>Appends <paramref name="element"/>.</summary>
	public void AddElement(TrackElement element)
	{
		elements.Add(element);
	}

	/// <summary>Appends the observation of <paramref name="point2DIdx"/> in <paramref name="imageId"/>.</summary>
	public void AddElement(uint imageId, uint point2DIdx)
	{
		elements.Add(new TrackElement(imageId, point2DIdx));
	}

	/// <summary>Appends <paramref name="newElements"/> in order.</summary>
	public void AddElements(IEnumerable<TrackElement> newElements)
	{
		elements.AddRange(newElements);
	}

	/// <summary>Deletes the element at <paramref name="idx"/> (THROW_CHECK_LT(idx, size)).</summary>
	public void DeleteElement(int idx)
	{
		Check.Lt(idx, elements.Count);
		elements.RemoveAt(idx);
	}

	/// <summary>Deletes every element equal to (<paramref name="imageId"/>, <paramref name="point2DIdx"/>).</summary>
	public void DeleteElement(uint imageId, uint point2DIdx)
	{
		// std::remove_if + erase: a stable removal, as RemoveAll is.
		var target = new TrackElement(imageId, point2DIdx);
		elements.RemoveAll(element => element == target);
	}

	/// <summary>Ensures the capacity is at least <paramref name="numElements"/> (std::vector::reserve).</summary>
	public void Reserve(int numElements)
	{
		if (numElements > elements.Capacity)
		{
			elements.Capacity = numElements;
		}
	}

	/// <summary>Shrinks the capacity to the length to save memory (shrink_to_fit).</summary>
	public void Compress()
	{
		elements.Capacity = elements.Count;
	}

	/// <summary>A deep copy (C++ copy construction).</summary>
	public Track Clone()
	{
		var copy = new Track();
		copy.elements = new List<TrackElement>(elements);
		return copy;
	}

	/// <summary>Element-wise equality, COLMAP's operator==.</summary>
	public static bool operator ==(Track? left, Track? right) => left is null ? right is null : left.Equals(right);

	/// <summary>Element-wise inequality.</summary>
	public static bool operator !=(Track? left, Track? right) => !(left == right);

	/// <inheritdoc/>
	public bool Equals(Track? other)
	{
		return other is not null && CollectionsMarshal.AsSpan(elements).SequenceEqual(CollectionsMarshal.AsSpan(other.elements));
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Track other && Equals(other);

	/// <summary>Hash of the elements; a Track is mutable, so do not key a map on one you then change.</summary>
	public override int GetHashCode()
	{
		var hash = default(HashCode);
		foreach (TrackElement element in elements)
		{
			hash.Add(element);
		}

		return hash.ToHashCode();
	}

	/// <summary>COLMAP's operator&lt;&lt;: "Track(elements=[TrackElement(...), ...])".</summary>
	public override string ToString()
	{
		var text = new StringBuilder("Track(elements=[");
		for (int i = 0; i < elements.Count; i++)
		{
			if (i > 0)
			{
				text.Append(", ");
			}

			text.Append(elements[i].ToString());
		}

		return text.Append("])").ToString();
	}
}
