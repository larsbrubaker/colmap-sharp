// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TrackTests: colmap/scene/track_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Scene/Track.cs. `stream << x` is x.ToString();
// `Track other = track` is track.Clone(); Elements().capacity() is Elements.Capacity.

using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public class TrackTests
{
	[Test]
	public async Task TrackElement_Empty()
	{
		var trackEl = new TrackElement();
		using (Assert.Multiple())
		{
			await Assert.That(trackEl.ImageId).IsEqualTo(InvalidImageId);
			await Assert.That(trackEl.Point2DIdx).IsEqualTo(InvalidPoint2DIdx);
		}
	}

	[Test]
	public async Task TrackElement_Equals()
	{
		var trackEl = new TrackElement();
		TrackElement other = trackEl;
		bool equalAtStart = trackEl == other;
		trackEl.ImageId = 1;
		bool notEqualAfterChange = trackEl != other;
		other.ImageId = 1;
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(trackEl == other).IsTrue();
		}
	}

	[Test]
	public async Task TrackElement_Print()
	{
		var trackEl = new TrackElement(1, 2);
		await Assert.That(trackEl.ToString()).IsEqualTo("TrackElement(image_id=1, point2D_idx=2)");
	}

	[Test]
	public async Task Track_Default()
	{
		var track = new Track();
		using (Assert.Multiple())
		{
			await Assert.That(track.Length).IsEqualTo(0);
			await Assert.That(track.Elements.Count).IsEqualTo(track.Length);
		}
	}

	[Test]
	public async Task Track_Equals()
	{
		var track = new Track();
		Track other = track.Clone();
		bool equalAtStart = track == other;
		track.AddElement(0, 1);
		bool notEqualAfterChange = track != other;
		other.AddElement(0, 1);
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(track == other).IsTrue();
		}
	}

	[Test]
	public async Task Track_Print()
	{
		var track = new Track();
		track.AddElement(1, 2);
		track.AddElement(2, 3);
		await Assert.That(track.ToString()).IsEqualTo(
			"Track(elements=[TrackElement(image_id=1, point2D_idx=2), "
			+ "TrackElement(image_id=2, point2D_idx=3)])");
	}

	[Test]
	public async Task Track_SetElements()
	{
		var track = new Track();
		var elements = new List<TrackElement> { new(0, 1), new(0, 2) };
		track.SetElements(elements);
		using (Assert.Multiple())
		{
			await Assert.That(track.Length).IsEqualTo(2);
			await Assert.That(track.Elements.Count).IsEqualTo(track.Length);
			await Assert.That(track.Element(0).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(0).Point2DIdx).IsEqualTo(1u);
			await Assert.That(track.Element(1).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(1).Point2DIdx).IsEqualTo(2u);
			for (int i = 0; i < track.Length; ++i)
			{
				await Assert.That(track.Element(i).ImageId).IsEqualTo(track.Elements[i].ImageId);
				await Assert.That(track.Element(i).Point2DIdx).IsEqualTo(track.Elements[i].Point2DIdx);
			}
		}
	}

	[Test]
	public async Task Track_AddElement()
	{
		var track = new Track();
		track.AddElement(0, 1);
		track.AddElement(new TrackElement(0, 2));
		var elements = new List<TrackElement> { new(0, 1), new(0, 2) };
		track.AddElements(elements);
		using (Assert.Multiple())
		{
			await Assert.That(track.Length).IsEqualTo(4);
			await Assert.That(track.Elements.Count).IsEqualTo(track.Length);
			await Assert.That(track.Element(0).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(0).Point2DIdx).IsEqualTo(1u);
			await Assert.That(track.Element(1).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(1).Point2DIdx).IsEqualTo(2u);
			await Assert.That(track.Element(2).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(2).Point2DIdx).IsEqualTo(1u);
			await Assert.That(track.Element(3).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(3).Point2DIdx).IsEqualTo(2u);
			for (int i = 0; i < track.Length; ++i)
			{
				await Assert.That(track.Element(i).ImageId).IsEqualTo(track.Elements[i].ImageId);
				await Assert.That(track.Element(i).Point2DIdx).IsEqualTo(track.Elements[i].Point2DIdx);
			}
		}
	}

	[Test]
	public async Task Track_DeleteElement()
	{
		var track = new Track();
		track.AddElement(0, 1);
		track.AddElement(0, 2);
		track.AddElement(0, 3);
		track.AddElement(0, 3);
		int lengthAfterAdd = track.Length;
		int countAfterAdd = track.Elements.Count;
		track.DeleteElement(0);
		int lengthAfterDelete = track.Length;
		int countAfterDelete = track.Elements.Count;
		TrackElement[] afterIndexDelete = [.. track.Elements];
		track.DeleteElement(0, 3);
		using (Assert.Multiple())
		{
			await Assert.That(lengthAfterAdd).IsEqualTo(4);
			await Assert.That(countAfterAdd).IsEqualTo(lengthAfterAdd);
			await Assert.That(lengthAfterDelete).IsEqualTo(3);
			await Assert.That(countAfterDelete).IsEqualTo(lengthAfterDelete);
			await Assert.That(afterIndexDelete[0].ImageId).IsEqualTo(0u);
			await Assert.That(afterIndexDelete[0].Point2DIdx).IsEqualTo(2u);
			await Assert.That(afterIndexDelete[1].ImageId).IsEqualTo(0u);
			await Assert.That(afterIndexDelete[1].Point2DIdx).IsEqualTo(3u);
			await Assert.That(afterIndexDelete[2].ImageId).IsEqualTo(0u);
			await Assert.That(afterIndexDelete[2].Point2DIdx).IsEqualTo(3u);
			await Assert.That(track.Length).IsEqualTo(1);
			await Assert.That(track.Elements.Count).IsEqualTo(track.Length);
			await Assert.That(track.Element(0).ImageId).IsEqualTo(0u);
			await Assert.That(track.Element(0).Point2DIdx).IsEqualTo(2u);
		}
	}

	[Test]
	public async Task Track_Reserve()
	{
		var track = new Track();
		track.Reserve(2);
		await Assert.That(track.Elements.Capacity).IsEqualTo(2);
	}

	[Test]
	public async Task Track_Compress()
	{
		var track = new Track();
		track.AddElement(0, 1);
		track.AddElement(0, 2);
		track.AddElement(0, 3);
		track.AddElement(0, 3);
		int capacityAfterAdd = track.Elements.Capacity;
		track.DeleteElement(0);
		track.DeleteElement(0);
		int capacityAfterDelete = track.Elements.Capacity;
		track.Compress();
		using (Assert.Multiple())
		{
			await Assert.That(capacityAfterAdd).IsEqualTo(4);
			await Assert.That(capacityAfterDelete).IsEqualTo(4);
			await Assert.That(track.Elements.Capacity).IsEqualTo(2);
		}
	}
}
