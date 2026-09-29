// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/ordered_groups.h (OrderedGroups<double*>, the
// ParameterBlockOrdering typedef) (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// A user-supplied ordering of parameter blocks into integer groups, lowest group first, set
// as SolverOptions.LinearSolverOrdering. For the Schur solvers the lowest group is eliminated
// first (it must be an independent set); SchurOrdering.cs applies it (ApplyOrdering). COLMAP's
// GlobalPositioner is the one caller: scales in group 0, points in group 1, frame centers and
// rig cameras in group 2.
//
// Elements are identified like Problem's blocks, by an ArraySegment's (array, offset). Ceres
// keeps each group in a std::set<double*>, so the blocks of one group come out in heap-address
// order, which no two runs share. Here they come out in the order they were added to the group
// (divergence 36).

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::ParameterBlockOrdering: parameter blocks in ordered integer groups.</summary>
public sealed class ParameterBlockOrdering
{
	// Group id -> (insertion sequence -> element); SortedDictionary keeps both orders.
	private readonly SortedDictionary<int, SortedDictionary<long, BlockKey>> groupToElements = [];
	private readonly Dictionary<BlockKey, (int Group, long Sequence)> elementToGroup = [];
	private long nextSequence;

	/// <summary>Number of elements over all groups.</summary>
	public int NumElements => elementToGroup.Count;

	/// <summary>Number of non-empty groups.</summary>
	public int NumGroups => groupToElements.Count;

	/// <summary>
	/// Puts the block starting at <paramref name="element"/> in <paramref name="group"/>,
	/// moving it out of its current group. False (and nothing changes) for a negative group.
	/// </summary>
	public bool AddElementToGroup(ArraySegment<double> element, int group)
	{
		if (group < 0)
		{
			return false;
		}

		BlockKey key = BlockKey.Of(element);
		if (elementToGroup.TryGetValue(key, out (int Group, long Sequence) current))
		{
			if (current.Group == group)
			{
				return true;
			}

			RemoveFromGroup(current.Group, current.Sequence);
		}

		long sequence = nextSequence++;
		elementToGroup[key] = (group, sequence);
		if (!groupToElements.TryGetValue(group, out SortedDictionary<long, BlockKey>? elements))
		{
			elements = [];
			groupToElements.Add(group, elements);
		}

		elements.Add(sequence, key);
		return true;
	}

	/// <summary>Removes every element and group.</summary>
	public void Clear()
	{
		groupToElements.Clear();
		elementToGroup.Clear();
	}

	/// <summary>Removes the element; false if it was not a member.</summary>
	public bool Remove(ArraySegment<double> element)
	{
		BlockKey key = BlockKey.Of(element);
		if (!elementToGroup.TryGetValue(key, out (int Group, long Sequence) current))
		{
			return false;
		}

		RemoveFromGroup(current.Group, current.Sequence);
		elementToGroup.Remove(key);
		return true;
	}

	/// <summary>Removes each of <paramref name="elements"/>; returns how many were members.</summary>
	public int Remove(IReadOnlyList<ArraySegment<double>> elements)
	{
		if (NumElements == 0 || elements.Count == 0)
		{
			return 0;
		}

		int numRemoved = 0;
		foreach (ArraySegment<double> element in elements)
		{
			numRemoved += Remove(element) ? 1 : 0;
		}

		return numRemoved;
	}

	/// <summary>
	/// Reverses the order of the groups: the highest keeps its id and the others follow with
	/// consecutive ids above it.
	/// </summary>
	public void Reverse()
	{
		if (NumGroups == 0)
		{
			return;
		}

		var reversed = new List<KeyValuePair<int, SortedDictionary<long, BlockKey>>>(groupToElements.Reverse());
		groupToElements.Clear();
		int newGroupId = reversed[0].Key;
		foreach (KeyValuePair<int, SortedDictionary<long, BlockKey>> group in reversed)
		{
			foreach (KeyValuePair<long, BlockKey> element in group.Value)
			{
				elementToGroup[element.Value] = (newGroupId, element.Key);
			}

			groupToElements.Add(newGroupId, group.Value);
			newGroupId++;
		}
	}

	/// <summary>The element's group, or -1 if it is not a member.</summary>
	public int GroupId(ArraySegment<double> element) =>
		elementToGroup.TryGetValue(BlockKey.Of(element), out (int Group, long Sequence) current) ? current.Group : -1;

	/// <summary>True if the element is in some group.</summary>
	public bool IsMember(ArraySegment<double> element) => elementToGroup.ContainsKey(BlockKey.Of(element));

	/// <summary>Number of elements in <paramref name="group"/> (0 if there is no such group).</summary>
	public int GroupSize(int group) => groupToElements.TryGetValue(group, out SortedDictionary<long, BlockKey>? elements) ? elements.Count : 0;

	/// <summary>The lowest group id; the ordering must not be empty.</summary>
	public int MinNonZeroGroup()
	{
		Check.Ne(NumGroups, 0, "NumGroups");
		return groupToElements.First().Key;
	}

	/// <summary>A copy (the solver edits a copy rather than the caller's ordering).</summary>
	internal ParameterBlockOrdering Clone()
	{
		var copy = new ParameterBlockOrdering { nextSequence = nextSequence };
		foreach (KeyValuePair<int, SortedDictionary<long, BlockKey>> group in groupToElements)
		{
			copy.groupToElements.Add(group.Key, new SortedDictionary<long, BlockKey>(group.Value));
		}

		foreach (KeyValuePair<BlockKey, (int Group, long Sequence)> element in elementToGroup)
		{
			copy.elementToGroup.Add(element.Key, element.Value);
		}

		return copy;
	}

	/// <summary>The groups in increasing id order, each with its elements in insertion order.</summary>
	internal IEnumerable<(int Group, IEnumerable<(double[] Array, int Offset)> Elements)> Groups()
	{
		foreach (KeyValuePair<int, SortedDictionary<long, BlockKey>> group in groupToElements)
		{
			yield return (group.Key, group.Value.Values.Select(k => (k.Array, k.Offset)));
		}
	}

	private void RemoveFromGroup(int group, long sequence)
	{
		SortedDictionary<long, BlockKey> elements = groupToElements[group];
		elements.Remove(sequence);
		if (elements.Count == 0)
		{
			groupToElements.Remove(group);
		}
	}

	// The (array, offset) identity of a block; arrays compare by reference.
	private readonly record struct BlockKey(double[] Array, int Offset)
	{
		public static BlockKey Of(ArraySegment<double> segment) =>
			new(segment.Array ?? throw new ArgumentException("A parameter block needs an array."), segment.Offset);
	}
}
