// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/ordered_groups_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// OrderedGroupsTests (Ceres' test, not COLMAP's): ColmapSharp/Solver/ParameterBlockOrdering.cs,
// same cases and values. Ceres' elements are double* into one array (x, x + 1, x + 2); here
// they are one-long slices of one array at offsets 0, 1, 2. MinNonZeroGroup's death test
// becomes a check that it throws.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class OrderedGroupsTests
{
	private static ArraySegment<double> At(double[] x, int i) => new(x, i, 1);

	[Test]
	public async Task EmptyOrderedGroupBehavesCorrectly()
	{
		var ordering = new ParameterBlockOrdering();
		await Assert.That(ordering.NumGroups).IsEqualTo(0);
		await Assert.That(ordering.NumElements).IsEqualTo(0);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(0);
		double[] x = new double[1];
		await Assert.That(ordering.GroupId(x)).IsEqualTo(-1);
		await Assert.That(ordering.Remove(x)).IsFalse();
	}

	[Test]
	public async Task EverythingInOneGroup()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];
		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 1);
		ordering.AddElementToGroup(At(x, 2), 1);
		ordering.AddElementToGroup(At(x, 0), 1);

		await Assert.That(ordering.NumGroups).IsEqualTo(1);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(1);

		ordering.Remove(At(x, 0));
		await Assert.That(ordering.NumGroups).IsEqualTo(1);
		await Assert.That(ordering.NumElements).IsEqualTo(2);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(2);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);

		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(-1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(1);
	}

	[Test]
	public async Task StartInOneGroupAndThenSplit()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];
		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 1);
		ordering.AddElementToGroup(At(x, 2), 1);
		ordering.AddElementToGroup(At(x, 0), 1);

		await Assert.That(ordering.NumGroups).IsEqualTo(1);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(1);

		ordering.AddElementToGroup(At(x, 0), 5);
		await Assert.That(ordering.NumGroups).IsEqualTo(2);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(2);
		await Assert.That(ordering.GroupSize(5)).IsEqualTo(1);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);

		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(5);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(1);
	}

	[Test]
	public async Task AddAndRemoveEveryThingFromOneGroup()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];
		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 1);
		ordering.AddElementToGroup(At(x, 2), 1);
		ordering.AddElementToGroup(At(x, 0), 1);

		await Assert.That(ordering.NumGroups).IsEqualTo(1);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(1);

		ordering.AddElementToGroup(At(x, 0), 5);
		ordering.AddElementToGroup(At(x, 1), 5);
		ordering.AddElementToGroup(At(x, 2), 5);
		await Assert.That(ordering.NumGroups).IsEqualTo(1);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(0);
		await Assert.That(ordering.GroupSize(5)).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(0)).IsEqualTo(0);

		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(5);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(5);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(5);
	}

	[Test]
	public async Task ReverseOrdering()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];
		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 2);
		ordering.AddElementToGroup(At(x, 2), 2);

		await Assert.That(ordering.NumGroups).IsEqualTo(2);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(1);
		await Assert.That(ordering.GroupSize(2)).IsEqualTo(2);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(2);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(2);

		ordering.Reverse();

		await Assert.That(ordering.NumGroups).IsEqualTo(2);
		await Assert.That(ordering.NumElements).IsEqualTo(3);
		await Assert.That(ordering.GroupSize(3)).IsEqualTo(1);
		await Assert.That(ordering.GroupSize(2)).IsEqualTo(2);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(3);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(2);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(2);
	}

	[Test]
	public async Task ReverseOrderingWithEmptyOrderedGroups()
	{
		var ordering = new ParameterBlockOrdering();

		// This should be a no-op.
		ordering.Reverse();

		// Ensure the properties of an empty OrderedGroups still hold after Reverse().
		await Assert.That(ordering.NumGroups).IsEqualTo(0);
		await Assert.That(ordering.NumElements).IsEqualTo(0);
		await Assert.That(ordering.GroupSize(1)).IsEqualTo(0);
		double[] x = new double[1];
		await Assert.That(ordering.GroupId(x)).IsEqualTo(-1);
		await Assert.That(ordering.Remove(x)).IsFalse();
	}

	[Test]
	public async Task BulkRemove()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];
		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 2);
		ordering.AddElementToGroup(At(x, 2), 2);

		List<ArraySegment<double>> elementsToRemove = [At(x, 0), At(x, 2)];

		await Assert.That(ordering.Remove(elementsToRemove)).IsEqualTo(2);
		await Assert.That(ordering.NumElements).IsEqualTo(1);
		await Assert.That(ordering.GroupId(At(x, 0))).IsEqualTo(-1);
		await Assert.That(ordering.GroupId(At(x, 1))).IsEqualTo(2);
		await Assert.That(ordering.GroupId(At(x, 2))).IsEqualTo(-1);
	}

	[Test]
	public async Task BulkRemoveWithNoElements()
	{
		var ordering = new ParameterBlockOrdering();

		double[] x = new double[3];
		List<ArraySegment<double>> elementsToRemove = [At(x, 0), At(x, 2)];

		await Assert.That(ordering.Remove(elementsToRemove)).IsEqualTo(0);

		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 2);
		ordering.AddElementToGroup(At(x, 2), 2);

		elementsToRemove.Clear();
		await Assert.That(ordering.Remove(elementsToRemove)).IsEqualTo(0);
	}

	[Test]
	public async Task MinNonZeroGroup()
	{
		var ordering = new ParameterBlockOrdering();
		double[] x = new double[3];

		ordering.AddElementToGroup(At(x, 0), 1);
		ordering.AddElementToGroup(At(x, 1), 1);
		ordering.AddElementToGroup(At(x, 2), 2);

		await Assert.That(ordering.MinNonZeroGroup()).IsEqualTo(1);
		ordering.Remove(At(x, 0));

		await Assert.That(ordering.MinNonZeroGroup()).IsEqualTo(1);
		ordering.Remove(At(x, 1));

		await Assert.That(ordering.MinNonZeroGroup()).IsEqualTo(2);
		ordering.Remove(At(x, 2));

		// No non-zero groups left.
		await Assert.That(() => ordering.MinNonZeroGroup()).Throws<Exception>();
	}
}
