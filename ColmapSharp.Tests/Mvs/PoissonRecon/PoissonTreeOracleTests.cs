// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonTreeOracleTests (C#-only, not a COLMAP test): the octree stages of the PoissonRecon
// port - PointExtent/PoissonXForm (unit-cube and normal transforms, PowOneThird),
// PoissonSolutionParameters (testAndSet), PoissonSampleSet (point insertion and sample
// accumulation, with and without confidence weights), FemTree (shape, offsets, node indices,
// resetNodeIndices, pruned traversal), NeighborKey (const, resetting and creating keys, child
// windows, cache invalidation across node creation), SortedTreeNodes, PoissonDensity and
// PoissonSplat (the normal field with its depth/weight sums, and the color field) -
// against TestData/oracle/poisson_tree.json, which oracle/fixture_poisson_tree.py records from
// oracle/poisson_tree_harness.cc (the vendored PoissonRecon built with -ffp-contract=off).
// Tier A: integers equal, floats bit-identical, and each value's kind (integer or float) as
// the harness printed it. Dumps longer than 1024 values are stored as checksums of 256-value
// chunks (the harness's PrintChunks); a mismatch there reports the first differing chunk.
//
// The harness generates its input points (a noisy ellipsoid shell with some zero, non-finite
// and coincident samples) and prints them as each run's "input" case, so this test replays
// exactly the same floats. Each producer below mirrors the matching block of the harness.

using System.Globalization;
using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public class PoissonTreeOracleTests
{
	private const string Fixture = "poisson_tree.json";

	// The harness checksums long dumps in chunks of this many values (PrintChunks).
	private const int ChunkSize = 256;

	[Test]
	[Arguments("depth5", 5)]
	[Arguments("depth3", 3)]
	public async Task TreeStage_MatchesHarness(string name, int depth)
	{
		var produced = new Cases(name);
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		RunTree(cases.GetProperty(name + "/input"), depth, produced);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
		await Assert.That(produced.Count).IsEqualTo(19);
	}

	[Test]
	[Arguments("density5", 5)]
	[Arguments("density3", 3)]
	[Arguments("density6", 6)]
	public async Task DensityStage_MatchesHarness(string name, int depth)
	{
		var produced = new Cases(name);
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		PoissonSampleSet set = Build(cases.GetProperty(name + "/input"), depth, confidence: false, out PoissonSolutionParameters parameters);
		FemTree tree = set.Tree;
		tree.ResetNodeIndices(0);
		int nodesBefore = tree.NodeCount;

		// Solve: setDensityEstimator< 1 , Reconstructor::WeightDegree >( samples , kernelDepth , samplesPerNode ).
		DensityEstimator density = PoissonDensity.SetDensityEstimator(set, 1, 2, (int)parameters.KernelDepth, parameters.SamplesPerNode);

		produced.F("density", DumpField(tree, density, 1));
		produced.I("densitytree", DumpTree(tree));
		produced.I("densityinfo", [density.Count, density.KernelDepth, density.CoDimension, nodesBefore, tree.NodeCount]);

		// Solve: the normal field (then negated) and the color field (then scaled per level).
		SparseNodeData normals = PoissonSplat.SetNormalField(set, density, (int)parameters.BaseDepth, (int)parameters.Depth, parameters.LowDepthCutOff, out var pointDepthAndWeight);
		produced.F("normals", DumpField(tree, normals, 3));
		produced.F("pointdepthandweight", [pointDepthAndWeight.DepthSum, pointDepthAndWeight.WeightSum, pointDepthAndWeight.TotalWeight]);
		produced.I("normaltree", DumpTree(tree));
		SparseNodeData colors = PoissonSplat.SetAuxField(set, parameters.PerLevelDataScaleFactor);
		produced.F("colors", DumpField(tree, colors, 4));
		produced.I("splatinfo", [normals.Count, colors.Count, tree.NodeCount]);
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task ConfidenceWeights_MatchHarness()
	{
		const string name = "confidence3";
		var produced = new Cases(name);
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		PoissonSampleSet set = Build(cases.GetProperty(name + "/input"), 3, confidence: true, out _);
		produced.F("samples", DumpSamples(set));
		await Assert.That(CompareRun(cases, name, produced)).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task PowOneThird_IsCorrectlyRoundedAndMatchesLibmWhereLibmIs()
	{
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		double[] xs = cases.GetProperty("powonethird/x").EnumerateArray().Select(ReadDouble).ToArray();
		double[] libm = cases.GetProperty("powonethird/y").EnumerateArray().Select(ReadDouble).ToArray();
		double[] correct = cases.GetProperty("powonethird/correct").EnumerateArray().Select(ReadDouble).ToArray();
		double[] ours = xs.Select(PowOneThird.Pow).ToArray();

		int notCorrect = 0;
		int libmDiffers = 0;
		int libmNotCorrect = 0;
		for (int i = 0; i < xs.Length; i++)
		{
			notCorrect += ours[i] != correct[i] ? 1 : 0;
			libmDiffers += ours[i] != libm[i] ? 1 : 0;
			libmNotCorrect += libm[i] != correct[i] ? 1 : 0;
		}

		using (Assert.Multiple())
		{
			// Every result is the correctly rounded one, so it equals Apple's libm exactly where
			// libm is correctly rounded; the fixture's libm misses 7 of 4000 (divergence 75).
			await Assert.That(notCorrect).IsEqualTo(0);
			await Assert.That(libmDiffers).IsEqualTo(libmNotCorrect);
			await Assert.That(libmNotCorrect).IsEqualTo(7);
		}
	}

	[Test]
	public async Task LogF_MatchesLibm()
	{
		JsonElement cases = OracleFixture.Load(Fixture).GetProperty("cases");
		double[] xs = cases.GetProperty("logf/x").EnumerateArray().Select(ReadDouble).ToArray();
		var produced = new Cases("logf");
		produced.F("y", xs.Select(x => (double)PoissonSplat.LogF((float)x)).ToList());
		await Assert.That(CompareRun(cases, "logf", produced, skip: "logf/x")).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task Cancellation_StopsTheTreeBuild()
	{
		JsonElement input = OracleFixture.Load(Fixture).GetProperty("cases").GetProperty("depth3/input");
		float[] flat = ReadFloats(input);
		int n = flat.Length / 9;
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		var parameters = new PoissonSolutionParameters { Depth = 3, FullDepth = 3 };
		await Assert.That(() => PoissonSampleSet.Build(flat.AsSpan(0, 3 * n), flat.AsSpan(3 * n, 3 * n), flat.AsSpan(6 * n, 3 * n), 3, parameters, cancelled.Token))
			.Throws<OperationCanceledException>();
	}

	// The harness's parameter set-up: COLMAP's options (depth, fullDepth = depth when < 5), the
	// rest PoissonRecon.cpp's defaults (PoissonSolutionParameters' defaults).
	private static PoissonSampleSet Build(JsonElement input, int depth, bool confidence, out PoissonSolutionParameters parameters)
	{
		float[] flat = ReadFloats(input);
		int n = flat.Length / 9;
		parameters = new PoissonSolutionParameters
		{
			Depth = (uint)depth,
			FullDepth = depth < 5 ? (uint)depth : 5u,
			Confidence = confidence,
		};
		return PoissonSampleSet.Build(flat.AsSpan(0, 3 * n), flat.AsSpan(3 * n, 3 * n), flat.AsSpan(6 * n, 3 * n), 3, parameters);
	}

	// Mirrors Run( ... , kTree ) in oracle/poisson_tree_harness.cc.
	private static void RunTree(JsonElement input, int depth, Cases emit)
	{
		PoissonSampleSet set = Build(input, depth, confidence: false, out PoissonSolutionParameters parameters);
		FemTree tree = set.Tree;

		var xform = new List<double>();
		AddMatrix(xform, set.ModelToUnitCube);
		AddMatrix(xform, set.UnitCubeToModel);
		AddMatrix(xform, set.NormalXForm);
		emit.F("xform", xform);
		emit.I("params", [parameters.Depth, parameters.SolveDepth, parameters.FullDepth, parameters.BaseDepth, parameters.KernelDepth]);
		emit.I("pointcount", [set.PointCount, tree.NodeCount]);
		emit.I("tree", DumpTree(tree));
		emit.F("samples", DumpSamples(set));

		tree.ResetNodeIndices(0);
		emit.I("reset", DumpTree(tree));
		emit.I("samplenodes", Enumerable.Range(0, set.Count).Select(i => (double)tree.NodeIndex(set.Node(i))).ToList());

		int maxDepth = tree.MaxDepth(tree.Root);
		emit.I("neighbors11", DumpNeighbors(tree, 1, 1, maxDepth, resetOnMissing: false));
		emit.I("neighbors12", DumpNeighbors(tree, 1, 2, maxDepth, resetOnMissing: false));
		emit.I("neighbors22", DumpNeighbors(tree, 2, 2, maxDepth, resetOnMissing: false));

		var sorted = new SortedTreeNodes();
		int[] map = sorted.Reset(tree, tree.Root);
		var sortedValues = new List<double> { sorted.Levels };
		for (int d = 0; d < sorted.Levels; d++)
		{
			for (int slice = 0; slice <= (1 << d); slice++)
			{
				sortedValues.Add(sorted.Begin(d, slice));
			}
		}

		sortedValues.AddRange(map.Select(m => (double)m));
		emit.I("sorted", sortedValues);
		emit.I("sortedtree", DumpTree(tree));

		var finest = new List<int>();
		tree.ProcessNodes(tree.SpaceRoot, node =>
		{
			if (!tree.HasChildren(node) && tree.Depth(node) == maxDepth)
			{
				finest.Add(node);
			}
		});
		var createKey = new NeighborKey(tree, 1, 1, resetOnMissing: true);
		createKey.Set(maxDepth);
		foreach (int node in finest)
		{
			createKey.GetNeighbors(node, createNodes: true);
		}

		emit.I("created", DumpTree(tree));
		RunKeyCases(tree, emit);
	}

	// Mirrors the harness's "Neighbor-key caching and traversal details" block.
	private static void RunKeyCases(FemTree tree, Cases emit)
	{
		int md = tree.MaxDepth(tree.Root);

		var pruned = new List<double>();
		tree.ProcessNodes(tree.Root, n =>
		{
			pruned.Add(tree.NodeIndex(n));
			return tree.Depth(n) < 2 || tree.NodeIndex(n) % 3 == 0;
		});
		emit.I("pruned", pruned);
		emit.I("resetkey11", DumpNeighbors(tree, 1, 1, md, resetOnMissing: true));

		var childKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		childKey.Set(md);
		var childNeighbors = new int[27];
		var children = new List<double>();
		int i = 0;
		tree.ProcessNodes(tree.Root, n =>
		{
			if (!tree.HasChildren(n) || tree.Depth(n) >= md || i++ % 5 != 0)
			{
				return;
			}

			childKey.GetNeighbors(n);
			for (int c = 0; c < 8; c++)
			{
				children.Add(childKey.GetChildNeighbors(c, tree.Depth(n), childNeighbors));
				AddWindow(children, tree, childNeighbors);
			}
		});
		emit.I("childneighbors", children);

		var resetKey = new NeighborKey(tree, 2, 2, resetOnMissing: true);
		var createKey = new NeighborKey(tree, 2, 2, resetOnMissing: true);
		var constKey = new NeighborKey(tree, 2, 2, resetOnMissing: false);
		resetKey.Set(md);
		createKey.Set(md);
		constKey.Set(md);
		var targets = new List<int>();
		int j = 0;
		tree.ProcessNodes(tree.Root, n =>
		{
			if (tree.Depth(n) == md && j++ % 11 == 0 && targets.Count < 12)
			{
				targets.Add(n);
			}
		});
		var cache = new List<double>();
		foreach (int n in targets)
		{
			AddWindow(cache, tree, resetKey.GetNeighbors(n));
			AddWindow(cache, tree, constKey.GetNeighbors(n));
			createKey.GetNeighbors(n, createNodes: true);
			AddWindow(cache, tree, resetKey.GetNeighbors(n));
			int parent = tree.Parent(n);
			int grandparent = tree.Parent(parent);
			int other = tree.FirstChild(grandparent) + ((tree.ChildIndexInParent(parent) + 1) % 8);
			constKey.GetNeighbors(other);
			AddWindow(cache, tree, constKey.GetNeighbors(n));
		}

		emit.I("cache", cache);
		emit.I("cachetree", DumpTree(tree));

		// The radius overloads, through a const key of radius 1.
		var radiusKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		radiusKey.Set(md);
		var radii = new List<double>();
		int[] w22 = new int[125], p22 = new int[125], w12 = new int[64], w33 = new int[343], w32 = new int[216];

		// The C++ windows start cleared (all null); the pNeighbors form leaves p22 untouched at the root.
		Array.Fill(p22, FemTree.None);
		int k = 0;
		tree.ProcessNodes(tree.Root, n =>
		{
			if (k++ % 9 != 0)
			{
				return;
			}

			radiusKey.GetNeighbors(2, 2, n, w22);
			AddWindow(radii, tree, w22);
			radiusKey.GetNeighbors(1, 2, n, w12);
			AddWindow(radii, tree, w12);
			radiusKey.GetNeighbors(3, 3, n, w33);
			AddWindow(radii, tree, w33);
			radiusKey.GetNeighbors(3, 2, n, w32);
			AddWindow(radii, tree, w32);
			radiusKey.GetNeighbors(2, 2, n, p22, w22);
			AddWindow(radii, tree, p22);
			AddWindow(radii, tree, w22);
		});
		emit.I("radii", radii);
	}

	private static List<double> DumpSamples(PoissonSampleSet set)
	{
		FemTree tree = set.Tree;
		var samples = new List<double>();
		for (int i = 0; i < set.Count; i++)
		{
			samples.Add(tree.NodeIndex(set.Node(i)));
			samples.AddRange([set.Position(i, 0), set.Position(i, 1), set.Position(i, 2), set.Weight(i)]);
			samples.AddRange([set.Normal(i, 0), set.Normal(i, 1), set.Normal(i, 2)]);
			samples.AddRange([set.Aux(i, 0), set.Aux(i, 1), set.Aux(i, 2)]);
		}

		return samples;
	}

	// Per node with an entry, in pre-order: node index, the field's entry and its values.
	private static List<double> DumpField(FemTree tree, SparseNodeData field, int width)
	{
		var values = new List<double>();
		tree.ProcessNodes(tree.Root, node =>
		{
			int slot = field.Index(tree.NodeIndex(node));
			if (slot == -1)
			{
				return;
			}

			values.AddRange([tree.NodeIndex(node), slot]);
			for (int k = 0; k < width; k++)
			{
				values.Add(field.Value(slot, k));
			}
		});
		return values;
	}

	// The harness's PrintChunks hash of values [start, start + ChunkSize): FNV-1a over each
	// value's 64 bits (the integer, or the double's bit pattern), masked to 52 bits.
	private static long ChunkHash(Case actual, int start)
	{
		ulong h = 14695981039346656037ul;
		for (int i = start; i < actual.Values.Count && i < start + ChunkSize; i++)
		{
			double value = actual.Values[i];
			ulong v = actual.IsFloat ? unchecked((ulong)BitConverter.DoubleToInt64Bits(value)) : unchecked((ulong)(long)value);
			for (int b = 0; b < 8; b++)
			{
				h = unchecked((h ^ ((v >> (8 * b)) & 0xFF)) * 1099511628211ul);
			}
		}

		return (long)(h & ((1ul << 52) - 1));
	}

	// Compares a checksummed fixture case ({count, float, chunks}) and, on a mismatch, names
	// the first differing chunk and shows the produced values there.
	private static string? FirstChunkMismatch(JsonElement expected, Case actual)
	{
		int count = expected.GetProperty("count").GetInt32();
		bool expectedFloat = expected.GetProperty("float").GetBoolean();
		if (expectedFloat != actual.IsFloat)
		{
			return $"expected a{(expectedFloat ? " float" : "n integer")} case, produced a{(actual.IsFloat ? " float" : "n integer")} one";
		}

		if (count != actual.Values.Count)
		{
			return $"expected {count} values, got {actual.Values.Count}";
		}

		int chunk = 0;
		foreach (JsonElement hash in expected.GetProperty("chunks").EnumerateArray())
		{
			int start = chunk * ChunkSize;
			if (hash.GetInt64() != ChunkHash(actual, start))
			{
				string shown = string.Join(", ", actual.Values.Skip(start).Take(ChunkSize).Select(v => v.ToString("R", CultureInfo.InvariantCulture)));
				return $"values {start}..{Math.Min(start + ChunkSize, count) - 1} differ (checksum); produced: {shown}";
			}

			chunk++;
		}

		return null;
	}

	private static List<double> DumpTree(FemTree tree)
	{
		var values = new List<double>();
		tree.ProcessNodes(tree.Root, node =>
		{
			values.AddRange([tree.Depth(node), tree.Offset(node, 0), tree.Offset(node, 1), tree.Offset(node, 2), tree.NodeIndex(node)]);
		});
		return values;
	}

	private static List<double> DumpNeighbors(FemTree tree, int left, int right, int maxDepth, bool resetOnMissing)
	{
		var key = new NeighborKey(tree, left, right, resetOnMissing);
		key.Set(maxDepth);
		var values = new List<double>();
		int i = 0;
		tree.ProcessNodes(tree.Root, node =>
		{
			if (i++ % 7 != 0)
			{
				return;
			}

			AddWindow(values, tree, key.GetNeighbors(node));
		});
		return values;
	}

	private static void AddWindow(List<double> values, FemTree tree, int[] window)
	{
		foreach (int m in window)
		{
			values.Add(m == FemTree.None ? -1 : tree.NodeIndex(m));
		}
	}

	private static void AddMatrix(List<double> values, PoissonXForm x)
	{
		for (int i = 0; i < x.Dim; i++)
		{
			for (int j = 0; j < x.Dim; j++)
			{
				values.Add(x[i, j]);
			}
		}
	}

	private static float[] ReadFloats(JsonElement array) => array.EnumerateArray().Select(e => (float)ReadDouble(e)).ToArray();

	// Numbers, or "NaN"/"Infinity"/"-Infinity" strings for non-finite values (see the fixture script).
	private static double ReadDouble(JsonElement e) =>
		e.ValueKind == JsonValueKind.String ? double.Parse(e.GetString()!, CultureInfo.InvariantCulture) : e.GetDouble();

	// Compares every fixture case of a run (except its input and `skip`) with the produced
	// ones, in both directions by name. Returns the mismatches, one per line.
	private static string CompareRun(JsonElement cases, string name, Cases produced, string? skip = null)
	{
		var mismatches = new List<string>();
		foreach (JsonProperty property in cases.EnumerateObject())
		{
			if (!property.Name.StartsWith(name + "/", StringComparison.Ordinal) || property.Name == name + "/input" || property.Name == skip)
			{
				continue;
			}

			if (!produced.TryGetValue(property.Name, out Case? actual))
			{
				mismatches.Add($"{property.Name}: no C# producer");
				continue;
			}

			string? mismatch = FirstMismatch(property.Value, actual);
			if (mismatch != null)
			{
				mismatches.Add($"{property.Name}: {mismatch}");
			}
		}

		foreach (string key in produced.Keys)
		{
			if (!cases.TryGetProperty(key, out _))
			{
				mismatches.Add($"{key}: not in the fixture");
			}
		}

		return string.Join("\n", mismatches);
	}

	private static string? FirstMismatch(JsonElement expected, Case actual)
	{
		if (expected.ValueKind == JsonValueKind.Object)
		{
			return FirstChunkMismatch(expected, actual);
		}

		int count = expected.GetArrayLength();
		if (count != actual.Values.Count)
		{
			return $"expected {count} values, got {actual.Values.Count}";
		}

		int i = 0;
		foreach (JsonElement e in expected.EnumerateArray())
		{
			// The fixture writes harness "f" values as JSON floats (Python repr always has a '.'
			// or an exponent, or is a NaN/Infinity string) and "i" values as bare integers.
			string raw = e.GetRawText();
			bool expectedFloat = e.ValueKind == JsonValueKind.String || raw.Contains('.') || raw.Contains('e') || raw.Contains('E');
			if (expectedFloat != actual.IsFloat)
			{
				return $"value {i}: expected a{(expectedFloat ? " float" : "n integer")} ({raw}), produced a{(actual.IsFloat ? " float" : "n integer")} case";
			}

			double x = ReadDouble(e);
			if (BitConverter.DoubleToInt64Bits(x) != BitConverter.DoubleToInt64Bits(actual.Values[i]))
			{
				return $"value {i}: expected {x.ToString("R", CultureInfo.InvariantCulture)}, got {actual.Values[i].ToString("R", CultureInfo.InvariantCulture)}";
			}

			i++;
		}

		return null;
	}

	private sealed record Case(bool IsFloat, List<double> Values);

	// Produced cases of one run, keyed "<run>/<case>", with the harness's value kind.
	private sealed class Cases(string run) : Dictionary<string, Case>
	{
		public void F(string key, List<double> values) => Add(run + "/" + key, new Case(true, values));

		public void I(string key, List<double> values) => Add(run + "/" + key, new Case(false, values));
	}
}
