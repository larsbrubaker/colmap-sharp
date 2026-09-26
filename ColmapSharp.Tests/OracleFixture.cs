// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// OracleFixture: finds and reads the checked-in oracle fixtures under
// ColmapSharp.Tests/TestData/oracle (CLAUDE.md, "Verification nets"). Each fixture is a JSON
// file written by a script in oracle/ (the script's header says how); the test project copies
// TestData/** next to the test assembly, so tests read them from there and never run Python.
// C#-only test infrastructure, shared by every oracle test (first user: RandomOracleTests).

using System.Collections.Concurrent;
using System.Text.Json;

namespace ColmapSharp.Tests;

/// <summary>
/// Loads oracle fixtures from TestData/oracle and converts their JSON arrays to C# arrays.
/// </summary>
internal static class OracleFixture
{
	private static readonly ConcurrentDictionary<string, JsonDocument> Cache = new();

	/// <summary>Absolute path of a fixture file, e.g. <c>PathOf("random.json")</c>.</summary>
	public static string PathOf(string fileName)
	{
		return Path.Combine(AppContext.BaseDirectory, "TestData", "oracle", fileName);
	}

	/// <summary>
	/// The parsed root element of a fixture file. Parsed once per test run and shared, so
	/// callers must only read it.
	/// </summary>
	public static JsonElement Load(string fileName)
	{
		return Cache.GetOrAdd(fileName, Parse).RootElement;
	}

	/// <summary>A JSON array of numbers as doubles (exact: the scripts write round-trip reprs).</summary>
	public static double[] Doubles(JsonElement array)
	{
		return array.EnumerateArray().Select(e => e.GetDouble()).ToArray();
	}

	/// <summary>A JSON array of integers as signed 64-bit values.</summary>
	public static long[] Int64s(JsonElement array)
	{
		return array.EnumerateArray().Select(e => e.GetInt64()).ToArray();
	}

	/// <summary>A JSON array of integers as unsigned 64-bit values.</summary>
	public static ulong[] UInt64s(JsonElement array)
	{
		return array.EnumerateArray().Select(e => e.GetUInt64()).ToArray();
	}

	private static JsonDocument Parse(string fileName)
	{
		string path = PathOf(fileName);
		if (!File.Exists(path))
		{
			throw new FileNotFoundException(
				$"Oracle fixture '{fileName}' is missing from the test output ({path}). It is checked in under "
				+ "ColmapSharp.Tests/TestData/oracle; the script in oracle/ that writes it says how to regenerate it.",
				path);
		}

		return JsonDocument.Parse(File.ReadAllText(path));
	}
}
