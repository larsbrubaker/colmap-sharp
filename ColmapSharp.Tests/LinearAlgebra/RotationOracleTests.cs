// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RotationOracleTests (C#-only; no COLMAP *_test.cc covers Eigen itself): Quaterniond and
// AngleAxisd against Eigen, through the pycolmap 4.2.0 Rotation3d binding. The fixture is
// written by oracle/linear_algebra_rotations.py into
// TestData/oracle/linear_algebra_rotations.json; that script lists which Eigen call each
// field comes from.
//
// Tiers, per operation (Rigid3d/Sim3d algebra downstream is Tier A, so these are pinned
// as tightly as the C++ allows):
// - Tier A, bit-identical on every case: product, matrix, from_matrix, from_general,
//   norm, inverse, angle, angle_to (ExactFields).
// - Tier B (ToleranceFields), each for a reason on the C++ side that we deliberately do
//   not reproduce, see docs/CPP_DIVERGENCES.md:
//   rotated         - the macOS arm64 wheel contracts the cross products inside q * v into
//                     FMAs (entry 6). With FMA emulated, our formula is exact on all
//                     cases; oracle/linear_algebra_rotations.py prints that evidence.
//   from_axis_angle - the C++ sin(a/2) can round 1 ulp away from libm sin(), which .NET
//                     calls; the cause is only a hypothesis (entry 7).
// Tolerance: 1e-14 relative per coefficient, a few ulps; the observed gaps are 1-2 ulps.

using System.Text.Json;

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class RotationOracleTests
{
	private static readonly Lazy<JsonElement[]> Cases = new(LoadCases);

	private static JsonElement[] LoadCases()
	{
		string path = Path.Combine(AppContext.BaseDirectory, "TestData", "oracle", "linear_algebra_rotations.json");
		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
		return document.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToArray();
	}

	private static double[] Read(JsonElement element, string name)
	{
		JsonElement value = element.GetProperty(name);
		if (value.ValueKind == JsonValueKind.Number)
		{
			return [value.GetDouble()];
		}

		return value.EnumerateArray().Select(e => e.GetDouble()).ToArray();
	}

	private static Quaterniond Quat(JsonElement element, string name)
	{
		double[] xyzw = Read(element, name);
		return Quaterniond.FromCoeffs(new Vector4d(xyzw[0], xyzw[1], xyzw[2], xyzw[3]));
	}

	private static Vector3d Vec3(JsonElement element, string name)
	{
		double[] v = Read(element, name);
		return new Vector3d(v[0], v[1], v[2]);
	}

	private static Matrix3d RowMajor(JsonElement element, string name)
	{
		double[] m = Read(element, name);
		return new Matrix3d(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
	}

	private static double[] Flatten(Quaterniond q) => [q.X, q.Y, q.Z, q.W];

	private static double[] Flatten(Vector3d v) => [v.X, v.Y, v.Z];

	private static double[] Flatten(Matrix3d m) => [m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2]];

	private static double[] Compute(JsonElement c, string field)
	{
		Quaterniond q = Quat(c, "q");
		Quaterniond other = Quat(c, "other");
		return field switch
		{
			"product" => Flatten(q * other),
			"rotated" => Flatten(q * Vec3(c, "v")),
			"matrix" => Flatten(q.ToRotationMatrix()),
			"from_matrix" => Flatten(Quaterniond.FromRotationMatrix(RowMajor(c, "matrix"))),
			"from_general" => Flatten(Quaterniond.FromRotationMatrix(RowMajor(c, "general"))),
			"from_axis_angle" => Flatten(AxisAngleVectorToQuaternion(Vec3(c, "axis_angle"))),
			"norm" => [q.Norm],
			"inverse" => Flatten(q.Inverse()),
			"angle" => [AngleAxisd.FromQuaternion(q).Angle],
			"angle_to" => [q.AngularDistance(other)],
			_ => throw new ArgumentException(field),
		};
	}

	// Rotation3d(axis_angle): Quaterniond(AngleAxisd(v.norm(), v.normalized())).
	private static Quaterniond AxisAngleVectorToQuaternion(Vector3d v)
	{
		return new AngleAxisd(v.Norm, v.Normalized()).ToQuaternion();
	}

	[Test]
	[Arguments("product")]
	[Arguments("matrix")]
	[Arguments("from_matrix")]
	[Arguments("from_general")]
	[Arguments("norm")]
	[Arguments("inverse")]
	[Arguments("angle")]
	[Arguments("angle_to")]
	public async Task ExactFields(string field)
	{
		await AssertAllCases(field, (expected, actual) =>
			BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual));
	}

	[Test]
	[Arguments("rotated")]
	[Arguments("from_axis_angle")]
	public async Task ToleranceFields(string field)
	{
		await AssertAllCases(field, (expected, actual) =>
			Math.Abs(expected - actual) <= 1e-14 * Math.Max(1.0, Math.Abs(expected)));
	}

	[Test]
	public async Task TraceZeroMatricesTakeTheDiagonalBranch()
	{
		// Matrices whose trace is exactly 0: the fixture shows Eigen does not take the
		// trace branch there, which is why FromRotationMatrix tests trace > 0, not >= 0.
		string path = Path.Combine(AppContext.BaseDirectory, "TestData", "oracle", "linear_algebra_rotations.json");
		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
		var mismatches = new List<string>();
		int count = 0;
		foreach (JsonElement c in document.RootElement.GetProperty("trace_zero_cases").EnumerateArray())
		{
			count++;
			double[] expected = Read(c, "from_matrix");
			double[] actual = Flatten(Quaterniond.FromRotationMatrix(RowMajor(c, "matrix")));
			for (int i = 0; i < 4; i++)
			{
				if (BitConverter.DoubleToInt64Bits(expected[i]) != BitConverter.DoubleToInt64Bits(actual[i]))
				{
					mismatches.Add($"case {count} [{i}] expected {expected[i]:R} got {actual[i]:R}");
				}
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(count).IsGreaterThan(0);
			await Assert.That(string.Join("\n", mismatches)).IsEqualTo(string.Empty);
		}
	}

	private static async Task AssertAllCases(string field, Func<double, double, bool> agrees)
	{
		var mismatches = new List<string>();
		foreach (JsonElement c in Cases.Value)
		{
			double[] expected = Read(c, field);
			double[] actual = Compute(c, field);
			for (int i = 0; i < expected.Length; i++)
			{
				if (!agrees(expected[i], actual[i]))
				{
					mismatches.Add($"q={string.Join(",", Read(c, "q").Select(x => x.ToString("R")))} [{i}] expected {expected[i]:R} got {actual[i]:R}");
				}
			}
		}

		// One string so a failure lists the first mismatching cases, not just a count.
		await Assert.That(string.Join("\n", mismatches.Take(10)) + $"\n{mismatches.Count} mismatches").IsEqualTo("\n0 mismatches");
	}
}
