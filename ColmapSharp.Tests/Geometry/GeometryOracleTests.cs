// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GeometryOracleTests (C#-only; rigid3_test.cc, sim3_test.cc and gps_test.cc only check
// self-consistency): Rigid3d, Sim3d and GPSTransform against COLMAP through the pycolmap
// 4.2.0 bindings. The fixture is written by oracle/geometry_transforms.py into
// TestData/oracle/geometry_transforms.json; that script lists which COLMAP call each field
// comes from.
//
// Tiers, per function, as the fixture shows them:
// - Tier A, bit-identical on every case (ExactFields and Strings): Rigid3d composition's
//   and Inverse's rotation, ToMatrix, FromMatrix, Adjoint; Sim3d composition's and
//   Inverse's scale and rotation, ToMatrix, FromMatrix; EllipsoidToUTM's zone; the
//   operator<< strings and Sim3d::ToFile's text.
// - Tier B (ToleranceFields), each for an FMA contraction in the macOS arm64 wheel that
//   ColmapSharp deliberately does not reproduce (divergence 6):
//   everything that rotates a vector with q * v (Rigid3d/Sim3d point transform, the
//   translations of composition and Inverse, TgtOriginInSrc); AdjointInverse and the
//   covariance helpers GetCovarianceForRigid3dInverse, GetCovarianceForComposedRigid3d and
//   GetCovarianceForRelativeRigid3d (Eigen's 3x3, 6x6 and 6x12 products); and the GPS
//   ellipsoid/ECEF/ENU/UTM conversions, whose multiply-adds the wheel fuses (for example
//   N * (1 - e2) + alt in EllipsoidToECEF: fusing it takes that coordinate from
//   13/80 mismatches to 3/80). UTMToEllipsoid is Tier B for an unexplained 1-ulp
//   latitude difference on 2/80 points (divergence 11).
// Tolerances: |expected - actual| <= relative * max(1, |expected|), and for coordinates in
// meters in the GPS conversions (ECEF, ENU, UTM and altitude) additionally within 1e-8 m:
// those come out of ~6.4e6 m ECEF values whose last-bit differences (9.3e-10 m) carry
// through subtractions to small results. The observed gaps are at most a few ulps of the
// inputs: 3.6e-15 relative for q * v, 3e-14 for the covariances (1e-13 allowed), 9.3e-10 m for GPS.

using System.Text.Json;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry;

public class GeometryOracleTests
{
	private const string Fixture = "geometry_transforms.json";

	private static double[] Read(JsonElement element, string name)
	{
		JsonElement value = element.GetProperty(name);
		if (value.ValueKind == JsonValueKind.Number)
		{
			return [value.GetDouble()];
		}

		return OracleFixture.Doubles(value);
	}

	private static Quaterniond Quat(JsonElement c, string name)
	{
		double[] xyzw = Read(c, name);
		return Quaterniond.FromCoeffs(new Vector4d(xyzw[0], xyzw[1], xyzw[2], xyzw[3]));
	}

	private static Vector3d Vec3(JsonElement c, string name)
	{
		double[] v = Read(c, name);
		return new Vector3d(v[0], v[1], v[2]);
	}

	private static Vector3d[] Points(JsonElement c, string name)
	{
		double[] v = Read(c, name);
		var points = new Vector3d[v.Length / 3];
		for (int i = 0; i < points.Length; i++)
		{
			points[i] = new Vector3d(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
		}

		return points;
	}

	private static Matrix3x4d Matrix3x4RowMajor(JsonElement c, string name)
	{
		double[] m = Read(c, name);
		return new Matrix3x4d(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
	}

	private static Matrix6d Matrix6RowMajor(JsonElement c, string name)
	{
		// Row-major values read as column-major give the transpose; transpose back.
		return Matrix6d.FromColumnMajor(Read(c, name)).Transpose();
	}

	private static double[] Flatten(Quaterniond q) => [q.X, q.Y, q.Z, q.W];

	private static double[] Flatten(Vector3d v) => [v.X, v.Y, v.Z];

	private static double[] Flatten(Vector3d[] points) => points.SelectMany(Flatten).ToArray();

	private static double[] Flatten(Matrix3x4d m)
	{
		var values = new double[12];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				values[i * 4 + j] = m[i, j];
			}
		}

		return values;
	}

	private static double[] Flatten(Matrix6d m)
	{
		var values = new double[36];
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				values[i * 6 + j] = m[i, j];
			}
		}

		return values;
	}

	private static double[] ComputeRigid(JsonElement c, string field)
	{
		var r = new Rigid3d(Quat(c, "q"), Vec3(c, "t"));
		var other = new Rigid3d(Quat(c, "other_q"), Vec3(c, "other_t"));
		return field switch
		{
			"apply" => Flatten(r * Vec3(c, "x")),
			"compose_q" => Flatten((r * other).Rotation),
			"compose_t" => Flatten((r * other).Translation),
			"inverse_q" => Flatten(r.Inverse().Rotation),
			"inverse_t" => Flatten(r.Inverse().Translation),
			"matrix" => Flatten(r.ToMatrix()),
			"from_matrix_q" => Flatten(Rigid3d.FromMatrix(Matrix3x4RowMajor(c, "general")).Rotation),
			"from_matrix_t" => Flatten(Rigid3d.FromMatrix(Matrix3x4RowMajor(c, "general")).Translation),
			"tgt_origin_in_src" => Flatten(r.TgtOriginInSrc()),
			"adjoint" => Flatten(r.Adjoint()),
			"adjoint_inverse" => Flatten(r.AdjointInverse()),
			"cov_inverse" => Flatten(Rigid3d.GetCovarianceForRigid3dInverse(r, Matrix6RowMajor(c, "cov"))),
			"cov_composed" => Flatten(Rigid3d.GetCovarianceForComposedRigid3d(r, MatrixXd.FromRowMajor(12, 12, Read(c, "cov12")))),
			"cov_relative" => Flatten(Rigid3d.GetCovarianceForRelativeRigid3d(r, other, MatrixXd.FromRowMajor(12, 12, Read(c, "cov12")))),
			_ => throw new ArgumentException(field),
		};
	}

	private static double[] ComputeSim(JsonElement c, string field)
	{
		var s = new Sim3d(Read(c, "s")[0], Quat(c, "q"), Vec3(c, "t"));
		var other = new Sim3d(Read(c, "other_s")[0], Quat(c, "other_q"), Vec3(c, "other_t"));
		return field switch
		{
			"apply" => Flatten(s * Vec3(c, "x")),
			"compose_s" => [(s * other).Scale],
			"compose_q" => Flatten((s * other).Rotation),
			"compose_t" => Flatten((s * other).Translation),
			"inverse_s" => [s.Inverse().Scale],
			"inverse_q" => Flatten(s.Inverse().Rotation),
			"inverse_t" => Flatten(s.Inverse().Translation),
			"matrix" => Flatten(s.ToMatrix()),
			"from_matrix_s" => [Sim3d.FromMatrix(Matrix3x4RowMajor(c, "general")).Scale],
			"from_matrix_q" => Flatten(Sim3d.FromMatrix(Matrix3x4RowMajor(c, "general")).Rotation),
			"from_matrix_t" => Flatten(Sim3d.FromMatrix(Matrix3x4RowMajor(c, "general")).Translation),
			_ => throw new ArgumentException(field),
		};
	}

	private static double[] ComputeGps(JsonElement c, string field)
	{
		var gps = new GPSTransform(Enum.Parse<GPSTransform.Ellipsoid>(c.GetProperty("ellipsoid").GetString()!));
		Vector3d[] lla = Points(c, "lla");
		Vector3d reference = Vec3(c, "ref");
		Vector3d[] ecef = Points(c, "ecef");
		Vector3d[] enu = Points(c, "enu");
		return field switch
		{
			"ecef" => Flatten(gps.EllipsoidToECEF(lla)),
			"ecef_to_ellipsoid" => Flatten(gps.ECEFToEllipsoid(ecef)),
			"enu" => Flatten(gps.EllipsoidToENU(lla, reference.X, reference.Y, reference.Z)),
			"ecef_to_enu" => Flatten(gps.ECEFToENU(ecef, ecef[0])),
			"enu_to_ellipsoid" => Flatten(gps.ENUToEllipsoid(enu, reference.X, reference.Y, reference.Z)),
			"enu_to_ecef" => Flatten(gps.ENUToECEF(enu, reference.X, reference.Y, reference.Z)),
			"utm" => Flatten(gps.EllipsoidToUTM(lla).Points),
			"utm_zone" => [gps.EllipsoidToUTM(lla).Zone],
			"utm_to_ellipsoid" => Flatten(gps.UTMToEllipsoid(
				Points(c, "utm"), c.GetProperty("zone").GetInt32(), c.GetProperty("is_north").GetBoolean())),
			_ => throw new ArgumentException(field),
		};
	}

	private static double[] Compute(string section, JsonElement c, string field) => section switch
	{
		"rigid" => ComputeRigid(c, field),
		"sim" => ComputeSim(c, field),
		"gps" => ComputeGps(c, field),
		_ => throw new ArgumentException(section),
	};

	private static double[] Expected(JsonElement c, string field)
	{
		return field == "utm_zone" ? [c.GetProperty("zone").GetInt32()] : Read(c, field);
	}

	// Compares every case of one field; returns the mismatch descriptions (empty when all match).
	private static List<string> Mismatches(string section, string field, Func<double, double, int, bool> equal)
	{
		var mismatches = new List<string>();
		int index = 0;
		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty(section).EnumerateArray())
		{
			double[] expected = Expected(c, field);
			double[] actual = Compute(section, c, field);
			for (int i = 0; i < expected.Length; i++)
			{
				if (!equal(expected[i], actual[i], i))
				{
					mismatches.Add($"case {index} [{i}]: expected {expected[i]:R}, got {actual[i]:R}");
				}
			}

			index++;
		}

		return mismatches;
	}

	[Test]
	[Arguments("rigid", "compose_q")]
	[Arguments("rigid", "inverse_q")]
	[Arguments("rigid", "matrix")]
	[Arguments("rigid", "from_matrix_q")]
	[Arguments("rigid", "from_matrix_t")]
	[Arguments("rigid", "adjoint")]
	[Arguments("sim", "compose_s")]
	[Arguments("sim", "compose_q")]
	[Arguments("sim", "inverse_s")]
	[Arguments("sim", "inverse_q")]
	[Arguments("sim", "matrix")]
	[Arguments("sim", "from_matrix_s")]
	[Arguments("sim", "from_matrix_q")]
	[Arguments("sim", "from_matrix_t")]
	[Arguments("gps", "utm_zone")]
	public async Task ExactFields(string section, string field)
	{
		List<string> mismatches = Mismatches(section, field, (expected, actual, component) =>
			BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual));
		await Assert.That(string.Join("\n", mismatches)).IsEqualTo(string.Empty);
	}

	[Test]
	[Arguments("rigid", "apply", 1e-14)]
	[Arguments("rigid", "compose_t", 1e-14)]
	[Arguments("rigid", "inverse_t", 1e-14)]
	[Arguments("rigid", "tgt_origin_in_src", 1e-14)]
	[Arguments("rigid", "adjoint_inverse", 1e-14)]
	[Arguments("rigid", "cov_inverse", 1e-13)]
	[Arguments("rigid", "cov_composed", 1e-13)]
	[Arguments("rigid", "cov_relative", 1e-13)]
	[Arguments("sim", "apply", 1e-14)]
	[Arguments("sim", "compose_t", 1e-14)]
	[Arguments("sim", "inverse_t", 1e-14)]
	[Arguments("gps", "ecef", 1e-14)]
	[Arguments("gps", "ecef_to_ellipsoid", 1e-14)]
	[Arguments("gps", "enu", 1e-14)]
	[Arguments("gps", "ecef_to_enu", 1e-14)]
	[Arguments("gps", "enu_to_ellipsoid", 1e-14)]
	[Arguments("gps", "enu_to_ecef", 1e-14)]
	[Arguments("gps", "utm", 1e-14)]
	[Arguments("gps", "utm_to_ellipsoid", 1e-14)]
	public async Task ToleranceFields(string section, string field, double relative)
	{
		const double MetersTolerance = 1e-8;
		List<string> mismatches = Mismatches(section, field, (expected, actual, component) =>
		{
			double difference = Math.Abs(expected - actual);
			return difference <= relative * Math.Max(1.0, Math.Abs(expected))
				|| (IsMeters(section, field, component) && difference <= MetersTolerance);
		});
		await Assert.That(string.Join("\n", mismatches)).IsEqualTo(string.Empty);
	}

	// Whether coefficient `component` of a field is a coordinate in meters: every GPS output
	// except the (lat, lon) of the ellipsoidal ones.
	private static bool IsMeters(string section, string field, int component)
	{
		if (section != "gps")
		{
			return false;
		}

		return !field.EndsWith("ellipsoid", StringComparison.Ordinal) || component % 3 == 2;
	}

	[Test]
	public async Task Strings()
	{
		var mismatches = new List<string>();
		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("rigid").EnumerateArray())
		{
			string actual = new Rigid3d(Quat(c, "q"), Vec3(c, "t")).ToString();
			string expected = c.GetProperty("str").GetString()!;
			if (actual != expected)
			{
				mismatches.Add($"expected {expected}, got {actual}");
			}
		}

		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("sim").EnumerateArray())
		{
			var sim = new Sim3d(Read(c, "s")[0], Quat(c, "q"), Vec3(c, "t"));
			string expected = c.GetProperty("str").GetString()!;
			if (sim.ToString() != expected)
			{
				mismatches.Add($"expected {expected}, got {sim}");
			}

			string path = Path.Combine(Path.GetTempPath(), $"ColmapSharpSim3d_{Guid.NewGuid():N}.txt");
			try
			{
				sim.ToFile(path);
				string written = File.ReadAllText(path);
				string expectedFile = c.GetProperty("to_file").GetString()! + "\n";
				if (written != expectedFile)
				{
					mismatches.Add($"ToFile: expected {expectedFile}, got {written}");
				}
			}
			finally
			{
				File.Delete(path);
			}
		}

		await Assert.That(string.Join("\n", mismatches)).IsEqualTo(string.Empty);
	}
}
