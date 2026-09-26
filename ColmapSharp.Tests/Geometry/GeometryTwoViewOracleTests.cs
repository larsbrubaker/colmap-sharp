// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GeometryTwoViewOracleTests (C#-only; the ported pose, essential matrix, homography and
// triangulation tests check self-consistency and a few hand values): Pose.cs,
// EssentialMatrix.cs, HomographyMatrix.cs, Triangulation.cs and PosePrior.cs against
// COLMAP through the pycolmap 4.2.0 bindings. The fixture is written by
// oracle/geometry_two_view.py into TestData/oracle/geometry_two_view.json; that script lists
// which COLMAP call each field comes from.
//
// Tier B throughout (SVD, eigen and slerp paths, and 3x3 products whose grouping and FMA
// contraction differ from Eigen's in the macOS wheel, docs/CPP_DIVERGENCES.md entry 6).
// Each field is compared as |expected - actual| <= tolerance * max(1, |expected|); the
// tolerance is stated per field below with the largest gap observed when it was set.
// ComputeRot90FromGravity is exact. The homography scenes carry a little noise: noise-free,
// a planar scene has two physically valid decompositions that both triangulate every point
// exactly, and COLMAP's pick between them (and ours) comes down to rounding (see
// HomographyMatrix.cs). Results that COLMAP defines only up to sign are not in
// the fixture (the epipole), and the ones compared here are sign-independent by
// construction (see the headers of the ported files).

using System.Text.Json;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry;

public class GeometryTwoViewOracleTests
{
	private const string Fixture = "geometry_two_view.json";

	private static double[] Read(JsonElement element, string name)
	{
		JsonElement value = element.GetProperty(name);
		return value.ValueKind == JsonValueKind.Number ? [value.GetDouble()] : OracleFixture.Doubles(value);
	}

	private static Quaterniond Quat(double[] xyzw, int offset = 0) =>
		Quaterniond.FromCoeffs(new Vector4d(xyzw[offset], xyzw[offset + 1], xyzw[offset + 2], xyzw[offset + 3]));

	private static Vector3d Vec3(double[] values, int offset = 0) => new(values[offset], values[offset + 1], values[offset + 2]);

	private static Vector2d Vec2(double[] values, int offset = 0) => new(values[offset], values[offset + 1]);

	private static Matrix3d Mat3(double[] rowMajor) => new(
		rowMajor[0], rowMajor[1], rowMajor[2],
		rowMajor[3], rowMajor[4], rowMajor[5],
		rowMajor[6], rowMajor[7], rowMajor[8]);

	private static Matrix3x4d Mat34(double[] rowMajor) => Matrix3x4d.FromColumns(
		new Vector3d(rowMajor[0], rowMajor[4], rowMajor[8]),
		new Vector3d(rowMajor[1], rowMajor[5], rowMajor[9]),
		new Vector3d(rowMajor[2], rowMajor[6], rowMajor[10]),
		new Vector3d(rowMajor[3], rowMajor[7], rowMajor[11]));

	private static Vector3d[] Vec3s(double[] values) =>
		Enumerable.Range(0, values.Length / 3).Select(i => Vec3(values, 3 * i)).ToArray();

	private static IEnumerable<double> Flatten(Matrix3d m)
	{
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				yield return m[row, col];
			}
		}
	}

	private static IEnumerable<double> Flatten(IEnumerable<Vector3d> vectors) => vectors.SelectMany(v => new[] { v.X, v.Y, v.Z });

	// Largest |expected - actual| / max(1, |expected|) over a field.
	private static double Gap(IEnumerable<double> expected, IEnumerable<double> actual)
	{
		double[] e = expected.ToArray();
		double[] a = actual.ToArray();
		if (e.Length != a.Length)
		{
			return double.PositiveInfinity;
		}

		double gap = 0;
		for (int i = 0; i < e.Length; ++i)
		{
			gap = Math.Max(gap, Math.Abs(e[i] - a[i]) / Math.Max(1.0, Math.Abs(e[i])));
		}

		return gap;
	}

	// The fixture's quaternions are COLMAP's own; a rotation compares up to sign (q and -q).
	private static double QuatGap(double[] expectedXyzw, Quaterniond actual)
	{
		double[] a = [actual.X, actual.Y, actual.Z, actual.W];
		return Math.Min(Gap(expectedXyzw, a), Gap(expectedXyzw, a.Select(v => -v)));
	}

	[Test]
	public async Task TwoView_MatchesPycolmap()
	{
		var gaps = new Dictionary<string, double>();
		void Record(string name, double gap) => gaps[name] = Math.Max(gaps.GetValueOrDefault(name), gap);

		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("two_view").EnumerateArray())
		{
			var rel = new Rigid3d(Quat(Read(c, "rel_q")), Vec3(Read(c, "rel_t")));
			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(rel);
			Record("essential_matrix_from_pose", Gap(Read(c, "E"), Flatten(e)));

			double[] noisy1 = Read(c, "noisy1");
			double[] noisy2 = Read(c, "noisy2");
			Vector2d[] points1 = Enumerable.Range(0, noisy1.Length / 2).Select(i => Vec2(noisy1, 2 * i)).ToArray();
			Vector2d[] points2 = Enumerable.Range(0, noisy2.Length / 2).Select(i => Vec2(noisy2, 2 * i)).ToArray();
			var sampson = new List<double>();
			EssentialMatrix.ComputeSquaredSampsonError(points1, points2, Mat3(Read(c, "E")), sampson);
			Record("sampson", Gap(Read(c, "sampson"), sampson));

			Matrix3x4d p1 = Mat34(Read(c, "P1"));
			Matrix3x4d p2 = Mat34(Read(c, "P2"));
			Matrix3x4d p3 = Mat34(Read(c, "P3"));
			Vector3d[] rays1 = Vec3s(Read(c, "rays1"));
			Vector3d[] rays2 = Vec3s(Read(c, "rays2"));
			Vector3d[] rays3 = Vec3s(Read(c, "rays3"));

			var triPoints = new List<Vector3d>();
			var triBearings = new List<Vector3d>();
			var triMid = new List<Vector3d>();
			var triMulti = new List<Vector3d>();
			for (int i = 0; i < points1.Length; ++i)
			{
				Triangulation.TriangulatePoint(p1, p2, points1[i], points2[i], out Vector3d x1);
				Triangulation.TriangulatePoint(p1, p2, rays1[i], rays2[i], out Vector3d x2);
				Triangulation.TriangulateMidPoint(rel, rays1[i], rays2[i], out Vector3d x3);
				Triangulation.TriangulateMultiViewPoint([p1, p2, p3], [rays1[i], rays2[i], rays3[i]], out Vector3d x4);
				triPoints.Add(x1);
				triBearings.Add(x2);
				triMid.Add(x3);
				triMulti.Add(x4);
			}

			Record("triangulate_point", Gap(Read(c, "tri_points"), Flatten(triPoints)));
			Record("triangulate_point_bearings", Gap(Read(c, "tri_bearings"), Flatten(triBearings)));
			Record("triangulate_mid_point", Gap(Read(c, "tri_mid"), Flatten(triMid)));
			Record("triangulate_multi_view_point", Gap(Read(c, "tri_multi"), Flatten(triMulti)));

			Vector3d c1 = Vec3(Read(c, "c1"));
			Vector3d c2 = Vec3(Read(c, "c2"));
			double[] angles = Triangulation.CalculateTriangulationAngles(c1, c2, Vec3s(Read(c, "points")));
			Record("triangulation_angle", Gap(Read(c, "tri_angles"), angles));
		}

		// Bounds (largest observed gap when set, see the file header for the metric).
		var bounds = new Dictionary<string, double>
		{
			["essential_matrix_from_pose"] = 1e-15, // observed 1.1e-16
			["sampson"] = 1e-15, // observed 6.8e-19
			["triangulate_point"] = 1e-12, // observed 1.9e-14
			["triangulate_point_bearings"] = 1e-12, // observed 9.4e-15
			["triangulate_mid_point"] = 1e-12, // observed 2.3e-14
			["triangulate_multi_view_point"] = 1e-12, // observed 1.0e-14
			["triangulation_angle"] = 1e-12, // observed 2.4e-14 (acos of a near-1 cosine)
		};
		await AssertWithin(gaps, bounds);
	}

	[Test]
	public async Task PoseFromHomographyMatrix_MatchesPycolmap()
	{
		var gaps = new Dictionary<string, double>();
		void Record(string name, double gap) => gaps[name] = Math.Max(gaps.GetValueOrDefault(name), gap);

		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("homography").EnumerateArray())
		{
			var points3D = new List<Vector3d>();
			HomographyMatrix.PoseFromHomographyMatrix(
				Mat3(Read(c, "H")),
				Mat3(Read(c, "K1")),
				Mat3(Read(c, "K2")),
				Vec3s(Read(c, "rays1")),
				Vec3s(Read(c, "rays2")),
				out Rigid3d cam2FromCam1,
				out Vector3d normal,
				points3D);
			Record("rotation", QuatGap(Read(c, "q"), cam2FromCam1.Rotation));
			Record("translation", Gap(Read(c, "t"), [cam2FromCam1.Translation.X, cam2FromCam1.Translation.Y, cam2FromCam1.Translation.Z]));
			Record("normal", Gap(Read(c, "normal"), [normal.X, normal.Y, normal.Z]));
			Record("points3D", Gap(Read(c, "points3D"), Flatten(points3D)));
		}

		var bounds = new Dictionary<string, double>
		{
			["rotation"] = 1e-12, // observed 2.1e-14
			["translation"] = 1e-12, // observed 5.9e-14
			["normal"] = 1e-11, // observed 1.7e-13
			["points3D"] = 1e-10, // observed 1.6e-12 (depths up to ~10)
		};
		await AssertWithin(gaps, bounds);
	}

	[Test]
	public async Task Pose_MatchesPycolmap()
	{
		var gaps = new Dictionary<string, double>();
		void Record(string name, double gap) => gaps[name] = Math.Max(gaps.GetValueOrDefault(name), gap);
		var rot90Mismatches = new List<int>();

		int index = 0;
		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("pose").EnumerateArray())
		{
			double[] quatValues = Read(c, "quats");
			Quaterniond[] quats = Enumerable.Range(0, quatValues.Length / 4).Select(i => Quat(quatValues, 4 * i)).ToArray();
			Quaterniond average = Pose.AverageQuaternions(quats, Read(c, "weights"));
			Record("average_quaternions", QuatGap(Read(c, "average"), average));

			var a = new Rigid3d(Quat(Read(c, "a_q")), Vec3(Read(c, "a_t")));
			var b = new Rigid3d(Quat(Read(c, "b_q")), Vec3(Read(c, "b_t")));
			Rigid3d interp = Pose.InterpolateCameraPoses(a, b, Read(c, "t")[0]);
			Record("interpolate_rotation", QuatGap(Read(c, "interp_q"), interp.Rotation));
			Record("interpolate_translation", Gap(Read(c, "interp_t"), [interp.Translation.X, interp.Translation.Y, interp.Translation.Z]));

			if (PosePrior.ComputeRot90FromGravity(Vec3(Read(c, "gravity"))) != c.GetProperty("rot90").GetInt32())
			{
				rot90Mismatches.Add(index);
			}

			index++;
		}

		var bounds = new Dictionary<string, double>
		{
			["average_quaternions"] = 1e-14, // observed 7.8e-16
			["interpolate_rotation"] = 1e-15, // observed 3.3e-16
			["interpolate_translation"] = 1e-15, // observed 0 (bit-identical)
		};
		await Assert.That(rot90Mismatches).IsEmpty();
		await AssertWithin(gaps, bounds);
	}

	private static async Task AssertWithin(Dictionary<string, double> gaps, Dictionary<string, double> bounds)
	{
		string[] over = bounds
			.Where(b => !(gaps.GetValueOrDefault(b.Key, double.NaN) <= b.Value))
			.Select(b => $"{b.Key}: gap {gaps.GetValueOrDefault(b.Key, double.NaN):R} > {b.Value:R}")
			.ToArray();
		await Assert.That(over).IsEmpty();
	}
}
