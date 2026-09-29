// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GPSTransform: colmap/geometry/gps.h and gps.cc - conversions between ellipsoidal GPS
// coordinates (latitude, longitude in degrees, altitude in meters), ECEF, local ENU and UTM,
// on the GRS80 or WGS84 ellipsoid. Photo GPS tags reach the pose priors through it. Uses
// Mathematics/MathUtils (DegToRad/RadToDeg) and LinearAlgebra/Matrix3d. Tests:
// ColmapSharp.Tests/Geometry/GpsTests.cs (gps_test.cc 1:1) and GeometryOracleTests.cs
// (C#-only, against pycolmap).
//
// Tiers (pinned by GeometryOracleTests): EllipsoidToUTM's zone is Tier A, bit-identical
// to pycolmap. UTMToEllipsoid is Tier B: its latitude can be 1 ulp off for a reason not
// established (divergence 11). The other conversions are Tier B: the macOS wheel
// fuses some of their multiply-adds into FMAs (divergence 6), which
// ColmapSharp does not, so coordinates can differ in the last bit of the ~6.4e6 m ECEF
// values (up to 9.3e-10 m). sin, cos and friends are the platform libm's on both sides
// (.NET's Math calls libm) and agree on the fixture. Plain scalar arithmetic otherwise
// follows gps.cc term for term.
//
// Translation notes:
// - std::vector<Eigen::Vector3d> in and out becomes IReadOnlyList<Vector3d> in and a
//   Vector3d[] out; EllipsoidToUTM's std::pair becomes a (Points, Zone) tuple.
// - MAKE_ENUM_CLASS(Ellipsoid, 0, GRS80, WGS84) becomes a plain enum with those values.
// - COLMAP also stores the semiminor axis b_ = (1 - f) * a but never reads it; it is not
//   kept here.
// - UTMToEllipsoid passes its bool hemisphere to N0(double), as the C++ does, so north
//   (true, 1.0 > 0) gets a false northing of 0 and south 10,000 km.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

using static ColmapSharp.Mathematics.MathUtils;

namespace ColmapSharp.Geometry;

/// <summary>
/// Transform ellipsoidal GPS coordinates to Cartesian coordinate systems and vice versa.
/// Port of colmap::GPSTransform.
/// </summary>
public sealed class GPSTransform
{
	/// <summary>The reference ellipsoid. Port of GPSTransform::Ellipsoid.</summary>
	public enum Ellipsoid
	{
		/// <summary>Geodetic Reference System 1980.</summary>
		GRS80 = 0,

		/// <summary>World Geodetic System 1984.</summary>
		WGS84 = 1,
	}

	private readonly double a; // Semimajor axis.
	private readonly double f; // Flattening.
	private readonly double e2; // Numerical eccentricity squared.

	/// <summary>Creates the transform for the given ellipsoid (GRS80 by default, as in COLMAP).</summary>
	/// <exception cref="ArgumentException">The ellipsoid is not defined.</exception>
	public GPSTransform(Ellipsoid ellipsoid = Ellipsoid.GRS80)
	{
		switch (ellipsoid)
		{
			case Ellipsoid.GRS80:
				a = 6378137.0;
				f = 1.0 / 298.257222100882711243162837;
				break;
			case Ellipsoid.WGS84:
				a = 6378137.0;
				f = 1.0 / 298.257223563;
				break;
			default:
				throw new ArgumentException("Ellipsoid not defined", nameof(ellipsoid));
		}

		e2 = f * (2.0 - f);
	}

	/// <summary>Convert ellipsoidal (lat/lon/alt) to ECEF coordinates.</summary>
	public Vector3d[] EllipsoidToECEF(IReadOnlyList<Vector3d> latLonAlt)
	{
		var xyzInEcef = new Vector3d[latLonAlt.Count];
		for (int i = 0; i < latLonAlt.Count; ++i)
		{
			double lat = DegToRad(latLonAlt[i].X);
			double lon = DegToRad(latLonAlt[i].Y);
			double alt = latLonAlt[i].Z;

			double sinLat = Math.Sin(lat);
			double sinLon = Math.Sin(lon);
			double cosLat = Math.Cos(lat);
			double cosLon = Math.Cos(lon);

			// Prime vertical radius of curvature.
			double n = a / Math.Sqrt(1 - e2 * sinLat * sinLat);

			xyzInEcef[i] = new Vector3d(
				(n + alt) * cosLat * cosLon,
				(n + alt) * cosLat * sinLon,
				(n * (1 - e2) + alt) * sinLat);
		}

		return xyzInEcef;
	}

	/// <summary>Convert ECEF to ellipsoidal (lat/lon/alt) coordinates.</summary>
	public Vector3d[] ECEFToEllipsoid(IReadOnlyList<Vector3d> xyzInEcef)
	{
		var latLonAlt = new Vector3d[xyzInEcef.Count];
		for (int i = 0; i < latLonAlt.Length; ++i)
		{
			double x = xyzInEcef[i].X;
			double y = xyzInEcef[i].Y;
			double z = xyzInEcef[i].Z;

			double radiusXy = Math.Sqrt(x * x + y * y);
			const double Eps = 1e-12;

			// Iteratively solve for latitude and altitude.
			double lat = Math.Atan2(z, radiusXy);
			double alt = 0.0;

			for (int j = 0; j < 100; ++j)
			{
				double sinLat = Math.Sin(lat);
				double n = a / Math.Sqrt(1 - e2 * sinLat * sinLat);
				double prevAlt = alt;
				alt = radiusXy / Math.Cos(lat) - n;
				double prevLat = lat;
				lat = Math.Atan((z / radiusXy) * 1 / (1 - e2 * n / (n + alt)));

				if (Math.Abs(prevLat - lat) < Eps && Math.Abs(prevAlt - alt) < Eps)
				{
					break;
				}
			}

			latLonAlt[i] = new Vector3d(RadToDeg(lat), RadToDeg(Math.Atan2(y, x)), alt);
		}

		return latLonAlt;
	}

	/// <summary>
	/// Convert ellipsoidal (lat/lon/alt) to ENU coordinates. The reference point
	/// (refLat, refLon, refAlt) defines the ENU origin.
	/// </summary>
	public Vector3d[] EllipsoidToENU(IReadOnlyList<Vector3d> latLonAlt, double refLat, double refLon, double refAlt)
	{
		Vector3d[] xyzInEcef = EllipsoidToECEF(latLonAlt);
		Vector3d refEcef = EllipsoidToECEF([new Vector3d(refLat, refLon, refAlt)])[0];
		return ECEFToENU(xyzInEcef, refEcef);
	}

	/// <summary>
	/// Convert ECEF to ENU coordinates. The reference point <paramref name="refEcef"/>
	/// defines the ENU origin.
	/// </summary>
	public Vector3d[] ECEFToENU(IReadOnlyList<Vector3d> xyzInEcef, Vector3d refEcef)
	{
		// Reference: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
		var xyzInEnu = new Vector3d[xyzInEcef.Count];

		// Compute lat/lon of reference point for rotation matrix.
		Vector3d refEll = ECEFToEllipsoid([refEcef])[0];
		Matrix3d r = EcefToEnuRotation(refEll.X, refEll.Y);

		for (int i = 0; i < xyzInEcef.Count; ++i)
		{
			xyzInEnu[i] = r * (xyzInEcef[i] - refEcef);
		}

		return xyzInEnu;
	}

	/// <summary>
	/// Convert ENU to ellipsoidal (lat/lon/alt) coordinates. The reference point
	/// (refLat, refLon, refAlt) defines the ENU origin.
	/// </summary>
	public Vector3d[] ENUToEllipsoid(IReadOnlyList<Vector3d> xyzInEnu, double refLat, double refLon, double refAlt)
	{
		return ECEFToEllipsoid(ENUToECEF(xyzInEnu, refLat, refLon, refAlt));
	}

	/// <summary>
	/// Convert ENU to ECEF coordinates. The reference point (refLat, refLon, refAlt)
	/// defines the ENU origin.
	/// </summary>
	public Vector3d[] ENUToECEF(IReadOnlyList<Vector3d> xyzInEnu, double refLat, double refLon, double refAlt)
	{
		var xyzInEcef = new Vector3d[xyzInEnu.Count];

		// Compute ECEF coordinates of the reference point.
		Vector3d refXyzInEcef = EllipsoidToECEF([new Vector3d(refLat, refLon, refAlt)])[0];

		// Build ENU to ECEF rotation matrix (transpose of ECEF to ENU).
		Matrix3d r = EcefToEnuRotation(refLat, refLon).Transpose();

		for (int i = 0; i < xyzInEnu.Count; ++i)
		{
			xyzInEcef[i] = (r * xyzInEnu[i]) + refXyzInEcef;
		}

		return xyzInEcef;
	}

	/// <summary>
	/// Convert ellipsoidal (lat/lon/alt) to UTM coordinates. Returns the converted
	/// coordinates and the zone number. If the points span multiple zones, the zone with the
	/// most points is chosen (the lowest such zone on a tie). The conversion uses a
	/// 4th-order expansion formula. The easting offset is 500 km, and the northing offset is
	/// 10,000 km for the Southern Hemisphere.
	/// </summary>
	/// <exception cref="ArgumentException">A latitude is outside [-90, 90] or a longitude outside [-180, 180].</exception>
	public (Vector3d[] Points, int Zone) EllipsoidToUTM(IReadOnlyList<Vector3d> latLonAlt)
	{
		// Reference:
		// https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

		var p = new UTMParams(a / 1000.0, f); // Convert to kilometers.

		// Select the predominant zone when points span multiple zones.
		var zoneCounts = new int[60];
		foreach (Vector3d lla in latLonAlt)
		{
			Check.Ge(lla.X, -90.0);
			Check.Le(lla.X, 90.0);
			Check.Ge(lla.Y, -180.0);
			Check.Le(lla.Y, 180.0);

			// Longitude 180 maps to zone 61, one past the end of the counts. In C++ that is
			// an out-of-bounds write; here it is an IndexOutOfRangeException.
			int zIndex = UTMParams.MeridianToZone(lla.Y) - 1;
			++zoneCounts[zIndex];
		}

		// std::max_element returns the first maximum.
		int best = 0;
		for (int i = 1; i < zoneCounts.Length; i++)
		{
			if (zoneCounts[best] < zoneCounts[i])
			{
				best = i;
			}
		}

		int zone = best + 1;
		double lambda0 = DegToRad(UTMParams.ZoneToCentralMeridian(zone));

		var xyzInUtm = new Vector3d[latLonAlt.Count];
		for (int index = 0; index < latLonAlt.Count; index++)
		{
			Vector3d lla = latLonAlt[index];
			double phi = DegToRad(lla.X);
			double lambda = DegToRad(lla.Y);

			double t = Math.Sinh(Math.Atanh(Math.Sin(phi))
				- 2 * Math.Sqrt(p.N[1]) / (1 + p.N[1])
				* Math.Atanh(2 * Math.Sqrt(p.N[1]) / (1 + p.N[1]) * Math.Sin(phi)));
			double xi = Math.Atan(t / Math.Cos(lambda - lambda0));
			double eta = Math.Atanh(Math.Sin(lambda - lambda0) / Math.Sqrt(1 + t * t));

			double e = eta;
			double n = xi;
			for (int i = 0; i < UTMParams.Order; ++i)
			{
				double doubledIndex = 2.0 * (i + 1);
				e += p.Alpha[i] * Math.Cos(doubledIndex * xi) * Math.Sinh(doubledIndex * eta);
				n += p.Alpha[i] * Math.Sin(doubledIndex * xi) * Math.Cosh(doubledIndex * eta);
			}

			e = UTMParams.E0 + UTMParams.K0 * p.A * e;
			n = UTMParams.N0(lla.X) + UTMParams.K0 * p.A * n;

			xyzInUtm[index] = new Vector3d(e * 1000, n * 1000, lla.Z); // Convert to meters.
		}

		return (xyzInUtm, zone);
	}

	/// <summary>
	/// Convert UTM to ellipsoidal (lat/lon/alt) coordinates. Requires the zone number and
	/// hemisphere (true for north, false for south).
	/// </summary>
	/// <exception cref="ArgumentException">The zone is outside [1, 60].</exception>
	public Vector3d[] UTMToEllipsoid(IReadOnlyList<Vector3d> xyzInUtm, int zone, bool isNorth)
	{
		// Reference:
		// https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

		Check.Ge(zone, 1);
		Check.Le(zone, 60);

		var p = new UTMParams(a / 1000.0, f); // Convert to kilometers.

		var latLonAlt = new Vector3d[xyzInUtm.Count];
		for (int index = 0; index < xyzInUtm.Count; index++)
		{
			Vector3d ena = xyzInUtm[index];
			double xi = (ena.Y / 1000.0 - UTMParams.N0(isNorth ? 1.0 : 0.0)) / (UTMParams.K0 * p.A);
			double eta = (ena.X / 1000.0 - UTMParams.E0) / (UTMParams.K0 * p.A);

			double xiPrime = 0.0;
			double etaPrime = 0.0;
			for (int i = 0; i < UTMParams.Order; ++i)
			{
				double doubledIndex = 2.0 * (i + 1);
				xiPrime += p.Beta[i] * Math.Sin(doubledIndex * xi) * Math.Cosh(doubledIndex * eta);
				etaPrime += p.Beta[i] * Math.Cos(doubledIndex * xi) * Math.Sinh(doubledIndex * eta);
			}

			xiPrime = xi - xiPrime;
			etaPrime = eta - etaPrime;
			double chi = Math.Asin(Math.Sin(xiPrime) / Math.Cosh(etaPrime));

			double phi = chi;
			for (int i = 0; i < UTMParams.Order; ++i)
			{
				double doubledIndex = 2.0 * (i + 1);
				phi += p.Delta[i] * Math.Sin(doubledIndex * chi);
			}

			double lat = RadToDeg(phi);
			double lon = UTMParams.ZoneToCentralMeridian(zone)
				+ RadToDeg(Math.Atan(Math.Sinh(etaPrime) / Math.Cos(xiPrime)));

			latLonAlt[index] = new Vector3d(lat, lon, ena.Z);
		}

		return latLonAlt;
	}

	// The ECEF-to-ENU rotation for a reference latitude and longitude in degrees, the same
	// comma-initialized matrix both ECEFToENU and ENUToECEF build in gps.cc.
	private static Matrix3d EcefToEnuRotation(double refLat, double refLon)
	{
		double cosLat = Math.Cos(DegToRad(refLat));
		double sinLat = Math.Sin(DegToRad(refLat));
		double cosLon = Math.Cos(DegToRad(refLon));
		double sinLon = Math.Sin(DegToRad(refLon));

		return new Matrix3d(
			-sinLon, cosLon, 0.0,
			-sinLat * cosLon, -sinLat * sinLon, cosLat,
			cosLat * cosLon, cosLat * sinLon, sinLat);
	}

	// Series coefficients of the UTM projection. Notation from:
	// https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
	private sealed class UTMParams
	{
		// Order of the series expansion, determining the precision.
		public const int Order = 4;

		// UTM scale factor at the central meridian.
		public const double K0 = 0.9996;

		// Easting of the origin in km.
		public const double E0 = 500;

		// Powers of n, where N[i] = n^i.
		public readonly double[] N = new double[Order + 1];

		// Alpha, beta and delta coefficients for the series expansions.
		public readonly double[] Alpha;
		public readonly double[] Beta;
		public readonly double[] Delta;

		// Multiplicative factor.
		public readonly double A;

		public UTMParams(double a, double f)
		{
			N[0] = 1;
			N[1] = f / (2.0 - f);
			for (int i = 2; i < Order + 1; ++i)
			{
				N[i] = N[1] * N[i - 1];
			}

			// The constant quotients (1.0 / 2.0 and so on) fold at compile time in both
			// languages to the same correctly rounded double.
			Alpha =
			[
				1.0 / 2.0 * N[1] - 2.0 / 3.0 * N[2] + 5.0 / 16.0 * N[3] + 41.0 / 180.0 * N[4],
				13.0 / 48.0 * N[2] - 3.0 / 5.0 * N[3] + 557.0 / 1440.0 * N[4],
				61.0 / 240.0 * N[3] - 103.0 / 140.0 * N[4],
				49561.0 / 161280.0 * N[4],
			];

			Beta =
			[
				1.0 / 2.0 * N[1] - 2.0 / 3.0 * N[2] + 37.0 / 96.0 * N[3] - 1.0 / 360.0 * N[4],
				1.0 / 48.0 * N[2] + 1.0 / 15.0 * N[3] - 437.0 / 1440.0 * N[4],
				17.0 / 480.0 * N[3] - 37.0 / 840.0 * N[4],
				4397.0 / 161280.0 * N[4],
			];

			Delta =
			[
				2.0 * N[1] - 2.0 / 3.0 * N[2] - 2.0 * N[3] - 116.0 / 45.0 * N[4],
				7.0 / 3.0 * N[2] - 8.0 / 5.0 * N[3] - 227.0 / 45.0 * N[4],
				56.0 / 15.0 * N[3] - 136.0 / 35.0 * N[4],
				4279.0 / 630.0 * N[4],
			];

			A = a / (1.0 + N[1]) * (1.0 + N[2] / 4.0 + N[4] / 64.0);
		}

		// Northing of the origin in km.
		public static double N0(double latOrHemi) => latOrHemi > 0 ? 0 : 1e4;

		public static double ZoneToCentralMeridian(int zone) => 6 * zone - 183;

		public static int MeridianToZone(double meridian) => (int)Math.Floor((meridian + 180) / 6) + 1;
	}
}
