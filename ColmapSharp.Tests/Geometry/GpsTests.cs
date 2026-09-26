// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GpsTests: colmap/geometry/gps_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, same reference values and tolerances (EigenMatrixNear is relative, see
// EigenMatchers.cs). Tests ColmapSharp/Geometry/GPSTransform.cs. The comparisons against
// pycolmap are in GeometryOracleTests (C#-only), which states the tier.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class GpsTests
{
	private const double EastOffset = 5.0e5;

	[Test]
	public async Task GPS_EllipsoidToECEFGRS80()
	{
		Vector3d[] ell =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		Vector3d[] refXyz =
		[
			new Vector3d(4.1772397090808507e6, 0.85515377993121441e6, 4.7282674046563692e6),
			new Vector3d(4.1772186604902023e6, 0.8551759313518483e6, 4.7282818502697079e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.GRS80);

		Vector3d[] xyz = gpsTform.EllipsoidToECEF(ell);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(xyz[i], refXyz[i], 1e-8)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_EllipsoidToECEFWGS84()
	{
		Vector3d[] ell =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		Vector3d[] refXyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		Vector3d[] xyz = gpsTform.EllipsoidToECEF(ell);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(xyz[i], refXyz[i], 1e-8)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ECEFToEllipsoid_GRS80()
	{
		Vector3d[] xyz =
		[
			new Vector3d(4.1772397090808507e6, 0.85515377993121441e6, 4.7282674046563692e6),
			new Vector3d(4.1772186604902023e6, 0.8551759313518483e6, 4.7282818502697079e6),
		];
		Vector3d[] refEll =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.GRS80);

		Vector3d[] ell = gpsTform.ECEFToEllipsoid(xyz);

		using (Assert.Multiple())
		{
			for (int i = 0; i < xyz.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(ell[i], refEll[i], 1e-5)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ECEFToEllipsoid_WGS84()
	{
		Vector3d[] xyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];
		Vector3d[] refEll =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		Vector3d[] ell = gpsTform.ECEFToEllipsoid(xyz);

		using (Assert.Multiple())
		{
			for (int i = 0; i < xyz.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(ell[i], refEll[i], 1e-5)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ECEFToEllipsoidipsoidToECEF_GRS80()
	{
		Vector3d[] xyz =
		[
			new Vector3d(4.177239709080851e6, 0.855153779931214e6, 4.728267404656370e6),
			new Vector3d(4.177218660490202e6, 0.855175931351848e6, 4.728281850269709e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.GRS80);

		Vector3d[] ell = gpsTform.ECEFToEllipsoid(xyz);
		Vector3d[] xyz2 = gpsTform.EllipsoidToECEF(ell);

		using (Assert.Multiple())
		{
			for (int i = 0; i < xyz.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(xyz[i], xyz2[i], 1e-5)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ECEFToEllipsoidipsoidToECEF_WGS84()
	{
		Vector3d[] xyz =
		[
			new Vector3d(4.177239709080851e6, 0.855153779931214e6, 4.728267404656370e6),
			new Vector3d(4.177218660490202e6, 0.855175931351848e6, 4.728281850269709e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		Vector3d[] ell = gpsTform.ECEFToEllipsoid(xyz);
		Vector3d[] xyz2 = gpsTform.EllipsoidToECEF(ell);

		using (Assert.Multiple())
		{
			for (int i = 0; i < xyz.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(xyz[i], xyz2[i], 1e-5)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_EllipsoidToENUWGS84()
	{
		Vector3d[] ell =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		Vector3d[] refXyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		// Get lat0, lon0 origin from ref
		Vector3d oriEll = gpsTform.ECEFToEllipsoid([refXyz[0]])[0];

		// Get ENU ref from ECEF ref
		Vector3d[] refEnu = gpsTform.ECEFToENU(refXyz, refXyz[0]);

		// Get ENU from Ell
		Vector3d[] enu = gpsTform.EllipsoidToENU(ell, oriEll.X, oriEll.Y, oriEll.Z);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(enu[i], refEnu[i], 1e-8)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ECEFToENU()
	{
		Vector3d[] ell =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		Vector3d[] refXyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		Vector3d[] xyz = gpsTform.EllipsoidToECEF(ell);

		// Get ENU from ECEF ref
		Vector3d[] refEnu = gpsTform.ECEFToENU(refXyz, refXyz[0]);

		// Get ENU from ECEF
		Vector3d[] enu = gpsTform.ECEFToENU(xyz, xyz[0]);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(enu[i], refEnu[i], 1e-8)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ENUToEllipsoidWGS84()
	{
		Vector3d[] refEll =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];

		Vector3d[] xyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		// Get lat0, lon0 origin from ref
		Vector3d[] oriEll = gpsTform.ECEFToEllipsoid(xyz);
		double lat0 = oriEll[0].X;
		double lon0 = oriEll[0].Y;
		double alt0 = oriEll[0].Z;

		// Get ENU from ECEF
		Vector3d[] enu = gpsTform.ECEFToENU(xyz, xyz[0]);

		// Unused in COLMAP's test too; kept so the same calls run.
		_ = gpsTform.ENUToECEF(enu, lat0, lon0, alt0);

		// Get Ell from ENU
		Vector3d[] ell = gpsTform.ENUToEllipsoid(enu, lat0, lon0, alt0);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(ell[i], refEll[i], 1e-5)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_ENUToECEF()
	{
		Vector3d[] ell =
		[
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		Vector3d[] refXyz =
		[
			new Vector3d(4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6),
			new Vector3d(4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);

		// Get lat0, lon0 origin from Ell
		double lat0 = ell[0].X;
		double lon0 = ell[0].Y;
		double alt0 = ell[0].Z;

		// Get ENU from Ell
		Vector3d[] enu = gpsTform.EllipsoidToENU(ell, lat0, lon0, alt0);

		// Get XYZ from ENU
		Vector3d[] xyz = gpsTform.ENUToECEF(enu, lat0, lon0, alt0);

		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(xyz[i], refXyz[i], 1e-8)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_EllipsoidToUTMWGS84()
	{
		Vector3d[] ell =
		[
			// (48.1476954472, 11.5695882694, 561.1851) zone32
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			// (48.1478904861, 11.5699366083, 561.1509) zone32
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
			// (48.1478904861, 12.5699366083, 561.1509) zone33
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 12 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];

		// Calculated from GeographicLib
		// echo 48.1476954472 11.5695882694 | TransverseMercatorProj -l 9 -p 9
		// echo 48.1478904861 11.5699366083 | TransverseMercatorProj -l 9 -p 9
		// echo 48.1478904861 12.5699366083 | TransverseMercatorProj -l 9 -p 9
		Vector3d[] refUtm =
		[
			new Vector3d(1.91125018424899e5 + EastOffset, 5.335909515367108e6, 561.1851),
			new Vector3d(1.91150201163177e5 + EastOffset, 5.335932057413140e6, 561.1509),
			new Vector3d(2.65520501819149e5 + EastOffset, 5.338903134602814e6, 561.1509),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);
		(Vector3d[] utm, int zone) = gpsTform.EllipsoidToUTM(ell);

		const double Tolerance = 1e-8; // 10nm
		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(utm[i], refUtm[i], Tolerance)).IsTrue();
			}

			await Assert.That(zone).IsEqualTo(32);
		}
	}

	[Test]
	public async Task GPS_EllipsoidToUTMGRS80()
	{
		Vector3d[] ell =
		[
			// (48.1476954472, 11.5695882694, 561.1851) zone32
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			// (48.1478904861, 11.5699366083, 561.1509) zone32
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
			// (48.1478904861, 12.5699366083, 561.1509) zone33
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 12 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];

		// Calculated from GeographicLib with -e 6378137.0 1.0/298.257222100882711243162837
		// echo 48.1476954472 11.5695882694 | TransverseMercatorProj -l 9 -p 9
		// echo 48.1478904861 11.5699366083 | TransverseMercatorProj -l 9 -p 9
		// echo 48.1478904861 12.5699366083 | TransverseMercatorProj -l 9 -p 9
		Vector3d[] refUtm =
		[
			new Vector3d(1.91125018426643e5 + EastOffset, 5.335909515244992e6, 561.1851),
			new Vector3d(1.91150201164921e5 + EastOffset, 5.335932057291023e6, 561.1509),
			new Vector3d(2.65520501821572e5 + EastOffset, 5.338903134480723e6, 561.1509),
		];

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.GRS80);
		(Vector3d[] utm, int zone) = gpsTform.EllipsoidToUTM(ell);

		const double Tolerance = 1e-8; // 10nm
		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(utm[i], refUtm[i], Tolerance)).IsTrue();
			}

			await Assert.That(zone).IsEqualTo(32);
		}
	}

	[Test]
	public async Task GPS_UTMToEllipsoidWGS84()
	{
		Vector3d[] utm =
		[
			new Vector3d(1.91125018424899e5 + EastOffset, 5.335909515367108e6, 561.1851),
			new Vector3d(1.91150201163177e5 + EastOffset, 5.335932057413140e6, 561.1509),
			new Vector3d(2.65520501819149e5 + EastOffset, 5.338903134602814e6, 561.1509),
		];

		// Calculated from GeographicLib
		// echo 191125.018424899 5335909.515367108|TransverseMercatorProj -l 9 -p 9 -r
		// echo 191150.201163177 5335932.057413140|TransverseMercatorProj -l 9 -p 9 -r
		// echo 265520.501819149 5338903.134602814|TransverseMercatorProj -l 9 -p 9 -r
		Vector3d[] refEll =
		[
			// (48.1476954472, 11.5695882694, 561.1851) zone32
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			// (48.1478904861, 11.5699366083, 561.1509) zone32
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
			// (48.1478904861, 12.5699366083, 561.1509) zone33
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 12 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);
		Vector3d[] ell = gpsTform.UTMToEllipsoid(utm, 32, true);

		const double Tolerance = 1e-8;
		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(ell[i], refEll[i], Tolerance)).IsTrue();
			}
		}
	}

	[Test]
	public async Task GPS_UTMToEllipsoidGRS80()
	{
		Vector3d[] utm =
		[
			new Vector3d(1.91125018426643e5 + EastOffset, 5.335909515244992e6, 561.1851),
			new Vector3d(1.91150201164921e5 + EastOffset, 5.335932057291023e6, 561.1509),
			new Vector3d(2.65520501821572e5 + EastOffset, 5.338903134480723e6, 561.1509),
		];

		// Calculated from GeographicLib with -e 6378137.0 1.0/298.257222100882711243162837
		// echo 191125.018424899 5335909.515367108|TransverseMercatorProj -l 9 -p 9 -r
		// echo 191150.201163177 5335932.057413140|TransverseMercatorProj -l 9 -p 9 -r
		// echo 265520.501819149 5338903.134602814|TransverseMercatorProj -l 9 -p 9 -r
		Vector3d[] refEll =
		[
			// (48.1476954472, 11.5695882694, 561.1851) zone32
			new Vector3d(48 + 8.0 / 60 + 51.70361 / 3600, 11 + 34.0 / 60 + 10.51777 / 3600, 561.1851),
			// (48.1478904861, 11.5699366083, 561.1509) zone32
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 11 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
			// (48.1478904861, 12.5699366083, 561.1509) zone33
			new Vector3d(48 + 8.0 / 60 + 52.40575 / 3600, 12 + 34.0 / 60 + 11.77179 / 3600, 561.1509),
		];
		// COLMAP's test converts the GRS80 coordinates with a WGS84 transform, not GRS80;
		// kept as written (1:1 port).

		var gpsTform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);
		Vector3d[] ell = gpsTform.UTMToEllipsoid(utm, 32, true);

		const double Tolerance = 1e-8;
		using (Assert.Multiple())
		{
			for (int i = 0; i < ell.Length; ++i)
			{
				await Assert.That(EigenMatrixNear(ell[i], refEll[i], Tolerance)).IsTrue();
			}
		}
	}
}
