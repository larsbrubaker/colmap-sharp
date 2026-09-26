// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SyntheticOptions: the option structs of colmap/scene/synthetic.h (SyntheticDatasetOptions,
// SyntheticNoiseOptions, SyntheticImageOptions), with COLMAP's defaults. The generator that
// reads them is Synthetic.cs (+ Synthetic.Matches.cs). Tests:
// ColmapSharp.Tests/Scene/SyntheticTests.cs.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

namespace ColmapSharp.Scene;

/// <summary>Port of SyntheticDatasetOptions::MatchConfig.</summary>
public enum SyntheticMatchConfig
{
	/// <summary>Exhaustive matches between all pairs of observations of a 3D point.</summary>
	Exhaustive = 1,

	/// <summary>
	/// Chain of matches between images with consecutive identifiers, i.e., there are only
	/// matches between image pairs (image_id, image_id+1).
	/// </summary>
	Chained = 2,

	/// <summary>
	/// Sparse matches with controllable sparsity, removing edges randomly while maintaining
	/// view graph connectivity.
	/// </summary>
	Sparse = 3,
}

/// <summary>Port of colmap::SyntheticDatasetOptions.</summary>
public sealed class SyntheticDatasetOptions
{
	/// <summary>Number of rigs.</summary>
	public int NumRigs { get; set; } = 2;

	/// <summary>Number of cameras per rig.</summary>
	public int NumCamerasPerRig { get; set; } = 1;

	/// <summary>Number of frames per rig.</summary>
	public int NumFramesPerRig { get; set; } = 5;

	/// <summary>Number of 3D points.</summary>
	public int NumPoints3D { get; set; } = 100;

	/// <summary>
	/// Target track length per 3D point. If -1 (default), all images observe all points
	/// (dense visibility). If &gt; 0, observations are pruned to exactly this many per point.
	/// Must be -1 or &gt;= 2.
	/// </summary>
	public int TrackLength { get; set; } = -1;

	/// <summary>Standard deviation of the sensor-from-rig translation.</summary>
	public double SensorFromRigTranslationStddev { get; set; } = 0.05;

	/// <summary>Random rotation in degrees around the z-axis of the sensor.</summary>
	public double SensorFromRigRotationStddev { get; set; } = 5.0;

	/// <summary>Camera width in pixels.</summary>
	public int CameraWidth { get; set; } = 1024;

	/// <summary>Camera height in pixels.</summary>
	public int CameraHeight { get; set; } = 768;

	/// <summary>Camera model.</summary>
	public CameraModelId CameraModelId { get; set; } = CameraModelId.SimpleRadial;

	/// <summary>Camera parameters.</summary>
	public double[] CameraParams { get; set; } = [1280, 512, 384, 0.05];

	/// <summary>Whether the cameras have a prior focal length.</summary>
	public bool CameraHasPriorFocalLength { get; set; }

	/// <summary>The type of feature descriptors to synthesize.</summary>
	public FeatureExtractorType FeatureType { get; set; } = FeatureExtractorType.Sift;

	/// <summary>Number of random 2D points per image that observe no 3D point.</summary>
	public int NumPoints2DWithoutPoint3D { get; set; } = 10;

	/// <summary>Fraction of inlier matches among all written matches.</summary>
	public double InlierMatchRatio { get; set; } = 1.0;

	/// <summary>Whether to include decomposed relative poses in two-view geometries.</summary>
	public bool TwoViewGeometryHasRelativePose { get; set; }

	/// <summary>How image pairs are matched.</summary>
	public SyntheticMatchConfig MatchConfig { get; set; } = SyntheticMatchConfig.Exhaustive;

	/// <summary>
	/// Sparsity parameter for SPARSE match config, in range [0, 1]. 0 = fully connected view
	/// graph, equivalent to EXHAUSTIVE (all edges); 1 = empty view graph (no edges).
	/// </summary>
	public double MatchSparsity { get; set; }

	/// <summary>Whether to write position priors to the database.</summary>
	public bool PriorPosition { get; set; }

	/// <summary>Coordinate system of the position priors.</summary>
	public PosePriorCoordinateSystem PriorPositionCoordinateSystem { get; set; } = PosePriorCoordinateSystem.Cartesian;

	/// <summary>Whether to write gravity priors to the database.</summary>
	public bool PriorGravity { get; set; }

	/// <summary>Gravity direction in world coordinates.</summary>
	public Vector3d PriorGravityInWorld { get; set; } = new(0, 1, 0);

	/// <summary>The synthesized image file extension.</summary>
	public string ImageExtension { get; set; } = ".png";
}

/// <summary>Port of colmap::SyntheticNoiseOptions.</summary>
public sealed class SyntheticNoiseOptions
{
	/// <summary>Standard deviation of the rig-from-world translation noise.</summary>
	public double RigFromWorldTranslationStddev { get; set; }

	/// <summary>Random rotation in degrees around the z-axis of the rig.</summary>
	public double RigFromWorldRotationStddev { get; set; }

	/// <summary>Standard deviation of the 3D point noise.</summary>
	public double Point3DStddev { get; set; }

	/// <summary>Standard deviation of the 2D point noise.</summary>
	public double Point2DStddev { get; set; }

	/// <summary>Translational standard deviation of the prior position in meters.</summary>
	public double PriorPositionStddev { get; set; } = 1.5;

	/// <summary>Rotational standard deviation of the prior gravity in degrees.</summary>
	public double PriorGravityStddev { get; set; } = 1.0;
}

/// <summary>Port of colmap::SyntheticImageOptions.</summary>
public sealed class SyntheticImageOptions
{
	/// <summary>Radius of the bright peak drawn at each feature.</summary>
	public int FeaturePeakRadius { get; set; } = 2;

	/// <summary>Radius of the patterned patch drawn around each feature.</summary>
	public int FeaturePatchRadius { get; set; } = 15;

	/// <summary>Maximum brightness of the patch pattern.</summary>
	public int FeaturePatchMaxBrightness { get; set; } = 128;
}
