// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSolutionParameters: Reconstructor::SolutionParameters<float> and
// Reconstructor::Poisson::SolutionParameters<float> from thirdparty/PoissonRecon/
// Reconstructors.h - the solver settings with PoissonRecon's defaults, and testAndSet, which
// derives the unset depths (-1 means "unset") from the system depth. Also the progress
// report type the Poisson stages publish. PoissonSampleSet calls TestAndSet right after
// computing the unit-cube transform, as Solve does.
//
// Translation notes: the depth fields are unsigned in the C++ and use (unsigned int)-1 as
// "unset", which compares greater than any real depth; they are uint here for the same
// comparisons. testAndSet's MK_WARN messages (a requested depth clamped) are not printed;
// the clamping itself is identical. The width-based depth (width > 0) is ported although
// COLMAP never sets a width.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>The stages of Poisson reconstruction, for progress reports.</summary>
public enum PoissonStage
{
	/// <summary>Reading the samples into the octree.</summary>
	TreeBuild,

	/// <summary>Kernel density estimation.</summary>
	Density,

	/// <summary>Splatting the normals (and colors) into the tree.</summary>
	Splat,

	/// <summary>Solving the screened Poisson system, depth by depth.</summary>
	Solve,

	/// <summary>Averaging the solved function at the samples for the surface's iso-value.</summary>
	IsoValue,

	/// <summary>Extracting the iso-surface.</summary>
	LevelSet,

	/// <summary>Trimming low-density parts of the surface.</summary>
	Trim,
}

/// <summary>A progress report: the stage, the fraction of it done, and the depth when the stage works per depth (else -1).</summary>
public readonly record struct PoissonProgress(PoissonStage Stage, double Fraction, int Depth = -1);

/// <summary>
/// Poisson solver settings. Port of <c>Reconstructor::Poisson::SolutionParameters&lt;float&gt;</c>
/// (with its base <c>Reconstructor::SolutionParameters</c>). The defaults are the values COLMAP's
/// <c>RunPoissonRecon</c> call produces with default <c>PoissonMeshingOptions</c>: depth 13 and
/// point weight 1 (COLMAP always passes both), full depth 5 (COLMAP passes --fullDepth only when
/// depth &lt; 5; callers must lower it then), alignment direction Dim-1 = 2 and Dirichlet erosion
/// on (PoissonRecon.cpp: AlignmentDir defaults to DEFAULT_DIMENSION-1 and dirichletErode is
/// !NoDirichletErode.set), and PoissonRecon.cpp's command-line defaults for everything else.
/// </summary>
public sealed class PoissonSolutionParameters
{
	/// <summary>The "unset" depth value, (unsigned int)-1.</summary>
	public const uint Unset = uint.MaxValue;

	/// <summary>Use the normal lengths as sample confidence weights.</summary>
	public bool Confidence { get; set; }

	/// <summary>Interpolate the points exactly rather than approximately.</summary>
	public bool ExactInterpolation { get; set; }

	/// <summary>The bounding box scale factor.</summary>
	public float Scale { get; set; } = 1.1f;

	/// <summary>Samples below this depth are ignored when splatting (0 = none).</summary>
	public float LowDepthCutOff { get; set; }

	/// <summary>If positive, the finest voxel width, which then sets Depth.</summary>
	public float Width { get; set; }

	/// <summary>The minimum number of samples per node.</summary>
	public float SamplesPerNode { get; set; } = 1.5f;

	/// <summary>The conjugate-gradient accuracy.</summary>
	public float CgSolverAccuracy { get; set; } = 1e-3f;

	/// <summary>The per-level scale of the auxiliary (color) data.</summary>
	public float PerLevelDataScaleFactor { get; set; } = 32.0f;

	/// <summary>The weight of the point interpolation term (PoissonRecon.cpp's --pointWeight).</summary>
	public float PointWeight { get; set; } = 1.0f;

	/// <summary>The weight of value interpolation (unused by COLMAP).</summary>
	public float ValueInterpolationWeight { get; set; }

	/// <summary>Erode Dirichlet constraints around the envelope (envelope only; unused by COLMAP).</summary>
	public bool DirichletErode { get; set; } = true;

	/// <summary>The maximum reconstruction depth.</summary>
	public uint Depth { get; set; } = 13;

	/// <summary>The maximum solution depth.</summary>
	public uint SolveDepth { get; set; } = Unset;

	/// <summary>The coarse multigrid solver depth.</summary>
	public uint BaseDepth { get; set; } = Unset;

	/// <summary>The depth up to which the tree is complete.</summary>
	public uint FullDepth { get; set; } = 5;

	/// <summary>The kernel density estimation depth.</summary>
	public uint KernelDepth { get; set; } = Unset;

	/// <summary>The envelope depth (envelope only).</summary>
	public uint EnvelopeDepth { get; set; } = Unset;

	/// <summary>V-cycles at the base depth.</summary>
	public uint BaseVCycles { get; set; } = 1;

	/// <summary>Gauss-Seidel iterations per depth.</summary>
	public uint Iters { get; set; } = 8;

	/// <summary>The axis the bounding box is aligned to (PoissonRecon.cpp: Dim-1).</summary>
	public uint AlignDir { get; set; } = 2;

	/// <summary>
	/// Derives the unset depths and clamps the set ones. Port of the Poisson
	/// <c>testAndSet&lt;Dim&gt;( unitCubeToModel )</c>, which first runs the base class's.
	/// </summary>
	public void TestAndSet(PoissonXForm unitCubeToModel)
	{
		const int Dim = 3;
		if (Width > 0)
		{
			float maxScale = 0;
			for (int i = 0; i < Dim; i++)
			{
				float l2 = 0;
				for (int j = 0; j < Dim; j++)
				{
					l2 += unitCubeToModel[i, j] * unitCubeToModel[i, j];
				}

				if (l2 > maxScale)
				{
					maxScale = l2;
				}
			}

			// Math.Log is the platform's libm, as log() is in the C++; a last-ulp difference could
			// only matter when log(maxScale/width)/log(2) lands on an integer. COLMAP never sets a
			// width, so this path is unused there.
			maxScale = MathF.Sqrt(maxScale);
			Depth = (uint)Math.Ceiling(StdMinMax.StdMax(0.0, Math.Log(maxScale / Width) / Math.Log(2.0)));
		}

		if (SolveDepth > Depth)
		{
			SolveDepth = Depth;
		}

		if (FullDepth > SolveDepth)
		{
			FullDepth = SolveDepth;
		}

		if (BaseDepth > FullDepth)
		{
			BaseDepth = FullDepth;
		}

		if (KernelDepth == Unset)
		{
			KernelDepth = Depth > 2 ? Depth - 2 : 0;
		}

		if (KernelDepth > Depth)
		{
			KernelDepth = Depth;
		}

		// Poisson::SolutionParameters::testAndSet
		if (EnvelopeDepth == Unset)
		{
			EnvelopeDepth = BaseDepth;
		}

		if (EnvelopeDepth > Depth)
		{
			EnvelopeDepth = Depth;
		}

		if (EnvelopeDepth < BaseDepth)
		{
			EnvelopeDepth = BaseDepth;
		}
	}
}
