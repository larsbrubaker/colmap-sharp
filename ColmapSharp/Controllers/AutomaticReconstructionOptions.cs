// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionOptions: the Options struct and enums of
// colmap/controllers/automatic_reconstruction.h, plus ReconstructionOptionSet, the part of
// colmap/controllers/option_manager.{h,cc} that AutomaticReconstructionController
// (AutomaticReconstruction.cs) uses: the option objects it hands to each stage and the
// ModifyFor{Individual,Video,Internet}Data / ModifyFor{Low,Medium,High,Extreme}Quality presets.
// The rest of OptionManager (the command-line/ini registry) belongs to the CLI and is not
// ported (docs/CPP_DIVERGENCES.md entry 67).
//
// Translation notes:
// - image_path / mask_path become IImageSource hosts (Images / Masks), as in ImageReader.cs
//   (docs/CPP_DIVERGENCES.md entry 82); workspace_path stays a folder on disk because the
//   sparse models, depth maps, fused points and meshes are written there as COLMAP does.
// - vocab_tree_path, use_gpu, gpu_index and ba_backend are not options here: vocabulary-tree
//   matching, CUDA stages and the Caspar backend are out of scope (docs/CPP_DIVERGENCES.md
//   entry 134).
// - Texture / TextureSink and the MeshTextureMapping option set are C#-only: they add COLMAP's
//   separate mesh_texturer step to the dense stages (docs/CPP_DIVERGENCES.md entry 135).
// - ComputeDevice is C#-only: the host's GPU for PatchMatch stereo (docs/CPP_DIVERGENCES.md
//   entry 136), in place of use_gpu / gpu_index.
// - `int /= 1.5` in ModifyForMediumQuality converts to double and truncates back, as C++
//   does; the casts below reproduce that.

using ColmapSharp.Compute;
using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Mvs;

namespace ColmapSharp.Controllers;

/// <summary>Port of AutomaticReconstructionController::Options.</summary>
public sealed class AutomaticReconstructionOptions
{
	/// <summary>Port of AutomaticReconstructionController::DataType.</summary>
	public enum DataType
	{
		/// <summary>INDIVIDUAL.</summary>
		Individual = 0,

		/// <summary>VIDEO.</summary>
		Video = 1,

		/// <summary>INTERNET.</summary>
		Internet = 2,
	}

	/// <summary>Port of AutomaticReconstructionController::Quality.</summary>
	public enum QualityLevel
	{
		/// <summary>LOW.</summary>
		Low = 0,

		/// <summary>MEDIUM.</summary>
		Medium = 1,

		/// <summary>HIGH.</summary>
		High = 2,

		/// <summary>EXTREME.</summary>
		Extreme = 3,
	}

	/// <summary>Port of AutomaticReconstructionController::Feature.</summary>
	public enum FeatureType
	{
		/// <summary>SIFT.</summary>
		Sift = 0,

		/// <summary>ALIKED (ONNX; out of scope, the extractor rejects it).</summary>
		Aliked = 1,

		/// <summary>LOMA (ONNX; out of scope, the extractor rejects it).</summary>
		Loma = 2,

		/// <summary>LOMA128 (ONNX; out of scope, the extractor rejects it).</summary>
		Loma128 = 3,
	}

	/// <summary>Port of AutomaticReconstructionController::Mapper.</summary>
	public enum MapperType
	{
		/// <summary>INCREMENTAL.</summary>
		Incremental = 0,

		/// <summary>HIERARCHICAL.</summary>
		Hierarchical = 1,

		/// <summary>GLOBAL.</summary>
		Global = 2,
	}

	/// <summary>Port of AutomaticReconstructionController::Mesher.</summary>
	public enum MesherType
	{
		/// <summary>POISSON.</summary>
		Poisson = 0,

		/// <summary>DELAUNAY.</summary>
		Delaunay = 1,

		/// <summary>ADVANCING_FRONT (CGAL only; skipped like a COLMAP build without CGAL).</summary>
		AdvancingFront = 2,
	}

	/// <summary>The path to the workspace folder in which all results are stored.</summary>
	public string WorkspacePath { get; set; } = "";

	/// <summary>The images used as input (COLMAP's image_path).</summary>
	public IImageSource? Images { get; set; }

	/// <summary>
	/// Optional list of image names to reconstruct. The list must contain the names of the
	/// images in <see cref="Images"/>.
	/// </summary>
	public List<string> ImageNames { get; set; } = [];

	/// <summary>The masks used as input (COLMAP's mask_path), or null for none.</summary>
	public IImageSource? Masks { get; set; }

	/// <summary>The type of input data used to choose optimal mapper settings.</summary>
	public DataType Data { get; set; } = DataType.Individual;

	/// <summary>Whether to perform low- or high-quality reconstruction.</summary>
	public QualityLevel Quality { get; set; } = QualityLevel.High;

	/// <summary>Whether to use shared intrinsics or not.</summary>
	public bool SingleCamera { get; set; }

	/// <summary>Whether to use shared intrinsics or not for all images in the same sub-folder.</summary>
	public bool SingleCameraPerFolder { get; set; }

	/// <summary>Which camera model to use for images.</summary>
	public string CameraModel { get; set; } = "SIMPLE_RADIAL";

	/// <summary>Initial camera params for all images.</summary>
	public string CameraParams { get; set; } = "";

	/// <summary>Whether to perform feature extraction.</summary>
	public bool Extraction { get; set; } = true;

	/// <summary>Whether to perform feature matching.</summary>
	public bool Matching { get; set; } = true;

	/// <summary>Whether to perform sparse mapping.</summary>
	public bool Sparse { get; set; } = true;

	/// <summary>
	/// Whether to perform dense mapping. True, like a COLMAP build with CUDA and MVS: the
	/// PatchMatch algorithm runs on the CPU here (docs/CPP_DIVERGENCES.md entry 134).
	/// </summary>
	public bool Dense { get; set; } = true;

	/// <summary>The feature extraction/matching algorithm to be used.</summary>
	public FeatureType Feature { get; set; } = FeatureType.Sift;

	/// <summary>The mapping algorithm to be used.</summary>
	public MapperType Mapper { get; set; } = MapperType.Incremental;

	/// <summary>The meshing algorithm to be used.</summary>
	public MesherType Mesher { get; set; } = MesherType.Poisson;

	/// <summary>
	/// The Poisson mesher's trim: surface parts whose sample density is below this are cut away.
	/// COLMAP's automatic reconstruction always uses PoissonMeshingOptions' default (10), which
	/// trims a small photo set's sparse fused cloud to nothing; C#-only so a host can lower it
	/// (0 keeps the whole watertight surface).
	/// </summary>
	public double PoissonTrim { get; set; } = 10.0;

	/// <summary>
	/// Whether to texture each dense model's mesh with the photos' colors after meshing (only
	/// when <see cref="Dense"/> is on). C#-only: COLMAP's automatic reconstruction stops at the
	/// mesh and textures only through its separate mesh_texturer command
	/// (docs/CPP_DIVERGENCES.md entry 135).
	/// </summary>
	public bool Texture { get; set; } = true;

	/// <summary>
	/// Receives each texture atlas as dense/&lt;i&gt;/&lt;mesh name&gt;-textured/texture.png
	/// for the host to encode (COLMAP's texturer writes that PNG itself), or null to keep the
	/// atlases in memory only (AutomaticReconstructionController.TexturedMeshes). The mesh.ply
	/// written beside that path names texture.png either way, so with no sink the host must
	/// encode each TexturedModelMesh's atlas next to its MeshPath for the file to be usable on
	/// its own. An empty atlas (no face seen by any view) is not passed to the sink.
	/// </summary>
	public IBitmapSink? TextureSink { get; set; }

	/// <summary>
	/// The host's GPU compute device for PatchMatch stereo in the dense stages, or null to run
	/// it on the CPU. C#-only (COLMAP's dense stereo is CUDA; docs/CPP_DIVERGENCES.md entry
	/// 136). A problem the device cannot hold runs on the CPU, with the reason logged as a
	/// warning. Only when a device is set do the dense stage's PatchMatch progress messages gain
	/// " (GPU)" or " (CPU)" after the image name, saying where each problem ran. The controller
	/// runs its CPU stages synchronously on the calling thread, so a UI host should run it off
	/// its UI thread. <see cref="AutomaticReconstructionController.RunAsync"/> awaits the device
	/// and so uses any device; the synchronous
	/// <see cref="AutomaticReconstructionController.Run"/> uses it only when
	/// <see cref="IComputeDevice.SupportsBlockingWait"/> is true (blocking is safe on any
	/// thread), and otherwise warns and runs PatchMatch on the CPU. Hosts in the browser must
	/// call RunAsync.
	/// </summary>
	public IComputeDevice? ComputeDevice { get; set; }

	/// <summary>The number of threads to use in all stages.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>The random seed to use in all stages.</summary>
	public int RandomSeed { get; set; } = -1;
}

/// <summary>
/// The options of every stage the automatic reconstruction runs, with COLMAP's presets. Port
/// of the part of colmap::OptionManager that AutomaticReconstructionController uses.
/// </summary>
internal sealed class ReconstructionOptionSet
{
	public ImageReaderOptions ImageReader { get; private set; } = new();

	public FeatureExtractionOptions FeatureExtraction { get; private set; } = new();

	public FeatureMatchingOptions FeatureMatching { get; private set; } = new();

	public TwoViewGeometryOptions TwoViewGeometry { get; private set; } = new();

	public ExhaustivePairingOptions ExhaustivePairing { get; private set; } = new();

	public SequentialPairingOptions SequentialPairing { get; private set; } = new();

	public IncrementalPipelineOptions Mapper { get; private set; } = new();

	public PatchMatchOptions PatchMatchStereo { get; private set; } = new();

	public StereoFusionOptions StereoFusion { get; private set; } = new();

	public PoissonMeshingOptions PoissonMeshing { get; private set; } = new();

	public DelaunayMeshingOptions DelaunayMeshing { get; private set; } = new();

	public MeshTextureMappingOptions MeshTextureMapping { get; private set; } = new();

	/// <summary>Port of OptionManager::ResetOptions(reset_paths = false).</summary>
	public void ResetOptions()
	{
		ImageReader = new();
		FeatureExtraction = new();
		FeatureMatching = new();
		TwoViewGeometry = new();
		ExhaustivePairing = new();
		SequentialPairing = new();
		Mapper = new();
		PatchMatchStereo = new();
		StereoFusion = new();
		PoissonMeshing = new();
		DelaunayMeshing = new();
		MeshTextureMapping = new();
	}

	/// <summary>Port of OptionManager::ModifyForIndividualData.</summary>
	public void ModifyForIndividualData()
	{
		Mapper.MinFocalLengthRatio = 0.1;
		Mapper.MaxFocalLengthRatio = 10;
		Mapper.MaxExtraParam = double.MaxValue;
	}

	/// <summary>Port of OptionManager::ModifyForVideoData.</summary>
	public void ModifyForVideoData()
	{
		ResetOptions();
		Mapper.MapperOptions.InitMinTriAngle /= 2;
		Mapper.BaGlobalFramesRatio = 1.4;
		Mapper.BaGlobalPointsRatio = 1.4;
		Mapper.MinFocalLengthRatio = 0.1;
		Mapper.MaxFocalLengthRatio = 10;
		Mapper.MaxExtraParam = double.MaxValue;
		StereoFusion.MinNumPixels = 15;
	}

	/// <summary>Port of OptionManager::ModifyForInternetData.</summary>
	public void ModifyForInternetData()
	{
		StereoFusion.MinNumPixels = 10;
	}

	/// <summary>
	/// Port of OptionManager::ModifyForLowQuality. The vocab_tree_pairing lines have no
	/// counterpart (vocabulary-tree matching is out of scope).
	/// </summary>
	public void ModifyForLowQuality()
	{
		FeatureExtraction.MaxImageSize = (int)(0.3125 * FeatureExtraction.EffMaxImageSize());
		FeatureExtraction.Sift.MaxNumFeatures = 2048;
		SequentialPairing.LoopDetectionNumImages /= 2;
		Mapper.BaLocalMaxNumIterations = Mapper.EffBaLocalMaxNumIterations() / 2;
		Mapper.BaGlobalMaxNumIterations = Mapper.EffBaGlobalMaxNumIterations() / 2;
		Mapper.BaGlobalFramesRatio *= 1.2;
		Mapper.BaGlobalPointsRatio *= 1.2;
		Mapper.BaGlobalMaxRefinements = 2;
		PatchMatchStereo.MaxImageSize = 1000;
		PatchMatchStereo.WindowRadius = 4;
		PatchMatchStereo.WindowStep = 2;
		PatchMatchStereo.NumSamples /= 2;
		PatchMatchStereo.NumIterations = 3;
		PatchMatchStereo.GeomConsistency = false;
		StereoFusion.CheckNumImages /= 2;
		StereoFusion.MaxImageSize = 1000;
	}

	/// <summary>Port of OptionManager::ModifyForMediumQuality (vocab_tree_pairing lines dropped).</summary>
	public void ModifyForMediumQuality()
	{
		FeatureExtraction.MaxImageSize = (int)(0.5 * FeatureExtraction.EffMaxImageSize());
		FeatureExtraction.Sift.MaxNumFeatures = 4096;
		SequentialPairing.LoopDetectionNumImages = (int)(SequentialPairing.LoopDetectionNumImages / 1.5);
		Mapper.BaLocalMaxNumIterations = (int)(Mapper.EffBaLocalMaxNumIterations() / 1.5);
		Mapper.BaGlobalMaxNumIterations = (int)(Mapper.EffBaGlobalMaxNumIterations() / 1.5);
		Mapper.BaGlobalFramesRatio *= 1.1;
		Mapper.BaGlobalPointsRatio *= 1.1;
		Mapper.BaGlobalMaxRefinements = 2;
		PatchMatchStereo.MaxImageSize = 1600;
		PatchMatchStereo.WindowRadius = 4;
		PatchMatchStereo.WindowStep = 2;
		PatchMatchStereo.NumSamples = (int)(PatchMatchStereo.NumSamples / 1.5);
		PatchMatchStereo.NumIterations = 5;
		PatchMatchStereo.GeomConsistency = false;
		StereoFusion.CheckNumImages = (int)(StereoFusion.CheckNumImages / 1.5);
		StereoFusion.MaxImageSize = 1600;
	}

	/// <summary>Port of OptionManager::ModifyForHighQuality (vocab_tree_pairing line dropped).</summary>
	public void ModifyForHighQuality()
	{
		FeatureExtraction.Sift.EstimateAffineShape = true;
		FeatureExtraction.MaxImageSize = (int)(0.75 * FeatureExtraction.EffMaxImageSize());
		FeatureExtraction.Sift.MaxNumFeatures = 8192;
		FeatureMatching.GuidedMatching = true;
		Mapper.BaLocalMaxNumIterations = 30;
		Mapper.BaLocalMaxRefinements = 3;
		Mapper.BaGlobalMaxNumIterations = 75;
		PatchMatchStereo.MaxImageSize = 2400;
		StereoFusion.MaxImageSize = 2400;
	}

	/// <summary>Port of OptionManager::ModifyForExtremeQuality.</summary>
	public void ModifyForExtremeQuality()
	{
		// Most of the options are set to extreme quality by default.
		FeatureExtraction.Sift.EstimateAffineShape = true;
		FeatureExtraction.Sift.DomainSizePooling = true;
		FeatureMatching.GuidedMatching = true;
		Mapper.BaLocalMaxNumIterations = 40;
		Mapper.BaLocalMaxRefinements = 3;
		Mapper.BaGlobalMaxNumIterations = 100;
	}
}
