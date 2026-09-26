// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HierarchicalPipeline: port of colmap/controllers/hierarchical_pipeline.h and .cc.
// Hierarchical mapping first partitions the scene graph into overlapping leaf clusters
// (Scene/SceneClustering.cs), reconstructs each cluster independently with the incremental
// pipeline (IncrementalPipeline.cs), and finally merges the cluster reconstructions bottom-up
// through the cluster tree (ObservationManager.MergeAndFilterReconstructions). This helps on
// larger scenes, since incremental mapping becomes slow with an increasing number of images.
// Tests: ColmapSharp.Tests/Controllers/HierarchicalPipelineTests.cs
// (hierarchical_pipeline_test.cc 1:1).
//
// Tier C (outcome): the merged reconstruction against the ground truth within COLMAP's
// bounds.
//
// Translation notes:
// - image_path becomes ReadImage, handed to every cluster's IncrementalPipeline
//   (docs/CPP_DIVERGENCES.md entry 68).
// - COLMAP's ThreadPool of workers becomes Parallel.ForEach with the same worker count; each
//   cluster writes only its own ReconstructionManager, so the merge sees the same inputs in
//   any schedule. The cluster-pointer NodeHashMap is a Dictionary keyed by reference.
// - PRNG (entry 110): each cluster starts from a fresh PRNG (the default seed on its first
//   draw), as the first cluster on a new COLMAP worker thread does. COLMAP's workers
//   continue their streams across the clusters they pick up, so its results depend on the
//   schedule; here they do not.
// - Cancellation: COLMAP's Run never checks CheckIfStopped. Here BaseController's
//   CancellationToken and stop function reach every cluster's IncrementalPipeline, and a
//   stopped run returns before merging (docs/CPP_DIVERGENCES.md entry 108).
// - Progress (C#-only): Progress, when set, receives a ControllerProgress per reconstructed
//   cluster and one when merging is done.
// - LOG(WARNING) goes to Util/Log.cs; LOG(INFO) and the timers are dropped.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::HierarchicalPipelineOptions.</summary>
public sealed class HierarchicalPipelineOptions
{
	/// <summary>
	/// Decodes an image by name to extract point colors (COLMAP's image_path). Null (the
	/// default) leaves all point colors black.
	/// </summary>
	public Func<string, Bitmap?>? ReadImage { get; set; }

	/// <summary>The maximum number of trials to initialize a cluster.</summary>
	public int InitNumTrials { get; set; } = 10;

	/// <summary>
	/// The total number of threads for the hierarchical pipeline. This budget is divided
	/// across workers to avoid thread oversubscription.
	/// </summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>The number of workers used to reconstruct clusters in parallel.</summary>
	public int NumWorkers { get; set; } = -1;

	/// <summary>Options for clustering the scene graph.</summary>
	public SceneClustering.Options ClusteringOptions { get; set; } = new();

	/// <summary>Options used to reconstruct each cluster individually.</summary>
	public IncrementalPipelineOptions IncrementalOptions { get; set; } = new();

	/// <summary>Port of Check: false on a bad option; throws unless the branching is 2.</summary>
	public bool Check()
	{
		if (InitNumTrials <= -1 || NumThreads < -1 || NumWorkers < -1)
		{
			return false;
		}

		// COLMAP calls both nested Checks but ignores their results; a bad nested option is
		// caught later, when SceneClustering or IncrementalPipeline checks its own options.
		_ = ClusteringOptions.Check();
		Util.Check.Eq(ClusteringOptions.Branching, 2);
		_ = IncrementalOptions.Check();
		return true;
	}
}

/// <summary>
/// Port of colmap::HierarchicalPipeline: partitions the scene into overlapping clusters,
/// reconstructs them separately with incremental mapping and merges them into a globally
/// consistent reconstruction.
/// </summary>
public sealed class HierarchicalPipeline : BaseController
{
	/// <summary>The stage reported after each cluster is reconstructed.</summary>
	public const string ReconstructionStage = "Reconstructing clusters";

	/// <summary>The stage reported after the clusters are merged.</summary>
	public const string MergingStage = "Merging clusters";

	private readonly HierarchicalPipelineOptions _options;
	private readonly DatabaseCache _databaseCache;
	private readonly ReconstructionManager _reconstructionManager;

	/// <summary>Loads the database into a cache.</summary>
	public HierarchicalPipeline(
		HierarchicalPipelineOptions options, Database database, ReconstructionManager reconstructionManager)
	{
		_options = Check.NotNull(options);
		_reconstructionManager = Check.NotNull(reconstructionManager);
		Check.That(_options.Check());
		Check.NotNull(database);

		var databaseCacheOptions = new DatabaseCache.Options
		{
			MinNumMatches = _options.IncrementalOptions.MinNumMatches,
			IgnoreWatermarks = _options.IncrementalOptions.IgnoreWatermarks,
		};
		_databaseCache = DatabaseCache.Create(database, databaseCacheOptions);

		if (_options.IncrementalOptions.BaRefineSensorFromRig)
		{
			Log.Warning(
				"The hierarchical reconstruction pipeline currently does not work robustly when refining the rig "
				+ "extrinsics, because overlapping frames in different child clusters are optimized independently "
				+ "and can thus diverge significantly. The merging of clusters oftentimes fails in these cases.");
		}
	}

	/// <summary>Receives a report per reconstructed cluster and after merging (C#-only).</summary>
	public IProgress<ControllerProgress>? Progress { get; set; }

	/// <summary>Partitions, reconstructs and merges.</summary>
	public override void Run()
	{
		// Cluster scene graph.
		var imageIdToName = new Dictionary<uint, string>(_databaseCache.NumImages);
		foreach ((uint imageId, Image image) in _databaseCache.Images)
		{
			imageIdToName.Add(imageId, image.Name);
		}

		SceneClustering sceneClustering = SceneClustering.Create(_options.ClusteringOptions, _databaseCache);
		List<SceneClustering.Cluster> leafClusters = sceneClustering.GetLeafClusters();

		// Reconstruct clusters. Determine the number of workers and threads per worker. The
		// total thread budget is divided across workers to avoid oversubscription.
		if (_options.IncrementalOptions.NumThreads > 0)
		{
			Log.Warning(
				"Mapper.num_threads is ignored in hierarchical mapping. Use num_threads to control the total "
				+ "thread budget instead.");
		}

		int numTotalThreads = Threading.GetEffectiveNumThreads(_options.NumThreads);
		const int kDefaultNumWorkers = 8;
		int numEffWorkers = Math.Max(
			1,
			Math.Min(
				leafClusters.Count,
				Math.Min(_options.NumWorkers < 1 ? kDefaultNumWorkers : _options.NumWorkers, numTotalThreads)));
		int numThreadsPerWorker = Math.Max(1, numTotalThreads / numEffWorkers);

		// Start reconstructing the bigger clusters first for better resource usage. The order
		// only affects scheduling; each cluster has its own manager.
		List<SceneClustering.Cluster> scheduled = [.. leafClusters.OrderByDescending(c => c.ImageIds.Count)];

		// A separate reconstruction manager per cluster avoids race conditions.
		var reconstructionManagers = new Dictionary<SceneClustering.Cluster, ReconstructionManager>(
			ReferenceEqualityComparer.Instance);
		foreach (SceneClustering.Cluster cluster in scheduled)
		{
			reconstructionManagers[cluster] = new ReconstructionManager();
		}

		int numReconstructed = 0;
		Parallel.ForEach(
			scheduled,
			new ParallelOptions { MaxDegreeOfParallelism = numEffWorkers },
			cluster =>
			{
				ReconstructCluster(cluster, reconstructionManagers[cluster], imageIdToName, numThreadsPerWorker);
				int done = Interlocked.Increment(ref numReconstructed);
				Progress?.Report(new ControllerProgress(ReconstructionStage, done, scheduled.Count, ""));
			});

		// A stopped run leaves partial cluster reconstructions that are not merged (entry 108).
		if (CheckIfStopped())
		{
			return;
		}

		// Merge clusters.
		if (leafClusters.Count > 1)
		{
			MergeClusters(Check.NotNull(sceneClustering.GetRootCluster()), reconstructionManagers);
			Progress?.Report(new ControllerProgress(MergingStage, 1, 1, ""));
		}

		Check.Eq(reconstructionManagers.Count, 1);
		ReconstructionManager merged = reconstructionManagers.Values.First();
		Check.Gt(merged.Get(0).NumRegImages, 0);
		_reconstructionManager.Clear();
		for (int i = 0; i < merged.Size; ++i)
		{
			_reconstructionManager.Set(_reconstructionManager.Add(), merged.Get(i));
		}

		for (int i = 0; i < _reconstructionManager.Size; ++i)
		{
			_reconstructionManager.Get(i).UpdatePoint3DErrors();
		}
	}

	// Reconstructs one cluster using incremental mapping.
	private void ReconstructCluster(
		SceneClustering.Cluster cluster,
		ReconstructionManager reconstructionManager,
		Dictionary<uint, string> imageIdToName,
		int numThreadsPerWorker)
	{
		if (cluster.ImageIds.Count == 0)
		{
			return;
		}

		IncrementalPipelineOptions incrementalOptions = _options.IncrementalOptions.Clone();
		incrementalOptions.ReadImage = _options.ReadImage;
		incrementalOptions.MaxModelOverlap = 3;
		incrementalOptions.InitNumTrials = _options.InitNumTrials;
		incrementalOptions.NumThreads = numThreadsPerWorker;

		var clusterImageNames = new HashSet<string>(cluster.ImageIds.Count);
		foreach (uint imageId in cluster.ImageIds)
		{
			clusterImageNames.Add(imageIdToName[imageId]);
		}

		// Create a filtered database cache for this cluster.
		var clusterCacheOptions = new DatabaseCache.Options
		{
			MinNumMatches = _options.IncrementalOptions.MinNumMatches,
			ImageNames = clusterImageNames,
		};
		DatabaseCache clusterDatabaseCache = DatabaseCache.CreateFromCache(_databaseCache, clusterCacheOptions);

		var mapper = new IncrementalPipeline(incrementalOptions, clusterDatabaseCache, reconstructionManager)
		{
			CancellationToken = CancellationToken,
		};
		mapper.SetCheckIfStoppedFunc(CheckIfStopped);

		// The cluster draws from a fresh PRNG, as on a new COLMAP worker thread, not from
		// whatever the calling or reused pool thread drew before; the thread's own PRNG is
		// restored afterwards (docs/CPP_DIVERGENCES.md entry 110).
		Mt19937? threadPrng = RandomUtils.Prng;
		RandomUtils.Prng = null;
		try
		{
			mapper.Run();
		}
		finally
		{
			RandomUtils.Prng = threadPrng;
		}
	}

	// Merges the reconstructions of the cluster's children (recursively) into as few
	// reconstructions as possible and stores them under the cluster, removing the children.
	private static void MergeClusters(
		SceneClustering.Cluster cluster, Dictionary<SceneClustering.Cluster, ReconstructionManager> reconstructionManagers)
	{
		// Extract all reconstructions from all child clusters.
		var reconstructions = new List<Reconstruction>();
		foreach (SceneClustering.Cluster childCluster in cluster.ChildClusters)
		{
			if (childCluster.ChildClusters.Count > 0)
			{
				MergeClusters(childCluster, reconstructionManagers);
			}

			ReconstructionManager reconstructionManager = reconstructionManagers[childCluster];
			for (int i = 0; i < reconstructionManager.Size; ++i)
			{
				reconstructions.Add(reconstructionManager.Get(i));
			}
		}

		// Try to merge all child cluster reconstruction.
		while (reconstructions.Count > 1)
		{
			bool mergeSuccess = false;
			for (int i = 0; i < reconstructions.Count; ++i)
			{
				for (int j = 0; j < i; ++j)
				{
					const double kMaxReprojError = 8.0;
					if (ObservationManager.MergeAndFilterReconstructions(kMaxReprojError, reconstructions[j], reconstructions[i]))
					{
						reconstructions.RemoveAt(j);
						mergeSuccess = true;
						break;
					}
				}

				if (mergeSuccess)
				{
					break;
				}
			}

			if (!mergeSuccess)
			{
				break;
			}
		}

		// Insert a new reconstruction manager for merged cluster.
		var mergedManager = new ReconstructionManager();
		foreach (Reconstruction reconstruction in reconstructions)
		{
			mergedManager.Set(mergedManager.Add(), reconstruction);
		}

		reconstructionManagers[cluster] = mergedManager;

		// Delete all merged child cluster reconstruction managers.
		foreach (SceneClustering.Cluster childCluster in cluster.ChildClusters)
		{
			reconstructionManagers.Remove(childCluster);
		}
	}
}
