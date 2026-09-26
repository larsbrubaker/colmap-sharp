// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CachedWorkspace: the CachedWorkspace class of colmap/mvs/workspace.h and workspace.cc - a
// Workspace (Workspace.cs) that loads each image's bitmap, depth map and normal map on
// first use and keeps them in a least-recently-used cache capped at options.CacheSize GB
// (Util/Cache.cs MemoryConstrainedLRUCache). Fusion uses it for large scenes. Tests:
// ColmapSharp.Tests/Mvs/WorkspaceTests.cs (both workspace kinds, as workspace_test.cc
// parameterizes them).
//
// Tier A (exact) bookkeeping. The locking mirrors COLMAP's: the cache lock guards cache
// access, and each cached image's own lock guards its lazy loads, so two threads never load
// the same map twice. An image evicted while another thread still holds it stays valid for
// that thread (the reference keeps it alive, like COLMAP's shared_ptr).

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::CachedWorkspace: loads MVS data on demand under a memory budget.</summary>
public sealed class CachedWorkspace : Workspace
{
	private readonly object cacheLock = new();
	private readonly MemoryConstrainedLRUCache<int, CachedImage> cache;

	/// <summary>
	/// A cached workspace whose model is read from options.WorkspacePath, with bitmaps from
	/// <paramref name="bitmapSource"/>.
	/// </summary>
	public CachedWorkspace(Options options, IBitmapSource bitmapSource)
		: base(options, bitmapSource)
	{
		cache = NewCache(options);
	}

	/// <summary>A cached workspace over a model already in memory.</summary>
	public CachedWorkspace(Options options, Model model, IBitmapSource bitmapSource)
		: base(options, model, bitmapSource)
	{
		cache = NewCache(options);
	}

	private static MemoryConstrainedLRUCache<int, CachedImage> NewCache(Options options) =>
		new((long)(1024.0 * 1024.0 * 1024.0 * options.CacheSize), static _ => new CachedImage());

	/// <summary>Does nothing: data is loaded as needed.</summary>
	public override void Load(IReadOnlyList<string> imageNames)
	{
	}

	/// <summary>Drops every cached image.</summary>
	public void ClearCache()
	{
		lock (cacheLock)
		{
			cache.Clear();
		}
	}

	/// <summary>The bitmap of an image, loaded (and rescaled to the model's size) on first use.</summary>
	public override Bitmap GetBitmap(int imageIdx)
	{
		CachedImage cachedImage = GetCachedImage(imageIdx);
		lock (cachedImage.Lock)
		{
			if (cachedImage.Bitmap is null)
			{
				Bitmap bitmap = bitmapSource.Read(GetBitmapPath(imageIdx), options.ImageAsRgb);
				if (options.MaxImageSize > 0)
				{
					bitmap.Rescale(model.Images[imageIdx].GetWidth(), model.Images[imageIdx].GetHeight());
				}

				cachedImage.Bitmap = bitmap;
				cachedImage.NumBytes += bitmap.NumBytes;
				UpdateNumBytes(imageIdx);
			}

			return cachedImage.Bitmap;
		}
	}

	/// <summary>The depth map of an image, loaded (and downsized to the model's size) on first use.</summary>
	public override DepthMap GetDepthMap(int imageIdx)
	{
		CachedImage cachedImage = GetCachedImage(imageIdx);
		lock (cachedImage.Lock)
		{
			if (cachedImage.DepthMap is null)
			{
				var depthMap = new DepthMap();
				depthMap.Read(GetDepthMapPath(imageIdx));
				if (options.MaxImageSize > 0)
				{
					depthMap.Downsize(model.Images[imageIdx].GetWidth(), model.Images[imageIdx].GetHeight());
				}

				cachedImage.DepthMap = depthMap;
				cachedImage.NumBytes += depthMap.GetNumBytes();
				UpdateNumBytes(imageIdx);
			}

			return cachedImage.DepthMap;
		}
	}

	/// <summary>The normal map of an image, loaded (and downsized to the model's size) on first use.</summary>
	public override NormalMap GetNormalMap(int imageIdx)
	{
		CachedImage cachedImage = GetCachedImage(imageIdx);
		lock (cachedImage.Lock)
		{
			if (cachedImage.NormalMap is null)
			{
				var normalMap = new NormalMap();
				normalMap.Read(GetNormalMapPath(imageIdx));
				if (options.MaxImageSize > 0)
				{
					normalMap.Downsize(model.Images[imageIdx].GetWidth(), model.Images[imageIdx].GetHeight());
				}

				cachedImage.NormalMap = normalMap;
				cachedImage.NumBytes += normalMap.GetNumBytes();
				UpdateNumBytes(imageIdx);
			}

			return cachedImage.NormalMap;
		}
	}

	private CachedImage GetCachedImage(int imageIdx)
	{
		lock (cacheLock)
		{
			return cache.Get(imageIdx);
		}
	}

	private void UpdateNumBytes(int imageIdx)
	{
		lock (cacheLock)
		{
			// Handle the case where another thread has already evicted the image.
			if (cache.Exists(imageIdx))
			{
				cache.UpdateNumBytes(imageIdx);
			}
		}
	}

	private sealed class CachedImage : ICacheSizedValue
	{
		public readonly object Lock = new();

		public long NumBytes { get; set; }

		public Bitmap? Bitmap { get; set; }

		public DepthMap? DepthMap { get; set; }

		public NormalMap? NormalMap { get; set; }
	}
}
