// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IBitmapSource: how the MVS code gets image pixels. COLMAP calls Bitmap::Read (OpenImageIO)
// on image file paths; this library does no image decoding (docs/LICENSE_AUDIT.md), so the
// host supplies decoded bitmaps by path. Workspace.cs (bitmap loading and HasBitmap) and
// Model.Pmvs.cs (image sizes) use it. A path is only a key here: a host can serve bitmaps
// from memory without any file existing. Written for colmap-sharp; not a COLMAP port.

using ColmapSharp.Sensor;

namespace ColmapSharp.Mvs;

/// <summary>
/// Supplies decoded images by path (the host decodes; the library never opens image files).
/// <para>
/// Contract: <see cref="Read"/> returns a new <see cref="Bitmap"/> instance on every call,
/// owned by the caller, which may mutate it (the workspaces rescale bitmaps in place, as
/// COLMAP's freshly read bitmaps are). Never hand out a shared or cached instance.
/// <see cref="Read"/> and <see cref="Exists"/> must be safe to call concurrently from
/// multiple threads: Workspace.Load reads images in parallel, and CachedWorkspace loads on
/// demand from whichever thread asks.
/// </para>
/// </summary>
public interface IBitmapSource
{
	/// <summary>
	/// Whether an image exists for <paramref name="path"/> (COLMAP's ExistsFile on the image
	/// path). Must be thread-safe.
	/// </summary>
	bool Exists(string path);

	/// <summary>
	/// A new decoded image at <paramref name="path"/>, as 3-channel RGB if
	/// <paramref name="asRgb"/> and 1-channel grey otherwise (Bitmap::Read(path, as_rgb)),
	/// owned by the caller, which may mutate it. Throws if the image cannot be read. Must be
	/// thread-safe.
	/// </summary>
	Bitmap Read(string path, bool asRgb);
}
