// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BitmapSink: how the undistorters (Undistorters*.cs) hand out the images they produce.
// COLMAP encodes and writes them with Bitmap::Write (OpenImageIO), which is native and not
// ported (docs/LICENSE_AUDIT.md), so the host encodes: it receives the output path and the
// bitmap, whose "Compression" metadata ("jpeg:<quality>", Bitmap.SetJpegQuality) carries the
// requested JPEG quality. InMemoryBitmapStore keeps the bitmaps instead and serves them back
// as the MVS code's IBitmapSource (Mvs/IBitmapSource.cs), so an undistorted workspace can go
// straight into Mvs.Workspace with no image files at all. C#-only; see
// divergence 98.

using System.Collections.Concurrent;

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

namespace ColmapSharp.Controllers;

/// <summary>
/// Receives the images a controller writes (COLMAP's Bitmap::Write), keyed by the output
/// path the controller would have written. Called concurrently from worker threads, so an
/// implementation must be thread safe.
/// </summary>
public interface IBitmapSink
{
	/// <summary>
	/// Stores <paramref name="bitmap"/> as <paramref name="path"/> and returns whether that
	/// worked. The sink may keep the instance; the controller does not touch it afterwards.
	/// A "Compression" metadata entry of "jpeg:&lt;quality&gt;" asks for that JPEG quality.
	/// </summary>
	bool Write(string path, Bitmap bitmap);
}

/// <summary>
/// An <see cref="IBitmapSink"/> that keeps the bitmaps in memory and serves them back by
/// path as an <see cref="IBitmapSource"/>, e.g. to Mvs.Workspace over the undistorted
/// images of <see cref="ColmapUndistorter"/>. Thread safe.
/// </summary>
public sealed class InMemoryBitmapStore : IBitmapSink, IBitmapSource
{
	private readonly ConcurrentDictionary<string, Bitmap> bitmaps = new(StringComparer.Ordinal);

	/// <summary>Every stored path, in any order.</summary>
	public IReadOnlyList<string> Paths => [.. bitmaps.Keys];

	/// <summary>The stored bitmap itself (not a copy); throws if there is none.</summary>
	public Bitmap Get(string path) => bitmaps[path];

	/// <inheritdoc/>
	public bool Write(string path, Bitmap bitmap)
	{
		bitmaps[path] = bitmap;
		return true;
	}

	/// <inheritdoc/>
	public bool Exists(string path) => bitmaps.ContainsKey(path);

	/// <inheritdoc/>
	public Bitmap Read(string path, bool asRgb)
	{
		Bitmap bitmap = bitmaps[path];
		return asRgb ? bitmap.CloneAsRGB() : bitmap.CloneAsGrey();
	}
}
