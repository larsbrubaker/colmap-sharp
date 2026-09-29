// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ImageSource: how ImageReader (ImageReader.cs) gets at images and masks without touching
// the file system. COLMAP's ImageReader walks `image_path` / `mask_path` and decodes files
// with OpenImageIO (Bitmap::Read), which is native and not ported (docs/LICENSE_AUDIT.md).
// Here the host (MatterCAD) owns decoding: it lists the image names and hands back decoded
// Bitmaps, with the EXIF metadata filled through Sensor/ExifReader.cs so the camera and GPS
// logic reads the same attributes OpenImageIO would have provided.
// InMemoryImageSource is the simplest host: already-decoded bitmaps keyed by name. C#-only;
// see divergence 82.

using ColmapSharp.Sensor;

namespace ColmapSharp.Controllers;

/// <summary>
/// A named collection of images (or masks) that <see cref="ImageReader"/> reads from, in
/// place of COLMAP's image_path / mask_path folders. Names are relative paths with '/'
/// separators, like COLMAP's normalized image names ("folder/0.jpg").
/// </summary>
public interface IImageSource
{
	/// <summary>
	/// Every image name, in any order (COLMAP's GetRecursiveFileList over image_path).
	/// ImageReader sorts them.
	/// </summary>
	IReadOnlyList<string> ListNames();

	/// <summary>Whether an entry with this name exists (ExistsFile).</summary>
	bool Exists(string name);

	/// <summary>
	/// Decodes the named image, or returns null when it is missing or cannot be decoded
	/// (Bitmap::Read returning false). The result may be grey or RGB and must carry a
	/// metadata store (any Bitmap made with the sized constructor does); ImageReader
	/// converts it to the requested channel count. The returned bitmap is owned by the
	/// caller, so return a fresh instance (or a clone) on every call.
	/// </summary>
	Bitmap? Read(string name);
}

/// <summary>
/// An <see cref="IImageSource"/> over already-decoded bitmaps. A name mapped to null
/// exists but fails to decode, like a corrupt file.
/// </summary>
public sealed class InMemoryImageSource : IImageSource
{
	private readonly Dictionary<string, Bitmap?> images = new(StringComparer.Ordinal);

	/// <summary>Adds (or replaces) an entry; null means "exists but cannot be decoded".</summary>
	public void Add(string name, Bitmap? bitmap) => images[name] = bitmap;

	/// <inheritdoc/>
	public IReadOnlyList<string> ListNames() => [.. images.Keys];

	/// <inheritdoc/>
	public bool Exists(string name) => images.ContainsKey(name);

	/// <inheritdoc/>
	public Bitmap? Read(string name) =>
		images.TryGetValue(name, out Bitmap? bitmap) ? bitmap?.Clone() : null;
}
