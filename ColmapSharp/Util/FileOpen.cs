// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FileOpen: std::ifstream / std::ofstream followed by COLMAP's THROW_CHECK_FILE_OPEN
// (colmap/util/logging.h), shared by every reader and writer that takes a path
// (Scene/ReconstructionIO*.cs, Util/Ply*.cs). A file that cannot be opened throws COLMAP's
// "Could not open" check failure.

namespace ColmapSharp.Util;

/// <summary>Opens files the way COLMAP's path-based readers and writers do.</summary>
internal static class FileOpen
{
	/// <summary>
	/// std::ifstream + THROW_CHECK_FILE_OPEN: opens <paramref name="path"/> for reading, or
	/// throws COLMAP's "Could not open" check failure.
	/// </summary>
	public static FileStream OpenRead(string path)
	{
		try
		{
			return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
		}
		catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
		{
			throw FileOpenFailure(path, ex);
		}
	}

	/// <summary>
	/// std::ofstream(path, std::ios::trunc) + THROW_CHECK_FILE_OPEN: creates or truncates
	/// <paramref name="path"/> for writing, or throws COLMAP's "Could not open" check failure.
	/// </summary>
	public static FileStream OpenWrite(string path)
	{
		try
		{
			return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
		}
		catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
		{
			throw FileOpenFailure(path, ex);
		}
	}

	private static ArgumentException FileOpenFailure(string path, Exception inner) =>
		new($"Check failed: (file).is_open() Could not open \"{path}\". Is the path a directory or does the parent dir not exist?", inner);
}
