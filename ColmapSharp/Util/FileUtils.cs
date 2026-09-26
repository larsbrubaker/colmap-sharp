// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FileUtils: the path helpers of colmap/util/file.h/.cc that callers need so far,
// HasFileExtension and AddFileExtension (first user: the undistorters,
// Controllers/Undistorters.cs and PmvsUndistorter.cs). The rest of util/file comes with a
// caller. Named FileUtils so it does not shadow System.IO.File. Opening files with COLMAP's
// check is FileOpen.cs. Tests: ColmapSharp.Tests/Util/FileUtilsTests.cs (the matching
// file_test.cc cases 1:1).
//
// Tier A (exact): std::filesystem::path::extension() semantics are reproduced by hand: the
// extension starts at the file name's last '.', except that a name starting with its only
// '.' (a dot file), "." and ".." have none.

namespace ColmapSharp.Util;

/// <summary>Port of the path helpers of colmap/util/file.h.</summary>
public static class FileUtils
{
	/// <summary>
	/// Port of colmap::HasFileExtension: whether the file name's extension equals
	/// <paramref name="ext"/> lower-cased. Only <paramref name="ext"/> is lower-cased, so
	/// "a.JPG" does not have ".jpg", as in COLMAP.
	/// </summary>
	public static bool HasFileExtension(string fileName, string ext)
	{
		Check.That(ext.Length > 0);
		Check.Eq(ext[0], '.');
		string name = Path.GetFileName(fileName);
		int dot = name.LastIndexOf('.');
		string extension = dot > 0 && name != ".." ? name[dot..] : "";
		return extension == ext.ToLowerInvariant();
	}

	/// <summary>Port of colmap::AddFileExtension: appends <paramref name="ext"/> to the path as is.</summary>
	public static string AddFileExtension(string path, string ext) => path + ext;
}
