// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FileUtilsTests: the HasFileExtension and AddFileExtension cases of
// colmap/util/file_test.cc ported 1:1 (Suite_Name methods), for ColmapSharp/Util/FileUtils.cs.
// The other file_test.cc cases come with the rest of util/file. Tier A (exact).

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class FileUtilsTests
{
	[Test]
	public async Task HasFileExtension_Nominal()
	{
		await Assert.That(FileUtils.HasFileExtension("", ".jpg")).IsFalse();
		await Assert.That(FileUtils.HasFileExtension("testjpg", ".jpg")).IsFalse();
		await Assert.That(FileUtils.HasFileExtension("test.jpg", ".jpg")).IsTrue();
		await Assert.That(FileUtils.HasFileExtension("test.jpg", ".Jpg")).IsTrue();
		await Assert.That(FileUtils.HasFileExtension("test.jpg", ".JPG")).IsTrue();
		await Assert.That(FileUtils.HasFileExtension("test.", ".")).IsTrue();
	}

	[Test]
	public async Task AddFileExtension_Nominal()
	{
		await Assert.That(FileUtils.AddFileExtension("test", ".txt")).IsEqualTo("test.txt");
		await Assert.That(FileUtils.AddFileExtension("test.jpg", ".txt")).IsEqualTo("test.jpg.txt");
	}
}
