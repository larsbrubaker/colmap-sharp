// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FileComplianceTests: the 800-non-empty-line limit per source file (CLAUDE.md, "Layout and
// style"), ported from MatterCAD's Tests/MatterCADTests/Standard/FileComplianceTests.cs so
// both repos enforce the same rule the same way. C#-only; not a COLMAP test.
//
// Differences from MatterCAD's copy: the root is found by walking up to ColmapSharp.sln
// (not a fixed ../../.. from this file), scripts (.py, .sh) are measured as well as .cs,
// and the excluded trees are this repo's: build output, the C++ reference checkout and the
// oracle's Python venv.

using System.Runtime.CompilerServices;
using System.Text;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests;

/// <summary>
/// Tests that all source files in the repository conform to file size limits.
/// File size is measured as count of non-empty lines (excluding blank lines and whitespace-only lines).
/// Limit: 800 non-empty lines, with no exemptions.
/// </summary>
public class FileComplianceTests
{
	/// <summary>
	/// Default maximum non-empty lines for any source file.
	/// </summary>
	private const int DefaultLineLimit = 800;

	/// <summary>
	/// Explicit file size limits for specific files (frozen at their size when added).
	/// Paths are relative to the repository root using forward slashes.
	/// This repo starts with none and CLAUDE.md allows no exemptions: a port that would
	/// exceed the limit is split by responsibility instead. The mechanism is kept so the
	/// rule matches MatterCAD's; if an entry is ever added, its limit may only decrease,
	/// never increase, and it is removed once the file is at or under 800 lines.
	/// </summary>
	private static readonly Dictionary<string, int> ExplicitFileLimits = new();

	/// <summary>
	/// Directory names to exclude from scanning, anywhere in the tree.
	/// </summary>
	private static readonly HashSet<string> ExcludedDirectories = new(StringComparer.OrdinalIgnoreCase)
	{
		"bin",
		"obj",
		".git",
		".vs",
		".vscode",
		".claude",
		".cursor",
		"TestResults",
		"__pycache__",
	};

	/// <summary>
	/// Directory paths, relative to the repository root and using forward slashes, to exclude from scanning.
	/// </summary>
	private static readonly HashSet<string> ExcludedRelativeDirectories = new(StringComparer.OrdinalIgnoreCase)
	{
		// The upstream C++ checkout (scripts/fetch-reference.sh): reading material, not our source.
		"cpp-reference",
		// The oracle's Python venv, which holds pycolmap and numpy.
		"oracle/.venv",
	};

	/// <summary>
	/// File extensions to include in scanning.
	/// </summary>
	private static readonly HashSet<string> IncludedExtensions = new(StringComparer.OrdinalIgnoreCase)
	{
		".cs",
		".py",
		".sh",
	};

	[Test]
	public async Task AllFilesShouldComplyWithSizeLimits()
	{
		var projectRoot = ResolveProjectRoot();
		var files = GetAllProjectFiles(projectRoot);
		var violations = new List<string>();

		foreach (var filePath in files)
		{
			var lineCount = CountNonEmptyLines(filePath);
			var relativePath = GetRelativePath(projectRoot, filePath);
			var limit = GetFileLimit(relativePath);

			if (lineCount > limit)
			{
				violations.Add($"  {relativePath}: {lineCount} non-empty lines (limit: {limit}) - this must be refactored into multiple smaller files. Use the file-size-refactoring skill.");
			}
		}

		if (violations.Count > 0)
		{
			var message = new StringBuilder();
			message.AppendLine($"File size violations found ({violations.Count} files exceed their limits):");
			message.AppendLine();
			foreach (var violation in violations.OrderByDescending(v => v))
			{
				message.AppendLine(violation);
			}

			message.AppendLine();
			message.AppendLine("To fix: Refactor oversized files into smaller, cohesive modules (split by responsibility).");

			Assert.Fail(message.ToString());
		}

		await Assert.That(files.Count).IsGreaterThan(0);
	}

	[Test]
	public async Task ComplianceSummaryReport()
	{
		var projectRoot = ResolveProjectRoot();
		var files = GetAllProjectFiles(projectRoot);

		// Count files by extension
		var fileCounts = files
			.GroupBy(f => Path.GetExtension(f).ToLowerInvariant())
			.OrderBy(g => g.Key)
			.ToDictionary(g => g.Key, g => g.Count());

		var violations = new List<(string Path, int Lines, int Limit)>();

		foreach (var filePath in files)
		{
			var lineCount = CountNonEmptyLines(filePath);
			var relativePath = GetRelativePath(projectRoot, filePath);
			var limit = GetFileLimit(relativePath);

			if (lineCount > limit)
			{
				violations.Add((relativePath, lineCount, limit));
			}
		}

		// Output summary
		Console.WriteLine();
		Console.WriteLine("=== File Compliance Summary ===");
		Console.WriteLine($"  Total files analyzed: {files.Count}");
		foreach (var (ext, count) in fileCounts)
		{
			Console.WriteLine($"    {ext}: {count} files");
		}

		if (violations.Count > 0)
		{
			Console.WriteLine();
			Console.WriteLine($"  VIOLATIONS: {violations.Count}");
			foreach (var (path, lines, limit) in violations.OrderByDescending(v => v.Lines))
			{
				var excess = lines - limit;
				Console.WriteLine($"    {path}: {lines} lines (limit: {limit}, {excess} over)");
			}
		}
		else
		{
			Console.WriteLine();
			Console.WriteLine("  All files comply with size limits!");
		}

		Console.WriteLine("===============================");

		// This test always passes - it's informational
		await Assert.That(files.Count).IsGreaterThan(0);
	}

	[Test]
	public async Task ScanExcludesReferenceVenvAndBuildDirectories()
	{
		var root = CreateScratchRoot();
		try
		{
			WriteSourceFile(Path.Combine(root, "Kept.cs"));
			WriteSourceFile(Path.Combine(root, "oracle", "Kept.py"));
			WriteSourceFile(Path.Combine(root, "cpp-reference", "src", "Reference.cs"));
			WriteSourceFile(Path.Combine(root, "oracle", ".venv", "lib", "Vendored.py"));
			WriteSourceFile(Path.Combine(root, "obj", "Generated.cs"));

			var files = GetAllProjectFiles(root);

			await Assert.That(files.Select(f => Path.GetFileName(f)).OrderBy(f => f, StringComparer.Ordinal).ToList())
				.IsEquivalentTo(new List<string> { "Kept.cs", "Kept.py" });
		}
		finally
		{
			Directory.Delete(root, true);
		}
	}

	[Test]
	public async Task ScanToleratesDirectoryDeletedMidWalk()
	{
		var root = CreateScratchRoot();
		try
		{
			// Reproduces the state left by a race: a scratch directory showed up in the
			// parent's listing and was deleted before the scan descended into it.
			var vanished = Path.Combine(root, "VanishedMidWalk");
			var files = new List<string>();

			ScanDirectory(root, vanished, files);

			await Assert.That(files.Count).IsEqualTo(0);
		}
		finally
		{
			Directory.Delete(root, true);
		}
	}

	private static string CreateScratchRoot()
	{
		var root = Path.Combine(Path.GetTempPath(), "ColmapSharpFileComplianceTests", Path.GetRandomFileName());
		Directory.CreateDirectory(root);
		return root;
	}

	private static void WriteSourceFile(string filePath)
	{
		Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
		File.WriteAllText(filePath, "// scan fixture" + Environment.NewLine);
	}

	/// <summary>
	/// Count non-empty lines in a file (excluding blank lines and whitespace-only lines).
	/// </summary>
	/// <exception cref="IOException">
	/// The file could not be read. Deliberately NOT swallowed: answering zero for anything that
	/// could not be opened would be a silent pass for every oversized file that happened to be
	/// locked by an editor or a build while the scan ran - the one failure mode a size gate must
	/// not have. A file that vanished mid-walk is a different thing and is handled below.
	/// </exception>
	private static int CountNonEmptyLines(string filePath)
	{
		try
		{
			var lines = File.ReadAllLines(filePath);
			return lines.Count(line => !string.IsNullOrWhiteSpace(line));
		}
		catch (Exception exception) when (exception is FileNotFoundException || exception is DirectoryNotFoundException)
		{
			// Gone between the walk that listed it and this read: nothing to measure. Every
			// OTHER failure - a lock, a denial - is a file that still exists and still has to
			// be checked, so it falls through to the throw below.
			return 0;
		}
		catch (Exception exception)
		{
			throw new IOException(
				$"{filePath} could not be read, so its size could not be checked. Close whatever holds it open and run again.",
				exception);
		}
	}

	/// <summary>
	/// Get the line limit for a specific file path.
	/// </summary>
	private static int GetFileLimit(string relativePath)
	{
		// Normalize path separators to forward slashes for comparison
		var normalizedPath = relativePath.Replace('\\', '/');

		foreach (var (explicitPath, explicitLimit) in ExplicitFileLimits)
		{
			var normalizedExplicit = explicitPath.Replace('\\', '/');
			if (normalizedPath.Equals(normalizedExplicit, StringComparison.OrdinalIgnoreCase)
				|| normalizedPath.EndsWith("/" + normalizedExplicit, StringComparison.OrdinalIgnoreCase))
			{
				return explicitLimit;
			}
		}

		return DefaultLineLimit;
	}

	/// <summary>
	/// Get all relevant project files for testing.
	/// </summary>
	private static List<string> GetAllProjectFiles(string projectRoot)
	{
		var files = new List<string>();
		ScanDirectory(projectRoot, projectRoot, files);
		return files;
	}

	private static void ScanDirectory(string projectRoot, string directory, List<string> files)
	{
		// Skip excluded directories
		if (IsExcludedDirectory(projectRoot, directory))
		{
			return;
		}

		// Add matching files
		try
		{
			foreach (var file in Directory.GetFiles(directory))
			{
				var extension = Path.GetExtension(file);
				if (IncludedExtensions.Contains(extension))
				{
					files.Add(file);
				}
			}

			// Recurse into subdirectories
			foreach (var subDir in Directory.GetDirectories(directory))
			{
				ScanDirectory(projectRoot, subDir, files);
			}
		}
		catch (UnauthorizedAccessException)
		{
			// Skip directories we can't access
		}
		catch (DirectoryNotFoundException)
		{
			// Deleted between the parent's listing and this descent; nothing left to measure.
		}
	}

	/// <summary>
	/// True when the directory is excluded either by name anywhere in the tree or by its path
	/// relative to the repository root.
	/// </summary>
	private static bool IsExcludedDirectory(string projectRoot, string directory)
	{
		if (ExcludedDirectories.Contains(Path.GetFileName(directory)))
		{
			return true;
		}

		var relativePath = Path.GetRelativePath(projectRoot, directory).Replace('\\', '/');
		return ExcludedRelativeDirectories.Contains(relativePath);
	}

	/// <summary>
	/// Get a relative path from the repository root to the given file.
	/// </summary>
	private static string GetRelativePath(string basePath, string fullPath)
	{
		return Path.GetRelativePath(basePath, fullPath);
	}

	/// <summary>
	/// Resolve the repository root: the nearest directory at or above this source file (or,
	/// failing that, the test binary) that contains ColmapSharp.sln. Walking up rather than
	/// counting levels keeps this correct if the file moves.
	/// </summary>
	private static string ResolveProjectRoot([CallerFilePath] string sourceFilePath = "")
	{
		foreach (var start in new[] { Path.GetDirectoryName(sourceFilePath), AppContext.BaseDirectory })
		{
			for (var dir = string.IsNullOrEmpty(start) ? null : new DirectoryInfo(start); dir != null; dir = dir.Parent)
			{
				if (File.Exists(Path.Combine(dir.FullName, "ColmapSharp.sln")))
				{
					return dir.FullName;
				}
			}
		}

		throw new DirectoryNotFoundException(
			$"No directory containing ColmapSharp.sln above {sourceFilePath} or {AppContext.BaseDirectory}.");
	}
}
