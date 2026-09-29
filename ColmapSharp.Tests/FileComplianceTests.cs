// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FileComplianceTests: the 800-line limit per file (CLAUDE.md, "Layout and style"), counting
// every line, blank lines included: it is a length trigger that prompts a refactor, not a
// measure of content. Ported from MatterCAD's Tests/MatterCADTests/Standard/FileComplianceTests.cs.
// C#-only; not a COLMAP test.
//
// Differences from MatterCAD's copy: every line counts (MatterCAD counts non-empty lines), the
// root is found by walking up to ColmapSharp.sln (not a fixed ../../.. from this file),
// scripts (.py, .sh), the oracle's C/C++ harnesses (.c, .cc, .cpp, .h) and Markdown docs (.md:
// plans, notices, the divergence log) are measured as well as .cs, and the excluded trees are
// this repo's:
// build output, the C++ reference checkout, the oracle's Python venv and the demo's agg-sharp
// submodule (the demo's own projects under demo/ are measured like the library). It also fails
// on git conflict markers left in any text file (a merge once let them slip into a doc), and
// checks the WGSL shader header convention (see WgslHeaderProblem). WGSL shaders
// (ColmapSharp/Mvs/Shaders/*.wgsl) are measured against the same 800-line limit.

using System.Runtime.CompilerServices;
using System.Text;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests;

/// <summary>
/// Tests that all source files in the repository conform to file size limits.
/// File size is measured as the count of all lines, blank lines included.
/// Limit: 800 lines, with no exemptions. Markdown docs are measured like source files.
/// </summary>
public class FileComplianceTests
{
	/// <summary>
	/// Default maximum lines for any measured file.
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
		// The demo's agg-sharp submodule: a separate repository with its own rules.
		"demo/agg-sharp",
	};

	/// <summary>
	/// File extensions to include in scanning.
	/// </summary>
	private static readonly HashSet<string> IncludedExtensions = new(StringComparer.OrdinalIgnoreCase)
	{
		".cs",
		".py",
		".sh",
		".c",
		".cc",
		".cpp",
		".h",
		".wgsl",
		// Docs: plans, notices and the divergence log are read by people and agents alike, so
		// they get the same limit; one that grows past it is split by topic with an index file.
		".md",
	};

	/// <summary>
	/// Extensions scanned for git conflict markers: every text file kind the repo tracks.
	/// </summary>
	private static readonly HashSet<string> ConflictMarkerExtensions = new(IncludedExtensions, StringComparer.OrdinalIgnoreCase)
	{
		".json",
		".txt",
		".csproj",
		".props",
		".targets",
		".sln",
		".yml",
		".yaml",
		".gitignore",
		".editorconfig",
	};

	// Built rather than written out, so this file does not itself contain the markers.
	private static readonly string OursMarker = new string('<', 7) + " ";
	private static readonly string SeparatorMarker = new string('=', 7);
	private static readonly string TheirsMarker = new string('>', 7) + " ";

	[Test]
	public async Task AllFilesShouldComplyWithSizeLimits()
	{
		var projectRoot = ResolveProjectRoot();
		var files = GetAllProjectFiles(projectRoot);
		var violations = new List<string>();

		foreach (var filePath in files)
		{
			var lineCount = CountLines(filePath);
			var relativePath = GetRelativePath(projectRoot, filePath);
			var limit = GetFileLimit(relativePath);

			if (lineCount > limit)
			{
				violations.Add($"  {relativePath}: {lineCount} lines (limit: {limit}) - {SplitAdvice(relativePath)}");
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
			message.AppendLine("To fix: split oversized code into smaller, cohesive files by responsibility, and oversized docs by topic with an index file. Never delete blank lines or comments to fit.");

			Assert.Fail(message.ToString());
		}

		await Assert.That(files.Count).IsGreaterThan(0);
	}

	[Test]
	public async Task NoFileContainsConflictMarkers()
	{
		var projectRoot = ResolveProjectRoot();
		var violations = new List<string>();
		foreach (var filePath in GetAllProjectFiles(projectRoot, ConflictMarkerExtensions))
		{
			foreach (var line in FindConflictMarkers(filePath))
			{
				violations.Add($"  {GetRelativePath(projectRoot, filePath)}:{line}");
			}
		}

		if (violations.Count > 0)
		{
			Assert.Fail("Git conflict markers found - resolve the merge in these files:" + Environment.NewLine + string.Join(Environment.NewLine, violations));
		}

		await Assert.That(violations.Count).IsEqualTo(0);
	}

	/// <summary>
	/// Every WGSL shader carries the header CLAUDE.md asks of every file, in the WGSL form
	/// ColmapSharp/Mvs/PatchMatchShaders.cs defines: a leading `//` block that starts with the
	/// copyright line and names what the file mirrors and ports.
	/// </summary>
	[Test]
	public async Task WgslFilesHaveTheHeader()
	{
		var projectRoot = ResolveProjectRoot();
		var shaders = GetAllProjectFiles(projectRoot).Where(f => Path.GetExtension(f).Equals(".wgsl", StringComparison.OrdinalIgnoreCase)).ToList();
		var violations = new List<string>();
		foreach (var filePath in shaders)
		{
			var problem = WgslHeaderProblem(File.ReadAllLines(filePath));
			if (problem != null)
			{
				violations.Add($"  {GetRelativePath(projectRoot, filePath)}: {problem}");
			}
		}

		if (violations.Count > 0)
		{
			Assert.Fail("WGSL files without the header:" + Environment.NewLine + string.Join(Environment.NewLine, violations));
		}

		await Assert.That(shaders.Count).IsGreaterThan(0);
	}

	[Test]
	public async Task WgslHeaderCheckFindsEachMissingPart()
	{
		string[] good = ["// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).", "//", "// k.wgsl: a kernel.", "// Mirrors: A.cs", "// Ports: nothing", "", "fn f() {}"];

		using (Assert.Multiple())
		{
			await Assert.That(WgslHeaderProblem(good)).IsNull();
			await Assert.That(WgslHeaderProblem(good.Skip(1).ToArray())).IsNotNull();
			await Assert.That(WgslHeaderProblem(good.Where(l => !l.StartsWith("// Mirrors:", StringComparison.Ordinal)).ToArray())).IsNotNull();
			await Assert.That(WgslHeaderProblem(good.Where(l => !l.StartsWith("// Ports:", StringComparison.Ordinal)).ToArray())).IsNotNull();

			// A Mirrors line after the leading comment block does not count.
			string[] late = [good[0], good[1], good[2], good[4], "", good[3]];
			await Assert.That(WgslHeaderProblem(late)).IsNotNull();
		}
	}

	/// <summary>
	/// Null when <paramref name="lines"/> start with the WGSL header: a block of `//` lines
	/// whose first is the copyright line naming Lars Brubaker and which holds a `// Mirrors:`
	/// and a `// Ports:` line; otherwise what is missing.
	/// </summary>
	private static string? WgslHeaderProblem(string[] lines)
	{
		var header = lines.TakeWhile(l => l.StartsWith("//", StringComparison.Ordinal)).ToList();
		if (header.Count == 0 || !header[0].StartsWith("// Copyright (c)", StringComparison.Ordinal) || !header[0].Contains("Lars Brubaker", StringComparison.Ordinal))
		{
			return "the first line must be the copyright line (// Copyright (c) <year>, Lars Brubaker. ...)";
		}

		if (!header.Any(l => l.StartsWith("// Mirrors:", StringComparison.Ordinal)))
		{
			return "the header needs a '// Mirrors:' line naming the C# file(s) the shader must agree with";
		}

		if (!header.Any(l => l.StartsWith("// Ports:", StringComparison.Ordinal)))
		{
			return "the header needs a '// Ports:' line naming the COLMAP source it replaces (or why none)";
		}

		return null;
	}

	[Test]
	public async Task ConflictMarkerScanFindsEachMarkerKind()
	{
		var root = CreateScratchRoot();
		try
		{
			var path = Path.Combine(root, "docs", "Merged.md");
			Directory.CreateDirectory(Path.GetDirectoryName(path)!);
			File.WriteAllLines(path, ["# Title", OursMarker + "HEAD", "ours", SeparatorMarker, "theirs", TheirsMarker + "branch", "======== not a marker", "text <<<<<<< inline"]);
			WriteSourceFile(Path.Combine(root, "cpp-reference", "Upstream.md"));
			File.AppendAllText(Path.Combine(root, "cpp-reference", "Upstream.md"), SeparatorMarker + Environment.NewLine);

			var files = GetAllProjectFiles(root, ConflictMarkerExtensions);

			using (Assert.Multiple())
			{
				await Assert.That(FindConflictMarkers(path)).IsEquivalentTo(new List<int> { 2, 4, 6 });
				await Assert.That(files.Select(f => Path.GetFileName(f)).ToList()).IsEquivalentTo(new List<string> { "Merged.md" });
			}
		}
		finally
		{
			Directory.Delete(root, true);
		}
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
			var lineCount = CountLines(filePath);
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
			WriteSourceFile(Path.Combine(root, "oracle", "Harness.cc"));
			WriteSourceFile(Path.Combine(root, "oracle", "Harness.h"));
			WriteSourceFile(Path.Combine(root, "oracle", "Harness.c"));
			WriteSourceFile(Path.Combine(root, "oracle", "Harness.cpp"));
			WriteSourceFile(Path.Combine(root, "ColmapSharp", "Mvs", "Shaders", "Kernel.wgsl"));
			WriteSourceFile(Path.Combine(root, "cpp-reference", "src", "Reference.cc"));
			WriteSourceFile(Path.Combine(root, "docs", "Plan.md"));
			WriteSourceFile(Path.Combine(root, "cpp-reference", "README.md"));
			WriteSourceFile(Path.Combine(root, "demo", "agg-sharp", "README.md"));
			WriteSourceFile(Path.Combine(root, "cpp-reference", "src", "Reference.cs"));
			WriteSourceFile(Path.Combine(root, "oracle", ".venv", "lib", "Vendored.py"));
			WriteSourceFile(Path.Combine(root, "obj", "Generated.cs"));
			WriteSourceFile(Path.Combine(root, "demo", "agg-sharp", "Foreign.cs"));
			WriteSourceFile(Path.Combine(root, "demo", "ColmapDemo", "Demo.cs"));

			var files = GetAllProjectFiles(root);

			await Assert.That(files.Select(f => Path.GetFileName(f)).OrderBy(f => f, StringComparer.Ordinal).ToList())
				.IsEquivalentTo(new List<string> { "Demo.cs", "Harness.c", "Harness.cc", "Harness.cpp", "Harness.h", "Kept.cs", "Kept.py", "Kernel.wgsl", "Plan.md" });
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

	[Test]
	public async Task LineCountIncludesBlankLines()
	{
		var root = CreateScratchRoot();
		try
		{
			// The limit is a length trigger, so blank and whitespace-only lines count too.
			var path = Path.Combine(root, "Doc.md");
			File.WriteAllLines(path, ["# Title", "", "   ", "text"]);

			await Assert.That(CountLines(path)).IsEqualTo(4);
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
	/// Count every line in a file, blank and whitespace-only lines included.
	/// </summary>
	/// <exception cref="IOException">
	/// The file could not be read. Deliberately NOT swallowed: answering zero for anything that
	/// could not be opened would be a silent pass for every oversized file that happened to be
	/// locked by an editor or a build while the scan ran - the one failure mode a size gate must
	/// not have. A file that vanished mid-walk is a different thing and is handled below.
	/// </exception>
	private static int CountLines(string filePath)
	{
		try
		{
			return File.ReadAllLines(filePath).Length;
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
	/// What to do about an oversized file: a doc is split by topic, code by responsibility.
	/// </summary>
	private static string SplitAdvice(string relativePath) =>
		Path.GetExtension(relativePath).Equals(".md", StringComparison.OrdinalIgnoreCase)
			? "split this doc by topic into smaller files, with an index file that says what lives where (see docs/CPP_DIVERGENCES.md). Use the file-size-refactoring skill."
			: "this must be refactored into multiple smaller files. Use the file-size-refactoring skill.";

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
	private static List<string> GetAllProjectFiles(string projectRoot) => GetAllProjectFiles(projectRoot, IncludedExtensions);

	private static List<string> GetAllProjectFiles(string projectRoot, HashSet<string> extensions)
	{
		var files = new List<string>();
		ScanDirectory(projectRoot, projectRoot, files, extensions);
		return files;
	}

	/// <summary>
	/// The 1-based line numbers holding a git conflict marker: a line starting with seven '&lt;'
	/// or '&gt;' and a space, or a line of exactly seven '='.
	/// </summary>
	private static List<int> FindConflictMarkers(string filePath)
	{
		var found = new List<int>();
		var lines = File.ReadAllLines(filePath);
		for (int i = 0; i < lines.Length; i++)
		{
			var line = lines[i];
			if (line.StartsWith(OursMarker, StringComparison.Ordinal) || line.StartsWith(TheirsMarker, StringComparison.Ordinal) || line == SeparatorMarker)
			{
				found.Add(i + 1);
			}
		}

		return found;
	}

	private static void ScanDirectory(string projectRoot, string directory, List<string> files) => ScanDirectory(projectRoot, directory, files, IncludedExtensions);

	private static void ScanDirectory(string projectRoot, string directory, List<string> files, HashSet<string> extensions)
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
				if (extensions.Contains(extension))
				{
					files.Add(file);
				}
			}

			// Recurse into subdirectories
			foreach (var subDir in Directory.GetDirectories(directory))
			{
				ScanDirectory(projectRoot, subDir, files, extensions);
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
