// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchShaders: the WGSL sources of GPU PatchMatch and how a kernel's module text is put
// together. Not a COLMAP port (PORTING_PLAN.md Phase 13); the kernels replace
// patch_match_cuda.cu. The .wgsl files live in Mvs/Shaders/ and are embedded in the assembly
// under fixed resource names (ColmapSharp.csproj), so loading them needs no file system and
// no reflection over types: it stays trim- and AOT-clean for browser-wasm.
//
// A kernel's source is a generated constants header (Compute/WgslConstants.cs) followed by
// the named parts in the order given. WGSL resolves module-scope names in any order, so a
// part may use a constant, a binding or a function another part declares. The parts:
// - patch_match_common.wgsl: self-contained helpers (RNG, exact u32 -> f32, CUDA min/max).
// - patch_match_textures.wgsl: source image and depth sampling; needs common, the
//   PM_SRC_MAX_WIDTH / PM_SRC_MAX_HEIGHT constants and the bindings its header lists.
//
// WGSL file header convention (FileComplianceTests checks it): the file starts with a `//`
// comment block whose first line is the copyright line, which says what the file is, and which
// holds a `// Mirrors:` line naming the C# file(s) it must agree with and a `// Ports:` line
// naming the COLMAP source it replaces (or why there is none).

using System.Text;

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

/// <summary>Loads and composes the embedded WGSL of GPU PatchMatch.</summary>
internal static class PatchMatchShaders
{
	/// <summary>The self-contained helpers every kernel shares.</summary>
	public const string Common = "patch_match_common.wgsl";

	/// <summary>Source image and depth map sampling.</summary>
	public const string Textures = "patch_match_textures.wgsl";

	// ColmapSharp.csproj gives every Mvs/Shaders/*.wgsl this prefix as its LogicalName.
	private const string ResourcePrefix = "ColmapSharp.Mvs.Shaders.";

	/// <summary>
	/// The text of the embedded shader file <paramref name="part"/> (for example
	/// <see cref="Common"/>), with line endings normalized to <c>\n</c> so the text is the same
	/// whichever way git checked the file out.
	/// </summary>
	public static string Load(string part)
	{
		using Stream? stream = typeof(PatchMatchShaders).Assembly.GetManifestResourceStream(ResourcePrefix + part);
		if (stream == null)
		{
			throw new ArgumentException($"There is no embedded shader named \"{part}\"; shader files go in ColmapSharp/Mvs/Shaders/.", nameof(part));
		}

		using var reader = new StreamReader(stream, Encoding.UTF8);
		return reader.ReadToEnd().Replace("\r\n", "\n");
	}

	/// <summary>
	/// A kernel's WGSL module: a comment naming the parts, <paramref name="constants"/>'
	/// header, then each part's text in the given order under a marker comment. The same
	/// arguments always give byte-identical text.
	/// </summary>
	public static string Compose(WgslConstants constants, params string[] parts)
	{
		ArgumentNullException.ThrowIfNull(constants);
		if (parts.Length == 0)
		{
			throw new ArgumentException("A kernel needs at least one shader part.", nameof(parts));
		}

		if (parts.Distinct(StringComparer.Ordinal).Count() != parts.Length)
		{
			throw new ArgumentException($"A shader part is listed twice: {string.Join(", ", parts)}.", nameof(parts));
		}

		var text = new StringBuilder();
		text.Append("// Composed by ColmapSharp.Mvs.PatchMatchShaders.Compose from: ")
			.Append(string.Join(", ", parts)).Append('\n');
		text.Append("// ---- constants ----\n");
		text.Append(constants.ToWgsl());
		foreach (string part in parts)
		{
			string body = Load(part);
			text.Append("\n// ---- ").Append(part).Append(" ----\n");
			text.Append(body);
			if (!body.EndsWith('\n'))
			{
				text.Append('\n');
			}
		}

		return text.ToString();
	}
}
