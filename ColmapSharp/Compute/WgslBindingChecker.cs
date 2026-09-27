// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// WgslBindingChecker: compares the buffer bindings a WGSL module declares with the bindings a
// ComputeKernelDescriptor (ComputeDescriptors.cs) states for it. Not a COLMAP port
// (PORTING_PLAN.md Phase 13). IComputeDevice builds an explicit pipeline layout from the
// descriptor and never parses the source, so a descriptor that disagrees with its WGSL only
// fails on a real GPU, as a validation error far from its cause. This check finds it without
// a GPU: the test suite runs it on every composed PatchMatch kernel.
//
// It reads only module-scope `@group(G) @binding(B) var<uniform|storage[, read|read_write]>
// name: T;` declarations (the attributes in either order), after removing comments. A
// uniform or storage variable whose group or binding is not a decimal literal is reported
// rather than guessed at.

using System.Text;
using System.Text.RegularExpressions;

namespace ColmapSharp.Compute;

/// <summary>One buffer binding a WGSL module declares.</summary>
/// <param name="Group">The <c>@group</c> index.</param>
/// <param name="Binding">The <c>@binding</c> index.</param>
/// <param name="Type">The address space and access mode as a binding type.</param>
/// <param name="Name">The variable's name.</param>
internal readonly record struct WgslBindingDeclaration(int Group, int Binding, ComputeBindingType Type, string Name);

/// <summary>Checks that WGSL declarations and a kernel descriptor's bindings agree.</summary>
internal static partial class WgslBindingChecker
{
	/// <summary>
	/// Every uniform and storage buffer declaration in <paramref name="source"/>, in source
	/// order, plus a message for each such variable whose group or binding could not be read.
	/// </summary>
	public static (IReadOnlyList<WgslBindingDeclaration> Declarations, IReadOnlyList<string> Unreadable) Parse(string source)
	{
		string code = StripComments(source);
		var declarations = new List<WgslBindingDeclaration>();
		var unreadable = new List<string>();
		foreach (Match match in BufferVariable().Matches(code))
		{
			string name = match.Groups["name"].Value;
			string attributes = match.Groups["attrs"].Value;
			Match group = GroupAttribute().Match(attributes);
			Match binding = BindingAttribute().Match(attributes);
			if (!group.Success || !binding.Success)
			{
				unreadable.Add($"WGSL declares the buffer variable {name} without a literal @group and @binding.");
				continue;
			}

			ComputeBindingType type = match.Groups["space"].Value == "uniform"
				? ComputeBindingType.Uniform
				: match.Groups["access"].Value == "read_write" ? ComputeBindingType.Storage : ComputeBindingType.ReadOnlyStorage;
			declarations.Add(new WgslBindingDeclaration(
				int.Parse(group.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture),
				int.Parse(binding.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture),
				type,
				name));
		}

		return (declarations, unreadable);
	}

	/// <summary>
	/// Every disagreement between <paramref name="descriptor"/>'s WGSL and its bindings: a
	/// binding one side has and the other lacks, a different binding type, a (group, binding)
	/// used twice, or an unreadable declaration. Empty when they agree.
	/// </summary>
	public static IReadOnlyList<string> Compare(in ComputeKernelDescriptor descriptor)
	{
		(IReadOnlyList<WgslBindingDeclaration> declared, IReadOnlyList<string> unreadable) = Parse(descriptor.Source);
		var problems = new List<string>(unreadable);
		var byIndex = new SortedDictionary<(int Group, int Binding), WgslBindingDeclaration>();
		foreach (WgslBindingDeclaration declaration in declared)
		{
			if (!byIndex.TryAdd((declaration.Group, declaration.Binding), declaration))
			{
				problems.Add($"WGSL declares @group({declaration.Group}) @binding({declaration.Binding}) twice ({byIndex[(declaration.Group, declaration.Binding)].Name} and {declaration.Name}).");
			}
		}

		var described = new SortedDictionary<(int Group, int Binding), ComputeKernelBinding>();
		foreach (ComputeKernelBinding binding in descriptor.Bindings)
		{
			if (!described.TryAdd((binding.Group, binding.Binding), binding))
			{
				problems.Add($"The descriptor lists @group({binding.Group}) @binding({binding.Binding}) twice.");
			}
		}

		foreach (((int group, int index), WgslBindingDeclaration declaration) in byIndex)
		{
			if (!described.TryGetValue((group, index), out ComputeKernelBinding binding))
			{
				problems.Add($"WGSL declares @group({group}) @binding({index}) {declaration.Name} as {declaration.Type}, but the descriptor has no such binding.");
			}
			else if (binding.Type != declaration.Type)
			{
				problems.Add($"@group({group}) @binding({index}) {declaration.Name} is {declaration.Type} in the WGSL but {binding.Type} in the descriptor.");
			}
		}

		foreach (((int group, int index), ComputeKernelBinding binding) in described)
		{
			if (!byIndex.ContainsKey((group, index)))
			{
				problems.Add($"The descriptor lists @group({group}) @binding({index}) as {binding.Type}, but the WGSL declares no such binding.");
			}
		}

		return problems;
	}

	/// <summary>
	/// Throws when <paramref name="descriptor"/>'s WGSL and bindings disagree, listing every
	/// problem <see cref="Compare"/> finds.
	/// </summary>
	public static void Check(in ComputeKernelDescriptor descriptor)
	{
		IReadOnlyList<string> problems = Compare(descriptor);
		if (problems.Count > 0)
		{
			throw new InvalidOperationException(
				$"Kernel \"{descriptor.Label}\": its WGSL and its descriptor's bindings disagree:\n  " + string.Join("\n  ", problems));
		}
	}

	/// <summary>The source with <c>//</c> line comments and (nestable) block comments replaced by spaces.</summary>
	internal static string StripComments(string source)
	{
		var text = new StringBuilder(source.Length);
		int depth = 0;
		for (int i = 0; i < source.Length; ++i)
		{
			char c = source[i];
			char next = i + 1 < source.Length ? source[i + 1] : '\0';
			if (depth > 0)
			{
				if (c == '/' && next == '*')
				{
					++depth;
					++i;
				}
				else if (c == '*' && next == '/')
				{
					--depth;
					++i;
				}

				text.Append(c == '\n' ? '\n' : ' ');
			}
			else if (c == '/' && next == '*')
			{
				depth = 1;
				++i;
				text.Append(' ');
			}
			else if (c == '/' && next == '/')
			{
				while (i < source.Length && source[i] != '\n')
				{
					++i;
				}

				text.Append('\n');
			}
			else
			{
				text.Append(c);
			}
		}

		return text.ToString();
	}

	// A uniform or storage variable with the attributes in front of it. Function-scope
	// variables cannot be in these address spaces, so every match is module scope.
	[GeneratedRegex(@"(?<attrs>(?:@\w+\s*\([^)]*\)\s*)*)\bvar\s*<\s*(?<space>uniform|storage)\s*(?:,\s*(?<access>read_write|read)\s*)?,?\s*>\s*(?<name>\w+)\s*:")]
	private static partial Regex BufferVariable();

	[GeneratedRegex(@"@group\s*\(\s*(\d+)\s*,?\s*\)")]
	private static partial Regex GroupAttribute();

	[GeneratedRegex(@"@binding\s*\(\s*(\d+)\s*,?\s*\)")]
	private static partial Regex BindingAttribute();
}
