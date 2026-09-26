// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CheckTests: C#-only tests (COLMAP has no logging_test.cc case for THROW_CHECK messages)
// pinning the message shape of ColmapSharp/Util/Check.cs to COLMAP's LogMessageFatalThrow:
// "[file:line] Check failed: <expr> " and "... <a> <op> <b> (<va> vs. <vb>) ".

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class CheckTests
{
	[Test]
	public async Task That_FailureMessageMatchesColmap()
	{
		var elems = Array.Empty<int>();
		var exception = Assert.Throws<ArgumentException>(() => Check.That(elems.Length > 0, "extra"));
		await Assert.That(exception.Message).Matches(@"^\[CheckTests\.cs:\d+\] Check failed: elems\.Length > 0 extra$");
	}

	[Test]
	public async Task Ge_FailureMessageMatchesColmap()
	{
		double p = -1.5;
		var exception = Assert.Throws<ArgumentException>(() => Check.Ge(p, 0));
		await Assert.That(exception.Message).Matches(@"^\[CheckTests\.cs:\d+\] Check failed: p >= 0 \(-1\.5 vs\. 0\) $");
	}

	[Test]
	public async Task Ops_PassAndFailLikeTheirOperators()
	{
		using (Assert.Multiple())
		{
			Check.Eq(1, 1);
			Check.Ne(1, 2);
			Check.Lt(1, 2);
			Check.Le(2, 2);
			Check.Gt(2, 1);
			Check.Ge(2, 2);
			await Assert.That(() => Check.Eq(1, 2)).Throws<ArgumentException>();
			await Assert.That(() => Check.Ne(1, 1)).Throws<ArgumentException>();
			await Assert.That(() => Check.Lt(2, 2)).Throws<ArgumentException>();
			await Assert.That(() => Check.Le(3, 2)).Throws<ArgumentException>();
			await Assert.That(() => Check.Gt(2, 2)).Throws<ArgumentException>();
			await Assert.That(() => Check.Ge(1, 2)).Throws<ArgumentException>();
			// NaN fails every ordered comparison, as the C++ operators do.
			await Assert.That(() => Check.Ge(double.NaN, 0)).Throws<ArgumentException>();
		}
	}

	[Test]
	public async Task NotNull_ReturnsValueOrThrows()
	{
		string? present = "x";
		string? missing = null;
		await Assert.That(Check.NotNull(present)).IsEqualTo("x");
		var exception = Assert.Throws<ArgumentException>(() => Check.NotNull(missing));
		await Assert.That(exception.Message).Matches(@"^\[CheckTests\.cs:\d+\] 'missing' Must be non NULL$");
	}
}
