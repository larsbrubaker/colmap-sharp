// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of ColmapDemoApp.FirstLine (demo/ColmapDemo/ColmapDemoApp.Run.cs): the panel shows one
// line of an error (the whole exception goes to the console), cut short enough to fit.

namespace ColmapDemo.Tests;

public class PanelTextTests
{
	[Test]
	public async Task FirstLineKeepsOnlyTheFirstLine()
	{
		await Assert.That(ColmapDemoApp.FirstLine("The GPU failed\r\nValidation: a\nValidation: b")).IsEqualTo("The GPU failed");
	}

	[Test]
	public async Task FirstLineCutsALongLineWithAnEllipsis()
	{
		string line = ColmapDemoApp.FirstLine(new string('x', 500));

		await Assert.That(line.Length).IsEqualTo(160);
		await Assert.That(line).EndsWith("…");
	}
}
