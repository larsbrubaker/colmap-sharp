// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchTests: C#-only tests (COLMAP has no patch_match_test.cc; its only PatchMatch
// test is pycolmap's CUDA-gated test_patch_match_options_init) for
// ColmapSharp/Mvs/PatchMatchOptions.cs and the problem validation in
// ColmapSharp/Mvs/PatchMatch.cs. They pin COLMAP's defaults (float literals stored in
// doubles), every PatchMatchOptions::Check bound, and each PatchMatch::Check condition.
// Tier A.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchTests
{
	private const int Width = 4;
	private const int Height = 3;
	private static readonly float[] K = [100, 0, 2, 0, 100, 1.5f, 0, 0, 1];
	private static readonly float[] R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
	private static readonly float[] T = [0, 0, 0];

	private static Image NewImage(float[]? k = null, bool asRgb = false)
	{
		var image = new Image("img", Width, Height, k ?? K, R, T);
		image.SetBitmap(new Bitmap(Width, Height, asRgb));
		return image;
	}

	private static PatchMatch.Problem NewProblem(bool withMaps, int numImages = 3)
	{
		var problem = new PatchMatch.Problem { RefImageIdx = 0, SrcImageIdxs = [1, 2] };
		problem.Images = new List<Image>();
		for (int i = 0; i < numImages; i++)
		{
			problem.Images.Add(NewImage());
		}

		if (withMaps)
		{
			problem.DepthMaps = new List<DepthMap>();
			problem.NormalMaps = new List<NormalMap>();
			for (int i = 0; i < numImages; i++)
			{
				problem.DepthMaps.Add(new DepthMap(Width, Height, 1, 2));
				problem.NormalMaps.Add(new NormalMap(Width, Height));
			}
		}

		return problem;
	}

	private static PatchMatchOptions PhotometricOptions() => new() { GeomConsistency = false };

	[Test]
	public async Task PatchMatchOptions_Defaults()
	{
		var options = new PatchMatchOptions();
		await Assert.That(options.DepthMin).IsEqualTo(-1.0);
		await Assert.That(options.DepthMax).IsEqualTo(-1.0);
		await Assert.That(options.SigmaSpatial).IsEqualTo(-1.0);

		// COLMAP initializes these doubles from float literals.
		await Assert.That(options.SigmaColor).IsEqualTo((double)0.2f);
		await Assert.That(options.NccSigma).IsEqualTo((double)0.6f);
		await Assert.That(options.IncidentAngleSigma).IsEqualTo((double)0.9f);
		await Assert.That(options.GeomConsistencyRegularizer).IsEqualTo((double)0.3f);
		await Assert.That(options.FilterMinNcc).IsEqualTo((double)0.1f);
		await Assert.That(options.SigmaColor).IsNotEqualTo(0.2);

		await Assert.That(options.MinTriangulationAngle).IsEqualTo(1.0);
		await Assert.That(options.GeomConsistencyMaxCost).IsEqualTo(3.0);
		await Assert.That(options.FilterMinTriangulationAngle).IsEqualTo(3.0);
		await Assert.That(options.FilterGeomConsistencyMaxCost).IsEqualTo(1.0);
		await Assert.That(options.CacheSize).IsEqualTo(32.0);
		await Assert.That(options.MaxImageSize).IsEqualTo(-1);
		await Assert.That(options.WindowRadius).IsEqualTo(5);
		await Assert.That(options.WindowStep).IsEqualTo(1);
		await Assert.That(options.NumSamples).IsEqualTo(15);
		await Assert.That(options.NumIterations).IsEqualTo(5);
		await Assert.That(options.FilterMinNumConsistent).IsEqualTo(2);
		await Assert.That(options.NumThreads).IsEqualTo(-1);
		await Assert.That(options.GeomConsistency).IsTrue();
		await Assert.That(options.Filter).IsTrue();
		await Assert.That(options.AllowMissingFiles).IsFalse();
		await Assert.That(options.WriteConsistencyGraph).IsFalse();
		await Assert.That(options.Check()).IsTrue();
	}

	[Test]
	public async Task PatchMatchOptions_CheckDepthRange()
	{
		await Assert.That(new PatchMatchOptions { DepthMin = 1, DepthMax = 2 }.Check()).IsTrue();
		await Assert.That(new PatchMatchOptions { DepthMin = 0, DepthMax = 0 }.Check()).IsTrue();
		await Assert.That(new PatchMatchOptions { DepthMin = 2, DepthMax = 1 }.Check()).IsFalse();

		// Setting only one bound still checks both.
		await Assert.That(new PatchMatchOptions { DepthMin = 1 }.Check()).IsFalse();
		await Assert.That(new PatchMatchOptions { DepthMax = 1 }.Check()).IsFalse();
		await Assert.That(new PatchMatchOptions { DepthMin = -0.5, DepthMax = 1 }.Check()).IsFalse();
	}

	[Test]
	public async Task PatchMatchOptions_CheckBounds()
	{
		var cases = new (Action<PatchMatchOptions> Set, bool Valid)[]
		{
			(o => o.WindowRadius = PatchMatchOptions.MaxPatchMatchWindowRadius, true),
			(o => o.WindowRadius = PatchMatchOptions.MaxPatchMatchWindowRadius + 1, false),
			(o => o.WindowRadius = 0, false),
			(o => o.SigmaColor = 0, false),
			(o => o.WindowStep = 0, false),
			(o => o.WindowStep = 2, true),
			(o => o.WindowStep = 3, false),
			(o => o.NumSamples = 0, false),
			(o => o.NccSigma = 0, false),
			(o => o.MinTriangulationAngle = 0, true),
			(o => o.MinTriangulationAngle = -1, false),
			(o => o.MinTriangulationAngle = 180, false),
			(o => o.IncidentAngleSigma = 0, false),
			(o => o.NumIterations = 0, false),
			(o => o.GeomConsistencyRegularizer = 0, true),
			(o => o.GeomConsistencyRegularizer = -0.1, false),
			(o => o.GeomConsistencyMaxCost = -0.1, false),
			(o => o.FilterMinNcc = -1, true),
			(o => o.FilterMinNcc = 1, true),
			(o => o.FilterMinNcc = -1.1, false),
			(o => o.FilterMinNcc = 1.1, false),
			(o => o.FilterMinTriangulationAngle = 180, true),
			(o => o.FilterMinTriangulationAngle = -1, false),
			(o => o.FilterMinTriangulationAngle = 181, false),
			(o => o.FilterMinNumConsistent = 0, true),
			(o => o.FilterMinNumConsistent = -1, false),
			(o => o.FilterGeomConsistencyMaxCost = -0.1, false),
			(o => o.CacheSize = 0, false),
			(o => o.NumThreads = -2, false),
			(o => o.NumThreads = 4, true),
		};

		for (int i = 0; i < cases.Length; i++)
		{
			var options = new PatchMatchOptions();
			cases[i].Set(options);
			await Assert.That(options.Check()).IsEqualTo(cases[i].Valid).Because($"case {i}");
		}
	}

	[Test]
	public void PatchMatch_CheckValidProblem()
	{
		// Check throws on an invalid problem; these must pass.
		new PatchMatch(PhotometricOptions(), NewProblem(withMaps: false)).Check();
		new PatchMatch(new PatchMatchOptions(), NewProblem(withMaps: true)).Check();
	}

	[Test]
	public async Task PatchMatch_CheckInvalidOptions()
	{
		var options = PhotometricOptions();
		options.WindowRadius = 0;
		await Assert.That(() => new PatchMatch(options, NewProblem(withMaps: false)).Check()).Throws<ArgumentException>();
	}

	[Test]
	public async Task PatchMatch_CheckImageSet()
	{
		PatchMatch.Problem noSources = NewProblem(withMaps: false);
		noSources.SrcImageIdxs = [];
		await Assert.That(() => new PatchMatch(PhotometricOptions(), noSources).Check()).Throws<ArgumentException>();

		PatchMatch.Problem duplicate = NewProblem(withMaps: false);
		duplicate.SrcImageIdxs = [1, 1];
		await Assert.That(() => new PatchMatch(PhotometricOptions(), duplicate).Check()).Throws<ArgumentException>();

		PatchMatch.Problem refAsSource = NewProblem(withMaps: false);
		refAsSource.SrcImageIdxs = [0, 1];
		await Assert.That(() => new PatchMatch(PhotometricOptions(), refAsSource).Check()).Throws<ArgumentException>();

		PatchMatch.Problem outOfRange = NewProblem(withMaps: false);
		outOfRange.SrcImageIdxs = [1, 3];
		await Assert.That(() => new PatchMatch(PhotometricOptions(), outOfRange).Check()).Throws<ArgumentException>();

		PatchMatch.Problem negative = NewProblem(withMaps: false);
		negative.RefImageIdx = -1;
		await Assert.That(() => new PatchMatch(PhotometricOptions(), negative).Check()).Throws<ArgumentException>();

		PatchMatch.Problem noImages = NewProblem(withMaps: false);
		noImages.Images = null;
		await Assert.That(() => new PatchMatch(PhotometricOptions(), noImages).Check()).Throws<ArgumentException>();
	}

	[Test]
	public async Task PatchMatch_CheckImages()
	{
		PatchMatch.Problem noBitmap = NewProblem(withMaps: false);
		noBitmap.Images![2] = new Image("img", Width, Height, K, R, T);
		await Assert.That(() => new PatchMatch(PhotometricOptions(), noBitmap).Check()).Throws<ArgumentException>();

		PatchMatch.Problem rgb = NewProblem(withMaps: false);
		rgb.Images![1] = NewImage(asRgb: true);
		await Assert.That(() => new PatchMatch(PhotometricOptions(), rgb).Check()).Throws<ArgumentException>();

		// Only fx, fy, cx and cy may be set; skew or a projective row is rejected.
		PatchMatch.Problem skew = NewProblem(withMaps: false);
		skew.Images![1] = NewImage([100, 0.5f, 2, 0, 100, 1.5f, 0, 0, 1]);
		await Assert.That(() => new PatchMatch(PhotometricOptions(), skew).Check()).Throws<ArgumentException>();

		PatchMatch.Problem lastRow = NewProblem(withMaps: false);
		lastRow.Images![0] = NewImage([100, 0, 2, 0, 100, 1.5f, 0, 0, 2]);
		await Assert.That(() => new PatchMatch(PhotometricOptions(), lastRow).Check()).Throws<ArgumentException>();

		// An unused image is not checked.
		PatchMatch.Problem unusedRgb = NewProblem(withMaps: false, numImages: 4);
		unusedRgb.Images![3] = NewImage(asRgb: true);
		new PatchMatch(PhotometricOptions(), unusedRgb).Check();
	}

	[Test]
	public async Task PatchMatch_CheckGeomConsistencyMaps()
	{
		// Geometric consistency needs depth and normal maps for every image.
		await Assert.That(() => new PatchMatch(new PatchMatchOptions(), NewProblem(withMaps: false)).Check()).Throws<ArgumentException>();

		PatchMatch.Problem fewerMaps = NewProblem(withMaps: true);
		fewerMaps.DepthMaps!.RemoveAt(2);
		await Assert.That(() => new PatchMatch(new PatchMatchOptions(), fewerMaps).Check()).Throws<ArgumentException>();

		PatchMatch.Problem wrongDepthSize = NewProblem(withMaps: true);
		wrongDepthSize.DepthMaps![1] = new DepthMap(Width + 1, Height, 1, 2);
		await Assert.That(() => new PatchMatch(new PatchMatchOptions(), wrongDepthSize).Check()).Throws<ArgumentException>();

		PatchMatch.Problem wrongRefNormalSize = NewProblem(withMaps: true);
		wrongRefNormalSize.NormalMaps![0] = new NormalMap(Width, Height + 1);
		await Assert.That(() => new PatchMatch(new PatchMatchOptions(), wrongRefNormalSize).Check()).Throws<ArgumentException>();

		// Only the reference image's normal map is checked.
		PatchMatch.Problem wrongSrcNormalSize = NewProblem(withMaps: true);
		wrongSrcNormalSize.NormalMaps![1] = new NormalMap();
		new PatchMatch(new PatchMatchOptions(), wrongSrcNormalSize).Check();
	}

	[Test]
	public async Task PatchMatch_CopiesProblem()
	{
		// The PatchMatch copies the problem's source list, as the C++ copy does.
		PatchMatch.Problem problem = NewProblem(withMaps: false);
		var patchMatch = new PatchMatch(PhotometricOptions(), problem);
		problem.SrcImageIdxs.Add(0);
		patchMatch.Check();
		await Assert.That(() => new PatchMatch(PhotometricOptions(), problem).Check()).Throws<ArgumentException>();
	}
}
