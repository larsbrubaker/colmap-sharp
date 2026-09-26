// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchKernelTests: C#-only invariant tests (COLMAP tests patch_match_cuda.cu only
// through the CUDA pipeline) for the kernel math in ColmapSharp/Mvs/
// PatchMatchKernel.Geometry.cs, PatchMatchKernel.Photometric.cs and PatchMatchLikelihood.cs:
// each helper is checked against the geometry or probability it must compute (points on
// the propagated plane, homography versus projection matrix, NCC of identical and
// affinely related patches, the NCC prior integrating to 1), not against copied formulas.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchKernelTests
{
	private const int Width = 32;
	private const int Height = 24;
	private static readonly float[] K = [40, 0, 15.5f, 0, 42, 11.5f, 0, 0, 1];
	private static readonly float[] Identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];

	private static Image NewImage(float[] r, float[] t, Func<int, int, byte>? pixel = null)
	{
		var image = new Image("img", Width, Height, K, r, t);
		var bitmap = new Bitmap(Width, Height, asRgb: false);
		pixel ??= (x, y) => (byte)((x * 37 + y * 101 + x * y * 7) % 200 + 20);
		for (int y = 0; y < Height; ++y)
		{
			for (int x = 0; x < Width; ++x)
			{
				bitmap.RowMajorData[y * Width + x] = pixel(x, y);
			}
		}

		image.SetBitmap(bitmap);
		return image;
	}

	private static float[] RotY(float angle)
	{
		float c = MathF.Cos(angle);
		float s = MathF.Sin(angle);
		return [c, 0, s, 0, 1, 0, -s, 0, c];
	}

	/// <summary>A reference camera at the origin and one rotated, translated source camera.</summary>
	private static (List<Image> Images, PatchMatchTransforms Transforms) TwoViews()
	{
		List<Image> images = [NewImage(Identity, [0, 0, 0]), NewImage(RotY(0.15f), [-0.8f, 0.1f, 0.05f])];
		return (images, new PatchMatchTransforms(images, 0, [1]));
	}

	private static float[] Unit(float x, float y, float z)
	{
		float n = MathF.Sqrt(x * x + y * y + z * z);
		return [x / n, y / n, z / n];
	}

	[Test]
	public async Task PropagateDepth_StaysOnThePlane()
	{
		(_, PatchMatchTransforms transforms) = TwoViews();
		for (int rotation = 0; rotation < 4; ++rotation)
		{
			var frame = new PatchMatchFrame(transforms, rotation);

			// A fronto-parallel plane keeps its depth.
			await Assert.That(PatchMatchKernel.PropagateDepth(frame, 3.0f, [0, 0, -1], 5, 6)).IsEqualTo(3.0f);

			// A slanted plane (in the column's y-z plane): the propagated point satisfies the
			// plane equation n · X2 = n · X1.
			float[] normal = Unit(0, -0.6f, -1);
			float depth2 = PatchMatchKernel.PropagateDepth(frame, 4.0f, normal, 9, 10);
			float y1 = 4.0f * (frame.InvK2 * 9 + frame.InvK3);
			float y2 = depth2 * (frame.InvK2 * 10 + frame.InvK3);
			await Assert.That(Math.Abs(normal[1] * y2 + normal[2] * depth2 - (normal[1] * y1 + normal[2] * 4.0f))).IsLessThan(1e-5f);
		}
	}

	[Test]
	public async Task PropagateDepth_ParallelPlaneKeepsDepth()
	{
		(_, PatchMatchTransforms transforms) = TwoViews();
		var frame = new PatchMatchFrame(transforms, 0);

		// A plane containing the second viewing ray has no intersection: depth1 is kept.
		float x4 = frame.InvK2 * 10 + frame.InvK3;
		float[] normal = Unit(0, 1, -x4);
		await Assert.That(PatchMatchKernel.PropagateDepth(frame, 2.5f, normal, 9, 10)).IsEqualTo(2.5f);
	}

	[Test]
	public async Task ComposeHomography_MapsPlanePointsLikeTheProjectionMatrix()
	{
		(_, PatchMatchTransforms transforms) = TwoViews();
		for (int rotation = 0; rotation < 4; ++rotation)
		{
			var frame = new PatchMatchFrame(transforms, rotation);
			float[] poses = transforms.Poses(rotation);
			float[] normal = Unit(0.3f, -0.2f, -1);
			const int Row = 7;
			const int Col = 12;
			const float Depth = 5.0f;
			float[] h = new float[9];
			PatchMatchKernel.ComposeHomography(poses, frame, 0, Row, Col, Depth, normal, h);

			// Another pixel's ray meets the plane through the (Row, Col) point at X2; H must map
			// that pixel where P projects X2.
			float[] x1 = new float[3];
			PatchMatchKernel.ComputePointAtDepth(frame, Row, Col, Depth, x1);
			const float Row2 = 10;
			const float Col2 = 4;
			float[] ray = [frame.InvK0 * Col2 + frame.InvK1, frame.InvK2 * Row2 + frame.InvK3, 1];
			float lambda = PatchMatchKernel.DotProduct3(normal, x1) / PatchMatchKernel.DotProduct3(normal, ray);
			float[] x2 = [lambda * ray[0], lambda * ray[1], lambda * ray[2]];

			float[] p = poses[PatchMatchTransforms.POffset..(PatchMatchTransforms.POffset + 12)];
			float z = p[8] * x2[0] + p[9] * x2[1] + p[10] * x2[2] + p[11];
			float u = (p[0] * x2[0] + p[1] * x2[1] + p[2] * x2[2] + p[3]) / z;
			float v = (p[4] * x2[0] + p[5] * x2[1] + p[6] * x2[2] + p[7]) / z;

			(float hu, float hv) = PatchMatchKernel.Mat33DotVec3Homogeneous(h, Col2, Row2);
			await Assert.That(Math.Abs(hu - u)).IsLessThan(1e-3f);
			await Assert.That(Math.Abs(hv - v)).IsLessThan(1e-3f);
		}
	}

	[Test]
	public async Task ComputeViewingAngles_MatchesTheGeometry()
	{
		(_, PatchMatchTransforms transforms) = TwoViews();
		float[] poses = transforms.Poses(0);
		float[] c = poses[PatchMatchTransforms.COffset..(PatchMatchTransforms.COffset + 3)];
		float[] point = [0.3f, -0.2f, 4.0f];
		float[] normal = Unit(0.1f, 0.2f, -1);

		(float cosTri, float cosInc) = PatchMatchKernel.ComputeViewingAngles(poses, point, normal, 0);

		// The angle at the point between the rays to the two camera centres, and between the
		// normal and the ray to the source camera.
		double[] toRef = [-point[0], -point[1], -point[2]];
		double[] toSrc = [c[0] - point[0], c[1] - point[1], c[2] - point[2]];
		double Norm(double[] a) => Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
		double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		double expectedTri = Dot(toRef, toSrc) / (Norm(toRef) * Norm(toSrc));
		double expectedInc = Dot(toSrc, [normal[0], normal[1], normal[2]]) / Norm(toSrc);
		await Assert.That(Math.Abs(cosTri - expectedTri)).IsLessThan(1e-6);
		await Assert.That(Math.Abs(cosInc - expectedInc)).IsLessThan(1e-6);
	}

	[Test]
	public async Task RandomNormalsAndPerturbations()
	{
		(_, PatchMatchTransforms transforms) = TwoViews();
		var frame = new PatchMatchFrame(transforms, 0);
		float[] normal = new float[3];
		float[] perturbed = new float[3];
		for (int i = 0; i < 500; ++i)
		{
			int row = i % Height;
			int col = i * 7 % Width;
			var random = new PatchMatchRandom(3, row, col, PatchMatchRandom.InitNormalPhase);
			PatchMatchKernel.GenerateRandomNormal(frame, row, col, ref random, normal);
			float[] ray = [frame.InvK0 * col + frame.InvK1, frame.InvK2 * row + frame.InvK3, 1];
			await Assert.That(Math.Abs(PatchMatchKernel.DotProduct3(normal, normal) - 1)).IsLessThan(1e-5f);
			await Assert.That(PatchMatchKernel.DotProduct3(normal, ray)).IsLessThanOrEqualTo(0.0f);

			PatchMatchKernel.PerturbNormal(frame, row, col, 0.5f * MathF.PI, normal, ref random, perturbed);
			await Assert.That(Math.Abs(PatchMatchKernel.DotProduct3(perturbed, perturbed) - 1)).IsLessThan(1e-5f);
			await Assert.That(PatchMatchKernel.DotProduct3(perturbed, ray)).IsLessThan(0.0f);

			float depth = PatchMatchKernel.PerturbDepth(0.25f, 8.0f, ref random);
			await Assert.That(depth).IsGreaterThan(6.0f - 1e-5f);
			await Assert.That(depth).IsLessThanOrEqualTo(10.0f);
		}

		// No perturbation keeps the normal.
		var zero = new PatchMatchRandom(3, 0, 0, 0);
		float[] n = Unit(0.1f, 0.1f, -1);
		PatchMatchKernel.PerturbNormal(frame, 5, 5, 0.0f, n, ref zero, perturbed);
		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(Math.Abs(perturbed[i] - n[i])).IsLessThan(1e-6f);
		}
	}

	[Test]
	public async Task GeomConsistencyCost_ConsistentDepthIsFree()
	{
		// Source camera translated along x; a fronto-parallel plane at depth 4 has depth 4 in
		// both views.
		List<Image> images = [NewImage(Identity, [0, 0, 0]), NewImage(Identity, [-0.5f, 0, 0])];
		var transforms = new PatchMatchTransforms(images, 0, [1]);
		var frame = new PatchMatchFrame(transforms, 0);
		var srcDepth = new DepthMap(Width, Height, 0, 10);
		srcDepth.Fill(4.0f);
		var srcDepthMaps = new PatchMatchSourceDepthMaps([srcDepth], Width, Height);
		float[] poses = transforms.Poses(0);

		float cost = PatchMatchKernel.ComputeGeomConsistencyCost(poses, srcDepthMaps, frame, 10, 12, 4.0f, 0, 3.0f);
		await Assert.That(cost).IsLessThan(1e-3f);

		// A wrong depth reprojects away from the pixel (here by 5 px at depth 2), capped.
		float wrong = PatchMatchKernel.ComputeGeomConsistencyCost(poses, srcDepthMaps, frame, 10, 12, 2.0f, 0, 3.0f);
		await Assert.That(wrong).IsEqualTo(3.0f);

		// Projecting outside the source depth map costs the maximum.
		float outside = PatchMatchKernel.ComputeGeomConsistencyCost(poses, srcDepthMaps, frame, 10, 1, 4.0f, 0, 3.0f);
		await Assert.That(outside).IsEqualTo(3.0f);
	}

	[Test]
	public async Task GeomConsistencyCost_DegenerateBackProjectionCostsTheMaximum()
	{
		// CUDA's min(float, float) is fminf, which returns the non-NaN operand. A source
		// whose inverse projection sends the point to the origin (0 * inf = NaN in the
		// backward projection) must cost max_cost, as on the GPU.
		(_, PatchMatchTransforms transforms) = TwoViews();
		var frame = new PatchMatchFrame(transforms, 0);
		float[] poses = new float[PatchMatchTransforms.NumTformParams];
		float[] p = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
		p.CopyTo(poses, PatchMatchTransforms.POffset);
		var srcDepth = new DepthMap(Width, Height, 0, 10);
		srcDepth.Fill(4.0f);
		var srcDepthMaps = new PatchMatchSourceDepthMaps([srcDepth], Width, Height);

		float cost = PatchMatchKernel.ComputeGeomConsistencyCost(poses, srcDepthMaps, frame, 10, 12, 4.0f, 0, 3.0f);
		await Assert.That(cost).IsEqualTo(3.0f);
	}

	[Test]
	public async Task RotateNormalMap_RotatesAboutZ()
	{
		var normalMap = new Mat<float>(3, 2, 3);
		normalMap.FillWithVector([0.6f, -0.8f, 0.5f]);
		PatchMatchKernel.RotateNormalMap(normalMap);
		float[] slice = new float[3];
		normalMap.GetSlice(1, 2, slice);
		await Assert.That(slice).IsEquivalentTo(new[] { -0.8f, -0.6f, 0.5f });

		for (int i = 0; i < 3; ++i)
		{
			PatchMatchKernel.RotateNormalMap(normalMap);
		}

		normalMap.GetSlice(0, 0, slice);
		await Assert.That(slice).IsEquivalentTo(new[] { 0.6f, -0.8f, 0.5f });
	}

	private static PatchMatchPhotoConsistency PhotoConsistency(List<Image> images, int windowRadius, int windowStep)
	{
		var transforms = new PatchMatchTransforms(images, 0, [1]);
		var refImage = new PatchMatchRefImage(Width, Height);
		refImage.Filter(images[0].GetBitmap().RowMajorData, windowRadius, windowStep, windowRadius, 0.2f);
		var srcImages = new PatchMatchSourceImages([images[1]]);
		return new PatchMatchPhotoConsistency(
			refImage, srcImages, transforms.Poses(0), new PatchMatchFrame(transforms, 0), windowRadius, windowStep, windowRadius, 0.2f);
	}

	[Test]
	public async Task PhotoConsistency_IdenticalAndAffinePatches()
	{
		float[] frontal = [0, 0, -1];
		foreach ((int radius, int step) in new[] { (3, 1), (5, 2) })
		{
			// The same camera and image: the warp is the identity and NCC is 1.
			PatchMatchPhotoConsistency same = PhotoConsistency([NewImage(Identity, [0, 0, 0]), NewImage(Identity, [0, 0, 0])], radius, step);
			await Assert.That(same.Compute(12, 16, 3.0f, frontal, 0)).IsLessThan(1e-4f);

			// NCC ignores an affine intensity change.
			Func<int, int, byte> pattern = (x, y) => (byte)((x * 13 + y * 29 + x * y) % 100);
			PatchMatchPhotoConsistency affine = PhotoConsistency(
				[NewImage(Identity, [0, 0, 0], pattern), NewImage(Identity, [0, 0, 0], (x, y) => (byte)(2 * pattern(x, y) + 10))], radius, step);
			await Assert.That(affine.Compute(12, 16, 3.0f, frontal, 0)).IsLessThan(1e-4f);

			// An inverted source is perfectly anti-correlated: cost 2.
			PatchMatchPhotoConsistency inverted = PhotoConsistency(
				[NewImage(Identity, [0, 0, 0], pattern), NewImage(Identity, [0, 0, 0], (x, y) => (byte)(200 - pattern(x, y)))], radius, step);
			await Assert.That(inverted.Compute(12, 16, 3.0f, frontal, 0)).IsGreaterThan(2.0f - 1e-4f);

			// A textureless source has no variance: the maximum cost.
			PatchMatchPhotoConsistency flat = PhotoConsistency(
				[NewImage(Identity, [0, 0, 0]), NewImage(Identity, [0, 0, 0], (x, y) => 90)], radius, step);
			await Assert.That(flat.Compute(12, 16, 3.0f, frontal, 0)).IsEqualTo(PatchMatchPhotoConsistency.MaxCost);
		}
	}

	[Test]
	public async Task PhotoConsistency_ComputesRadiiBeyondCudaTemplates()
	{
		// COLMAP's CUDA kernel is instantiated for radii 1-20 only; Check accepts up to 32 and
		// the port computes them (docs/CPP_DIVERGENCES.md, entry 97). With the same camera
		// and image the windows (zero outside the image on both sides) match.
		float[] frontal = [0, 0, -1];
		foreach ((int radius, int step) in new[] { (25, 2), (PatchMatchOptions.MaxPatchMatchWindowRadius, 1) })
		{
			PatchMatchPhotoConsistency same = PhotoConsistency([NewImage(Identity, [0, 0, 0]), NewImage(Identity, [0, 0, 0])], radius, step);
			await Assert.That(same.Compute(12, 16, 3.0f, frontal, 0)).IsLessThan(1e-4f);

			// And a different source image is told apart.
			PatchMatchPhotoConsistency other = PhotoConsistency(
				[NewImage(Identity, [0, 0, 0]), NewImage(Identity, [0, 0, 0], (x, y) => (byte)((x * 11 + y * 3) % 90 + 40))], radius, step);
			await Assert.That(other.Compute(12, 16, 3.0f, frontal, 0)).IsGreaterThan(0.1f);
		}
	}

	[Test]
	public async Task LockstepMatchesScalarBitForBit()
	{
		// CLAUDE.md allows Vector128 lanes only when each lane is bit-identical to the scalar
		// path. Every pixel (borders included, where samples leave the image), random
		// depths and normals, both window steps, all four rotations' frames, and two source
		// images of different sizes, so the layers have different padding and offsets.
		Image wide = SizedImage(Width + 6, Height - 4, RotY(0.12f), [-0.6f, 0.15f, 0.1f]);
		Image tall = SizedImage(Width - 5, Height + 3, RotY(-0.1f), [0.5f, -0.1f, 0.05f]);
		List<Image> images = [NewImage(Identity, [0, 0, 0]), wide, tall];
		var transforms = new PatchMatchTransforms(images, 0, [1, 2]);
		var srcImages = new PatchMatchSourceImages([wide, tall]);
		int mismatches = 0;
		int compared = 0;
		foreach ((int radius, int step) in new[] { (3, 1), (4, 2) })
		{
			var refImage = new PatchMatchRefImage(Width, Height);
			refImage.Filter(images[0].GetBitmap().RowMajorData, radius, step, radius, 0.2f);
			for (int rotation = 0; rotation < 4; ++rotation)
			{
				// The reference image of rotation k is the original rotated k times.
				var frame = new PatchMatchFrame(transforms, rotation);
				var pcc = new PatchMatchPhotoConsistency(refImage, srcImages, transforms.Poses(rotation), frame, radius, step, radius, 0.2f);
				int frameWidth = refImage.Image.GetWidth();
				int frameHeight = refImage.Image.GetHeight();
				float[] refColors = new float[pcc.WindowCount];
				float[] weights = new float[pcc.WindowCount];
				float[] depths = new float[4];
				float[] normals = new float[12];
				float[] normal = new float[3];
				float[] lockstep = new float[4];
				for (int row = 0; row < frameHeight; ++row)
				{
					for (int col = 0; col < frameWidth; ++col)
					{
						var random = new PatchMatchRandom(11, row, col, rotation);
						for (int h = 0; h < 4; ++h)
						{
							depths[h] = PatchMatchKernel.GenerateRandomDepth(1.0f, 9.0f, ref random);
							PatchMatchKernel.GenerateRandomNormal(frame, row, col, ref random, normal);
							normal.CopyTo(normals, 3 * h);
						}

						float weightSum = pcc.PrepareWindow(row, col, refColors, weights);
						for (int srcImageIdx = 0; srcImageIdx < 2; ++srcImageIdx)
						{
							pcc.ComputeFourLockstep(row, col, depths, normals, srcImageIdx, refColors, weights, weightSum, lockstep);
							for (int h = 0; h < 4; ++h)
							{
								float scalar = pcc.Compute(row, col, depths[h], normals.AsSpan(3 * h, 3), srcImageIdx, refColors, weights, weightSum);
								compared++;
								if (BitConverter.SingleToInt32Bits(scalar) != BitConverter.SingleToInt32Bits(lockstep[h]))
								{
									mismatches++;
								}
							}
						}
					}
				}

				var rotated = new PatchMatchRefImage(refImage.Image.GetHeight(), refImage.Image.GetWidth());
				refImage.Image.Rotate(rotated.Image);
				refImage.SumImage.Rotate(rotated.SumImage);
				refImage.SquaredSumImage.Rotate(rotated.SquaredSumImage);
				refImage = rotated;
			}
		}

		await Assert.That(compared).IsEqualTo(2 * 4 * Width * Height * 2 * 4);
		await Assert.That(mismatches).IsEqualTo(0);

#if DEBUG
		// The vector blend (all four lanes inside the layer) ran, not only the per-lane
		// fallback.
		await Assert.That(srcImages.VectorBlendCount).IsGreaterThan(1000);
#endif
	}

	private static Image SizedImage(int width, int height, float[] r, float[] t)
	{
		float[] k = [40, 0, (width - 1) / 2.0f, 0, 42, (height - 1) / 2.0f, 0, 0, 1];
		var image = new Image("img", width, height, k, r, t);
		var bitmap = new Bitmap(width, height, asRgb: false);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				bitmap.RowMajorData[y * width + x] = (byte)((x * 53 + y * 29 + x * y * 3) % 210 + 15);
			}
		}

		image.SetBitmap(bitmap);
		return image;
	}

	[Test]
	public async Task PhotoConsistency_FindsTheTrueDepth()
	{
		// The source image is the reference image rendered from a camera shifted by 0.5
		// along x for a fronto-parallel plane at depth 4, i.e. shifted 5 px left. The cost is
		// lowest at the true depth.
		Func<int, int, byte> pattern = (x, y) => (byte)((x * 37 + y * 101 + x * y * 7) % 200 + 20);
		List<Image> images = [NewImage(Identity, [0, 0, 0], pattern), NewImage(Identity, [-0.5f, 0, 0], (x, y) => pattern(x + 5, y))];
		PatchMatchPhotoConsistency pcc = PhotoConsistency(images, 3, 1);
		float[] frontal = [0, 0, -1];
		float atTruth = pcc.Compute(12, 12, 4.0f, frontal, 0);
		await Assert.That(atTruth).IsLessThan(1e-4f);
		await Assert.That(pcc.Compute(12, 12, 3.0f, frontal, 0)).IsGreaterThan(atTruth + 0.1f);
		await Assert.That(pcc.Compute(12, 12, 6.0f, frontal, 0)).IsGreaterThan(atTruth + 0.1f);
	}

	[Test]
	public async Task ComputeInitialCost_FillsEveryPixelAndSource()
	{
		PatchMatchPhotoConsistency pcc = PhotoConsistency([NewImage(Identity, [0, 0, 0]), NewImage(RotY(0.05f), [-0.3f, 0, 0])], 2, 1);
		var depthMap = new Mat<float>(Width, Height, 1);
		depthMap.Fill(4.0f);
		var normalMap = new Mat<float>(Width, Height, 3);
		normalMap.FillWithVector([0, 0, -1]);
		var costMap = new Mat<float>(Width, Height, 1);
		PatchMatchKernel.ComputeInitialCost(costMap, depthMap, normalMap, pcc);

		for (int row = 0; row < Height; row += 5)
		{
			for (int col = 0; col < Width; col += 3)
			{
				await Assert.That(costMap.Get(row, col, 0)).IsEqualTo(pcc.Compute(row, col, 4.0f, [0, 0, -1], 0));
			}
		}
	}

	[Test]
	public async Task ThreadCountDoesNotChangeResults()
	{
		// Mat moves, the prefilter and the initial cost write each output once, so one thread
		// and many give bit-identical results.
		var mat = new Mat<float>(71, 45, 2);
		for (int i = 0; i < mat.Data.Length; ++i)
		{
			mat.Data[i] = i;
		}

		foreach (bool rotate in new[] { false, true })
		{
			var single = new Mat<float>(45, 71, 2);
			var many = new Mat<float>(45, 71, 2);
			if (rotate)
			{
				mat.Rotate(single, 1);
				mat.Rotate(many, 7);
			}
			else
			{
				mat.Transpose(single, 1);
				mat.Transpose(many, 7);
			}

			await Assert.That(many.Data).IsEquivalentTo(single.Data, CollectionOrdering.Matching);
		}

		Image image = NewImage(Identity, [0, 0, 0]);
		var refSingle = new PatchMatchRefImage(Width, Height);
		refSingle.Filter(image.GetBitmap().RowMajorData, 4, 1, 4, 0.2f, numThreads: 1);
		var refMany = new PatchMatchRefImage(Width, Height);
		refMany.Filter(image.GetBitmap().RowMajorData, 4, 1, 4, 0.2f, numThreads: 5);
		await Assert.That(refMany.SumImage.Data).IsEquivalentTo(refSingle.SumImage.Data, CollectionOrdering.Matching);
		await Assert.That(refMany.SquaredSumImage.Data).IsEquivalentTo(refSingle.SquaredSumImage.Data, CollectionOrdering.Matching);

		PatchMatchPhotoConsistency pcc = PhotoConsistency([image, NewImage(RotY(0.05f), [-0.3f, 0, 0])], 2, 1);
		var depthMap = new Mat<float>(Width, Height, 1);
		depthMap.Fill(4.0f);
		var normalMap = new Mat<float>(Width, Height, 3);
		normalMap.FillWithVector([0, 0, -1]);
		var costSingle = new Mat<float>(Width, Height, 1);
		var costMany = new Mat<float>(Width, Height, 1);
		PatchMatchKernel.ComputeInitialCost(costSingle, depthMap, normalMap, pcc, numThreads: 1);
		PatchMatchKernel.ComputeInitialCost(costMany, depthMap, normalMap, pcc, numThreads: 6);
		await Assert.That(costMany.Data).IsEquivalentTo(costSingle.Data, CollectionOrdering.Matching);
	}

	[Test]
	public async Task Likelihood_NccPriorIntegratesToOne()
	{
		foreach (float sigma in new[] { 0.3f, 0.6f, 1.0f })
		{
			var likelihood = new PatchMatchLikelihood(sigma, 0.0174f, 0.9f);
			const int N = 20000;
			double integral = 0;
			for (int i = 0; i < N; ++i)
			{
				integral += likelihood.ComputeNCCProb((i + 0.5f) * 2.0f / N) * (2.0 / N);
			}

			await Assert.That(Math.Abs(integral - 1.0)).IsLessThan(1e-4);
		}
	}

	[Test]
	public async Task Likelihood_ErfMatchesKnownValues()
	{
		// erf values from Abramowitz and Stegun, Table 7.1.
		await Assert.That(PatchMatchLikelihood.ErfF(0.0f)).IsEqualTo(0.0f);
		await Assert.That(Math.Abs(PatchMatchLikelihood.ErfF(0.5f) - 0.5204998778)).IsLessThan(1e-7);
		await Assert.That(Math.Abs(PatchMatchLikelihood.ErfF(1.0f) - 0.8427007929)).IsLessThan(1e-7);
		await Assert.That(Math.Abs(PatchMatchLikelihood.ErfF(2.0f) - 0.9953222650)).IsLessThan(1e-7);
		await Assert.That(Math.Abs(PatchMatchLikelihood.ErfF(-1.0f) + 0.8427007929)).IsLessThan(1e-7);
		await Assert.That(PatchMatchLikelihood.ErfF(7.0f)).IsEqualTo(1.0f);
	}

	[Test]
	public async Task Likelihood_MessagesAndPriors()
	{
		var likelihood = new PatchMatchLikelihood(0.6f, 1.0f * MathF.PI / 180, 0.9f);
		foreach (float cost in new[] { 0.0f, 0.5f, 1.0f, 2.0f })
		{
			foreach (float prev in new[] { 0.0f, 0.3f, 1.0f })
			{
				float alpha = likelihood.ComputeForwardMessage(cost, prev);
				float beta = likelihood.ComputeBackwardMessage(cost, prev);
				await Assert.That(alpha).IsBetween(0.0f, 1.0f);
				await Assert.That(beta).IsBetween(0.0f, 1.0f);
			}
		}

		// A good match makes "visible" more likely than a bad one.
		await Assert.That(likelihood.ComputeForwardMessage(0.0f, 0.5f)).IsGreaterThan(likelihood.ComputeForwardMessage(1.5f, 0.5f));

		// The previous probability dominates at full weight.
		await Assert.That(likelihood.ComputeSelProb(0.9f, 0.9f, 0.25f, 1.0f)).IsEqualTo(0.25f);
		await Assert.That(likelihood.ComputeSelProb(0.5f, 0.5f, 0.0f, 0.0f)).IsEqualTo(0.5f);

		// Triangulation prior: 1 at or beyond the minimum angle, 0 for a zero angle.
		await Assert.That(likelihood.ComputeTriProb(MathF.Cos(0.1f))).IsEqualTo(1.0f);
		await Assert.That(likelihood.ComputeTriProb(1.0f)).IsEqualTo(0.0f);

		// Incident prior: 1 head-on, exp(-1 / (2 sigma²)) at or beyond 90 degrees. A point on
		// the source camera centre gives a NaN cosine, which CUDA's fmaxf turns into 0.
		await Assert.That(likelihood.ComputeIncProb(1.0f)).IsEqualTo(1.0f);
		await Assert.That(likelihood.ComputeIncProb(float.NaN)).IsEqualTo(likelihood.ComputeIncProb(0.0f));
		await Assert.That(Math.Abs(likelihood.ComputeIncProb(-0.5f) - Math.Exp(-1 / (2 * 0.9 * 0.9)))).IsLessThan(1e-6);

		// Resolution prior. COLMAP warps the patch corners (±radius, spanning 2 radius) but
		// compares with a (2 radius + 1)² reference area, so even the identity scores
		// (6/7)² for radius 3; a 2x magnification spans 12² against 7².
		await Assert.That(Math.Abs(likelihood.ComputeResolutionProb(Identity, 10, 10, 3) - 36.0f / 49.0f)).IsLessThan(1e-6f);
		await Assert.That(Math.Abs(likelihood.ComputeResolutionProb([2, 0, 0, 0, 2, 0, 0, 0, 1], 10, 10, 3) - 49.0f / 144.0f)).IsLessThan(1e-6f);
	}

	[Test]
	public async Task FindMinCostAndCdf()
	{
		await Assert.That(PatchMatchKernel.FindMinCost([3, 1, 2, 1, 5])).IsEqualTo(3);
		float[] probs = [1, 3, 0, 4];
		PatchMatchKernel.TransformPDFToCDF(probs);
		await Assert.That(probs).IsEquivalentTo(new[] { 0.125f, 0.5f, 0.5f, 1.0f });
	}
}
