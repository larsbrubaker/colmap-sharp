// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchInputsTests: C#-only tests (COLMAP has no tests for these; its GPU versions
// are only exercised through the CUDA kernel) for the CPU PatchMatch kernel's inputs:
// PatchMatchRandom.cs (the counter-based generator, divergence 86), PatchMatchRefImage.cs
// (the bilateral prefilter), PatchMatchTextures.cs (bilinear source sampling with border 0,
// divergence 95; point-sampled depth maps) and PatchMatchTransforms.cs (the rotated
// calibrations and pose tables, and the rotated-to-original pixel map). The prefilter is
// checked against a brute-force double-precision evaluation of the same formula; the
// transforms against the geometry they must describe.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchInputsTests
{
	[Test]
	public async Task PatchMatchRandom_RangeAndMoments()
	{
		double sum = 0;
		float min = float.MaxValue;
		float max = float.MinValue;
		const int N = 200000;
		var random = new PatchMatchRandom(7, 3, 5, PatchMatchRandom.SweepPhase(1, 2));
		for (int i = 0; i < N; ++i)
		{
			float u = random.NextUniform();
			min = MathF.Min(min, u);
			max = MathF.Max(max, u);
			sum += u;
		}

		// curand_uniform's range is (0, 1].
		await Assert.That(min).IsGreaterThan(0.0f);
		await Assert.That(max).IsLessThanOrEqualTo(1.0f);
		await Assert.That(Math.Abs(sum / N - 0.5)).IsLessThan(0.005);
	}

	[Test]
	public async Task PatchMatchRandom_IsAFunctionOfItsKey()
	{
		static float[] Draws(ulong seed, int row, int col, int phase)
		{
			var random = new PatchMatchRandom(seed, row, col, phase);
			return [random.NextUniform(), random.NextUniform(), random.NextUniform()];
		}

		float[] reference = Draws(0, 10, 20, 3);
		await Assert.That(Draws(0, 10, 20, 3)).IsEquivalentTo(reference);
		await Assert.That(Draws(1, 10, 20, 3)).IsNotEquivalentTo(reference);
		await Assert.That(Draws(0, 11, 20, 3)).IsNotEquivalentTo(reference);
		await Assert.That(Draws(0, 10, 21, 3)).IsNotEquivalentTo(reference);
		await Assert.That(Draws(0, 20, 10, 3)).IsNotEquivalentTo(reference);
		await Assert.That(Draws(0, 10, 20, 4)).IsNotEquivalentTo(reference);
		await Assert.That(Draws(0, 10, 20, PatchMatchRandom.InitNormalPhase)).IsNotEquivalentTo(reference);
		await Assert.That(reference[0]).IsNotEqualTo(reference[1]);
	}

	[Test]
	public async Task ToOriginalPixel_UndoesMatRotate()
	{
		const int Width = 7;
		const int Height = 4;
		var frame = new Mat<int>(Width, Height, 1);
		for (int i = 0; i < Width * Height; ++i)
		{
			frame.Data[i] = i;
		}

		for (int rotation = 1; rotation <= 4; ++rotation)
		{
			var rotated = new Mat<int>(frame.GetHeight(), frame.GetWidth(), 1);
			frame.Rotate(rotated);
			frame = rotated;
			for (int row = 0; row < frame.GetHeight(); ++row)
			{
				for (int col = 0; col < frame.GetWidth(); ++col)
				{
					(int origRow, int origCol) = PatchMatchTransforms.ToOriginalPixel(rotation, row, col, Width, Height);
					await Assert.That(frame.Get(row, col)).IsEqualTo(origRow * Width + origCol);
				}
			}
		}
	}

	[Test]
	public async Task RefImageFilter_MatchesBruteForce()
	{
		const int Width = 9;
		const int Height = 6;
		var pixels = new byte[Width * Height];
		for (int i = 0; i < pixels.Length; ++i)
		{
			pixels[i] = (byte)((i * 37 + 11) % 256);
		}

		foreach ((int radius, int step) in new[] { (1, 1), (2, 1), (3, 2) })
		{
			const float SigmaSpatial = 2.0f;
			const float SigmaColor = 0.2f;
			var refImage = new PatchMatchRefImage(Width, Height);
			refImage.Filter(pixels, radius, step, SigmaSpatial, SigmaColor);

			for (int row = 0; row < Height; ++row)
			{
				for (int col = 0; col < Width; ++col)
				{
					// Border addressing: pixels outside the image are 0.
					double Color(int r, int c) => r < 0 || c < 0 || r >= Height || c >= Width ? 0 : pixels[r * Width + c] / 255.0;
					double center = Color(row, col);
					double sum = 0, squaredSum = 0, weightSum = 0;
					for (int dr = -radius; dr <= radius; dr += step)
					{
						for (int dc = -radius; dc <= radius; dc += step)
						{
							double color = Color(row + dr, col + dc);
							double weight = Math.Exp(-(dr * dr + dc * dc) / (2.0 * SigmaSpatial * SigmaSpatial)
								- (center - color) * (center - color) / (2.0 * SigmaColor * SigmaColor));
							sum += weight * color;
							squaredSum += weight * color * color;
							weightSum += weight;
						}
					}

					await Assert.That(Math.Abs(refImage.SumImage.Get(row, col) - sum / weightSum)).IsLessThan(1e-5);
					await Assert.That(Math.Abs(refImage.SquaredSumImage.Get(row, col) - squaredSum / weightSum)).IsLessThan(1e-5);
					await Assert.That(refImage.Image.Get(row, col)).IsEqualTo(pixels[row * Width + col]);
				}
			}
		}
	}

	[Test]
	public async Task RefImageFilter_ConstantInterior()
	{
		// Away from the border a constant image has mean v and mean square v².
		var pixels = new byte[11 * 11];
		Array.Fill(pixels, (byte)128);
		var refImage = new PatchMatchRefImage(11, 11);
		refImage.Filter(pixels, 2, 1, 2.0f, 0.2f);
		float v = 128 / 255.0f;
		await Assert.That(Math.Abs(refImage.SumImage.Get(5, 5) - v)).IsLessThan(1e-6f);
		await Assert.That(Math.Abs(refImage.SquaredSumImage.Get(5, 5) - v * v)).IsLessThan(1e-6f);

		// At a corner the zero border pulls the mean down.
		await Assert.That(refImage.SumImage.Get(0, 0)).IsLessThan(v);
	}

	private static Image GreyImage(int width, int height, Func<int, int, byte> pixel)
	{
		float[] k = [10, 0, width / 2.0f, 0, 10, height / 2.0f, 0, 0, 1];
		float[] r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] t = [0, 0, 0];
		var image = new Image("img", width, height, k, r, t);
		var bitmap = new Bitmap(width, height, asRgb: false);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				bitmap.RowMajorData[y * width + x] = pixel(x, y);
			}
		}

		image.SetBitmap(bitmap);
		return image;
	}

	[Test]
	public async Task SourceImages_BilinearWithZeroBorder()
	{
		Image small = GreyImage(2, 2, (x, y) => (byte)(x == 0 && y == 0 ? 255 : y == 0 ? 51 : 102));
		Image large = GreyImage(3, 2, (x, y) => 255);
		var textures = new PatchMatchSourceImages([small, large]);
		await Assert.That(textures.MaxWidth).IsEqualTo(3);
		await Assert.That(textures.MaxHeight).IsEqualTo(2);

		// Texel centres read the texel exactly.
		await Assert.That(textures.Sample(0.5f, 0.5f, 0)).IsEqualTo(1.0f);
		await Assert.That(textures.Sample(1.5f, 0.5f, 0)).IsEqualTo(51 / 255.0f);
		await Assert.That(textures.Sample(0.5f, 1.5f, 0)).IsEqualTo(102 / 255.0f);

		// Halfway between two texels is their mean; at the image corner, half the weight
		// goes to the zero border on each axis.
		await Assert.That(Math.Abs(textures.Sample(1.0f, 0.5f, 0) - (1.0f + 51 / 255.0f) / 2)).IsLessThan(1e-7f);
		await Assert.That(Math.Abs(textures.Sample(0.0f, 0.0f, 0) - 0.25f)).IsLessThan(1e-7f);

		// The smaller image is padded with 0 to the layer size.
		await Assert.That(textures.Sample(2.5f, 0.5f, 0)).IsEqualTo(0.0f);
		await Assert.That(textures.Sample(2.5f, 0.5f, 1)).IsEqualTo(1.0f);

		// Far outside, and NaN, read the border.
		await Assert.That(textures.Sample(-5.0f, 0.5f, 1)).IsEqualTo(0.0f);
		await Assert.That(textures.Sample(0.5f, 1e9f, 1)).IsEqualTo(0.0f);
		await Assert.That(textures.Sample(float.NaN, 0.5f, 1)).IsEqualTo(0.0f);
	}

	[Test]
	public async Task SourceDepthMaps_PointSampledWithZeroBorder()
	{
		var depthMap = new DepthMap(2, 2, 0, 10);
		depthMap.Set(0, 0, 1);
		depthMap.Set(0, 1, 2);
		depthMap.Set(1, 0, 3);
		depthMap.Set(1, 1, 4);
		var textures = new PatchMatchSourceDepthMaps([depthMap], 3, 2);

		await Assert.That(textures.Sample(0.5f, 0.5f, 0)).IsEqualTo(1.0f);
		await Assert.That(textures.Sample(1.99f, 0.0f, 0)).IsEqualTo(2.0f);
		await Assert.That(textures.Sample(0.2f, 1.7f, 0)).IsEqualTo(3.0f);
		await Assert.That(textures.Sample(2.5f, 0.5f, 0)).IsEqualTo(0.0f);
		await Assert.That(textures.Sample(-0.01f, 0.5f, 0)).IsEqualTo(0.0f);
		await Assert.That(textures.Sample(0.5f, 2.0f, 0)).IsEqualTo(0.0f);
	}

	private static Image PosedImage(int width, int height, float[] k, float[] r, float[] t) => new("img", width, height, k, r, t);

	[Test]
	public async Task Transforms_RotatedCalibrationProjectsRotatedPixels()
	{
		const int Width = 40;
		const int Height = 30;
		float[] k = [50, 0, 18.5f, 0, 55, 13.25f, 0, 0, 1];
		float[] identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float c = MathF.Cos(0.1f);
		float s = MathF.Sin(0.1f);
		float[] srcR = [c, 0, s, 0, 1, 0, -s, 0, c];
		Image refImage = PosedImage(Width, Height, k, identity, [0, 0, 0]);
		Image srcImage = PosedImage(Width, Height, k, srcR, [-1, 0.2f, 0.1f]);
		var transforms = new PatchMatchTransforms([refImage, srcImage], 0, [1]);

		await Assert.That(transforms.RefK(0).ToArray()).IsEquivalentTo(new[] { 50f, 18.5f, 55f, 13.25f });
		await Assert.That(transforms.RefInvK(0)[0]).IsEqualTo(1.0f / 50);
		await Assert.That(transforms.RefInvK(0)[1]).IsEqualTo(-18.5f / 50);

		// A reference-frame point X projects to (u, v). After k rotations the camera frame is
		// RZ90^k X, and the pixel must be where Mat.Rotate moved (v, u) to.
		float[] x = [0.7f, -0.4f, 5.0f];
		float u = k[0] * x[0] / x[2] + k[2];
		float v = k[4] * x[1] / x[2] + k[5];

		// The source pixel of X, which no rotation may change.
		float[] p0 = transforms.Poses(0)[PatchMatchTransforms.POffset..(PatchMatchTransforms.POffset + 12)];
		(float srcU, float srcV) = Project(p0, x);

		float[] xk = [.. x];
		for (int rotation = 0; rotation < 4; ++rotation)
		{
			ReadOnlySpan<float> refK = transforms.RefK(rotation);
			float uk = refK[0] * xk[0] / xk[2] + refK[1];
			float vk = refK[2] * xk[1] / xk[2] + refK[3];

			// Where pixel (row v, col u) of the original lands after `rotation` rotations.
			float row = v;
			float col = u;
			int frameWidth = Width;
			int frameHeight = Height;
			for (int i = 0; i < rotation; ++i)
			{
				(row, col) = (frameWidth - 1 - col, row);
				(frameWidth, frameHeight) = (frameHeight, frameWidth);
			}

			await Assert.That(Math.Abs(uk - col)).IsLessThan(1e-3f);
			await Assert.That(Math.Abs(vk - row)).IsLessThan(1e-3f);

			float[] pk = transforms.Poses(rotation)[PatchMatchTransforms.POffset..(PatchMatchTransforms.POffset + 12)];
			(float srcUk, float srcVk) = Project(pk, xk);
			await Assert.That(Math.Abs(srcUk - srcU)).IsLessThan(1e-3f);
			await Assert.That(Math.Abs(srcVk - srcV)).IsLessThan(1e-3f);

			// The source's projection centre, seen from the rotated frame, is that frame's
			// -Rᵀ T.
			float[] pose = transforms.Poses(rotation);
			float[] cExpected = new float[3];
			MvsGeometry.ComputeProjectionCenter(pose.AsSpan(PatchMatchTransforms.ROffset, 9), pose.AsSpan(PatchMatchTransforms.TOffset, 3), cExpected);
			await Assert.That(pose.AsSpan(PatchMatchTransforms.COffset, 3).ToArray()).IsEquivalentTo(cExpected);

			// Rotate X by 90 degrees counter-clockwise about z for the next frame.
			xk = [xk[1], -xk[0], xk[2]];
		}
	}

	private static (float U, float V) Project(float[] p, float[] x)
	{
		float z = p[8] * x[0] + p[9] * x[1] + p[10] * x[2] + p[11];
		return ((p[0] * x[0] + p[1] * x[1] + p[2] * x[2] + p[3]) / z, (p[4] * x[0] + p[5] * x[1] + p[6] * x[2] + p[7]) / z);
	}
}
