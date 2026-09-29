// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MatTests: colmap/mvs/mat_test.cc ported 1:1 (Suite_Name), testing ColmapSharp/Mvs/Mat.cs.
// Tier A (exact). The C#-only cases at the end pin the .bin file format byte for byte
// (header "w&h&d&" then little-endian floats, as mat.cc writes it) and the header parsing
// and truncation behavior of Read.

using System.Text;

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class MatTests
{
	[Test]
	public async Task Mat_Empty()
	{
		var mat = new Mat<int>();
		await Assert.That(mat.GetWidth()).IsEqualTo(0);
		await Assert.That(mat.GetHeight()).IsEqualTo(0);
		await Assert.That(mat.GetDepth()).IsEqualTo(0);
		await Assert.That(mat.GetNumBytes()).IsEqualTo(0);
	}

	[Test]
	public async Task Mat_NonEmpty()
	{
		var mat = new Mat<int>(1, 2, 3);
		await Assert.That(mat.GetWidth()).IsEqualTo(1);
		await Assert.That(mat.GetHeight()).IsEqualTo(2);
		await Assert.That(mat.GetDepth()).IsEqualTo(3);
		await Assert.That(mat.GetNumBytes()).IsEqualTo(24);
	}

	[Test]
	public async Task Mat_GetSet()
	{
		var mat = new Mat<int>(1, 2, 3);

		await Assert.That(mat.GetNumBytes()).IsEqualTo(24);

		mat.Set(0, 0, 0, 1);
		mat.Set(0, 0, 1, 2);
		mat.Set(0, 0, 2, 3);
		mat.Set(1, 0, 0, 4);
		mat.Set(1, 0, 1, 5);
		mat.Set(1, 0, 2, 6);

		await Assert.That(mat.Get(0, 0, 0)).IsEqualTo(1);
		await Assert.That(mat.Get(0, 0, 1)).IsEqualTo(2);
		await Assert.That(mat.Get(0, 0, 2)).IsEqualTo(3);
		await Assert.That(mat.Get(1, 0, 0)).IsEqualTo(4);
		await Assert.That(mat.Get(1, 0, 1)).IsEqualTo(5);
		await Assert.That(mat.Get(1, 0, 2)).IsEqualTo(6);

		int[] slice = new int[3];
		mat.GetSlice(0, 0, slice);
		await Assert.That(slice[0]).IsEqualTo(1);
		await Assert.That(slice[1]).IsEqualTo(2);
		await Assert.That(slice[2]).IsEqualTo(3);
		mat.GetSlice(1, 0, slice);
		await Assert.That(slice[0]).IsEqualTo(4);
		await Assert.That(slice[1]).IsEqualTo(5);
		await Assert.That(slice[2]).IsEqualTo(6);
	}

	[Test]
	public async Task Mat_Fill()
	{
		var mat = new Mat<int>(1, 2, 3);

		await Assert.That(mat.GetNumBytes()).IsEqualTo(24);

		mat.Fill(10);
		mat.Set(0, 0, 0, 10);
		mat.Set(0, 0, 1, 10);
		mat.Set(0, 0, 2, 10);
		mat.Set(1, 0, 0, 10);
		mat.Set(1, 0, 1, 10);
		mat.Set(1, 0, 2, 10);
	}

	// C#-only: the exact bytes Mat<float>::Write produces, and Read of them.
	[Test]
	public async Task Mat_WriteReadByteExact()
	{
		var mat = new Mat<float>(2, 1, 2);
		mat.Set(0, 0, 0, 1.5f);
		mat.Set(0, 1, 0, -2.0f);
		mat.Set(0, 0, 1, 0.1f);
		mat.Set(0, 1, 1, float.MaxValue);

		string path = Path.Combine(MvsTestUtils.CreateTestDir(), "mat.bin");
		mat.Write(path);

		var expected = new List<byte>(Encoding.ASCII.GetBytes("2&1&2&"));
		foreach (float value in new[] { 1.5f, -2.0f, 0.1f, float.MaxValue })
		{
			expected.AddRange(BitConverter.GetBytes(value));
		}

		await Assert.That(File.ReadAllBytes(path)).IsEquivalentTo(expected.ToArray(), CollectionOrdering.Matching);

		var read = new Mat<float>();
		read.Read(path);
		await Assert.That(read.GetWidth()).IsEqualTo(2);
		await Assert.That(read.GetHeight()).IsEqualTo(1);
		await Assert.That(read.GetDepth()).IsEqualTo(2);
		await Assert.That(read.Data).IsEquivalentTo(mat.Data, CollectionOrdering.Matching);
	}

	// C#-only: `file >> width >> c ...` skips whitespace before each field, so a header
	// with spaces reads; the first byte after the third '&' is data even if it is a space.
	[Test]
	public async Task Mat_ReadHeaderWithWhitespace()
	{
		string path = Path.Combine(MvsTestUtils.CreateTestDir(), "mat.bin");
		var bytes = new List<byte>(Encoding.ASCII.GetBytes(" 1 & 1\n& 1&"));
		bytes.AddRange(BitConverter.GetBytes(BitConverter.Int32BitsToSingle(0x20202020)));
		File.WriteAllBytes(path, bytes.ToArray());

		var mat = new Mat<float>();
		mat.Read(path);
		await Assert.That(mat.GetWidth()).IsEqualTo(1);
		await Assert.That(BitConverter.SingleToInt32Bits(mat.Get(0, 0))).IsEqualTo(0x20202020);
	}

	// C#-only: a zero dimension fails COLMAP's THROW_CHECK_GT, and a truncated payload
	// throws (divergence 62).
	[Test]
	public async Task Mat_ReadInvalid()
	{
		string dir = MvsTestUtils.CreateTestDir();
		string zeroPath = Path.Combine(dir, "zero.bin");
		File.WriteAllBytes(zeroPath, Encoding.ASCII.GetBytes("0&1&1&"));
		await Assert.That(() => new Mat<float>().Read(zeroPath)).Throws<ArgumentException>();

		string shortPath = Path.Combine(dir, "short.bin");
		File.WriteAllBytes(shortPath, Encoding.ASCII.GetBytes("2&1&1&abcd"));
		await Assert.That(() => new Mat<float>().Read(shortPath)).Throws<EndOfStreamException>();
	}
}
