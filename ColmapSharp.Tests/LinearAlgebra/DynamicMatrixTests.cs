// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DynamicMatrixTests (C#-only; COLMAP has no test for Eigen itself): MatrixXd and VectorXd
// in ColmapSharp/LinearAlgebra. Covers the layout contract (column-major storage,
// row-major FromRowMajor like Eigen's comma initializer), block/row/column access and the
// span views, products against hand-computed values, the transposed products against
// explicit transposes (bit-identical: same terms in the same order), and the round trip to
// the fixed-size types.

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Assertions.Enums;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class DynamicMatrixTests
{
	private static readonly MatrixXd A23 = MatrixXd.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]);

	[Test]
	public async Task Layout_ColumnMajorStorageAndRowMajorInitializer()
	{
		using (Assert.Multiple())
		{
			await Assert.That(A23.AsSpan().ToArray()).IsEquivalentTo(new double[] { 1, 4, 2, 5, 3, 6 }, CollectionOrdering.Matching);
			await Assert.That(A23[1, 2]).IsEqualTo(6.0);
			await Assert.That(A23.Row(1).AsSpan().ToArray()).IsEquivalentTo(new double[] { 4, 5, 6 }, CollectionOrdering.Matching);
			await Assert.That(A23.Col(1).AsSpan().ToArray()).IsEquivalentTo(new double[] { 2, 5 }, CollectionOrdering.Matching);
			await Assert.That(A23.Transpose().AsSpan().ToArray()).IsEquivalentTo(new double[] { 1, 2, 3, 4, 5, 6 }, CollectionOrdering.Matching);
			await Assert.That(() => A23[2, 0]).Throws<ArgumentOutOfRangeException>();
			await Assert.That(() => A23[0, 3]).Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task Blocks_CopyAndWriteBack()
	{
		var m = MatrixXd.FromRowMajor(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
		MatrixXd block = m.Block(1, 1, 2, 2);
		block[0, 0] = -1;
		var target = MatrixXd.Zero(3, 4);
		target.SetBlock(1, 2, block);
		m.ColumnSpan(3)[0] = 40;
		using (Assert.Multiple())
		{
			await Assert.That(block.AsSpan().ToArray()).IsEquivalentTo(new double[] { -1, 10, 7, 11 }, CollectionOrdering.Matching);
			await Assert.That(m[1, 1]).IsEqualTo(6.0);
			await Assert.That(m[0, 3]).IsEqualTo(40.0);
			await Assert.That(target[1, 2]).IsEqualTo(-1.0);
			await Assert.That(target[2, 3]).IsEqualTo(11.0);
			await Assert.That(target[0, 0]).IsEqualTo(0.0);
			await Assert.That(m.LeftCols(1).AsSpan().ToArray()).IsEquivalentTo(new double[] { 1, 5, 9 }, CollectionOrdering.Matching);
			await Assert.That(m.BottomRows(1).AsSpan().ToArray()).IsEquivalentTo(new double[] { 9, 10, 11, 12 }, CollectionOrdering.Matching);
			await Assert.That(m.ReverseRows()[0, 0]).IsEqualTo(9.0);
			await Assert.That(m.ReverseCols()[0, 0]).IsEqualTo(40.0);
			await Assert.That(() => m.Block(2, 0, 2, 1)).Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task Products_MatchHandComputedValues()
	{
		var b = MatrixXd.FromRowMajor(3, 2, [1, 0, 0, 1, 2, -1]);
		MatrixXd ab = A23 * b;
		VectorXd av = A23 * new VectorXd([1, 1, 1]);
		using (Assert.Multiple())
		{
			await Assert.That(ab.AsSpan().ToArray()).IsEquivalentTo(new double[] { 7, 16, -1, -1 }, CollectionOrdering.Matching);
			await Assert.That(av.AsSpan().ToArray()).IsEquivalentTo(new double[] { 6, 15 }, CollectionOrdering.Matching);
			await Assert.That((A23 + A23).AsSpan().ToArray()).IsEquivalentTo((A23 * 2).AsSpan().ToArray(), CollectionOrdering.Matching);
			await Assert.That((A23 - A23).Norm()).IsEqualTo(0.0);
			await Assert.That(A23.SquaredNorm()).IsEqualTo(91.0);
			await Assert.That(MatrixXd.Identity(3).Trace()).IsEqualTo(3.0);
			await Assert.That(() => A23 * A23).Throws<ArgumentException>();
		}
	}

	[Test]
	public async Task TransposedProducts_EqualExplicitTranspose()
	{
		var a = MatrixXd.FromRowMajor(4, 3, [0.3, -1.2, 2.5, 1.1, 0.7, -0.4, -2.2, 0.9, 1.6, 0.05, -0.8, 3.1]);
		var b = MatrixXd.FromRowMajor(4, 2, [1.5, -0.3, 0.2, 2.2, -1.1, 0.6, 0.9, 0.4]);
		var v = new VectorXd([0.5, -1.5, 2.0, 0.25]);
		MatrixXd gram = a.TransposeTimesSelf();
		using (Assert.Multiple())
		{
			await Assert.That(a.TransposeTimes(b).AsSpan().ToArray()).IsEquivalentTo((a.Transpose() * b).AsSpan().ToArray(), CollectionOrdering.Matching);
			await Assert.That(gram.AsSpan().ToArray()).IsEquivalentTo((a.Transpose() * a).AsSpan().ToArray(), CollectionOrdering.Matching);
			await Assert.That(gram.AsSpan().ToArray()).IsEquivalentTo(gram.Transpose().AsSpan().ToArray(), CollectionOrdering.Matching);
			await Assert.That(a.TransposeTimes(v).AsSpan().ToArray()).IsEquivalentTo((a.Transpose() * v).AsSpan().ToArray(), CollectionOrdering.Matching);
		}
	}

	[Test]
	public async Task FixedSizeConversions_RoundTrip()
	{
		var m3 = new Matrix3d(2, -1, 0.5, 3, 4, -2, 1, 0.25, 5);
		var m34 = new Matrix3x4d(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
		var m4 = Matrix4d.Identity;
		var m2 = new Matrix2d(1, 2, 3, 4);
		MatrixXd x3 = MatrixXd.From(m3);
		using (Assert.Multiple())
		{
			await Assert.That(x3[0, 2]).IsEqualTo(0.5);
			await Assert.That(x3.ToMatrix3d()).IsEqualTo(m3);
			await Assert.That(MatrixXd.From(m34).ToMatrix3x4d()).IsEqualTo(m34);
			await Assert.That(MatrixXd.From(m4).ToMatrix4d()).IsEqualTo(m4);
			await Assert.That(MatrixXd.From(m2).ToMatrix2d()).IsEqualTo(m2);
			await Assert.That(MatrixXd.From(m34)[2, 3]).IsEqualTo(12.0);
			await Assert.That(VectorXd.From(new Vector3d(1, 2, 3)).ToVector3d()).IsEqualTo(new Vector3d(1, 2, 3));
			await Assert.That(VectorXd.From(new Vector2d(1, 2)).ToVector2d()).IsEqualTo(new Vector2d(1, 2));
			await Assert.That(VectorXd.From(new Vector4d(1, 2, 3, 4)).ToVector4d()).IsEqualTo(new Vector4d(1, 2, 3, 4));
			await Assert.That(() => x3.ToMatrix4d()).Throws<ArgumentException>();
			await Assert.That((m3 * m3).IsApprox((x3 * x3).ToMatrix3d())).IsTrue();
		}
	}

	[Test]
	public async Task VectorXd_NormsAndNormalized()
	{
		var v = new VectorXd([3, 4]);
		using (Assert.Multiple())
		{
			await Assert.That(v.Norm()).IsEqualTo(5.0);
			await Assert.That(v.Normalized().AsSpan().ToArray()).IsEquivalentTo(new double[] { 0.6, 0.8 }, CollectionOrdering.Matching);
			await Assert.That(VectorXd.Zero(3).Normalized().Norm()).IsEqualTo(0.0);
			await Assert.That(v.Dot(new VectorXd([1, -1]))).IsEqualTo(-1.0);
			await Assert.That(VectorXd.Unit(3, 1)[1]).IsEqualTo(1.0);
			await Assert.That(new VectorXd([1, 2, 3, 4]).Segment(1, 2).AsSpan().ToArray()).IsEquivalentTo(new double[] { 2, 3 }, CollectionOrdering.Matching);
			await Assert.That(new VectorXd([-7, 2]).MaxAbs()).IsEqualTo(7.0);
		}
	}
}
