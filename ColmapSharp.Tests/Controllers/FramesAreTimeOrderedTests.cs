// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FramesAreTimeOrderedTests: C#-only (divergences 142, 143 and 144; not in COLMAP).
// AutomaticReconstructionOptions.FramesAreTimeOrdered decides the features that need the photos
// in filming order - Object mode's temporal mask repair (SegmentationOptions.TemporalWindow 2),
// KLT video tracking and silhouette placement's time order - apart from Data, which also picks
// the matcher. Null keeps the old rule (time-ordered iff Data = Video); true lets a host match
// exhaustively (Data = Individual) and still get them. Checked on the controller's own switches,
// which the stages read; milliseconds.

using ColmapSharp.Controllers;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class FramesAreTimeOrderedTests
{
	[Test]
	public async Task CSharpOnly_IndividualDataWithTimeOrderedFramesGetsTemporalMasksAndTracking()
	{
		string workspace = NewWorkspace();
		try
		{
			AutomaticReconstructionController controller = Controller(
				workspace, AutomaticReconstructionOptions.DataType.Individual, framesAreTimeOrdered: true);
			await Assert.That(controller.MaskTemporalWindow).IsEqualTo(2);
			await Assert.That(controller.UsesVideoTracking).IsTrue();

			// And the other way: video data whose frames the host says are not in order gets neither.
			AutomaticReconstructionController unordered = Controller(
				workspace, AutomaticReconstructionOptions.DataType.Video, framesAreTimeOrdered: false);
			await Assert.That(unordered.MaskTemporalWindow).IsEqualTo(0);
			await Assert.That(unordered.UsesVideoTracking).IsFalse();
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	[Test]
	public async Task CSharpOnly_NullDefaultFollowsDataType()
	{
		string workspace = NewWorkspace();
		try
		{
			await Assert.That(new AutomaticReconstructionOptions().FramesAreTimeOrdered).IsNull();

			AutomaticReconstructionController video = Controller(workspace, AutomaticReconstructionOptions.DataType.Video, null);
			await Assert.That(video.MaskTemporalWindow).IsEqualTo(2);
			await Assert.That(video.UsesVideoTracking).IsTrue();

			foreach (AutomaticReconstructionOptions.DataType data in new[]
			{
				AutomaticReconstructionOptions.DataType.Individual, AutomaticReconstructionOptions.DataType.Internet,
			})
			{
				AutomaticReconstructionController other = Controller(workspace, data, null);
				await Assert.That(other.MaskTemporalWindow).IsEqualTo(0);
				await Assert.That(other.UsesVideoTracking).IsFalse();
			}
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	private static AutomaticReconstructionController Controller(
		string workspace, AutomaticReconstructionOptions.DataType data, bool? framesAreTimeOrdered) =>
		new(
			new AutomaticReconstructionOptions
			{
				WorkspacePath = workspace,
				Images = new InMemoryImageSource(),
				Subject = AutomaticReconstructionOptions.SubjectType.Object,
				Data = data,
				VideoTracking = true,
				FramesAreTimeOrdered = framesAreTimeOrdered,
			},
			new ReconstructionManager());

	private static string NewWorkspace()
	{
		string workspace = Path.Combine(Path.GetTempPath(), "colmap-sharp-time-ordered-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(workspace);
		return workspace;
	}
}
