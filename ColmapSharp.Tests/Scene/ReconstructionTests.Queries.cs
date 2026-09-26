// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests, second half of colmap/scene/reconstruction_test.cc: normalization,
// transforms, cropping, lookups, statistics, validation and image directories, testing
// ColmapSharp/Scene/Reconstruction.Queries.cs. Helpers and translation notes are in
// ReconstructionTests.cs. The last cases are C#-only (labeled): they pin Clone's pointer
// rewiring and the ascending-id iteration order that docs/CPP_DIVERGENCES.md entry 21
// documents (the ported ConstructCopy/AssignCopy in ReconstructionTests.IO.cs cover copies
// of a synthetic dataset).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using static ColmapSharp.Tests.ReconstructionMatchers;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	private static async Task ExpectCamZ(Reconstruction reconstruction, uint imageId, double z)
	{
		await Assert.That(reconstruction.Image(imageId).CamFromWorld().Translation.Z).IsEqualTo(z).Within(1e-6);
	}

	[Test]
	public async Task Reconstruction_Normalize()
	{
		Reconstruction reconstruction = GenerateReconstruction(7);
		foreach (uint frameId in reconstruction.Frames.Keys)
		{
			reconstruction.DeRegisterFrame(frameId);
		}

		Sim3d tform = reconstruction.Normalize(fixedScale: false);
		await Assert.That(tform.Scale).IsEqualTo(1.0);
		await Assert.That(tform.Rotation.Coeffs).IsEqualTo(Quaterniond.Identity.Coeffs);
		await Assert.That(tform.Translation).IsEqualTo(Vector3d.Zero);
		reconstruction.Frame(1).SetRigFromWorld(TranslatedZ(-20.0));
		reconstruction.Frame(2).SetRigFromWorld(TranslatedZ(-10.0));
		reconstruction.Frame(3).SetRigFromWorld(TranslatedZ(0.0));
		reconstruction.RegisterFrame(1);
		reconstruction.RegisterFrame(2);
		reconstruction.RegisterFrame(3);
		reconstruction.Normalize(fixedScale: true);
		await ExpectCamZ(reconstruction, 1, -10);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 10);
		reconstruction.Normalize(fixedScale: false);
		await ExpectCamZ(reconstruction, 1, -5);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 5);
		reconstruction.Normalize(fixedScale: false, 5);
		await ExpectCamZ(reconstruction, 1, -2.5);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 2.5);
		reconstruction.Normalize(fixedScale: false, 10, 0.0, 1.0);
		await ExpectCamZ(reconstruction, 1, -5);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 5);
		tform = reconstruction.Normalize(fixedScale: false, 20);
		await ExpectCamZ(reconstruction, 1, -10);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 10);
		reconstruction.Transform(tform.Inverse());
		await ExpectCamZ(reconstruction, 1, -5);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 5);
		reconstruction.Transform(tform);
		await ExpectCamZ(reconstruction, 1, -10);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 10);
		reconstruction.Image(4).FramePtr.SetRigFromWorld(TranslatedZ(-7.5));
		reconstruction.Image(5).FramePtr.SetRigFromWorld(TranslatedZ(-5.0));
		reconstruction.Image(6).FramePtr.SetRigFromWorld(TranslatedZ(5.0));
		reconstruction.Image(7).FramePtr.SetRigFromWorld(TranslatedZ(7.5));
		reconstruction.RegisterFrame(4);
		reconstruction.RegisterFrame(5);
		reconstruction.RegisterFrame(6);
		reconstruction.RegisterFrame(7);
		reconstruction.Normalize(fixedScale: false, 10, 0.0, 1.0);
		await ExpectCamZ(reconstruction, 1, -5);
		await ExpectCamZ(reconstruction, 2, 0);
		await ExpectCamZ(reconstruction, 3, 5);
		await ExpectCamZ(reconstruction, 4, -3.75);
		await ExpectCamZ(reconstruction, 5, -2.5);
		await ExpectCamZ(reconstruction, 6, 2.5);
		await ExpectCamZ(reconstruction, 7, 3.75);
	}

	[Test]
	public async Task Reconstruction_ComputeBoundsAndCentroidEmpty()
	{
		var reconstruction = new Reconstruction();
		Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
		AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
		await Assert.That(centroid.X).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(centroid.Y).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(centroid.Z).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Min.X).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Min.Y).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Min.Z).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Max.X).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Max.Y).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Max.Z).IsEqualTo(0.0).Within(1e-6);
	}

	[Test]
	public async Task Reconstruction_ComputeBoundsAndCentroid()
	{
		var reconstruction = new Reconstruction();
		reconstruction.AddPoint3D(new Vector3d(3.0, 0.0, 0.0), new Track());
		reconstruction.AddPoint3D(new Vector3d(0.0, 3.0, 0.0), new Track());
		reconstruction.AddPoint3D(new Vector3d(0.0, 0.0, 3.0), new Track());
		Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
		AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
		await Assert.That(centroid.X).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(centroid.Y).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(centroid.Z).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(bbox.Min.X).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Min.Y).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Min.Z).IsEqualTo(0.0).Within(1e-6);
		await Assert.That(bbox.Max.X).IsEqualTo(3.0).Within(1e-6);
		await Assert.That(bbox.Max.Y).IsEqualTo(3.0).Within(1e-6);
		await Assert.That(bbox.Max.Z).IsEqualTo(3.0).Within(1e-6);
	}

	[Test]
	public async Task Reconstruction_Crop()
	{
		Reconstruction reconstruction = GenerateReconstruction(3);
		ulong pointId = reconstruction.AddPoint3D(new Vector3d(0.0, 0.0, 0.0), new Track());
		reconstruction.AddObservation(pointId, new TrackElement(1, 1));
		pointId = reconstruction.AddPoint3D(new Vector3d(0.5, 0.5, 0.0), new Track());
		reconstruction.AddObservation(pointId, new TrackElement(1, 2));
		pointId = reconstruction.AddPoint3D(new Vector3d(1.0, 1.0, 0.0), new Track());
		reconstruction.AddObservation(pointId, new TrackElement(2, 3));
		pointId = reconstruction.AddPoint3D(new Vector3d(0.0, 0.0, 0.5), new Track());
		reconstruction.AddObservation(pointId, new TrackElement(2, 4));
		pointId = reconstruction.AddPoint3D(new Vector3d(0.5, 0.5, 1.0), new Track());
		reconstruction.AddObservation(pointId, new TrackElement(3, 5));

		// Check correct reconstruction setup
		await Assert.That(reconstruction.NumCameras).IsEqualTo(1);
		await Assert.That(reconstruction.NumImages).IsEqualTo(3);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(3);
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(5);

		// Test emtpy reconstruction after cropping.
		Reconstruction cropped1 = reconstruction.Crop(new AlignedBox3d(new Vector3d(-1, -1, -1), new Vector3d(-0.5, -0.5, -0.5)));
		await Assert.That(cropped1.NumCameras).IsEqualTo(1);
		await Assert.That(cropped1.NumImages).IsEqualTo(3);
		await Assert.That(cropped1.NumRegFrames).IsEqualTo(0);
		await Assert.That(cropped1.NumPoints3D).IsEqualTo(0);

		// Test reconstruction with contents after cropping
		Reconstruction cropped2 = reconstruction.Crop(new AlignedBox3d(new Vector3d(0.0, 0.0, 0.0), new Vector3d(0.75, 0.75, 0.75)));
		await Assert.That(cropped2.NumCameras).IsEqualTo(1);
		await Assert.That(cropped2.NumImages).IsEqualTo(3);
		await Assert.That(cropped2.NumRegFrames).IsEqualTo(2);
		await Assert.That(cropped2.NumPoints3D).IsEqualTo(3);
		await Assert.That(cropped2.Image(1).HasPose).IsTrue();
		await Assert.That(cropped2.Image(2).HasPose).IsTrue();
		await Assert.That(cropped2.Image(3).HasPose).IsFalse();
	}

	[Test]
	public async Task Reconstruction_Transform()
	{
		Reconstruction reconstruction = GenerateReconstruction(3);
		ulong point3DId = reconstruction.AddPoint3D(new Vector3d(1, 1, 1), new Track());
		reconstruction.AddObservation(point3DId, new TrackElement(1, 1));
		reconstruction.AddObservation(point3DId, new TrackElement(2, 1));
		reconstruction.Transform(new Sim3d(2, Quaterniond.Identity, new Vector3d(0, 1, 2)));
		await Assert.That(reconstruction.Image(1).ProjectionCenter() == new Vector3d(0, 1, 2)).IsTrue();
		await Assert.That(reconstruction.Point3D(point3DId).Xyz == new Vector3d(2, 3, 4)).IsTrue();
	}

	[Test]
	public async Task Reconstruction_FindImageWithName()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		await Assert.That(ReferenceEquals(reconstruction.FindImageWithName("image1"), reconstruction.Image(1))).IsTrue();
		await Assert.That(ReferenceEquals(reconstruction.FindImageWithName("image2"), reconstruction.Image(2))).IsTrue();
		await Assert.That(reconstruction.FindImageWithName("image3") is null).IsTrue();
	}

	[Test]
	public async Task Reconstruction_FindCommonRegImageIds()
	{
		Reconstruction reconstruction1 = GenerateReconstruction(5);
		Reconstruction reconstruction2 = GenerateReconstruction(5);
		reconstruction1.DeRegisterFrame(1);
		reconstruction1.Image(2).Name = "foo";
		reconstruction2.DeRegisterFrame(3);
		reconstruction2.Image(4).Name = "bar";
		List<(uint ImageId, uint OtherImageId)> commonImageIds = reconstruction1.FindCommonRegImageIds(reconstruction2);
		await Assert.That(commonImageIds.Count).IsEqualTo(1);
		await Assert.That(commonImageIds[0].ImageId).IsEqualTo(5u);
		await Assert.That(commonImageIds[0].OtherImageId).IsEqualTo(5u);
		await Assert.That(commonImageIds.SequenceEqual(reconstruction2.FindCommonRegImageIds(reconstruction1))).IsTrue();
	}

	[Test]
	public async Task Reconstruction_ComputeNumObservations()
	{
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(2);
		ulong point3DId1 = reconstruction.AddPoint3D(xyz, new Track());
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(0L);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(1L);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 1));
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(2L);
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.ComputeNumObservations()).IsEqualTo(3L);
	}

	[Test]
	public async Task Reconstruction_ComputeMeanTrackLength()
	{
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(2);
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(0.0);
		ulong point3DId1 = reconstruction.AddPoint3D(xyz, new Track());
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(0.0);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(1.0);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 1));
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(2.0);
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.ComputeMeanTrackLength()).IsEqualTo(3.0);
	}

	[Test]
	public async Task Reconstruction_ComputeMeanObservationsPerRegImage()
	{
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(2);
		await Assert.That(reconstruction.ComputeMeanObservationsPerRegImage()).IsEqualTo(0.0);
		ulong point3DId1 = reconstruction.AddPoint3D(xyz, new Track());
		await Assert.That(reconstruction.ComputeMeanObservationsPerRegImage()).IsEqualTo(0.0);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 0));
		await Assert.That(reconstruction.ComputeMeanObservationsPerRegImage()).IsEqualTo(0.5);
		reconstruction.AddObservation(point3DId1, new TrackElement(1, 1));
		await Assert.That(reconstruction.ComputeMeanObservationsPerRegImage()).IsEqualTo(1.0);
		reconstruction.AddObservation(point3DId1, new TrackElement(2, 0));
		await Assert.That(reconstruction.ComputeMeanObservationsPerRegImage()).IsEqualTo(1.5);
	}

	[Test]
	public async Task Reconstruction_ComputeMeanReprojectionError()
	{
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(2);
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(0.0);
		ulong point3DId1 = reconstruction.AddPoint3D(xyz, new Track());
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(0.0);
		reconstruction.Point3D(point3DId1).Error = 0.0;
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(0.0);
		reconstruction.Point3D(point3DId1).Error = 1.0;
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(1.0);
		reconstruction.Point3D(point3DId1).Error = 2.0;
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(2.0);
	}

	[Test]
	public async Task Reconstruction_UpdatePoint3DErrors()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		await Assert.That(reconstruction.ComputeMeanReprojectionError()).IsEqualTo(0.0);
		var track = new Track();
		track.AddElement(1, 0);
		reconstruction.Image(1).Point2DAt(0).Xy = new Vector2d(0.5, 0.5);
		ulong point3DId = reconstruction.AddPoint3D(new Vector3d(0, 0, 1), track);
		await Assert.That(reconstruction.Point3D(point3DId).Error).IsEqualTo(-1.0);
		reconstruction.UpdatePoint3DErrors();
		await Assert.That(reconstruction.Point3D(point3DId).Error).IsEqualTo(0.0);
		reconstruction.Point3D(point3DId).Xyz = new Vector3d(0, 1, 1);
		reconstruction.UpdatePoint3DErrors();
		await Assert.That(reconstruction.Point3D(point3DId).Error).IsEqualTo(1.0);
	}

	[Test]
	public async Task Reconstruction_IsValid()
	{
		Vector3d xyz = RandomEigen.RandomEigenVector3d();
		Reconstruction reconstruction = GenerateReconstruction(2);
		var track = new Track();
		track.AddElement(1, 0);
		track.AddElement(2, 1);
		reconstruction.AddPoint3D(xyz, track);
		await Assert.That(reconstruction.IsValid()).IsTrue();

		// Test empty frame pointer for image.
		{
			Reconstruction reconstructionCopy = reconstruction.Clone();
			reconstructionCopy.Image(1).ResetFramePtr();
			await Assert.That(reconstructionCopy.IsValid()).IsFalse();
		}

		// Test breaking track consistency by directly modifying point3D track.
		{
			Reconstruction reconstructionCopy = reconstruction.Clone();
			reconstructionCopy.Point3D(1).Track.SetElement(0, new TrackElement(1, 5));
			await Assert.That(reconstructionCopy.IsValid()).IsFalse();
		}

		// Test registered frame without pose.
		{
			Reconstruction reconstructionCopy = reconstruction.Clone();
			reconstructionCopy.Frame(1).ResetPose();
			await Assert.That(reconstructionCopy.IsValid()).IsFalse();
		}
	}

	[Test]
	public async Task Reconstruction_CreateImageDirs()
	{
		Reconstruction reconstruction = GenerateReconstruction(2);
		reconstruction.Image(1).Name = "subdir1/image1.jpg";
		reconstruction.Image(2).Name = "subdir2/subdir3/image2.jpg";

		// CreateTestDir() in COLMAP: a fresh directory per test.
		string testDir = Path.Combine(Path.GetTempPath(), "ColmapSharpTests", nameof(Reconstruction_CreateImageDirs), Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(testDir);
		try
		{
			reconstruction.CreateImageDirs(testDir);

			await Assert.That(Directory.Exists(Path.Combine(testDir, "subdir1"))).IsTrue();
			await Assert.That(Directory.Exists(Path.Combine(testDir, "subdir2", "subdir3"))).IsTrue();
		}
		finally
		{
			Directory.Delete(testDir, recursive: true);
		}
	}

	/// <summary>C#-only: Clone deep-copies every object and rewires the references to the copy's own objects.</summary>
	[Test]
	public async Task CSharpOnly_CloneRewiresReferences()
	{
		Reconstruction reconstruction = GenerateReconstruction(3);
		var track = new Track();
		track.AddElement(1, 0);
		track.AddElement(2, 1);
		reconstruction.AddPoint3D(new Vector3d(1, 2, 3), track);

		Reconstruction copy = reconstruction.Clone();
		await Assert.That(ReconstructionEq(copy, reconstruction)).IsTrue();
		await Assert.That(copy.RegFrameIds.SequenceEqual(reconstruction.RegFrameIds)).IsTrue();
		await ExpectValidPtrs(copy);
		await Assert.That(ReferenceEquals(copy.Image(1), reconstruction.Image(1))).IsFalse();
		await Assert.That(copy.IsValid()).IsTrue();

		copy.Frame(1).SetRigFromWorld(TranslatedZ(1));
		await Assert.That(ReconstructionEq(copy, reconstruction)).IsFalse();
		await Assert.That(reconstruction.Frame(1).RigFromWorld() == new Rigid3d()).IsTrue();
	}

	/// <summary>
	/// C#-only: objects iterate in ascending id order whatever the insertion order, so Crop
	/// hands out new point ids in ascending order of the old ones
	/// (docs/CPP_DIVERGENCES.md, entry 21).
	/// </summary>
	[Test]
	public async Task CSharpOnly_IterationIsInAscendingIdOrder()
	{
		Reconstruction reconstruction = GenerateReconstruction(1);
		var point = new Point3D { Xyz = new Vector3d(0.25, 0, 0) };
		reconstruction.AddPoint3D(9, point);
		reconstruction.AddPoint3D(4, new Point3D { Xyz = new Vector3d(0.5, 0, 0) });
		reconstruction.AddPoint3D(6, new Point3D { Xyz = new Vector3d(0.75, 0, 0) });
		await Assert.That(reconstruction.Points3D.Keys.SequenceEqual(new ulong[] { 4, 6, 9 })).IsTrue();

		Reconstruction cropped = reconstruction.Crop(new AlignedBox3d(Vector3d.Zero, Vector3d.Ones));
		await Assert.That(cropped.Points3D.Keys.SequenceEqual(new ulong[] { 1, 2, 3 })).IsTrue();
		await Assert.That(cropped.Points3D.Values.Select(p => p.Xyz.X).SequenceEqual(new[] { 0.5, 0.75, 0.25 })).IsTrue();
	}
}
