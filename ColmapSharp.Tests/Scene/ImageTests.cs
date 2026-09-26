// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ImageTests: colmap/scene/image_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Scene/Image.cs. C++ copy construction and copy
// assignment are both Image.Clone(); `image.Point2D(i)` is Point2DAt(i); pointer identity
// (`image.CameraPtr() == &camera`) is reference identity; operator<< is ToString().

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ImageTests
{
	[Test]
	public async Task Image_Default()
	{
		var image = new Image();
		using (Assert.Multiple())
		{
			await Assert.That(image.ImageId).IsEqualTo(InvalidImageId);
			await Assert.That(image.Name).IsEqualTo("");
			await Assert.That(image.CameraId).IsEqualTo(InvalidCameraId);
			await Assert.That(image.DataId).IsEqualTo(new DataId(new SensorId(SensorType.Camera, InvalidCameraId), InvalidImageId));
			await Assert.That(image.HasCameraId).IsFalse();
			await Assert.That(image.HasCameraPtr).IsFalse();
			await Assert.That(image.HasPose).IsFalse();
			await Assert.That(image.NumPoints2D).IsEqualTo(0u);
			await Assert.That(image.NumPoints3D).IsEqualTo(0u);
			await Assert.That(image.Points2D.Count).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Image_Equals()
	{
		var image = new Image();
		Image other = image.Clone();
		await Assert.That(image == other).IsTrue();
		image.Name = "test";
		await Assert.That(image != other).IsTrue();
		other.Name = "test";
		await Assert.That(image == other).IsTrue();
	}

	[Test]
	public async Task Image_Print()
	{
		var image = new Image { ImageId = 1 };
		image.SetCameraId(2);
		image.SetFrameId(3);
		image.Name = "test";
		await Assert.That(image.ToString()).IsEqualTo(
			"Image(image_id=1, camera_id=2, frame_id=3, name=\"test\", has_pose=0, triangulated=0/0)");
	}

	[Test]
	public async Task Image_ImageId()
	{
		var image = new Image();
		await Assert.That(image.ImageId).IsEqualTo(InvalidImageId);
		image.ImageId = 1;
		await Assert.That(image.ImageId).IsEqualTo(1u);
	}

	[Test]
	public async Task Image_Name()
	{
		// C++ has SetName and a mutable Name(); both are the Name property here.
		var image = new Image();
		await Assert.That(image.Name).IsEqualTo("");
		image.Name = "test1";
		await Assert.That(image.Name).IsEqualTo("test1");
		image.Name = "test2";
		await Assert.That(image.Name).IsEqualTo("test2");
	}

	[Test]
	public async Task Image_CameraId()
	{
		var image = new Image();
		await Assert.That(image.CameraId).IsEqualTo(InvalidCameraId);
		image.SetCameraId(1);
		await Assert.That(image.CameraId).IsEqualTo(1u);
	}

	[Test]
	public async Task Image_DataId()
	{
		var image = new Image { ImageId = 1 };
		image.SetCameraId(2);
		await Assert.That(image.DataId).IsEqualTo(new DataId(new SensorId(SensorType.Camera, 2), 1));
	}

	[Test]
	public async Task Image_CameraPtr()
	{
		var image = new Image();
		using (Assert.Multiple())
		{
			await Assert.That(image.HasCameraPtr).IsFalse();
			await Assert.That(() => image.CameraPtr).ThrowsException();
			var camera = new Camera { CameraId = 1 };
			await Assert.That(() => image.SetCameraPtr(camera)).ThrowsException();
			image.SetCameraId(2);
			await Assert.That(() => image.SetCameraPtr(camera)).ThrowsException();
			image.SetCameraId(1);
			image.SetCameraPtr(camera);
			await Assert.That(image.HasCameraPtr).IsTrue();
			await Assert.That(ReferenceEquals(image.CameraPtr, camera)).IsTrue();
			image.ResetCameraPtr();
			await Assert.That(image.HasCameraPtr).IsFalse();
			await Assert.That(() => image.CameraPtr).ThrowsException();
		}
	}

	[Test]
	public async Task Image_FramePtr()
	{
		Frame frame = CreateTrivialFrame(withPose: false);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(1);
		image.SetFrameId(1);
		await Assert.That(() => image.SetFramePtr(frame)).ThrowsException();
		frame.AddDataId(image.DataId);
		image.SetFramePtr(frame);
	}

	[Test]
	public async Task Image_SetResetPose()
	{
		Frame frame = CreateTrivialFrame(withPose: false);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(1);
		frame.AddDataId(image.DataId);
		image.SetFrameId(1);
		image.SetFramePtr(frame);
		using (Assert.Multiple())
		{
			await Assert.That(image.HasPose).IsFalse();
			await Assert.That(() => image.CamFromWorld()).ThrowsException();
			frame.SetRigFromWorld(new Rigid3d());
			await Assert.That(image.HasPose).IsTrue();
			await Assert.That(image.CamFromWorld().Rotation.Coeffs).IsEqualTo(Quaterniond.Identity.Coeffs);
			await Assert.That(image.CamFromWorld().Translation).IsEqualTo(Vector3d.Zero);
			image.FramePtr.ResetPose();
			await Assert.That(image.HasPose).IsFalse();
			await Assert.That(() => image.CamFromWorld()).ThrowsException();
		}
	}

	[Test]
	public async Task Image_ConstructCopy()
	{
		Image image = CreatePosedImage("test");
		Image imageCopy = image.Clone();
		using (Assert.Multiple())
		{
			await Assert.That(image == imageCopy).IsTrue();
			await Assert.That(new Rigid3d() == imageCopy.CamFromWorld()).IsTrue();
			imageCopy.FramePtr.ResetPose();
			await Assert.That(image.HasPose).IsFalse();
			await Assert.That(imageCopy.HasPose).IsFalse();
		}
	}

	[Test]
	public async Task Image_AssignCopy()
	{
		Image image = CreatePosedImage("test");
		var imageCopy = new Image();
		imageCopy = image.Clone();
		using (Assert.Multiple())
		{
			await Assert.That(image == imageCopy).IsTrue();
			await Assert.That(imageCopy.CamFromWorld() == new Rigid3d()).IsTrue();
			imageCopy.FramePtr.ResetPose();
			await Assert.That(image.HasPose).IsFalse();
			await Assert.That(imageCopy.HasPose).IsFalse();
		}
	}

	[Test]
	public async Task Image_NumPoints2D()
	{
		var image = new Image();
		await Assert.That(image.NumPoints2D).IsEqualTo(0u);
		image.SetPoints2D(new Vector2d[10]);
		await Assert.That(image.NumPoints2D).IsEqualTo(10u);
	}

	[Test]
	public async Task Image_NumPoints3D()
	{
		var image = new Image();
		image.SetPoints2D(new Vector2d[10]);
		await Assert.That(image.NumPoints3D).IsEqualTo(0u);
		image.SetPoint3DForPoint2D(0, 0);
		await Assert.That(image.NumPoints3D).IsEqualTo(1u);
		image.SetPoint3DForPoint2D(0, 1);
		image.SetPoint3DForPoint2D(1, 2);
		await Assert.That(image.NumPoints3D).IsEqualTo(2u);
	}

	[Test]
	public async Task Image_Points2D()
	{
		var image = new Image();
		await Assert.That(image.Points2D.Count).IsEqualTo(0);
		var points2D = new Vector2d[10];
		points2D[0] = new Vector2d(1.0, 2.0);
		image.SetPoints2D(points2D);
		using (Assert.Multiple())
		{
			await Assert.That(image.Points2D.Count).IsEqualTo(10);
			await Assert.That(image.Point2DAt(0).Xy.X).IsEqualTo(1.0);
			await Assert.That(image.Point2DAt(0).Xy.Y).IsEqualTo(2.0);
			await Assert.That(image.NumPoints3D).IsEqualTo(0u);
		}
	}

	[Test]
	public async Task Image_Points2DWith3D()
	{
		var image = new Image();
		await Assert.That(image.Points2D.Count).IsEqualTo(0);
		var points2D = Enumerable.Range(0, 10).Select(_ => new Point2D()).ToList();
		points2D[0] = new Point2D { Xy = new Vector2d(1.0, 2.0), Point3DId = 1 };
		image.SetPoints2D(points2D);
		using (Assert.Multiple())
		{
			await Assert.That(image.Points2D.Count).IsEqualTo(10);
			await Assert.That(image.Point2DAt(0).Xy.X).IsEqualTo(1.0);
			await Assert.That(image.Point2DAt(0).Xy.Y).IsEqualTo(2.0);
			await Assert.That(image.NumPoints3D).IsEqualTo(1u);
		}
	}

	// C#-only (image_test.cc has no counterpart): C++ SetPoints2D takes the vector by value,
	// so two images set from the same list must not share point state.
	[Test]
	public async Task SetPoints2D_CopiesTheList()
	{
		var points2D = Enumerable.Range(0, 2).Select(_ => new Point2D()).ToList();
		var image1 = new Image();
		var image2 = new Image();
		image1.SetPoints2D(points2D);
		image2.SetPoints2D(points2D);
		image1.SetPoint3DForPoint2D(0, 5);
		using (Assert.Multiple())
		{
			await Assert.That(image1.Point2DAt(0).HasPoint3D).IsTrue();
			await Assert.That(image2.Point2DAt(0).HasPoint3D).IsFalse();
			await Assert.That(points2D[0].HasPoint3D).IsFalse();
		}
	}

	[Test]
	public async Task Image_Points3D()
	{
		var image = new Image();
		image.SetPoints2D(new Vector2d[2]);
		using (Assert.Multiple())
		{
			await CheckPoints3D(image, has0: false, has1: false, numPoints3D: 0);
			image.SetPoint3DForPoint2D(0, 0);
			await CheckPoints3D(image, has0: true, has1: false, numPoints3D: 1);
			await Assert.That(image.HasPoint3D(0)).IsTrue();
			image.SetPoint3DForPoint2D(0, 1);
			await CheckPoints3D(image, has0: true, has1: false, numPoints3D: 1);
			await Assert.That(image.HasPoint3D(0)).IsFalse();
			await Assert.That(image.HasPoint3D(1)).IsTrue();
			image.SetPoint3DForPoint2D(1, 0);
			await CheckPoints3D(image, has0: true, has1: true, numPoints3D: 2);
			await Assert.That(image.HasPoint3D(0)).IsTrue();
			await Assert.That(image.HasPoint3D(1)).IsTrue();
			image.ResetPoint3DForPoint2D(0);
			await CheckPoints3D(image, has0: false, has1: true, numPoints3D: 1);
			await Assert.That(image.HasPoint3D(0)).IsTrue();
			await Assert.That(image.HasPoint3D(1)).IsFalse();
			image.ResetPoint3DForPoint2D(1);
			await CheckPoints3D(image, has0: false, has1: false, numPoints3D: 0);
			await Assert.That(image.HasPoint3D(0)).IsFalse();
			await Assert.That(image.HasPoint3D(1)).IsFalse();
			image.ResetPoint3DForPoint2D(0);
			await CheckPoints3D(image, has0: false, has1: false, numPoints3D: 0);
			await Assert.That(image.HasPoint3D(0)).IsFalse();
			await Assert.That(image.HasPoint3D(1)).IsFalse();
		}
	}

	[Test]
	public async Task Image_ProjectionCenter()
	{
		Image image = CreatePosedImage(name: "");
		await Assert.That(image.ProjectionCenter()).IsEqualTo(Vector3d.Zero);
	}

	[Test]
	public async Task Image_ViewingDirection()
	{
		Image image = CreatePosedImage(name: "");
		await Assert.That(image.ViewingDirection()).IsEqualTo(new Vector3d(0, 0, 1));
	}

	[Test]
	public async Task Image_ProjectPoint()
	{
		Image image = CreatePosedImage(name: "");
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 1, 1);
		image.SetCameraId(camera.CameraId);
		image.SetCameraPtr(camera);
		Vector2d? result = image.ProjectPoint(new Vector3d(2, 0, 1));
		await Assert.That(result.HasValue).IsTrue();
		using (Assert.Multiple())
		{
			await Assert.That(result!.Value).IsEqualTo(new Vector2d(2.5, 0.5));
			await Assert.That(image.ProjectPoint(new Vector3d(2, 0, 0)).HasValue).IsFalse();
			await Assert.That(image.ProjectPoint(new Vector3d(2, 0, -1)).HasValue).IsFalse();
		}
	}

	// The rig and frame every pose test of image_test.cc builds: rig 1 with CAMERA 1 as
	// reference sensor, frame 1 of that rig, optionally posed at the identity. The rig is
	// set before the pose here; the two setters are independent.
	private static Frame CreateTrivialFrame(bool withPose)
	{
		var rig = new Rig { RigId = 1 };
		rig.AddRefSensor(new SensorId(SensorType.Camera, 1));
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(1);
		if (withPose)
		{
			frame.SetRigFromWorld(new Rigid3d());
		}

		frame.SetRigPtr(rig);
		return frame;
	}

	// Image 1 of camera 1 in a posed trivial frame, as ConstructCopy, AssignCopy,
	// ProjectionCenter, ViewingDirection and ProjectPoint set it up.
	private static Image CreatePosedImage(string name)
	{
		Frame frame = CreateTrivialFrame(withPose: true);
		var image = new Image { ImageId = 1 };
		image.SetCameraId(1);
		image.Name = name;
		frame.AddDataId(image.DataId);
		image.SetFrameId(1);
		image.SetFramePtr(frame);
		return image;
	}

	private static async Task CheckPoints3D(Image image, bool has0, bool has1, uint numPoints3D)
	{
		await Assert.That(image.Point2DAt(0).HasPoint3D).IsEqualTo(has0);
		await Assert.That(image.Point2DAt(1).HasPoint3D).IsEqualTo(has1);
		await Assert.That(image.NumPoints3D).IsEqualTo(numPoints3D);
	}
}
