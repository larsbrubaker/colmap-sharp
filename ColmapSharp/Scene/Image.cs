// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Image: port of colmap/scene/image.h and image.cc, one camera exposure: its name, its
// camera (Scene/Camera.cs), its frame (Scene/Frame.cs, which carries the pose) and its 2D
// feature points (Scene/Point2D.cs) with their 3D point links. Tests:
// ColmapSharp.Tests/Scene/ImageTests.cs (image_test.cc 1:1).
//
// Design (later phases build on it):
// - Ownership. The Reconstruction (a later Phase 4 port) owns cameras, frames and images.
//   C++'s raw `Camera* camera_ptr_` and `Frame* frame_ptr_` become non-owning C#
//   references (CameraPtr, FramePtr) that the Reconstruction sets and resets under COLMAP's
//   rules. The image has no pose of its own: CamFromWorld is the frame's rig_from_world
//   composed with the rig's sensor_from_rig, so a pose edit (or Phase 8's in-place bundle
//   adjustment write) on the frame is seen by all its images, as in COLMAP.
// - Copying. C++ copy construction/assignment is Clone(): it copies the ids, name and
//   points and shares the camera and frame references, exactly as COLMAP's copy does.
// - Points. std::vector<Point2D> is a List; Point2DAt returns a ref into it (C++ returns
//   Point2D&), which stays valid until the list is resized. C++ names the accessor Point2D,
//   which C# cannot use for a method returning the Point2D type.

using System.Globalization;
using System.Runtime.InteropServices;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Image. An image is the product of one camera exposure at a certain
/// location (parameterized as the pose). An image may share a camera with multiple other
/// images, if its intrinsics are the same.
/// </summary>
public sealed class Image : IEquatable<Image>
{
	// All image points, including points that are not part of a 3D point track.
	private List<Point2D> _points2D = [];

	private uint _cameraId = InvalidCameraId;
	private uint _frameId = InvalidFrameId;
	private Camera? _cameraPtr;
	private Frame? _framePtr;

	/// <summary>The unique identifier of the image; <see cref="InvalidImageId"/> until set.</summary>
	public uint ImageId { get; set; } = InvalidImageId;

	/// <summary>The name of the image, i.e. the relative path.</summary>
	public string Name { get; set; } = "";

	/// <summary>
	/// The identifier of the camera; <see cref="InvalidCameraId"/> until set. Multiple
	/// images may share the same camera.
	/// </summary>
	public uint CameraId => _cameraId;

	/// <summary>Whether the camera id has been set.</summary>
	public bool HasCameraId => _cameraId != InvalidCameraId;

	/// <summary>The data id (CAMERA, camera_id) / image_id.</summary>
	public DataId DataId => new(new SensorId(SensorType.Camera, _cameraId), ImageId);

	/// <summary>
	/// The shared camera, typically only set when the image was added to a reconstruction;
	/// throws if unset.
	/// </summary>
	public Camera CameraPtr => Check.NotNull(_cameraPtr);

	/// <summary>Whether the camera reference is set.</summary>
	public bool HasCameraPtr => _cameraPtr is not null;

	/// <summary>The identifier of the frame; <see cref="InvalidFrameId"/> until set.</summary>
	public uint FrameId => _frameId;

	/// <summary>Whether the frame id has been set.</summary>
	public bool HasFrameId => _frameId != InvalidFrameId;

	/// <summary>The frame of the image; throws if unset.</summary>
	public Frame FramePtr => Check.NotNull(_framePtr);

	/// <summary>Whether the frame reference is set.</summary>
	public bool HasFramePtr => _framePtr is not null;

	/// <summary>Whether the image was captured by the reference sensor of its frame's rig.</summary>
	public bool IsRefInFrame => Check.NotNull(_framePtr).RigPtr.IsRefSensor(new SensorId(SensorType.Camera, _cameraId));

	/// <summary>Whether the image has a pose, i.e. a frame with a pose.</summary>
	public bool HasPose => _framePtr is not null && _framePtr.HasPose;

	/// <summary>The number of image points.</summary>
	public uint NumPoints2D => (uint)_points2D.Count;

	/// <summary>
	/// The number of triangulations, i.e. the number of points that are part of a 3D point
	/// track.
	/// </summary>
	public uint NumPoints3D { get; private set; }

	/// <summary>All image points (mutable, like C++'s non-const Points2D()).</summary>
	public List<Point2D> Points2D => _points2D;

	/// <summary>Sets the camera id; throws for an invalid id or when the camera reference is set.</summary>
	public void SetCameraId(uint cameraId)
	{
		Check.Ne(cameraId, InvalidCameraId);
		Check.That(!HasCameraPtr);
		_cameraId = cameraId;
	}

	/// <summary>
	/// Sets the shared camera. Without a camera set yet, its id must equal
	/// <see cref="CameraId"/>; replacing a set camera adopts the new camera's id.
	/// </summary>
	public void SetCameraPtr(Camera camera)
	{
		Check.NotNull(camera);
		Check.Ne(camera.CameraId, InvalidCameraId);
		if (!HasCameraPtr)
		{
			Check.Eq(camera.CameraId, _cameraId);
			_cameraPtr = camera;
		}
		else
		{
			_cameraId = camera.CameraId;
			_cameraPtr = camera;
		}
	}

	/// <summary>Clears the camera reference.</summary>
	public void ResetCameraPtr() => _cameraPtr = null;

	/// <summary>Sets the frame id; throws when the frame reference is set.</summary>
	public void SetFrameId(uint frameId)
	{
		Check.That(!HasFramePtr);
		_frameId = frameId;
	}

	/// <summary>
	/// Sets the frame. The frame must contain this image's data id; without a frame set yet,
	/// its id must equal <see cref="FrameId"/>, and replacing a set frame adopts its id.
	/// </summary>
	public void SetFramePtr(Frame frame)
	{
		Check.NotNull(frame);
		Check.Ne(frame.FrameId, InvalidFrameId);
		Check.That(frame.HasDataId(DataId), $"Image {ImageId} does not exist in frame {frame.FrameId}");
		if (!HasFramePtr)
		{
			Check.Eq(frame.FrameId, _frameId);
			_framePtr = frame;
		}
		else
		{
			_frameId = frame.FrameId;
			_framePtr = frame;
		}
	}

	/// <summary>Clears the frame reference.</summary>
	public void ResetFramePtr() => _framePtr = null;

	/// <summary>
	/// The composition of sensor_from_rig and rig_from_world; equal to rig_from_world for a
	/// trivial frame. Throws if the frame is unset or has no pose.
	/// </summary>
	public Rigid3d CamFromWorld() => Check.NotNull(_framePtr).SensorFromWorld(new SensorId(SensorType.Camera, _cameraId));

	/// <summary>A reference to one image point; throws for an index out of range (C++ at()).</summary>
	public ref Point2D Point2DAt(uint point2DIdx)
	{
		if (point2DIdx >= (uint)_points2D.Count)
		{
			throw new ArgumentOutOfRangeException(nameof(point2DIdx), "vector");
		}

		return ref CollectionsMarshal.AsSpan(_points2D)[(int)point2DIdx];
	}

	/// <summary>
	/// Sets the point coordinates, resizing the point list to match. Existing points keep
	/// their 3D point links and <see cref="NumPoints3D"/> is not recomputed, as in COLMAP;
	/// call it on an image without triangulations.
	/// </summary>
	public void SetPoints2D(IReadOnlyList<Vector2d> points)
	{
		if (points.Count < _points2D.Count)
		{
			_points2D.RemoveRange(points.Count, _points2D.Count - points.Count);
		}

		while (_points2D.Count < points.Count)
		{
			_points2D.Add(new Point2D());
		}

		Span<Point2D> points2D = CollectionsMarshal.AsSpan(_points2D);
		for (int point2DIdx = 0; point2DIdx < points.Count; ++point2DIdx)
		{
			points2D[point2DIdx].Xy = points[point2DIdx];
		}
	}

	/// <summary>
	/// Copies the given points (with their 3D point links) as the image's points and counts
	/// the triangulated ones. The image must not have points yet. C++ takes the vector by
	/// value, so the caller's list is copied, never shared.
	/// </summary>
	public void SetPoints2D(IEnumerable<Point2D> points)
	{
		Check.That(_points2D.Count == 0);
		_points2D = new List<Point2D>(points);
		NumPoints3D = 0;
		foreach (Point2D point2D in _points2D)
		{
			if (point2D.HasPoint3D)
			{
				NumPoints3D += 1;
			}
		}
	}

	/// <summary>Sets the point as triangulated, i.e. part of the 3D point track <paramref name="point3DId"/>.</summary>
	public void SetPoint3DForPoint2D(uint point2DIdx, ulong point3DId)
	{
		Check.Ne(point3DId, InvalidPoint3DId);
		ref Point2D point2D = ref Point2DAt(point2DIdx);
		if (!point2D.HasPoint3D)
		{
			NumPoints3D += 1;
		}

		point2D.Point3DId = point3DId;
	}

	/// <summary>Sets the point as not triangulated, i.e. not part of a 3D point track.</summary>
	public void ResetPoint3DForPoint2D(uint point2DIdx)
	{
		ref Point2D point2D = ref Point2DAt(point2DIdx);
		if (point2D.HasPoint3D)
		{
			point2D.Point3DId = InvalidPoint3DId;
			NumPoints3D -= 1;
		}
	}

	/// <summary>Whether one of the image points is part of the given 3D point track.</summary>
	public bool HasPoint3D(ulong point3DId)
	{
		foreach (Point2D point2D in _points2D)
		{
			if (point2D.Point3DId == point3DId)
			{
				return true;
			}
		}

		return false;
	}

	/// <summary>The projection center in world space.</summary>
	public Vector3d ProjectionCenter() => CamFromWorld().TgtOriginInSrc();

	/// <summary>The viewing direction of the image (third row of the rotation matrix).</summary>
	public Vector3d ViewingDirection() => CamFromWorld().Rotation.ToRotationMatrix().Row(2);

	/// <summary>
	/// Reprojects the 3D point onto the image in pixels (throws if the camera is not set).
	/// Null if the 3D point is behind the camera.
	/// </summary>
	public Vector2d? ProjectPoint(Vector3d point3D)
	{
		Check.That(HasCameraPtr);
		Vector3d point3DInCam = CamFromWorld() * point3D;
		return _cameraPtr!.ImgFromCam(point3DInCam);
	}

	/// <summary>
	/// A copy (C++ copy construction): same ids, name, points and triangulation count,
	/// sharing the camera and frame references.
	/// </summary>
	public Image Clone() => new()
	{
		ImageId = ImageId,
		Name = Name,
		_cameraId = _cameraId,
		_cameraPtr = _cameraPtr,
		_frameId = _frameId,
		_framePtr = _framePtr,
		NumPoints3D = NumPoints3D,
		_points2D = new List<Point2D>(_points2D),
	};

	/// <summary>
	/// The C++ operator==: same ids, name, triangulation count, points and pose presence,
	/// and, if posed, exactly equal rig_from_world of the frames.
	/// </summary>
	public bool Equals(Image? other)
	{
		if (other is null)
		{
			return false;
		}

		bool result = ImageId == other.ImageId
			&& _cameraId == other._cameraId
			&& _frameId == other._frameId
			&& Name == other.Name
			&& NumPoints3D == other.NumPoints3D
			&& HasPose == other.HasPose
			&& _points2D.SequenceEqual(other._points2D);
		if (!HasPose)
		{
			return result;
		}

		return result && _framePtr!.RigFromWorld() == other._framePtr!.RigFromWorld();
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => Equals(obj as Image);

	/// <summary>Hash of the image id (everything else may change in place).</summary>
	public override int GetHashCode() => ImageId.GetHashCode();

	/// <summary>The C++ operator==.</summary>
	public static bool operator ==(Image? a, Image? b) => a is null ? b is null : a.Equals(b);

	/// <summary>The C++ operator!=.</summary>
	public static bool operator !=(Image? a, Image? b) => !(a == b);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g. "Image(image_id=1, camera_id=2, frame_id=3,
	/// name="test", has_pose=0, triangulated=0/0)".
	/// </summary>
	public override string ToString()
	{
		string imageIdStr = ImageId != InvalidImageId ? ImageId.ToString(CultureInfo.InvariantCulture) : "Invalid";
		string cameraIdStr = HasCameraId ? CameraId.ToString(CultureInfo.InvariantCulture) : "Invalid";
		string frameIdStr = HasFrameId ? FrameId.ToString(CultureInfo.InvariantCulture) : "Invalid";
		return string.Create(
			CultureInfo.InvariantCulture,
			$"Image(image_id={imageIdStr}, camera_id={cameraIdStr}, frame_id={frameIdStr}, name=\"{Name}\", has_pose={(HasPose ? 1 : 0)}, triangulated={NumPoints3D}/{NumPoints2D})");
	}
}
