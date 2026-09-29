// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticObjectScene: the synthetic object scenes of the reconstruction benchmark
// (docs/QUALITY_PLAN.md, stage 0a). Not a COLMAP port; COLMAP has nothing like it. It renders a
// single object turning in front of a plain grey wall, seen by a fixed camera, and returns
// everything a benchmark needs to score a reconstruction of it: the frames, the true cameras,
// the true mesh and a true foreground mask per frame. It lives in the library (like
// PatchMatchSyntheticScene.cs) so the test suite and a benchmark runner share one copy.
//
// The scenes (SyntheticObjectKind):
// - DarkObject: the stand-in for the driving real capture, a black matte computer mouse
//   spinning on its cable. A superellipsoid of mouse proportions with albedo ~0.06 and faint
//   smooth noise, darker seam grooves and a textured label on the underside, plus a weak
//   specular highlight. Few features, most of them on the label.
// - TexturedSphere: a sphere covered in solid value noise, the sanity baseline.
// - TexturelessBox: a uniform matte box, which SfM should find hard.
//
// The rig, shared by all three: the object turns about a near-vertical axis through its center
// on a twisting pendulum, theta(t) = A sin(w t) e^(-lambda t) + drift t over t in [0, 1], so its
// speed varies and it reverses direction. The camera is fixed, slightly below the object and
// looking up at it (so the underside shows), with small seeded jitter per frame (hand shake).
// A fixed light gives Lambertian shading plus ambient; the wall is unlit (its value is a faint
// gradient around 0.5) and nothing casts shadows. Frames are 3x3 supersampled, get Gaussian
// sensor noise per channel, and are quantized to 8 bits.
//
// Frames: the object moves and the camera is still, but SfM recovers the equivalent rigid
// scene - the object fixed and the camera moving around it. So the world frame here is the
// object's own frame, and CamFromWorld[k] = cam_from_lab[k] * lab_from_object[k]. The mesh is in
// that world frame and never moves. The wall is not part of the world: it is fixed to the lab,
// so in the world frame it moves with the camera, which is the trap a real turntable capture
// sets (background features agree with no rigid scene). Frames are rendered from exactly the
// returned poses, so the truth is the truth of the pixels.
//
// Determinism: the same (kind, frame count, size, seed) gives the same pixels. The seed drives
// the camera jitter and the sensor noise; the object's shape, texture and motion are fixed. The
// noise is an Irwin-Hall sum of mt19937 draws (no transcendental functions), and frames render
// in parallel, each into its own slot from its own generator, so thread count does not matter.
// Across platforms the poses go through Math.Sin/Cos/Pow, which .NET does not promise to round
// identically everywhere, so cross-platform hashes may differ in the last bits.
//
// Rendering (a z-buffer rasterizer with perspective-correct interpolation) is in
// SyntheticObjectScene.Render.cs; the meshes and surface albedo are in
// SyntheticObjectScene.Shapes.cs; the older sphere-on-textured-wall views that
// AutomaticReconstructionTests uses, and the shared value noise, are in
// SyntheticObjectScene.SphereOnWall.cs.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Testing;

/// <summary>Which object <see cref="SyntheticObjectScene.Generate"/> renders.</summary>
public enum SyntheticObjectKind
{
	/// <summary>A dark, nearly textureless mouse-shaped superellipsoid with seams and an underside label.</summary>
	DarkObject,

	/// <summary>A sphere covered in solid value noise (the well-textured baseline).</summary>
	TexturedSphere,

	/// <summary>A uniform matte box.</summary>
	TexturelessBox,
}

/// <summary>
/// A rendered synthetic object scene with its ground truth: frames, one shared pinhole camera,
/// per-frame poses (world = the object's frame), the object mesh in world coordinates and a
/// foreground mask per frame. See the file header for the rig and conventions.
/// </summary>
public sealed partial class SyntheticObjectScene
{
	// Twisting pendulum, theta(t) = A sin(w t) e^(-lambda t) + drift t, t in [0, 1].
	private const double PendulumAmplitude = 150 * Math.PI / 180;
	private const double PendulumOmega = 2 * Math.PI * 1.5;
	private const double PendulumDecay = 0.8;
	private const double PendulumDrift = 60 * Math.PI / 180;

	// The turning axis leans this far from lab vertical, and the object hangs nose-down by
	// RestTilt (about lab z, its nose being +x), so the underside faces the camera whenever the
	// nose turns away.
	private const double AxisTilt = 6 * Math.PI / 180;
	private const double RestTilt = 35 * Math.PI / 180;

	// Camera: this far from the object's center, this far below it, focal length this multiple
	// of the larger image side (the same framing as RenderTexturedSphereOnWall).
	private const double CameraDistance = 2.8;
	private const double CameraElevation = -25 * Math.PI / 180;
	private const double FocalScale = 1.2;

	// Per-frame jitter: translation up to this many units per axis, rotation up to this many
	// radians per axis, both uniform.
	internal const double TranslationJitter = 0.01;
	internal const double RotationJitter = 0.25 * Math.PI / 180;

	// Working memory (bytes) that frames rendering in parallel may use together.
	private const long ParallelMemoryBudget = 256L << 20;

	private SyntheticObjectScene(
		SyntheticObjectKind kind,
		Camera camera,
		Bitmap[] frames,
		Rigid3d[] camFromWorld,
		Bitmap[] masks,
		Vector3d[] meshVertices,
		PlyMeshFace[] meshTriangles)
	{
		Kind = kind;
		Camera = camera;
		Frames = frames;
		CamFromWorld = camFromWorld;
		Masks = masks;
		MeshVertices = meshVertices;
		MeshTriangles = meshTriangles;
	}

	/// <summary>The object that was rendered.</summary>
	public SyntheticObjectKind Kind { get; }

	/// <summary>The one camera every frame shares: PINHOLE, camera id 1, principal point at the image center.</summary>
	public Camera Camera { get; }

	/// <summary>The rendered frames, 8-bit RGB.</summary>
	public IReadOnlyList<Bitmap> Frames { get; }

	/// <summary>True pose of each frame: world (the object's frame) to camera.</summary>
	public IReadOnlyList<Rigid3d> CamFromWorld { get; }

	/// <summary>
	/// True foreground of each frame: 8-bit grey, 255 where the ray through the pixel center
	/// hits the object and 0 where it sees the wall.
	/// </summary>
	public IReadOnlyList<Bitmap> Masks { get; }

	/// <summary>The object's closed, outward-oriented triangle mesh, in world coordinates.</summary>
	public IReadOnlyList<Vector3d> MeshVertices { get; }

	/// <summary>The mesh's triangles, as indices into <see cref="MeshVertices"/>.</summary>
	public IReadOnlyList<PlyMeshFace> MeshTriangles { get; }

	/// <summary>
	/// Renders <paramref name="numFrames"/> frames of <paramref name="kind"/> at
	/// <paramref name="width"/> x <paramref name="height"/>. <paramref name="seed"/> drives the
	/// camera jitter and the sensor noise. <paramref name="sensorNoiseSigma"/> is the noise's
	/// standard deviation in 8-bit levels.
	/// </summary>
	public static SyntheticObjectScene Generate(
		SyntheticObjectKind kind,
		int numFrames,
		int width,
		int height,
		uint seed,
		double sensorNoiseSigma = 1.5,
		CancellationToken cancellationToken = default)
	{
		Check.That(numFrames > 0);
		Check.That(width > 0 && height > 0);
		Check.That(sensorNoiseSigma >= 0);

		ObjectShape shape = ObjectShape.Create(kind);
		double focal = FocalScale * Math.Max(width, height);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, focal, width, height);

		var jitter = new Mt19937(seed);
		var camFromLab = new Rigid3d[numFrames];
		var camFromWorld = new Rigid3d[numFrames];
		for (int k = 0; k < numFrames; ++k)
		{
			camFromLab[k] = JitteredCamFromLab(jitter);
			camFromWorld[k] = camFromLab[k] * LabFromObject(FrameTime(k, numFrames));
		}

		var frames = new Bitmap[numFrames];
		var masks = new Bitmap[numFrames];
		var parallelOptions = new ParallelOptions
		{
			CancellationToken = cancellationToken,
			MaxDegreeOfParallelism = MaxParallelFrames(width, height),
		};
		Parallel.For(0, numFrames, parallelOptions, k =>
		{
			var noise = new Mt19937(unchecked(seed * 2654435761u + (uint)k + 1u));
			(frames[k], masks[k]) = RenderFrame(shape, camera, camFromWorld[k], camFromLab[k], noise, sensorNoiseSigma);
		});

		return new SyntheticObjectScene(kind, camera, frames, camFromWorld, masks, shape.Vertices, shape.Triangles);
	}

	/// <summary>Time t in [0, 1] of frame <paramref name="k"/> of <paramref name="numFrames"/>.</summary>
	internal static double FrameTime(int k, int numFrames) => numFrames == 1 ? 0 : k / (double)(numFrames - 1);

	// How many frames render at once: each holds a z-buffer band and its outputs, and all of
	// them together stay within ParallelMemoryBudget (at least one frame, at most one per core).
	internal static int MaxParallelFrames(int width, int height)
	{
		long perFrame = (long)BandRows * Supersampling * width * Supersampling * ZBufferBytesPerSample
			+ 4L * width * height;
		return (int)Math.Clamp(ParallelMemoryBudget / perFrame, 1, Environment.ProcessorCount);
	}

	/// <summary>The pendulum angle theta(t) in radians, t in [0, 1] over the capture.</summary>
	public static double PendulumAngle(double t) =>
		PendulumAmplitude * Math.Sin(PendulumOmega * t) * Math.Exp(-PendulumDecay * t) + PendulumDrift * t;

	// The object's pose in the lab at time t: its rest tilt, then the pendulum turn about the
	// leaning axis. The object turns about its own center, so there is no translation.
	internal static Rigid3d LabFromObject(double t)
	{
		var axis = new Vector3d(Math.Sin(AxisTilt), Math.Cos(AxisTilt), 0);
		Quaterniond turn = Quaterniond.FromAngleAxis(new AngleAxisd(PendulumAngle(t), axis));
		Quaterniond rest = Quaterniond.FromAngleAxis(new AngleAxisd(-RestTilt, Vector3d.UnitZ));
		return new Rigid3d((turn * rest).Normalized(), Vector3d.Zero);
	}

	// The lab has y up; the camera sits in front (-z) and below, looking at the origin, with
	// COLMAP's camera axes (x right, y down, z forward). Jitter moves the center and turns the
	// camera a little, drawn from the rig's generator in a fixed order.
	private static Rigid3d JitteredCamFromLab(Mt19937 jitter)
	{
		var center = new Vector3d(
			TranslationJitter * Uniform(jitter),
			CameraDistance * Math.Sin(CameraElevation) + TranslationJitter * Uniform(jitter),
			-CameraDistance * Math.Cos(CameraElevation) + TranslationJitter * Uniform(jitter));
		Vector3d forward = (-center).Normalized();
		Vector3d right = forward.Cross(Vector3d.UnitY).Normalized();
		Vector3d down = forward.Cross(right);
		Matrix3d lookAt = Matrix3d.FromRows(right, down, forward);

		var shake = new Vector3d(
			RotationJitter * Uniform(jitter), RotationJitter * Uniform(jitter), RotationJitter * Uniform(jitter));
		Quaterniond rotation = Quaterniond.FromAngleAxis(new AngleAxisd(shake.Norm, shake.Norm > 0 ? shake / shake.Norm : Vector3d.UnitX))
			* Quaterniond.FromRotationMatrix(lookAt);
		rotation = rotation.Normalized();
		return new Rigid3d(rotation, -(rotation * center));
	}

	// Uniform in [-1, 1] from one mt19937 draw.
	private static double Uniform(Mt19937 random) => random.Next() / (double)uint.MaxValue * 2 - 1;

	// Approximately standard normal: the Irwin-Hall sum of twelve uniforms in [0, 1), minus 6.
	// Only additions, so it is the same on every platform.
	private static double Gaussian(Mt19937 random)
	{
		double sum = 0;
		for (int i = 0; i < 12; ++i)
		{
			sum += random.Next() / 4294967296.0;
		}

		return sum - 6;
	}
}
