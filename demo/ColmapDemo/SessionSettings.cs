// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SessionSettings: everything one ReconstructionSession (ReconstructionSession.cs) runs with, and
// ApplyTo, the one place those settings become the library's AutomaticReconstructionOptions. The
// settings panel (ColmapDemoApp.Settings.cs) edits a DemoSettings (DemoSettings.cs), which makes one
// of these per run; Describe is the line the run log starts with, so a tester's log says what ran.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using ColmapSharp.Scene;

namespace ColmapDemo
{
	/// <summary>The demo's run settings. Defaults are chosen for a quick run on a small photo set.</summary>
	public sealed class SessionSettings
	{
		/// <summary>
		/// Photos are shrunk so their longer side is at most this many pixels before anything
		/// else sees them. The Low preset already caps SIFT (3200 * 0.3125 = 1000) and PatchMatch
		/// and fusion (1000), so shrinking up front mostly saves decode memory and upload time.
		/// </summary>
		public int MaxImageSize { get; set; } = 1000;

		/// <summary>The library's quality preset; Low keeps a demo run in the tens of seconds.</summary>
		public AutomaticReconstructionOptions.QualityLevel Quality { get; set; } = AutomaticReconstructionOptions.QualityLevel.Low;

		/// <summary>
		/// Poisson trim. COLMAP's 10 cuts the sparse fused cloud of a handful of photos down to
		/// nothing (PORTING_PLAN.md Phase 11); 5 keeps the surface near the samples while still
		/// dropping most of the far-flung "balloon" Poisson closes the surface with.
		/// </summary>
		public double PoissonTrim { get; set; } = 5;

		/// <summary>
		/// Poisson octree depth (COLMAP's default is 13). Measured on 6 synthetic 320x240 views
		/// (8104 fused points, trim 5): depth 13 meshed in 48.8 s, depth 11 in 10.9 s with the very
		/// same mesh (60805 faces; the octree adapts to the sampling, so the extra levels were
		/// never used), depth 10 in 7.7 s with 60697 faces. On all three, 99.5% of the fused points
		/// lie within 1% of the model's size from a mesh vertex. 11 keeps the full result on this set
		/// and leaves headroom for the denser clouds of 1000 px photos, at a quarter of the time.
		/// </summary>
		public int PoissonDepth { get; set; } = 11;

		/// <summary>
		/// Whether every photo shares one camera (COLMAP's single_camera): right for frames of one
		/// video, which all come from the same lens at the same zoom, and it gives the mapper one
		/// set of intrinsics to refine from all of them instead of one guess per frame.
		/// </summary>
		public bool SingleCamera { get; set; }

		/// <summary>What is being captured: a scene (COLMAP's pipeline) or one object on a plain background.</summary>
		public AutomaticReconstructionOptions.SubjectType Subject { get; set; } = AutomaticReconstructionOptions.SubjectType.Scene;

		/// <summary>
		/// How the photos are matched. Individual (every photo with every other) is the default for
		/// video frames too: on an orbit clip, sequential matching (Video) without loop detection
		/// placed fewer frames (16 vs 24 of 40 on a phone clip), since the last frames never get
		/// matched back to the first. Video is what turns on the library's temporal outline repair
		/// in object mode and its point tracking.
		/// </summary>
		public AutomaticReconstructionOptions.DataType Data { get; set; } = AutomaticReconstructionOptions.DataType.Individual;

		/// <summary>
		/// Whether the photos, in name order, are one video's frames in filming order (the library's
		/// FramesAreTimeOrdered; false leaves it to Data). It turns on object mode's temporal outline
		/// repair, and tracking and outline placement need it; matching stays exhaustive.
		/// </summary>
		public bool FramesAreTimeOrdered { get; set; }

		/// <summary>Whether the frames count as time-ordered for the library: set, or matched in recording order.</summary>
		public bool TimeOrdered => this.FramesAreTimeOrdered || this.Data == AutomaticReconstructionOptions.DataType.Video;

		/// <summary>Follow points through video frames (the library's VideoTracking; needs <see cref="TimeOrdered"/>).</summary>
		public bool VideoTracking { get; set; }

		/// <summary>
		/// Place frames feature matching missed from their outlines (the library's
		/// SilhouettePlacement; the library refuses it without object mode and <see cref="TimeOrdered"/>).
		/// </summary>
		public bool SilhouettePlacement { get; set; }

		/// <summary>The camera model every photo's camera gets.</summary>
		public string CameraModel { get; set; } = "SIMPLE_RADIAL";

		/// <summary>
		/// The camera's focal length as a 35 mm-equivalent in millimetres (what phone specs and
		/// EXIF's FocalLengthIn35mmFilm give), or 0 when unknown. When set (this or
		/// <see cref="KnownFocalPixels"/>), all photos share one camera that starts at that focal and
		/// bundle adjustment keeps the focal fixed (distortion is still refined: knowing the focal
		/// says nothing about the lens's distortion).
		/// that focal and bundle adjustment keeps it (and the distortion) fixed.
		/// </summary>
		public double KnownFocal35mm { get; set; }

		/// <summary>
		/// The camera's focal length in pixels of the photos as the run sees them (after shrinking to
		/// <see cref="MaxImageSize"/>), or 0 when unknown; used when <see cref="KnownFocal35mm"/> is 0.
		/// </summary>
		public double KnownFocalPixels { get; set; }

		/// <summary>Whether to compute depth maps and a surface; off stops at the sparse points.</summary>
		public bool Dense { get; set; } = true;

		/// <summary>The meshing algorithm.</summary>
		public AutomaticReconstructionOptions.MesherType Mesher { get; set; } = AutomaticReconstructionOptions.MesherType.Poisson;

		/// <summary>Whether to paint the photos onto the mesh.</summary>
		public bool Texture { get; set; } = true;

		/// <summary>The sparse mapping algorithm.</summary>
		public AutomaticReconstructionOptions.MapperType Mapper { get; set; } = AutomaticReconstructionOptions.MapperType.Incremental;

		/// <summary>The random seed of every stage, or -1 for the library's default.</summary>
		public int RandomSeed { get; set; } = -1;

		/// <summary>Where each run's temp workspace folder is made.</summary>
		public string WorkspaceRoot { get; set; } = Path.Combine(Path.GetTempPath(), "ColmapDemo");

		/// <summary>
		/// Keep the run's workspace (depth maps, fused.ply, meshes) instead of deleting it when the
		/// run ends: a developer option, on when COLMAP_DEMO_KEEP_WORKSPACE=1.
		/// </summary>
		public bool KeepWorkspace { get; set; } = Environment.GetEnvironmentVariable("COLMAP_DEMO_KEEP_WORKSPACE") == "1";

		/// <summary>The GPU for PatchMatch, or null to run it on the CPU.</summary>
		public IComputeDevice ComputeDevice { get; set; }

		/// <summary>
		/// What the run awaits between units of work (each decoded photo, each pipeline stage, each
		/// PatchMatch problem), or null for <see cref="Task.Yield"/>. A host whose run shares the UI
		/// thread (the browser) supplies one that really gives the page a turn to paint.
		/// </summary>
		public Func<ValueTask> YieldAsync { get; set; }

		/// <summary>
		/// The full parameter list of <paramref name="cameraModel"/> for a
		/// <paramref name="width"/> x <paramref name="height"/> photo whose 35 mm-equivalent focal
		/// length is <paramref name="focal35mm"/>: the focal in pixels by COLMAP's rule for EXIF's
		/// FocalLengthIn35mmFilm (Bitmap.Exif.cs: the CIPA diagonal ratio, a 35 mm frame being 43.27 mm
		/// across its diagonal), 		/// its longer side), the principal point at the centre and no distortion, as COLMAP's
		/// Camera::CreateFromModelName lays them out.
		/// </summary>
		public static string KnownFocalCameraParams(string cameraModel, double focal35mm, int width, int height) =>
			KnownFocalPixelsCameraParams(cameraModel, focal35mm / 43.27 * Math.Sqrt(((double)width * width) + ((double)height * height)), width, height);

		/// <summary><see cref="KnownFocalCameraParams"/> for a focal already in pixels.</summary>
		public static string KnownFocalPixelsCameraParams(string cameraModel, double focalPixels, int width, int height) =>
			// Round-trip precision: Camera.ParamsToString keeps COLMAP's 6 significant digits, which
			// would round the focal the user gave.
			string.Join(", ", Camera.CreateFromModelName(1, cameraModel, focalPixels, width, height).Params
				.Select(v => v.ToString("R", CultureInfo.InvariantCulture)));

		/// <summary>What the run says, before any stage, when a known focal meets photos of different sizes.</summary>
		public const string MixedSizesKnownFocalMessage = "Photos of different sizes can't share one fixed focal length \u2014 untick \u2018I know this camera's focal length\u2019 or use photos of one size.";

		/// <summary>
		/// <see cref="MixedSizesKnownFocalMessage"/> when <paramref name="sizes"/> (each photo's width
		/// and height after shrinking) are not all one size, else null. The library takes one
		/// CameraParams string for every camera, and its principal point is a photo's centre, so a
		/// known focal holds for one photo size only.
		/// </summary>
		public static string KnownFocalSizeError(IEnumerable<(int Width, int Height)> sizes) =>
			sizes.Distinct().Count() > 1 ? MixedSizesKnownFocalMessage : null;

		/// <summary>Whether the run is told the focal length (and so keeps it fixed).</summary>
		public bool HasKnownFocal => this.KnownFocal35mm > 0 || this.KnownFocalPixels > 0;

		/// <summary>
		/// Sets everything the user controls on <paramref name="options"/>. The known focal needs
		/// the photos' size, so it is applied only when <paramref name="photoWidth"/> is known (the
		/// first photo's size; a known camera means one camera, whose photos share a size).
		/// </summary>
		public void ApplyTo(AutomaticReconstructionOptions options, int photoWidth = 0, int photoHeight = 0)
		{
			options.Subject = this.Subject;
			options.Data = this.Data;
			options.SingleCamera = this.SingleCamera || this.HasKnownFocal;
			options.FramesAreTimeOrdered = this.FramesAreTimeOrdered ? true : null;
			options.Quality = this.Quality;
			options.CameraModel = this.CameraModel;
			options.Dense = this.Dense;
			options.Mesher = this.Mesher;
			options.Texture = this.Texture;
			options.Mapper = this.Mapper;
			options.RandomSeed = this.RandomSeed;
			options.VideoTracking = this.VideoTracking;
			options.SilhouettePlacement = this.SilhouettePlacement;
			options.PoissonMeshing.Trim = this.PoissonTrim;
			options.PoissonMeshing.Depth = this.PoissonDepth;
			if (this.HasKnownFocal && photoWidth > 0 && photoHeight > 0)
			{
				options.CameraParams = this.KnownFocal35mm > 0
					? KnownFocalCameraParams(this.CameraModel, this.KnownFocal35mm, photoWidth, photoHeight)
					: KnownFocalPixelsCameraParams(this.CameraModel, this.KnownFocalPixels, photoWidth, photoHeight);
				options.BaRefineFocalLength = false;
			}
		}

		/// <summary>One line naming every setting the run uses, for the log.</summary>
		public string Describe()
		{
			var c = CultureInfo.InvariantCulture;
			return string.Join(", ", new[]
			{
				$"subject={this.Subject}",
				$"matching={this.Data}",
				$"quality={this.Quality}",
				$"max image size={this.MaxImageSize}",
				$"single camera={this.SingleCamera || this.HasKnownFocal}",
				$"frames time-ordered={this.TimeOrdered}",
				$"camera model={this.CameraModel}",
				this.KnownFocal35mm > 0 ? string.Format(c, "known focal={0} mm (35 mm equiv.), focal fixed", this.KnownFocal35mm)
					: this.KnownFocalPixels > 0 ? string.Format(c, "known focal={0} px, focal fixed", this.KnownFocalPixels) : "focal=estimated",
				$"video tracking={this.VideoTracking}",
				$"silhouette placement={this.SilhouettePlacement}",
				$"dense={this.Dense}",
				$"mesher={this.Mesher}",
				string.Format(c, "poisson depth={0} trim={1}", this.PoissonDepth, this.PoissonTrim),
				$"texture={this.Texture}",
				$"mapper={this.Mapper}",
				$"seed={this.RandomSeed}",
				$"gpu={(this.ComputeDevice != null ? "on" : "off")}",
				$"keep workspace={this.KeepWorkspace}",
			});
		}
	}
}
