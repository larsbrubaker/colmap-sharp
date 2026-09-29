// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DemoSettings: what the settings panel (ColmapDemoApp.Settings.cs) edits - every choice a user or
// tester can make about the next run, with the demo's defaults. ToSessionSettings turns it into the
// SessionSettings (SessionSettings.cs) one ReconstructionSession runs with; FramesPerVideo is read by
// the video intake instead. Summary gives the panel's read-only "This run" rows. No widgets here,
// so tests can check the mapping without a window.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using ColmapSharp.Compute;
using ColmapSharp.Controllers;

namespace ColmapDemo
{
	/// <summary>The unit a known focal length is typed in.</summary>
	public enum FocalUnit
	{
		/// <summary>Millimetres, 35 mm equivalent (what phones list).</summary>
		Millimetres35,

		/// <summary>Pixels of the photos as the run sees them.</summary>
		Pixels,
	}

	/// <summary>The settings panel's model: every option of the next run, with the demo's defaults.</summary>
	public sealed class DemoSettings
	{
		/// <summary>The focal length offered first: a typical phone main camera, 35 mm equivalent.</summary>
		public const double TypicalPhoneFocal35mm = 26;

		public AutomaticReconstructionOptions.SubjectType Subject { get; set; } = AutomaticReconstructionOptions.SubjectType.Scene;

		/// <summary>Set when a dropped video picked <see cref="Subject"/>, for the panel's note; cleared by any subject choice.</summary>
		public bool SubjectPickedForVideo { get; set; }

		/// <summary>
		/// How photos are matched (Advanced). A video's frames keep Individual (exhaustive) matching and
		/// are marked time-ordered instead; Video (recording order) marks any photos time-ordered.
		/// </summary>
		public AutomaticReconstructionOptions.DataType Matching { get; set; } = AutomaticReconstructionOptions.DataType.Individual;

		public AutomaticReconstructionOptions.QualityLevel Quality { get; set; } = new SessionSettings().Quality;

		public int MaxImageSize { get; set; } = new SessionSettings().MaxImageSize;

		/// <summary>All photos share one camera. A run of one video's frames alone always does.</summary>
		public bool SameCamera { get; set; }

		/// <summary>
		/// Whether the user gives the focal length. Turning it on moves the mapper off Global, which
		/// always re-estimates the focal (the library refuses the pair); the panel then disables Global.
		/// </summary>
		public bool KnowFocalLength
		{
			get => this.knowFocalLength;
			set
			{
				this.knowFocalLength = value;
				if (value && this.Mapper == AutomaticReconstructionOptions.MapperType.Global)
				{
					this.Mapper = AutomaticReconstructionOptions.MapperType.Incremental;
				}
			}
		}

		private bool knowFocalLength;

		public double FocalLength { get; set; } = TypicalPhoneFocal35mm;

		public FocalUnit FocalUnit { get; set; } = FocalUnit.Millimetres35;

		public string CameraModel { get; set; } = new SessionSettings().CameraModel;

		/// <summary>How many frames each added video is cut into.</summary>
		public int FramesPerVideo { get; set; } = VideoFrameSampler.DefaultTargetFrames;

		/// <summary>Follow points through video frames; applies only to time-ordered frames (<see cref="TimeOrdered"/>).</summary>
		public bool VideoTracking { get; set; }

		/// <summary>
		/// Place frames matching missed from their outlines; needs object mode and time-ordered frames.
		/// Experimental and off by default: it can misplace frames of round or symmetric objects.
		/// </summary>
		public bool SilhouettePlacement { get; set; }

		/// <summary>Full mesh (true) or the quick camera check, sparse points only (false).</summary>
		public bool Dense { get; set; } = true;

		public bool Texture { get; set; } = true;

		public AutomaticReconstructionOptions.MesherType Mesher { get; set; } = AutomaticReconstructionOptions.MesherType.Poisson;

		public int PoissonDepth { get; set; } = new SessionSettings().PoissonDepth;

		public double PoissonTrim { get; set; } = new SessionSettings().PoissonTrim;

		public AutomaticReconstructionOptions.MapperType Mapper { get; set; } = AutomaticReconstructionOptions.MapperType.Incremental;

		/// <summary>Whether PatchMatch uses the GPU when there is one; off runs it on the CPU, for comparison.</summary>
		public bool UseGpu { get; set; } = true;

		public int RandomSeed { get; set; } = -1;

		public bool KeepWorkspace { get; set; } = new SessionSettings().KeepWorkspace;

		/// <summary>Whether the subject is one object (outline placement needs it).</summary>
		public bool IsObject => this.Subject == AutomaticReconstructionOptions.SubjectType.Object;

		/// <summary>
		/// Whether a run's photos count as a video's frames in filming order: they are one video's
		/// frames alone (<paramref name="photosAreOneVideo"/>), or matching is in recording order.
		/// Tracking and outline placement apply only then.
		/// </summary>
		public bool TimeOrdered(bool photosAreOneVideo) => photosAreOneVideo || this.Matching == AutomaticReconstructionOptions.DataType.Video;

		/// <summary>A copy of every setting.</summary>
		public DemoSettings Clone() => (DemoSettings)this.MemberwiseClone();

		/// <summary>
		/// What a dropped video changes: a video is usually one object turned or walked round, so
		/// object mode. Its frames are marked time-ordered at run time (outline repair between
		/// neighbouring frames), so matching stays as it is.
		/// </summary>
		public void ApplyVideoDefaults()
		{
			this.Subject = AutomaticReconstructionOptions.SubjectType.Object;
			this.SubjectPickedForVideo = true;
		}

		/// <summary>
		/// The settings of one run. <paramref name="photosAreOneVideo"/> turns on one shared camera;
		/// <paramref name="gpu"/> is the host's device, used only when <see cref="UseGpu"/> is on.
		/// </summary>
		public SessionSettings ToSessionSettings(bool photosAreOneVideo, IComputeDevice gpu = null, Func<ValueTask> yieldAsync = null)
		{
			bool knowFocal = this.KnowFocalLength && this.FocalLength > 0;
			bool timeOrdered = this.TimeOrdered(photosAreOneVideo);
			return new SessionSettings
			{
				Subject = this.Subject,
				Data = this.Matching,
				Quality = this.Quality,
				MaxImageSize = this.MaxImageSize,
				SingleCamera = this.SameCamera || photosAreOneVideo || knowFocal,
				FramesAreTimeOrdered = photosAreOneVideo,
				CameraModel = this.CameraModel,
				KnownFocal35mm = knowFocal && this.FocalUnit == FocalUnit.Millimetres35 ? this.FocalLength : 0,
				KnownFocalPixels = knowFocal && this.FocalUnit == FocalUnit.Pixels ? this.FocalLength : 0,
				VideoTracking = this.VideoTracking && timeOrdered,
				SilhouettePlacement = this.SilhouettePlacement && this.IsObject && timeOrdered,
				Dense = this.Dense,
				Texture = this.Dense && this.Texture,
				Mesher = this.Mesher,
				PoissonDepth = this.PoissonDepth,
				PoissonTrim = this.PoissonTrim,
				Mapper = this.Mapper,
				RandomSeed = this.RandomSeed,
				KeepWorkspace = this.KeepWorkspace,
				ComputeDevice = this.UseGpu ? gpu : null,
				YieldAsync = yieldAsync,
			};
		}

		/// <summary>
		/// The "This run" rows the panel shows while a run goes, in plain words, from what the run
		/// actually uses (<paramref name="run"/>, from <see cref="ToSessionSettings"/>).
		/// </summary>
		public IReadOnlyList<(string Label, string Value)> Summary(SessionSettings run)
		{
			var c = CultureInfo.InvariantCulture;
			string focal = run.KnownFocal35mm > 0 ? string.Format(c, "{0} mm fixed", run.KnownFocal35mm)
				: run.KnownFocalPixels > 0 ? string.Format(c, "{0} px fixed", run.KnownFocalPixels)
				: "focal length estimated";
			string camera = (run.SingleCamera ? "Same camera" : "Camera per photo") + ", " + focal;
			string video = string.Format(c, "{0} frames", this.FramesPerVideo)
				+ (run.TimeOrdered ? ", in filming order" : string.Empty)
				+ (run.VideoTracking ? ", follow points" : string.Empty)
				+ (run.SilhouettePlacement ? ", place from outlines" : string.Empty);
			return new[]
			{
				("Capturing", run.Subject == AutomaticReconstructionOptions.SubjectType.Object ? "One object" : "A scene or room"),
				("Quality", QualityLabel(run.Quality) + string.Format(c, ", photos up to {0} px", run.MaxImageSize)),
				("Camera", camera),
				("Video", video),
				("Result", run.Dense ? (run.Texture ? "Full mesh, painted" : "Full mesh, unpainted") : "Quick camera check"),
				("Advanced", string.Format(c, "{0}, depth {1}, trim {2}, {3}, {4}, seed {5}, {6}", run.Mesher, run.PoissonDepth, run.PoissonTrim, run.Mapper, run.Data, run.RandomSeed, run.ComputeDevice != null ? "GPU" : "CPU")),
			};
		}

		/// <summary>The panel's name for a quality preset (Fast / Medium / High / Best).</summary>
		public static string QualityLabel(AutomaticReconstructionOptions.QualityLevel quality) => quality switch
		{
			AutomaticReconstructionOptions.QualityLevel.Low => "Fast",
			AutomaticReconstructionOptions.QualityLevel.Medium => "Medium",
			AutomaticReconstructionOptions.QualityLevel.High => "High",
			_ => "Best",
		};
	}
}
