// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePoseRegistrationTests: C#-only tests of silhouette pose registration
// (docs/QUALITY_PLAN.md, stages 4a and 4b; ColmapSharp/Sfm/Silhouette). Not a port; COLMAP has
// no silhouette registration. All run on the benchmark's rendered DarkObject
// (Mvs/Testing/SyntheticObjectScene, 480x360, 40 frames, motionDuration 0.27) with its true
// masks: some frames are registered with their true poses, the rest are placed from their
// silhouettes and compared with the truth. Tier C (outcome): the bars are pose error bounds.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm.Silhouette;
using TUnit.Core;

namespace ColmapSharp.Tests.Sfm.Silhouette;

public class SilhouettePoseRegistrationTests
{
	private static readonly Lazy<SyntheticObjectScene> Scene = new(() => SyntheticObjectScene.Generate(
		SyntheticObjectKind.DarkObject, 40, 480, 360, seed: 1, motionDuration: 0.27));

	[Test]
	public async Task CSharpOnly_InitializePoseInterpolatesAndExtrapolates()
	{
		Rigid3d a = At(0, new Vector3d(0, 0, 0)), b = At(0.4, new Vector3d(2, 0, 0));
		Rigid3d? mid = SilhouettePoseRegistration.InitializePose([a, null, null, b], 1, 2.0);
		await Assert.That(mid!.Value.Rotation.AngularDistance(At(0.4 / 3, default).Rotation)).IsLessThan(1e-12);
		await Assert.That((mid.Value.TgtOriginInSrc() - new Vector3d(2.0 / 3, 0, 0)).Norm).IsLessThan(1e-12);

		// Only earlier frames: the motion continues, capped at twice the spacing.
		Rigid3d? ahead = SilhouettePoseRegistration.InitializePose([a, b, null, null, null, null], 5, 2.0);
		await Assert.That(ahead!.Value.Rotation.AngularDistance(At(1.2, default).Rotation)).IsLessThan(1e-12);
		await Assert.That((ahead.Value.TgtOriginInSrc() - new Vector3d(6, 0, 0)).Norm).IsLessThan(1e-12);
		await Assert.That(SilhouettePoseRegistration.InitializePose([null, null], 0, 2.0)).IsNull();
	}

	[Test]
	public async Task CSharpOnly_EveryOtherFramePlacedFromSilhouettes()
	{
		SyntheticObjectScene scene = Scene.Value;
		Reconstruction reconstruction = Registered(scene, k => k % 2 == 0);
		SilhouettePoseResult result = SilhouettePoseRegistration.Register(reconstruction, Frames(scene, null), Options(scene));
		(double rotation, double position) = Errors(scene, reconstruction, result, "every other");

		await Assert.That(result.PlacedCount).IsEqualTo(20);
		await Assert.That(rotation).IsLessThan(2.0);
		await Assert.That(position).IsLessThan(2.0);
	}

	[Test]
	public async Task CSharpOnly_TwentyFrameGapPlacedFromSilhouettes()
	{
		SyntheticObjectScene scene = Scene.Value;
		var medians = new List<(double Rotation, double Position, int Placed)>();
		foreach ((int passes, bool turntable) in new[] { (1, false), (3, false), (3, true) })
		{
			Reconstruction reconstruction = Registered(scene, k => k < 10 || k >= 30);
			SilhouettePoseOptions options = Options(scene);
			options.MaxPasses = passes;
			options.Turntable = turntable;
			SilhouettePoseResult result = SilhouettePoseRegistration.Register(reconstruction, Frames(scene, null), options);
			if (result.Turntable is TurntableFit fit)
			{
				Console.WriteLine(
					$"turntable fit: valid {fit.Valid} {fit.Reason} radial {fit.RadialResidual:F4} planar {fit.PlanarResidual:F4} arc {fit.ArcDegrees:F1}");
			}

			string label = turntable ? "gap, turntable" : $"gap, {passes} pass(es)";
			(double rotation, double position) = Errors(scene, reconstruction, result, label);
			medians.Add((rotation, position, result.PlacedCount));
		}

		// No accuracy target was set for the gap; these pin the measured behaviour so a change
		// shows. The hull from frames 0-9 and 30-39 is loose over the gap's viewpoints (its IoU
		// at the true gap poses is 0.97-0.98), so a wrong pose can match the silhouette better
		// than the truth: silhouettes place every gap frame, but only to several degrees.
		await Assert.That(medians[1].Placed).IsGreaterThanOrEqualTo(medians[0].Placed);
		await Assert.That(medians[1].Placed).IsEqualTo(20);
		await Assert.That(medians[1].Rotation).IsLessThan(8.0);

		// The turntable search must beat the interpolated starts.
		await Assert.That(medians[2].Placed).IsGreaterThanOrEqualTo(medians[1].Placed);
		// Measured: 20/20, median 4.08 degrees / 14.0% against 5.49 / 17.5% interpolated. Frames
		// 12-17 land within 1.3 degrees; the middle (20-26) still sits 5.5-7.9 degrees off, where the
		// hull is loosest.
		await Assert.That(medians[2].Rotation).IsLessThan(medians[1].Rotation);
		await Assert.That(medians[2].Placed).IsEqualTo(20);
		await Assert.That(medians[2].Rotation).IsLessThan(4.5);
	}

	[Test]
	public async Task CSharpOnly_TurntableFitAcceptsPendulumAndRejectsStraightLine()
	{
		// The DarkObject turns about a fixed pivot, so its object-fixed camera centres lie on a
		// circle; a camera sliding along a line past the object is not a turntable.
		SyntheticObjectScene scene = Scene.Value;
		TurntableFit pendulum = SilhouetteTurntable.Fit(scene.CamFromWorld);
		Console.WriteLine(
			$"pendulum fit: valid {pendulum.Valid} {pendulum.Reason} radial {pendulum.RadialResidual:F4} planar {pendulum.PlanarResidual:F4} arc {pendulum.ArcDegrees:F1}");
		await Assert.That(pendulum.Valid).IsTrue();

		var line = new List<Rigid3d>();
		for (int k = 0; k < 10; k++)
		{
			line.Add(new Rigid3d(Quaterniond.Identity, new Vector3d(-(k * 0.3 - 1.5), 0.01 * (k % 3), 5)));
		}

		TurntableFit sliding = SilhouetteTurntable.Fit(line);
		Console.WriteLine($"line fit: valid {sliding.Valid} {sliding.Reason} radius {sliding.Radius:G4}");
		await Assert.That(sliding.Valid).IsFalse();
	}

	[Test]
	public async Task CSharpOnly_RefinementFromTruthStopsEarly()
	{
		// A pose that starts at the truth is already at the optimum (up to the hull's excess),
		// so refinement must stop in a few steps per level instead of running every round.
		SyntheticObjectScene scene = Scene.Value;
		var views = new List<VisualHullView>();
		for (int k = 0; k < scene.Frames.Count; k += 2)
		{
			views.Add(new VisualHullView(scene.Camera, scene.CamFromWorld[k], scene.Masks[k]));
		}

		SilhouettePoseOptions options = Options(scene);
		var model = SilhouetteHullModel.FromHull(VisualHull.Build(views, options.HullBox!.Value, options.HullOptions));
		foreach (int k in new[] { 1, 9, 21 })
		{
			SilhouettePoseRefinement refined = SilhouettePoseRefiner.Refine(
				scene.Camera, scene.Masks[k], scene.CamFromWorld[k], model, options);
			double rotation = refined.CamFromWorld.Rotation.AngularDistance(scene.CamFromWorld[k].Rotation) * 180 / Math.PI;
			Console.WriteLine($"from truth, frame {k}: {refined.Iterations} steps, IoU {refined.InitialIou:F4} -> {refined.Iou:F4}, rot {rotation:F3} deg");
			// Before the stall test every start ran all 30 rounds of every level (90 steps).
			await Assert.That(refined.Iterations).IsLessThanOrEqualTo(8 * options.PyramidLevels);
			// The drift from the truth is the hull's excess, not the search: frame 9's hull IoU at
			// the true pose is only 0.970, and refinement moves 0.77 degrees to a pose that fits
			// that hull better (0.977). Frames 1 and 21, where the hull is tight, stay near 0.2.
			await Assert.That(rotation).IsLessThan(1.0);
		}
	}

	[Test]
	public async Task CSharpOnly_ConsistencyGateRejectsRolledPose()
	{
		// Frame 11 at its true pose carves nothing the even frames need; the same frame rolled
		// 10 degrees about its optical axis still shows a plausible silhouette, but its cone cuts
		// into object the even frames see.
		SyntheticObjectScene scene = Scene.Value;
		var references = new List<VisualHullView>();
		for (int k = 0; k < scene.Frames.Count; k += 2)
		{
			references.Add(new VisualHullView(scene.Camera, scene.CamFromWorld[k], scene.Masks[k]));
		}

		SilhouettePoseOptions options = Options(scene);
		var gate = new SilhouetteConsistencyGate(references, options.HullBox!.Value, options.HullOptions);
		const int K = 11;
		Rigid3d truth = scene.CamFromWorld[K];
		Quaterniond roll = Quaterniond.FromAngleAxis(new AngleAxisd(10 * Math.PI / 180, new Vector3d(0, 0, 1)));
		var rolled = new Rigid3d(roll * truth.Rotation, roll * truth.Translation);

		bool acceptsRolled = gate.TryAccept(new VisualHullView(scene.Camera, rolled, scene.Masks[K]), out double rolledMedian, out double rolledMax);
		bool acceptsTruth = gate.TryAccept(new VisualHullView(scene.Camera, truth, scene.Masks[K]), out double truthMedian, out double truthMax);
		Console.WriteLine(
			$"gate: truth median drop {truthMedian:F4} worst {truthMax:F4} -> {acceptsTruth}; "
			+ $"rolled median drop {rolledMedian:F4} worst {rolledMax:F4} -> {acceptsRolled}");
		await Assert.That(acceptsTruth).IsTrue();
		await Assert.That(acceptsRolled).IsFalse();
	}

	[Test]
	public async Task CSharpOnly_TurnLimitRejectsSilhouetteTwin()
	{
		// A symmetric silhouette fits its 180-degree twin as well as the truth; the continuity
		// bound must keep the twin out and let the truth in, with and without a prediction.
		SyntheticObjectScene scene = Scene.Value;
		var options = new SilhouettePoseOptions();
		TurntableFit fit = SilhouetteTurntable.Fit(scene.CamFromWorld);
		const int K = 11;
		Rigid3d truth = scene.CamFromWorld[K];
		Rigid3d twin = fit.PoseAt(truth, fit.AngleOf(truth.TgtOriginInSrc()) + Math.PI);

		Rigid3d?[] everyOther = [.. Enumerable.Range(0, 40).Select(k => k % 2 == 0 ? scene.CamFromWorld[k] : (Rigid3d?)null)];
		Rigid3d predicted = SilhouettePoseRegistration.InitializePose(everyOther, K, 2.0)!.Value;
		double limit = SilhouettePoseRegistration.TurnLimitDegrees(everyOther, K, true, options);
		double truthTurn = truth.Rotation.AngularDistance(predicted.Rotation) * 180 / Math.PI;
		double twinTurn = twin.Rotation.AngularDistance(predicted.Rotation) * 180 / Math.PI;
		Console.WriteLine($"interpolated: limit {limit:F1}, truth turn {truthTurn:F2}, twin turn {twinTurn:F1}");
		await Assert.That(truthTurn).IsLessThan(limit);
		await Assert.That(twinTurn).IsGreaterThan(limit);

		// Only frames 0-9 posed and no prediction: the bound comes from the motion's rate.
		Rigid3d?[] firstTen = [.. Enumerable.Range(0, 40).Select(k => k < 10 ? scene.CamFromWorld[k] : (Rigid3d?)null)];
		const int G = 12;
		Rigid3d gapTruth = scene.CamFromWorld[G];
		Rigid3d gapTwin = fit.PoseAt(gapTruth, fit.AngleOf(gapTruth.TgtOriginInSrc()) + Math.PI);
		double rateLimit = SilhouettePoseRegistration.TurnLimitDegrees(firstTen, G, false, options);
		double gapTruthTurn = gapTruth.Rotation.AngularDistance(scene.CamFromWorld[9].Rotation) * 180 / Math.PI;
		double gapTwinTurn = gapTwin.Rotation.AngularDistance(scene.CamFromWorld[9].Rotation) * 180 / Math.PI;
		Console.WriteLine($"rate: limit {rateLimit:F1}, truth turn {gapTruthTurn:F2}, twin turn {gapTwinTurn:F1}");
		await Assert.That(gapTruthTurn).IsLessThan(rateLimit);
		await Assert.That(gapTwinTurn).IsGreaterThan(rateLimit);
	}

	[Test]
	public async Task CSharpOnly_ExistingUnregisteredImagesAreRegisteredInPlace()
	{
		// The odd frames are already images in the reconstruction (as after a mapper run that
		// could not place them): they are registered through their own frames, and no image is added.
		SyntheticObjectScene scene = Scene.Value;
		Reconstruction reconstruction = Registered(scene, k => k % 2 == 0);
		for (int k = 1; k < 40; k += 2)
		{
			var image = new Image { ImageId = (uint)k + 1, Name = Name(k) };
			image.SetCameraId(scene.Camera.CameraId);
			reconstruction.AddImageWithTrivialFrame(image);
		}

		SilhouettePoseOptions options = Options(scene);
		options.Turntable = false;
		options.MaxPasses = 1;
		SilhouettePoseResult result = SilhouettePoseRegistration.Register(reconstruction, Frames(scene, null), options);
		await Assert.That(reconstruction.NumImages).IsEqualTo(40);
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(20 + result.PlacedCount);
		await Assert.That(result.PlacedCount).IsEqualTo(20);
		Rigid3d placed = reconstruction.Image(12).CamFromWorld();
		await Assert.That(placed.Rotation.AngularDistance(scene.CamFromWorld[11].Rotation) * 180 / Math.PI).IsLessThan(2.0);
	}

	[Test]
	public async Task CSharpOnly_TruncatedMaskRefinesFromTruth()
	{
		// The frame is cropped through the object, so the hull's silhouette runs off the image.
		// Contour points there are the crop, not the object, and must not pull the pose.
		SyntheticObjectScene scene = Scene.Value;
		var views = new List<VisualHullView>();
		for (int k = 0; k < scene.Frames.Count; k += 2)
		{
			views.Add(new VisualHullView(scene.Camera, scene.CamFromWorld[k], scene.Masks[k]));
		}

		SilhouettePoseOptions options = Options(scene);
		SilhouetteHullModel model = SilhouetteHullModel.FromHull(VisualHull.Build(views, options.HullBox!.Value, options.HullOptions));
		const int K = 21;
		Bitmap full = scene.Masks[K];
		long sumX = 0, count = 0;
		for (int i = 0; i < full.RowMajorData.Length; i++)
		{
			if (full.RowMajorData[i] >= 128)
			{
				sumX += i % full.Width;
				count++;
			}
		}

		int width = (int)(sumX / count);
		Camera cropped = scene.Camera.Clone();
		cropped.Width = width;
		var mask = new Bitmap(width, full.Height, asRgb: false);
		for (int y = 0; y < full.Height; y++)
		{
			Array.Copy(full.RowMajorData, y * full.Width, mask.RowMajorData, y * width, width);
		}

		SilhouettePoseRefinement refined = SilhouettePoseRefiner.Refine(cropped, mask, scene.CamFromWorld[K], model, options);
		double rotation = refined.CamFromWorld.Rotation.AngularDistance(scene.CamFromWorld[K].Rotation) * 180 / Math.PI;
		double shift = (refined.CamFromWorld.TgtOriginInSrc() - scene.CamFromWorld[K].TgtOriginInSrc()).Norm;
		double distance = (scene.CamFromWorld[K].TgtOriginInSrc() - model.Centroid).Norm;
		Console.WriteLine($"truncated at x={width}: rot {rotation:F3} deg, centre shift {shift / distance:P2} of distance, IoU {refined.Iou:F4}");
		await Assert.That(rotation).IsLessThan(0.5);
		await Assert.That(shift / distance).IsLessThan(0.01);
	}

	[Test]
	public async Task CSharpOnly_BittenMaskIsNotPlacedWildly()
	{
		SyntheticObjectScene scene = Scene.Value;
		const int Bitten = 11;
		Reconstruction reconstruction = Registered(scene, k => k % 2 == 0);
		SilhouettePoseResult result = SilhouettePoseRegistration.Register(
			reconstruction, Frames(scene, Bitten), Options(scene));
		SilhouettePoseFrameReport report = result.Frames.Single(f => f.Name == Name(Bitten));
		Console.WriteLine($"bitten frame: placed {report.Placed}, IoU {report.Iou:F4}, iterations {report.Iterations}, {report.Reason}");
		if (report.Placed)
		{
			Rigid3d estimate = reconstruction.FindImageWithName(Name(Bitten))!.CamFromWorld();
			double rotation = estimate.Rotation.AngularDistance(scene.CamFromWorld[Bitten].Rotation) * 180 / Math.PI;
			Console.WriteLine($"bitten frame rotation error {rotation:F3} deg");
			await Assert.That(rotation).IsLessThan(5.0);
		}
	}

	[Test]
	public async Task CSharpOnly_SequentialEqualsParallel()
	{
		SyntheticObjectScene scene = Scene.Value;
		var poses = new List<List<Rigid3d>>();
		foreach (bool parallel in new[] { false, true })
		{
			Reconstruction reconstruction = Registered(scene, k => k % 4 != 1);
			SilhouettePoseOptions options = Options(scene);
			options.Parallel = parallel;
			options.HullOptions.Parallel = parallel;
			options.MaxPasses = 1;
			// The turntable pass is sequential by design; this pins the Parallel.For path.
			options.Turntable = false;
			SilhouettePoseRegistration.Register(reconstruction, Frames(scene, null), options);
			poses.Add(Enumerable.Range(0, 40).Where(k => k % 4 == 1)
				.Select(k => reconstruction.FindImageWithName(Name(k)) is Image image ? image.CamFromWorld() : new Rigid3d()).ToList());
		}

		for (int i = 0; i < poses[0].Count; i++)
		{
			await Assert.That(poses[1][i].Rotation.Coeffs).IsEqualTo(poses[0][i].Rotation.Coeffs);
			await Assert.That(poses[1][i].Translation).IsEqualTo(poses[0][i].Translation);
		}
	}

	private static Rigid3d At(double angle, Vector3d center)
	{
		Quaterniond q = Quaterniond.FromAngleAxis(new AngleAxisd(angle, new Vector3d(0, 1, 0)));
		return new Rigid3d(q, -(q * center));
	}

	private static string Name(int k) => $"frame{k:D3}.png";

	private static Reconstruction Registered(SyntheticObjectScene scene, Func<int, bool> registered)
	{
		var reconstruction = new Reconstruction();
		reconstruction.AddCameraWithTrivialRig(scene.Camera);
		for (int k = 0; k < scene.Frames.Count; k++)
		{
			if (registered(k))
			{
				var image = new Image { ImageId = (uint)k + 1, Name = Name(k) };
				image.SetCameraId(scene.Camera.CameraId);
				reconstruction.AddImageWithTrivialFrame(image, scene.CamFromWorld[k]);
			}
		}

		return reconstruction;
	}

	// The true masks; frame `bitten` loses a disc a third of the silhouette's width wide at the
	// right end of its silhouette, like a segmenter that missed the object's tail.
	private static List<SilhouetteFrame> Frames(SyntheticObjectScene scene, int? bitten)
	{
		var frames = new List<SilhouetteFrame>();
		for (int k = 0; k < scene.Frames.Count; k++)
		{
			Bitmap mask = scene.Masks[k];
			if (k == bitten)
			{
				mask = Bite(mask);
			}

			frames.Add(new SilhouetteFrame(Name(k), mask));
		}

		return frames;
	}

	private static Bitmap Bite(Bitmap mask)
	{
		var bitten = new Bitmap(mask.Width, mask.Height, asRgb: false);
		byte[] source = mask.RowMajorData, data = bitten.RowMajorData;
		Array.Copy(source, data, data.Length);
		int minX = int.MaxValue, maxX = -1;
		long sumY = 0, count = 0;
		for (int y = 0; y < mask.Height; y++)
		{
			for (int x = 0; x < mask.Width; x++)
			{
				if (source[y * mask.Width + x] >= 128)
				{
					minX = Math.Min(minX, x);
					maxX = Math.Max(maxX, x);
					sumY += y;
					count++;
				}
			}
		}

		double radius = (maxX - minX) / 3.0, cx = maxX, cy = sumY / (double)count;
		for (int y = 0; y < mask.Height; y++)
		{
			for (int x = 0; x < mask.Width; x++)
			{
				if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius)
				{
					data[y * mask.Width + x] = 0;
				}
			}
		}

		return bitten;
	}

	private static SilhouettePoseOptions Options(SyntheticObjectScene scene) => new()
	{
		HullBox = VisualHullBounds.FromPoints(scene.MeshVertices),
		HullOptions = new VisualHullOptions { Resolution = 128 },
	};

	// Prints each frame's report and error against the truth; returns the median rotation
	// error (degrees) and camera-centre error (% of the object's bounding-box diagonal) over the
	// placed frames (NaN if none).
	private static (double Rotation, double Position) Errors(
		SyntheticObjectScene scene, Reconstruction reconstruction, SilhouettePoseResult result, string label)
	{
		var min = new Vector3d(double.MaxValue, double.MaxValue, double.MaxValue);
		var max = new Vector3d(double.MinValue, double.MinValue, double.MinValue);
		foreach (Vector3d v in scene.MeshVertices)
		{
			min = new Vector3d(Math.Min(min.X, v.X), Math.Min(min.Y, v.Y), Math.Min(min.Z, v.Z));
			max = new Vector3d(Math.Max(max.X, v.X), Math.Max(max.Y, v.Y), Math.Max(max.Z, v.Z));
		}

		double diagonal = (max - min).Norm;
		var rotations = new List<double>();
		var positions = new List<double>();
		foreach (SilhouettePoseFrameReport report in result.Frames)
		{
			int k = int.Parse(report.Name.AsSpan(5, 3));
			string error = "";
			if (report.Start is Rigid3d start)
			{
				error = $" start rot {start.Rotation.AngularDistance(scene.CamFromWorld[k].Rotation) * 180 / Math.PI:F3} deg"
					+ $" pos {(start.TgtOriginInSrc() - scene.CamFromWorld[k].TgtOriginInSrc()).Norm / diagonal * 100:F3}%;";
			}

			if (report.Placed)
			{
				Rigid3d estimate = reconstruction.FindImageWithName(report.Name)!.CamFromWorld();
				double rotation = estimate.Rotation.AngularDistance(scene.CamFromWorld[k].Rotation) * 180 / Math.PI;
				double position = (estimate.TgtOriginInSrc() - scene.CamFromWorld[k].TgtOriginInSrc()).Norm / diagonal * 100;
				rotations.Add(rotation);
				positions.Add(position);
				error += $" rot {rotation:F3} deg, pos {position:F3}%";
				if (result.Hull is not null)
				{
					// The hull's IoU at the true pose: below the refined IoU means the hull's excess
					// over the object, not the pose search, limits the accuracy.
					double truthIou = SilhouettePoseRefiner.Iou(
						scene.Camera, scene.Masks[k], scene.CamFromWorld[k], SilhouetteHullModel.FromHull(result.Hull));
					error += $", IoU at truth {truthIou:F4}";
				}
			}

			Console.WriteLine(
				$"{label}: {report.Name} pass {report.Pass} placed {report.Placed} IoU {report.InitialIou:F4} -> {report.Iou:F4}, "
				+ $"{report.Iterations} it, turned {report.RotationFromInitDegrees:F2} deg{error} {report.Reason}");
		}

		double rotationMedian = Median(rotations), positionMedian = Median(positions);
		Console.WriteLine($"{label}: placed {result.PlacedCount}/{result.Frames.Count}, median rot {rotationMedian:F3} deg, pos {positionMedian:F3}%");
		return (rotationMedian, positionMedian);
	}

	private static double Median(List<double> values)
	{
		if (values.Count == 0)
		{
			return double.NaN;
		}

		values.Sort();
		return values[values.Count / 2];
	}
}
