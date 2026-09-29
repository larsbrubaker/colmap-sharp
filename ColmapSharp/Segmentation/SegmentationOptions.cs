// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SegmentationOptions: the knobs of SilhouetteSegmenter (automatic foreground-object masks for
// a single object in front of a plain background). Not a COLMAP port; COLMAP takes masks from
// the user. The algorithms and their sources are listed in SilhouetteSegmenter.cs.

using ColmapSharp.Util;

namespace ColmapSharp.Segmentation;

/// <summary>
/// Options of <see cref="SilhouetteSegmenter"/>. The defaults suit a phone video or photo set
/// of one object in front of a plain wall. Distances and radii are in working-resolution
/// pixels, colour distances in CIE Lab units (ΔE 1976).
/// </summary>
public sealed class SegmentationOptions
{
	/// <summary>
	/// The longest side, in pixels, that frames are box-downsampled to before segmenting. The
	/// masks are upsampled back to each frame's full size.
	/// </summary>
	public int WorkingSize { get; set; } = 320;

	// Note: when the Otsu fallback is used, a foreground class that covers more than a quarter
	// of the image border counts as wall (BackgroundModel.MaxObjectBorderFraction). A close-up
	// object that fills the frame is therefore reported as not found and left unmasked.

	/// <summary>
	/// The fewest frames (of one size) for which the per-pixel temporal median is used as the
	/// background model. Below this the Otsu threshold on L* is used instead.
	/// </summary>
	public int MinMedianFrames { get; set; } = 5;

	/// <summary>
	/// The most frames that go into the temporal median; longer sequences are sampled evenly.
	/// Bounds the memory the median needs.
	/// </summary>
	public int MaxMedianFrames { get; set; } = 25;

	/// <summary>
	/// The camera counts as near-static when, in the median frame, at least this fraction of
	/// pixels lies within <see cref="BackgroundDistance"/> of the median background.
	/// </summary>
	public double StaticAgreement { get; set; } = 0.6;

	/// <summary>
	/// A pixel is initially foreground when its Lab distance from the background median is
	/// larger than this.
	/// </summary>
	public double BackgroundDistance { get; set; } = 15.0;

	/// <summary>
	/// The median model is rejected (and Otsu used) when the median background itself holds a
	/// dark or light blob of at least this fraction of the image that does not reach the image
	/// border: an object that barely moves (a mouse spinning on its cable) survives the median
	/// and would read as background.
	/// </summary>
	public double PersistentObjectMinArea { get; set; } = 0.01;

	/// <summary>Erosion radius of the initial mask that gives the sure-foreground of the trimap.</summary>
	public double TrimapErodeRadius { get; set; } = 2.0;

	/// <summary>
	/// Dilation radius of the initial mask outside which the trimap is sure-background (unless
	/// inside the object's convex hull). 12 beat 6 on the mouse captures: a wider unknown band
	/// lets GrabCut recover rim that the background difference missed, and never hurt.
	/// </summary>
	public double TrimapDilateRadius { get; set; } = 12.0;

	/// <summary>Gaussian components per colour model (GrabCut uses 5).</summary>
	public int GmmComponents { get; set; } = 5;

	/// <summary>Lloyd iterations of the deterministic k-means that seeds each colour model.</summary>
	public int KMeansIterations { get; set; } = 8;

	/// <summary>Added to each covariance diagonal (Lab units squared), so flat colours stay invertible.</summary>
	public double CovarianceRegularization { get; set; } = 1.0;

	/// <summary>GrabCut iterations (component assignment, model refit, graph cut).</summary>
	public int GrabCutIterations { get; set; } = 3;

	/// <summary>The Potts smoothness weight γ of GrabCut (the paper uses 50).</summary>
	public double Smoothness { get; set; } = 50.0;

	/// <summary>
	/// The opening radius as a fraction of the object's width (twice its largest inscribed
	/// radius). Opening removes parts thinner than about twice the radius, such as a cable.
	/// </summary>
	public double OpeningFraction { get; set; } = 0.1;

	/// <summary>
	/// The closing radius, applied after the opening, as a fraction of the object's width. It
	/// seals thin channels and notches cut into the rim (white label text on a dark object).
	/// </summary>
	public double CloseFraction { get; set; } = 0.1;

	/// <summary>Worker count for frames and the median (-1: all cores, 1: sequential).</summary>
	public int MaxDegreeOfParallelism { get; set; } = -1;

	/// <summary>Keep only the largest connected foreground component.</summary>
	public bool KeepLargestComponent { get; set; } = true;

	/// <summary>
	/// Fill holes in the foreground, so a bright specular highlight inside the object stays
	/// foreground. A true through-hole (a mug handle) is filled too.
	/// </summary>
	public bool FillHoles { get; set; } = true;

	/// <summary>Throws (Check failed) when an option is out of range.</summary>
	public void Validate()
	{
		Check.Gt(WorkingSize, 0);
		Check.Ge(MinMedianFrames, 1);
		Check.Ge(MaxMedianFrames, 1);
		Check.Ge(StaticAgreement, 0.0);
		Check.Le(StaticAgreement, 1.0);
		Check.Gt(BackgroundDistance, 0.0);
		Check.Ge(PersistentObjectMinArea, 0.0);
		Check.Le(PersistentObjectMinArea, 1.0);
		Check.Ge(TrimapErodeRadius, 0.0);
		Check.Ge(TrimapDilateRadius, 0.0);
		Check.Ge(GmmComponents, 1);
		Check.Ge(KMeansIterations, 0);
		Check.Gt(CovarianceRegularization, 0.0);
		Check.Ge(GrabCutIterations, 0);
		Check.Ge(Smoothness, 0.0);
		Check.Ge(OpeningFraction, 0.0);
		Check.Ge(CloseFraction, 0.0);
		Check.That(MaxDegreeOfParallelism == -1 || MaxDegreeOfParallelism >= 1, "MaxDegreeOfParallelism must be -1 or at least 1");
	}
}
